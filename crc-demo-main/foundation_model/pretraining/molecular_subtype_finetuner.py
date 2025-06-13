#!/usr/bin/env python3
"""
Molecular Subtype Fine-tuning Module
Implements advanced fine-tuning strategies for CRC molecular subtype prediction
Based on Pitroda classification: Canonical, Immune, Stromal
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from pathlib import Path
import yaml
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from sklearn.preprocessing import label_binarize
import wandb
from tqdm import tqdm
import pandas as pd

# Import augmentation utilities
from .augmentation_utils import (
    MixupCutmixAugmentation, 
    mixup_criterion,
    TestTimeAugmentation
)


class MolecularSubtypeClassifier(nn.Module):
    """
    Advanced classifier head for molecular subtype prediction
    Supports MLP architecture with dropout and optional metadata integration
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
        num_classes: int = 3,
        classifier_type: str = "mlp",
        hidden_dims: List[int] = [256],
        dropout: float = 0.3,
        use_metadata: bool = False,
        metadata_dim: int = 0
    ):
        super().__init__()
        self.classifier_type = classifier_type
        self.use_metadata = use_metadata
        
        input_dim = feature_dim + metadata_dim if use_metadata else feature_dim
        
        if classifier_type == "linear":
            self.classifier = nn.Linear(input_dim, num_classes)
        elif classifier_type == "mlp":
            layers = []
            prev_dim = input_dim
            
            # Hidden layers
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                ])
                prev_dim = hidden_dim
            
            # Output layer
            layers.append(nn.Linear(prev_dim, num_classes))
            self.classifier = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def forward(
        self, 
        features: torch.Tensor,
        metadata: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with optional metadata integration"""
        if self.use_metadata and metadata is not None:
            # Concatenate image features with metadata
            combined = torch.cat([features, metadata], dim=1)
        else:
            combined = features
        
        return self.classifier(combined)


class ClinicalMetadataProcessor(nn.Module):
    """
    Process and embed clinical metadata for integration with image features
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Embedding layers for categorical features
        self.categorical_embeddings = nn.ModuleDict()
        self.categorical_features = []
        self.numerical_features = []
        
        total_dim = 0
        
        # Parse features
        for feature in config['clinical_integration']['features']:
            if feature in ['sex', 'tumor_location', 'msi_status', 'stage']:
                # Categorical features
                self.categorical_features.append(feature)
                
                # Create embedding layer (simplified - in practice, need vocab size)
                embed_dim = config['clinical_integration']['embedding_dims']['categorical']
                self.categorical_embeddings[feature] = nn.Embedding(10, embed_dim)
                total_dim += embed_dim
            else:
                # Numerical features
                self.numerical_features.append(feature)
                total_dim += config['clinical_integration']['embedding_dims']['numerical']
        
        self.output_dim = total_dim
        
        # Optional fusion layer
        if config['clinical_integration']['fusion_method'] == 'attention':
            self.fusion = nn.MultiheadAttention(
                embed_dim=total_dim,
                num_heads=4,
                batch_first=True
            )
    
    def forward(self, metadata_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Process metadata dictionary into feature vector"""
        embeddings = []
        
        # Process categorical features
        for feature in self.categorical_features:
            if feature in metadata_dict:
                embedded = self.categorical_embeddings[feature](metadata_dict[feature])
                embeddings.append(embedded)
        
        # Process numerical features
        for feature in self.numerical_features:
            if feature in metadata_dict:
                # Normalize numerical features
                normalized = (metadata_dict[feature] - metadata_dict[feature].mean()) / (metadata_dict[feature].std() + 1e-6)
                embeddings.append(normalized.unsqueeze(-1))
        
        # Concatenate all features
        if embeddings:
            return torch.cat(embeddings, dim=-1)
        else:
            return torch.zeros(metadata_dict[list(metadata_dict.keys())[0]].shape[0], self.output_dim)


class MolecularSubtypeFinetuner:
    """
    Advanced fine-tuning module with differential LR, gradual unfreezing,
    semi-supervised learning, and clinical metadata integration
    """
    
    def __init__(
        self,
        foundation_model: nn.Module,
        config_path: str,
        device: torch.device
    ):
        """Initialize fine-tuner with pre-trained foundation model"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = device
        self.foundation_model = foundation_model.to(device)
        
        # Setup logging
        self.setup_logging()
        
        # Extract fine-tuning config
        self.ft_config = self.config['finetuning']
        self.clinical_config = self.config.get('clinical_integration', {})
        
        # Create classifier head
        self.classifier = self.create_classifier()
        
        # Create metadata processor if enabled
        if self.clinical_config.get('enabled', False):
            self.metadata_processor = ClinicalMetadataProcessor(self.config).to(device)
        else:
            self.metadata_processor = None
        
        # Setup optimizer with differential learning rates
        self.optimizer = self.setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self.setup_scheduler()
        
        # Loss function
        self.criterion = self.setup_loss_function()
        
        # Training state
        self.current_epoch = 0
        self.best_val_metric = -float('inf')
        self.patience_counter = 0
        self.frozen_layers = []
        
        # Augmentation setup
        self.use_mixup_cutmix = self.config['augmentations']['advanced'].get('use_mixup', False) or \
                               self.config['augmentations']['advanced'].get('use_cutmix', False)
        if self.use_mixup_cutmix:
            self.mixup_cutmix = MixupCutmixAugmentation(
                mixup_alpha=self.config['augmentations']['advanced'].get('mixup_alpha', 0.2),
                cutmix_alpha=self.config['augmentations']['advanced'].get('cutmix_alpha', 1.0),
                mixup_prob=0.5 if self.config['augmentations']['advanced'].get('use_mixup', False) else 0,
                cutmix_prob=0.5 if self.config['augmentations']['advanced'].get('use_cutmix', False) else 0,
                use_cuda=device.type == 'cuda'
            )
    
    def setup_logging(self):
        """Setup logging and experiment tracking"""
        log_dir = Path("./logs/finetuning")
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"finetuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize Weights & Biases if enabled
        if self.config['logging']['wandb']['enabled']:
            wandb.init(
                project="crc_molecular_subtyping",
                config=self.config,
                name=f"finetuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def create_classifier(self) -> nn.Module:
        """Create molecular subtype classifier head"""
        classifier_config = self.ft_config['molecular_subtyping']
        
        # Determine if using metadata
        use_metadata = self.clinical_config.get('enabled', False)
        metadata_dim = 0
        
        if use_metadata and self.metadata_processor is not None:
            metadata_dim = self.metadata_processor.output_dim
        
        classifier = MolecularSubtypeClassifier(
            feature_dim=self.config['model']['architecture']['feature_dim'],
            num_classes=classifier_config['num_classes'],
            classifier_type=classifier_config['classifier_type'],
            hidden_dims=classifier_config.get('mlp_hidden_dims', [256]),
            dropout=classifier_config.get('dropout', 0.3),
            use_metadata=use_metadata,
            metadata_dim=metadata_dim
        )
        
        return classifier.to(self.device)
    
    def setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer with differential learning rates"""
        if self.ft_config['differential_lr']['enabled']:
            # Different learning rates for backbone and classifier
            param_groups = [
                {
                    'params': self.foundation_model.parameters(),
                    'lr': self.ft_config['differential_lr']['backbone_lr']
                },
                {
                    'params': self.classifier.parameters(),
                    'lr': self.ft_config['differential_lr']['classifier_lr']
                }
            ]
            
            # Add metadata processor if exists
            if self.metadata_processor is not None:
                param_groups.append({
                    'params': self.metadata_processor.parameters(),
                    'lr': self.ft_config['differential_lr']['classifier_lr']
                })
        else:
            # Single learning rate for all parameters
            params = list(self.foundation_model.parameters()) + \
                    list(self.classifier.parameters())
            if self.metadata_processor is not None:
                params += list(self.metadata_processor.parameters())
            
            param_groups = [{'params': params, 'lr': self.config['training']['learning_rate']}]
        
        return AdamW(param_groups, weight_decay=self.config['training']['weight_decay'])
    
    def setup_scheduler(self):
        """Setup learning rate scheduler"""
        if self.ft_config['early_stopping']['enabled']:
            # Use ReduceLROnPlateau for adaptive scheduling
            return ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5,
                verbose=True
            )
        else:
            # Use cosine annealing with warm restarts
            return CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=10,
                T_mult=2,
                eta_min=1e-6
            )
    
    def setup_loss_function(self):
        """Setup loss function based on configuration"""
        if self.ft_config['class_balancing']['enabled']:
            if self.ft_config['class_balancing']['method'] == 'focal_loss':
                # Focal loss for imbalanced classes
                return FocalLoss(
                    gamma=self.ft_config['class_balancing']['focal_loss_gamma'],
                    alpha=None  # Can be set based on class frequencies
                )
            else:
                # Will use weighted cross-entropy (weights set during training)
                return nn.CrossEntropyLoss(reduction='none')
        else:
            return nn.CrossEntropyLoss()
    
    def freeze_backbone(self):
        """Freeze backbone for initial training"""
        self.logger.info("Freezing backbone layers")
        for param in self.foundation_model.parameters():
            param.requires_grad = False
        self.frozen_layers = list(self.foundation_model.parameters())
    
    def unfreeze_gradual(self, epoch: int):
        """Gradually unfreeze layers based on schedule"""
        if not self.ft_config['gradual_unfreezing']['enabled']:
            return
        
        schedule = self.ft_config['gradual_unfreezing']['unfreeze_schedule']
        
        if epoch in schedule:
            if epoch == schedule[0]:
                # Unfreeze last block
                self.logger.info("Unfreezing last backbone block")
                # This is simplified - in practice, need to identify last block
                # For ResNet, would be layer4
                if hasattr(self.foundation_model, 'base_encoder'):
                    if hasattr(self.foundation_model.base_encoder, 'layer4'):
                        for param in self.foundation_model.base_encoder.layer4.parameters():
                            param.requires_grad = True
            else:
                # Unfreeze all
                self.logger.info("Unfreezing entire backbone")
                for param in self.foundation_model.parameters():
                    param.requires_grad = True
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        unlabeled_loader: Optional[DataLoader] = None,
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Train one epoch with optional semi-supervised learning"""
        self.foundation_model.train()
        self.classifier.train()
        if self.metadata_processor:
            self.metadata_processor.train()
        
        total_loss = 0
        supervised_loss_total = 0
        consistency_loss_total = 0
        all_preds = []
        all_labels = []
        
        # Create iterator for unlabeled data if semi-supervised
        if self.ft_config['semi_supervised']['enabled'] and unlabeled_loader is not None:
            unlabeled_iter = iter(unlabeled_loader)
        
        with tqdm(train_loader, desc=f"Epoch {self.current_epoch}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Unpack batch
                if len(batch) == 3:
                    images, labels, metadata = batch
                else:
                    images, labels = batch
                    metadata = None
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Apply Mixup/CutMix if enabled
                if self.use_mixup_cutmix:
                    mixed_images, labels_a, labels_b, lam, mix_method = self.mixup_cutmix(images, labels)
                else:
                    mixed_images = images
                    labels_a = labels
                    labels_b = labels
                    lam = 1.0
                    mix_method = 'none'
                
                # Extract features
                with torch.cuda.amp.autocast(enabled=self.config['training']['mixed_precision']):
                    features = self.foundation_model(mixed_images)['features']
                    
                    # Process metadata if available
                    if metadata is not None and self.metadata_processor is not None:
                        metadata_features = self.metadata_processor(metadata)
                    else:
                        metadata_features = None
                    
                    # Get predictions
                    logits = self.classifier(features, metadata_features)
                    
                    # Supervised loss
                    if mix_method != 'none':
                        # Mixed loss for Mixup/CutMix
                        if isinstance(self.criterion, nn.CrossEntropyLoss) and class_weights is not None:
                            # Weighted mixed loss
                            loss_a = (self.criterion(logits, labels_a) * class_weights[labels_a]).mean()
                            loss_b = (self.criterion(logits, labels_b) * class_weights[labels_b]).mean()
                            loss = lam * loss_a + (1 - lam) * loss_b
                        else:
                            loss = mixup_criterion(self.criterion, logits, labels_a, labels_b, lam)
                    else:
                        # Standard loss
                        if isinstance(self.criterion, nn.CrossEntropyLoss) and class_weights is not None:
                            # Weighted cross-entropy
                            loss = (self.criterion(logits, labels) * class_weights[labels]).mean()
                        else:
                            loss = self.criterion(logits, labels)
                    
                    supervised_loss = loss.clone()
                
                # Semi-supervised consistency loss
                if self.ft_config['semi_supervised']['enabled'] and unlabeled_loader is not None:
                    try:
                        # Get unlabeled batch
                        unlabeled_batch = next(unlabeled_iter)
                        if isinstance(unlabeled_batch, tuple):
                            unlabeled_images1, unlabeled_images2 = unlabeled_batch
                        else:
                            # Create two augmented views
                            unlabeled_images1 = unlabeled_batch
                            unlabeled_images2 = unlabeled_batch  # Should be augmented differently
                        
                        unlabeled_images1 = unlabeled_images1.to(self.device)
                        unlabeled_images2 = unlabeled_images2.to(self.device)
                        
                        # Get features for both views
                        with torch.no_grad():
                            features1 = self.foundation_model(unlabeled_images1)['features']
                            features2 = self.foundation_model(unlabeled_images2)['features']
                        
                        # Consistency loss
                        logits1 = self.classifier(features1)
                        logits2 = self.classifier(features2)
                        
                        # Simple consistency: minimize MSE between predictions
                        consistency_loss = F.mse_loss(
                            F.softmax(logits1, dim=1),
                            F.softmax(logits2, dim=1)
                        )
                        
                        # Add to total loss
                        loss = loss + self.ft_config['semi_supervised']['consistency_weight'] * consistency_loss
                        consistency_loss_total += consistency_loss.item()
                        
                    except StopIteration:
                        # Reset iterator
                        unlabeled_iter = iter(unlabeled_loader)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    list(self.foundation_model.parameters()) + 
                    list(self.classifier.parameters()),
                    self.config['training']['gradient_clip']
                )
                
                self.optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                supervised_loss_total += supervised_loss.item()
                
                # Store predictions for metrics
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'sup_loss': f'{supervised_loss.item():.4f}',
                    'cons_loss': f'{consistency_loss_total/(batch_idx+1):.4f}' if consistency_loss_total > 0 else 'N/A'
                })
        
        # Calculate epoch metrics
        metrics = {
            'total_loss': total_loss / len(train_loader),
            'supervised_loss': supervised_loss_total / len(train_loader),
            'consistency_loss': consistency_loss_total / len(train_loader) if consistency_loss_total > 0 else 0,
            'accuracy': np.mean(np.array(all_preds) == np.array(all_labels)),
            'f1_macro': f1_score(all_labels, all_preds, average='macro')
        }
        
        return metrics
    
    def validate(
        self,
        val_loader: DataLoader
    ) -> Dict[str, float]:
        """Validate model and compute comprehensive metrics"""
        self.foundation_model.eval()
        self.classifier.eval()
        if self.metadata_processor:
            self.metadata_processor.eval()
        
        all_preds = []
        all_probs = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                # Unpack batch
                if len(batch) == 3:
                    images, labels, metadata = batch
                else:
                    images, labels = batch
                    metadata = None
                
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                features = self.foundation_model(images)['features']
                
                if metadata is not None and self.metadata_processor is not None:
                    metadata_features = self.metadata_processor(metadata)
                else:
                    metadata_features = None
                
                logits = self.classifier(features, metadata_features)
                
                # Loss
                loss = self.criterion(logits, labels)
                if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                    loss = loss.mean()
                total_loss += loss.item()
                
                # Predictions
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        
        # Calculate metrics
        metrics = {
            'val_loss': total_loss / len(val_loader),
            'accuracy': np.mean(all_preds == all_labels),
            'f1_macro': f1_score(all_labels, all_preds, average='macro'),
            'mcc': matthews_corrcoef(all_labels, all_preds)
        }
        
        # Per-class metrics
        for i in range(self.ft_config['molecular_subtyping']['num_classes']):
            mask = all_labels == i
            if mask.sum() > 0:
                metrics[f'accuracy_class_{i}'] = np.mean(all_preds[mask] == i)
                metrics[f'f1_class_{i}'] = f1_score(
                    all_labels == i,
                    all_preds == i,
                    average='binary'
                )
        
        # AUC for multi-class
        if self.ft_config['molecular_subtyping']['num_classes'] > 2:
            # One-vs-rest AUC
            labels_binarized = label_binarize(
                all_labels,
                classes=list(range(self.ft_config['molecular_subtyping']['num_classes']))
            )
            for i in range(self.ft_config['molecular_subtyping']['num_classes']):
                if len(np.unique(labels_binarized[:, i])) > 1:
                    metrics[f'auc_class_{i}'] = roc_auc_score(
                        labels_binarized[:, i],
                        all_probs[:, i]
                    )
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        metrics['confusion_matrix'] = cm
        
        return metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        unlabeled_loader: Optional[DataLoader] = None,
        class_weights: Optional[torch.Tensor] = None
    ):
        """Main training loop with all advanced features"""
        self.logger.info("Starting molecular subtype fine-tuning")
        
        # Initial freezing if enabled
        if self.ft_config['gradual_unfreezing']['enabled']:
            self.freeze_backbone()
        
        for epoch in range(self.ft_config['epochs']):
            self.current_epoch = epoch
            
            # Gradual unfreezing
            self.unfreeze_gradual(epoch)
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader, unlabeled_loader, class_weights)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log metrics
            self.logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['total_loss']:.4f}, "
                           f"Val Loss: {val_metrics['val_loss']:.4f}, "
                           f"Val F1: {val_metrics['f1_macro']:.4f}")
            
            if wandb.run:
                wandb.log({
                    **{f'train/{k}': v for k, v in train_metrics.items()},
                    **{f'val/{k}': v for k, v in val_metrics.items() if k != 'confusion_matrix'},
                    'epoch': epoch
                })
            
            # Learning rate scheduling
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_metrics[self.ft_config['early_stopping']['monitor']])
            else:
                self.scheduler.step()
            
            # Early stopping
            if self.ft_config['early_stopping']['enabled']:
                metric = val_metrics[self.ft_config['early_stopping']['monitor']]
                
                if metric > self.best_val_metric:
                    self.best_val_metric = metric
                    self.patience_counter = 0
                    self.save_best_model(val_metrics)
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.ft_config['early_stopping']['patience']:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Regular checkpointing
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_metrics)
    
    def save_best_model(self, metrics: Dict):
        """Save best model checkpoint"""
        checkpoint = {
            'foundation_model': self.foundation_model.state_dict(),
            'classifier': self.classifier.state_dict(),
            'metadata_processor': self.metadata_processor.state_dict() if self.metadata_processor else None,
            'metrics': metrics,
            'config': self.config,
            'epoch': self.current_epoch
        }
        
        save_path = Path("./checkpoints/molecular_subtype_best.pth")
        save_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, save_path)
        self.logger.info(f"Saved best model to {save_path}")
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save regular checkpoint"""
        checkpoint = {
            'foundation_model': self.foundation_model.state_dict(),
            'classifier': self.classifier.state_dict(),
            'metadata_processor': self.metadata_processor.state_dict() if self.metadata_processor else None,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'epoch': epoch
        }
        
        save_path = Path(f"./checkpoints/molecular_subtype_epoch_{epoch}.pth")
        save_path.parent.mkdir(exist_ok=True)
        torch.save(checkpoint, save_path)


class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance"""
    
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def create_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Create balanced sampler for imbalanced datasets"""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(labels),
        replacement=True
    )


# Testing and example usage
if __name__ == "__main__":
    # Example configuration
    config_path = "./foundation_model/configs/pretraining_config.yaml"
    
    # Load pre-trained foundation model
    # foundation_model = load_foundation_model()
    
    # Create fine-tuner
    # finetuner = MolecularSubtypeFinetuner(
    #     foundation_model=foundation_model,
    #     config_path=config_path,
    #     device=torch.device('cuda')
    # )
    
    # Train
    # finetuner.train(train_loader, val_loader, unlabeled_loader) 