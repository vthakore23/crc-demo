#!/usr/bin/env python3
"""
EPOC-Integrated Molecular Subtype Training
Complete training pipeline with clinical-grade validation and performance monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as transforms
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
import argparse
import logging
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, 
    classification_report, matthews_corrcoef
)
from sklearn.model_selection import StratifiedKFold, train_test_split
import wandb
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from foundation_model.molecular_subtype_foundation import (
    MolecularSubtypeFoundationModel,
    create_molecular_foundation_model
)
from foundation_model.clinical_inference import ConfidenceCalibrator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EPOCMolecularDataset(Dataset):
    """Enhanced dataset for EPOC molecular subtype training"""
    
    def __init__(self, 
                 manifest_df: pd.DataFrame, 
                 data_dir: str, 
                 transform=None, 
                 augment=False,
                 molecular_validation=True):
        
        self.manifest = manifest_df.copy()
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.augment = augment
        self.molecular_validation = molecular_validation
        
        # Pitroda molecular subtype mapping
        self.subtype_map = {
            'canonical': 0, 'Canonical': 0, 'CANONICAL': 0,
            'immune': 1, 'Immune': 1, 'IMMUNE': 1, 
            'stromal': 2, 'Stromal': 2, 'STROMAL': 2
        }
        
        self.idx_to_subtype = {0: 'Canonical', 1: 'Immune', 2: 'Stromal'}
        
        # Validate molecular labels if required
        if molecular_validation:
            self._validate_molecular_labels()
        
        # Enhanced augmentation pipeline
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomRotation(degrees=(-20, 20)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
                transforms.RandomChoice([
                    transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
                    transforms.RandomAutocontrast(p=0.3),
                    transforms.RandomEqualize(p=0.2)
                ])
            ])
        
        logger.info(f"Dataset initialized with {len(self.manifest)} samples")
        self._log_distribution()
    
    def _validate_molecular_labels(self):
        """Validate molecular subtype labels"""
        if 'molecular_subtype' not in self.manifest.columns:
            raise ValueError("Missing 'molecular_subtype' column in manifest")
        
        # Check for valid subtypes
        valid_subtypes = set(self.subtype_map.keys())
        actual_subtypes = set(self.manifest['molecular_subtype'].unique())
        
        invalid_subtypes = actual_subtypes - valid_subtypes
        if invalid_subtypes:
            logger.warning(f"Found invalid molecular subtypes: {invalid_subtypes}")
            # Remove invalid entries
            mask = self.manifest['molecular_subtype'].isin(valid_subtypes)
            self.manifest = self.manifest[mask].copy()
            logger.info(f"Removed {(~mask).sum()} samples with invalid subtypes")
        
        # Check for molecular validation columns if available
        validation_columns = ['rna_seq_subtype', 'cms_classification', 'pathologist_review']
        available_validation = [col for col in validation_columns if col in self.manifest.columns]
        
        if available_validation:
            logger.info(f"Available molecular validation columns: {available_validation}")
        else:
            logger.warning("No molecular validation columns found. Using provided labels.")
    
    def _log_distribution(self):
        """Log class distribution and statistics"""
        if 'molecular_subtype' in self.manifest.columns:
            dist = self.manifest['molecular_subtype'].value_counts()
            logger.info(f"Molecular subtype distribution: {dist.to_dict()}")
            
            # Check for class imbalance
            min_count = dist.min()
            max_count = dist.max()
            imbalance_ratio = max_count / min_count
            
            if imbalance_ratio > 3:
                logger.warning(f"Significant class imbalance detected (ratio: {imbalance_ratio:.1f})")
            else:
                logger.info(f"Class balance acceptable (ratio: {imbalance_ratio:.1f})")
    
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        
        # Load image
        image_path = self._get_image_path(row)
        image = self._load_image(image_path)
        
        # Apply augmentation
        if self.augment and hasattr(self, 'augment_transform'):
            # Apply augmentation with probability
            if np.random.random() > 0.3:  # 70% chance of augmentation
                image = self.augment_transform(image)
        
        # Apply main transform
        if self.transform:
            image = self.transform(image)
        
        # Get molecular subtype label
        molecular_subtype = row.get('molecular_subtype', 'Canonical')
        label = self.subtype_map.get(molecular_subtype, 0)
        
        return image, label
    
    def _get_image_path(self, row):
        """Get image path from manifest row"""
        possible_columns = ['image_path', 'file_path', 'filename', 'image_file']
        
        for col in possible_columns:
            if col in row and pd.notna(row[col]):
                image_path = self.data_dir / row[col]
                if image_path.exists():
                    return image_path
        
        # If no valid path found, try using index
        extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.tif']
        for ext in extensions:
            image_path = self.data_dir / f"{row.name}{ext}"
            if image_path.exists():
                return image_path
        
        raise FileNotFoundError(f"Could not find image for row {row.name}")
    
    def _load_image(self, image_path):
        """Load and validate image"""
        try:
            image = Image.open(image_path).convert('RGB')
            
            # Validate image dimensions
            if image.size[0] < 50 or image.size[1] < 50:
                logger.warning(f"Image too small: {image_path}")
                return Image.new('RGB', (224, 224), (128, 128, 128))
            
            return image
            
        except Exception as e:
            logger.warning(f"Error loading image {image_path}: {e}")
            # Return blank image as fallback
            return Image.new('RGB', (224, 224), (128, 128, 128))

class AdvancedTrainer:
    """Advanced trainer with clinical-grade features"""
    
    def __init__(self, 
                 model: MolecularSubtypeFoundationModel,
                 device: str = 'cuda',
                 class_weights: Optional[torch.Tensor] = None):
        
        self.model = model.to(device)
        self.device = device
        
        # Loss functions with class weighting
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.focal_loss = FocalLoss(alpha=1.0, gamma=2.0, class_weights=class_weights)
        
        # Optimizer with different learning rates for different components
        backbone_params = list(self.model.backbone.parameters())
        other_params = [p for p in self.model.parameters() if p not in backbone_params]
        
        self.optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': 5e-5},  # Lower LR for pretrained backbone
            {'params': other_params, 'lr': 1e-4}     # Higher LR for new components
        ], weight_decay=0.01)
        
        # Advanced scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=[5e-5, 1e-4],
            epochs=100,  # Will be updated based on actual epochs
            steps_per_epoch=100,  # Will be updated based on actual steps
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': [],
            'learning_rates': [],
            'per_class_metrics': []
        }
        
        # Best model tracking
        self.best_val_acc = 0
        self.best_model_state = None
        
        logger.info("Advanced trainer initialized")
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with advanced monitoring"""
        self.model.train()
        
        epoch_loss = 0
        all_predictions = []
        all_targets = []
        batch_losses = []
        
        # Progress tracking
        num_batches = len(train_loader)
        log_interval = max(1, num_batches // 10)
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            
            # Multi-component loss
            ce_loss = self.ce_loss(output['logits'], target)
            molecular_loss = self.ce_loss(output['molecular_logits'], target)
            focal_loss = self.focal_loss(output['logits'], target)
            
            # Uncertainty loss if available
            uncertainty_loss = 0
            if output['uncertainty'] is not None:
                alpha = output['uncertainty']['alpha']
                uncertainty_loss = self._dirichlet_loss(alpha, target)
            
            # Combined loss with adaptive weighting
            total_loss = (
                0.4 * ce_loss +
                0.3 * molecular_loss +
                0.2 * focal_loss +
                0.1 * uncertainty_loss
            )
            
            # Backward pass with gradient clipping
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            epoch_loss += total_loss.item()
            batch_losses.append(total_loss.item())
            
            # Collect predictions for metrics
            pred = output['logits'].argmax(dim=1)
            all_predictions.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # Log progress
            if batch_idx % log_interval == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, "
                          f"Loss: {total_loss.item():.4f}, LR: {current_lr:.2e}")
        
        # Calculate epoch metrics
        avg_loss = epoch_loss / len(train_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        # Per-class metrics
        per_class_f1 = f1_score(all_targets, all_predictions, average=None)
        
        return avg_loss, accuracy, f1, per_class_f1
    
    def validate(self, val_loader):
        """Comprehensive validation"""
        self.model.eval()
        
        total_loss = 0
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_uncertainties = []
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                
                # Calculate loss
                loss = self.ce_loss(output['logits'], target)
                total_loss += loss.item()
                
                # Get predictions and probabilities
                probs = F.softmax(output['logits'], dim=1)
                pred = probs.argmax(dim=1)
                
                all_predictions.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
                
                # Collect uncertainties if available
                if output['uncertainty'] is not None:
                    uncertainty = output['uncertainty']['uncertainty']
                    all_uncertainties.extend(uncertainty.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(all_targets, all_predictions)
        f1_weighted = f1_score(all_targets, all_predictions, average='weighted')
        f1_macro = f1_score(all_targets, all_predictions, average='macro')
        
        # Per-class metrics
        per_class_f1 = f1_score(all_targets, all_predictions, average=None)
        per_class_acc = []
        
        for class_idx in range(3):
            class_mask = np.array(all_targets) == class_idx
            if class_mask.sum() > 0:
                class_preds = np.array(all_predictions)[class_mask]
                class_targets = np.array(all_targets)[class_mask]
                class_accuracy = accuracy_score(class_targets, class_preds)
                per_class_acc.append(class_accuracy)
            else:
                per_class_acc.append(0.0)
        
        # AUC scores
        all_probabilities = np.array(all_probabilities)
        auc_scores = []
        
        for class_idx in range(3):
            y_true = np.array(all_targets) == class_idx
            y_score = all_probabilities[:, class_idx]
            
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, y_score)
                auc_scores.append(auc)
            else:
                auc_scores.append(0.5)
        
        # Matthews Correlation Coefficient
        mcc = matthews_corrcoef(all_targets, all_predictions)
        
        # Average uncertainty
        avg_uncertainty = np.mean(all_uncertainties) if all_uncertainties else 0
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'per_class_f1': per_class_f1,
            'per_class_acc': per_class_acc,
            'auc_scores': auc_scores,
            'mcc': mcc,
            'avg_uncertainty': avg_uncertainty,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
    
    def _dirichlet_loss(self, alpha, target):
        """Dirichlet loss for uncertainty estimation"""
        S = torch.sum(alpha, dim=1, keepdim=True)
        target_one_hot = F.one_hot(target, num_classes=alpha.shape[1]).float()
        A = torch.sum(target_one_hot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
        B = torch.sum((1 - target_one_hot) * (torch.digamma(alpha) - torch.digamma(S)), dim=1)
        return torch.mean(A + B)
    
    def save_checkpoint(self, path: str, epoch: int, val_metrics: Dict):
        """Save comprehensive checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'val_metrics': val_metrics,
            'training_history': self.history,
            'model_config': {
                'backbone': 'efficientnet_b3',
                'num_classes': 3,
                'use_spatial_transformer': True,
                'use_uncertainty': True
            }
        }
        
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved: {path}")

class FocalLoss(nn.Module):
    """Focal loss with class weighting"""
    
    def __init__(self, alpha=1.0, gamma=2.0, class_weights=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def create_data_loaders(args):
    """Create comprehensive data loaders with proper validation"""
    
    # Load manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
    
    manifest_df = pd.read_csv(manifest_path)
    logger.info(f"Loaded manifest with {len(manifest_df)} samples")
    
    # Validate required columns
    required_columns = ['molecular_subtype']
    missing_columns = [col for col in required_columns if col not in manifest_df.columns]
    
    if missing_columns:
        if args.create_dummy_labels:
            logger.warning(f"Missing columns {missing_columns}, creating dummy labels")
            manifest_df['molecular_subtype'] = np.random.choice(
                ['Canonical', 'Immune', 'Stromal'], 
                size=len(manifest_df),
                p=[0.4, 0.35, 0.25]  # Realistic distribution
            )
        else:
            raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Stratified split
    train_df, temp_df = train_test_split(
        manifest_df,
        test_size=args.val_split + args.test_split,
        stratify=manifest_df['molecular_subtype'],
        random_state=42
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=args.test_split / (args.val_split + args.test_split),
        stratify=temp_df['molecular_subtype'],
        random_state=42
    )
    
    logger.info(f"Data splits - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Advanced transforms
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = EPOCMolecularDataset(
        train_df, args.data_dir, train_transform, 
        augment=True, molecular_validation=not args.create_dummy_labels
    )
    val_dataset = EPOCMolecularDataset(
        val_df, args.data_dir, val_transform, 
        augment=False, molecular_validation=not args.create_dummy_labels
    )
    test_dataset = EPOCMolecularDataset(
        test_df, args.data_dir, val_transform, 
        augment=False, molecular_validation=not args.create_dummy_labels
    )
    
    # Calculate class weights
    subtype_counts = train_df['molecular_subtype'].value_counts()
    total_samples = len(train_df)
    
    class_weights = []
    for subtype in ['Canonical', 'Immune', 'Stromal']:
        count = subtype_counts.get(subtype, 1)
        weight = total_samples / (3 * count)
        class_weights.append(weight)
    
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    logger.info(f"Class weights: {class_weights}")
    
    # Create weighted sampler for training
    train_labels = [train_dataset.subtype_map[label] for label in train_df['molecular_subtype']]
    sample_weights = [class_weights[label] for label in train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, class_weights_tensor

def train_epoc_molecular_model(args):
    """Main training function"""
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Initialize wandb if requested
    if args.use_wandb:
        wandb.init(
            project="epoc-molecular-subtype",
            name=f"epoc_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            config=vars(args)
        )
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_weights = create_data_loaders(args)
    
    # Create model
    model_config = {
        'backbone': args.backbone,
        'num_classes': 3,
        'pretrained': True,
        'use_spatial_transformer': True,
        'use_uncertainty': True
    }
    
    model = create_molecular_foundation_model(model_config)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    # Create trainer
    trainer = AdvancedTrainer(model, device, class_weights.to(device))
    
    # Update scheduler with actual parameters
    trainer.scheduler = optim.lr_scheduler.OneCycleLR(
        trainer.optimizer,
        max_lr=[5e-5, 1e-4],
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    logger.info(f"Training on device: {device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    patience = 20
    
    for epoch in range(args.epochs):
        logger.info(f"\n{'='*50}")
        logger.info(f"Epoch {epoch+1}/{args.epochs}")
        logger.info(f"{'='*50}")
        
        # Training
        train_loss, train_acc, train_f1, train_per_class_f1 = trainer.train_epoch(train_loader, epoch)
        
        # Validation
        val_metrics = trainer.validate(val_loader)
        
        # Update history
        trainer.history['train_loss'].append(train_loss)
        trainer.history['train_acc'].append(train_acc)
        trainer.history['train_f1'].append(train_f1)
        trainer.history['val_loss'].append(val_metrics['loss'])
        trainer.history['val_acc'].append(val_metrics['accuracy'])
        trainer.history['val_f1'].append(val_metrics['f1_weighted'])
        trainer.history['learning_rates'].append(trainer.optimizer.param_groups[0]['lr'])
        trainer.history['per_class_metrics'].append({
            'train_per_class_f1': train_per_class_f1.tolist(),
            'val_per_class_f1': val_metrics['per_class_f1'].tolist(),
            'val_per_class_acc': val_metrics['per_class_acc']
        })
        
        # Log metrics
        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                   f"F1: {val_metrics['f1_weighted']:.4f}")
        logger.info(f"AUC   - Canonical: {val_metrics['auc_scores'][0]:.4f}, "
                   f"Immune: {val_metrics['auc_scores'][1]:.4f}, "
                   f"Stromal: {val_metrics['auc_scores'][2]:.4f}")
        logger.info(f"MCC: {val_metrics['mcc']:.4f}, Uncertainty: {val_metrics['avg_uncertainty']:.4f}")
        
        # Wandb logging
        if args.use_wandb:
            log_dict = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'train_f1': train_f1,
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_f1_weighted': val_metrics['f1_weighted'],
                'val_f1_macro': val_metrics['f1_macro'],
                'val_mcc': val_metrics['mcc'],
                'val_uncertainty': val_metrics['avg_uncertainty'],
                'learning_rate': trainer.optimizer.param_groups[0]['lr']
            }
            
            # Add per-class metrics
            for i, subtype in enumerate(['Canonical', 'Immune', 'Stromal']):
                log_dict[f'val_auc_{subtype}'] = val_metrics['auc_scores'][i]
                log_dict[f'val_f1_{subtype}'] = val_metrics['per_class_f1'][i]
                log_dict[f'val_acc_{subtype}'] = val_metrics['per_class_acc'][i]
            
            wandb.log(log_dict)
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            patience_counter = 0
            
            # Save best checkpoint
            best_model_path = output_dir / 'best_molecular_model.pth'
            trainer.save_checkpoint(best_model_path, epoch, val_metrics)
            
            # Save best model state for calibration
            trainer.best_model_state = model.state_dict().copy()
            
            logger.info(f"✅ New best model! Val Acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch+1}.pth'
            trainer.save_checkpoint(checkpoint_path, epoch, val_metrics)
    
    # Final evaluation on test set
    logger.info("\n" + "="*50)
    logger.info("FINAL TEST SET EVALUATION")
    logger.info("="*50)
    
    # Load best model for testing
    if trainer.best_model_state:
        model.load_state_dict(trainer.best_model_state)
    
    test_metrics = trainer.validate(test_loader)
    
    logger.info(f"Test Results:")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  F1 (Weighted): {test_metrics['f1_weighted']:.4f}")
    logger.info(f"  F1 (Macro): {test_metrics['f1_macro']:.4f}")
    logger.info(f"  MCC: {test_metrics['mcc']:.4f}")
    
    for i, subtype in enumerate(['Canonical', 'Immune', 'Stromal']):
        logger.info(f"  {subtype} - AUC: {test_metrics['auc_scores'][i]:.4f}, "
                   f"F1: {test_metrics['per_class_f1'][i]:.4f}, "
                   f"Acc: {test_metrics['per_class_acc'][i]:.4f}")
    
    # Generate comprehensive test report
    generate_test_report(test_metrics, output_dir)
    
    # Train confidence calibrator
    logger.info("Training confidence calibrator...")
    calibrator = train_confidence_calibrator(model, val_loader, device)
    
    # Save calibrator
    calibrator_path = output_dir / 'confidence_calibrator.pkl'
    calibrator.save(str(calibrator_path))
    
    # Save final model with complete metadata
    final_model_path = output_dir / 'final_molecular_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'best_val_acc': best_val_acc,
        'test_metrics': test_metrics,
        'class_weights': class_weights.tolist(),
        'training_history': trainer.history,
        'training_args': vars(args),
        'timestamp': datetime.now().isoformat()
    }, final_model_path)
    
    # Save training history
    history_path = output_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_json = {}
        for key, value in trainer.history.items():
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], np.ndarray):
                    history_json[key] = [v.tolist() for v in value]
                else:
                    history_json[key] = value
            else:
                history_json[key] = value
        
        json.dump(history_json, f, indent=2, default=str)
    
    logger.info(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Final test accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"All outputs saved to: {output_dir}")
    
    if args.use_wandb:
        # Log final metrics
        wandb.log({
            'final_test_acc': test_metrics['accuracy'],
            'final_test_f1': test_metrics['f1_weighted'],
            'final_test_mcc': test_metrics['mcc'],
            'best_val_acc': best_val_acc
        })
        wandb.finish()
    
    return model, trainer.history, test_metrics

def generate_test_report(test_metrics, output_dir):
    """Generate comprehensive test report with visualizations"""
    
    # Confusion matrix
    cm = confusion_matrix(test_metrics['targets'], test_metrics['predictions'])
    subtype_names = ['Canonical', 'Immune', 'Stromal']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=subtype_names, yticklabels=subtype_names)
    plt.title('Molecular Subtype Classification - Test Set Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_dir / 'test_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Per-class metrics visualization
    metrics_df = pd.DataFrame({
        'Subtype': subtype_names,
        'Accuracy': test_metrics['per_class_acc'],
        'F1-Score': test_metrics['per_class_f1'],
        'AUC': test_metrics['auc_scores']
    })
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, metric in enumerate(['Accuracy', 'F1-Score', 'AUC']):
        axes[i].bar(metrics_df['Subtype'], metrics_df[metric])
        axes[i].set_title(f'Per-Class {metric}')
        axes[i].set_ylabel(metric)
        axes[i].set_ylim(0, 1)
        
        # Add value labels
        for j, v in enumerate(metrics_df[metric]):
            axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_per_class_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save detailed classification report
    report = classification_report(
        test_metrics['targets'], 
        test_metrics['predictions'],
        target_names=subtype_names,
        output_dict=True
    )
    
    with open(output_dir / 'test_classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Test report generated in: {output_dir}")

def train_confidence_calibrator(model, val_loader, device):
    """Train confidence calibrator on validation set"""
    model.eval()
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            all_logits.append(output['logits'].cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets)
    
    # Train calibrator
    calibrator = ConfidenceCalibrator()
    calibrator.fit(logits, targets, method='temperature')
    
    return calibrator

def main():
    parser = argparse.ArgumentParser(description='Train EPOC Molecular Subtype Model')
    parser.add_argument('--manifest', type=str, required=True,
                       help='Path to EPOC manifest CSV file')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing histopathology images')
    parser.add_argument('--output_dir', type=str, default='./epoc_molecular_models',
                       help='Output directory for trained models')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Training batch size')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.1,
                       help='Test split ratio')
    parser.add_argument('--backbone', type=str, default='efficientnet_b3',
                       choices=['efficientnet_b0', 'efficientnet_b3', 'resnet50'],
                       help='Model backbone architecture')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Training device (cuda/cpu)')
    parser.add_argument('--use_wandb', action='store_true',
                       help='Use Weights & Biases for experiment tracking')
    parser.add_argument('--create_dummy_labels', action='store_true',
                       help='Create dummy molecular labels if not available (for testing)')
    
    args = parser.parse_args()
    
    logger.info("Starting EPOC Molecular Subtype Training")
    logger.info(f"Arguments: {vars(args)}")
    
    # Validate inputs
    if not Path(args.manifest).exists():
        raise FileNotFoundError(f"Manifest file not found: {args.manifest}")
    
    if not Path(args.data_dir).exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    # Start training
    model, history, test_metrics = train_epoc_molecular_model(args)
    
    logger.info("Training completed successfully!")
    logger.info(f"Final test accuracy: {test_metrics['accuracy']:.4f}")

if __name__ == "__main__":
    main() 