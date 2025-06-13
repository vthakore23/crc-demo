#!/usr/bin/env python3
"""
Enhanced Training Pipeline
Implements practical training enhancements for improved accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
import wandb
from typing import Dict, List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import json

class EnhancedDataset(Dataset):
    """Enhanced dataset with advanced augmentation"""
    
    def __init__(self, 
                 image_paths: List[Path],
                 labels: List[int],
                 transform=None,
                 mode: str = 'train'):
        self.image_paths = image_paths
        self.labels = labels
        self.mode = mode
        self.transform = transform or self._get_default_transform()
        
    def _get_default_transform(self):
        """Get default augmentation pipeline"""
        if self.mode == 'train':
            return A.Compose([
                # Spatial augmentations
                A.RandomResizedCrop(192, 192, scale=(0.8, 1.2)),
                A.Rotate(limit=90, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.3),
                
                # Color augmentations
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.GaussianBlur(blur_limit=(3, 7), p=0.2),
                
                # Stain augmentation (simulate H&E variations)
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
                
                # Normalization
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(192, 192),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentations
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        label = self.labels[idx]
        
        return {
            'image': image,
            'label': label,
            'path': str(image_path)
        }


class MultiTaskLoss(nn.Module):
    """Multi-task loss for auxiliary tasks"""
    
    def __init__(self, task_weights: Dict[str, float]):
        super().__init__()
        self.task_weights = task_weights
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute multi-task loss"""
        total_loss = 0
        task_losses = {}
        
        # Main task: molecular subtype classification
        if 'molecular_subtype' in predictions:
            ce_loss = F.cross_entropy(predictions['molecular_subtype'], targets['molecular_subtype'])
            focal_loss = self.focal_loss(predictions['molecular_subtype'], targets['molecular_subtype'])
            subtype_loss = 0.7 * ce_loss + 0.3 * focal_loss
            
            total_loss += self.task_weights.get('molecular_subtype', 1.0) * subtype_loss
            task_losses['molecular_subtype'] = subtype_loss
        
        # Auxiliary task: survival prediction (if available)
        if 'survival' in predictions and 'survival' in targets:
            survival_loss = self.cox_loss(predictions['survival'], targets['survival_time'], targets['survival_event'])
            total_loss += self.task_weights.get('survival', 0.2) * survival_loss
            task_losses['survival'] = survival_loss
        
        # Auxiliary task: mutation prediction (if available)
        if 'mutations' in predictions and 'mutations' in targets:
            mutation_loss = F.binary_cross_entropy_with_logits(predictions['mutations'], targets['mutations'])
            total_loss += self.task_weights.get('mutations', 0.2) * mutation_loss
            task_losses['mutations'] = mutation_loss
        
        return total_loss, task_losses
    
    def focal_loss(self, inputs, targets, alpha=0.25, gamma=2):
        """Focal loss for handling class imbalance"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def cox_loss(self, risk_scores, survival_time, event):
        """Cox proportional hazards loss"""
        # Simplified Cox loss implementation
        sorted_idx = torch.argsort(survival_time, descending=True)
        sorted_risk = risk_scores[sorted_idx]
        sorted_event = event[sorted_idx]
        
        # Compute partial likelihood
        exp_risk = torch.exp(sorted_risk)
        risk_sum = torch.cumsum(exp_risk, dim=0)
        
        # Only consider events
        event_risk = sorted_risk[sorted_event == 1]
        event_risk_sum = torch.log(risk_sum[sorted_event == 1])
        
        loss = -torch.mean(event_risk - event_risk_sum)
        return loss


class EnhancedTrainer:
    """Enhanced training with curriculum learning and active learning"""
    
    def __init__(self, 
                 model: nn.Module,
                 config: Dict,
                 device: str = 'cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Initialize optimizer with different learning rates
        self.optimizer = self._create_optimizer()
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=config.get('T_0', 10),
            T_mult=config.get('T_mult', 2)
        )
        
        # Loss function
        self.criterion = MultiTaskLoss(config.get('task_weights', {'molecular_subtype': 1.0}))
        
        # Initialize wandb
        if config.get('use_wandb', True):
            wandb.init(project=config.get('project_name', 'crc-molecular-subtyping'),
                      config=config)
        
        self.best_accuracy = 0
        self.patience_counter = 0
        
    def _create_optimizer(self):
        """Create optimizer with layer-wise learning rates"""
        # Different learning rates for different parts
        params = [
            {'params': self.model.base_model.backbone.parameters(), 
             'lr': self.config.get('backbone_lr', 1e-5)},
            {'params': self.model.base_model.classifier.parameters(), 
             'lr': self.config.get('classifier_lr', 1e-4)},
            {'params': self.model.uncertainty_head.parameters(), 
             'lr': self.config.get('uncertainty_lr', 1e-4)}
        ]
        
        return optim.AdamW(params, weight_decay=self.config.get('weight_decay', 0.01))
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch in pbar:
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Prepare targets
            targets = {'molecular_subtype': labels}
            
            # Compute loss
            loss, task_losses = self.criterion({'molecular_subtype': outputs['logits']}, targets)
            
            # Add uncertainty regularization
            if 'epistemic_uncertainty' in outputs:
                uncertainty_reg = outputs['epistemic_uncertainty'].mean()
                loss += self.config.get('uncertainty_weight', 0.1) * uncertainty_reg
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                         self.config.get('grad_clip', 1.0))
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs['probabilities'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        predictions = []
        uncertainties = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                
                # Compute loss
                targets = {'molecular_subtype': labels}
                loss, _ = self.criterion({'molecular_subtype': outputs['logits']}, targets)
                
                # Update metrics
                total_loss += loss.item()
                _, predicted = outputs['probabilities'].max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                predictions.extend(predicted.cpu().numpy())
                if 'total_uncertainty' in outputs:
                    uncertainties.extend(outputs['total_uncertainty'].cpu().numpy())
        
        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': 100. * correct / total
        }
        
        # Compute per-class metrics
        from sklearn.metrics import classification_report
        true_labels = []
        for batch in val_loader:
            true_labels.extend(batch['label'].numpy())
        
        report = classification_report(true_labels, predictions, output_dict=True)
        
        return metrics, report
    
    def curriculum_learning_schedule(self, epoch: int) -> float:
        """Curriculum learning: start with easy samples"""
        if epoch < self.config.get('warmup_epochs', 5):
            # Start with high confidence samples only
            return 0.3  # Use top 30% confident samples
        elif epoch < self.config.get('warmup_epochs', 5) * 2:
            return 0.6  # Use top 60% confident samples
        else:
            return 1.0  # Use all samples
    
    def active_learning_sample(self, unlabeled_loader: DataLoader, budget: int = 100):
        """Select most uncertain samples for labeling"""
        self.model.eval()
        
        uncertainties = []
        sample_indices = []
        
        with torch.no_grad():
            for idx, batch in enumerate(unlabeled_loader):
                images = batch['image'].to(self.device)
                outputs = self.model(images)
                
                if 'total_uncertainty' in outputs:
                    uncertainty = outputs['total_uncertainty']
                else:
                    # Use entropy as uncertainty
                    probs = outputs['probabilities']
                    uncertainty = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
                
                uncertainties.extend(uncertainty.cpu().numpy())
                sample_indices.extend(range(idx * unlabeled_loader.batch_size, 
                                          (idx + 1) * unlabeled_loader.batch_size))
        
        # Select top uncertain samples
        uncertainties = np.array(uncertainties)
        top_uncertain_idx = np.argsort(uncertainties)[-budget:]
        
        return [sample_indices[i] for i in top_uncertain_idx]
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int):
        """Full training loop"""
        
        for epoch in range(num_epochs):
            # Adjust learning rate
            self.scheduler.step()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics, val_report = self.validate(val_loader)
            
            # Log metrics
            if self.config.get('use_wandb', True):
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_acc': train_metrics['accuracy'],
                    'val_loss': val_metrics['loss'],
                    'val_acc': val_metrics['accuracy'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
            
            # Save best model
            if val_metrics['accuracy'] > self.best_accuracy:
                self.best_accuracy = val_metrics['accuracy']
                self.patience_counter = 0
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_accuracy': self.best_accuracy,
                    'config': self.config
                }, f'models/enhanced_molecular_predictor_best.pth')
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter > self.config.get('patience', 20):
                print(f"Early stopping at epoch {epoch}")
                break
            
            print(f"Epoch {epoch}: Train Acc: {train_metrics['accuracy']:.2f}%, "
                  f"Val Acc: {val_metrics['accuracy']:.2f}%")


def create_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create enhanced data loaders"""
    # Load data paths and labels
    data_df = pd.read_csv(config['data_csv'])
    
    # Stratified K-fold for better validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(data_df['path'], data_df['label'])):
        if fold == config.get('fold', 0):
            train_df = data_df.iloc[train_idx]
            val_df = data_df.iloc[val_idx]
            break
    
    # Create datasets
    train_dataset = EnhancedDataset(
        image_paths=[Path(p) for p in train_df['path']],
        labels=train_df['label'].tolist(),
        mode='train'
    )
    
    val_dataset = EnhancedDataset(
        image_paths=[Path(p) for p in val_df['path']],
        labels=val_df['label'].tolist(),
        mode='val'
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=True,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 4),
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Configuration
    config = {
        'data_csv': 'data/molecular_subtype_labels.csv',
        'batch_size': 32,
        'num_epochs': 100,
        'backbone_lr': 1e-5,
        'classifier_lr': 1e-4,
        'uncertainty_lr': 1e-4,
        'weight_decay': 0.01,
        'T_0': 10,
        'T_mult': 2,
        'patience': 20,
        'grad_clip': 1.0,
        'uncertainty_weight': 0.1,
        'task_weights': {
            'molecular_subtype': 1.0,
            'survival': 0.2,
            'mutations': 0.2
        },
        'use_wandb': True,
        'project_name': 'crc-molecular-enhanced'
    }
    
    # Create model
    from models.enhanced_molecular_predictor import create_enhanced_predictor
    model = create_enhanced_predictor()
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(config)
    
    # Create trainer
    trainer = EnhancedTrainer(model, config)
    
    # Train
    trainer.train(train_loader, val_loader, config['num_epochs']) 