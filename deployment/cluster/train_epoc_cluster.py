#!/usr/bin/env python3
"""
EPOC Cluster Training Script - Enhanced
Integrates all state-of-the-art enhancements for 96% accuracy target
"""

import os
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import argparse
import wandb
from typing import Dict, List, Tuple
import yaml
from tqdm import tqdm

# Import enhanced components
from models.enhanced_molecular_predictor import EnhancedMolecularPredictor, EnsembleMolecularPredictor
from models.state_of_the_art_molecular_classifier import create_state_of_the_art_model
from enhanced_training_pipeline import EnhancedTrainer, EnhancedDataset, MultiTaskLoss
from scripts.preprocessing.enhanced_preprocessing import EnhancedPreprocessor, create_preprocessing_config
from scripts.augmentation.advanced_histopathology_augmentation import AdvancedHistopathologyAugmentation

class EPOCClusterTrainer:
    """
    Enhanced cluster trainer for EPOC molecular subtype prediction
    Implements all practical enhancements from 96% accuracy roadmap
    """
    
    def __init__(self, config: Dict, rank: int, world_size: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.device = torch.device(f'cuda:{rank}')
        
        # Initialize distributed training
        self.setup_distributed()
        
        # Initialize logging
        self.setup_logging()
        
        # Initialize wandb (only on rank 0)
        if rank == 0 and config.get('use_wandb', True):
            wandb.init(
                project=config.get('project_name', 'epoc-molecular-subtyping'),
                config=config,
                name=f"epoc_enhanced_training_{config.get('experiment_id', 'default')}"
            )
        
        # Load enhanced model
        self.model = self.create_enhanced_model()
        self.model = DDP(self.model, device_ids=[rank])
        
        # Setup training components
        self.setup_training_components()
        
    def setup_distributed(self):
        """Setup distributed training"""
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            self.rank = int(os.environ['RANK'])
            self.world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=self.world_size,
            rank=self.rank
        )
        
        torch.cuda.set_device(self.rank)
        
    def setup_logging(self):
        """Setup logging for distributed training"""
        logging.basicConfig(
            level=logging.INFO,
            format=f'[Rank {self.rank}] %(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'logs/training_rank_{self.rank}.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_enhanced_model(self) -> nn.Module:
        """Create enhanced molecular predictor model"""
        model_type = self.config.get('model_type', 'enhanced')
        
        if model_type == 'enhanced':
            model = EnhancedMolecularPredictor(num_classes=3)
        elif model_type == 'ensemble':
            model_configs = [
                {'type': 'enhanced', 'num_classes': 3},
                {'type': 'standard', 'architecture': 'swin', 'num_classes': 3},
                {'type': 'standard', 'architecture': 'convnext', 'num_classes': 3}
            ]
            model = EnsembleMolecularPredictor(model_configs)
        else:
            model = create_state_of_the_art_model(
                num_classes=3,
                use_uncertainty=True,
                architecture=self.config.get('architecture', 'ensemble')
            )
        
        return model.to(self.device)
    
    def setup_training_components(self):
        """Setup optimizer, scheduler, and loss function"""
        # Enhanced optimizer with layer-wise learning rates
        self.optimizer = torch.optim.AdamW([
            {'params': self.model.module.base_model.parameters(), 
             'lr': self.config.get('backbone_lr', 1e-5)},
            {'params': self.model.module.uncertainty_head.parameters(), 
             'lr': self.config.get('uncertainty_lr', 1e-4)}
        ], weight_decay=self.config.get('weight_decay', 0.01))
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.get('T_0', 10),
            T_mult=self.config.get('T_mult', 2)
        )
        
        # Enhanced loss function
        self.criterion = MultiTaskLoss(
            self.config.get('task_weights', {'molecular_subtype': 1.0})
        )
        
    def load_epoc_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load and preprocess EPOC data"""
        data_path = Path(self.config['data_path'])
        
        # Load manifest with molecular annotations
        manifest_path = data_path / 'epoc_manifest.csv'
        if not manifest_path.exists():
            raise FileNotFoundError(f"EPOC manifest not found: {manifest_path}")
        
        manifest_df = pd.read_csv(manifest_path)
        
        # Validate required columns
        required_cols = ['wsi_path', 'molecular_subtype', 'patient_id']
        missing_cols = [col for col in required_cols if col not in manifest_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in manifest: {missing_cols}")
        
        # Split data
        train_df, val_df = self.split_data(manifest_df)
        
        # Setup preprocessing
        preprocessing_config = create_preprocessing_config()
        preprocessing_config.update(self.config.get('preprocessing', {}))
        
        # Create datasets with enhanced augmentation
        train_dataset = EPOCDataset(
            train_df, 
            preprocessing_config=preprocessing_config,
            mode='train',
            config=self.config
        )
        
        val_dataset = EPOCDataset(
            val_df,
            preprocessing_config=preprocessing_config, 
            mode='val',
            config=self.config
        )
        
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 32),
            sampler=train_sampler,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.get('batch_size', 32),
            sampler=val_sampler,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def split_data(self, manifest_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data ensuring patient-level separation"""
        # Patient-level split to avoid data leakage
        unique_patients = manifest_df['patient_id'].unique()
        np.random.seed(42)
        np.random.shuffle(unique_patients)
        
        val_split = self.config.get('val_split', 0.2)
        val_size = int(len(unique_patients) * val_split)
        
        val_patients = unique_patients[:val_size]
        train_patients = unique_patients[val_size:]
        
        train_df = manifest_df[manifest_df['patient_id'].isin(train_patients)]
        val_df = manifest_df[manifest_df['patient_id'].isin(val_patients)]
        
        self.logger.info(f"Train patients: {len(train_patients)}, Val patients: {len(val_patients)}")
        self.logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
        
        return train_df, val_df
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch with enhanced features"""
        self.model.train()
        train_loader.sampler.set_epoch(epoch)
        
        total_loss = 0
        correct = 0
        total = 0
        
        if self.rank == 0:
            pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
        else:
            pbar = train_loader
        
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass with enhanced model
            outputs = self.model(images)
            
            # Prepare targets for multi-task loss
            targets = {'molecular_subtype': labels}
            
            # Add auxiliary tasks if available
            if 'survival_time' in batch:
                targets['survival_time'] = batch['survival_time'].to(self.device)
                targets['survival_event'] = batch['survival_event'].to(self.device)
            
            if 'mutations' in batch:
                targets['mutations'] = batch['mutations'].to(self.device)
            
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
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.get('grad_clip', 1.0)
            )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            _, predicted = outputs['probabilities'].max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar (rank 0 only)
            if self.rank == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        # Reduce metrics across all processes
        total_loss = self.reduce_metric(total_loss)
        correct = self.reduce_metric(correct)
        total = self.reduce_metric(total)
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate with enhanced metrics"""
        self.model.eval()
        
        total_loss = 0
        correct = 0
        total = 0
        predictions = []
        uncertainties = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation') if self.rank == 0 else val_loader:
                images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Enhanced inference with TTA
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
        
        # Reduce metrics across all processes
        total_loss = self.reduce_metric(total_loss)
        correct = self.reduce_metric(correct)
        total = self.reduce_metric(total)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': 100. * correct / total,
            'avg_uncertainty': np.mean(uncertainties) if uncertainties else 0.0
        }
    
    def reduce_metric(self, metric: float) -> float:
        """Reduce metric across all processes"""
        metric_tensor = torch.tensor(metric).to(self.device)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        return metric_tensor.item() / self.world_size
    
    def train(self):
        """Full training loop"""
        # Load data
        train_loader, val_loader = self.load_epoc_data()
        
        # Training loop
        best_accuracy = 0
        patience_counter = 0
        
        for epoch in range(self.config.get('num_epochs', 100)):
            # Adjust learning rate
            self.scheduler.step()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log metrics (rank 0 only)
            if self.rank == 0:
                if self.config.get('use_wandb', True):
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_metrics['loss'],
                        'train_acc': train_metrics['accuracy'],
                        'val_loss': val_metrics['loss'],
                        'val_acc': val_metrics['accuracy'],
                        'val_uncertainty': val_metrics['avg_uncertainty'],
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })
                
                self.logger.info(
                    f"Epoch {epoch}: Train Acc: {train_metrics['accuracy']:.2f}%, "
                    f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                    f"Val Uncertainty: {val_metrics['avg_uncertainty']:.3f}"
                )
                
                # Save best model
                if val_metrics['accuracy'] > best_accuracy:
                    best_accuracy = val_metrics['accuracy']
                    patience_counter = 0
                    
                    self.save_checkpoint(epoch, val_metrics['accuracy'])
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter > self.config.get('patience', 20):
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
        
        # Cleanup
        self.cleanup()
    
    def save_checkpoint(self, epoch: int, accuracy: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': accuracy,
            'config': self.config
        }
        
        checkpoint_path = f'checkpoints/epoc_enhanced_epoch_{epoch}_acc_{accuracy:.2f}.pth'
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def cleanup(self):
        """Cleanup distributed training"""
        dist.destroy_process_group()


class EPOCDataset(Dataset):
    """Enhanced EPOC dataset with preprocessing and augmentation"""
    
    def __init__(self, manifest_df: pd.DataFrame, preprocessing_config: Dict, 
                 mode: str = 'train', config: Dict = None):
        self.manifest_df = manifest_df
        self.mode = mode
        self.config = config or {}
        
        # Initialize preprocessing
        self.preprocessor = EnhancedPreprocessor(preprocessing_config)
        
        # Initialize augmentation
        if mode == 'train':
            self.augmenter = AdvancedHistopathologyAugmentation(
                augmentation_strength=config.get('augmentation_strength', 0.5)
            )
        else:
            self.augmenter = None
        
        # Label mapping
        self.label_map = {'Canonical': 0, 'Immune': 1, 'Stromal': 2}
    
    def __len__(self):
        return len(self.manifest_df)
    
    def __getitem__(self, idx):
        row = self.manifest_df.iloc[idx]
        
        # Load and preprocess image
        wsi_path = row['wsi_path']
        image = self.preprocessor.preprocess_image(wsi_path)
        
        if image is None:
            # Fallback to dummy image if preprocessing fails
            image = np.zeros((256, 256, 3), dtype=np.uint8)
        
        # Apply augmentation (training only)
        if self.augmenter and self.mode == 'train':
            image = self.augmenter.apply_augmentation(image)
        
        # Convert to tensor
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # Get label
        molecular_subtype = row['molecular_subtype']
        label = self.label_map.get(molecular_subtype, 0)
        
        sample = {
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'patient_id': row['patient_id'],
            'wsi_path': wsi_path
        }
        
        # Add auxiliary data if available
        if 'survival_time' in row:
            sample['survival_time'] = torch.tensor(row['survival_time'], dtype=torch.float)
            sample['survival_event'] = torch.tensor(row['survival_event'], dtype=torch.long)
        
        return sample


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='EPOC Enhanced Cluster Training')
    parser.add_argument('--config', type=str, required=True, help='Config file path')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup distributed training
    if 'WORLD_SIZE' in os.environ:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
    else:
        world_size = 1
        rank = 0
    
    # Initialize trainer
    trainer = EPOCClusterTrainer(config, rank, world_size)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main() 