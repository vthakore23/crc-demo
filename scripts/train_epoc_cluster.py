#!/usr/bin/env python3
"""
EPOC Cohort Training Script for Computing Cluster
Designed for distributed training on HPC systems with SLURM
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
import time
import shutil

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms, models

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
import seaborn as sns


class EPOCDataset(Dataset):
    """
    Dataset for EPOC cohort with proper support for WSI tiles
    Expected data structure:
    - CSV manifest with patient info and molecular subtypes
    - Pre-extracted tile directories per patient
    """
    
    def __init__(self, manifest_csv, tile_dir, transform=None, tiles_per_patient=50):
        """
        Args:
            manifest_csv: Path to CSV with columns:
                - patient_id
                - molecular_subtype (canonical, immune, stromal)
                - clinical_data (optional)
            tile_dir: Directory containing patient subdirectories with tiles
            transform: Image transformations
            tiles_per_patient: Number of tiles to sample per patient
        """
        self.manifest = pd.read_csv(manifest_csv)
        self.tile_dir = Path(tile_dir)
        self.transform = transform
        self.tiles_per_patient = tiles_per_patient
        
        # Map molecular subtypes to class indices
        self.subtype_map = {
            'canonical': 0, 'immune': 1, 'stromal': 2
        }
        
        # Build tile index
        self.tile_index = []
        for _, row in self.manifest.iterrows():
            patient_id = row['patient_id']
            subtype = row['molecular_subtype']
            patient_tile_dir = self.tile_dir / str(patient_id)
            
            if patient_tile_dir.exists():
                tiles = list(patient_tile_dir.glob('*.png')) + list(patient_tile_dir.glob('*.jpg'))
                # Sample tiles if too many
                if len(tiles) > self.tiles_per_patient:
                    tiles = np.random.choice(tiles, self.tiles_per_patient, replace=False)
                
                for tile_path in tiles:
                    self.tile_index.append({
                        'path': tile_path,
                        'label': self.subtype_map.get(subtype, 0),
                        'patient_id': patient_id,
                        'subtype': subtype
                    })
        
        print(f"Dataset created with {len(self.tile_index)} tiles from {len(self.manifest)} patients")
        
    def __len__(self):
        return len(self.tile_index)
    
    def __getitem__(self, idx):
        tile_info = self.tile_index[idx]
        
        # Load image
        from PIL import Image
        image = Image.open(tile_info['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, tile_info['label'], tile_info['patient_id']


class EPOCModel(nn.Module):
    """
    Enhanced model for EPOC molecular subtype classification
    Supports multiple backbones and includes uncertainty estimation
    """
    
    def __init__(self, num_classes=4, backbone='efficientnet_b0', pretrained=True, 
                 use_attention=True, dropout_rate=0.5):
        super().__init__()
        
        # Select backbone
        if backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif backbone == 'densenet121':
            self.backbone = models.densenet121(pretrained=pretrained)
            num_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8)
        )
        
        # Attention mechanism
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Linear(128, 256),
                nn.Sigmoid()
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, num_classes)
        )
        
        # Uncertainty estimation
        self.uncertainty_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        features = self.feature_processor(features)
        
        # Apply attention
        if self.use_attention:
            attention_weights = self.attention(features)
            features = features * attention_weights
        
        # Get predictions
        logits = self.classifier(features)
        uncertainty = self.uncertainty_head(features)
        
        return logits, uncertainty, features


class DistributedTrainer:
    """
    Distributed trainer for multi-GPU/multi-node training on cluster
    """
    
    def __init__(self, rank, world_size, args):
        self.rank = rank
        self.world_size = world_size
        self.args = args
        self.device = torch.device(f'cuda:{rank}')
        
        # Setup distributed training
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # Create model
        self.model = EPOCModel(
            num_classes=args.num_classes,
            backbone=args.backbone,
            pretrained=args.pretrained,
            use_attention=args.use_attention,
            dropout_rate=args.dropout_rate
        ).to(self.device)
        
        # Wrap with DDP
        self.model = DDP(self.model, device_ids=[rank])
        
        # Setup logging
        if rank == 0:
            self.setup_logging()
            self.writer = SummaryWriter(self.args.log_dir / 'tensorboard')
        
    def setup_logging(self):
        """Setup logging for rank 0 process"""
        log_file = self.args.log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def train(self):
        """Main training loop"""
        # Create data loaders
        train_dataset = EPOCDataset(
            self.args.train_manifest,
            self.args.tile_dir,
            transform=self.get_train_transform(),
            tiles_per_patient=self.args.tiles_per_patient
        )
        
        val_dataset = EPOCDataset(
            self.args.val_manifest,
            self.args.tile_dir,
            transform=self.get_val_transform(),
            tiles_per_patient=self.args.tiles_per_patient
        )
        
        # Distributed samplers
        train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank)
        val_sampler = DistributedSampler(val_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.args.batch_size,
            sampler=train_sampler,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.args.batch_size,
            sampler=val_sampler,
            num_workers=self.args.num_workers,
            pin_memory=True
        )
        
        # Optimizer and scheduler
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay
        )
        
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.args.learning_rate * 10,
            epochs=self.args.epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # Loss function
        criterion = nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing)
        
        # Training loop
        best_val_auc = 0
        patience_counter = 0
        
        for epoch in range(self.args.epochs):
            train_sampler.set_epoch(epoch)  # Important for proper shuffling
            
            # Train
            train_metrics = self.train_epoch(
                train_loader, criterion, optimizer, scheduler, epoch
            )
            
            # Validate
            val_metrics = self.validate_epoch(val_loader, criterion, epoch)
            
            # Log metrics
            if self.rank == 0:
                self.log_metrics(train_metrics, val_metrics, epoch)
                
                # Save checkpoint
                if val_metrics['auc'] > best_val_auc:
                    best_val_auc = val_metrics['auc']
                    patience_counter = 0
                    self.save_checkpoint(epoch, val_metrics)
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= self.args.patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
        
        # Cleanup
        if self.rank == 0:
            self.writer.close()
        dist.destroy_process_group()
    
    def train_epoch(self, loader, criterion, optimizer, scheduler, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(loader, desc=f"[Rank {self.rank}] Epoch {epoch} Train") if self.rank == 0 else loader
        
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            logits, uncertainty, _ = self.model(images)
            
            # Calculate loss
            cls_loss = criterion(logits, labels)
            uncertainty_loss = nn.MSELoss()(
                uncertainty.squeeze(),
                (logits.max(1)[0] == labels).float()
            )
            loss = cls_loss + 0.1 * uncertainty_loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
            
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            _, preds = logits.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            if self.rank == 0 and batch_idx % 10 == 0:
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
        
        # Calculate metrics
        metrics = {
            'loss': total_loss / len(loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='macro')
        }
        
        return metrics
    
    def validate_epoch(self, loader, criterion, epoch):
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(loader, desc=f"[Rank {self.rank}] Epoch {epoch} Val") if self.rank == 0 else loader
            
            for images, labels, _ in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                logits, uncertainty, _ = self.model(images)
                
                # Calculate loss
                loss = criterion(logits, labels)
                total_loss += loss.item()
                
                # Get predictions
                probs = torch.softmax(logits, dim=1)
                _, preds = logits.max(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        metrics = {
            'loss': total_loss / len(loader),
            'accuracy': accuracy_score(all_labels, all_preds),
            'f1': f1_score(all_labels, all_preds, average='macro'),
            'auc': roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        }
        
        return metrics
    
    def save_checkpoint(self, epoch, metrics):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),  # Unwrap DDP
            'metrics': metrics,
            'args': vars(self.args),
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = self.args.checkpoint_dir / f'best_model_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as 'best_model.pth'
        shutil.copy(checkpoint_path, self.args.checkpoint_dir / 'best_model.pth')
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def log_metrics(self, train_metrics, val_metrics, epoch):
        """Log metrics to tensorboard and console"""
        # Console logging
        self.logger.info(f"\nEpoch {epoch} Summary:")
        self.logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1']:.4f}")
        self.logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1']:.4f}, AUC: {val_metrics['auc']:.4f}")
        
        # Tensorboard logging
        self.writer.add_scalars('Loss', {
            'train': train_metrics['loss'],
            'val': val_metrics['loss']
        }, epoch)
        
        self.writer.add_scalars('Accuracy', {
            'train': train_metrics['accuracy'],
            'val': val_metrics['accuracy']
        }, epoch)
        
        self.writer.add_scalar('Val/AUC', val_metrics['auc'], epoch)
        self.writer.add_scalar('Val/F1', val_metrics['f1'], epoch)
    
    def get_train_transform(self):
        """Get training data transforms"""
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_val_transform(self):
        """Get validation data transforms"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def generate_slurm_script(args):
    """Generate SLURM job submission script"""
    script_content = f"""#!/bin/bash
#SBATCH --job-name=epoc_training
#SBATCH --partition={args.partition}
#SBATCH --nodes={args.nodes}
#SBATCH --ntasks-per-node={args.gpus_per_node}
#SBATCH --cpus-per-task={args.cpus_per_gpu}
#SBATCH --mem={args.memory}
#SBATCH --time={args.time_limit}
#SBATCH --output={args.log_dir}/slurm_%j.out
#SBATCH --error={args.log_dir}/slurm_%j.err
#SBATCH --gres=gpu:{args.gpus_per_node}

# Load modules
module load cuda/11.8
module load python/3.9
module load openmpi/4.1.4

# Activate virtual environment
source {args.venv_path}/bin/activate

# Set environment variables
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS

# Run training
srun python {Path(__file__).absolute()} \\
    --train-manifest {args.train_manifest} \\
    --val-manifest {args.val_manifest} \\
    --tile-dir {args.tile_dir} \\
    --checkpoint-dir {args.checkpoint_dir} \\
    --log-dir {args.log_dir} \\
    --epochs {args.epochs} \\
    --batch-size {args.batch_size} \\
    --learning-rate {args.learning_rate} \\
    --num-workers {args.num_workers} \\
    --distributed
"""
    
    script_path = args.output_dir / 'train_epoc.sbatch'
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    print(f"SLURM script generated: {script_path}")
    print(f"Submit with: sbatch {script_path}")
    
    return script_path


def main():
    parser = argparse.ArgumentParser(description='EPOC Cohort Training on Computing Cluster')
    
    # Data arguments
    parser.add_argument('--train-manifest', type=str, required=True,
                        help='Path to training manifest CSV')
    parser.add_argument('--val-manifest', type=str, required=True,
                        help='Path to validation manifest CSV')
    parser.add_argument('--tile-dir', type=str, required=True,
                        help='Directory containing patient tile subdirectories')
    parser.add_argument('--tiles-per-patient', type=int, default=50,
                        help='Number of tiles to sample per patient')
    
    # Model arguments
    parser.add_argument('--num-classes', type=int, default=4,
                        help='Number of molecular subtypes')
    parser.add_argument('--backbone', type=str, default='efficientnet_b0',
                        choices=['efficientnet_b0', 'resnet50', 'densenet121'])
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained backbone')
    parser.add_argument('--use-attention', action='store_true', default=True,
                        help='Use attention mechanism')
    parser.add_argument('--dropout-rate', type=float, default=0.5,
                        help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size per GPU')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Base learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing factor')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping value')
    parser.add_argument('--patience', type=int, default=10,
                        help='Early stopping patience')
    
    # System arguments
    parser.add_argument('--checkpoint-dir', type=Path, required=True,
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=Path, required=True,
                        help='Directory for logs')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers per GPU')
    parser.add_argument('--distributed', action='store_true',
                        help='Use distributed training')
    
    # SLURM arguments
    parser.add_argument('--generate-slurm', action='store_true',
                        help='Generate SLURM submission script')
    parser.add_argument('--partition', type=str, default='gpu',
                        help='SLURM partition')
    parser.add_argument('--nodes', type=int, default=1,
                        help='Number of nodes')
    parser.add_argument('--gpus-per-node', type=int, default=4,
                        help='GPUs per node')
    parser.add_argument('--cpus-per-gpu', type=int, default=8,
                        help='CPUs per GPU')
    parser.add_argument('--memory', type=str, default='32GB',
                        help='Memory per node')
    parser.add_argument('--time-limit', type=str, default='24:00:00',
                        help='Time limit (HH:MM:SS)')
    parser.add_argument('--venv-path', type=Path, default='~/.conda/envs/epoc',
                        help='Path to virtual environment')
    parser.add_argument('--output-dir', type=Path, default='.',
                        help='Output directory for generated scripts')
    
    args = parser.parse_args()
    
    # Create directories
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    
    if args.generate_slurm:
        # Generate SLURM script
        generate_slurm_script(args)
    else:
        # Run training
        if args.distributed:
            # Get distributed training info from environment
            rank = int(os.environ.get('SLURM_PROCID', 0))
            world_size = int(os.environ.get('SLURM_NTASKS', 1))
            
            # Initialize trainer and start training
            trainer = DistributedTrainer(rank, world_size, args)
            trainer.train()
        else:
            # Single GPU training
            trainer = DistributedTrainer(0, 1, args)
            trainer.train()


if __name__ == '__main__':
    main() 