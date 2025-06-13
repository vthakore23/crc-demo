"""
Distributed Training Infrastructure for CRC Molecular Subtype Classification
Supports multi-GPU and multi-node training on computing clusters
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

import wandb
from tqdm import tqdm

from models.cluster_ready_model import ClusterReadyMolecularModel
from data.wsi_dataset import EPOCDataset
from utils.checkpoint_manager import CheckpointManager
from utils.metrics import ClinicalMetrics

logger = logging.getLogger(__name__)


class DistributedTrainer:
    """Handles distributed training across multiple GPUs/nodes"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = int(os.environ.get('WORLD_SIZE', 1))
        
        # Initialize process group
        self.setup_distributed()
        
        # Set device
        self.device = torch.device(f'cuda:{self.local_rank}')
        torch.cuda.set_device(self.device)
        
        # Initialize model
        self.model = self._build_model()
        
        # Initialize optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Mixed precision training
        self.scaler = GradScaler() if config['use_amp'] else None
        
        # Metrics and logging
        self.metrics = ClinicalMetrics(config)
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config['checkpoint_dir'],
            keep_last_n=config['keep_checkpoints']
        )
        
        # Initialize wandb on rank 0
        if self.rank == 0 and config['use_wandb']:
            wandb.init(
                project=config['wandb_project'],
                name=config['experiment_name'],
                config=config
            )
    
    def setup_distributed(self):
        """Initialize distributed process group"""
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '29500'
        
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=self.world_size,
            rank=self.rank
        )
        
        logger.info(f"Initialized process group: rank={self.rank}, world_size={self.world_size}")
    
    def _build_model(self) -> nn.Module:
        """Build and wrap model for distributed training"""
        model = ClusterReadyMolecularModel(self.config['model'])
        
        # Load pretrained weights if available
        if self.config.get('pretrained_path'):
            state_dict = torch.load(
                self.config['pretrained_path'],
                map_location='cpu'
            )
            model.load_state_dict(state_dict['model_state_dict'])
            logger.info(f"Loaded pretrained model from {self.config['pretrained_path']}")
        
        # Move to device and wrap with DDP
        model = model.to(self.device)
        model = DDP(
            model,
            device_ids=[self.local_rank],
            output_device=self.local_rank,
            find_unused_parameters=self.config.get('find_unused_parameters', False)
        )
        
        return model
    
    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer with parameter groups"""
        # Separate parameters for different learning rates
        backbone_params = []
        head_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        param_groups = [
            {'params': backbone_params, 'lr': self.config['backbone_lr']},
            {'params': head_params, 'lr': self.config['head_lr']}
        ]
        
        if self.config['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.config['weight_decay']
            )
        elif self.config['optimizer'] == 'sgd':
            optimizer = torch.optim.SGD(
                param_groups,
                momentum=self.config['momentum'],
                weight_decay=self.config['weight_decay']
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config['optimizer']}")
        
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler"""
        if self.config['scheduler'] == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['epochs'],
                eta_min=self.config['min_lr']
            )
        elif self.config['scheduler'] == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['step_size'],
                gamma=self.config['gamma']
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        best_val_metric = 0.0
        
        for epoch in range(self.config['epochs']):
            # Training
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validation
            if (epoch + 1) % self.config['val_frequency'] == 0:
                val_metrics = self.validate(val_loader)
                
                # Log metrics
                if self.rank == 0:
                    self.log_metrics(train_metrics, val_metrics, epoch)
                    
                    # Save checkpoint if best
                    if val_metrics['accuracy'] > best_val_metric:
                        best_val_metric = val_metrics['accuracy']
                        self.save_checkpoint(
                            epoch,
                            val_metrics,
                            is_best=True
                        )
            
            # Regular checkpoint
            if (epoch + 1) % self.config['checkpoint_frequency'] == 0:
                self.save_checkpoint(epoch, train_metrics, is_best=False)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
        
        # Cleanup
        self.cleanup()
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        predictions = []
        targets = []
        
        # Progress bar only on rank 0
        if self.rank == 0:
            pbar = tqdm(total=len(train_loader), desc=f'Epoch {epoch}')
        
        for batch_idx, (patches, labels, metadata) in enumerate(train_loader):
            # Move to device
            patches = patches.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass with mixed precision
            if self.scaler is not None:
                with autocast():
                    outputs = self.model(patches, metadata)
                    loss = self.compute_loss(outputs, labels)
            else:
                outputs = self.model(patches, metadata)
                loss = self.compute_loss(outputs, labels)
            
            # Gradient accumulation
            loss = loss / self.config['accumulation_steps']
            
            # Backward pass
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Optimizer step
            if (batch_idx + 1) % self.config['accumulation_steps'] == 0:
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
            
            # Track metrics
            total_loss += loss.item() * self.config['accumulation_steps']
            predictions.extend(outputs['predictions'].cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
            # Update progress bar
            if self.rank == 0:
                pbar.update(1)
                pbar.set_postfix({'loss': loss.item()})
        
        if self.rank == 0:
            pbar.close()
        
        # Gather metrics across all processes
        metrics = self.metrics.compute_epoch_metrics(
            predictions=np.array(predictions),
            targets=np.array(targets),
            loss=total_loss / len(train_loader)
        )
        
        return self.gather_metrics(metrics)
    
    def validate(self, val_loader: DataLoader) -> Dict:
        """Validation loop"""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        all_uncertainties = []
        
        with torch.no_grad():
            for patches, labels, metadata in val_loader:
                patches = patches.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(patches, metadata)
                loss = self.compute_loss(outputs, labels)
                
                # Track metrics
                total_loss += loss.item()
                all_predictions.extend(outputs['predictions'].cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
                all_uncertainties.extend(outputs['uncertainty'].cpu().numpy())
        
        # Compute comprehensive metrics
        metrics = self.metrics.compute_validation_metrics(
            predictions=np.array(all_predictions),
            targets=np.array(all_targets),
            uncertainties=np.array(all_uncertainties),
            loss=total_loss / len(val_loader)
        )
        
        return self.gather_metrics(metrics)
    
    def compute_loss(self, outputs: Dict, labels: torch.Tensor) -> torch.Tensor:
        """Compute multi-task loss"""
        # Classification loss
        ce_loss = nn.CrossEntropyLoss()(outputs['logits'], labels)
        
        # Uncertainty loss (evidential)
        evidence = outputs['evidence']
        uncertainty_loss = self.evidential_loss(evidence, labels)
        
        # Pathway-specific losses
        pathway_losses = []
        for pathway in ['canonical', 'immune', 'stromal']:
            if f'{pathway}_score' in outputs:
                pathway_target = (labels == self.config['pathway_indices'][pathway]).float()
                pathway_loss = nn.BCEWithLogitsLoss()(
                    outputs[f'{pathway}_score'],
                    pathway_target
                )
                pathway_losses.append(pathway_loss)
        
        # Combine losses
        total_loss = (
            self.config['ce_weight'] * ce_loss +
            self.config['uncertainty_weight'] * uncertainty_loss +
            self.config['pathway_weight'] * sum(pathway_losses)
        )
        
        return total_loss
    
    def evidential_loss(self, evidence: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Evidential deep learning loss for uncertainty estimation"""
        # Convert evidence to Dirichlet parameters
        alpha = evidence + 1
        
        # Compute expected probability
        S = torch.sum(alpha, dim=1, keepdim=True)
        p = alpha / S
        
        # One-hot encode labels
        y_one_hot = torch.nn.functional.one_hot(labels, num_classes=3).float()
        
        # MSE loss
        mse_loss = torch.sum((y_one_hot - p) ** 2, dim=1)
        
        # KL divergence regularization
        alpha_tilde = y_one_hot + (1 - y_one_hot) * alpha
        kl_loss = self.kl_divergence(alpha, alpha_tilde)
        
        return torch.mean(mse_loss + self.config['kl_weight'] * kl_loss)
    
    def gather_metrics(self, metrics: Dict) -> Dict:
        """Gather metrics across all processes"""
        if self.world_size == 1:
            return metrics
        
        gathered_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                tensor = torch.tensor(value, device=self.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
                gathered_metrics[key] = tensor.item()
            else:
                gathered_metrics[key] = value
        
        return gathered_metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict, is_best: bool = False):
        """Save model checkpoint"""
        if self.rank != 0:
            return
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config
        }
        
        if is_best:
            path = self.checkpoint_manager.save_best(checkpoint)
            logger.info(f"Saved best checkpoint to {path}")
        else:
            path = self.checkpoint_manager.save_regular(checkpoint, epoch)
            logger.info(f"Saved checkpoint to {path}")
    
    def log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log metrics to wandb and console"""
        if self.rank != 0:
            return
        
        # Console logging
        logger.info(f"\nEpoch {epoch} Results:")
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        logger.info(f"Per-class F1: {val_metrics['per_class_f1']}")
        
        # Wandb logging
        if wandb.run is not None:
            wandb.log({
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/canonical_f1': val_metrics['per_class_f1']['canonical'],
                'val/immune_f1': val_metrics['per_class_f1']['immune'],
                'val/stromal_f1': val_metrics['per_class_f1']['stromal'],
                'val/uncertainty_quality': val_metrics['uncertainty_quality'],
                'learning_rate': self.optimizer.param_groups[0]['lr']
            })
    
    def cleanup(self):
        """Cleanup distributed training"""
        dist.destroy_process_group()
        if wandb.run is not None:
            wandb.finish()


def launch_distributed_training(config_path: str):
    """Launch distributed training across multiple processes"""
    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set environment variables
    os.environ['CUDA_VISIBLE_DEVICES'] = config['gpus']
    
    # Launch processes
    world_size = len(config['gpus'].split(','))
    mp.spawn(
        train_worker,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )


def train_worker(rank: int, world_size: int, config: Dict):
    """Worker function for each process"""
    # Set process-specific environment variables
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # Initialize trainer
    trainer = DistributedTrainer(config)
    
    # Create data loaders
    train_dataset = EPOCDataset(
        manifest_path=config['train_manifest'],
        transform_config=config['train_transforms']
    )
    val_dataset = EPOCDataset(
        manifest_path=config['val_manifest'],
        transform_config=config['val_transforms']
    )
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=False
    )
    
    # Start training
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    launch_distributed_training(args.config) 