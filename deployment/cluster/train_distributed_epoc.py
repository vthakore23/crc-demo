#!/usr/bin/env python3
"""
Distributed Training Pipeline for CRC Molecular Subtype Classification
EPOC Trial Validation

Features:
- Multi-node multi-GPU training with PyTorch DDP
- Dynamic batch sizing and gradient accumulation
- Mixed precision training with fault-tolerant checkpointing
- Distributed WSI processing and monitoring
"""

import os
import sys
import json
import yaml
import time
import logging
import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast
from torch.nn import SyncBatchNorm
import torch.profiler

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
import ray
from accelerate import Accelerator
from accelerate.utils import set_seed
import psutil
import nvidia_ml_py3 as nvml

# Import custom modules
sys.path.append(str(Path(__file__).parent.parent.parent))
from src.models.distributed_wrapper import DistributedModelWrapper
from src.data.wsi_dataset_distributed import DistributedWSIDataset
from src.utils.checkpoint_manager import FaultTolerantCheckpointManager
from src.utils.monitoring import ClusterMonitor
from src.utils.memory_manager import GPUMemoryManager
from src.validation.epoc_validator import EPOCValidator


@dataclass
class DistributedTrainingConfig:
    """Configuration for distributed training"""
    # Model configuration
    model_name: str = "resnet50_attention_mil"
    num_classes: int = 3
    input_size: Tuple[int, int] = (224, 224)
    patch_size: int = 256
    magnifications: List[int] = [10, 20, 40]
    
    # Training configuration
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_val: float = 1.0
    accumulation_steps: int = 1
    warmup_epochs: int = 5
    
    # Distributed configuration
    backend: str = "nccl"
    find_unused_parameters: bool = False
    sync_batchnorm: bool = True
    gradient_as_bucket_view: bool = True
    static_graph: bool = False
    
    # Mixed precision
    use_amp: bool = True
    amp_dtype: str = "float16"  # float16, bfloat16
    gradient_clipping: bool = True
    
    # Data configuration
    num_workers: int = 8
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    
    # WSI processing
    tile_size: int = 512
    overlap: int = 0
    tissue_threshold: float = 0.1
    background_threshold: float = 0.9
    stain_normalize: bool = True
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine_warmup"
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    # Monitoring
    log_every_n_steps: int = 50
    val_every_n_epochs: int = 1
    save_every_n_epochs: int = 5
    profiler_enabled: bool = False
    
    # Paths
    data_root: str = "/data/epoc_wsi"
    output_dir: str = "/results/distributed_training"
    checkpoint_dir: str = "/checkpoints"
    log_dir: str = "/logs"
    
    # Fault tolerance
    max_retries: int = 3
    checkpoint_every_n_batches: int = 1000
    health_check_interval: int = 100


class DistributedTrainer:
    """Distributed trainer for WSI analysis"""
    
    def __init__(self, config: DistributedTrainingConfig):
        self.config = config
        self.setup_logging()
        self.setup_distributed()
        self.setup_monitoring()
        
        # Initialize components
        self.accelerator = Accelerator(
            mixed_precision=config.amp_dtype if config.use_amp else "no",
            gradient_accumulation_steps=config.accumulation_steps,
            log_with="wandb" if self.is_main_process else None,
            logging_dir=config.log_dir
        )
        
        self.device = self.accelerator.device
        self.setup_model()
        self.setup_optimizer()
        self.setup_data()
        
        # Advanced components
        self.memory_manager = GPUMemoryManager()
        self.checkpoint_manager = FaultTolerantCheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            max_checkpoints=10
        )
        self.epoc_validator = EPOCValidator(config)
        
        # Monitoring
        self.metrics_history = []
        self.training_start_time = None
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_level = logging.INFO if self.is_main_process else logging.WARNING
        logging.basicConfig(
            level=log_level,
            format=f'[Rank {self.get_rank()}] %(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(
                    Path(self.config.log_dir) / f"training_rank_{self.get_rank()}.log"
                )
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def setup_distributed(self):
        """Setup distributed training environment"""
        if not dist.is_initialized():
            # Auto-detect distributed environment
            if 'WORLD_SIZE' in os.environ:
                dist.init_process_group(
                    backend=self.config.backend,
                    init_method='env://'
                )
            else:
                # Single node training
                os.environ['MASTER_ADDR'] = 'localhost'
                os.environ['MASTER_PORT'] = '12355'
                os.environ['WORLD_SIZE'] = str(torch.cuda.device_count())
                os.environ['RANK'] = '0'
                
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            
        self.logger.info(f"Distributed setup: rank={self.rank}, world_size={self.world_size}")
        
    def setup_monitoring(self):
        """Setup system and training monitoring"""
        self.monitor = ClusterMonitor(
            log_dir=self.config.log_dir,
            rank=self.rank,
            world_size=self.world_size
        )
        
        # Initialize GPU monitoring
        if torch.cuda.is_available():
            nvml.nvmlInit()
            
    def setup_model(self):
        """Setup model with distributed wrapper"""
        self.model_wrapper = DistributedModelWrapper(
            model_name=self.config.model_name,
            num_classes=self.config.num_classes,
            config=self.config
        )
        
        self.model = self.model_wrapper.get_model()
        
        # Convert BatchNorm to SyncBatchNorm for distributed training
        if self.config.sync_batchnorm and self.world_size > 1:
            self.model = SyncBatchNorm.convert_sync_batchnorm(self.model)
            
        # Prepare model with accelerator
        self.model = self.accelerator.prepare(self.model)
        
    def setup_optimizer(self):
        """Setup optimizer and scheduler"""
        # Create parameter groups with different learning rates
        backbone_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'backbone' in name or 'encoder' in name:
                backbone_params.append(param)
            else:
                classifier_params.append(param)
                
        param_groups = [
            {'params': backbone_params, 'lr': self.config.learning_rate * 0.1},
            {'params': classifier_params, 'lr': self.config.learning_rate}
        ]
        
        if self.config.optimizer.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                param_groups,
                weight_decay=self.config.weight_decay,
                eps=1e-8
            )
        elif self.config.optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                param_groups,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
                nesterov=True
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
            
        # Setup scheduler
        total_steps = self.config.epochs * len(self.train_loader)
        warmup_steps = self.config.warmup_epochs * len(self.train_loader)
        
        if self.config.scheduler == 'cosine_warmup':
            from transformers import get_cosine_schedule_with_warmup
            self.scheduler = get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
            
        # Prepare optimizer and scheduler
        self.optimizer, self.scheduler = self.accelerator.prepare(
            self.optimizer, self.scheduler
        )
        
    def setup_data(self):
        """Setup distributed data loading"""
        # Create datasets
        train_dataset = DistributedWSIDataset(
            data_root=self.config.data_root,
            split='train',
            config=self.config,
            transform_config='train'
        )
        
        val_dataset = DistributedWSIDataset(
            data_root=self.config.data_root,
            split='val',
            config=self.config,
            transform_config='val'
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor
        )
        
        # Prepare data loaders
        self.train_loader, self.val_loader = self.accelerator.prepare(
            self.train_loader, self.val_loader
        )
        
    @contextmanager
    def profiler_context(self):
        """Context manager for PyTorch profiler"""
        if self.config.profiler_enabled and self.is_main_process:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler(
                    str(Path(self.config.log_dir) / "profiler")
                ),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
            ) as prof:
                yield prof
        else:
            yield None
            
    def train(self):
        """Main training loop with fault tolerance"""
        self.training_start_time = time.time()
        
        # Resume from checkpoint if available
        start_epoch = self.load_checkpoint()
        
        best_val_metric = 0.0
        patience_counter = 0
        
        try:
            with self.profiler_context() as profiler:
                for epoch in range(start_epoch, self.config.epochs):
                    # Training epoch
                    train_metrics = self.train_epoch(epoch, profiler)
                    
                    # Validation
                    if (epoch + 1) % self.config.val_every_n_epochs == 0:
                        val_metrics = self.validate_epoch(epoch)
                        
                        # EPOC-specific validation
                        epoc_metrics = self.epoc_validator.validate(
                            self.model, self.val_loader, epoch
                        )
                        
                        # Log metrics
                        self.log_metrics(train_metrics, val_metrics, epoc_metrics, epoch)
                        
                        # Check for improvement
                        current_metric = val_metrics.get('f1_macro', 0.0)
                        if current_metric > best_val_metric:
                            best_val_metric = current_metric
                            patience_counter = 0
                            self.save_best_checkpoint(epoch, val_metrics)
                        else:
                            patience_counter += 1
                            
                    # Regular checkpoint saving
                    if (epoch + 1) % self.config.save_every_n_epochs == 0:
                        self.save_checkpoint(epoch, train_metrics)
                        
                    # Health check
                    if epoch % self.config.health_check_interval == 0:
                        self.health_check()
                        
                    # Update profiler
                    if profiler:
                        profiler.step()
                        
        except Exception as e:
            self.logger.error(f"Training failed with error: {str(e)}")
            self.logger.error(traceback.format_exc())
            self.save_emergency_checkpoint()
            raise
            
        finally:
            self.cleanup()
            
    def train_epoch(self, epoch: int, profiler=None) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        epoch_metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'samples_processed': 0
        }
        
        # Progress bar for main process
        if self.is_main_process:
            pbar = tqdm(
                total=len(self.train_loader),
                desc=f'Epoch {epoch}/{self.config.epochs}'
            )
            
        batch_times = []
        
        for batch_idx, batch in enumerate(self.train_loader):
            batch_start_time = time.time()
            
            # Dynamic batch sizing based on GPU memory
            if batch_idx == 0:
                self.adjust_batch_size()
                
            # Forward pass
            with self.accelerator.accumulate(self.model):
                outputs = self.model(batch)
                loss = self.compute_loss(outputs, batch)
                
                # Backward pass
                self.accelerator.backward(loss)
                
                # Gradient clipping
                if self.config.gradient_clipping:
                    self.accelerator.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip_val
                    )
                    
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
            # Update metrics
            batch_loss = loss.item()
            batch_acc = self.compute_accuracy(outputs, batch)
            batch_size = len(batch['labels'])
            
            epoch_metrics['loss'] += batch_loss * batch_size
            epoch_metrics['accuracy'] += batch_acc * batch_size
            epoch_metrics['samples_processed'] += batch_size
            
            # Logging
            if batch_idx % self.config.log_every_n_steps == 0:
                self.log_training_step(
                    epoch, batch_idx, batch_loss, batch_acc,
                    self.optimizer.param_groups[0]['lr']
                )
                
            # Memory monitoring
            if batch_idx % 100 == 0:
                self.monitor.log_gpu_memory()
                
            # Emergency checkpoint
            if batch_idx % self.config.checkpoint_every_n_batches == 0:
                self.save_emergency_checkpoint(epoch, batch_idx)
                
            # Update progress bar
            if self.is_main_process:
                batch_time = time.time() - batch_start_time
                batch_times.append(batch_time)
                
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'acc': f'{batch_acc:.4f}',
                    'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                    'time': f'{np.mean(batch_times[-10:]):.2f}s'
                })
                
            # Profiler step
            if profiler:
                profiler.step()
                
        if self.is_main_process:
            pbar.close()
            
        # Gather metrics across all processes
        if self.world_size > 1:
            epoch_metrics = self.gather_metrics(epoch_metrics)
            
        # Normalize metrics
        total_samples = epoch_metrics['samples_processed']
        epoch_metrics['loss'] /= total_samples
        epoch_metrics['accuracy'] /= total_samples
        
        return epoch_metrics
        
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validation for one epoch"""
        self.model.eval()
        
        val_metrics = {
            'loss': 0.0,
            'accuracy': 0.0,
            'samples_processed': 0
        }
        
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation', disable=not self.is_main_process):
                # Forward pass
                outputs = self.model(batch)
                loss = self.compute_loss(outputs, batch)
                
                # Collect predictions
                predictions = outputs['predictions'].cpu().numpy()
                labels = batch['labels'].cpu().numpy()
                uncertainties = outputs.get('uncertainty', torch.zeros_like(outputs['predictions'])).cpu().numpy()
                
                all_predictions.extend(predictions)
                all_labels.extend(labels)
                all_uncertainties.extend(uncertainties)
                
                # Update metrics
                batch_loss = loss.item()
                batch_acc = self.compute_accuracy(outputs, batch)
                batch_size = len(batch['labels'])
                
                val_metrics['loss'] += batch_loss * batch_size
                val_metrics['accuracy'] += batch_acc * batch_size
                val_metrics['samples_processed'] += batch_size
                
        # Compute comprehensive metrics
        from sklearn.metrics import classification_report, confusion_matrix, f1_score
        
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        f1_per_class = f1_score(all_labels, all_predictions, average=None)
        
        val_metrics.update({
            'f1_macro': f1_macro,
            'f1_canonical': f1_per_class[0],
            'f1_immune': f1_per_class[1],
            'f1_stromal': f1_per_class[2],
            'uncertainty_quality': self.compute_uncertainty_quality(all_uncertainties, all_predictions, all_labels)
        })
        
        # Gather metrics across all processes
        if self.world_size > 1:
            val_metrics = self.gather_metrics(val_metrics)
            
        # Normalize metrics
        total_samples = val_metrics['samples_processed']
        val_metrics['loss'] /= total_samples
        val_metrics['accuracy'] /= total_samples
        
        return val_metrics
        
    def compute_loss(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute multi-task loss with label smoothing and regularization"""
        labels = batch['labels']
        
        # Main classification loss with label smoothing
        logits = outputs['logits']
        ce_loss = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)(logits, labels)
        
        # Uncertainty loss (evidential learning)
        uncertainty_loss = torch.tensor(0.0, device=self.device)
        if 'evidence' in outputs:
            uncertainty_loss = self.compute_evidential_loss(outputs['evidence'], labels)
            
        # Contrastive loss for similar patches
        contrastive_loss = torch.tensor(0.0, device=self.device)
        if 'features' in outputs and 'patch_similarities' in batch:
            contrastive_loss = self.compute_contrastive_loss(
                outputs['features'], batch['patch_similarities']
            )
            
        # Total loss
        total_loss = (
            ce_loss + 
            0.1 * uncertainty_loss + 
            0.05 * contrastive_loss
        )
        
        return total_loss
        
    def compute_accuracy(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> float:
        """Compute batch accuracy"""
        predictions = torch.argmax(outputs['logits'], dim=1)
        labels = batch['labels']
        correct = (predictions == labels).sum().item()
        total = len(labels)
        return correct / total
        
    def compute_uncertainty_quality(self, uncertainties: np.ndarray, predictions: np.ndarray, labels: np.ndarray) -> float:
        """Compute uncertainty calibration quality"""
        # Use Expected Calibration Error (ECE)
        from sklearn.metrics import accuracy_score
        
        # Sort by confidence (inverse of uncertainty)
        confidence = 1 - uncertainties
        sorted_indices = np.argsort(confidence)
        
        ece = 0.0
        n_bins = 10
        
        for i in range(n_bins):
            start_idx = i * len(sorted_indices) // n_bins
            end_idx = (i + 1) * len(sorted_indices) // n_bins
            
            bin_indices = sorted_indices[start_idx:end_idx]
            if len(bin_indices) == 0:
                continue
                
            bin_confidence = confidence[bin_indices].mean()
            bin_accuracy = accuracy_score(labels[bin_indices], predictions[bin_indices])
            bin_weight = len(bin_indices) / len(sorted_indices)
            
            ece += bin_weight * abs(bin_confidence - bin_accuracy)
            
        return 1.0 - ece  # Return quality (higher is better)
        
    def adjust_batch_size(self):
        """Dynamically adjust batch size based on GPU memory"""
        available_memory = self.memory_manager.get_available_memory()
        recommended_batch_size = self.memory_manager.estimate_batch_size(
            self.model, self.config.input_size
        )
        
        if recommended_batch_size != self.config.batch_size:
            self.logger.info(
                f"Adjusting batch size from {self.config.batch_size} to {recommended_batch_size}"
            )
            # Note: This would require recreating data loaders in practice
            
    def health_check(self):
        """Perform health check on training process"""
        issues = []
        
        # Check GPU memory
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if memory_usage > 0.95:
                issues.append(f"High GPU memory usage: {memory_usage:.1%}")
                
        # Check gradient norms
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        if total_norm > 100.0:
            issues.append(f"Large gradient norm: {total_norm:.2f}")
            
        # Check learning rate
        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr < 1e-8:
            issues.append(f"Very small learning rate: {current_lr:.2e}")
            
        if issues:
            self.logger.warning(f"Health check issues: {'; '.join(issues)}")
            
    def gather_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Gather metrics across all processes"""
        if self.world_size == 1:
            return metrics
            
        gathered_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                tensor = torch.tensor(value, device=self.device)
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                gathered_metrics[key] = tensor.item()
            else:
                gathered_metrics[key] = value
                
        return gathered_metrics
        
    def log_metrics(self, train_metrics: Dict, val_metrics: Dict, epoc_metrics: Dict, epoch: int):
        """Log comprehensive metrics"""
        if not self.is_main_process:
            return
            
        # Console logging
        self.logger.info(f"\nEpoch {epoch} Results:")
        self.logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        self.logger.info(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        self.logger.info(f"F1 Scores - Canonical: {val_metrics['f1_canonical']:.4f}, "
                        f"Immune: {val_metrics['f1_immune']:.4f}, Stromal: {val_metrics['f1_stromal']:.4f}")
        
        # WandB logging
        if self.accelerator.is_main_process:
            log_dict = {
                'epoch': epoch,
                'train/loss': train_metrics['loss'],
                'train/accuracy': train_metrics['accuracy'],
                'val/loss': val_metrics['loss'],
                'val/accuracy': val_metrics['accuracy'],
                'val/f1_macro': val_metrics['f1_macro'],
                'val/f1_canonical': val_metrics['f1_canonical'],
                'val/f1_immune': val_metrics['f1_immune'],
                'val/f1_stromal': val_metrics['f1_stromal'],
                'val/uncertainty_quality': val_metrics['uncertainty_quality'],
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'gpu_memory_used': torch.cuda.memory_allocated() / 1024**3,
                'training_time': time.time() - self.training_start_time
            }
            
            # Add EPOC-specific metrics
            log_dict.update({f'epoc/{k}': v for k, v in epoc_metrics.items()})
            
            self.accelerator.log(log_dict, step=epoch)
            
    def log_training_step(self, epoch: int, batch_idx: int, loss: float, acc: float, lr: float):
        """Log training step metrics"""
        if not self.is_main_process:
            return
            
        step = epoch * len(self.train_loader) + batch_idx
        
        if self.accelerator.is_main_process:
            self.accelerator.log({
                'train/step_loss': loss,
                'train/step_accuracy': acc,
                'train/learning_rate': lr,
                'train/step': step
            }, step=step)
            
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save regular checkpoint"""
        if not self.is_main_process:
            return
            
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': asdict(self.config),
            'accelerator_state': self.accelerator.get_state_dict()
        }
        
        self.checkpoint_manager.save_checkpoint(checkpoint_data, epoch)
        
    def save_best_checkpoint(self, epoch: int, metrics: Dict[str, float]):
        """Save best checkpoint"""
        if not self.is_main_process:
            return
            
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'config': asdict(self.config),
            'accelerator_state': self.accelerator.get_state_dict()
        }
        
        self.checkpoint_manager.save_best_checkpoint(checkpoint_data, metrics['f1_macro'])
        
    def save_emergency_checkpoint(self, epoch: int = None, batch_idx: int = None):
        """Save emergency checkpoint in case of failure"""
        if not self.is_main_process:
            return
            
        checkpoint_data = {
            'epoch': epoch or -1,
            'batch_idx': batch_idx or -1,
            'model_state_dict': self.accelerator.unwrap_model(self.model).state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': asdict(self.config),
            'accelerator_state': self.accelerator.get_state_dict(),
            'timestamp': time.time()
        }
        
        emergency_path = Path(self.config.checkpoint_dir) / "emergency_checkpoint.pth"
        torch.save(checkpoint_data, emergency_path)
        self.logger.info(f"Saved emergency checkpoint to {emergency_path}")
        
    def load_checkpoint(self) -> int:
        """Load checkpoint and return start epoch"""
        checkpoint_path = self.checkpoint_manager.get_latest_checkpoint()
        
        if checkpoint_path is None:
            self.logger.info("No checkpoint found, starting from scratch")
            return 0
            
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Load model state
        self.accelerator.unwrap_model(self.model).load_state_dict(
            checkpoint['model_state_dict']
        )
        
        # Load optimizer and scheduler
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
        # Load accelerator state
        if 'accelerator_state' in checkpoint:
            self.accelerator.load_state_dict(checkpoint['accelerator_state'])
            
        start_epoch = checkpoint['epoch'] + 1
        self.logger.info(f"Resumed training from epoch {start_epoch}")
        
        return start_epoch
        
    def cleanup(self):
        """Cleanup distributed training"""
        if self.accelerator.is_main_process and hasattr(self, 'accelerator'):
            self.accelerator.end_training()
            
        if dist.is_initialized():
            dist.destroy_process_group()
            
        self.logger.info("Training cleanup completed")
        
    @property
    def is_main_process(self) -> bool:
        """Check if this is the main process"""
        return self.rank == 0
        
    def get_rank(self) -> int:
        """Get process rank"""
        return self.rank


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Distributed CRC Subtype Training")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        if args.config.endswith('.yaml') or args.config.endswith('.yml'):
            config_dict = yaml.safe_load(f)
        else:
            config_dict = json.load(f)
            
    config = DistributedTrainingConfig(**config_dict)
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Initialize trainer
    trainer = DistributedTrainer(config)
    
    # Start training
    trainer.train()


if __name__ == "__main__":
    main() 