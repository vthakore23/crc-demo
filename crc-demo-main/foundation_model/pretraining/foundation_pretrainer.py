#!/usr/bin/env python3
"""
Foundation Model Pre-training for Multi-Scale CRC Analysis
Implements state-of-the-art self-supervised learning methods:
- MAE (Masked Autoencoder)
- SimCLR (Contrastive Learning)
- DINO (Self-Distillation)
- MoCo v3 (Momentum Contrast)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from pathlib import Path
import yaml
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import wandb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from app.multiscale_fusion_network import MultiScaleFeatureExtractor
from app.pathology_augmentation import PathologyAugmentation


class FoundationPretrainer:
    """
    Master class for foundation model pre-training
    Orchestrates multiple self-supervised learning methods
    """
    
    def __init__(self, config_path: str):
        """Initialize pre-trainer with configuration"""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        self.setup_logging()
        
        # Initialize device and distributed training
        self.setup_distributed()
        
        # Create model
        self.model = self.create_model()
        
        # Initialize pre-training methods
        self.mae_trainer = None
        self.simclr_trainer = None
        self.dino_trainer = None
        self.moco_trainer = None
        
        # Initialize data loaders
        self.train_loader = None
        self.val_loader = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_metric = 0
        
    def setup_logging(self):
        """Setup logging and experiment tracking"""
        # Create directories
        log_dir = Path(self.config['output']['log_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Python logging
        log_file = log_dir / f"pretraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize Weights & Biases
        if self.config['logging']['wandb']['enabled']:
            wandb.init(
                project=self.config['logging']['wandb']['project'],
                entity=self.config['logging']['wandb']['entity'],
                config=self.config,
                name=f"foundation_pretraining_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
    
    def setup_distributed(self):
        """Setup distributed training if enabled"""
        self.distributed = self.config['training']['distributed']['enabled']
        
        if self.distributed:
            # Initialize process group
            dist.init_process_group(backend=self.config['training']['distributed']['backend'])
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = torch.device(f'cuda:{self.rank}')
            torch.cuda.set_device(self.device)
        else:
            self.rank = 0
            self.world_size = 1
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.logger.info(f"Initialized training on {self.device} (rank {self.rank}/{self.world_size})")
    
    def create_model(self) -> nn.Module:
        """Create the multi-scale fusion model"""
        # Get base encoder
        base_encoder_name = self.config['model']['architecture']['base_encoder']
        
        if base_encoder_name.startswith('resnet'):
            from torchvision import models
            base_encoder = getattr(models, base_encoder_name)(pretrained=False)
            # Remove classification head
            base_encoder.fc = nn.Identity()
            base_encoder.output_dim = 2048 if '50' in base_encoder_name or '101' in base_encoder_name else 512
        elif base_encoder_name.startswith('efficientnet'):
            import timm
            base_encoder = timm.create_model(base_encoder_name, pretrained=False, num_classes=0)
            base_encoder.output_dim = base_encoder.num_features
        elif base_encoder_name.startswith('vit'):
            # Vision Transformer support
            import timm
            base_encoder = timm.create_model(base_encoder_name, pretrained=False, num_classes=0)
            base_encoder.output_dim = base_encoder.num_features
            self.logger.info(f"Using Vision Transformer: {base_encoder_name}")
        elif base_encoder_name.startswith('convnext'):
            # ConvNeXt support (modern CNN architecture)
            import timm
            base_encoder = timm.create_model(base_encoder_name, pretrained=False, num_classes=0)
            base_encoder.output_dim = base_encoder.num_features
            self.logger.info(f"Using ConvNeXt: {base_encoder_name}")
        else:
            raise ValueError(f"Unknown base encoder: {base_encoder_name}")
        
        # Create multi-scale feature extractor
        model = MultiScaleFeatureExtractor(
            base_encoder=base_encoder,
            scales=self.config['model']['architecture']['scales'],
            feature_dim=self.config['model']['architecture']['feature_dim']
        )
        
        # Move to device and wrap with DDP if distributed
        model = model.to(self.device)
        if self.distributed:
            model = DDP(model, device_ids=[self.rank])
        
        return model
    
    def run_pretraining(self):
        """Run the complete pre-training pipeline"""
        self.logger.info("🚀 Starting Foundation Model Pre-training")
        self.logger.info("="*60)
        
        # Phase 1: MAE Pre-training
        if self.config['pretraining']['mae']['enabled']:
            self.logger.info("\n📦 Phase 1: Masked Autoencoder (MAE) Pre-training")
            self.mae_trainer = MAETrainer(self.model, self.config, self.device)
            self.run_mae_pretraining()
        
        # Phase 2: SimCLR Pre-training
        if self.config['pretraining']['simclr']['enabled']:
            self.logger.info("\n🔄 Phase 2: SimCLR Contrastive Pre-training")
            self.simclr_trainer = SimCLRTrainer(self.model, self.config, self.device)
            self.run_simclr_pretraining()
        
        # Phase 3: DINO Pre-training
        if self.config['pretraining']['dino']['enabled']:
            self.logger.info("\n🦕 Phase 3: DINO Self-Distillation Pre-training")
            self.dino_trainer = DINOTrainer(self.model, self.config, self.device)
            self.run_dino_pretraining()
        
        # Phase 4: MoCo v3 Pre-training
        if self.config['pretraining']['moco']['enabled']:
            self.logger.info("\n🎯 Phase 4: MoCo v3 Momentum Contrast Pre-training")
            self.moco_trainer = MoCoV3Trainer(self.model, self.config, self.device)
            self.run_moco_pretraining()
        
        # Final evaluation
        self.logger.info("\n✅ Pre-training Complete! Running final evaluation...")
        self.evaluate_pretrained_model()
        
    def run_mae_pretraining(self):
        """Run MAE pre-training phase"""
        mae_config = self.config['pretraining']['mae']
        epochs = mae_config['epochs']
        
        # Create data loader with MAE-specific transforms
        train_loader = self.create_dataloader('mae')
        
        # Training loop
        for epoch in range(epochs):
            self.logger.info(f"\nMAE Epoch {epoch+1}/{epochs}")
            
            # Train one epoch
            train_loss = self.mae_trainer.train_epoch(train_loader)
            
            # Log metrics
            self.logger.info(f"MAE Loss: {train_loss:.4f}")
            if wandb.run:
                wandb.log({
                    'mae/train_loss': train_loss,
                    'mae/epoch': epoch
                })
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['checkpointing']['save_frequency'] == 0:
                self.save_checkpoint('mae', epoch, train_loss)
    
    def run_simclr_pretraining(self):
        """Run SimCLR pre-training phase"""
        simclr_config = self.config['pretraining']['simclr']
        epochs = simclr_config['epochs']
        
        # Create data loader with SimCLR-specific transforms
        train_loader = self.create_dataloader('simclr')
        
        # Training loop
        for epoch in range(epochs):
            self.logger.info(f"\nSimCLR Epoch {epoch+1}/{epochs}")
            
            # Train one epoch
            train_loss = self.simclr_trainer.train_epoch(train_loader)
            
            # Validate with k-NN
            if (epoch + 1) % self.config['validation']['validation_frequency'] == 0:
                knn_acc = self.evaluate_knn()
                self.logger.info(f"SimCLR k-NN Accuracy: {knn_acc:.2%}")
                
                if wandb.run:
                    wandb.log({
                        'simclr/knn_accuracy': knn_acc,
                        'simclr/epoch': epoch
                    })
            
            # Log metrics
            self.logger.info(f"SimCLR Loss: {train_loss:.4f}")
            if wandb.run:
                wandb.log({
                    'simclr/train_loss': train_loss,
                    'simclr/epoch': epoch
                })
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['checkpointing']['save_frequency'] == 0:
                self.save_checkpoint('simclr', epoch, train_loss)
    
    def evaluate_pretrained_model(self):
        """Comprehensive evaluation of the pre-trained model"""
        self.logger.info("\n🔍 Evaluating Pre-trained Model")
        
        results = {}
        
        # 1. k-NN evaluation
        knn_acc = self.evaluate_knn()
        results['knn_accuracy'] = knn_acc
        self.logger.info(f"k-NN Accuracy: {knn_acc:.2%}")
        
        # 2. Linear probe evaluation
        linear_acc = self.evaluate_linear_probe()
        results['linear_probe_accuracy'] = linear_acc
        self.logger.info(f"Linear Probe Accuracy: {linear_acc:.2%}")
        
        # 3. Feature quality metrics
        feature_metrics = self.evaluate_feature_quality()
        results.update(feature_metrics)
        
        # 4. Downstream task preview
        downstream_results = self.evaluate_downstream_tasks()
        results.update(downstream_results)
        
        # Log all results
        if wandb.run:
            wandb.log({f'final/{k}': v for k, v in results.items()})
        
        # Save final model
        self.save_final_model(results)
        
        return results
    
    def save_final_model(self, metrics: Dict):
        """Save the final pre-trained model with metadata"""
        checkpoint_dir = Path(self.config['output']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare checkpoint
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'pretraining_phases': {
                'mae': self.config['pretraining']['mae']['enabled'],
                'simclr': self.config['pretraining']['simclr']['enabled'],
                'dino': self.config['pretraining']['dino']['enabled'],
                'moco': self.config['pretraining']['moco']['enabled']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Save checkpoint
        save_path = checkpoint_dir / 'foundation_model_pretrained.pth'
        torch.save(checkpoint, save_path)
        self.logger.info(f"💾 Saved final model to {save_path}")
        
        # Also save a lighter version without optimizer states
        model_only = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config['model'],
            'metrics': metrics
        }
        
        light_path = checkpoint_dir / 'foundation_model_weights.pth'
        torch.save(model_only, light_path)
        self.logger.info(f"💾 Saved model weights to {light_path}")


class MAETrainer:
    """Masked Autoencoder trainer for vision transformers"""
    
    def __init__(self, model: nn.Module, config: Dict, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.mae_config = config['pretraining']['mae']
        
        # Create MAE-specific components
        self.mask_ratio = self.mae_config['mask_ratio']
        self.patch_size = self.mae_config['patch_size']
        
        # Decoder
        self.decoder = self._create_decoder()
        self.decoder = self.decoder.to(device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.decoder.parameters()),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Loss function
        self.criterion = nn.MSELoss() if self.mae_config['reconstruction_loss'] == 'l2' else nn.L1Loss()
    
    def _create_decoder(self) -> nn.Module:
        """Create MAE decoder"""
        decoder_dim = self.mae_config['decoder_dim']
        decoder_depth = self.mae_config['decoder_depth']
        
        layers = []
        # Input projection
        layers.append(nn.Linear(self.config['model']['architecture']['feature_dim'], decoder_dim))
        layers.append(nn.ReLU(inplace=True))
        
        # Transformer blocks (simplified)
        for _ in range(decoder_depth):
            layers.extend([
                nn.Linear(decoder_dim, decoder_dim),
                nn.LayerNorm(decoder_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
        
        # Output projection
        layers.append(nn.Linear(decoder_dim, self.patch_size * self.patch_size * 3))
        
        return nn.Sequential(*layers)
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train one epoch of MAE"""
        self.model.train()
        self.decoder.train()
        
        total_loss = 0
        num_batches = 0
        
        with tqdm(dataloader, desc="MAE Training") as pbar:
            for batch_idx, images in enumerate(pbar):
                if isinstance(images, (list, tuple)):
                    images = images[0]  # Get first element if tuple
                
                images = images.to(self.device)
                
                # Patchify
                patches = self.patchify(images)
                batch_size, num_patches, _ = patches.shape
                
                # Random masking
                masked_patches, mask, ids_restore = self.random_masking(patches)
                
                # Encode with multi-scale model
                # We'll use the finest scale features
                model_output = self.model(images)
                encoded_features = model_output['features']  # [B, C]
                
                # Expand features to patch dimension
                encoded_patches = encoded_features.unsqueeze(1).expand(-1, num_patches, -1)
                
                # Apply masking
                visible_patches = encoded_patches[~mask].reshape(batch_size, -1, encoded_features.shape[-1])
                
                # Decode
                pred_patches = self.decoder(visible_patches)
                
                # Compute loss only on masked patches
                target_patches = patches[mask].reshape(-1, self.patch_size * self.patch_size * 3)
                pred_patches_flat = pred_patches.reshape(-1, self.patch_size * self.patch_size * 3)
                
                # Ensure same number of predictions
                min_size = min(target_patches.shape[0], pred_patches_flat.shape[0])
                loss = self.criterion(pred_patches_flat[:min_size], target_patches[:min_size])
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.decoder.parameters()),
                    self.config['training']['gradient_clip']
                )
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert images to patches"""
        p = self.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p * p * 3)
        return x
    
    def random_masking(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly mask patches"""
        N, L, D = x.shape
        len_keep = int(L * (1 - self.mask_ratio))
        
        # Generate random noise
        noise = torch.rand(N, L, device=x.device)
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], dtype=torch.bool, device=x.device)
        mask[:, :len_keep] = 0
        
        # Unshuffle to get final mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x, mask, ids_restore


class SimCLRTrainer:
    """SimCLR contrastive learning trainer"""
    
    def __init__(self, model: nn.Module, config: Dict, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.simclr_config = config['pretraining']['simclr']
        
        # Projection head
        self.projection_head = self._create_projection_head()
        self.projection_head = self.projection_head.to(device)
        
        # Temperature
        self.temperature = self.simclr_config['temperature']
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.projection_head.parameters()),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.simclr_config['epochs'],
            eta_min=config['training']['scheduler']['min_lr']
        )
    
    def _create_projection_head(self) -> nn.Module:
        """Create SimCLR projection head"""
        dims = [self.config['model']['architecture']['feature_dim']] + \
               self.simclr_config['projection_head_dims']
        
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # No activation after last layer
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.BatchNorm1d(dims[i + 1]))
        
        return nn.Sequential(*layers)
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train one epoch of SimCLR"""
        self.model.train()
        self.projection_head.train()
        
        total_loss = 0
        num_batches = 0
        
        with tqdm(dataloader, desc="SimCLR Training") as pbar:
            for batch_idx, (images1, images2) in enumerate(pbar):
                images1 = images1.to(self.device)
                images2 = images2.to(self.device)
                
                batch_size = images1.shape[0]
                
                # Get features from both views
                features1 = self.model(images1)['features']
                features2 = self.model(images2)['features']
                
                # Project features
                z1 = self.projection_head(features1)
                z2 = self.projection_head(features2)
                
                # Normalize
                z1 = F.normalize(z1, dim=1)
                z2 = F.normalize(z2, dim=1)
                
                # Concatenate
                z = torch.cat([z1, z2], dim=0)  # [2N, D]
                
                # Compute similarity matrix
                sim = torch.mm(z, z.t()) / self.temperature  # [2N, 2N]
                
                # Create labels
                labels = torch.cat([
                    torch.arange(batch_size) + batch_size,
                    torch.arange(batch_size)
                ], dim=0).to(self.device)
                
                # Mask out self-similarities
                mask = torch.eye(2 * batch_size, dtype=torch.bool).to(self.device)
                sim.masked_fill_(mask, -9e15)
                
                # Compute loss
                loss = F.cross_entropy(sim, labels)
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.projection_head.parameters()),
                    self.config['training']['gradient_clip']
                )
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        self.scheduler.step()
        return total_loss / num_batches


class DINOTrainer:
    """DINO self-distillation trainer"""
    
    def __init__(self, model: nn.Module, config: Dict, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.dino_config = config['pretraining']['dino']
        
        # Create teacher model (EMA of student)
        self.teacher_model = self._create_teacher_model()
        
        # DINO head
        self.student_head = self._create_dino_head()
        self.teacher_head = self._create_dino_head()
        
        self.student_head = self.student_head.to(device)
        self.teacher_head = self.teacher_head.to(device)
        
        # Temperature parameters
        self.student_temp = self.dino_config['student_temp']
        self.teacher_temp = self.dino_config['teacher_temp']
        self.teacher_momentum = self.dino_config['teacher_momentum']
        
        # Center for teacher outputs
        self.register_buffer('center', torch.zeros(1, self.config['model']['architecture']['num_classes']))
        
        # Optimizer (only for student)
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.student_head.parameters()),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    
    def _create_teacher_model(self) -> nn.Module:
        """Create teacher model as EMA of student"""
        import copy
        teacher = copy.deepcopy(self.model)
        # Freeze teacher
        for p in teacher.parameters():
            p.requires_grad = False
        return teacher.to(self.device)
    
    def _create_dino_head(self) -> nn.Module:
        """Create DINO projection head"""
        num_classes = self.config['model']['architecture']['num_classes']
        feature_dim = self.config['model']['architecture']['feature_dim']
        
        return nn.Sequential(
            nn.Linear(feature_dim, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes)
        )
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train one epoch of DINO"""
        self.model.train()
        self.student_head.train()
        
        total_loss = 0
        num_batches = 0
        
        with tqdm(dataloader, desc="DINO Training") as pbar:
            for batch_idx, images in enumerate(pbar):
                # DINO uses multiple crops
                global_crops = images[:2]  # 2 global crops
                local_crops = images[2:] if len(images) > 2 else []  # Local crops
                
                # Move to device
                global_crops = [img.to(self.device) for img in global_crops]
                local_crops = [img.to(self.device) for img in local_crops]
                
                # Teacher forward (global crops only)
                with torch.no_grad():
                    teacher_output = []
                    for crop in global_crops:
                        feat = self.teacher_model(crop)['features']
                        out = self.teacher_head(feat)
                        teacher_output.append(out)
                    
                    teacher_output = torch.cat(teacher_output, dim=0)
                    
                    # Center and sharpen teacher predictions
                    teacher_output = teacher_output - self.center
                    teacher_output = F.softmax(teacher_output / self.teacher_temp, dim=-1)
                
                # Student forward (all crops)
                student_output = []
                for crop in global_crops + local_crops:
                    feat = self.model(crop)['features']
                    out = self.student_head(feat)
                    student_output.append(out)
                
                student_output = torch.cat(student_output, dim=0)
                student_output = student_output / self.student_temp
                student_log_softmax = F.log_softmax(student_output, dim=-1)
                
                # Split student output
                batch_size = global_crops[0].shape[0]
                student_global = student_log_softmax[:2 * batch_size]
                student_local = student_log_softmax[2 * batch_size:]
                
                # Compute loss
                loss = 0
                n_loss_terms = 0
                
                # Global-to-global distillation
                for i in range(2):  # 2 global crops
                    for j in range(2):  # 2 teacher outputs
                        if i != j:
                            loss += -torch.mean(torch.sum(
                                teacher_output[j * batch_size:(j + 1) * batch_size] * 
                                student_global[i * batch_size:(i + 1) * batch_size], 
                                dim=-1
                            ))
                            n_loss_terms += 1
                
                # Local-to-global distillation
                if len(local_crops) > 0:
                    for i in range(len(local_crops)):
                        for j in range(2):  # 2 teacher outputs
                            loss += -torch.mean(torch.sum(
                                teacher_output[j * batch_size:(j + 1) * batch_size] * 
                                student_local[i * batch_size:(i + 1) * batch_size], 
                                dim=-1
                            ))
                            n_loss_terms += 1
                
                loss = loss / n_loss_terms
                
                # Update student
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + list(self.student_head.parameters()),
                    self.config['training']['gradient_clip']
                )
                self.optimizer.step()
                
                # EMA update teacher
                self._update_teacher()
                
                # Update center
                self._update_center(teacher_output)
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def _update_teacher(self):
        """Update teacher with EMA"""
        with torch.no_grad():
            # Model parameters
            for param_q, param_k in zip(self.model.parameters(), self.teacher_model.parameters()):
                param_k.data.mul_(self.teacher_momentum).add_((1 - self.teacher_momentum) * param_q.data)
            
            # Head parameters
            for param_q, param_k in zip(self.student_head.parameters(), self.teacher_head.parameters()):
                param_k.data.mul_(self.teacher_momentum).add_((1 - self.teacher_momentum) * param_q.data)
    
    def _update_center(self, teacher_output: torch.Tensor):
        """Update center with EMA"""
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center.mul_(0.9).add_(0.1 * batch_center)


class MoCoV3Trainer:
    """MoCo v3 momentum contrast trainer"""
    
    def __init__(self, model: nn.Module, config: Dict, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.moco_config = config['pretraining']['moco']
        
        # Create momentum encoder
        self.momentum_model = self._create_momentum_encoder()
        
        # Projection heads
        self.projection_head = self._create_projection_head()
        self.momentum_projection_head = self._create_projection_head()
        
        self.projection_head = self.projection_head.to(device)
        self.momentum_projection_head = self.momentum_projection_head.to(device)
        
        # Prediction head (only for query)
        self.prediction_head = self._create_prediction_head()
        self.prediction_head = self.prediction_head.to(device)
        
        # Parameters
        self.temperature = self.moco_config['temperature']
        self.momentum = self.moco_config['momentum']
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + 
            list(self.projection_head.parameters()) + 
            list(self.prediction_head.parameters()),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    
    def _create_momentum_encoder(self) -> nn.Module:
        """Create momentum encoder"""
        import copy
        momentum_model = copy.deepcopy(self.model)
        # Initialize momentum encoder with query encoder
        for param_q, param_k in zip(self.model.parameters(), momentum_model.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        return momentum_model.to(self.device)
    
    def _create_projection_head(self) -> nn.Module:
        """Create MoCo projection head"""
        feature_dim = self.config['model']['architecture']['feature_dim']
        projection_dim = self.config['model']['architecture']['projection_dim']
        
        return nn.Sequential(
            nn.Linear(feature_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, projection_dim)
        )
    
    def _create_prediction_head(self) -> nn.Module:
        """Create MoCo prediction head"""
        projection_dim = self.config['model']['architecture']['projection_dim']
        
        return nn.Sequential(
            nn.Linear(projection_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, projection_dim)
        )
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train one epoch of MoCo v3"""
        self.model.train()
        self.projection_head.train()
        self.prediction_head.train()
        
        total_loss = 0
        num_batches = 0
        
        with tqdm(dataloader, desc="MoCo v3 Training") as pbar:
            for batch_idx, (images1, images2) in enumerate(pbar):
                images1 = images1.to(self.device)
                images2 = images2.to(self.device)
                
                # Query features
                q1_feat = self.model(images1)['features']
                q2_feat = self.model(images2)['features']
                
                q1 = self.projection_head(q1_feat)
                q2 = self.projection_head(q2_feat)
                
                q1 = self.prediction_head(q1)
                q2 = self.prediction_head(q2)
                
                # Key features (with momentum encoder)
                with torch.no_grad():
                    # Momentum update
                    self._momentum_update()
                    
                    k1_feat = self.momentum_model(images1)['features']
                    k2_feat = self.momentum_model(images2)['features']
                    
                    k1 = self.momentum_projection_head(k1_feat)
                    k2 = self.momentum_projection_head(k2_feat)
                
                # Compute loss
                loss = self._compute_loss(q1, k2) + self._compute_loss(q2, k1)
                
                if self.moco_config['symmetric_loss']:
                    loss = loss / 2
                
                # Backward
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.model.parameters()) + 
                    list(self.projection_head.parameters()) + 
                    list(self.prediction_head.parameters()),
                    self.config['training']['gradient_clip']
                )
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                # Update progress bar
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return total_loss / num_batches
    
    def _compute_loss(self, q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Compute MoCo loss"""
        # Normalize
        q = F.normalize(q, dim=1)
        k = F.normalize(k, dim=1)
        
        # Gather all targets
        if self.distributed:
            k = self._gather_from_all(k)
        
        # Compute similarity
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.temperature
        
        # Labels: positives are diagonal
        batch_size = q.shape[0]
        labels = torch.arange(batch_size, dtype=torch.long).to(self.device)
        
        # Compute cross entropy
        loss = F.cross_entropy(logits, labels)
        
        return loss
    
    def _momentum_update(self):
        """Momentum update of the momentum encoder"""
        with torch.no_grad():
            # Model
            for param_q, param_k in zip(self.model.parameters(), self.momentum_model.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
            
            # Projection head
            for param_q, param_k in zip(self.projection_head.parameters(), 
                                      self.momentum_projection_head.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)
    
    def _gather_from_all(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather tensors from all processes"""
        gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, tensor)
        gathered = torch.cat(gathered, dim=0)
        return gathered


# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Foundation Model Pre-training')
    parser.add_argument('--config', type=str, 
                       default='foundation_model/configs/pretraining_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Create and run pre-trainer
    pretrainer = FoundationPretrainer(args.config)
    pretrainer.run_pretraining() 