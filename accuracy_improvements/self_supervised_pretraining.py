#!/usr/bin/env python3
"""
Self-Supervised Pre-Training Framework for Pathology Images
Implements SimCLR, Barlow Twins, and Masked Autoencoding for better representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import random
from pathlib import Path
import logging
from typing import Tuple, Dict, List
import math
import timm
from PIL import Image
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimCLRTransform:
    """SimCLR-style augmentations for contrastive learning"""
    
    def __init__(self, image_size=224):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, x):
        return self.transform(x), self.transform(x)

class PathologyContrastiveDataset(Dataset):
    """Dataset for contrastive learning on pathology images"""
    
    def __init__(self, image_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.transform = transform or SimCLRTransform()
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Generate two augmented views
        view1, view2 = self.transform(image)
        
        return view1, view2, idx

class ProjectionHead(nn.Module):
    """Projection head for contrastive learning"""
    
    def __init__(self, input_dim=1024, hidden_dim=512, output_dim=128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return F.normalize(self.projection(x), dim=1)

class SimCLRModel(nn.Module):
    """SimCLR model for contrastive learning"""
    
    def __init__(self, backbone='resnet50', projection_dim=128):
        super().__init__()
        
        # Backbone encoder
        if backbone.startswith('resnet'):
            self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
            backbone_dim = self.backbone.num_features
        elif backbone.startswith('efficientnet'):
            self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
            backbone_dim = self.backbone.num_features
        elif backbone.startswith('vit'):
            self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
            backbone_dim = self.backbone.num_features
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Projection head
        self.projection_head = ProjectionHead(
            input_dim=backbone_dim,
            hidden_dim=backbone_dim // 2,
            output_dim=projection_dim
        )
    
    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        return features, projections

class BarlowTwinsModel(nn.Module):
    """Barlow Twins model for self-supervised learning"""
    
    def __init__(self, backbone='resnet50', projection_dim=2048):
        super().__init__()
        
        # Backbone encoder
        if backbone.startswith('resnet'):
            self.backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
            backbone_dim = self.backbone.num_features
        else:
            self.backbone = timm.create_model('resnet50', pretrained=True, num_classes=0)
            backbone_dim = self.backbone.num_features
        
        # Projection head for Barlow Twins
        self.projection_head = nn.Sequential(
            nn.Linear(backbone_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim),
            nn.BatchNorm1d(projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        projections = self.projection_head(features)
        return features, projections

class MaskedAutoencoderViT(nn.Module):
    """Masked Autoencoder with Vision Transformer"""
    
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12, 
                 num_heads=12, decoder_embed_dim=512, decoder_depth=8, 
                 decoder_num_heads=16, mask_ratio=0.75):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.mask_ratio = mask_ratio
        
        # Encoder (ViT)
        self.encoder = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        
        # Decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            timm.models.vision_transformer.Block(
                decoder_embed_dim, decoder_num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=nn.LayerNorm)
            for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * 3)  # RGB
        
    def random_masking(self, x, mask_ratio):
        """Random masking for MAE"""
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        """Forward pass through encoder with masking"""
        # Patch embedding
        x = self.encoder.patch_embed(x)
        x = x + self.encoder.pos_embed[:, 1:, :]
        
        # Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Add class token
        cls_token = self.encoder.cls_token + self.encoder.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Apply encoder blocks
        for blk in self.encoder.blocks:
            x = blk(x)
        x = self.encoder.norm(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x, ids_restore):
        """Forward pass through decoder"""
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Add mask tokens
        mask_tokens = torch.zeros(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], x.shape[2], device=x.device)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        # Add position embedding
        x = x + self.decoder_pos_embed
        
        # Apply decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Predictor
        x = self.decoder_pred(x)
        x = x[:, 1:, :]  # Remove class token
        
        return x
    
    def forward(self, imgs, mask_ratio=None):
        mask_ratio = mask_ratio or self.mask_ratio
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        return pred, mask

class ContrastiveLoss(nn.Module):
    """InfoNCE loss for contrastive learning"""
    
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1, z2):
        """
        z1, z2: normalized projections [batch_size, projection_dim]
        """
        batch_size = z1.shape[0]
        
        # Concatenate z1 and z2
        z = torch.cat([z1, z2], dim=0)  # [2*batch_size, projection_dim]
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # [2*batch_size, 2*batch_size]
        
        # Create labels for positive pairs
        labels = torch.cat([torch.arange(batch_size, 2*batch_size),
                           torch.arange(0, batch_size)]).to(z.device)
        
        # Remove self-similarities
        sim_matrix = sim_matrix - torch.eye(2*batch_size, device=z.device) * 1e9
        
        # Compute loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss

class BarlowTwinsLoss(nn.Module):
    """Barlow Twins loss"""
    
    def __init__(self, lambda_param=0.0051):
        super().__init__()
        self.lambda_param = lambda_param
    
    def forward(self, z1, z2):
        """
        z1, z2: projections [batch_size, projection_dim]
        """
        batch_size = z1.shape[0]
        
        # Normalize
        z1_norm = (z1 - z1.mean(0)) / z1.std(0)
        z2_norm = (z2 - z2.mean(0)) / z2.std(0)
        
        # Cross-correlation matrix
        c = torch.mm(z1_norm.t(), z2_norm) / batch_size
        
        # Loss
        c_diff = (c - torch.eye(c.shape[0], device=c.device)).pow(2)
        
        # Diagonal elements (invariance term)
        invariance_loss = c_diff.diag().sum()
        
        # Off-diagonal elements (redundancy reduction term)
        redundancy_loss = self.lambda_param * c_diff.fill_diagonal_(0).sum()
        
        return invariance_loss + redundancy_loss

class MAELoss(nn.Module):
    """Masked Autoencoder loss"""
    
    def __init__(self, patch_size=16):
        super().__init__()
        self.patch_size = patch_size
    
    def patchify(self, imgs):
        """Convert images to patches"""
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def forward(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove
        """
        target = self.patchify(imgs)
        
        # Compute loss only on masked patches
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

class SelfSupervisedTrainer:
    """Trainer for self-supervised pre-training"""
    
    def __init__(self, method='simclr', backbone='resnet50', device='cuda'):
        self.method = method
        self.device = device
        
        # Initialize model based on method
        if method == 'simclr':
            self.model = SimCLRModel(backbone=backbone).to(device)
            self.loss_fn = ContrastiveLoss()
        elif method == 'barlow_twins':
            self.model = BarlowTwinsModel(backbone=backbone).to(device)
            self.loss_fn = BarlowTwinsLoss()
        elif method == 'mae':
            self.model = MaskedAutoencoderViT().to(device)
            self.loss_fn = MAELoss()
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        # Optimizer
        self.optimizer = optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.05)
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # Training history
        self.history = {'train_loss': [], 'epoch': []}
    
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            if self.method in ['simclr', 'barlow_twins']:
                view1, view2, _ = batch
                view1, view2 = view1.to(self.device), view2.to(self.device)
                
                # Forward pass
                if self.method == 'simclr':
                    _, z1 = self.model(view1)
                    _, z2 = self.model(view2)
                    loss = self.loss_fn(z1, z2)
                else:  # barlow_twins
                    _, z1 = self.model(view1)
                    _, z2 = self.model(view2)
                    loss = self.loss_fn(z1, z2)
                    
            elif self.method == 'mae':
                images, _, _ = batch
                images = images.to(self.device)
                
                pred, mask = self.model(images)
                loss = self.loss_fn(images, pred, mask)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        self.scheduler.step()
        
        return avg_loss
    
    def train(self, dataloader, epochs=100, save_path=None):
        """Full training loop"""
        logger.info(f"Starting {self.method} pre-training for {epochs} epochs")
        
        best_loss = float('inf')
        
        for epoch in range(epochs):
            avg_loss = self.train_epoch(dataloader)
            
            self.history['train_loss'].append(avg_loss)
            self.history['epoch'].append(epoch)
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                if save_path:
                    self.save_checkpoint(save_path, epoch, avg_loss)
        
        logger.info(f"Pre-training completed. Best loss: {best_loss:.4f}")
    
    def save_checkpoint(self, path, epoch, loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'method': self.method,
            'history': self.history
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint['epoch'], checkpoint['loss']
    
    def extract_features(self, dataloader):
        """Extract features using pre-trained model"""
        self.model.eval()
        features = []
        labels = []
        
        with torch.no_grad():
            for batch in dataloader:
                if self.method in ['simclr', 'barlow_twins']:
                    images, _, batch_labels = batch
                    images = images.to(self.device)
                    
                    if hasattr(self.model, 'backbone'):
                        feats = self.model.backbone(images)
                    else:
                        feats, _ = self.model(images)
                    
                    features.append(feats.cpu())
                    labels.append(batch_labels)
        
        return torch.cat(features), torch.cat(labels)

def create_ssl_dataloader(image_paths, batch_size=64, num_workers=4):
    """Create dataloader for self-supervised learning"""
    transform = SimCLRTransform()
    dataset = PathologyContrastiveDataset(image_paths, transform=transform)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader

if __name__ == "__main__":
    # Example usage
    logger.info("🧪 Testing Self-Supervised Pre-Training Framework")
    
    # Create dummy data
    dummy_paths = ['dummy_path'] * 100
    dataloader = create_ssl_dataloader(dummy_paths, batch_size=8)
    
    # Test different methods
    methods = ['simclr', 'barlow_twins', 'mae']
    
    for method in methods:
        logger.info(f"Testing {method}")
        trainer = SelfSupervisedTrainer(method=method, device='cpu')
        
        # Test one batch
        for batch in dataloader:
            try:
                if method in ['simclr', 'barlow_twins']:
                    view1, view2, _ = batch
                    if method == 'simclr':
                        _, z1 = trainer.model(view1)
                        _, z2 = trainer.model(view2)
                        loss = trainer.loss_fn(z1, z2)
                    else:
                        _, z1 = trainer.model(view1)
                        _, z2 = trainer.model(view2)
                        loss = trainer.loss_fn(z1, z2)
                else:  # mae
                    images, _, _ = batch
                    pred, mask = trainer.model(images)
                    loss = trainer.loss_fn(images, pred, mask)
                
                logger.info(f"✅ {method} test successful, loss: {loss.item():.4f}")
                break
            except Exception as e:
                logger.error(f"❌ {method} test failed: {e}")
                break
    
    logger.info("🎉 Self-Supervised Pre-Training Framework ready!") 