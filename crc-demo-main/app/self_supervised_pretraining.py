#!/usr/bin/env python3
"""
Self-Supervised Pre-training Infrastructure for CRC Analysis
Implements SimCLR and MAE for pre-training on unlabeled WSI data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
import random
from torchvision import transforms
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class WSIAugmentationPair:
    """Generate augmentation pairs for contrastive learning"""
    
    def __init__(self, base_transform, strong_augment=True):
        self.base_transform = base_transform
        
        # Augmentations for SimCLR
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        
        if strong_augment:
            self.augment = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
        else:
            self.augment = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([color_jitter], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __call__(self, image):
        """Return two augmented versions of the same image"""
        return self.augment(image), self.augment(image)


class UnlabeledWSIDataset(Dataset):
    """Dataset for unlabeled WSI tiles"""
    
    def __init__(self, data_path: str, transform=None, extensions=('.png', '.jpg', '.jpeg')):
        self.data_path = Path(data_path)
        self.transform = transform or WSIAugmentationPair(None)
        
        # Find all image files
        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(self.data_path.rglob(f'*{ext}'))
        
        print(f"Found {len(self.image_paths)} unlabeled images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Apply augmentation pair transform
        if self.transform:
            return self.transform(image)
        
        return image


class SimCLRPreTrainer:
    """Self-supervised contrastive pre-training with SimCLR"""
    
    def __init__(self, encoder, projection_dim=128, temperature=0.07):
        self.encoder = encoder
        self.temperature = temperature
        
        # Get encoder output dimension
        if hasattr(encoder, 'output_dim'):
            encoder_dim = encoder.output_dim
        else:
            # Infer from a forward pass
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                encoder_output = encoder(dummy_input)
                # Handle dict output from MultiScaleFeatureExtractor
                if isinstance(encoder_output, dict):
                    encoder_dim = encoder_output['features'].shape[1]
                else:
                    encoder_dim = encoder_output.shape[1]
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim)
        )
    
    def contrastive_loss(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        NT-Xent loss for contrastive learning
        
        Args:
            z_i: Projections from first augmentation [N, projection_dim]
            z_j: Projections from second augmentation [N, projection_dim]
        """
        batch_size = z_i.shape[0]
        device = z_i.device
        
        # Normalize projections
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate representations
        z = torch.cat([z_i, z_j], dim=0)  # [2N, projection_dim]
        
        # Compute similarity matrix
        sim = torch.mm(z, z.t()) / self.temperature  # [2N, 2N]
        
        # Create labels for positive pairs
        labels = torch.cat([torch.arange(batch_size) + batch_size,
                           torch.arange(batch_size)], dim=0)
        labels = labels.to(device)
        
        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(device)
        sim.masked_fill_(mask, -9e15)
        
        # Compute loss
        loss = F.cross_entropy(sim, labels, reduction='mean')
        
        return loss
    
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
                   device: torch.device) -> float:
        """Train one epoch of SimCLR"""
        self.encoder.train()
        self.projection_head.train()
        
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images_i, images_j) in enumerate(dataloader):
            images_i = images_i.to(device)
            images_j = images_j.to(device)
            
            # Extract features
            h_i = self.encoder(images_i)
            h_j = self.encoder(images_j)
            
            # Handle dict output from MultiScaleFeatureExtractor
            if isinstance(h_i, dict):
                h_i = h_i['features']
                h_j = h_j['features']
            
            # Flatten if needed
            if len(h_i.shape) > 2:
                h_i = F.adaptive_avg_pool2d(h_i, 1).squeeze(-1).squeeze(-1)
                h_j = F.adaptive_avg_pool2d(h_j, 1).squeeze(-1).squeeze(-1)
            
            # Project features
            z_i = self.projection_head(h_i)
            z_j = self.projection_head(h_j)
            
            # Compute loss
            loss = self.contrastive_loss(z_i, z_j)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches


class MAEPreTrainer:
    """Masked Autoencoder pre-training"""
    
    def __init__(self, encoder, decoder_dim=512, mask_ratio=0.75, patch_size=16):
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        
        # Calculate number of patches
        self.num_patches = (224 // patch_size) ** 2
        
        # Get encoder output dimension
        if hasattr(encoder, 'output_dim'):
            encoder_dim = encoder.output_dim
        else:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                encoder_output = encoder(dummy_input)
                # Handle dict output from MultiScaleFeatureExtractor
                if isinstance(encoder_output, dict):
                    encoder_dim = encoder_output['features'].shape[1]
                else:
                    encoder_dim = encoder_output.shape[1]
        
        # Decoder for reconstruction
        self.decoder = nn.Sequential(
            nn.Linear(encoder_dim, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, decoder_dim),
            nn.ReLU(),
            nn.Linear(decoder_dim, patch_size * patch_size * 3)
        )
        
        # Mask token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder_dim))
        nn.init.normal_(self.mask_token, std=0.02)
    
    def random_masking(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly mask patches
        
        Args:
            x: Input features [N, L, D]
            mask_ratio: Percentage of patches to mask
            
        Returns:
            x_masked: Masked features
            mask: Binary mask [N, L]
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        # Generate random noise for each sample
        noise = torch.rand(N, L, device=x.device)
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, 
                               index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        
        # Unshuffle to get final mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """Convert images to patches"""
        p = self.patch_size
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], 3, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p * p * 3)
        return x
    
    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert patches back to images"""
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        x = x.reshape(x.shape[0], h, w, p, p, 3)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 3, h * p, w * p)
        return imgs
    
    def train_epoch(self, dataloader: DataLoader, optimizer: torch.optim.Optimizer,
                   device: torch.device) -> float:
        """Train one epoch of MAE"""
        self.encoder.train()
        self.decoder.train()
        
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
            # Patchify images
            patches = self.patchify(images)
            
            # Encode patches (simplified - in practice would use ViT)
            batch_size = images.shape[0]
            encoded_patches = []
            
            for i in range(self.num_patches):
                patch = patches[:, i, :].reshape(batch_size, 3, self.patch_size, self.patch_size)
                feat = self.encoder(F.interpolate(patch, size=(224, 224), mode='bilinear'))
                # Handle dict output from MultiScaleFeatureExtractor
                if isinstance(feat, dict):
                    feat = feat['features']
                if len(feat.shape) > 2:
                    feat = F.adaptive_avg_pool2d(feat, 1).squeeze(-1).squeeze(-1)
                encoded_patches.append(feat)
            
            x = torch.stack(encoded_patches, dim=1)  # [N, L, D]
            
            # Random masking
            x_masked, mask, ids_restore = self.random_masking(x, self.mask_ratio)
            
            # Decode
            pred_patches = []
            for i in range(x_masked.shape[1]):
                pred = self.decoder(x_masked[:, i, :])
                pred_patches.append(pred)
            
            pred = torch.stack(pred_patches, dim=1)  # [N, L_keep, patch_size^2 * 3]
            
            # Calculate reconstruction loss on masked patches
            target = patches[mask == 1].reshape(-1, self.patch_size * self.patch_size * 3)
            if pred.shape[1] * pred.shape[0] != target.shape[0]:
                # Adjust for shape mismatch
                loss = F.mse_loss(pred.reshape(-1, pred.shape[-1])[:target.shape[0]], target)
            else:
                loss = F.mse_loss(pred.reshape(-1, pred.shape[-1]), target)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        return total_loss / num_batches


def pretrain_on_unlabeled_data(model, data_path: str, epochs: int = 100, 
                              batch_size: int = 256, device: str = 'cuda'):
    """
    Main pre-training function for self-supervised learning
    
    Args:
        model: Encoder model to pre-train
        data_path: Path to unlabeled WSI tiles
        epochs: Number of training epochs
        batch_size: Batch size for training
        device: Device to train on
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print("🚀 Self-Supervised Pre-training Pipeline")
    print("="*50)
    
    # Create dataset
    print(f"\n📁 Loading unlabeled data from {data_path}")
    dataset = UnlabeledWSIDataset(data_path)
    
    if len(dataset) == 0:
        print("❌ No unlabeled images found!")
        return
    
    # Phase 1: SimCLR Pre-training
    print("\n🔄 Phase 1: SimCLR Contrastive Pre-training")
    print("-"*40)
    
    simclr_dataset = UnlabeledWSIDataset(
        data_path, 
        transform=WSIAugmentationPair(None, strong_augment=True)
    )
    simclr_loader = DataLoader(
        simclr_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    simclr = SimCLRPreTrainer(model)
    simclr.projection_head = simclr.projection_head.to(device)
    
    # Optimizer for SimCLR
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(simclr.projection_head.parameters()),
        lr=1e-3,
        weight_decay=1e-6
    )
    
    for epoch in range(epochs // 2):
        print(f"\nEpoch {epoch + 1}/{epochs // 2}")
        loss = simclr.train_epoch(simclr_loader, optimizer, device)
        print(f"Average Loss: {loss:.4f}")
    
    print("\n✅ SimCLR pre-training complete!")
    
    # Phase 2: MAE Pre-training
    print("\n🔄 Phase 2: Masked Autoencoder Pre-training")
    print("-"*40)
    
    mae_dataset = UnlabeledWSIDataset(
        data_path,
        transform=transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    )
    mae_loader = DataLoader(
        mae_dataset,
        batch_size=batch_size // 4,  # Smaller batch for MAE
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    mae = MAEPreTrainer(model)
    mae.decoder = mae.decoder.to(device)
    mae.mask_token = mae.mask_token.to(device)
    
    # Optimizer for MAE
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(mae.decoder.parameters()),
        lr=5e-4,
        weight_decay=1e-6
    )
    
    for epoch in range(epochs // 2):
        print(f"\nEpoch {epoch + 1}/{epochs // 2}")
        loss = mae.train_epoch(mae_loader, optimizer, device)
        print(f"Average Loss: {loss:.4f}")
    
    print("\n✅ MAE pre-training complete!")
    
    # Save pre-trained model
    save_path = Path('models/pretrained_encoder.pth')
    save_path.parent.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'simclr_projection': simclr.projection_head.state_dict(),
        'mae_decoder': mae.decoder.state_dict(),
        'epochs': epochs,
        'pretrain_type': 'SimCLR + MAE'
    }, save_path)
    
    print(f"\n💾 Pre-trained model saved to {save_path}")
    print("\n🎯 Pre-training complete! Model ready for fine-tuning.")
    
    return model


# Example usage
if __name__ == "__main__":
    # Example: Pre-train a ResNet50 encoder
    from torchvision import models
    
    # Create encoder
    encoder = models.resnet50(pretrained=False)
    encoder.fc = nn.Identity()  # Remove classification head
    encoder.output_dim = 2048
    
    # Pre-train on unlabeled data
    pretrain_on_unlabeled_data(
        model=encoder,
        data_path="data/unlabeled_wsi_tiles",
        epochs=100,
        batch_size=256
    ) 