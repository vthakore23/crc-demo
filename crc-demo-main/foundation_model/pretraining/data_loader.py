#!/usr/bin/env python3
"""
Data Loaders for Foundation Model Pre-training
Supports TCGA, CAMELYON, and custom pathology datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union, Callable
import random
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import staintools
import openslide
import warnings
warnings.filterwarnings('ignore')


class PathologyPretrainingDataset(Dataset):
    """
    Unified dataset for pathology pre-training
    Supports multiple data sources and augmentation strategies
    """
    
    def __init__(
        self,
        data_sources: List[Dict],
        transform: Optional[Callable] = None,
        patch_size: int = 256,
        scales: List[float] = [1.0],
        stain_normalize: bool = True,
        pretraining_mode: str = 'simclr'  # 'mae', 'simclr', 'dino', 'moco'
    ):
        """
        Args:
            data_sources: List of data source configurations
            transform: Custom transform pipeline
            patch_size: Size of extracted patches
            scales: Multi-scale factors
            stain_normalize: Whether to apply stain normalization
            pretraining_mode: Type of pre-training
        """
        self.patch_size = patch_size
        self.scales = scales
        self.stain_normalize = stain_normalize
        self.pretraining_mode = pretraining_mode
        
        # Initialize stain normalizer if needed
        if stain_normalize:
            self.stain_normalizer = staintools.StainNormalizer(method='macenko')
            self.target_stain_set = False
        
        # Load all patches from data sources
        self.patches = []
        self.metadata = []
        
        for source in data_sources:
            if source['enabled']:
                self._load_data_source(source)
        
        print(f"Loaded {len(self.patches)} patches for pre-training")
        
        # Setup transforms based on pre-training mode
        if transform is not None:
            self.transform = transform
        else:
            self.transform = self._get_default_transform()
    
    def _load_data_source(self, source: Dict):
        """Load patches from a specific data source"""
        source_type = source.get('type', 'patches')
        
        if source_type == 'patches':
            # Pre-extracted patches
            self._load_patches(source)
        elif source_type == 'slides':
            # Whole slide images
            self._load_wsi(source)
        else:
            raise ValueError(f"Unknown source type: {source_type}")
    
    def _load_patches(self, source: Dict):
        """Load pre-extracted patches"""
        data_path = Path(source['path'])
        
        # Find all image files
        extensions = ['.png', '.jpg', '.jpeg', '.tif', '.tiff']
        patch_files = []
        
        for ext in extensions:
            patch_files.extend(data_path.rglob(f'*{ext}'))
        
        # Limit to specified number of samples
        if 'num_samples' in source:
            patch_files = random.sample(patch_files, 
                                      min(len(patch_files), source['num_samples']))
        
        # Add to dataset
        for patch_file in patch_files:
            self.patches.append(str(patch_file))
            self.metadata.append({
                'source': source.get('name', 'unknown'),
                'magnification': source.get('magnification', '20x'),
                'tissue_type': source.get('tissue_type', 'colorectal')
            })
    
    def _load_wsi(self, source: Dict):
        """Extract patches from whole slide images"""
        slides_path = Path(source['path'])
        slide_files = list(slides_path.glob('*.svs')) + \
                     list(slides_path.glob('*.tif')) + \
                     list(slides_path.glob('*.ndpi'))
        
        # Limit slides if specified
        if 'num_slides' in source:
            slide_files = random.sample(slide_files, 
                                      min(len(slide_files), source['num_slides']))
        
        stride = source.get('stride', self.patch_size // 2)
        target_patches_per_slide = source.get('patches_per_slide', 100)
        
        for slide_file in slide_files:
            try:
                slide = openslide.OpenSlide(str(slide_file))
                
                # Extract patches at tissue regions
                patches = self._extract_tissue_patches(
                    slide, 
                    patch_size=self.patch_size,
                    stride=stride,
                    max_patches=target_patches_per_slide
                )
                
                # Save patches temporarily or keep in memory
                for patch in patches:
                    self.patches.append(patch)  # PIL Image
                    self.metadata.append({
                        'source': source.get('name', 'unknown'),
                        'slide': slide_file.name,
                        'magnification': source.get('magnification', '20x'),
                        'tissue_type': source.get('tissue_type', 'colorectal')
                    })
                
            except Exception as e:
                print(f"Error processing slide {slide_file}: {e}")
                continue
    
    def _extract_tissue_patches(
        self, 
        slide: openslide.OpenSlide,
        patch_size: int,
        stride: int,
        max_patches: int
    ) -> List[Image.Image]:
        """Extract patches from tissue regions in WSI"""
        # Get thumbnail for tissue detection
        thumbnail_size = (1024, 1024)
        thumbnail = slide.get_thumbnail(thumbnail_size)
        thumbnail_np = np.array(thumbnail)
        
        # Simple tissue detection
        gray = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2GRAY)
        _, tissue_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, 
                                     np.ones((10, 10), np.uint8))
        
        # Find tissue regions
        contours, _ = cv2.findContours(tissue_mask, cv2.RETR_EXTERNAL, 
                                      cv2.CHAIN_APPROX_SIMPLE)
        
        # Scale factors
        scale_x = slide.dimensions[0] / thumbnail_size[0]
        scale_y = slide.dimensions[1] / thumbnail_size[1]
        
        patches = []
        
        for contour in contours:
            if cv2.contourArea(contour) < 100:  # Skip small regions
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Scale to slide coordinates
            x_slide = int(x * scale_x)
            y_slide = int(y * scale_y)
            w_slide = int(w * scale_x)
            h_slide = int(h * scale_y)
            
            # Extract patches from this region
            for py in range(y_slide, y_slide + h_slide - patch_size, stride):
                for px in range(x_slide, x_slide + w_slide - patch_size, stride):
                    try:
                        patch = slide.read_region((px, py), 0, (patch_size, patch_size))
                        patch = patch.convert('RGB')
                        
                        # Check if patch contains enough tissue
                        patch_np = np.array(patch)
                        if np.mean(patch_np) < 240:  # Not mostly white
                            patches.append(patch)
                            
                            if len(patches) >= max_patches:
                                return patches
                    except:
                        continue
        
        return patches
    
    def _get_default_transform(self) -> Callable:
        """Get default transform based on pre-training mode"""
        if self.pretraining_mode == 'mae':
            # Simple normalization for MAE
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        
        elif self.pretraining_mode in ['simclr', 'moco']:
            # Strong augmentations for contrastive learning
            return SimCLRAugmentation(size=224)
        
        elif self.pretraining_mode == 'dino':
            # DINO-specific augmentations
            return DINOAugmentation(
                global_crops_scale=(0.4, 1.0),
                local_crops_scale=(0.05, 0.4),
                local_crops_number=8
            )
        
        else:
            # Basic transform
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self) -> int:
        return len(self.patches)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Get item based on pre-training mode"""
        # Load patch
        patch = self.patches[idx]
        if isinstance(patch, str):
            patch = Image.open(patch).convert('RGB')
        
        # Apply stain normalization if enabled
        if self.stain_normalize:
            patch = self._apply_stain_normalization(patch)
        
        # Apply transforms based on mode
        if self.pretraining_mode in ['simclr', 'moco']:
            # Return two augmented views
            return self.transform(patch)
        elif self.pretraining_mode == 'dino':
            # Return multiple crops
            return self.transform(patch)
        else:
            # Single augmented view
            return self.transform(patch)
    
    def _apply_stain_normalization(self, image: Image.Image) -> Image.Image:
        """Apply Macenko stain normalization"""
        try:
            img_np = np.array(image)
            
            # Set target stain if not set
            if not self.target_stain_set:
                self.stain_normalizer.fit(img_np)
                self.target_stain_set = True
            
            # Normalize
            normalized = self.stain_normalizer.transform(img_np)
            return Image.fromarray(normalized)
        except:
            # Return original if normalization fails
            return image


class SimCLRAugmentation:
    """SimCLR augmentation pipeline"""
    
    def __init__(self, size: int = 224, s: float = 0.5):
        self.size = size
        
        # Color jitter
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        
        # Main augmentation pipeline
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=45),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=size//20*2+1)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Additional pathology-specific augmentations
        self.pathology_augment = A.Compose([
            A.ElasticTransform(alpha=120, sigma=9, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, 
                               val_shift_limit=10, p=0.3),
            A.Rotate(limit=45, p=0.5),
        ])
    
    def __call__(self, image: Image.Image) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return two augmented views"""
        # Apply pathology augmentations
        img_np = np.array(image)
        augmented = self.pathology_augment(image=img_np)['image']
        image = Image.fromarray(augmented)
        
        # Generate two views
        view1 = self.train_transform(image)
        view2 = self.train_transform(image)
        
        return view1, view2


class DINOAugmentation:
    """DINO augmentation pipeline with global and local crops"""
    
    def __init__(
        self,
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
        local_crops_scale: Tuple[float, float] = (0.05, 0.4),
        local_crops_number: int = 8,
        size: int = 224
    ):
        self.local_crops_number = local_crops_number
        
        # Global crops
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=global_crops_scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=global_crops_scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Local crops
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def __call__(self, image: Image.Image) -> List[torch.Tensor]:
        """Return global and local crops"""
        crops = []
        
        # Global crops
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        
        # Local crops
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        
        return crops


def create_pretraining_dataloaders(
    config: Dict,
    pretraining_mode: str,
    distributed: bool = False
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create data loaders for pre-training
    
    Args:
        config: Configuration dictionary
        pretraining_mode: Type of pre-training ('mae', 'simclr', 'dino', 'moco')
        distributed: Whether to use distributed training
        
    Returns:
        train_loader, val_loader (if validation is enabled)
    """
    # Prepare data sources
    data_sources = []
    
    # TCGA
    if config['datasets']['tcga']['enabled']:
        data_sources.append({
            'name': 'tcga',
            'enabled': True,
            'type': 'patches',
            'path': config['datasets']['tcga']['path'],
            'num_samples': config['datasets']['tcga']['num_samples'],
            'magnification': config['datasets']['tcga']['magnification']
        })
    
    # CAMELYON
    if config['datasets']['camelyon']['enabled']:
        data_sources.append({
            'name': 'camelyon',
            'enabled': True,
            'type': 'patches',
            'path': config['datasets']['camelyon']['path'],
            'num_samples': config['datasets']['camelyon']['num_samples'],
            'magnification': config['datasets']['camelyon']['magnification']
        })
    
    # Internal data
    if config['datasets']['internal']['enabled']:
        data_sources.append({
            'name': 'internal',
            'enabled': True,
            'type': 'slides',
            'path': config['datasets']['internal']['path'],
            'num_slides': 100,
            'patches_per_slide': 200
        })
    
    # Create dataset
    train_dataset = PathologyPretrainingDataset(
        data_sources=data_sources,
        patch_size=config['datasets']['tcga']['patch_size'],
        scales=config['model']['architecture']['scales'],
        stain_normalize=config['datasets']['tcga'].get('stain_normalization', 'macenko') == 'macenko',
        pretraining_mode=pretraining_mode
    )
    
    # Create sampler for distributed training
    if distributed:
        train_sampler = DistributedSampler(train_dataset)
        shuffle = False
    else:
        train_sampler = None
        shuffle = True
    
    # Create data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=config['training']['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    # Create validation loader if needed
    val_loader = None
    # TODO: Implement validation data loading
    
    return train_loader, val_loader


# Testing
if __name__ == "__main__":
    # Test data loading
    config = {
        'datasets': {
            'tcga': {
                'enabled': True,
                'path': '/data/tcga_patches',
                'num_samples': 1000,
                'patch_size': 256,
                'magnification': '20x',
                'stain_normalization': 'macenko'
            },
            'camelyon': {
                'enabled': False,
                'path': '/data/camelyon_patches',
                'num_samples': 500,
                'patch_size': 256,
                'magnification': '20x'
            },
            'internal': {
                'enabled': False,
                'path': '/data/internal_slides',
                'num_samples': 100,
                'patch_size': 256
            }
        },
        'model': {
            'architecture': {
                'scales': [1.0, 0.5, 0.25]
            }
        },
        'training': {
            'batch_size': 32,
            'num_workers': 4
        }
    }
    
    # Test different pre-training modes
    for mode in ['mae', 'simclr', 'dino', 'moco']:
        print(f"\nTesting {mode} data loader...")
        train_loader, val_loader = create_pretraining_dataloaders(
            config, 
            pretraining_mode=mode,
            distributed=False
        )
        
        # Get one batch
        for batch in train_loader:
            if mode in ['simclr', 'moco']:
                view1, view2 = batch
                print(f"  View1 shape: {view1.shape}")
                print(f"  View2 shape: {view2.shape}")
            elif mode == 'dino':
                print(f"  Number of crops: {len(batch)}")
                for i, crop in enumerate(batch):
                    print(f"  Crop {i} shape: {crop.shape}")
            else:
                print(f"  Batch shape: {batch.shape}")
            break 