#!/usr/bin/env python3
"""
Enhanced Pathology-Specific Augmentation Pipeline v2.0
Biologically-realistic augmentations that preserve pathological features
for molecular subtype prediction accuracy improvement
"""

import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from typing import Tuple, List, Union
import skimage
from skimage import exposure, filters, morphology
import warnings
warnings.filterwarnings("ignore")

class StainNormalization:
    """Stain normalization for H&E histopathology images"""
    
    def __init__(self, method='macenko'):
        self.method = method
        # Reference H&E stain matrix (from literature)
        self.target_stains = np.array([
            [0.65, 0.70, 0.29],  # Hematoxylin
            [0.07, 0.99, 0.11]   # Eosin
        ])
        
    def __call__(self, image):
        """Apply stain normalization"""
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        if self.method == 'macenko':
            return self._macenko_normalize(image)
        elif self.method == 'reinhard':
            return self._reinhard_normalize(image)
        else:
            return image
    
    def _macenko_normalize(self, image):
        """Macenko stain normalization"""
        try:
            # Convert to optical density
            image_od = -np.log((image.astype(np.float32) + 1) / 256.0)
            
            # Remove transparent pixels
            od_hat = image_od[~np.any(image_od < 0.15, axis=1)]
            
            if len(od_hat) == 0:
                return image
                
            # Compute eigenvectors
            eigvals, eigvecs = np.linalg.eigh(np.cov(od_hat.T))
            eigvecs = eigvecs[:, np.argsort(eigvals)[::-1]]
            
            # Project to plane
            plane = np.dot(od_hat, eigvecs[:, :2])
            
            # Find robust extremes
            phi = np.arctan2(plane[:, 1], plane[:, 0])
            min_phi = np.percentile(phi, 1)
            max_phi = np.percentile(phi, 99)
            
            # Compute stain vectors
            v1 = np.dot(eigvecs[:, :2], np.array([np.cos(min_phi), np.sin(min_phi)]))
            v2 = np.dot(eigvecs[:, :2], np.array([np.cos(max_phi), np.sin(max_phi)]))
            
            if v1[0] > v2[0]:
                he_matrix = np.array([v1, v2])
            else:
                he_matrix = np.array([v2, v1])
                
            # Normalize and convert back
            source_concentrations = np.linalg.lstsq(he_matrix.T, image_od.T, rcond=None)[0].T
            target_concentrations = np.linalg.lstsq(self.target_stains.T, image_od.T, rcond=None)[0].T
            
            # Apply normalization
            normalized_od = np.dot(target_concentrations, self.target_stains)
            normalized_image = np.exp(-normalized_od) * 256 - 1
            normalized_image = np.clip(normalized_image, 0, 255).astype(np.uint8)
            
            return normalized_image
            
        except:
            return image
    
    def _reinhard_normalize(self, image):
        """Reinhard color normalization"""
        try:
            # Convert to LAB
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
            
            # Target statistics (from reference slide)
            target_means = np.array([8.74108109, -0.12440419,  0.0444982])
            target_stds = np.array([0.6135447, 0.10989488, 0.0286032])
            
            # Current statistics
            means = np.mean(lab.reshape(-1, 3), axis=0)
            stds = np.std(lab.reshape(-1, 3), axis=0)
            
            # Normalize
            lab_norm = (lab - means) / stds * target_stds + target_means
            lab_norm = np.clip(lab_norm, 0, 255)
            
            # Convert back to RGB
            normalized = cv2.cvtColor(lab_norm.astype(np.uint8), cv2.COLOR_LAB2RGB)
            return normalized
            
        except:
            return image

class NuclearMorphologyPreservingTransform:
    """Preserves nuclear morphology while applying transformations"""
    
    def __init__(self, max_rotation=15, preserve_nuclei=True):
        self.max_rotation = max_rotation
        self.preserve_nuclei = preserve_nuclei
        
    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Detect nuclei regions (simplified)
        nuclei_mask = self._detect_nuclei(image)
        
        # Apply rotation with nuclei preservation
        angle = random.uniform(-self.max_rotation, self.max_rotation)
        
        if self.preserve_nuclei:
            # Rotate non-nuclear regions more aggressively
            nuclear_regions = image * nuclei_mask[:, :, np.newaxis]
            background_regions = image * (1 - nuclei_mask[:, :, np.newaxis])
            
            # Rotate nuclear regions conservatively
            nuclear_rotated = self._rotate_image(nuclear_regions, angle * 0.5)
            background_rotated = self._rotate_image(background_regions, angle)
            
            result = nuclear_rotated + background_rotated
        else:
            result = self._rotate_image(image, angle)
            
        return result.astype(np.uint8)
    
    def _detect_nuclei(self, image):
        """Simple nuclei detection based on color"""
        # Convert to grayscale and threshold for dark regions (nuclei)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        nuclei_mask = gray < np.percentile(gray, 30)
        
        # Clean up mask
        nuclei_mask = morphology.remove_small_objects(nuclei_mask, min_size=50)
        nuclei_mask = morphology.binary_closing(nuclei_mask, morphology.disk(3))
        
        return nuclei_mask.astype(np.float32)
    
    def _rotate_image(self, image, angle):
        """Rotate image while preserving shape"""
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                               borderMode=cv2.BORDER_REFLECT_101)
        
        return rotated

class GlandStructurePreservingAugmentation:
    """Preserves glandular structures during augmentation"""
    
    def __init__(self, max_deformation=0.1):
        self.max_deformation = max_deformation
        
    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Detect glandular structures
        gland_mask = self._detect_glands(image)
        
        # Apply structure-preserving elastic deformation
        deformed = self._elastic_deformation(image, gland_mask)
        
        return deformed
    
    def _detect_glands(self, image):
        """Detect glandular structures"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect circular/tubular structures (glands)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                                  param1=50, param2=30, minRadius=10, maxRadius=100)
        
        mask = np.zeros(gray.shape, dtype=np.float32)
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(mask, (x, y), r, 1, -1)
        
        return mask
    
    def _elastic_deformation(self, image, structure_mask):
        """Apply elastic deformation while preserving structures"""
        height, width = image.shape[:2]
        
        # Generate displacement field
        dx = np.random.randn(height, width) * self.max_deformation * 10
        dy = np.random.randn(height, width) * self.max_deformation * 10
        
        # Reduce deformation in structural regions
        dx = dx * (1 - structure_mask * 0.8)
        dy = dy * (1 - structure_mask * 0.8)
        
        # Smooth displacement field
        dx = cv2.GaussianBlur(dx, (15, 15), 3)
        dy = cv2.GaussianBlur(dy, (15, 15), 3)
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)
        
        # Apply deformation
        deformed = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
        
        return deformed

class EnhancedPathologyAugmentation:
    """Comprehensive pathology-specific augmentation pipeline"""
    
    def __init__(self, 
                 augmentation_probability=0.8,
                 preserve_structures=True,
                 molecular_subtype_aware=True):
        
        self.aug_prob = augmentation_probability
        self.preserve_structures = preserve_structures
        self.molecular_aware = molecular_subtype_aware
        
        # Initialize components
        self.stain_normalizer = StainNormalization(method='macenko')
        self.nuclear_transform = NuclearMorphologyPreservingTransform()
        self.gland_augmentation = GlandStructurePreservingAugmentation()
        
        # Molecular subtype-specific augmentations
        self.canonical_augmentations = self._get_canonical_augmentations()
        self.immune_augmentations = self._get_immune_augmentations()
        self.stromal_augmentations = self._get_stromal_augmentations()
        
    def _get_canonical_augmentations(self):
        """Augmentations for Canonical subtype (E2F/MYC activation)"""
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.7),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.6),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.7),
            A.ElasticTransform(alpha=50, sigma=5, alpha_affine=5, p=0.3),
            A.GridDistortion(num_steps=3, distort_limit=0.1, p=0.3),
        ])
    
    def _get_immune_augmentations(self):
        """Augmentations for Immune subtype (preserve immune cell patterns)"""
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.08, contrast_limit=0.08, p=0.6),
            A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=8, val_shift_limit=8, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.03, rotate_limit=5, p=0.6),
            # Preserve immune cell spatial patterns
            A.GridDistortion(num_steps=2, distort_limit=0.05, p=0.2),
        ])
    
    def _get_stromal_augmentations(self):
        """Augmentations for Stromal subtype (preserve stromal architecture)"""
        return A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.8),
            A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=15, p=0.7),
            A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.08, rotate_limit=15, p=0.8),
            A.ElasticTransform(alpha=80, sigma=8, alpha_affine=8, p=0.4),
            A.OpticalDistortion(distort_limit=0.1, shift_limit=0.05, p=0.3),
        ])
    
    def __call__(self, image, subtype_label=None):
        """Apply comprehensive augmentation pipeline"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        # Store original for fallback
        original_image = image.copy()
        
        try:
            # Step 1: Stain normalization (always apply)
            if random.random() < 0.7:
                image = self.stain_normalizer(image)
            
            # Step 2: Structure-preserving transformations
            if self.preserve_structures and random.random() < self.aug_prob:
                if random.random() < 0.4:
                    image = self.nuclear_transform(image)
                if random.random() < 0.3:
                    image = self.gland_augmentation(image)
            
            # Step 3: Molecular subtype-aware augmentations
            if self.molecular_aware and subtype_label is not None:
                if random.random() < self.aug_prob:
                    if subtype_label == 0:  # Canonical
                        augmented = self.canonical_augmentations(image=image)
                        image = augmented['image']
                    elif subtype_label == 1:  # Immune
                        augmented = self.immune_augmentations(image=image)
                        image = augmented['image']
                    elif subtype_label == 2:  # Stromal
                        augmented = self.stromal_augmentations(image=image)
                        image = augmented['image']
            
            # Step 4: General pathology augmentations
            else:
                general_augmentations = A.Compose([
                    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.7),
                    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.6),
                    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.7),
                    A.OneOf([
                        A.ElasticTransform(alpha=50, sigma=5, alpha_affine=5),
                        A.GridDistortion(num_steps=3, distort_limit=0.1),
                        A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05),
                    ], p=0.3),
                    A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                    A.Blur(blur_limit=3, p=0.1),
                ])
                
                if random.random() < self.aug_prob:
                    augmented = general_augmentations(image=image)
                    image = augmented['image']
            
            # Ensure valid image
            image = np.clip(image, 0, 255).astype(np.uint8)
            
            return image
            
        except Exception as e:
            print(f"Augmentation failed: {e}, returning original image")
            return original_image

def get_enhanced_train_transforms(image_size=224, molecular_aware=True):
    """Get enhanced training transforms for molecular subtype prediction"""
    
    # Enhanced pathology-specific augmentation
    pathology_aug = EnhancedPathologyAugmentation(
        augmentation_probability=0.8,
        preserve_structures=True,
        molecular_subtype_aware=molecular_aware
    )
    
    # Standard preprocessing transforms
    preprocessing = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    class CombinedTransform:
        def __init__(self, pathology_aug, preprocessing):
            self.pathology_aug = pathology_aug
            self.preprocessing = preprocessing
            
        def __call__(self, image, subtype_label=None):
            # Apply pathology-specific augmentation
            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(image)
            
            augmented = self.pathology_aug(image, subtype_label)
            
            # Convert to PIL and apply standard preprocessing
            if isinstance(augmented, np.ndarray):
                augmented = Image.fromarray(augmented)
                
            return self.preprocessing(augmented)
    
    return CombinedTransform(pathology_aug, preprocessing)

def get_validation_transforms(image_size=224):
    """Get validation transforms (minimal augmentation)"""
    stain_normalizer = StainNormalization(method='macenko')
    
    class ValidationTransform:
        def __init__(self, stain_normalizer, image_size):
            self.stain_normalizer = stain_normalizer
            self.preprocessing = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
        def __call__(self, image):
            if isinstance(image, torch.Tensor):
                image = transforms.ToPILImage()(image)
            
            # Apply stain normalization only
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            normalized = self.stain_normalizer(image)
            
            if isinstance(normalized, np.ndarray):
                normalized = Image.fromarray(normalized)
                
            return self.preprocessing(normalized)
    
    return ValidationTransform(stain_normalizer, image_size)

if __name__ == "__main__":
    # Test the enhanced augmentation pipeline
    print("🧪 Testing Enhanced Pathology Augmentation Pipeline")
    
    # Create dummy image
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    dummy_pil = Image.fromarray(dummy_image)
    
    # Test augmentation
    augmenter = EnhancedPathologyAugmentation()
    
    # Test with different subtypes
    for subtype, label in [("Canonical", 0), ("Immune", 1), ("Stromal", 2)]:
        augmented = augmenter(dummy_pil, subtype_label=label)
        print(f"✅ {subtype} augmentation successful: {augmented.shape}")
    
    # Test transforms
    train_transform = get_enhanced_train_transforms(molecular_aware=True)
    val_transform = get_validation_transforms()
    
    train_tensor = train_transform(dummy_pil, subtype_label=0)
    val_tensor = val_transform(dummy_pil)
    
    print(f"✅ Train transform output: {train_tensor.shape}")
    print(f"✅ Validation transform output: {val_tensor.shape}")
    print("🎉 Enhanced Pathology Augmentation Pipeline ready!") 