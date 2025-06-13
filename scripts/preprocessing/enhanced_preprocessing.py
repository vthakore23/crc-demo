#!/usr/bin/env python3
"""
Enhanced Preprocessing Pipeline
Implements practical preprocessing improvements for better accuracy
"""

import numpy as np
import cv2
from PIL import Image
import torch
from typing import Tuple, List, Optional, Dict
from pathlib import Path
import staintools
from scipy import ndimage
import skimage.filters
from concurrent.futures import ProcessPoolExecutor
import json


class StainNormalizer:
    """Advanced H&E stain normalization"""
    
    def __init__(self, method: str = 'macenko', reference_image_path: Optional[str] = None):
        self.method = method
        self.normalizer = self._create_normalizer()
        
        if reference_image_path:
            reference_image = cv2.imread(reference_image_path)
            reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
            self.fit(reference_image)
    
    def _create_normalizer(self):
        """Create stain normalizer based on method"""
        if self.method == 'macenko':
            return staintools.StainNormalizer(method='macenko')
        elif self.method == 'vahadane':
            return staintools.StainNormalizer(method='vahadane')
        elif self.method == 'reinhard':
            return staintools.ReinhardColorNormalizer()
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def fit(self, reference_image: np.ndarray):
        """Fit normalizer to reference image"""
        # Ensure image is uint8
        if reference_image.dtype != np.uint8:
            reference_image = (reference_image * 255).astype(np.uint8)
        
        # Standardize brightness first
        reference_image = staintools.LuminosityStandardizer.standardize(reference_image)
        self.normalizer.fit(reference_image)
        self.fitted = True
    
    def transform(self, image: np.ndarray) -> np.ndarray:
        """Transform image using fitted normalizer"""
        if not hasattr(self, 'fitted'):
            raise RuntimeError("Normalizer must be fitted before transform")
        
        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        try:
            # Standardize brightness
            image = staintools.LuminosityStandardizer.standardize(image)
            # Normalize stain
            normalized = self.normalizer.transform(image)
            return normalized
        except Exception as e:
            print(f"Stain normalization failed: {e}, returning original image")
            return image


class QualityControl:
    """Quality control for histopathology images"""
    
    @staticmethod
    def check_tissue_content(image: np.ndarray, min_tissue_percent: float = 0.2) -> bool:
        """Check if image contains sufficient tissue"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate tissue percentage
        tissue_pixels = np.sum(binary < 128)  # Assuming tissue is darker
        total_pixels = binary.size
        tissue_percent = tissue_pixels / total_pixels
        
        return tissue_percent >= min_tissue_percent
    
    @staticmethod
    def check_focus_quality(image: np.ndarray, threshold: float = 100.0) -> bool:
        """Check if image is in focus using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return variance > threshold
    
    @staticmethod
    def check_brightness(image: np.ndarray, min_brightness: float = 50, max_brightness: float = 200) -> bool:
        """Check if image brightness is within acceptable range"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray)
        return min_brightness <= mean_brightness <= max_brightness
    
    @staticmethod
    def detect_artifacts(image: np.ndarray, artifact_threshold: float = 0.1) -> bool:
        """Detect common artifacts in histopathology images"""
        # Check for pen marks (usually very dark or colored)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Detect blue pen marks
        blue_lower = np.array([100, 50, 50])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        
        # Detect black pen marks
        black_lower = np.array([0, 0, 0])
        black_upper = np.array([180, 255, 30])
        black_mask = cv2.inRange(hsv, black_lower, black_upper)
        
        # Combine masks
        artifact_mask = cv2.bitwise_or(blue_mask, black_mask)
        artifact_percent = np.sum(artifact_mask > 0) / artifact_mask.size
        
        return artifact_percent < artifact_threshold


class TissueDetector:
    """Advanced tissue detection and extraction"""
    
    def __init__(self, min_tissue_size: int = 1000):
        self.min_tissue_size = min_tissue_size
    
    def detect_tissue(self, image: np.ndarray) -> np.ndarray:
        """Detect tissue regions in the image"""
        # Convert to HSV for better tissue detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Create tissue mask
        # Tissue typically has specific hue and saturation ranges
        tissue_mask = cv2.inRange(hsv, 
                                 np.array([0, 20, 30]), 
                                 np.array([180, 255, 255]))
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
        
        # Remove small regions
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(tissue_mask, connectivity=8)
        
        # Keep only large tissue regions
        cleaned_mask = np.zeros_like(tissue_mask)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_tissue_size:
                cleaned_mask[labels == i] = 255
        
        return cleaned_mask
    
    def extract_tissue_patches(self, image: np.ndarray, patch_size: int = 256, 
                             stride: int = 128, min_tissue_ratio: float = 0.5) -> List[np.ndarray]:
        """Extract patches containing tissue"""
        tissue_mask = self.detect_tissue(image)
        patches = []
        
        h, w = image.shape[:2]
        
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                # Extract patch
                patch = image[y:y+patch_size, x:x+patch_size]
                patch_mask = tissue_mask[y:y+patch_size, x:x+patch_size]
                
                # Check tissue ratio
                tissue_ratio = np.sum(patch_mask > 0) / patch_mask.size
                
                if tissue_ratio >= min_tissue_ratio:
                    patches.append(patch)
        
        return patches


class ColorAugmenter:
    """Color augmentation specific to H&E images"""
    
    @staticmethod
    def augment_hed(image: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """Augment H&E stain concentrations"""
        # Separate stains using color deconvolution
        hed = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
        
        # Add random perturbations
        perturbation = np.random.normal(0, alpha, hed.shape)
        hed_augmented = hed + perturbation * 255
        
        # Clip values
        hed_augmented = np.clip(hed_augmented, 0, 255).astype(np.uint8)
        
        # Convert back to RGB
        augmented = cv2.cvtColor(hed_augmented, cv2.COLOR_Lab2RGB)
        
        return augmented
    
    @staticmethod
    def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image contrast"""
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        adjusted = (image - mean) * factor + mean
        return np.clip(adjusted, 0, 255).astype(np.uint8)


class EnhancedPreprocessor:
    """Complete preprocessing pipeline"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.stain_normalizer = StainNormalizer(
            method=config.get('stain_norm_method', 'macenko')
        )
        self.quality_control = QualityControl()
        self.tissue_detector = TissueDetector(
            min_tissue_size=config.get('min_tissue_size', 1000)
        )
        self.color_augmenter = ColorAugmenter()
        
        # Fit stain normalizer if reference provided
        if 'reference_image' in config:
            ref_image = cv2.imread(config['reference_image'])
            ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
            self.stain_normalizer.fit(ref_image)
    
    def preprocess_image(self, image_path: str) -> Optional[np.ndarray]:
        """Complete preprocessing pipeline for a single image"""
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Quality control
        if not self._pass_quality_control(image):
            return None
        
        # Stain normalization
        if self.config.get('apply_stain_norm', True) and hasattr(self.stain_normalizer, 'fitted'):
            try:
                image = self.stain_normalizer.transform(image)
            except:
                pass  # Skip normalization if it fails
        
        # Tissue detection and cropping
        if self.config.get('crop_to_tissue', True):
            tissue_mask = self.tissue_detector.detect_tissue(image)
            
            # Find bounding box of tissue
            coords = np.where(tissue_mask > 0)
            if len(coords[0]) > 0:
                y_min, y_max = coords[0].min(), coords[0].max()
                x_min, x_max = coords[1].min(), coords[1].max()
                
                # Add padding
                padding = self.config.get('tissue_padding', 50)
                y_min = max(0, y_min - padding)
                y_max = min(image.shape[0], y_max + padding)
                x_min = max(0, x_min - padding)
                x_max = min(image.shape[1], x_max + padding)
                
                # Crop to tissue region
                image = image[y_min:y_max, x_min:x_max]
        
        return image
    
    def _pass_quality_control(self, image: np.ndarray) -> bool:
        """Run quality control checks"""
        checks = {
            'tissue_content': self.quality_control.check_tissue_content(
                image, self.config.get('min_tissue_percent', 0.2)
            ),
            'focus_quality': self.quality_control.check_focus_quality(
                image, self.config.get('focus_threshold', 100.0)
            ),
            'brightness': self.quality_control.check_brightness(
                image, self.config.get('min_brightness', 50), 
                self.config.get('max_brightness', 200)
            ),
            'artifacts': self.quality_control.detect_artifacts(
                image, self.config.get('artifact_threshold', 0.1)
            )
        }
        
        # Log failed checks
        failed_checks = [k for k, v in checks.items() if not v]
        if failed_checks:
            print(f"Quality control failed: {failed_checks}")
        
        return all(checks.values())
    
    def process_batch(self, image_paths: List[str], num_workers: int = 4) -> List[Optional[np.ndarray]]:
        """Process multiple images in parallel"""
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(executor.map(self.preprocess_image, image_paths))
        
        return results
    
    def extract_patches(self, image: np.ndarray, patch_size: int = 256, 
                       stride: int = 128) -> List[np.ndarray]:
        """Extract patches from preprocessed image"""
        return self.tissue_detector.extract_tissue_patches(
            image, patch_size, stride, 
            min_tissue_ratio=self.config.get('min_tissue_ratio', 0.5)
        )


def create_preprocessing_config() -> Dict:
    """Create default preprocessing configuration"""
    config = {
        # Stain normalization
        'stain_norm_method': 'macenko',  # 'macenko', 'vahadane', 'reinhard'
        'apply_stain_norm': True,
        'reference_image': 'data/reference_he_image.png',
        
        # Quality control
        'min_tissue_percent': 0.2,
        'focus_threshold': 100.0,
        'min_brightness': 50,
        'max_brightness': 200,
        'artifact_threshold': 0.1,
        
        # Tissue detection
        'crop_to_tissue': True,
        'min_tissue_size': 1000,
        'tissue_padding': 50,
        
        # Patch extraction
        'patch_size': 256,
        'stride': 128,
        'min_tissue_ratio': 0.5
    }
    
    return config


if __name__ == "__main__":
    # Example usage
    config = create_preprocessing_config()
    preprocessor = EnhancedPreprocessor(config)
    
    # Process single image
    image_path = "data/sample_wsi_patch.png"
    processed_image = preprocessor.preprocess_image(image_path)
    
    if processed_image is not None:
        print(f"Processed image shape: {processed_image.shape}")
        
        # Extract patches
        patches = preprocessor.extract_patches(processed_image)
        print(f"Extracted {len(patches)} patches")
        
        # Save preprocessed image
        cv2.imwrite("data/preprocessed_sample.png", 
                   cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)) 