"""
Pathology-specific data augmentation for CRC molecular subtyping
Implements domain-aware augmentations to improve model robustness
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates
import random
from skimage import color, exposure


class PathologyAugmentation:
    """Advanced augmentation specifically designed for H&E pathology images"""
    
    def __init__(self):
        self.stain_augmenter = StainAugmentation()
        self.artifact_generator = ArtifactSimulator()
        self.tissue_deformation = TissueDeformation()
        
    def augment(self, image, severity='medium'):
        """Apply pathology-specific augmentations"""
        if isinstance(image, np.ndarray):
            img = image.copy()
        else:
            img = np.array(image)
            
        # Apply augmentations with probabilities
        augmentations = []
        
        # Stain variations (very common in real data)
        if random.random() < 0.7:
            img = self.stain_augmenter.random_he_variation(img, severity)
            augmentations.append('stain_variation')
            
        # Tissue deformations
        if random.random() < 0.3:
            img = self.tissue_deformation.elastic_transform(img)
            augmentations.append('elastic_deformation')
            
        # Common artifacts
        if random.random() < 0.15:
            img = self.artifact_generator.add_fold(img)
            augmentations.append('tissue_fold')
            
        if random.random() < 0.1:
            img = self.artifact_generator.add_bubble(img)
            augmentations.append('air_bubble')
            
        if random.random() < 0.05:
            img = self.artifact_generator.add_blur_region(img)
            augmentations.append('focus_blur')
            
        # Brightness/contrast variations
        if random.random() < 0.3:
            img = self.stain_augmenter.adjust_brightness_contrast(img)
            augmentations.append('brightness_contrast')
            
        return img, augmentations


class StainAugmentation:
    """H&E stain-specific augmentations"""
    
    def __init__(self):
        # Stain matrices for H&E deconvolution (from literature)
        self.stain_matrix_reference = np.array([
            [0.644, 0.717, 0.267],  # Hematoxylin
            [0.093, 0.954, 0.283],  # Eosin
            [0.636, 0.000, 0.772]   # DAB (not used in H&E)
        ])
        
    def random_he_variation(self, image, severity='medium'):
        """Apply realistic H&E stain variations"""
        severity_params = {
            'light': (0.9, 1.1),
            'medium': (0.8, 1.2),
            'strong': (0.7, 1.3)
        }
        
        alpha_range = severity_params.get(severity, severity_params['medium'])
        
        # Convert to OD (optical density) space
        od = self.rgb_to_od(image)
        
        # Separate stains
        h_channel, e_channel = self.separate_stains(od)
        
        # Apply random variations
        h_factor = np.random.uniform(*alpha_range)
        e_factor = np.random.uniform(*alpha_range)
        
        # Modify stain concentrations
        h_channel_modified = h_channel * h_factor
        e_channel_modified = e_channel * e_factor
        
        # Reconstruct image
        modified_od = self.combine_stains(h_channel_modified, e_channel_modified)
        modified_rgb = self.od_to_rgb(modified_od)
        
        return np.clip(modified_rgb, 0, 255).astype(np.uint8)
    
    def rgb_to_od(self, image):
        """Convert RGB to optical density"""
        image = image.astype(np.float32) + 1
        od = -np.log(image / 255.0)
        return np.maximum(od, 1e-6)
    
    def od_to_rgb(self, od):
        """Convert optical density back to RGB"""
        rgb = 255.0 * np.exp(-od)
        return rgb
    
    def separate_stains(self, od):
        """Separate H and E channels using color deconvolution"""
        # Simplified separation - in practice would use full deconvolution
        h_channel = od[:, :, 0] * 0.7 + od[:, :, 1] * 0.3
        e_channel = od[:, :, 1] * 0.6 + od[:, :, 2] * 0.4
        return h_channel, e_channel
    
    def combine_stains(self, h_channel, e_channel):
        """Combine separated stains back to RGB"""
        od = np.zeros((h_channel.shape[0], h_channel.shape[1], 3))
        od[:, :, 0] = h_channel * 0.7
        od[:, :, 1] = h_channel * 0.3 + e_channel * 0.6
        od[:, :, 2] = e_channel * 0.4
        return od
    
    def adjust_brightness_contrast(self, image, brightness_range=(-20, 20), contrast_range=(0.8, 1.2)):
        """Adjust brightness and contrast within realistic ranges"""
        brightness = np.random.uniform(*brightness_range)
        contrast = np.random.uniform(*contrast_range)
        
        # Apply adjustments
        adjusted = image.astype(np.float32)
        adjusted = adjusted * contrast + brightness
        
        return np.clip(adjusted, 0, 255).astype(np.uint8)


class ArtifactSimulator:
    """Simulate common pathology artifacts"""
    
    def add_fold(self, image):
        """Add tissue fold artifact"""
        h, w = image.shape[:2]
        
        # Create fold line
        start_point = (np.random.randint(0, w), np.random.randint(0, h//2))
        end_point = (np.random.randint(0, w), np.random.randint(h//2, h))
        
        # Create fold mask
        fold_mask = np.zeros((h, w), dtype=np.float32)
        thickness = np.random.randint(5, 20)
        cv2.line(fold_mask, start_point, end_point, 1.0, thickness)
        
        # Gaussian blur the mask
        fold_mask = gaussian_filter(fold_mask, sigma=thickness/4)
        fold_mask = np.expand_dims(fold_mask, axis=2)
        
        # Darken fold area
        darkened = image.astype(np.float32) * (1 - fold_mask * 0.4)
        
        return darkened.astype(np.uint8)
    
    def add_bubble(self, image):
        """Add air bubble artifact"""
        h, w = image.shape[:2]
        
        # Random bubble parameters
        center = (np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4))
        radius = np.random.randint(10, min(h, w)//8)
        
        # Create bubble mask
        y, x = np.ogrid[:h, :w]
        mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
        
        # Create gradient effect
        dist_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2)
        gradient = np.clip(1 - (dist_from_center - radius*0.7) / (radius*0.3), 0, 1)
        
        # Apply bubble effect
        bubble_image = image.copy()
        bubble_image[mask] = (
            image[mask] * 0.3 + 
            np.array([240, 240, 250]) * 0.7
        ).astype(np.uint8)
        
        return bubble_image
    
    def add_blur_region(self, image):
        """Add out-of-focus region"""
        h, w = image.shape[:2]
        
        # Create random blur region
        blur_center = (np.random.randint(w//4, 3*w//4), np.random.randint(h//4, 3*h//4))
        blur_radius = np.random.randint(30, min(h, w)//4)
        
        # Create smooth transition mask
        y, x = np.ogrid[:h, :w]
        dist = np.sqrt((x - blur_center[0])**2 + (y - blur_center[1])**2)
        mask = np.clip(1 - (dist - blur_radius) / blur_radius, 0, 1)
        mask = gaussian_filter(mask, sigma=10)
        
        # Apply varying blur
        blurred = cv2.GaussianBlur(image, (21, 21), 0)
        
        # Blend based on mask
        result = (image * (1 - mask[:, :, np.newaxis]) + 
                 blurred * mask[:, :, np.newaxis]).astype(np.uint8)
        
        return result


class TissueDeformation:
    """Simulate tissue deformations during slide preparation"""
    
    def elastic_transform(self, image, alpha=50, sigma=5):
        """Apply elastic deformation to simulate tissue stretching/compression"""
        shape = image.shape[:2]
        
        # Random displacement fields
        dx = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        dy = gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma) * alpha
        
        # Create coordinate arrays
        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        
        # Apply transformation to each channel
        if len(image.shape) == 3:
            transformed = np.zeros_like(image)
            for i in range(image.shape[2]):
                transformed[:, :, i] = map_coordinates(
                    image[:, :, i], indices, order=1, mode='reflect'
                ).reshape(shape)
        else:
            transformed = map_coordinates(
                image, indices, order=1, mode='reflect'
            ).reshape(shape)
            
        return transformed 