#!/usr/bin/env python3
"""
Whole Slide Image (WSI) Processing for Molecular Subtype Classification
Handles multi-gigapixel WSI files with intelligent patch extraction and preprocessing
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
from dataclasses import dataclass
import concurrent.futures
from functools import lru_cache
import gc

# Handle different WSI libraries
try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    logging.warning("OpenSlide not available. WSI support limited.")

try:
    from skimage import filters, morphology, measure, segmentation
    from skimage.color import rgb2gray, hed2rgb
    from skimage.feature import local_binary_pattern
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logging.warning("Scikit-image not available. Advanced image processing limited.")

import torch
import torchvision.transforms as transforms
from PIL import Image

logger = logging.getLogger(__name__)

@dataclass
class WSIMetadata:
    """Metadata for whole slide image"""
    filename: str
    dimensions: Tuple[int, int]
    level_count: int
    level_dimensions: List[Tuple[int, int]]
    level_downsamples: List[float]
    magnification: float
    pixel_spacing: Tuple[float, float]
    vendor: str
    scanner: str

@dataclass
class PatchInfo:
    """Information about extracted patch"""
    coordinates: Tuple[int, int]  # (x, y) at level 0
    level: int
    size: Tuple[int, int]
    tissue_ratio: float
    quality_score: float
    predicted_subtype: Optional[str] = None
    confidence: Optional[float] = None

class StainNormalizer:
    """Stain normalization using Macenko method"""
    
    def __init__(self, target_image_path: Optional[str] = None):
        self.target_stains = None
        self.target_concentrations = None
        
        if target_image_path and Path(target_image_path).exists():
            self._set_target_image(target_image_path)
    
    def _set_target_image(self, image_path: str):
        """Set target image for normalization"""
        try:
            target_image = np.array(Image.open(image_path))
            self.target_stains, self.target_concentrations = self._get_stain_matrix(target_image)
            logger.info(f"Stain normalization target set: {image_path}")
        except Exception as e:
            logger.warning(f"Could not set stain target: {e}")
    
    def _get_stain_matrix(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract stain matrix using Macenko method"""
        if not SKIMAGE_AVAILABLE:
            return None, None
            
        # Convert to optical density
        image = image.astype(np.float32)
        image = np.maximum(image, 1)
        od = -np.log(image / 255.0)
        
        # Remove transparent pixels
        od_flat = od.reshape(-1, 3)
        mask = np.sum(od_flat, axis=1) > 0.15
        od_pixels = od_flat[mask]
        
        if len(od_pixels) < 100:
            return None, None
        
        # SVD to get stain directions
        _, _, V = np.linalg.svd(od_pixels, full_matrices=False)
        V = V[:2, :]  # First two components
        
        # Make sure vectors point in correct direction
        if V[0, 0] < 0:
            V[0, :] *= -1
        if V[1, 0] < 0:
            V[1, :] *= -1
            
        # Robust angle estimation
        angles = np.arctan2(V[:, 1], V[:, 0])
        angles = np.sort(angles)
        
        # Ensure minimum angle separation
        if angles[1] - angles[0] < 0.3:
            angles[1] = angles[0] + 0.5
        
        # Reconstruct stain matrix
        stain_matrix = np.array([
            [np.cos(angles[0]), np.cos(angles[1])],
            [np.sin(angles[0]), np.sin(angles[1])]
        ]).T
        
        # Get concentrations
        concentrations = np.linalg.lstsq(stain_matrix.T, od_pixels.T, rcond=None)[0]
        
        return stain_matrix, concentrations
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize image staining"""
        if self.target_stains is None:
            return image
            
        try:
            # Get source stain matrix
            source_stains, source_concentrations = self._get_stain_matrix(image)
            
            if source_stains is None:
                return image
            
            # Convert to optical density
            image_float = image.astype(np.float32)
            image_float = np.maximum(image_float, 1)
            od = -np.log(image_float / 255.0)
            
            # Separate stains
            od_flat = od.reshape(-1, 3)
            concentrations = np.linalg.lstsq(source_stains.T, od_flat.T, rcond=None)[0]
            
            # Normalize concentrations
            max_conc = np.percentile(concentrations, 99, axis=1, keepdims=True)
            concentrations = concentrations / (max_conc + 1e-6)
            
            # Apply target stain matrix
            normalized_od = self.target_stains.T @ concentrations
            normalized_od = normalized_od.T.reshape(od.shape)
            
            # Convert back to RGB
            normalized_image = np.exp(-normalized_od) * 255
            normalized_image = np.clip(normalized_image, 0, 255).astype(np.uint8)
            
            return normalized_image
            
        except Exception as e:
            logger.warning(f"Stain normalization failed: {e}")
            return image

class TissueSegmentation:
    """Tissue detection and segmentation for WSI"""
    
    def __init__(self, tissue_threshold: float = 0.8):
        self.tissue_threshold = tissue_threshold
    
    def segment_tissue(self, image: np.ndarray, return_mask: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Segment tissue regions from background"""
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu thresholding
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
        
        # Remove small components
        num_labels, labels = cv2.connectedComponents(cleaned)
        for i in range(1, num_labels):
            mask = labels == i
            if np.sum(mask) < 1000:  # Remove small components
                cleaned[mask] = 0
        
        # Invert mask (tissue = 255, background = 0)
        tissue_mask = 255 - cleaned
        
        if return_mask:
            return tissue_mask
        else:
            # Apply mask to original image
            result = image.copy()
            if len(result.shape) == 3:
                result[tissue_mask == 0] = [255, 255, 255]  # White background
            else:
                result[tissue_mask == 0] = 255
            return result, tissue_mask
    
    def calculate_tissue_ratio(self, image: np.ndarray) -> float:
        """Calculate tissue to total area ratio"""
        tissue_mask = self.segment_tissue(image, return_mask=True)
        tissue_pixels = np.sum(tissue_mask > 0)
        total_pixels = tissue_mask.shape[0] * tissue_mask.shape[1]
        return tissue_pixels / total_pixels

class WSIProcessor:
    """Main WSI processing class for molecular subtype analysis"""
    
    def __init__(self, 
                 patch_size: int = 224,
                 patch_level: int = 0,
                 overlap: float = 0.0,
                 tissue_threshold: float = 0.5,
                 quality_threshold: float = 0.3,
                 max_patches: int = 1000,
                 stain_normalize: bool = True):
        
        self.patch_size = patch_size
        self.patch_level = patch_level
        self.overlap = overlap
        self.tissue_threshold = tissue_threshold
        self.quality_threshold = quality_threshold
        self.max_patches = max_patches
        self.stain_normalize = stain_normalize
        
        # Initialize components
        self.tissue_segmenter = TissueSegmentation(tissue_threshold)
        self.stain_normalizer = StainNormalizer() if stain_normalize else None
        
        # Patch extraction transform
        self.patch_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((patch_size, patch_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"WSI Processor initialized - patch_size: {patch_size}, level: {patch_level}")
    
    def load_wsi(self, wsi_path: str) -> Optional['openslide.OpenSlide']:
        """Load WSI file"""
        if not OPENSLIDE_AVAILABLE:
            logger.error("OpenSlide not available for WSI loading")
            return None
            
        try:
            slide = openslide.OpenSlide(str(wsi_path))
            return slide
        except Exception as e:
            logger.error(f"Failed to load WSI {wsi_path}: {e}")
            return None
    
    def get_wsi_metadata(self, slide: 'openslide.OpenSlide') -> WSIMetadata:
        """Extract metadata from WSI"""
        try:
            properties = slide.properties
            
            # Get magnification
            try:
                magnification = float(properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER, 40))
            except:
                magnification = 40.0
            
            # Get pixel spacing
            try:
                mpp_x = float(properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
                mpp_y = float(properties.get(openslide.PROPERTY_NAME_MPP_Y, 0.25))
                pixel_spacing = (mpp_x, mpp_y)
            except:
                pixel_spacing = (0.25, 0.25)
            
            metadata = WSIMetadata(
                filename="",
                dimensions=slide.dimensions,
                level_count=slide.level_count,
                level_dimensions=slide.level_dimensions,
                level_downsamples=slide.level_downsamples,
                magnification=magnification,
                pixel_spacing=pixel_spacing,
                vendor=properties.get(openslide.PROPERTY_NAME_VENDOR, "Unknown"),
                scanner=properties.get("aperio.ScanScope ID", "Unknown")
            )
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to extract WSI metadata: {e}")
            return None
    
    def generate_patch_coordinates(self, slide: 'openslide.OpenSlide') -> List[Tuple[int, int]]:
        """Generate coordinates for patch extraction"""
        level_dimension = slide.level_dimensions[self.patch_level]
        downsample = slide.level_downsamples[self.patch_level]
        
        # Calculate step size
        step_size = int(self.patch_size * (1 - self.overlap))
        
        coordinates = []
        for y in range(0, level_dimension[1] - self.patch_size, step_size):
            for x in range(0, level_dimension[0] - self.patch_size, step_size):
                # Convert to level 0 coordinates
                x_level0 = int(x * downsample)
                y_level0 = int(y * downsample)
                coordinates.append((x_level0, y_level0))
        
        return coordinates
    
    def extract_patch(self, slide: 'openslide.OpenSlide', coordinates: Tuple[int, int]) -> Optional[np.ndarray]:
        """Extract a single patch from WSI"""
        try:
            # Extract patch
            patch_size_level0 = int(self.patch_size * slide.level_downsamples[self.patch_level])
            patch = slide.read_region(coordinates, self.patch_level, (self.patch_size, self.patch_size))
            
            # Convert to RGB
            patch_rgb = patch.convert('RGB')
            patch_array = np.array(patch_rgb)
            
            # Apply stain normalization if enabled
            if self.stain_normalizer:
                patch_array = self.stain_normalizer.normalize(patch_array)
            
            return patch_array
            
        except Exception as e:
            logger.warning(f"Failed to extract patch at {coordinates}: {e}")
            return None
    
    def assess_patch_quality(self, patch: np.ndarray) -> Tuple[float, float]:
        """Assess patch quality and tissue content"""
        
        # Calculate tissue ratio
        tissue_ratio = self.tissue_segmenter.calculate_tissue_ratio(patch)
        
        # Calculate quality score based on multiple factors
        quality_score = 0.0
        
        # 1. Tissue content (40% weight)
        quality_score += 0.4 * tissue_ratio
        
        # 2. Contrast and sharpness (30% weight)
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        contrast = np.std(gray) / 255.0
        
        # Laplacian variance for sharpness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness = min(laplacian_var / 1000.0, 1.0)  # Normalize
        
        quality_score += 0.2 * contrast + 0.1 * sharpness
        
        # 3. Color distribution (20% weight)
        # Avoid too bright (overexposed) or too dark (underexposed) patches
        mean_intensity = np.mean(patch)
        if 50 < mean_intensity < 200:
            color_quality = 1.0
        else:
            color_quality = max(0, 1.0 - abs(mean_intensity - 125) / 125)
        
        quality_score += 0.2 * color_quality
        
        # 4. Avoid artifacts (10% weight)
        # Check for pen marks (very dark or colored regions)
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        pen_mask = (hsv[:, :, 2] < 50) | (hsv[:, :, 1] > 200)  # Very dark or very saturated
        artifact_ratio = np.sum(pen_mask) / (patch.shape[0] * patch.shape[1])
        artifact_penalty = max(0, 1.0 - artifact_ratio * 10)
        
        quality_score += 0.1 * artifact_penalty
        
        return tissue_ratio, quality_score
    
    def select_representative_patches(self, 
                                   patch_infos: List[PatchInfo], 
                                   max_patches: Optional[int] = None) -> List[PatchInfo]:
        """Select most representative patches using intelligent sampling"""
        
        if max_patches is None:
            max_patches = self.max_patches
        
        if len(patch_infos) <= max_patches:
            return patch_infos
        
        # Filter by quality and tissue content
        good_patches = [p for p in patch_infos 
                       if p.quality_score >= self.quality_threshold 
                       and p.tissue_ratio >= self.tissue_threshold]
        
        if len(good_patches) <= max_patches:
            return good_patches
        
        # Sort by quality score and take top patches
        good_patches.sort(key=lambda x: x.quality_score, reverse=True)
        
        # Use stratified sampling to ensure spatial diversity
        selected_patches = []
        
        # Divide image into grid and sample from each cell
        coords = [(p.coordinates[0], p.coordinates[1]) for p in good_patches]
        if coords:
            min_x, min_y = np.min(coords, axis=0)
            max_x, max_y = np.max(coords, axis=0)
            
            grid_size = int(np.sqrt(max_patches))
            step_x = (max_x - min_x) / grid_size if max_x > min_x else 1
            step_y = (max_y - min_y) / grid_size if max_y > min_y else 1
            
            for i in range(grid_size):
                for j in range(grid_size):
                    # Define grid cell bounds
                    cell_min_x = min_x + i * step_x
                    cell_max_x = min_x + (i + 1) * step_x
                    cell_min_y = min_y + j * step_y
                    cell_max_y = min_y + (j + 1) * step_y
                    
                    # Find patches in this cell
                    cell_patches = [p for p in good_patches 
                                  if cell_min_x <= p.coordinates[0] < cell_max_x 
                                  and cell_min_y <= p.coordinates[1] < cell_max_y]
                    
                    # Select best patch from cell
                    if cell_patches:
                        best_patch = max(cell_patches, key=lambda x: x.quality_score)
                        selected_patches.append(best_patch)
                        
                        if len(selected_patches) >= max_patches:
                            break
                
                if len(selected_patches) >= max_patches:
                    break
        
        # If still not enough patches, add remaining high-quality ones
        if len(selected_patches) < max_patches:
            remaining_patches = [p for p in good_patches if p not in selected_patches]
            remaining_patches.sort(key=lambda x: x.quality_score, reverse=True)
            selected_patches.extend(remaining_patches[:max_patches - len(selected_patches)])
        
        return selected_patches[:max_patches]
    
    def process_wsi(self, wsi_path: str, output_dir: Optional[str] = None) -> Dict:
        """Complete WSI processing pipeline"""
        
        logger.info(f"Processing WSI: {wsi_path}")
        
        # Load WSI
        slide = self.load_wsi(wsi_path)
        if slide is None:
            return {'error': 'Failed to load WSI'}
        
        try:
            # Get metadata
            metadata = self.get_wsi_metadata(slide)
            
            # Generate patch coordinates
            coordinates = self.generate_patch_coordinates(slide)
            logger.info(f"Generated {len(coordinates)} patch coordinates")
            
            # Extract and assess patches
            patch_infos = []
            
            for i, coord in enumerate(coordinates):
                if i % 100 == 0:
                    logger.info(f"Processing patch {i}/{len(coordinates)}")
                
                # Extract patch
                patch = self.extract_patch(slide, coord)
                if patch is None:
                    continue
                
                # Assess quality
                tissue_ratio, quality_score = self.assess_patch_quality(patch)
                
                patch_info = PatchInfo(
                    coordinates=coord,
                    level=self.patch_level,
                    size=(self.patch_size, self.patch_size),
                    tissue_ratio=tissue_ratio,
                    quality_score=quality_score
                )
                
                patch_infos.append(patch_info)
            
            # Select representative patches
            selected_patches = self.select_representative_patches(patch_infos)
            logger.info(f"Selected {len(selected_patches)} representative patches")
            
            # Save results if output directory provided
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                # Save patch information
                patch_data = []
                for patch_info in selected_patches:
                    patch_data.append({
                        'coordinates': patch_info.coordinates,
                        'level': patch_info.level,
                        'size': patch_info.size,
                        'tissue_ratio': patch_info.tissue_ratio,
                        'quality_score': patch_info.quality_score
                    })
                
                with open(output_path / 'patch_info.json', 'w') as f:
                    json.dump(patch_data, f, indent=2)
                
                # Save metadata
                metadata_dict = {
                    'filename': Path(wsi_path).name,
                    'dimensions': metadata.dimensions,
                    'level_count': metadata.level_count,
                    'level_dimensions': metadata.level_dimensions,
                    'level_downsamples': metadata.level_downsamples,
                    'magnification': metadata.magnification,
                    'pixel_spacing': metadata.pixel_spacing,
                    'vendor': metadata.vendor,
                    'scanner': metadata.scanner
                }
                
                with open(output_path / 'metadata.json', 'w') as f:
                    json.dump(metadata_dict, f, indent=2)
            
            # Prepare results
            results = {
                'metadata': metadata,
                'total_patches_generated': len(coordinates),
                'patches_processed': len(patch_infos),
                'patches_selected': len(selected_patches),
                'selected_patches': selected_patches,
                'average_tissue_ratio': np.mean([p.tissue_ratio for p in selected_patches]) if selected_patches else 0,
                'average_quality_score': np.mean([p.quality_score for p in selected_patches]) if selected_patches else 0
            }
            
            return results
            
        finally:
            slide.close()
            gc.collect()  # Force garbage collection
    
    def extract_patches_for_model(self, wsi_path: str, patch_infos: List[PatchInfo]) -> torch.Tensor:
        """Extract patches and return as tensor for model inference"""
        
        slide = self.load_wsi(wsi_path)
        if slide is None:
            return None
        
        try:
            patches = []
            
            for patch_info in patch_infos:
                patch = self.extract_patch(slide, patch_info.coordinates)
                if patch is not None:
                    # Convert to tensor
                    patch_tensor = self.patch_transform(patch)
                    patches.append(patch_tensor)
            
            if patches:
                return torch.stack(patches)
            else:
                return None
                
        finally:
            slide.close()

def create_wsi_processor(config: Dict) -> WSIProcessor:
    """Factory function to create WSI processor"""
    return WSIProcessor(
        patch_size=config.get('patch_size', 224),
        patch_level=config.get('patch_level', 0),
        overlap=config.get('overlap', 0.0),
        tissue_threshold=config.get('tissue_threshold', 0.5),
        quality_threshold=config.get('quality_threshold', 0.3),
        max_patches=config.get('max_patches', 1000),
        stain_normalize=config.get('stain_normalize', True)
    )

if __name__ == "__main__":
    # Example usage
    processor_config = {
        'patch_size': 224,
        'patch_level': 0,
        'overlap': 0.1,
        'tissue_threshold': 0.6,
        'quality_threshold': 0.4,
        'max_patches': 500,
        'stain_normalize': True
    }
    
    processor = create_wsi_processor(processor_config)
    print("WSI Processor created successfully!")
    
    # Example processing (would need actual WSI file)
    # results = processor.process_wsi('example.svs', 'output_dir')
    # print(f"Processing completed: {results['patches_selected']} patches selected") 