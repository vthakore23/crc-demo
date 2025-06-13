import numpy as np
import cv2
from typing import List, Tuple, Optional
from PIL import Image
import openslide
import logging
from pathlib import Path

# Try to import various tissue processing tools
try:
    from tiatoolbox.tools.tissuemask import MorphologicalMasker
    TIATOOLBOX_AVAILABLE = True
except ImportError:
    TIATOOLBOX_AVAILABLE = False

logger = logging.getLogger(__name__)

class TissueSegmenter:
    """Segment tissue regions from whole slide images"""
    
    def __init__(self):
        if TIATOOLBOX_AVAILABLE:
            self.masker = MorphologicalMasker()
        else:
            self.masker = None
            
    def segment_tissue(self, wsi_path: str, level: int = 2) -> np.ndarray:
        """Segment tissue regions from WSI
        
        Args:
            wsi_path: Path to whole slide image
            level: Pyramid level for processing
            
        Returns:
            Binary mask of tissue regions
        """
        # Open slide
        slide = openslide.OpenSlide(wsi_path)
        
        # Get thumbnail for tissue detection
        thumbnail = slide.get_thumbnail(slide.level_dimensions[level])
        thumbnail_np = np.array(thumbnail)
        
        if self.masker is not None:
            # Use tiatoolbox if available
            tissue_mask = self.masker.fit_transform([thumbnail_np])[0]
        else:
            # Fallback to simple tissue detection
            tissue_mask = self._simple_tissue_detection(thumbnail_np)
            
        return tissue_mask
    
    def _simple_tissue_detection(self, image: np.ndarray) -> np.ndarray:
        """Simple tissue detection using color thresholding"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Otsu's thresholding
        _, tissue_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if necessary (tissue should be white)
        if np.mean(tissue_mask) > 127:
            tissue_mask = 255 - tissue_mask
            
        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_CLOSE, kernel)
        tissue_mask = cv2.morphologyEx(tissue_mask, cv2.MORPH_OPEN, kernel)
        
        return tissue_mask.astype(bool)

class StainNormalizer:
    """Stain normalization for H&E slides"""
    
    def __init__(self, method: str = 'macenko'):
        self.method = method
        self.reference_mean = None
        self.reference_std = None
    
    def fit(self, reference_image: np.ndarray):
        """Fit normalizer to reference image"""
        # Simple normalization using statistics
        self.reference_mean = np.mean(reference_image, axis=(0, 1))
        self.reference_std = np.std(reference_image, axis=(0, 1))
    
    def transform(self, image: np.ndarray) -> np.ndarray:
        """Transform image to match reference staining"""
        if self.reference_mean is None:
            # If not fitted, use standard H&E statistics
            self.reference_mean = np.array([148.60, 41.56, 111.51])  # Typical H&E
            self.reference_std = np.array([41.61, 9.01, 17.75])
            
        # Normalize
        mean = np.mean(image, axis=(0, 1))
        std = np.std(image, axis=(0, 1))
        
        # Avoid division by zero
        std = np.where(std < 1e-8, 1, std)
        
        normalized = (image - mean) / std
        normalized = normalized * self.reference_std + self.reference_mean
        
        return np.clip(normalized, 0, 255).astype(np.uint8)

class WSIPreprocessor:
    """Complete preprocessing pipeline for WSIs"""
    
    def __init__(self, config):
        self.config = config
        self.tissue_segmenter = TissueSegmenter()
        self.stain_normalizer = StainNormalizer()
        
    def process_wsi(self, wsi_path: str) -> List[np.ndarray]:
        """Process whole slide image and extract normalized tissue tiles
        
        Args:
            wsi_path: Path to WSI file
            
        Returns:
            List of normalized tissue tiles
        """
        tiles = []
        
        try:
            # Open slide
            slide = openslide.OpenSlide(str(wsi_path))
            
            # Get tissue mask
            tissue_mask = self.tissue_segmenter.segment_tissue(str(wsi_path))
            
            # Extract tiles
            tile_size = self.config.wsi_processing.get('tile_size', 256)
            tiles = self._extract_tiles(slide, tissue_mask, tile_size)
            
            # Normalize staining if enabled
            if self.config.wsi_processing.get('stain_normalization', True):
                tiles = [self.stain_normalizer.transform(tile) for tile in tiles]
                
            logger.info(f"Extracted {len(tiles)} tiles from {wsi_path}")
            
        except Exception as e:
            logger.error(f"Error processing {wsi_path}: {str(e)}")
            
        return tiles
    
    def _extract_tiles(self, slide: openslide.OpenSlide, tissue_mask: np.ndarray, 
                      tile_size: int) -> List[np.ndarray]:
        """Extract tiles from regions with tissue"""
        tiles = []
        
        # Get dimensions
        width, height = slide.dimensions
        mask_h, mask_w = tissue_mask.shape
        
        # Calculate scaling factors
        scale_x = width / mask_w
        scale_y = height / mask_h
        
        # Find tissue regions in mask
        tissue_coords = np.where(tissue_mask)
        
        # Sample tiles from tissue regions
        stride = tile_size  # Non-overlapping tiles
        
        for y in range(0, height - tile_size, stride):
            for x in range(0, width - tile_size, stride):
                # Check if this region contains tissue
                mask_x = int(x / scale_x)
                mask_y = int(y / scale_y)
                
                if mask_x < mask_w and mask_y < mask_h and tissue_mask[mask_y, mask_x]:
                    # Extract tile
                    tile = slide.read_region((x, y), 0, (tile_size, tile_size))
                    tile = np.array(tile.convert('RGB'))
                    
                    # Check if tile has enough tissue
                    if self._is_valid_tile(tile):
                        tiles.append(tile)
                        
                    # Limit number of tiles for memory
                    if len(tiles) >= self.config.wsi_processing.get('max_tiles', 1000):
                        return tiles
                        
        return tiles
    
    def _is_valid_tile(self, tile: np.ndarray, tissue_threshold: float = 0.5) -> bool:
        """Check if tile contains sufficient tissue"""
        # Convert to grayscale
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        
        # Simple tissue detection
        _, binary = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        
        # Calculate tissue ratio
        tissue_ratio = np.sum(binary > 0) / (tile.shape[0] * tile.shape[1])
        
        return tissue_ratio > tissue_threshold 

def extract_tiles(wsi_path, output_dir, tile_size=512, stride=256, min_tissue_percent=0.5):
    """Extract tiles from WSI with tissue detection"""
    try:
        # Try to open with tiffslide first
        try:
            from tiffslide import TiffSlide
            slide = TiffSlide(str(wsi_path))
            
            # Get dimensions
            dimensions = slide.dimensions
            
            # Try to get magnification, with fallback
            try:
                magnification = slide.properties.get('aperio.AppMag', 
                                                   slide.properties.get('openslide.objective-power', 20))
                if magnification is None:
                    magnification = 20  # Default to 20x if not found
                    logging.warning(f"No magnification found in metadata, using default {magnification}x")
            except:
                magnification = 20  # Default magnification
                logging.warning(f"Could not read magnification, using default {magnification}x")
                
        except ImportError:
            logging.warning("tiffslide not available, using PIL")
            # Fallback to PIL
            from PIL import Image
            img = Image.open(wsi_path)
            dimensions = img.size
            magnification = 20  # Default for PIL
            slide = img

        # Get tissue mask
        tissue_mask = extract_tiles.tissue_segmenter.segment_tissue(str(wsi_path))
        
        # Extract tiles
        tiles = extract_tiles._extract_tiles(slide, tissue_mask, tile_size)
        
        # Normalize staining if enabled
        if extract_tiles.config.wsi_processing.get('stain_normalization', True):
            tiles = [extract_tiles.stain_normalizer.transform(tile) for tile in tiles]
            
        logger.info(f"Extracted {len(tiles)} tiles from {wsi_path}")
        
    except Exception as e:
        logger.error(f"Error processing {wsi_path}: {str(e)}")
        
    return tiles 