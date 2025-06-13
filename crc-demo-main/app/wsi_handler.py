"""
WSI (Whole Slide Image) Handler for CRC Analysis Platform
Handles SVS, NDPI, and other WSI formats for the web interface
"""

import numpy as np
from PIL import Image
from pathlib import Path
import logging
from typing import Tuple, List, Optional
import torch
import cv2

# Try to import OpenSlide
try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    logging.warning("OpenSlide not available. WSI support limited.")

logger = logging.getLogger(__name__)


def is_wsi_file(filename: str) -> bool:
    """Check if a file is a whole slide image format"""
    wsi_extensions = ['.svs', '.ndpi', '.vms', '.vmu', '.scn', '.mrxs', '.tiff', '.svslide', '.bif']
    return any(filename.lower().endswith(ext) for ext in wsi_extensions)


def load_wsi_region(wsi_path: str, level: int = -1, region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
    """Load a region or thumbnail from a WSI file
    
    Args:
        wsi_path: Path to WSI file
        level: Level to read from (-1 for thumbnail)
        region: Optional (x, y, width, height) tuple for specific region
        
    Returns:
        numpy array of the image region
    """
    if not OPENSLIDE_AVAILABLE:
        raise ImportError("OpenSlide is required for WSI support. Install with: pip install openslide-python")
    
    try:
        slide = openslide.OpenSlide(wsi_path)
        
        if level == -1:
            # Get thumbnail
            thumbnail = slide.get_thumbnail((1024, 1024))
            return np.array(thumbnail)
        else:
            # Get specific region
            if region:
                x, y, w, h = region
                img = slide.read_region((x, y), level, (w, h))
            else:
                # Get full image at specified level
                img = slide.read_region((0, 0), level, slide.level_dimensions[level])
            
            # Convert RGBA to RGB
            img = img.convert('RGB')
            return np.array(img)
            
    except Exception as e:
        raise Exception(f"Error loading WSI: {str(e)}")


class WSIHandler:
    """Handler for Whole Slide Images in the web interface"""
    
    def __init__(self, tile_size: int = 512, overlap: int = 0):
        self.tile_size = tile_size
        self.overlap = overlap
        
    def can_handle_wsi(self, file_extension: str) -> bool:
        """Check if we can handle this WSI format"""
        wsi_extensions = ['.svs', '.ndpi', '.vms', '.vmu', '.scn', '.mrxs', '.tiff', '.svslide', '.bif']
        return file_extension.lower() in wsi_extensions and OPENSLIDE_AVAILABLE
    
    def load_image_for_display(self, file_path: str, max_size: Tuple[int, int] = (2048, 2048)) -> np.ndarray:
        """Load a WSI file and create a display image"""
        
        if not OPENSLIDE_AVAILABLE:
            raise ImportError("OpenSlide is required to handle WSI files. Install with: pip install openslide-python")
        
        try:
            # Open the slide
            slide = openslide.OpenSlide(file_path)
            
            # Get dimensions
            dimensions = slide.dimensions
            logger.info(f"WSI dimensions: {dimensions}")
            
            # Calculate appropriate level for display
            # OpenSlide levels go from 0 (highest resolution) to n (lowest)
            best_level = 0
            for level in range(slide.level_count):
                level_dims = slide.level_dimensions[level]
                if level_dims[0] <= max_size[0] and level_dims[1] <= max_size[1]:
                    best_level = level
                    break
            
            # Get the image at the selected level
            level_dims = slide.level_dimensions[best_level]
            region = slide.read_region((0, 0), best_level, level_dims)
            
            # Convert to RGB numpy array
            image = np.array(region.convert('RGB'))
            
            # If still too large, resize
            if image.shape[0] > max_size[1] or image.shape[1] > max_size[0]:
                from PIL import Image as PILImage
                pil_image = PILImage.fromarray(image)
                pil_image.thumbnail(max_size, PILImage.Resampling.LANCZOS)
                image = np.array(pil_image)
            
            logger.info(f"Loaded WSI thumbnail: {image.shape}")
            
            slide.close()
            return image
            
        except Exception as e:
            logger.error(f"Failed to load WSI: {e}")
            raise
    
    def extract_representative_tiles(self, file_path: str, n_tiles: int = 16) -> List[np.ndarray]:
        """Extract representative tiles from a WSI for analysis"""
        
        if not OPENSLIDE_AVAILABLE:
            raise ImportError("OpenSlide is required to handle WSI files")
        
        tiles = []
        
        try:
            slide = openslide.OpenSlide(file_path)
            dimensions = slide.dimensions
            
            # Calculate tissue mask at low resolution
            thumbnail = slide.get_thumbnail((1024, 1024))
            tissue_mask = self._detect_tissue(np.array(thumbnail))
            
            # Scale mask coordinates to full resolution
            scale_x = dimensions[0] / tissue_mask.shape[1]
            scale_y = dimensions[1] / tissue_mask.shape[0]
            
            # Find tissue regions
            tissue_coords = np.argwhere(tissue_mask > 0)
            
            if len(tissue_coords) == 0:
                # No tissue found, extract from center
                center_x = dimensions[0] // 2
                center_y = dimensions[1] // 2
                tile = slide.read_region(
                    (center_x - self.tile_size//2, center_y - self.tile_size//2),
                    0, (self.tile_size, self.tile_size)
                )
                tiles.append(np.array(tile.convert('RGB')))
            else:
                # Sample tiles from tissue regions
                n_coords = min(n_tiles, len(tissue_coords))
                sample_indices = np.random.choice(len(tissue_coords), n_coords, replace=False)
                
                for idx in sample_indices:
                    y, x = tissue_coords[idx]
                    # Convert to full resolution coordinates
                    full_x = int(x * scale_x)
                    full_y = int(y * scale_y)
                    
                    # Ensure we don't go out of bounds
                    full_x = min(full_x, dimensions[0] - self.tile_size)
                    full_y = min(full_y, dimensions[1] - self.tile_size)
                    
                    # Extract tile
                    tile = slide.read_region(
                        (full_x, full_y), 0, (self.tile_size, self.tile_size)
                    )
                    tiles.append(np.array(tile.convert('RGB')))
            
            slide.close()
            logger.info(f"Extracted {len(tiles)} tiles from WSI")
            
        except Exception as e:
            logger.error(f"Failed to extract tiles: {e}")
            raise
        
        return tiles
    
    def _detect_tissue(self, image: np.ndarray, threshold: float = 0.8) -> np.ndarray:
        """Simple tissue detection using Otsu's thresholding"""
        import cv2
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu's thresholding
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Invert if necessary (tissue should be white)
        if np.mean(binary) > 127:
            binary = 255 - binary
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        return binary


def process_uploaded_file(uploaded_file, temp_dir: Path = Path("temp")) -> Tuple[np.ndarray, Optional[List[np.ndarray]]]:
    """
    Process an uploaded file, handling both regular images and WSI files
    
    Returns:
        Tuple of (display_image, tiles_for_analysis)
        - display_image: numpy array for display
        - tiles_for_analysis: list of tiles if WSI, None if regular image
    """
    
    # Create temp directory if needed
    temp_dir.mkdir(exist_ok=True)
    
    # Save uploaded file temporarily
    file_extension = Path(uploaded_file.name).suffix
    temp_path = temp_dir / f"temp_upload{file_extension}"
    
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    
    try:
        # Check if it's a WSI file
        wsi_handler = WSIHandler()
        
        if wsi_handler.can_handle_wsi(file_extension):
            # Handle as WSI
            logger.info(f"Processing WSI file: {uploaded_file.name}")
            
            # Get display image
            display_image = wsi_handler.load_image_for_display(str(temp_path))
            
            # Extract tiles for analysis
            tiles = wsi_handler.extract_representative_tiles(str(temp_path))
            
            return display_image, tiles
        else:
            # Handle as regular image
            logger.info(f"Processing regular image file: {uploaded_file.name}")
            image = Image.open(temp_path).convert('RGB')
            image_np = np.array(image)
            
            return image_np, None
            
    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()


def analyze_wsi_tiles(tiles: List[np.ndarray], model, transform) -> dict:
    """
    Analyze multiple tiles from a WSI and aggregate results
    """
    import torch
    
    all_predictions = []
    all_probabilities = []
    
    for tile in tiles:
        # Convert to PIL image
        tile_pil = Image.fromarray(tile)
        
        # Transform and predict
        img_tensor = transform(tile_pil).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            _, predicted = torch.max(probs, 1)
            
            all_predictions.append(predicted.item())
            all_probabilities.append(probs[0].numpy())
    
    # Aggregate results
    all_probabilities = np.array(all_probabilities)
    mean_probabilities = np.mean(all_probabilities, axis=0)
    
    # Get consensus prediction
    consensus_class = np.argmax(mean_probabilities)
    consensus_confidence = mean_probabilities[consensus_class] * 100
    
    # Calculate tissue composition across all tiles
    classes = ['Tumor', 'Stroma', 'Complex', 'Lymphocytes',
               'Debris', 'Mucosa', 'Adipose', 'Empty']
    
    tissue_counts = np.bincount(all_predictions, minlength=len(classes))
    tissue_percentages = tissue_counts / len(all_predictions)
    
    return {
        'primary_class': classes[consensus_class],
        'confidence': consensus_confidence,
        'probabilities': mean_probabilities,
        'tile_predictions': all_predictions,
        'tissue_distribution': {
            classes[i]: tissue_percentages[i] * 100 
            for i in range(len(classes))
        },
        'n_tiles_analyzed': len(tiles)
    } 