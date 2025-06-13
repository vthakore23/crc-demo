import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Dict, List, Tuple, Optional
import logging
import cv2

# Try to import optional dependencies
try:
    from skimage.feature import graycomatrix, graycoprops
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

try:
    from radiomics import featureextractor
    import SimpleITK as sitk
    RADIOMICS_AVAILABLE = True
except ImportError:
    RADIOMICS_AVAILABLE = False
    
logger = logging.getLogger(__name__)

class DeepFeatureExtractor:
    """Extract deep features from tiles using pretrained CNNs"""
    
    def __init__(self, model_name='resnet50', device='cpu'):
        self.device = device
        self.model_name = model_name
        
        # Load pretrained model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 2048
        elif model_name == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 1280
        elif model_name == 'densenet121':
            self.model = models.densenet121(pretrained=True)
            self.model.classifier = nn.Identity()
            self.feature_dim = 1024
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        self.model.eval()
        self.model.to(device)
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, tiles: List[np.ndarray], batch_size: int = 32) -> torch.Tensor:
        """Extract features from tiles in batches"""
        features = []
        
        with torch.no_grad():
            for i in range(0, len(tiles), batch_size):
                batch_tiles = tiles[i:i+batch_size]
                
                # Transform batch
                batch_tensors = []
                for tile in batch_tiles:
                    tensor = self.transform(tile)
                    batch_tensors.append(tensor)
                
                batch = torch.stack(batch_tensors).to(self.device)
                
                # Extract features
                batch_features = self.model(batch).squeeze()
                
                if len(batch_features.shape) == 1:
                    batch_features = batch_features.unsqueeze(0)
                
                features.append(batch_features.cpu())
        
        return torch.cat(features, dim=0)

class HandcraftedFeatureExtractor:
    """Extract handcrafted features from tiles"""
    
    def __init__(self):
        self.feature_names = []
        
    def extract_features(self, tile: np.ndarray) -> Dict[str, float]:
        """Extract handcrafted features from a single tile"""
        features = {}
        
        # Color features
        color_features = self._extract_color_features(tile)
        features.update(color_features)
        
        # Texture features
        texture_features = self._extract_texture_features(tile)
        features.update(texture_features)
        
        # Morphological features
        morph_features = self._extract_morphological_features(tile)
        features.update(morph_features)
        
        return features
    
    def _extract_color_features(self, tile: np.ndarray) -> Dict[str, float]:
        """Extract color-based features"""
        features = {}
        
        # RGB statistics
        for i, channel in enumerate(['R', 'G', 'B']):
            features[f'{channel}_mean'] = np.mean(tile[:, :, i])
            features[f'{channel}_std'] = np.std(tile[:, :, i])
            features[f'{channel}_skew'] = self._skewness(tile[:, :, i])
            features[f'{channel}_kurtosis'] = self._kurtosis(tile[:, :, i])
        
        # HSV statistics
        hsv = cv2.cvtColor(tile, cv2.COLOR_RGB2HSV)
        for i, channel in enumerate(['H', 'S', 'V']):
            features[f'{channel}_mean'] = np.mean(hsv[:, :, i])
            features[f'{channel}_std'] = np.std(hsv[:, :, i])
        
        # LAB statistics
        lab = cv2.cvtColor(tile, cv2.COLOR_RGB2LAB)
        for i, channel in enumerate(['L', 'A', 'B']):
            features[f'LAB_{channel}_mean'] = np.mean(lab[:, :, i])
            features[f'LAB_{channel}_std'] = np.std(lab[:, :, i])
        
        return features
    
    def _extract_texture_features(self, tile: np.ndarray) -> Dict[str, float]:
        """Extract texture features using GLCM and other methods"""
        features = {}
        
        # Convert to grayscale
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        
        # Haralick features (simplified)
        # In practice, use mahotas or skimage for proper GLCM
        features['texture_contrast'] = np.std(gray)
        features['texture_homogeneity'] = 1 / (1 + np.var(gray))
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # Local Binary Patterns (simplified)
        # In practice, use skimage.feature.local_binary_pattern
        features['lbp_variance'] = np.var(gray)
        
        return features
    
    def _extract_morphological_features(self, tile: np.ndarray) -> Dict[str, float]:
        """Extract morphological features"""
        features = {}
        
        # Convert to binary
        gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Largest contour statistics
            largest_contour = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            
            features['largest_object_area'] = area / binary.size
            features['largest_object_perimeter'] = perimeter
            features['largest_object_circularity'] = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            
            # Number of objects
            features['num_objects'] = len(contours)
            
            # Average object size
            areas = [cv2.contourArea(c) for c in contours]
            features['mean_object_area'] = np.mean(areas) / binary.size
            features['std_object_area'] = np.std(areas) / binary.size
        else:
            # Default values if no contours found
            features['largest_object_area'] = 0
            features['largest_object_perimeter'] = 0
            features['largest_object_circularity'] = 0
            features['num_objects'] = 0
            features['mean_object_area'] = 0
            features['std_object_area'] = 0
        
        return features
    
    def _skewness(self, data: np.ndarray) -> float:
        """Calculate skewness"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3

class HybridFeatureExtractor:
    """Combine deep and handcrafted features"""
    
    def __init__(self, deep_model='resnet50', device='cpu'):
        self.deep_extractor = DeepFeatureExtractor(deep_model, device)
        self.handcrafted_extractor = HandcraftedFeatureExtractor()
        
    def extract_features(self, tiles: List[np.ndarray], 
                        use_deep: bool = True, 
                        use_handcrafted: bool = True) -> np.ndarray:
        """Extract hybrid features from tiles"""
        all_features = []
        
        for tile in tiles:
            tile_features = []
            
            if use_deep:
                # Deep features
                deep_feat = self.deep_extractor.extract_features([tile])
                tile_features.append(deep_feat.squeeze().numpy())
            
            if use_handcrafted:
                # Handcrafted features
                handcrafted_feat = self.handcrafted_extractor.extract_features(tile)
                handcrafted_array = np.array(list(handcrafted_feat.values()))
                tile_features.append(handcrafted_array)
            
            # Concatenate features
            if len(tile_features) > 0:
                combined = np.concatenate(tile_features)
                all_features.append(combined)
        
        return np.array(all_features)

class RadiomicsFeatureExtractor:
    """Extract radiomics features from tiles"""
    
    def __init__(self):
        if not RADIOMICS_AVAILABLE:
            logger.warning("Radiomics not available. Install pyradiomics to use this feature.")
            self.extractor = None
            return
            
        # Configure radiomics settings
        self.settings = {
            'binWidth': 25,
            'resampledPixelSpacing': None,
            'interpolator': 'sitkBSpline',
            'enableCExtensions': True
        }
        
        # Enable feature classes
        self.enabled_features = {
            'firstorder': [],  # All first order features
            'shape2D': [],
            'glcm': [],  # Gray Level Co-occurrence Matrix
            'glrlm': [],  # Gray Level Run Length Matrix
            'glszm': [],  # Gray Level Size Zone Matrix
            'gldm': []   # Gray Level Dependence Matrix
        }
        
        self.extractor = featureextractor.RadiomicsFeatureExtractor(
            self.settings, 
            **self.enabled_features
        )
    
    def extract_features(self, tiles: List[np.ndarray], 
                        masks: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """Extract radiomics features from tiles"""
        if not RADIOMICS_AVAILABLE or self.extractor is None:
            # Return empty features if radiomics not available
            return np.zeros((len(tiles), 1))
            
        all_features = []
        
        for i, tile in enumerate(tiles):
            # Convert to SimpleITK image
            image_sitk = sitk.GetImageFromArray(tile)
            
            # Create or get mask
            if masks and i < len(masks):
                mask = masks[i]
            else:
                mask = np.ones(tile.shape[:2], dtype=np.uint8)
            mask_sitk = sitk.GetImageFromArray(mask)
            
            # Extract features
            features = self.extractor.execute(image_sitk, mask_sitk)
            
            # Filter out diagnostic information
            feature_values = []
            for key, value in features.items():
                if not key.startswith('diagnostics_'):
                    feature_values.append(float(value))
            
            all_features.append(feature_values)
        
        return np.array(all_features)

class FeatureFusion:
    """Combine deep and radiomics features"""
    
    def __init__(self, config):
        self.config = config
        self.deep_extractor = DeepFeatureExtractor(device=config.DEVICE)
        self.radiomics_extractor = RadiomicsFeatureExtractor()
    
    def extract_features(self, tiles: List[np.ndarray], 
                        masks: Optional[List[np.ndarray]] = None) -> np.ndarray:
        """Extract and combine all features"""
        features = []
        
        # Deep features
        if self.config.USE_DEEP_FEATURES:
            deep_features = self.deep_extractor.extract_features(tiles)
            features.append(deep_features.numpy())
        
        # Radiomics features
        if self.config.USE_RADIOMICS:
            radiomics_features = self.radiomics_extractor.extract_features(tiles, masks)
            features.append(radiomics_features)
        
        # Combine features
        if len(features) > 1:
            combined = np.concatenate(features, axis=1)
        else:
            combined = features[0]
        
        return combined

class MILFeatureAggregator:
    """Aggregate tile-level features using attention MIL"""
    
    def __init__(self, feature_dim: int, hidden_dim: int, attention_dim: int):
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Tensor of shape (n_tiles, feature_dim)
        Returns:
            Tensor of shape (1, feature_dim)
        """
        # Calculate attention scores
        attention_scores = self.attention(features)
        attention_weights = torch.softmax(attention_scores, dim=0)
        
        # Weighted sum of features
        weighted_features = features * attention_weights
        aggregated = torch.sum(weighted_features, dim=0, keepdim=True)
        
        return aggregated, attention_weights 

class FeatureExtractor:
    """Extract deep features from tiles using pretrained CNN"""
    
    def __init__(self, model_name='resnet50', device='cpu'):
        self.device = device
        
        # Load pretrained model
        if model_name == 'resnet50':
            self.model = models.resnet50(pretrained=True)
            # Remove the final classification layer
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 2048
        elif model_name == 'efficientnet':
            self.model = models.efficientnet_b0(pretrained=True)
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            self.feature_dim = 1280
        
        self.model.eval()
        self.model.to(device)
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, tiles: List[np.ndarray]) -> torch.Tensor:
        """Extract features from tiles"""
        features = []
        
        with torch.no_grad():
            for tile in tiles:
                # Transform
                if isinstance(tile, np.ndarray):
                    img_tensor = self.transform(tile).unsqueeze(0).to(self.device)
                else:
                    img_tensor = self.transform(tile).unsqueeze(0).to(self.device)
                
                # Extract features
                feat = self.model(img_tensor).squeeze()
                features.append(feat.cpu())
        
        return torch.stack(features)
    
    def extract_single_feature(self, image):
        """Extract features from a single image (PIL Image or numpy array)"""
        with torch.no_grad():
            # Transform based on input type
            if isinstance(image, np.ndarray):
                # Convert numpy array to PIL Image
                from PIL import Image as PILImage
                image = PILImage.fromarray(image)
            
            # Apply transforms
            img_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            feat = self.model(img_tensor).squeeze()
            
        return feat.cpu() 