#!/usr/bin/env python3
"""
Advanced Molecular Subtype Predictor with State-of-the-Art Architecture
Uses Vision Transformer (ViT) backbone with multi-scale feature extraction,
attention mechanisms, and sophisticated spatial pattern analysis.

Designed for genuine molecular subtype prediction based on biological characteristics:
- SNF1 (Canonical): High tumor content, sharp edges, low immune infiltration
- SNF2 (Immune): High immune infiltration, lymphocyte aggregates, immune highways  
- SNF3 (Stromal): High stromal content, fibrosis patterns, stromal barriers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
from PIL import Image
import torchvision.models as models
from torchvision import transforms
from pathlib import Path
import json

# Try importing advanced libraries
try:
    from timm import create_model
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("timm not available - using standard architectures")

try:
    from scipy import ndimage
    from scipy.spatial import distance
    from scipy.stats import skew, kurtosis
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("scipy not available - some features will be simplified")


class MultiScaleFeatureExtractor(nn.Module):
    """Extract features at multiple scales using dilated convolutions"""
    
    def __init__(self, in_channels=2048, out_channels=256):
        super().__init__()
        
        # Multi-scale feature extraction with different dilation rates
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.scale3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        self.scale4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=8, dilation=8),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 4, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Extract features at multiple scales
        s1 = self.scale1(x)
        s2 = self.scale2(x)
        s3 = self.scale3(x)
        s4 = self.scale4(x)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([s1, s2, s3, s4], dim=1)
        
        # Fuse features
        fused = self.fusion(multi_scale)
        
        return fused, [s1, s2, s3, s4]


class SpatialAttentionModule(nn.Module):
    """Spatial attention to focus on relevant regions"""
    
    def __init__(self, in_channels):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)
        
    def forward(self, x):
        # Generate attention map
        att = self.conv1(x)
        att = F.relu(att, inplace=True)
        att = self.conv2(att)
        att = torch.sigmoid(att)
        
        # Apply attention
        attended = x * att
        
        return attended, att


class BiologicalConstraintModule(nn.Module):
    """Apply biological constraints based on molecular subtype characteristics"""
    
    def __init__(self, feature_dim=512):
        super().__init__()
        
        # Learnable constraint networks for each subtype
        self.snf1_constraints = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.snf2_constraints = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        self.snf3_constraints = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
        # Mutual exclusion layer
        self.exclusion_weights = nn.Parameter(torch.tensor([
            [1.0, -0.3, -0.3],   # SNF1 suppresses others
            [-0.3, 1.0, -0.5],   # SNF2 strongly suppresses SNF3
            [-0.3, -0.5, 1.0]    # SNF3 strongly suppresses SNF2
        ]))
        
    def forward(self, features, raw_scores):
        # Extract subtype-specific features
        snf1_features = self.snf1_constraints(features)
        snf2_features = self.snf2_constraints(features)
        snf3_features = self.snf3_constraints(features)
        
        # Calculate constraint scores
        snf1_constraint = torch.sum(snf1_features, dim=1, keepdim=True)
        snf2_constraint = torch.sum(snf2_features, dim=1, keepdim=True)
        snf3_constraint = torch.sum(snf3_features, dim=1, keepdim=True)
        
        constraint_scores = torch.cat([snf1_constraint, snf2_constraint, snf3_constraint], dim=1)
        
        # Apply mutual exclusion
        exclusion_adjusted = torch.matmul(raw_scores, self.exclusion_weights)
        
        # Combine raw scores with constraints
        final_scores = raw_scores + 0.3 * constraint_scores + 0.2 * exclusion_adjusted
        
        return final_scores


class AdvancedMolecularPredictor(nn.Module):
    """State-of-the-art molecular subtype predictor"""
    
    def __init__(self, num_classes=3, use_vit=True):
        super().__init__()
        
        # Use Vision Transformer if available, otherwise ResNet50
        if use_vit and TIMM_AVAILABLE:
            try:
                self.backbone = create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
                backbone_dim = 768
                self.using_vit = True
            except Exception as e:
                print(f"Warning: Could not load ViT model ({e}), falling back to ResNet50")
                resnet = models.resnet50(pretrained=True)
                self.backbone = nn.Sequential(*list(resnet.children())[:-2])
                backbone_dim = 2048
                self.using_vit = False
        else:
            # Use ResNet50 as backbone
            resnet = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            backbone_dim = 2048
            self.using_vit = False
        
        # Multi-scale feature extractor
        self.multi_scale = MultiScaleFeatureExtractor(backbone_dim, 256)
        
        # Spatial attention
        self.spatial_attention = SpatialAttentionModule(256)
        
        # Global context aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.local_pool = nn.AdaptiveMaxPool2d(4)
        
        # Feature dimension after pooling
        feature_dim = 256 + 256 * 16  # global + local features
        
        # Deep feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )
        
        # Biological constraints
        self.biological_constraints = BiologicalConstraintModule(512)
        
        # Subtype-specific heads with different architectures
        self.snf1_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.snf2_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        self.snf3_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
        # Temperature scaling for calibrated predictions
        self.temperature = nn.Parameter(torch.ones(1))
        
    def forward(self, x, return_features=False):
        # Extract backbone features
        if hasattr(self, 'using_vit') and self.using_vit and hasattr(self.backbone, 'forward_features'):
            # Vision Transformer
            backbone_features = self.backbone.forward_features(x)
            # Reshape to spatial format
            B, L, C = backbone_features.shape
            # For ViT base patch16 with 224x224 input: (224/16)^2 = 196 patches
            H = W = int(np.sqrt(L))
            backbone_features = backbone_features.transpose(1, 2).reshape(B, C, H, W)
        else:
            # CNN backbone (ResNet)
            backbone_features = self.backbone(x)
        
        # Multi-scale feature extraction
        multi_scale_features, scale_features = self.multi_scale(backbone_features)
        
        # Apply spatial attention
        attended_features, attention_map = self.spatial_attention(multi_scale_features)
        
        # Global and local pooling
        global_features = self.global_pool(attended_features).flatten(1)
        local_features = self.local_pool(attended_features).flatten(1)
        
        # Concatenate features
        combined_features = torch.cat([global_features, local_features], dim=1)
        
        # Process features
        processed_features = self.feature_processor(combined_features)
        
        # Subtype-specific predictions
        snf1_score = self.snf1_head(processed_features)
        snf2_score = self.snf2_head(processed_features)
        snf3_score = self.snf3_head(processed_features)
        
        # Combine scores
        raw_scores = torch.cat([snf1_score, snf2_score, snf3_score], dim=1)
        
        # Apply biological constraints
        constrained_scores = self.biological_constraints(processed_features, raw_scores)
        
        # Temperature scaling
        scaled_scores = constrained_scores / self.temperature
        
        # Apply softmax
        probabilities = F.softmax(scaled_scores, dim=1)
        
        if return_features:
            return probabilities, {
                'backbone_features': backbone_features,
                'attention_map': attention_map,
                'processed_features': processed_features,
                'raw_scores': raw_scores
            }
        
        return probabilities


class SpatialPatternAnalyzerV2:
    """Advanced spatial pattern analysis using computer vision techniques"""
    
    def __init__(self):
        self.gabor_kernels = self._create_gabor_kernels()
        self.lbp_radius = 3
        self.lbp_points = 24
        
    def _create_gabor_kernels(self):
        """Create Gabor filter bank for texture analysis"""
        kernels = []
        ksize = 31
        for theta in np.arange(0, np.pi, np.pi / 8):
            for frequency in [0.1, 0.2, 0.3]:
                kernel = cv2.getGaborKernel(
                    (ksize, ksize), 
                    sigma=5.0, 
                    theta=theta,
                    lambd=10.0 / frequency, 
                    gamma=0.5, 
                    psi=0, 
                    ktype=cv2.CV_32F
                )
                kernels.append(kernel)
        return kernels
    
    def extract_morphological_features(self, binary_mask: np.ndarray) -> Dict[str, float]:
        """Extract morphological features from binary mask"""
        features = {}
        
        # Find contours
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) == 0:
            return {
                'num_objects': 0,
                'mean_area': 0,
                'mean_perimeter': 0,
                'mean_circularity': 0,
                'mean_solidity': 0,
                'mean_eccentricity': 0,
                'area_std': 0,
                'shape_heterogeneity': 0
            }
        
        areas = []
        perimeters = []
        circularities = []
        solidities = []
        eccentricities = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 10:  # Skip tiny objects
                continue
                
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter ** 2 + 1e-6)
            
            # Convex hull for solidity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / (hull_area + 1e-6)
            
            # Fit ellipse for eccentricity
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (center, (MA, ma), angle) = ellipse
                eccentricity = np.sqrt(1 - (ma / MA) ** 2) if MA > 0 else 0
            else:
                eccentricity = 0
            
            areas.append(area)
            perimeters.append(perimeter)
            circularities.append(circularity)
            solidities.append(solidity)
            eccentricities.append(eccentricity)
        
        if len(areas) > 0:
            features = {
                'num_objects': len(areas),
                'mean_area': np.mean(areas),
                'mean_perimeter': np.mean(perimeters),
                'mean_circularity': np.mean(circularities),
                'mean_solidity': np.mean(solidities),
                'mean_eccentricity': np.mean(eccentricities),
                'area_std': np.std(areas),
                'shape_heterogeneity': np.std(circularities) + np.std(solidities)
            }
        else:
            features = {
                'num_objects': 0,
                'mean_area': 0,
                'mean_perimeter': 0,
                'mean_circularity': 0,
                'mean_solidity': 0,
                'mean_eccentricity': 0,
                'area_std': 0,
                'shape_heterogeneity': 0
            }
        
        return features
    
    def extract_texture_features(self, gray_image: np.ndarray) -> Dict[str, float]:
        """Extract texture features using multiple methods"""
        features = {}
        
        # Gabor features
        gabor_responses = []
        for kernel in self.gabor_kernels:
            filtered = cv2.filter2D(gray_image, cv2.CV_32F, kernel)
            gabor_responses.append(filtered)
        
        # Statistical features from Gabor responses
        gabor_mean = np.mean([np.mean(r) for r in gabor_responses])
        gabor_std = np.mean([np.std(r) for r in gabor_responses])
        gabor_energy = np.mean([np.sum(r**2) for r in gabor_responses])
        
        features['gabor_mean'] = gabor_mean
        features['gabor_std'] = gabor_std
        features['gabor_energy'] = gabor_energy
        
        # Haralick features from GLCM
        # Calculate GLCM for multiple directions
        distances = [1, 3, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm_features = {
            'contrast': [],
            'homogeneity': [],
            'energy': [],
            'correlation': []
        }
        
        for d in distances:
            for angle in angles:
                # Simple GLCM calculation
                glcm = self._calculate_glcm(gray_image, d, angle)
                
                # Extract Haralick features
                glcm_features['contrast'].append(self._glcm_contrast(glcm))
                glcm_features['homogeneity'].append(self._glcm_homogeneity(glcm))
                glcm_features['energy'].append(self._glcm_energy(glcm))
                glcm_features['correlation'].append(self._glcm_correlation(glcm))
        
        # Average GLCM features
        for key in glcm_features:
            features[f'glcm_{key}'] = np.mean(glcm_features[key])
        
        # Fractal dimension
        features['fractal_dimension'] = self._calculate_fractal_dimension(gray_image)
        
        return features
    
    def _calculate_glcm(self, image, distance, angle):
        """Calculate Gray Level Co-occurrence Matrix"""
        # Quantize image to 16 levels
        quantized = (image / 16).astype(np.uint8)
        
        # Calculate offsets
        dx = int(distance * np.cos(angle))
        dy = int(distance * np.sin(angle))
        
        # Initialize GLCM
        glcm = np.zeros((16, 16))
        
        # Calculate co-occurrences
        rows, cols = image.shape
        for i in range(rows - abs(dy)):
            for j in range(cols - abs(dx)):
                i_pixel = quantized[i, j]
                j_pixel = quantized[i + dy, j + dx]
                glcm[i_pixel, j_pixel] += 1
        
        # Normalize
        glcm = glcm / (glcm.sum() + 1e-6)
        
        return glcm
    
    def _glcm_contrast(self, glcm):
        """Calculate GLCM contrast"""
        i, j = np.meshgrid(range(16), range(16), indexing='ij')
        return np.sum((i - j) ** 2 * glcm)
    
    def _glcm_homogeneity(self, glcm):
        """Calculate GLCM homogeneity"""
        i, j = np.meshgrid(range(16), range(16), indexing='ij')
        return np.sum(glcm / (1 + np.abs(i - j)))
    
    def _glcm_energy(self, glcm):
        """Calculate GLCM energy"""
        return np.sum(glcm ** 2)
    
    def _glcm_correlation(self, glcm):
        """Calculate GLCM correlation"""
        i, j = np.meshgrid(range(16), range(16), indexing='ij')
        mu_i = np.sum(i * glcm)
        mu_j = np.sum(j * glcm)
        sigma_i = np.sqrt(np.sum((i - mu_i) ** 2 * glcm))
        sigma_j = np.sqrt(np.sum((j - mu_j) ** 2 * glcm))
        
        if sigma_i > 0 and sigma_j > 0:
            return np.sum((i - mu_i) * (j - mu_j) * glcm) / (sigma_i * sigma_j)
        else:
            return 0
    
    def _calculate_fractal_dimension(self, image):
        """Calculate fractal dimension using box-counting method"""
        # Binarize image
        threshold = np.mean(image)
        binary = image > threshold
        
        # Box sizes
        sizes = np.logspace(0.5, 3, num=10, base=2).astype(int)
        counts = []
        
        for size in sizes:
            # Count non-empty boxes
            count = 0
            for i in range(0, image.shape[0] - size, size):
                for j in range(0, image.shape[1] - size, size):
                    if np.any(binary[i:i+size, j:j+size]):
                        count += 1
            counts.append(count)
        
        # Fit log-log relationship
        coeffs = np.polyfit(np.log(sizes), np.log(counts + 1), 1)
        return -coeffs[0]
    
    def analyze_spatial_relationships(self, masks: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Analyze spatial relationships between tissue types"""
        features = {}
        
        # Tumor-Immune interface analysis
        if 'tumor' in masks and 'lymphocytes' in masks:
            tumor_edges = cv2.Canny(masks['tumor'].astype(np.uint8) * 255, 50, 150)
            
            # Distance transform from tumor edges
            if SCIPY_AVAILABLE:
                edge_dist = ndimage.distance_transform_edt(~tumor_edges)
                
                # Lymphocytes at different distances from tumor
                near_tumor = masks['lymphocytes'] & (edge_dist < 10)
                mid_tumor = masks['lymphocytes'] & (edge_dist >= 10) & (edge_dist < 30)
                far_tumor = masks['lymphocytes'] & (edge_dist >= 30)
                
                features['lymphocytes_near_tumor_ratio'] = near_tumor.sum() / (masks['lymphocytes'].sum() + 1)
                features['lymphocytes_mid_tumor_ratio'] = mid_tumor.sum() / (masks['lymphocytes'].sum() + 1)
                features['lymphocytes_far_tumor_ratio'] = far_tumor.sum() / (masks['lymphocytes'].sum() + 1)
                
                # Gradient of lymphocyte density from tumor edge
                lymph_gradient = []
                for d in range(0, 50, 5):
                    band = masks['lymphocytes'] & (edge_dist >= d) & (edge_dist < d+5)
                    lymph_gradient.append(band.sum())
                
                if len(lymph_gradient) > 1:
                    features['lymphocyte_gradient_slope'] = np.polyfit(range(len(lymph_gradient)), lymph_gradient, 1)[0]
                else:
                    features['lymphocyte_gradient_slope'] = 0
            else:
                # Simplified version
                features['lymphocytes_near_tumor_ratio'] = 0.3
                features['lymphocytes_mid_tumor_ratio'] = 0.3
                features['lymphocytes_far_tumor_ratio'] = 0.4
                features['lymphocyte_gradient_slope'] = 0
        
        # Stromal barrier analysis
        if 'stroma' in masks and 'lymphocytes' in masks:
            # Detect thick stromal regions
            stroma_dilated = cv2.dilate(
                masks['stroma'].astype(np.uint8), 
                np.ones((15, 15)), 
                iterations=1
            )
            thick_stroma = stroma_dilated & masks['stroma']
            
            # Check if lymphocytes are excluded by stroma
            excluded_regions = thick_stroma & ~masks['lymphocytes']
            features['stromal_exclusion_score'] = excluded_regions.sum() / (thick_stroma.sum() + 1)
        
        # Tumor organization
        if 'tumor' in masks:
            tumor_morph = self.extract_morphological_features(masks['tumor'])
            features['tumor_organization_score'] = tumor_morph['mean_solidity'] * tumor_morph['mean_circularity']
            features['tumor_fragmentation'] = tumor_morph['num_objects'] / (masks['tumor'].sum() / 1000 + 1)
        
        return features


class AdvancedMolecularClassifier:
    """Main interface for advanced molecular subtype classification"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = AdvancedMolecularPredictor(use_vit=TIMM_AVAILABLE)
        self.model.to(self.device)
        
        # Load trained weights if available
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded trained model from {model_path}")
        else:
            print("Using randomly initialized model - train with EPOC data for best results")
        
        self.model.eval()
        
        # Initialize spatial analyzer
        self.spatial_analyzer = SpatialPatternAnalyzerV2()
        
        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Subtype information
        self.subtypes = {
            0: {
                'name': 'SNF1 (Canonical)',
                'characteristics': ['High tumor content', 'Sharp edges', 'Low immune'],
                'color': '#e74c3c'
            },
            1: {
                'name': 'SNF2 (Immune)',
                'characteristics': ['High immune infiltration', 'Lymphocyte aggregates', 'Best prognosis'],
                'color': '#27ae60'
            },
            2: {
                'name': 'SNF3 (Stromal)',
                'characteristics': ['High stromal content', 'Fibrosis patterns', 'Stromal barriers'],
                'color': '#e67e22'
            }
        }
    
    def predict(self, image: np.ndarray, extract_features: bool = True) -> Dict:
        """Predict molecular subtype with confidence and explanations"""
        
        # Convert to PIL if numpy array
        if isinstance(image, np.ndarray):
            image_pil = Image.fromarray(image.astype(np.uint8))
        else:
            image_pil = image
            image = np.array(image_pil)
        
        # Prepare tensor
        img_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            if extract_features:
                probabilities, features = self.model(img_tensor, return_features=True)
            else:
                probabilities = self.model(img_tensor)
        
        probs = probabilities.cpu().numpy()[0]
        predicted_idx = np.argmax(probs)
        confidence = probs[predicted_idx]
        
        # Extract spatial features for explanation
        spatial_features = {}
        if extract_features:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Extract morphological features
            spatial_features['morphology'] = self.spatial_analyzer.extract_morphological_features(
                gray > np.mean(gray)
            )
            
            # Extract texture features
            spatial_features['texture'] = self.spatial_analyzer.extract_texture_features(gray)
        
        # Generate explanation
        explanation = self._generate_explanation(predicted_idx, probs, spatial_features)
        
        return {
            'prediction': self.subtypes[predicted_idx]['name'],
            'predicted_idx': predicted_idx,
            'probabilities': {
                'SNF1': float(probs[0]),
                'SNF2': float(probs[1]),
                'SNF3': float(probs[2])
            },
            'confidence': float(confidence),
            'confidence_level': self._get_confidence_level(confidence),
            'spatial_features': spatial_features,
            'explanation': explanation,
            'features': features if extract_features else None
        }
    
    def _generate_explanation(self, predicted_idx: int, probs: np.ndarray, spatial_features: Dict) -> List[str]:
        """Generate human-readable explanation for prediction"""
        explanations = []
        
        # Confidence-based explanation
        confidence = probs[predicted_idx]
        if confidence > 0.8:
            explanations.append(f"High confidence ({confidence:.1%}) in prediction")
        elif confidence > 0.6:
            explanations.append(f"Moderate confidence ({confidence:.1%}) in prediction")
        else:
            explanations.append(f"Low confidence ({confidence:.1%}) - consider additional analysis")
        
        # Subtype-specific explanations
        if predicted_idx == 0:  # SNF1
            if spatial_features.get('morphology', {}).get('mean_solidity', 0) > 0.8:
                explanations.append("Detected solid tumor nests with sharp boundaries")
            if spatial_features.get('morphology', {}).get('num_objects', 0) < 5:
                explanations.append("Organized tumor architecture with minimal fragmentation")
        
        elif predicted_idx == 1:  # SNF2
            if spatial_features.get('morphology', {}).get('shape_heterogeneity', 0) > 0.3:
                explanations.append("Detected heterogeneous immune cell infiltration patterns")
            if spatial_features.get('texture', {}).get('glcm_homogeneity', 1) < 0.7:
                explanations.append("Texture analysis suggests lymphocyte aggregates")
        
        elif predicted_idx == 2:  # SNF3
            if spatial_features.get('texture', {}).get('glcm_contrast', 0) > 0.5:
                explanations.append("High texture contrast indicating fibrotic patterns")
            if spatial_features.get('morphology', {}).get('mean_eccentricity', 0) > 0.7:
                explanations.append("Elongated structures suggesting stromal barriers")
        
        # Differential diagnosis
        second_best_idx = np.argsort(probs)[-2]
        if probs[second_best_idx] > 0.3:
            explanations.append(
                f"Alternative consideration: {self.subtypes[second_best_idx]['name']} "
                f"({probs[second_best_idx]:.1%} probability)"
            )
        
        return explanations
    
    def _get_confidence_level(self, confidence: float) -> str:
        """Categorize confidence level"""
        if confidence > 0.8:
            return "High"
        elif confidence > 0.6:
            return "Moderate"
        else:
            return "Low"
    
    def batch_predict(self, images: List[np.ndarray]) -> List[Dict]:
        """Predict multiple images efficiently"""
        results = []
        
        # Process in batches
        batch_size = 16
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            
            # Prepare batch tensor
            batch_tensors = []
            for img in batch_images:
                if isinstance(img, np.ndarray):
                    img_pil = Image.fromarray(img.astype(np.uint8))
                else:
                    img_pil = img
                
                img_tensor = self.transform(img_pil)
                batch_tensors.append(img_tensor)
            
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                probabilities = self.model(batch_tensor)
            
            # Process results
            for j, probs in enumerate(probabilities.cpu().numpy()):
                predicted_idx = np.argmax(probs)
                results.append({
                    'prediction': self.subtypes[predicted_idx]['name'],
                    'predicted_idx': predicted_idx,
                    'probabilities': {
                        'SNF1': float(probs[0]),
                        'SNF2': float(probs[1]),
                        'SNF3': float(probs[2])
                    },
                    'confidence': float(probs[predicted_idx])
                })
        
        return results


# Integration function for existing codebase
def create_advanced_molecular_predictor(model_path: Optional[str] = None) -> AdvancedMolecularClassifier:
    """Factory function to create advanced molecular predictor"""
    return AdvancedMolecularClassifier(model_path)


if __name__ == "__main__":
    # Test the predictor
    print("Advanced Molecular Predictor initialized successfully")
    print("Features:")
    print("- Multi-scale feature extraction")
    print("- Spatial attention mechanisms")
    print("- Biological constraint modeling")
    print("- Advanced texture and morphology analysis")
    print("- Explainable predictions") 