"""
Virtual IHC Prediction Network
Predicts immunohistochemistry (IHC) markers from H&E stained images
Ready for training on paired H&E/IHC data when available
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional
from torchvision import models


class IHCMarkerHead(nn.Module):
    """
    Prediction head for a specific IHC marker
    """
    
    def __init__(self, in_features=2048, hidden_dim=512):
        super().__init__()
        
        # Feature refinement
        self.feature_refine = nn.Sequential(
            nn.Conv2d(in_features, hidden_dim, 1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        
        # Marker-specific decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, 1),  # Single channel output
            nn.Sigmoid()  # Probability map
        )
        
    def forward(self, features):
        """Generate marker probability map from features"""
        refined = self.feature_refine(features)
        marker_map = self.decoder(refined)
        return marker_map


class VirtualIHCNetwork(nn.Module):
    """
    Predicts multiple IHC markers from H&E stained images
    """
    
    def __init__(self, backbone='resnet50', pretrained=True):
        super().__init__()
        
        # Shared feature extractor
        if backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            # Remove final layers
            self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
            
        # IHC marker heads
        self.marker_heads = nn.ModuleDict({
            'cd3': IHCMarkerHead(feature_dim),     # T-cells (all)
            'cd8': IHCMarkerHead(feature_dim),     # Cytotoxic T-cells
            'cd20': IHCMarkerHead(feature_dim),    # B-cells
            'cd68': IHCMarkerHead(feature_dim),    # Macrophages
            'ki67': IHCMarkerHead(feature_dim),    # Proliferation
            'sma': IHCMarkerHead(feature_dim),     # Smooth muscle actin (fibrosis)
            'panck': IHCMarkerHead(feature_dim),   # Pan-cytokeratin (epithelial)
            'vim': IHCMarkerHead(feature_dim)      # Vimentin (mesenchymal)
        })
        
        # Marker relationships for consistency
        self.marker_relationships = {
            'cd8': 'cd3',  # CD8+ is subset of CD3+
            'cd20': None,   # Independent
            'cd68': None,   # Independent
            'ki67': None,   # Can overlap with others
            'sma': None,    # Stromal marker
            'panck': None,  # Epithelial marker
            'vim': None     # Mesenchymal marker
        }
        
    def forward(self, x, markers=None):
        """
        Predict IHC marker expressions
        
        Args:
            x: H&E image tensor [B, 3, H, W]
            markers: List of markers to predict (None = all)
            
        Returns:
            Dictionary of marker probability maps
        """
        # Extract shared features
        features = self.backbone(x)
        
        # Predict specified markers
        if markers is None:
            markers = list(self.marker_heads.keys())
            
        predictions = {}
        for marker in markers:
            if marker in self.marker_heads:
                predictions[marker] = self.marker_heads[marker](features)
                
        # Apply biological constraints
        predictions = self._apply_biological_constraints(predictions)
        
        return predictions
    
    def _apply_biological_constraints(self, predictions):
        """Apply known biological relationships between markers"""
        # CD8+ cells must be subset of CD3+ cells
        if 'cd8' in predictions and 'cd3' in predictions:
            predictions['cd8'] = torch.min(predictions['cd8'], predictions['cd3'])
            
        # Pan-CK and Vimentin are usually mutually exclusive
        if 'panck' in predictions and 'vim' in predictions:
            # Soft constraint - reduce overlap
            overlap = predictions['panck'] * predictions['vim']
            predictions['panck'] = predictions['panck'] - 0.5 * overlap
            predictions['vim'] = predictions['vim'] - 0.5 * overlap
            
        return predictions
    
    def predict_molecular_features(self, x):
        """
        Predict molecular subtype-relevant features from virtual IHC
        
        Returns aggregated features useful for molecular subtyping
        """
        # Get all marker predictions
        markers = self.forward(x)
        
        # Aggregate into molecular features
        features = {}
        
        # Immune infiltration score (CD3, CD8, CD20)
        immune_markers = []
        for marker in ['cd3', 'cd8', 'cd20']:
            if marker in markers:
                # Global average of marker expression
                expr = markers[marker].mean(dim=(2, 3))  # [B, 1]
                immune_markers.append(expr)
                
        if immune_markers:
            features['immune_infiltration'] = torch.cat(immune_markers, dim=1).mean(dim=1)
        else:
            features['immune_infiltration'] = torch.zeros(x.size(0))
            
        # Macrophage presence (CD68)
        if 'cd68' in markers:
            features['macrophage_density'] = markers['cd68'].mean(dim=(2, 3)).squeeze()
            
        # Proliferation index (Ki67)
        if 'ki67' in markers:
            features['proliferation_index'] = markers['ki67'].mean(dim=(2, 3)).squeeze()
            
        # Fibrosis score (SMA)
        if 'sma' in markers:
            features['fibrosis_score'] = markers['sma'].mean(dim=(2, 3)).squeeze()
            
        # EMT score (Vim/PanCK ratio)
        if 'vim' in markers and 'panck' in markers:
            vim_expr = markers['vim'].mean(dim=(2, 3))
            panck_expr = markers['panck'].mean(dim=(2, 3))
            features['emt_score'] = vim_expr / (panck_expr + vim_expr + 1e-6)
            features['emt_score'] = features['emt_score'].squeeze()
            
        return features


class VirtualIHCLoss(nn.Module):
    """
    Loss function for training virtual IHC predictor
    """
    
    def __init__(self, marker_weights=None):
        super().__init__()
        self.marker_weights = marker_weights or {}
        
    def forward(self, predictions, targets, masks=None):
        """
        Compute loss between predicted and real IHC
        
        Args:
            predictions: Dict of predicted marker maps
            targets: Dict of ground truth marker maps
            masks: Optional tissue masks to focus on relevant regions
            
        Returns:
            Total loss and per-marker losses
        """
        total_loss = 0
        marker_losses = {}
        
        for marker, pred in predictions.items():
            if marker in targets:
                target = targets[marker]
                
                # Binary cross-entropy for probability maps
                loss = F.binary_cross_entropy(pred, target, reduction='none')
                
                # Apply tissue mask if provided
                if masks is not None and marker in masks:
                    loss = loss * masks[marker]
                    
                # Average over spatial dimensions
                loss = loss.mean()
                
                # Apply marker-specific weight
                weight = self.marker_weights.get(marker, 1.0)
                weighted_loss = loss * weight
                
                total_loss += weighted_loss
                marker_losses[marker] = loss.item()
                
        return total_loss, marker_losses


class VirtualIHCPredictor:
    """
    Wrapper class for virtual IHC prediction with pre/post-processing
    """
    
    def __init__(self, model_path=None):
        self.model = VirtualIHCNetwork()
        
        if model_path:
            self.load_model(model_path)
        else:
            print("Virtual IHC Network initialized (untrained)")
            print("Train on paired H&E/IHC data for accurate predictions")
            
        self.model.eval()
        
    def load_model(self, model_path):
        """Load pre-trained model weights"""
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded Virtual IHC model from {model_path}")
        
    def predict(self, image, transform=None):
        """
        Predict IHC markers from H&E image
        
        Args:
            image: H&E image (numpy array or PIL Image)
            transform: Optional image transformation
            
        Returns:
            Dictionary of marker predictions and molecular features
        """
        # Prepare image
        if transform:
            img_tensor = transform(image).unsqueeze(0)
        else:
            # Default transform
            from torchvision import transforms
            default_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            img_tensor = default_transform(image).unsqueeze(0)
            
        # Predict markers
        with torch.no_grad():
            marker_maps = self.model(img_tensor)
            molecular_features = self.model.predict_molecular_features(img_tensor)
            
        # Convert to numpy
        results = {
            'marker_maps': {
                marker: prob_map.squeeze().cpu().numpy()
                for marker, prob_map in marker_maps.items()
            },
            'molecular_features': {
                feat: score.cpu().numpy()
                for feat, score in molecular_features.items()
            }
        }
        
        # Add interpretations
        results['interpretations'] = self._interpret_results(results)
        
        return results
    
    def _interpret_results(self, results):
        """Provide biological interpretation of predictions"""
        interp = {}
        features = results['molecular_features']
        
        # Immune status
        if 'immune_infiltration' in features:
            score = features['immune_infiltration']
            if score > 0.3:
                interp['immune_status'] = 'Hot (High infiltration)'
            elif score > 0.1:
                interp['immune_status'] = 'Warm (Moderate infiltration)'
            else:
                interp['immune_status'] = 'Cold (Low infiltration)'
                
        # Fibrosis level
        if 'fibrosis_score' in features:
            score = features['fibrosis_score']
            if score > 0.4:
                interp['fibrosis_level'] = 'High (Desmoplastic)'
            elif score > 0.2:
                interp['fibrosis_level'] = 'Moderate'
            else:
                interp['fibrosis_level'] = 'Low'
                
        # EMT status
        if 'emt_score' in features:
            score = features['emt_score']
            if score > 0.6:
                interp['emt_status'] = 'High (Mesenchymal-like)'
            elif score > 0.4:
                interp['emt_status'] = 'Intermediate'
            else:
                interp['emt_status'] = 'Low (Epithelial-like)'
                
        return interp


# Example usage for molecular subtyping integration
def create_virtual_ihc_features(image, model_path=None):
    """
    Extract molecular-relevant features using virtual IHC
    
    Returns features that can be integrated with molecular subtyping
    """
    predictor = VirtualIHCPredictor(model_path)
    results = predictor.predict(image)
    
    # Extract key features for molecular subtyping
    features = results['molecular_features']
    
    # Map to expected molecular subtypes
    subtype_features = {
        'snf1_score': 1.0 - features.get('immune_infiltration', 0.5),  # Low immune
        'snf2_score': features.get('immune_infiltration', 0.5),        # High immune
        'snf3_score': features.get('fibrosis_score', 0.5)              # High fibrosis
    }
    
    return subtype_features, results['interpretations'] 