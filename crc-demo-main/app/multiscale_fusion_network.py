#!/usr/bin/env python3
"""
Multi-Scale Feature Fusion Network for CRC Molecular Subtyping
Captures hierarchical patterns from cellular to tissue level
Key innovation: Process WSI tiles at multiple scales to capture different biological patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from typing import List, Tuple, Dict, Optional
import cv2
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class MultiScaleFeatureExtractor(nn.Module):
    """
    Extract features at multiple scales from WSI tiles
    Captures patterns from cellular (224x224) to regional (56x56) level
    """
    
    def __init__(self, base_encoder, scales=[1.0, 0.5, 0.25], feature_dim=512):
        super().__init__()
        self.scales = scales
        self.feature_dim = feature_dim
        
        # Use provided base encoder (e.g., RevolutionaryMolecularNet)
        self.base_encoder = base_encoder
        
        # Determine encoder output dimension
        # Handle different encoder types
        if hasattr(base_encoder, 'output_dim'):
            self.encoder_out_dim = base_encoder.output_dim
        elif hasattr(base_encoder, 'fc'):
            # ResNet-like architecture
            if hasattr(base_encoder.fc, 'in_features'):
                self.encoder_out_dim = base_encoder.fc.in_features
            else:
                self.encoder_out_dim = 512
        else:
            # Default for custom encoders
            self.encoder_out_dim = 512
        
        # Debug encoder output dimension
        print(f"[DEBUG] MultiScaleFeatureExtractor init: encoder_out_dim = {self.encoder_out_dim}")
        
        # Scale-specific adaptation layers
        self.scale_adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(self.encoder_out_dim, feature_dim, 1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True),
                nn.Dropout2d(0.1)
            ) for _ in scales
        ])
        
        # Cross-scale attention mechanism
        self.cross_scale_attention = CrossScaleAttention(
            feature_dim, 
            num_scales=len(scales)
        )
        
        # Feature pyramid network for multi-scale fusion
        self.fpn = FeaturePyramidNetwork(feature_dim, len(scales))
        
        # Scale-aware pooling
        self.scale_pooling = ScaleAwarePooling(feature_dim)
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract multi-scale features with attention-based fusion
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            Dictionary with 'features', 'scale_features', 'attention_weights', 'scale_maps'
        """
        batch_size = x.shape[0]
        scale_features = []
        scale_maps = []
        
        # Extract features at each scale
        for i, scale in enumerate(self.scales):
            if scale != 1.0:
                # Resize input for different scales
                scaled_size = int(224 * scale)
                scaled_x = F.interpolate(
                    x, 
                    size=(scaled_size, scaled_size), 
                    mode='bilinear', 
                    align_corners=False
                )
            else:
                scaled_x = x
            
            # Extract features using base encoder
            # Handle different encoder types
            if hasattr(self.base_encoder, 'cnn_backbone'):
                # Revolutionary encoder that needs bio_features
                # For multi-scale, we pass dummy bio features
                dummy_bio = torch.zeros(batch_size, 20).to(scaled_x.device)
                features = self.base_encoder(scaled_x, dummy_bio)
            else:
                # Standard encoder (ResNet, etc.)
                features = self.base_encoder(scaled_x)
            
            # Debug: check feature shapes
            if i == 0 and not hasattr(self, '_debug_encoder'):
                print(f"[DEBUG] Scale {scale}: encoder output shape: {features.shape}")
                self._debug_encoder = True
            
            # Handle different output shapes from encoder
            if len(features.shape) == 4:  # Conv features [B, C, H, W]
                # Preserve spatial dimensions for FPN
                spatial_features = features
            else:  # Already pooled features [B, C]
                # Reshape to spatial format for consistency
                spatial_features = features.view(batch_size, -1, 1, 1)
                spatial_features = F.interpolate(
                    spatial_features, 
                    size=(7, 7), 
                    mode='nearest'
                )
            
            # Ensure consistent spatial size across scales
            if spatial_features.shape[2:] != (7, 7):
                spatial_features = F.adaptive_avg_pool2d(spatial_features, (7, 7))
            
            # Debug spatial features
            if i == 0 and not hasattr(self, '_debug_spatial'):
                print(f"[DEBUG] Spatial features shape before adapter: {spatial_features.shape}")
                print(f"[DEBUG] Expected channels: {self.encoder_out_dim}")
                self._debug_spatial = True
            
            # Apply scale-specific adaptation
            adapted_features = self.scale_adapters[i](spatial_features)
            scale_features.append(adapted_features)
            scale_maps.append(spatial_features)
        
        # Apply cross-scale attention
        attended_features, attention_weights = self.cross_scale_attention(scale_features)
        
        # Fuse with Feature Pyramid Network
        fused_features = self.fpn(attended_features)
        
        # Scale-aware pooling for final representation
        final_features = self.scale_pooling(fused_features, scale_features)
        
        return {
            'features': final_features,
            'scale_features': scale_features,
            'attention_weights': attention_weights,
            'scale_maps': scale_maps,
            'fused_map': fused_features
        }


class CrossScaleAttention(nn.Module):
    """
    Attention mechanism to capture relationships across different scales
    Key insight: Cellular patterns inform tissue-level organization
    """
    
    def __init__(self, feature_dim, num_scales, num_heads=8):
        super().__init__()
        self.num_scales = num_scales
        self.num_heads = num_heads
        self.feature_dim = feature_dim
        
        # Multi-head attention for cross-scale interaction
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Scale embedding to differentiate between scales
        self.scale_embedding = nn.Parameter(
            torch.randn(num_scales, feature_dim) * 0.02
        )
        
        # Output projection with residual
        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
    def forward(self, scale_features: List[torch.Tensor]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Apply cross-scale attention to learn scale relationships
        
        Args:
            scale_features: List of features at different scales [B, C, H, W]
            
        Returns:
            Attended features and attention weights
        """
        batch_size = scale_features[0].shape[0]
        
        # Flatten spatial dimensions and add scale embeddings
        flattened_features = []
        position_indices = []
        
        for i, features in enumerate(scale_features):
            B, C, H, W = features.shape
            # Flatten: [B, C, H, W] -> [B, H*W, C]
            features_flat = features.view(B, C, -1).transpose(1, 2)
            
            # Add scale embedding
            features_flat = features_flat + self.scale_embedding[i].unsqueeze(0).unsqueeze(0)
            flattened_features.append(features_flat)
            position_indices.append(H * W)
        
        # Concatenate all scales: [B, total_positions, C]
        all_features = torch.cat(flattened_features, dim=1)
        
        # Self-attention across all scales and positions
        attended_features, attention_weights = self.attention(
            all_features, all_features, all_features
        )
        
        # Add residual connection
        attended_features = attended_features + all_features
        
        # Project output
        attended_features = self.output_proj(attended_features)
        
        # Reshape back to scale-specific features
        attended_scale_features = []
        start_idx = 0
        
        for i, num_positions in enumerate(position_indices):
            # Extract features for this scale
            scale_attended = attended_features[:, start_idx:start_idx + num_positions, :]
            
            # Reshape back to spatial format
            H = W = int(np.sqrt(num_positions))
            scale_attended = scale_attended.transpose(1, 2).view(batch_size, -1, H, W)
            attended_scale_features.append(scale_attended)
            
            start_idx += num_positions
        
        return attended_scale_features, attention_weights


class FeaturePyramidNetwork(nn.Module):
    """
    FPN for fusing multi-scale features with top-down and bottom-up pathways
    Inspired by object detection but adapted for histopathology
    """
    
    def __init__(self, feature_dim, num_scales):
        super().__init__()
        self.num_scales = num_scales
        
        # Lateral connections (1x1 convs)
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(feature_dim, feature_dim, 1)
            for _ in range(num_scales)
        ])
        
        # Top-down pathway (3x3 convs)
        self.td_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_scales - 1)
        ])
        
        # Bottom-up pathway for bidirectional fusion
        self.bu_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True)
            )
            for _ in range(num_scales - 1)
        ])
        
        # Final fusion with channel attention
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim * num_scales, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            ChannelAttention(feature_dim),
            nn.Conv2d(feature_dim, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multi-scale features using bidirectional FPN
        
        Args:
            scale_features: Features at different scales (finest to coarsest)
            
        Returns:
            Fused feature map
        """
        # Apply lateral connections
        laterals = [
            self.lateral_convs[i](scale_features[i]) 
            for i in range(self.num_scales)
        ]
        
        # Top-down pathway (coarse to fine)
        td_features = laterals.copy()
        for i in range(self.num_scales - 1, 0, -1):
            # Upsample coarser features
            upsampled = F.interpolate(
                td_features[i], 
                size=td_features[i-1].shape[2:],
                mode='bilinear',
                align_corners=False
            )
            
            # Add to finer features
            td_features[i-1] = td_features[i-1] + upsampled
            
            # Apply convolution
            if i - 1 < len(self.td_convs):
                td_features[i-1] = self.td_convs[i-1](td_features[i-1])
        
        # Bottom-up pathway (fine to coarse)
        bu_features = laterals.copy()
        for i in range(self.num_scales - 1):
            # Downsample finer features
            downsampled = self.bu_convs[i](bu_features[i])
            
            # Ensure size match
            if downsampled.shape[2:] != bu_features[i+1].shape[2:]:
                downsampled = F.adaptive_avg_pool2d(
                    downsampled, 
                    bu_features[i+1].shape[2:]
                )
            
            # Add to coarser features
            bu_features[i+1] = bu_features[i+1] + downsampled
        
        # Combine top-down and bottom-up features
        combined_features = []
        for td, bu in zip(td_features, bu_features):
            combined = td + bu
            combined_features.append(combined)
        
        # Resize all features to same size (use finest scale size)
        target_size = combined_features[0].shape[2:]
        resized_features = []
        
        for features in combined_features:
            if features.shape[2:] != target_size:
                resized = F.interpolate(
                    features, 
                    size=target_size,
                    mode='bilinear',
                    align_corners=False
                )
                resized_features.append(resized)
            else:
                resized_features.append(features)
        
        # Concatenate and fuse all scales
        concat_features = torch.cat(resized_features, dim=1)
        fused = self.fusion_conv(concat_features)
        
        return fused


class ChannelAttention(nn.Module):
    """Channel attention module to weight feature importance"""
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        
        # Max pooling
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply
        out = (avg_out + max_out).view(b, c, 1, 1)
        return x * out.expand_as(x)


class ScaleAwarePooling(nn.Module):
    """
    Scale-aware pooling that weights features based on their scale
    Cellular details vs tissue architecture
    """
    
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        
        # Learn scale importance weights
        self.scale_weights = nn.Sequential(
            nn.Linear(feature_dim * 3, 128),  # 3 scales
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 3),
            nn.Softmax(dim=1)
        )
        
        # Final projection
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
    def forward(self, fused_features: torch.Tensor, 
                scale_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Pool features with scale-aware weighting
        
        Args:
            fused_features: FPN output [B, C, H, W]
            scale_features: Original features at each scale
            
        Returns:
            Final feature vector [B, C]
        """
        # Global average pooling on fused features
        fused_pooled = F.adaptive_avg_pool2d(fused_features, 1).squeeze(-1).squeeze(-1)
        
        # Pool each scale
        scale_pooled = []
        for features in scale_features:
            pooled = F.adaptive_avg_pool2d(features, 1).squeeze(-1).squeeze(-1)
            scale_pooled.append(pooled)
        
        # Concatenate scale features
        scale_concat = torch.cat(scale_pooled, dim=1)
        
        # Learn scale importance
        scale_weights = self.scale_weights(scale_concat)
        
        # Weighted combination of scale features
        # scale_weights shape: [B, 3]
        # scale_pooled: list of 3 tensors each [B, feature_dim]
        weighted_features = torch.zeros_like(scale_pooled[0])
        for i, feat in enumerate(scale_pooled):
            weight = scale_weights[:, i:i+1]  # [B, 1]
            weighted_features += weight * feat
        
        # Combine with fused features
        final_features = fused_pooled + weighted_features
        
        # Final projection
        return self.projection(final_features)


class GlandularPatternAttention(nn.Module):
    """
    Specialized attention for glandular patterns in CRC
    Focuses on crypt architecture and organization
    """
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Gland boundary detection
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
        # Gland feature extraction
        self.gland_features = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=4, dilation=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Attention refinement
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract glandular patterns with attention
        
        Returns:
            Gland features and attention map
        """
        # Detect gland boundaries
        gland_attention = self.boundary_conv(x)
        
        # Extract gland features
        gland_feats = self.gland_features(x)
        
        # Apply attention
        attended = gland_feats * gland_attention
        
        # Refine
        refined = self.refine(attended)
        
        return refined, gland_attention


class MultiScaleCRCPredictor(nn.Module):
    """
    Complete multi-scale model for CRC molecular subtyping
    Integrates all components for end-to-end prediction
    """
    
    def __init__(self, base_encoder, num_classes=3, feature_dim=512, 
                 scales=[1.0, 0.5, 0.25], use_clinical=True):
        super().__init__()
        
        self.use_clinical = use_clinical
        
        # Multi-scale feature extractor
        self.multi_scale_extractor = MultiScaleFeatureExtractor(
            base_encoder=base_encoder,
            scales=scales,
            feature_dim=feature_dim
        )
        
        # Glandular pattern attention
        self.gland_attention = GlandularPatternAttention(
            in_channels=feature_dim,
            out_channels=feature_dim // 2
        )
        
        # Feature dimensions
        # Multi-scale extractor outputs feature_dim
        # Gland attention outputs feature_dim // 2
        self.base_feature_dim = feature_dim + feature_dim // 2  # Main + gland features
        self.clinical_dim = 32 if use_clinical else 0
        
        # Total dimension for classifier
        total_feature_dim = self.base_feature_dim + self.clinical_dim
        
        # Store for debugging
        self.expected_feature_dim = total_feature_dim
        
        # Create flexible classifier that adapts to input size
        self.classifier_base = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Input projection layers for different configurations
        self.proj_without_clinical = nn.Linear(self.base_feature_dim, 512)
        self.proj_with_clinical = nn.Linear(total_feature_dim, 512)
        
        # Uncertainty heads for different configurations
        self.uncertainty_without_clinical = nn.Sequential(
            nn.Linear(self.base_feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
        self.uncertainty_with_clinical = nn.Sequential(
            nn.Linear(total_feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x: torch.Tensor, 
                clinical_features: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-scale processing
        
        Args:
            x: Input images [B, C, H, W]
            clinical_features: Optional clinical features [B, clinical_dim]
            
        Returns:
            Dictionary with predictions and intermediate outputs
        """
        # Extract multi-scale features
        ms_output = self.multi_scale_extractor(x)
        ms_features = ms_output['features']
        
        # Extract glandular patterns from finest scale
        finest_scale = ms_output['scale_features'][0]
        gland_features, gland_attention_map = self.gland_attention(finest_scale)
        
        # Pool gland features
        gland_features_pooled = F.adaptive_avg_pool2d(gland_features, 1)
        # Ensure correct dimensions [B, C]
        if len(gland_features_pooled.shape) == 4:
            gland_features_pooled = gland_features_pooled.squeeze(-1).squeeze(-1)
        
        # Ensure ms_features is 2D [B, C]
        if len(ms_features.shape) > 2:
            ms_features = ms_features.view(ms_features.shape[0], -1)
        
        # Debug feature dimensions
        if not hasattr(self, '_debug_printed'):
            print(f"[DEBUG] ms_features shape: {ms_features.shape}")
            print(f"[DEBUG] gland_features_pooled shape: {gland_features_pooled.shape}")
            print(f"[DEBUG] Expected total dim: {self.expected_feature_dim}")
            self._debug_printed = True
        
        # Combine all features
        combined_features = torch.cat([ms_features, gland_features_pooled], dim=1)
        
        # Add clinical features if provided
        if self.use_clinical and clinical_features is not None:
            combined_features = torch.cat([combined_features, clinical_features], dim=1)
        
        # Verify dimensions
        if combined_features.shape[1] != self.expected_feature_dim:
            print(f"[WARNING] Feature dimension mismatch!")
            print(f"  Expected: {self.expected_feature_dim}")
            print(f"  Got: {combined_features.shape[1]}")
            print(f"  ms_features: {ms_features.shape[1]}")
            print(f"  gland_features: {gland_features_pooled.shape[1]}")
            if self.use_clinical and clinical_features is not None:
                print(f"  clinical_features: {clinical_features.shape[1]}")
        
        # Choose appropriate projection based on whether clinical features are present
        has_clinical = self.use_clinical and clinical_features is not None
        
        if has_clinical:
            projected = self.proj_with_clinical(combined_features)
            log_uncertainty = self.uncertainty_with_clinical(combined_features)
        else:
            projected = self.proj_without_clinical(combined_features)
            log_uncertainty = self.uncertainty_without_clinical(combined_features)
        
        # Final prediction
        logits = self.classifier_base(projected)
        
        # Uncertainty estimation
        uncertainty = F.softplus(log_uncertainty)
        
        return {
            'logits': logits,
            'probabilities': F.softmax(logits, dim=1),
            'uncertainty': uncertainty,
            'features': combined_features,
            'attention_weights': ms_output['attention_weights'],
            'gland_attention': gland_attention_map,
            'scale_features': ms_output['scale_features']
        }


def create_multiscale_model(base_encoder, config: Dict) -> MultiScaleCRCPredictor:
    """
    Factory function to create multi-scale model
    
    Args:
        base_encoder: Base feature extractor (e.g., RevolutionaryMolecularNet)
        config: Configuration dictionary
        
    Returns:
        Configured multi-scale model
    """
    model = MultiScaleCRCPredictor(
        base_encoder=base_encoder,
        num_classes=config.get('num_classes', 3),
        feature_dim=config.get('feature_dim', 512),
        scales=config.get('scales', [1.0, 0.5, 0.25]),
        use_clinical=config.get('use_clinical', True)
    )
    
    return model


# Test the implementation
if __name__ == "__main__":
    print("🔬 Multi-Scale Feature Fusion Network for CRC")
    print("="*60)
    
    # Create dummy base encoder
    from torchvision import models
    base_encoder = models.resnet50(pretrained=True)
    base_encoder.fc = nn.Identity()
    
    # Configuration
    config = {
        'num_classes': 3,
        'feature_dim': 512,
        'scales': [1.0, 0.5, 0.25],
        'use_clinical': True
    }
    
    # Create model
    model = create_multiscale_model(base_encoder, config)
    
    # Test forward pass
    batch_size = 2
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_clinical = torch.randn(batch_size, 32)
    
    with torch.no_grad():
        output = model(dummy_images, dummy_clinical)
    
    print("\n✅ Model created successfully!")
    print(f"Output shapes:")
    print(f"  - Logits: {output['logits'].shape}")
    print(f"  - Probabilities: {output['probabilities'].shape}")
    print(f"  - Uncertainty: {output['uncertainty'].shape}")
    print(f"  - Features: {output['features'].shape}")
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel statistics:")
    print(f"  - Total parameters: {total_params:,}")
    print(f"  - Trainable parameters: {trainable_params:,}")
    
    print("\n🎯 Expected benefits:")
    print("  - Captures cellular patterns (Scale 1.0)")
    print("  - Captures glandular architecture (Scale 0.5)")
    print("  - Captures regional organization (Scale 0.25)")
    print("  - Cross-scale attention for pattern relationships")
    print("  - Specialized glandular pattern detection")
    print("  - Uncertainty quantification built-in")
    
    print("\n💡 Integration ready for 96% accuracy target!") 