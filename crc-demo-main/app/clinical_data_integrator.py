"""
Clinical Data Integration Framework for Multi-Modal CRC Analysis
Combines histopathology features with clinical variables for improved predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List


class ClinicalDataIntegrator(nn.Module):
    """
    Integrates clinical data with histopathology features for enhanced molecular subtyping
    Ready for EPOC clinical variables integration
    """
    
    def __init__(self, clinical_features_dim=10, hidden_dim=32, output_dim=16):
        super().__init__()
        
        # Clinical feature encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_features_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim)
        )
        
        # Feature names for documentation and validation
        self.clinical_feature_names = [
            'age',                    # Patient age (years)
            'sex',                    # Sex (0=female, 1=male)
            'cea_level',             # Carcinoembryonic antigen level
            'time_to_metastasis',    # Months from primary to metastasis
            'num_metastases',        # Number of liver metastases
            'primary_t_stage',       # Primary tumor T stage (1-4)
            'primary_n_stage',       # Primary tumor N stage (0-2)
            'synchronous',           # Synchronous vs metachronous (0/1)
            'prior_chemo',           # Prior chemotherapy (0/1)
            'liver_only'             # Liver-only disease (0/1)
        ]
        
        # Normalization parameters (will be learned from data)
        self.register_buffer('feature_means', torch.zeros(clinical_features_dim))
        self.register_buffer('feature_stds', torch.ones(clinical_features_dim))
        
    def encode_clinical_features(self, clinical_data: Dict[str, float]) -> torch.Tensor:
        """
        Encode clinical features into a fixed-size representation
        
        Args:
            clinical_data: Dictionary with clinical variables
            
        Returns:
            Encoded clinical features tensor
        """
        # Extract and normalize features
        features = []
        
        # Age (normalize to 0-1 range, assuming 0-100 years)
        features.append(clinical_data.get('age', 60) / 100.0)
        
        # Sex (already binary)
        features.append(float(clinical_data.get('sex', 0)))
        
        # CEA level (log transform and normalize)
        cea = clinical_data.get('cea_level', 5.0)
        features.append(np.log1p(cea) / 10.0)
        
        # Time to metastasis (normalize by 60 months = 5 years)
        features.append(clinical_data.get('time_to_metastasis', 12) / 60.0)
        
        # Number of metastases (normalize by 10)
        features.append(clinical_data.get('num_metastases', 1) / 10.0)
        
        # T stage (normalize by 4)
        features.append(clinical_data.get('primary_t_stage', 3) / 4.0)
        
        # N stage (normalize by 3)
        features.append(clinical_data.get('primary_n_stage', 1) / 3.0)
        
        # Binary features (already 0/1)
        features.append(float(clinical_data.get('synchronous', 0)))
        features.append(float(clinical_data.get('prior_chemo', 0)))
        features.append(float(clinical_data.get('liver_only', 1)))
        
        # Convert to tensor
        feature_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Apply learned normalization if available
        if self.training:
            # Update running statistics during training
            self.update_statistics(feature_tensor)
        
        # Normalize using running statistics
        normalized = (feature_tensor - self.feature_means) / (self.feature_stds + 1e-6)
        
        # Encode through neural network
        encoded = self.clinical_encoder(normalized)
        
        return encoded
    
    def update_statistics(self, features: torch.Tensor):
        """Update running statistics for normalization"""
        # Simple exponential moving average
        momentum = 0.1
        batch_mean = features.mean(dim=0)
        batch_std = features.std(dim=0)
        
        self.feature_means = (1 - momentum) * self.feature_means + momentum * batch_mean
        self.feature_stds = (1 - momentum) * self.feature_stds + momentum * batch_std
    
    def forward(self, clinical_data_batch: List[Dict[str, float]]) -> torch.Tensor:
        """
        Process a batch of clinical data
        
        Args:
            clinical_data_batch: List of clinical data dictionaries
            
        Returns:
            Batch of encoded clinical features
        """
        # Process all patients as a batch
        batch_features = []
        
        for clinical_data in clinical_data_batch:
            # Extract features for each patient
            features = []
            
            # Age (normalize to 0-1 range, assuming 0-100 years)
            features.append(clinical_data.get('age', 60) / 100.0)
            
            # Sex (already binary)
            features.append(float(clinical_data.get('sex', 0)))
            
            # CEA level (log transform and normalize)
            cea = clinical_data.get('cea_level', 5.0)
            features.append(np.log1p(cea) / 10.0)
            
            # Time to metastasis (normalize by 60 months = 5 years)
            features.append(clinical_data.get('time_to_metastasis', 12) / 60.0)
            
            # Number of metastases (normalize by 10)
            features.append(clinical_data.get('num_metastases', 1) / 10.0)
            
            # T stage (normalize by 4)
            features.append(clinical_data.get('primary_t_stage', 3) / 4.0)
            
            # N stage (normalize by 3)
            features.append(clinical_data.get('primary_n_stage', 1) / 3.0)
            
            # Binary features (already 0/1)
            features.append(float(clinical_data.get('synchronous', 0)))
            features.append(float(clinical_data.get('prior_chemo', 0)))
            features.append(float(clinical_data.get('liver_only', 1)))
            
            batch_features.append(features)
        
        # Convert to tensor
        feature_tensor = torch.FloatTensor(batch_features)
        
        # Normalize using running statistics
        normalized = (feature_tensor - self.feature_means) / (self.feature_stds + 1e-6)
        
        # Encode through neural network (all at once)
        encoded = self.clinical_encoder(normalized)
        
        return encoded


class MultiModalFusionNetwork(nn.Module):
    """
    Fuses histopathology and clinical features for final prediction
    """
    
    def __init__(self, histology_dim=512, clinical_dim=16, num_classes=3):
        super().__init__()
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(histology_dim, clinical_dim)
        
        # Fusion layers
        fusion_dim = histology_dim + clinical_dim
        self.fusion_network = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Gating mechanism to weight modalities
        self.histology_gate = nn.Sequential(
            nn.Linear(histology_dim, 1),
            nn.Sigmoid()
        )
        
        self.clinical_gate = nn.Sequential(
            nn.Linear(clinical_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, histology_features: torch.Tensor, 
                clinical_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse multi-modal features for prediction
        
        Args:
            histology_features: Features from histopathology analysis
            clinical_features: Encoded clinical features
            
        Returns:
            Class logits
        """
        # Apply cross-modal attention
        hist_attended, clin_attended = self.cross_attention(
            histology_features, clinical_features
        )
        
        # Compute modality importance gates
        hist_weight = self.histology_gate(hist_attended)
        clin_weight = self.clinical_gate(clin_attended)
        
        # Weighted combination
        hist_weighted = hist_attended * hist_weight
        clin_weighted = clin_attended * clin_weight
        
        # Concatenate and fuse
        fused = torch.cat([hist_weighted, clin_weighted], dim=1)
        
        # Final prediction
        output = self.fusion_network(fused)
        
        return output


class CrossModalAttention(nn.Module):
    """
    Attention mechanism between histology and clinical features
    """
    
    def __init__(self, histology_dim, clinical_dim):
        super().__init__()
        
        # Project to common dimension
        common_dim = 128
        self.hist_proj = nn.Linear(histology_dim, common_dim)
        self.clin_proj = nn.Linear(clinical_dim, common_dim)
        
        # Attention layers
        self.attention = nn.MultiheadAttention(common_dim, num_heads=4)
        
        # Output projections
        self.hist_out = nn.Linear(common_dim, histology_dim)
        self.clin_out = nn.Linear(common_dim, clinical_dim)
        
    def forward(self, histology_features, clinical_features):
        """Apply cross-modal attention"""
        # Project to common space
        hist_proj = self.hist_proj(histology_features).unsqueeze(0)
        clin_proj = self.clin_proj(clinical_features).unsqueeze(0)
        
        # Cross attention: histology attending to clinical
        hist_attended, _ = self.attention(
            query=hist_proj,
            key=clin_proj,
            value=clin_proj
        )
        
        # Cross attention: clinical attending to histology
        clin_attended, _ = self.attention(
            query=clin_proj,
            key=hist_proj,
            value=hist_proj
        )
        
        # Project back to original dimensions
        hist_attended = self.hist_out(hist_attended.squeeze(0))
        clin_attended = self.clin_out(clin_attended.squeeze(0))
        
        # Residual connections
        hist_final = histology_features + hist_attended
        clin_final = clinical_features + clin_attended
        
        return hist_final, clin_final


class ClinicalDataValidator:
    """
    Validates and preprocesses clinical data
    """
    
    @staticmethod
    def validate_clinical_data(clinical_data: Dict[str, float]) -> Dict[str, float]:
        """
        Validate and fill missing clinical data with reasonable defaults
        """
        defaults = {
            'age': 60,
            'sex': 0,
            'cea_level': 5.0,
            'time_to_metastasis': 12,
            'num_metastases': 1,
            'primary_t_stage': 3,
            'primary_n_stage': 1,
            'synchronous': 0,
            'prior_chemo': 0,
            'liver_only': 1
        }
        
        # Validate ranges
        validated = {}
        for key, default in defaults.items():
            value = clinical_data.get(key, default)
            
            # Apply range constraints
            if key == 'age':
                value = np.clip(value, 18, 100)
            elif key == 'sex':
                value = int(value in [1, '1', 'M', 'Male', 'male'])
            elif key == 'cea_level':
                value = max(0, value)
            elif key == 'time_to_metastasis':
                value = max(0, value)
            elif key == 'num_metastases':
                value = max(1, int(value))
            elif key == 'primary_t_stage':
                value = np.clip(int(value), 1, 4)
            elif key == 'primary_n_stage':
                value = np.clip(int(value), 0, 2)
            elif key in ['synchronous', 'prior_chemo', 'liver_only']:
                value = int(bool(value))
                
            validated[key] = value
            
        return validated


# Example usage function
def create_multimodal_predictor(histology_model_dim=512):
    """
    Create a complete multi-modal prediction system
    """
    # Clinical data encoder
    clinical_encoder = ClinicalDataIntegrator()
    
    # Multi-modal fusion
    fusion_network = MultiModalFusionNetwork(
        histology_dim=histology_model_dim,
        clinical_dim=16,
        num_classes=3  # SNF1, SNF2, SNF3
    )
    
    return clinical_encoder, fusion_network 