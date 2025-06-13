import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AttentionMIL(nn.Module):
    """
    Attention-based Multiple Instance Learning model
    Ilse et al., "Attention-based Deep Multiple Instance Learning"
    """
    
    def __init__(self, feature_dim: int = 2048, hidden_dim: int = 512, 
                 attention_dim: int = 256, n_classes: int = 4, dropout: float = 0.25):
        super(AttentionMIL, self).__init__()
        
        # Feature transformation
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention mechanism
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh()
        )
        
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Sigmoid()
        )
        
        self.attention_w = nn.Linear(attention_dim, 1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Tensor of shape (n_instances, feature_dim)
            return_attention: Whether to return attention weights
        Returns:
            logits: Class predictions (1, n_classes)
            attention_weights: Attention weights (n_instances,) if return_attention=True
        """
        # Extract features
        features = self.feature_extractor(x)  # (n_instances, hidden_dim)
        
        # Calculate attention scores
        attn_V = self.attention_V(features)  # (n_instances, attention_dim)
        attn_U = self.attention_U(features)  # (n_instances, attention_dim)
        
        attn_scores = self.attention_w(attn_V * attn_U)  # (n_instances, 1)
        attn_scores = torch.softmax(attn_scores, dim=0)
        
        # Apply attention pooling
        weighted_features = features * attn_scores  # (n_instances, hidden_dim)
        aggregated = torch.sum(weighted_features, dim=0, keepdim=True)  # (1, hidden_dim)
        
        # Classification
        logits = self.classifier(aggregated)
        
        if return_attention:
            return logits, attn_scores.squeeze()
        return logits

class TransMIL(nn.Module):
    """
    Transformer-based Multiple Instance Learning
    Shao et al., "TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification"
    """
    
    def __init__(self, feature_dim: int = 2048, hidden_dim: int = 512, 
                 n_heads: int = 8, n_classes: int = 4, dropout: float = 0.1):
        super(TransMIL, self).__init__()
        
        # Feature projection
        self.feature_proj = nn.Linear(feature_dim, hidden_dim)
        
        # Positional encoding
        self.position_embedding = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (n_instances, feature_dim)
        Returns:
            logits: Class predictions (1, n_classes)
        """
        # Project features
        x = self.feature_proj(x)  # (n_instances, hidden_dim)
        n_instances = x.shape[0]
        
        # Add positional encoding
        x = x.unsqueeze(0)  # (1, n_instances, hidden_dim)
        x = x + self.position_embedding[:, :n_instances, :]
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(1, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (1, n_instances+1, hidden_dim)
        
        # Transformer encoding
        encoded = self.transformer(x)
        
        # Extract CLS token representation
        cls_output = encoded[:, 0]  # (1, hidden_dim)
        
        # Classification
        logits = self.classifier(cls_output)
        
        return logits

class CLAM_SB(nn.Module):
    """
    CLAM Single Branch model
    Lu et al., "Data-efficient and weakly supervised computational pathology on whole-slide images"
    """
    
    def __init__(self, feature_dim: int = 2048, hidden_dim: int = 512,
                 dropout: float = 0.25, n_classes: int = 4):
        super(CLAM_SB, self).__init__()
        
        # Attention network
        self.attention_net = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Instance classifier
        self.instance_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # Binary: positive/negative
        )
        
        # Bag classifier
        self.bag_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
        
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (n_instances, feature_dim)
            return_features: Whether to return aggregated features
        Returns:
            bag_logits: Bag-level predictions (1, n_classes)
            instance_logits: Instance-level predictions (n_instances, 2)
            attention_scores: Attention weights (n_instances,)
        """
        # Attention scores
        attention_scores = self.attention_net(x)  # (n_instances, 1)
        attention_scores = torch.softmax(attention_scores, dim=0)
        
        # Instance predictions
        instance_logits = self.instance_classifier(x)  # (n_instances, 2)
        
        # Weighted aggregation
        weighted_features = x * attention_scores  # (n_instances, feature_dim)
        aggregated = torch.sum(weighted_features, dim=0, keepdim=True)  # (1, feature_dim)
        
        # Bag prediction
        bag_logits = self.bag_classifier(aggregated)  # (1, n_classes)
        
        if return_features:
            return bag_logits, instance_logits, attention_scores.squeeze(), aggregated
        return bag_logits, instance_logits, attention_scores.squeeze()

class DSMIL(nn.Module):
    """
    Dual-Stream Multiple Instance Learning
    Li et al., "Dual-stream Multiple Instance Learning Network for Whole Slide Image Classification"
    """
    
    def __init__(self, feature_dim: int = 2048, hidden_dim: int = 512,
                 n_classes: int = 4, dropout: float = 0.5):
        super(DSMIL, self).__init__()
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention networks for max and mean pooling
        self.attention_max = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.attention_mean = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (n_instances, feature_dim)
        Returns:
            logits: Class predictions (1, n_classes)
        """
        # Extract features
        features = self.feature_extractor(x)  # (n_instances, hidden_dim)
        
        # Max pooling stream
        attn_max = self.attention_max(features)  # (n_instances, 1)
        attn_max = torch.softmax(attn_max, dim=0)
        max_features = torch.sum(features * attn_max, dim=0)  # (hidden_dim,)
        
        # Mean pooling stream
        attn_mean = self.attention_mean(features)  # (n_instances, 1)
        attn_mean = torch.softmax(attn_mean, dim=0)
        mean_features = torch.sum(features * attn_mean, dim=0)  # (hidden_dim,)
        
        # Concatenate streams
        combined = torch.cat([max_features, mean_features])  # (hidden_dim * 2,)
        
        # Classification
        logits = self.classifier(combined.unsqueeze(0))  # (1, n_classes)
        
        return logits

class GatedAttentionMIL(nn.Module):
    """
    Gated Attention MIL
    Ilse et al., "Attention-based Deep Multiple Instance Learning" (Gated variant)
    """
    
    def __init__(self, feature_dim: int = 2048, hidden_dim: int = 512,
                 attention_dim: int = 256, n_classes: int = 4, dropout: float = 0.25):
        super(GatedAttentionMIL, self).__init__()
        
        # Feature transformation
        self.feature_extractor = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Gated attention mechanism
        self.attention_V = nn.Linear(hidden_dim, attention_dim)
        self.attention_U = nn.Linear(hidden_dim, attention_dim)
        self.attention_w = nn.Linear(attention_dim, 1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_classes)
        )
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Tensor of shape (n_instances, feature_dim)
            return_attention: Whether to return attention weights
        Returns:
            logits: Class predictions (1, n_classes)
            attention_weights: Attention weights (n_instances,) if return_attention=True
        """
        # Extract features
        features = self.feature_extractor(x)  # (n_instances, hidden_dim)
        
        # Gated attention
        attn_V = torch.tanh(self.attention_V(features))  # (n_instances, attention_dim)
        attn_U = torch.sigmoid(self.attention_U(features))  # (n_instances, attention_dim)
        
        # Gated combination
        gated = attn_V * attn_U  # (n_instances, attention_dim)
        attn_scores = self.attention_w(gated)  # (n_instances, 1)
        attn_scores = torch.softmax(attn_scores, dim=0)
        
        # Apply attention pooling
        weighted_features = features * attn_scores  # (n_instances, hidden_dim)
        aggregated = torch.sum(weighted_features, dim=0, keepdim=True)  # (1, hidden_dim)
        
        # Classification
        logits = self.classifier(aggregated)
        
        if return_attention:
            return logits, attn_scores.squeeze()
        return logits

class ClassicalMLClassifier:
    """Ensemble of classical ML models"""
    
    def __init__(self, config):
        self.config = config
        self.models = {
            'rf': RandomForestClassifier(
                n_estimators=200,
                max_depth=None,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            'xgb': xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                objective='multi:softprob',
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        }
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None) -> Dict:
        """Train all models"""
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name} model...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train, y_train, 
                cv=5, scoring='f1_macro'
            )
            
            results[name] = {
                'model': model,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Validation performance
            if X_val is not None and y_val is not None:
                y_pred = model.predict(X_val)
                val_score = roc_auc_score(
                    y_val, 
                    model.predict_proba(X_val), 
                    multi_class='ovr'
                )
                results[name]['val_score'] = val_score
            
            logger.info(f"{name} - CV Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        
        return results
    
    def predict(self, X: np.ndarray, model_name: str = 'rf') -> np.ndarray:
        """Make predictions using specified model"""
        return self.models[model_name].predict(X)
    
    def predict_proba(self, X: np.ndarray, model_name: str = 'rf') -> np.ndarray:
        """Get prediction probabilities"""
        return self.models[model_name].predict_proba(X) 