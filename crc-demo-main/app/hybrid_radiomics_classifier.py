#!/usr/bin/env python3
"""
Hybrid PyRadiomics-Deep Learning Classifier for Molecular Subtypes
Combines handcrafted radiomic features with deep learning features for improved accuracy and interpretability.

Integrates:
1. PyRadiomics feature extraction from tumor regions
2. Deep learning features from ResNet50
3. Feature selection techniques (LASSO, Boruta, SHAP)
4. Ensemble methods for robust prediction
5. Clinical interpretability through feature importance analysis
"""

import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.feature_selection import SelectKBest, f_classif, RFECV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb
from PIL import Image
import cv2
import json
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# PyRadiomics imports
try:
    import radiomics
    from radiomics import featureextractor
    import SimpleITK as sitk
    RADIOMICS_AVAILABLE = True
except ImportError:
    RADIOMICS_AVAILABLE = False
    print("Warning: PyRadiomics not available. Install with: pip install pyradiomics")

# Feature selection imports
try:
    from boruta import BorutaPy
    BORUTA_AVAILABLE = True
except ImportError:
    BORUTA_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from app.molecular_subtype_mapper import MolecularSubtypeMapper


class PyRadiomicsExtractor:
    """Extract handcrafted radiomic features from histopathology images"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        
        if not RADIOMICS_AVAILABLE:
            raise ImportError("PyRadiomics is required. Install with: pip install pyradiomics")
        
        # Configure PyRadiomics feature extractor
        self.extractor = featureextractor.RadiomicsFeatureExtractor()
        
        # Configure feature classes - optimized for histopathology
        self.extractor.enableFeatureClassByName('firstorder')       # Intensity statistics
        self.extractor.enableFeatureClassByName('glcm')            # Gray Level Co-occurrence Matrix
        self.extractor.enableFeatureClassByName('glrlm')           # Gray Level Run Length Matrix
        self.extractor.enableFeatureClassByName('glszm')           # Gray Level Size Zone Matrix
        self.extractor.enableFeatureClassByName('gldm')            # Gray Level Dependence Matrix
        self.extractor.enableFeatureClassByName('ngtdm')           # Neighbouring Gray Tone Difference Matrix
        self.extractor.enableFeatureClassByName('shape')           # Shape-based features
        
        # Configure filters for multi-scale analysis
        self.extractor.enableImageTypeByName('Original')
        self.extractor.enableImageTypeByName('LoG')  # Laplacian of Gaussian
        self.extractor.enableImageTypeByName('Wavelet')                 # Wavelet decomposition
        
        # Histopathology-specific settings
        settings = {
            'binWidth': 25,                    # Intensity discretization
            'sigma': [3.0, 4.0, 5.0],         # LoG kernel sizes
            'start_level': 0,                  # Wavelet decomposition start level
            'level': 1,                        # Wavelet decomposition level
            'wavelet': 'coif1',                # Wavelet type
            'setting': {
                'Original': {},
                'LoG': {},
                'Wavelet': {}
            }
        }
        
        for key, value in settings.items():
            if key != 'setting':
                self.extractor.settings[key] = value
        
        # Configure logging
        radiomics.setVerbosity(60 if verbose else 40)  # ERROR or WARNING level
        
        # Feature name mapping for interpretability
        self.feature_categories = {
            'tumor_morphology': ['shape_', 'original_shape_'],
            'tumor_intensity': ['original_firstorder_', 'log_firstorder_'],
            'tumor_texture': ['original_glcm_', 'original_glrlm_', 'original_glszm_'],
            'tumor_heterogeneity': ['original_gldm_', 'original_ngtdm_'],
            'multiscale_patterns': ['wavelet_', 'log_glcm_', 'log_glrlm_']
        }
    
    def extract_features_from_region(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Extract radiomic features from a masked region"""
        if mask.sum() == 0:
            return {}
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image.copy()
        
        # Convert to SimpleITK images
        sitk_image = sitk.GetImageFromArray(gray_image.astype(np.float32))
        sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
        
        try:
            # Extract features
            features = self.extractor.execute(sitk_image, sitk_mask)
            
            # Clean feature names and convert to float
            clean_features = {}
            for key, value in features.items():
                if key.startswith('original_') or key.startswith('wavelet_') or key.startswith('log_'):
                    try:
                        clean_features[key] = float(value)
                    except (ValueError, TypeError):
                        continue
                        
            return clean_features
            
        except Exception as e:
            if self.verbose:
                print(f"Warning: Failed to extract radiomic features: {e}")
            return {}
    
    def extract_multiscale_features(self, image: np.ndarray, tumor_mask: np.ndarray, 
                                   stroma_mask: np.ndarray, lymphocyte_mask: np.ndarray) -> Dict[str, Any]:
        """Extract radiomic features from different tissue regions"""
        features = {
            'tumor_features': {},
            'stroma_features': {},
            'lymphocyte_features': {},
            'interface_features': {},
            'heterogeneity_features': {}
        }
        
        # Extract features from each tissue type
        if tumor_mask.sum() > 100:  # Minimum region size
            features['tumor_features'] = self.extract_features_from_region(image, tumor_mask)
        
        if stroma_mask.sum() > 100:
            features['stroma_features'] = self.extract_features_from_region(image, stroma_mask)
        
        if lymphocyte_mask.sum() > 50:
            features['lymphocyte_features'] = self.extract_features_from_region(image, lymphocyte_mask)
        
        # Extract interface features (tumor-stroma boundary)
        interface_mask = self._create_interface_mask(tumor_mask, stroma_mask)
        if interface_mask.sum() > 50:
            features['interface_features'] = self.extract_features_from_region(image, interface_mask)
        
        # Compute heterogeneity features across regions
        features['heterogeneity_features'] = self._compute_heterogeneity_features(
            features['tumor_features'], features['stroma_features'], features['lymphocyte_features']
        )
        
        return features
    
    def _create_interface_mask(self, tumor_mask: np.ndarray, stroma_mask: np.ndarray, 
                              boundary_width: int = 10) -> np.ndarray:
        """Create mask for tumor-stroma interface region"""
        # Dilate tumor mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (boundary_width, boundary_width))
        dilated_tumor = cv2.dilate(tumor_mask.astype(np.uint8), kernel, iterations=1)
        
        # Interface is the overlap between dilated tumor and stroma
        interface_mask = (dilated_tumor > 0) & (stroma_mask > 0)
        
        return interface_mask.astype(bool)
    
    def _compute_heterogeneity_features(self, tumor_features: Dict, stroma_features: Dict, 
                                       lymphocyte_features: Dict) -> Dict[str, float]:
        """Compute heterogeneity features across tissue regions"""
        heterogeneity = {}
        
        # Find common features across regions
        all_regions = [tumor_features, stroma_features, lymphocyte_features]
        all_regions = [r for r in all_regions if r]  # Remove empty dictionaries
        
        if len(all_regions) < 2:
            return heterogeneity
        
        # Get common feature names
        common_features = set(all_regions[0].keys())
        for region in all_regions[1:]:
            common_features = common_features.intersection(set(region.keys()))
        
        # Compute heterogeneity metrics
        for feature_name in common_features:
            values = [region[feature_name] for region in all_regions]
            
            if len(values) > 1:
                heterogeneity[f'{feature_name}_heterogeneity_std'] = np.std(values)
                heterogeneity[f'{feature_name}_heterogeneity_range'] = np.max(values) - np.min(values)
                heterogeneity[f'{feature_name}_heterogeneity_cv'] = np.std(values) / (np.mean(values) + 1e-6)
        
        return heterogeneity


class HybridFeatureSelector:
    """Advanced feature selection combining multiple methods"""
    
    def __init__(self, n_features_to_select: int = 50, random_state: int = 42):
        self.n_features_to_select = n_features_to_select
        self.random_state = random_state
        self.selected_features = []
        self.feature_importance_scores = {}
        
    def select_features(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Select features using multiple methods and ensemble approach"""
        
        # Method 1: LASSO feature selection
        lasso_features = self._lasso_selection(X, y, feature_names)
        
        # Method 2: Random Forest importance
        rf_features = self._random_forest_selection(X, y, feature_names)
        
        # Method 3: Statistical tests (f_classif)
        statistical_features = self._statistical_selection(X, y, feature_names)
        
        # Method 4: Boruta (if available)
        boruta_features = []
        if BORUTA_AVAILABLE:
            boruta_features = self._boruta_selection(X, y, feature_names)
        
        # Ensemble selection: features selected by multiple methods
        all_selections = [lasso_features, rf_features, statistical_features]
        if boruta_features:
            all_selections.append(boruta_features)
        
        # Count votes for each feature
        feature_votes = {}
        for features in all_selections:
            for feature in features:
                feature_votes[feature] = feature_votes.get(feature, 0) + 1
        
        # Select features with at least 2 votes, up to n_features_to_select
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        selected = [f for f, votes in sorted_features if votes >= 2][:self.n_features_to_select]
        
        # If not enough features, add highest voted ones
        if len(selected) < self.n_features_to_select:
            remaining = self.n_features_to_select - len(selected)
            additional = [f for f, _ in sorted_features if f not in selected][:remaining]
            selected.extend(additional)
        
        self.selected_features = selected
        
        # Get indices
        selected_indices = [feature_names.index(f) for f in selected if f in feature_names]
        
        return X[:, selected_indices], selected
    
    def _lasso_selection(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[str]:
        """LASSO feature selection"""
        try:
            lasso = LassoCV(cv=3, random_state=self.random_state, max_iter=1000)
            lasso.fit(X, y)
            
            # Get features with non-zero coefficients
            selected_mask = lasso.coef_ != 0
            selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
            
            # Store importance scores
            for i, feature in enumerate(feature_names):
                self.feature_importance_scores[f'{feature}_lasso'] = abs(lasso.coef_[i])
            
            return selected_features
        except Exception:
            return []
    
    def _random_forest_selection(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[str]:
        """Random Forest feature importance selection"""
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            rf.fit(X, y)
            
            # Get top features by importance
            importances = rf.feature_importances_
            top_indices = np.argsort(importances)[::-1][:self.n_features_to_select]
            selected_features = [feature_names[i] for i in top_indices]
            
            # Store importance scores
            for i, feature in enumerate(feature_names):
                self.feature_importance_scores[f'{feature}_rf'] = importances[i]
            
            return selected_features
        except Exception:
            return []
    
    def _statistical_selection(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[str]:
        """Statistical feature selection using f_classif"""
        try:
            selector = SelectKBest(f_classif, k=min(self.n_features_to_select, X.shape[1]))
            selector.fit(X, y)
            
            selected_mask = selector.get_support()
            selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
            
            # Store importance scores
            scores = selector.scores_
            for i, feature in enumerate(feature_names):
                self.feature_importance_scores[f'{feature}_stat'] = scores[i] if not np.isnan(scores[i]) else 0
            
            return selected_features
        except Exception:
            return []
    
    def _boruta_selection(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> List[str]:
        """Boruta feature selection"""
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=0, random_state=self.random_state)
            boruta_selector.fit(X, y)
            
            selected_mask = boruta_selector.support_
            selected_features = [feature_names[i] for i in range(len(feature_names)) if selected_mask[i]]
            
            # Store importance scores
            for i, feature in enumerate(feature_names):
                self.feature_importance_scores[f'{feature}_boruta'] = 1.0 if selected_mask[i] else 0.0
            
            return selected_features
        except Exception:
            return []


class HybridRadiomicsClassifier:
    """
    Hybrid classifier combining PyRadiomics features with deep learning features
    for molecular subtype prediction with enhanced interpretability
    """
    
    def __init__(self, tissue_model, model_save_path: str = "models/hybrid_radiomics_model.pkl"):
        self.tissue_model = tissue_model
        self.model_save_path = Path(model_save_path)
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        if RADIOMICS_AVAILABLE:
            self.radiomics_extractor = PyRadiomicsExtractor(verbose=False)
        else:
            self.radiomics_extractor = None
            print("Warning: PyRadiomics not available. Using deep features only.")
        
        self.molecular_mapper = MolecularSubtypeMapper(tissue_model)
        self.feature_selector = HybridFeatureSelector(n_features_to_select=50)
        self.scaler = StandardScaler()
        
        # Model ensemble
        self.models = {
            'random_forest': RandomForestClassifier(n_estimators=200, random_state=42),
            'xgboost': xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'),
            'logistic': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        self.ensemble_model = None
        self.is_trained = False
        self.feature_names = []
        self.selected_features = []
        
        # SHAP explainer for interpretability
        self.shap_explainer = None
        
    def extract_hybrid_features(self, image: np.ndarray, transform) -> Dict[str, Any]:
        """Extract both radiomic and deep learning features"""
        features = {
            'deep_features': {},
            'radiomic_features': {},
            'spatial_features': {},
            'combined_features': {}
        }
        
        # Extract deep learning features using existing pipeline
        # Convert PIL image to numpy array if needed
        if hasattr(image, 'mode'):  # PIL Image
            image = np.array(image)
        tissue_probs, spatial_features = self.molecular_mapper.extract_deep_features(image, transform)
        
        # Extract tissue masks for radiomic analysis
        masks, tissue_maps = self.molecular_mapper.extract_tissue_masks(image, transform)
        
        # Deep learning features
        features['deep_features'] = {
            'tissue_probs': tissue_probs,
            'spatial_features': spatial_features.flatten() if len(spatial_features.shape) > 1 else spatial_features
        }
        
        # Extract PyRadiomics features if available
        if self.radiomics_extractor is not None:
            radiomic_features = self.radiomics_extractor.extract_multiscale_features(
                image, 
                masks['tumor'], 
                masks['stroma'], 
                masks['lymphocytes']
            )
            features['radiomic_features'] = radiomic_features
        
        # Extract spatial pattern features
        spatial_patterns = self.molecular_mapper.analyze_spatial_patterns(image, transform)
        features['spatial_features'] = self._flatten_spatial_features(spatial_patterns)
        
        # Combine all features into a single vector
        features['combined_features'] = self._combine_all_features(features)
        
        return features
    
    def _flatten_spatial_features(self, spatial_patterns: Dict) -> Dict[str, float]:
        """Flatten nested spatial pattern features"""
        flattened = {}
        
        def flatten_dict(d, prefix=''):
            for key, value in d.items():
                if isinstance(value, dict):
                    flatten_dict(value, f"{prefix}{key}_")
                elif isinstance(value, (int, float, bool)):
                    flattened[f"{prefix}{key}"] = float(value)
        
        flatten_dict(spatial_patterns, 'spatial_')
        return flattened
    
    def _combine_all_features(self, features: Dict) -> Dict[str, float]:
        """Combine all feature types into a single feature vector"""
        combined = {}
        
        # Add deep learning features
        if 'deep_features' in features:
            # Tissue probabilities
            tissue_probs = features['deep_features'].get('tissue_probs', [])
            for i, prob in enumerate(tissue_probs):
                combined[f'deep_tissue_class_{i}'] = float(prob)
            
            # Spatial features from deep network
            spatial_feats = features['deep_features'].get('spatial_features', [])
            if len(spatial_feats.shape) == 0:
                spatial_feats = [spatial_feats]
            for i, feat in enumerate(spatial_feats.flatten()):
                combined[f'deep_spatial_{i}'] = float(feat)
        
        # Add radiomic features
        if 'radiomic_features' in features and features['radiomic_features']:
            for region_type, region_features in features['radiomic_features'].items():
                if isinstance(region_features, dict):
                    for feat_name, feat_value in region_features.items():
                        combined[f'radiomic_{region_type}_{feat_name}'] = float(feat_value)
        
        # Add spatial pattern features
        if 'spatial_features' in features:
            combined.update(features['spatial_features'])
        
        return combined
    
    def prepare_training_data(self, images: List[np.ndarray], labels: List[int], transform) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data with hybrid features"""
        print("Extracting hybrid features from training data...")
        
        all_features = []
        valid_labels = []
        
        for i, (image, label) in enumerate(zip(images, labels)):
            try:
                features = self.extract_hybrid_features(image, transform)
                combined_features = features['combined_features']
                
                if combined_features:  # Only add if features were extracted successfully
                    all_features.append(combined_features)
                    valid_labels.append(label)
                    
                if (i + 1) % 10 == 0:
                    print(f"Processed {i + 1}/{len(images)} images")
                    
            except Exception as e:
                print(f"Warning: Failed to extract features from image {i}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No features could be extracted from the training data")
        
        # Convert to DataFrame for easier handling
        feature_df = pd.DataFrame(all_features)
        
        # Handle missing values
        feature_df = feature_df.fillna(feature_df.mean())
        
        # Get feature names and convert to numpy
        self.feature_names = list(feature_df.columns)
        X = feature_df.values
        y = np.array(valid_labels)
        
        print(f"Extracted {X.shape[1]} features from {X.shape[0]} samples")
        
        return X, y, self.feature_names
    
    def train(self, images: List[np.ndarray], labels: List[int], transform, validation_split: float = 0.2):
        """Train the hybrid model with cross-validation"""
        print("Training Hybrid PyRadiomics-Deep Learning Classifier...")
        
        # Prepare training data
        X, y, feature_names = self.prepare_training_data(images, labels, transform)
        
        # Feature selection
        print("Performing feature selection...")
        X_selected, selected_features = self.feature_selector.select_features(X, y, feature_names)
        self.selected_features = selected_features
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # Train individual models
        print("Training ensemble models...")
        trained_models = []
        
        for name, model in self.models.items():
            try:
                print(f"Training {name}...")
                
                # Cross-validation
                cv_scores = cross_val_score(model, X_scaled, y, cv=3, scoring='accuracy')
                print(f"{name} CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                
                # Train on full dataset
                model.fit(X_scaled, y)
                trained_models.append((name, model))
                
            except Exception as e:
                print(f"Warning: Failed to train {name}: {e}")
                continue
        
        # Create ensemble
        if len(trained_models) >= 2:
            self.ensemble_model = VotingClassifier(
                estimators=trained_models,
                voting='soft'  # Use probability averaging
            )
            self.ensemble_model.fit(X_scaled, y)
        elif len(trained_models) == 1:
            self.ensemble_model = trained_models[0][1]
        else:
            raise ValueError("No models could be trained successfully")
        
        # Initialize SHAP explainer for interpretability
        if SHAP_AVAILABLE:
            try:
                # Use a smaller background dataset for SHAP
                background_size = min(100, X_scaled.shape[0])
                background = X_scaled[:background_size]
                self.shap_explainer = shap.KernelExplainer(self.ensemble_model.predict_proba, background)
            except Exception as e:
                print(f"Warning: Could not initialize SHAP explainer: {e}")
        
        self.is_trained = True
        
        # Save model
        self._save_model()
        
        print("Training completed successfully!")
        
        return {
            'n_features_selected': len(self.selected_features),
            'selected_features': self.selected_features[:10],  # Top 10 for display
            'feature_importance': self.get_feature_importance()
        }
    
    def predict(self, image: np.ndarray, transform, return_probabilities: bool = True, 
                explain: bool = False) -> Dict[str, Any]:
        """Predict molecular subtype with hybrid approach"""
        if not self.is_trained:
            # Try to load saved model
            if not self._load_model():
                raise ValueError("Model not trained. Please train the model first or provide training data.")
        
        # Extract features
        features = self.extract_hybrid_features(image, transform)
        combined_features = features['combined_features']
        
        # Convert to DataFrame and align with training features
        feature_df = pd.DataFrame([combined_features])
        
        # Add missing features with zero values
        for feature in self.selected_features:
            if feature not in feature_df.columns:
                feature_df[feature] = 0.0
        
        # Select only the features used during training
        feature_df = feature_df[self.selected_features]
        
        # Scale features
        X_scaled = self.scaler.transform(feature_df.values)
        
        # Predict
        prediction = self.ensemble_model.predict(X_scaled)[0]
        probabilities = self.ensemble_model.predict_proba(X_scaled)[0]
        
        # Get subtype names
        subtype_names = ['SNF1 (Canonical)', 'SNF2 (Immune)', 'SNF3 (Stromal)']
        
        results = {
            'subtype': subtype_names[prediction],
            'subtype_idx': int(prediction),
            'confidence': float(probabilities[prediction]) * 100,
            'probabilities': probabilities.tolist(),
            'model_type': 'Hybrid PyRadiomics-Deep Learning',
            'feature_summary': {
                'total_features_extracted': len(combined_features),
                'features_used_for_prediction': len(self.selected_features),
                'deep_features': len([f for f in combined_features.keys() if f.startswith('deep_')]),
                'radiomic_features': len([f for f in combined_features.keys() if f.startswith('radiomic_')]),
                'spatial_features': len([f for f in combined_features.keys() if f.startswith('spatial_')])
            }
        }
        
        # Add probabilities by name
        results['probabilities_by_subtype'] = {
            name: float(prob) for name, prob in zip(subtype_names, probabilities)
        }
        
        # Add feature importance explanation if requested
        if explain and self.shap_explainer is not None:
            try:
                shap_values = self.shap_explainer.shap_values(X_scaled)
                if isinstance(shap_values, list):
                    shap_values = shap_values[prediction]  # Get values for predicted class
                
                # Get top contributing features
                feature_contributions = list(zip(self.selected_features, shap_values[0]))
                feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
                
                results['explanation'] = {
                    'top_contributing_features': feature_contributions[:10],
                    'prediction_drivers': self._interpret_feature_contributions(feature_contributions[:5])
                }
            except Exception as e:
                print(f"Warning: Could not generate SHAP explanation: {e}")
        
        return results
    
    def _interpret_feature_contributions(self, contributions: List[Tuple[str, float]]) -> List[str]:
        """Interpret feature contributions in clinical terms"""
        interpretations = []
        
        for feature_name, contribution in contributions:
            if abs(contribution) < 0.01:  # Skip very small contributions
                continue
            
            direction = "increased" if contribution > 0 else "decreased"
            
            # Map feature names to clinical interpretations
            if 'radiomic_tumor_' in feature_name:
                if 'shape_' in feature_name:
                    interpretations.append(f"{direction.capitalize()} tumor morphological complexity")
                elif 'firstorder_' in feature_name:
                    interpretations.append(f"{direction.capitalize()} tumor intensity heterogeneity")
                elif 'glcm_' in feature_name:
                    interpretations.append(f"{direction.capitalize()} tumor texture homogeneity")
            elif 'radiomic_stroma_' in feature_name:
                interpretations.append(f"{direction.capitalize()} stromal tissue characteristics")
            elif 'radiomic_lymphocyte_' in feature_name:
                interpretations.append(f"{direction.capitalize()} immune infiltration patterns")
            elif 'deep_tissue_' in feature_name:
                interpretations.append(f"{direction.capitalize()} tissue composition from deep learning")
            elif 'spatial_' in feature_name:
                interpretations.append(f"{direction.capitalize()} spatial organization patterns")
        
        return interpretations[:3]  # Return top 3 interpretations
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the ensemble model"""
        if not self.is_trained:
            return {}
        
        importance_dict = {}
        
        # Get importance from Random Forest if available
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_') and name == 'random_forest':
                for i, importance in enumerate(model.feature_importances_):
                    feature_name = self.selected_features[i]
                    importance_dict[feature_name] = float(importance)
        
        return importance_dict
    
    def _save_model(self):
        """Save the trained model"""
        import pickle
        
        model_data = {
            'ensemble_model': self.ensemble_model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        with open(self.model_save_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {self.model_save_path}")
    
    def _load_model(self) -> bool:
        """Load a saved model"""
        import pickle
        
        if not self.model_save_path.exists():
            return False
        
        try:
            with open(self.model_save_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.ensemble_model = model_data['ensemble_model']
            self.scaler = model_data['scaler']
            self.feature_selector = model_data['feature_selector']
            self.selected_features = model_data['selected_features']
            self.feature_names = model_data['feature_names']
            self.is_trained = model_data['is_trained']
            
            print(f"Model loaded from {self.model_save_path}")
            return True
            
        except Exception as e:
            print(f"Failed to load model: {e}")
            return False


# Utility functions for clinical integration
def create_clinical_report(predictions: Dict[str, Any], patient_id: str = None) -> str:
    """Create a clinical report from hybrid predictions"""
    
    report = f"""
    MOLECULAR SUBTYPE PREDICTION REPORT
    {'='*50}
    Patient ID: {patient_id or 'N/A'}
    Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    PREDICTION RESULTS:
    Molecular Subtype: {predictions['subtype']}
    Confidence: {predictions['confidence']:.1f}%
    Model: {predictions['model_type']}
    
    SUBTYPE PROBABILITIES:
    SNF1 (Canonical): {predictions['probabilities_by_subtype']['SNF1 (Canonical)']:.3f}
    SNF2 (Immune): {predictions['probabilities_by_subtype']['SNF2 (Immune)']:.3f}  
    SNF3 (Stromal): {predictions['probabilities_by_subtype']['SNF3 (Stromal)']:.3f}
    
    FEATURE ANALYSIS:
    Total Features Extracted: {predictions['feature_summary']['total_features_extracted']}
    Features Used: {predictions['feature_summary']['features_used_for_prediction']}
    - Deep Learning Features: {predictions['feature_summary']['deep_features']}
    - Radiomic Features: {predictions['feature_summary']['radiomic_features']}
    - Spatial Pattern Features: {predictions['feature_summary']['spatial_features']}
    """
    
    if 'explanation' in predictions:
        report += "\n    KEY PREDICTION DRIVERS:\n"
        for driver in predictions['explanation']['prediction_drivers']:
            report += f"    - {driver}\n"
    
    # Add clinical recommendations based on subtype
    subtype_idx = predictions['subtype_idx']
    if subtype_idx == 0:  # SNF1
        report += """
    CLINICAL IMPLICATIONS (SNF1 - Canonical):
    - Intermediate prognosis (37% 10-year survival)
    - Standard chemotherapy response
    - Monitor for systemic progression
        """
    elif subtype_idx == 1:  # SNF2
        report += """
    CLINICAL IMPLICATIONS (SNF2 - Immune):
    - Favorable prognosis (64% 10-year survival)
    - Consider surgical resection for oligometastatic disease
    - Potential immunotherapy benefit
        """
    else:  # SNF3
        report += """
    CLINICAL IMPLICATIONS (SNF3 - Stromal):
    - Poor prognosis (20% 10-year survival)
    - Likely cetuximab resistance
    - Aggressive systemic therapy recommended
        """
    
    report += f"""
    
    QUALITY METRICS:
    Model combines handcrafted radiomic features with deep learning
    for enhanced accuracy and clinical interpretability.
    
    NOTE: This prediction is for research purposes and should be 
    validated with additional clinical data and expert pathologist review.
    {'='*50}
    """
    
    return report


# Example usage
if __name__ == "__main__":
    print("Hybrid PyRadiomics-Deep Learning Classifier for Molecular Subtypes")
    print("This module combines handcrafted radiomic features with deep learning")
    print("for improved accuracy and clinical interpretability.")
    
    if not RADIOMICS_AVAILABLE:
        print("\nWarning: PyRadiomics not installed. Please install with:")
        print("pip install pyradiomics")
    
    print("\nExample usage:")
    print("from app.hybrid_radiomics_classifier import HybridRadiomicsClassifier")
    print("classifier = HybridRadiomicsClassifier(tissue_model)")
    print("# Train with your data: classifier.train(images, labels, transform)")
    print("# Predict: result = classifier.predict(image, transform, explain=True)") 