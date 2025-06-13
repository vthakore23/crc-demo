"""
Uncertainty-Aware Ensemble Predictor for CRC Molecular Subtyping
Combines multiple models with calibrated confidence and rejection capability
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import warnings
from scipy import stats


class TemperatureScaling(nn.Module):
    """Temperature scaling for probability calibration"""
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
        
    def forward(self, logits):
        """Apply temperature scaling to logits"""
        return logits / self.temperature


class UncertaintyAwareEnsemble:
    """
    Ensemble predictor with uncertainty quantification and rejection capability
    Combines multiple models for robust predictions
    """
    
    def __init__(self, rejection_threshold=0.85):
        self.rejection_threshold = rejection_threshold
        self.models = []
        self.model_weights = []
        self.calibrators = {}
        self.performance_history = {}
        
    def add_model(self, model, name, weight=1.0):
        """Add a model to the ensemble"""
        self.models.append((name, model))
        self.model_weights.append(weight)
        self.calibrators[name] = TemperatureScaling()
        self.performance_history[name] = []
        
    def predict_with_uncertainty(self, image, transform=None, return_all_predictions=False):
        """
        Make ensemble prediction with uncertainty quantification
        
        Returns:
            prediction: Final ensemble prediction or rejection
            uncertainty: Ensemble uncertainty measure
            details: Detailed prediction info if requested
        """
        
        # Collect predictions from all models
        all_predictions = []
        all_probabilities = []
        all_uncertainties = []
        model_details = {}
        
        for (name, model), weight in zip(self.models, self.model_weights):
            try:
                # Get model prediction
                pred_result = self._get_model_prediction(name, model, image, transform)
                
                if pred_result is not None:
                    all_predictions.append(pred_result['prediction'])
                    all_probabilities.append(pred_result['probabilities'])
                    all_uncertainties.append(pred_result['uncertainty'])
                    
                    model_details[name] = {
                        'prediction': pred_result['prediction'],
                        'probabilities': pred_result['probabilities'],
                        'uncertainty': pred_result['uncertainty'],
                        'confidence': pred_result['confidence'],
                        'weight': weight
                    }
                    
            except Exception as e:
                warnings.warn(f"Model {name} failed: {str(e)}")
                continue
                
        if not all_predictions:
            return "No valid predictions", 1.0, {}
            
        # Convert to numpy arrays
        all_probabilities = np.array(all_probabilities)
        all_uncertainties = np.array(all_uncertainties)
        weights = np.array([w for (_, _), w in zip(self.models, self.model_weights) 
                          if len(model_details) > 0])
        
        # Normalize weights based on model uncertainties
        uncertainty_weights = 1.0 / (all_uncertainties + 1e-6)
        combined_weights = weights * uncertainty_weights
        combined_weights = combined_weights / combined_weights.sum()
        
        # Compute weighted ensemble prediction
        ensemble_probs = np.average(all_probabilities, axis=0, weights=combined_weights)
        ensemble_pred_idx = np.argmax(ensemble_probs)
        
        # Compute ensemble uncertainty metrics
        uncertainty_metrics = self._compute_ensemble_uncertainty(
            all_probabilities, all_predictions, combined_weights
        )
        
        # Determine if we should reject
        should_reject, rejection_reason = self._check_rejection_criteria(
            uncertainty_metrics, ensemble_probs
        )
        
        # Format output
        subtype_names = ['SNF1 (Canonical)', 'SNF2 (Immune)', 'SNF3 (Stromal)']
        
        if should_reject:
            prediction = f"Uncertain - {rejection_reason}"
            confidence = uncertainty_metrics['ensemble_confidence']
        else:
            prediction = subtype_names[ensemble_pred_idx]
            confidence = float(ensemble_probs[ensemble_pred_idx])
            
        # Prepare detailed results
        results = {
            'prediction': prediction,
            'predicted_idx': ensemble_pred_idx,
            'confidence': confidence * 100,
            'probabilities': {
                'SNF1': float(ensemble_probs[0]),
                'SNF2': float(ensemble_probs[1]),
                'SNF3': float(ensemble_probs[2])
            },
            'uncertainty_metrics': uncertainty_metrics,
            'rejection': {
                'rejected': should_reject,
                'reason': rejection_reason
            },
            'ensemble_method': 'Uncertainty-weighted averaging',
            'num_models': len(all_predictions)
        }
        
        if return_all_predictions:
            results['model_predictions'] = model_details
            results['model_agreement'] = self._compute_model_agreement(all_predictions)
            
        return results
        
    def _get_model_prediction(self, name, model, image, transform):
        """Get prediction from a specific model with uncertainty"""
        
        # Handle different model types
        if hasattr(model, 'predict'):
            # Models with predict method (revolutionary, advanced)
            result = model.predict(image)
            
            # Extract uncertainty from confidence
            probs = np.array([
                result.get('probabilities', {}).get('SNF1', 0),
                result.get('probabilities', {}).get('SNF2', 0),
                result.get('probabilities', {}).get('SNF3', 0)
            ])
            
            # Compute entropy-based uncertainty
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = -np.log(1/3)
            uncertainty = entropy / max_entropy
            
            return {
                'prediction': result.get('predicted_idx', 0),
                'probabilities': probs,
                'confidence': result.get('confidence', 0) / 100,
                'uncertainty': uncertainty
            }
            
        elif hasattr(model, 'classify_molecular_subtype'):
            # Molecular subtype mapper
            result = model.classify_molecular_subtype(image, transform)
            
            probs = result.get('probabilities', np.array([0.33, 0.33, 0.34]))
            
            # Compute uncertainty
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            max_entropy = -np.log(1/3)
            uncertainty = entropy / max_entropy
            
            return {
                'prediction': result.get('subtype_idx', 0),
                'probabilities': probs,
                'confidence': result.get('confidence', 0) / 100,
                'uncertainty': uncertainty
            }
            
        else:
            # Direct neural network model
            with torch.no_grad():
                if isinstance(image, np.ndarray):
                    img_tensor = transform(image).unsqueeze(0)
                else:
                    img_tensor = transform(image).unsqueeze(0)
                    
                # Get logits
                logits = model(img_tensor)
                
                # Apply temperature scaling
                if name in self.calibrators:
                    logits = self.calibrators[name](logits)
                    
                # Get probabilities
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]
                
                # MC Dropout uncertainty if available
                if hasattr(model, 'training'):
                    mc_probs = self._mc_dropout_inference(model, img_tensor, n_samples=10)
                    uncertainty = self._compute_mc_uncertainty(mc_probs)
                else:
                    # Entropy-based uncertainty
                    entropy = -np.sum(probs * np.log(probs + 1e-10))
                    max_entropy = -np.log(1/3)
                    uncertainty = entropy / max_entropy
                    
                return {
                    'prediction': np.argmax(probs),
                    'probabilities': probs,
                    'confidence': float(np.max(probs)),
                    'uncertainty': uncertainty
                }
                
    def _mc_dropout_inference(self, model, input_tensor, n_samples=10):
        """Monte Carlo dropout inference for uncertainty estimation"""
        model.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                logits = model(input_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()
                predictions.append(probs)
                
        model.eval()  # Disable dropout
        return np.array(predictions)
        
    def _compute_mc_uncertainty(self, mc_predictions):
        """Compute uncertainty from MC dropout predictions"""
        # Mean prediction
        mean_pred = mc_predictions.mean(axis=0)
        
        # Predictive entropy
        entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-10), axis=1)
        
        # Mutual information (epistemic uncertainty)
        individual_entropy = -np.sum(mc_predictions * np.log(mc_predictions + 1e-10), axis=2).mean(axis=0)
        mutual_info = entropy - individual_entropy.mean()
        
        return float(mutual_info.mean())
        
    def _compute_ensemble_uncertainty(self, all_probs, all_preds, weights):
        """Compute comprehensive uncertainty metrics for ensemble"""
        
        # Weighted mean prediction
        mean_probs = np.average(all_probs, axis=0, weights=weights)
        
        # Prediction variance (epistemic uncertainty)
        pred_variance = np.average((all_probs - mean_probs)**2, axis=0, weights=weights)
        total_variance = pred_variance.sum()
        
        # Prediction entropy (aleatoric uncertainty)
        entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-10))
        max_entropy = -np.log(1/len(mean_probs))
        normalized_entropy = entropy / max_entropy
        
        # Model disagreement
        pred_indices = [np.argmax(p) for p in all_probs]
        disagreement = 1.0 - (np.array(pred_indices) == stats.mode(pred_indices)[0]).mean()
        
        # Jensen-Shannon divergence between predictions
        js_divergence = self._compute_js_divergence(all_probs)
        
        # Confidence metrics
        max_prob = np.max(mean_probs)
        margin = np.sort(mean_probs)[-1] - np.sort(mean_probs)[-2]
        
        return {
            'total_uncertainty': float(normalized_entropy + total_variance),
            'aleatoric_uncertainty': float(normalized_entropy),
            'epistemic_uncertainty': float(total_variance),
            'model_disagreement': float(disagreement),
            'js_divergence': float(js_divergence),
            'ensemble_confidence': float(max_prob),
            'prediction_margin': float(margin),
            'variance_per_class': pred_variance.tolist()
        }
        
    def _compute_js_divergence(self, all_probs):
        """Compute Jensen-Shannon divergence between model predictions"""
        # Average distribution
        avg_probs = all_probs.mean(axis=0)
        
        # KL divergence from each model to average
        kl_divs = []
        for probs in all_probs:
            kl = np.sum(probs * np.log(probs / (avg_probs + 1e-10) + 1e-10))
            kl_divs.append(kl)
            
        # JS divergence is average of KL divergences
        return np.mean(kl_divs)
        
    def _check_rejection_criteria(self, uncertainty_metrics, ensemble_probs):
        """Determine if prediction should be rejected"""
        
        reasons = []
        
        # High total uncertainty
        if uncertainty_metrics['total_uncertainty'] > self.rejection_threshold:
            reasons.append("high total uncertainty")
            
        # Low confidence
        if uncertainty_metrics['ensemble_confidence'] < 0.4:
            reasons.append("low confidence")
            
        # High model disagreement
        if uncertainty_metrics['model_disagreement'] > 0.6:
            reasons.append("high model disagreement")
            
        # Low prediction margin
        if uncertainty_metrics['prediction_margin'] < 0.1:
            reasons.append("ambiguous prediction")
            
        # High epistemic uncertainty
        if uncertainty_metrics['epistemic_uncertainty'] > 0.3:
            reasons.append("high model uncertainty")
            
        should_reject = len(reasons) > 0
        rejection_reason = "Recommend additional testing: " + ", ".join(reasons) if reasons else ""
        
        return should_reject, rejection_reason
        
    def _compute_model_agreement(self, predictions):
        """Compute pairwise model agreement"""
        n_models = len(predictions)
        if n_models < 2:
            return 1.0
            
        agreement_count = 0
        total_pairs = 0
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                if predictions[i] == predictions[j]:
                    agreement_count += 1
                total_pairs += 1
                
        return agreement_count / total_pairs if total_pairs > 0 else 0
        
    def calibrate_on_validation_data(self, val_data, true_labels):
        """Calibrate temperature scaling on validation data"""
        # This would be implemented when validation data is available
        pass
        
    def update_model_weights(self, performance_metrics):
        """Update model weights based on recent performance"""
        # Dynamically adjust weights based on model performance
        for i, (name, _) in enumerate(self.models):
            if name in performance_metrics:
                # Update performance history
                self.performance_history[name].append(performance_metrics[name])
                
                # Compute new weight based on recent performance
                if len(self.performance_history[name]) > 0:
                    recent_performance = np.mean(self.performance_history[name][-10:])
                    self.model_weights[i] = recent_performance
                    
        # Normalize weights
        total_weight = sum(self.model_weights)
        if total_weight > 0:
            self.model_weights = [w / total_weight for w in self.model_weights] 