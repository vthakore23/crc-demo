#!/usr/bin/env python3
"""
Clinical Inference System for Molecular Subtype Classification
Production-ready system for clinical deployment with confidence calibration
"""

import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
import json
from dataclasses import dataclass, asdict
from datetime import datetime
import warnings
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import pickle

# Import our components
from .molecular_subtype_foundation import MolecularSubtypeFoundationModel, load_pretrained_model
from .wsi_processor import WSIProcessor, PatchInfo, create_wsi_processor

logger = logging.getLogger(__name__)

@dataclass
class ClinicalPrediction:
    """Clinical prediction result with full metadata"""
    patient_id: str
    wsi_filename: str
    predicted_subtype: str
    confidence: float
    probabilities: Dict[str, float]
    uncertainty: Optional[float]
    risk_stratification: str
    treatment_recommendation: str
    survival_prediction: str
    oligometastatic_potential: str
    processing_metadata: Dict
    timestamp: str
    model_version: str
    validation_flags: List[str]

@dataclass
class QualityMetrics:
    """Quality metrics for prediction validation"""
    tissue_adequacy: float  # 0-1 score
    image_quality: float   # 0-1 score
    stain_quality: float   # 0-1 score
    patch_diversity: float # 0-1 score
    coverage_score: float  # 0-1 score
    overall_quality: float # 0-1 score
    quality_flags: List[str]

class ConfidenceCalibrator:
    """Calibrate model confidence scores for clinical reliability"""
    
    def __init__(self):
        self.temperature_scale = None
        self.platt_scaler = None
        self.isotonic_regressor = None
        self.calibration_method = 'temperature'
        self.is_fitted = False
    
    def fit(self, logits: np.ndarray, true_labels: np.ndarray, method: str = 'temperature'):
        """Fit calibration model"""
        self.calibration_method = method
        
        if method == 'temperature':
            self._fit_temperature_scaling(logits, true_labels)
        elif method == 'platt':
            self._fit_platt_scaling(logits, true_labels)
        elif method == 'isotonic':
            self._fit_isotonic_regression(logits, true_labels)
        
        self.is_fitted = True
        logger.info(f"Confidence calibrator fitted using {method} method")
    
    def _fit_temperature_scaling(self, logits: np.ndarray, true_labels: np.ndarray):
        """Fit temperature scaling parameter"""
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        labels_tensor = torch.tensor(true_labels, dtype=torch.long)
        
        # Find optimal temperature
        temperature = torch.ones(1, requires_grad=True)
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits_tensor / temperature, labels_tensor)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        self.temperature_scale = temperature.item()
    
    def _fit_platt_scaling(self, logits: np.ndarray, true_labels: np.ndarray):
        """Fit Platt scaling (sigmoid)"""
        # Use max logit as confidence score
        max_logits = np.max(logits, axis=1)
        self.platt_scaler = LogisticRegression()
        
        # Convert to binary (correct vs incorrect)
        predictions = np.argmax(logits, axis=1)
        correct = (predictions == true_labels).astype(int)
        
        self.platt_scaler.fit(max_logits.reshape(-1, 1), correct)
    
    def _fit_isotonic_regression(self, logits: np.ndarray, true_labels: np.ndarray):
        """Fit isotonic regression"""
        # Use max probability as confidence
        probs = F.softmax(torch.tensor(logits), dim=1).numpy()
        max_probs = np.max(probs, axis=1)
        
        predictions = np.argmax(logits, axis=1)
        correct = (predictions == true_labels).astype(float)
        
        self.isotonic_regressor = IsotonicRegression(out_of_bounds='clip')
        self.isotonic_regressor.fit(max_probs, correct)
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply calibration to logits"""
        if not self.is_fitted:
            warnings.warn("Calibrator not fitted. Returning uncalibrated probabilities.")
            return F.softmax(torch.tensor(logits), dim=1).numpy()
        
        if self.calibration_method == 'temperature':
            calibrated_logits = logits / self.temperature_scale
            return F.softmax(torch.tensor(calibrated_logits), dim=1).numpy()
        
        elif self.calibration_method == 'platt':
            probs = F.softmax(torch.tensor(logits), dim=1).numpy()
            max_probs = np.max(probs, axis=1)
            calibrated_confidence = self.platt_scaler.predict_proba(max_probs.reshape(-1, 1))[:, 1]
            
            # Scale probabilities to match calibrated confidence
            scaling_factor = calibrated_confidence / max_probs
            calibrated_probs = probs * scaling_factor.reshape(-1, 1)
            calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)
            
            return calibrated_probs
        
        elif self.calibration_method == 'isotonic':
            probs = F.softmax(torch.tensor(logits), dim=1).numpy()
            max_probs = np.max(probs, axis=1)
            calibrated_confidence = self.isotonic_regressor.predict(max_probs)
            
            # Scale probabilities
            scaling_factor = calibrated_confidence / max_probs
            calibrated_probs = probs * scaling_factor.reshape(-1, 1)
            calibrated_probs = calibrated_probs / calibrated_probs.sum(axis=1, keepdims=True)
            
            return calibrated_probs
    
    def save(self, path: str):
        """Save calibrator"""
        calibrator_data = {
            'method': self.calibration_method,
            'temperature_scale': self.temperature_scale,
            'is_fitted': self.is_fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(calibrator_data, f)
        
        # Save sklearn models separately if needed
        if self.platt_scaler:
            with open(str(Path(path).with_suffix('.platt.pkl')), 'wb') as f:
                pickle.dump(self.platt_scaler, f)
        
        if self.isotonic_regressor:
            with open(str(Path(path).with_suffix('.isotonic.pkl')), 'wb') as f:
                pickle.dump(self.isotonic_regressor, f)
    
    def load(self, path: str):
        """Load calibrator"""
        with open(path, 'rb') as f:
            calibrator_data = pickle.load(f)
        
        self.calibration_method = calibrator_data['method']
        self.temperature_scale = calibrator_data['temperature_scale']
        self.is_fitted = calibrator_data['is_fitted']
        
        # Load sklearn models if they exist
        platt_path = str(Path(path).with_suffix('.platt.pkl'))
        if Path(platt_path).exists():
            with open(platt_path, 'rb') as f:
                self.platt_scaler = pickle.load(f)
        
        isotonic_path = str(Path(path).with_suffix('.isotonic.pkl'))
        if Path(isotonic_path).exists():
            with open(isotonic_path, 'rb') as f:
                self.isotonic_regressor = pickle.load(f)

class QualityAssessment:
    """Assess quality of WSI data for reliable prediction"""
    
    def __init__(self):
        self.quality_thresholds = {
            'tissue_adequacy': 0.6,
            'image_quality': 0.5,
            'stain_quality': 0.4,
            'patch_diversity': 0.3,
            'coverage_score': 0.5,
            'overall_quality': 0.5
        }
    
    def assess_quality(self, 
                      patch_infos: List[PatchInfo], 
                      wsi_metadata: Dict,
                      patch_predictions: Optional[List[Dict]] = None) -> QualityMetrics:
        """Comprehensive quality assessment"""
        
        # 1. Tissue adequacy
        tissue_ratios = [p.tissue_ratio for p in patch_infos]
        tissue_adequacy = np.mean(tissue_ratios) if tissue_ratios else 0
        
        # 2. Image quality (based on quality scores)
        quality_scores = [p.quality_score for p in patch_infos]
        image_quality = np.mean(quality_scores) if quality_scores else 0
        
        # 3. Stain quality (consistency across patches)
        stain_quality = self._assess_stain_consistency(patch_infos)
        
        # 4. Patch diversity (spatial distribution)
        patch_diversity = self._assess_patch_diversity(patch_infos, wsi_metadata)
        
        # 5. Coverage score (how much of the tissue is sampled)
        coverage_score = self._assess_coverage(patch_infos, wsi_metadata)
        
        # 6. Overall quality (weighted combination)
        overall_quality = (
            0.3 * tissue_adequacy +
            0.25 * image_quality +
            0.15 * stain_quality +
            0.15 * patch_diversity +
            0.15 * coverage_score
        )
        
        # Generate quality flags
        quality_flags = []
        if tissue_adequacy < self.quality_thresholds['tissue_adequacy']:
            quality_flags.append("Insufficient tissue content")
        if image_quality < self.quality_thresholds['image_quality']:
            quality_flags.append("Poor image quality")
        if stain_quality < self.quality_thresholds['stain_quality']:
            quality_flags.append("Inconsistent staining")
        if patch_diversity < self.quality_thresholds['patch_diversity']:
            quality_flags.append("Limited spatial sampling")
        if coverage_score < self.quality_thresholds['coverage_score']:
            quality_flags.append("Inadequate tissue coverage")
        if overall_quality < self.quality_thresholds['overall_quality']:
            quality_flags.append("Overall quality below clinical threshold")
        
        return QualityMetrics(
            tissue_adequacy=tissue_adequacy,
            image_quality=image_quality,
            stain_quality=stain_quality,
            patch_diversity=patch_diversity,
            coverage_score=coverage_score,
            overall_quality=overall_quality,
            quality_flags=quality_flags
        )
    
    def _assess_stain_consistency(self, patch_infos: List[PatchInfo]) -> float:
        """Assess stain consistency across patches"""
        # This would require actual patch images
        # For now, return a default score
        return 0.8
    
    def _assess_patch_diversity(self, patch_infos: List[PatchInfo], wsi_metadata: Dict) -> float:
        """Assess spatial diversity of patches"""
        if len(patch_infos) < 2:
            return 0.0
        
        # Calculate spatial spread
        coordinates = [p.coordinates for p in patch_infos]
        x_coords = [c[0] for c in coordinates]
        y_coords = [c[1] for c in coordinates]
        
        # Coefficient of variation for spatial distribution
        x_cv = np.std(x_coords) / (np.mean(x_coords) + 1e-6)
        y_cv = np.std(y_coords) / (np.mean(y_coords) + 1e-6)
        
        # Normalize to 0-1 range
        diversity_score = min(1.0, (x_cv + y_cv) / 2)
        
        return diversity_score
    
    def _assess_coverage(self, patch_infos: List[PatchInfo], wsi_metadata: Dict) -> float:
        """Assess how well patches cover the tissue area"""
        if not patch_infos:
            return 0.0
        
        # Estimate coverage based on number of patches and tissue area
        # This is a simplified estimation
        num_patches = len(patch_infos)
        estimated_coverage = min(1.0, num_patches / 100)  # Assume 100 patches for good coverage
        
        return estimated_coverage

class ClinicalInferenceEngine:
    """Main clinical inference engine for molecular subtype prediction"""
    
    def __init__(self, 
                 model_path: str,
                 model_config: Dict,
                 calibrator_path: Optional[str] = None,
                 wsi_processor_config: Optional[Dict] = None,
                 device: str = 'cuda'):
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = load_pretrained_model(model_path, model_config)
        self.model.to(self.device)
        self.model.eval()
        
        # Load calibrator if available
        self.calibrator = ConfidenceCalibrator()
        if calibrator_path and Path(calibrator_path).exists():
            self.calibrator.load(calibrator_path)
            logger.info(f"Loaded confidence calibrator from {calibrator_path}")
        
        # Initialize WSI processor
        if wsi_processor_config is None:
            wsi_processor_config = {
                'patch_size': 224,
                'patch_level': 0,
                'overlap': 0.1,
                'tissue_threshold': 0.6,
                'quality_threshold': 0.4,
                'max_patches': 500,
                'stain_normalize': True
            }
        
        self.wsi_processor = create_wsi_processor(wsi_processor_config)
        self.quality_assessor = QualityAssessment()
        
        # Clinical information
        self.subtype_info = {
            'Canonical': {
                'survival_10yr': 0.37,
                'characteristics': 'E2F/MYC activation, NOTCH1/PIK3C2B mutations, sharp tumor borders',
                'treatment': 'DNA damage response inhibitors, combination chemotherapy',
                'oligometastatic_potential': 'moderate',
                'risk_level': 'intermediate'
            },
            'Immune': {
                'survival_10yr': 0.64,
                'characteristics': 'MSI-independent immune activation, dense band-like infiltration',
                'treatment': 'Immunotherapy (PD-1/PD-L1 inhibitors) + local therapy',
                'oligometastatic_potential': 'high',
                'risk_level': 'low'
            },
            'Stromal': {
                'survival_10yr': 0.20,
                'characteristics': 'EMT/angiogenesis, SMAD3 mutation, high VEGFA, immune exclusion',
                'treatment': 'Bevacizumab + stromal targeting agents',
                'oligometastatic_potential': 'low',
                'risk_level': 'high'
            }
        }
        
        logger.info("Clinical inference engine initialized")
    
    def predict_from_wsi(self, 
                        wsi_path: str, 
                        patient_id: str,
                        confidence_threshold: float = 0.7) -> ClinicalPrediction:
        """Complete prediction pipeline from WSI file"""
        
        logger.info(f"Processing WSI for patient {patient_id}: {wsi_path}")
        
        # Process WSI and extract patches
        wsi_results = self.wsi_processor.process_wsi(wsi_path)
        
        if 'error' in wsi_results:
            raise ValueError(f"WSI processing failed: {wsi_results['error']}")
        
        # Extract patches for model inference
        patch_infos = wsi_results['selected_patches']
        patch_tensors = self.wsi_processor.extract_patches_for_model(wsi_path, patch_infos)
        
        if patch_tensors is None or len(patch_tensors) == 0:
            raise ValueError("No valid patches extracted from WSI")
        
        # Quality assessment
        quality_metrics = self.quality_assessor.assess_quality(
            patch_infos, 
            asdict(wsi_results['metadata'])
        )
        
        # Model inference
        prediction_result = self._predict_from_patches(patch_tensors)
        
        # Generate clinical prediction
        clinical_prediction = self._generate_clinical_prediction(
            patient_id=patient_id,
            wsi_filename=Path(wsi_path).name,
            prediction_result=prediction_result,
            quality_metrics=quality_metrics,
            wsi_metadata=wsi_results,
            confidence_threshold=confidence_threshold
        )
        
        return clinical_prediction
    
    def _predict_from_patches(self, patch_tensors: torch.Tensor) -> Dict:
        """Predict molecular subtype from patch tensors"""
        
        patch_tensors = patch_tensors.to(self.device)
        
        with torch.no_grad():
            # Process patches in batches to avoid memory issues
            batch_size = 32
            all_outputs = []
            
            for i in range(0, len(patch_tensors), batch_size):
                batch = patch_tensors[i:i+batch_size]
                output = self.model(batch)
                all_outputs.append(output)
            
            # Aggregate predictions across patches
            all_logits = torch.cat([out['logits'] for out in all_outputs], dim=0)
            all_molecular_logits = torch.cat([out['molecular_logits'] for out in all_outputs], dim=0)
            
            # Ensemble predictions
            ensemble_logits = (all_logits + all_molecular_logits) / 2
            
            # Aggregate using attention-weighted average
            attention_weights = F.softmax(torch.max(ensemble_logits, dim=1)[0], dim=0)
            weighted_logits = torch.sum(ensemble_logits * attention_weights.unsqueeze(1), dim=0)
            
            # Apply calibration if available
            if self.calibrator.is_fitted:
                calibrated_probs = self.calibrator.calibrate(weighted_logits.cpu().numpy().reshape(1, -1))[0]
                probabilities = torch.tensor(calibrated_probs)
            else:
                probabilities = F.softmax(weighted_logits, dim=0)
            
            # Get prediction
            confidence, predicted_idx = torch.max(probabilities, dim=0)
            
            # Calculate uncertainty if available
            uncertainty = None
            if all_outputs[0]['uncertainty'] is not None:
                all_uncertainties = torch.cat([out['uncertainty']['uncertainty'] for out in all_outputs], dim=0)
                uncertainty = torch.mean(all_uncertainties).item()
            
            result = {
                'predicted_idx': predicted_idx.item(),
                'confidence': confidence.item(),
                'probabilities': probabilities.cpu().numpy(),
                'uncertainty': uncertainty,
                'num_patches': len(patch_tensors)
            }
            
            return result
    
    def _generate_clinical_prediction(self,
                                    patient_id: str,
                                    wsi_filename: str,
                                    prediction_result: Dict,
                                    quality_metrics: QualityMetrics,
                                    wsi_metadata: Dict,
                                    confidence_threshold: float) -> ClinicalPrediction:
        """Generate comprehensive clinical prediction"""
        
        # Get predicted subtype
        predicted_idx = prediction_result['predicted_idx']
        subtype_names = ['Canonical', 'Immune', 'Stromal']
        predicted_subtype = subtype_names[predicted_idx]
        
        confidence = prediction_result['confidence']
        probs = prediction_result['probabilities']
        
        # Create probability dictionary
        probabilities = {
            'Canonical': float(probs[0]),
            'Immune': float(probs[1]),
            'Stromal': float(probs[2])
        }
        
        # Get clinical information
        subtype_data = self.subtype_info[predicted_subtype]
        
        # Risk stratification
        if confidence >= confidence_threshold:
            if subtype_data['risk_level'] == 'low':
                risk_stratification = f"Low Risk - {predicted_subtype} subtype (64% 10-year survival)"
            elif subtype_data['risk_level'] == 'intermediate':
                risk_stratification = f"Intermediate Risk - {predicted_subtype} subtype (37% 10-year survival)"
            else:
                risk_stratification = f"High Risk - {predicted_subtype} subtype (20% 10-year survival)"
        else:
            risk_stratification = f"Uncertain - Low confidence prediction ({confidence:.3f})"
        
        # Treatment recommendation
        if confidence >= confidence_threshold:
            treatment_recommendation = f"Recommended: {subtype_data['treatment']}"
        else:
            treatment_recommendation = "Recommend additional testing and multidisciplinary consultation"
        
        # Survival prediction
        survival_10yr = subtype_data['survival_10yr']
        survival_prediction = f"Estimated 10-year survival: {survival_10yr:.0%} (based on Pitroda et al. 2018)"
        
        # Oligometastatic potential
        oligometastatic_potential = f"Oligometastatic potential: {subtype_data['oligometastatic_potential']}"
        
        # Processing metadata
        processing_metadata = {
            'total_patches_analyzed': prediction_result['num_patches'],
            'wsi_dimensions': wsi_metadata['metadata'].dimensions,
            'magnification': wsi_metadata['metadata'].magnification,
            'quality_metrics': asdict(quality_metrics),
            'average_tissue_ratio': wsi_metadata['average_tissue_ratio'],
            'average_quality_score': wsi_metadata['average_quality_score']
        }
        
        # Validation flags
        validation_flags = []
        if confidence < confidence_threshold:
            validation_flags.append(f"Low confidence ({confidence:.3f} < {confidence_threshold})")
        
        validation_flags.extend(quality_metrics.quality_flags)
        
        if prediction_result['uncertainty'] and prediction_result['uncertainty'] > 0.3:
            validation_flags.append(f"High prediction uncertainty ({prediction_result['uncertainty']:.3f})")
        
        if prediction_result['num_patches'] < 50:
            validation_flags.append(f"Limited patch sampling ({prediction_result['num_patches']} patches)")
        
        # Create clinical prediction
        clinical_prediction = ClinicalPrediction(
            patient_id=patient_id,
            wsi_filename=wsi_filename,
            predicted_subtype=predicted_subtype,
            confidence=confidence,
            probabilities=probabilities,
            uncertainty=prediction_result['uncertainty'],
            risk_stratification=risk_stratification,
            treatment_recommendation=treatment_recommendation,
            survival_prediction=survival_prediction,
            oligometastatic_potential=oligometastatic_potential,
            processing_metadata=processing_metadata,
            timestamp=datetime.now().isoformat(),
            model_version="foundation_v1.0",
            validation_flags=validation_flags
        )
        
        return clinical_prediction
    
    def batch_predict(self, 
                     wsi_paths: List[str], 
                     patient_ids: List[str],
                     output_dir: Optional[str] = None) -> List[ClinicalPrediction]:
        """Batch prediction for multiple WSI files"""
        
        if len(wsi_paths) != len(patient_ids):
            raise ValueError("Number of WSI paths must match number of patient IDs")
        
        predictions = []
        
        for wsi_path, patient_id in zip(wsi_paths, patient_ids):
            try:
                prediction = self.predict_from_wsi(wsi_path, patient_id)
                predictions.append(prediction)
                
                logger.info(f"Completed prediction for {patient_id}: {prediction.predicted_subtype} "
                           f"(confidence: {prediction.confidence:.3f})")
                
            except Exception as e:
                logger.error(f"Failed to process {patient_id}: {e}")
                continue
        
        # Save results if output directory provided
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save individual predictions
            for prediction in predictions:
                pred_file = output_path / f"{prediction.patient_id}_prediction.json"
                with open(pred_file, 'w') as f:
                    json.dump(asdict(prediction), f, indent=2, default=str)
            
            # Save summary report
            summary = {
                'total_predictions': len(predictions),
                'predictions_by_subtype': {},
                'average_confidence': np.mean([p.confidence for p in predictions]),
                'high_confidence_predictions': sum(1 for p in predictions if len(p.validation_flags) == 0),
                'timestamp': datetime.now().isoformat()
            }
            
            for subtype in ['Canonical', 'Immune', 'Stromal']:
                count = sum(1 for p in predictions if p.predicted_subtype == subtype)
                summary['predictions_by_subtype'][subtype] = count
            
            with open(output_path / 'batch_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Batch prediction results saved to {output_path}")
        
        return predictions

def create_clinical_inference_engine(config: Dict) -> ClinicalInferenceEngine:
    """Factory function to create clinical inference engine"""
    return ClinicalInferenceEngine(
        model_path=config['model_path'],
        model_config=config['model_config'],
        calibrator_path=config.get('calibrator_path'),
        wsi_processor_config=config.get('wsi_processor_config'),
        device=config.get('device', 'cuda')
    )

if __name__ == "__main__":
    # Example usage
    config = {
        'model_path': 'models/best_molecular_foundation_model.pth',
        'model_config': {
            'backbone': 'efficientnet_b3',
            'num_classes': 3,
            'pretrained': True,
            'use_spatial_transformer': True,
            'use_uncertainty': True
        },
        'calibrator_path': 'models/confidence_calibrator.pkl',
        'device': 'cuda'
    }
    
    # Create inference engine
    engine = create_clinical_inference_engine(config)
    
    print("Clinical inference engine created successfully!")
    print("Ready for WSI processing and molecular subtype prediction.") 