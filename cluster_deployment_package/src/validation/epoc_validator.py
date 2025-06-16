"""
EPOC Trial Validation Framework
Comprehensive validation for CRC molecular subtype classification
Includes clinical concordance, cross-institutional validation, and regulatory compliance
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, asdict
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, cohen_kappa_score
)
from scipy import stats
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of validation analysis"""
    metric_name: str
    value: float
    confidence_interval: Tuple[float, float]
    p_value: Optional[float] = None
    n_samples: int = 0
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ConcordanceAnalysis:
    """Concordance analysis between model predictions and ground truth"""
    overall_concordance: float
    per_class_concordance: Dict[str, float]
    kappa_score: float
    weighted_kappa: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CrossInstitutionAnalysis:
    """Cross-institutional validation results"""
    institution_performance: Dict[str, Dict[str, float]]
    institution_concordance: Dict[str, float]
    scanner_normalization_effect: Dict[str, float]
    batch_effect_analysis: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EPOCValidator:
    """EPOC Trial Validation Framework"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.class_names = ['Canonical', 'Immune-inflamed', 'Stromal']
        self.institutions = config.get('epoc', {}).get('trial', {}).get('sites', [])
        self.validation_results = {}
        
        # Clinical validation thresholds
        self.concordance_threshold = config.get('epoc', {}).get('clinical_validation', {}).get('concordance_threshold', 0.8)
        self.pathologist_agreement_threshold = 0.75
        self.uncertainty_threshold = config.get('production', {}).get('quality_assurance', {}).get('uncertainty_threshold', 0.3)
        
        logger.info(f"Initialized EPOC validator with {len(self.institutions)} institutions")
        
    def validate(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
        epoch: int
    ) -> Dict[str, Any]:
        """Comprehensive validation for EPOC trial"""
        
        logger.info(f"Starting EPOC validation for epoch {epoch}")
        
        # Collect predictions and ground truth
        predictions, labels, uncertainties, metadata = self._collect_predictions(model, data_loader)
        
        # Core performance metrics
        performance_metrics = self._compute_performance_metrics(predictions, labels, uncertainties)
        
        # Clinical concordance analysis
        concordance_analysis = self._analyze_concordance(predictions, labels, metadata)
        
        # Cross-institutional validation
        cross_institution_analysis = self._analyze_cross_institution(predictions, labels, metadata)
        
        # Uncertainty calibration
        uncertainty_analysis = self._analyze_uncertainty_calibration(predictions, labels, uncertainties)
        
        # Scanner normalization validation
        scanner_analysis = self._analyze_scanner_effects(predictions, labels, metadata)
        
        # Biomarker correlation (if available)
        biomarker_analysis = self._analyze_biomarker_correlation(predictions, labels, metadata)
        
        # Generate clinical report
        clinical_report = self._generate_clinical_report(
            performance_metrics, concordance_analysis, cross_institution_analysis
        )
        
        # Regulatory compliance check
        compliance_analysis = self._check_regulatory_compliance(
            performance_metrics, concordance_analysis
        )
        
        validation_results = {
            'epoch': epoch,
            'performance_metrics': performance_metrics,
            'concordance_analysis': concordance_analysis.to_dict(),
            'cross_institution_analysis': cross_institution_analysis.to_dict(),
            'uncertainty_analysis': uncertainty_analysis,
            'scanner_analysis': scanner_analysis,
            'biomarker_analysis': biomarker_analysis,
            'clinical_report': clinical_report,
            'compliance_analysis': compliance_analysis,
            'validation_passed': self._check_validation_criteria(performance_metrics, concordance_analysis)
        }
        
        self.validation_results[epoch] = validation_results
        
        logger.info(f"EPOC validation completed for epoch {epoch}")
        return validation_results
        
    def _collect_predictions(
        self,
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict]]:
        """Collect model predictions and metadata"""
        
        model.eval()
        all_predictions = []
        all_labels = []
        all_uncertainties = []
        all_metadata = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Forward pass
                outputs = model(batch)
                
                # Get predictions
                logits = outputs['logits']
                probabilities = F.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)
                
                # Get uncertainties
                if 'uncertainty' in outputs:
                    uncertainties = outputs['uncertainty']
                else:
                    # Compute entropy-based uncertainty
                    uncertainties = -torch.sum(probabilities * torch.log(probabilities + 1e-8), dim=1)
                    
                # Collect data
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                all_uncertainties.extend(uncertainties.cpu().numpy())
                
                # Collect metadata
                for i in range(len(batch['labels'])):
                    metadata = {
                        'slide_id': batch['slide_ids'][i] if 'slide_ids' in batch else f"slide_{i}",
                        'site': batch.get('site', ['unknown'] * len(batch['labels']))[i],
                        'scanner': batch.get('scanner', ['unknown'] * len(batch['labels']))[i],
                        'stain': batch.get('stain', ['he'] * len(batch['labels']))[i],
                        'probabilities': probabilities[i].cpu().numpy()
                    }
                    all_metadata.append(metadata)
                    
        return (
            np.array(all_predictions),
            np.array(all_labels),
            np.array(all_uncertainties),
            all_metadata
        )
        
    def _compute_performance_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        uncertainties: np.ndarray
    ) -> Dict[str, ValidationResult]:
        """Compute comprehensive performance metrics"""
        
        metrics = {}
        
        # Accuracy
        accuracy = accuracy_score(labels, predictions)
        acc_ci = self._compute_confidence_interval(labels == predictions)
        metrics['accuracy'] = ValidationResult(
            metric_name='accuracy',
            value=accuracy,
            confidence_interval=acc_ci,
            n_samples=len(labels)
        )
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, labels=[0, 1, 2]
        )
        
        for i, class_name in enumerate(self.class_names):
            metrics[f'{class_name.lower()}_precision'] = ValidationResult(
                metric_name=f'{class_name}_precision',
                value=precision[i],
                confidence_interval=self._compute_confidence_interval_proportion(
                    (predictions == i) & (labels == i), predictions == i
                ),
                n_samples=int(support[i])
            )
            
            metrics[f'{class_name.lower()}_recall'] = ValidationResult(
                metric_name=f'{class_name}_recall',
                value=recall[i],
                confidence_interval=self._compute_confidence_interval_proportion(
                    (predictions == i) & (labels == i), labels == i
                ),
                n_samples=int(support[i])
            )
            
            metrics[f'{class_name.lower()}_f1'] = ValidationResult(
                metric_name=f'{class_name}_f1',
                value=f1[i],
                confidence_interval=(f1[i] - 0.05, f1[i] + 0.05),  # Approximate CI
                n_samples=int(support[i])
            )
            
        # Macro-averaged metrics
        macro_precision = np.mean(precision)
        macro_recall = np.mean(recall)
        macro_f1 = np.mean(f1)
        
        metrics['macro_precision'] = ValidationResult(
            metric_name='macro_precision',
            value=macro_precision,
            confidence_interval=(macro_precision - 0.05, macro_precision + 0.05),
            n_samples=len(labels)
        )
        
        metrics['macro_recall'] = ValidationResult(
            metric_name='macro_recall',
            value=macro_recall,
            confidence_interval=(macro_recall - 0.05, macro_recall + 0.05),
            n_samples=len(labels)
        )
        
        metrics['macro_f1'] = ValidationResult(
            metric_name='macro_f1',
            value=macro_f1,
            confidence_interval=(macro_f1 - 0.05, macro_f1 + 0.05),
            n_samples=len(labels)
        )
        
        # AUC scores (if probabilities available)
        try:
            # Convert to one-hot for AUC calculation
            labels_onehot = np.eye(3)[labels]
            probabilities = np.array([meta['probabilities'] for meta in self.current_metadata])
            
            auc_scores = []
            for i in range(3):
                auc = roc_auc_score(labels_onehot[:, i], probabilities[:, i])
                auc_scores.append(auc)
                
                metrics[f'{self.class_names[i].lower()}_auc'] = ValidationResult(
                    metric_name=f'{self.class_names[i]}_auc',
                    value=auc,
                    confidence_interval=(auc - 0.05, auc + 0.05),
                    n_samples=len(labels)
                )
                
            macro_auc = np.mean(auc_scores)
            metrics['macro_auc'] = ValidationResult(
                metric_name='macro_auc',
                value=macro_auc,
                confidence_interval=(macro_auc - 0.05, macro_auc + 0.05),
                n_samples=len(labels)
            )
            
        except Exception as e:
            logger.warning(f"Could not compute AUC scores: {str(e)}")
            
        return metrics
        
    def _analyze_concordance(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        metadata: List[Dict]
    ) -> ConcordanceAnalysis:
        """Analyze concordance between predictions and ground truth"""
        
        # Overall concordance
        overall_concordance = accuracy_score(labels, predictions)
        
        # Per-class concordance
        per_class_concordance = {}
        confidence_intervals = {}
        
        for i, class_name in enumerate(self.class_names):
            class_mask = labels == i
            if np.sum(class_mask) > 0:
                class_concordance = accuracy_score(labels[class_mask], predictions[class_mask])
                per_class_concordance[class_name] = class_concordance
                
                # Compute confidence interval
                class_correct = (labels[class_mask] == predictions[class_mask])
                ci = self._compute_confidence_interval(class_correct)
                confidence_intervals[class_name] = ci
            else:
                per_class_concordance[class_name] = 0.0
                confidence_intervals[class_name] = (0.0, 0.0)
                
        # Cohen's kappa
        kappa = cohen_kappa_score(labels, predictions)
        weighted_kappa = cohen_kappa_score(labels, predictions, weights='quadratic')
        
        # Overall confidence interval
        overall_correct = (labels == predictions)
        overall_ci = self._compute_confidence_interval(overall_correct)
        confidence_intervals['overall'] = overall_ci
        
        return ConcordanceAnalysis(
            overall_concordance=overall_concordance,
            per_class_concordance=per_class_concordance,
            kappa_score=kappa,
            weighted_kappa=weighted_kappa,
            confidence_intervals=confidence_intervals
        )
        
    def _analyze_cross_institution(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        metadata: List[Dict]
    ) -> CrossInstitutionAnalysis:
        """Analyze performance across different institutions"""
        
        # Group by institution
        institution_data = {}
        for i, meta in enumerate(metadata):
            site = meta.get('site', 'unknown')
            if site not in institution_data:
                institution_data[site] = {'predictions': [], 'labels': [], 'indices': []}
            institution_data[site]['predictions'].append(predictions[i])
            institution_data[site]['labels'].append(labels[i])
            institution_data[site]['indices'].append(i)
            
        # Compute performance per institution
        institution_performance = {}
        institution_concordance = {}
        
        for site, data in institution_data.items():
            site_pred = np.array(data['predictions'])
            site_labels = np.array(data['labels'])
            
            if len(site_pred) > 0:
                # Basic metrics
                accuracy = accuracy_score(site_labels, site_pred)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    site_labels, site_pred, average='macro', zero_division=0
                )
                
                institution_performance[site] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'n_samples': len(site_pred)
                }
                
                institution_concordance[site] = accuracy
                
        # Scanner normalization effect
        scanner_effects = self._analyze_scanner_effects(predictions, labels, metadata)
        
        # Batch effect analysis (simplified)
        batch_effect_analysis = {
            'institution_variance': np.var(list(institution_concordance.values())),
            'min_performance': min(institution_concordance.values()) if institution_concordance else 0.0,
            'max_performance': max(institution_concordance.values()) if institution_concordance else 0.0,
            'performance_range': max(institution_concordance.values()) - min(institution_concordance.values()) if institution_concordance else 0.0
        }
        
        return CrossInstitutionAnalysis(
            institution_performance=institution_performance,
            institution_concordance=institution_concordance,
            scanner_normalization_effect=scanner_effects,
            batch_effect_analysis=batch_effect_analysis
        )
        
    def _analyze_uncertainty_calibration(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        uncertainties: np.ndarray
    ) -> Dict[str, Any]:
        """Analyze uncertainty calibration quality"""
        
        # Expected Calibration Error (ECE)
        ece = self._compute_ece(predictions, labels, uncertainties)
        
        # Reliability diagram data
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        # Convert uncertainties to confidence (1 - uncertainty)
        confidences = 1 - uncertainties
        accuracies = (predictions == labels).astype(float)
        
        ece_bins = []
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece_bins.append({
                    'confidence': avg_confidence_in_bin,
                    'accuracy': accuracy_in_bin,
                    'proportion': prop_in_bin,
                    'count': in_bin.sum()
                })
                
        # Uncertainty quality metrics
        high_uncertainty_mask = uncertainties > self.uncertainty_threshold
        high_uncertainty_accuracy = accuracies[high_uncertainty_mask].mean() if high_uncertainty_mask.sum() > 0 else 0.0
        low_uncertainty_accuracy = accuracies[~high_uncertainty_mask].mean() if (~high_uncertainty_mask).sum() > 0 else 0.0
        
        return {
            'ece': ece,
            'reliability_diagram': ece_bins,
            'high_uncertainty_accuracy': high_uncertainty_accuracy,
            'low_uncertainty_accuracy': low_uncertainty_accuracy,
            'uncertainty_separation': low_uncertainty_accuracy - high_uncertainty_accuracy,
            'high_uncertainty_fraction': high_uncertainty_mask.mean()
        }
        
    def _analyze_scanner_effects(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        metadata: List[Dict]
    ) -> Dict[str, float]:
        """Analyze scanner-specific effects"""
        
        scanner_performance = {}
        
        # Group by scanner
        scanner_data = {}
        for i, meta in enumerate(metadata):
            scanner = meta.get('scanner', 'unknown')
            if scanner not in scanner_data:
                scanner_data[scanner] = {'predictions': [], 'labels': []}
            scanner_data[scanner]['predictions'].append(predictions[i])
            scanner_data[scanner]['labels'].append(labels[i])
            
        # Compute performance per scanner
        for scanner, data in scanner_data.items():
            scanner_pred = np.array(data['predictions'])
            scanner_labels = np.array(data['labels'])
            
            if len(scanner_pred) > 0:
                accuracy = accuracy_score(scanner_labels, scanner_pred)
                scanner_performance[scanner] = accuracy
                
        return scanner_performance
        
    def _analyze_biomarker_correlation(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        metadata: List[Dict]
    ) -> Dict[str, Any]:
        """Analyze correlation with biomarkers (placeholder)"""
        
        # This would analyze correlation with RNA-seq, IHC, mutation status, etc.
        # Implementation depends on availability of biomarker data
        
        return {
            'rna_seq_correlation': 0.0,  # Placeholder
            'ihc_correlation': 0.0,      # Placeholder
            'mutation_correlation': 0.0  # Placeholder
        }
        
    def _generate_clinical_report(
        self,
        performance_metrics: Dict[str, ValidationResult],
        concordance_analysis: ConcordanceAnalysis,
        cross_institution_analysis: CrossInstitutionAnalysis
    ) -> Dict[str, Any]:
        """Generate clinical validation report"""
        
        report = {
            'summary': {
                'overall_accuracy': performance_metrics['accuracy'].value,
                'accuracy_ci': performance_metrics['accuracy'].confidence_interval,
                'concordance': concordance_analysis.overall_concordance,
                'kappa_score': concordance_analysis.kappa_score,
                'cross_institution_variance': cross_institution_analysis.batch_effect_analysis['institution_variance']
            },
            'per_class_performance': {},
            'institution_summary': {
                'best_performing': max(cross_institution_analysis.institution_concordance.items(), key=lambda x: x[1]) if cross_institution_analysis.institution_concordance else ('N/A', 0.0),
                'worst_performing': min(cross_institution_analysis.institution_concordance.items(), key=lambda x: x[1]) if cross_institution_analysis.institution_concordance else ('N/A', 0.0),
                'performance_range': cross_institution_analysis.batch_effect_analysis['performance_range']
            },
            'clinical_significance': {
                'meets_concordance_threshold': concordance_analysis.overall_concordance >= self.concordance_threshold,
                'acceptable_cross_institution_variance': cross_institution_analysis.batch_effect_analysis['performance_range'] < 0.1,
                'ready_for_clinical_use': False  # Will be determined by overall assessment
            }
        }
        
        # Per-class performance
        for class_name in self.class_names:
            class_key = class_name.lower()
            report['per_class_performance'][class_name] = {
                'precision': performance_metrics[f'{class_key}_precision'].value,
                'recall': performance_metrics[f'{class_key}_recall'].value,
                'f1': performance_metrics[f'{class_key}_f1'].value,
                'concordance': concordance_analysis.per_class_concordance.get(class_name, 0.0)
            }
            
        # Overall clinical readiness assessment
        clinical_ready = (
            report['summary']['overall_accuracy'] >= 0.85 and
            report['summary']['concordance'] >= self.concordance_threshold and
            report['institution_summary']['performance_range'] < 0.1 and
            all(perf['f1'] >= 0.7 for perf in report['per_class_performance'].values())
        )
        
        report['clinical_significance']['ready_for_clinical_use'] = clinical_ready
        
        return report
        
    def _check_regulatory_compliance(
        self,
        performance_metrics: Dict[str, ValidationResult],
        concordance_analysis: ConcordanceAnalysis
    ) -> Dict[str, Any]:
        """Check regulatory compliance requirements"""
        
        # FDA/CE marking requirements (simplified)
        fda_requirements = {
            'minimum_accuracy': 0.80,
            'minimum_sensitivity_per_class': 0.75,
            'minimum_specificity_per_class': 0.75,
            'minimum_sample_size': 100,
            'cross_validation_required': True
        }
        
        compliance_status = {}
        
        # Check accuracy requirement
        compliance_status['accuracy_compliant'] = performance_metrics['accuracy'].value >= fda_requirements['minimum_accuracy']
        
        # Check per-class requirements
        compliance_status['per_class_compliant'] = True
        for class_name in self.class_names:
            class_key = class_name.lower()
            recall = performance_metrics[f'{class_key}_recall'].value
            precision = performance_metrics[f'{class_key}_precision'].value
            
            if recall < fda_requirements['minimum_sensitivity_per_class'] or precision < fda_requirements['minimum_specificity_per_class']:
                compliance_status['per_class_compliant'] = False
                break
                
        # Check sample size
        compliance_status['sample_size_compliant'] = performance_metrics['accuracy'].n_samples >= fda_requirements['minimum_sample_size']
        
        # Overall compliance
        compliance_status['overall_compliant'] = all([
            compliance_status['accuracy_compliant'],
            compliance_status['per_class_compliant'],
            compliance_status['sample_size_compliant']
        ])
        
        return {
            'fda_compliance': compliance_status,
            'ce_compliance': compliance_status,  # Similar requirements
            'requirements_met': compliance_status['overall_compliant']
        }
        
    def _check_validation_criteria(
        self,
        performance_metrics: Dict[str, ValidationResult],
        concordance_analysis: ConcordanceAnalysis
    ) -> bool:
        """Check if validation criteria are met"""
        
        criteria = [
            performance_metrics['accuracy'].value >= 0.80,
            concordance_analysis.overall_concordance >= self.concordance_threshold,
            concordance_analysis.kappa_score >= 0.6,
            performance_metrics['macro_f1'].value >= 0.75
        ]
        
        return all(criteria)
        
    def _compute_confidence_interval(self, binary_outcomes: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval for binary outcomes"""
        n = len(binary_outcomes)
        p = np.mean(binary_outcomes)
        
        # Wilson score interval
        z = stats.norm.ppf((1 + confidence) / 2)
        denominator = 1 + z**2 / n
        centre_adjusted_probability = (p + z**2 / (2 * n)) / denominator
        adjusted_standard_deviation = np.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / denominator
        
        lower_bound = centre_adjusted_probability - z * adjusted_standard_deviation
        upper_bound = centre_adjusted_probability + z * adjusted_standard_deviation
        
        return (max(0.0, lower_bound), min(1.0, upper_bound))
        
    def _compute_confidence_interval_proportion(self, numerator: np.ndarray, denominator: np.ndarray) -> Tuple[float, float]:
        """Compute confidence interval for proportions"""
        n_success = np.sum(numerator)
        n_total = np.sum(denominator)
        
        if n_total == 0:
            return (0.0, 0.0)
            
        p = n_success / n_total
        
        # Wilson score interval
        z = 1.96  # 95% confidence
        denominator_adj = 1 + z**2 / n_total
        centre_adjusted = (p + z**2 / (2 * n_total)) / denominator_adj
        adjusted_std = np.sqrt((p * (1 - p) + z**2 / (4 * n_total)) / n_total) / denominator_adj
        
        lower = centre_adjusted - z * adjusted_std
        upper = centre_adjusted + z * adjusted_std
        
        return (max(0.0, lower), min(1.0, upper))
        
    def _compute_ece(self, predictions: np.ndarray, labels: np.ndarray, uncertainties: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error"""
        confidences = 1 - uncertainties
        accuracies = (predictions == labels).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                
        return ece
        
    def save_validation_report(self, epoch: int, output_dir: str):
        """Save comprehensive validation report"""
        if epoch not in self.validation_results:
            logger.error(f"No validation results found for epoch {epoch}")
            return
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save JSON report
        report_path = output_path / f"epoc_validation_epoch_{epoch}.json"
        with open(report_path, 'w') as f:
            json.dump(self.validation_results[epoch], f, indent=2, default=str)
            
        logger.info(f"Saved EPOC validation report to {report_path}")
        
    def generate_plots(self, epoch: int, output_dir: str):
        """Generate validation plots"""
        if epoch not in self.validation_results:
            return
            
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = self.validation_results[epoch]
        
        # Performance metrics plot
        self._plot_performance_metrics(results['performance_metrics'], output_path, epoch)
        
        # Confusion matrix
        # self._plot_confusion_matrix(results, output_path, epoch)
        
        # Cross-institution analysis
        self._plot_cross_institution_analysis(results['cross_institution_analysis'], output_path, epoch)
        
    def _plot_performance_metrics(self, metrics: Dict[str, ValidationResult], output_path: Path, epoch: int):
        """Plot performance metrics with confidence intervals"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        metric_names = []
        metric_values = []
        ci_lower = []
        ci_upper = []
        
        for metric_name, result in metrics.items():
            if 'macro' in metric_name or metric_name == 'accuracy':
                metric_names.append(metric_name)
                metric_values.append(result.value)
                ci_lower.append(result.confidence_interval[0])
                ci_upper.append(result.confidence_interval[1])
                
        y_pos = np.arange(len(metric_names))
        
        ax.barh(y_pos, metric_values, xerr=[np.array(metric_values) - np.array(ci_lower),
                                          np.array(ci_upper) - np.array(metric_values)],
               capsize=5)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(metric_names)
        ax.set_xlabel('Performance Score')
        ax.set_title(f'EPOC Validation Metrics - Epoch {epoch}')
        ax.set_xlim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_path / f'performance_metrics_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_cross_institution_analysis(self, analysis: Dict[str, Any], output_path: Path, epoch: int):
        """Plot cross-institution performance"""
        if not analysis['institution_performance']:
            return
            
        fig, ax = plt.subplots(figsize=(10, 6))
        
        institutions = list(analysis['institution_performance'].keys())
        accuracies = [analysis['institution_performance'][inst]['accuracy'] for inst in institutions]
        
        bars = ax.bar(institutions, accuracies)
        ax.set_ylabel('Accuracy')
        ax.set_title(f'Cross-Institution Performance - Epoch {epoch}')
        ax.set_ylim(0, 1)
        
        # Add threshold line
        ax.axhline(y=self.concordance_threshold, color='red', linestyle='--', 
                  label=f'Concordance Threshold ({self.concordance_threshold})')
        ax.legend()
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path / f'cross_institution_performance_epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close() 