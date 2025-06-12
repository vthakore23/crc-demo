#!/usr/bin/env python3
"""
Comprehensive Molecular Subtype Model Evaluation
Clinical validation and performance assessment
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report,
    matthews_corrcoef, cohen_kappa_score
)
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from foundation_model.molecular_subtype_foundation import load_pretrained_model
from foundation_model.clinical_inference import ClinicalInferenceEngine, create_clinical_inference_engine
from scripts.train_epoc_molecular_model import EPOCMolecularDataset, create_data_loaders

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClinicalValidator:
    """Clinical validation of molecular subtype predictions"""
    
    def __init__(self, model_path: str, model_config: Dict, device: str = 'cuda'):
        self.model_path = model_path
        self.model_config = model_config
        self.device = device
        
        # Load model
        self.model = load_pretrained_model(model_path, model_config)
        self.model.to(device)
        self.model.eval()
        
        # Clinical thresholds
        self.clinical_thresholds = {
            'confidence_threshold': 0.70,
            'minimum_accuracy': 0.80,
            'minimum_sensitivity': 0.75,
            'minimum_specificity': 0.75,
            'minimum_ppv': 0.70,
            'minimum_npv': 0.85
        }
        
        self.subtype_names = ['Canonical', 'Immune', 'Stromal']
        
    def comprehensive_evaluation(self, test_loader, output_dir: str):
        """Perform comprehensive clinical evaluation"""
        
        logger.info("Starting comprehensive clinical evaluation...")
        
        # Get predictions
        results = self._get_predictions(test_loader)
        
        # Calculate metrics
        metrics = self._calculate_clinical_metrics(results)
        
        # Clinical validation
        validation_results = self._clinical_validation(metrics)
        
        # Generate reports
        self._generate_clinical_report(metrics, validation_results, output_dir)
        
        # Visualizations
        self._create_visualizations(results, metrics, output_dir)
        
        return metrics, validation_results
    
    def _get_predictions(self, test_loader):
        """Get model predictions on test set"""
        all_predictions = []
        all_targets = []
        all_probabilities = []
        all_uncertainties = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                
                # Get probabilities and predictions
                probs = torch.softmax(output['logits'], dim=1)
                preds = probs.argmax(dim=1)
                
                all_predictions.extend(preds.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
                
                # Get uncertainties if available
                if output['uncertainty'] is not None:
                    uncertainties = output['uncertainty']['uncertainty']
                    all_uncertainties.extend(uncertainties.cpu().numpy())
        
        return {
            'predictions': np.array(all_predictions),
            'targets': np.array(all_targets),
            'probabilities': np.array(all_probabilities),
            'uncertainties': np.array(all_uncertainties) if all_uncertainties else None
        }
    
    def _calculate_clinical_metrics(self, results):
        """Calculate comprehensive clinical metrics"""
        preds = results['predictions']
        targets = results['targets']
        probs = results['probabilities']
        
        # Overall metrics
        overall_metrics = {
            'accuracy': accuracy_score(targets, preds),
            'f1_macro': f1_score(targets, preds, average='macro'),
            'f1_weighted': f1_score(targets, preds, average='weighted'),
            'precision_macro': precision_score(targets, preds, average='macro'),
            'recall_macro': recall_score(targets, preds, average='macro'),
            'mcc': matthews_corrcoef(targets, preds),
            'kappa': cohen_kappa_score(targets, preds)
        }
        
        # Per-class metrics
        per_class_metrics = {}
        for i, subtype in enumerate(self.subtype_names):
            # Binary classification metrics for each class
            y_true_binary = (targets == i).astype(int)
            y_pred_binary = (preds == i).astype(int)
            y_prob = probs[:, i]
            
            # Calculate metrics
            tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary).ravel()
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            # AUC
            if len(np.unique(y_true_binary)) > 1:
                auc = roc_auc_score(y_true_binary, y_prob)
            else:
                auc = 0.5
            
            per_class_metrics[subtype] = {
                'sensitivity': sensitivity,
                'specificity': specificity,
                'ppv': ppv,
                'npv': npv,
                'auc': auc,
                'f1': f1_score(y_true_binary, y_pred_binary),
                'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0)
            }
        
        # Confidence analysis
        confidence_scores = np.max(probs, axis=1)
        confidence_metrics = {
            'mean_confidence': np.mean(confidence_scores),
            'std_confidence': np.std(confidence_scores),
            'high_confidence_ratio': np.mean(confidence_scores >= self.clinical_thresholds['confidence_threshold']),
            'low_confidence_ratio': np.mean(confidence_scores < self.clinical_thresholds['confidence_threshold'])
        }
        
        # High confidence subset analysis
        high_conf_mask = confidence_scores >= self.clinical_thresholds['confidence_threshold']
        if np.sum(high_conf_mask) > 0:
            high_conf_accuracy = accuracy_score(targets[high_conf_mask], preds[high_conf_mask])
        else:
            high_conf_accuracy = 0
        
        confidence_metrics['high_confidence_accuracy'] = high_conf_accuracy
        
        return {
            'overall': overall_metrics,
            'per_class': per_class_metrics,
            'confidence': confidence_metrics,
            'confusion_matrix': confusion_matrix(targets, preds)
        }
    
    def _clinical_validation(self, metrics):
        """Validate against clinical requirements"""
        validation_results = {
            'clinical_readiness': True,
            'passed_checks': [],
            'failed_checks': [],
            'warnings': []
        }
        
        # Check overall accuracy
        if metrics['overall']['accuracy'] >= self.clinical_thresholds['minimum_accuracy']:
            validation_results['passed_checks'].append(f"Overall accuracy: {metrics['overall']['accuracy']:.3f} >= {self.clinical_thresholds['minimum_accuracy']}")
        else:
            validation_results['failed_checks'].append(f"Overall accuracy: {metrics['overall']['accuracy']:.3f} < {self.clinical_thresholds['minimum_accuracy']}")
            validation_results['clinical_readiness'] = False
        
        # Check per-class metrics
        for subtype in self.subtype_names:
            class_metrics = metrics['per_class'][subtype]
            
            # Sensitivity check
            if class_metrics['sensitivity'] >= self.clinical_thresholds['minimum_sensitivity']:
                validation_results['passed_checks'].append(f"{subtype} sensitivity: {class_metrics['sensitivity']:.3f} >= {self.clinical_thresholds['minimum_sensitivity']}")
            else:
                validation_results['failed_checks'].append(f"{subtype} sensitivity: {class_metrics['sensitivity']:.3f} < {self.clinical_thresholds['minimum_sensitivity']}")
                validation_results['clinical_readiness'] = False
            
            # Specificity check
            if class_metrics['specificity'] >= self.clinical_thresholds['minimum_specificity']:
                validation_results['passed_checks'].append(f"{subtype} specificity: {class_metrics['specificity']:.3f} >= {self.clinical_thresholds['minimum_specificity']}")
            else:
                validation_results['failed_checks'].append(f"{subtype} specificity: {class_metrics['specificity']:.3f} < {self.clinical_thresholds['minimum_specificity']}")
                validation_results['clinical_readiness'] = False
            
            # PPV check
            if class_metrics['ppv'] >= self.clinical_thresholds['minimum_ppv']:
                validation_results['passed_checks'].append(f"{subtype} PPV: {class_metrics['ppv']:.3f} >= {self.clinical_thresholds['minimum_ppv']}")
            else:
                validation_results['failed_checks'].append(f"{subtype} PPV: {class_metrics['ppv']:.3f} < {self.clinical_thresholds['minimum_ppv']}")
                validation_results['clinical_readiness'] = False
        
        # Confidence analysis
        high_conf_ratio = metrics['confidence']['high_confidence_ratio']
        if high_conf_ratio >= 0.7:
            validation_results['passed_checks'].append(f"High confidence predictions: {high_conf_ratio:.3f} >= 0.7")
        else:
            validation_results['warnings'].append(f"Low proportion of high confidence predictions: {high_conf_ratio:.3f} < 0.7")
        
        return validation_results
    
    def _generate_clinical_report(self, metrics, validation_results, output_dir):
        """Generate comprehensive clinical report"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Clinical validation report
        report = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_path': str(self.model_path),
            'clinical_readiness': validation_results['clinical_readiness'],
            'overall_metrics': metrics['overall'],
            'per_class_metrics': metrics['per_class'],
            'confidence_analysis': metrics['confidence'],
            'clinical_validation': validation_results,
            'clinical_recommendations': self._generate_recommendations(metrics, validation_results)
        }
        
        # Save report
        with open(output_path / 'clinical_validation_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Generate markdown report
        self._generate_markdown_report(report, output_path)
        
        logger.info(f"Clinical validation report saved to: {output_path}")
    
    def _generate_recommendations(self, metrics, validation_results):
        """Generate clinical recommendations"""
        recommendations = []
        
        if validation_results['clinical_readiness']:
            recommendations.append("✅ Model meets clinical validation criteria")
            recommendations.append("✅ Suitable for clinical deployment with appropriate oversight")
        else:
            recommendations.append("❌ Model does not meet clinical validation criteria")
            recommendations.append("❌ Additional training and validation required before clinical use")
        
        # Specific recommendations based on performance
        for subtype in self.subtype_names:
            class_metrics = metrics['per_class'][subtype]
            
            if class_metrics['sensitivity'] < 0.8:
                recommendations.append(f"⚠️  {subtype}: Low sensitivity ({class_metrics['sensitivity']:.3f}) - consider additional training data")
            
            if class_metrics['ppv'] < 0.7:
                recommendations.append(f"⚠️  {subtype}: Low PPV ({class_metrics['ppv']:.3f}) - high false positive rate")
        
        # Confidence recommendations
        if metrics['confidence']['high_confidence_ratio'] < 0.7:
            recommendations.append("⚠️  Low proportion of high-confidence predictions - consider confidence calibration")
        
        return recommendations
    
    def _generate_markdown_report(self, report, output_path):
        """Generate human-readable markdown report"""
        
        md_content = f"""# Clinical Validation Report - Molecular Subtype Classification

**Evaluation Date:** {report['evaluation_timestamp']}  
**Model:** {report['model_path']}  
**Clinical Readiness:** {'✅ APPROVED' if report['clinical_readiness'] else '❌ NOT APPROVED'}

## Overall Performance

| Metric | Value | Clinical Threshold | Status |
|--------|-------|-------------------|---------|
| Accuracy | {report['overall_metrics']['accuracy']:.3f} | ≥0.80 | {'✅' if report['overall_metrics']['accuracy'] >= 0.80 else '❌'} |
| F1 (Macro) | {report['overall_metrics']['f1_macro']:.3f} | - | - |
| F1 (Weighted) | {report['overall_metrics']['f1_weighted']:.3f} | - | - |
| Matthews Correlation | {report['overall_metrics']['mcc']:.3f} | - | - |
| Cohen's Kappa | {report['overall_metrics']['kappa']:.3f} | - | - |

## Per-Class Performance

"""
        
        for subtype in self.subtype_names:
            metrics = report['per_class_metrics'][subtype]
            md_content += f"""
### {subtype} Subtype

| Metric | Value | Clinical Threshold | Status |
|--------|-------|-------------------|---------|
| Sensitivity | {metrics['sensitivity']:.3f} | ≥0.75 | {'✅' if metrics['sensitivity'] >= 0.75 else '❌'} |
| Specificity | {metrics['specificity']:.3f} | ≥0.75 | {'✅' if metrics['specificity'] >= 0.75 else '❌'} |
| PPV | {metrics['ppv']:.3f} | ≥0.70 | {'✅' if metrics['ppv'] >= 0.70 else '❌'} |
| NPV | {metrics['npv']:.3f} | ≥0.85 | {'✅' if metrics['npv'] >= 0.85 else '❌'} |
| AUC | {metrics['auc']:.3f} | - | - |
| F1-Score | {metrics['f1']:.3f} | - | - |
"""
        
        md_content += f"""
## Confidence Analysis

| Metric | Value |
|--------|-------|
| Mean Confidence | {report['confidence_analysis']['mean_confidence']:.3f} |
| High Confidence Ratio | {report['confidence_analysis']['high_confidence_ratio']:.3f} |
| High Confidence Accuracy | {report['confidence_analysis']['high_confidence_accuracy']:.3f} |

## Clinical Recommendations

"""
        
        for rec in report['clinical_recommendations']:
            md_content += f"- {rec}\n"
        
        md_content += f"""
## Validation Summary

### Passed Checks ({len(report['clinical_validation']['passed_checks'])})
"""
        for check in report['clinical_validation']['passed_checks']:
            md_content += f"- ✅ {check}\n"
        
        md_content += f"""
### Failed Checks ({len(report['clinical_validation']['failed_checks'])})
"""
        for check in report['clinical_validation']['failed_checks']:
            md_content += f"- ❌ {check}\n"
        
        if report['clinical_validation']['warnings']:
            md_content += f"""
### Warnings ({len(report['clinical_validation']['warnings'])})
"""
            for warning in report['clinical_validation']['warnings']:
                md_content += f"- ⚠️ {warning}\n"
        
        # Save markdown report
        with open(output_path / 'clinical_validation_report.md', 'w') as f:
            f.write(md_content)
    
    def _create_visualizations(self, results, metrics, output_dir):
        """Create comprehensive visualizations"""
        output_path = Path(output_dir)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(metrics['confusion_matrix'], 
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.subtype_names, 
                   yticklabels=self.subtype_names)
        plt.title('Confusion Matrix - Molecular Subtype Classification')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Per-class metrics comparison
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        metrics_to_plot = ['sensitivity', 'specificity', 'ppv', 'npv']
        metric_names = ['Sensitivity', 'Specificity', 'PPV', 'NPV']
        
        for i, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
            ax = axes[i//2, i%2]
            
            values = [metrics['per_class'][subtype][metric] for subtype in self.subtype_names]
            bars = ax.bar(self.subtype_names, values)
            
            # Color bars based on clinical thresholds
            threshold = self.clinical_thresholds.get(f'minimum_{metric}', 0.7)
            for bar, value in zip(bars, values):
                bar.set_color('green' if value >= threshold else 'red')
            
            ax.set_title(f'{name} by Molecular Subtype')
            ax.set_ylabel(name)
            ax.set_ylim(0, 1)
            ax.axhline(y=threshold, color='black', linestyle='--', alpha=0.7, 
                      label=f'Clinical Threshold ({threshold})')
            
            # Add value labels
            for j, v in enumerate(values):
                ax.text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
            
            ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'per_class_clinical_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. ROC Curves
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, subtype in enumerate(self.subtype_names):
            y_true = (results['targets'] == i).astype(int)
            y_prob = results['probabilities'][:, i]
            
            if len(np.unique(y_true)) > 1:
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc = roc_auc_score(y_true, y_prob)
                
                axes[i].plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc:.3f})')
                axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.6)
                axes[i].set_xlabel('False Positive Rate')
                axes[i].set_ylabel('True Positive Rate')
                axes[i].set_title(f'{subtype} ROC Curve')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Confidence distribution
        confidence_scores = np.max(results['probabilities'], axis=1)
        correct_predictions = (results['predictions'] == results['targets'])
        
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.hist(confidence_scores, bins=30, alpha=0.7, edgecolor='black')
        plt.axvline(x=self.clinical_thresholds['confidence_threshold'], 
                   color='red', linestyle='--', label='Clinical Threshold')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Distribution of Confidence Scores')
        plt.legend()
        
        plt.subplot(2, 2, 2)
        plt.hist(confidence_scores[correct_predictions], bins=30, alpha=0.7, 
                label='Correct', edgecolor='black', color='green')
        plt.hist(confidence_scores[~correct_predictions], bins=30, alpha=0.7,
                label='Incorrect', edgecolor='black', color='red')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence by Correctness')
        plt.legend()
        
        # Confidence vs Accuracy
        plt.subplot(2, 2, 3)
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        accuracies = []
        for i in range(len(bins) - 1):
            mask = (confidence_scores >= bins[i]) & (confidence_scores < bins[i+1])
            if mask.sum() > 0:
                acc = correct_predictions[mask].mean()
                accuracies.append(acc)
            else:
                accuracies.append(0)
        
        plt.plot(bin_centers, accuracies, 'o-', linewidth=2, markersize=8)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Perfect Calibration')
        plt.xlabel('Confidence Score')
        plt.ylabel('Accuracy')
        plt.title('Confidence Calibration')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Subtype-specific confidence
        plt.subplot(2, 2, 4)
        for i, subtype in enumerate(self.subtype_names):
            subtype_mask = results['targets'] == i
            subtype_confidence = confidence_scores[subtype_mask]
            plt.hist(subtype_confidence, bins=20, alpha=0.6, label=subtype)
        
        plt.axvline(x=self.clinical_thresholds['confidence_threshold'], 
                   color='red', linestyle='--', label='Clinical Threshold')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Confidence by True Subtype')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(output_path / 'confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to: {output_path}")

def benchmark_model(args):
    """Main benchmarking function"""
    
    logger.info("Starting molecular subtype model benchmark...")
    
    # Load model configuration
    if Path(args.model_path).exists():
        checkpoint = torch.load(args.model_path, map_location='cpu')
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
        else:
            # Default configuration
            model_config = {
                'backbone': 'efficientnet_b3',
                'num_classes': 3,
                'pretrained': True,
                'use_spatial_transformer': True,
                'use_uncertainty': True
            }
    else:
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    # Create validator
    validator = ClinicalValidator(args.model_path, model_config, args.device)
    
    # Load test data
    logger.info("Loading test data...")
    
    # Create a simple test loader for evaluation
    class SimpleArgs:
        def __init__(self):
            self.manifest = args.test_manifest
            self.data_dir = args.test_data_dir
            self.batch_size = args.batch_size
            self.val_split = 0.0  # No validation split for testing
            self.test_split = 0.0  # Use all data as test
            self.create_dummy_labels = args.create_dummy_labels
    
    simple_args = SimpleArgs()
    
    # For evaluation, we just need test data
    try:
        # Load test manifest directly
        test_df = pd.read_csv(args.test_manifest)
        
        # Create test dataset and loader
        from torchvision import transforms
        
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = EPOCMolecularDataset(
            test_df, args.test_data_dir, test_transform,
            augment=False, molecular_validation=not args.create_dummy_labels
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        logger.info(f"Loaded {len(test_dataset)} test samples")
        
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        raise
    
    # Perform comprehensive evaluation
    metrics, validation_results = validator.comprehensive_evaluation(test_loader, args.output_dir)
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("CLINICAL VALIDATION SUMMARY")
    logger.info("="*50)
    
    logger.info(f"Clinical Readiness: {'✅ APPROVED' if validation_results['clinical_readiness'] else '❌ NOT APPROVED'}")
    logger.info(f"Overall Accuracy: {metrics['overall']['accuracy']:.3f}")
    logger.info(f"High Confidence Ratio: {metrics['confidence']['high_confidence_ratio']:.3f}")
    logger.info(f"High Confidence Accuracy: {metrics['confidence']['high_confidence_accuracy']:.3f}")
    
    logger.info("\nPer-Class Performance:")
    for subtype in ['Canonical', 'Immune', 'Stromal']:
        class_metrics = metrics['per_class'][subtype]
        logger.info(f"  {subtype}: Sensitivity={class_metrics['sensitivity']:.3f}, "
                   f"Specificity={class_metrics['specificity']:.3f}, "
                   f"PPV={class_metrics['ppv']:.3f}, AUC={class_metrics['auc']:.3f}")
    
    # Save final summary
    summary = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'model_path': args.model_path,
        'test_data': args.test_manifest,
        'clinical_ready': validation_results['clinical_readiness'],
        'key_metrics': {
            'overall_accuracy': metrics['overall']['accuracy'],
            'high_confidence_ratio': metrics['confidence']['high_confidence_ratio'],
            'high_confidence_accuracy': metrics['confidence']['high_confidence_accuracy']
        },
        'per_class_summary': {
            subtype: {
                'sensitivity': metrics['per_class'][subtype]['sensitivity'],
                'specificity': metrics['per_class'][subtype]['specificity'],
                'auc': metrics['per_class'][subtype]['auc']
            }
            for subtype in ['Canonical', 'Immune', 'Stromal']
        }
    }
    
    output_path = Path(args.output_dir)
    with open(output_path / 'evaluation_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"\n✅ Evaluation completed! Results saved to: {args.output_dir}")
    
    return metrics, validation_results

def main():
    parser = argparse.ArgumentParser(description='Evaluate Molecular Subtype Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--test_manifest', type=str, required=True,
                       help='Path to test manifest CSV')
    parser.add_argument('--test_data_dir', type=str, required=True,
                       help='Directory containing test images')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Output directory for evaluation results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for evaluation (cuda/cpu)')
    parser.add_argument('--create_dummy_labels', action='store_true',
                       help='Create dummy labels if not available (for testing)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    
    if not Path(args.test_manifest).exists():
        raise FileNotFoundError(f"Test manifest not found: {args.test_manifest}")
    
    if not Path(args.test_data_dir).exists():
        raise FileNotFoundError(f"Test data directory not found: {args.test_data_dir}")
    
    # Run benchmark
    metrics, validation_results = benchmark_model(args)
    
    # Final status
    if validation_results['clinical_readiness']:
        logger.info("🎉 Model APPROVED for clinical deployment!")
    else:
        logger.info("❌ Model requires additional validation before clinical use.")

if __name__ == "__main__":
    main() 