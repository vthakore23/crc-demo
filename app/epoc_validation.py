#!/usr/bin/env python3
"""
EPOC Trial Validation Module
Batch validation of molecular subtype predictions against EPOC trial data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, cohen_kappa_score
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from app.molecular_subtype_mapper import MolecularSubtypeMapper
from app.hybrid_radiomics_classifier import HybridRadiomicsClassifier
from PIL import Image
import json

class EPOCValidator:
    """Validate molecular subtype predictions against EPOC trial ground truth"""
    
    def __init__(self, tissue_model=None, transform=None, model_path="models/quick_model.pth", 
                 use_hybrid_classifier=True):
        """Initialize validator with trained models"""
        self.use_hybrid_classifier = use_hybrid_classifier
        
        # If model is provided, use it; otherwise try to load from file
        if tissue_model is not None:
            self.tissue_model = tissue_model
            self.transform = transform
        else:
            # Dynamic import to avoid circular dependency
            import sys
            sys.path.append('.')
            from torchvision import models, transforms
            import torch.nn as nn
            
            # Define a simple classifier if needed
            class SimpleClassifier(nn.Module):
                def __init__(self, num_classes=8):
                    super().__init__()
                    self.backbone = models.resnet50(weights=None)
                    num_features = self.backbone.fc.in_features
                    self.backbone.fc = nn.Sequential(
                        nn.Dropout(0.5),
                        nn.Linear(num_features, 512),
                        nn.ReLU(),
                        nn.Dropout(0.5),
                        nn.Linear(512, num_classes)
                    )
                    
                def forward(self, x):
                    return self.backbone(x)
            
            self.tissue_model = SimpleClassifier(num_classes=8)
            
            # Try to load weights
            try:
                state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                self.tissue_model.load_state_dict(state_dict, strict=False)
                print(f"Successfully loaded model from {model_path}")
            except Exception as e:
                print(f"Failed to load {model_path}: {str(e)}. Using random initialization.")
            
            self.tissue_model.eval()
            
            # Define transform
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        
        # Initialize molecular subtype mapper
        self.subtype_mapper = MolecularSubtypeMapper(self.tissue_model)
        
        # Initialize hybrid classifier if requested
        if self.use_hybrid_classifier:
            try:
                self.hybrid_classifier = HybridRadiomicsClassifier(self.tissue_model)
                print("Hybrid PyRadiomics classifier initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize hybrid classifier: {e}")
                print("Falling back to standard molecular subtype mapper")
                self.use_hybrid_classifier = False
                self.hybrid_classifier = None
        else:
            self.hybrid_classifier = None
        
    def process_epoc_cohort(self, epoc_manifest_csv, wsi_directory):
        """
        Process entire EPOC cohort
        
        Args:
            epoc_manifest_csv: Path to CSV with columns:
                - patient_id
                - wsi_path (relative to wsi_directory)
                - molecular_subtype (ground truth: canonical/immune/stromal)
                - treatment_arm (chemo/chemo+cetuximab)
                - pfs_months
                - os_months
                - pfs_event (0/1)
                - os_event (0/1)
                - recurrence_site (liver/lung/peritoneal/other)
            wsi_directory: Base directory containing WSI files
        """
        # Load EPOC metadata
        epoc_df = pd.read_csv(epoc_manifest_csv)
        results = []
        
        print(f"Processing {len(epoc_df)} EPOC cases...")
        
        for idx, row in tqdm(epoc_df.iterrows(), total=len(epoc_df)):
            # Load WSI (simplified - in practice you'd tile the WSI)
            wsi_path = Path(wsi_directory) / row['wsi_path']
            
            try:
                # For demo: assume pre-extracted representative tiles
                # In practice: use OpenSlide to extract tiles from WSI
                image = Image.open(wsi_path).convert('RGB')
                
                # Predict molecular subtype using hybrid classifier if available
                if self.use_hybrid_classifier and self.hybrid_classifier is not None:
                    try:
                        prediction = self.hybrid_classifier.predict(
                            np.array(image), self.transform, explain=True
                        )
                        prediction_method = "Hybrid PyRadiomics-Deep Learning"
                    except Exception as e:
                        print(f"Warning: Hybrid prediction failed for {row['patient_id']}: {e}")
                        # Fallback to standard mapper
                        prediction = self.subtype_mapper.classify_molecular_subtype(
                            image, self.transform
                        )
                        prediction_method = "Standard Deep Learning"
                else:
                    prediction = self.subtype_mapper.classify_molecular_subtype(
                        image, self.transform
                    )
                    prediction_method = "Standard Deep Learning"
                
                # Store results - handle different prediction formats
                if self.use_hybrid_classifier and 'probabilities_by_subtype' in prediction:
                    # Hybrid classifier format
                    predicted_subtype = prediction['subtype'].split()[0]  # Extract canonical/2/3
                    results.append({
                        'patient_id': row['patient_id'],
                        'true_subtype': row['molecular_subtype'],
                        'predicted_subtype': predicted_subtype,
                        'confidence': prediction['confidence'],
                        'snf1_prob': prediction['probabilities_by_subtype'].get('canonical (Canonical)', 0),
                        'snf2_prob': prediction['probabilities_by_subtype'].get('immune (Immune)', 0),
                        'snf3_prob': prediction['probabilities_by_subtype'].get('stromal (Stromal)', 0),
                        'treatment_arm': row['treatment_arm'],
                        'pfs_months': row['pfs_months'],
                        'os_months': row['os_months'],
                        'pfs_event': row['pfs_event'],
                        'os_event': row['os_event'],
                        'recurrence_site': row['recurrence_site'],
                        'prediction_method': prediction_method,
                        'feature_summary': prediction.get('feature_summary', {}),
                        'explanation': prediction.get('explanation', {})
                    })
                else:
                    # Standard mapper format
                    results.append({
                        'patient_id': row['patient_id'],
                        'true_subtype': row['molecular_subtype'],
                        'predicted_subtype': prediction['subtype'].split()[0],  # Extract canonical/2/3
                        'confidence': prediction['confidence'],
                        'snf1_prob': prediction['probabilities'][0],
                        'snf2_prob': prediction['probabilities'][1],
                        'snf3_prob': prediction['probabilities'][2],
                        'treatment_arm': row['treatment_arm'],
                        'pfs_months': row['pfs_months'],
                        'os_months': row['os_months'],
                        'pfs_event': row['pfs_event'],
                        'os_event': row['os_event'],
                        'recurrence_site': row['recurrence_site'],
                        'prediction_method': prediction_method,
                        'feature_summary': {},
                        'explanation': {}
                    })
                
            except Exception as e:
                print(f"Error processing {row['patient_id']}: {e}")
                continue
        
        return pd.DataFrame(results)
    
    def calculate_validation_metrics(self, results_df):
        """Calculate comprehensive validation metrics"""
        
        # Convert subtype names to numeric for sklearn
        subtype_map = {'canonical': 0, 'immune': 1, 'stromal': 2}
        y_true = results_df['true_subtype'].map(subtype_map)
        y_pred = results_df['predicted_subtype'].map(subtype_map)
        
        # Basic classification metrics
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
            'cohen_kappa': cohen_kappa_score(y_true, y_pred)
        }
        
        # Per-class metrics
        for subtype in ['canonical', 'immune', 'stromal']:
            subtype_mask = results_df['true_subtype'] == subtype
            if subtype_mask.sum() > 0:
                metrics[f'{subtype}_precision'] = (
                    (results_df[subtype_mask]['predicted_subtype'] == subtype).sum() / 
                    (results_df['predicted_subtype'] == subtype).sum()
                )
                metrics[f'{subtype}_recall'] = (
                    (results_df[subtype_mask]['predicted_subtype'] == subtype).sum() / 
                    subtype_mask.sum()
                )
        
        # ROC-AUC for each subtype (one-vs-rest)
        for i, subtype in enumerate(['canonical', 'immune', 'stromal']):
            y_true_binary = (y_true == i).astype(int)
            y_score = results_df[f'{subtype.lower()}_prob']
            metrics[f'{subtype}_auroc'] = roc_auc_score(y_true_binary, y_score)
        
        # Average confidence by correctness
        correct_mask = results_df['true_subtype'] == results_df['predicted_subtype']
        metrics['avg_confidence_correct'] = results_df[correct_mask]['confidence'].mean()
        metrics['avg_confidence_incorrect'] = results_df[~correct_mask]['confidence'].mean()
        
        return metrics
    
    def analyze_clinical_correlations(self, results_df):
        """Analyze clinical outcome correlations"""
        
        # Survival analysis by predicted subtype
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Kaplan-Meier curves for OS by predicted subtype
        ax = axes[0, 0]
        for subtype in ['canonical', 'immune', 'stromal']:
            mask = results_df['predicted_subtype'] == subtype
            if mask.sum() > 5:  # Need sufficient samples
                kmf = KaplanMeierFitter()
                kmf.fit(
                    results_df[mask]['os_months'],
                    results_df[mask]['os_event'],
                    label=f'{subtype} (n={mask.sum()})'
                )
                kmf.plot_survival_function(ax=ax)
        ax.set_title('Overall Survival by Predicted Subtype')
        ax.set_xlabel('Months')
        ax.set_ylabel('Survival Probability')
        
        # 2. Treatment response by subtype
        ax = axes[0, 1]
        response_data = []
        for subtype in ['canonical', 'immune', 'stromal']:
            for treatment in ['chemo', 'chemo+cetuximab']:
                mask = (results_df['predicted_subtype'] == subtype) & \
                       (results_df['treatment_arm'] == treatment)
                if mask.sum() > 0:
                    median_pfs = results_df[mask]['pfs_months'].median()
                    response_data.append({
                        'Subtype': subtype,
                        'Treatment': treatment,
                        'Median PFS': median_pfs,
                        'N': mask.sum()
                    })
        
        response_df = pd.DataFrame(response_data)
        response_pivot = response_df.pivot(
            index='Subtype', 
            columns='Treatment', 
            values='Median PFS'
        )
        response_pivot.plot(kind='bar', ax=ax)
        ax.set_title('Median PFS by Subtype and Treatment')
        ax.set_ylabel('Median PFS (months)')
        ax.set_xlabel('Predicted Subtype')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
        
        # 3. Confusion matrix
        ax = axes[1, 0]
        cm = confusion_matrix(
            results_df['true_subtype'], 
            results_df['predicted_subtype'],
            labels=['canonical', 'immune', 'stromal']
        )
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['canonical', 'immune', 'stromal'],
                    yticklabels=['canonical', 'immune', 'stromal'],
                    ax=ax)
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Subtype')
        ax.set_ylabel('True Subtype')
        
        # 4. Recurrence patterns
        ax = axes[1, 1]
        recurrence_data = []
        for subtype in ['canonical', 'immune', 'stromal']:
            mask = results_df['predicted_subtype'] == subtype
            if mask.sum() > 0:
                for site in ['liver', 'lung', 'peritoneal', 'other']:
                    count = (results_df[mask]['recurrence_site'] == site).sum()
                    recurrence_data.append({
                        'Subtype': subtype,
                        'Site': site,
                        'Percentage': 100 * count / mask.sum()
                    })
        
        rec_df = pd.DataFrame(recurrence_data)
        rec_pivot = rec_df.pivot(index='Site', columns='Subtype', values='Percentage')
        rec_pivot.plot(kind='bar', ax=ax)
        ax.set_title('Recurrence Site Distribution by Predicted Subtype')
        ax.set_ylabel('Percentage (%)')
        ax.set_xlabel('Recurrence Site')
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
    
    def generate_validation_report(self, results_df, output_dir):
        """Generate comprehensive validation report"""
        
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Calculate metrics
        metrics = self.calculate_validation_metrics(results_df)
        
        # Generate clinical correlation plots
        clinical_fig = self.analyze_clinical_correlations(results_df)
        clinical_fig.savefig(output_dir / 'clinical_correlations.png', dpi=300, bbox_inches='tight')
        
        # Create summary report
        report = {
            'validation_date': pd.Timestamp.now().isoformat(),
            'n_samples': len(results_df),
            'metrics': metrics,
            'subtype_distribution': {
                'true': results_df['true_subtype'].value_counts().to_dict(),
                'predicted': results_df['predicted_subtype'].value_counts().to_dict()
            },
            'clinical_insights': {
                'snf2_surgical_benefit': self._calculate_snf2_benefit(results_df),
                'snf3_cetuximab_resistance': self._calculate_snf3_resistance(results_df)
            },
            'model_performance': {
                'prediction_methods': results_df['prediction_method'].value_counts().to_dict(),
                'hybrid_classifier_usage': self.use_hybrid_classifier,
                'feature_analysis': self._analyze_hybrid_features(results_df) if self.use_hybrid_classifier else {}
            }
        }
        
        # Save report
        with open(output_dir / 'validation_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save detailed results
        results_df.to_csv(output_dir / 'detailed_predictions.csv', index=False)
        
        # Print summary
        print("\n" + "="*60)
        print("EPOC VALIDATION SUMMARY")
        print("="*60)
        print(f"Total samples: {len(results_df)}")
        print(f"Overall accuracy: {metrics['accuracy']:.3f}")
        print(f"Macro F1-score: {metrics['macro_f1']:.3f}")
        print(f"Cohen's Kappa: {metrics['cohen_kappa']:.3f}")
        print("\nPer-subtype AUROC:")
        for subtype in ['canonical', 'immune', 'stromal']:
            print(f"  {subtype}: {metrics[f'{subtype}_auroc']:.3f}")
        
        # Print hybrid classifier information
        if self.use_hybrid_classifier and 'model_performance' in report:
            model_perf = report['model_performance']
            print(f"\nModel Performance:")
            print(f"Hybrid classifier enabled: {model_perf['hybrid_classifier_usage']}")
            prediction_methods = model_perf.get('prediction_methods', {})
            for method, count in prediction_methods.items():
                print(f"  {method}: {count} samples")
            
            feature_analysis = model_perf.get('feature_analysis', {})
            if 'feature_distributions' in feature_analysis:
                feat_dist = feature_analysis['feature_distributions']
                print(f"\nFeature Analysis:")
                print(f"  Avg total features: {feat_dist.get('avg_total_features', 0):.1f}")
                print(f"  Avg radiomic features: {feat_dist.get('avg_radiomic_features', 0):.1f}")
                print(f"  Avg deep learning features: {feat_dist.get('avg_deep_features', 0):.1f}")
                print(f"  Feature extraction success rate: {feat_dist.get('feature_usage_rate', 0):.1%}")
        
        print("\nFiles saved to:", output_dir)
        print("="*60)
        
        return report
    
    def _calculate_snf2_benefit(self, results_df):
        """Calculate surgical benefit for predicted immune patients"""
        snf2_mask = results_df['predicted_subtype'] == 'immune'
        if snf2_mask.sum() > 10:
            median_os = results_df[snf2_mask]['os_months'].median()
            return {
                'n_patients': int(snf2_mask.sum()),
                'median_os_months': float(median_os),
                'oligometastatic_eligible': True
            }
        return None
    
    def _calculate_snf3_resistance(self, results_df):
        """Calculate cetuximab resistance in predicted stromal patients"""
        snf3_cetux_mask = (
            (results_df['predicted_subtype'] == 'stromal') & 
            (results_df['treatment_arm'] == 'chemo+cetuximab')
        )
        if snf3_cetux_mask.sum() > 5:
            median_pfs = results_df[snf3_cetux_mask]['pfs_months'].median()
            return {
                'n_patients': int(snf3_cetux_mask.sum()),
                'median_pfs_months': float(median_pfs),
                'likely_resistant': median_pfs < 6  # Example threshold
            }
        return None
    
    def _analyze_hybrid_features(self, results_df):
        """Analyze hybrid feature usage and importance"""
        analysis = {
            'feature_distributions': {},
            'prediction_explanations': {},
            'model_interpretability': {}
        }
        
        # Analyze feature summaries
        feature_summaries = [row for row in results_df['feature_summary'] if row and isinstance(row, dict)]
        if feature_summaries:
            total_features = [fs.get('total_features_extracted', 0) for fs in feature_summaries]
            deep_features = [fs.get('deep_features', 0) for fs in feature_summaries]
            radiomic_features = [fs.get('radiomic_features', 0) for fs in feature_summaries]
            spatial_features = [fs.get('spatial_features', 0) for fs in feature_summaries]
            
            analysis['feature_distributions'] = {
                'avg_total_features': np.mean(total_features),
                'avg_deep_features': np.mean(deep_features),
                'avg_radiomic_features': np.mean(radiomic_features),
                'avg_spatial_features': np.mean(spatial_features),
                'feature_usage_rate': len(feature_summaries) / len(results_df)
            }
        
        # Analyze prediction explanations
        explanations = [row for row in results_df['explanation'] if row and isinstance(row, dict)]
        if explanations:
            explanation_drivers = []
            for exp in explanations:
                drivers = exp.get('prediction_drivers', [])
                explanation_drivers.extend(drivers)
            
            analysis['prediction_explanations'] = {
                'total_explanations': len(explanations),
                'common_drivers': list(set(explanation_drivers))[:10],  # Top 10 common drivers
                'explanation_rate': len(explanations) / len(results_df)
            }
        
        return analysis


# Example usage
if __name__ == "__main__":
    # Initialize validator
    validator = EPOCValidator()
    
    # Example: Process EPOC cohort
    # results_df = validator.process_epoc_cohort(
    #     epoc_manifest_csv="path/to/epoc_manifest.csv",
    #     wsi_directory="path/to/epoc_wsis/"
    # )
    
    # Generate validation report
    # report = validator.generate_validation_report(
    #     results_df,
    #     output_dir="epoc_validation_results"
    # )
    
    print("EPOC Validator ready. Use process_epoc_cohort() with your data.") 