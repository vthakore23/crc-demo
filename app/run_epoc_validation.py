#!/usr/bin/env python3
"""
Run EPOC Validation
Example script for validating molecular subtype predictions on EPOC trial data
"""

from app.epoc_validation import EPOCValidator
import pandas as pd

def create_example_manifest():
    """Create an example EPOC manifest CSV structure"""
    example_data = {
        'patient_id': ['EPOC_001', 'EPOC_002', 'EPOC_003'],
        'wsi_path': ['patient001.tif', 'patient002.tif', 'patient003.tif'],
        'molecular_subtype': ['canonical', 'immune', 'stromal'],
        'treatment_arm': ['chemo', 'chemo+cetuximab', 'chemo'],
        'pfs_months': [12.5, 24.3, 8.1],
        'os_months': [18.2, 36.5, 12.4],
        'pfs_event': [1, 0, 1],
        'os_event': [1, 0, 1],
        'recurrence_site': ['liver', 'lung', 'peritoneal']
    }
    
    df = pd.DataFrame(example_data)
    df.to_csv('example_epoc_manifest.csv', index=False)
    print("Created example_epoc_manifest.csv")
    print("\nExpected CSV structure:")
    print(df.head())
    print("\n" + "="*60)

def main():
    """Run EPOC validation when data is available"""
    
    print("="*60)
    print("EPOC TRIAL VALIDATION SETUP")
    print("="*60)
    print("\nThis script will validate your molecular subtype predictions")
    print("against EPOC trial ground truth data.")
    print("\nRequired data structure:")
    print("1. EPOC manifest CSV with columns:")
    print("   - patient_id")
    print("   - wsi_path (path to WSI file)")
    print("   - molecular_subtype (canonical/immune/stromal)")
    print("   - treatment_arm (chemo/chemo+cetuximab)")
    print("   - pfs_months, os_months")
    print("   - pfs_event, os_event (0/1)")
    print("   - recurrence_site")
    print("\n2. Directory containing WSI files")
    print("="*60)
    
    # Create example manifest
    create_example_manifest()
    
    # When you have EPOC data, uncomment and modify:
    """
    # Initialize validator
    validator = EPOCValidator()
    
    # Process EPOC cohort
    results_df = validator.process_epoc_cohort(
        epoc_manifest_csv="path/to/your/epoc_manifest.csv",
        wsi_directory="path/to/your/epoc_wsis/"
    )
    
    # Generate validation report
    report = validator.generate_validation_report(
        results_df,
        output_dir="epoc_validation_results"
    )
    
    # The validation will produce:
    # 1. epoc_validation_results/validation_report.json - Summary metrics
    # 2. epoc_validation_results/detailed_predictions.csv - All predictions
    # 3. epoc_validation_results/clinical_correlations.png - Survival curves, etc.
    """
    
    print("\nValidation metrics that will be calculated:")
    print("- Overall accuracy, F1-score, Cohen's Kappa")
    print("- Per-subtype precision, recall, AUROC")
    print("- Survival analysis (Kaplan-Meier curves)")
    print("- Treatment response by subtype")
    print("- Recurrence pattern analysis")
    print("- Clinical insights (immune surgical benefit, stromal resistance)")
    print("\nReady for EPOC validation when data is available!")
    print("="*60)

if __name__ == "__main__":
    main() 