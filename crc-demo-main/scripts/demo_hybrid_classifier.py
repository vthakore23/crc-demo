#!/usr/bin/env python3
"""
Demo Script: Hybrid PyRadiomics-Deep Learning Classifier
Demonstrates the key features and improvements of the hybrid approach
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import time

def create_demo_tissue_model():
    """Create a simple tissue classification model for demonstration"""
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
    
    model = SimpleClassifier(num_classes=8)
    model.eval()
    return model

def create_demo_image(subtype='SNF2', size=(512, 512)):
    """Create a demo histopathology image"""
    np.random.seed(42)  # For reproducible demo
    
    # Different patterns for each subtype
    if subtype == 'SNF1':
        # Canonical - organized tumor pattern
        image = np.full((*size, 3), [200, 150, 150], dtype=np.uint8)  # Pink base
        # Add organized tumor regions
        for i in range(0, size[0], 40):
            for j in range(0, size[1], 40):
                if (i + j) % 80 == 0:
                    image[i:i+30, j:j+30] = [180, 100, 140]  # Darker tumor
                    
    elif subtype == 'SNF2':
        # Immune - scattered immune infiltration
        image = np.full((*size, 3), [220, 200, 200], dtype=np.uint8)  # Light background
        # Add scattered immune cells
        for _ in range(200):
            x, y = np.random.randint(0, size[0]-10), np.random.randint(0, size[1]-10)
            image[x:x+5, y:y+5] = [100, 100, 200]  # Blue immune cells
            
    else:  # SNF3
        # Stromal - fibrous pattern
        image = np.full((*size, 3), [160, 140, 120], dtype=np.uint8)  # Brown base
        # Add fibrous strands
        for i in range(0, size[0], 15):
            thickness = np.random.randint(3, 8)
            image[i:i+thickness, :] = [140, 120, 100]  # Darker fibers
    
    # Add some noise for realism
    noise = np.random.randint(-20, 20, image.shape)
    image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
    
    return image

def demo_feature_comparison():
    """Demonstrate the difference between standard and hybrid approaches"""
    print("="*80)
    print("HYBRID PYRADIOMICS CLASSIFIER DEMONSTRATION")
    print("="*80)
    
    # Check if PyRadiomics is available
    try:
        import radiomics
        from app.hybrid_radiomics_classifier import HybridRadiomicsClassifier, create_clinical_report
        pyradiomics_available = True
        print("✓ PyRadiomics available - Full hybrid functionality enabled")
    except ImportError:
        pyradiomics_available = False
        print("⚠ PyRadiomics not available - Install with: pip install pyradiomics")
        print("Demonstration will show architecture without radiomic features")
    
    # Import standard classifier
    from app.molecular_subtype_mapper import MolecularSubtypeMapper
    
    # Create tissue model and transform
    print("\n1. Initializing Models...")
    tissue_model = create_demo_tissue_model()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Initialize both classifiers
    standard_mapper = MolecularSubtypeMapper(tissue_model)
    
    if pyradiomics_available:
        hybrid_classifier = HybridRadiomicsClassifier(tissue_model)
        print("✓ Both standard and hybrid classifiers initialized")
    else:
        hybrid_classifier = None
        print("✓ Standard classifier initialized")
    
    # Test on different subtype examples
    subtypes = ['SNF1', 'SNF2', 'SNF3']
    
    print("\n2. Comparing Predictions on Demo Images...")
    print("-" * 80)
    
    for subtype in subtypes:
        print(f"\nTesting {subtype} pattern:")
        
        # Create demo image
        demo_image = create_demo_image(subtype)
        
        # Standard prediction
        start_time = time.time()
        standard_result = standard_mapper.classify_molecular_subtype(demo_image, transform)
        standard_time = time.time() - start_time
        
        print(f"  Standard Classifier:")
        print(f"    Prediction: {standard_result['subtype']}")
        print(f"    Confidence: {standard_result['confidence']:.1f}%")
        print(f"    Processing time: {standard_time:.3f}s")
        
        # Hybrid prediction (if available)
        if hybrid_classifier:
            try:
                start_time = time.time()
                hybrid_result = hybrid_classifier.predict(demo_image, transform, explain=True)
                hybrid_time = time.time() - start_time
                
                print(f"  Hybrid Classifier:")
                print(f"    Prediction: {hybrid_result['subtype']}")
                print(f"    Confidence: {hybrid_result['confidence']:.1f}%")
                print(f"    Processing time: {hybrid_time:.3f}s")
                
                # Show feature breakdown
                if 'feature_summary' in hybrid_result:
                    fs = hybrid_result['feature_summary']
                    print(f"    Features used: {fs.get('features_used_for_prediction', 'N/A')}")
                    print(f"    - Deep learning: {fs.get('deep_features', 0)}")
                    print(f"    - Radiomic: {fs.get('radiomic_features', 0)}")
                    print(f"    - Spatial: {fs.get('spatial_features', 0)}")
                
                # Show explanation if available
                if 'explanation' in hybrid_result and hybrid_result['explanation'].get('prediction_drivers'):
                    print(f"    Key drivers:")
                    for driver in hybrid_result['explanation']['prediction_drivers'][:2]:
                        print(f"      - {driver}")
                        
            except Exception as e:
                print(f"  Hybrid Classifier: Error - {e}")
        
        print("-" * 40)
    
    # Demonstrate clinical report generation
    if hybrid_classifier and pyradiomics_available:
        print("\n3. Clinical Report Generation...")
        
        # Use SNF2 example for clinical report
        demo_image = create_demo_image('SNF2')
        
        try:
            prediction = hybrid_classifier.predict(demo_image, transform, explain=True)
            clinical_report = create_clinical_report(prediction, patient_id="DEMO_001")
            
            print("✓ Clinical report generated:")
            print("-" * 40)
            # Show first part of report
            report_lines = clinical_report.split('\n')
            for line in report_lines[:20]:  # Show first 20 lines
                print(line)
            print("... (truncated)")
            print("-" * 40)
            
        except Exception as e:
            print(f"Clinical report generation failed: {e}")
    
    # Feature extraction demonstration
    print("\n4. Feature Extraction Capabilities...")
    
    if hybrid_classifier and pyradiomics_available:
        demo_image = create_demo_image('SNF1')
        
        try:
            features = hybrid_classifier.extract_hybrid_features(demo_image, transform)
            
            print("✓ Hybrid feature extraction successful:")
            print(f"  Total combined features: {len(features['combined_features'])}")
            
            # Count feature types
            radiomic_count = len([k for k in features['combined_features'].keys() if k.startswith('radiomic_')])
            deep_count = len([k for k in features['combined_features'].keys() if k.startswith('deep_')])
            spatial_count = len([k for k in features['combined_features'].keys() if k.startswith('spatial_')])
            
            print(f"  - Radiomic features: {radiomic_count}")
            print(f"  - Deep learning features: {deep_count}")
            print(f"  - Spatial pattern features: {spatial_count}")
            
            # Show some example feature names
            example_features = list(features['combined_features'].keys())[:5]
            print(f"  Example features: {', '.join(example_features)}")
            
        except Exception as e:
            print(f"Feature extraction demonstration failed: {e}")
    else:
        print("Hybrid feature extraction requires PyRadiomics installation")
    
    # Benefits summary
    print("\n5. Key Benefits of Hybrid Approach...")
    print("✓ Combines interpretable radiomic features with deep learning")
    print("✓ Enhanced clinical interpretability through SHAP explanations")
    print("✓ Robust feature selection using multiple methods")
    print("✓ Improved accuracy through ensemble modeling")
    print("✓ Automated clinical report generation")
    print("✓ Better performance on limited datasets")
    
    # Installation instructions
    if not pyradiomics_available:
        print("\n6. To Enable Full Hybrid Functionality:")
        print("   pip install pyradiomics>=3.0.1")
        print("   pip install SimpleITK>=2.1.0")
        print("   pip install xgboost>=1.7.0")
        print("   pip install shap>=0.41.0")
        print("   pip install boruta>=0.3")
    
    print("\n" + "="*80)
    print("DEMONSTRATION COMPLETED")
    print("="*80)
    
    if pyradiomics_available:
        print("To train your own hybrid model:")
        print("  python scripts/train_hybrid_classifier.py")
        print("\nTo validate on EPOC data with hybrid features:")
        print("  validator = EPOCValidator(tissue_model, use_hybrid_classifier=True)")
    else:
        print("Install PyRadiomics dependencies to access full hybrid functionality")

def demo_validation_integration():
    """Demonstrate EPOC validation with hybrid classifier"""
    print("\n" + "="*60)
    print("EPOC VALIDATION WITH HYBRID CLASSIFIER")
    print("="*60)
    
    try:
        from app.epoc_validation import EPOCValidator
        
        # Create tissue model
        tissue_model = create_demo_tissue_model()
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Initialize validator with hybrid classifier
        print("Initializing EPOC validator with hybrid classifier...")
        
        validator = EPOCValidator(
            tissue_model=tissue_model,
            transform=transform,
            use_hybrid_classifier=True
        )
        
        print("✓ EPOC validator with hybrid classifier initialized")
        print("\nTo validate on real EPOC data:")
        print("  results_df = validator.process_epoc_cohort(")
        print("      epoc_manifest_csv='path/to/epoc_manifest.csv',")
        print("      wsi_directory='path/to/wsi_files/'")
        print("  )")
        print("  report = validator.generate_validation_report(results_df, 'output_dir')")
        
    except Exception as e:
        print(f"EPOC validation demo failed: {e}")

if __name__ == "__main__":
    try:
        demo_feature_comparison()
        demo_validation_integration()
        
        print("\nFor more information, see: HYBRID_RADIOMICS_INTEGRATION.md")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc() 