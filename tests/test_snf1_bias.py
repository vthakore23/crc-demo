#!/usr/bin/env python3
"""
Test script to investigate canonical bias in molecular subtype predictions
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('app')

from molecular_subtype_mapper import MolecularSubtypeMapper
from crc_unified_platform import CRCClassifier
from torchvision import transforms
import pandas as pd

def load_tissue_model():
    """Load the trained tissue classifier"""
    model = CRCClassifier(num_classes=8)
    model_paths = [
        "models/best_tissue_classifier.pth",
        "models/best_model.pth",
        "models/quick_model.pth",
        "models/final_model.pth"
    ]
    
    for path in model_paths:
        if Path(path).exists():
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print(f"Loaded model from {path}")
                break
            except Exception as e:
                print(f"Failed to load {path}: {e}")
    
    model.eval()
    return model

def create_test_images():
    """Create synthetic test images with different tissue patterns"""
    test_images = {}
    
    # Immune-like pattern (immune-rich)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    # Purple/lymphocytes (many small dots)
    for _ in range(500):
        x, y = np.random.randint(0, 224, 2)
        cv2.circle(img, (x, y), 3, (128, 0, 128), -1)
    # Some tumor regions (pink)
    for _ in range(5):
        x, y = np.random.randint(30, 194, 2)
        cv2.circle(img, (x, y), 20, (255, 192, 203), -1)
    test_images['immune_pattern'] = Image.fromarray(img)
    
    # Stromal-like pattern (stromal-rich)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    # Stromal patterns (fibrous, pink)
    for i in range(0, 224, 10):
        cv2.line(img, (i, 0), (i+20, 224), (255, 182, 193), 3)
    # Few tumor nests
    for _ in range(3):
        x, y = np.random.randint(40, 184, 2)
        cv2.circle(img, (x, y), 25, (255, 105, 180), -1)
    test_images['stromal_pattern'] = Image.fromarray(img)
    
    # Canonical-like pattern (tumor-rich, organized)
    img = np.zeros((224, 224, 3), dtype=np.uint8)
    # Large tumor regions with clear borders
    cv2.rectangle(img, (30, 30), (194, 194), (255, 192, 203), -1)
    # Clean edges
    cv2.rectangle(img, (30, 30), (194, 194), (255, 0, 0), 2)
    test_images['canonical_pattern'] = Image.fromarray(img)
    
    return test_images

def analyze_predictions(mapper, transform, test_images):
    """Analyze predictions for test images"""
    results = []
    
    for name, image in test_images.items():
        print(f"\nAnalyzing {name}...")
        
        # Get detailed prediction
        pred_results = mapper.classify_molecular_subtype(image, transform, detailed_analysis=True)
        
        # Extract key information
        result = {
            'image': name,
            'predicted': pred_results['subtype'],
            'confidence': pred_results['confidence'],
            'canonical_prob': pred_results['probabilities'][0],
            'immune_prob': pred_results['probabilities'][1],
            'stromal_prob': pred_results['probabilities'][2],
            'tumor_pct': pred_results['tissue_composition']['tumor'],
            'stroma_pct': pred_results['tissue_composition']['stroma'],
            'lymph_pct': pred_results['tissue_composition']['lymphocytes'],
            'immune_score': pred_results['molecular_signatures']['immune_infiltration'],
            'fibrosis_score': pred_results['molecular_signatures']['fibrosis_level'],
            'confidence_reasons': pred_results['confidence_metrics'].get('confidence_reasons', [])
        }
        results.append(result)
        
        # Print detailed info
        print(f"  Predicted: {result['predicted']}")
        print(f"  Confidence: {result['confidence']:.1f}%")
        print(f"  Probabilities: canonical={result['canonical_prob']:.3f}, immune={result['immune_prob']:.3f}, stromal={result['stromal_prob']:.3f}")
        print(f"  Tissue comp: Tumor={result['tumor_pct']:.2f}, Stroma={result['stroma_pct']:.2f}, Lymph={result['lymph_pct']:.2f}")
        print(f"  Confidence reasons: {result['confidence_reasons']}")
    
    return pd.DataFrame(results)

def test_real_images():
    """Test with real demo images if available"""
    demo_images = [
        'demo_tissue_sample.png',
        'demo_tissue_sample.tif'
    ]
    
    real_results = []
    
    for img_path in demo_images:
        if Path(img_path).exists():
            print(f"\nTesting real image: {img_path}")
            image = Image.open(img_path).convert('RGB')
            
            # Create mapper and transform
            tissue_model = load_tissue_model()
            mapper = MolecularSubtypeMapper(tissue_model)
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Get prediction
            results = mapper.classify_molecular_subtype(image, transform)
            
            print(f"  Predicted: {results['subtype']}")
            print(f"  Confidence: {results['confidence']:.1f}%")
            print(f"  Probabilities: {results['probabilities']}")
            
            real_results.append({
                'image': img_path,
                'predicted': results['subtype'],
                'confidence': results['confidence'],
                'probabilities': results['probabilities']
            })
    
    return real_results

def main():
    print("=" * 80)
    print("TESTING FOR CANONICAL BIAS IN MOLECULAR PREDICTIONS")
    print("=" * 80)
    
    # Load model
    tissue_model = load_tissue_model()
    mapper = MolecularSubtypeMapper(tissue_model)
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Test synthetic images
    print("\n1. Testing with synthetic images...")
    test_images = create_test_images()
    df_results = analyze_predictions(mapper, transform, test_images)
    
    print("\n" + "=" * 80)
    print("SUMMARY OF SYNTHETIC IMAGE TESTS:")
    print(df_results.to_string())
    
    # Check for bias
    canonical_count = (df_results['predicted'].str.contains('canonical')).sum()
    total_count = len(df_results)
    
    print(f"\ncanonical predictions: {canonical_count}/{total_count} ({canonical_count/total_count*100:.0f}%)")
    
    if canonical_count == total_count:
        print("\n⚠️ WARNING: All predictions are canonical! There may be a bias.")
        print("\nPossible causes:")
        print("1. The tissue classifier may be biased toward tumor classification")
        print("2. The scoring weights may favor canonical characteristics")
        print("3. The spatial pattern detection may not be working properly")
    
    # Test real images
    print("\n" + "=" * 80)
    print("2. Testing with real demo images...")
    real_results = test_real_images()
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS:")
    print("1. Check if tissue classifier is properly loaded and working")
    print("2. Verify that spatial pattern analysis is enabled")
    print("3. Review the scoring weights in compute_spatial_enhanced_signatures()")
    print("4. Test with more diverse real images")
    print("5. Consider retraining with balanced EPOC data when available")

if __name__ == "__main__":
    # Import cv2 only if needed
    try:
        import cv2
    except ImportError:
        print("Warning: OpenCV not available, using simplified test images")
        # Fallback implementation without cv2
        def create_test_images():
            test_images = {}
            for pattern in ['canonical_pattern', 'immune_pattern', 'stromal_pattern']:
                # Create simple colored image
                img = np.ones((224, 224, 3), dtype=np.uint8) * 128
                test_images[pattern] = Image.fromarray(img)
            return test_images
    
    main() 