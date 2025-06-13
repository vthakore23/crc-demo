#!/usr/bin/env python3
"""
Simple test to verify molecular predictor gives balanced predictions
"""

import numpy as np
from PIL import Image
import sys
import os
from collections import defaultdict

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

def create_test_image(pattern='random'):
    """Create a simple test image"""
    img = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
    
    if pattern == 'pink':
        # More pink/red tones (tumor-like)
        img[:, :, 0] = np.random.randint(180, 255, (224, 224))  # High red
        img[:, :, 1] = np.random.randint(100, 150, (224, 224))  # Med green
        img[:, :, 2] = np.random.randint(150, 200, (224, 224))  # High blue
    elif pattern == 'purple':
        # Purple tones (lymphocyte-like)
        img[:, :, 0] = np.random.randint(100, 150, (224, 224))  # Med red
        img[:, :, 1] = np.random.randint(50, 100, (224, 224))   # Low green
        img[:, :, 2] = np.random.randint(150, 255, (224, 224))  # High blue
    elif pattern == 'white':
        # White/light (stromal-like)
        img[:, :, :] = np.random.randint(200, 255, (224, 224, 3))
    
    # Convert to PIL Image
    return Image.fromarray(img)

def test_molecular_balance():
    """Test molecular predictor for balance"""
    print("Testing Molecular Predictor Balance")
    print("=" * 50)
    
    # Try the existing predictor first (which should be working)
    try:
        from molecular_subtype_mapper import MolecularSubtypeMapper
        from crc_unified_platform import load_models
        tissue_model, _, _, _ = load_models()
        predictor = MolecularSubtypeMapper(tissue_model)
        print("✓ Using existing molecular predictor (with balance fixes)")
        use_advanced = False
    except Exception as e:
        print(f"✗ Failed to load existing predictor: {e}")
        # Fallback to advanced predictor
        try:
            from molecular_predictor_advanced import AdvancedMolecularClassifier
            predictor = AdvancedMolecularClassifier()
            print("✓ Initialized Advanced Molecular Predictor")
            use_advanced = True
        except ImportError as e2:
            print(f"✗ Failed to import advanced predictor: {e2}")
            return False
    
    # Test with different image patterns
    patterns = ['random', 'pink', 'purple', 'white']
    predictions = defaultdict(int)
    
    print(f"\nTesting with {len(patterns) * 5} images...")
    
    for pattern in patterns:
        for i in range(5):
            # Create test image
            img = create_test_image(pattern)
            
            if use_advanced:
                result = predictor.predict(img, extract_features=False)
                prediction = result['prediction']
                confidence = result['confidence'] * 100
            else:
                from torchvision import transforms
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                result = predictor.classify_molecular_subtype(img, transform, detailed_analysis=False)
                prediction = result['subtype']
                confidence = result['confidence']
            
            predictions[prediction] += 1
            print(f"  {pattern:>8} image {i+1}: {prediction} ({confidence:.1f}%)")
    
    # Analyze results
    print("\n" + "=" * 50)
    print("RESULTS:")
    print("=" * 50)
    
    total = sum(predictions.values())
    print(f"\nPrediction Distribution (Total: {total}):")
    
    for subtype in sorted(predictions.keys()):
        count = predictions[subtype]
        percentage = (count / total) * 100
        print(f"  {subtype}: {count}/{total} ({percentage:.1f}%)")
    
    # Check balance
    expected = total / 3
    max_count = max(predictions.values())
    min_count = min(predictions.values())
    
    is_balanced = (max_count - min_count) <= 4  # Allow some variation
    
    print(f"\nBalance Check:")
    print(f"  Expected per subtype: ~{expected:.1f}")
    print(f"  Range: {min_count} to {max_count}")
    print(f"  Balanced: {'✓ YES' if is_balanced else '✗ NO'}")
    
    if is_balanced:
        print("\n✅ SUCCESS: Molecular predictor shows balanced predictions!")
        print("   No bias towards any particular subtype detected.")
    else:
        print("\n⚠️  WARNING: Some bias may still exist")
        print("   Consider checking the prediction logic.")
    
    return is_balanced

if __name__ == "__main__":
    test_molecular_balance() 