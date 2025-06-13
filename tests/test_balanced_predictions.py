#!/usr/bin/env python3
"""
Test script to verify molecular predictor gives balanced predictions
without bias towards any particular subtype
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict
import sys
import os

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

try:
    from molecular_predictor_advanced import AdvancedMolecularClassifier
    print("✓ Successfully imported Advanced Molecular Predictor")
except ImportError as e:
    print(f"✗ Failed to import Advanced Molecular Predictor: {e}")
    exit(1)


def generate_test_images(n_images=30):
    """Generate diverse test images with different patterns"""
    images = []
    
    for i in range(n_images):
        # Create random tissue-like patterns
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Add various patterns to simulate different tissue types
        pattern_type = i % 6
        
        if pattern_type == 0:
            # Dense cellular pattern (could be tumor or immune)
            for _ in range(200):
                x, y = np.random.randint(20, 204, 2)
                color = np.random.choice([
                    [200, 150, 200],  # Purple (tumor-like)
                    [150, 100, 180],  # Dark purple (lymphocyte-like)
                ])
                cv2.circle(img, (x, y), np.random.randint(3, 8), color, -1)
                
        elif pattern_type == 1:
            # Fibrous pattern (stromal-like)
            for _ in range(50):
                pt1 = (np.random.randint(0, 224), np.random.randint(0, 224))
                pt2 = (np.random.randint(0, 224), np.random.randint(0, 224))
                color = [220, 200, 210]  # Pink/white
                cv2.line(img, pt1, pt2, color, np.random.randint(1, 3))
                
        elif pattern_type == 2:
            # Mixed pattern
            # Add some cells
            for _ in range(100):
                x, y = np.random.randint(10, 214, 2)
                cv2.circle(img, (x, y), 5, [180, 120, 180], -1)
            # Add some stroma
            for _ in range(30):
                pt1 = (np.random.randint(0, 224), np.random.randint(0, 224))
                pt2 = (pt1[0] + np.random.randint(-50, 50), 
                       pt1[1] + np.random.randint(-50, 50))
                cv2.line(img, pt1, pt2, [230, 210, 220], 2)
                
        elif pattern_type == 3:
            # Organized tumor nests
            for cx in range(30, 200, 60):
                for cy in range(30, 200, 60):
                    cv2.circle(img, (cx, cy), 25, [200, 150, 190], -1)
                    cv2.circle(img, (cx, cy), 25, [150, 100, 140], 2)
                    
        elif pattern_type == 4:
            # Scattered immune cells
            for _ in range(300):
                x, y = np.random.randint(5, 219, 2)
                cv2.circle(img, (x, y), 3, [100, 50, 150], -1)
                
        else:
            # Complex mixed pattern
            # Background stroma
            img[:, :] = [240, 220, 230]
            # Add tumor regions
            for _ in range(5):
                cx, cy = np.random.randint(40, 184, 2)
                cv2.ellipse(img, (cx, cy), (30, 20), np.random.randint(0, 180),
                           0, 360, [200, 150, 180], -1)
            # Add immune cells
            for _ in range(150):
                x, y = np.random.randint(10, 214, 2)
                cv2.circle(img, (x, y), 2, [120, 80, 160], -1)
        
        # Add some noise for realism
        noise = np.random.normal(0, 10, img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Apply slight blur for realism
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        images.append(img)
    
    return images


def test_prediction_balance():
    """Test if predictions are balanced across subtypes"""
    print("\n" + "="*60)
    print("Testing Molecular Predictor Balance")
    print("="*60)
    
    # Initialize predictor
    predictor = AdvancedMolecularClassifier()
    print("✓ Initialized Advanced Molecular Predictor (untrained)")
    
    # Generate test images
    print("\nGenerating 30 diverse test images...")
    test_images = generate_test_images(30)
    
    # Collect predictions
    predictions = defaultdict(int)
    confidences = []
    all_probabilities = []
    
    print("\nRunning predictions...")
    for i, img in enumerate(test_images):
        result = predictor.predict(img)
        
        predictions[result['prediction']] += 1
        confidences.append(result['confidence'])
        all_probabilities.append([
            result['probabilities']['SNF1'],
            result['probabilities']['SNF2'],
            result['probabilities']['SNF3']
        ])
        
        if i % 10 == 0:
            print(f"  Processed {i+1}/30 images...")
    
    # Analyze results
    print("\n" + "-"*60)
    print("RESULTS:")
    print("-"*60)
    
    # Distribution of predictions
    print("\nPrediction Distribution:")
    total = sum(predictions.values())
    for subtype in ['SNF1 (Canonical)', 'SNF2 (Immune)', 'SNF3 (Stromal)']:
        count = predictions[subtype]
        percentage = (count / total) * 100
        print(f"  {subtype}: {count}/30 ({percentage:.1f}%)")
    
    # Check if balanced (each should be ~33%)
    expected = total / 3
    chi_square = sum((predictions[st] - expected)**2 / expected 
                     for st in predictions.keys())
    
    # Critical value for chi-square with df=2 at p=0.05 is 5.99
    is_balanced = chi_square < 5.99
    
    print(f"\nChi-square statistic: {chi_square:.2f}")
    print(f"Balanced distribution: {'✓ YES' if is_balanced else '✗ NO'}")
    
    # Confidence statistics
    print(f"\nConfidence Statistics:")
    print(f"  Mean: {np.mean(confidences):.1f}%")
    print(f"  Std: {np.std(confidences):.1f}%")
    print(f"  Min: {np.min(confidences):.1f}%")
    print(f"  Max: {np.max(confidences):.1f}%")
    
    # Probability statistics
    all_probabilities = np.array(all_probabilities)
    print(f"\nAverage Probabilities:")
    print(f"  SNF1: {np.mean(all_probabilities[:, 0]):.3f} ± {np.std(all_probabilities[:, 0]):.3f}")
    print(f"  SNF2: {np.mean(all_probabilities[:, 1]):.3f} ± {np.std(all_probabilities[:, 1]):.3f}")
    print(f"  SNF3: {np.mean(all_probabilities[:, 2]):.3f} ± {np.std(all_probabilities[:, 2]):.3f}")
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Prediction distribution
    subtypes = list(predictions.keys())
    counts = list(predictions.values())
    colors = ['#e74c3c', '#27ae60', '#e67e22']
    
    axes[0].bar(range(len(subtypes)), counts, color=colors)
    axes[0].set_xticks(range(len(subtypes)))
    axes[0].set_xticklabels([st.split(' ')[0] for st in subtypes])
    axes[0].axhline(y=10, color='gray', linestyle='--', label='Expected (balanced)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Prediction Distribution')
    axes[0].legend()
    
    # Confidence distribution
    axes[1].hist(confidences, bins=20, color='skyblue', edgecolor='black')
    axes[1].axvline(x=np.mean(confidences), color='red', linestyle='--', 
                    label=f'Mean: {np.mean(confidences):.1f}%')
    axes[1].set_xlabel('Confidence (%)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Confidence Distribution')
    axes[1].legend()
    
    # Average probabilities
    avg_probs = np.mean(all_probabilities, axis=0)
    std_probs = np.std(all_probabilities, axis=0)
    
    x = range(3)
    axes[2].bar(x, avg_probs, yerr=std_probs, color=colors, capsize=10)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(['SNF1', 'SNF2', 'SNF3'])
    axes[2].axhline(y=0.333, color='gray', linestyle='--', label='Expected (1/3)')
    axes[2].set_ylabel('Average Probability')
    axes[2].set_title('Average Prediction Probabilities')
    axes[2].legend()
    axes[2].set_ylim(0, 0.5)
    
    plt.tight_layout()
    plt.savefig('molecular_prediction_balance_test.png', dpi=150)
    print(f"\n✓ Results saved to molecular_prediction_balance_test.png")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    
    if is_balanced:
        print("✅ SUCCESS: Molecular predictor shows balanced predictions!")
        print("   - No bias towards any particular subtype")
        print("   - Each subtype gets roughly equal predictions")
        print("   - Confidence levels are moderate (as expected for untrained model)")
    else:
        print("⚠️  WARNING: Predictions may still show some bias")
        print("   Please check the distribution and adjust if needed")
    
    print("\n💡 Next steps:")
    print("   1. Run the platform to see balanced predictions in action")
    print("   2. Train with EPOC data for high accuracy:")
    print("      python app/train_advanced_molecular_predictor.py")
    print("="*60)


if __name__ == "__main__":
    # Check if we have cv2
    try:
        import cv2
    except ImportError:
        print("Error: OpenCV (cv2) is required for this test")
        print("Please install it with: pip install opencv-python")
        exit(1)
    
    test_prediction_balance() 