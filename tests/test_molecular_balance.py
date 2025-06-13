#!/usr/bin/env python3
"""Test script to verify molecular predictions are balanced after fixes"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app'))

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torch

# Change to app directory and import
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app'))
sys.path.insert(0, '.')
from crc_unified_platform import load_models
from molecular_subtype_mapper import MolecularSubtypeMapper
os.chdir('..')  # Change back

def generate_test_images():
    """Generate synthetic test images representing different subtypes"""
    images = []
    expected_subtypes = []
    
    # SNF2 (Immune) - Purple/blue with scattered pattern
    img1 = np.zeros((512, 512, 3), dtype=np.uint8)
    # Add purple lymphocyte clusters
    for _ in range(100):
        x, y = np.random.randint(50, 462, 2)
        cv2.circle(img1, (x, y), np.random.randint(5, 15), (128, 0, 128), -1)
    # Add some pink tumor areas
    for _ in range(20):
        x, y = np.random.randint(50, 462, 2)
        cv2.ellipse(img1, (x, y), (30, 20), np.random.randint(0, 180), 0, 360, (200, 150, 200), -1)
    images.append(img1)
    expected_subtypes.append("SNF2")
    
    # SNF1 (Canonical) - Mostly pink with sharp edges
    img2 = np.zeros((512, 512, 3), dtype=np.uint8)
    # Large pink tumor regions
    cv2.rectangle(img2, (50, 50), (250, 250), (255, 200, 200), -1)
    cv2.rectangle(img2, (300, 300), (450, 450), (255, 180, 180), -1)
    cv2.ellipse(img2, (350, 150), (80, 60), 45, 0, 360, (255, 190, 190), -1)
    # Minimal immune cells
    for _ in range(10):
        x, y = np.random.randint(0, 512, 2)
        cv2.circle(img2, (x, y), 3, (100, 0, 100), -1)
    images.append(img2)
    expected_subtypes.append("SNF1")
    
    # SNF3 (Stromal) - White/light with encased tumor
    img3 = np.zeros((512, 512, 3), dtype=np.uint8) + 230  # Light background
    # Add encased tumor regions
    cv2.ellipse(img3, (150, 150), (60, 40), 0, 0, 360, (200, 150, 150), -1)
    cv2.ellipse(img3, (150, 150), (80, 60), 0, 0, 360, (245, 245, 245), 3)
    cv2.ellipse(img3, (350, 350), (50, 70), 45, 0, 360, (200, 150, 150), -1)
    cv2.ellipse(img3, (350, 350), (70, 90), 45, 0, 360, (245, 245, 245), 5)
    # Add some blocked immune cells at periphery
    for _ in range(30):
        angle = np.random.rand() * 2 * np.pi
        r = 100 + np.random.randint(-10, 10)
        x = int(256 + r * np.cos(angle))
        y = int(256 + r * np.sin(angle))
        cv2.circle(img3, (x, y), 3, (100, 0, 100), -1)
    images.append(img3)
    expected_subtypes.append("SNF3")
    
    # Mixed pattern - should be less confident
    img4 = np.zeros((512, 512, 3), dtype=np.uint8)
    # Mix of tumor and immune
    cv2.rectangle(img4, (100, 100), (200, 200), (255, 180, 180), -1)
    for _ in range(50):
        x, y = np.random.randint(0, 512, 2)
        cv2.circle(img4, (x, y), 5, (128, 0, 128), -1)
    # Some stroma
    cv2.rectangle(img4, (300, 100), (400, 400), (240, 240, 240), -1)
    images.append(img4)
    expected_subtypes.append("Mixed")
    
    return images, expected_subtypes

def test_molecular_predictions():
    """Test molecular predictions on synthetic images"""
    print("Loading models...")
    tissue_model, tissue_loaded, subtype_mapper, _ = load_models()
    
    if not tissue_loaded:
        print("WARNING: Tissue model not loaded properly!")
    
    print("\nGenerating test images...")
    test_images, expected_subtypes = generate_test_images()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\nTesting molecular predictions:")
    print("=" * 80)
    
    predictions = []
    
    for i, (img, expected) in enumerate(zip(test_images, expected_subtypes)):
        print(f"\nTest Image {i+1} (Expected: {expected}):")
        
        # Get molecular prediction
        result = subtype_mapper.classify_molecular_subtype(img, transform, detailed_analysis=True)
        
        predicted_subtype = result['subtype']
        confidence = result['confidence']
        probabilities = result['probabilities']
        
        predictions.append(predicted_subtype)
        
        print(f"  Predicted: {predicted_subtype}")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  Probabilities: SNF1={probabilities[0]:.3f}, SNF2={probabilities[1]:.3f}, SNF3={probabilities[2]:.3f}")
        
        # Print tissue composition
        tissue_comp = result['tissue_composition']
        print(f"  Tissue composition:")
        print(f"    - Tumor: {tissue_comp['tumor']:.1%}")
        print(f"    - Stroma: {tissue_comp['stroma']:.1%}")
        print(f"    - Lymphocytes: {tissue_comp['lymphocytes']:.1%}")
        print(f"    - Empty: {tissue_comp['empty']:.1%}")
        
        # Print key features
        if 'confidence_metrics' in result and 'confidence_reasons' in result['confidence_metrics']:
            reasons = result['confidence_metrics']['confidence_reasons']
            if reasons:
                print(f"  Key indicators: {', '.join(reasons[:3])}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY:")
    unique_predictions = set(predictions)
    print(f"Unique predictions: {len(unique_predictions)} ({', '.join(unique_predictions)})")
    
    if len(unique_predictions) == 1:
        print("WARNING: All images predicted as same subtype - bias issue persists!")
    else:
        print("SUCCESS: Diverse predictions achieved!")
    
    # Save test images for visual inspection
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    for i, (img, expected, predicted) in enumerate(zip(test_images[:4], expected_subtypes[:4], predictions[:4])):
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Expected: {expected}\nPredicted: {predicted.split()[0]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('molecular_balance_test.png')
    print("\nTest images saved to molecular_balance_test.png")

if __name__ == "__main__":
    test_molecular_predictions() 