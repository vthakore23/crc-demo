#!/usr/bin/env python3
"""Test molecular predictions on realistic tissue patterns"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app'))

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# Change to app directory and import
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app'))
sys.path.insert(0, '.')
from crc_unified_platform import load_models
from molecular_subtype_mapper import MolecularSubtypeMapper
os.chdir('..')  # Change back

def create_realistic_tissue_patterns():
    """Create more realistic tissue patterns for each subtype"""
    images = []
    descriptions = []
    
    # SNF2 (Immune-rich) - Dense purple infiltration throughout
    print("Creating SNF2 (Immune) pattern...")
    img_snf2 = np.ones((512, 512, 3), dtype=np.uint8) * 240  # Light background
    
    # Add tumor regions (pink)
    for i in range(3):
        x, y = np.random.randint(100, 400, 2)
        cv2.ellipse(img_snf2, (x, y), (60, 40), np.random.randint(0, 180), 
                   0, 360, (255, 180, 200), -1)
    
    # Add many lymphocytes throughout (purple)
    for _ in range(200):
        x, y = np.random.randint(0, 512, 2)
        size = np.random.randint(3, 8)
        cv2.circle(img_snf2, (x, y), size, (128, 0, 128), -1)
    
    # Add lymphoid aggregates
    for _ in range(5):
        cx, cy = np.random.randint(50, 450, 2)
        for _ in range(30):
            dx, dy = np.random.randint(-20, 20, 2)
            cv2.circle(img_snf2, (cx+dx, cy+dy), 4, (100, 0, 150), -1)
    
    images.append(img_snf2)
    descriptions.append("SNF2 (Immune) - Dense lymphocyte infiltration")
    
    # SNF1 (Canonical) - Organized tumor with sharp borders
    print("Creating SNF1 (Canonical) pattern...")
    img_snf1 = np.ones((512, 512, 3), dtype=np.uint8) * 250  # Very light background
    
    # Large organized tumor regions with sharp edges
    points1 = np.array([[100, 100], [300, 120], [280, 320], [80, 300]], np.int32)
    cv2.fillPoly(img_snf1, [points1], (255, 160, 180))
    
    points2 = np.array([[350, 200], [450, 180], [470, 380], [330, 400]], np.int32)
    cv2.fillPoly(img_snf1, [points2], (255, 170, 190))
    
    # Very few immune cells
    for _ in range(15):
        x, y = np.random.randint(0, 512, 2)
        cv2.circle(img_snf1, (x, y), 3, (150, 50, 150), -1)
    
    # Add clean borders
    cv2.polylines(img_snf1, [points1], True, (255, 140, 160), 2)
    cv2.polylines(img_snf1, [points2], True, (255, 140, 160), 2)
    
    images.append(img_snf1)
    descriptions.append("SNF1 (Canonical) - Organized tumor, minimal immune")
    
    # SNF3 (Stromal) - Heavy fibrosis with encased tumor
    print("Creating SNF3 (Stromal) pattern...")
    img_snf3 = np.ones((512, 512, 3), dtype=np.uint8) * 235  # Light gray background
    
    # Add wavy stromal patterns (fibrosis)
    for y in range(0, 512, 15):
        for x in range(0, 512, 2):
            intensity = int(240 + 10 * np.sin(x * 0.05 + y * 0.1))
            cv2.circle(img_snf3, (x, y), 1, (intensity, intensity, intensity), -1)
    
    # Small encased tumor islands
    for _ in range(4):
        cx, cy = np.random.randint(100, 400, 2)
        # Tumor core
        cv2.ellipse(img_snf3, (cx, cy), (30, 20), np.random.randint(0, 180), 
                   0, 360, (255, 150, 170), -1)
        # Stromal barrier (thick white ring)
        cv2.ellipse(img_snf3, (cx, cy), (45, 35), np.random.randint(0, 180), 
                   0, 360, (245, 245, 245), 8)
    
    # Lymphocytes blocked at periphery
    for _ in range(80):
        angle = np.random.rand() * 2 * np.pi
        r = 200 + np.random.randint(0, 50)
        x = int(256 + r * np.cos(angle))
        y = int(256 + r * np.sin(angle))
        if 0 <= x < 512 and 0 <= y < 512:
            cv2.circle(img_snf3, (x, y), 3, (120, 0, 120), -1)
    
    images.append(img_snf3)
    descriptions.append("SNF3 (Stromal) - Fibrotic with encased tumor")
    
    # Mixed/Ambiguous pattern
    print("Creating mixed pattern...")
    img_mixed = np.ones((512, 512, 3), dtype=np.uint8) * 245
    
    # Some tumor
    cv2.rectangle(img_mixed, (150, 150), (250, 250), (255, 180, 190), -1)
    
    # Some stroma
    cv2.rectangle(img_mixed, (300, 200), (400, 300), (235, 235, 235), -1)
    
    # Moderate immune cells
    for _ in range(50):
        x, y = np.random.randint(0, 512, 2)
        cv2.circle(img_mixed, (x, y), 4, (130, 20, 130), -1)
    
    images.append(img_mixed)
    descriptions.append("Mixed pattern - Unclear subtype")
    
    return images, descriptions

def main():
    """Test molecular predictions on realistic patterns"""
    print("Loading models...")
    tissue_model, tissue_loaded, subtype_mapper, _ = load_models()
    
    print(f"Tissue model loaded: {tissue_loaded}")
    
    print("\nCreating realistic tissue patterns...")
    test_images, descriptions = create_realistic_tissue_patterns()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\nAnalyzing tissue patterns:")
    print("=" * 80)
    
    results = []
    
    for img, desc in zip(test_images, descriptions):
        print(f"\n{desc}:")
        
        # Get prediction
        result = subtype_mapper.classify_molecular_subtype(img, transform, detailed_analysis=True)
        
        predicted = result['subtype']
        confidence = result['confidence']
        probs = result['probabilities']
        
        results.append({
            'description': desc,
            'predicted': predicted,
            'confidence': confidence,
            'probabilities': probs
        })
        
        print(f"  Predicted: {predicted}")
        print(f"  Confidence: {confidence:.1f}%")
        print(f"  Probabilities: SNF1={probs[0]:.3f}, SNF2={probs[1]:.3f}, SNF3={probs[2]:.3f}")
        
        # Show tissue composition
        tissue = result['tissue_composition']
        print(f"  Main tissues: Tumor={tissue['tumor']:.1%}, Stroma={tissue['stroma']:.1%}, Lymphocytes={tissue['lymphocytes']:.1%}")
        
        # Show key features
        if 'spatial_patterns' in result:
            patterns = result['spatial_patterns']
            print(f"  Spatial patterns detected:")
            if patterns['immune_highways']['highway_present']:
                print(f"    - Immune highways: Yes")
            if patterns['stromal_barriers']['strong_barriers_present']:
                print(f"    - Stromal barriers: Yes")
            if patterns['interface_sharpness']['sharp_interfaces']:
                print(f"    - Sharp interfaces: Yes")
            if patterns['lymphoid_aggregates']['aggregates_present']:
                print(f"    - Lymphoid aggregates: Yes")
    
    # Summary
    print("\n" + "=" * 80)
    print("PREDICTION SUMMARY:")
    
    predicted_subtypes = [r['predicted'] for r in results]
    unique_predictions = set(predicted_subtypes)
    
    print(f"Unique predictions: {len(unique_predictions)}")
    for subtype in ['SNF1 (Canonical)', 'SNF2 (Immune)', 'SNF3 (Stromal)']:
        count = predicted_subtypes.count(subtype)
        print(f"  {subtype}: {count} predictions")
    
    # Visual summary
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, (img, result) in enumerate(zip(test_images, results)):
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"{result['description']}\nPredicted: {result['predicted'].split()[0]} ({result['confidence']:.0f}%)")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('real_tissue_predictions.png', dpi=150)
    print(f"\nVisual summary saved to real_tissue_predictions.png")
    
    # Check for bias
    if len(unique_predictions) == 1:
        print("\n⚠️  WARNING: All predictions are the same subtype - bias issue detected!")
    elif len(unique_predictions) == 2:
        print("\n⚡ PARTIAL SUCCESS: Two different subtypes predicted")
    else:
        print("\n✅ SUCCESS: All three subtypes can be predicted!")

if __name__ == "__main__":
    main() 