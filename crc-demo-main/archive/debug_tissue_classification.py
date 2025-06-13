#!/usr/bin/env python3
"""Debug tissue classification issue"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app'))

import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image

# Import from app
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'app'))
sys.path.insert(0, '.')
from crc_unified_platform import load_models, analyze_tissue_patch
os.chdir('..')

def test_tissue_classification():
    """Test tissue classification on simple patterns"""
    print("Loading models...")
    tissue_model, tissue_loaded, _, _ = load_models()
    
    # Create simple test image
    img = np.ones((224, 224, 3), dtype=np.uint8) * 200  # Light background
    
    # Add pink tumor region
    cv2.rectangle(img, (50, 50), (150, 150), (255, 180, 200), -1)
    
    # Add purple lymphocytes
    for _ in range(20):
        x, y = np.random.randint(20, 200, 2)
        cv2.circle(img, (x, y), 5, (128, 0, 128), -1)
    
    print("\nTesting tissue classification...")
    
    # Method 1: Direct analysis
    result = analyze_tissue_patch(img, tissue_model)
    print(f"Primary class: {result['primary_class']}")
    print(f"Confidence: {result['confidence']:.1f}%")
    print(f"All predictions:")
    for pred in result['all_predictions']:
        print(f"  {pred['class']}: {pred['confidence']:.1f}%")
    
    # Method 2: Manual prediction
    print("\nManual prediction test:")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_pil = Image.fromarray(img)
    img_tensor = transform(img_pil).unsqueeze(0)
    
    with torch.no_grad():
        outputs = tissue_model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        print(f"Raw outputs shape: {outputs.shape}")
        print(f"Raw outputs: {outputs.squeeze().tolist()}")
        print(f"Probabilities: {probs.squeeze().tolist()}")
        
        # Get prediction
        confidence, predicted = torch.max(probs, 1)
        classes = ['Tumor', 'Stroma', 'Complex', 'Lymphocytes', 'Debris', 'Mucosa', 'Adipose', 'Empty']
        print(f"\nPredicted class: {classes[predicted.item()]} ({confidence.item()*100:.1f}%)")
        
        # Show all class probabilities
        print("\nAll class probabilities:")
        for i, (cls, prob) in enumerate(zip(classes, probs.squeeze().tolist())):
            print(f"  {i}: {cls}: {prob*100:.1f}%")

if __name__ == "__main__":
    test_tissue_classification() 