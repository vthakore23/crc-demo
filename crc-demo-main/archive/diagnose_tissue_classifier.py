#!/usr/bin/env python3
"""
Diagnostic script to check tissue classifier behavior
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.append('app')

from crc_unified_platform import CRCClassifier, get_transform
import torch.nn.functional as F
from torchvision import transforms

def test_tissue_classifier():
    """Test the tissue classifier with various inputs"""
    print("=" * 80)
    print("TISSUE CLASSIFIER DIAGNOSTIC TEST")
    print("=" * 80)
    
    # Load model
    model = CRCClassifier(num_classes=8)
    model_path = Path('models/quick_model.pth')
    
    if model_path.exists():
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded model from {model_path}")
    else:
        print("WARNING: No model found! Using random weights.")
    
    model.eval()
    
    # Test with different colored patches
    transform = get_transform()
    tissue_classes = ['Tumor', 'Stroma', 'Complex', 'Lymphocytes', 
                     'Debris', 'Mucosa', 'Adipose', 'Empty']
    
    print("\n1. Testing with solid color patches:")
    print("-" * 50)
    
    # Test different solid colors
    test_colors = {
        'Pink (tumor-like)': [255, 192, 203],
        'Purple (lymphocyte-like)': [128, 0, 128],
        'White (stroma-like)': [240, 240, 240],
        'Red (blood/debris)': [255, 0, 0],
        'Yellow (adipose-like)': [255, 255, 200],
        'Gray (empty-like)': [128, 128, 128],
        'Black': [0, 0, 0],
        'Random noise': None
    }
    
    for color_name, rgb in test_colors.items():
        if rgb is None:
            # Random noise
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        else:
            # Solid color
            img = np.ones((224, 224, 3), dtype=np.uint8)
            img[:, :] = rgb
        
        pil_img = Image.fromarray(img)
        img_tensor = transform(pil_img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)[0]
        
        top_idx = probs.argmax().item()
        top_prob = probs[top_idx].item()
        
        print(f"\n{color_name}:")
        print(f"  Top prediction: {tissue_classes[top_idx]} ({top_prob:.2%})")
        
        # Show top 3
        top3_probs, top3_idx = torch.topk(probs, 3)
        for i in range(3):
            print(f"  {i+1}. {tissue_classes[top3_idx[i]]}: {top3_probs[i]:.2%}")
    
    # Test with real image if available
    print("\n\n2. Testing with real demo image:")
    print("-" * 50)
    
    demo_path = Path('demo_tissue_sample.png')
    if demo_path.exists():
        demo_img = Image.open(demo_path).convert('RGB')
        demo_tensor = transform(demo_img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(demo_tensor)
            probs = F.softmax(outputs, dim=1)[0]
        
        print("\nTissue composition:")
        for i, tissue in enumerate(tissue_classes):
            print(f"  {tissue}: {probs[i]:.2%}")
    
    # Check model statistics
    print("\n\n3. Model weight statistics:")
    print("-" * 50)
    
    # Check final layer bias
    if hasattr(model.backbone, 'fc'):
        fc_layers = list(model.backbone.fc.children())
        for i, layer in enumerate(fc_layers):
            if hasattr(layer, 'bias') and layer.bias is not None:
                bias = layer.bias.detach().cpu().numpy()
                print(f"\nLayer {i} bias statistics:")
                if len(bias) == 8:  # Final layer
                    for j, tissue in enumerate(tissue_classes):
                        print(f"  {tissue}: {bias[j]:.3f}")
                    print(f"\n  Tumor bias rank: {np.argsort(bias)[::-1].tolist().index(0) + 1}/8")
    
    # Test batch behavior
    print("\n\n4. Testing batch behavior:")
    print("-" * 50)
    
    # Create batch of different images
    batch_imgs = []
    batch_labels = []
    
    for i, (color_name, rgb) in enumerate(list(test_colors.items())[:4]):
        if rgb is None:
            img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        else:
            img = np.ones((224, 224, 3), dtype=np.uint8)
            img[:, :] = rgb
        
        pil_img = Image.fromarray(img)
        img_tensor = transform(pil_img)
        batch_imgs.append(img_tensor)
        batch_labels.append(color_name)
    
    batch_tensor = torch.stack(batch_imgs)
    
    with torch.no_grad():
        outputs = model(batch_tensor)
        probs = F.softmax(outputs, dim=1)
    
    print("\nBatch predictions:")
    for i, label in enumerate(batch_labels):
        top_idx = probs[i].argmax().item()
        top_prob = probs[i][top_idx].item()
        print(f"  {label}: {tissue_classes[top_idx]} ({top_prob:.2%})")
    
    # Summary
    print("\n\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY:")
    print("=" * 80)
    
    # Count how many predictions were tumor
    tumor_count = sum(1 for p in probs if p.argmax().item() == 0)
    
    if tumor_count >= len(probs) * 0.8:
        print("\n⚠️ ISSUE DETECTED: Tissue classifier is heavily biased toward tumor classification!")
        print("\nPossible causes:")
        print("1. The model was trained on imbalanced data with too many tumor samples")
        print("2. The model weights are not properly loaded (using random initialization)")
        print("3. The preprocessing/normalization is incorrect")
        print("4. The model is overfitting to tumor features")
        
        print("\nRecommended fixes:")
        print("1. Retrain the tissue classifier with balanced data")
        print("2. Verify the model file is correct and properly loaded")
        print("3. Check if the correct model checkpoint is being used")
        print("4. Use data augmentation and regularization during training")
    else:
        print("\n✓ Tissue classifier appears to be working normally")
        print("  Predictions are reasonably distributed across tissue types")

if __name__ == "__main__":
    test_tissue_classifier() 