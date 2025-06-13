#!/usr/bin/env python3
"""
Test if quick_model_fixed.pth is less biased
"""

import torch
import numpy as np
from PIL import Image
from pathlib import Path
import sys
sys.path.append('app')

from crc_unified_platform import CRCClassifier, get_transform
import torch.nn.functional as F

def test_model(model_path):
    """Test a specific model for bias"""
    print(f"\nTesting model: {model_path}")
    print("-" * 50)
    
    # Load model
    model = CRCClassifier(num_classes=8)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Test with different colors
    transform = get_transform()
    tissue_classes = ['Tumor', 'Stroma', 'Complex', 'Lymphocytes', 
                     'Debris', 'Mucosa', 'Adipose', 'Empty']
    
    test_colors = {
        'Purple (lymphocyte)': [128, 0, 128],
        'White (stroma)': [240, 240, 240],
        'Pink (tumor)': [255, 192, 203]
    }
    
    tumor_predictions = 0
    
    for color_name, rgb in test_colors.items():
        img = np.ones((224, 224, 3), dtype=np.uint8)
        img[:, :] = rgb
        
        pil_img = Image.fromarray(img)
        img_tensor = transform(pil_img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)[0]
        
        top_idx = probs.argmax().item()
        top_prob = probs[top_idx].item()
        
        print(f"{color_name}: {tissue_classes[top_idx]} ({top_prob:.1%})")
        
        if top_idx == 0:  # Tumor
            tumor_predictions += 1
    
    bias_score = tumor_predictions / len(test_colors) * 100
    print(f"\nTumor bias: {bias_score:.0f}% of test patches classified as tumor")
    
    return bias_score

def main():
    print("=" * 80)
    print("COMPARING TISSUE CLASSIFIER MODELS")
    print("=" * 80)
    
    models_to_test = [
        'models/quick_model.pth',
        'models/quick_model_fixed.pth',
        'models/quick_model_backup.pth'
    ]
    
    results = {}
    
    for model_path in models_to_test:
        if Path(model_path).exists():
            bias = test_model(model_path)
            results[model_path] = bias
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    # Find least biased model
    if results:
        best_model = min(results.items(), key=lambda x: abs(x[1] - 33.3))
        print(f"\nLeast biased model: {best_model[0]}")
        print(f"Tumor bias: {best_model[1]:.0f}%")
        
        if best_model[1] < 50:
            print("\n✓ This model appears more balanced!")
            print("Consider using this model for molecular predictions:")
            print(f"  cp {best_model[0]} models/best_tissue_classifier.pth")
        else:
            print("\n⚠️ All models show significant tumor bias")
            print("A properly trained tissue classifier is needed")

if __name__ == "__main__":
    main() 