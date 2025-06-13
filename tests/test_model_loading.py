#!/usr/bin/env python3
"""Test tissue model loading"""

import torch
import os
from pathlib import Path

# Check model files
models_dir = Path("models")
print("Available models:")
for model_file in models_dir.glob("*.pth"):
    print(f"  - {model_file}")
    try:
        # Try loading the model
        state_dict = torch.load(model_file, map_location='cpu', weights_only=False)
        if isinstance(state_dict, dict):
            if 'model_state_dict' in state_dict:
                print(f"    ✓ Contains model_state_dict")
                print(f"    Keys: {list(state_dict.keys())[:5]}...")
            else:
                print(f"    ✓ Direct state dict with {len(state_dict)} keys")
                print(f"    Sample keys: {list(state_dict.keys())[:3]}...")
        else:
            print(f"    ? Unknown format: {type(state_dict)}")
    except Exception as e:
        print(f"    ✗ Error loading: {e}")
    print()

# Test loading from app directory
print("\nTesting from app directory:")
os.chdir("app")
model_paths = [
    "../models/best_tissue_classifier.pth",
    "../models/balanced_tissue_classifier.pth",
    "../models/quick_model.pth"
]

for path in model_paths:
    if Path(path).exists():
        print(f"  ✓ Found: {path}")
    else:
        print(f"  ✗ Not found: {path}")

os.chdir("..") 