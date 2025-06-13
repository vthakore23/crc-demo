#!/usr/bin/env python3
"""
Update model loading in all files to handle PyTorch weights_only setting
"""

import os
import re
from pathlib import Path

def update_file(filepath):
    """Update torch.load calls in a file"""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Pattern to find torch.load calls
    pattern = r'torch\.load\((.*?)\)'
    
    # Check if file has torch.load calls
    if 'torch.load' not in content:
        return False
    
    # Replace torch.load calls to include weights_only=False
    def replace_torch_load(match):
        args = match.group(1)
        # Check if weights_only is already specified
        if 'weights_only' in args:
            return match.group(0)
        
        # Add weights_only=False
        if ',' in args:
            # Has multiple arguments
            return f'torch.load({args}, weights_only=False)'
        else:
            # Single argument
            return f'torch.load({args}, weights_only=False)'
    
    updated_content = re.sub(pattern, replace_torch_load, content)
    
    if updated_content != content:
        with open(filepath, 'w') as f:
            f.write(updated_content)
        return True
    
    return False

def main():
    """Update all Python files in the project"""
    print("Updating model loading code to handle PyTorch security settings...")
    
    # Files to update
    files_to_update = [
        'app/crc_unified_platform.py',
        'app/molecular_subtype_mapper.py',
        'test_snf1_bias.py',
        'diagnose_tissue_classifier.py',
        'test_fixed_model.py',
        'app/epoc_explainable_dashboard.py',
        'app/crc_metastasis_subtype_classifier.py',
        'app/molecular_predictor_v2.py',
        'app/epoc_validation.py',
        'app/run_epoc_validation.py',
        'app/test_molecular_accuracy_current.py'
    ]
    
    updated_count = 0
    for filepath in files_to_update:
        if Path(filepath).exists():
            if update_file(filepath):
                print(f"✓ Updated: {filepath}")
                updated_count += 1
            else:
                print(f"  No changes needed: {filepath}")
        else:
            print(f"  File not found: {filepath}")
    
    print(f"\nUpdated {updated_count} files.")
    
    # Now copy the model again with proper settings
    print("\nCopying balanced model with proper settings...")
    os.system("cp models/balanced_tissue_classifier.pth models/best_tissue_classifier.pth")
    
    print("\nDone! The model loading issue should be fixed.")

if __name__ == "__main__":
    main() 