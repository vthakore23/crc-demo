#!/usr/bin/env python3
"""Fix quick_model.pth format to be a direct state dict"""

import torch

# Load the balanced model
print("Loading balanced tissue classifier...")
checkpoint = torch.load('models/balanced_tissue_classifier.pth', map_location='cpu', weights_only=False)

# Extract just the state dict
state_dict = checkpoint['model_state_dict']

# Save as direct state dict (like the original quick_model format)
print("Saving state dict in quick_model format...")
torch.save(state_dict, 'models/quick_model.pth')

print("Done! quick_model.pth now contains the balanced tissue classifier in the correct format.")

# Verify it loads correctly
print("\nVerifying the saved model...")
loaded_state = torch.load('models/quick_model.pth', map_location='cpu', weights_only=False)
print(f"Loaded state dict with {len(loaded_state)} keys")
print(f"First few keys: {list(loaded_state.keys())[:5]}") 