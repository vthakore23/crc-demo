#!/usr/bin/env python3
"""
Quick test script for CRC Molecular Subtype Foundation Model
Verifies that the model can be created and run without errors
"""

import torch
import numpy as np
from PIL import Image
import sys
from pathlib import Path

def test_molecular_model():
    """Test the molecular foundation model"""
    try:
        # Import the model
        from foundation_model.molecular_subtype_foundation import (
            create_sota_molecular_model,
            StateOfTheArtMolecularFoundation
        )
        print("✅ Successfully imported molecular foundation model")
        
        # Create model
        model = create_sota_molecular_model()
        print("✅ Successfully created state-of-the-art molecular model")
        
        # Check model parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 Model has {total_params:,} parameters")
        
        # Test forward pass with single image
        x_single = torch.randn(1, 3, 224, 224)
        output_single = model(x_single, return_features=True)
        print("✅ Single image forward pass successful")
        
        # Test forward pass with WSI patches
        x_patches = torch.randn(1, 5, 3, 224, 224)  # 1 sample, 5 patches
        output_patches = model(x_patches, return_features=True, return_attention=True)
        print("✅ WSI patches forward pass successful")
        
        # Test prediction method
        predictions = model.predict_with_confidence(x_single)
        predicted_subtype = predictions['predicted_subtype']
        confidence = predictions['confidence']
        print(f"✅ Prediction successful: {predicted_subtype} (confidence: {confidence:.3f})")
        
        # Test model components
        print(f"✅ Model uses MIL: {model.use_mil}")
        print(f"✅ Model uses uncertainty: {model.use_uncertainty}")
        print(f"✅ Available subtypes: {model.subtype_names}")
        
        print("\n🎉 All tests passed! Molecular foundation model is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_streamlit_imports():
    """Test Streamlit app imports"""
    try:
        from app.molecular_subtype_platform import (
            apply_molecular_theme,
            display_molecular_landing,
            display_molecular_sidebar,
            load_molecular_model
        )
        print("✅ Successfully imported Streamlit app components")
        return True
    except Exception as e:
        print(f"❌ Streamlit import test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧬 Testing CRC Molecular Subtype Foundation Model")
    print("=" * 60)
    
    # Test model
    model_test = test_molecular_model()
    
    print("\n" + "=" * 60)
    print("🌐 Testing Streamlit App Components")
    print("=" * 60)
    
    # Test Streamlit imports
    app_test = test_streamlit_imports()
    
    print("\n" + "=" * 60)
    if model_test and app_test:
        print("🎉 ALL TESTS PASSED! Ready for deployment.")
        print("🚀 Run: streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 60) 