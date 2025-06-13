#!/usr/bin/env python3
"""
CRC Molecular Subtype Predictor - Launch Script
State-of-the-art molecular subtype prediction for oligometastatic CRC
"""

import sys
import os
import subprocess
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import torch
        import streamlit
        import timm
        logger.info("✅ Core dependencies found")
        return True
    except ImportError as e:
        logger.error(f"❌ Missing dependency: {e}")
        logger.info("Please install dependencies: pip install -r requirements.txt")
        return False

def check_model_files():
    """Check if model files exist"""
    model_dir = Path("models")
    foundation_model = Path("foundation_model")
    
    if not foundation_model.exists():
        logger.error("❌ foundation_model directory not found")
        return False
    
    if not model_dir.exists():
        logger.info("📁 Creating models directory")
        model_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("✅ Model structure validated")
    return True

def setup_environment():
    """Setup the environment for running the molecular predictor"""
    logger.info("🚀 Setting up CRC Molecular Subtype Predictor...")
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Check model files
    if not check_model_files():
        return False
    
    # Set environment variables
    os.environ['STREAMLIT_SERVER_PORT'] = '8501'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    logger.info("✅ Environment setup complete")
    return True

def run_streamlit_app():
    """Run the Streamlit application"""
    try:
        logger.info("🌟 Launching CRC Molecular Subtype Predictor...")
        logger.info("🌐 Application will be available at: http://localhost:8501")
        
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false",
            "--theme.base", "dark"
        ])
        
    except KeyboardInterrupt:
        logger.info("🛑 Application stopped by user")
    except Exception as e:
        logger.error(f"❌ Error running application: {e}")

def main():
    """Main function"""
    print("=" * 60)
    print("🧬 CRC MOLECULAR SUBTYPE PREDICTOR")
    print("State-of-the-art AI for oligometastatic CRC assessment")
    print("=" * 60)
    
    # Setup environment
    if not setup_environment():
        logger.error("❌ Environment setup failed")
        sys.exit(1)
    
    # Display system info
    logger.info("📊 System Information:")
    try:
        import torch
        logger.info(f"   PyTorch: {torch.__version__}")
        logger.info(f"   CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"   CUDA Device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        logger.warning("   PyTorch not installed")
    
    try:
        import streamlit
        logger.info(f"   Streamlit: {streamlit.__version__}")
    except ImportError:
        logger.warning("   Streamlit not installed")
    
    print("\n" + "=" * 60)
    print("🎯 MOLECULAR SUBTYPES:")
    print("   🎯 Canonical - E2F/MYC activation (37% 10-yr survival)")
    print("   🛡️ Immune - MSI-independent activation (64% 10-yr survival)")  
    print("   🌊 Stromal - EMT/angiogenesis (20% 10-yr survival)")
    print("=" * 60)
    
    # Run application
    run_streamlit_app()

if __name__ == "__main__":
    main() 