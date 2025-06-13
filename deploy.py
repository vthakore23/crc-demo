#!/usr/bin/env python3
"""
CRC Analysis Platform - Smart Deployment Script
Automatically detects environment and launches the appropriate configuration
"""

import sys
import subprocess
import os
from pathlib import Path

def check_pyradiomics():
    """Check if PyRadiomics is available"""
    try:
        import radiomics
        return True, radiomics.__version__
    except ImportError:
        return False, None

def check_environment():
    """Check current Python environment"""
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    env_name = os.environ.get('CONDA_DEFAULT_ENV', 'unknown')
    return python_version, env_name

def run_streamlit(port=8501):
    """Launch Streamlit app"""
    try:
        cmd = ["streamlit", "run", "app.py", "--server.port", str(port)]
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
    except Exception as e:
        print(f"❌ Error running Streamlit: {e}")

def main():
    print("🚀 CRC Analysis Platform - Smart Deployment")
    print("=" * 50)
    
    # Check environment
    python_version, env_name = check_environment()
    pyradiomics_available, pyradiomics_version = check_pyradiomics()
    
    print(f"🐍 Python Version: {python_version}")
    print(f"🏠 Environment: {env_name}")
    
    if pyradiomics_available:
        print(f"✅ PyRadiomics: v{pyradiomics_version} (Full hybrid functionality)")
        print("🧬 Features: 33,440+ (651 radiomic + 32,776 deep learning)")
        deployment_type = "Full Hybrid"
    else:
        print("⚠️  PyRadiomics: Not available (Standard functionality)")
        print("🧬 Features: 32,790 (deep learning + spatial patterns)")
        deployment_type = "Standard"
    
    print("\n" + "=" * 50)
    print(f"🎯 Deployment Type: {deployment_type}")
    print("=" * 50)
    
    if not pyradiomics_available:
        print("\n💡 To enable full PyRadiomics functionality:")
        print("   conda create -n pyradiomics python=3.11 -y")
        print("   conda activate pyradiomics") 
        print("   pip install -r requirements_local.txt")
        print("   python deploy.py")
        
        print("\n📋 Requirements file guide:")
        print("   • requirements.txt         → Cloud-compatible (current)")
        print("   • requirements_local.txt   → Full features with PyRadiomics")
        print("   • requirements_cloud.txt   → Legacy cloud file")
        
        response = input(f"\n🤔 Continue with standard deployment? (y/n): ")
        if response.lower() not in ['y', 'yes']:
            print("👋 Deployment cancelled")
            return
    else:
        print(f"\n✅ Hybrid features enabled! You have the full PyRadiomics installation.")
    
    # Choose port
    try:
        port = int(input(f"\n🌐 Port (default 8501): ") or "8501")
    except ValueError:
        port = 8501
    
    print(f"\n🚀 Launching CRC Analysis Platform...")
    print(f"📍 URL: http://localhost:{port}")
    print("🔄 Loading models and initializing interface...")
    print("\n" + "=" * 50)
    
    # Launch app
    run_streamlit(port)

if __name__ == "__main__":
    main() 