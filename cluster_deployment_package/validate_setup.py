#!/usr/bin/env python3
"""
Setup Validation Script
Run this script to verify that all dependencies and environment setup is correct
before starting training.
"""

import sys
import os
import subprocess
from pathlib import Path

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("❌ Python 3.10+ required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    return True

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.device_count()} GPUs")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
            return True
        else:
            print("❌ CUDA not available")
            return False
    except ImportError:
        print("❌ PyTorch not installed")
        return False

def check_distributed():
    """Check distributed training setup"""
    try:
        import torch.distributed as dist
        print("✅ Distributed training support available")
        return True
    except ImportError:
        print("❌ Distributed training not available")
        return False

def check_dependencies():
    """Check required dependencies"""
    required_packages = [
        'torch', 'torchvision', 'numpy', 'pandas', 'opencv-python',
        'scikit-learn', 'scipy', 'matplotlib', 'seaborn', 'tqdm',
        'wandb', 'tensorboard', 'ray', 'fastapi', 'uvicorn',
        'pyyaml', 'h5py', 'lmdb', 'openslide-python'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - missing")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def check_file_structure():
    """Check that all required files are present"""
    required_files = [
        'train_distributed_epoc.py',
        'slurm_submit.sh',
        'production_inference.py',
        'training_config.yaml',
        'requirements.txt',
        'src/models/distributed_wrapper.py',
        'src/data/wsi_dataset_distributed.py',
        'src/utils/checkpoint_manager.py',
        'src/utils/monitoring.py',
        'src/validation/epoc_validator.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path} - missing")
            missing_files.append(file_path)
    
    return len(missing_files) == 0

def check_data_paths():
    """Check if data paths in config are accessible"""
    try:
        import yaml
        with open('training_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        data_root = config.get('data_root', '/data/epoc_wsi')
        if Path(data_root).exists():
            print(f"✅ Data root accessible: {data_root}")
            return True
        else:
            print(f"⚠️  Data root not found: {data_root}")
            print("   Update training_config.yaml with correct data_root path")
            return False
    except Exception as e:
        print(f"❌ Error checking data paths: {e}")
        return False

def check_slurm():
    """Check if SLURM is available"""
    try:
        subprocess.run(['sbatch', '--version'], capture_output=True, check=True)
        print("✅ SLURM available")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  SLURM not available (may not be needed for all clusters)")
        return False

def main():
    """Run all validation checks"""
    print("🔍 Validating CRC Subtype Predictor Setup...")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("CUDA Support", check_cuda),
        ("Distributed Training", check_distributed),
        ("Dependencies", check_dependencies),
        ("File Structure", check_file_structure),
        ("Data Paths", check_data_paths),
        ("SLURM", check_slurm)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        if check_func():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Validation Summary: {passed}/{total} checks passed")
    
    if passed == total:
        print("🎉 All checks passed! Ready for training.")
        return 0
    elif passed >= total - 1:
        print("⚠️  Setup mostly ready. Check warnings above.")
        return 0
    else:
        print("❌ Setup incomplete. Fix issues above before training.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 