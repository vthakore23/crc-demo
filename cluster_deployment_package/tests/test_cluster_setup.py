#!/usr/bin/env python3
"""
Cluster Setup Validation Tests
Comprehensive tests to validate the cluster environment before training
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import subprocess
import json
import time
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClusterValidator:
    """Validates cluster setup for CRC molecular subtype training"""
    
    def __init__(self):
        self.results = {
            'environment': {},
            'hardware': {},
            'software': {},
            'distributed': {},
            'data': {},
            'model': {},
            'overall_status': 'UNKNOWN'
        }
    
    def test_environment(self):
        """Test basic environment setup"""
        logger.info("Testing environment setup...")
        
        # Check Python version
        python_version = sys.version_info
        self.results['environment']['python_version'] = f"{python_version.major}.{python_version.minor}.{python_version.micro}"
        self.results['environment']['python_ok'] = python_version >= (3, 11)
        
        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        self.results['environment']['cuda_available'] = cuda_available
        
        if cuda_available:
            self.results['environment']['cuda_version'] = torch.version.cuda
            self.results['environment']['gpu_count'] = torch.cuda.device_count()
            
            # Get GPU information
            gpu_info = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    'name': props.name,
                    'memory_gb': props.total_memory / (1024**3),
                    'compute_capability': f"{props.major}.{props.minor}"
                })
            self.results['environment']['gpu_info'] = gpu_info
        
        # Check required environment variables
        required_vars = ['SLURM_JOB_ID', 'SLURM_PROCID', 'SLURM_NTASKS']
        env_vars = {}
        for var in required_vars:
            env_vars[var] = os.environ.get(var, 'NOT_SET')
        self.results['environment']['slurm_vars'] = env_vars
        
        logger.info(f"Python: {self.results['environment']['python_version']}")
        logger.info(f"CUDA: {cuda_available}")
        if cuda_available:
            logger.info(f"GPUs: {self.results['environment']['gpu_count']}")
    
    def test_hardware(self):
        """Test hardware resources"""
        logger.info("Testing hardware resources...")
        
        # Memory check
        try:
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            
            for line in meminfo.split('\n'):
                if 'MemTotal:' in line:
                    mem_kb = int(line.split()[1])
                    mem_gb = mem_kb / (1024**2)
                    self.results['hardware']['total_memory_gb'] = round(mem_gb, 1)
                    self.results['hardware']['memory_sufficient'] = mem_gb >= 256
                    break
        except:
            self.results['hardware']['total_memory_gb'] = 'UNKNOWN'
            self.results['hardware']['memory_sufficient'] = False
        
        # CPU check
        try:
            cpu_count = os.cpu_count()
            self.results['hardware']['cpu_count'] = cpu_count
            self.results['hardware']['cpu_sufficient'] = cpu_count >= 16
        except:
            self.results['hardware']['cpu_count'] = 'UNKNOWN'
            self.results['hardware']['cpu_sufficient'] = False
        
        # Storage check
        try:
            statvfs = os.statvfs('/')
            free_bytes = statvfs.f_frsize * statvfs.f_bavail
            free_gb = free_bytes / (1024**3)
            self.results['hardware']['storage_free_gb'] = round(free_gb, 1)
            self.results['hardware']['storage_sufficient'] = free_gb >= 1000
        except:
            self.results['hardware']['storage_free_gb'] = 'UNKNOWN'
            self.results['hardware']['storage_sufficient'] = False
        
        logger.info(f"Memory: {self.results['hardware']['total_memory_gb']} GB")
        logger.info(f"CPUs: {self.results['hardware']['cpu_count']}")
        logger.info(f"Storage: {self.results['hardware']['storage_free_gb']} GB free")
    
    def test_software_dependencies(self):
        """Test software dependencies"""
        logger.info("Testing software dependencies...")
        
        # Test PyTorch
        try:
            torch_version = torch.__version__
            self.results['software']['torch_version'] = torch_version
            self.results['software']['torch_ok'] = True
        except:
            self.results['software']['torch_version'] = 'NOT_INSTALLED'
            self.results['software']['torch_ok'] = False
        
        # Test key packages
        packages_to_test = [
            'timm', 'einops', 'openslide', 'h5py', 'pandas', 
            'scikit-image', 'albumentations', 'wandb'
        ]
        
        package_status = {}
        for package in packages_to_test:
            try:
                __import__(package)
                package_status[package] = 'OK'
            except ImportError:
                package_status[package] = 'MISSING'
        
        self.results['software']['packages'] = package_status
        self.results['software']['all_packages_ok'] = all(
            status == 'OK' for status in package_status.values()
        )
        
        # Test SLURM
        try:
            result = subprocess.run(['squeue', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            self.results['software']['slurm_available'] = result.returncode == 0
            if result.returncode == 0:
                self.results['software']['slurm_version'] = result.stdout.strip()
        except:
            self.results['software']['slurm_available'] = False
            self.results['software']['slurm_version'] = 'NOT_AVAILABLE'
        
        logger.info(f"PyTorch: {torch_version}")
        logger.info(f"Missing packages: {[k for k, v in package_status.items() if v == 'MISSING']}")
    
    def test_distributed_setup(self, gpus=None, backend='nccl'):
        """Test distributed training setup"""
        logger.info("Testing distributed training setup...")
        
        if not torch.cuda.is_available():
            self.results['distributed']['status'] = 'CUDA_NOT_AVAILABLE'
            return
        
        gpu_count = torch.cuda.device_count()
        if gpus is None:
            gpus = gpu_count
        
        if gpus > gpu_count:
            self.results['distributed']['status'] = 'INSUFFICIENT_GPUS'
            self.results['distributed']['requested'] = gpus
            self.results['distributed']['available'] = gpu_count
            return
        
        try:
            # Test single GPU
            device = torch.device('cuda:0')
            test_tensor = torch.randn(1000, 1000, device=device)
            result = torch.mm(test_tensor, test_tensor.t())
            self.results['distributed']['single_gpu_test'] = 'PASS'
            
            # Test multi-GPU if available
            if gpu_count > 1:
                test_tensor_multi = torch.randn(1000, 1000)
                if torch.cuda.device_count() >= 2:
                    test_tensor_multi = test_tensor_multi.cuda(1)
                    self.results['distributed']['multi_gpu_test'] = 'PASS'
                else:
                    self.results['distributed']['multi_gpu_test'] = 'SKIP'
            
            self.results['distributed']['status'] = 'READY'
            self.results['distributed']['backend'] = backend
            self.results['distributed']['gpus_tested'] = min(gpus, gpu_count)
            
        except Exception as e:
            self.results['distributed']['status'] = 'FAILED'
            self.results['distributed']['error'] = str(e)
        
        logger.info(f"Distributed status: {self.results['distributed']['status']}")
    
    def test_data_access(self, data_dir='/data/epoc'):
        """Test data directory access"""
        logger.info("Testing data access...")
        
        data_path = Path(data_dir)
        
        # Check if data directory exists
        self.results['data']['directory_exists'] = data_path.exists()
        
        if data_path.exists():
            # Check read access
            self.results['data']['readable'] = os.access(data_path, os.R_OK)
            
            # Check write access
            try:
                test_file = data_path / 'test_write.tmp'
                test_file.write_text('test')
                test_file.unlink()
                self.results['data']['writable'] = True
            except:
                self.results['data']['writable'] = False
            
            # Check for WSI files
            wsi_extensions = ['.svs', '.ndpi', '.mrxs']
            wsi_count = 0
            for ext in wsi_extensions:
                wsi_count += len(list(data_path.rglob(f'*{ext}')))
            
            self.results['data']['wsi_files_found'] = wsi_count
            self.results['data']['has_wsi_data'] = wsi_count > 0
        else:
            self.results['data']['readable'] = False
            self.results['data']['writable'] = False
            self.results['data']['wsi_files_found'] = 0
            self.results['data']['has_wsi_data'] = False
        
        logger.info(f"Data directory: {data_path} (exists: {self.results['data']['directory_exists']})")
        logger.info(f"WSI files found: {self.results['data']['wsi_files_found']}")
    
    def test_model_loading(self):
        """Test model architecture loading"""
        logger.info("Testing model loading...")
        
        try:
            # Add current directory to path for imports
            sys.path.insert(0, str(Path.cwd()))
            
            from cluster.models.cluster_ready_model import ClusterReadyMolecularModel
            
            # Test model configuration
            config = {
                'backbone': 'efficientnet_b3',
                'feature_dim': 512,
                'local_aggregation': 'attention',
                'global_aggregation': 'transformer',
                'num_heads': 8,
                'depth': 4,
                'pathway_indices': {'canonical': 0, 'immune': 1, 'stromal': 2}
            }
            
            # Create model
            model = ClusterReadyMolecularModel(config)
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            self.results['model']['loading_status'] = 'SUCCESS'
            self.results['model']['total_parameters'] = total_params
            self.results['model']['trainable_parameters'] = trainable_params
            self.results['model']['model_size_mb'] = total_params * 4 / (1024**2)  # Assuming float32
            
            # Test forward pass
            if torch.cuda.is_available():
                model = model.cuda()
                dummy_input = {
                    'level_0': torch.randn(1, 10, 3, 256, 256).cuda()
                }
                
                with torch.no_grad():
                    output = model(dummy_input)
                
                self.results['model']['forward_pass'] = 'SUCCESS'
                self.results['model']['output_keys'] = list(output.keys())
            else:
                self.results['model']['forward_pass'] = 'SKIPPED_NO_CUDA'
            
        except Exception as e:
            self.results['model']['loading_status'] = 'FAILED'
            self.results['model']['error'] = str(e)
            self.results['model']['forward_pass'] = 'FAILED'
        
        logger.info(f"Model loading: {self.results['model']['loading_status']}")
        if self.results['model']['loading_status'] == 'SUCCESS':
            logger.info(f"Model parameters: {self.results['model']['total_parameters']:,}")
    
    def run_all_tests(self, gpus=None, data_dir='/data/epoc'):
        """Run all validation tests"""
        logger.info("Starting cluster validation tests...")
        
        self.test_environment()
        self.test_hardware()
        self.test_software_dependencies()
        self.test_distributed_setup(gpus)
        self.test_data_access(data_dir)
        self.test_model_loading()
        
        # Determine overall status
        critical_checks = [
            self.results['environment']['python_ok'],
            self.results['environment']['cuda_available'],
            self.results['hardware']['memory_sufficient'],
            self.results['software']['torch_ok'],
            self.results['distributed']['status'] == 'READY',
            self.results['model']['loading_status'] == 'SUCCESS'
        ]
        
        if all(critical_checks):
            self.results['overall_status'] = 'READY'
        elif any(critical_checks):
            self.results['overall_status'] = 'PARTIAL'
        else:
            self.results['overall_status'] = 'FAILED'
        
        logger.info(f"Overall status: {self.results['overall_status']}")
        return self.results
    
    def generate_report(self, output_file=None):
        """Generate validation report"""
        report = f"""
# Cluster Validation Report

## Overall Status: {self.results['overall_status']}

## Environment
- Python Version: {self.results['environment']['python_version']} ({'✅' if self.results['environment']['python_ok'] else '❌'})
- CUDA Available: {self.results['environment']['cuda_available']} ({'✅' if self.results['environment']['cuda_available'] else '❌'})
- GPU Count: {self.results['environment'].get('gpu_count', 'N/A')}

## Hardware
- Memory: {self.results['hardware']['total_memory_gb']} GB ({'✅' if self.results['hardware']['memory_sufficient'] else '❌'})
- CPUs: {self.results['hardware']['cpu_count']} ({'✅' if self.results['hardware']['cpu_sufficient'] else '❌'})
- Storage: {self.results['hardware']['storage_free_gb']} GB free ({'✅' if self.results['hardware']['storage_sufficient'] else '❌'})

## Software
- PyTorch: {self.results['software']['torch_version']} ({'✅' if self.results['software']['torch_ok'] else '❌'})
- All Packages: {'✅' if self.results['software']['all_packages_ok'] else '❌'}
- SLURM: {'✅' if self.results['software']['slurm_available'] else '❌'}

## Distributed Training
- Status: {self.results['distributed']['status']}
- Single GPU Test: {self.results['distributed'].get('single_gpu_test', 'N/A')}
- Multi GPU Test: {self.results['distributed'].get('multi_gpu_test', 'N/A')}

## Data Access
- Directory Exists: {'✅' if self.results['data']['directory_exists'] else '❌'}
- Readable: {'✅' if self.results['data']['readable'] else '❌'}
- Writable: {'✅' if self.results['data']['writable'] else '❌'}
- WSI Files Found: {self.results['data']['wsi_files_found']}

## Model
- Loading: {'✅' if self.results['model']['loading_status'] == 'SUCCESS' else '❌'}
- Parameters: {self.results['model'].get('total_parameters', 'N/A'):,}
- Forward Pass: {'✅' if self.results['model'].get('forward_pass') == 'SUCCESS' else '❌'}

## Recommendations
"""
        
        # Add recommendations based on results
        if self.results['overall_status'] == 'READY':
            report += "- ✅ System is ready for training!\n"
        else:
            if not self.results['environment']['cuda_available']:
                report += "- ❌ CUDA not available - check GPU drivers\n"
            if not self.results['hardware']['memory_sufficient']:
                report += "- ❌ Insufficient memory - need at least 256GB\n"
            if not self.results['software']['all_packages_ok']:
                report += "- ❌ Missing packages - run pip install -r requirements_cluster.txt\n"
            if not self.results['data']['directory_exists']:
                report += "- ❌ Data directory not found - check data paths\n"
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to: {output_file}")
        
        print(report)
        return report


def main():
    parser = argparse.ArgumentParser(description='Validate cluster setup for CRC training')
    parser.add_argument('--gpus', type=int, help='Number of GPUs to test')
    parser.add_argument('--data_dir', default='/data/epoc', help='Data directory to test')
    parser.add_argument('--output_report', help='Output report file')
    parser.add_argument('--json_output', help='Output results as JSON')
    
    args = parser.parse_args()
    
    validator = ClusterValidator()
    results = validator.run_all_tests(gpus=args.gpus, data_dir=args.data_dir)
    
    # Generate report
    validator.generate_report(args.output_report)
    
    # Save JSON results
    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(results, f, indent=2)
    
    # Exit with appropriate code
    if results['overall_status'] == 'READY':
        return 0
    elif results['overall_status'] == 'PARTIAL':
        return 1
    else:
        return 2


if __name__ == "__main__":
    exit(main()) 