# CRC Subtype Predictor - Cluster Deployment Guide

## Overview
Distributed training system for CRC molecular subtype prediction from WSI data for EPOC clinical trials.

## Requirements
- PyTorch 2.0+, CUDA 11.8+
- Multi-node cluster with InfiniBand/high-speed interconnect
- Minimum 4 nodes × 8 GPUs (32 GPUs total)
- 10TB+ shared storage for WSI data

## Quick Start

### 1. Environment Setup
```bash
# Load modules (adapt to your cluster)
module load cuda/11.8 python/3.11 gcc/11.2 openmpi/4.1.4

# Install dependencies
pip install -r requirements_cluster.txt
```

### 2. Data Preparation
```bash
# WSI data structure
/data/epoc_wsi/
├── train/
│   ├── canonical/
│   ├── immune/
│   └── stromal/
└── val/
    ├── canonical/
    ├── immune/
    └── stromal/
```

### 3. Submit Training Job
```bash
# Edit slurm_submit.sh for your cluster configuration
sbatch slurm_submit.sh
```

## Configuration

### SLURM Job Parameters
```bash
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --mem=500G
#SBATCH --time=72:00:00
```

### Training Configuration
Edit `config/epoc_training_config.yaml`:
```yaml
# Essential parameters
epochs: 100
batch_size: 32
  learning_rate: 3e-4
data_root: "/data/epoc_wsi"
output_dir: "/results/distributed_training"
  
# Distributed settings
  backend: "nccl"
use_amp: true
  sync_batchnorm: true
```

## Key Files
- `train_distributed_epoc.py` - Main training script
- `slurm_submit.sh` - SLURM submission script
- `production_inference.py` - Production inference pipeline
- `config/epoc_training_config.yaml` - Training configuration

## Monitoring
```bash
# Check job status
squeue -u $USER

# Monitor training logs
tail -f /logs/training_rank_0.log

# Check GPU utilization
watch nvidia-smi
```

## Production Inference
```bash
# Start inference service
python production_inference.py --config config/epoc_training_config.yaml
```

## Troubleshooting

### Common Issues
- **OOM errors**: Reduce batch_size or enable gradient_checkpointing
- **NCCL timeout**: Check InfiniBand configuration
- **WSI loading errors**: Verify data paths and file permissions

### Performance Optimization
- Enable `torch.compile` for GPU efficiency
- Use `persistent_workers=True` for data loading
- Adjust `accumulation_steps` based on available memory

## Contact
For technical issues, contact the cluster support team. 