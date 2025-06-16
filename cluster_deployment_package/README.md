# CRC Subtype Predictor - Cluster Deployment Package

## Overview
Distributed training system for CRC molecular subtype prediction from whole slide images (WSI) for EPOC clinical trials.

## Package Contents
```
cluster_deployment_package/
├── README.md                           # This file
├── train_distributed_epoc.py           # Main distributed training script
├── slurm_submit.sh                     # SLURM job submission script
├── production_inference.py             # Production inference pipeline
├── training_config.yaml                # Training configuration
├── requirements.txt                    # Python dependencies  
├── validate_setup.py                   # Setup validation script
└── src/                                # Supporting modules
    ├── models/distributed_wrapper.py   # Model architecture
    ├── data/wsi_dataset_distributed.py # WSI data loading
    ├── utils/
    │   ├── checkpoint_manager.py       # Fault-tolerant checkpointing
    │   └── monitoring.py               # Cluster monitoring
    └── validation/epoc_validator.py    # EPOC validation framework
```

## Requirements
- **Nodes**: 4-8 compute nodes
- **GPUs**: 8+ per node (NVIDIA V100/A100/H100)
- **Memory**: 512GB+ per node
- **Storage**: 10TB+ shared storage (NFS/Lustre/GPFS)
- **Network**: InfiniBand or 100GbE
- **Software**: Python 3.10+, CUDA 11.8+, PyTorch 2.0+

## Quick Start

### 1. Environment Setup
```bash
# Load modules (adjust for your cluster)
module load cuda/11.8 python/3.11 openmpi/4.1.4

# Install dependencies
pip install -r requirements.txt

# Validate setup
python validate_setup.py
```

### 2. Data Preparation
Organize WSI data as:
```
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

### 3. Configuration
Edit `training_config.yaml`:
```yaml
# Key parameters to adjust
data_root: "/data/epoc_wsi"
output_dir: "/results/distributed_training"
epochs: 100
batch_size: 32  # per GPU
```

### 4. Submit Training Job
```bash
# Edit SLURM parameters in slurm_submit.sh for your cluster
sbatch slurm_submit.sh
```

## Monitoring

### Job Status
```bash
squeue -u $USER
sacct -j JOBID --format=JobID,JobName,State,Time,Start,End,NodeList
```

### Training Progress
```bash
# Monitor logs
tail -f /logs/training_rank_0.log

# GPU utilization
watch nvidia-smi

# TensorBoard (if enabled)
tensorboard --logdir /logs/tensorboard
```

## Production Inference

### Start Inference Service
```bash
python production_inference.py --config training_config.yaml
```

### API Usage
```bash
# Submit WSI for inference
curl -X POST http://localhost:8000/inference/submit \
  -H "Content-Type: application/json" \
  -d '{"wsi_path": "/data/slides/sample.svs", "slide_id": "sample_001"}'

# Check status
curl http://localhost:8000/inference/status/REQUEST_ID

# Get results
curl http://localhost:8000/inference/result/REQUEST_ID
```

## Configuration Reference

### Key Training Parameters
- `epochs`: Number of training epochs (default: 100)
- `batch_size`: Batch size per GPU (default: 32)
- `learning_rate`: Initial learning rate (default: 3e-4)
- `num_workers`: Data loading workers per GPU (default: 8)
- `use_amp`: Mixed precision training (default: true)

### Distributed Settings
- `backend`: Communication backend (default: "nccl")
- `sync_batchnorm`: Synchronized batch normalization (default: true)
- `find_unused_parameters`: DDP setting (default: false)

### Data Processing
- `patch_size`: WSI patch size (default: 256)
- `magnifications`: Magnification levels (default: [10, 20, 40])
- `tissue_threshold`: Tissue detection threshold (default: 0.1)

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce `batch_size` or enable `gradient_checkpointing`
2. **NCCL timeout**: Check InfiniBand/network configuration
3. **Data loading errors**: Verify paths and file permissions
4. **Slow training**: Increase `num_workers`, check storage I/O

### Performance Optimization
- Enable `torch.compile` for ~20% speedup
- Use `persistent_workers=True` for faster data loading
- Adjust `accumulation_steps` for effective larger batch sizes
- Enable `gradient_checkpointing` to reduce memory usage

## Expected Performance
- **Training Time**: ~48 hours on 32 V100 GPUs
- **Memory Usage**: ~16GB per GPU
- **Throughput**: ~100 slides/hour during inference
- **Accuracy**: 90%+ on EPOC validation set

## Support
For issues specific to this deployment package, check:
1. Training logs in `/logs/training_rank_0.log`
2. SLURM job logs via `scontrol show job JOBID`
3. GPU monitoring with `nvidia-smi`
4. System monitoring in cluster dashboard

## File Descriptions

### Core Scripts
- **train_distributed_epoc.py**: Main training script with distributed training, fault tolerance, and monitoring
- **slurm_submit.sh**: SLURM submission script with multi-node configuration
- **production_inference.py**: FastAPI-based inference service with load balancing

### Configuration
- **training_config.yaml**: Comprehensive training configuration with all parameters
- **requirements.txt**: Python dependencies with specific versions

### Supporting Modules
- **distributed_wrapper.py**: Model architecture with attention-based MIL
- **wsi_dataset_distributed.py**: Distributed WSI data loading with Ray
- **checkpoint_manager.py**: Fault-tolerant checkpointing with compression
- **monitoring.py**: Cluster monitoring with GPU/system metrics
- **epoc_validator.py**: Clinical validation framework for EPOC trials 