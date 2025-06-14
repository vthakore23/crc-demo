# System Setup - Randi Cluster Configuration

## Prerequisites

### Account Requirements
- Active UChicago CRI account with Randi cluster access
- Allocation on appropriate partition (GPU nodes)
- SSH key configured for cluster access

### Verify Cluster Access
```bash
ssh username@randi.cri.uchicago.edu
```

## Environment Setup

### 1. Load Required Modules

The Randi cluster uses environment modules. Load the following:

```bash
module load cuda/11.8
module load python/3.11
module load gcc/11.2.0
module load openmpi/4.1.4
```

### 2. Create Project Directory

```bash
# Navigate to your scratch space (recommended for large datasets)
cd /scratch/username

# Create project directory
mkdir crc_molecular_training
cd crc_molecular_training
```

### 3. Python Environment Setup

```bash
# Create virtual environment
python -m venv crc_env

# Activate environment
source crc_env/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 4. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install additional requirements
pip install -r requirements.txt
```

## Storage Configuration

### Recommended Directory Structure

```
/scratch/username/crc_molecular_training/
├── data/                    # EPOC WSI data (10TB+)
│   ├── raw/                # Original WSI files
│   ├── processed/          # Preprocessed patches
│   └── manifests/          # Data split files
├── models/                 # Model checkpoints
├── logs/                   # Training logs
├── results/                # Output files
└── scripts/                # Training scripts
```

### Storage Quotas

- **Home Directory**: 50GB (not suitable for training data)
- **Scratch Space**: 100TB (use for training data and checkpoints)
- **Project Space**: Contact CRI for allocation if needed

## GPU Node Specifications

### Available GPU Nodes on Randi
- **5 GPU nodes**: 8x NVIDIA A100 (40GB) per node
- **1 SXM node**: 8x NVIDIA A100 (80GB) connected via NVSwitch
- **Total GPU Memory**: 320GB (40GB nodes) or 640GB (80GB node)

### Recommended Configuration
- **Training**: Use 4-8 A100 GPUs across 1-2 nodes
- **Memory**: 512GB RAM per node minimum
- **Storage**: NVMe scratch space for data pipeline

## Network Configuration

### InfiniBand Setup
Randi uses InfiniBand HDR100 (100 Gbps) for inter-node communication:

```bash
# Verify InfiniBand status
ibstat

# Test bandwidth (if needed)
ib_write_bw
```

## SLURM Configuration

### Partition Information
```bash
# View available partitions
sinfo

# Check GPU availability
sinfo -p gpu --format="%.15N %.6D %.6t %.15C %.8z %.6m %.8d %.6w %.8f %20E"
```

### Resource Limits
- **Max Job Time**: 7 days
- **Max GPUs per Job**: 8 GPUs
- **Max Nodes per Job**: 2 nodes (for multi-node training)

## Validation Tests

### 1. CUDA Availability
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### 2. GPU Memory Test
```bash
nvidia-smi
```

### 3. InfiniBand Test
```bash
# Test inter-node communication (if using multiple nodes)
mpirun -np 2 --hostfile hostfile python -c "import torch.distributed as dist; dist.init_process_group('nccl')"
```

## Common Module Commands

```bash
# List available modules
module avail

# Show loaded modules
module list

# Unload all modules
module purge

# Save module configuration
module save crc_training

# Load saved configuration
module restore crc_training
```

## Environment Variables

Add to your `~/.bashrc` or job script:

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO
export PYTHONPATH=/scratch/username/crc_molecular_training:$PYTHONPATH
```

## Troubleshooting

### Module Loading Issues
```bash
# If modules fail to load
module purge
module load cuda/11.8 python/3.11 gcc/11.2.0
```

### CUDA Issues
```bash
# Check CUDA installation
nvcc --version
which nvcc
```

### Storage Issues
```bash
# Check disk usage
df -h /scratch/username
quota -u username
```

## Next Steps

1. Verify all components are working with `scripts/validate_setup.sh`
2. Proceed to `TRAINING_GUIDE.md` for training procedures
3. Configure data paths in `config/randi_training_config.yaml`

---

**Support**: For Randi-specific issues, contact help@rcc.uchicago.edu 