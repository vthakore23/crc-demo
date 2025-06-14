# System Setup - UChicago RCC Configuration

## Prerequisites

### Account Requirements
- Active UChicago RCC account with cluster access
- Allocation on appropriate partition (CPU or GPU)
- SSH key configured for cluster access

### Verify Cluster Access

**For Midway2 (Primary HPC Cluster):**
```bash
ssh username@midway2.rcc.uchicago.edu
```

**For Randi (GPU Cluster):**
```bash
ssh username@randi.cri.uchicago.edu
```

## RCC Infrastructure Overview

Based on the [UChicago RCC resources](https://rcc.uchicago.edu/resources/high-performance-computing), the available systems include:

### Midway2 Cluster
- **Total Resources**: 16,016+ cores across 572+ nodes
- **CPU Types**: Intel Broadwell (28 cores @ 2.4 GHz) and Skylake (40 cores @ 2.4 GHz)
- **Memory**: 64-96 GB per standard node, up to 1TB on large memory nodes
- **Network**: InfiniBand FDR/EDR (up to 100Gbps) and 40Gbps GigE
- **Storage**: 2.2+ PB total storage infrastructure

### GPU Resources
- **Midway2 GPU Nodes**: NVIDIA K80 accelerator cards (4 per node)
- **Randi GPU Nodes**: NVIDIA A100 GPUs (if available)
- **Integration**: Fully integrated with InfiniBand network

## Environment Setup

### 1. Load Required Modules

The RCC uses environment modules. Common modules include:

```bash
# Check available modules
module avail

# Load common modules for deep learning
module load cuda/11.8
module load python/3.11
module load gcc/11.2.0
module load openmpi/4.1.4
```

### 2. Create Project Directory

**For Midway2:**
```bash
# Navigate to your scratch space
cd /scratch/midway2/username

# Create project directory
mkdir crc_molecular_training
cd crc_molecular_training
```

**For Randi:**
```bash
# Navigate to your scratch space
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
# Install PyTorch with CUDA support (if using GPU nodes)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install additional requirements
pip install -r requirements.txt
```

## Storage Configuration

### RCC Storage Options

Based on RCC infrastructure:
- **Home Directory**: Limited space (typically 50GB)
- **Scratch Space**: High-performance temporary storage
- **Project Storage**: Long-term storage (contact RCC for allocation)

### Recommended Directory Structure

```
/scratch/[cluster]/username/crc_molecular_training/
├── data/                    # EPOC WSI data (10TB+)
│   ├── raw/                # Original WSI files
│   ├── processed/          # Preprocessed patches
│   └── manifests/          # Data split files
├── models/                 # Model checkpoints
├── logs/                   # Training logs
├── results/                # Output files
└── scripts/                # Training scripts
```

## Hardware Specifications

### Midway2 Specifications
- **CPU Nodes**: Intel Broadwell (28 cores) or Skylake (40 cores)
- **Memory**: 64-96 GB standard, up to 1TB on large memory nodes
- **GPU Nodes**: NVIDIA K80 accelerators (4 per node)
- **Network**: InfiniBand FDR/EDR or 40Gbps GigE

### Randi Specifications (if available)
- **GPU Nodes**: NVIDIA A100 GPUs
- **Memory**: High-memory configurations available
- **Network**: InfiniBand for high-performance communication

### Recommended Configuration
- **For CPU Training**: Midway2 Skylake nodes (40 cores, 96GB RAM)
- **For GPU Training**: GPU nodes with K80 or A100 accelerators
- **For Data Processing**: Large memory nodes (up to 1TB RAM)

## Network Configuration

### InfiniBand Setup
RCC uses InfiniBand for high-performance interconnect:

```bash
# Verify InfiniBand status (if available)
ibstat

# Test bandwidth (if needed)
ib_write_bw
```

## SLURM Configuration

### Check Available Partitions
```bash
# View available partitions
sinfo

# Check specific partition details
sinfo -p gpu --format="%.15N %.6D %.6t %.15C %.8z %.6m %.8d %.6w %.8f %20E"
sinfo -p broadwl --format="%.15N %.6D %.6t %.15C %.8z %.6m %.8d %.6w %.8f %20E"
```

### Common Partitions
- **broadwl**: Intel Broadwell nodes (28 cores)
- **skylake**: Intel Skylake nodes (40 cores)
- **gpu**: GPU-enabled nodes
- **bigmem**: Large memory nodes (up to 1TB)

## Validation Tests

### 1. System Information
```bash
# Check CPU information
lscpu

# Check memory
free -h

# Check available modules
module avail | grep -E "(cuda|python|gcc)"
```

### 2. CUDA Availability (GPU nodes)
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"
```

### 3. GPU Information (if available)
```bash
nvidia-smi
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
# For GPU training
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8
export NCCL_DEBUG=INFO

# Project path
export PYTHONPATH=/scratch/[cluster]/username/crc_molecular_training:$PYTHONPATH
```

## Troubleshooting

### Module Loading Issues
```bash
# If modules fail to load
module purge
module load cuda/11.8 python/3.11 gcc/11.2.0
```

### CUDA Issues (GPU nodes)
```bash
# Check CUDA installation
nvcc --version
which nvcc
```

### Storage Issues
```bash
# Check disk usage
df -h /scratch/[cluster]/username
quota -u username
```

### Network Connectivity
```bash
# Test cluster connectivity
ping midway2.rcc.uchicago.edu
ping randi.cri.uchicago.edu
```

## Getting Help

### RCC Support Resources
- **Email**: help@rcc.uchicago.edu
- **Phone**: (773) 795-2667
- **Walk-in**: Regenstein Library, Suite 216
- **User Guide**: Available on RCC website

### Account and Allocation
- **Request Account**: Visit RCC website
- **Storage Allocation**: Contact RCC for additional storage
- **Compute Allocation**: Review allocation policies on RCC website

## Next Steps

1. Verify all components are working with `scripts/validate_setup.sh`
2. Choose appropriate cluster (Midway2 vs Randi) based on requirements
3. Proceed to `TRAINING_GUIDE.md` for training procedures
4. Configure data paths in `config/rcc_training_config.yaml`

---

**Support**: For RCC-specific issues, contact help@rcc.uchicago.edu or visit the [RCC website](https://rcc.uchicago.edu/resources/high-performance-computing) 