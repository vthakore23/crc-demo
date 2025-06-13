# CRC Molecular Subtype Classification - System Requirements

## 🎯 Overview

This document specifies the complete system requirements for deploying and running the CRC molecular subtype classification system on a computing cluster with EPOC WSI data.

## 🖥️ Hardware Requirements

### Minimum Requirements

| Component | Specification | Justification |
|-----------|---------------|---------------|
| **GPUs** | 32× NVIDIA V100 (32GB) or A100 (40GB) | Distributed training across 4 nodes |
| **CPU** | 128 cores total (32 per node) | Data preprocessing and loading |
| **Memory** | 2TB total (512GB per node) | Large batch processing and caching |
| **Storage** | 10TB NVMe SSD | WSI data and processed patches |
| **Network** | InfiniBand EDR (100Gb/s) | Inter-node communication |

### Recommended Configuration

| Component | Specification | Benefits |
|-----------|---------------|----------|
| **GPUs** | 32× NVIDIA A100 (80GB) | Larger batch sizes, faster training |
| **CPU** | 256 cores total (64 per node) | Faster data preprocessing |
| **Memory** | 4TB total (1TB per node) | Larger caches, more workers |
| **Storage** | 20TB NVMe SSD RAID | Redundancy and higher throughput |
| **Network** | InfiniBand HDR (200Gb/s) | Reduced communication overhead |

### Storage Layout

```
/data/epoc/                    # 5TB - Raw and processed WSI data
├── raw/                       # 2TB - Original WSI files
├── processed/                 # 2TB - Extracted patches (HDF5)
└── manifests/                 # 1GB - Data split manifests

/scratch/crc_molecular/        # 3TB - Training artifacts
├── checkpoints/               # 500GB - Model checkpoints
├── cache/                     # 2TB - Fast data cache
└── tensorboard/               # 500GB - Training logs

/results/crc_molecular/        # 2TB - Final results
├── models/                    # 100GB - Production models
├── evaluation/                # 500GB - Validation results
└── reports/                   # 1.4TB - Clinical reports and visualizations
```

## 💻 Software Requirements

### Operating System
- **Linux Distribution**: CentOS 7+, Ubuntu 18.04+, or RHEL 7+
- **Kernel**: 3.10+ (4.15+ recommended)
- **Architecture**: x86_64

### CUDA Environment
- **CUDA Toolkit**: 11.8 or 12.0+
- **cuDNN**: 8.6+
- **NVIDIA Driver**: 520.61.05+ (for CUDA 11.8)
- **NCCL**: 2.15+ (for multi-GPU communication)

### Python Environment
- **Python**: 3.11+ (3.11.5 recommended)
- **pip**: 23.0+
- **Virtual Environment**: venv or conda

### Job Scheduler
- **SLURM**: 20.11+ (preferred)
- **PBS/Torque**: 19.0+ (alternative)
- **LSF**: 10.1+ (alternative)

### Container Runtime (Optional)
- **Docker**: 20.10+
- **Singularity**: 3.8+
- **Podman**: 4.0+

## 📦 Python Dependencies

### Core Deep Learning
```
torch==2.1.0+cu118
torchvision==0.16.0+cu118
timm==0.9.12
einops==0.7.0
```

### Medical Imaging
```
openslide-python==1.3.1
scikit-image==0.22.0
opencv-python==4.9.0.80
albumentations==1.3.1
staintools==2.1.2
```

### Data Processing
```
h5py==3.10.0
pandas==2.1.4
numpy==1.26.2
scipy==1.11.4
```

### Distributed Training
```
wandb==0.16.2
tensorboard==2.15.1
mpi4py==3.1.5
```

### Clinical Analysis
```
lifelines==0.27.8
statsmodels==0.14.1
scikit-learn==1.3.2
```

## 🌐 Network Requirements

### Bandwidth
- **Inter-node**: 100Gb/s minimum (InfiniBand EDR)
- **Storage**: 10GB/s sustained read/write
- **Internet**: 1Gb/s for package downloads and monitoring

### Latency
- **Inter-node**: <2μs (InfiniBand)
- **Storage**: <1ms average
- **GPU-GPU**: NVLink preferred

### Ports
- **SSH**: 22 (cluster access)
- **SLURM**: 6817-6818 (scheduler communication)
- **NCCL**: 29500-29600 (distributed training)
- **TensorBoard**: 6006 (monitoring)
- **Weights & Biases**: 443 (HTTPS)

## 📊 Performance Specifications

### Training Performance
- **Throughput**: 1000+ patches/second per GPU
- **Memory Usage**: <90% GPU memory utilization
- **Network Utilization**: <80% bandwidth usage
- **Storage I/O**: 2GB/s sustained read rate

### Inference Performance
- **Latency**: <5 minutes per WSI
- **Throughput**: 100+ WSIs per hour
- **Memory**: <16GB GPU memory per inference
- **CPU**: <50% utilization during inference

## 🔒 Security Requirements

### Data Protection
- **Encryption**: AES-256 for data at rest
- **Network**: TLS 1.3 for data in transit
- **Access Control**: RBAC with audit logging
- **Backup**: 3-2-1 backup strategy

### Compliance
- **HIPAA**: Patient data protection
- **GDPR**: Data privacy (if applicable)
- **Institutional**: Local data governance policies
- **Audit**: Complete access and processing logs

## 🔧 System Configuration

### Kernel Parameters
```bash
# /etc/sysctl.conf
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
vm.swappiness = 1
```

### GPU Configuration
```bash
# Set GPU persistence mode
nvidia-smi -pm 1

# Set GPU power limit (if needed)
nvidia-smi -pl 300

# Enable MIG mode (for A100 if needed)
nvidia-smi -mig 1
```

### SLURM Configuration
```bash
# slurm.conf (key parameters)
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
GresTypes=gpu
AccountingStorageType=accounting_storage/slurmdbd
JobAcctGatherType=jobacct_gather/cgroup
```

### Environment Modules
```bash
# Required modules
module load cuda/11.8
module load python/3.11
module load openmpi/4.1.4
module load hdf5/1.12.2
```

## 📈 Monitoring Requirements

### System Monitoring
- **CPU/Memory**: Ganglia, Nagios, or Prometheus
- **GPU**: nvidia-ml-py, gpustat
- **Network**: InfiniBand monitoring tools
- **Storage**: iostat, iotop

### Application Monitoring
- **Training**: Weights & Biases, TensorBoard
- **Performance**: PyTorch Profiler
- **Logs**: ELK stack or similar
- **Alerts**: PagerDuty or similar

## 🧪 Validation Checklist

### Pre-deployment Validation
- [ ] GPU drivers and CUDA installation
- [ ] Python environment and dependencies
- [ ] SLURM job submission and execution
- [ ] Inter-node communication (NCCL test)
- [ ] Storage performance (iozone benchmark)
- [ ] Network performance (ib_write_bw test)

### Post-deployment Validation
- [ ] Distributed training test (small dataset)
- [ ] Data loading performance test
- [ ] Model checkpoint saving/loading
- [ ] Monitoring system integration
- [ ] Backup and recovery procedures

## 🚨 Troubleshooting

### Common Issues

**CUDA Out of Memory**
```bash
# Reduce batch size in config
sed -i 's/batch_size: 32/batch_size: 16/' cluster/configs/cluster_training_config.yaml
```

**Slow Data Loading**
```bash
# Increase workers and use faster storage
export TMPDIR=/scratch/tmp
```

**Network Communication Failures**
```bash
# Test InfiniBand connectivity
ibstat
ib_write_bw -d mlx5_0
```

**SLURM Job Failures**
```bash
# Check job logs and resource usage
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,MaxRSS,MaxVMSize
```

## 📞 Support Contacts

### Technical Support
- **System Administration**: sysadmin@institution.edu
- **SLURM Support**: hpc-support@institution.edu
- **Network Issues**: network-ops@institution.edu

### Application Support
- **Model Training**: ml-team@institution.edu
- **Data Processing**: data-team@institution.edu
- **Clinical Integration**: clinical-ai@institution.edu

---

**Note**: These requirements are based on training with ~2000 EPOC WSI cases. Scale requirements proportionally for larger datasets. 