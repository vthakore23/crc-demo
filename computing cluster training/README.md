# CRC Molecular Subtype Classification - UChicago RCC Training Package

## Overview

This package contains all necessary components for training the CRC molecular subtype classification model on the University of Chicago Research Computing Center (RCC) high-performance computing infrastructure. The model is designed to predict molecular subtypes from EPOC WSI (Whole Slide Image) data using state-of-the-art deep learning techniques.

## Compatible Systems

This package is designed to work with UChicago RCC computing resources:

- **Midway2**: Primary HPC cluster with 16,016+ cores across 572+ nodes
- **Randi**: Specialized cluster with NVIDIA A100 GPUs
- **GPU Nodes**: NVIDIA K80 or A100 accelerator cards
- **Storage**: 2.2+ PB available storage infrastructure

## Package Contents

```
computing cluster training/
├── README.md                           # This file
├── SYSTEM_SETUP.md                     # RCC cluster setup instructions
├── TRAINING_GUIDE.md                   # Step-by-step training procedures
├── DATA_PREPARATION.md                 # EPOC data preparation guidelines
├── TROUBLESHOOTING.md                  # Common issues and solutions
├── scripts/
│   ├── setup_environment.sh           # Environment setup script
│   ├── submit_training_job.sh          # SLURM job submission
│   ├── monitor_training.sh             # Training monitoring utilities
│   └── validate_setup.sh               # Pre-training validation
├── config/
│   ├── rcc_training_config.yaml       # RCC-specific configuration
│   ├── slurm_job_template.sh          # SLURM job template
│   └── module_requirements.txt        # Required environment modules
└── models/
    ├── model_architecture.py          # Neural network definitions
    ├── training_pipeline.py           # Training orchestration
    └── data_loader.py                 # EPOC data handling
```

## Quick Start

1. **Review System Requirements** - See `SYSTEM_SETUP.md`
2. **Prepare Environment** - Follow `TRAINING_GUIDE.md` Section 1
3. **Configure Data Paths** - Update `config/rcc_training_config.yaml`
4. **Submit Training Job** - Execute `scripts/submit_training_job.sh`
5. **Monitor Progress** - Use `scripts/monitor_training.sh`

## Hardware Requirements (RCC Infrastructure)

### Minimum Requirements
- **Compute Nodes**: 4-8 nodes recommended
- **CPUs**: Intel Broadwell (28 cores) or Skylake (40 cores) per node
- **Memory**: 64-96GB RAM per node minimum
- **GPUs**: NVIDIA K80 or A100 (if using GPU nodes)
- **Storage**: 10TB+ for training data and checkpoints
- **Network**: InfiniBand FDR/EDR (up to 100Gbps) or 40Gbps GigE

### Optimal Configuration
- **GPU Nodes**: 4-8 NVIDIA A100 GPUs (if available on Randi)
- **CPU Nodes**: Intel Skylake 40-core nodes on Midway2
- **Memory**: Large shared memory nodes (up to 1TB) for data preprocessing
- **Storage**: High-performance storage from RCC's 2.2+ PB infrastructure

## Expected Training Time

- **Data Preprocessing**: 2-4 hours
- **Model Training**: 24-48 hours (depending on available resources)
- **Validation**: 2-4 hours
- **Total**: 28-56 hours

## RCC Account Requirements

- Active UChicago RCC account
- Allocation on appropriate partition (CPU or GPU)
- Access to required software modules
- Storage allocation for training data

## Support

For technical assistance:
- **RCC Support**: help@rcc.uchicago.edu or (773) 795-2667
- **Walk-in Support**: Regenstein Library, Suite 216
- **RCC User Guide**: Available on RCC website
- **Model Training Issues**: Contact project team
- **EPOC Data Questions**: Refer to clinical team

## Getting Started with RCC

1. **Request Account**: Visit RCC website for account setup
2. **Review RCC User Guide**: Familiarize yourself with RCC systems
3. **Choose Appropriate Cluster**: Midway2 for CPU, Randi for GPU training
4. **Prepare EPOC WSI data** according to `DATA_PREPARATION.md`
5. **Execute training pipeline** following `TRAINING_GUIDE.md`

## Next Steps

1. Review all documentation in this package
2. Contact RCC support for account setup if needed
3. Determine optimal cluster configuration (Midway2 vs Randi)
4. Prepare EPOC WSI data according to guidelines
5. Execute training pipeline with appropriate resource allocation

---

**Note**: This package is configured for the University of Chicago Research Computing Center infrastructure. All paths, module names, and configurations can be adapted for either Midway2 or Randi clusters based on resource availability and requirements. 