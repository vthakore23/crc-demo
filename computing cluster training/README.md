# CRC Molecular Subtype Classification - Randi Cluster Training Package

## Overview

This package contains all necessary components for training the CRC molecular subtype classification model on the University of Chicago CRI Randi cluster. The model is designed to predict molecular subtypes from EPOC WSI (Whole Slide Image) data using state-of-the-art deep learning techniques.

## Package Contents

```
computing cluster training/
├── README.md                           # This file
├── SYSTEM_SETUP.md                     # Randi cluster setup instructions
├── TRAINING_GUIDE.md                   # Step-by-step training procedures
├── DATA_PREPARATION.md                 # EPOC data preparation guidelines
├── TROUBLESHOOTING.md                  # Common issues and solutions
├── scripts/
│   ├── setup_environment.sh           # Environment setup script
│   ├── submit_training_job.sh          # SLURM job submission
│   ├── monitor_training.sh             # Training monitoring utilities
│   └── validate_setup.sh               # Pre-training validation
├── config/
│   ├── randi_training_config.yaml     # Randi-specific configuration
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
3. **Configure Data Paths** - Update `config/randi_training_config.yaml`
4. **Submit Training Job** - Execute `scripts/submit_training_job.sh`
5. **Monitor Progress** - Use `scripts/monitor_training.sh`

## Hardware Requirements (Randi Cluster)

- **Compute Nodes**: 4-8 GPU nodes recommended
- **GPUs**: 8x NVIDIA A100 (40GB or 80GB)
- **Memory**: 512GB RAM per node minimum
- **Storage**: 10TB scratch space for training data
- **Network**: InfiniBand HDR100 (available on Randi)

## Expected Training Time

- **Data Preprocessing**: 2-4 hours
- **Model Training**: 24-48 hours (depending on data size)
- **Validation**: 2-4 hours
- **Total**: 28-56 hours

## Support

For technical assistance:
- **Randi Cluster Support**: help@rcc.uchicago.edu
- **Model Training Issues**: Contact project team
- **EPOC Data Questions**: Refer to clinical team

## Next Steps

1. Review all documentation in this package
2. Contact CRI support for account setup if needed
3. Prepare EPOC WSI data according to `DATA_PREPARATION.md`
4. Execute training pipeline following `TRAINING_GUIDE.md`

---

**Note**: This package is specifically configured for the University of Chicago CRI Randi cluster environment. All paths, module names, and configurations are tailored for this system. 