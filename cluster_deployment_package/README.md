# CRC Molecular Subtype Classification - Cluster Deployment Package

## 🎯 Overview

This package contains everything needed to deploy and train the CRC molecular subtype classification model on your computing cluster using the EPOC WSI dataset. The system is designed for distributed training across multiple GPUs/nodes and includes comprehensive monitoring, validation, and clinical integration capabilities.

## 📦 Package Contents

```
cluster_deployment_package/
├── README.md                           # This file
├── DEPLOYMENT_CHECKLIST.md            # Step-by-step deployment guide
├── SYSTEM_REQUIREMENTS.md             # Hardware/software requirements
├── cluster/                           # Core training infrastructure
│   ├── distributed_trainer.py         # Main distributed training script
│   ├── submit_training.sh             # SLURM job submission
│   ├── requirements_cluster.txt       # Python dependencies
│   ├── configs/
│   │   └── cluster_training_config.yaml  # Training configuration
│   ├── models/
│   │   └── cluster_ready_model.py     # Multi-scale model architecture
│   ├── data/
│   │   └── wsi_processing_pipeline.py # WSI preprocessing pipeline
│   └── utils/                         # Utility functions
├── scripts/                           # Setup and utility scripts
├── docs/                             # Detailed documentation
└── tests/                            # Validation tests
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Load required modules (adjust for your cluster)
module load cuda/11.8 python/3.11 openmpi/4.1.4

# Create virtual environment
python -m venv crc_molecular_env
source crc_molecular_env/bin/activate

# Install dependencies
pip install -r cluster/requirements_cluster.txt
```

### 2. Data Preparation
```bash
# Set up data directories
mkdir -p /data/epoc/{raw,processed}
mkdir -p /scratch/{checkpoints,cache,tensorboard}/crc_molecular

# Process EPOC WSI data
sbatch scripts/preprocess_epoc_data.sh \
    --input_dir /path/to/epoc/wsi \
    --output_dir /data/epoc/processed \
    --num_workers 32
```

### 3. Launch Training
```bash
# Submit distributed training job
sbatch cluster/submit_training.sh
```

### 4. Monitor Progress
```bash
# Check job status
squeue -u $USER

# Monitor training metrics
wandb sync logs/
tensorboard --logdir /scratch/tensorboard/crc_molecular
```

## 📋 System Requirements

### Hardware
- **GPUs**: 32+ NVIDIA V100/A100 GPUs (4 nodes × 8 GPUs)
- **Memory**: 512GB RAM per node
- **Storage**: 10TB+ high-speed storage (NVMe preferred)
- **Network**: InfiniBand or 100Gb Ethernet for inter-node communication

### Software
- **OS**: Linux (CentOS 7+, Ubuntu 18.04+)
- **CUDA**: 11.8+
- **Python**: 3.11+
- **Scheduler**: SLURM or PBS
- **Container**: Docker/Singularity (optional)

### Data Requirements
- **EPOC Dataset**: ~2000 WSI files (.svs, .ndpi, .mrxs formats)
- **Annotations**: Molecular subtype labels (Canonical/Immune/Stromal)
- **Clinical Metadata**: Patient demographics, survival data, molecular markers
- **Storage Space**: ~5TB for processed patches

## 🔧 Configuration

### Key Configuration Files

1. **Training Config** (`cluster/configs/cluster_training_config.yaml`)
   - Model architecture parameters
   - Training hyperparameters
   - Distributed training settings
   - Data paths and preprocessing options

2. **SLURM Script** (`cluster/submit_training.sh`)
   - Resource allocation
   - Module loading
   - Environment setup
   - Job submission parameters

3. **Requirements** (`cluster/requirements_cluster.txt`)
   - All Python dependencies with versions
   - CUDA-specific packages
   - Medical imaging libraries

### Customization Points

- **Batch Size**: Adjust based on GPU memory (default: 32 per GPU)
- **Learning Rates**: Backbone (1e-5) and head (1e-3) learning rates
- **Data Paths**: Update paths in config file for your storage layout
- **Resource Allocation**: Modify SLURM parameters for your cluster

## 📊 Expected Performance

### Training Metrics
- **Training Time**: ~48 hours on 32 V100 GPUs
- **Memory Usage**: ~16GB per GPU
- **Storage I/O**: ~2GB/s sustained read
- **Network**: ~10GB/s inter-node communication

### Model Performance Targets
- **Accuracy**: 90%+ on EPOC validation set
- **Per-class F1**: 0.85+ for each molecular subtype
- **Inference Speed**: <5 minutes per WSI
- **Uncertainty Calibration**: ECE < 0.1

## 🔍 Monitoring & Debugging

### Real-time Monitoring
- **Weights & Biases**: Training metrics, loss curves, validation scores
- **TensorBoard**: Detailed training logs and visualizations
- **SLURM**: Job status, resource utilization, error logs

### Common Issues & Solutions

1. **Out of Memory**
   - Reduce batch size in config
   - Enable gradient checkpointing
   - Use mixed precision training

2. **Slow Data Loading**
   - Increase num_workers
   - Use faster storage (NVMe)
   - Pre-process data to HDF5 format

3. **Network Bottlenecks**
   - Check InfiniBand connectivity
   - Adjust NCCL settings
   - Monitor inter-node bandwidth

## 📈 Validation & Testing

### Automated Tests
```bash
# Run system validation
python tests/test_cluster_setup.py

# Validate data pipeline
python tests/test_data_processing.py

# Test model architecture
python tests/test_model_components.py
```

### Clinical Validation
- 5-fold cross-validation with stratification
- Survival analysis (C-index calculation)
- Subgroup analysis (MSI, BRAF, age groups)
- Calibration assessment

## 🚨 Important Notes

### Data Security
- Ensure HIPAA compliance for patient data
- Use encrypted storage for WSI files
- Implement audit logging for all access
- Follow institutional data governance policies

### Reproducibility
- Fixed random seeds in configuration
- Version-controlled code and dependencies
- Comprehensive logging of all parameters
- Checkpoint saving for recovery

### Clinical Integration
- Model outputs include uncertainty estimates
- Generates interpretable heatmaps
- Provides pathway-specific scores
- Compatible with clinical reporting systems

## 📞 Support & Contact

### Technical Issues
- **Model Architecture**: See `docs/model_architecture.md`
- **Data Processing**: See `docs/data_pipeline.md`
- **Distributed Training**: See `docs/distributed_training.md`

### Clinical Questions
- **Validation Protocol**: See `docs/clinical_validation.md`
- **Interpretation Guide**: See `docs/clinical_interpretation.md`
- **Regulatory Compliance**: See `docs/regulatory_requirements.md`

## 📝 Citation

If you use this system in your research, please cite:

```bibtex
@article{crc_molecular_2024,
  title={Multi-scale Deep Learning for CRC Molecular Subtype Classification},
  author={[Your Team]},
  journal={[Journal]},
  year={2024},
  note={Based on Pitroda et al. JAMA Oncology 2018}
}
```

## 📄 License

This software is provided under the MIT License. See LICENSE file for details.

---

**For immediate support during deployment, contact: [your-email@institution.edu]** 