# CRC Molecular Subtype Classification - Deployment Instructions

## 🚀 Quick Deployment Guide

This package contains everything needed to deploy the CRC molecular subtype classification system on your computing cluster. Follow these steps for a successful deployment.

## 📋 Pre-Deployment Checklist

### ✅ System Requirements Verified
- [ ] 32+ NVIDIA V100/A100 GPUs available
- [ ] 2TB+ total RAM (512GB per node minimum)
- [ ] 10TB+ high-speed storage (NVMe preferred)
- [ ] InfiniBand or 100Gb Ethernet network
- [ ] SLURM or PBS job scheduler configured
- [ ] CUDA 11.8+ and Python 3.11+ installed

### ✅ Data Preparation
- [ ] EPOC WSI files accessible on cluster storage
- [ ] Clinical metadata CSV file available
- [ ] Molecular subtype annotations validated
- [ ] Data access permissions configured

### ✅ Network Access
- [ ] Internet connectivity for package downloads
- [ ] Weights & Biases account (optional but recommended)
- [ ] SSH access to cluster head node

## 🛠️ Step-by-Step Deployment

### Step 1: Extract and Setup (15 minutes)

```bash
# 1. Extract deployment package
tar -xzf crc_molecular_cluster_deployment.tar.gz
cd cluster_deployment_package

# 2. Load required modules (adjust for your cluster)
module load cuda/11.8 python/3.11 openmpi/4.1.4

# 3. Create virtual environment
python -m venv crc_molecular_env
source crc_molecular_env/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r cluster/requirements_cluster.txt
```

### Step 2: Configure Paths (10 minutes)

```bash
# 1. Edit configuration file
vim cluster/configs/cluster_training_config.yaml

# Update these paths for your cluster:
# paths:
#   raw_data: "/path/to/epoc/wsi/"
#   processed_data: "/path/to/processed/data/"
#   cache_dir: "/scratch/crc_molecular/cache/"
#   results_dir: "/results/crc_molecular/"

# 2. Update SLURM script
vim cluster/submit_training.sh

# Update these SLURM parameters:
# #SBATCH --account=your_account
# #SBATCH --partition=your_partition
# #SBATCH --nodes=4
# #SBATCH --gres=gpu:8
```

### Step 3: Validate Setup (20 minutes)

```bash
# 1. Run system validation
python tests/test_cluster_setup.py \
    --gpus 8 \
    --data_dir /path/to/epoc/data \
    --output_report validation_report.txt

# 2. Check validation results
cat validation_report.txt

# 3. If validation fails, address issues before proceeding
```

### Step 4: Process EPOC Data (2-4 hours)

```bash
# 1. Create data directories
mkdir -p /data/epoc/{raw,processed,manifests}
mkdir -p /scratch/crc_molecular/{checkpoints,cache,tensorboard}
mkdir -p /results/crc_molecular

# 2. Submit preprocessing job
sbatch scripts/preprocess_epoc_data.sh \
    --input_dir /path/to/epoc/wsi \
    --output_dir /data/epoc/processed \
    --num_workers 32

# 3. Monitor preprocessing progress
watch -n 30 'squeue -u $USER'
tail -f logs/preprocessing_*.out
```

### Step 5: Launch Training (48+ hours)

```bash
# 1. Verify preprocessing completed successfully
ls /data/epoc/processed/manifests/

# 2. Submit training job
sbatch cluster/submit_training.sh

# 3. Monitor training progress
squeue -u $USER
tail -f logs/crc_molecular_*.out

# 4. Access monitoring dashboards
# - TensorBoard: http://cluster-node:6006
# - Weights & Biases: https://wandb.ai/your-project
```

## 📊 Monitoring Training

### Real-time Monitoring

```bash
# Check job status
squeue -u $USER

# Monitor GPU utilization
srun --jobid=$SLURM_JOB_ID nvidia-smi

# View training logs
tail -f logs/crc_molecular_*.out

# Check training metrics
wandb sync logs/
```

### Key Metrics to Watch

- **Training Loss**: Should decrease steadily
- **Validation Accuracy**: Should improve over epochs
- **GPU Utilization**: Should be >90% across all GPUs
- **Memory Usage**: Should be <90% of available GPU memory
- **Processing Time**: ~48 hours for full training

### Expected Checkpoints

| Time | Expected Progress |
|------|------------------|
| 6 hours | Training loss decreased by >50% |
| 24 hours | Validation accuracy >70% |
| 48 hours | Validation accuracy >85% |
| 72 hours | Training converged, final validation |

## 🔧 Troubleshooting

### Common Issues

**Job Fails to Start**
```bash
# Check SLURM configuration
sinfo
squeue -u $USER

# Verify resource availability
sinfo -N -o "%N %c %m %f %G"
```

**Out of Memory Errors**
```bash
# Reduce batch size
sed -i 's/batch_size: 32/batch_size: 16/' cluster/configs/cluster_training_config.yaml

# Enable gradient checkpointing
sed -i 's/gradient_checkpointing: false/gradient_checkpointing: true/' cluster/configs/cluster_training_config.yaml
```

**Slow Data Loading**
```bash
# Increase number of workers
sed -i 's/num_workers: 8/num_workers: 16/' cluster/configs/cluster_training_config.yaml

# Use faster storage for cache
export TMPDIR=/scratch/tmp
```

**Network Communication Issues**
```bash
# Test inter-node connectivity
srun --nodes=2 --ntasks-per-node=1 --gres=gpu:1 python -c "
import torch
import torch.distributed as dist
dist.init_process_group('nccl')
print('NCCL communication test passed')
"
```

### Getting Help

**Check Logs**
```bash
# SLURM job logs
cat logs/crc_molecular_*.err

# Application logs
grep -i error logs/crc_molecular_*.out

# System logs
journalctl -u slurmd
```

**Performance Analysis**
```bash
# Check resource usage
sacct -j $SLURM_JOB_ID --format=JobID,State,ExitCode,MaxRSS,MaxVMSize

# GPU memory usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Network utilization
ib_write_bw -d mlx5_0
```

## ✅ Post-Training Validation

### Model Evaluation

```bash
# 1. Run comprehensive evaluation
python cluster/evaluate_model.py \
    --checkpoint /scratch/crc_molecular/checkpoints/best_model.pth \
    --test_manifest /data/epoc/processed/test_manifest.json \
    --output_dir /results/crc_molecular/evaluation

# 2. Generate clinical validation report
python cluster/clinical_validation.py \
    --results_dir /results/crc_molecular/evaluation \
    --output_report /results/crc_molecular/clinical_report.pdf
```

### Success Criteria

- [ ] **Training Completed**: No errors in final logs
- [ ] **Accuracy Target**: >90% on test set
- [ ] **Per-class Performance**: F1 >0.85 for all subtypes
- [ ] **Clinical Metrics**: C-index >0.75 for survival
- [ ] **Model Export**: Production model saved successfully

### Final Steps

```bash
# 1. Export production model
python cluster/export_production_model.py \
    --checkpoint /scratch/crc_molecular/checkpoints/best_model.pth \
    --output /results/crc_molecular/production_model.pth

# 2. Create deployment archive
tar -czf crc_molecular_trained_model.tar.gz \
    /results/crc_molecular/production_model.pth \
    /results/crc_molecular/clinical_report.pdf \
    cluster/configs/cluster_training_config.yaml

# 3. Backup results
cp -r /results/crc_molecular /backup/crc_molecular_$(date +%Y%m%d)
```

## 📞 Support

### Technical Issues
- **Cluster Problems**: Contact your HPC support team
- **SLURM Issues**: Check SLURM documentation or support
- **CUDA/GPU Issues**: Verify driver installation and compatibility

### Application Issues
- **Model Training**: Review training logs and configuration
- **Data Processing**: Check WSI file formats and accessibility
- **Performance**: Monitor resource utilization and bottlenecks

### Emergency Contacts
- **System Administrator**: [your-sysadmin@institution.edu]
- **HPC Support**: [hpc-support@institution.edu]
- **Project Lead**: [project-lead@institution.edu]

## 📋 Deployment Checklist

### Pre-Deployment
- [ ] System requirements verified
- [ ] EPOC data accessible
- [ ] Cluster access confirmed
- [ ] Dependencies installed

### During Deployment
- [ ] Configuration updated
- [ ] Validation tests passed
- [ ] Data preprocessing completed
- [ ] Training job submitted

### Post-Deployment
- [ ] Training completed successfully
- [ ] Model performance validated
- [ ] Clinical report generated
- [ ] Production model exported
- [ ] Results backed up

---

**Deployment Status: [ ] Complete [ ] In Progress [ ] Failed**

**Completion Date: _______________**

**Deployed By: _______________**

**Contact for Issues: _______________** 