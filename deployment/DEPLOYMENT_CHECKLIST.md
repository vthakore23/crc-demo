# CRC Molecular Subtype Classification - Deployment Checklist

## 📋 Pre-Deployment Requirements

### ✅ Hardware Verification
- [ ] **GPU Resources**: 32+ NVIDIA V100/A100 GPUs available
- [ ] **Memory**: 512GB RAM per node (4 nodes minimum)
- [ ] **Storage**: 10TB+ high-speed storage (NVMe preferred)
- [ ] **Network**: InfiniBand or 100Gb Ethernet connectivity
- [ ] **Cooling**: Adequate cooling for sustained GPU workload

### ✅ Software Environment
- [ ] **Operating System**: Linux (CentOS 7+, Ubuntu 18.04+)
- [ ] **CUDA**: Version 11.8 or higher installed
- [ ] **Python**: Version 3.11+ available
- [ ] **MPI**: OpenMPI 4.1.4+ for distributed training
- [ ] **Scheduler**: SLURM or PBS configured and running
- [ ] **Container Runtime**: Docker/Singularity (optional but recommended)

### ✅ Data Preparation
- [ ] **EPOC Dataset**: WSI files accessible on cluster storage
- [ ] **Annotations**: Molecular subtype labels available
- [ ] **Clinical Metadata**: Patient demographics and survival data
- [ ] **Data Permissions**: Appropriate access controls configured
- [ ] **Backup Strategy**: Data backup and recovery plan in place

## 🚀 Step-by-Step Deployment

### Step 1: Environment Setup (30 minutes)

```bash
# 1.1 Load required modules
module purge
module load cuda/11.8
module load python/3.11
module load openmpi/4.1.4

# 1.2 Create project directory
mkdir -p /projects/crc_molecular
cd /projects/crc_molecular

# 1.3 Extract deployment package
tar -xzf crc_molecular_deployment.tar.gz
cd cluster_deployment_package

# 1.4 Create virtual environment
python -m venv crc_molecular_env
source crc_molecular_env/bin/activate

# 1.5 Install dependencies
pip install --upgrade pip
pip install -r cluster/requirements_cluster.txt
```

**Verification:**
- [ ] All modules loaded successfully
- [ ] Virtual environment created and activated
- [ ] All Python packages installed without errors
- [ ] CUDA available: `python -c "import torch; print(torch.cuda.is_available())"`

### Step 2: Storage Configuration (20 minutes)

```bash
# 2.1 Create data directories
mkdir -p /data/epoc/{raw,processed,manifests}
mkdir -p /scratch/crc_molecular/{checkpoints,cache,tensorboard}
mkdir -p /results/crc_molecular

# 2.2 Set permissions
chmod -R 755 /data/epoc
chmod -R 755 /scratch/crc_molecular
chmod -R 755 /results/crc_molecular

# 2.3 Test storage performance
dd if=/dev/zero of=/scratch/crc_molecular/test_file bs=1G count=10
rm /scratch/crc_molecular/test_file
```

**Verification:**
- [ ] All directories created successfully
- [ ] Appropriate permissions set
- [ ] Storage write speed >1GB/s
- [ ] Sufficient free space (>10TB)

### Step 3: Data Processing Setup (45 minutes)

```bash
# 3.1 Configure data paths in config file
vim cluster/configs/cluster_training_config.yaml
# Update paths section:
# paths:
#   raw_data: "/data/epoc/raw/"
#   processed_data: "/data/epoc/processed/"
#   cache_dir: "/scratch/crc_molecular/cache/"
#   results_dir: "/results/crc_molecular/"

# 3.2 Create EPOC data manifest
python scripts/create_epoc_manifest.py \
    --wsi_dir /data/epoc/raw \
    --annotations_file /data/epoc/annotations.csv \
    --output_manifest /data/epoc/epoc_manifest.json

# 3.3 Test data processing pipeline
python tests/test_data_processing.py \
    --manifest /data/epoc/epoc_manifest.json \
    --num_samples 10
```

**Verification:**
- [ ] Configuration file updated with correct paths
- [ ] EPOC manifest created successfully
- [ ] Data processing test completed without errors
- [ ] Sample patches generated and validated

### Step 4: Distributed Training Test (30 minutes)

```bash
# 4.1 Test single-node multi-GPU setup
python tests/test_distributed_setup.py \
    --gpus 8 \
    --backend nccl

# 4.2 Test multi-node communication (if applicable)
srun --nodes=2 --ntasks-per-node=1 --gres=gpu:1 \
    python tests/test_multi_node.py

# 4.3 Validate model architecture
python tests/test_model_components.py
```

**Verification:**
- [ ] Single-node multi-GPU test passed
- [ ] Multi-node communication working (if applicable)
- [ ] Model architecture loads without errors
- [ ] Memory usage within expected limits

### Step 5: Launch Data Preprocessing (2-4 hours)

```bash
# 5.1 Submit preprocessing job
sbatch scripts/preprocess_epoc_data.sh \
    --input_dir /data/epoc/raw \
    --output_dir /data/epoc/processed \
    --num_workers 32

# 5.2 Monitor preprocessing progress
watch -n 30 'squeue -u $USER'
tail -f logs/preprocessing_*.out
```

**Verification:**
- [ ] Preprocessing job submitted successfully
- [ ] Job running without errors
- [ ] Patch extraction progressing
- [ ] Output files being generated

### Step 6: Training Configuration (15 minutes)

```bash
# 6.1 Update training configuration
vim cluster/configs/cluster_training_config.yaml
# Verify/update:
# - data.train_manifest path
# - data.val_manifest path
# - distributed.nodes and gpus_per_node
# - resources.partition and account

# 6.2 Update SLURM script
vim cluster/submit_training.sh
# Verify/update:
# - #SBATCH --nodes
# - #SBATCH --gres=gpu
# - #SBATCH --account
# - #SBATCH --partition
```

**Verification:**
- [ ] Training config paths are correct
- [ ] Resource allocation matches cluster setup
- [ ] SLURM parameters are appropriate
- [ ] Account and partition names are valid

### Step 7: Launch Training (48+ hours)

```bash
# 7.1 Submit training job
sbatch cluster/submit_training.sh

# 7.2 Verify job submission
squeue -u $USER
sacct -j $SLURM_JOB_ID

# 7.3 Monitor initial progress
tail -f logs/crc_molecular_*.out
```

**Verification:**
- [ ] Training job submitted successfully
- [ ] Job allocated requested resources
- [ ] Training started without errors
- [ ] Initial metrics being logged

## 🔍 Monitoring & Validation

### Real-time Monitoring Setup

```bash
# Setup Weights & Biases monitoring
export WANDB_API_KEY="your_wandb_key"
wandb login

# Setup TensorBoard
tensorboard --logdir /scratch/crc_molecular/tensorboard \
    --host 0.0.0.0 --port 6006 &
```

### Key Metrics to Monitor

- [ ] **Training Loss**: Decreasing steadily
- [ ] **Validation Accuracy**: Improving over epochs
- [ ] **GPU Utilization**: >90% across all GPUs
- [ ] **Memory Usage**: <90% of available GPU memory
- [ ] **Network I/O**: Consistent inter-node communication
- [ ] **Storage I/O**: No bottlenecks in data loading

### Validation Checkpoints

**After 6 hours:**
- [ ] Training loss decreased by >50%
- [ ] No out-of-memory errors
- [ ] All GPUs actively training

**After 24 hours:**
- [ ] Validation accuracy >70%
- [ ] Training stable (no divergence)
- [ ] Checkpoints saving successfully

**After 48 hours:**
- [ ] Validation accuracy >85%
- [ ] Per-class F1 scores >0.8
- [ ] Model converging

## 🚨 Troubleshooting

### Common Issues & Solutions

**Out of Memory Errors:**
```bash
# Reduce batch size in config
sed -i 's/batch_size: 32/batch_size: 16/' cluster/configs/cluster_training_config.yaml
```

**Slow Data Loading:**
```bash
# Increase number of workers
sed -i 's/num_workers: 8/num_workers: 16/' cluster/configs/cluster_training_config.yaml
```

**Network Communication Issues:**
```bash
# Check InfiniBand status
ibstat
# Test inter-node connectivity
srun --nodes=2 --ntasks-per-node=1 ib_write_bw
```

**Job Failures:**
```bash
# Check job logs
sacct -j $SLURM_JOB_ID --format=JobID,JobName,State,ExitCode
# Review error logs
cat logs/crc_molecular_*.err
```

## ✅ Post-Training Validation

### Model Evaluation
```bash
# Run comprehensive evaluation
python cluster/evaluate_model.py \
    --checkpoint /scratch/crc_molecular/checkpoints/best_model.pth \
    --test_manifest /data/epoc/processed/test_manifest.json \
    --output_dir /results/crc_molecular/evaluation

# Generate clinical validation report
python cluster/clinical_validation.py \
    --results_dir /results/crc_molecular/evaluation \
    --output_report /results/crc_molecular/clinical_report.pdf
```

### Performance Verification
- [ ] **Accuracy**: >90% on test set
- [ ] **Per-class F1**: >0.85 for all subtypes
- [ ] **Calibration**: ECE <0.1
- [ ] **Inference Speed**: <5 minutes per WSI
- [ ] **Uncertainty Quality**: Reliable confidence scores

### Clinical Validation
- [ ] **Cross-validation**: 5-fold CV completed
- [ ] **Survival Analysis**: C-index >0.75
- [ ] **Subgroup Analysis**: Consistent performance across groups
- [ ] **Clinical Report**: Generated and reviewed

## 📦 Deployment Completion

### Final Steps
- [ ] **Model Export**: Production model saved
- [ ] **Documentation**: All logs and reports archived
- [ ] **Cleanup**: Temporary files removed
- [ ] **Backup**: Final model and results backed up
- [ ] **Handover**: Documentation provided to end users

### Success Criteria
- [ ] Training completed successfully
- [ ] Model meets performance targets
- [ ] Clinical validation passed
- [ ] Production deployment ready
- [ ] Documentation complete

---

**Deployment Status: [ ] Complete [ ] In Progress [ ] Failed**

**Completion Date: _______________**

**Deployed By: _______________**

**Validated By: _______________** 