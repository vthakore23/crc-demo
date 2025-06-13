# EPOC Cohort Training Guide for Computing Cluster

## Overview

This guide explains how to train the CRC molecular subtype classification model using EPOC cohort data on a high-performance computing (HPC) cluster. The process involves preparing data, configuring the training environment, and running distributed training across multiple GPUs/nodes.

## Prerequisites

### 1. Data Requirements
- **WSI Images**: Whole Slide Images from EPOC cohort patients
- **Molecular Labels**: Ground truth molecular subtypes (CMS1-4 or canonical-3)
- **Clinical Data**: Treatment outcomes, survival data (optional but valuable)

### 2. Computing Requirements
- **GPU Cluster**: Access to SLURM-managed cluster with NVIDIA GPUs
- **Storage**: Sufficient space for WSI tiles (~500GB-2TB depending on cohort size)
- **Software**: CUDA 11.8+, Python 3.9+, PyTorch 2.0+

## Step-by-Step Training Process

### Step 1: Data Preparation

#### 1.1 Extract Tiles from WSIs
```bash
# On the cluster, run tile extraction
python scripts/extract_wsi_tiles.py \
    --wsi-dir /path/to/epoc/wsis \
    --output-dir /path/to/epoc/tiles \
    --tile-size 224 \
    --overlap 0.1 \
    --magnification 20 \
    --workers 32
```

#### 1.2 Create Training Manifest
The manifest CSV should contain:
```csv
patient_id,molecular_subtype,wsi_path,treatment_arm,pfs_months,os_months
EPOC_001,CMS1,patient_001/,chemo,12.3,24.5
EPOC_002,CMS2,patient_002/,chemo+cetuximab,18.7,36.2
...
```

#### 1.3 Split Data
```python
# Create train/val/test splits
python scripts/prepare_epoc_splits.py \
    --manifest epoc_manifest.csv \
    --output-dir /path/to/epoc/splits \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --stratify-by molecular_subtype
```

### Step 2: Environment Setup on Cluster

#### 2.1 Create Virtual Environment
```bash
# Login to cluster
ssh username@cluster.example.com

# Load modules
module load python/3.9
module load cuda/11.8

# Create conda environment
conda create -n epoc python=3.9 -y
conda activate epoc

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements_cluster.txt
```

#### 2.2 Test GPU Access
```python
# test_gpu.py
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
```

### Step 3: Configure Training

#### 3.1 Generate SLURM Script
```bash
python scripts/train_epoc_cluster.py \
    --generate-slurm \
    --train-manifest /path/to/epoc/splits/train.csv \
    --val-manifest /path/to/epoc/splits/val.csv \
    --tile-dir /path/to/epoc/tiles \
    --checkpoint-dir /path/to/checkpoints \
    --log-dir /path/to/logs \
    --partition gpu-v100 \
    --nodes 2 \
    --gpus-per-node 4 \
    --time-limit 48:00:00 \
    --output-dir ./
```

This generates `train_epoc.sbatch` with proper cluster configuration.

#### 3.2 Key Training Parameters
```bash
# Important parameters to adjust:
--batch-size 64          # Per GPU batch size (adjust based on GPU memory)
--epochs 100             # Total training epochs
--learning-rate 1e-4     # Base learning rate
--tiles-per-patient 50   # Number of tiles per patient
--backbone efficientnet_b0  # Model architecture
--num-classes 4          # Number of molecular subtypes
```

### Step 4: Submit Training Job

#### 4.1 Submit to SLURM
```bash
# Submit the job
sbatch train_epoc.sbatch

# Check job status
squeue -u $USER

# Monitor job output
tail -f logs/slurm_JOBID.out
```

#### 4.2 Monitor Training Progress
```bash
# TensorBoard monitoring (from login node)
module load tensorboard
tensorboard --logdir=/path/to/logs/tensorboard --port=6006

# Then SSH tunnel from local machine:
ssh -L 6006:localhost:6006 username@cluster.example.com
# Open http://localhost:6006 in browser
```

### Step 5: Training Stages

The training process goes through these stages:

1. **Data Loading**: 
   - Loads tile manifests
   - Distributes data across GPUs
   - ~5-10 minutes

2. **Model Initialization**:
   - Loads pretrained backbone
   - Initializes distributed training
   - ~2-5 minutes

3. **Training Loop**:
   - Processes batches across all GPUs
   - Validates every epoch
   - ~30-60 minutes per epoch

4. **Checkpointing**:
   - Saves best model based on validation AUC
   - Saves training state for resumption
   - Automatic

### Step 6: Expected Outputs

After training completes, you'll have:

```
checkpoints/
├── best_model.pth              # Best model weights
├── best_model_epoch_42.pth     # Specific epoch checkpoint
└── training_state.json         # Training configuration

logs/
├── training_20240112_143022.log  # Detailed training log
├── slurm_12345.out               # SLURM output
└── tensorboard/                  # TensorBoard logs

results/
├── validation_metrics.json       # Final validation metrics
├── confusion_matrix.png          # Confusion matrix plot
└── training_curves.png           # Loss/accuracy curves
```

### Step 7: Post-Training Validation

#### 7.1 Test Set Evaluation
```bash
python scripts/evaluate_epoc_model.py \
    --model-path checkpoints/best_model.pth \
    --test-manifest /path/to/epoc/splits/test.csv \
    --tile-dir /path/to/epoc/tiles \
    --output-dir results/test_evaluation
```

#### 7.2 Clinical Correlation Analysis
```python
# Run EPOC validation
python scripts/run_epoc_validation.py \
    --model-path checkpoints/best_model.pth \
    --epoc-manifest epoc_full_manifest.csv \
    --wsi-directory /path/to/epoc/wsis \
    --output-dir epoc_validation_results
```

## Important Considerations

### Memory Management
- **Tile Loading**: Use `--tiles-per-patient` to control memory usage
- **Batch Size**: Reduce if encountering OOM errors
- **Gradient Accumulation**: Use if batch size is too small

### Performance Optimization
- **Mixed Precision**: Add `--amp` flag for faster training
- **Data Loading**: Increase `--num-workers` for faster I/O
- **Prefetching**: Data is automatically prefetched

### Molecular Subtype Validation
**CRITICAL**: The model requires ground truth molecular labels (from RNA-seq, IHC, etc.) for training. The accuracy metrics are only meaningful if these labels are validated molecular subtypes, not morphological guesses.

### Common Issues and Solutions

1. **CUDA Out of Memory**:
   ```bash
   # Reduce batch size or tiles per patient
   --batch-size 32 --tiles-per-patient 25
   ```

2. **Slow Data Loading**:
   ```bash
   # Increase workers and ensure data is on fast storage
   --num-workers 8
   ```

3. **Training Instability**:
   ```bash
   # Reduce learning rate or increase warmup
   --learning-rate 5e-5 --warmup-epochs 5
   ```

## Example Complete Workflow

```bash
# 1. Prepare data
cd /cluster/project/epoc
python scripts/extract_wsi_tiles.py --wsi-dir raw_wsis/ --output-dir tiles/

# 2. Create splits
python scripts/prepare_epoc_splits.py --manifest epoc_manifest.csv

# 3. Generate SLURM script
python scripts/train_epoc_cluster.py --generate-slurm \
    --train-manifest splits/train.csv \
    --val-manifest splits/val.csv \
    --tile-dir tiles/ \
    --checkpoint-dir checkpoints/ \
    --log-dir logs/

# 4. Submit job
sbatch train_epoc.sbatch

# 5. Monitor
watch squeue -u $USER
tail -f logs/slurm_*.out

# 6. Evaluate
python scripts/evaluate_epoc_model.py \
    --model-path checkpoints/best_model.pth \
    --test-manifest splits/test.csv
```

## Integration with CRC Platform

After training, integrate the model:

1. **Copy Model to Platform**:
   ```bash
   cp checkpoints/best_model.pth ~/CRC_Analysis_Project/models/epoc_trained_model.pth
   ```

2. **Update Configuration**:
   ```python
   # In app/config.py
   MOLECULAR_MODEL_PATH = "models/epoc_trained_model.pth"
   MODEL_TRAINED_ON = "EPOC Cohort"
   MOLECULAR_CLASSES = ["CMS1", "CMS2", "CMS3", "CMS4"]
   ```

3. **Deploy Updated Platform**:
   ```bash
   cd ~/CRC_Analysis_Project
   streamlit run app.py
   ```

## Support and Troubleshooting

For cluster-specific issues:
- Check cluster documentation
- Contact HPC support team
- Review SLURM error logs

For model/training issues:
- Check training logs in `logs/`
- Review TensorBoard metrics
- Ensure data quality and labels

Remember: The model's clinical utility depends entirely on having validated molecular subtype labels in your training data! 