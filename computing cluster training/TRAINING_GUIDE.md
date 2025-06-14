# Training Guide - CRC Molecular Subtype Classification

## Overview

This guide provides step-by-step instructions for training the CRC molecular subtype classification model on the Randi cluster using EPOC WSI data.

## Prerequisites

- Completed system setup (see `SYSTEM_SETUP.md`)
- EPOC WSI data prepared (see `DATA_PREPARATION.md`)
- Active allocation on Randi GPU partition

## Training Workflow

### Phase 1: Environment Preparation

#### 1.1 Connect to Randi
```bash
ssh username@randi.cri.uchicago.edu
cd /scratch/username/crc_molecular_training
```

#### 1.2 Load Environment
```bash
module load cuda/11.8 python/3.11 gcc/11.2.0 openmpi/4.1.4
source crc_env/bin/activate
```

#### 1.3 Verify Setup
```bash
./scripts/validate_setup.sh
```

### Phase 2: Data Configuration

#### 2.1 Update Configuration File
Edit `config/randi_training_config.yaml`:

```yaml
# Data paths (update with your actual paths)
data:
  root_dir: "/scratch/username/crc_molecular_training/data"
  manifest_file: "/scratch/username/crc_molecular_training/data/manifests/epoc_manifest.csv"
  output_dir: "/scratch/username/crc_molecular_training/results"

# Training parameters
training:
  batch_size: 32          # Per GPU
  num_epochs: 100
  learning_rate: 1e-4
  num_workers: 8          # Data loading workers per GPU

# Hardware configuration
hardware:
  num_gpus: 8             # Total GPUs to use
  nodes: 1                # Number of nodes
  memory_per_node: "512G" # Memory allocation
```

#### 2.2 Verify Data Paths
```bash
# Check data directory structure
ls -la /scratch/username/crc_molecular_training/data/
ls -la /scratch/username/crc_molecular_training/data/manifests/
```

### Phase 3: Job Submission

#### 3.1 Review Job Script
Examine `scripts/submit_training_job.sh` and modify if needed:

```bash
cat scripts/submit_training_job.sh
```

Key parameters to verify:
- `--gres=gpu:8` (number of GPUs)
- `--mem=512G` (memory allocation)
- `--time=48:00:00` (wall time)

#### 3.2 Submit Training Job
```bash
sbatch scripts/submit_training_job.sh
```

#### 3.3 Check Job Status
```bash
# View job queue
squeue -u username

# Check job details
scontrol show job JOBID
```

### Phase 4: Monitoring

#### 4.1 Monitor Training Progress
```bash
# View real-time logs
tail -f logs/training_JOBID.out

# Check GPU utilization
ssh gpu-node-name
nvidia-smi
```

#### 4.2 Training Metrics
Monitor the following metrics:
- **Loss curves**: Training and validation loss
- **Accuracy**: Per-epoch accuracy improvements
- **GPU utilization**: Should be >80%
- **Memory usage**: Monitor for out-of-memory errors

#### 4.3 Checkpoint Management
```bash
# List saved checkpoints
ls -la models/checkpoints/

# Check latest checkpoint
ls -lt models/checkpoints/ | head -5
```

### Phase 5: Validation

#### 5.1 Training Completion
When training completes, verify:
```bash
# Check final logs
tail -50 logs/training_JOBID.out

# Verify model files
ls -la models/final/
```

#### 5.2 Model Evaluation
```bash
# Run evaluation script
python scripts/evaluate_model.py --config config/randi_training_config.yaml
```

#### 5.3 Results Analysis
```bash
# Check results directory
ls -la results/
cat results/training_summary.txt
```

## Expected Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| Data Preprocessing | 2-4 hours | WSI patch extraction and preparation |
| Model Training | 24-48 hours | Deep learning training process |
| Validation | 2-4 hours | Model evaluation and testing |
| **Total** | **28-56 hours** | Complete training pipeline |

## Resource Utilization

### Optimal Configuration
- **GPUs**: 8x NVIDIA A100 (40GB or 80GB)
- **Memory**: 512GB RAM
- **Storage**: 10TB scratch space
- **Network**: InfiniBand for multi-GPU communication

### Performance Targets
- **GPU Utilization**: >80%
- **Memory Usage**: <90% of available
- **Training Speed**: ~1000 samples/second
- **Convergence**: Validation accuracy plateau after 50-80 epochs

## Troubleshooting

### Common Issues

#### Job Fails to Start
```bash
# Check resource availability
sinfo -p gpu

# Review job script for errors
sbatch --test-only scripts/submit_training_job.sh
```

#### Out of Memory Errors
```bash
# Reduce batch size in config
sed -i 's/batch_size: 32/batch_size: 16/' config/randi_training_config.yaml
```

#### Slow Training
```bash
# Check data loading bottlenecks
# Increase num_workers in config
sed -i 's/num_workers: 8/num_workers: 16/' config/randi_training_config.yaml
```

#### Network Issues
```bash
# Test InfiniBand connectivity
ibstat
ib_write_bw
```

### Log Analysis

#### Training Logs Location
```bash
# SLURM output logs
ls logs/training_*.out
ls logs/training_*.err

# Application logs
ls logs/model_training.log
```

#### Key Log Patterns
- **Success**: "Training completed successfully"
- **GPU Issues**: "CUDA out of memory"
- **Data Issues**: "FileNotFoundError" or "Data loading error"
- **Network Issues**: "NCCL timeout" or "Communication error"

## Post-Training Steps

### 1. Model Validation
```bash
# Run comprehensive evaluation
python scripts/validate_final_model.py
```

### 2. Results Documentation
```bash
# Generate training report
python scripts/generate_report.py --output results/training_report.pdf
```

### 3. Model Export
```bash
# Export for inference
python scripts/export_model.py --format onnx --output models/production/
```

### 4. Cleanup
```bash
# Archive training data (optional)
tar -czf training_archive_$(date +%Y%m%d).tar.gz logs/ models/checkpoints/

# Clean temporary files
rm -rf /tmp/training_*
```

## Next Steps

1. **Model Integration**: Deploy trained model for inference
2. **Performance Analysis**: Review training metrics and accuracy
3. **Clinical Validation**: Test with additional EPOC data
4. **Production Deployment**: Integrate with clinical workflow

---

**Support**: For training issues, refer to `TROUBLESHOOTING.md` or contact the technical team. 