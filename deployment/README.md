# CRC Subtype Predictor - Deployment

## Contents

### Cluster Training (`cluster/`)
- `train_distributed_epoc.py` - Main distributed training script
- `slurm_submit.sh` - SLURM job submission script
- `production_inference.py` - Production inference pipeline
- `config/epoc_training_config.yaml` - Training configuration
- `requirements_cluster.txt` - Python dependencies

### Quick Start
1. Install dependencies: `pip install -r cluster/requirements_cluster.txt`
2. Configure paths in `cluster/config/epoc_training_config.yaml`
3. Submit job: `sbatch cluster/slurm_submit.sh`

### Requirements
- PyTorch 2.0+, CUDA 11.8+
- Multi-node cluster with InfiniBand
- Minimum 4 nodes × 8 GPUs (32 GPUs total)
- 10TB+ shared storage for WSI data

See `DEPLOYMENT_GUIDE.md` for detailed instructions.
