#!/bin/bash
#SBATCH --job-name=crc_molecular_training
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --mem=512GB
#SBATCH --partition=gpu
#SBATCH --account=medical_ai
#SBATCH --output=logs/crc_molecular_%j.out
#SBATCH --error=logs/crc_molecular_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=team@medical-ai.org

# CRC Molecular Subtype Classification - Distributed Training
# This script launches multi-node distributed training on the cluster

echo "Starting CRC Molecular Subtype Distributed Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks: $SLURM_NTASKS"
date

# Load required modules
module purge
module load cuda/11.8
module load python/3.11
module load openmpi/4.1.4

# Set up environment
export PYTHONPATH=$PWD:$PYTHONPATH
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Distributed training environment
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# Create necessary directories
mkdir -p logs
mkdir -p /scratch/checkpoints/crc_molecular
mkdir -p /scratch/tensorboard/crc_molecular
mkdir -p /scratch/cache/crc_molecular

# Activate virtual environment
source /path/to/venv/bin/activate

# Install dependencies if needed
pip install -r requirements_cluster.txt

# Log system information
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Pre-process data if not already done
if [ ! -f "data/processed/manifests/train_manifest.json" ]; then
    echo "Processing WSI data..."
    srun --ntasks=1 --nodes=1 python cluster/data/preprocess_epoc.py \
        --config cluster/configs/preprocessing_config.yaml \
        --num_workers 32
fi

# Launch distributed training
echo "Launching distributed training..."
srun python cluster/distributed_trainer.py \
    --config cluster/configs/cluster_training_config.yaml \
    --resume_from_checkpoint ${RESUME_CHECKPOINT:-none}

# Post-training evaluation
if [ $? -eq 0 ]; then
    echo "Training completed successfully. Running evaluation..."
    
    # Run comprehensive evaluation on test set
    srun --ntasks=1 --nodes=1 --gres=gpu:1 python cluster/evaluate_model.py \
        --config cluster/configs/cluster_training_config.yaml \
        --checkpoint /scratch/checkpoints/crc_molecular/best_model.pth \
        --output_dir results/evaluation_${SLURM_JOB_ID}
    
    # Generate clinical validation report
    python cluster/clinical_validation.py \
        --results_dir results/evaluation_${SLURM_JOB_ID} \
        --output_report results/clinical_report_${SLURM_JOB_ID}.pdf
fi

echo "Job completed at $(date)" 