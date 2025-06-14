#!/bin/bash
#SBATCH --job-name=crc_molecular_training
#SBATCH --account=pi-username
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:8
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=logs/training_%j.out
#SBATCH --error=logs/training_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=username@uchicago.edu

# Job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node List: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"

# Load required modules
module purge
module load cuda/11.8
module load python/3.11
module load gcc/11.2.0
module load openmpi/4.1.4

# Verify modules loaded
echo "Loaded modules:"
module list

# Navigate to project directory
cd /scratch/$USER/crc_molecular_training

# Activate virtual environment
source crc_env/bin/activate

# Verify Python environment
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=4
export NCCL_DEBUG=INFO
export PYTHONPATH=/scratch/$USER/crc_molecular_training:$PYTHONPATH

# Create necessary directories
mkdir -p logs models/checkpoints results

# Run training
echo "Starting training at $(date)"
python models/training_pipeline.py \
    --config config/randi_training_config.yaml \
    --job-id $SLURM_JOB_ID \
    --output-dir results/ \
    --checkpoint-dir models/checkpoints/ \
    --log-dir logs/

# Check training completion status
if [ $? -eq 0 ]; then
    echo "Training completed successfully at $(date)"
    
    # Run post-training validation
    echo "Running post-training validation..."
    python scripts/validate_final_model.py --config config/randi_training_config.yaml
    
    # Generate training report
    echo "Generating training report..."
    python scripts/generate_report.py --job-id $SLURM_JOB_ID --output results/
    
else
    echo "Training failed at $(date)"
    echo "Check logs for error details: logs/training_${SLURM_JOB_ID}.err"
    exit 1
fi

echo "Job completed at $(date)" 