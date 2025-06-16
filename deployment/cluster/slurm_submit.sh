#!/bin/bash
#SBATCH --job-name=epoc_crc_training
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu
#SBATCH --output=/logs/slurm_%j.out
#SBATCH --error=/logs/slurm_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=researcher@institution.edu

# === EPOC Trial CRC Subtype Distributed Training ===
# Enterprise-grade multi-node training with fault tolerance
# Supports SLURM, PBS, and Kubernetes environments

set -e  # Exit on any error
set -u  # Exit on undefined variables

# Configuration
export MASTER_PORT=29500
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8
export PYTHONPATH="${SLURM_SUBMIT_DIR}:${PYTHONPATH:-}"

# Distributed training configuration
export WORLD_SIZE=$((SLURM_NNODES * SLURM_NTASKS_PER_NODE))
export NODE_RANK=$SLURM_NODEID
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID

# Data and checkpoint paths
DATA_ROOT="/shared/data/epoc_wsi"
CHECKPOINT_DIR="/shared/checkpoints/epoc_training_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/shared/logs/epoc_training_$(date +%Y%m%d_%H%M%S)"
CONFIG_PATH="${SLURM_SUBMIT_DIR}/deployment/cluster/config/epoc_training_config.yaml"

# Create directories
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

echo "=== EPOC CRC Subtype Training Job Started ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total world size: $WORLD_SIZE"
echo "Node rank: $NODE_RANK"
echo "Data root: $DATA_ROOT"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "Log dir: $LOG_DIR"
echo "=============================================="

# Function to setup environment
setup_environment() {
    echo "Setting up environment..."
    
    # Load modules (adjust for your cluster)
    module purge
    module load cuda/12.1
    module load python/3.10
    module load gcc/11.2.0
    module load openmpi/4.1.4
    
    # Activate virtual environment
    source /shared/envs/epoc_training/bin/activate
    
    # Install any missing dependencies
    pip install --upgrade pip
    pip install -r "${SLURM_SUBMIT_DIR}/requirements.txt"
    
    # Verify CUDA setup
    python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
    python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"
    
    echo "Environment setup complete."
}

# Function to check data availability
check_data() {
    echo "Checking data availability..."
    
    if [ ! -d "$DATA_ROOT" ]; then
        echo "ERROR: Data directory $DATA_ROOT not found!"
        exit 1
    fi
    
    # Count WSI files
    wsi_count=$(find "$DATA_ROOT" -name "*.svs" -o -name "*.ndpi" -o -name "*.tiff" -o -name "*.mrxs" | wc -l)
    echo "Found $wsi_count WSI files"
    
    if [ "$wsi_count" -eq 0 ]; then
        echo "ERROR: No WSI files found in $DATA_ROOT"
        exit 1
    fi
    
    echo "Data check complete."
}

# Function to setup monitoring
setup_monitoring() {
    echo "Setting up monitoring..."
    
    # Start GPU monitoring in background
    nvidia-smi dmon -s pucvmet -o DT > "$LOG_DIR/gpu_usage_node_${NODE_RANK}.log" &
    export GPU_MONITOR_PID=$!
    
    # Start system monitoring
    sar -u -r -n DEV 60 > "$LOG_DIR/system_usage_node_${NODE_RANK}.log" &
    export SAR_PID=$!
    
    # Create monitoring script
    cat > "$LOG_DIR/monitor.sh" << 'EOF'
#!/bin/bash
while true; do
    echo "$(date): Node $NODE_RANK Status"
    echo "GPU Memory:"
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
    echo "System Memory:"
    free -h | grep Mem
    echo "Network:"
    cat /proc/net/dev | grep -E "(ib|eth)" | head -2
    echo "---"
    sleep 300  # Log every 5 minutes
done
EOF
    chmod +x "$LOG_DIR/monitor.sh"
    "$LOG_DIR/monitor.sh" > "$LOG_DIR/detailed_monitor_node_${NODE_RANK}.log" &
    export MONITOR_PID=$!
    
    echo "Monitoring setup complete."
}

# Function to cleanup on exit
cleanup() {
    echo "Cleaning up..."
    
    # Kill monitoring processes
    [ ! -z "${GPU_MONITOR_PID:-}" ] && kill $GPU_MONITOR_PID 2>/dev/null || true
    [ ! -z "${SAR_PID:-}" ] && kill $SAR_PID 2>/dev/null || true
    [ ! -z "${MONITOR_PID:-}" ] && kill $MONITOR_PID 2>/dev/null || true
    
    # Save final status
    echo "Job completed at $(date)" >> "$LOG_DIR/job_status.log"
    
    # Compress logs
    tar -czf "$LOG_DIR/logs_node_${NODE_RANK}.tar.gz" "$LOG_DIR"/*.log
    
    echo "Cleanup complete."
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Function to run training with fault tolerance
run_training() {
    local max_retries=3
    local retry_count=0
    local success=false
    
    while [ $retry_count -lt $max_retries ] && [ "$success" = false ]; do
        echo "Training attempt $((retry_count + 1)) of $max_retries"
        
        # Check if checkpoint exists for resume
        if [ -d "$CHECKPOINT_DIR" ] && [ "$(ls -A $CHECKPOINT_DIR)" ]; then
            echo "Found existing checkpoints, resuming training..."
            RESUME_FLAG="--resume"
        else
            echo "Starting training from scratch..."
            RESUME_FLAG=""
        fi
        
        # Run training
        set +e  # Don't exit on error for retry logic
        
        if [ $NODE_RANK -eq 0 ]; then
            # Master node
            export MASTER_ADDR=$(hostname -i)
            echo "Master node address: $MASTER_ADDR"
            
            # Start Ray cluster head (for distributed WSI processing)
            ray start --head --port=6379 --dashboard-host=0.0.0.0 --dashboard-port=8265 &
            export RAY_HEAD_PID=$!
            
            # Wait for other nodes to connect
            sleep 30
        else
            # Worker nodes
            export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
            echo "Connecting to master at: $MASTER_ADDR"
            
            # Connect to Ray cluster
            ray start --address="$MASTER_ADDR:6379" &
            export RAY_WORKER_PID=$!
        fi
        
        # Launch distributed training
        srun python "${SLURM_SUBMIT_DIR}/deployment/cluster/train_distributed_enterprise.py" \
            --config "$CONFIG_PATH" \
            $RESUME_FLAG \
            --local_rank $LOCAL_RANK \
            2>&1 | tee "$LOG_DIR/training_node_${NODE_RANK}_attempt_${retry_count}.log"
        
        local exit_code=$?
        
        # Cleanup Ray
        if [ ! -z "${RAY_HEAD_PID:-}" ]; then
            kill $RAY_HEAD_PID 2>/dev/null || true
        fi
        if [ ! -z "${RAY_WORKER_PID:-}" ]; then
            kill $RAY_WORKER_PID 2>/dev/null || true
        fi
        ray stop || true
        
        set -e  # Re-enable exit on error
        
        if [ $exit_code -eq 0 ]; then
            echo "Training completed successfully!"
            success=true
        else
            echo "Training failed with exit code $exit_code"
            retry_count=$((retry_count + 1))
            
            if [ $retry_count -lt $max_retries ]; then
                echo "Retrying in 60 seconds..."
                sleep 60
                
                # Check node health before retry
                if ! nvidia-smi > /dev/null 2>&1; then
                    echo "GPU health check failed, aborting retries"
                    break
                fi
            fi
        fi
    done
    
    if [ "$success" = false ]; then
        echo "Training failed after $max_retries attempts"
        exit 1
    fi
}

# Function to validate results
validate_results() {
    echo "Validating training results..."
    
    # Check if best checkpoint exists
    if [ -f "$CHECKPOINT_DIR/best_checkpoint.pth" ]; then
        echo "Best checkpoint found: $CHECKPOINT_DIR/best_checkpoint.pth"
        
        # Run validation script
        python "${SLURM_SUBMIT_DIR}/scripts/validate_checkpoint.py" \
            --checkpoint "$CHECKPOINT_DIR/best_checkpoint.pth" \
            --config "$CONFIG_PATH" \
            --output "$LOG_DIR/final_validation.json"
        
        echo "Validation complete. Results saved to $LOG_DIR/final_validation.json"
    else
        echo "WARNING: No best checkpoint found!"
    fi
}

# Function to send notification
send_notification() {
    local status=$1
    local message=$2
    
    # Email notification
    echo "$message" | mail -s "EPOC Training Job $SLURM_JOB_ID $status" researcher@institution.edu
    
    # Slack notification (if webhook is configured)
    if [ ! -z "${SLACK_WEBHOOK_URL:-}" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"EPOC Training Job $SLURM_JOB_ID $status: $message\"}" \
            "$SLACK_WEBHOOK_URL"
    fi
}

# Main execution
main() {
    echo "Starting EPOC CRC Subtype Training Job at $(date)"
    
    # Initial setup
    setup_environment
    check_data
    setup_monitoring
    
    # Create comprehensive config for this run
    cat > "$LOG_DIR/run_config.yaml" << EOF
job_info:
  slurm_job_id: $SLURM_JOB_ID
  nodes: $SLURM_NNODES
  tasks_per_node: $SLURM_NTASKS_PER_NODE
  world_size: $WORLD_SIZE
  node_rank: $NODE_RANK
  start_time: $(date -Iseconds)
  
paths:
  data_root: $DATA_ROOT
  checkpoint_dir: $CHECKPOINT_DIR
  log_dir: $LOG_DIR
  config_path: $CONFIG_PATH
  
environment:
  cuda_version: $(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
  python_version: $(python --version 2>&1 | awk '{print $2}')
  pytorch_version: $(python -c "import torch; print(torch.__version__)")
  gpu_count: $(nvidia-smi -L | wc -l)
  
system:
  hostname: $(hostname)
  kernel: $(uname -r)
  cpu_count: $(nproc)
  memory_gb: $(free -g | awk '/^Mem:/{print $2}')
EOF
    
    # Start training
    echo "Launching distributed training..."
    send_notification "STARTED" "Training job started on $SLURM_NNODES nodes with $WORLD_SIZE total processes"
    
    if run_training; then
        echo "Training completed successfully!"
        validate_results
        send_notification "COMPLETED" "Training job completed successfully. Logs at $LOG_DIR"
        exit 0
    else
        echo "Training failed!"
        send_notification "FAILED" "Training job failed. Check logs at $LOG_DIR"
        exit 1
    fi
}

# Execute main function
main "$@" 