#!/bin/bash
#SBATCH --job-name=epoc_preprocessing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256GB
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --output=logs/preprocessing_%j.out
#SBATCH --error=logs/preprocessing_%j.err

# EPOC WSI Data Preprocessing Script
# Processes raw WSI files into multi-scale patches for training

echo "Starting EPOC WSI preprocessing"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
date

# Parse command line arguments
INPUT_DIR=""
OUTPUT_DIR=""
NUM_WORKERS=32
CONFIG_FILE="cluster/configs/preprocessing_config.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --input_dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Validate arguments
if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
    echo "Error: --input_dir and --output_dir are required"
    exit 1
fi

echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Number of workers: $NUM_WORKERS"
echo "Config file: $CONFIG_FILE"

# Load required modules
module purge
module load python/3.11

# Activate virtual environment
source crc_molecular_env/bin/activate

# Set environment variables
export PYTHONPATH=$PWD:$PYTHONPATH
export OMP_NUM_THREADS=1

# Create output directories
mkdir -p "$OUTPUT_DIR"/{patches,manifests,quality_reports}
mkdir -p logs

# Log system information
echo "Python version: $(python --version)"
echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "Available CPUs: $(nproc)"
echo "Disk space: $(df -h $OUTPUT_DIR | tail -1 | awk '{print $4}')"

# Create EPOC manifest from raw data
echo "Creating EPOC data manifest..."
python scripts/create_epoc_manifest.py \
    --wsi_dir "$INPUT_DIR" \
    --output_manifest "$OUTPUT_DIR/manifests/epoc_manifest.json" \
    --include_clinical_metadata

if [ $? -ne 0 ]; then
    echo "Error: Failed to create EPOC manifest"
    exit 1
fi

# Run WSI preprocessing pipeline
echo "Starting WSI preprocessing pipeline..."
python cluster/data/wsi_processing_pipeline.py \
    --manifest "$OUTPUT_DIR/manifests/epoc_manifest.json" \
    --output_dir "$OUTPUT_DIR" \
    --config "$CONFIG_FILE" \
    --num_workers "$NUM_WORKERS" \
    --log_level INFO

if [ $? -ne 0 ]; then
    echo "Error: WSI preprocessing failed"
    exit 1
fi

# Create training/validation/test splits
echo "Creating data splits..."
python scripts/create_data_splits.py \
    --processed_dir "$OUTPUT_DIR" \
    --split_ratios 0.7,0.15,0.15 \
    --stratify_by molecular_subtype,institution \
    --output_dir "$OUTPUT_DIR/manifests"

if [ $? -ne 0 ]; then
    echo "Error: Failed to create data splits"
    exit 1
fi

# Generate quality report
echo "Generating quality report..."
python scripts/generate_quality_report.py \
    --processed_dir "$OUTPUT_DIR" \
    --output_report "$OUTPUT_DIR/quality_reports/preprocessing_report.html"

# Generate summary statistics
echo "Generating summary statistics..."
python scripts/data_summary_stats.py \
    --processed_dir "$OUTPUT_DIR" \
    --output_file "$OUTPUT_DIR/quality_reports/data_summary.json"

# Validate processed data
echo "Validating processed data..."
python tests/test_processed_data.py \
    --data_dir "$OUTPUT_DIR" \
    --num_samples 100

if [ $? -ne 0 ]; then
    echo "Warning: Data validation found issues"
fi

# Final summary
echo ""
echo "=== PREPROCESSING SUMMARY ==="
echo "Input WSI files: $(find $INPUT_DIR -name "*.svs" -o -name "*.ndpi" -o -name "*.mrxs" | wc -l)"
echo "Processed patients: $(ls $OUTPUT_DIR/patches/*.h5 2>/dev/null | wc -l)"
echo "Total patches extracted: $(python -c "
import json
with open('$OUTPUT_DIR/manifests/train_manifest.json') as f:
    train = json.load(f)
with open('$OUTPUT_DIR/manifests/val_manifest.json') as f:
    val = json.load(f)
with open('$OUTPUT_DIR/manifests/test_manifest.json') as f:
    test = json.load(f)
print(len(train['patient_ids']) + len(val['patient_ids']) + len(test['patient_ids']))
")"
echo "Output directory size: $(du -sh $OUTPUT_DIR | cut -f1)"
echo "Processing time: $SECONDS seconds"
echo ""

echo "Preprocessing completed successfully!"
echo "Training manifests available at: $OUTPUT_DIR/manifests/"
echo "Quality reports available at: $OUTPUT_DIR/quality_reports/"

date 