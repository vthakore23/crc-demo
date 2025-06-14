# Data Preparation Guide - EPOC WSI Data

## Overview

This guide provides instructions for preparing EPOC (Evaluation of Preoperative Chemotherapy) Whole Slide Image (WSI) data for training the CRC molecular subtype classification model on the Randi cluster.

## Data Requirements

### EPOC WSI Specifications
- **Format**: Standard WSI formats (.svs, .ndpi, .tiff)
- **Resolution**: 20x or 40x magnification
- **Staining**: H&E (Hematoxylin and Eosin)
- **Size**: Typically 50,000 x 50,000 pixels or larger
- **File Size**: 500MB - 2GB per WSI

### Molecular Annotation Requirements
- **Subtypes**: Canonical, Immune, Stromal classifications
- **Format**: CSV manifest file with WSI paths and labels
- **Validation**: Molecular ground truth from clinical analysis

## Directory Structure Setup

### 1. Create Data Directories

```bash
# Navigate to project directory
cd /scratch/username/crc_molecular_training

# Create data directory structure
mkdir -p data/{raw,processed,manifests,quality_control}
mkdir -p data/processed/{patches,features,metadata}
```

### 2. Expected Directory Layout

```
data/
├── raw/                          # Original WSI files
│   ├── patient_001.svs
│   ├── patient_002.svs
│   └── ...
├── processed/                    # Processed data
│   ├── patches/                  # Extracted tissue patches
│   ├── features/                 # Extracted features
│   └── metadata/                 # Processing metadata
├── manifests/                    # Data split files
│   ├── epoc_manifest.csv         # Main data manifest
│   ├── train_split.csv           # Training data
│   ├── val_split.csv             # Validation data
│   └── test_split.csv            # Test data
└── quality_control/              # QC reports and logs
    ├── qc_report.html
    └── failed_files.txt
```

## Data Transfer

### 1. Transfer WSI Files to Randi

```bash
# Using rsync (recommended for large files)
rsync -avz --progress /path/to/local/wsi/files/ username@randi.cri.uchicago.edu:/scratch/username/crc_molecular_training/data/raw/

# Using scp for smaller transfers
scp *.svs username@randi.cri.uchicago.edu:/scratch/username/crc_molecular_training/data/raw/
```

### 2. Verify Transfer Integrity

```bash
# Check file counts
ls -1 data/raw/ | wc -l

# Check file sizes
du -sh data/raw/*

# Verify file integrity (if checksums available)
md5sum data/raw/* > data/quality_control/file_checksums.md5
```

## Manifest File Preparation

### 1. Create EPOC Manifest File

Create `data/manifests/epoc_manifest.csv` with the following format:

```csv
patient_id,wsi_path,molecular_subtype,clinical_stage,treatment_response,survival_months
EPOC_001,/scratch/username/crc_molecular_training/data/raw/patient_001.svs,Canonical,T3N1M0,Complete,36
EPOC_002,/scratch/username/crc_molecular_training/data/raw/patient_002.svs,Immune,T2N0M0,Partial,42
EPOC_003,/scratch/username/crc_molecular_training/data/raw/patient_003.svs,Stromal,T4N2M1,None,18
```

### 2. Required Columns

| Column | Description | Values |
|--------|-------------|---------|
| `patient_id` | Unique patient identifier | EPOC_001, EPOC_002, etc. |
| `wsi_path` | Full path to WSI file | Absolute path on Randi |
| `molecular_subtype` | Target classification | Canonical, Immune, Stromal |
| `clinical_stage` | TNM staging (optional) | T1-4, N0-3, M0-1 |
| `treatment_response` | Response to therapy (optional) | Complete, Partial, None |
| `survival_months` | Survival data (optional) | Numeric months |

### 3. Validate Manifest File

```bash
# Check manifest format
python scripts/validate_manifest.py --manifest data/manifests/epoc_manifest.csv

# Verify file paths exist
python scripts/check_file_paths.py --manifest data/manifests/epoc_manifest.csv
```

## Data Quality Control

### 1. WSI Quality Assessment

```bash
# Run quality control checks
python scripts/quality_control.py \
    --input-dir data/raw/ \
    --output-dir data/quality_control/ \
    --manifest data/manifests/epoc_manifest.csv
```

### 2. Quality Metrics

The QC process checks for:
- **File integrity**: Readable WSI files
- **Image quality**: Focus, artifacts, staining quality
- **Tissue content**: Minimum tissue percentage (>30%)
- **Resolution**: Appropriate magnification levels
- **File size**: Within expected ranges

### 3. Handle Failed Files

```bash
# Review failed files
cat data/quality_control/failed_files.txt

# Remove failed files from manifest
python scripts/filter_manifest.py \
    --input data/manifests/epoc_manifest.csv \
    --failed data/quality_control/failed_files.txt \
    --output data/manifests/epoc_manifest_filtered.csv
```

## Data Preprocessing

### 1. Patch Extraction

```bash
# Extract tissue patches from WSI files
python scripts/extract_patches.py \
    --manifest data/manifests/epoc_manifest_filtered.csv \
    --output-dir data/processed/patches/ \
    --patch-size 256 \
    --overlap 0.25 \
    --tissue-threshold 0.3
```

### 2. Preprocessing Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `patch_size` | 256 | Patch dimensions (256x256 pixels) |
| `overlap` | 0.25 | Overlap between adjacent patches |
| `tissue_threshold` | 0.3 | Minimum tissue content per patch |
| `magnification` | 20x | Target magnification level |
| `background_threshold` | 0.8 | Background detection threshold |

### 3. Stain Normalization

```bash
# Apply H&E stain normalization
python scripts/stain_normalization.py \
    --input-dir data/processed/patches/ \
    --output-dir data/processed/patches_normalized/ \
    --method macenko \
    --reference-image data/reference/he_reference.png
```

## Data Splitting

### 1. Create Train/Validation/Test Splits

```bash
# Generate data splits
python scripts/create_splits.py \
    --manifest data/manifests/epoc_manifest_filtered.csv \
    --output-dir data/manifests/ \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15 \
    --stratify molecular_subtype \
    --random-seed 42
```

### 2. Verify Split Balance

```bash
# Check class distribution in splits
python scripts/analyze_splits.py --manifest-dir data/manifests/
```

Expected output:
```
Training set: 70% (Canonical: 45%, Immune: 30%, Stromal: 25%)
Validation set: 15% (Canonical: 44%, Immune: 31%, Stromal: 25%)
Test set: 15% (Canonical: 46%, Immune: 29%, Stromal: 25%)
```

## Storage Optimization

### 1. Data Compression

```bash
# Compress processed patches (optional)
tar -czf data/processed/patches_compressed.tar.gz data/processed/patches/

# Create HDF5 datasets for faster loading
python scripts/create_hdf5_dataset.py \
    --input-dir data/processed/patches/ \
    --output data/processed/epoc_dataset.h5 \
    --compression gzip
```

### 2. Storage Usage

Monitor storage usage:
```bash
# Check data directory sizes
du -sh data/*

# Monitor scratch space usage
df -h /scratch/username/
```

## Validation Checklist

Before proceeding to training, verify:

- [ ] All WSI files transferred successfully
- [ ] Manifest file format is correct
- [ ] File paths in manifest are valid
- [ ] Quality control passed for >90% of files
- [ ] Patch extraction completed without errors
- [ ] Data splits are balanced across molecular subtypes
- [ ] Sufficient storage space available (10TB+ recommended)
- [ ] Preprocessing logs show no critical errors

## Troubleshooting

### Common Issues

#### Large File Transfer Failures
```bash
# Resume interrupted transfers
rsync -avz --progress --partial /path/to/local/files/ username@randi.cri.uchicago.edu:/scratch/username/crc_molecular_training/data/raw/
```

#### WSI Reading Errors
```bash
# Check OpenSlide installation
python -c "import openslide; print(openslide.__version__)"

# Test individual file reading
python -c "import openslide; slide = openslide.open_slide('data/raw/patient_001.svs'); print(slide.dimensions)"
```

#### Storage Space Issues
```bash
# Clean temporary files
rm -rf /tmp/preprocessing_*

# Compress old data
tar -czf data/archive/raw_backup_$(date +%Y%m%d).tar.gz data/raw/
```

#### Manifest Validation Errors
```bash
# Check for missing files
python scripts/validate_manifest.py --manifest data/manifests/epoc_manifest.csv --verbose

# Fix path issues
sed -i 's|/old/path/|/scratch/username/crc_molecular_training/data/raw/|g' data/manifests/epoc_manifest.csv
```

## Next Steps

1. **Verify Data Preparation**: Run validation scripts
2. **Update Configuration**: Modify `config/randi_training_config.yaml` with correct data paths
3. **Test Pipeline**: Run small-scale test with subset of data
4. **Proceed to Training**: Follow `TRAINING_GUIDE.md`

---

**Note**: EPOC data contains patient information. Ensure compliance with institutional data handling policies and HIPAA requirements. 