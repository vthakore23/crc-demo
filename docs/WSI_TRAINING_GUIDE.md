# WSI Training Guide for CRC Analysis Project

## Overview

This guide documents the complete pipeline for training the CRC model using Whole Slide Images (WSI) from the EBHI-SEG dataset in preparation for EPOC validation.

## Dataset: EBHI-SEG

The EBHI-SEG (Enteroscope Biopsy Histopathological H&E Image Dataset) contains:
- **Total Images**: 4,454 histopathological images (224×224 pixels, PNG format)
- **Categories**: 6 pathological types with corresponding segmentation masks

### Category Distribution:
- Adenocarcinoma: 1,590 images → **Canonical**
- High-grade IN: 372 images → **Canonical**
- Low-grade IN: 1,276 images → **Stromal**
- Normal: 152 images → **Normal**
- Polyp: 948 images → **Stromal**
- Serrated adenoma: 116 images → **Immune**

## Pipeline Workflow

### 1. Data Processing

```bash
# Process the extracted EBHI-SEG dataset
python scripts/process_ebhi_seg.py
```

This script:
- Maps pathological categories to molecular subtypes
- Applies image preprocessing (CLAHE enhancement)
- Creates train/val/test splits (70%/15%/15%)
- Saves processed data to `data/ebhi_seg_processed/`

### 2. Model Training

```bash
# Train the EPOC-ready model
python scripts/train_model_with_ebhi.py
```

Features:
- **Architecture**: EfficientNet-B0 backbone with attention mechanism
- **Optimizations**: Label smoothing, OneCycleLR scheduling, early stopping
- **Outputs**: Model weights, training history, performance metrics
- **Location**: `models/epoc_ready/`

### 3. Model Architecture

```python
CRCModelForEPOC:
├── EfficientNet-B0 (backbone)
├── Feature Enhancement Layers
├── Attention Mechanism
├── Classification Head (4 classes)
└── Confidence Estimation Head
```

## Training Configuration

### Hyperparameters:
- **Batch Size**: 32
- **Initial LR**: 1e-4 (with OneCycleLR up to 1e-3)
- **Epochs**: 50 (with early stopping)
- **Image Size**: 224×224
- **Augmentation**: Random crops, flips, rotation, color jitter

### Data Augmentation:
- Random resized crop (80-100% scale)
- Horizontal/vertical flips
- Rotation (±20°)
- Color jitter (brightness, contrast, saturation)
- Random affine transformations

## Expected Performance

Based on the EBHI-SEG dataset characteristics:
- **Validation Accuracy**: 85-92%
- **Validation AUC**: 0.90-0.95
- **Per-class Performance**: Varies by subtype distribution

## Integration with EPOC

The trained model is specifically designed for EPOC validation:

1. **Confidence Scores**: Built-in confidence estimation for uncertainty quantification
2. **Feature Extraction**: Enhanced features suitable for downstream analysis
3. **Multi-class Support**: Handles 4 molecular subtypes + Normal class
4. **Robust Preprocessing**: Consistent with clinical histopathology standards

## Next Steps

1. **Evaluate Model**:
   ```bash
   python scripts/evaluate_epoc_model.py
   ```

2. **Generate EPOC Submission**:
   ```bash
   python scripts/generate_epoc_submission.py
   ```

3. **Clinical Integration**:
   - Load trained model in CRC unified platform
   - Apply to EPOC validation dataset
   - Generate comprehensive reports

## Model Files

After training, you'll find:
- `models/epoc_ready/best_epoc_model.pth` - Best model weights
- `models/epoc_ready/model_info.json` - Model configuration
- `models/epoc_ready/training_history_*.png` - Training plots
- `models/epoc_ready/confusion_matrix_*.png` - Performance visualization
- `models/epoc_ready/classification_report_*.json` - Detailed metrics

## Troubleshooting

### Memory Issues:
- Reduce batch size in training script
- Enable gradient checkpointing
- Use mixed precision training

### Data Loading:
- Ensure EBHI-SEG is extracted to `/Users/vijaythakore/Downloads/EBHI-SEG`
- Check file permissions
- Verify image file integrity

### Training Issues:
- Monitor loss curves for overfitting
- Adjust learning rate if loss plateaus
- Check class balance in splits

## References

- EBHI-SEG Dataset: [Kaggle Link](https://www.kaggle.com/datasets/mahdiislam/colorectal-cancer-wsi/data)
- EfficientNet: [Paper](https://arxiv.org/abs/1905.11946)
- OneCycleLR: [Super-Convergence](https://arxiv.org/abs/1708.07120) 