# CRC Molecular Subtype Classification - Cluster-Ready Architecture

## Executive Summary

This document outlines a comprehensive reorganization of the CRC molecular subtype classification system to prepare for:
1. **EPOC Validated WSI Data Processing** at scale
2. **Distributed Training on Computing Clusters**
3. **Clinical-Grade Production Deployment**

## 🏗️ Proposed Architecture

### 1. Data Pipeline Infrastructure

#### 1.1 WSI Data Management System

```
data/
├── raw/
│   ├── epoc/
│   │   ├── wsi/              # Original WSI files (.svs, .ndpi, .mrxs)
│   │   ├── annotations/       # Pathologist annotations (QuPath/ASAP)
│   │   └── metadata/          # Clinical metadata (JSON/CSV)
│   ├── external/              # Other datasets for validation
│   └── checksums/             # Data integrity verification
├── processed/
│   ├── patches/               # Extracted patches (HDF5 format)
│   │   ├── level_0/          # 20x magnification
│   │   ├── level_1/          # 10x magnification
│   │   └── level_2/          # 5x magnification
│   ├── features/              # Pre-computed features
│   └── manifests/             # Data split manifests
└── cache/                     # Fast SSD cache for active training
```

#### 1.2 Distributed Data Processing Pipeline

```python
# config/data_pipeline.yaml
pipeline:
  stages:
    - name: quality_check
      workers: 4
      checks:
        - tissue_detection
        - focus_quality
        - stain_quality
        - artifact_detection
    
    - name: patch_extraction
      workers: 16
      params:
        patch_size: [256, 512, 1024]
        overlap: 0.25
        tissue_threshold: 0.8
        
    - name: normalization
      method: macenko
      reference: "data/reference_stain.npy"
      
    - name: feature_extraction
      models:
        - name: histossl
          weights: "pretrained/histossl_best.pth"
        - name: ciga
          weights: "pretrained/ciga_best.pth"
```

### 2. Distributed Training Infrastructure

#### 2.1 Multi-Scale Architecture

```python
# models/cluster_ready_model.py
class ClusterReadyMolecularModel(nn.Module):
    """Distributed training ready molecular subtype classifier"""
    
    def __init__(self, config):
        super().__init__()
        
        # Multi-scale feature extractors
        self.patch_encoder = PatchEncoder(
            backbone=config.backbone,
            pretrained_path=config.pretrained_path
        )
        
        # Hierarchical aggregation
        self.local_aggregator = LocalAggregator(
            method='attention',
            dim=config.feature_dim
        )
        
        self.global_aggregator = GlobalAggregator(
            method='transformer',
            num_heads=8,
            depth=4
        )
        
        # Molecular pathway heads
        self.canonical_head = PathwayHead('canonical', config)
        self.immune_head = PathwayHead('immune', config)  
        self.stromal_head = PathwayHead('stromal', config)
        
        # Uncertainty estimation
        self.uncertainty_module = EvidentialUncertainty(
            num_classes=3,
            evidence_dim=config.evidence_dim
        )
```

#### 2.2 Distributed Training Script

```python
# train_distributed.py
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

class DistributedTrainer:
    def __init__(self, config):
        self.config = config
        self.setup_distributed()
        
    def setup_distributed(self):
        dist.init_process_group(
            backend='nccl',
            init_method=self.config.init_method,
            world_size=self.config.world_size,
            rank=self.config.rank
        )
        
    def train(self):
        # Model setup
        model = ClusterReadyMolecularModel(self.config)
        model = model.to(self.config.device)
        model = DDP(model, device_ids=[self.config.local_rank])
        
        # Data loading with distributed sampler
        train_sampler = DistributedSampler(
            self.train_dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank
        )
        
        # Gradient accumulation for large batches
        accumulation_steps = self.config.effective_batch_size // (
            self.config.batch_size * self.config.world_size
        )
```

### 3. Clinical Validation Framework

#### 3.1 Cross-Validation Strategy

```python
# validation/clinical_crossval.py
class ClinicalCrossValidator:
    """Stratified cross-validation preserving patient splits"""
    
    def __init__(self, n_folds=5):
        self.n_folds = n_folds
        self.stratify_by = ['molecular_subtype', 'institution', 'stage']
        
    def create_folds(self, patient_data):
        # Ensure no patient appears in multiple folds
        # Stratify by clinical variables
        # Balance molecular subtypes across folds
        pass
```

#### 3.2 Clinical Performance Metrics

```python
# metrics/clinical_metrics.py
class ClinicalMetrics:
    """Comprehensive clinical performance evaluation"""
    
    def __init__(self):
        self.metrics = {
            'discrimination': ['auroc', 'auprc', 'f1_weighted'],
            'calibration': ['brier_score', 'ece', 'mce'],
            'clinical_utility': ['net_benefit', 'decision_curve'],
            'reliability': ['uncertainty_accuracy', 'ood_detection']
        }
        
    def compute_survival_concordance(self, predictions, outcomes):
        """C-index for survival stratification"""
        pass
```

### 4. Production Deployment System

#### 4.1 Model Registry

```yaml
# config/model_registry.yaml
models:
  - name: crc_molecular_v2.0
    version: 2.0.3
    trained_on: "EPOC_2025"
    performance:
      canonical_f1: 0.89
      immune_f1: 0.92
      stromal_f1: 0.87
    artifacts:
      weights: "s3://models/crc_molecular_v2.0.3.pth"
      config: "s3://models/crc_molecular_v2.0.3_config.json"
    clinical_validation:
      institutions: ["EPOC_A", "EPOC_B", "EPOC_C"]
      patient_count: 1847
      concordance_index: 0.76
```

#### 4.2 Clinical Integration Service

```python
# services/clinical_service.py
class ClinicalInferenceService:
    """Production inference with full audit trail"""
    
    def __init__(self, model_registry):
        self.model = self.load_validated_model(model_registry)
        self.preprocessor = ClinicalPreprocessor()
        self.postprocessor = ClinicalPostprocessor()
        
    async def predict(self, wsi_path, patient_metadata):
        # Create audit entry
        audit_id = self.create_audit_entry(wsi_path, patient_metadata)
        
        # Process WSI
        patches = await self.preprocessor.extract_patches(wsi_path)
        
        # Run inference with uncertainty
        predictions = self.model.predict_with_uncertainty(patches)
        
        # Generate clinical report
        report = self.postprocessor.generate_report(
            predictions,
            patient_metadata,
            include_heatmaps=True
        )
        
        # Log to audit trail
        self.complete_audit_entry(audit_id, report)
        
        return report
```

### 5. Monitoring and Quality Assurance

#### 5.1 Real-time Performance Monitoring

```python
# monitoring/performance_monitor.py
class PerformanceMonitor:
    """Track model performance in production"""
    
    def __init__(self):
        self.metrics_db = MetricsDatabase()
        self.alert_thresholds = {
            'accuracy_drop': 0.05,
            'confidence_shift': 0.1,
            'processing_time': 300  # seconds
        }
        
    def log_prediction(self, prediction, metadata):
        # Log to time-series database
        # Check for performance degradation
        # Alert if thresholds exceeded
        pass
```

#### 5.2 Data Drift Detection

```python
# monitoring/drift_detector.py
class DataDriftDetector:
    """Detect distribution shifts in production data"""
    
    def __init__(self, reference_stats):
        self.reference_stats = reference_stats
        self.detectors = {
            'feature_drift': KolmogorovSmirnovDetector(),
            'prediction_drift': ChiSquareDetector(),
            'uncertainty_drift': WassersteinDetector()
        }
```

### 6. Cluster Job Configuration

#### 6.1 SLURM Job Script

```bash
#!/bin/bash
#SBATCH --job-name=crc_molecular_training
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --time=48:00:00
#SBATCH --mem=512GB
#SBATCH --partition=gpu

# Load modules
module load cuda/11.8
module load python/3.11

# Setup distributed environment
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=29500
export WORLD_SIZE=32

# Launch distributed training
srun python train_distributed.py \
    --config config/cluster_training.yaml \
    --data_manifest data/processed/manifests/epoc_train.json \
    --checkpoint_dir /scratch/checkpoints \
    --tensorboard_dir /scratch/tensorboard
```

#### 6.2 Kubernetes Deployment

```yaml
# k8s/training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: crc-molecular-training
spec:
  parallelism: 4
  template:
    spec:
      containers:
      - name: trainer
        image: crc-molecular:latest
        resources:
          requests:
            nvidia.com/gpu: 8
            memory: "256Gi"
            cpu: "32"
        volumeMounts:
        - name: data
          mountPath: /data
        - name: checkpoints
          mountPath: /checkpoints
```

### 7. Key Improvements for EPOC Data

1. **Automated Quality Control**
   - Tissue quality assessment
   - Staining consistency checks
   - Artifact detection and masking

2. **Multi-Resolution Processing**
   - Hierarchical patch extraction
   - Cross-magnification features
   - Efficient memory management

3. **Clinical Metadata Integration**
   - Patient demographics
   - Treatment history
   - Molecular markers (MSI, BRAF, etc.)

4. **Interpretability Tools**
   - Attention heatmaps
   - Feature importance maps
   - Pathway activation visualization

5. **Regulatory Compliance**
   - HIPAA-compliant data handling
   - Audit trail generation
   - Model versioning and validation

## Implementation Timeline

| Phase | Duration | Key Deliverables |
|-------|----------|-----------------|
| 1. Infrastructure Setup | 2 weeks | Cluster config, distributed data pipeline |
| 2. Model Architecture | 3 weeks | Multi-scale model, uncertainty estimation |
| 3. Training Pipeline | 2 weeks | Distributed training, checkpointing |
| 4. Validation Framework | 2 weeks | Cross-validation, clinical metrics |
| 5. Production System | 3 weeks | Deployment, monitoring, API |
| 6. Clinical Testing | 4 weeks | EPOC data validation, report generation |

## Success Metrics

- **Technical**: 90%+ accuracy on EPOC validation set
- **Clinical**: C-index > 0.75 for survival stratification  
- **Operational**: <5 min processing time per WSI
- **Reliability**: 99.9% uptime, <2% prediction rejection rate

This architecture ensures scalability, clinical validity, and production readiness for the CRC molecular subtype classification system. 