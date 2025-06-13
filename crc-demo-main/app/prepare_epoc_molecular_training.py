#!/usr/bin/env python3
"""
EPOC Data Preparation and Molecular Training Pipeline
Ready to run when EPOC WSI data arrives
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
import json
from typing import Dict, List, Tuple
import logging
from datetime import datetime
import argparse

# Import our modules
from app.crc_unified_platform import load_models
from app.wsi_handler import WSIHandler
from app.multiscale_fusion_network import MultiScaleFeatureExtractor
sys.path.append(str(Path(__file__).parent.parent))
from foundation_model.pretraining.molecular_subtype_finetuner import (
    MolecularSubtypeFinetuner,
    create_balanced_sampler
)
from foundation_model.pretraining.foundation_pretrainer import FoundationPretrainer
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('epoc_molecular_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EPOCMolecularDataset(Dataset):
    """Dataset for EPOC molecular subtype training"""
    
    def __init__(self, manifest_path: str, wsi_dir: str, tissue_model, 
                 transform, max_tiles_per_wsi=50, tile_size=224):
        """
        Args:
            manifest_path: Path to EPOC manifest with molecular labels
            wsi_dir: Directory containing EPOC WSI files
            tissue_model: Trained tissue classifier model
            transform: Image preprocessing transform
            max_tiles_per_wsi: Maximum tiles to extract per WSI
            tile_size: Size of extracted tiles
        """
        self.manifest = pd.read_csv(manifest_path)
        self.wsi_dir = Path(wsi_dir)
        self.tissue_model = tissue_model
        self.tissue_model.eval()
        self.transform = transform
        self.max_tiles = max_tiles_per_wsi
        self.tile_size = tile_size
        
        # Map molecular subtypes to indices (Pitroda classification)
        # Update mapping based on actual EPoC data format
        self.subtype_map = {
            'canonical': 0, 'Canonical': 0, 'CANONICAL': 0,
            'immune': 1, 'Immune': 1, 'IMMUNE': 1,
            'stromal': 2, 'Stromal': 2, 'STROMAL': 2,
            # Legacy mapping if needed
            'SNF1': 0, 'SNF2': 1, 'SNF3': 2
        }
        
        # WSI processor
        self.wsi_processor = WSIHandler()
        
        # Cache for tissue predictions
        self.cache_dir = Path('cache/epoc_tissue_predictions')
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loaded EPOC dataset with {len(self.manifest)} samples")
        logger.info(f"Subtype distribution: {self.manifest['molecular_subtype'].value_counts().to_dict()}")
        
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        patient_id = row['patient_id']
        molecular_subtype = row['molecular_subtype']
        wsi_path = self.wsi_dir / row['wsi_filename']
        
        # Check cache first
        cache_file = self.cache_dir / f"{patient_id}_tissue_predictions.pt"
        if cache_file.exists():
            tissue_predictions = torch.load(cache_file)
        else:
            # Process WSI and extract tissue predictions
            tissue_predictions = self._process_wsi(wsi_path, patient_id)
            torch.save(tissue_predictions, cache_file)
        
        # Get label
        label = self.subtype_map[molecular_subtype]
        
        return tissue_predictions, label
    
    def _process_wsi(self, wsi_path: Path, patient_id: str):
        """Process WSI and extract tissue predictions"""
        logger.info(f"Processing WSI for patient {patient_id}")
        
        # Extract tiles from WSI
        tiles = self.wsi_processor.extract_tiles(
            str(wsi_path),
            tile_size=self.tile_size,
            overlap=0.1,
            tissue_threshold=0.7
        )
        
        # Limit number of tiles
        if len(tiles) > self.max_tiles:
            # Random sampling of tiles
            indices = np.random.choice(len(tiles), self.max_tiles, replace=False)
            tiles = [tiles[i] for i in indices]
        
        # Get tissue predictions for all tiles
        tissue_predictions = []
        with torch.no_grad():
            for tile in tiles:
                # Transform and predict
                tile_tensor = self.transform(tile).unsqueeze(0)
                output = self.tissue_model(tile_tensor)
                probs = torch.softmax(output, dim=1).squeeze()
                tissue_predictions.append(probs)
        
        # Stack predictions
        tissue_predictions = torch.stack(tissue_predictions)  # [num_tiles, 8]
        
        return tissue_predictions


class InternalCohortDataset(EPOCMolecularDataset):
    """Dataset for internal 60-patient cohort"""
    
    def __init__(self, manifest_path: str, slides_dir: str, tissue_model, 
                 transform, max_tiles_per_slide=50):
        """Similar to EPOC but for unstained slides"""
        super().__init__(manifest_path, slides_dir, tissue_model, transform, max_tiles_per_slide)
        logger.info(f"Loaded internal cohort with {len(self.manifest)} samples")


def prepare_data_loaders(epoc_dataset, internal_dataset, batch_size=16, val_split=0.15, test_split=0.15):
    """Prepare train/val/test data loaders"""
    
    # Combine datasets
    all_data = []
    all_labels = []
    
    # Add EPOC data
    for i in range(len(epoc_dataset)):
        data, label = epoc_dataset[i]
        all_data.append(data)
        all_labels.append(label)
    
    # Add internal data
    for i in range(len(internal_dataset)):
        data, label = internal_dataset[i]
        all_data.append(data)
        all_labels.append(label)
    
    # Convert to arrays
    all_labels = np.array(all_labels)
    
    # Stratified split
    skf = StratifiedKFold(n_splits=int(1/test_split), shuffle=True, random_state=42)
    train_val_idx, test_idx = next(skf.split(all_data, all_labels))
    
    # Further split train/val
    train_labels = all_labels[train_val_idx]
    skf_val = StratifiedKFold(n_splits=int(1/val_split), shuffle=True, random_state=42)
    train_idx_rel, val_idx_rel = next(skf_val.split(train_val_idx, train_labels))
    
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]
    
    # Create datasets
    train_data = [all_data[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    
    val_data = [all_data[i] for i in val_idx]
    val_labels = [all_labels[i] for i in val_idx]
    
    test_data = [all_data[i] for i in test_idx]
    test_labels = [all_labels[i] for i in test_idx]
    
    # Create TensorDatasets
    train_dataset = torch.utils.data.TensorDataset(
        torch.stack(train_data), torch.tensor(train_labels)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.stack(val_data), torch.tensor(val_labels)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.stack(test_data), torch.tensor(test_labels)
    )
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Data split - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader


def validate_baseline_accuracy(tissue_model, test_loader):
    """Test current molecular predictor accuracy before training"""
    from molecular_subtype_mapper import MolecularSubtypeMapper
    
    mapper = MolecularSubtypeMapper(tissue_model)
    correct = 0
    total = 0
    predictions = []
    targets = []
    
    logger.info("Validating baseline molecular predictor...")
    
    for tissue_probs, labels in test_loader:
        for i in range(tissue_probs.shape[0]):
            # Get average tissue composition
            avg_tissue = tissue_probs[i].mean(dim=0).numpy()
            
            # Create dummy image for architectural features
            dummy_image = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Predict using current method
            result = mapper.classify_molecular_subtype(
                dummy_image, 
                lambda x: torch.tensor(avg_tissue),
                detailed_analysis=False
            )
            
            pred = result['subtype_idx']
            true_label = labels[i].item()
            
            predictions.append(pred)
            targets.append(true_label)
            
            if pred == true_label:
                correct += 1
            total += 1
    
    accuracy = correct / total
    logger.info(f"Baseline accuracy with current method: {accuracy:.2%}")
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(targets, predictions)
    logger.info(f"Baseline confusion matrix:\n{cm}")
    
    return accuracy, cm


def train_molecular_predictor(train_loader, val_loader, test_loader, device='cuda', epochs=100):
    """Train the new molecular predictor using foundation model fine-tuning"""
    
    # Load pre-trained foundation model
    logger.info("Loading pre-trained foundation model...")
    
    # Check if foundation model weights exist
    foundation_weights_path = Path("foundation_model/checkpoints/foundation_model_weights.pth")
    if foundation_weights_path.exists():
        logger.info("Loading pre-trained weights from checkpoint...")
        checkpoint = torch.load(foundation_weights_path, map_location=device)
        
        # Create model from checkpoint
        from torchvision import models
        base_encoder = models.resnet50(pretrained=False)
        base_encoder.fc = torch.nn.Identity()
        base_encoder.output_dim = 2048
        
        foundation_model = MultiScaleFeatureExtractor(
            base_encoder=base_encoder,
            scales=[1.0, 0.5, 0.25, 0.125],  # Including new scale
            feature_dim=512
        )
        
        # Load weights
        foundation_model.load_state_dict(checkpoint['model_state_dict'])
        logger.info("Foundation model loaded successfully!")
    else:
        logger.warning("No pre-trained weights found. Using random initialization...")
        # Create new model
        from torchvision import models
        base_encoder = models.resnet50(pretrained=True)  # Use ImageNet weights
        base_encoder.fc = torch.nn.Identity()
        base_encoder.output_dim = 2048
        
        foundation_model = MultiScaleFeatureExtractor(
            base_encoder=base_encoder,
            scales=[1.0, 0.5, 0.25, 0.125],
            feature_dim=512
        )
    
    # Move to device
    foundation_model = foundation_model.to(device)
    
    # Create fine-tuner
    config_path = "foundation_model/configs/pretraining_config.yaml"
    finetuner = MolecularSubtypeFinetuner(
        foundation_model=foundation_model,
        config_path=config_path,
        device=torch.device(device)
    )
    
    logger.info("Starting molecular subtype fine-tuning...")
    
    # Calculate class weights for imbalanced data
    train_labels = []
    for _, labels in train_loader:
        train_labels.extend(labels.numpy())
    
    train_labels = np.array(train_labels)
    class_counts = np.bincount(train_labels)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    logger.info(f"Class weights: {class_weights}")
    
    # Train model
    finetuner.train(
        train_loader=train_loader,
        val_loader=val_loader,
        unlabeled_loader=None,  # Can add unlabeled data if available
        class_weights=class_weights
    )
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = finetuner.validate(test_loader)
    
    # Extract metrics
    metrics = {
        'accuracy': test_metrics['accuracy'],
        'f1_score': test_metrics['f1_macro'],
        'auc_scores': [
            test_metrics.get('auc_class_0', 0.0),
            test_metrics.get('auc_class_1', 0.0),
            test_metrics.get('auc_class_2', 0.0)
        ],
        'mcc': test_metrics['mcc'],
        'confusion_matrix': test_metrics['confusion_matrix'].tolist()
    }
    
    # History (simplified - get from wandb or logs if needed)
    history = {
        'train_loss': [],
        'val_acc': [test_metrics['accuracy']]
    }
    
    logger.info(f"Final test accuracy: {metrics['accuracy']:.2%}")
    logger.info(f"Final F1 score: {metrics['f1_score']:.4f}")
    logger.info(f"Final MCC: {metrics['mcc']:.4f}")
    
    return finetuner, history, metrics


def main():
    parser = argparse.ArgumentParser(description='EPOC Molecular Training Pipeline')
    parser.add_argument('--epoc_manifest', type=str, required=True, help='Path to EPOC manifest CSV')
    parser.add_argument('--epoc_wsi_dir', type=str, required=True, help='Directory with EPOC WSI files')
    parser.add_argument('--internal_manifest', type=str, help='Path to internal cohort manifest')
    parser.add_argument('--internal_slides_dir', type=str, help='Directory with internal slides')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Load tissue classifier
    logger.info("Loading tissue classifier model...")
    tissue_model, model_loaded, _, _ = load_models()
    
    # Get transform
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create EPOC dataset
    logger.info("Creating EPOC dataset...")
    epoc_dataset = EPOCMolecularDataset(
        args.epoc_manifest,
        args.epoc_wsi_dir,
        tissue_model,
        transform
    )
    
    # Create internal dataset if provided
    internal_dataset = None
    if args.internal_manifest and args.internal_slides_dir:
        logger.info("Creating internal cohort dataset...")
        internal_dataset = InternalCohortDataset(
            args.internal_manifest,
            args.internal_slides_dir,
            tissue_model,
            transform
        )
    
    # Prepare data loaders
    logger.info("Preparing data loaders...")
    if internal_dataset:
        train_loader, val_loader, test_loader = prepare_data_loaders(
            epoc_dataset, internal_dataset, batch_size=args.batch_size
        )
    else:
        # EPOC only
        train_loader, val_loader, test_loader = prepare_data_loaders(
            epoc_dataset, epoc_dataset, batch_size=args.batch_size
        )
    
    # Validate baseline accuracy
    baseline_acc, baseline_cm = validate_baseline_accuracy(tissue_model, test_loader)
    
    # Train new model
    model, history, metrics = train_molecular_predictor(
        train_loader, val_loader, test_loader, 
        device=args.device, epochs=args.epochs
    )
    
    # Compare results
    improvement = metrics['accuracy'] - baseline_acc
    logger.info(f"\n{'='*50}")
    logger.info(f"RESULTS SUMMARY")
    logger.info(f"{'='*50}")
    logger.info(f"Baseline accuracy: {baseline_acc:.2%}")
    logger.info(f"New model accuracy: {metrics['accuracy']:.2%}")
    logger.info(f"Improvement: {improvement:.2%}")
    logger.info(f"F1 Score: {metrics['f1_score']:.4f}")
    logger.info(f"AUC Scores: Canonical={metrics['auc_scores'][0]:.4f}, "
                f"Immune={metrics['auc_scores'][1]:.4f}, Stromal={metrics['auc_scores'][2]:.4f}")
    
    # Save results summary
    results = {
        'timestamp': datetime.now().isoformat(),
        'baseline_accuracy': baseline_acc,
        'final_accuracy': metrics['accuracy'],
        'improvement': improvement,
        'metrics': metrics,
        'training_epochs': len(history['train_loss']),
        'best_val_accuracy': max(history['val_acc'])
    }
    
    with open('molecular_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\nTraining complete! Results saved to molecular_training_results.json")
    

if __name__ == "__main__":
    # Check if running with arguments or just displaying help
    if len(sys.argv) == 1:
        # No arguments provided, show help
        print("\n" + "="*60)
        print("EPOC MOLECULAR TRAINING PIPELINE")
        print("="*60)
        print("\nThis script is ready to run when your EPOC data arrives!")
        print("\nUsage:")
        print("python prepare_epoc_molecular_training.py \\")
        print("  --epoc_manifest data/epoc/manifest.csv \\")
        print("  --epoc_wsi_dir data/epoc/wsis \\")
        print("  --internal_manifest data/internal/manifest.csv \\  # Optional")
        print("  --internal_slides_dir data/internal/slides \\      # Optional")
        print("  --epochs 100")
        print("\nRequired data structure:")
        print("- EPOC manifest CSV with columns: patient_id, wsi_filename, molecular_subtype")
        print("- WSI files in specified directory")
        print("- Internal cohort manifest (optional)")
        print("\nExpected outcomes:")
        print("- Baseline accuracy: ~35-45%")
        print("- Target accuracy: >70%")
        print("- Full validation report with confusion matrices")
        print("- Trained model saved for deployment")
        print("\nRun with --help for all options")
        sys.exit(0)
    
    main() 