import os
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from datetime import datetime

from config import Config
from preprocessing import WSIPreprocessor
from feature_extraction import FeatureFusion, MILFeatureAggregator
from models import AttentionMIL, TransMIL, ClassicalMLClassifier
from training import Trainer, ClinicalEvaluator

def setup_logging(config):
    """Setup logging configuration"""
    log_file = config.OUTPUT_ROOT / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

class WSIDataset(torch.utils.data.Dataset):
    """Dataset for WSI processing"""
    
    def __init__(self, wsi_paths, labels_df, config, mode='train'):
        self.wsi_paths = wsi_paths
        self.labels = labels_df.set_index('slide_id')
        self.config = config
        self.mode = mode
        
        # Initialize processors
        self.preprocessor = WSIPreprocessor(config)
        self.feature_extractor = FeatureFusion(config)
        self.feature_aggregator = MILFeatureAggregator(
            feature_dim=config.MIL_FEATURE_DIM,
            hidden_dim=config.MIL_HIDDEN_DIM,
            attention_dim=config.MIL_ATTENTION_DIM
        )
    
    def __len__(self):
        return len(self.wsi_paths)
    
    def __getitem__(self, idx):
        wsi_path = self.wsi_paths[idx]
        slide_id = Path(wsi_path).stem
        
        # Get label
        label = self.labels.loc[slide_id, 'subtype']
        
        # Process WSI
        tiles, tissue_mask = self.preprocessor.process_wsi(wsi_path)
        
        # Extract features
        features = self.feature_extractor.extract_features(tiles)
        
        # Convert to tensor
        features = torch.tensor(features, dtype=torch.float32)
        
        # Aggregate features if using MIL
        if hasattr(self.config, 'USE_MIL') and self.config.USE_MIL:
            features, _ = self.feature_aggregator.forward(features)
        
        return features, label, slide_id

def load_data(config, split='discovery'):
    """Load WSI paths and labels"""
    wsi_dir = config.DATA_ROOT / split / "wsis"
    label_file = config.DATA_ROOT / split / "labels.csv"
    clinical_file = config.DATA_ROOT / split / "clinical_data.csv"
    
    # Load WSIs
    wsi_paths = list(wsi_dir.glob("*.svs")) + list(wsi_dir.glob("*.ndpi"))
    
    # Load labels
    labels_df = pd.read_csv(label_file)
    
    # Load clinical data if available
    clinical_data = None
    if clinical_file.exists():
        clinical_data = pd.read_csv(clinical_file)
    
    return wsi_paths, labels_df, clinical_data

def train_models(config, train_dataset, val_dataset):
    """Train both deep learning and classical models"""
    results = {}
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    
    # 1. Train Attention MIL
    attention_mil = AttentionMIL(
        feature_dim=config.MIL_FEATURE_DIM,
        hidden_dim=config.MIL_HIDDEN_DIM,
        attention_dim=config.MIL_ATTENTION_DIM,
        n_classes=config.MIL_N_CLASSES
    )
    
    attention_trainer = Trainer(attention_mil, config, "attention_mil")
    attention_trainer.train(train_loader, val_loader)
    results['attention_mil'] = attention_mil
    
    # 2. Train TransMIL
    trans_mil = TransMIL(
        feature_dim=config.MIL_FEATURE_DIM,
        hidden_dim=config.MIL_HIDDEN_DIM,
        n_classes=config.MIL_N_CLASSES
    )
    
    trans_trainer = Trainer(trans_mil, config, "trans_mil")
    trans_trainer.train(train_loader, val_loader)
    results['trans_mil'] = trans_mil
    
    # 3. Train Classical Models
    classical = ClassicalMLClassifier(config)
    
    # Extract features for classical models
    X_train = []
    y_train = []
    for features, labels, _ in train_loader:
        X_train.append(features.numpy())
        y_train.append(labels.numpy())
    
    X_train = np.vstack(X_train)
    y_train = np.concatenate(y_train)
    
    classical_results = classical.train(X_train, y_train)
    results['classical'] = classical_results
    
    return results

def main():
    """Main execution function"""
    # Initialize configuration
    config = Config()
    logger = setup_logging(config)
    
    try:
        # Load discovery cohort data
        logger.info("Loading discovery cohort data...")
        train_paths, train_labels, train_clinical = load_data(config, 'discovery')
        
        # Create datasets
        train_dataset = WSIDataset(train_paths, train_labels, config, mode='train')
        
        # Train models
        logger.info("Training models...")
        model_results = train_models(config, train_dataset, None)
        
        # Load EPOC validation data
        logger.info("Loading EPOC validation data...")
        val_paths, val_labels, val_clinical = load_data(config, 'epoc')
        val_dataset = WSIDataset(val_paths, val_labels, config, mode='test')
        
        # Evaluate on EPOC cohort
        logger.info("Evaluating on EPOC cohort...")
        evaluator = ClinicalEvaluator(config)
        
        validation_results = {}
        for model_name, model in model_results.items():
            if isinstance(model, (AttentionMIL, TransMIL)):
                val_loader = DataLoader(val_dataset, batch_size=1)
                results = evaluator.evaluate_model(
                    model, val_loader, val_clinical
                )
                validation_results[model_name] = results
        
        logger.info("Pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 