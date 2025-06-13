from pathlib import Path
import torch

class Config:
    """Configuration settings for the pipeline"""
    # WSI Processing
    tile_size = 512
    stride = 256
    magnification = 20
    min_tissue_percent = 0.5
    
    # Model settings
    feature_dim = 2048  # ResNet50 output
    hidden_dim = 512
    attention_dim = 256
    n_classes = 4
    dropout = 0.25
    
    # Training
    batch_size = 32
    learning_rate = 1e-4
    epochs = 20  # Reduced for faster training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Feature Extraction
    USE_RADIOMICS = True
    USE_DEEP_FEATURES = True
    USE_HANDCRAFTED = True
    
    # Molecular Subtypes (from literature)
    SUBTYPES = {
        0: "Canonical",  # E2F/MYC activation
        1: "Immune",     # MSI-like with immune activation
        2: "Stromal",    # VEGFA amplification, mesenchymal
        3: "Mixed"       # Mixed molecular features
    }
    
    # Clinical Risk Factors
    CRS_PARAMS = [
        'node_positive_primary',
        'disease_free_interval_months',
        'cea_level',
        'num_metastases',
        'largest_metastasis_cm'
    ]
    
    # Paths
    model_path = Path("models")
    results_path = Path("results")
    temp_path = Path("temp")
    
    def __init__(self):
        self.model_path.mkdir(exist_ok=True)
        self.results_path.mkdir(exist_ok=True)
        self.temp_path.mkdir(exist_ok=True) 