#!/usr/bin/env python3
"""
--- ADVANCED TRAINING SCRIPT with REAL DATA ---
CRC Molecular Subtype Model - Two-Stage Transfer Learning

This script implements an advanced, two-stage training pipeline:
1.  **Pre-training:** The EfficientNet-B1 model is first pre-trained on the
    real-world EBHI-SEG histopathology dataset. This teaches the model to
    recognize genuine tissue features and cell structures.
2.  **Fine-tuning:** The feature-rich backbone from the pre-trained model is then
    fine-tuned on our synthetic molecular subtype data.

This transfer learning approach creates a more robust and clinically relevant model
than training on synthetic data alone.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
from PIL import Image, ImageDraw
from tqdm import tqdm
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# --- General Configuration ---
IMG_SIZE = 224
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRANSFORM = EfficientNet_B1_Weights.DEFAULT.transforms()

# --- Phase 1: Pre-training on EBHI-SEG Configuration ---
EBHI_DATA_DIR = "/Users/vijaythakore/Downloads/EBHI-SEG"
PRETRAIN_EPOCHS = 10
PRETRAIN_LR = 0.001
PRETRAINED_BACKBONE_SAVE_PATH = "models/pretrained_ebhi_backbone.pth"

# --- Phase 2: Fine-tuning on Synthetic Data Configuration ---
SYNTHETIC_DATA_DIR = "data/synthetic_molecular"
SYNTHETIC_NUM_SAMPLES = 1000
FINETUNE_EPOCHS = 50
FINETUNE_LR = 0.001
EARLY_STOPPING_PATIENCE = 5
FINAL_MODEL_SAVE_PATH = "models/final_crc_subtype_model_real_data.pth"

# --- Dataset Classes ---
class EBHIDataset(Dataset):
    """Dataset for the real-world EBHI-SEG histopathology images."""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.image_paths = list(self.data_dir.glob('*/image/*.png'))
        self.transform = transform
        
        # Correctly create mapping only from directories, ignoring files like readme.md
        class_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls_name.name: i for i, cls_name in enumerate(class_dirs)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        # Get the class name from the parent of the 'image' directory
        class_name = img_path.parent.parent.name
        label = self.class_to_idx[class_name]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class SyntheticMolecularDataset(Dataset):
    """Dataset for the synthetic molecular subtype images."""
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.image_paths = list(self.data_dir.glob('*/*.png'))
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = int(img_path.parent.name)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# --- Model Definition ---
def get_model(num_classes, pretrained_weights=True):
    """Defines and returns an EfficientNet-B1 model."""
    weights = EfficientNet_B1_Weights.DEFAULT if pretrained_weights else None
    model = efficientnet_b1(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(512, num_classes),
    )
    return model

# --- Helper Functions ---
def generate_synthetic_image(subtype, size):
    """Generates a synthetic image for a given molecular subtype."""
    # (Reusing the generation logic from the previous script)
    image = Image.new('RGB', (size, size), color=(245, 245, 245))
    draw = ImageDraw.Draw(image)
    if subtype == 0:  # Canonical
        for _ in range(np.random.randint(150, 200)):
            x, y = np.random.randint(0, size, 2); r = np.random.randint(2, 4)
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(180, 40, 140))
    elif subtype == 1:  # Immune
        for _ in range(np.random.randint(3, 6)):
            cx, cy = np.random.randint(size*0.2, size*0.8, 2)
            for _ in range(np.random.randint(40, 60)):
                a, d = np.random.rand()*2*np.pi, np.random.rand()*size*0.15
                x, y = cx+d*np.cos(a), cy+d*np.sin(a); r = np.random.randint(3, 6)
                draw.ellipse((x-r, y-r, x+r, y+r), fill=(40, 120, 180))
    elif subtype == 2:  # Stromal
        for _ in range(np.random.randint(80, 120)):
            x1, y1 = np.random.randint(0, size, 2); x2, y2 = x1+np.random.randint(-40,40), y1+np.random.randint(-40,40)
            draw.line((x1, y1, x2, y2), fill=(40, 180, 100), width=np.random.randint(1, 3))
    return image

# --- Training Phases ---
def run_pretraining():
    """Phase 1: Pre-train the model on the EBHI-SEG dataset."""
    print("--- Phase 1: Pre-training on Real Histopathology Data (EBHI-SEG) ---")
    
    # Setup dataset
    ebhi_dataset = EBHIDataset(data_dir=EBHI_DATA_DIR, transform=TRANSFORM)
    if not ebhi_dataset.image_paths:
        print(f"❌ Error: No images found in {EBHI_DATA_DIR}. Please ensure the dataset is present.")
        return False
        
    num_ebhi_classes = len(ebhi_dataset.class_to_idx)
    print(f"Found {len(ebhi_dataset)} images across {num_ebhi_classes} classes.")
    
    train_loader = DataLoader(ebhi_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model
    model = get_model(num_classes=num_ebhi_classes, pretrained_weights=True).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=PRETRAIN_LR) # Train all layers
    criterion = nn.CrossEntropyLoss()
    
    # Pre-training loop
    for epoch in range(PRETRAIN_EPOCHS):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Pre-train Epoch {epoch+1}/{PRETRAIN_EPOCHS}")
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=running_loss/len(progress_bar))
            
    # Save the backbone
    Path(PRETRAINED_BACKBONE_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.backbone.state_dict(), PRETRAINED_BACKBONE_SAVE_PATH)
    print(f"✅ Pre-training complete. Backbone saved to {PRETRAINED_BACKBONE_SAVE_PATH}\n")
    return True

def run_finetuning():
    """Phase 2: Fine-tune the pre-trained model on synthetic molecular data."""
    print("--- Phase 2: Fine-tuning on Synthetic Molecular Subtype Data ---")

    # Generate synthetic data
    print("Generating synthetic molecular subtype images...")
    Path(SYNTHETIC_DATA_DIR).mkdir(parents=True, exist_ok=True)
    for subtype_id, name in {0: 'Canonical', 1: 'Immune', 2: 'Stromal'}.items():
        subtype_dir = Path(SYNTHETIC_DATA_DIR) / str(subtype_id)
        subtype_dir.mkdir(exist_ok=True)
        for i in range(SYNTHETIC_NUM_SAMPLES // 3):
            img = generate_synthetic_image(subtype_id, IMG_SIZE)
            img.save(subtype_dir / f'synth_{i}.png')
            
    # Setup dataset and dataloaders
    full_dataset = SyntheticMolecularDataset(data_dir=SYNTHETIC_DATA_DIR, transform=TRANSFORM)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Synthetic data ready. Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # Initialize model and load pre-trained backbone
    model = get_model(num_classes=3, pretrained_weights=False).to(DEVICE)
    try:
        model.backbone.load_state_dict(torch.load(PRETRAINED_BACKBONE_SAVE_PATH))
        print("Successfully loaded pre-trained backbone.")
    except FileNotFoundError:
        print("⚠️ Pre-trained backbone not found. Training from scratch.")
        
    # Freeze backbone, train only classifier
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    optimizer = optim.Adam(model.classifier.parameters(), lr=FINETUNE_LR)
    criterion = nn.CrossEntropyLoss()
    
    # Fine-tuning loop
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(FINETUNE_EPOCHS):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Finetune Epoch {epoch+1}/{FINETUNE_EPOCHS} [Train]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch+1}/{FINETUNE_EPOCHS} | Val Loss: {val_loss:.4f}")
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), FINAL_MODEL_SAVE_PATH)
            print(f"✅ Validation loss decreased. Saving final model to {FINAL_MODEL_SAVE_PATH}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                print(f"🛑 Early stopping triggered after {epoch+1} epochs.")
                break
    
    print("\n--- Fine-tuning Finished ---")

if __name__ == '__main__':
    if run_pretraining():
        run_finetuning() 