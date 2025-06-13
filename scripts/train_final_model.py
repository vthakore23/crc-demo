#!/usr/bin/env python3
"""
FINAL TRAINING SCRIPT - CRC Molecular Subtype Model

This script provides a complete, reproducible pipeline to train the
EfficientNet-B1 model for classifying CRC molecular subtypes based on synthetic data.

The process includes:
1.  Generating synthetic histopathology-like image patterns.
2.  Defining the EfficientNet-B1 model architecture.
3.  Setting up data loaders for training and validation.
4.  Implementing a training loop with validation, loss tracking, and accuracy monitoring.
5.  Using early stopping to prevent overfitting and save the best model.
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

# --- Configuration ---
IMG_SIZE = 224
NUM_CLASSES = 3  # 0: Canonical, 1: Immune, 2: Stromal
SUBTYPE_NAMES = {0: 'Canonical', 1: 'Immune', 2: 'Stromal'}
NUM_SAMPLES = 1000
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 5
MODEL_SAVE_PATH = "models/final_crc_subtype_model.pth"
DATA_DIR = "data/synthetic_patterns"

# --- 1. Synthetic Data Generation ---
def generate_synthetic_image(subtype, size):
    """Generates a synthetic image pattern for a given subtype."""
    image = Image.new('RGB', (size, size), color='white')
    draw = ImageDraw.Draw(image)
    
    if subtype == 0:  # Canonical: Uniform, dense, small circles
        for _ in range(np.random.randint(150, 200)):
            x, y = np.random.randint(0, size, 2)
            r = np.random.randint(2, 4)
            draw.ellipse((x-r, y-r, x+r, y+r), fill=(180, 40, 140)) # Purple-ish
            
    elif subtype == 1:  # Immune: Clustered, larger circles
        num_clusters = np.random.randint(3, 6)
        for _ in range(num_clusters):
            cx, cy = np.random.randint(size*0.2, size*0.8, 2)
            for _ in range(np.random.randint(40, 60)):
                angle = np.random.rand() * 2 * np.pi
                dist = np.random.rand() * size * 0.15
                x, y = cx + dist * np.cos(angle), cy + dist * np.sin(angle)
                r = np.random.randint(3, 6)
                draw.ellipse((x-r, y-r, x+r, y+r), fill=(40, 120, 180)) # Blue-ish

    elif subtype == 2:  # Stromal: Elongated, fibrous lines
        for _ in range(np.random.randint(80, 120)):
            x1, y1 = np.random.randint(0, size, 2)
            x2 = x1 + np.random.randint(-40, 40)
            y2 = y1 + np.random.randint(-40, 40)
            width = np.random.randint(1, 3)
            draw.line((x1, y1, x2, y2), fill=(40, 180, 100), width=width) # Green-ish
            
    return image

class SyntheticDataset(Dataset):
    """PyTorch Dataset for synthetic images."""
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

# --- 2. Model Definition ---
def get_model(num_classes=NUM_CLASSES, pretrained=True):
    """Defines and returns the EfficientNet-B1 model."""
    weights = EfficientNet_B1_Weights.DEFAULT if pretrained else None
    model = efficientnet_b1(weights=weights)
    
    # Freeze backbone layers
    if pretrained:
        for param in model.parameters():
            param.requires_grad = False
            
    # Replace the classifier
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(in_features, 512),
        nn.ReLU(),
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(512, num_classes),
    )
    
    # Unfreeze classifier layers
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    return model

# --- 3. Training and Validation ---
def train_model():
    """Main function to run the training pipeline."""
    
    # Prepare data directories
    print("--- Phase 1: Generating Synthetic Data ---")
    Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
    for subtype_id, name in SUBTYPE_NAMES.items():
        subtype_dir = Path(DATA_DIR) / str(subtype_id)
        subtype_dir.mkdir(exist_ok=True)
        print(f"Generating images for subtype: {name}")
        for i in tqdm(range(NUM_SAMPLES // NUM_CLASSES)):
            img = generate_synthetic_image(subtype_id, IMG_SIZE)
            img.save(subtype_dir / f'img_{i}.png')
    print("Synthetic data generation complete.\n")

    # Setup dataset and dataloaders
    print("--- Phase 2: Setting up DataLoaders ---")
    transform = EfficientNet_B1_Weights.DEFAULT.transforms()
    
    full_dataset = SyntheticDataset(data_dir=DATA_DIR, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}\n")
    
    # Initialize model, optimizer, and loss function
    print("--- Phase 3: Initializing Model ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    optimizer = optim.Adam(model.classifier.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    print(f"Model initialized on device: {device}\n")

    # Training loop
    print("--- Phase 4: Starting Model Training ---")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_acc = 100 * correct_val / total_val
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Early stopping and model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            Path(MODEL_SAVE_PATH).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"✅ Validation loss decreased. Saving model to {MODEL_SAVE_PATH}")
        else:
            epochs_no_improve += 1
            print(f"⚠️ Validation loss did not improve. Counter: {epochs_no_improve}/{EARLY_STOPPING_PATIENCE}")
            
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f"\n🛑 Early stopping triggered after {epoch+1} epochs.")
            break
            
    print("\n--- Training Finished ---")

if __name__ == '__main__':
    train_model() 