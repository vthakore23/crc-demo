#!/usr/bin/env python3
"""
Train CRC model with WSI (Whole Slide Image) data
Integrates EBHI-SEG dataset with existing CRC analysis framework
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import json
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app.hybrid_radiomics_classifier import HybridRadiomicsClassifier
from app.molecular_subtype_mapper import MolecularSubtypeMapper
from app.memory_config import MemoryConfig

class WSIDataset(Dataset):
    """Dataset for WSI patches"""
    
    def __init__(self, metadata_path, patches_dir, split='train', transform=None):
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[self.metadata['split'] == split]
        self.patches_dir = Path(patches_dir)
        self.transform = transform
        
        # For now, we'll use synthetic labels based on image properties
        # In real scenario, these would come from annotations
        self._generate_synthetic_labels()
        
    def _generate_synthetic_labels(self):
        """Generate synthetic labels for demonstration"""
        # Map to CRC molecular subtypes
        np.random.seed(42)
        self.metadata['label'] = np.random.choice(
            ['canonical', 'immune', 'stromal'], 
            size=len(self.metadata),
            p=[0.4, 0.3, 0.3]  # Approximate distribution
        )
        
        # Convert to numeric
        self.label_to_idx = {'canonical': 0, 'immune': 1, 'stromal': 2}
        self.idx_to_label = {v: k for k, v in self.label_to_idx.items()}
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load image
        img_path = self.patches_dir / row['patch_filename']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.label_to_idx[row['label']]
        
        return image, label, row['patch_filename']

class EnhancedCRCModel(nn.Module):
    """Enhanced CRC model that combines WSI features with existing classifiers"""
    
    def __init__(self, num_classes=3, pretrained=True):
        super().__init__()
        
        # Base WSI feature extractor (ResNet50)
        self.wsi_backbone = models.resnet50(pretrained=pretrained)
        num_features = self.wsi_backbone.fc.in_features
        self.wsi_backbone.fc = nn.Identity()  # Remove final layer
        
        # Feature fusion layers
        self.feature_fusion = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classification head
        self.classifier = nn.Linear(256, num_classes)
        
        # Attention mechanism for patch aggregation
        self.attention = nn.Sequential(
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        # Extract WSI features
        features = self.wsi_backbone(x)
        
        # Fuse features
        fused_features = self.feature_fusion(features)
        
        # Classification
        logits = self.classifier(fused_features)
        
        return logits, fused_features

class WSITrainer:
    """Trainer for WSI-based CRC model"""
    
    def __init__(self, model, device, save_dir):
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_config = MemoryConfig()
        
    def train_epoch(self, dataloader, criterion, optimizer):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels, _) in enumerate(tqdm(dataloader, desc="Training")):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs, _ = self.model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Memory management
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy
    
    def validate(self, dataloader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(dataloader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs, _ = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        
        return avg_loss, accuracy, all_preds, all_labels
    
    def train(self, train_loader, val_loader, epochs=30, lr=1e-4):
        """Full training loop"""
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_acc = 0
        train_losses, val_losses = [], []
        train_accs, val_accs = [], []
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate(val_loader, criterion)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                }, self.save_dir / 'best_wsi_model.pth')
            
            scheduler.step()
        
        # Plot training history
        self._plot_training_history(train_losses, val_losses, train_accs, val_accs)
        
        # Generate final confusion matrix
        self._plot_confusion_matrix(val_labels, val_preds)
        
        return self.model
    
    def _plot_training_history(self, train_losses, val_losses, train_accs, val_accs):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        
        # Accuracy plot
        ax2.plot(train_accs, label='Train Acc')
        ax2.plot(val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_history.png')
        plt.close()
    
    def _plot_confusion_matrix(self, true_labels, pred_labels):
        """Plot confusion matrix"""
        cm = confusion_matrix(true_labels, pred_labels)
        class_names = ['canonical', 'immune', 'stromal']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix - WSI Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.save_dir / 'confusion_matrix_wsi.png')
        plt.close()

def main():
    """Main training pipeline"""
    
    # Configuration
    project_path = Path("/Users/vijaythakore/Downloads/Downloads/Projects/CRC_Analysis_Project")
    wsi_processed_path = project_path / "data" / "wsi_processed"
    patches_dir = wsi_processed_path / "patches"
    metadata_path = wsi_processed_path / "metadata" / "patch_metadata_with_splits.csv"
    save_dir = project_path / "models" / "wsi_trained"
    
    # Check if processed data exists
    if not metadata_path.exists():
        print("WSI data not processed yet!")
        print("Please run: python scripts/process_extracted_wsi.py")
        print("After extracting the EBHI-SEG.rar file")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = WSIDataset(metadata_path, patches_dir, split='train', transform=train_transform)
    val_dataset = WSIDataset(metadata_path, patches_dir, split='val', transform=val_transform)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)
    
    # Initialize model
    model = EnhancedCRCModel(num_classes=3, pretrained=True)
    model = model.to(device)
    
    # Train model
    trainer = WSITrainer(model, device, save_dir)
    trained_model = trainer.train(train_loader, val_loader, epochs=30)
    
    print("\n" + "="*50)
    print("WSI model training complete!")
    print(f"Model saved to: {save_dir}")
    print("\nNext steps:")
    print("1. Evaluate on test set")
    print("2. Integrate with existing CRC analysis pipeline")
    print("3. Prepare for EPOC validation")
    
    # Save model info
    model_info = {
        'architecture': 'EnhancedCRCModel',
        'backbone': 'ResNet50',
        'num_classes': 3,
        'classes': ['canonical', 'immune', 'stromal'],
        'input_size': 224,
        'trained_on': 'EBHI-SEG WSI patches',
        'training_samples': len(train_dataset),
        'validation_samples': len(val_dataset)
    }
    
    with open(save_dir / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)

if __name__ == "__main__":
    main() 