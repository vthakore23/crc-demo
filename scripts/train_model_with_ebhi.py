#!/usr/bin/env python3
"""
Train CRC Model with EBHI-SEG Dataset
Prepares the model for EPOC validation using histopathological images
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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from app import memory_config

class EBHIDataset(Dataset):
    """Dataset for EBHI-SEG processed images"""
    
    def __init__(self, data_dir, split='train', transform=None, include_masks=False):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.include_masks = include_masks
        
        # Load metadata
        metadata_path = self.data_dir / 'metadata' / 'dataset_metadata.csv'
        self.metadata = pd.read_csv(metadata_path)
        self.metadata = self.metadata[self.metadata['split'] == split]
        
        # Create label mapping
        self.classes = sorted(self.metadata['subtype'].unique())
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load image
        img_path = row['processed_image']
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.class_to_idx[row['subtype']]
        
        # Load mask if requested
        if self.include_masks:
            mask_path = row['processed_mask']
            mask = Image.open(mask_path).convert('L')
            if self.transform:
                mask = self.transform(mask)
            return image, label, mask, row['filename']
        
        return image, label, row['filename']

class CRCModelForEPOC(nn.Module):
    """Enhanced CRC model specifically designed for EPOC validation"""
    
    def __init__(self, num_classes=4, pretrained=True, use_attention=True):
        super().__init__()
        
        # Use EfficientNet-B0 for better performance with histopathological images
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        
        # Feature enhancement layers
        self.feature_enhancement = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Attention mechanism for important features
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(256, 128),
                nn.Tanh(),
                nn.Linear(128, 256),
                nn.Sigmoid()
            )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Enhance features
        enhanced_features = self.feature_enhancement(features)
        
        # Apply attention if enabled
        if self.use_attention:
            attention_weights = self.attention(enhanced_features)
            enhanced_features = enhanced_features * attention_weights
        
        # Classification
        logits = self.classifier(enhanced_features)
        
        # Confidence estimation
        confidence = self.confidence_head(enhanced_features)
        
        return logits, confidence, enhanced_features

class EPOCTrainer:
    """Trainer specifically designed for EPOC validation preparation"""
    
    def __init__(self, model, device, save_dir):
        self.model = model
        self.device = device
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_config = memory_config
        self.training_history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_auc': [], 'val_auc': [],
            'learning_rates': []
        }
        
    def train_epoch(self, dataloader, criterion, optimizer, scheduler, epoch):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(dataloader, desc=f"Training Epoch {epoch}")
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            logits, confidence, _ = self.model(images)
            
            # Combined loss: classification + confidence regularization
            cls_loss = criterion(logits, labels)
            conf_loss = nn.MSELoss()(confidence.squeeze(), 
                                     (logits.max(1)[0] == labels).float())
            loss = cls_loss + 0.1 * conf_loss
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Step scheduler for OneCycleLR (must be called after each batch)
            if scheduler is not None:
                scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Store for AUC calculation
            probs = torch.softmax(logits, dim=1)
            all_probs.extend(probs.detach().cpu().numpy())
            all_preds.extend(predicted.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
            
            # Memory management
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        
        # Calculate AUC for multi-class
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        except:
            auc = 0.0
        
        return avg_loss, accuracy, auc
    
    def validate(self, dataloader, criterion, epoch):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []
        all_confidences = []
        
        with torch.no_grad():
            pbar = tqdm(dataloader, desc=f"Validation Epoch {epoch}")
            for images, labels, _ in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                logits, confidence, _ = self.model(images)
                
                # Calculate losses
                cls_loss = criterion(logits, labels)
                conf_loss = nn.MSELoss()(confidence.squeeze(), 
                                       (logits.max(1)[0] == labels).float())
                loss = cls_loss + 0.1 * conf_loss
                
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Store predictions
                probs = torch.softmax(logits, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidence.cpu().numpy())
                
                pbar.set_postfix({'acc': f'{100.*correct/total:.2f}%'})
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        
        # Calculate AUC
        try:
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
        except:
            auc = 0.0
        
        return avg_loss, accuracy, auc, all_preds, all_labels, all_confidences
    
    def train(self, train_loader, val_loader, epochs=50, lr=1e-4):
        """Full training loop with EPOC-specific optimizations"""
        
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing for better generalization
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        
        # Learning rate scheduling
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=lr*10,
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        best_val_auc = 0
        best_val_acc = 0
        patience = 10
        patience_counter = 0
        
        for epoch in range(1, epochs + 1):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch}/{epochs}")
            print('='*50)
            
            # Train
            train_loss, train_acc, train_auc = self.train_epoch(
                train_loader, criterion, optimizer, scheduler, epoch
            )
            
            # Validate
            val_loss, val_acc, val_auc, val_preds, val_labels, val_confs = self.validate(
                val_loader, criterion, epoch
            )
            
            # Store history
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['train_acc'].append(train_acc)
            self.training_history['val_acc'].append(val_acc)
            self.training_history['train_auc'].append(train_auc)
            self.training_history['val_auc'].append(val_auc)
            self.training_history['learning_rates'].append(
                optimizer.param_groups[0]['lr']
            )
            
            # Print metrics
            print(f"\nTrain - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%, AUC: {train_auc:.4f}")
            print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, AUC: {val_auc:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model based on AUC
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_auc': val_auc,
                    'training_history': self.training_history
                }, self.save_dir / 'best_epoc_model.pth')
                
                print(f"✓ New best model saved! (AUC: {val_auc:.4f})")
                
                # Generate plots for best model
                if epoch > 5:  # Wait for some epochs before plotting
                    self._generate_plots(val_labels, val_preds, val_confs, epoch)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs")
                    break
        
        # Final summary
        print("\n" + "="*50)
        print("Training Complete!")
        print(f"Best Validation AUC: {best_val_auc:.4f}")
        print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
        print("="*50)
        
        return self.model
    
    def _generate_plots(self, true_labels, pred_labels, confidences, epoch):
        """Generate comprehensive plots for model evaluation"""
        
        # 1. Training history
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.training_history['train_loss'], label='Train')
        axes[0, 0].plot(self.training_history['val_loss'], label='Val')
        axes[0, 0].set_title('Loss History')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.training_history['train_acc'], label='Train')
        axes[0, 1].plot(self.training_history['val_acc'], label='Val')
        axes[0, 1].set_title('Accuracy History')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC plot
        axes[1, 0].plot(self.training_history['train_auc'], label='Train')
        axes[1, 0].plot(self.training_history['val_auc'], label='Val')
        axes[1, 0].set_title('AUC History')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate plot
        axes[1, 1].plot(self.training_history['learning_rates'])
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / f'training_history_epoch_{epoch}.png', dpi=300)
        plt.close()
        
        # 2. Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        plt.figure(figsize=(10, 8))
        class_names = ['Canonical', 'Immune', 'Normal', 'Stromal']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.save_dir / f'confusion_matrix_epoch_{epoch}.png', dpi=300)
        plt.close()
        
        # 3. Confidence distribution
        plt.figure(figsize=(10, 6))
        confidences_flat = np.array(confidences).flatten()
        plt.hist(confidences_flat, bins=50, alpha=0.7, edgecolor='black')
        plt.axvline(confidences_flat.mean(), color='red', linestyle='--', 
                   label=f'Mean: {confidences_flat.mean():.3f}')
        plt.title('Model Confidence Distribution')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.save_dir / f'confidence_distribution_epoch_{epoch}.png', dpi=300)
        plt.close()
        
        # 4. Generate classification report
        report = classification_report(true_labels, pred_labels, 
                                     target_names=class_names,
                                     output_dict=True)
        
        # Save report as JSON
        with open(self.save_dir / f'classification_report_epoch_{epoch}.json', 'w') as f:
            json.dump(report, f, indent=2)

def main():
    """Main training pipeline for EPOC preparation"""
    
    # Configuration
    project_path = Path("/Users/vijaythakore/Downloads/Downloads/Projects/CRC_Analysis_Project")
    data_path = project_path / "data" / "ebhi_seg_processed"
    save_dir = project_path / "models" / "epoc_ready"
    
    # Check if processed data exists
    if not data_path.exists():
        print("EBHI-SEG data not processed yet!")
        print("Please run: python scripts/process_ebhi_seg.py")
        return
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Data transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = EBHIDataset(data_path, split='train', transform=train_transform)
    val_dataset = EBHIDataset(data_path, split='val', transform=val_transform)
    
    print(f"\nDataset Statistics:")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize model
    model = CRCModelForEPOC(num_classes=len(train_dataset.classes), 
                           pretrained=True, 
                           use_attention=True)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Parameters:")
    print(f"Total: {total_params:,}")
    print(f"Trainable: {trainable_params:,}")
    
    # Train model
    trainer = EPOCTrainer(model, device, save_dir)
    trained_model = trainer.train(train_loader, val_loader, epochs=50, lr=1e-4)
    
    # Save model information
    model_info = {
        'architecture': 'CRCModelForEPOC',
        'backbone': 'EfficientNet-B0',
        'num_classes': len(train_dataset.classes),
        'classes': train_dataset.classes,
        'input_size': 224,
        'trained_on': 'EBHI-SEG Dataset',
        'training_samples': len(train_dataset),
        'validation_samples': len(val_dataset),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'training_completed': datetime.now().isoformat(),
        'best_val_auc': float(max(trainer.training_history['val_auc'])),
        'best_val_acc': float(max(trainer.training_history['val_acc']))
    }
    
    with open(save_dir / 'model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("\n" + "="*70)
    print("EPOC-READY MODEL TRAINING COMPLETE!")
    print("="*70)
    print(f"\nModel saved to: {save_dir}")
    print(f"Best Validation AUC: {model_info['best_val_auc']:.4f}")
    print(f"Best Validation Accuracy: {model_info['best_val_acc']:.2f}%")
    print("\nThe model is now ready for EPOC validation!")
    print("\nNext steps:")
    print("1. Evaluate on test set: python scripts/evaluate_epoc_model.py")
    print("2. Generate EPOC submission: python scripts/generate_epoc_submission.py")
    print("3. Integrate with CRC platform for clinical deployment")

if __name__ == "__main__":
    main() 