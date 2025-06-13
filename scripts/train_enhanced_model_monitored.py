#!/usr/bin/env python3
"""
Monitored Enhanced Training - Prevents Overfitting
Implements proper early stopping and monitoring to prevent overfitting
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import timm
from PIL import Image
import cv2
import random
import logging
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPathologyDataset(Dataset):
    """Enhanced dataset with realistic molecular subtype patterns"""
    
    def __init__(self, size=1500, transform=None, validation=False):
        self.size = size
        self.transform = transform
        self.validation = validation
        
        # More realistic distribution
        subtype_distribution = [0.35, 0.25, 0.30, 0.10]  # Canonical, Immune, Stromal, Normal
        
        self.samples = []
        for i in range(size):
            subtype = np.random.choice(4, p=subtype_distribution)
            self.samples.append({
                'subtype': subtype,
                'index': i
            })
    
    def generate_realistic_pathology_image(self, subtype):
        """Generate more realistic pathology images"""
        # Create base H&E-like image
        base = np.random.rand(224, 224, 3) * 150 + 80  # More realistic intensity range
        
        # Add subtype-specific patterns with more variation
        if subtype == 0:  # Canonical - varied glandular structures
            num_glands = random.randint(4, 10)
            for _ in range(num_glands):
                center = (random.randint(30, 194), random.randint(30, 194))
                radius = random.randint(8, 25)
                # Vary color intensity
                color_variation = random.uniform(0.8, 1.2)
                color = (
                    int(160 * color_variation), 
                    int(90 * color_variation), 
                    int(120 * color_variation)
                )
                cv2.circle(base, center, radius, color, -1)
                # Add some internal structure
                if radius > 12:
                    inner_radius = radius // 3
                    inner_color = (color[0] + 30, color[1] + 20, color[2] + 25)
                    cv2.circle(base, center, inner_radius, inner_color, -1)
                    
        elif subtype == 1:  # Immune - immune cell infiltration
            num_cells = random.randint(40, 80)
            for _ in range(num_cells):
                x, y = random.randint(5, 219), random.randint(5, 219)
                size = random.randint(2, 6)
                # Immune cells have different characteristics
                color_variation = random.uniform(0.7, 1.3)
                color = (
                    int(100 * color_variation), 
                    int(140 * color_variation), 
                    int(110 * color_variation)
                )
                cv2.circle(base, (x, y), size, color, -1)
                
        elif subtype == 2:  # Stromal - fibrous tissue patterns
            num_fibers = random.randint(8, 20)
            for _ in range(num_fibers):
                # Create more realistic fiber patterns
                start_angle = random.uniform(0, 2 * np.pi)
                length = random.randint(30, 80)
                thickness = random.randint(2, 5)
                
                start_x = random.randint(0, 224)
                start_y = random.randint(0, 224)
                end_x = int(start_x + length * np.cos(start_angle))
                end_y = int(start_y + length * np.sin(start_angle))
                
                # Clamp to image bounds
                end_x = max(0, min(224, end_x))
                end_y = max(0, min(224, end_y))
                
                color_variation = random.uniform(0.8, 1.2)
                color = (
                    int(140 * color_variation), 
                    int(110 * color_variation), 
                    int(150 * color_variation)
                )
                cv2.line(base, (start_x, start_y), (end_x, end_y), color, thickness)
                
        else:  # Normal - organized tissue
            grid_size = random.randint(10, 16)
            for i in range(0, 224, grid_size):
                for j in range(0, 224, grid_size):
                    if random.random() > 0.3:
                        cell_size = random.randint(6, grid_size-2)
                        color_variation = random.uniform(0.9, 1.1)
                        color = (
                            int(200 * color_variation), 
                            int(180 * color_variation), 
                            int(190 * color_variation)
                        )
                        cv2.rectangle(base, (i, j), (i+cell_size, j+cell_size), color, -1)
        
        # Add realistic noise and processing
        noise = np.random.normal(0, 5, base.shape)  # Reduced noise
        base = base + noise
        base = np.clip(base, 0, 255).astype(np.uint8)
        
        # Apply realistic smoothing
        base = cv2.GaussianBlur(base, (3, 3), 0.5)
        
        return base
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Generate image
        image_data = self.generate_realistic_pathology_image(sample['subtype'])
        image = Image.fromarray(image_data)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, sample['subtype'], sample['index']

class MonitoredMolecularModel(nn.Module):
    """Enhanced molecular model with better regularization"""
    
    def __init__(self, num_classes=4, dropout_rate=0.5):  # Increased dropout
        super().__init__()
        
        # Primary backbone - EfficientNet (smaller to reduce overfitting)
        self.efficientnet = timm.create_model('efficientnet_b2', pretrained=True, num_classes=0)
        
        # Secondary backbone - ResNet
        self.resnet = timm.create_model('resnet34', pretrained=True, num_classes=0)  # Smaller ResNet
        
        # Get feature dimensions
        self.efficient_dim = self.efficientnet.num_features
        self.resnet_dim = self.resnet.num_features
        
        # Feature fusion with more regularization
        self.fusion = nn.Sequential(
            nn.Linear(self.efficient_dim + self.resnet_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.8)
        )
        
        # Molecular subtype classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.6),
            nn.Linear(128, num_classes)
        )
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.4),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # Extract features from both backbones
        efficient_features = self.efficientnet(x)
        resnet_features = self.resnet(x)
        
        # Fuse features
        combined = torch.cat([efficient_features, resnet_features], dim=1)
        fused_features = self.fusion(combined)
        
        # Classification
        logits = self.classifier(fused_features)
        
        # Confidence estimation
        confidence = self.confidence_head(fused_features)
        
        return {
            'logits': logits,
            'confidence': confidence,
            'features': fused_features
        }

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

class FocalLoss(nn.Module):
    """Focal Loss for class imbalance"""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def create_transforms():
    """Create balanced augmentation transforms"""
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),  # Reduced rotation
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),  # Reduced jitter
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def monitor_training():
    """Main monitored training function"""
    logger.info("🚀 Starting Monitored Enhanced Training (Overfitting Prevention)")
    
    # Configuration
    config = {
        'batch_size': 32,  # Larger batch size for better gradients
        'learning_rate': 1e-4,
        'epochs': 100,  # More epochs but with early stopping
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
        'save_path': 'models/monitored_enhanced_model.pth',
        'early_stopping_patience': 8,
        'lr_patience': 5
    }
    
    logger.info(f"Device: {config['device']}")
    
    # Create transforms
    train_transform, val_transform = create_transforms()
    
    # Create datasets with proper train/val split
    train_dataset = EnhancedPathologyDataset(size=1200, transform=train_transform)  # Reduced size
    val_dataset = EnhancedPathologyDataset(size=300, transform=val_transform, validation=True)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    logger.info(f"Datasets created - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create model with regularization
    model = MonitoredMolecularModel(num_classes=4, dropout_rate=0.5).to(config['device'])
    total_params = sum(p.numel() for p in model.parameters())
    
    logger.info(f"Model created with {total_params:,} parameters")
    
    # Setup training with regularization
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), 
                           lr=config['learning_rate'], 
                           weight_decay=0.01)  # L2 regularization
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=config['lr_patience'], verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['early_stopping_patience'])
    
    # Training loop with monitoring
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'epoch': []}
    best_val_acc = 0
    
    logger.info("🔍 Starting monitored training with overfitting prevention...")
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target, _) in enumerate(train_loader):
            data, target = data.to(config['device']), target.to(config['device'])
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output['logits'], target)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output['logits'], 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        train_acc = 100. * train_correct / train_total
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target, _ in val_loader:
                data, target = data.to(config['device']), target.to(config['device'])
                output = model(data)
                val_loss += criterion(output['logits'], target).item()
                
                _, predicted = torch.max(output['logits'], 1)
                val_total += target.size(0)
                val_correct += (predicted == target).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_loss /= len(val_loader)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['epoch'].append(epoch)
        
        # Calculate overfitting metrics
        acc_gap = train_acc - val_acc
        loss_ratio = val_loss / train_loss if train_loss > 0 else 1.0
        
        logger.info(f'Epoch {epoch+1}/{config["epochs"]}:')
        logger.info(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logger.info(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        logger.info(f'  Acc Gap: {acc_gap:.2f}%, Loss Ratio: {loss_ratio:.3f}')
        
        # Overfitting warning
        if acc_gap > 15:
            logger.warning(f"⚠️  Large accuracy gap detected: {acc_gap:.1f}%")
        if loss_ratio > 1.5:
            logger.warning(f"⚠️  Validation loss much higher than training loss: {loss_ratio:.2f}x")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history,
                'config': config
            }, config['save_path'])
            
            logger.info(f"🎯 New best model saved: {best_val_acc:.2f}%")
        
        # Early stopping check
        if early_stopping(val_loss, model):
            logger.info(f"⏹️  Early stopping triggered at epoch {epoch+1}")
            logger.info(f"   Final validation accuracy: {val_acc:.2f}%")
            logger.info(f"   Best validation accuracy: {best_val_acc:.2f}%")
            break
    
    # Create training visualization
    create_training_visualization(history, config)
    
    logger.info("✅ Monitored training completed!")
    logger.info(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"💾 Model saved to: {config['save_path']}")
    
    return model, best_val_acc, history

def create_training_visualization(history, config):
    """Create training progress visualization"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Monitored Training Progress - Overfitting Prevention', fontsize=16, fontweight='bold')
    
    epochs = history['epoch']
    
    # Loss plot
    ax1 = axes[0, 0]
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training & Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2 = axes[0, 1]
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training & Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Overfitting metrics
    ax3 = axes[1, 0]
    acc_gaps = [t - v for t, v in zip(history['train_acc'], history['val_acc'])]
    ax3.plot(epochs, acc_gaps, 'g-', linewidth=2)
    ax3.axhline(y=10, color='orange', linestyle='--', label='Warning Threshold')
    ax3.axhline(y=15, color='red', linestyle='--', label='Overfitting Threshold')
    ax3.set_title('Accuracy Gap (Overfitting Monitor)')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Train Acc - Val Acc (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Loss ratio
    ax4 = axes[1, 1]
    loss_ratios = [v/t if t > 0 else 1.0 for t, v in zip(history['train_loss'], history['val_loss'])]
    ax4.plot(epochs, loss_ratios, 'm-', linewidth=2)
    ax4.axhline(y=1.0, color='blue', linestyle='-', alpha=0.5, label='Equal Loss')
    ax4.axhline(y=1.5, color='orange', linestyle='--', label='Warning Threshold')
    ax4.set_title('Loss Ratio (Val/Train)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Loss / Training Loss')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/monitored_training_progress.png', dpi=300, bbox_inches='tight')
    logger.info("📊 Training visualization saved to results/monitored_training_progress.png")

if __name__ == "__main__":
    # Run monitored training
    model, best_accuracy, history = monitor_training()
    
    print(f"\n🏆 MONITORED TRAINING RESULTS:")
    print(f"   Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"   Model Architecture: Monitored Multi-Scale Ensemble")
    print(f"   Overfitting Prevention: ✅ ACTIVE")
    print(f"   Early Stopping: ✅ ENABLED")
    print(f"   EPOC Readiness: {'✅ READY' if best_accuracy > 85 else '⚠️ NEEDS IMPROVEMENT'}") 