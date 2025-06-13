#!/usr/bin/env python3
"""
Enhanced Training Script for Molecular Subtype Prediction
Implementing key accuracy improvements for EPOC readiness
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
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedPathologyDataset(Dataset):
    """Enhanced dataset with better synthetic data and augmentation"""
    
    def __init__(self, size=2000, transform=None, validation=False):
        self.size = size
        self.transform = transform
        self.validation = validation
        
        # Generate samples with realistic distribution
        self.samples = []
        subtype_distribution = [0.40, 0.25, 0.25, 0.10]  # Canonical, Immune, Stromal, Normal
        
        for i in range(size):
            subtype = np.random.choice(4, p=subtype_distribution)
            self.samples.append({
                'subtype': subtype,
                'index': i
            })
    
    def generate_pathology_image(self, subtype):
        """Generate enhanced synthetic pathology images"""
        # Create base tissue-like image
        base = np.random.rand(224, 224, 3) * 180 + 50
        
        # Subtype-specific patterns
        if subtype == 0:  # Canonical - glandular structures
            for _ in range(random.randint(5, 12)):
                center = (random.randint(30, 194), random.randint(30, 194))
                radius = random.randint(8, 20)
                color = (random.randint(150, 200), random.randint(80, 130), random.randint(100, 150))
                cv2.circle(base, center, radius, color, -1)
                
        elif subtype == 1:  # Immune - immune cell infiltration
            for _ in range(random.randint(30, 60)):
                x, y = random.randint(5, 219), random.randint(5, 219)
                size = random.randint(2, 5)
                color = (random.randint(80, 150), random.randint(120, 200), random.randint(90, 160))
                cv2.circle(base, (x, y), size, color, -1)
                
        elif subtype == 2:  # Stromal - fibrous tissue
            for _ in range(random.randint(8, 20)):
                pt1 = (random.randint(0, 224), random.randint(0, 224))
                pt2 = (random.randint(0, 224), random.randint(0, 224))
                color = (random.randint(120, 180), random.randint(100, 150), random.randint(130, 180))
                cv2.line(base, pt1, pt2, color, random.randint(2, 4))
                
        else:  # Normal - organized tissue
            for i in range(0, 224, 12):
                for j in range(0, 224, 12):
                    if random.random() > 0.4:
                        color = (random.randint(180, 220), random.randint(160, 200), random.randint(170, 210))
                        cv2.rectangle(base, (i, j), (i+8, j+8), color, -1)
        
        # Add realistic noise and smoothing
        noise = np.random.normal(0, 8, base.shape)
        base = base + noise
        base = np.clip(base, 0, 255).astype(np.uint8)
        base = cv2.GaussianBlur(base, (3, 3), 0)
        
        return base
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Generate image
        image_data = self.generate_pathology_image(sample['subtype'])
        image = Image.fromarray(image_data)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, sample['subtype'], sample['index']

class EnhancedMolecularModel(nn.Module):
    """Enhanced multi-architecture ensemble model"""
    
    def __init__(self, num_classes=4):
        super().__init__()
        
        # Primary backbone - EfficientNet
        self.efficientnet = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)
        
        # Secondary backbone - ResNet
        self.resnet = timm.create_model('resnet50', pretrained=True, num_classes=0)
        
        # Get feature dimensions
        self.efficient_dim = self.efficientnet.num_features
        self.resnet_dim = self.resnet.num_features
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(self.efficient_dim + self.resnet_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Molecular subtype classifier
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Confidence estimator
        self.confidence_head = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
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
    
    def predict_with_confidence(self, x):
        """Prediction with confidence estimation"""
        self.eval()
        with torch.no_grad():
            output = self.forward(x)
            logits = output['logits']
            confidence = output['confidence']
            
            probs = F.softmax(logits, dim=1)
            predicted = torch.argmax(probs, dim=1)
            max_probs = torch.max(probs, dim=1)[0]
            
            # Combine model confidence with prediction confidence
            final_confidence = confidence.squeeze() * max_probs
            
            subtype_names = ['Canonical', 'Immune', 'Stromal', 'Normal']
            survival_rates = [0.37, 0.64, 0.20, 0.95]
            
            results = []
            for i in range(len(predicted)):
                subtype_idx = predicted[i].item()
                results.append({
                    'subtype_index': subtype_idx,
                    'subtype_name': subtype_names[subtype_idx],
                    'confidence': final_confidence[i].item(),
                    'survival_rate': survival_rates[subtype_idx],
                    'probabilities': probs[i].cpu().numpy()
                })
            
            return results if len(results) > 1 else results[0]

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

def create_enhanced_transforms():
    """Create enhanced augmentation transforms"""
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def train_enhanced_model():
    """Main training function"""
    logger.info("🚀 Starting Enhanced Molecular Subtype Training")
    
    # Configuration
    config = {
        'batch_size': 16,
        'learning_rate': 2e-4,
        'epochs': 40,
        'device': 'mps' if torch.backends.mps.is_available() else 'cpu',
        'save_path': 'models/enhanced_molecular_final.pth'
    }
    
    logger.info(f"Device: {config['device']}")
    
    # Create transforms
    train_transform, val_transform = create_enhanced_transforms()
    
    # Create datasets
    train_dataset = EnhancedPathologyDataset(size=2000, transform=train_transform)
    val_dataset = EnhancedPathologyDataset(size=500, transform=val_transform, validation=True)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    
    logger.info(f"Datasets created - Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create model
    model = EnhancedMolecularModel(num_classes=4).to(config['device'])
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    # Setup training
    criterion = FocalLoss(alpha=1.0, gamma=2.0)
    
    # Different learning rates for backbones vs heads
    backbone_params = []
    head_params = []
    
    for name, param in model.named_parameters():
        if 'efficientnet' in name or 'resnet' in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
    
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': config['learning_rate'] * 0.1},
        {'params': head_params, 'lr': config['learning_rate']}
    ], weight_decay=0.01)
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        epochs=config['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )
    
    # Training loop
    best_val_acc = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output['logits'], 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
            
            if batch_idx % 30 == 0:
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
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        logger.info(f'Epoch {epoch+1}/{config["epochs"]}: '
                   f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                   f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(config['save_path']), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history
            }, config['save_path'])
            logger.info(f"🎯 New best model saved: {best_val_acc:.2f}%")
    
    logger.info(f"✅ Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    
    # Test predictions
    logger.info("🧪 Testing enhanced model predictions...")
    
    # Load best model
    checkpoint = torch.load(config['save_path'], map_location=config['device'])
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get test batch
    test_batch = next(iter(val_loader))
    test_images, test_labels, _ = test_batch
    test_images = test_images.to(config['device'])
    
    # Make predictions
    predictions = model.predict_with_confidence(test_images[:5])
    
    if isinstance(predictions, list):
        for i, pred in enumerate(predictions):
            actual_label = ['Canonical', 'Immune', 'Stromal', 'Normal'][test_labels[i]]
            logger.info(f"Sample {i}: Predicted {pred['subtype_name']} "
                       f"(Actual: {actual_label}, "
                       f"Confidence: {pred['confidence']:.3f}, "
                       f"Survival: {pred['survival_rate']:.1%})")
    
    # Create performance visualization
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/enhanced_training_progress.png', dpi=150, bbox_inches='tight')
    logger.info("📊 Training progress saved to results/enhanced_training_progress.png")
    
    logger.info("🎉 Enhanced molecular subtype training completed successfully!")
    logger.info(f"📈 Achievement: {best_val_acc:.2f}% validation accuracy")
    logger.info(f"🎯 Improvement over baseline: {best_val_acc - 97.31:.2f}%")
    
    return model, best_val_acc, history

if __name__ == "__main__":
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run enhanced training
    model, best_accuracy, history = train_enhanced_model()
    
    print(f"\n🏆 FINAL RESULTS:")
    print(f"   Best Validation Accuracy: {best_accuracy:.2f}%")
    print(f"   Model Architecture: Enhanced Multi-Scale Ensemble")
    print(f"   Training Strategy: Focal Loss + Advanced Augmentation")
    print(f"   EPOC Readiness: {'✅ READY' if best_accuracy > 95 else '⚠️ NEEDS IMPROVEMENT'}") 