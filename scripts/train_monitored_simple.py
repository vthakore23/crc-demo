#!/usr/bin/env python3
"""
Simple Monitored Training - Prevents Overfitting
Clean implementation with proper monitoring and early stopping
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
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RealisticDataset(Dataset):
    """Simple but realistic dataset for molecular subtype classification"""
    
    def __init__(self, size=1000, transform=None):
        self.size = size
        self.transform = transform
        self.samples = []
        
        # Balanced distribution across 3 molecular subtypes only
        subtype_distribution = [0.33, 0.33, 0.34]  # Canonical, Immune, Stromal
        
        for i in range(size):
            subtype = np.random.choice(3, p=subtype_distribution)
            self.samples.append({'subtype': subtype, 'index': i})
    
    def generate_image(self, subtype):
        """Generate distinguishable synthetic patterns per molecular subtype"""
        base = np.random.rand(224, 224, 3) * 100 + 100
        
        if subtype == 0:  # Canonical - E2F/MYC pathway (circular patterns)
            for _ in range(random.randint(3, 8)):
                center = (random.randint(30, 194), random.randint(30, 194))
                radius = random.randint(10, 25)
                cv2.circle(base, center, radius, (200, 100, 150), -1)
                # Add inner structure to simulate cellular organization
                if radius > 15:
                    inner_radius = radius // 2
                    cv2.circle(base, center, inner_radius, (220, 120, 170), -1)
                
        elif subtype == 1:  # Immune - MSI-independent immune activation (scattered dots)
            for _ in range(random.randint(25, 60)):
                x, y = random.randint(5, 219), random.randint(5, 219)
                size = random.randint(2, 6)
                cv2.circle(base, (x, y), size, (100, 200, 120), -1)
                # Add some clustering to simulate immune infiltration
                if random.random() > 0.7:
                    for _ in range(3):
                        nearby_x = x + random.randint(-10, 10)
                        nearby_y = y + random.randint(-10, 10)
                        if 0 <= nearby_x < 224 and 0 <= nearby_y < 224:
                            cv2.circle(base, (nearby_x, nearby_y), random.randint(1, 4), (120, 220, 140), -1)
                
        elif subtype == 2:  # Stromal - EMT/VEGFA (linear/fibrous patterns)
            for _ in range(random.randint(8, 20)):
                pt1 = (random.randint(0, 224), random.randint(0, 224))
                pt2 = (random.randint(0, 224), random.randint(0, 224))
                thickness = random.randint(2, 6)
                cv2.line(base, pt1, pt2, (150, 120, 200), thickness)
                # Add branching to simulate angiogenesis
                if random.random() > 0.6:
                    mid_x, mid_y = (pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2
                    branch_pt = (mid_x + random.randint(-30, 30), mid_y + random.randint(-30, 30))
                    branch_pt = (max(0, min(223, branch_pt[0])), max(0, min(223, branch_pt[1])))
                    cv2.line(base, (mid_x, mid_y), branch_pt, (170, 140, 220), thickness//2)
        
        # Add realistic noise and processing
        noise = np.random.normal(0, 8, base.shape)
        base = np.clip(base + noise, 0, 255).astype(np.uint8)
        base = cv2.GaussianBlur(base, (3, 3), 0.5)
        
        return base
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_data = self.generate_image(sample['subtype'])
        image = Image.fromarray(image_data)
        
        if self.transform:
            image = self.transform(image)
        
        return image, sample['subtype'], sample['index']

class SimpleEnsembleModel(nn.Module):
    """Simple but effective ensemble model for 3-class molecular subtype prediction"""
    
    def __init__(self, num_classes=3):  # Changed to 3 classes
        super().__init__()
        
        # Single backbone to avoid overfitting
        self.backbone = timm.create_model('efficientnet_b1', pretrained=True, num_classes=0)
        backbone_dim = self.backbone.num_features
        
        # Simple classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(backbone_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Confidence head
        self.confidence_head = nn.Sequential(
            nn.Linear(backbone_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.classifier(features)
        confidence = self.confidence_head(features)
        
        return {
            'logits': logits,
            'confidence': confidence,
            'features': features
        }

def train_with_monitoring():
    """Main training function with monitoring"""
    logger.info("🚀 Starting Simple Monitored Training")
    
    # Configuration
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    batch_size = 32
    learning_rate = 2e-4
    epochs = 50
    
    logger.info(f"Device: {device}")
    
    # Create transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = RealisticDataset(size=800, transform=train_transform)
    val_dataset = RealisticDataset(size=200, transform=val_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Datasets: Train={len(train_dataset)}, Val={len(val_dataset)}")
    
    # Create model
    model = SimpleEnsembleModel(num_classes=3).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Training loop with monitoring
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0
    patience = 8
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target, _) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output['logits'], target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(output['logits'], 1)
            train_total += target.size(0)
            train_correct += (predicted == target).sum().item()
        
        train_acc = 100. * train_correct / train_total
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target, _ in val_loader:
                data, target = data.to(device), target.to(device)
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
        
        # Monitor overfitting
        acc_gap = train_acc - val_acc
        
        logger.info(f'Epoch {epoch+1:2d}: Train Acc: {train_acc:5.1f}%, Val Acc: {val_acc:5.1f}%, Gap: {acc_gap:4.1f}%')
        
        # Early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history
            }, 'models/simple_monitored_model.pth')
            
            logger.info(f"🎯 New best: {best_val_acc:.1f}%")
            
        else:
            patience_counter += 1
            
        # Overfitting warnings
        if acc_gap > 20:
            logger.warning(f"⚠️  Large accuracy gap: {acc_gap:.1f}%")
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"⏹️  Early stopping at epoch {epoch+1}")
            break
    
    # Create visualization
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
    plt.savefig('results/simple_training_progress.png', dpi=150)
    logger.info("📊 Saved training progress")
    
    # Test the model
    logger.info("🧪 Testing final model...")
    model.eval()
    
    test_dataset = RealisticDataset(size=400, transform=val_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    test_correct = 0
    test_total = 0
    class_correct = [0, 0, 0]
    class_total = [0, 0, 0]
    
    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output['logits'], 1)
            
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()
            
            # Per-class accuracy
            for i in range(len(target)):
                label = target[i].item()
                class_total[label] += 1
                if predicted[i] == target[i]:
                    class_correct[label] += 1
    
    test_acc = 100. * test_correct / test_total
    
    # Calculate per-class accuracies
    class_accuracies = []
    subtype_names = ['Canonical', 'Immune', 'Stromal']
    
    for i in range(3):
        if class_total[i] > 0:
            class_acc = 100. * class_correct[i] / class_total[i]
        else:
            class_acc = 0
        class_accuracies.append(class_acc)
        logger.info(f"  {subtype_names[i]}: {class_acc:.1f}%")
    
    # Save final results
    final_results = {
        'best_validation_accuracy': float(best_val_acc),
        'test_accuracy': float(test_acc),
        'class_accuracies': {
            'Canonical': float(class_accuracies[0]),
            'Immune': float(class_accuracies[1]),
            'Stromal': float(class_accuracies[2])
        },
        'training_history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'train_acc': [float(x) for x in history['train_acc']],
            'val_loss': [float(x) for x in history['val_loss']],
            'val_acc': [float(x) for x in history['val_acc']]
        },
        'model_info': {
            'architecture': 'EfficientNet-B1 Ensemble',
            'parameters': total_params,
            'device': device
        }
    }
    
    with open('results/final_training_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    logger.info("✅ Training completed successfully!")
    logger.info(f"🏆 Final Results:")
    logger.info(f"   Best Validation: {best_val_acc:.1f}%")
    logger.info(f"   Test Accuracy: {test_acc:.1f}%")
    logger.info(f"   Model: {total_params:,} parameters")
    
    return final_results

if __name__ == "__main__":
    results = train_with_monitoring()
    
    print(f"\n🎉 TRAINING COMPLETE!")
    print(f"Best Validation: {results['best_validation_accuracy']:.1f}%")
    print(f"Test Accuracy: {results['test_accuracy']:.1f}%")
    print("Per-class Performance:")
    for subtype, acc in results['class_accuracies'].items():
        print(f"  {subtype}: {acc:.1f}%") 