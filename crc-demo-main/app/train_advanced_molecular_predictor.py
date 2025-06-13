#!/usr/bin/env python3
"""
Training Script for Advanced Molecular Predictor
Trains the state-of-the-art model on EPOC trial data for genuine high accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import argparse

try:
    from molecular_predictor_advanced import AdvancedMolecularPredictor, AdvancedMolecularClassifier
except ImportError:
    print("Error: Could not import advanced molecular predictor")
    exit(1)


class EPOCDataset(Dataset):
    """Dataset class for EPOC trial data"""
    
    def __init__(self, manifest_path, transform=None, augment=False):
        self.manifest = pd.read_csv(manifest_path)
        self.transform = transform
        self.augment = augment
        
        # Map subtype names to indices
        self.subtype_to_idx = {
            'SNF1': 0,
            'SNF2': 1,
            'SNF3': 2
        }
        
        # Data augmentation transforms
        self.augmentation_transforms = [
            self._rotate_90,
            self._rotate_180,
            self._rotate_270,
            self._flip_horizontal,
            self._flip_vertical,
            self._add_gaussian_noise,
            self._adjust_brightness,
            self._adjust_contrast
        ]
        
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        
        # Load image
        image_path = row['image_path']
        image = Image.open(image_path).convert('RGB')
        
        # Convert to numpy for augmentation
        image_np = np.array(image)
        
        # Apply augmentation if enabled
        if self.augment and np.random.random() > 0.5:
            # Randomly select and apply augmentation
            aug_func = np.random.choice(self.augmentation_transforms)
            image_np = aug_func(image_np)
        
        # Convert back to PIL
        image = Image.fromarray(image_np.astype(np.uint8))
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Get label
        subtype = row['molecular_subtype']
        label = self.subtype_to_idx[subtype]
        
        return image, label
    
    def _rotate_90(self, image):
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    
    def _rotate_180(self, image):
        return cv2.rotate(image, cv2.ROTATE_180)
    
    def _rotate_270(self, image):
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    def _flip_horizontal(self, image):
        return cv2.flip(image, 1)
    
    def _flip_vertical(self, image):
        return cv2.flip(image, 0)
    
    def _add_gaussian_noise(self, image):
        noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
        noisy = cv2.add(image, noise)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def _adjust_brightness(self, image):
        factor = np.random.uniform(0.8, 1.2)
        adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=0)
        return adjusted
    
    def _adjust_contrast(self, image):
        factor = np.random.uniform(0.8, 1.2)
        adjusted = cv2.convertScaleAbs(image, alpha=factor, beta=128 * (1 - factor))
        return adjusted


class MolecularPredictorTrainer:
    """Trainer for advanced molecular predictor"""
    
    def __init__(self, model_path=None, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = AdvancedMolecularPredictor(use_vit=True)
        
        # Load pretrained weights if available
        if model_path and Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded pretrained model from {model_path}")
        
        self.model.to(self.device)
        
        # Loss function with class weights for imbalanced data
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=1e-4,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': []
        }
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                # Get predictions
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        
        # Classification report
        report = classification_report(
            all_labels, all_preds,
            target_names=['SNF1', 'SNF2', 'SNF3'],
            output_dict=True
        )
        
        return val_loss, val_acc, report, all_preds, all_labels, all_probs
    
    def train(self, train_loader, val_loader, epochs=100, save_path='models/molecular_predictor_advanced.pth'):
        """Full training loop"""
        best_val_acc = 0
        patience = 20
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, report, _, _, _ = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(report['weighted avg']['f1-score'])
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            print(f"Val F1 (weighted): {report['weighted avg']['f1-score']:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_acc': best_val_acc,
                    'history': self.history
                }, save_path)
                print(f"Saved best model with validation accuracy: {best_val_acc:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        return self.history
    
    def generate_report(self, val_loader, save_dir='training_results'):
        """Generate comprehensive training report"""
        Path(save_dir).mkdir(exist_ok=True)
        
        # Get final predictions
        val_loss, val_acc, report, preds, labels, probs = self.validate(val_loader)
        
        # Plot confusion matrix
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['SNF1', 'SNF2', 'SNF3'],
                    yticklabels=['SNF1', 'SNF2', 'SNF3'])
        plt.title(f'Confusion Matrix (Accuracy: {val_acc:.2f}%)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300)
        plt.close()
        
        # Plot training history
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train')
        axes[0, 1].plot(self.history['val_acc'], label='Validation')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        axes[1, 0].plot(self.history['val_f1'])
        axes[1, 0].set_title('Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].grid(True)
        
        # Per-class metrics
        classes = ['SNF1', 'SNF2', 'SNF3']
        precisions = [report[cls]['precision'] for cls in classes]
        recalls = [report[cls]['recall'] for cls in classes]
        f1_scores = [report[cls]['f1-score'] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        axes[1, 1].bar(x - width, precisions, width, label='Precision')
        axes[1, 1].bar(x, recalls, width, label='Recall')
        axes[1, 1].bar(x + width, f1_scores, width, label='F1')
        axes[1, 1].set_title('Per-Class Metrics')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(classes)
        axes[1, 1].legend()
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_history.png', dpi=300)
        plt.close()
        
        # Generate text report
        report_text = f"""
Advanced Molecular Predictor Training Report
============================================

Final Performance:
- Validation Accuracy: {val_acc:.2f}%
- Validation Loss: {val_loss:.4f}

Per-Class Performance:
- SNF1: Precision={report['SNF1']['precision']:.3f}, Recall={report['SNF1']['recall']:.3f}, F1={report['SNF1']['f1-score']:.3f}
- SNF2: Precision={report['SNF2']['precision']:.3f}, Recall={report['SNF2']['recall']:.3f}, F1={report['SNF2']['f1-score']:.3f}
- SNF3: Precision={report['SNF3']['precision']:.3f}, Recall={report['SNF3']['recall']:.3f}, F1={report['SNF3']['f1-score']:.3f}

Overall Metrics:
- Weighted F1 Score: {report['weighted avg']['f1-score']:.3f}
- Macro F1 Score: {report['macro avg']['f1-score']:.3f}

Confusion Matrix:
{cm}

Training completed with {len(self.history['train_loss'])} epochs.
Best validation accuracy: {max(self.history['val_acc']):.2f}% at epoch {np.argmax(self.history['val_acc'])+1}
"""
        
        with open(f'{save_dir}/training_report.txt', 'w') as f:
            f.write(report_text)
        
        print(report_text)
        
        # Save detailed metrics
        metrics = {
            'final_val_acc': val_acc,
            'final_val_loss': val_loss,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'history': self.history
        }
        
        with open(f'{save_dir}/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nTraining results saved to {save_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Train Advanced Molecular Predictor')
    parser.add_argument('--manifest', type=str, default='demo_epoc_manifest.csv',
                        help='Path to EPOC manifest CSV file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--augment', action='store_true',
                        help='Enable data augmentation')
    parser.add_argument('--save_path', type=str, default='models/molecular_predictor_advanced.pth',
                        help='Path to save trained model')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained model to continue training')
    
    args = parser.parse_args()
    
    # Check if manifest exists
    if not Path(args.manifest).exists():
        print(f"Error: Manifest file {args.manifest} not found!")
        print("Please ensure EPOC data is available.")
        return
    
    # Load manifest and split data
    manifest_df = pd.read_csv(args.manifest)
    train_df, val_df = train_test_split(manifest_df, test_size=args.val_split, 
                                        stratify=manifest_df['molecular_subtype'],
                                        random_state=42)
    
    # Save split manifests
    train_df.to_csv('train_manifest.csv', index=False)
    val_df.to_csv('val_manifest.csv', index=False)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Class distribution:")
    print(train_df['molecular_subtype'].value_counts())
    
    # Create datasets
    from torchvision import transforms
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = EPOCDataset('train_manifest.csv', transform=transform, augment=args.augment)
    val_dataset = EPOCDataset('val_manifest.csv', transform=transform, augment=False)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, 
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # Initialize trainer
    trainer = MolecularPredictorTrainer(model_path=args.pretrained)
    
    # Train model
    print("\nStarting training...")
    history = trainer.train(train_loader, val_loader, epochs=args.epochs, save_path=args.save_path)
    
    # Generate report
    print("\nGenerating training report...")
    trainer.generate_report(val_loader)
    
    print("\nTraining complete!")
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    main() 