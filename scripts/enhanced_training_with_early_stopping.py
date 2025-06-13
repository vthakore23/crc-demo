#!/usr/bin/env python3
"""
Enhanced Training with Early Stopping
Implements proper early stopping and overfitting prevention for molecular subtype training
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        """
        Args:
            patience (int): Number of epochs with no improvement to wait
            min_delta (float): Minimum change to qualify as improvement
            restore_best_weights (bool): Whether to restore best weights on stop
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_metric = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, current_metric, model):
        """
        Returns True if training should stop
        """
        if self.best_metric is None:
            self.best_metric = current_metric
            self.save_checkpoint(model)
            return False
        
        # Check for improvement
        if current_metric > self.best_metric + self.min_delta:
            self.best_metric = current_metric
            self.counter = 0
            self.save_checkpoint(model)
            print(f"💾 New best validation accuracy: {current_metric:.2f}%")
            return False
        else:
            self.counter += 1
            print(f"⏰ Early stopping counter: {self.counter}/{self.patience}")
            
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    print(f"🔄 Restored best weights (val_acc: {self.best_metric:.2f}%)")
                return True
            return False
    
    def save_checkpoint(self, model):
        """Save the current best model weights"""
        self.best_weights = model.state_dict().copy()

class TrainingMonitor:
    """Monitor training for overfitting and other issues"""
    
    def __init__(self, overfitting_threshold=0.15):
        """
        Args:
            overfitting_threshold: Max allowed gap between train and val accuracy
        """
        self.overfitting_threshold = overfitting_threshold
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'overfitting_gap': [], 'learning_rate': []
        }
        
    def update(self, train_acc, val_acc, train_loss, val_loss, lr):
        """Update training history"""
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['learning_rate'].append(lr)
        
        # Calculate overfitting gap
        gap = train_acc - val_acc
        self.history['overfitting_gap'].append(gap)
        
        return self.analyze_training_state(gap)
    
    def analyze_training_state(self, overfitting_gap):
        """Analyze current training state"""
        warnings = []
        
        # Check for overfitting
        if overfitting_gap > self.overfitting_threshold * 100:
            warnings.append(f"🚨 Overfitting detected! Gap: {overfitting_gap:.1f}%")
        
        # Check for validation loss increase
        if len(self.history['val_loss']) >= 3:
            recent_val_losses = self.history['val_loss'][-3:]
            if all(recent_val_losses[i] < recent_val_losses[i+1] for i in range(len(recent_val_losses)-1)):
                warnings.append("📈 Validation loss increasing for 3 epochs")
        
        # Check for training stagnation
        if len(self.history['val_acc']) >= 5:
            recent_val_accs = self.history['val_acc'][-5:]
            if max(recent_val_accs) - min(recent_val_accs) < 0.5:
                warnings.append("😴 Training stagnating (val acc plateau)")
        
        return warnings

def train_with_early_stopping(model, train_loader, val_loader, 
                            num_epochs=50, patience=7, device='cuda'):
    """
    Train model with early stopping and overfitting prevention
    """
    print("🛡️ Training with Early Stopping & Overfitting Prevention")
    print("=" * 60)
    
    # Initialize components
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5, verbose=True
    )
    
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)
    monitor = TrainingMonitor(overfitting_threshold=0.15)
    
    model = model.to(device)
    classes = ['Tumor', 'Stroma', 'Complex', 'Lymphocytes', 
              'Debris', 'Mucosa', 'Adipose', 'Empty']
    
    print(f"📊 Training Configuration:")
    print(f"  • Device: {device}")
    print(f"  • Early Stopping Patience: {patience} epochs")
    print(f"  • Overfitting Threshold: 15%")
    print(f"  • Learning Rate Schedule: ReduceLROnPlateau")
    print()
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1:2d}/{num_epochs} [Train]')
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        class_correct = list(0. for i in range(8))
        class_total = list(0. for i in range(8))
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1:2d}/{num_epochs} [Val]')
            for inputs, labels in val_pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
                # Per-class accuracy
                c = (predicted == labels).squeeze()
                for i in range(labels.size(0)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*val_correct/val_total:.2f}%'
                })
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Update learning rate scheduler
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Monitor training progress
        warnings = monitor.update(train_acc, val_acc, avg_train_loss, avg_val_loss, current_lr)
        
        # Print epoch summary
        print(f'\n📈 Epoch {epoch+1}/{num_epochs} Summary:')
        print(f'  Train: Loss {avg_train_loss:.4f}, Acc {train_acc:.2f}%')
        print(f'  Val:   Loss {avg_val_loss:.4f}, Acc {val_acc:.2f}%')
        print(f'  Gap:   {train_acc - val_acc:.1f}% | LR: {current_lr:.2e}')
        
        # Show warnings
        if warnings:
            print("\n⚠️ Training Warnings:")
            for warning in warnings:
                print(f"  {warning}")
        
        # Per-class validation accuracy
        print('\n🎯 Per-class Validation Accuracy:')
        for i in range(8):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                status = "✅" if acc > 80 else "⚠️" if acc > 60 else "❌"
                print(f'  {status} {classes[i]:12s}: {acc:5.1f}%')
        
        # Early stopping check
        if early_stopping(val_acc, model):
            print(f"\n🛑 Early stopping triggered after {epoch+1} epochs")
            print(f"💎 Best validation accuracy: {early_stopping.best_metric:.2f}%")
            break
        
        print("-" * 60)
    
    # Save training history
    save_path = Path("results/enhanced_training_history.json")
    save_path.parent.mkdir(exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(monitor.history, f, indent=2)
    
    print(f"\n✅ Training completed!")
    print(f"📊 Training history saved: {save_path}")
    print(f"🏆 Final validation accuracy: {early_stopping.best_metric:.2f}%")
    
    return model, monitor.history

def main():
    """Example usage"""
    print("Enhanced Training with Early Stopping")
    print("This script provides proper overfitting prevention")
    print("\nKey Features:")
    print("• Early stopping with patience")
    print("• Overfitting detection")
    print("• Learning rate scheduling") 
    print("• Training state monitoring")
    print("• Best model restoration")

if __name__ == "__main__":
    main() 