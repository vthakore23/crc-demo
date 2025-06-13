#!/usr/bin/env python3
"""
Train a balanced tissue classifier for CRC analysis
This fixes the tumor bias issue and enables accurate molecular subtyping
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
from pathlib import Path
import random
from datetime import datetime
import json
from tqdm import tqdm
import cv2

# Add parent directory to path
import sys
sys.path.append('..')
from crc_unified_platform import CRCClassifier

class SyntheticTissueDataset(Dataset):
    """
    Creates synthetic tissue patches for all 8 classes with realistic patterns
    Based on histopathological characteristics of each tissue type
    """
    
    def __init__(self, samples_per_class=1000, image_size=224, augment=True, split='train'):
        self.samples_per_class = samples_per_class
        self.image_size = image_size
        self.augment = augment
        self.split = split
        self.classes = [
            'Tumor', 'Stroma', 'Complex', 'Lymphocytes',
            'Debris', 'Mucosa', 'Adipose', 'Empty'
        ]
        self.num_classes = len(self.classes)
        
        # Define color palettes and patterns for each tissue type
        self.tissue_configs = {
            'Tumor': {
                'base_colors': [(255, 192, 203), (255, 182, 193), (255, 105, 180)],
                'pattern': 'glandular',
                'cell_size': (15, 25),
                'density': 0.8,
                'organization': 'high'
            },
            'Stroma': {
                'base_colors': [(255, 228, 225), (255, 240, 245), (255, 250, 250)],
                'pattern': 'fibrous',
                'fiber_thickness': (2, 5),
                'density': 0.7,
                'direction_variance': 30
            },
            'Complex': {
                'base_colors': [(255, 200, 200), (255, 230, 230), (200, 150, 200)],
                'pattern': 'mixed',
                'components': ['tumor', 'stroma', 'lymphocytes'],
                'mixing_ratio': [0.4, 0.4, 0.2]
            },
            'Lymphocytes': {
                'base_colors': [(128, 0, 128), (148, 0, 211), (138, 43, 226)],
                'pattern': 'scattered',
                'cell_size': (3, 6),
                'density': 0.6,
                'clustering': True
            },
            'Debris': {
                'base_colors': [(139, 69, 19), (160, 82, 45), (205, 133, 63)],
                'pattern': 'irregular',
                'opacity': 0.7,
                'texture': 'grainy'
            },
            'Mucosa': {
                'base_colors': [(255, 218, 185), (255, 228, 196), (255, 239, 213)],
                'pattern': 'columnar',
                'cell_height': (20, 30),
                'organization': 'regular'
            },
            'Adipose': {
                'base_colors': [(255, 255, 224), (255, 255, 240), (255, 248, 220)],
                'pattern': 'circular',
                'cell_size': (30, 50),
                'density': 0.9,
                'border_color': (240, 240, 240)
            },
            'Empty': {
                'base_colors': [(245, 245, 245), (250, 250, 250), (255, 255, 255)],
                'pattern': 'blank',
                'noise_level': 0.1
            }
        }
        
        # Data augmentation transforms
        self.augmentation = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=90),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomResizedCrop(size=image_size, scale=(0.8, 1.0))
        ]) if augment else None
        
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return self.samples_per_class * self.num_classes
    
    def __getitem__(self, idx):
        class_idx = idx // self.samples_per_class
        class_name = self.classes[class_idx]
        
        # Generate synthetic image for this class
        image = self.generate_tissue_image(class_name)
        
        # Apply augmentation
        if self.augmentation and self.split == 'train':
            image = self.augmentation(image)
        
        # Convert to tensor and normalize
        image_tensor = self.normalize(image)
        
        return image_tensor, class_idx
    
    def generate_tissue_image(self, tissue_type):
        """Generate a synthetic tissue image based on tissue type"""
        img = Image.new('RGB', (self.image_size, self.image_size), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        config = self.tissue_configs[tissue_type]
        
        if tissue_type == 'Tumor':
            self._draw_tumor_tissue(img, draw, config)
        elif tissue_type == 'Stroma':
            self._draw_stromal_tissue(img, draw, config)
        elif tissue_type == 'Complex':
            self._draw_complex_tissue(img, draw, config)
        elif tissue_type == 'Lymphocytes':
            self._draw_lymphocyte_tissue(img, draw, config)
        elif tissue_type == 'Debris':
            self._draw_debris_tissue(img, draw, config)
        elif tissue_type == 'Mucosa':
            self._draw_mucosal_tissue(img, draw, config)
        elif tissue_type == 'Adipose':
            self._draw_adipose_tissue(img, draw, config)
        elif tissue_type == 'Empty':
            self._draw_empty_tissue(img, draw, config)
        
        # Add realistic texture and noise
        img = self._add_tissue_texture(img, tissue_type)
        
        return img
    
    def _draw_tumor_tissue(self, img, draw, config):
        """Draw tumor tissue with glandular patterns"""
        base_color = random.choice(config['base_colors'])
        
        # Create glandular structures
        num_glands = random.randint(8, 15)
        for _ in range(num_glands):
            x = random.randint(20, self.image_size - 20)
            y = random.randint(20, self.image_size - 20)
            size = random.randint(*config['cell_size'])
            
            # Gland with lumen
            draw.ellipse([x-size, y-size, x+size, y+size], 
                        fill=base_color, outline=(180, 140, 180), width=2)
            # Central lumen (white center)
            lumen_size = size // 3
            draw.ellipse([x-lumen_size, y-lumen_size, x+lumen_size, y+lumen_size], 
                        fill=(255, 255, 255))
        
        # Add some solid tumor areas
        for _ in range(random.randint(3, 6)):
            x = random.randint(10, self.image_size - 40)
            y = random.randint(10, self.image_size - 40)
            w, h = random.randint(20, 40), random.randint(20, 40)
            color_var = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in base_color)
            draw.ellipse([x, y, x+w, y+h], fill=color_var)
    
    def _draw_stromal_tissue(self, img, draw, config):
        """Draw stromal tissue with fibrous patterns"""
        base_color = random.choice(config['base_colors'])
        
        # Draw wavy fibers
        num_fibers = random.randint(20, 40)
        for _ in range(num_fibers):
            start_x = random.randint(0, self.image_size)
            start_y = random.randint(0, self.image_size)
            
            # Create wavy path
            points = []
            x, y = start_x, start_y
            angle = random.uniform(0, 360)
            
            for step in range(20):
                points.append((int(x), int(y)))
                angle += random.uniform(-config['direction_variance'], config['direction_variance'])
                x += 5 * np.cos(np.radians(angle))
                y += 5 * np.sin(np.radians(angle))
                
                if x < 0 or x > self.image_size or y < 0 or y > self.image_size:
                    break
            
            if len(points) > 2:
                thickness = random.randint(*config['fiber_thickness'])
                color_var = tuple(max(0, min(255, c + random.randint(-10, 10))) for c in base_color)
                draw.line(points, fill=color_var, width=thickness)
    
    def _draw_complex_tissue(self, img, draw, config):
        """Draw mixed tissue patterns"""
        # Divide into regions
        regions = random.randint(3, 6)
        for i in range(regions):
            component = random.choices(config['components'], weights=config['mixing_ratio'])[0]
            
            # Define region
            x = random.randint(0, self.image_size - 50)
            y = random.randint(0, self.image_size - 50)
            w = random.randint(30, 80)
            h = random.randint(30, 80)
            
            # Create sub-image for region
            region_img = Image.new('RGB', (w, h), (255, 255, 255))
            region_draw = ImageDraw.Draw(region_img)
            
            # Draw appropriate tissue type
            if component == 'tumor':
                self._draw_tumor_tissue(region_img, region_draw, self.tissue_configs['Tumor'])
            elif component == 'stroma':
                self._draw_stromal_tissue(region_img, region_draw, self.tissue_configs['Stroma'])
            else:
                self._draw_lymphocyte_tissue(region_img, region_draw, self.tissue_configs['Lymphocytes'])
            
            # Paste region with some blending
            img.paste(region_img, (x, y))
    
    def _draw_lymphocyte_tissue(self, img, draw, config):
        """Draw lymphocyte infiltration patterns"""
        base_color = random.choice(config['base_colors'])
        
        # Create clusters of small cells
        if config['clustering']:
            num_clusters = random.randint(5, 10)
            for _ in range(num_clusters):
                cx = random.randint(20, self.image_size - 20)
                cy = random.randint(20, self.image_size - 20)
                cluster_size = random.randint(15, 30)
                
                # Draw cells in cluster
                for _ in range(random.randint(10, 30)):
                    dx = random.randint(-cluster_size, cluster_size)
                    dy = random.randint(-cluster_size, cluster_size)
                    if np.sqrt(dx**2 + dy**2) <= cluster_size:
                        size = random.randint(*config['cell_size'])
                        color_var = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in base_color)
                        draw.ellipse([cx+dx-size, cy+dy-size, cx+dx+size, cy+dy+size], 
                                   fill=color_var)
        
        # Add scattered cells
        num_scattered = int(self.image_size * self.image_size * config['density'] / 200)
        for _ in range(num_scattered):
            x = random.randint(5, self.image_size - 5)
            y = random.randint(5, self.image_size - 5)
            size = random.randint(*config['cell_size'])
            color_var = tuple(max(0, min(255, c + random.randint(-20, 20))) for c in base_color)
            draw.ellipse([x-size, y-size, x+size, y+size], fill=color_var)
    
    def _draw_debris_tissue(self, img, draw, config):
        """Draw debris/necrotic patterns"""
        base_color = random.choice(config['base_colors'])
        
        # Create irregular patches
        num_patches = random.randint(10, 20)
        for _ in range(num_patches):
            x = random.randint(0, self.image_size - 30)
            y = random.randint(0, self.image_size - 30)
            
            # Irregular shape using multiple overlapping circles
            for _ in range(random.randint(3, 8)):
                dx = random.randint(-15, 15)
                dy = random.randint(-15, 15)
                size = random.randint(5, 20)
                color_var = tuple(max(0, min(255, c + random.randint(-30, 30))) for c in base_color)
                draw.ellipse([x+dx-size, y+dy-size, x+dx+size, y+dy+size], 
                           fill=color_var)
    
    def _draw_mucosal_tissue(self, img, draw, config):
        """Draw mucosal tissue with columnar epithelium"""
        base_color = random.choice(config['base_colors'])
        
        # Draw columnar cells
        cell_width = 8
        for x in range(0, self.image_size, cell_width):
            height = random.randint(*config['cell_height'])
            y_start = random.randint(self.image_size//3, 2*self.image_size//3)
            
            # Cell body
            color_var = tuple(max(0, min(255, c + random.randint(-10, 10))) for c in base_color)
            draw.rectangle([x, y_start, x+cell_width-1, y_start+height], 
                         fill=color_var, outline=(200, 180, 180))
            
            # Nucleus (darker spot near base)
            nucleus_y = y_start + height - height//4
            nucleus_color = tuple(max(0, c - 50) for c in color_var)
            draw.ellipse([x+2, nucleus_y-3, x+cell_width-2, nucleus_y+3], 
                       fill=nucleus_color)
    
    def _draw_adipose_tissue(self, img, draw, config):
        """Draw adipose tissue with fat cells"""
        base_color = random.choice(config['base_colors'])
        border_color = config['border_color']
        
        # Create hexagonal packing of adipocytes
        cell_radius = random.randint(*config['cell_size']) // 2
        
        for y in range(0, self.image_size, int(cell_radius * 1.5)):
            for x in range(0, self.image_size, int(cell_radius * 2)):
                # Offset every other row
                offset = cell_radius if (y // int(cell_radius * 1.5)) % 2 else 0
                cx = x + offset
                cy = y
                
                if cx - cell_radius >= 0 and cx + cell_radius <= self.image_size:
                    # Draw adipocyte
                    draw.ellipse([cx-cell_radius, cy-cell_radius, 
                                cx+cell_radius, cy+cell_radius],
                               fill=base_color, outline=border_color, width=2)
                    
                    # Add small nucleus at periphery
                    nucleus_angle = random.uniform(0, 2*np.pi)
                    nx = cx + int((cell_radius-5) * np.cos(nucleus_angle))
                    ny = cy + int((cell_radius-5) * np.sin(nucleus_angle))
                    draw.ellipse([nx-2, ny-2, nx+2, ny+2], fill=(180, 140, 180))
    
    def _draw_empty_tissue(self, img, draw, config):
        """Draw empty/background tissue"""
        base_color = random.choice(config['base_colors'])
        
        # Fill with base color
        draw.rectangle([0, 0, self.image_size, self.image_size], fill=base_color)
        
        # Add subtle noise
        img_array = np.array(img)
        noise_level = int(255 * config['noise_level'])
        noise = np.random.randint(-noise_level, noise_level, img_array.shape)
        img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
        return img
    
    def _add_tissue_texture(self, img, tissue_type):
        """Add realistic texture to tissue image"""
        # Add slight blur for realism
        if tissue_type not in ['Empty', 'Debris']:
            img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        # Convert to numpy for gradient
        img_np = np.array(img)
        
        # Add subtle gradient
        gradient = np.linspace(0.9, 1.1, self.image_size)
        gradient = np.tile(gradient[:, np.newaxis], (1, self.image_size))
        img_np = img_np * gradient[:, :, np.newaxis]
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_np)


def train_model(model, train_loader, val_loader, num_epochs=50, device='cuda'):
    """Train the tissue classifier with proper techniques"""
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0
    best_model_state = None
    
    print(f"Training on {device}")
    model = model.to(device)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
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
            
            # Update progress bar
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
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
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
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Best Val Acc: {best_val_acc:.2f}%')
        
        # Print per-class accuracy
        print('\n  Per-class Validation Accuracy:')
        classes = ['Tumor', 'Stroma', 'Complex', 'Lymphocytes', 
                  'Debris', 'Mucosa', 'Adipose', 'Empty']
        for i in range(8):
            if class_total[i] > 0:
                acc = 100 * class_correct[i] / class_total[i]
                print(f'    {classes[i]}: {acc:.2f}%')
        print()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, history


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model and generate detailed metrics"""
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating'):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Calculate metrics
    from sklearn.metrics import confusion_matrix, classification_report
    
    classes = ['Tumor', 'Stroma', 'Complex', 'Lymphocytes', 
              'Debris', 'Mucosa', 'Adipose', 'Empty']
    
    cm = confusion_matrix(all_labels, all_predictions)
    report = classification_report(all_labels, all_predictions, 
                                 target_names=classes, digits=3)
    
    return cm, report


def test_color_patches(model, device='cuda'):
    """Test model on solid color patches to check for bias"""
    model.eval()
    model = model.to(device)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_colors = {
        'Pink (tumor-like)': [255, 192, 203],
        'Purple (lymphocyte-like)': [128, 0, 128],
        'White (stroma-like)': [240, 240, 240],
        'Red (debris-like)': [255, 0, 0],
        'Yellow (adipose-like)': [255, 255, 200],
        'Beige (mucosa-like)': [255, 228, 196],
        'Gray (empty-like)': [200, 200, 200]
    }
    
    classes = ['Tumor', 'Stroma', 'Complex', 'Lymphocytes', 
              'Debris', 'Mucosa', 'Adipose', 'Empty']
    
    print("\nTesting on solid color patches:")
    print("-" * 50)
    
    for color_name, rgb in test_colors.items():
        # Create solid color image
        img = np.ones((224, 224, 3), dtype=np.uint8)
        img[:, :] = rgb
        
        pil_img = Image.fromarray(img)
        img_tensor = transform(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = F.softmax(outputs, dim=1)[0]
        
        top_idx = probs.argmax().item()
        top_prob = probs[top_idx].item()
        
        print(f"\n{color_name}:")
        print(f"  Prediction: {classes[top_idx]} ({top_prob:.1%})")
        
        # Show top 3 predictions
        top3_probs, top3_idx = torch.topk(probs, 3)
        print("  Top 3:")
        for i in range(3):
            print(f"    {classes[top3_idx[i]]}: {top3_probs[i]:.1%}")


def main():
    """Main training pipeline"""
    print("=" * 80)
    print("BALANCED TISSUE CLASSIFIER TRAINING")
    print("=" * 80)
    print()
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nCreating balanced synthetic datasets...")
    train_dataset = SyntheticTissueDataset(samples_per_class=500, augment=True, split='train')
    val_dataset = SyntheticTissueDataset(samples_per_class=100, augment=False, split='val')
    test_dataset = SyntheticTissueDataset(samples_per_class=100, augment=False, split='test')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"Training samples: {len(train_dataset)} ({len(train_dataset)//8} per class)")
    print(f"Validation samples: {len(val_dataset)} ({len(val_dataset)//8} per class)")
    print(f"Test samples: {len(test_dataset)} ({len(test_dataset)//8} per class)")
    
    # Create model
    print("\nInitializing ResNet50 model...")
    model = CRCClassifier(num_classes=8, dropout_rate=0.5)
    
    # Train model
    print("\nStarting training...")
    model, history = train_model(model, train_loader, val_loader, num_epochs=30, device=device)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    cm, report = evaluate_model(model, test_loader, device=device)
    print("\nClassification Report:")
    print(report)
    
    # Test on color patches
    test_color_patches(model, device=device)
    
    # Save model
    print("\nSaving trained model...")
    save_path = Path('../models/balanced_tissue_classifier.pth')
    save_path.parent.mkdir(exist_ok=True)
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'training_history': history,
        'confusion_matrix': cm,
        'classification_report': report,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'num_classes': 8,
            'architecture': 'ResNet50',
            'training_samples_per_class': 500,
            'augmentation': True,
            'best_val_accuracy': max(history['val_acc'])
        }
    }, save_path)
    
    print(f"\nModel saved to: {save_path}")
    print(f"Best validation accuracy: {max(history['val_acc']):.2f}%")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy History')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../models/training_history.png')
    print("\nTraining history plot saved to: models/training_history.png")
    
    # Final recommendation
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print("\nTo use this model in the platform:")
    print("1. Copy the model to the correct location:")
    print("   cp models/balanced_tissue_classifier.pth models/best_tissue_classifier.pth")
    print("\n2. Test molecular predictions:")
    print("   python test_snf1_bias.py")
    print("\n3. Run the platform:")
    print("   streamlit run app/crc_unified_platform.py")


if __name__ == "__main__":
    main() 