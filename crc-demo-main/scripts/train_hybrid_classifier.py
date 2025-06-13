#!/usr/bin/env python3
"""
Training Script for Hybrid PyRadiomics-Deep Learning Classifier
Demonstrates how to train the hybrid model with combined radiomic and deep learning features
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
import json
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from app.hybrid_radiomics_classifier import HybridRadiomicsClassifier, create_clinical_report

# Configure matplotlib for non-interactive use
plt.switch_backend('Agg')


class SimpleClassifier(nn.Module):
    """Simple tissue classifier for demonstration"""
    def __init__(self, num_classes=8):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)


def load_demo_dataset(data_dir: str = "demo_data"):
    """Load demonstration dataset with synthetic molecular subtype data"""
    
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"Creating synthetic demo dataset in {data_dir}")
        create_synthetic_dataset(data_dir)
    
    # Load images and labels
    images = []
    labels = []
    patient_ids = []
    
    for subtype_idx, subtype in enumerate(['SNF1', 'SNF2', 'SNF3']):
        subtype_dir = data_path / subtype
        if subtype_dir.exists():
            for img_path in subtype_dir.glob('*.png'):
                try:
                    image = Image.open(img_path).convert('RGB')
                    images.append(np.array(image))
                    labels.append(subtype_idx)
                    patient_ids.append(img_path.stem)
                except Exception as e:
                    print(f"Warning: Could not load {img_path}: {e}")
    
    if not images:
        raise ValueError(f"No images found in {data_dir}. Please provide training data.")
    
    print(f"Loaded {len(images)} images:")
    for i, subtype in enumerate(['SNF1', 'SNF2', 'SNF3']):
        count = sum(1 for label in labels if label == i)
        print(f"  {subtype}: {count} samples")
    
    return images, labels, patient_ids


def create_synthetic_dataset(data_dir: str):
    """Create synthetic histopathology images for demonstration"""
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True)
    
    print("Creating synthetic histopathology images for demonstration...")
    
    # Define synthetic tissue patterns for each subtype
    subtypes = {
        'SNF1': {'dominant_color': [200, 100, 150], 'pattern': 'solid'},      # Pink tumor
        'SNF2': {'dominant_color': [100, 100, 200], 'pattern': 'scattered'},   # Blue immune
        'SNF3': {'dominant_color': [150, 150, 100], 'pattern': 'fibrous'}      # Brown stroma
    }
    
    for subtype, properties in subtypes.items():
        subtype_dir = data_path / subtype
        subtype_dir.mkdir(exist_ok=True)
        
        # Generate 20 synthetic images per subtype
        for i in range(20):
            image = create_synthetic_histology_image(
                properties['dominant_color'], 
                properties['pattern'],
                size=(512, 512)
            )
            
            image_path = subtype_dir / f"patient_{subtype}_{i:03d}.png"
            Image.fromarray(image).save(image_path)
    
    print(f"Created synthetic dataset in {data_dir} with 60 total images")


def create_synthetic_histology_image(dominant_color, pattern, size=(512, 512)):
    """Create a synthetic histopathology image with specified characteristics"""
    np.random.seed()  # Ensure randomness
    
    image = np.zeros((*size, 3), dtype=np.uint8)
    
    # Base color with variation
    base_color = np.array(dominant_color)
    
    if pattern == 'solid':
        # Solid tumor regions
        for _ in range(5):
            center = (np.random.randint(100, size[0]-100), np.random.randint(100, size[1]-100))
            radius = np.random.randint(50, 100)
            y, x = np.ogrid[:size[0], :size[1]]
            mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
            
            color_variation = np.random.randint(-30, 30, 3)
            region_color = np.clip(base_color + color_variation, 0, 255)
            image[mask] = region_color
            
    elif pattern == 'scattered':
        # Scattered immune infiltration
        for _ in range(100):
            x = np.random.randint(0, size[0])
            y = np.random.randint(0, size[1])
            size_cell = np.random.randint(3, 8)
            
            x_start, x_end = max(0, x-size_cell), min(size[0], x+size_cell)
            y_start, y_end = max(0, y-size_cell), min(size[1], y+size_cell)
            
            color_variation = np.random.randint(-20, 20, 3)
            cell_color = np.clip(base_color + color_variation, 0, 255)
            image[x_start:x_end, y_start:y_end] = cell_color
            
    elif pattern == 'fibrous':
        # Fibrous stromal pattern
        for i in range(0, size[0], 20):
            thickness = np.random.randint(5, 15)
            x_start, x_end = i, min(size[0], i + thickness)
            
            color_variation = np.random.randint(-25, 25, 3)
            fiber_color = np.clip(base_color + color_variation, 0, 255)
            image[x_start:x_end, :] = fiber_color
    
    # Add background noise
    background_color = [220, 200, 200]  # Light pink background
    mask = np.all(image == 0, axis=2)
    image[mask] = background_color
    
    # Add some texture noise
    noise = np.random.randint(-10, 10, image.shape)
    image = np.clip(image.astype(int) + noise, 0, 255).astype(np.uint8)
    
    return image


def train_hybrid_classifier(data_dir: str = "demo_data", 
                           model_save_path: str = "models/hybrid_radiomics_model.pkl"):
    """Train the hybrid PyRadiomics classifier"""
    
    print("="*60)
    print("HYBRID PYRADIOMICS CLASSIFIER TRAINING")
    print("="*60)
    
    # Load dataset
    print("Loading dataset...")
    images, labels, patient_ids = load_demo_dataset(data_dir)
    
    # Split into train/test
    train_images, test_images, train_labels, test_labels = train_test_split(
        images, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    print(f"Training set: {len(train_images)} images")
    print(f"Test set: {len(test_images)} images")
    
    # Initialize tissue model
    print("Initializing tissue classification model...")
    tissue_model = SimpleClassifier(num_classes=8)
    tissue_model.eval()  # Use in evaluation mode (random initialization for demo)
    
    # Define transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Initialize hybrid classifier
    print("Initializing hybrid classifier...")
    try:
        hybrid_classifier = HybridRadiomicsClassifier(
            tissue_model=tissue_model,
            model_save_path=model_save_path
        )
        print("✓ Hybrid classifier initialized with PyRadiomics support")
    except Exception as e:
        print(f"Warning: PyRadiomics not available: {e}")
        print("Please install PyRadiomics: pip install pyradiomics")
        return
    
    # Train the model
    print("\nStarting training...")
    try:
        training_results = hybrid_classifier.train(
            images=train_images,
            labels=train_labels,
            transform=transform,
            validation_split=0.2
        )
        
        print("✓ Training completed successfully!")
        print(f"Selected {training_results['n_features_selected']} features")
        print(f"Top features: {', '.join(training_results['selected_features'][:5])}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        return
    
    # Test the model
    print("\nEvaluating on test set...")
    test_predictions = []
    test_probabilities = []
    
    for i, test_image in enumerate(test_images):
        try:
            prediction = hybrid_classifier.predict(
                test_image, transform, explain=True
            )
            test_predictions.append(prediction['subtype_idx'])
            test_probabilities.append(prediction['probabilities'])
            
            if i == 0:  # Show detailed results for first test case
                print(f"\nSample prediction (Test case 1):")
                print(f"Predicted: {prediction['subtype']}")
                print(f"Confidence: {prediction['confidence']:.1f}%")
                print(f"Feature summary: {prediction['feature_summary']}")
                
                if 'explanation' in prediction:
                    print("Prediction drivers:")
                    for driver in prediction['explanation']['prediction_drivers']:
                        print(f"  - {driver}")
                
        except Exception as e:
            print(f"Warning: Prediction failed for test case {i}: {e}")
            test_predictions.append(0)  # Default to class 0
            test_probabilities.append([1.0, 0.0, 0.0])
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score
    
    accuracy = accuracy_score(test_labels, test_predictions)
    f1 = f1_score(test_labels, test_predictions, average='macro')
    
    print(f"\nTest Results:")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro F1-score: {f1:.3f}")
    
    # Generate classification report
    subtype_names = ['SNF1 (Canonical)', 'SNF2 (Immune)', 'SNF3 (Stromal)']
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, test_predictions, target_names=subtype_names))
    
    # Create confusion matrix
    cm = confusion_matrix(test_labels, test_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['SNF1', 'SNF2', 'SNF3'],
                yticklabels=['SNF1', 'SNF2', 'SNF3'])
    plt.title('Confusion Matrix - Hybrid PyRadiomics Classifier')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    # Save confusion matrix
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    plt.savefig(results_dir / "hybrid_classifier_confusion_matrix.png", dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {results_dir / 'hybrid_classifier_confusion_matrix.png'}")
    
    # Generate clinical report for a test case
    if test_images:
        sample_prediction = hybrid_classifier.predict(
            test_images[0], transform, explain=True
        )
        clinical_report = create_clinical_report(sample_prediction, patient_id="TEST_001")
        
        with open(results_dir / "sample_clinical_report.txt", "w") as f:
            f.write(clinical_report)
        
        print(f"Sample clinical report saved to {results_dir / 'sample_clinical_report.txt'}")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"Model saved to: {model_save_path}")
    print(f"Results saved to: {results_dir}")
    print("\nTo use the trained model:")
    print("from app.hybrid_radiomics_classifier import HybridRadiomicsClassifier")
    print("classifier = HybridRadiomicsClassifier(tissue_model)")
    print("result = classifier.predict(image, transform, explain=True)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Hybrid PyRadiomics Classifier")
    parser.add_argument("--data_dir", default="demo_data", 
                       help="Directory containing training data")
    parser.add_argument("--model_path", default="models/hybrid_radiomics_model.pkl",
                       help="Path to save trained model")
    
    args = parser.parse_args()
    
    try:
        train_hybrid_classifier(args.data_dir, args.model_path)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc() 