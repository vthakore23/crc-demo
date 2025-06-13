#!/usr/bin/env python3
"""
Advanced Molecular Subtype Predictor V2
Learnable weights, multi-scale features, and validation-ready architecture
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns


class MolecularFeatureExtractor(nn.Module):
    """Extract multi-scale features from tissue predictions"""
    
    def __init__(self, input_dim=8):
        super().__init__()
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
    def forward(self, tissue_probs):
        """Extract enriched features from tissue probabilities"""
        # Basic tissue scores
        tumor_score = tissue_probs[:, 0] + tissue_probs[:, 2] * 0.5
        immune_score = tissue_probs[:, 3] + tissue_probs[:, 5] * 0.3
        stromal_score = tissue_probs[:, 1] + tissue_probs[:, 2] * 0.5
        
        # Ratios and interactions
        tumor_immune_ratio = tumor_score / (immune_score + 0.01)
        tumor_stromal_ratio = tumor_score / (stromal_score + 0.01)
        immune_stromal_ratio = immune_score / (stromal_score + 0.01)
        
        # Diversity metrics
        entropy = -torch.sum(tissue_probs * torch.log(tissue_probs + 1e-8), dim=1)
        max_prob = torch.max(tissue_probs, dim=1)[0]
        
        # Combine all features
        features = torch.stack([
            tumor_score, immune_score, stromal_score,
            tumor_immune_ratio, tumor_stromal_ratio, immune_stromal_ratio,
            entropy, max_prob
        ], dim=1)
        
        # Transform to higher-dimensional space
        enriched_features = self.feature_transform(features)
        
        return enriched_features, {
            'tumor_score': tumor_score,
            'immune_score': immune_score, 
            'stromal_score': stromal_score
        }


class AttentionAggregator(nn.Module):
    """Aggregate tile-level predictions with attention"""
    
    def __init__(self, feature_dim=32):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 16),
            nn.Tanh(),
            nn.Linear(16, 1)
        )
        
    def forward(self, tile_features):
        """
        tile_features: [batch_size, num_tiles, feature_dim]
        returns: [batch_size, feature_dim]
        """
        # Calculate attention weights
        attention_scores = self.attention(tile_features)  # [B, T, 1]
        attention_weights = F.softmax(attention_scores, dim=1)  # [B, T, 1]
        
        # Weighted aggregation
        aggregated = torch.sum(tile_features * attention_weights, dim=1)  # [B, F]
        
        return aggregated, attention_weights


class LearnableMolecularPredictor(nn.Module):
    """Advanced molecular subtype predictor with learnable parameters"""
    
    def __init__(self, num_subtypes=3, feature_dim=32):
        super().__init__()
        
        # Feature extraction
        self.feature_extractor = MolecularFeatureExtractor()
        
        # Attention-based aggregation for multiple tiles
        self.aggregator = AttentionAggregator(feature_dim)
        
        # Subtype-specific networks
        self.snf1_network = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.snf2_network = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.snf3_network = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Biological constraint layer
        self.biological_constraints = BiologicalConstraintLayer()
        
        # Initialize weights properly
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, tissue_probs, return_attention=False):
        """
        tissue_probs: [batch_size, num_tiles, 8] or [batch_size, 8]
        """
        if len(tissue_probs.shape) == 2:
            # Single tile, add tile dimension
            tissue_probs = tissue_probs.unsqueeze(1)
            
        batch_size, num_tiles, _ = tissue_probs.shape
        
        # Extract features for each tile
        tile_features = []
        tissue_scores_list = []
        
        for i in range(num_tiles):
            features, tissue_scores = self.feature_extractor(tissue_probs[:, i, :])
            tile_features.append(features)
            tissue_scores_list.append(tissue_scores)
            
        # Stack tile features
        tile_features = torch.stack(tile_features, dim=1)  # [B, T, F]
        
        # Aggregate with attention
        aggregated_features, attention_weights = self.aggregator(tile_features)
        
        # Compute subtype scores
        snf1_score = self.snf1_network(aggregated_features)
        snf2_score = self.snf2_network(aggregated_features)
        snf3_score = self.snf3_network(aggregated_features)
        
        # Combine scores
        scores = torch.cat([snf1_score, snf2_score, snf3_score], dim=1)
        
        # Apply biological constraints
        avg_tissue_scores = {
            key: torch.mean(torch.stack([ts[key] for ts in tissue_scores_list], dim=1), dim=1)
            for key in tissue_scores_list[0].keys()
        }
        scores = self.biological_constraints(scores, avg_tissue_scores)
        
        # Apply softmax for probabilities
        probabilities = F.softmax(scores, dim=1)
        
        if return_attention:
            return probabilities, attention_weights
        return probabilities


class BiologicalConstraintLayer(nn.Module):
    """Apply known biological constraints to predictions"""
    
    def __init__(self):
        super().__init__()
        # Learnable constraint strengths
        self.constraint_weights = nn.Parameter(torch.tensor([0.3, 0.3, 0.3]))
        
    def forward(self, scores, tissue_scores):
        """Apply biological rules with learnable strengths"""
        snf1_scores, snf2_scores, snf3_scores = scores[:, 0], scores[:, 1], scores[:, 2]
        
        # SNF2 should correlate with high immune
        immune_boost_snf2 = tissue_scores['immune_score'] * self.constraint_weights[0]
        snf2_scores = snf2_scores + immune_boost_snf2
        
        # SNF3 should correlate with high stromal
        stromal_boost_snf3 = tissue_scores['stromal_score'] * self.constraint_weights[1]
        snf3_scores = snf3_scores + stromal_boost_snf3
        
        # SNF1 should correlate with high tumor, low immune
        tumor_boost_snf1 = (tissue_scores['tumor_score'] * (1 - tissue_scores['immune_score'])) * self.constraint_weights[2]
        snf1_scores = snf1_scores + tumor_boost_snf1
        
        return torch.stack([snf1_scores, snf2_scores, snf3_scores], dim=1)


class MolecularSubtypeTrainer:
    """Training and validation logic for molecular predictor"""
    
    def __init__(self, model: LearnableMolecularPredictor, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', patience=10, factor=0.5
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Track training history
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_f1': [], 'val_f1': []
        }
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        predictions = []
        targets = []
        
        for batch_idx, (tissue_probs, labels) in enumerate(train_loader):
            tissue_probs = tissue_probs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(tissue_probs)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions, average='weighted')
        
        return avg_loss, accuracy, f1
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        targets = []
        probabilities = []
        
        with torch.no_grad():
            for tissue_probs, labels in val_loader:
                tissue_probs = tissue_probs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(tissue_probs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predictions.extend(outputs.argmax(dim=1).cpu().numpy())
                targets.extend(labels.cpu().numpy())
                probabilities.extend(outputs.cpu().numpy())
                
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(targets, predictions)
        f1 = f1_score(targets, predictions, average='weighted')
        
        # Calculate AUC for each class
        targets_bin = label_binarize(targets, classes=[0, 1, 2])
        auc_scores = []
        for i in range(3):
            auc = roc_auc_score(targets_bin[:, i], np.array(probabilities)[:, i])
            auc_scores.append(auc)
            
        return avg_loss, accuracy, f1, auc_scores, predictions, targets
    
    def train(self, train_loader, val_loader, epochs=100, save_path='models/molecular_predictor_v2.pth'):
        """Full training loop"""
        best_val_acc = 0
        patience_counter = 0
        patience = 20
        
        for epoch in range(epochs):
            # Train
            train_loss, train_acc, train_f1 = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc, val_f1, auc_scores, _, _ = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_acc)
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            
            # Print progress
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
            print(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
            print(f"AUC - SNF1: {auc_scores[0]:.4f}, SNF2: {auc_scores[1]:.4f}, SNF3: {auc_scores[2]:.4f}")
            print("-" * 50)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'history': self.history
                }, save_path)
                print(f"Saved best model with validation accuracy: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
                
        return self.history
    
    def generate_validation_report(self, val_loader, save_dir='validation_results'):
        """Generate comprehensive validation report"""
        Path(save_dir).mkdir(exist_ok=True)
        
        # Get predictions
        _, accuracy, f1, auc_scores, predictions, targets = self.validate(val_loader)
        
        # Confusion matrix
        cm = confusion_matrix(targets, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['SNF1', 'SNF2', 'SNF3'],
                    yticklabels=['SNF1', 'SNF2', 'SNF3'])
        plt.title(f'Confusion Matrix (Accuracy: {accuracy:.2%})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{save_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Training history
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        
        # Accuracy
        axes[0, 1].plot(self.history['train_acc'], label='Train')
        axes[0, 1].plot(self.history['val_acc'], label='Validation')
        axes[0, 1].set_title('Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        
        # F1 Score
        axes[1, 0].plot(self.history['train_f1'], label='Train')
        axes[1, 0].plot(self.history['val_f1'], label='Validation')
        axes[1, 0].set_title('F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        
        # AUC by class
        axes[1, 1].bar(['SNF1', 'SNF2', 'SNF3'], auc_scores)
        axes[1, 1].set_title('AUC by Subtype')
        axes[1, 1].set_ylabel('AUC Score')
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate text report
        report = f"""
Molecular Subtype Prediction Validation Report
==============================================

Overall Performance:
- Accuracy: {accuracy:.2%}
- Weighted F1 Score: {f1:.4f}

Per-Class Performance:
- SNF1 AUC: {auc_scores[0]:.4f}
- SNF2 AUC: {auc_scores[1]:.4f}
- SNF3 AUC: {auc_scores[2]:.4f}

Confusion Matrix:
{cm}

Best Validation Accuracy: {max(self.history['val_acc']):.2%} at epoch {np.argmax(self.history['val_acc'])+1}

Training completed with {len(self.history['train_loss'])} epochs.
"""
        
        with open(f'{save_dir}/validation_report.txt', 'w') as f:
            f.write(report)
            
        print(report)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'auc_scores': auc_scores,
            'confusion_matrix': cm.tolist()
        }


# Integration with existing pipeline
class MolecularPredictorV2Integration:
    """Integrate V2 predictor with existing codebase"""
    
    def __init__(self, tissue_model, model_path='models/molecular_predictor_v2.pth'):
        self.tissue_model = tissue_model
        self.tissue_model.eval()
        
        # Load molecular predictor
        self.molecular_predictor = LearnableMolecularPredictor()
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            self.molecular_predictor.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded molecular predictor with {checkpoint['best_val_acc']:.2%} validation accuracy")
        else:
            print("No trained molecular predictor found. Using random initialization.")
            
        self.molecular_predictor.eval()
        
    def predict_molecular_subtype(self, wsi_tiles, transform):
        """Predict molecular subtype from WSI tiles"""
        # Extract tissue predictions for all tiles
        tissue_predictions = []
        
        with torch.no_grad():
            for tile in wsi_tiles:
                # Preprocess tile
                tile_tensor = transform(tile).unsqueeze(0)
                
                # Get tissue predictions
                tissue_output = self.tissue_model(tile_tensor)
                tissue_probs = F.softmax(tissue_output, dim=1)
                tissue_predictions.append(tissue_probs)
                
        # Stack all tissue predictions
        tissue_predictions = torch.cat(tissue_predictions, dim=0)  # [num_tiles, 8]
        
        # Add batch dimension
        tissue_predictions = tissue_predictions.unsqueeze(0)  # [1, num_tiles, 8]
        
        # Get molecular predictions with attention
        molecular_probs, attention_weights = self.molecular_predictor(
            tissue_predictions, return_attention=True
        )
        
        # Get prediction
        predicted_idx = molecular_probs.argmax(dim=1).item()
        confidence = molecular_probs[0, predicted_idx].item() * 100
        
        subtype_names = ['SNF1 (Canonical)', 'SNF2 (Immune)', 'SNF3 (Stromal)']
        
        return {
            'subtype': subtype_names[predicted_idx],
            'confidence': confidence,
            'probabilities': molecular_probs[0].numpy(),
            'attention_weights': attention_weights[0].numpy(),
            'predicted_idx': predicted_idx
        }


if __name__ == "__main__":
    # Example usage
    print("Molecular Predictor V2 - Ready for EPOC validation")
    print("Features:")
    print("- Learnable weights instead of arbitrary values")
    print("- Multi-scale feature extraction")
    print("- Attention-based tile aggregation")
    print("- Biological constraint layer")
    print("- Comprehensive validation metrics")
    print("\nExpected performance after training:")
    print("- Initial (random): ~33% accuracy")
    print("- After EPOC training: >70% accuracy target") 