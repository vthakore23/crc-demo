import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (classification_report, roc_auc_score, 
                           confusion_matrix, balanced_accuracy_score)
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop

class ModelTrainer:
    """Unified trainer for deep learning models"""
    
    def __init__(self, model: nn.Module, config, experiment_name: str = "default"):
        self.model = model.to(config.DEVICE)
        self.config = config
        self.experiment_name = experiment_name
        
        # Setup optimization
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=config.LEARNING_RATE,
            weight_decay=0.01
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.EPOCHS
        )
        
        # Early stopping
        self.early_stopping = EarlyStopping(patience=10)
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0
        self.best_epoch = 0
        
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, (features, labels, masks, _) in enumerate(pbar):
            features = features.to(self.config.DEVICE)
            labels = labels.to(self.config.DEVICE)
            masks = masks.to(self.config.DEVICE)
            
            self.optimizer.zero_grad()
            
            # Process each bag individually
            batch_loss = 0
            for i in range(features.shape[0]):
                # Get non-padded features
                mask = masks[i].bool()
                bag_features = features[i][mask]
                
                if bag_features.shape[0] == 0:
                    continue
                
                # Forward pass
                if hasattr(self.model, 'forward'):
                    output = self.model(bag_features)
                else:
                    output = self.model(bag_features.unsqueeze(0))
                
                # Handle different output formats
                if isinstance(output, tuple):
                    output = output[0]
                
                loss = self.criterion(output, labels[i].unsqueeze(0))
                batch_loss += loss
                
                # Track accuracy
                _, predicted = torch.max(output, 1)
                correct += (predicted == labels[i]).sum().item()
                total += 1
            
            # Backward pass
            if total > 0:
                batch_loss = batch_loss / total
                batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                total_loss += batch_loss.item()
                
                pbar.set_postfix({
                    'loss': batch_loss.item(), 
                    'acc': correct/total if total > 0 else 0
                })
        
        epoch_loss = total_loss / len(dataloader)
        epoch_acc = correct / total if total > 0 else 0
        
        return epoch_loss, epoch_acc
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float, Dict]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for features, labels, masks, _ in tqdm(dataloader, desc="Validating"):
                features = features.to(self.config.DEVICE)
                labels = labels.to(self.config.DEVICE)
                masks = masks.to(self.config.DEVICE)
                
                for i in range(features.shape[0]):
                    mask = masks[i].bool()
                    bag_features = features[i][mask]
                    
                    if bag_features.shape[0] == 0:
                        continue
                    
                    # Forward pass
                    if hasattr(self.model, 'forward'):
                        output = self.model(bag_features)
                    else:
                        output = self.model(bag_features.unsqueeze(0))
                    
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    loss = self.criterion(output, labels[i].unsqueeze(0))
                    total_loss += loss.item()
                    
                    # Predictions
                    probs = torch.softmax(output, dim=1)
                    _, predicted = torch.max(output, 1)
                    
                    correct += (predicted == labels[i]).sum().item()
                    total += 1
                    
                    all_preds.append(predicted.cpu().numpy()[0])
                    all_labels.append(labels[i].cpu().numpy())
                    all_probs.append(probs.cpu().numpy()[0])
        
        # Calculate metrics
        avg_loss = total_loss / total if total > 0 else float('inf')
        accuracy = correct / total if total > 0 else 0
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'predictions': all_preds,
            'labels': all_labels,
            'probabilities': all_probs
        }
        
        return avg_loss, accuracy, metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: Optional[int] = None):
        """Full training loop"""
        epochs = epochs or self.config.EPOCHS
        
        logger.info(f"Starting training for {self.experiment_name}")
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Device: {self.config.DEVICE}")
        logger.info(f"Epochs: {epochs}")
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_acc, val_metrics = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step()
            
            # Logging
            logger.info(f"Epoch {epoch+1}/{epochs}")
            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.save_checkpoint(epoch, val_metrics)
            
            # Early stopping
            if self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Plot training curves
        self.plot_training_curves()
        
        # Load best model
        self.load_best_checkpoint()
        
        return self.best_val_acc
    
    def save_checkpoint(self, epoch: int, metrics: Dict):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'metrics': metrics,
            'config': self.config
        }
        
        path = self.config.MODEL_SAVE_PATH / f"{self.experiment_name}_best.pth"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_best_checkpoint(self):
        """Load the best checkpoint"""
        path = self.config.MODEL_SAVE_PATH / f"{self.experiment_name}_best.pth"
        if path.exists():
            checkpoint = torch.load(path, map_location=self.config.DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best checkpoint from epoch {checkpoint['epoch']}")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss', linewidth=2)
        ax1.plot(self.val_losses, label='Val Loss', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curve
        ax2.plot(self.val_accuracies, label='Val Accuracy', linewidth=2)
        ax2.axhline(y=self.best_val_acc, color='r', linestyle='--', 
                   label=f'Best: {self.best_val_acc:.3f}')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.config.RESULTS_PATH / f"{self.experiment_name}_curves.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved training curves to {save_path}")

class ClinicalEvaluator:
    """Comprehensive evaluation including clinical metrics"""
    
    def __init__(self, config):
        self.config = config
    
    def evaluate_model(self, model: nn.Module, test_loader: DataLoader, 
                      clinical_data: Optional[pd.DataFrame] = None):
        """Comprehensive model evaluation"""
        model.eval()
        
        all_preds = []
        all_probs = []
        all_labels = []
        all_slide_ids = []
        attention_weights = {}
        
        with torch.no_grad():
            for features, labels, slide_ids in test_loader:
                features = features.to(self.config.DEVICE)
                
                for i in range(features.shape[0]):
                    bag_features = features[i]
                    
                    # Get predictions and attention weights
                    if hasattr(model, 'forward') and 'return_attention' in model.forward.__code__.co_varnames:
                        output, attn = model(bag_features, return_attention=True)
                        attention_weights[slide_ids[i]] = attn.cpu().numpy()
                    else:
                        output = model(bag_features)
                    
                    probs = torch.softmax(output, dim=1)
                    _, predicted = torch.max(output, 1)
                    
                    all_preds.append(predicted.cpu().numpy()[0])
                    all_probs.append(probs.cpu().numpy()[0])
                    all_labels.append(labels[i].numpy())
                    all_slide_ids.append(slide_ids[i])
        
        # Calculate metrics
        results = self._calculate_metrics(all_labels, all_preds, all_probs)
        
        # Add clinical correlation if available
        if clinical_data is not None:
            clinical_results = self._clinical_correlation(
                all_slide_ids, all_preds, all_labels, clinical_data
            )
            results.update(clinical_results)
        
        # Generate comprehensive report
        self._generate_report(results, attention_weights)
        
        return results
    
    def _calculate_metrics(self, y_true: List, y_pred: List, 
                          y_probs: List) -> Dict:
        """Calculate comprehensive metrics"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_probs = np.array(y_probs)
        
        # Basic metrics
        metrics = {
            'accuracy': np.mean(y_true == y_pred),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, 
                                                        target_names=list(self.config.SUBTYPES.values()),
                                                        output_dict=True),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # ROC-AUC (one-vs-rest)
        if len(np.unique(y_true)) > 2:
            auc_scores = {}
            for i, subtype in self.config.SUBTYPES.items():
                y_true_binary = (y_true == i).astype(int)
                if len(np.unique(y_true_binary)) > 1:
                    auc = roc_auc_score(y_true_binary, y_probs[:, i])
                    auc_scores[subtype] = auc
            metrics['roc_auc_per_class'] = auc_scores
            metrics['roc_auc_macro'] = np.mean(list(auc_scores.values()))
        
        return metrics
    
    def _clinical_correlation(self, slide_ids: List, predictions: List,
                            true_labels: List, clinical_data: pd.DataFrame) -> Dict:
        """Correlate predictions with clinical outcomes"""
        results = {}
        
        # Create results dataframe
        df_results = pd.DataFrame({
            'slide_id': slide_ids,
            'predicted_subtype': predictions,
            'true_subtype': true_labels
        })
        
        # Merge with clinical data
        df_merged = df_results.merge(clinical_data, on='slide_id', how='left')
        
        # Survival analysis
        if 'survival_months' in df_merged.columns and 'survival_event' in df_merged.columns:
            survival_results = self._survival_analysis(df_merged)
            results['survival_analysis'] = survival_results
        
        # Treatment response analysis
        if 'treatment_response' in df_merged.columns:
            response_results = self._treatment_response_analysis(df_merged)
            results['treatment_response'] = response_results
        
        return results
    
    def _survival_analysis(self, df: pd.DataFrame) -> Dict:
        """Kaplan-Meier and Cox regression analysis"""
        results = {}
        kmf = KaplanMeierFitter()
        
        # Plot survival curves by subtype
        plt.figure(figsize=(10, 6))
        for subtype in df['predicted_subtype'].unique():
            mask = df['predicted_subtype'] == subtype
            if mask.sum() > 0:
                kmf.fit(
                    df.loc[mask, 'survival_months'],
                    df.loc[mask, 'survival_event'],
                    label=f'Subtype {subtype}'
                )
                kmf.plot()
        
        plt.title('Overall Survival by Molecular Subtype')
        plt.xlabel('Months')
        plt.ylabel('Survival Probability')
        save_path = self.config.RESULTS_PATH / 'survival_curves.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        # Cox regression
        if len(df) > 20:  # Need sufficient samples
            cph = CoxPHFitter()
            cox_data = df[['survival_months', 'survival_event', 'predicted_subtype']]
            cph.fit(cox_data, duration_col='survival_months', event_col='survival_event')
            results['cox_summary'] = cph.print_summary()
        
        return results
    
    def _treatment_response_analysis(self, df: pd.DataFrame) -> Dict:
        """Analyze treatment response by subtype"""
        response_by_subtype = pd.crosstab(
            df['predicted_subtype'], 
            df['treatment_response'],
            normalize='index'
        )
        
        # Plot response rates
        plt.figure(figsize=(10, 6))
        response_by_subtype.plot(kind='bar', stacked=True)
        plt.title('Treatment Response by Molecular Subtype')
        plt.xlabel('Subtype')
        plt.ylabel('Proportion')
        save_path = self.config.RESULTS_PATH / 'response_rates.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        return {'response_rates': response_by_subtype.to_dict()}

class MetricsCalculator:
    """Calculate and visualize various metrics"""
    
    @staticmethod
    def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                         y_probs: np.ndarray, class_names: List[str]) -> Dict:
        """Calculate comprehensive metrics"""
        metrics = {
            'accuracy': np.mean(y_true == y_pred),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(
                y_true, y_pred, target_names=class_names, output_dict=True
            )
        }
        
        # ROC-AUC for multi-class
        try:
            if len(class_names) > 2:
                metrics['roc_auc_macro'] = roc_auc_score(
                    y_true, y_probs, multi_class='ovr', average='macro'
                )
                metrics['roc_auc_weighted'] = roc_auc_score(
                    y_true, y_probs, multi_class='ovr', average='weighted'
                )
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_probs[:, 1])
        except:
            logger.warning("Could not calculate ROC-AUC")
        
        return metrics
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                            save_path: Optional[Path] = None):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   square=True, cbar_kws={'label': 'Count'})
        
        # Add normalized values as text
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                plt.text(j + 0.5, i + 0.7, f'({cm_normalized[i, j]:.2%})',
                        ha='center', va='center', fontsize=9, color='gray')
        
        plt.title('Confusion Matrix', fontsize=16)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    @staticmethod
    def plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray, 
                       class_names: List[str], save_path: Optional[Path] = None):
        """Plot ROC curves for multi-class classification"""
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve for each class
        for i in range(len(class_names)):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, linewidth=2,
                    label=f'{class_names[i]} (AUC = {roc_auc:.3f})')
        
        # Plot random classifier
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Multi-class Classification', fontsize=16)
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show() 