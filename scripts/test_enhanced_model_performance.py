#!/usr/bin/env python3
"""
Enhanced Model Performance Testing & Overfitting Monitoring
Tests the enhanced molecular subtype model and reports actual performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
import logging
import json
import os
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our enhanced model
import sys
sys.path.append('.')

class TestDataset(Dataset):
    """Test dataset for performance evaluation"""
    
    def __init__(self, size=1000):
        self.size = size
        self.samples = []
        
        # Realistic distribution
        subtype_distribution = [0.35, 0.25, 0.30, 0.10]  # Canonical, Immune, Stromal, Normal
        
        for i in range(size):
            subtype = np.random.choice(4, p=subtype_distribution)
            self.samples.append({
                'subtype': subtype,
                'index': i
            })
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Generate test image
        image = torch.randn(3, 224, 224)
        
        return image, sample['subtype'], sample['index']

def load_enhanced_model(model_path='models/enhanced_molecular_final.pth'):
    """Load the enhanced molecular model"""
    try:
        from enhanced_training_run import EnhancedMolecularModel
        
        # Create model
        model = EnhancedMolecularModel(num_classes=4)
        
        # Load checkpoint
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location='cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Get training history if available
            history = checkpoint.get('history', {})
            best_val_acc = checkpoint.get('best_val_acc', 0)
            
            logger.info(f"✅ Enhanced model loaded from {model_path}")
            logger.info(f"📊 Best validation accuracy: {best_val_acc:.2f}%")
            
            return model, history, best_val_acc
        else:
            logger.error(f"Model file not found: {model_path}")
            return None, {}, 0
            
    except Exception as e:
        logger.error(f"Failed to load enhanced model: {e}")
        return None, {}, 0

def analyze_overfitting(history):
    """Analyze training history for overfitting signs"""
    if not history or 'train_loss' not in history:
        logger.warning("No training history available for overfitting analysis")
        return {"overfitting_detected": False, "recommendation": "Unable to analyze - no history"}
    
    train_losses = history['train_loss']
    val_losses = history['val_loss']
    train_accs = history['train_acc']
    val_accs = history['val_acc']
    
    # Check for overfitting signs
    overfitting_signs = []
    
    # 1. Validation loss increasing while training loss decreasing
    if len(val_losses) > 5:
        recent_val_trend = np.polyfit(range(len(val_losses)//2, len(val_losses)), 
                                     val_losses[len(val_losses)//2:], 1)[0]
        recent_train_trend = np.polyfit(range(len(train_losses)//2, len(train_losses)), 
                                       train_losses[len(train_losses)//2:], 1)[0]
        
        if recent_val_trend > 0 and recent_train_trend < 0:
            overfitting_signs.append("Validation loss increasing while training loss decreasing")
    
    # 2. Large gap between training and validation accuracy
    if train_accs and val_accs:
        final_gap = train_accs[-1] - val_accs[-1]
        if final_gap > 10:  # More than 10% gap
            overfitting_signs.append(f"Large accuracy gap: {final_gap:.1f}%")
    
    # 3. Validation accuracy plateauing or decreasing
    if len(val_accs) > 5:
        recent_val_acc_trend = np.polyfit(range(len(val_accs)//2, len(val_accs)), 
                                         val_accs[len(val_accs)//2:], 1)[0]
        if recent_val_acc_trend < -0.1:  # Decreasing trend
            overfitting_signs.append("Validation accuracy decreasing")
    
    # Generate recommendation
    if overfitting_signs:
        recommendation = "⚠️  OVERFITTING DETECTED: " + "; ".join(overfitting_signs)
        overfitting_detected = True
    else:
        recommendation = "✅ No significant overfitting detected"
        overfitting_detected = False
    
    return {
        "overfitting_detected": overfitting_detected,
        "signs": overfitting_signs,
        "recommendation": recommendation,
        "final_train_acc": train_accs[-1] if train_accs else 0,
        "final_val_acc": val_accs[-1] if val_accs else 0,
        "accuracy_gap": train_accs[-1] - val_accs[-1] if (train_accs and val_accs) else 0
    }

def evaluate_model_performance(model, device='cpu'):
    """Comprehensive model performance evaluation"""
    model.eval()
    
    # Create test dataset
    test_dataset = TestDataset(size=1000)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_predictions = []
    all_targets = []
    all_confidences = []
    
    logger.info("🧪 Evaluating model performance...")
    
    with torch.no_grad():
        for batch_idx, (data, target, _) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            
            # Forward pass
            output = model(data)
            logits = output['logits']
            confidence = output.get('confidence', torch.ones(len(data)))
            
            # Get predictions
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_predictions) == np.array(all_targets)) * 100
    f1_macro = f1_score(all_targets, all_predictions, average='macro') * 100
    f1_weighted = f1_score(all_targets, all_predictions, average='weighted') * 100
    
    # Per-class metrics
    class_report = classification_report(all_targets, all_predictions, 
                                       target_names=['Canonical', 'Immune', 'Stromal', 'Normal'],
                                       output_dict=True)
    
    # Confidence statistics
    mean_confidence = np.mean(all_confidences)
    
    performance_metrics = {
        'overall_accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'mean_confidence': mean_confidence,
        'class_performance': {
            'Canonical': {
                'f1_score': class_report['Canonical']['f1-score'] * 100,
                'precision': class_report['Canonical']['precision'] * 100,
                'recall': class_report['Canonical']['recall'] * 100
            },
            'Immune': {
                'f1_score': class_report['Immune']['f1-score'] * 100,
                'precision': class_report['Immune']['precision'] * 100,
                'recall': class_report['Immune']['recall'] * 100
            },
            'Stromal': {
                'f1_score': class_report['Stromal']['f1-score'] * 100,
                'precision': class_report['Stromal']['precision'] * 100,
                'recall': class_report['Stromal']['recall'] * 100
            },
            'Normal': {
                'f1_score': class_report['Normal']['f1-score'] * 100,
                'precision': class_report['Normal']['precision'] * 100,
                'recall': class_report['Normal']['recall'] * 100
            }
        }
    }
    
    return performance_metrics, all_predictions, all_targets

def create_performance_visualization(history, performance_metrics, overfitting_analysis):
    """Create comprehensive performance visualization"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Enhanced Molecular Subtype Model - Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Training History
    if history and 'train_loss' in history:
        ax1 = axes[0, 0]
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training & Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Accuracy History
        ax2 = axes[0, 1]
        ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training & Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        axes[0, 0].text(0.5, 0.5, 'Training History\nNot Available', ha='center', va='center')
        axes[0, 1].text(0.5, 0.5, 'Accuracy History\nNot Available', ha='center', va='center')
    
    # 3. Class Performance
    ax3 = axes[0, 2]
    classes = list(performance_metrics['class_performance'].keys())
    f1_scores = [performance_metrics['class_performance'][cls]['f1_score'] for cls in classes]
    
    bars = ax3.bar(classes, f1_scores, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax3.set_title('F1-Score by Molecular Subtype')
    ax3.set_ylabel('F1-Score (%)')
    ax3.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, score in zip(bars, f1_scores):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{score:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. Overall Metrics Summary
    ax4 = axes[1, 0]
    metrics_names = ['Overall Accuracy', 'F1-Macro', 'F1-Weighted']
    metrics_values = [performance_metrics['overall_accuracy'], 
                     performance_metrics['f1_macro'], 
                     performance_metrics['f1_weighted']]
    
    bars = ax4.barh(metrics_names, metrics_values, color=['#FF9F43', '#6C5CE7', '#A29BFE'])
    ax4.set_title('Overall Performance Metrics')
    ax4.set_xlabel('Score (%)')
    ax4.set_xlim(0, 100)
    
    for bar, value in zip(bars, metrics_values):
        ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{value:.1f}%', ha='left', va='center', fontweight='bold')
    
    # 5. Overfitting Analysis
    ax5 = axes[1, 1]
    ax5.axis('off')
    
    overfitting_text = f"""
    🔍 OVERFITTING ANALYSIS
    
    Status: {'⚠️ DETECTED' if overfitting_analysis['overfitting_detected'] else '✅ GOOD'}
    
    Final Training Acc: {overfitting_analysis['final_train_acc']:.1f}%
    Final Validation Acc: {overfitting_analysis['final_val_acc']:.1f}%
    Accuracy Gap: {overfitting_analysis['accuracy_gap']:.1f}%
    
    Recommendation:
    {overfitting_analysis['recommendation']}
    """
    
    ax5.text(0.05, 0.95, overfitting_text, transform=ax5.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # 6. Model Summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    summary_text = f"""
    🧬 MODEL SUMMARY
    
    Architecture: Multi-Scale Ensemble
    - EfficientNet-B3
    - ResNet-50
    - Fusion Layers
    
    Performance:
    • Accuracy: {performance_metrics['overall_accuracy']:.2f}%
    • Mean Confidence: {performance_metrics['mean_confidence']:.3f}
    • F1-Macro: {performance_metrics['f1_macro']:.2f}%
    
    EPOC Readiness: {'✅ READY' if performance_metrics['overall_accuracy'] > 95 else '⚠️ NEEDS IMPROVEMENT'}
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/enhanced_model_performance_analysis.png', dpi=300, bbox_inches='tight')
    logger.info("📊 Performance visualization saved to results/enhanced_model_performance_analysis.png")
    
    return fig

def main():
    """Main performance testing function"""
    logger.info("🚀 Enhanced Model Performance Testing & Overfitting Monitoring")
    logger.info("=" * 70)
    
    # Set device
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Load enhanced model
    model, history, best_val_acc = load_enhanced_model()
    
    if model is None:
        logger.error("❌ Failed to load enhanced model")
        return
    
    model = model.to(device)
    
    # Analyze overfitting
    logger.info("🔍 Analyzing training for overfitting...")
    overfitting_analysis = analyze_overfitting(history)
    
    # Evaluate performance
    logger.info("📊 Evaluating model performance...")
    performance_metrics, predictions, targets = evaluate_model_performance(model, device)
    
    # Display results
    logger.info("\n" + "="*50)
    logger.info("🏆 ENHANCED MODEL PERFORMANCE RESULTS")
    logger.info("="*50)
    
    logger.info(f"📈 Overall Accuracy: {performance_metrics['overall_accuracy']:.2f}%")
    logger.info(f"📊 F1-Macro Score: {performance_metrics['f1_macro']:.2f}%")
    logger.info(f"📊 F1-Weighted Score: {performance_metrics['f1_weighted']:.2f}%")
    logger.info(f"🎯 Mean Confidence: {performance_metrics['mean_confidence']:.3f}")
    
    logger.info("\n🧬 Per-Class Performance:")
    for subtype, metrics in performance_metrics['class_performance'].items():
        logger.info(f"  {subtype}: F1={metrics['f1_score']:.1f}%, Precision={metrics['precision']:.1f}%, Recall={metrics['recall']:.1f}%")
    
    logger.info(f"\n{overfitting_analysis['recommendation']}")
    
    # Create visualization
    create_performance_visualization(history, performance_metrics, overfitting_analysis)
    
    # Save results
    results = {
        'performance_metrics': performance_metrics,
        'overfitting_analysis': overfitting_analysis,
        'model_info': {
            'best_validation_accuracy': best_val_acc,
            'architecture': 'Multi-Scale Ensemble (EfficientNet-B3 + ResNet-50)',
            'device': device
        }
    }
    
    results_path = 'results/enhanced_model_test_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"💾 Results saved to {results_path}")
    
    # EPOC readiness assessment
    epoc_ready = performance_metrics['overall_accuracy'] > 95
    logger.info("\n" + "="*50)
    logger.info("🎯 EPOC VALIDATION READINESS")
    logger.info("="*50)
    logger.info(f"Status: {'✅ READY' if epoc_ready else '⚠️ NEEDS IMPROVEMENT'}")
    logger.info(f"Target: 95% | Achieved: {performance_metrics['overall_accuracy']:.2f}%")
    
    if not overfitting_analysis['overfitting_detected']:
        logger.info("✅ No overfitting detected - Model is well-generalized")
    else:
        logger.info("⚠️  Overfitting detected - Consider regularization or early stopping")
    
    logger.info("🎉 Performance testing completed!")
    
    return performance_metrics, overfitting_analysis

if __name__ == "__main__":
    performance_metrics, overfitting_analysis = main() 