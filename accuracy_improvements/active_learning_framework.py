#!/usr/bin/env python3
"""
Active Learning Framework for Molecular Subtype Prediction
Implements uncertainty sampling, diversity sampling, and pseudo-labeling
for iterative accuracy improvement with limited labels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import logging
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import random
from collections import defaultdict
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UncertaintyEstimator:
    """Various uncertainty estimation methods"""
    
    @staticmethod
    def entropy_uncertainty(logits):
        """Predictive entropy uncertainty"""
        probs = F.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy
    
    @staticmethod
    def max_entropy_uncertainty(logits):
        """Maximum class probability uncertainty (1 - max_prob)"""
        probs = F.softmax(logits, dim=1)
        max_probs, _ = torch.max(probs, dim=1)
        return 1 - max_probs
    
    @staticmethod
    def bald_uncertainty(logits_list):
        """Bayesian Active Learning by Disagreement (BALD)"""
        # logits_list: List of logits from multiple forward passes (MC dropout)
        probs_list = [F.softmax(logits, dim=1) for logits in logits_list]
        mean_probs = torch.stack(probs_list).mean(dim=0)
        
        # Predictive entropy
        pred_entropy = -torch.sum(mean_probs * torch.log(mean_probs + 1e-8), dim=1)
        
        # Expected entropy
        expected_entropy = 0
        for probs in probs_list:
            expected_entropy += -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        expected_entropy /= len(probs_list)
        
        # BALD = Predictive entropy - Expected entropy
        bald = pred_entropy - expected_entropy
        return bald
    
    @staticmethod
    def ensemble_disagreement(logits_list):
        """Disagreement among ensemble members"""
        probs_list = [F.softmax(logits, dim=1) for logits in logits_list]
        probs_stack = torch.stack(probs_list)  # [num_models, batch_size, num_classes]
        
        # Variance across ensemble members
        disagreement = torch.var(probs_stack, dim=0).sum(dim=1)
        return disagreement

class DiversitySampler:
    """Diversity-based sampling strategies"""
    
    def __init__(self, method='k_means'):
        self.method = method
    
    def sample(self, features, n_samples, labeled_indices=None):
        """Sample diverse instances based on feature representations"""
        if self.method == 'k_means':
            return self._k_means_sampling(features, n_samples, labeled_indices)
        elif self.method == 'max_distance':
            return self._max_distance_sampling(features, n_samples, labeled_indices)
        elif self.method == 'core_set':
            return self._core_set_sampling(features, n_samples, labeled_indices)
        else:
            raise ValueError(f"Unknown diversity method: {self.method}")
    
    def _k_means_sampling(self, features, n_samples, labeled_indices):
        """K-means cluster centers as diverse samples"""
        features_np = features.cpu().numpy()
        
        # Remove already labeled samples
        if labeled_indices is not None:
            unlabeled_mask = np.ones(len(features_np), dtype=bool)
            unlabeled_mask[labeled_indices] = False
            unlabeled_features = features_np[unlabeled_mask]
            unlabeled_indices = np.where(unlabeled_mask)[0]
        else:
            unlabeled_features = features_np
            unlabeled_indices = np.arange(len(features_np))
        
        if len(unlabeled_features) <= n_samples:
            return unlabeled_indices.tolist()
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_samples, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(unlabeled_features)
        
        # Select samples closest to cluster centers
        selected_indices = []
        for i in range(n_samples):
            cluster_mask = cluster_labels == i
            if np.any(cluster_mask):
                cluster_features = unlabeled_features[cluster_mask]
                cluster_indices = unlabeled_indices[cluster_mask]
                
                # Find closest to center
                center = kmeans.cluster_centers_[i]
                distances = np.linalg.norm(cluster_features - center, axis=1)
                closest_idx = np.argmin(distances)
                selected_indices.append(cluster_indices[closest_idx])
        
        return selected_indices
    
    def _max_distance_sampling(self, features, n_samples, labeled_indices):
        """Maximum distance sampling (farthest point sampling)"""
        features_np = features.cpu().numpy()
        
        # Remove already labeled samples
        if labeled_indices is not None:
            unlabeled_mask = np.ones(len(features_np), dtype=bool)
            unlabeled_mask[labeled_indices] = False
            unlabeled_features = features_np[unlabeled_mask]
            unlabeled_indices = np.where(unlabeled_mask)[0]
        else:
            unlabeled_features = features_np
            unlabeled_indices = np.arange(len(features_np))
        
        if len(unlabeled_features) <= n_samples:
            return unlabeled_indices.tolist()
        
        # Start with random point
        selected_indices = [random.choice(unlabeled_indices)]
        
        for _ in range(n_samples - 1):
            # Compute distances to all selected points
            selected_features = features_np[selected_indices]
            distances = pairwise_distances(unlabeled_features, selected_features)
            min_distances = np.min(distances, axis=1)
            
            # Remove already selected
            for idx in selected_indices:
                if idx in unlabeled_indices:
                    pos = np.where(unlabeled_indices == idx)[0][0]
                    min_distances[pos] = -1
            
            # Select farthest point
            farthest_idx = np.argmax(min_distances)
            selected_indices.append(unlabeled_indices[farthest_idx])
        
        return selected_indices
    
    def _core_set_sampling(self, features, n_samples, labeled_indices):
        """Core-set selection for diversity"""
        features_np = features.cpu().numpy()
        
        if labeled_indices is not None:
            unlabeled_mask = np.ones(len(features_np), dtype=bool)
            unlabeled_mask[labeled_indices] = False
            unlabeled_indices = np.where(unlabeled_mask)[0]
            labeled_features = features_np[labeled_indices]
        else:
            unlabeled_indices = np.arange(len(features_np))
            labeled_features = np.array([]).reshape(0, features_np.shape[1])
        
        unlabeled_features = features_np[unlabeled_indices]
        
        if len(unlabeled_features) <= n_samples:
            return unlabeled_indices.tolist()
        
        # Core-set selection: minimize maximum distance to selected points
        selected_indices = []
        
        for _ in range(n_samples):
            if len(selected_indices) == 0 and len(labeled_features) == 0:
                # First selection - random
                idx = random.choice(range(len(unlabeled_indices)))
                selected_indices.append(unlabeled_indices[idx])
            else:
                # Select point that minimizes maximum distance
                current_selected = features_np[labeled_indices.tolist() + selected_indices] if len(labeled_indices) > 0 else features_np[selected_indices]
                
                best_idx = None
                best_score = float('inf')
                
                for i, idx in enumerate(unlabeled_indices):
                    if idx not in selected_indices:
                        candidate_features = np.vstack([current_selected, features_np[idx:idx+1]])
                        
                        # Compute maximum distance from any unlabeled point to selected set
                        distances = pairwise_distances(unlabeled_features, candidate_features)
                        min_distances = np.min(distances, axis=1)
                        max_min_distance = np.max(min_distances)
                        
                        if max_min_distance < best_score:
                            best_score = max_min_distance
                            best_idx = idx
                
                if best_idx is not None:
                    selected_indices.append(best_idx)
        
        return selected_indices

class PseudoLabeler:
    """Pseudo-labeling for semi-supervised learning"""
    
    def __init__(self, confidence_threshold=0.8, max_pseudo_ratio=0.5):
        self.confidence_threshold = confidence_threshold
        self.max_pseudo_ratio = max_pseudo_ratio
    
    def generate_pseudo_labels(self, model, dataloader, device='cuda'):
        """Generate pseudo-labels for unlabeled data"""
        model.eval()
        pseudo_labels = []
        confidences = []
        indices = []
        
        with torch.no_grad():
            for batch_idx, (data, _, batch_indices) in enumerate(dataloader):
                data = data.to(device)
                
                # Forward pass
                if hasattr(model, 'predict_with_confidence'):
                    # Use model's built-in prediction method
                    results = model.predict_with_confidence(data)
                    if isinstance(results, list):
                        for i, result in enumerate(results):
                            pseudo_labels.append(result['subtype_index'])
                            confidences.append(result['confidence'])
                            indices.append(batch_indices[i].item())
                    else:
                        pseudo_labels.append(results['subtype_index'])
                        confidences.append(results['confidence'])
                        indices.append(batch_indices.item())
                else:
                    # Standard forward pass
                    output = model(data)
                    if isinstance(output, dict):
                        logits = output['logits']
                    else:
                        logits = output
                    
                    probs = F.softmax(logits, dim=1)
                    max_probs, predicted = torch.max(probs, 1)
                    
                    for i in range(len(predicted)):
                        pseudo_labels.append(predicted[i].item())
                        confidences.append(max_probs[i].item())
                        indices.append(batch_indices[i].item())
        
        # Filter by confidence
        pseudo_data = []
        for i, (label, conf, idx) in enumerate(zip(pseudo_labels, confidences, indices)):
            if conf >= self.confidence_threshold:
                pseudo_data.append({
                    'index': idx,
                    'label': label,
                    'confidence': conf
                })
        
        # Limit number of pseudo-labels
        total_samples = len(pseudo_labels)
        max_pseudo = int(total_samples * self.max_pseudo_ratio)
        
        if len(pseudo_data) > max_pseudo:
            # Sort by confidence and take top samples
            pseudo_data.sort(key=lambda x: x['confidence'], reverse=True)
            pseudo_data = pseudo_data[:max_pseudo]
        
        logger.info(f"Generated {len(pseudo_data)} pseudo-labels from {total_samples} samples")
        
        return pseudo_data

class ActiveLearningManager:
    """Main active learning manager"""
    
    def __init__(self, 
                 initial_budget=100,
                 query_budget=50,
                 max_iterations=10,
                 uncertainty_method='entropy',
                 diversity_method='k_means',
                 use_pseudo_labeling=True,
                 pseudo_confidence=0.8):
        
        self.initial_budget = initial_budget
        self.query_budget = query_budget
        self.max_iterations = max_iterations
        self.uncertainty_method = uncertainty_method
        self.diversity_method = diversity_method
        self.use_pseudo_labeling = use_pseudo_labeling
        self.pseudo_confidence = pseudo_confidence
        
        # Initialize components
        self.uncertainty_estimator = UncertaintyEstimator()
        self.diversity_sampler = DiversitySampler(method=diversity_method)
        self.pseudo_labeler = PseudoLabeler(confidence_threshold=pseudo_confidence)
        
        # Tracking
        self.labeled_indices = set()
        self.unlabeled_indices = set()
        self.pseudo_labeled_indices = set()
        self.iteration_history = []
    
    def initialize_labeled_set(self, total_samples, strategy='random'):
        """Initialize labeled set"""
        if strategy == 'random':
            labeled_indices = random.sample(range(total_samples), self.initial_budget)
        elif strategy == 'balanced':
            # Assume we have some way to get balanced samples
            labeled_indices = random.sample(range(total_samples), self.initial_budget)
        else:
            raise ValueError(f"Unknown initialization strategy: {strategy}")
        
        self.labeled_indices = set(labeled_indices)
        self.unlabeled_indices = set(range(total_samples)) - self.labeled_indices
        
        logger.info(f"Initialized with {len(self.labeled_indices)} labeled samples")
        return list(self.labeled_indices)
    
    def query_samples(self, model, unlabeled_dataloader, feature_extractor=None, device='cuda'):
        """Query samples using active learning strategy"""
        if len(self.unlabeled_indices) == 0:
            logger.warning("No unlabeled samples available")
            return []
        
        # Extract features and uncertainties
        features_list = []
        uncertainties_list = []
        indices_list = []
        
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, _, batch_indices) in enumerate(unlabeled_dataloader):
                data = data.to(device)
                
                # Get features
                if feature_extractor:
                    features = feature_extractor(data)
                elif hasattr(model, 'backbone'):
                    features = model.backbone(data)
                else:
                    # Use penultimate layer
                    output = model(data)
                    if isinstance(output, dict) and 'bag_features' in output:
                        features = output['bag_features']
                    else:
                        features = output
                
                # Get uncertainties
                if hasattr(model, 'forward'):
                    output = model(data)
                    if isinstance(output, dict):
                        logits = output['logits']
                    else:
                        logits = output
                else:
                    logits = model(data)
                
                if self.uncertainty_method == 'entropy':
                    uncertainty = self.uncertainty_estimator.entropy_uncertainty(logits)
                elif self.uncertainty_method == 'max_entropy':
                    uncertainty = self.uncertainty_estimator.max_entropy_uncertainty(logits)
                else:
                    uncertainty = self.uncertainty_estimator.entropy_uncertainty(logits)
                
                features_list.append(features)
                uncertainties_list.append(uncertainty)
                indices_list.extend(batch_indices.tolist())
        
        # Concatenate all features and uncertainties
        all_features = torch.cat(features_list, dim=0)
        all_uncertainties = torch.cat(uncertainties_list, dim=0)
        
        # Convert to indices relative to unlabeled set
        unlabeled_indices_list = list(self.unlabeled_indices)
        
        # Hybrid strategy: uncertainty + diversity
        n_uncertainty = self.query_budget // 2
        n_diversity = self.query_budget - n_uncertainty
        
        # 1. Uncertainty-based selection
        uncertainty_scores = all_uncertainties.cpu().numpy()
        uncertainty_order = np.argsort(uncertainty_scores)[::-1]  # High to low
        uncertainty_selected = []
        
        for idx in uncertainty_order:
            if len(uncertainty_selected) >= n_uncertainty:
                break
            if indices_list[idx] in self.unlabeled_indices:
                uncertainty_selected.append(indices_list[idx])
        
        # 2. Diversity-based selection
        diversity_selected = self.diversity_sampler.sample(
            all_features, 
            n_diversity, 
            labeled_indices=list(self.labeled_indices) + uncertainty_selected
        )
        
        # Combine selections
        selected_indices = list(set(uncertainty_selected + diversity_selected))
        
        # Ensure we don't exceed budget
        if len(selected_indices) > self.query_budget:
            selected_indices = selected_indices[:self.query_budget]
        
        logger.info(f"Selected {len(selected_indices)} samples: "
                   f"{len(uncertainty_selected)} by uncertainty, "
                   f"{len(diversity_selected)} by diversity")
        
        return selected_indices
    
    def update_labeled_set(self, new_labeled_indices, pseudo_labeled_data=None):
        """Update labeled and unlabeled sets"""
        # Add new labeled samples
        self.labeled_indices.update(new_labeled_indices)
        self.unlabeled_indices -= set(new_labeled_indices)
        
        # Add pseudo-labeled samples
        if pseudo_labeled_data:
            pseudo_indices = [item['index'] for item in pseudo_labeled_data]
            self.pseudo_labeled_indices.update(pseudo_indices)
            self.unlabeled_indices -= set(pseudo_indices)
        
        logger.info(f"Updated sets - Labeled: {len(self.labeled_indices)}, "
                   f"Pseudo-labeled: {len(self.pseudo_labeled_indices)}, "
                   f"Unlabeled: {len(self.unlabeled_indices)}")
    
    def run_active_learning_cycle(self, 
                                 model, 
                                 train_function,
                                 unlabeled_dataloader,
                                 validation_dataloader,
                                 device='cuda'):
        """Run one cycle of active learning"""
        
        # Query new samples
        new_labeled_indices = self.query_samples(model, unlabeled_dataloader, device=device)
        
        # Generate pseudo-labels if enabled
        pseudo_labeled_data = None
        if self.use_pseudo_labeling:
            pseudo_labeled_data = self.pseudo_labeler.generate_pseudo_labels(
                model, unlabeled_dataloader, device=device
            )
        
        # Update labeled sets
        self.update_labeled_set(new_labeled_indices, pseudo_labeled_data)
        
        # Retrain model with updated labeled set
        # Note: train_function should handle the updated labeled indices
        train_results = train_function(
            labeled_indices=list(self.labeled_indices),
            pseudo_labeled_data=pseudo_labeled_data
        )
        
        # Evaluate on validation set
        val_results = self.evaluate_model(model, validation_dataloader, device)
        
        # Record iteration
        iteration_info = {
            'labeled_count': len(self.labeled_indices),
            'pseudo_labeled_count': len(self.pseudo_labeled_indices),
            'unlabeled_count': len(self.unlabeled_indices),
            'train_results': train_results,
            'val_results': val_results,
            'new_labeled_indices': new_labeled_indices
        }
        
        self.iteration_history.append(iteration_info)
        
        return iteration_info
    
    def evaluate_model(self, model, dataloader, device='cuda'):
        """Evaluate model performance"""
        model.eval()
        correct = 0
        total = 0
        class_correct = defaultdict(int)
        class_total = defaultdict(int)
        
        with torch.no_grad():
            for data, target, _ in dataloader:
                data, target = data.to(device), target.to(device)
                
                output = model(data)
                if isinstance(output, dict):
                    logits = output['logits']
                else:
                    logits = output
                
                _, predicted = torch.max(logits, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # Per-class accuracy
                for i in range(len(target)):
                    label = target[i].item()
                    class_total[label] += 1
                    if predicted[i] == target[i]:
                        class_correct[label] += 1
        
        accuracy = 100 * correct / total
        class_accuracies = {
            cls: 100 * class_correct[cls] / class_total[cls] 
            for cls in class_total.keys()
        }
        
        return {
            'accuracy': accuracy,
            'class_accuracies': class_accuracies,
            'total_samples': total
        }
    
    def save_state(self, path):
        """Save active learning state"""
        state = {
            'labeled_indices': self.labeled_indices,
            'unlabeled_indices': self.unlabeled_indices,
            'pseudo_labeled_indices': self.pseudo_labeled_indices,
            'iteration_history': self.iteration_history,
            'config': {
                'initial_budget': self.initial_budget,
                'query_budget': self.query_budget,
                'max_iterations': self.max_iterations,
                'uncertainty_method': self.uncertainty_method,
                'diversity_method': self.diversity_method,
                'use_pseudo_labeling': self.use_pseudo_labeling,
                'pseudo_confidence': self.pseudo_confidence
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(state, f)
        
        logger.info(f"Active learning state saved to {path}")
    
    def load_state(self, path):
        """Load active learning state"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        
        self.labeled_indices = state['labeled_indices']
        self.unlabeled_indices = state['unlabeled_indices']
        self.pseudo_labeled_indices = state['pseudo_labeled_indices']
        self.iteration_history = state['iteration_history']
        
        # Update config
        config = state['config']
        self.initial_budget = config['initial_budget']
        self.query_budget = config['query_budget']
        self.max_iterations = config['max_iterations']
        self.uncertainty_method = config['uncertainty_method']
        self.diversity_method = config['diversity_method']
        self.use_pseudo_labeling = config['use_pseudo_labeling']
        self.pseudo_confidence = config['pseudo_confidence']
        
        logger.info(f"Active learning state loaded from {path}")

if __name__ == "__main__":
    # Test active learning framework
    logger.info("🧪 Testing Active Learning Framework")
    
    # Create dummy data
    total_samples = 1000
    
    # Initialize active learning manager
    al_manager = ActiveLearningManager(
        initial_budget=50,
        query_budget=25,
        uncertainty_method='entropy',
        diversity_method='k_means',
        use_pseudo_labeling=True
    )
    
    # Initialize labeled set
    initial_labeled = al_manager.initialize_labeled_set(total_samples, strategy='random')
    logger.info(f"✅ Initialized with {len(initial_labeled)} labeled samples")
    
    # Test uncertainty estimation
    dummy_logits = torch.randn(10, 3)  # 10 samples, 3 classes
    entropy_unc = UncertaintyEstimator.entropy_uncertainty(dummy_logits)
    max_ent_unc = UncertaintyEstimator.max_entropy_uncertainty(dummy_logits)
    
    logger.info(f"✅ Entropy uncertainty shape: {entropy_unc.shape}")
    logger.info(f"✅ Max entropy uncertainty shape: {max_ent_unc.shape}")
    
    # Test diversity sampling
    dummy_features = torch.randn(100, 64)  # 100 samples, 64-dim features
    diversity_sampler = DiversitySampler(method='k_means')
    diverse_indices = diversity_sampler.sample(dummy_features, n_samples=10)
    
    logger.info(f"✅ Diversity sampling selected {len(diverse_indices)} samples")
    
    # Test pseudo-labeler
    pseudo_labeler = PseudoLabeler(confidence_threshold=0.8)
    logger.info("✅ Pseudo-labeler initialized")
    
    logger.info("🎉 Active Learning Framework ready!") 