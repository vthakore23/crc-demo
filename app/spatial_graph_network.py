"""
Spatial Graph Neural Network for Tissue Region Modeling
Models spatial relationships between tissue regions as graphs for enhanced analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from scipy.spatial import Delaunay
import cv2


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network layer for processing tissue graphs
    """
    
    def __init__(self, in_features, out_features, dropout=0.6, alpha=0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # Learnable weights
        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        # Attention mechanism
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        
    def forward(self, h, adj):
        """
        h: node features [N, in_features]
        adj: adjacency matrix [N, N]
        """
        Wh = torch.mm(h, self.W)  # [N, out_features]
        N = Wh.size()[0]
        
        # Attention mechanism
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))
        
        # Masked attention
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, Wh)
        
        return F.elu(h_prime)
    
    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]
        
        # Repeat for all pairs
        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        
        # Concatenate
        all_combinations_matrix = torch.cat([
            Wh_repeated_in_chunks,
            Wh_repeated_alternating
        ], dim=1)
        
        return all_combinations_matrix.view(N, N, 2 * self.out_features)


class TissueGraphNetwork(nn.Module):
    """
    Graph Neural Network for modeling spatial relationships between tissue regions
    """
    
    def __init__(self, node_features_dim=512, hidden_dim=256, num_classes=3):
        super().__init__()
        
        # Node feature encoder
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim)
        )
        
        # Edge feature encoder (distance, angle, shared boundary)
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        
        # Graph attention layers
        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim) 
            for _ in range(3)
        ])
        
        # Global graph pooling
        self.global_pool = GlobalAttentionPool(hidden_dim)
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, node_features, edge_index, edge_features, batch=None):
        """
        Process tissue graph
        
        Args:
            node_features: Features for each tissue region [N, node_features_dim]
            edge_index: Graph connectivity [2, E]
            edge_features: Features for edges [E, 4]
            batch: Batch assignment for nodes (for multiple graphs)
            
        Returns:
            Class predictions
        """
        # Encode nodes
        x = self.node_encoder(node_features)
        
        # Create adjacency matrix from edge_index
        num_nodes = x.size(0)
        adj = self._edge_index_to_adj_matrix(edge_index, num_nodes)
        
        # Apply GNN layers
        for gnn in self.gnn_layers:
            x_new = gnn(x, adj)
            x = x + x_new  # Residual connection
            x = F.dropout(x, p=0.5, training=self.training)
            
        # Global pooling
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
        graph_embedding = self.global_pool(x, batch)
        
        # Classification
        out = self.classifier(graph_embedding)
        
        return out
    
    def _edge_index_to_adj_matrix(self, edge_index, num_nodes):
        """Convert edge index to adjacency matrix"""
        adj = torch.zeros((num_nodes, num_nodes), device=edge_index.device)
        adj[edge_index[0], edge_index[1]] = 1
        adj[edge_index[1], edge_index[0]] = 1  # Undirected graph
        adj.fill_diagonal_(1)  # Self-loops
        return adj


class GlobalAttentionPool(nn.Module):
    """
    Global attention pooling for graph-level representation
    """
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x, batch):
        """
        Pool node features to graph-level representation
        
        Args:
            x: Node features [N, hidden_dim]
            batch: Batch assignment [N]
            
        Returns:
            Graph-level features [B, hidden_dim]
        """
        # Compute attention scores
        att_scores = self.attention_weights(x).view(-1)
        
        # Softmax per graph
        att_scores = self._softmax_per_graph(att_scores, batch)
        
        # Weighted sum per graph
        out = self._weighted_sum_per_graph(x, att_scores, batch)
        
        return out
    
    def _softmax_per_graph(self, scores, batch):
        """Apply softmax separately for each graph in batch"""
        batch_size = batch.max().item() + 1
        softmax_scores = torch.zeros_like(scores)
        
        for i in range(batch_size):
            mask = batch == i
            if mask.any():
                softmax_scores[mask] = F.softmax(scores[mask], dim=0)
                
        return softmax_scores
    
    def _weighted_sum_per_graph(self, x, scores, batch):
        """Compute weighted sum for each graph"""
        batch_size = batch.max().item() + 1
        feature_dim = x.size(1)
        output = torch.zeros(batch_size, feature_dim, device=x.device)
        
        for i in range(batch_size):
            mask = batch == i
            if mask.any():
                output[i] = torch.sum(x[mask] * scores[mask].unsqueeze(1), dim=0)
                
        return output


class TissueGraphBuilder:
    """
    Builds graphs from tissue segmentation masks
    """
    
    def __init__(self, min_region_size=100):
        self.min_region_size = min_region_size
        
    def build_tissue_graph(self, tissue_masks: Dict[str, np.ndarray], 
                          tissue_features: Optional[Dict[str, np.ndarray]] = None) -> Dict:
        """
        Build graph representation from tissue masks
        
        Args:
            tissue_masks: Dictionary of tissue type masks
            tissue_features: Optional pre-computed features for each region
            
        Returns:
            Graph data dictionary
        """
        # Extract regions from masks
        regions = self._extract_regions(tissue_masks)
        
        if len(regions) == 0:
            return None
            
        # Compute region features
        node_features = self._compute_region_features(regions, tissue_features)
        
        # Build spatial graph
        edge_index, edge_features = self._build_spatial_edges(regions)
        
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features,
            'regions': regions,
            'num_nodes': len(regions)
        }
    
    def _extract_regions(self, tissue_masks):
        """Extract individual tissue regions from masks"""
        regions = []
        region_id = 0
        
        for tissue_type, mask in tissue_masks.items():
            if tissue_type in ['empty', 'debris']:
                continue
                
            # Find connected components
            binary_mask = (mask > 0.5).astype(np.uint8)
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < self.min_region_size:
                    continue
                    
                # Compute region properties
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                    
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Compute bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                regions.append({
                    'id': region_id,
                    'tissue_type': tissue_type,
                    'centroid': (cx, cy),
                    'area': area,
                    'bbox': (x, y, w, h),
                    'contour': contour,
                    'perimeter': cv2.arcLength(contour, True),
                    'solidity': area / cv2.contourArea(cv2.convexHull(contour))
                })
                region_id += 1
                
        return regions
    
    def _compute_region_features(self, regions, tissue_features=None):
        """Compute feature vector for each region"""
        features = []
        
        # Tissue type encoding
        tissue_types = ['tumor', 'stroma', 'lymphocytes', 'complex', 'mucosa', 'adipose']
        
        for region in regions:
            # One-hot encode tissue type
            tissue_encoding = np.zeros(len(tissue_types))
            if region['tissue_type'] in tissue_types:
                tissue_encoding[tissue_types.index(region['tissue_type'])] = 1
                
            # Morphological features
            morph_features = np.array([
                region['area'] / 10000,  # Normalize
                region['perimeter'] / 1000,
                region['solidity'],
                region['bbox'][2] / 100,  # Width
                region['bbox'][3] / 100,  # Height
                region['bbox'][2] / (region['bbox'][3] + 1e-6)  # Aspect ratio
            ])
            
            # Combine features
            region_features = np.concatenate([tissue_encoding, morph_features])
            
            # Add pre-computed features if available
            if tissue_features and region['id'] in tissue_features:
                region_features = np.concatenate([
                    region_features, 
                    tissue_features[region['id']]
                ])
                
            features.append(region_features)
            
        return torch.FloatTensor(features)
    
    def _build_spatial_edges(self, regions):
        """Build edges based on spatial proximity"""
        num_regions = len(regions)
        edges = []
        edge_features = []
        
        # Extract centroids
        points = np.array([r['centroid'] for r in regions])
        
        if num_regions < 3:
            # Too few regions for triangulation
            edge_index = torch.LongTensor([[], []])
            edge_feat = torch.FloatTensor([])
            return edge_index, edge_feat
            
        # Delaunay triangulation for spatial connectivity
        try:
            tri = Delaunay(points)
            
            # Extract edges from triangulation
            for simplex in tri.simplices:
                for i in range(3):
                    for j in range(i+1, 3):
                        edge = [simplex[i], simplex[j]]
                        if edge not in edges and edge[::-1] not in edges:
                            edges.append(edge)
                            
                            # Compute edge features
                            feat = self._compute_edge_features(
                                regions[edge[0]], regions[edge[1]]
                            )
                            edge_features.append(feat)
        except:
            # Fallback: connect k-nearest neighbors
            k = min(5, num_regions - 1)
            for i in range(num_regions):
                distances = [
                    np.linalg.norm(np.array(regions[i]['centroid']) - 
                                 np.array(regions[j]['centroid']))
                    for j in range(num_regions)
                ]
                nearest = np.argsort(distances)[1:k+1]
                
                for j in nearest:
                    edge = [i, j]
                    if edge not in edges and edge[::-1] not in edges:
                        edges.append(edge)
                        feat = self._compute_edge_features(regions[i], regions[j])
                        edge_features.append(feat)
                        
        # Convert to tensors
        if edges:
            edge_index = torch.LongTensor(edges).t()
            edge_feat = torch.FloatTensor(edge_features)
        else:
            edge_index = torch.LongTensor([[], []])
            edge_feat = torch.FloatTensor([])
            
        return edge_index, edge_feat
    
    def _compute_edge_features(self, region1, region2):
        """Compute features for an edge between two regions"""
        # Euclidean distance
        dist = np.linalg.norm(
            np.array(region1['centroid']) - np.array(region2['centroid'])
        )
        
        # Angle
        dx = region2['centroid'][0] - region1['centroid'][0]
        dy = region2['centroid'][1] - region1['centroid'][1]
        angle = np.arctan2(dy, dx)
        
        # Check if regions share boundary
        # Simplified: check if bounding boxes overlap
        bbox1 = region1['bbox']
        bbox2 = region2['bbox']
        
        overlap_x = max(0, min(bbox1[0]+bbox1[2], bbox2[0]+bbox2[2]) - 
                        max(bbox1[0], bbox2[0]))
        overlap_y = max(0, min(bbox1[1]+bbox1[3], bbox2[1]+bbox2[3]) - 
                        max(bbox1[1], bbox2[1]))
        
        shares_boundary = float(overlap_x > 0 and overlap_y > 0)
        
        # Tissue type compatibility (same type = 1)
        same_tissue = float(region1['tissue_type'] == region2['tissue_type'])
        
        return np.array([
            dist / 1000,  # Normalize
            np.sin(angle),
            np.cos(angle),
            shares_boundary
        ])


# Integration function
def create_tissue_graph_predictor(node_features_dim=12):
    """
    Create a complete tissue graph prediction system
    """
    # Graph builder
    graph_builder = TissueGraphBuilder()
    
    # Graph neural network
    gnn = TissueGraphNetwork(
        node_features_dim=node_features_dim,
        hidden_dim=256,
        num_classes=3  # canonical, immune, stromal
    )
    
    return graph_builder, gnn 