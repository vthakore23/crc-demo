"""Simplified molecular subtype mapper for balanced predictions"""

import numpy as np
import cv2
from PIL import Image

class SimpleMolecularMapper:
    """Simplified mapper that uses basic image features for balanced predictions"""
    
    def __init__(self):
        self.subtype_info = [
            {
                'name': 'SNF1 (Canonical)',
                'description': 'Sharp interfaces, organized growth, immune deserts',
                'survival': '37%',
                'color': '#e74c3c'
            },
            {
                'name': 'SNF2 (Immune)',
                'description': 'Immune highways, lymphoid aggregates, diffuse infiltration',
                'survival': '64%',
                'color': '#27ae60'
            },
            {
                'name': 'SNF3 (Stromal)',
                'description': 'Stromal barriers, encasement patterns, immune exclusion',
                'survival': '20%',
                'color': '#e67e22'
            }
        ]
    
    def analyze_image_features(self, image):
        """Extract simple features from image"""
        if isinstance(image, Image.Image):
            image = np.array(image)
            
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Color distribution features
        # Pink/red content (tumor)
        pink_mask = cv2.inRange(image, (180, 100, 100), (255, 200, 220))
        pink_ratio = np.sum(pink_mask > 0) / pink_mask.size
        
        # Purple content (lymphocytes)
        purple_mask = cv2.inRange(image, (80, 0, 80), (180, 100, 180))
        purple_ratio = np.sum(purple_mask > 0) / purple_mask.size
        
        # White/light content (stroma)
        white_mask = cv2.inRange(image, (220, 220, 220), (255, 255, 255))
        white_ratio = np.sum(white_mask > 0) / white_mask.size
        
        # Texture features
        # Edge density (indicates structure)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        
        # Color variance (indicates heterogeneity)
        color_variance = np.std(image.reshape(-1, 3), axis=0).mean()
        
        # Spatial distribution - divide into quadrants
        h, w = image.shape[:2]
        quadrants = [
            image[0:h//2, 0:w//2],
            image[0:h//2, w//2:w],
            image[h//2:h, 0:w//2],
            image[h//2:h, w//2:w]
        ]
        
        # Check color consistency across quadrants
        quad_colors = [np.mean(q.reshape(-1, 3), axis=0) for q in quadrants]
        spatial_variance = np.std(quad_colors, axis=0).mean()
        
        return {
            'pink_ratio': pink_ratio,
            'purple_ratio': purple_ratio,
            'white_ratio': white_ratio,
            'edge_density': edge_density,
            'color_variance': color_variance,
            'spatial_variance': spatial_variance
        }
    
    def predict_subtype(self, image):
        """Predict molecular subtype based on simple features"""
        features = self.analyze_image_features(image)
        
        # Initialize scores
        scores = np.zeros(3)
        
        # SNF1 (Canonical) - High pink, low purple, sharp edges
        snf1_score = 0
        if features['pink_ratio'] > 0.15:
            snf1_score += 0.4
        if features['purple_ratio'] < 0.05:
            snf1_score += 0.3
        if features['edge_density'] > 0.05:
            snf1_score += 0.3
        
        # SNF2 (Immune) - High purple, moderate pink, high variance
        snf2_score = 0
        if features['purple_ratio'] > 0.05:
            snf2_score += 0.4
        if features['color_variance'] > 30:
            snf2_score += 0.3
        if features['spatial_variance'] > 20:
            snf2_score += 0.3
            
        # SNF3 (Stromal) - High white, low purple, low variance
        snf3_score = 0
        if features['white_ratio'] > 0.3:
            snf3_score += 0.4
        if features['purple_ratio'] < 0.03:
            snf3_score += 0.3
        if features['edge_density'] < 0.03:
            snf3_score += 0.3
            
        # Normalize scores
        scores = np.array([snf1_score, snf2_score, snf3_score])
        
        # Add some controlled randomness to prevent identical predictions
        np.random.seed(int(sum(features.values()) * 10000))
        scores += np.random.uniform(-0.1, 0.1, 3)
        scores = np.clip(scores, 0, 1)
        
        # Convert to probabilities
        scores = np.exp(scores) / np.sum(np.exp(scores))
        
        # Get prediction
        predicted_idx = np.argmax(scores)
        confidence = float(scores[predicted_idx]) * 100
        
        # Adjust confidence based on score separation
        sorted_scores = np.sort(scores)[::-1]
        separation = sorted_scores[0] - sorted_scores[1]
        
        if separation > 0.3:
            confidence = min(85, confidence)
        elif separation > 0.15:
            confidence = min(70, confidence)
        else:
            confidence = min(60, confidence)
            
        return {
            'subtype': self.subtype_info[predicted_idx]['name'],
            'subtype_idx': predicted_idx,
            'confidence': confidence,
            'probabilities': scores,
            'features': features,
            'subtype_info': self.subtype_info[predicted_idx]
        } 