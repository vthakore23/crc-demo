"""
Enhanced Spatial Analysis for CRC Molecular Subtyping
Adds TME ecological features and hierarchical spatial context
"""

import numpy as np
import cv2
from scipy import ndimage, spatial
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from collections import Counter


class EnhancedSpatialAnalyzer:
    """Advanced spatial pattern analysis with ecological TME features"""
    
    def __init__(self):
        self.tme_analyzer = TMEEcologicalAnalyzer()
        self.hierarchical_analyzer = HierarchicalSpatialAnalyzer()
        
    def analyze_comprehensive_spatial_features(self, tissue_masks, image):
        """Extract comprehensive spatial features including TME ecology"""
        
        # Basic spatial patterns (existing functionality)
        basic_patterns = {
            'immune_highways': self._detect_immune_highways(tissue_masks['lymphocytes']),
            'stromal_barriers': self._analyze_stromal_barriers(
                tissue_masks['stroma'], tissue_masks['lymphocytes']
            ),
            'interface_sharpness': self._measure_interface_sharpness(
                tissue_masks['tumor'], tissue_masks['stroma']
            ),
            'lymphoid_aggregates': self._detect_lymphoid_aggregates(tissue_masks['lymphocytes'])
        }
        
        # TME ecological features (new)
        tme_features = self.tme_analyzer.analyze_tme_ecology(tissue_masks, image)
        
        # Hierarchical spatial features (new)
        hierarchical_features = self.hierarchical_analyzer.extract_multiscale_features(
            tissue_masks, image
        )
        
        return {
            'basic_patterns': basic_patterns,
            'tme_ecology': tme_features,
            'hierarchical_features': hierarchical_features
        }
    
    def _detect_immune_highways(self, lymphocyte_mask):
        """Existing immune highway detection - enhanced"""
        # Skeleton analysis for linear structures
        try:
            skeleton = cv2.ximgproc.thinning(
                (lymphocyte_mask * 255).astype(np.uint8)
            )
        except AttributeError:
            # Fallback: use morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
            skeleton = cv2.morphologyEx(
                (lymphocyte_mask * 255).astype(np.uint8),
                cv2.MORPH_GRADIENT,
                kernel
            )
        
        # Find continuous paths
        contours, _ = cv2.findContours(
            skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        highways = []
        for contour in contours:
            if cv2.arcLength(contour, False) > 50:  # Minimum length
                highways.append(contour)
                
        return {
            'highway_present': len(highways) > 0,
            'highway_count': len(highways),
            'total_highway_length': sum(cv2.arcLength(h, False) for h in highways),
            'highway_connectivity': self._compute_connectivity(highways)
        }
    
    def _analyze_stromal_barriers(self, stroma_mask, lymphocyte_mask):
        """Enhanced stromal barrier analysis"""
        # Compute distance transform
        dist_transform = cv2.distanceTransform(
            (1 - stroma_mask).astype(np.uint8), cv2.DIST_L2, 5
        )
        
        # Find thick stromal regions
        thick_stroma = dist_transform < 10  # Within 10 pixels of stroma
        
        # Check if stroma blocks lymphocytes
        blocked_regions = thick_stroma & (lymphocyte_mask < 0.1)
        
        # Analyze barrier patterns
        barrier_components = cv2.connectedComponentsWithStats(
            blocked_regions.astype(np.uint8), connectivity=8
        )
        
        return {
            'strong_barriers_present': barrier_components[0] > 2,
            'barrier_count': barrier_components[0] - 1,
            'encasement_pattern_count': self._count_encasement_patterns(
                stroma_mask, lymphocyte_mask
            ),
            'barrier_thickness': np.mean(dist_transform[thick_stroma]) if thick_stroma.any() else 0
        }
    
    def _measure_interface_sharpness(self, tumor_mask, stroma_mask):
        """Measure sharpness of tumor-stroma interface"""
        # Find interface pixels
        tumor_dilated = cv2.dilate(tumor_mask.astype(np.uint8), np.ones((3,3)))
        stroma_dilated = cv2.dilate(stroma_mask.astype(np.uint8), np.ones((3,3)))
        interface = tumor_dilated & stroma_dilated
        
        if not interface.any():
            return {'sharp_interfaces': False, 'sharpness_score': 0}
            
        # Compute gradient magnitude at interface
        gradient_x = cv2.Sobel(tumor_mask.astype(np.float32), cv2.CV_64F, 1, 0)
        gradient_y = cv2.Sobel(tumor_mask.astype(np.float32), cv2.CV_64F, 0, 1)
        gradient_mag = np.sqrt(gradient_x**2 + gradient_y**2)
        
        interface_sharpness = gradient_mag[interface].mean()
        
        return {
            'sharp_interfaces': interface_sharpness > 0.3,
            'sharpness_score': float(interface_sharpness),
            'interface_length': interface.sum(),
            'interface_complexity': self._compute_interface_complexity(interface)
        }
    
    def _detect_lymphoid_aggregates(self, lymphocyte_mask):
        """Detect lymphoid aggregates with size and density analysis"""
        # Find connected components
        components = cv2.connectedComponentsWithStats(
            (lymphocyte_mask > 0.5).astype(np.uint8), connectivity=8
        )
        
        aggregates = []
        for i in range(1, components[0]):
            area = components[2][i, cv2.CC_STAT_AREA]
            if area > 100:  # Minimum size for aggregate
                x, y, w, h = components[2][i, :4]
                density = lymphocyte_mask[y:y+h, x:x+w].mean()
                if density > 0.7:  # High density required
                    aggregates.append({
                        'area': area,
                        'density': density,
                        'circularity': self._compute_circularity(components[1] == i)
                    })
                    
        return {
            'aggregates_present': len(aggregates) > 0,
            'aggregate_count': len(aggregates),
            'total_aggregate_area': sum(a['area'] for a in aggregates),
            'mean_aggregate_density': np.mean([a['density'] for a in aggregates]) if aggregates else 0
        }
    
    def _compute_connectivity(self, contours):
        """Compute connectivity score for highway networks"""
        if len(contours) < 2:
            return 0
            
        # Build graph of nearby contours
        G = nx.Graph()
        for i, c1 in enumerate(contours):
            for j, c2 in enumerate(contours[i+1:], i+1):
                dist = self._min_contour_distance(c1, c2)
                if dist < 50:  # Connection threshold
                    G.add_edge(i, j, weight=1/dist)
                    
        # Compute connectivity metrics
        if G.number_of_nodes() > 0:
            return nx.average_clustering(G)
        return 0
    
    def _count_encasement_patterns(self, stroma_mask, lymphocyte_mask):
        """Count stromal encasement patterns around lymphocytes"""
        # Find lymphocyte regions surrounded by stroma
        lymph_components = cv2.connectedComponentsWithStats(
            (lymphocyte_mask > 0.5).astype(np.uint8)
        )
        
        encased_count = 0
        for i in range(1, lymph_components[0]):
            component_mask = lymph_components[1] == i
            dilated = cv2.dilate(component_mask.astype(np.uint8), np.ones((5,5)))
            border = dilated - component_mask
            
            if border.any() and (stroma_mask[border] > 0.8).mean() > 0.7:
                encased_count += 1
                
        return encased_count
    
    def _compute_interface_complexity(self, interface_mask):
        """Compute fractal dimension of interface"""
        # Box-counting method for fractal dimension
        sizes = [2, 4, 8, 16, 32]
        counts = []
        
        for size in sizes:
            count = 0
            for i in range(0, interface_mask.shape[0], size):
                for j in range(0, interface_mask.shape[1], size):
                    if interface_mask[i:i+size, j:j+size].any():
                        count += 1
            counts.append(count)
            
        # Fit log-log relationship
        if len(counts) > 1 and all(c > 0 for c in counts):
            coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
            return -coeffs[0]  # Fractal dimension
        return 1.0
    
    def _compute_circularity(self, mask):
        """Compute circularity metric"""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            area = cv2.contourArea(contours[0])
            perimeter = cv2.arcLength(contours[0], True)
            if perimeter > 0:
                return 4 * np.pi * area / (perimeter ** 2)
        return 0
    
    def _min_contour_distance(self, contour1, contour2):
        """Compute minimum distance between two contours"""
        distances = spatial.distance.cdist(contour1.reshape(-1, 2), contour2.reshape(-1, 2))
        return distances.min()


class TMEEcologicalAnalyzer:
    """Analyze tumor microenvironment ecology"""
    
    def analyze_tme_ecology(self, tissue_masks, image):
        """Extract TME ecological features"""
        
        # Compute cell-type diversity
        diversity_metrics = self._compute_diversity_metrics(tissue_masks)
        
        # Analyze spatial mixing
        mixing_metrics = self._compute_mixing_metrics(tissue_masks)
        
        # Compute interaction patterns
        interaction_metrics = self._compute_interaction_metrics(tissue_masks)
        
        # Analyze spatial gradients
        gradient_metrics = self._compute_gradient_metrics(tissue_masks)
        
        return {
            'diversity': diversity_metrics,
            'mixing': mixing_metrics,
            'interactions': interaction_metrics,
            'gradients': gradient_metrics
        }
    
    def _compute_diversity_metrics(self, tissue_masks):
        """Compute ecological diversity metrics"""
        # Get tissue proportions
        proportions = []
        total_tissue = 0
        
        for tissue_type, mask in tissue_masks.items():
            if tissue_type != 'empty':
                area = mask.sum()
                proportions.append(area)
                total_tissue += area
                
        if total_tissue > 0:
            proportions = np.array(proportions) / total_tissue
            
            # Shannon diversity
            shannon = -np.sum(proportions * np.log(proportions + 1e-10))
            
            # Simpson diversity
            simpson = 1 - np.sum(proportions ** 2)
            
            # Evenness
            evenness = shannon / np.log(len(proportions)) if len(proportions) > 1 else 0
            
            return {
                'shannon_diversity': float(shannon),
                'simpson_diversity': float(simpson),
                'evenness': float(evenness)
            }
            
        return {'shannon_diversity': 0, 'simpson_diversity': 0, 'evenness': 0}
    
    def _compute_mixing_metrics(self, tissue_masks):
        """Compute spatial mixing between tissue types"""
        # Focus on tumor-immune mixing
        tumor_mask = tissue_masks.get('tumor', np.zeros_like(list(tissue_masks.values())[0]))
        immune_mask = tissue_masks.get('lymphocytes', np.zeros_like(tumor_mask))
        
        if tumor_mask.any() and immune_mask.any():
            # Compute interface length
            tumor_boundary = cv2.Canny((tumor_mask * 255).astype(np.uint8), 100, 200)
            immune_boundary = cv2.Canny((immune_mask * 255).astype(np.uint8), 100, 200)
            
            # Distance-based mixing
            tumor_dist = cv2.distanceTransform(
                (1 - tumor_mask).astype(np.uint8), cv2.DIST_L2, 5
            )
            immune_dist = cv2.distanceTransform(
                (1 - immune_mask).astype(np.uint8), cv2.DIST_L2, 5
            )
            
            # Mixing score based on overlapping gradients
            mixing_zone = (tumor_dist < 20) & (immune_dist < 20)
            mixing_score = mixing_zone.sum() / (tumor_mask.sum() + immune_mask.sum())
            
            return {
                'tumor_immune_mixing': float(mixing_score),
                'interface_density': float((tumor_boundary & immune_boundary).sum() / tumor_boundary.sum()) 
                                   if tumor_boundary.any() else 0,
                'mixing_zone_area': float(mixing_zone.sum())
            }
            
        return {'tumor_immune_mixing': 0, 'interface_density': 0, 'mixing_zone_area': 0}
    
    def _compute_interaction_metrics(self, tissue_masks):
        """Compute cell-type interaction patterns"""
        interactions = {}
        
        # Pairwise interactions
        tissue_types = list(tissue_masks.keys())
        for i, type1 in enumerate(tissue_types):
            for type2 in tissue_types[i+1:]:
                if type1 != 'empty' and type2 != 'empty':
                    interaction_score = self._compute_pairwise_interaction(
                        tissue_masks[type1], tissue_masks[type2]
                    )
                    interactions[f'{type1}_{type2}_interaction'] = float(interaction_score)
                    
        return interactions
    
    def _compute_pairwise_interaction(self, mask1, mask2):
        """Compute interaction score between two tissue types"""
        # Dilate masks to find interaction zones
        dilated1 = cv2.dilate(mask1.astype(np.uint8), np.ones((5,5)))
        dilated2 = cv2.dilate(mask2.astype(np.uint8), np.ones((5,5)))
        
        # Interaction zone
        interaction = dilated1 & dilated2
        
        # Normalize by total area
        total_area = mask1.sum() + mask2.sum()
        if total_area > 0:
            return interaction.sum() / total_area
        return 0
    
    def _compute_gradient_metrics(self, tissue_masks):
        """Compute spatial gradients of tissue distributions"""
        gradients = {}
        
        for tissue_type, mask in tissue_masks.items():
            if tissue_type not in ['empty', 'debris'] and mask.any():
                # Gaussian smoothing for gradient computation
                smoothed = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)
                
                # Compute gradients
                grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=5)
                grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=5)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                
                gradients[f'{tissue_type}_gradient_mean'] = float(grad_mag.mean())
                gradients[f'{tissue_type}_gradient_max'] = float(grad_mag.max())
                
        return gradients


class HierarchicalSpatialAnalyzer:
    """Extract hierarchical spatial features at multiple scales"""
    
    def extract_multiscale_features(self, tissue_masks, image):
        """Extract features at tile, regional, and global levels"""
        
        # Tile-level features (existing approach)
        tile_features = self._extract_tile_features(tissue_masks)
        
        # Regional features (mesoscale)
        regional_features = self._extract_regional_features(tissue_masks, image)
        
        # Global features (whole image)
        global_features = self._extract_global_features(tissue_masks, image)
        
        return {
            'tile_level': tile_features,
            'regional_level': regional_features,
            'global_level': global_features
        }
    
    def _extract_tile_features(self, tissue_masks):
        """Extract local tile-level features"""
        # Divide into tiles and compute local statistics
        tile_size = 64
        features = []
        
        h, w = tissue_masks['tumor'].shape
        for i in range(0, h - tile_size, tile_size):
            for j in range(0, w - tile_size, tile_size):
                tile_feat = {}
                for tissue_type, mask in tissue_masks.items():
                    tile_mask = mask[i:i+tile_size, j:j+tile_size]
                    tile_feat[f'{tissue_type}_density'] = tile_mask.mean()
                features.append(tile_feat)
                
        # Aggregate statistics
        if features:
            aggregated = {}
            for key in features[0].keys():
                values = [f[key] for f in features]
                aggregated[f'{key}_mean'] = np.mean(values)
                aggregated[f'{key}_std'] = np.std(values)
                aggregated[f'{key}_max'] = np.max(values)
            return aggregated
            
        return {}
    
    def _extract_regional_features(self, tissue_masks, image):
        """Extract mesoscale regional features"""
        # Identify coherent tissue regions
        regions = {}
        
        for tissue_type, mask in tissue_masks.items():
            if tissue_type not in ['empty', 'debris']:
                # Find connected components
                components = cv2.connectedComponentsWithStats(
                    (mask > 0.5).astype(np.uint8)
                )
                
                # Analyze large regions
                large_regions = []
                for i in range(1, components[0]):
                    if components[2][i, cv2.CC_STAT_AREA] > 500:
                        large_regions.append({
                            'area': components[2][i, cv2.CC_STAT_AREA],
                            'aspect_ratio': components[2][i, cv2.CC_STAT_WIDTH] / 
                                          (components[2][i, cv2.CC_STAT_HEIGHT] + 1e-6),
                            'solidity': self._compute_solidity(components[1] == i)
                        })
                        
                if large_regions:
                    regions[f'{tissue_type}_num_regions'] = len(large_regions)
                    regions[f'{tissue_type}_mean_region_size'] = np.mean([r['area'] for r in large_regions])
                    regions[f'{tissue_type}_region_heterogeneity'] = np.std([r['area'] for r in large_regions])
                    
        return regions
    
    def _extract_global_features(self, tissue_masks, image):
        """Extract global architectural features"""
        # Global spatial arrangement
        features = {}
        
        # Compute global texture features
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Haralick features
        glcm = self._compute_glcm(gray)
        features['global_homogeneity'] = self._glcm_homogeneity(glcm)
        features['global_contrast'] = self._glcm_contrast(glcm)
        features['global_energy'] = self._glcm_energy(glcm)
        
        # Global tissue arrangement
        tissue_centroids = {}
        for tissue_type, mask in tissue_masks.items():
            if mask.any():
                y, x = np.where(mask > 0.5)
                tissue_centroids[tissue_type] = (x.mean(), y.mean())
                
        # Compute spatial dispersion
        if len(tissue_centroids) > 1:
            centroids_array = np.array(list(tissue_centroids.values()))
            features['tissue_spatial_dispersion'] = np.std(centroids_array, axis=0).mean()
            
        return features
    
    def _compute_solidity(self, mask):
        """Compute solidity of a region"""
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if contours:
            area = cv2.contourArea(contours[0])
            hull = cv2.convexHull(contours[0])
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                return area / hull_area
        return 0
    
    def _compute_glcm(self, gray_image):
        """Compute gray-level co-occurrence matrix"""
        # Simplified GLCM computation
        levels = 256
        glcm = np.zeros((levels, levels))
        
        # Horizontal pairs
        for i in range(gray_image.shape[0]):
            for j in range(gray_image.shape[1] - 1):
                glcm[gray_image[i, j], gray_image[i, j + 1]] += 1
                
        # Normalize
        glcm = glcm / glcm.sum()
        return glcm
    
    def _glcm_homogeneity(self, glcm):
        """Compute GLCM homogeneity"""
        i, j = np.ogrid[0:glcm.shape[0], 0:glcm.shape[1]]
        return (glcm / (1.0 + np.abs(i - j))).sum()
    
    def _glcm_contrast(self, glcm):
        """Compute GLCM contrast"""
        i, j = np.ogrid[0:glcm.shape[0], 0:glcm.shape[1]]
        return ((i - j) ** 2 * glcm).sum()
    
    def _glcm_energy(self, glcm):
        """Compute GLCM energy"""
        return (glcm ** 2).sum() 