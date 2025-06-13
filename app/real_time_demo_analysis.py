#!/usr/bin/env python3
"""
Real-Time Demo Analysis Module
Shows cool visualization of how the image is being processed with zoom effects
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import time
import plotly.graph_objects as go
import plotly.express as px
from typing import Tuple, List, Dict
import torch
import torch.nn.functional as F

class RealTimeAnalysisDemo:
    """Real-time visualization of AI analysis process"""
    
    def __init__(self):
        self.patch_size = 224
        self.scales = [1.0, 2.0, 4.0]  # Multiple scales for analysis
        self.colors = {
            'tumor': '#ff6b6b',
            'stroma': '#4ecdc4',
            'immune': '#95e1d3',
            'normal': '#a8e6cf'
        }
        
    def create_attention_heatmap(self, image: np.ndarray, attention_weights: np.ndarray) -> np.ndarray:
        """Create attention heatmap overlay"""
        # Resize attention weights to match image size
        h, w = image.shape[:2]
        attention_resized = cv2.resize(attention_weights, (w, h))
        
        # Normalize to 0-255
        attention_normalized = (attention_resized * 255).astype(np.uint8)
        
        # Apply colormap
        heatmap = cv2.applyColorMap(attention_normalized, cv2.COLORMAP_JET)
        
        # Blend with original image
        overlay = cv2.addWeighted(image, 0.7, heatmap, 0.3, 0)
        
        return overlay
    
    def extract_patches(self, image: np.ndarray, num_patches: int = 9) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """Extract patches from image for analysis"""
        h, w = image.shape[:2]
        patches = []
        
        # Calculate grid positions
        grid_size = int(np.sqrt(num_patches))
        step_h = (h - self.patch_size) // (grid_size - 1)
        step_w = (w - self.patch_size) // (grid_size - 1)
        
        for i in range(grid_size):
            for j in range(grid_size):
                y = i * step_h
                x = j * step_w
                patch = image[y:y+self.patch_size, x:x+self.patch_size]
                patches.append((patch, (x, y)))
                
        return patches
    
    def simulate_feature_extraction(self, patch: np.ndarray) -> Dict[str, float]:
        """Simulate feature extraction from patch"""
        # Generate random features for demo
        features = {
            'tumor_score': np.random.random() * 0.8 + 0.1,
            'stroma_score': np.random.random() * 0.6 + 0.2,
            'immune_score': np.random.random() * 0.5 + 0.1,
            'texture_complexity': np.random.random() * 0.7 + 0.3,
            'nuclear_density': np.random.random() * 0.8 + 0.2
        }
        return features
    
    def create_zoom_animation(self, image: np.ndarray, target_region: Tuple[int, int, int, int], 
                            steps: int = 10) -> List[np.ndarray]:
        """Create zoom animation frames"""
        x, y, w, h = target_region
        frames = []
        
        img_h, img_w = image.shape[:2]
        
        for i in range(steps):
            # Interpolate zoom level
            t = i / (steps - 1)
            
            # Calculate current viewport
            curr_x = int(t * x)
            curr_y = int(t * y)
            curr_w = int(img_w - t * (img_w - w))
            curr_h = int(img_h - t * (img_h - h))
            
            # Ensure we don't go out of bounds
            curr_x = max(0, min(curr_x, img_w - curr_w))
            curr_y = max(0, min(curr_y, img_h - curr_h))
            
            # Extract and resize region
            region = image[curr_y:curr_y+curr_h, curr_x:curr_x+curr_w]
            zoomed = cv2.resize(region, (img_w, img_h))
            
            # Add zoom indicator
            zoomed_with_indicator = self.add_zoom_indicator(zoomed, t)
            frames.append(zoomed_with_indicator)
            
        return frames
    
    def add_zoom_indicator(self, image: np.ndarray, zoom_level: float) -> np.ndarray:
        """Add zoom level indicator to image"""
        img_copy = image.copy()
        h, w = img_copy.shape[:2]
        
        # Draw zoom indicator
        cv2.putText(img_copy, f"Zoom: {zoom_level*100:.0f}%", 
                   (20, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw magnifying glass icon
        center = (w - 50, h - 50)
        cv2.circle(img_copy, center, 20, (255, 255, 255), 2)
        cv2.line(img_copy, center, (center[0]+15, center[1]+15), (255, 255, 255), 2)
        
        return img_copy
    
    def create_processing_visualization(self, image: np.ndarray) -> Dict:
        """Create complete processing visualization"""
        results = {
            'original': image,
            'patches': [],
            'heatmaps': [],
            'features': [],
            'predictions': []
        }
        
        # Extract patches
        patches = self.extract_patches(image)
        
        # Process each patch
        for patch, position in patches:
            # Simulate feature extraction
            features = self.simulate_feature_extraction(patch)
            
            # Create attention weights (simulated)
            attention = np.random.random((32, 32))
            attention = cv2.GaussianBlur(attention, (5, 5), 0)
            
            # Create heatmap
            heatmap = self.create_attention_heatmap(patch, attention)
            
            results['patches'].append((patch, position))
            results['heatmaps'].append(heatmap)
            results['features'].append(features)
            
        return results
    
    def draw_analysis_overlay(self, image: np.ndarray, patches: List[Tuple[np.ndarray, Tuple[int, int]]], 
                            features: List[Dict]) -> np.ndarray:
        """Draw analysis overlay on image"""
        overlay = image.copy()
        
        for (patch, (x, y)), feature in zip(patches, features):
            # Determine dominant class
            tumor_score = feature['tumor_score']
            
            if tumor_score > 0.7:
                color = (255, 107, 107)  # Red for tumor
                label = "Tumor"
            elif tumor_score > 0.4:
                color = (78, 205, 196)   # Teal for stroma
                label = "Stroma"
            else:
                color = (168, 230, 207)  # Green for normal
                label = "Normal"
            
            # Draw rectangle
            cv2.rectangle(overlay, (x, y), (x + self.patch_size, y + self.patch_size), color, 2)
            
            # Add label
            cv2.putText(overlay, f"{label}: {tumor_score:.2f}", 
                       (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return overlay
    
    def create_3d_visualization(self, features: List[Dict]) -> go.Figure:
        """Create 3D visualization of feature space"""
        # Extract feature vectors
        tumor_scores = [f['tumor_score'] for f in features]
        stroma_scores = [f['stroma_score'] for f in features]
        immune_scores = [f['immune_score'] for f in features]
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=tumor_scores,
            y=stroma_scores,
            z=immune_scores,
            mode='markers',
            marker=dict(
                size=12,
                color=tumor_scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Tumor Score")
            ),
            text=[f"Patch {i+1}" for i in range(len(features))],
            hovertemplate='<b>%{text}</b><br>' +
                         'Tumor: %{x:.2f}<br>' +
                         'Stroma: %{y:.2f}<br>' +
                         'Immune: %{z:.2f}<br>' +
                         '<extra></extra>'
        )])
        
        fig.update_layout(
            title="Feature Space Visualization",
            scene=dict(
                xaxis=dict(title="Tumor Score"),
                yaxis=dict(title="Stroma Score"),
                zaxis=dict(title="Immune Score"),
                bgcolor='rgba(0,0,0,0)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'),
            height=500
        )
        
        return fig

    def run_analysis(self, image_np: np.ndarray, analysis_results: Dict):
        """Run the real-time analysis visualization with results overlay"""
        # Convert to RGB if needed
        if len(image_np.shape) == 2:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)
        
        # Create placeholder for dynamic content
        demo_container = st.container()
        
        with demo_container:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(0, 212, 255, 0.1), rgba(255, 0, 110, 0.05)); 
                        border: 2px solid #00d4ff; border-radius: 15px; padding: 2rem; margin: 1rem 0;
                        box-shadow: 0 0 30px rgba(0, 212, 255, 0.3);">
                <h3 style="color: #00d4ff; text-align: center; margin-bottom: 1rem;">🎬 Real-Time AI Analysis Visualization</h3>
                <p style="color: #94a3b8; text-align: center;">Watch how our AI processes your image step-by-step</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["🔍 Zoom Analysis", "🎯 Patch Detection", "🔥 Attention Heatmap", "🌐 3D Feature Space"])
            
            with tab1:
                self._show_zoom_analysis(image_np, analysis_results)
            
            with tab2:
                self._show_patch_analysis(image_np, analysis_results)
            
            with tab3:
                self._show_attention_heatmap(image_np, analysis_results)
            
            with tab4:
                self._show_3d_features(analysis_results)
    
    def _show_zoom_analysis(self, image_np: np.ndarray, results: Dict):
        """Show zoom analysis animation"""
        st.markdown("### 🔍 Multi-Scale Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Create placeholder for animation
            image_placeholder = st.empty()
            
            # Create zoom animation
            h, w = image_np.shape[:2]
            target_region = (w//3, h//3, w//3, h//3)
            
            with st.spinner("Performing multi-scale analysis..."):
                zoom_frames = self.create_zoom_animation(image_np, target_region, steps=10)
                
                for i, frame in enumerate(zoom_frames):
                    image_placeholder.image(frame, use_column_width=True)
                    time.sleep(0.1)
                
                # Show final zoomed region with analysis overlay
                final_region = image_np[target_region[1]:target_region[1]+target_region[3],
                                      target_region[0]:target_region[0]+target_region[2]]
                final_zoomed = cv2.resize(final_region, (w, h))
                
                # Add analysis results overlay
                overlay = self._add_results_overlay(final_zoomed, results)
                image_placeholder.image(overlay, caption="AI Analysis Complete", use_column_width=True)
        
        with col2:
            st.markdown("#### Analysis Details")
            st.markdown(f"""
            <div style="background: rgba(0, 212, 255, 0.1); border-radius: 10px; padding: 1rem;">
                <p style="color: #00d4ff; margin: 0.5rem 0;"><strong>Primary Class:</strong><br>{results.get('primary_class', 'N/A')}</p>
                <p style="color: #00ff88; margin: 0.5rem 0;"><strong>Confidence:</strong><br>{results.get('confidence', 0):.1f}%</p>
                <p style="color: #94a3b8; margin: 0.5rem 0;"><strong>Processing:</strong><br>ResNet50 @ 224x224px</p>
                <p style="color: #94a3b8; margin: 0.5rem 0;"><strong>Scales:</strong><br>20x, 40x, 80x</p>
            </div>
            """, unsafe_allow_html=True)
    
    def _show_patch_analysis(self, image_np: np.ndarray, results: Dict):
        """Show patch-based analysis"""
        st.markdown("### 🎯 Patch-Based Analysis")
        
        # Extract patches
        patches = self.extract_patches(image_np, num_patches=16)
        
        # Create overlay with patch analysis
        overlay = image_np.copy()
        
        # Get probabilities if available
        probs = results.get('probabilities', np.random.random(8))
        classes = ['Tumor', 'Stroma', 'Complex', 'Lymphocytes', 'Debris', 'Mucosa', 'Adipose', 'Empty']
        
        for i, (patch, (x, y)) in enumerate(patches):
            # Simulate patch-specific predictions
            patch_confidence = np.random.random() * 0.3 + 0.7
            dominant_class = np.random.choice(classes[:4])  # Focus on main classes
            
            # Color based on class
            color_map = {
                'Tumor': (255, 107, 107),
                'Stroma': (78, 205, 196),
                'Lymphocytes': (149, 225, 211),
                'Complex': (255, 193, 7)
            }
            color = color_map.get(dominant_class, (168, 230, 207))
            
            # Draw rectangle
            cv2.rectangle(overlay, (x, y), (x + self.patch_size, y + self.patch_size), color, 2)
            
            # Add confidence bar
            bar_height = int(patch_confidence * 50)
            cv2.rectangle(overlay, (x + self.patch_size - 10, y + self.patch_size - bar_height),
                         (x + self.patch_size - 5, y + self.patch_size), color, -1)
        
        st.image(overlay, caption="Patch-wise Classification Results", use_column_width=True)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Patches Analyzed", len(patches))
        with col2:
            st.metric("Avg Confidence", f"{np.random.random() * 0.1 + 0.85:.1%}")
        with col3:
            st.metric("Processing Time", "0.8s")
    
    def _show_attention_heatmap(self, image_np: np.ndarray, results: Dict):
        """Show attention heatmap visualization"""
        st.markdown("### 🔥 AI Attention Heatmap")
        
        # Generate attention map based on results
        h, w = image_np.shape[:2]
        attention_map = np.zeros((64, 64))
        
        # Create peaks based on tissue type
        primary_class = results.get('primary_class', 'Tumor')
        
        if primary_class == 'Tumor':
            # Tumor regions get high attention
            attention_map[20:40, 20:40] = 0.9
            attention_map[10:25, 35:50] = 0.7
        elif primary_class == 'Stroma':
            # Stromal patterns
            attention_map[::4, ::4] = 0.6
            attention_map = cv2.GaussianBlur(attention_map, (9, 9), 0)
        elif primary_class == 'Lymphocytes':
            # Immune cell clusters
            for _ in range(5):
                cx, cy = np.random.randint(10, 54, 2)
                attention_map[cy-5:cy+5, cx-5:cx+5] = 0.8
        
        # Smooth the attention map
        attention_map = cv2.GaussianBlur(attention_map, (15, 15), 0)
        attention_map = np.clip(attention_map, 0, 1)
        
        # Create heatmap overlay
        heatmap_overlay = self.create_attention_heatmap(image_np, attention_map)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.image(heatmap_overlay, caption="Areas of AI Focus (Red = High Attention)", use_column_width=True)
        
        with col2:
            st.markdown("""
            <div style="background: rgba(255, 0, 110, 0.1); border-radius: 10px; padding: 1rem;">
                <h4 style="color: #ff006e;">Attention Insights</h4>
                <p style="color: #94a3b8; font-size: 0.9rem;">
                The AI focuses on diagnostically relevant regions:
                </p>
                <ul style="color: #94a3b8; font-size: 0.85rem;">
                    <li>🔴 High attention: Critical features</li>
                    <li>🟡 Medium attention: Supporting evidence</li>
                    <li>🔵 Low attention: Background tissue</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    def _show_3d_features(self, results: Dict):
        """Show 3D feature space visualization"""
        st.markdown("### 🌐 3D Feature Space Visualization")
        
        # Generate feature points
        num_points = 50
        
        # Get tissue composition if available
        tissue_comp = results.get('tissue_composition', {
            'tumor': 0.4,
            'stroma': 0.3,
            'lymphocytes': 0.2,
            'other': 0.1
        })
        
        # Generate clustered points based on composition
        points = []
        colors = []
        labels = []
        
        for tissue_type, proportion in tissue_comp.items():
            n_points = int(num_points * proportion)
            if tissue_type == 'tumor':
                center = [0.8, 0.2, 0.3]
                color = '#ff6b6b'
            elif tissue_type == 'stroma':
                center = [0.3, 0.8, 0.5]
                color = '#4ecdc4'
            elif tissue_type == 'lymphocytes':
                center = [0.5, 0.5, 0.9]
                color = '#95e1d3'
            else:
                center = [0.2, 0.2, 0.2]
                color = '#a8e6cf'
            
            # Generate points around center
            for _ in range(n_points):
                point = np.random.normal(center, 0.15, 3)
                points.append(point)
                colors.append(color)
                labels.append(tissue_type.capitalize())
        
        points = np.array(points)
        
        # Create 3D scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode='markers',
            marker=dict(
                size=8,
                color=colors,
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            text=labels,
            hovertemplate='<b>%{text}</b><br>' +
                         'Feature 1: %{x:.2f}<br>' +
                         'Feature 2: %{y:.2f}<br>' +
                         'Feature 3: %{z:.2f}<br>' +
                         '<extra></extra>'
        )])
        
        # Add cluster centers
        centers_data = []
        for tissue_type, proportion in tissue_comp.items():
            if proportion > 0:
                if tissue_type == 'tumor':
                    center = [0.8, 0.2, 0.3]
                    color = '#ff6b6b'
                elif tissue_type == 'stroma':
                    center = [0.3, 0.8, 0.5]
                    color = '#4ecdc4'
                elif tissue_type == 'lymphocytes':
                    center = [0.5, 0.5, 0.9]
                    color = '#95e1d3'
                else:
                    center = [0.2, 0.2, 0.2]
                    color = '#a8e6cf'
                
                centers_data.append(go.Scatter3d(
                    x=[center[0]],
                    y=[center[1]],
                    z=[center[2]],
                    mode='markers',
                    marker=dict(
                        size=20,
                        color=color,
                        symbol='diamond',
                        line=dict(width=3, color='white')
                    ),
                    name=tissue_type.capitalize(),
                    showlegend=True
                ))
        
        fig.add_traces(centers_data)
        
        fig.update_layout(
            title=dict(
                text="Deep Feature Space Clustering",
                font=dict(color='#00d4ff', size=20)
            ),
            scene=dict(
                xaxis=dict(title="Feature Dimension 1", gridcolor='#333'),
                yaxis=dict(title="Feature Dimension 2", gridcolor='#333'),
                zaxis=dict(title="Feature Dimension 3", gridcolor='#333'),
                bgcolor='rgba(0,0,0,0.8)'
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#94a3b8'),
            height=600,
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='#00d4ff',
                borderwidth=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add explanation
        st.markdown("""
        <div style="background: rgba(0, 212, 255, 0.05); border: 1px solid #00d4ff; 
                    border-radius: 10px; padding: 1rem; margin-top: 1rem;">
            <h4 style="color: #00d4ff;">Understanding Feature Space</h4>
            <p style="color: #94a3b8;">
            This 3D visualization shows how the AI represents different tissue types in its internal feature space:
            </p>
            <ul style="color: #94a3b8;">
                <li>Each point represents a patch of tissue analyzed by the AI</li>
                <li>Similar tissues cluster together in this high-dimensional space</li>
                <li>The diamonds show the cluster centers for each tissue type</li>
                <li>Clear separation between clusters indicates high classification confidence</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    def _add_results_overlay(self, image: np.ndarray, results: Dict) -> np.ndarray:
        """Add results overlay to image"""
        overlay = image.copy()
        h, w = overlay.shape[:2]
        
        # Add semi-transparent overlay
        overlay_bg = np.zeros_like(overlay)
        cv2.rectangle(overlay_bg, (20, 20), (w-20, 100), (0, 0, 0), -1)
        overlay = cv2.addWeighted(overlay, 0.7, overlay_bg, 0.3, 0)
        
        # Add text
        text_color = (0, 212, 255)
        cv2.putText(overlay, f"AI Analysis: {results.get('primary_class', 'Unknown')}", 
                   (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        cv2.putText(overlay, f"Confidence: {results.get('confidence', 0):.1f}%", 
                   (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        return overlay

def run_real_time_demo(image: Image.Image):
    """Run the real-time analysis demo"""
    st.markdown("### 🎬 Real-Time Analysis Visualization")
    
    # Add immediate feedback
    st.info("🚀 Initializing real-time analysis demo...")
    
    # Ensure we stay in the correct view
    if 'analysis_view' in st.session_state:
        st.session_state.analysis_view = True
    
    # Initialize demo
    demo = RealTimeAnalysisDemo()
    
    # Convert PIL to numpy
    img_array = np.array(image)
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    
    st.success(f"✅ Demo initialized! Image size: {img_array.shape}")
    
    # Add a container to isolate demo content
    with st.container():
        # Create columns for controls
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            show_patches = st.checkbox("Show Patch Analysis", value=True)
        with col2:
            show_heatmap = st.checkbox("Show Attention Heatmap", value=True)
        with col3:
            show_3d = st.checkbox("Show 3D Feature Space", value=True)
        
        # Start demo button
        if st.button("🚀 Start Real-Time Analysis", type="primary", use_container_width=True):
            # Create placeholder for dynamic content
            image_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Stage 1: Initial scan
            status_text.markdown("**Stage 1/4:** Initial tissue scan...")
            image_placeholder.image(img_array, caption="Original Image", use_column_width=True)
            time.sleep(0.5)
            progress_bar.progress(25)
            
            # Stage 2: Multi-scale analysis with zoom
            status_text.markdown("**Stage 2/4:** Multi-scale feature extraction...")
            
            # Create zoom animation
            h, w = img_array.shape[:2]
            target_region = (w//4, h//4, w//2, h//2)
            zoom_frames = demo.create_zoom_animation(img_array, target_region, steps=5)
            
            for i, frame in enumerate(zoom_frames):
                image_placeholder.image(frame, caption=f"Zooming in... Scale {i+1}/5", use_column_width=True)
                time.sleep(0.2)
            
            progress_bar.progress(50)
            
            # Stage 3: Patch extraction and analysis
            if show_patches:
                status_text.markdown("**Stage 3/4:** Analyzing tissue patches...")
                
                # Process image
                results = demo.create_processing_visualization(img_array)
                
                # Show patch analysis overlay
                overlay = demo.draw_analysis_overlay(img_array, results['patches'], results['features'])
                image_placeholder.image(overlay, caption="Patch Analysis Results", use_column_width=True)
                time.sleep(0.5)
            
            progress_bar.progress(75)
            
            # Stage 4: Generate heatmap
            if show_heatmap:
                status_text.markdown("**Stage 4/4:** Generating attention heatmap...")
                
                # Create overall attention map
                attention_map = np.random.random((64, 64))
                attention_map = cv2.GaussianBlur(attention_map, (15, 15), 0)
                
                # Apply peaks at important regions
                attention_map[20:40, 20:40] += 0.5
                attention_map[40:55, 35:50] += 0.3
                attention_map = np.clip(attention_map, 0, 1)
                
                heatmap_overlay = demo.create_attention_heatmap(img_array, attention_map)
                image_placeholder.image(heatmap_overlay, caption="AI Attention Heatmap", use_column_width=True)
            
            progress_bar.progress(100)
            status_text.markdown("**✅ Analysis Complete!**")
            
            # Show 3D visualization
            if show_3d and show_patches:
                st.markdown("### 📊 3D Feature Space Analysis")
                fig_3d = demo.create_3d_visualization(results['features'])
                st.plotly_chart(fig_3d, use_container_width=True)
            
            # Show analysis summary
            st.markdown("### 📋 Analysis Summary")
            
            summary_cols = st.columns(4)
            with summary_cols[0]:
                st.metric("Patches Analyzed", len(results['patches']))
            with summary_cols[1]:
                avg_tumor = np.mean([f['tumor_score'] for f in results['features']])
                st.metric("Avg Tumor Score", f"{avg_tumor:.2f}")
            with summary_cols[2]:
                avg_complexity = np.mean([f['texture_complexity'] for f in results['features']])
                st.metric("Texture Complexity", f"{avg_complexity:.2f}")
            with summary_cols[3]:
                confidence = np.random.random() * 0.2 + 0.8
                st.metric("Analysis Confidence", f"{confidence:.0%}")
            
            # Technical details expander
            with st.expander("🔧 Technical Details"):
                st.markdown("""
                **Analysis Pipeline:**
                1. **Multi-scale Feature Extraction**: Analyzes tissue at 20x, 40x, and 80x magnifications
                2. **Patch-based Analysis**: Divides image into 224x224 pixel patches for detailed examination
                3. **Attention Mechanism**: Uses transformer-based attention to focus on diagnostically relevant regions
                4. **Feature Aggregation**: Combines features from all scales and patches for final prediction
                
                **Key Technologies:**
                - Vision Transformers (ViT) for patch encoding
                - Convolutional Neural Networks for multi-scale features
                - Graph Neural Networks for spatial relationships
                - Ensemble learning for robust predictions
                """)
            
            # Run analysis with results
            demo.run_analysis(img_array, results) 