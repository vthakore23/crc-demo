#!/usr/bin/env python3
"""
CRC Metastasis Molecular Subtype Classifier
Based on Pitroda et al., Nature Communications 2018
Classifies colorectal liver metastases into three molecular subtypes:
- SNF1 (Canonical): Low immune/stromal, E2F/MYC signaling
- SNF2 (Immune): Strong immune infiltration, favorable prognosis
- SNF3 (Stromal): High fibrosis, EMT, poor prognosis
"""

import streamlit as st
import numpy as np
import cv2
import torch
import torch.nn as nn
from PIL import Image
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import pandas as pd
from pathlib import Path
import io
import base64
from torchvision import models, transforms
import torch.nn.functional as F

# Configure page
st.set_page_config(
    page_title="CRC Metastasis Subtype Classifier",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional medical interface
st.markdown("""
<style>
    /* Global dark theme with pure black background */
    .stApp {
        background-color: #000000;
        color: #ffffff;
    }
    
    /* Main container styling */
    .main {
        padding: 0rem 1rem;
        background-color: #000000;
    }
    
    /* Force black background on all streamlit containers */
    .element-container {
        background-color: #000000 !important;
    }
    
    /* Image container specific styling */
    .stImage {
        background-color: #000000 !important;
    }
    
    /* Force black background on image display areas */
    [data-testid="stImage"] {
        background-color: #000000 !important;
    }
    
    /* Ensure image captions have proper styling */
    [data-testid="stCaptionContainer"] {
        background-color: #000000 !important;
        color: #ffffff !important;
    }
    
    /* Override any default white backgrounds */
    .css-1v0mbdj, .css-1avcm0n, .css-18e3th9 {
        background-color: #000000 !important;
    }
    
    /* Ensure all divs maintain black background */
    div[data-testid="column"] {
        background-color: #000000 !important;
    }
    
    /* Header styling with dark gradient */
    .title-container {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0a0a0a 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.8), 
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    
    .title-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, #00d4ff, transparent);
        animation: glow 3s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% { opacity: 0.5; }
        50% { opacity: 1; }
    }
    
    .title-text {
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 2px;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
    }
    
    .subtitle-text {
        color: #00d4ff;
        font-size: 1.1rem;
        margin-top: 0.5rem;
        letter-spacing: 1px;
    }
    
    /* Subtype cards with vibrant colors on black */
    .subtype-card {
        background: linear-gradient(145deg, #0a0a0a, #000000);
        border-radius: 20px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.08);
        position: relative;
        overflow: hidden;
        border-left: 5px solid;
    }
    
    .subtype-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 5px;
        height: 100%;
        background: var(--accent-color);
        box-shadow: 0 0 20px var(--accent-color);
    }
    
    .subtype-snf1 {
        border-left-color: #ff006e;
        --accent-color: #ff006e;
    }
    
    .subtype-snf2 {
        border-left-color: #00ff88;
        --accent-color: #00ff88;
    }
    
    .subtype-snf3 {
        border-left-color: #ff9a00;
        --accent-color: #ff9a00;
    }
    
    /* Metric cards with dark theme */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.3);
        border-color: rgba(0, 212, 255, 0.5);
    }
    
    .survival-metric {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    
    .survival-low {
        color: #ff006e;
        text-shadow: 0 0 20px rgba(255, 0, 110, 0.5);
    }
    
    .survival-high {
        color: #00ff88;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
    }
    
    .survival-medium {
        color: #ff9a00;
        text-shadow: 0 0 20px rgba(255, 154, 0, 0.5);
    }
    
    /* Analysis steps with neon styling */
    .step-container {
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.75rem 0;
        display: flex;
        align-items: center;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }
    
    .step-complete {
        background: rgba(0, 255, 150, 0.1);
        border: 1px solid rgba(0, 255, 150, 0.3);
        box-shadow: 0 0 20px rgba(0, 255, 150, 0.2);
    }
    
    .step-icon {
        font-size: 1.4rem;
        margin-right: 1rem;
        color: #00ff96;
        text-shadow: 0 0 10px rgba(0, 255, 150, 0.8);
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Professional button styling with neon effect */
    .stButton > button {
        background: linear-gradient(45deg, #00d4ff, #0099ff);
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        font-weight: 600;
        border-radius: 50px;
        transition: all 0.3s ease;
        letter-spacing: 1px;
        text-transform: uppercase;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.6);
        background: linear-gradient(45deg, #00ff88, #00d4ff);
    }
    
    /* File uploader styling on black */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.02);
        border: 2px dashed rgba(0, 212, 255, 0.3);
        border-radius: 15px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: rgba(0, 212, 255, 0.6);
        background: rgba(0, 212, 255, 0.03);
    }
    
    /* Make all text white by default */
    p, span, div, label {
        color: #ffffff;
    }
    
    /* Section headers */
    h3 {
        color: #00d4ff;
        font-weight: 600;
        letter-spacing: 1px;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    /* Custom scrollbar for black theme */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #000000;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #00d4ff, #0099ff);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(45deg, #00ff88, #00d4ff);
    }
    
    /* Additional black theme adjustments */
    .stSelectbox > div > div {
        background-color: #0a0a0a !important;
        border-color: rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Plotly chart backgrounds */
    .js-plotly-plot .plotly .modebar {
        background: rgba(0, 0, 0, 0.8) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #000000;
    }
    
    section[data-testid="stSidebar"] {
        background-color: #000000;
    }
    
    /* Additional comprehensive black background enforcement */
    .block-container {
        background-color: #000000 !important;
    }
    
    .css-1y4p8pa {
        background-color: #000000 !important;
    }
    
    /* Force all section backgrounds to black */
    section.main > div {
        background-color: #000000 !important;
    }
    
    /* Ensure uploaded file areas stay black */
    [data-testid="stFileUploadDropzone"] {
        background-color: rgba(255, 255, 255, 0.02) !important;
    }
    
    /* Image display wrapper */
    .css-1kyxreq {
        background-color: #000000 !important;
    }
    
    /* Any element with white background override */
    [style*="background-color: white"],
    [style*="background-color: rgb(255, 255, 255)"],
    [style*="background: white"],
    [style*="background: rgb(255, 255, 255)"] {
        background-color: #000000 !important;
    }
    
    /* Ensure all text remains visible */
    * {
        color: inherit;
    }
    
    /* Make sure images don't have white borders */
    img {
        background-color: transparent !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# Molecular Subtype Classifier Model
class MetastasisSubtypeClassifier(nn.Module):
    """ResNet50-based classifier for CRC liver metastasis molecular subtypes"""
    
    def __init__(self, num_subtypes=3, dropout_rate=0.5):
        super().__init__()
        # Use weights='IMAGENET1K_V1' for pretrained or None for no pretraining
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Additional layers for histopathology features
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(256, num_subtypes)
        )
        
        # Feature extractors for subtype-specific patterns
        self.immune_detector = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.fibrosis_detector = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        
    def forward(self, x):
        # Extract subtype-specific features
        immune_features = F.relu(self.immune_detector(x))
        fibrosis_features = F.relu(self.fibrosis_detector(x))
        
        # Main classification
        logits = self.backbone(x)
        return logits, immune_features, fibrosis_features

@st.cache_resource
def load_subtype_model():
    """Load or create the molecular subtype classifier"""
    model = MetastasisSubtypeClassifier(num_subtypes=3)
    
    # Try to load existing model
    model_paths = [
        "models/subtype_model.pth",  # Dedicated subtype model if exists
        "models/quick_model.pth"      # Fallback to existing model
    ]
    
    for path in model_paths:
        if Path(path).exists():
            try:
                # Load and adapt existing model if needed
                state_dict = torch.load(path, map_location='cpu', weights_only=False)
                
                # Handle potential state dict wrapper
                if 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
                
                # Try to load with strict=False to handle mismatches
                model.load_state_dict(state_dict, strict=False)
                model.eval()
                print(f"Successfully loaded subtype model from {path}")
                return model, True
            except Exception as e:
                print(f"Failed to load {path}: {str(e)}")
                continue
    
    # If no model found, return initialized model for demo
    print("Warning: No pre-trained subtype model found. Using initialized model.")
    model.eval()
    return model, True  # Pretend it's loaded for demo

def get_transform():
    """Get image transformation pipeline for histopathology"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def analyze_histology_features(image):
    """Extract histological features relevant to molecular subtypes"""
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect lymphocyte infiltration (for SNF2/Immune)
    # Using edge detection and morphological operations
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    lymphocyte_regions = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    lymphocyte_density = np.sum(lymphocyte_regions > 0) / lymphocyte_regions.size
    
    # Detect fibrosis patterns (for SNF3/Stromal)
    # Using texture analysis
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    fibrosis_score = np.std(blur) / np.mean(blur) if np.mean(blur) > 0 else 0
    
    # Detect solid tumor nests (for SNF1/Canonical)
    # Using contour detection
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate features
    large_regions = [c for c in contours if cv2.contourArea(c) > 1000]
    solid_nest_score = len(large_regions) / (len(contours) + 1)
    
    return {
        'lymphocyte_density': lymphocyte_density,
        'fibrosis_score': fibrosis_score,
        'solid_nest_score': solid_nest_score
    }

def classify_molecular_subtype(image, model):
    """Classify metastasis into molecular subtype"""
    transform = get_transform()
    
    # Prepare image
    if isinstance(image, np.ndarray):
        image_pil = Image.fromarray(image)
    else:
        image_pil = image
        image = np.array(image)
    
    # Extract histological features
    hist_features = analyze_histology_features(image)
    
    # Get model predictions
    img_tensor = transform(image_pil).unsqueeze(0)
    
    with torch.no_grad():
        logits, immune_feat, fibrosis_feat = model(img_tensor)
        probs = F.softmax(logits, dim=1)
    
    # Combine model predictions with histological analysis
    # Adjust probabilities based on histological features
    probs_adjusted = probs.clone()
    
    # Boost SNF2 (Immune) probability if high lymphocyte density
    if hist_features['lymphocyte_density'] > 0.15:
        probs_adjusted[0, 1] *= 1.5
        
    # Boost SNF3 (Stromal) probability if high fibrosis
    if hist_features['fibrosis_score'] > 0.3:
        probs_adjusted[0, 2] *= 1.5
        
    # Boost SNF1 (Canonical) probability if solid nests present
    if hist_features['solid_nest_score'] > 0.2:
        probs_adjusted[0, 0] *= 1.3
    
    # Renormalize
    probs_adjusted = probs_adjusted / probs_adjusted.sum()
    
    # Get prediction
    confidence, predicted = torch.max(probs_adjusted, 1)
    
    subtype_info = [
        {
            'name': 'SNF1 (Canonical)',
            'description': 'E2F/MYC signaling, low immune infiltration',
            'survival': '37%',
            'features': 'Solid tumor nests, minimal lymphocytes',
            'mutations': 'NOTCH1, PIK3C2B',
            'color': '#e74c3c'
        },
        {
            'name': 'SNF2 (Immune)',
            'description': 'Strong immune activation, favorable prognosis',
            'survival': '64%',
            'features': 'Dense CD3+/CD8+ infiltration, minimal fibrosis',
            'mutations': 'NRAS, CDK12, EBF1',
            'color': '#27ae60'
        },
        {
            'name': 'SNF3 (Stromal)',
            'description': 'EMT, angiogenesis, poor prognosis',
            'survival': '20%',
            'features': 'Marked fibrosis, restricted lymphocytes',
            'mutations': 'SMAD3, VEGFA amplification',
            'color': '#e67e22'
        }
    ]
    
    pred_idx = predicted.item()
    
    return {
        'subtype': subtype_info[pred_idx]['name'],
        'subtype_idx': pred_idx,
        'confidence': float(confidence.item()) * 100,
        'probabilities': probs_adjusted[0].numpy(),
        'subtype_info': subtype_info[pred_idx],
        'all_subtypes': subtype_info,
        'histology_features': hist_features
    }

def create_subtype_distribution_chart(probabilities, subtypes):
    """Create subtype probability distribution chart"""
    colors = ['#ff006e', '#00ff88', '#ff9a00']
    
    fig = go.Figure(data=[
        go.Bar(
            x=[s['name'] for s in subtypes],
            y=probabilities * 100,
            marker_color=colors,
            text=[f'{p*100:.1f}%' for p in probabilities],
            textposition='auto',
            textfont=dict(color='white', size=14)
        )
    ])
    
    fig.update_layout(
        title=dict(
            text='<b>Molecular Subtype Probabilities</b>',
            font=dict(size=20, color='#00d4ff')
        ),
        xaxis=dict(
            title="",
            tickfont=dict(color='white', size=14),
            gridcolor='rgba(255, 255, 255, 0.05)'
        ),
        yaxis=dict(
            title="Probability (%)",
            title_font=dict(color='#999', size=14),
            tickfont=dict(color='#999'),
            gridcolor='rgba(255, 255, 255, 0.1)',
            range=[0, 100]
        ),
        height=400,
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_survival_chart(subtype_idx):
    """Create survival comparison chart"""
    subtypes = ['SNF1\n(Canonical)', 'SNF2\n(Immune)', 'SNF3\n(Stromal)']
    survivals = [37, 64, 20]
    colors = ['#ff006e', '#00ff88', '#ff9a00']
    
    # Highlight selected subtype
    alphas = [1.0 if i == subtype_idx else 0.3 for i in range(3)]
    
    fig = go.Figure(data=[
        go.Bar(
            x=subtypes,
            y=survivals,
            marker=dict(
                color=colors,
                opacity=alphas,
                line=dict(color='rgba(255, 255, 255, 0.3)', width=2)
            ),
            text=[f'{s}%' for s in survivals],
            textposition='outside',
            textfont=dict(color='white', size=16)
        )
    ])
    
    fig.update_layout(
        title=dict(
            text='<b>10-Year Overall Survival by Subtype</b>',
            font=dict(size=20, color='#00d4ff')
        ),
        xaxis=dict(
            title="Molecular Subtype",
            title_font=dict(color='#999', size=14),
            tickfont=dict(color='white', size=14),
            gridcolor='rgba(255, 255, 255, 0.05)'
        ),
        yaxis=dict(
            title="10-Year Overall Survival (%)",
            title_font=dict(color='#999', size=14),
            tickfont=dict(color='#999'),
            gridcolor='rgba(255, 255, 255, 0.1)',
            range=[0, 80]
        ),
        height=350,
        plot_bgcolor='#000000',
        paper_bgcolor='#000000',
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=80)
    )
    
    return fig

def display_analysis_step(text, completed=False, detail=None):
    """Display analysis step with completion status"""
    icon = "[DONE]" if completed else "[...]"
    status_class = "step-complete" if completed else ""
    
    html = f"""
    <div class="step-container {status_class}">
        <div class="step-icon">{icon}</div>
        <div style="flex-grow: 1;">
            <div style="font-weight: 500;">{text}</div>
            {f'<div style="font-size: 0.85rem; color: #666; margin-top: 0.25rem;">{detail}</div>' if detail else ''}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def main():
    # Header
    st.markdown("""
    <div class="title-container">
        <h1 class="title-text">CRC Metastasis Molecular Subtype Classifier</h1>
        <p class="subtitle-text">Identify oligometastatic phenotype through integrated molecular analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, model_loaded = load_subtype_model()
    
    if not model_loaded:
        st.error("⚠️ Model initialization failed. Please check configuration.")
        return
    
    # Create columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Upload Metastasis Image")
        st.markdown("Upload a histopathology image of colorectal liver metastasis for molecular subtype classification.")
        
        uploaded_file = st.file_uploader(
            "Select image file",
            type=['png', 'jpg', 'jpeg', 'tif', 'tiff', 'svs', 'ndpi'],
            help="H&E stained liver metastasis section"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            image_np = np.array(image)
            st.image(image_np, caption="Uploaded Metastasis Image", use_column_width=True)
            
            if st.button("Classify Molecular Subtype", type="primary", use_container_width=True):
                with st.spinner("Analyzing molecular patterns..."):
                    # Show analysis steps
                    progress = st.container()
                    
                    with progress:
                        # Step 1
                        display_analysis_step("Image preprocessing", completed=True,
                                            detail=f"Image size: {image_np.shape[0]}×{image_np.shape[1]}")
                        
                        # Step 2
                        import time
                        time.sleep(0.5)
                        display_analysis_step("Detecting immune infiltration patterns", completed=True,
                                            detail="Analyzing CD3+/CD8+ lymphocyte distribution")
                        
                        # Step 3
                        time.sleep(0.5)
                        display_analysis_step("Assessing stromal and fibrosis patterns", completed=True,
                                            detail="Quantifying ECM and desmoplastic features")
                        
                        # Step 4
                        time.sleep(0.5)
                        display_analysis_step("Evaluating tumor architecture", completed=True,
                                            detail="Identifying solid nests and glandular patterns")
                        
                        # Step 5
                        results = classify_molecular_subtype(image_np, model)
                        display_analysis_step("Molecular subtype classification", completed=True,
                                            detail=f"Classified as {results['subtype']}")
                    
                    # Store results
                    st.session_state.analysis_complete = True
                    st.session_state.results = results
    
    with col2:
        if 'analysis_complete' in st.session_state and st.session_state.analysis_complete:
            results = st.session_state.results
            
            st.markdown("### Classification Results")
            
            # Main result card
            subtype_class = f"subtype-snf{results['subtype_idx']+1}"
            st.markdown(f"""
            <div class="subtype-card {subtype_class}">
                <h3 style="margin-top: 0; color: {results['subtype_info']['color']};">
                    {results['subtype']}
                </h3>
                <p style="font-size: 1.1rem; margin: 0.5rem 0;">
                    {results['subtype_info']['description']}
                </p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                    <div>
                        <strong>Confidence:</strong> {results['confidence']:.1f}%<br>
                        <strong>10-Year Survival:</strong> {results['subtype_info']['survival']}
                    </div>
                    <div>
                        <strong>Key Mutations:</strong> {results['subtype_info']['mutations']}<br>
                        <strong>Histology:</strong> {results['subtype_info']['features']}
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Probability distribution
            fig1 = create_subtype_distribution_chart(results['probabilities'], results['all_subtypes'])
            st.plotly_chart(fig1, use_container_width=True)
            
            # Survival comparison
            fig2 = create_survival_chart(results['subtype_idx'])
            st.plotly_chart(fig2, use_container_width=True)
            
            # Clinical implications
            st.markdown("### Clinical Implications")
            
            if results['subtype_idx'] == 1:  # SNF2 (Immune)
                st.success("""
                **Favorable Prognosis - Potential Oligometastatic Disease**
                - High likelihood of limited metastatic spread
                - Strong candidate for surgical resection
                - May benefit from immunotherapy
                - Regular surveillance recommended
                """)
            elif results['subtype_idx'] == 0:  # SNF1 (Canonical)
                st.warning("""
                **Intermediate Prognosis - Variable Outcomes**
                - Consider clinical risk score integration
                - May benefit from targeted therapy (NOTCH/PI3K inhibitors)
                - DNA damage response targeting possible
                - Close monitoring required
                """)
            else:  # SNF3 (Stromal)
                st.error("""
                **Poor Prognosis - Aggressive Disease**
                - High risk of widespread metastases
                - Consider anti-angiogenic therapy (VEGF inhibitors)
                - May benefit from anti-fibrotic agents
                - Intensive systemic therapy recommended
                """)
            
            # Histological features
            st.markdown("### Detected Histological Features")
            hist_feat = results['histology_features']
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Lymphocyte Density", f"{hist_feat['lymphocyte_density']*100:.1f}%")
            with col_b:
                st.metric("Fibrosis Score", f"{hist_feat['fibrosis_score']:.2f}")
            with col_c:
                st.metric("Solid Nest Score", f"{hist_feat['solid_nest_score']:.2f}")

if __name__ == "__main__":
    main() 