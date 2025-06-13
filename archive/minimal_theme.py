#!/usr/bin/env python3
"""Minimal theme for CRC Analysis Platform"""

import streamlit as st

def apply_minimal_theme():
    """Apply minimal dark theme that doesn't break layout"""
    theme_css = """<style>
        /* Dark background */
        .stApp {
            background-color: #0f1419;
        }
        
        /* White text */
        .stApp, .stApp p, .stApp span, .stApp div, .stApp h1, .stApp h2, .stApp h3 {
            color: white;
        }
        
        /* Dark sidebar */
        section[data-testid="stSidebar"] {
            background-color: #1a1f2e;
        }
        
        /* Style buttons */
        .stButton > button {
            background-color: #3b82f6;
            color: white;
            border: none;
            border-radius: 8px;
        }
        
        .stButton > button:hover {
            background-color: #2563eb;
        }
        
        /* Style metrics */
        [data-testid="metric-container"] {
            background-color: rgba(255, 255, 255, 0.05);
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
    </style>"""
    
    st.markdown(theme_css, unsafe_allow_html=True) 