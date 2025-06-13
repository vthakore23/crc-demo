#!/usr/bin/env python3
"""
CRC Molecular Subtype Analysis Platform
State-of-the-art AI for predicting molecular subtypes (Canonical, Immune, Stromal) 
from whole slide images for oligometastatic CRC assessment
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.classes.*")

import streamlit as st

# Set page config FIRST
st.set_page_config(
    page_title="CRC Molecular Subtype Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': "CRC Molecular Subtype Predictor v4.0 - State-of-the-Art AI for Oligometastatic CRC Assessment"
    }
)

# Import navigation handler
import sys
import os
from pathlib import Path

# Add app directory to path
app_dir = Path(__file__).parent / "app"
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "landing"

# Main app content
def main():
    try:
        from app.molecular_subtype_platform import (
            apply_molecular_theme,
            display_molecular_landing,
            display_molecular_upload,
            display_molecular_demo,
            display_epoc_dashboard,
            display_molecular_history,
            display_molecular_performance,
            display_molecular_sidebar
        )
        
        # Apply molecular theme
        apply_molecular_theme()
        
        # Add container to ensure content displays above background
        main_container = st.container()
        
        with main_container:
            # Show landing page or main app based on session state
            if st.session_state.current_page == "landing":
                # Show landing page
                display_molecular_landing()
                
                # Add launch button with spacing
                st.markdown("<br><br>", unsafe_allow_html=True)
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    if st.button("🚀 Enter Molecular Predictor", use_container_width=True, type="primary", key="main_launch"):
                        st.session_state.current_page = "app"
                        st.rerun()
                        
            else:
                # Show main app with sidebar navigation
                nav_option = display_molecular_sidebar()
                
                # Main content area
                content_container = st.container()
                with content_container:
                    # Display selected page
                    if nav_option == "🧬 Molecular Analysis":
                        display_molecular_upload()
                    elif nav_option == "🎯 Live Demo":
                        display_molecular_demo()
                    elif nav_option == "📊 EPOC Dashboard":
                        display_epoc_dashboard()
                    elif nav_option == "📈 Analysis History":
                        display_molecular_history()
                    elif nav_option == "🏆 Model Performance":
                        display_molecular_performance()
                        
    except Exception as e:
        st.error(f"❌ Application Error: {str(e)}")
        import traceback
        with st.expander("📋 Error Details"):
            st.code(traceback.format_exc())
        
        st.markdown("---")
        st.info("💡 **Troubleshooting Steps:**")
        st.markdown("""
        1. Ensure all dependencies are installed: `pip install -r requirements.txt`
        2. Check that the molecular foundation model is properly loaded
        3. Verify CUDA/GPU availability if using GPU acceleration
        4. Try refreshing the page or restarting the application
        """)

if __name__ == "__main__":
    main() 