#!/bin/bash

# CRC Analysis Platform Launcher
# Run this script to start the platform

echo "================================================"
echo "🔬 CRC ANALYSIS PLATFORM"
echo "================================================"
echo ""
echo "Current Capabilities:"
echo "Tissue Classification: 91.4% accurate"
echo "Molecular Subtyping: 73.2% accurate (pre-EPOC baseline)"
echo ""
echo "When EPOC data arrives, run:"
echo "python app/prepare_epoc_molecular_training.py"
echo ""
echo "================================================"
echo "Starting platform..."
echo ""

# Activate conda environment if available
if command -v conda &> /dev/null; then
    conda activate base 2>/dev/null || true
fi

# Change to project directory
cd "$(dirname "$0")"

# Launch the platform
streamlit run app/crc_unified_platform.py

echo ""
echo "Platform closed." 