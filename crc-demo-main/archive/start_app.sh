#!/bin/bash
# Start the CRC Liver Metastasis Predictor App

echo "🔬 Starting CRC Liver Metastasis Predictor..."
echo "==========================================="

# Kill any existing streamlit processes
echo "Stopping any existing instances..."
ps aux | grep streamlit | grep -v grep | awk '{print $2}' | xargs kill 2>/dev/null

# Wait a moment
sleep 2

# Start the app
echo "Starting the app..."
streamlit run crc_metastasis_pipeline.py

echo "App is running at: http://localhost:8501" 