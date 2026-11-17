#!/bin/bash
set -e

# Azure App Service startup script for WikiSage
echo "Starting WikiSage on Azure App Service..."

# Set default port if not provided
export PORT=${PORT:-8501}

# Always install dependencies (Oryx build is disabled)
echo "Installing Python dependencies..."
pip install --no-cache-dir -r requirements.txt 2>&1 | tail -5
echo "Dependencies installed."

# Start the Streamlit application
echo "Starting Streamlit application on port $PORT..."
python -m streamlit run app.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false
