#!/bin/bash

# Start Web Server for Doctor Image Generator
# This script starts the simple web UI

echo "========================================"
echo "🏥 Doctor Image Generator - Web Server"
echo "========================================"
echo ""

# Check if ComfyUI is running
if ! pgrep -f "main.py --listen 0.0.0.0 --port 8188" > /dev/null; then
    echo "⚠️  Warning: ComfyUI doesn't appear to be running"
    echo "Please start ComfyUI first with: ./start_comfyui.sh"
    echo ""
    read -p "Do you want to continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if Flask is installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "📦 Installing Flask dependencies..."
    pip install flask flask-cors
fi

# Create necessary directories
mkdir -p uploads outputs

echo "🚀 Starting web server..."
echo "🌐 Access the UI at: http://localhost:2000"
echo "🌐 Or from network: http://$(hostname -I | awk '{print $1}'):2000"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================"
echo ""

# Start the server
python3 web_server.py

