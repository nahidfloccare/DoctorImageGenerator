#!/bin/bash

# Start ComfyUI server for Doctor Image Generator

echo "Starting ComfyUI server..."
echo "Access the interface at: http://localhost:8188"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd ComfyUI
python3 main.py --listen 0.0.0.0 --port 8188

