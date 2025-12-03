#!/bin/bash

# Doctor Image Generator - Setup Script
# This script sets up ComfyUI with Flux.1-dev and ReActor for photorealistic doctor image generation

set -e  # Exit on error

echo "=========================================="
echo "Doctor Image Generator Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${RED}Error: This script is designed for Linux systems.${NC}"
    exit 1
fi

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}Warning: nvidia-smi not found. This project requires an NVIDIA GPU.${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ NVIDIA GPU detected${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# Check Python version
echo ""
echo "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 is not installed.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# Set up directory structure
echo ""
echo "Creating directory structure..."
mkdir -p models/checkpoints
mkdir -p models/vae
mkdir -p models/clip
mkdir -p models/unet
mkdir -p models/insightface
mkdir -p models/reactor
mkdir -p models/upscale_models
mkdir -p inputs
mkdir -p outputs
mkdir -p workflows
mkdir -p prompts
echo -e "${GREEN}✓ Directories created${NC}"

# Check if ComfyUI already exists
if [ -d "ComfyUI" ]; then
    echo ""
    echo -e "${YELLOW}ComfyUI directory already exists. Skipping clone.${NC}"
else
    echo ""
    echo "Cloning ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git
    echo -e "${GREEN}✓ ComfyUI cloned${NC}"
fi

# Install ComfyUI dependencies
echo ""
echo "Installing ComfyUI dependencies..."
cd ComfyUI
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install -r requirements.txt
cd ..
echo -e "${GREEN}✓ ComfyUI dependencies installed${NC}"

# Install ComfyUI Manager (for easier node installation)
echo ""
echo "Installing ComfyUI Manager..."
if [ -d "ComfyUI/custom_nodes/ComfyUI-Manager" ]; then
    echo -e "${YELLOW}ComfyUI Manager already exists. Skipping.${NC}"
else
    cd ComfyUI/custom_nodes
    git clone https://github.com/ltdrdata/ComfyUI-Manager.git
    cd ../..
    echo -e "${GREEN}✓ ComfyUI Manager installed${NC}"
fi

# Install ReActor Node
echo ""
echo "Installing ReActor Node for face swapping..."
if [ -d "ComfyUI/custom_nodes/comfyui-reactor-node" ]; then
    echo -e "${YELLOW}ReActor Node already exists. Skipping.${NC}"
else
    cd ComfyUI/custom_nodes
    git clone https://github.com/Gourieff/comfyui-reactor-node.git
    cd comfyui-reactor-node
    python3 -m pip install -r requirements.txt
    cd ../../..
    echo -e "${GREEN}✓ ReActor Node installed${NC}"
fi

# Install IP-Adapter for Flux
echo ""
echo "Installing IP-Adapter support..."
if [ -d "ComfyUI/custom_nodes/ComfyUI_IPAdapter_plus" ]; then
    echo -e "${YELLOW}IP-Adapter already exists. Skipping.${NC}"
else
    cd ComfyUI/custom_nodes
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git
    cd ../..
    echo -e "${GREEN}✓ IP-Adapter installed${NC}"
fi

# Install additional required nodes
echo ""
echo "Installing additional custom nodes..."

# ComfyUI Essentials
if [ ! -d "ComfyUI/custom_nodes/ComfyUI_essentials" ]; then
    cd ComfyUI/custom_nodes
    git clone https://github.com/cubiq/ComfyUI_essentials.git
    cd ../..
fi

# Install project Python dependencies
echo ""
echo "Installing project dependencies..."
python3 -m pip install -r requirements.txt
echo -e "${GREEN}✓ Project dependencies installed${NC}"

# Download InsightFace models for ReActor
echo ""
echo "Downloading InsightFace models..."
mkdir -p ComfyUI/models/insightface
python3 << 'PYTHON_SCRIPT'
import os
from huggingface_hub import hf_hub_download

token = os.getenv("HF_TOKEN", "your_huggingface_token_here")
target_dir = "ComfyUI/models/insightface"

try:
    # Download inswapper model for face swapping
    print("Downloading inswapper_128.onnx...")
    hf_hub_download(
        repo_id="ezioruan/inswapper_128.onnx",
        filename="inswapper_128.onnx",
        local_dir=target_dir,
        token=token
    )
    print("✓ InsightFace models downloaded")
except Exception as e:
    print(f"Warning: Could not download InsightFace model: {e}")
    print("You may need to download it manually from https://huggingface.co/ezioruan/inswapper_128.onnx")
PYTHON_SCRIPT
echo -e "${GREEN}✓ InsightFace setup complete${NC}"

# Download Flux.1-dev model
echo ""
echo "=========================================="
echo "IMPORTANT: Downloading Flux.1-dev Model"
echo "=========================================="
echo "This is a large file (~23GB). It may take 30-60 minutes depending on your connection."
echo ""

HF_TOKEN="${HF_TOKEN:-your_huggingface_token_here}"

python3 << PYTHON_SCRIPT
import os
from huggingface_hub import hf_hub_download, snapshot_download

token = "$HF_TOKEN"

print("Downloading Flux.1-dev model components...")
print("This will take some time. Please be patient.")
print("")

try:
    # Download Flux.1-dev model
    print("1. Downloading main FLUX model...")
    snapshot_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        local_dir="models/flux1-dev",
        token=token,
        allow_patterns=["*.safetensors", "*.json", "*.txt"]
    )
    print("✓ Flux.1-dev downloaded")
    
    # Download VAE
    print("\n2. Downloading VAE...")
    hf_hub_download(
        repo_id="black-forest-labs/FLUX.1-dev",
        filename="ae.safetensors",
        local_dir="ComfyUI/models/vae",
        token=token
    )
    print("✓ VAE downloaded")
    
    # Create symlink to ComfyUI models directory
    flux_path = os.path.abspath("models/flux1-dev")
    comfy_checkpoint = "ComfyUI/models/checkpoints/flux1-dev"
    
    if not os.path.exists(comfy_checkpoint):
        os.makedirs(os.path.dirname(comfy_checkpoint), exist_ok=True)
        if os.path.exists(flux_path):
            # Copy or link the main model file
            import shutil
            for file in os.listdir(flux_path):
                if file.endswith('.safetensors'):
                    src = os.path.join(flux_path, file)
                    dst = os.path.join("ComfyUI/models/checkpoints", file)
                    if not os.path.exists(dst):
                        shutil.copy2(src, dst)
                        print(f"✓ Copied {file} to ComfyUI checkpoints")
    
    print("\n✓ All Flux.1-dev components downloaded successfully!")
    
except Exception as e:
    print(f"\nError downloading models: {e}")
    print("\nYou may need to download the models manually:")
    print("1. Visit: https://huggingface.co/black-forest-labs/FLUX.1-dev")
    print("2. Download flux1-dev.safetensors")
    print("3. Place it in: ComfyUI/models/checkpoints/")
    exit(1)

PYTHON_SCRIPT

if [ $? -ne 0 ]; then
    echo -e "${RED}Model download failed. Please check the error messages above.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Flux.1-dev model setup complete${NC}"

# Download upscale model
echo ""
echo "Downloading 4x-UltraSharp upscale model..."
python3 << 'PYTHON_SCRIPT'
import os
from huggingface_hub import hf_hub_download

try:
    print("Downloading upscale model...")
    hf_hub_download(
        repo_id="Kim2091/UltraSharp",
        filename="4x-UltraSharp.pth",
        local_dir="ComfyUI/models/upscale_models"
    )
    print("✓ Upscale model downloaded")
except Exception as e:
    print(f"Warning: Could not download upscale model: {e}")
    print("You can continue without it, but images won't be upscaled.")
PYTHON_SCRIPT

# Create config file
echo ""
echo "Creating configuration file..."
cat > config.yaml << 'EOF'
# Doctor Image Generator Configuration

# Model Settings
model_settings:
  flux_model: "flux1-dev.safetensors"
  flux_path: "ComfyUI/models/checkpoints"
  steps: 20
  cfg_scale: 3.5
  sampler: "euler"
  scheduler: "simple"
  denoise: 1.0

# Image Generation Settings
image_settings:
  width: 1024
  height: 1024
  batch_size: 1

# Face Swap Settings (ReActor)
face_swap:
  enabled: true
  blend_strength: 0.85
  face_restore: true
  upscale_visibility: 1.0

# Upscale Settings
upscale:
  enabled: true
  scale: 2
  model: "4x-UltraSharp.pth"

# Prompt Settings
prompt_settings:
  negative_prompt: "low quality, blurry, distorted, deformed, ugly, bad anatomy, duplicate, watermark, signature, text, jpeg artifacts, worst quality"
  style_prefix: "Professional medical photography, high quality, photorealistic, detailed,"
  
# Instagram Aesthetic Keywords
instagram_style:
  lighting:
    - "soft studio lighting"
    - "golden hour lighting"
    - "natural window light"
    - "cinematic lighting"
    - "professional lighting"
  
  camera:
    - "shot on Sony A7R IV"
    - "85mm lens"
    - "f/1.8 aperture"
    - "bokeh background"
    - "shallow depth of field"
  
  vibe:
    - "candid"
    - "professional yet approachable"
    - "medical lifestyle photography"
    - "authentic moment"
    - "confident and warm"

# Paths
paths:
  comfyui: "ComfyUI"
  inputs: "inputs"
  outputs: "outputs"
  workflows: "workflows"
EOF
echo -e "${GREEN}✓ Configuration file created${NC}"

# Make scripts executable
echo ""
echo "Setting up launch scripts..."
chmod +x start_comfyui.sh
echo -e "${GREEN}✓ Scripts configured${NC}"

# Final message
echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Start ComfyUI server:"
echo "   ./start_comfyui.sh"
echo ""
echo "2. Generate images using Python API:"
echo "   python3 generate.py --doctor-image inputs/doctor.jpg --scenario consultant"
echo ""
echo "3. Or access ComfyUI web interface:"
echo "   http://localhost:8188"
echo ""
echo "For more information, see README.md"
echo ""

