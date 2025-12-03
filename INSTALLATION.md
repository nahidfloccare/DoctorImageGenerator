# Installation Guide

Complete step-by-step installation instructions for the Doctor Image Generator.

## System Requirements

### Minimum Requirements
- **OS**: Linux (Ubuntu 20.04+, Debian 11+, or similar)
- **GPU**: NVIDIA GPU with 12GB VRAM
- **RAM**: 16GB system RAM
- **Storage**: 50GB free disk space
- **Python**: 3.10 or higher
- **CUDA**: 11.8 or 12.1

### Recommended Requirements
- **GPU**: NVIDIA RTX 3090, 4090, or A5000/A6000
- **RAM**: 32GB system RAM
- **Storage**: 100GB+ SSD
- **Internet**: Fast connection for model downloads (~25GB)

### Tested Configurations

✅ **Working:**
- Ubuntu 22.04 + RTX 4090 (24GB) - Excellent
- Ubuntu 20.04 + RTX 3090 (24GB) - Excellent
- Debian 11 + RTX A5000 (24GB) - Excellent
- Ubuntu 22.04 + RTX 3080 (10GB) - Good (reduce resolution)

⚠️ **Limited:**
- RTX 3060 (12GB) - Works at 768x768 resolution
- RTX 2080 Ti (11GB) - Works with optimizations

❌ **Not Supported:**
- AMD GPUs (ComfyUI requires CUDA)
- Apple Silicon (MPS support experimental)
- CPUs only (too slow, 30+ minutes per image)

## Pre-Installation

### 1. Install NVIDIA Drivers

```bash
# Check if drivers are installed
nvidia-smi

# If not, install (Ubuntu/Debian)
sudo apt update
sudo apt install nvidia-driver-535
sudo reboot

# Verify after reboot
nvidia-smi
```

You should see your GPU listed with driver version 535+.

### 2. Install CUDA Toolkit (if needed)

```bash
# Check CUDA version
nvcc --version

# If not installed or version < 11.8:
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run
sudo sh cuda_12.1.0_530.30.02_linux.run

# Add to PATH (add to ~/.bashrc)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 3. Install Python 3.10+

```bash
# Check Python version
python3 --version

# If < 3.10, install (Ubuntu 22.04)
sudo apt update
sudo apt install python3.10 python3.10-venv python3-pip

# Make it default (optional)
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
```

### 4. Install Git

```bash
sudo apt update
sudo apt install git git-lfs
```

### 5. Install System Dependencies

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-dev \
    wget \
    curl \
    vim
```

## Main Installation

### Method 1: Automated Setup (Recommended)

```bash
# Navigate to project directory
cd /home/nahid/documents/DoctorImageGenerator

# Make setup script executable
chmod +x setup.sh

# Run setup (will take 20-60 minutes)
./setup.sh
```

The setup script will:
1. ✅ Clone ComfyUI
2. ✅ Install Python dependencies
3. ✅ Install custom nodes (ReActor, IP-Adapter, etc.)
4. ✅ Download Flux.1-dev model (~23GB)
5. ✅ Download auxiliary models (VAE, upscalers, face models)
6. ✅ Set up directory structure
7. ✅ Create configuration files

**What to expect:**
- Initial packages: 2-5 minutes
- ComfyUI setup: 5-10 minutes
- Model downloads: 20-40 minutes (depends on internet speed)
- Total: ~30-60 minutes

### Method 2: Manual Installation

If the automated script fails, follow these steps:

#### Step 1: Clone ComfyUI

```bash
cd /home/nahid/documents/DoctorImageGenerator
git clone https://github.com/comfyanonymous/ComfyUI.git
```

#### Step 2: Install ComfyUI Dependencies

```bash
cd ComfyUI
python3 -m pip install --upgrade pip
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install -r requirements.txt
cd ..
```

#### Step 3: Install Custom Nodes

```bash
cd ComfyUI/custom_nodes

# ComfyUI Manager
git clone https://github.com/ltdrdata/ComfyUI-Manager.git

# ReActor (Face Swap)
git clone https://github.com/Gourieff/comfyui-reactor-node.git
cd comfyui-reactor-node
pip install -r requirements.txt
cd ..

# IP-Adapter
git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git

# ComfyUI Essentials
git clone https://github.com/cubiq/ComfyUI_essentials.git

cd ../..
```

#### Step 4: Install Project Dependencies

```bash
pip install -r requirements.txt
```

#### Step 5: Download Models

**Set your Hugging Face token:**
```bash
export HF_TOKEN="your_huggingface_token_here"
```

**Download Flux.1-dev:**
```bash
# Using huggingface-cli
huggingface-cli login --token $HF_TOKEN

# Download model
huggingface-cli download \
    black-forest-labs/FLUX.1-dev \
    --local-dir models/flux1-dev \
    --include "*.safetensors" "*.json"

# Copy to ComfyUI
cp models/flux1-dev/*.safetensors ComfyUI/models/checkpoints/
```

**Download VAE:**
```bash
huggingface-cli download \
    black-forest-labs/FLUX.1-dev \
    --local-dir ComfyUI/models/vae \
    --include "ae.safetensors"
```

**Download InsightFace (Face Swap):**
```bash
mkdir -p ComfyUI/models/insightface
wget -O ComfyUI/models/insightface/inswapper_128.onnx \
    https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx
```

**Download Upscaler:**
```bash
mkdir -p ComfyUI/models/upscale_models
wget -O ComfyUI/models/upscale_models/4x-UltraSharp.pth \
    https://huggingface.co/Kim2091/UltraSharp/resolve/main/4x-UltraSharp.pth
```

## Post-Installation

### 1. Verify Installation

```bash
python test_setup.py
```

This will check:
- ✅ Python version
- ✅ GPU availability
- ✅ Required packages
- ✅ Directory structure
- ✅ Model files
- ✅ Configuration

### 2. Test ComfyUI Server

```bash
./start_comfyui.sh
```

Wait for: `To see the GUI go to: http://127.0.0.1:8188`

Open browser and navigate to http://localhost:8188

You should see the ComfyUI interface.

Press `Ctrl+C` to stop the server.

### 3. Generate Test Image

```bash
# Place a doctor's photo
cp /path/to/doctor_photo.jpg inputs/doctor.jpg

# Start ComfyUI server (in one terminal)
./start_comfyui.sh

# In another terminal, generate image
python generate.py --doctor-image inputs/doctor.jpg --scenario consultant
```

If successful, you'll see an image in `outputs/`.

## Troubleshooting Installation

### Issue: "CUDA out of memory" during model download

**Solution:**
Close other GPU applications, or use CPU for download:
```bash
export CUDA_VISIBLE_DEVICES=""
./setup.sh
```

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution:**
Install PyTorch manually:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: "Repository not found" when downloading Flux

**Solution:**
1. Verify your HF token has access to Flux.1-dev
2. Accept the license at: https://huggingface.co/black-forest-labs/FLUX.1-dev
3. Ensure token is correct in setup.sh

### Issue: "OSError: [Errno 28] No space left on device"

**Solution:**
Free up disk space:
```bash
# Check available space
df -h

# Clean apt cache
sudo apt clean

# Remove unused Docker images (if any)
docker system prune -a

# Move models to larger drive
mv models /path/to/larger/drive/models
ln -s /path/to/larger/drive/models models
```

### Issue: Git clone fails with "Permission denied"

**Solution:**
```bash
# Add SSH key to GitHub
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
# Add to GitHub: Settings → SSH Keys

# Or use HTTPS with token
git config --global credential.helper store
```

### Issue: "ImportError: libcudnn.so.8: cannot open shared object file"

**Solution:**
Install cuDNN:
```bash
sudo apt install libcudnn8 libcudnn8-dev
```

### Issue: ReActor node not working

**Solution:**
1. Ensure InsightFace model is downloaded:
   ```bash
   ls -lh ComfyUI/models/insightface/inswapper_128.onnx
   ```

2. Install onnxruntime-gpu:
   ```bash
   pip install onnxruntime-gpu
   ```

3. Restart ComfyUI

## Upgrading

### Update ComfyUI

```bash
cd ComfyUI
git pull
pip install -r requirements.txt
cd ..
```

### Update Custom Nodes

```bash
cd ComfyUI/custom_nodes
for dir in */; do
    cd "$dir"
    git pull
    if [ -f requirements.txt ]; then
        pip install -r requirements.txt
    fi
    cd ..
done
cd ../..
```

### Update Flux Model (when new version releases)

```bash
# Backup old model
mv ComfyUI/models/checkpoints/flux1-dev.safetensors \
   ComfyUI/models/checkpoints/flux1-dev.safetensors.backup

# Download new version
huggingface-cli download \
    black-forest-labs/FLUX.1-dev \
    --local-dir models/flux1-dev \
    --include "*.safetensors"

cp models/flux1-dev/*.safetensors ComfyUI/models/checkpoints/
```

## Uninstallation

To completely remove the installation:

```bash
cd /home/nahid/documents/DoctorImageGenerator

# Stop any running servers
pkill -f "python.*main.py"

# Remove all files
rm -rf ComfyUI/ models/ outputs/ inputs/*.jpg

# Keep your code and configs if you want
# rm -rf *.py *.sh *.md *.yaml prompts/ workflows/
```

## Cloud Installation

### RunPod

1. Create pod with:
   - GPU: RTX 4090 or A5000
   - Template: RunPod PyTorch
   - Volume: 50GB+

2. Connect via SSH

3. Run automated setup:
   ```bash
   git clone <your-repo>
   cd DoctorImageGenerator
   ./setup.sh
   ```

4. Expose port:
   ```bash
   # RunPod will auto-expose 8188
   ./start_comfyui.sh
   ```

### Google Cloud / AWS

Similar process, ensure:
- NVIDIA GPU instance (T4, V100, A100)
- CUDA drivers pre-installed
- Open port 8188 in firewall

## Next Steps

After successful installation:

1. Read [QUICKSTART.md](QUICKSTART.md) for usage guide
2. Run test generation
3. Explore scenarios: `python generate.py --list-scenarios`
4. Generate your first portfolio

## Getting Help

- Check `test_setup.py` output for diagnostics
- Review ComfyUI logs in terminal
- Ensure GPU has sufficient VRAM
- Verify all models downloaded correctly

For issues, provide:
- Output of `test_setup.py`
- GPU info from `nvidia-smi`
- Error messages from ComfyUI terminal

