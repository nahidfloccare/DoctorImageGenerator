# Doctor Image Generator - Photorealistic Instagram Images

A state-of-the-art AI pipeline for generating photorealistic Instagram-style images of doctors using **Flux.1-dev**, **Flux Redux**, and **ReActor** face swapping technology.

## 🎯 Features

- **Photorealistic Generation**: Uses Flux.1-dev, the current king of open-source photorealism
- **Identity Preservation**: Maintains the doctor's facial features accurately using ReActor face swapping
- **Zero-Shot**: No training required - just upload 1 photo of the doctor
- **Instagram-Ready**: Professional medical lifestyle photography aesthetic
- **Multiple Scenarios**: Pre-built templates for various professional contexts

## 🏗️ Architecture

```
Input: Doctor's Photo (Headshot)
    ↓
Flux.1-dev (Base Photorealistic Generation)
    ↓
Flux Redux/IP-Adapter (Identity Guidance)
    ↓
ReActor (Face Detail Refinement & Swapping)
    ↓
Upscaler (4K Quality Enhancement)
    ↓
Output: Instagram-Ready Image
```

## 📋 Prerequisites

- **GPU**: NVIDIA GPU with at least 12GB VRAM (16GB+ recommended)
- **Storage**: ~50GB free space for models
- **OS**: Linux (tested on Ubuntu 22.04)
- **Python**: 3.10+
- **CUDA**: 11.8 or 12.1

## 🚀 Quick Start

### 1. Clone and Setup

```bash
cd /home/nahid/documents/DoctorImageGenerator
chmod +x setup.sh
./setup.sh
```

This will:
- Install ComfyUI
- Download Flux.1-dev and required models
- Install ReActor node
- Set up the Python environment

### 2. Start ComfyUI Server

```bash
./start_comfyui.sh
```

ComfyUI will be available at `http://localhost:8188`

### 3. Generate Images via Python API

```python
from doctor_image_gen import DoctorImageGenerator

# Initialize generator
generator = DoctorImageGenerator()

# Generate image
result = generator.generate(
    doctor_image_path="doctor_headshot.jpg",
    scenario="consultant",  # or: speaker, candid_walk, desk_shot, etc.
    custom_prompt="Additional details here"
)

print(f"Generated image: {result['output_path']}")
```

### 4. Use the CLI

```bash
python generate.py --doctor-image doctor.jpg --scenario consultant --output output.png
```

## 🎨 Available Scenarios

1. **consultant**: Doctor consulting with patient, holding tablet
2. **speaker**: Speaking at medical conference
3. **candid_walk**: Walking down hospital corridor
4. **desk_shot**: Working at desk with papers
5. **examination**: Examining patient
6. **team_meeting**: In a medical team discussion
7. **lab_coat**: Professional portrait in lab coat
8. **outdoor**: Outdoor professional shot

## 📁 Project Structure

```
DoctorImageGenerator/
├── setup.sh                 # Main setup script
├── start_comfyui.sh        # ComfyUI launcher
├── requirements.txt        # Python dependencies
├── doctor_image_gen.py     # Main Python API
├── generate.py             # CLI interface
├── workflows/
│   └── doctor_flux_workflow.json  # ComfyUI workflow
├── prompts/
│   └── templates.py        # Prompt templates
├── models/                 # Model storage (created during setup)
├── inputs/                 # Input doctor images
├── outputs/                # Generated images
└── ComfyUI/               # ComfyUI installation
```

## 🔧 Configuration

Edit `config.yaml` to customize:

```yaml
model_settings:
  flux_model: "flux1-dev.safetensors"
  steps: 20
  cfg_scale: 3.5
  
face_swap:
  enabled: true
  blend_strength: 0.85
  
upscale:
  enabled: true
  scale: 4
  model: "4x-UltraSharp"
```

## 💡 Tips for Best Results

1. **Input Photo Quality**: Use high-resolution, well-lit headshot (passport style)
2. **Face Forward**: Doctor should be facing the camera directly
3. **Good Lighting**: Even lighting without harsh shadows
4. **High Resolution**: Minimum 512x512px, preferably 1024x1024px

## 🐛 Troubleshooting

### Out of Memory Error
- Reduce image resolution in config
- Lower the number of steps
- Close other GPU applications

### Face Not Matching
- Ensure input photo is clear and high quality
- Increase face swap blend strength
- Try preprocessing the face image

## 📝 License

This project uses several open-source models and tools:
- Flux.1-dev: Apache 2.0
- ComfyUI: GPL-3.0
- ReActor: GPL-3.0

## 🙏 Credits

- **Flux.1-dev**: Black Forest Labs
- **ComfyUI**: comfyanonymous
- **ReActor**: Gourieff
- **InsightFace**: deepinsight

## 📞 Support

For issues and questions, please refer to the documentation or create an issue.

