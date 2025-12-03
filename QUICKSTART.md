# Quick Start Guide - Doctor Image Generator

Get up and running in 3 steps!

## Step 1: Run Setup (20-60 minutes)

This will install ComfyUI, download models (~23GB), and set up everything:

```bash
chmod +x setup.sh
./setup.sh
```

**What it does:**
- ✅ Installs ComfyUI and dependencies
- ✅ Downloads Flux.1-dev model (~23GB)
- ✅ Installs ReActor face swapping node
- ✅ Downloads upscale models
- ✅ Sets up directory structure

**Requirements:**
- Linux OS (Ubuntu 20.04+)
- NVIDIA GPU with 12GB+ VRAM
- ~50GB free disk space
- Python 3.10+

## Step 2: Verify Setup

```bash
python test_setup.py
```

This will check if everything is installed correctly.

## Step 3: Generate Your First Image

### 3a. Start ComfyUI Server

In one terminal:

```bash
./start_comfyui.sh
```

Wait for "To see the GUI go to: http://127.0.0.1:8188"

### 3b. Generate Image

In another terminal:

```bash
# Place your doctor's headshot in inputs/doctor.jpg
cp /path/to/doctor_photo.jpg inputs/doctor.jpg

# Generate!
python generate.py --doctor-image inputs/doctor.jpg --scenario consultant
```

**Output:** `outputs/doctor_consultant_TIMESTAMP.png`

## Available Scenarios

List all scenarios:

```bash
python generate.py --list-scenarios
```

Quick reference:
- `consultant` - Doctor consulting with patient
- `speaker` - Speaking at medical conference
- `candid_walk` - Walking in hospital corridor
- `desk_shot` - Working at executive desk
- `lab_coat_portrait` - Professional portrait
- `outdoor_professional` - Outdoor portrait
- And 6 more!

## Generate Multiple Images (Batch Mode)

```bash
python generate.py \
  --doctor-image inputs/doctor.jpg \
  --batch consultant speaker desk_shot candid_walk
```

This will generate 4 different scenarios automatically.

## Tips for Best Results

### Input Photo Requirements

✅ **Good:**
- High resolution (1024x1024px minimum)
- Face forward, looking at camera
- Well-lit, even lighting
- Clear facial features
- Passport/headshot style
- Plain or professional background

❌ **Avoid:**
- Low resolution or blurry
- Face at angle or profile
- Harsh shadows
- Sunglasses or obscured face
- Multiple people in photo

### Custom Prompts

Add specific details:

```bash
python generate.py \
  --doctor-image inputs/doctor.jpg \
  --scenario consultant \
  --custom "wearing blue surgical scrubs, friendly smile, modern hospital"
```

## Troubleshooting

### "CUDA out of memory"
- Close other GPU applications
- Reduce resolution in `config.yaml`:
  ```yaml
  image_settings:
    width: 768
    height: 768
  ```

### "Model not found"
- Re-run setup: `./setup.sh`
- Check `ComfyUI/models/checkpoints/` has flux model

### Face doesn't match well
- Use higher quality input photo
- Ensure face is clearly visible and forward-facing
- Adjust in `config.yaml`:
  ```yaml
  face_swap:
    blend_strength: 0.90  # Increase from 0.85
  ```

### ComfyUI won't start
- Check port 8188 is not in use: `lsof -i :8188`
- Check logs in ComfyUI terminal
- Try: `cd ComfyUI && python main.py`

## Advanced Usage

### Python API

```python
from doctor_image_gen import DoctorImageGenerator

generator = DoctorImageGenerator()

result = generator.generate(
    doctor_image_path="inputs/doctor.jpg",
    scenario="consultant",
    photography_style="cinematic",
    custom_prompt="wearing glasses, warm smile"
)

print(f"Generated: {result['output_path']}")
```

### Custom Configuration

Edit `config.yaml` to adjust:
- Generation steps (speed vs quality)
- Image dimensions
- Face swap strength
- Upscale settings

### Use ComfyUI GUI

1. Start server: `./start_comfyui.sh`
2. Open browser: http://localhost:8188
3. Load workflow: `workflows/doctor_flux_workflow.json`
4. Customize nodes and generate

## Getting Help

1. Check `README.md` for detailed documentation
2. Run `test_setup.py` to diagnose issues
3. Check ComfyUI logs in the terminal

## What's Next?

- Experiment with different scenarios
- Try custom prompts for specific contexts
- Adjust configuration for your hardware
- Generate professional medical content at scale

Enjoy creating photorealistic doctor images! 🎉

