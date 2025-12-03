# Architecture & Technical Details

## System Overview

The Doctor Image Generator uses a hybrid AI pipeline combining multiple state-of-the-art models to achieve photorealistic results with accurate identity preservation.

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Doctor's Photo                     │
│                   (Headshot, 1024x1024px)                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   STAGE 1: Pre-processing                    │
│  • Face detection and cropping                               │
│  • Image normalization                                       │
│  • Quality validation                                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            STAGE 2: Prompt Engineering                       │
│  • Scenario selection from templates                         │
│  • Instagram aesthetic keywords injection                    │
│  • Photography style composition                             │
│  • Negative prompt optimization                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         STAGE 3: Base Image Generation (Flux.1-dev)         │
│  • Photorealistic generation                                 │
│  • Scene composition and lighting                            │
│  • Body pose and environment                                 │
│  • Professional photography aesthetic                        │
│  Output: Base image with generic face                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         STAGE 4: Face Swapping (ReActor + InsightFace)      │
│  • Face detection in generated image                         │
│  • Feature extraction from doctor's photo                    │
│  • Precise face swapping                                     │
│  • Skin tone and lighting matching                           │
│  • Seamless blending                                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           STAGE 5: Enhancement & Upscaling                   │
│  • Face restoration (GFPGAN)                                 │
│  • 4x upscaling (UltraSharp)                                 │
│  • Detail enhancement                                        │
│  • Final quality optimization                                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              OUTPUT: Photorealistic Doctor Image             │
│         Instagram-ready, identity-preserved, 4K quality      │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. ComfyUI (Interface Layer)

**Purpose:** Node-based workflow orchestration  
**Why:** Allows complex multi-model pipelines with visual editing

- Node graph system for model chaining
- GPU memory management
- Async processing queue
- WebSocket API for external control

### 2. Flux.1-dev (Base Generation)

**Purpose:** Photorealistic image generation  
**Model:** Black Forest Labs Flux.1-dev (~23GB)  
**Why chosen:** Best open-source photorealism as of 2024

**Capabilities:**
- Superior lighting and texture understanding
- Professional photography style awareness
- Accurate scene composition
- Natural human poses and expressions

**Technical specs:**
- Architecture: Diffusion transformer
- Resolution: Up to 2048x2048
- Steps: 20-50 (sweet spot: 20-30)
- CFG Scale: 3.5 (Flux works best at lower CFG)

### 3. ReActor (Face Swapping)

**Purpose:** Identity preservation through face replacement  
**Technology:** InsightFace + Custom blending

**Why needed:** Flux can't memorize specific faces from single photos

**Process:**
1. Face detection using RetinaFace
2. Feature extraction with InsightFace
3. Face mesh warping and alignment
4. Pixel-level face swap
5. Seamless boundary blending
6. Skin tone matching

**Models used:**
- `inswapper_128.onnx` - Face swapping model
- `retinaface_resnet50` - Face detection
- `GFPGANv1.4.pth` - Face restoration

### 4. Prompt Engineering System

**Purpose:** Translate scenarios into effective prompts

**Architecture:**
```python
Template + Photography Keywords + Custom Additions
    ↓
Prompt Compiler
    ↓
Optimized Prompt for Flux
```

**Instagram Aesthetic Keywords:**
- Lighting: "soft studio lighting", "golden hour", etc.
- Camera: "shot on Sony A7R IV", "85mm f/1.8", etc.
- Style: "medical lifestyle photography", "candid moment"

**Negative Prompting:**
Explicitly avoiding common AI artifacts:
- Bad anatomy, distortions
- Digital artifacts (watermarks, signatures)
- Non-photorealistic styles (cartoon, anime)

### 5. Upscaling Pipeline

**Purpose:** Enhance final output to 4K quality

**Models:**
- 4x-UltraSharp: General purpose upscaler
- GFPGAN: Face-specific enhancement

**Process:**
1. 4x upscale (1024px → 4096px)
2. Face restoration pass
3. Resize to target resolution
4. Sharpening and detail enhancement

## Data Flow

### Input Processing
```
Doctor Photo (JPG/PNG)
    ↓
PIL Image Load
    ↓
Face Detection & Validation
    ↓
Crop to 1:1 aspect ratio
    ↓
Resize to 512x512
    ↓
Store in ComfyUI/input/
```

### Generation Flow
```
User Request
    ↓
Python API (doctor_image_gen.py)
    ↓
Prompt Builder (templates.py)
    ↓
Workflow JSON Update
    ↓
ComfyUI HTTP API
    ↓
WebSocket Queue
    ↓
Node Execution (GPU)
    ↓
Image Output
    ↓
File Save (outputs/)
```

## API Architecture

### Python API (`doctor_image_gen.py`)

**Class: DoctorImageGenerator**

```python
__init__(comfyui_url, config_path)
  ↓
generate(doctor_image, scenario, **options)
  ├── Load workflow JSON
  ├── Build prompts
  ├── Update workflow nodes
  ├── Queue to ComfyUI
  ├── Wait for completion
  └── Return result metadata

batch_generate(doctor_image, scenarios)
  └── Loop generate() for each scenario
```

**Key Methods:**
- `_queue_prompt()` - Submit to ComfyUI
- `_wait_for_completion()` - Poll for results
- `_get_image()` - Retrieve generated image
- `_update_workflow()` - Inject parameters into workflow

### CLI Interface (`generate.py`)

Argparse-based command line interface:
- Single image generation
- Batch mode
- Scenario listing
- Custom prompt support

## Configuration System

### YAML-based (`config.yaml`)

**Sections:**
1. **model_settings** - Flux parameters (steps, CFG, sampler)
2. **image_settings** - Dimensions, batch size
3. **face_swap** - ReActor configuration
4. **upscale** - Enhancement settings
5. **prompt_settings** - Style and negative prompts
6. **paths** - Directory structure

**Hot-reloadable:** Changes apply to next generation

## Performance Optimization

### GPU Memory Management

**Typical VRAM usage:**
- Flux.1-dev loading: ~11GB
- Generation: ~13GB peak
- Face swap: +1GB
- Upscaling: +2GB

**Optimization strategies:**
1. Model offloading to CPU when idle
2. Sequential node execution (not parallel)
3. Batch size = 1 for maximum quality
4. Automatic VRAM clearing between generations

### Speed Optimizations

**Generation time breakdown:**
- Model loading: 5-10s (first time only)
- Generation (20 steps): 30-60s
- Face swap: 2-5s
- Upscaling: 10-20s
- **Total: ~50-90s per image**

**Speed vs Quality trade-offs:**
- Reduce steps: 20 → 15 (30% faster, slight quality loss)
- Skip upscaling: 50% faster, lower resolution
- Lower resolution: 1024 → 768 (40% faster)

## Workflow Node Structure

### ComfyUI Workflow JSON

**Key Nodes:**

1. **CheckpointLoaderSimple** (Node 1)
   - Loads Flux.1-dev model
   
2. **CLIPTextEncode** (Nodes 2, 3)
   - Encodes positive/negative prompts
   
3. **EmptyLatentImage** (Node 4)
   - Creates latent space canvas
   
4. **KSampler** (Node 5)
   - Main generation logic
   - Denoising loop
   
5. **VAEDecode** (Node 6)
   - Converts latent to pixel space
   
6. **LoadImage** (Node 7)
   - Loads doctor's reference photo
   
7. **ReActorFaceSwap** (Node 8)
   - Face swapping operation
   
8. **ImageUpscaleWithModel** (Node 9)
   - 4x upscaling
   
9. **ImageResize** (Node 10)
   - Final size adjustment
   
10. **SaveImage** (Node 11)
    - Output to file

**Node connections:**
```
[1] → [2,3] → [5] → [6] → [8] → [9] → [10] → [11]
         ↓      ↑
        [4] ──┘
         
[7] → [8]  (reference face)
```

## Zero-Shot Design

**Key feature:** No per-doctor training required

**How it works:**
1. Flux generates photorealistic doctor in scenario
2. ReActor swaps the generated face with real doctor's face
3. No LoRA training, no fine-tuning, no dataset collection

**Benefits:**
- Instant usage (no 30-60 min training)
- No overfitting risks
- Scales to unlimited doctors
- No training data requirements

**Trade-offs:**
- Slightly less consistent than LoRA across multiple images
- Requires high-quality input photo
- Face swap may occasionally need manual review

## Extension Points

### Adding New Scenarios

1. Edit `prompts/templates.py`
2. Add to `SCENARIOS` dict:
```python
"new_scenario": {
    "name": "Display Name",
    "prompt": "Detailed prompt with {photography_keywords}",
    "focus": "key, themes, here"
}
```

### Custom Models

1. Download model to appropriate directory:
   - Checkpoints: `ComfyUI/models/checkpoints/`
   - LoRAs: `ComfyUI/models/loras/`
   - Upscalers: `ComfyUI/models/upscale_models/`

2. Update `config.yaml`:
```yaml
model_settings:
  flux_model: "your_model.safetensors"
```

### API Integration

**REST endpoints:**
- POST `/prompt` - Queue generation
- GET `/history/{prompt_id}` - Check status
- GET `/view` - Download image

**WebSocket:**
- Real-time progress updates
- Queue position monitoring
- Error notifications

## Security Considerations

### Input Validation

- Image format validation (PIL)
- Face detection verification
- File size limits
- Path traversal prevention

### Privacy

- Images stored locally only
- No cloud upload required
- Manual deletion of outputs
- No model fine-tuning = no data retention

### API Security

- Local-only by default (127.0.0.1)
- No authentication (add reverse proxy if exposing)
- Rate limiting recommended for production

## Future Enhancements

### Potential Additions

1. **Video Generation**
   - AnimateDiff integration
   - Short video clips of doctor scenarios

2. **LoRA Training**
   - Optional per-doctor fine-tuning
   - Better consistency across images

3. **Background Removal/Replacement**
   - Custom background selection
   - Green screen integration

4. **Batch Face Swap**
   - Multiple doctors in one image
   - Team photo generation

5. **Style Transfer**
   - Apply specific Instagram filters
   - Match existing brand aesthetics

6. **API Server**
   - FastAPI REST interface
   - Multi-user queue management
   - Database tracking

## References

- **Flux.1-dev**: https://huggingface.co/black-forest-labs/FLUX.1-dev
- **ComfyUI**: https://github.com/comfyanonymous/ComfyUI
- **ReActor**: https://github.com/Gourieff/comfyui-reactor-node
- **InsightFace**: https://github.com/deepinsight/insightface

