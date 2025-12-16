# Technical Design Document (TDD)
## Doctor Image Generator

**Version:** 1.0  
**Date:** December 15, 2025  
**Author:** Nahid@floccare.ai 

---

## 1. Executive Summary

The Doctor Image Generator is an AI-powered professional medical photography system that generates photorealistic doctor portraits while preserving the identity of the original subject. The system uses a multi-stage pipeline combining FLUX diffusion models, PuLID identity preservation, face/hand refinement, and REAL1SM realistic enhancement.

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Web Browser                                  │
│                    (http://localhost:8000)                          │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Flask Web Server                                │
│                      (web_server.py)                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │   Upload    │  │   Generate  │  │   Progress  │                  │
│  │   Handler   │  │     API     │  │   Stream    │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  Doctor Image Generator                              │
│                  (doctor_image_gen.py)                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                  │
│  │  Workflow   │  │   Prompt    │  │   Config    │                  │
│  │   Loader    │  │   Builder   │  │   Manager   │                  │
│  └─────────────┘  └─────────────┘  └─────────────┘                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        ComfyUI Backend                               │
│                    (http://localhost:8188)                           │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    Doctor.json Workflow                      │    │
│  │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐      │    │
│  │  │PuLID│→│FLUX │→│Hand │→│React│→│Face │→│REAL │      │    │
│  │  │     │  │Gen  │  │Det  │  │or   │  │Det  │  │1SM  │      │    │
│  │  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘  └─────┘      │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         GPU (NVIDIA A100)                            │
│                          40GB VRAM                                   │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Overview

| Component | Technology | Port | Purpose |
|-----------|------------|------|---------|
| Web UI | Flask + HTML/JS | 8000 | User interface |
| API Server | Flask REST | 8000 | Handle requests |
| Image Generator | Python | N/A | Workflow orchestration |
| ComfyUI | Python + PyTorch | 8188 | AI image generation |
| GPU | NVIDIA A100 | N/A | Model inference |

---

## 3. Image Generation Pipeline

### 3.1 Workflow Stages

```
Stage 1: Identity Encoding (PuLID)
├── Load 1-3 reference images
├── Extract face embeddings via InsightFace (CPU)
├── Encode identity with EVA-CLIP
└── Output: Identity conditioning tensor

Stage 2: Main Generation (FLUX)
├── Model: flux1-dev.safetensors (23GB)
├── Steps: 100
├── Sampler: euler
├── Scheduler: beta
├── CFG: 1.0
├── Resolution: 1024x1024
└── Output: Base generated image

Stage 3: Hand Refinement (FaceDetailer)
├── Detector: hand_yolov9c.pt
├── Steps: 50
├── Denoise: 0.25
└── Output: Image with refined hands

Stage 4: Face Swap (ReActor)
├── Swap Model: inswapper_128.onnx
├── Face Restore: CodeFormer
├── CodeFormer Weight: 0.5
└── Output: Face-swapped image

Stage 5: Face Refinement (FaceDetailer)
├── Detector: face_yolov8m.pt
├── Steps: 150
├── Denoise: 0.66
├── Scheduler: beta
└── Output: Image with refined face

Stage 6: Realistic Refinement (REAL1SM)
├── Model: REAL1SM_V3_FP8.safetensors (12GB)
├── Steps: 35
├── Denoise: 0.4
├── Scheduler: karras
└── Output: Photorealistic refined image

Stage 7: Upscaling
├── Method: lanczos
├── Scale: 4x
├── Output Resolution: 4096x4096
└── Output: Final high-resolution image
```

### 3.2 Node Graph (Doctor.json)

```
[LoadImage x3] ──→ [ResizeAndPad] ──→ [ImageBatch] ──→ [ApplyPulidFlux]
                                                              │
[CheckpointLoader: flux1-dev] ─────────────────────────────────┘
                                                              │
[DualCLIPLoader] ──→ [CLIPTextEncode] ────────────────────────┤
                                                              ▼
                                                        [KSampler: 100 steps]
                                                              │
                                                        [VAEDecode]
                                                              │
                                                        [FaceDetailer: Hands]
                                                              │
                                                        [FreeMemoryImage]
                                                              │
                                                        [ReActorFaceSwap]
                                                              │
                                                        [FaceDetailer: Face]
                                                              │
                                                        [VAEEncode]
                                                              │
[CheckpointLoader: REAL1SM] ──────────────────────────→ [KSampler: 35 steps]
                                                              │
                                                        [VAEDecode]
                                                              │
                                                        [ImageScaleBy: 4x]
                                                              │
                                                        [SaveImage]
```

---

## 4. Models & Resources

### 4.1 AI Models

| Model | Size | Purpose | Location |
|-------|------|---------|----------|
| flux1-dev.safetensors | 23GB | Main generation | models/checkpoints/ |
| REAL1SM_V3_FP8.safetensors | 12GB | Realistic refinement | models/checkpoints/ |
| pulid_flux_v0.9.1.safetensors | 1.1GB | Identity preservation | models/pulid/ |
| ae.safetensors | 320MB | VAE encoder/decoder | models/vae/ |
| t5xxl_fp16.safetensors | ~10GB | Text encoder | models/clip/ |
| clip_l.safetensors | ~500MB | CLIP encoder | models/clip/ |
| inswapper_128.onnx | 529MB | Face swap | models/insightface/ |
| codeformer-v0.1.0.pth | ~400MB | Face restoration | models/facerestore_models/ |
| hand_yolov9c.pt | 50MB | Hand detection | models/ultralytics/bbox/ |
| face_yolov8m.pt | 50MB | Face detection | models/ultralytics/bbox/ |

### 4.2 Hardware Requirements

| Resource | Minimum | Recommended | Current |
|----------|---------|-------------|---------|
| GPU VRAM | 24GB | 40GB+ | 40GB (A100) |
| System RAM | 32GB | 64GB+ | 85GB |
| Storage | 100GB | 200GB+ | Available |
| GPU | RTX 3090 | A100/H100 | A100-SXM4-40GB |

### 4.3 VRAM Usage Estimation

| Stage | Estimated VRAM |
|-------|----------------|
| FLUX model loaded | ~24GB |
| PuLID + InsightFace | ~4GB |
| Generation active | ~30GB |
| REAL1SM loaded | ~12GB |
| Peak usage | ~35GB |

---

## 5. API Reference

### 5.1 Web Server Endpoints

#### POST /api/generate
Generate a doctor image.

**Request:**
```
Content-Type: multipart/form-data

images: File[] (1-3 reference images)
positive_prompt: string
negative_prompt: string (optional)
```

**Response:**
```json
{
  "status": "success",
  "image_url": "/outputs/generated_image.png",
  "message": "Image generated successfully!",
  "session_id": "session_1234567890_abc123"
}
```

#### GET /api/progress/{session_id}
Server-Sent Events stream for progress updates.

**Response Stream:**
```json
{"type": "progress", "percentage": 45, "value": 45, "max": 100}
{"type": "complete"}
```

#### GET /api/health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "comfyui": "connected",
  "gpu": "available"
}
```

---

## 6. Configuration

### 6.1 config.yaml Structure

```yaml
model_settings:
  flux_model: "flux1-dev.safetensors"
  steps: 100
  cfg_scale: 1.0
  sampler: "euler"
  scheduler: "beta"

pulid:
  model: "pulid_flux_v0.9.1.safetensors"
  weight: 0.85

refinement:
  model: "REAL1SM_V3_FP8.safetensors"
  steps: 35
  denoise: 0.4

upscale:
  scale: 4.0
  method: "lanczos"

reactor:
  swap_model: "inswapper_128.onnx"
  face_restore_model: "codeformer-v0.1.0.pth"
```

---

## 7. Deployment

### 7.1 Startup Commands

```bash
# Start ComfyUI (with CUDA memory fix)
cd ComfyUI
python3 main.py --listen 0.0.0.0 --port 8188 --disable-cuda-malloc

# Start Web Server
cd /path/to/DoctorImageGenerator
python3 web_server.py
```

### 7.2 Process Management

| Process | Command | Port |
|---------|---------|------|
| ComfyUI | `python3 main.py --listen 0.0.0.0 --port 8188 --disable-cuda-malloc` | 8188 |
| Web Server | `python3 web_server.py` | 8000 |

---

## 8. Performance Metrics

### 8.1 Generation Time Breakdown

| Stage | Approximate Time |
|-------|------------------|
| PuLID encoding | ~10s |
| Main generation (100 steps) | ~60s |
| Hand refinement (50 steps) | ~25s |
| Face swap + restore | ~5s |
| Face refinement (150 steps) | ~45s |
| REAL1SM refinement (35 steps) | ~20s |
| Upscaling (4x) | ~5s |
| **Total** | **~2.5-3 minutes** |

### 8.2 Output Quality

| Metric | Value |
|--------|-------|
| Base Resolution | 1024x1024 |
| Final Resolution | 4096x4096 |
| Identity Preservation | High (PuLID 0.85) |
| Realism Enhancement | REAL1SM refinement |
| Face Quality | CodeFormer restored |

---

## 9. Known Issues & Mitigations

### 9.1 CUDA Memory Errors

**Issue:** "CUDA error: invalid argument" during generation

**Mitigations:**
1. Use `--disable-cuda-malloc` flag
2. Set InsightFace provider to CPU
3. Add FreeMemory nodes between stages
4. Restart ComfyUI to clear VRAM fragmentation

### 9.2 Identity Loss

**Issue:** Generated face doesn't match reference

**Mitigations:**
1. Use 3 reference images from different angles
2. Increase PuLID weight (0.85-0.95)
3. Ensure reference images are high quality
4. Face should be clearly visible in references

---

## 10. File Structure

```
DoctorImageGenerator/
├── ComfyUI/                    # ComfyUI installation
│   ├── models/                 # AI models
│   │   ├── checkpoints/        # Main models (FLUX, REAL1SM)
│   │   ├── pulid/              # PuLID models
│   │   ├── vae/                # VAE models
│   │   ├── clip/               # Text encoders
│   │   ├── insightface/        # Face swap models
│   │   └── ultralytics/        # Detection models
│   ├── custom_nodes/           # ComfyUI extensions
│   ├── input/                  # ComfyUI input images
│   └── output/                 # ComfyUI output images
├── templates/                  # HTML templates
│   └── index.html              # Web UI
├── workflows/                  # ComfyUI workflows
│   └── Doctor.json             # Main workflow
├── uploads/                    # Temporary uploads
├── outputs/                    # Generated images
├── config.yaml                 # Configuration
├── doctor_image_gen.py         # Python API
├── web_server.py               # Flask server
├── Doctor.json                 # Workflow (root copy)
└── TDD_Report.md               # This document
```

---

## 11. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Dec 15, 2025 | Initial release with FLUX + PuLID + REAL1SM pipeline |

---

## 12. Appendix

### A. ComfyUI Custom Nodes Required

- ComfyUI-PuLID-Flux
- ComfyUI-ReActor
- ComfyUI-Impact-Pack (FaceDetailer)
- ComfyUI-Manager

### B. Environment

- Python: 3.10.12
- PyTorch: 2.5.1+cu121
- CUDA: 12.1
- OS: Linux (Ubuntu)

---

*Document generated automatically. Last updated: December 15, 2025*

