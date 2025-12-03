#!/usr/bin/env python3
"""
Test Setup Script
Validates that the Doctor Image Generator is properly configured
"""

import os
import sys
from pathlib import Path
import subprocess


def check_item(description: str, condition: bool, optional: bool = False):
    """Check and print status of an item"""
    if condition:
        print(f"✅ {description}")
        return True
    else:
        if optional:
            print(f"⚠️  {description} (optional)")
        else:
            print(f"❌ {description}")
        return not optional


def main():
    print("="*70)
    print("Doctor Image Generator - Setup Validation")
    print("="*70)
    print()
    
    all_ok = True
    
    # Check Python version
    print("🐍 Python Environment:")
    print("-" * 70)
    py_version = sys.version_info
    all_ok &= check_item(
        f"Python {py_version.major}.{py_version.minor}.{py_version.micro}",
        py_version.major == 3 and py_version.minor >= 10
    )
    
    # Check for required Python packages
    print("\n📦 Python Packages:")
    print("-" * 70)
    
    packages = [
        ("torch", True),
        ("transformers", True),
        ("huggingface_hub", True),
        ("Pillow", True),
        ("yaml", True),
        ("insightface", False),  # Optional for now
    ]
    
    for package, required in packages:
        try:
            __import__(package.replace("-", "_").lower())
            check_item(f"{package} installed", True)
        except ImportError:
            all_ok &= check_item(f"{package} installed", False, optional=not required)
    
    # Check NVIDIA GPU
    print("\n🎮 GPU:")
    print("-" * 70)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            gpu_info = result.stdout.strip()
            print(f"✅ NVIDIA GPU detected: {gpu_info}")
        else:
            all_ok &= check_item("NVIDIA GPU detected", False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        all_ok &= check_item("NVIDIA GPU detected", False)
    
    # Check directory structure
    print("\n📁 Directory Structure:")
    print("-" * 70)
    
    required_dirs = [
        "models",
        "inputs",
        "outputs",
        "workflows",
        "prompts"
    ]
    
    for dir_name in required_dirs:
        exists = Path(dir_name).exists()
        all_ok &= check_item(f"{dir_name}/ directory", exists)
    
    # Check for ComfyUI
    print("\n🎨 ComfyUI:")
    print("-" * 70)
    comfyui_exists = Path("ComfyUI").exists()
    all_ok &= check_item("ComfyUI installed", comfyui_exists, optional=True)
    
    if comfyui_exists:
        check_item("ComfyUI/main.py", Path("ComfyUI/main.py").exists(), optional=True)
        check_item("ComfyUI/custom_nodes", Path("ComfyUI/custom_nodes").exists(), optional=True)
    
    # Check for models
    print("\n🤖 Models:")
    print("-" * 70)
    
    models_to_check = [
        ("Flux.1-dev checkpoint", "ComfyUI/models/checkpoints/flux1-dev*.safetensors"),
        ("VAE", "ComfyUI/models/vae/ae.safetensors"),
        ("InsightFace", "ComfyUI/models/insightface/inswapper_128.onnx"),
        ("Upscaler", "ComfyUI/models/upscale_models/4x-UltraSharp.pth"),
    ]
    
    from glob import glob
    for model_name, pattern in models_to_check:
        found = len(glob(pattern)) > 0
        check_item(model_name, found, optional=True)
    
    # Check configuration
    print("\n⚙️  Configuration:")
    print("-" * 70)
    
    config_exists = Path("config.yaml").exists()
    all_ok &= check_item("config.yaml", config_exists)
    
    workflow_exists = Path("workflows/doctor_flux_workflow.json").exists()
    all_ok &= check_item("workflow JSON", workflow_exists)
    
    # Check main scripts
    print("\n📝 Scripts:")
    print("-" * 70)
    
    scripts = [
        "doctor_image_gen.py",
        "generate.py",
        "prompts/templates.py",
        "setup.sh",
        "start_comfyui.sh"
    ]
    
    for script in scripts:
        exists = Path(script).exists()
        all_ok &= check_item(script, exists)
    
    # Final summary
    print("\n" + "="*70)
    if all_ok:
        print("✅ ALL CRITICAL CHECKS PASSED")
        print("\nYour setup looks good! You can now:")
        print("1. Start ComfyUI: ./start_comfyui.sh")
        print("2. Generate images: python generate.py --doctor-image inputs/doctor.jpg --scenario consultant")
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nPlease review the errors above and:")
        print("1. Run setup.sh if you haven't: ./setup.sh")
        print("2. Install missing dependencies: pip install -r requirements.txt")
        print("3. Download required models (see setup.sh)")
    print("="*70)
    print()
    
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())

