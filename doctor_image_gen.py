"""
Doctor Image Generator - Python API
Main interface for generating photorealistic doctor images using Flux.1-dev + ReActor
"""

import os
import json
import time
import uuid
import yaml
from pathlib import Path
from typing import Dict, Optional, Union, List
import websocket
import urllib.request
import urllib.parse

from prompts.templates import build_prompt, get_scenario_list, SCENARIOS


class DoctorImageGenerator:
    """
    Main class for generating doctor images using ComfyUI backend
    """
    
    def __init__(self, 
                 comfyui_url: str = "http://127.0.0.1:8188",
                 config_path: str = "config.yaml"):
        """
        Initialize the generator
        
        Args:
            comfyui_url: URL where ComfyUI server is running
            config_path: Path to configuration file
        """
        self.comfyui_url = comfyui_url
        self.config = self._load_config(config_path)
        self.client_id = str(uuid.uuid4())
        
        # Paths
        self.inputs_dir = Path(self.config["paths"]["inputs"])
        self.outputs_dir = Path(self.config["paths"]["outputs"])
        self.workflows_dir = Path(self.config["paths"]["workflows"])
        
        # Create directories if they don't exist
        self.inputs_dir.mkdir(exist_ok=True, parents=True)
        self.outputs_dir.mkdir(exist_ok=True, parents=True)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _queue_prompt(self, prompt: dict) -> str:
        """Queue a prompt to ComfyUI server"""
        p = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(p).encode('utf-8')
        req = urllib.request.Request(f"{self.comfyui_url}/prompt", data=data)
        req.add_header('Content-Type', 'application/json')
        
        try:
            with urllib.request.urlopen(req) as response:
                result = json.loads(response.read())
                return result['prompt_id']
        except urllib.error.HTTPError as e:
            error_body = e.read().decode() if hasattr(e, 'read') else str(e)
            try:
                error_json = json.loads(error_body)
                print(f"\n❌ ComfyUI Error:")
                print(json.dumps(error_json, indent=2))
            except:
                print(f"\n❌ Error: {error_body}")
            raise RuntimeError(f"Failed to queue prompt: {e}")
    
    def _get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Get generated image from ComfyUI"""
        data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        url_values = urllib.parse.urlencode(data)
        
        try:
            with urllib.request.urlopen(f"{self.comfyui_url}/view?{url_values}") as response:
                return response.read()
        except Exception as e:
            raise RuntimeError(f"Failed to get image: {e}")
    
    def _get_history(self, prompt_id: str) -> dict:
        """Get history for a specific prompt"""
        try:
            with urllib.request.urlopen(f"{self.comfyui_url}/history/{prompt_id}") as response:
                return json.loads(response.read())
        except Exception as e:
            raise RuntimeError(f"Failed to get history: {e}")
    
    def _wait_for_completion(self, prompt_id: str, timeout: int = 300) -> dict:
        """Wait for prompt to complete and return results"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            history = self._get_history(prompt_id)
            
            if prompt_id in history:
                return history[prompt_id]
            
            time.sleep(1)
        
        raise TimeoutError(f"Image generation timed out after {timeout} seconds")
    
    def _load_workflow(self, workflow_name: str = None) -> dict:
        """Load ComfyUI workflow JSON - Uses FLUX-specific PuLID nodes with optional hand refinement"""
        # Check if detail enhancement is enabled in config
        if workflow_name is None:
            detail_config = self.config.get("detail_enhancement", {})
            mode = detail_config.get("mode", "none")
            
            if detail_config.get("enabled", False) and mode != "none":
                if mode == "facedetailer":
                    workflow_name = "current_comfyui_workflow.json"
                    print(f"🔍 Using your current ComfyUI workflow - PuLID + FaceDetailer (hands & face)")
                elif mode == "meshgraphormer":
                    workflow_name = "flux_pulid_with_meshgraphormer.json"
                    print(f"🔍 Using MeshGraphormer workflow - 3D hand mesh prediction for perfect hands")
                elif mode == "basic":
                    workflow_name = "flux_pulid_with_detailers.json"
                    print(f"🔍 Using basic detailer workflow - standard hand/face refinement")
                else:
                    workflow_name = "flux_pulid_flux_api.json"
                    print(f"⚡ Using fast workflow - no hand refinement")
            else:
                workflow_name = "flux_pulid_flux_api.json"
                print(f"⚡ Using fast workflow - no hand refinement")
        
        workflow_path = self.workflows_dir / workflow_name
        
        if not workflow_path.exists():
            # Fallback chain: meshgraphormer → basic → standard
            if "meshgraphormer" in workflow_name:
                print(f"⚠️ MeshGraphormer workflow not found or models not installed")
                print(f"   Falling back to basic detailer...")
                workflow_path = self.workflows_dir / "flux_pulid_with_detailers.json"
            
            if not workflow_path.exists() and "detailers" in workflow_name:
                print(f"⚠️ Basic detailer workflow not found")
                print(f"   Falling back to standard workflow (no hand refinement)")
                print(f"   Install Impact Pack for hand/face refinement")
                workflow_path = self.workflows_dir / "flux_pulid_flux_api.json"
            
            if not workflow_path.exists():
                raise FileNotFoundError(
                    f"Workflow not found: {workflow_path}\n"
                    "Please ensure the workflow file is in the workflows directory."
                )
        
        with open(workflow_path, 'r') as f:
            return json.load(f)
    
    def _update_workflow(self, 
                        workflow: dict, 
                        doctor_image_path: str,
                        positive_prompt: str,
                        negative_prompt: str) -> dict:
        """
        Update workflow with specific parameters for FLUX + PuLID
        """
        import random
        import shutil
        
        for node_id, node in workflow.items():
            node_type = node.get("class_type", "")
            
            # FLUX Model Loaders
            if node_type == "UNETLoader":
                node["inputs"]["unet_name"] = "flux1-dev-fp8.safetensors"
            
            elif node_type == "DualCLIPLoader":
                node["inputs"]["clip_name1"] = "t5xxl_fp16.safetensors"
                node["inputs"]["clip_name2"] = "clip_l.safetensors"
                node["inputs"]["type"] = "flux"
            
            elif node_type == "VAELoader":
                node["inputs"]["vae_name"] = "ae.safetensors"
            
            # PuLID FLUX Loaders
            elif node_type == "PulidFluxModelLoader":
                node["inputs"]["pulid_file"] = "pulid_flux_v0.9.0.safetensors"
            
            elif node_type == "PulidFluxEvaClipLoader":
                node["inputs"]["eva_clip_name"] = "EVA02_CLIP_L_336_psz14_s6B.pt"
            
            elif node_type == "PulidFluxInsightFaceLoader":
                node["inputs"]["provider"] = "CUDA"
            
            # Apply PuLID with config
            elif node_type == "ApplyPulidFlux":
                pulid_config = self.config.get("pulid", {})
                node["inputs"]["weight"] = pulid_config.get("weight", 1.0)
                node["inputs"]["start_at"] = pulid_config.get("start_at", 0.0)
                node["inputs"]["end_at"] = pulid_config.get("end_at", 1.0)
            
            # Update prompts (node 6 = positive, node 7 = negative in your workflow)
            elif node_type == "CLIPTextEncode":
                if node_id == "6":
                    # Positive prompt
                    node["inputs"]["text"] = positive_prompt
                    print(f"   ✓ Set Node 6 (Positive): {positive_prompt[:50]}...")
                elif node_id == "7":
                    # Negative prompt - pass the actual negative prompt
                    node["inputs"]["text"] = negative_prompt
                    print(f"   ✓ Set Node 7 (Negative): {negative_prompt}")
            
            # Load doctor's image
            elif node_type == "LoadImage":
                filename = os.path.basename(doctor_image_path)
                comfyui_input = Path("ComfyUI/input") / filename
                comfyui_input.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy2(doctor_image_path, comfyui_input)
                node["inputs"]["image"] = filename
            
            # KSampler with FLUX settings
            elif node_type == "KSampler":
                # Main generation KSampler (denoise=1.0)
                if node["inputs"].get("denoise", 1.0) == 1.0:
                    node["inputs"]["steps"] = self.config.get("model_settings", {}).get("steps", 100)
                    node["inputs"]["denoise"] = 1.0
                else:
                    # Refinement KSampler (denoise < 1.0, after upscaling)
                    node["inputs"]["steps"] = self.config.get("model_settings", {}).get("steps", 100)
                    node["inputs"]["denoise"] = 0.35
                
                node["inputs"]["cfg"] = 1.0  # FLUX requirement
                node["inputs"]["sampler_name"] = "euler"
                node["inputs"]["scheduler"] = "simple"  # FLUX requirement
                node["inputs"]["seed"] = random.randint(100000, 999999)
            
            # Image dimensions
            elif node_type == "EmptyLatentImage":
                node["inputs"]["width"] = 1024
                node["inputs"]["height"] = 1024
                node["inputs"]["batch_size"] = 1
            
            # FLUX Guidance
            elif node_type == "FluxGuidance":
                node["inputs"]["guidance"] = 3.5
            
            # FaceDetailer nodes for hand refinement
            elif node_type == "FaceDetailer":
                detail_config = self.config.get("detail_enhancement", {})
                node["inputs"]["seed"] = random.randint(100000, 999999)
                node["inputs"]["cfg"] = 1.0
                node["inputs"]["sampler_name"] = "euler"
                node["inputs"]["scheduler"] = "simple"
                node["inputs"]["denoise"] = detail_config.get("hand_refinement_denoise", 0.35)
                node["inputs"]["steps"] = detail_config.get("hand_refinement_steps", 50)
                node["inputs"]["guide_size"] = detail_config.get("hand_guide_size", 1024)
            
            # ImageScaleBy for upscaling
            elif node_type == "ImageScaleBy":
                # User's workflow uses bicubic 1.5x upscaling
                node["inputs"]["upscale_method"] = "bicubic"
                node["inputs"]["scale_by"] = 1.5
            
            # ResizeAndPadImage for input preprocessing
            elif node_type == "ResizeAndPadImage":
                node["inputs"]["target_width"] = 1024
                node["inputs"]["target_height"] = 1024
                node["inputs"]["padding_color"] = "white"
                node["inputs"]["interpolation"] = "area"
            
            # VAEEncode for refinement stage
            elif node_type == "VAEEncode":
                # No special configuration needed, just ensure it's connected properly
                pass
            
            # ImageBatch for combining multiple images
            elif node_type == "ImageBatch":
                # No special configuration needed
                pass
            
            # UltralyticsDetectorProvider nodes (for hand/face detection)
            elif node_type == "UltralyticsDetectorProvider":
                # Node configuration is already set in workflow, no changes needed
                pass
            
            # SAMLoader for segmentation
            elif node_type == "SAMLoader":
                node["inputs"]["model_name"] = "sam_vit_h_4b8939.pth"
                node["inputs"]["device_mode"] = "AUTO"
            
            # Detail Enhancement - Hand and Face Refiners (if Impact Pack is installed)
            elif node_type == "DetailerForEach":
                detail_config = self.config.get("detail_enhancement", {})
                # Update denoise levels for detailers
                if "hand_yolov8s" in node.get("inputs", {}).get("bbox_detector", ["", ""])[0]:
                    # This is the hand detailer
                    node["inputs"]["denoise"] = detail_config.get("hand_refinement_denoise", 0.5)
                    node["inputs"]["bbox_threshold"] = detail_config.get("hand_detection_threshold", 0.5)
                    node["inputs"]["seed"] = random.randint(100000, 999999)
                elif "person_yolov8m" in str(node.get("inputs", {})):
                    # This is the face/person detailer  
                    node["inputs"]["denoise"] = detail_config.get("face_refinement_denoise", 0.35)
                    node["inputs"]["seed"] = random.randint(100000, 999999)
        
        return workflow
    
    def generate(self,
                 doctor_image_path: str,
                 scenario: str = "consultant",
                 photography_style: str = "cinematic",
                 custom_prompt: str = "",
                 custom_negative_prompt: Optional[str] = None,
                 output_filename: Optional[str] = None) -> Dict[str, Union[str, dict]]:
        """
        Generate a doctor image
        
        Args:
            doctor_image_path: Path to the doctor's headshot photo
            scenario: Scenario to generate (see prompts/templates.py for options)
            photography_style: Photography style (cinematic, editorial, portrait, etc.)
            custom_prompt: Additional custom prompt text
            custom_negative_prompt: Custom negative prompt (uses default from config if None)
            output_filename: Custom output filename (auto-generated if None)
        
        Returns:
            Dict containing:
                - output_path: Path to generated image
                - prompt_id: ComfyUI prompt ID
                - scenario: Used scenario
                - metadata: Generation metadata
        """
        # Validate inputs
        if not os.path.exists(doctor_image_path):
            raise FileNotFoundError(f"Doctor image not found: {doctor_image_path}")
        
        if scenario not in SCENARIOS:
            raise ValueError(
                f"Unknown scenario: {scenario}\n"
                f"Available scenarios: {list(SCENARIOS.keys())}"
            )
        
        print(f"🎬 Generating image with scenario: {SCENARIOS[scenario]['name']}")
        print(f"📸 Photography style: {photography_style}")
        print(f"👨‍⚕️ Doctor image: {doctor_image_path}")
        
        # Build prompts
        positive_prompt, negative_prompt = build_prompt(
            scenario=scenario,
            photography_style=photography_style,
            custom_additions=custom_prompt
        )
        
        # Use custom negative prompt if provided
        if custom_negative_prompt:
            negative_prompt = custom_negative_prompt
            print(f"\n🚫 Using custom negative prompt: {negative_prompt}")
        else:
            print(f"\n🚫 Using default negative prompt: {negative_prompt}")
        
        print(f"\n📝 Positive prompt: {positive_prompt[:100]}...")
        
        # Load and update workflow
        print("\n⚙️ Loading workflow...")
        workflow = self._load_workflow()
        workflow = self._update_workflow(
            workflow=workflow,
            doctor_image_path=doctor_image_path,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt
        )
        
        # Queue prompt
        print("🚀 Queuing generation...")
        prompt_id = self._queue_prompt(workflow)
        print(f"📋 Prompt ID: {prompt_id}")
        
        # Wait for completion
        print("⏳ Generating image (this may take 30-120 seconds)...")
        history = self._wait_for_completion(prompt_id)
        
        # Get output images
        outputs = history.get("outputs", {})
        images = []
        
        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for image in node_output["images"]:
                    images.append(image)
        
        if not images:
            raise RuntimeError("No images were generated")
        
        # Save the first image
        image_data = self._get_image(
            filename=images[0]["filename"],
            subfolder=images[0].get("subfolder", ""),
            folder_type=images[0].get("type", "output")
        )
        
        # Generate output filename
        if output_filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            output_filename = f"doctor_{scenario}_{timestamp}.png"
        
        output_path = self.outputs_dir / output_filename
        
        # Write image
        with open(output_path, 'wb') as f:
            f.write(image_data)
        
        print(f"\n✅ Image generated successfully!")
        print(f"📁 Saved to: {output_path}")
        
        # Return metadata
        return {
            "status": "success",
            "output_path": str(output_path),
            "prompt_id": prompt_id,
            "scenario": scenario,
            "metadata": {
                "doctor_image": doctor_image_path,
                "scenario_name": SCENARIOS[scenario]["name"],
                "photography_style": photography_style,
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "custom_prompt": custom_prompt,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": self.config
            }
        }
    
    def list_scenarios(self) -> List[dict]:
        """List all available scenarios"""
        return get_scenario_list()
    
    def batch_generate(self,
                      doctor_image_path: str,
                      scenarios: List[str],
                      **kwargs) -> List[Dict]:
        """
        Generate multiple images with different scenarios
        
        Args:
            doctor_image_path: Path to doctor's headshot
            scenarios: List of scenario keys
            **kwargs: Additional arguments passed to generate()
        
        Returns:
            List of generation results
        """
        results = []
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{'='*60}")
            print(f"Generating {i}/{len(scenarios)}: {scenario}")
            print(f"{'='*60}")
            
            try:
                result = self.generate(
                    doctor_image_path=doctor_image_path,
                    scenario=scenario,
                    **kwargs
                )
                results.append(result)
            except Exception as e:
                print(f"❌ Error generating {scenario}: {e}")
                results.append({"error": str(e), "scenario": scenario})
        
        return results


# CLI-style usage example
if __name__ == "__main__":
    # Example usage
    generator = DoctorImageGenerator()
    
    # List available scenarios
    print("Available Scenarios:")
    print("=" * 60)
    for scenario in generator.list_scenarios():
        print(f"{scenario['key']:20} - {scenario['name']}")
    
    # Generate an image (example - won't work without actual setup)
    # result = generator.generate(
    #     doctor_image_path="inputs/doctor_headshot.jpg",
    #     scenario="consultant"
    # )

