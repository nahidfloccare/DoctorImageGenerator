"""
Doctor Image Generator - Python API
Main interface for generating photorealistic doctor images using FLUX + PuLID
Updated to use Doctor.json workflow with REAL1SM refinement
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
    Uses Doctor.json workflow with FLUX + PuLID + FaceDetailer + REAL1SM refinement
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
    
    def _wait_for_completion(self, prompt_id: str, timeout: int = 900) -> dict:
        """Wait for prompt to complete and return results"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            history = self._get_history(prompt_id)
            
            if prompt_id in history:
                return history[prompt_id]
            
            time.sleep(1)
        
        raise TimeoutError(f"Image generation timed out after {timeout} seconds")
    
    def _load_workflow(self, workflow_name: str = None) -> dict:
        """Load ComfyUI workflow JSON - Uses Doctor.json by default"""
        if workflow_name is None:
            workflow_name = self.config.get("workflow", {}).get("default", "Doctor.json")
        
        workflow_path = self.workflows_dir / workflow_name
        
        if not workflow_path.exists():
            # Try ComfyUI directory as fallback
            workflow_path = Path("ComfyUI") / workflow_name
            if not workflow_path.exists():
                raise FileNotFoundError(
                    f"Workflow not found: {workflow_name}\n"
                    "Please ensure Doctor.json is in the workflows directory."
                )
        
        print(f"🔍 Loading workflow: {workflow_path}")
        
        with open(workflow_path, 'r') as f:
            return json.load(f)
    
    def _update_workflow(self, 
                        workflow: dict, 
                        doctor_image_path: str,
                        positive_prompt: str,
                        negative_prompt: str,
                        additional_images: List[str] = None) -> dict:
        """
        Update Doctor.json workflow with specific parameters
        
        Current Workflow Structure (December 2025 - No LoRA):
        - Node 76: CheckpointLoaderSimple (flux1-dev.safetensors)
        - Node 23: PulidFluxModelLoader (pulid_flux_v0.9.1.safetensors)
        - Node 22: ApplyPulidFlux (weight=0.85, connects directly to checkpoint)
        - Node 6: CLIPTextEncode (positive prompt)
        - Node 7: CLIPTextEncode (negative prompt)
        - Node 11, 27, 28: LoadImage (reference images)
        - Node 3: KSampler (main generation, 100 steps, denoise=1.0, scheduler=beta)
        - Node 35: UltralyticsDetectorProvider (hand_yolov9c.pt)
        - Node 36: FaceDetailer (hands, denoise=0.25, 50 steps)
        - Node 105: FreeMemoryImage (after hands)
        - Node 82: ReActorFaceSwap (face swap with CodeFormer)
        - Node 87: UltralyticsDetectorProvider (face_yolov8m.pt)
        - Node 86: FaceDetailer (face refinement, denoise=0.66, 150 steps, beta scheduler)
        - Node 91: CheckpointLoaderSimple (REAL1SM_V3_FP8.safetensors)
        - Node 99: VAEEncode (encode for refinement)
        - Node 101: KSampler (REAL1SM refinement, 30 steps, denoise=0.3, karras)
        - Node 100: VAEDecode (decode refinement)
        - Node 90: ImageScaleBy (upscale 4x bicubic)
        - Node 104: FreeMemoryImage (before save)
        - Node 9: SaveImage
        """
        import random
        import shutil
        
        model_settings = self.config.get("model_settings", {})
        detail_config = self.config.get("detail_enhancement", {})
        pulid_config = self.config.get("pulid", {})
        lora_config = self.config.get("lora", {})
        reactor_config = self.config.get("reactor", {})
        refinement_config = self.config.get("refinement", {})
        upscale_config = self.config.get("upscale", {})
        
        for node_id, node in workflow.items():
            node_type = node.get("class_type", "")
            
            # Main Flux Checkpoint Loader (Node 76)
            if node_type == "CheckpointLoaderSimple" and node_id == "76":
                node["inputs"]["ckpt_name"] = model_settings.get("flux_model", "flux1-dev.safetensors")
                print(f"   ✓ Flux Checkpoint: {node['inputs']['ckpt_name']}")
            
            # REAL1SM Refinement Checkpoint Loader (Node 91)
            elif node_type == "CheckpointLoaderSimple" and node_id == "91":
                node["inputs"]["ckpt_name"] = refinement_config.get("model", "REAL1SM_V3_FP8.safetensors")
                print(f"   ✓ Refinement Checkpoint: {node['inputs']['ckpt_name']}")
            
            # LoRA Loader (Node 80) - Optional, may not exist in workflow
            elif node_type == "LoraLoaderModelOnly":
                if lora_config.get("enabled", False):
                    node["inputs"]["lora_name"] = lora_config.get("name", "UltraRealistic_v2.safetensors")
                    node["inputs"]["strength_model"] = lora_config.get("strength", 0.5)
                    print(f"   ✓ LoRA: {node['inputs']['lora_name']} @ {node['inputs']['strength_model']}")
                else:
                    print(f"   ℹ LoRA node found but disabled in config")
            
            # CLIP Loaders
            elif node_type == "DualCLIPLoader":
                node["inputs"]["clip_name1"] = "t5xxl_fp16.safetensors"
                node["inputs"]["clip_name2"] = "clip_l.safetensors"
                node["inputs"]["type"] = "flux"
            
            # VAE Loader
            elif node_type == "VAELoader":
                node["inputs"]["vae_name"] = "ae.safetensors"
            
            # PuLID Model Loader
            elif node_type == "PulidFluxModelLoader":
                node["inputs"]["pulid_file"] = pulid_config.get("model", "pulid_flux_v0.9.1.safetensors")
                print(f"   ✓ PuLID: {node['inputs']['pulid_file']}")
            
            # Apply PuLID (Node 22)
            elif node_type == "ApplyPulidFlux":
                node["inputs"]["weight"] = pulid_config.get("weight", 0.85)
                node["inputs"]["start_at"] = pulid_config.get("start_at", 0.0)
                node["inputs"]["end_at"] = pulid_config.get("end_at", 1.0)
                print(f"   ✓ PuLID weight: {node['inputs']['weight']}")
            
            # Positive Prompt (Node 6)
            elif node_type == "CLIPTextEncode" and node_id == "6":
                node["inputs"]["text"] = positive_prompt
                print(f"   ✓ Positive prompt: {positive_prompt[:60]}...")
            
            # Negative Prompt (Node 7)
            elif node_type == "CLIPTextEncode" and node_id == "7":
                node["inputs"]["text"] = negative_prompt
                print(f"   ✓ Negative prompt: {negative_prompt[:60] if negative_prompt else '(minimal)'}")
            
            # Load reference images (Nodes 11, 27, 28)
            elif node_type == "LoadImage":
                # Copy primary image to ComfyUI input
                filename = os.path.basename(doctor_image_path)
                comfyui_input = Path("ComfyUI/input") / filename
                comfyui_input.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy2(doctor_image_path, comfyui_input)
                node["inputs"]["image"] = filename
                print(f"   ✓ Reference image: {filename}")
            
            # Main KSampler (Node 3) - denoise=1.0, 100 steps, beta scheduler
            elif node_type == "KSampler" and node_id == "3":
                node["inputs"]["steps"] = model_settings.get("steps", 100)
                node["inputs"]["cfg"] = model_settings.get("cfg_scale", 1.0)
                node["inputs"]["sampler_name"] = model_settings.get("sampler", "euler")
                node["inputs"]["scheduler"] = model_settings.get("scheduler", "beta")
                node["inputs"]["denoise"] = 1.0
                node["inputs"]["seed"] = random.randint(100000, 999999999999)
                print(f"   ✓ Main KSampler: {node['inputs']['steps']} steps, {node['inputs']['scheduler']} scheduler")
            
            # REAL1SM Refinement KSampler (Node 101)
            elif node_type == "KSampler" and node_id == "101":
                node["inputs"]["steps"] = refinement_config.get("steps", 30)
                node["inputs"]["cfg"] = refinement_config.get("cfg", 1.0)
                node["inputs"]["sampler_name"] = refinement_config.get("sampler", "euler")
                node["inputs"]["scheduler"] = refinement_config.get("scheduler", "karras")
                node["inputs"]["denoise"] = refinement_config.get("denoise", 0.3)
                node["inputs"]["seed"] = random.randint(100000, 999999999999)
                print(f"   ✓ Refinement KSampler: {node['inputs']['steps']} steps, denoise={node['inputs']['denoise']}")
            
            # Hand FaceDetailer (Node 36) - denoise=0.25, 50 steps
            elif node_type == "FaceDetailer" and node_id == "36":
                node["inputs"]["seed"] = random.randint(100000, 999999999999)
                node["inputs"]["steps"] = detail_config.get("hand_refinement_steps", 50)
                node["inputs"]["cfg"] = 1.0
                node["inputs"]["sampler_name"] = "euler"
                node["inputs"]["scheduler"] = "simple"
                node["inputs"]["denoise"] = detail_config.get("hand_refinement_denoise", 0.25)
                node["inputs"]["guide_size"] = detail_config.get("hand_guide_size", 1024)
                node["inputs"]["bbox_threshold"] = 0.5
                node["inputs"]["bbox_crop_factor"] = 3.0
                print(f"   ✓ Hand FaceDetailer: denoise={node['inputs']['denoise']}, {node['inputs']['steps']} steps")
            
            # Face FaceDetailer (Node 86) - denoise=0.66, 150 steps, beta scheduler
            elif node_type == "FaceDetailer" and node_id == "86":
                node["inputs"]["seed"] = random.randint(100000, 999999999999)
                node["inputs"]["steps"] = detail_config.get("face_refinement_steps", 150)
                node["inputs"]["cfg"] = 1.0
                node["inputs"]["sampler_name"] = "euler"
                node["inputs"]["scheduler"] = detail_config.get("face_scheduler", "beta")
                node["inputs"]["denoise"] = detail_config.get("face_refinement_denoise", 0.66)
                node["inputs"]["guide_size"] = detail_config.get("face_guide_size", 1024)
                node["inputs"]["bbox_threshold"] = 0.5
                node["inputs"]["bbox_crop_factor"] = 3.0
                print(f"   ✓ Face FaceDetailer: denoise={node['inputs']['denoise']}, {node['inputs']['steps']} steps, {node['inputs']['scheduler']} scheduler")
            
            # ReActorFaceSwap (Node 82)
            elif node_type == "ReActorFaceSwap":
                node["inputs"]["enabled"] = True
                node["inputs"]["swap_model"] = reactor_config.get("swap_model", "inswapper_128.onnx")
                node["inputs"]["facedetection"] = reactor_config.get("facedetection", "retinaface_resnet50")
                node["inputs"]["face_restore_model"] = reactor_config.get("face_restore_model", "codeformer-v0.1.0.pth")
                node["inputs"]["codeformer_weight"] = reactor_config.get("codeformer_weight", 0.5)
                print(f"   ✓ ReActor: {node['inputs']['swap_model']}, restore={node['inputs']['face_restore_model']}")
            
            # Hand Detector Provider (Node 35)
            elif node_type == "UltralyticsDetectorProvider" and node_id == "35":
                node["inputs"]["model_name"] = detail_config.get("hand_detector", "bbox/hand_yolov9c.pt")
                print(f"   ✓ Hand detector: {node['inputs']['model_name']}")
            
            # Face Detector Provider (Node 87)
            elif node_type == "UltralyticsDetectorProvider" and node_id == "87":
                node["inputs"]["model_name"] = detail_config.get("face_detector", "bbox/face_yolov8m.pt")
                print(f"   ✓ Face detector: {node['inputs']['model_name']}")
            
            # Empty Latent Image
            elif node_type == "EmptyLatentImage":
                node["inputs"]["width"] = self.config.get("image_settings", {}).get("width", 1024)
                node["inputs"]["height"] = self.config.get("image_settings", {}).get("height", 1024)
                node["inputs"]["batch_size"] = 1
            
            # Upscale settings (Node 90) - 4x bicubic
            elif node_type == "ImageScaleBy":
                node["inputs"]["upscale_method"] = upscale_config.get("method", "bicubic")
                node["inputs"]["scale_by"] = upscale_config.get("scale", 4.0)
                print(f"   ✓ Upscale: {node['inputs']['scale_by']}x {node['inputs']['upscale_method']}")
            
            # FreeMemoryImage nodes (104, 105) - no config needed, just pass-through
            elif node_type == "FreeMemoryImage":
                pass  # These nodes don't need configuration
        
        return workflow
    
    def generate(self,
                 doctor_image_path: str,
                 scenario: str = "consultant",
                 photography_style: str = "professional",
                 custom_prompt: str = "",
                 custom_negative_prompt: Optional[str] = None,
                 output_filename: Optional[str] = None) -> Dict[str, Union[str, dict]]:
        """
        Generate a doctor image using Doctor.json workflow
        
        Args:
            doctor_image_path: Path to the doctor's headshot photo
            scenario: Scenario to generate (see prompts/templates.py for options)
            photography_style: Photography style (professional, cinematic, etc.)
            custom_prompt: Custom prompt (overrides scenario if provided)
            custom_negative_prompt: Custom negative prompt
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
        
        print(f"\n{'='*60}")
        print(f"🏥 Doctor Image Generator")
        print(f"{'='*60}")
        print(f"📸 Reference image: {doctor_image_path}")
        
        # Build prompts
        if custom_prompt:
            # Use custom prompt directly - it should already contain proper identity preservation language
            # If the prompt doesn't mention reference photo, add context
            if "input" not in custom_prompt.lower() and "reference" not in custom_prompt.lower():
                positive_prompt = f"Using the input doctor photo as reference, {custom_prompt}. Maintain the person's exact facial features, skin tone, and natural appearance."
            else:
                positive_prompt = custom_prompt
            print(f"📝 Using custom prompt")
        else:
            # Build from scenario template
            if scenario not in SCENARIOS:
                raise ValueError(f"Unknown scenario: {scenario}. Available: {list(SCENARIOS.keys())}")
            
            positive_prompt, _ = build_prompt(
                scenario=scenario,
                photography_style=photography_style,
                custom_additions=""
            )
            # Add identity preservation prefix
            positive_prompt = f"Using the input doctor photo as reference, create a portrait of the same doctor: {positive_prompt}. Maintain the person's exact facial features, skin tone, and natural appearance."
            print(f"🎬 Scenario: {SCENARIOS[scenario]['name']}")
        
        # Use minimal negative prompt like Doctor.json
        negative_prompt = custom_negative_prompt if custom_negative_prompt else "smooth skin, text, watermark, bokeh"
        
        print(f"\n⚙️ Loading Doctor.json workflow...")
        workflow = self._load_workflow("Doctor.json")
        
        print(f"\n🔧 Configuring workflow:")
        workflow = self._update_workflow(
            workflow=workflow,
            doctor_image_path=doctor_image_path,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt
        )
        
        # Queue prompt
        print(f"\n🚀 Queuing generation...")
        prompt_id = self._queue_prompt(workflow)
        print(f"📋 Prompt ID: {prompt_id}")
        
        # Wait for completion
        print(f"⏳ Generating image (estimated 5-10 minutes with refinement)...")
        history = self._wait_for_completion(prompt_id, timeout=900)
        
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
        print(f"{'='*60}\n")
        
        # Return metadata
        return {
            "status": "success",
            "output_path": str(output_path),
            "prompt_id": prompt_id,
            "scenario": scenario,
            "metadata": {
                "doctor_image": doctor_image_path,
                "scenario_name": SCENARIOS.get(scenario, {}).get("name", "Custom"),
                "photography_style": photography_style,
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt,
                "custom_prompt": custom_prompt,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "workflow": "Doctor.json",
                "refinement_model": self.config.get("refinement", {}).get("model", "REAL1SM_V3_FP8.safetensors")
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
