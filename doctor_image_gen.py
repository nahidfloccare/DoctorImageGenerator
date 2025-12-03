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
        except Exception as e:
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
    
    def _load_workflow(self, workflow_name: str = "doctor_flux_workflow.json") -> dict:
        """Load ComfyUI workflow JSON"""
        workflow_path = self.workflows_dir / workflow_name
        
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
        Update workflow with specific parameters
        
        This modifies the workflow JSON to use the provided inputs
        """
        # Note: The exact node IDs will depend on your workflow structure
        # This is a template that you'll need to adjust based on your actual workflow
        
        # Update checkpoint loader (node typically named "Load Checkpoint")
        for node_id, node in workflow.items():
            if node.get("class_type") == "CheckpointLoaderSimple":
                node["inputs"]["ckpt_name"] = self.config["model_settings"]["flux_model"]
            
            # Update positive prompt
            elif node.get("class_type") == "CLIPTextEncode" and "positive" in str(node):
                node["inputs"]["text"] = positive_prompt
            
            # Update negative prompt
            elif node.get("class_type") == "CLIPTextEncode" and "negative" in str(node):
                node["inputs"]["text"] = negative_prompt
            
            # Update image loader for doctor's face
            elif node.get("class_type") == "LoadImage":
                # Copy doctor image to ComfyUI input directory
                import shutil
                filename = os.path.basename(doctor_image_path)
                comfyui_input = Path("ComfyUI/input") / filename
                comfyui_input.parent.mkdir(exist_ok=True, parents=True)
                shutil.copy2(doctor_image_path, comfyui_input)
                node["inputs"]["image"] = filename
            
            # Update sampler settings
            elif node.get("class_type") == "KSampler":
                node["inputs"]["steps"] = self.config["model_settings"]["steps"]
                node["inputs"]["cfg"] = self.config["model_settings"]["cfg_scale"]
                node["inputs"]["sampler_name"] = self.config["model_settings"]["sampler"]
                node["inputs"]["scheduler"] = self.config["model_settings"]["scheduler"]
                node["inputs"]["denoise"] = self.config["model_settings"]["denoise"]
            
            # Update image dimensions
            elif node.get("class_type") == "EmptyLatentImage":
                node["inputs"]["width"] = self.config["image_settings"]["width"]
                node["inputs"]["height"] = self.config["image_settings"]["height"]
                node["inputs"]["batch_size"] = self.config["image_settings"]["batch_size"]
            
            # Update ReActor (face swap) settings
            elif node.get("class_type") == "ReActorFaceSwap":
                node["inputs"]["swap_model"] = "inswapper_128.onnx"
                node["inputs"]["facedetection"] = "retinaface_resnet50"
                node["inputs"]["face_restore_model"] = "GFPGANv1.4.pth"
                node["inputs"]["face_restore_visibility"] = self.config["face_swap"]["upscale_visibility"]
        
        return workflow
    
    def generate(self,
                 doctor_image_path: str,
                 scenario: str = "consultant",
                 photography_style: str = "cinematic",
                 custom_prompt: str = "",
                 output_filename: Optional[str] = None) -> Dict[str, Union[str, dict]]:
        """
        Generate a doctor image
        
        Args:
            doctor_image_path: Path to the doctor's headshot photo
            scenario: Scenario to generate (see prompts/templates.py for options)
            photography_style: Photography style (cinematic, editorial, portrait, etc.)
            custom_prompt: Additional custom prompt text
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
        
        print(f"\n📝 Prompt: {positive_prompt[:100]}...")
        
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

