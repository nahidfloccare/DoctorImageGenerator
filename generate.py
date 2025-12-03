#!/usr/bin/env python3
"""
Doctor Image Generator - Command Line Interface
Simple CLI for generating doctor images
"""

import argparse
import sys
from pathlib import Path
from doctor_image_gen import DoctorImageGenerator
from prompts.templates import get_scenario_list


def main():
    parser = argparse.ArgumentParser(
        description="Generate photorealistic Instagram-style doctor images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python generate.py --doctor-image inputs/doctor.jpg --scenario consultant
  
  # With custom prompt additions
  python generate.py --doctor-image inputs/doctor.jpg --scenario speaker --custom "wearing blue tie"
  
  # Batch generation with multiple scenarios
  python generate.py --doctor-image inputs/doctor.jpg --batch consultant speaker desk_shot
  
  # List all available scenarios
  python generate.py --list-scenarios
        """
    )
    
    parser.add_argument(
        "--doctor-image",
        type=str,
        help="Path to doctor's headshot photo (required unless using --list-scenarios)"
    )
    
    parser.add_argument(
        "--scenario",
        type=str,
        default="consultant",
        help="Scenario to generate (default: consultant). Use --list-scenarios to see all options"
    )
    
    parser.add_argument(
        "--batch",
        nargs="+",
        metavar="SCENARIO",
        help="Generate multiple scenarios in batch mode"
    )
    
    parser.add_argument(
        "--photography-style",
        type=str,
        default="cinematic",
        choices=["cinematic", "editorial", "portrait", "documentary", "lifestyle"],
        help="Photography style (default: cinematic)"
    )
    
    parser.add_argument(
        "--custom",
        type=str,
        default="",
        help="Additional custom prompt text"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Custom output filename (default: auto-generated)"
    )
    
    parser.add_argument(
        "--comfyui-url",
        type=str,
        default="http://127.0.0.1:8188",
        help="ComfyUI server URL (default: http://127.0.0.1:8188)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List all available scenarios and exit"
    )
    
    args = parser.parse_args()
    
    # List scenarios if requested
    if args.list_scenarios:
        print("\n" + "="*70)
        print("AVAILABLE SCENARIOS")
        print("="*70 + "\n")
        
        scenarios = get_scenario_list()
        for i, scenario in enumerate(scenarios, 1):
            print(f"{i:2}. {scenario['key']:20} - {scenario['name']}")
            print(f"    {'':20}   Focus: {scenario['focus']}\n")
        
        print("="*70)
        print(f"Total: {len(scenarios)} scenarios available")
        print("="*70 + "\n")
        return 0
    
    # Validate required arguments
    if not args.doctor_image:
        parser.error("--doctor-image is required (unless using --list-scenarios)")
    
    if not Path(args.doctor_image).exists():
        print(f"❌ Error: Doctor image not found: {args.doctor_image}")
        return 1
    
    try:
        # Initialize generator
        print("\n🚀 Initializing Doctor Image Generator...")
        generator = DoctorImageGenerator(
            comfyui_url=args.comfyui_url,
            config_path=args.config
        )
        
        # Batch mode
        if args.batch:
            print(f"\n📦 Batch Mode: Generating {len(args.batch)} scenarios")
            results = generator.batch_generate(
                doctor_image_path=args.doctor_image,
                scenarios=args.batch,
                photography_style=args.photography_style,
                custom_prompt=args.custom
            )
            
            # Summary
            print("\n" + "="*70)
            print("BATCH GENERATION SUMMARY")
            print("="*70)
            
            successful = [r for r in results if "error" not in r]
            failed = [r for r in results if "error" in r]
            
            print(f"\n✅ Successful: {len(successful)}/{len(results)}")
            for result in successful:
                print(f"   - {result['scenario']:20} → {result['output_path']}")
            
            if failed:
                print(f"\n❌ Failed: {len(failed)}/{len(results)}")
                for result in failed:
                    print(f"   - {result['scenario']:20} → {result['error']}")
            
            print("\n" + "="*70 + "\n")
            
            return 0 if not failed else 1
        
        # Single image generation
        else:
            result = generator.generate(
                doctor_image_path=args.doctor_image,
                scenario=args.scenario,
                photography_style=args.photography_style,
                custom_prompt=args.custom,
                output_filename=args.output
            )
            
            print("\n" + "="*70)
            print("GENERATION COMPLETE")
            print("="*70)
            print(f"Scenario: {result['metadata']['scenario_name']}")
            print(f"Output:   {result['output_path']}")
            print(f"Prompt:   {result['metadata']['positive_prompt'][:80]}...")
            print("="*70 + "\n")
            
            return 0
    
    except KeyboardInterrupt:
        print("\n\n❌ Generation cancelled by user")
        return 130
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

