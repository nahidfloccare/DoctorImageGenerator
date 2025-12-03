#!/usr/bin/env python3
"""
Batch Generation Example
Demonstrates how to generate multiple images programmatically
"""

from doctor_image_gen import DoctorImageGenerator
from pathlib import Path
import time


def generate_full_portfolio(doctor_image_path: str):
    """
    Generate a complete portfolio of images for a doctor
    
    This creates a diverse set of professional images suitable for:
    - Website hero images
    - Social media profiles
    - LinkedIn banner
    - Blog author photos
    - Marketing materials
    """
    
    # Initialize generator
    generator = DoctorImageGenerator()
    
    # Define portfolio scenarios
    portfolio_scenarios = [
        "lab_coat_portrait",      # Professional headshot
        "consultant",             # Patient interaction
        "desk_shot",             # Executive/thoughtful
        "candid_walk",           # Dynamic/energetic
        "outdoor_professional",   # Approachable/modern
    ]
    
    print("="*70)
    print("GENERATING DOCTOR PORTFOLIO")
    print("="*70)
    print(f"Doctor: {doctor_image_path}")
    print(f"Scenarios: {len(portfolio_scenarios)}")
    print("="*70)
    print()
    
    results = []
    start_time = time.time()
    
    for i, scenario in enumerate(portfolio_scenarios, 1):
        print(f"\n[{i}/{len(portfolio_scenarios)}] Generating: {scenario}")
        print("-" * 70)
        
        try:
            result = generator.generate(
                doctor_image_path=doctor_image_path,
                scenario=scenario,
                photography_style="cinematic",
                output_filename=f"portfolio_{scenario}.png"
            )
            
            results.append({
                "scenario": scenario,
                "status": "success",
                "output": result["output_path"],
                "prompt": result["metadata"]["positive_prompt"][:80] + "..."
            })
            
            print(f"✅ Success: {result['output_path']}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            results.append({
                "scenario": scenario,
                "status": "failed",
                "error": str(e)
            })
    
    # Summary
    elapsed = time.time() - start_time
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    print("\n" + "="*70)
    print("PORTFOLIO GENERATION COMPLETE")
    print("="*70)
    print(f"Total time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")
    print()
    
    if successful:
        print("✅ Generated Images:")
        for result in successful:
            print(f"   - {result['scenario']:25} → {result['output']}")
    
    if failed:
        print("\n❌ Failed Images:")
        for result in failed:
            print(f"   - {result['scenario']:25} → {result['error']}")
    
    print("="*70)
    print()
    
    return results


def generate_social_media_variants(doctor_image_path: str):
    """
    Generate images optimized for different social media platforms
    """
    
    generator = DoctorImageGenerator()
    
    social_scenarios = {
        "linkedin_profile": "lab_coat_portrait",
        "instagram_post": "candid_walk",
        "twitter_header": "outdoor_professional",
        "facebook_cover": "consultant",
    }
    
    print("Generating social media variants...")
    
    for platform, scenario in social_scenarios.items():
        print(f"\n📱 {platform}: {scenario}")
        
        result = generator.generate(
            doctor_image_path=doctor_image_path,
            scenario=scenario,
            output_filename=f"social_{platform}.png"
        )
        
        print(f"   ✅ {result['output_path']}")


def generate_with_custom_prompts(doctor_image_path: str):
    """
    Demonstrate custom prompt usage for specific requirements
    """
    
    generator = DoctorImageGenerator()
    
    custom_generations = [
        {
            "scenario": "consultant",
            "custom": "wearing blue surgical scrubs, friendly smile, pediatric ward",
            "output": "custom_pediatric_consultant.png"
        },
        {
            "scenario": "lab_coat_portrait",
            "custom": "wearing glasses, grey hair, experienced senior doctor",
            "output": "custom_senior_portrait.png"
        },
        {
            "scenario": "speaker",
            "custom": "TEDx stage, red carpet, inspirational talk about healthcare",
            "output": "custom_tedx_speaker.png"
        }
    ]
    
    print("Generating custom prompt variations...")
    
    for config in custom_generations:
        print(f"\n🎨 {config['output']}")
        print(f"   Custom: {config['custom']}")
        
        result = generator.generate(
            doctor_image_path=doctor_image_path,
            scenario=config["scenario"],
            custom_prompt=config["custom"],
            output_filename=config["output"]
        )
        
        print(f"   ✅ Generated")


def main():
    """Main example runner"""
    
    # Check if sample doctor image exists
    sample_image = "inputs/doctor.jpg"
    
    if not Path(sample_image).exists():
        print("❌ Error: Please place a doctor's headshot at inputs/doctor.jpg")
        print("\nExample usage:")
        print("  cp /path/to/doctor_photo.jpg inputs/doctor.jpg")
        print("  python examples/batch_generation_example.py")
        return 1
    
    # Run examples
    print("\n" + "="*70)
    print("BATCH GENERATION EXAMPLES")
    print("="*70)
    print()
    
    # Example 1: Full portfolio
    print("\n### Example 1: Full Portfolio Generation ###\n")
    generate_full_portfolio(sample_image)
    
    # Example 2: Social media variants
    # Uncomment to run:
    # print("\n### Example 2: Social Media Variants ###\n")
    # generate_social_media_variants(sample_image)
    
    # Example 3: Custom prompts
    # Uncomment to run:
    # print("\n### Example 3: Custom Prompts ###\n")
    # generate_with_custom_prompts(sample_image)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

