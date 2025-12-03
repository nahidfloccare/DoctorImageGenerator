"""
Prompt templates for Doctor Instagram Image Generation
Each template is carefully crafted for photorealistic Instagram aesthetic
"""

# Base photography keywords for all scenarios
PHOTOGRAPHY_BASE = {
    "lighting": [
        "soft studio lighting",
        "golden hour lighting",
        "natural window light",
        "cinematic lighting",
        "professional photography lighting"
    ],
    "camera": [
        "shot on Sony A7R IV",
        "85mm lens f/1.8",
        "shallow depth of field",
        "bokeh background",
        "professional photography"
    ],
    "quality": [
        "photorealistic",
        "8k uhd",
        "high detail",
        "sharp focus",
        "professional quality"
    ],
    "style": [
        "medical lifestyle photography",
        "candid moment",
        "authentic",
        "professional yet approachable",
        "warm and trustworthy"
    ]
}

# Scenario Templates
SCENARIOS = {
    "consultant": {
        "name": "Medical Consultation",
        "prompt": """A professional doctor consulting with a patient in a modern medical office, 
        holding a digital tablet showing medical data, warm genuine smile, 
        wearing professional medical attire with stethoscope, 
        contemporary clinic background with medical equipment visible, 
        soft natural daylight from large windows, 
        patient partially visible in foreground (blurred), 
        professional medical interaction, caring and confident demeanor,
        {photography_keywords}""",
        "focus": "consultation, interaction, warmth"
    },
    
    "speaker": {
        "name": "Medical Conference Speaker",
        "prompt": """A confident doctor standing at a prestigious medical conference podium, 
        holding a microphone, gesturing professionally while presenting, 
        wearing formal business attire or elegant medical coat, 
        large presentation screen visible in background (blurred), 
        audience silhouettes visible (out of focus), 
        stage lighting with professional conference setup, 
        authoritative yet approachable presence, engaged in speaking,
        {photography_keywords}""",
        "focus": "authority, expertise, public speaking"
    },
    
    "candid_walk": {
        "name": "Candid Hospital Walk",
        "prompt": """A doctor walking confidently down a bright modern hospital corridor, 
        candid shot capturing natural movement, genuine smile or focused expression, 
        wearing crisp white medical coat with stethoscope around neck, 
        holding medical chart or tablet, 
        bright clean hospital environment with natural light, 
        slight motion blur suggesting movement and dynamism, 
        other medical staff visible in background (blurred), 
        professional yet energetic atmosphere,
        {photography_keywords}""",
        "focus": "candid, movement, hospital environment"
    },
    
    "desk_shot": {
        "name": "Executive Desk Work",
        "prompt": """A focused doctor sitting at an elegant mahogany executive desk, 
        reviewing important medical papers and documents, 
        wearing professional attire with reading glasses resting on desk, 
        warm desk lamp providing ambient lighting, 
        medical diplomas and certificates visible on wall (slightly out of focus), 
        bookshelf with medical textbooks in background, 
        intense focus and concentration, thoughtful expression, 
        professional office setting with plants and decor,
        {photography_keywords}""",
        "focus": "professionalism, concentration, executive presence"
    },
    
    "examination": {
        "name": "Patient Examination",
        "prompt": """A caring doctor conducting a thorough patient examination in a medical room, 
        using medical examination tools professionally, 
        wearing medical coat with stethoscope, 
        gentle and focused expression showing expertise and care, 
        patient partially visible (blurred for privacy), 
        medical equipment and examination room visible in background, 
        soft clinical lighting creating professional atmosphere, 
        demonstrating medical competence and bedside manner,
        {photography_keywords}""",
        "focus": "medical expertise, care, examination"
    },
    
    "team_meeting": {
        "name": "Medical Team Discussion",
        "prompt": """A doctor leading or participating in a medical team meeting, 
        seated around a conference table with other healthcare professionals (slightly blurred), 
        engaging in serious medical discussion, 
        medical charts or x-rays visible on table or light box, 
        modern medical facility conference room, 
        collaborative and professional atmosphere, 
        confident body language and engaged expression, 
        natural office lighting,
        {photography_keywords}""",
        "focus": "teamwork, leadership, collaboration"
    },
    
    "lab_coat_portrait": {
        "name": "Professional Lab Coat Portrait",
        "prompt": """A professional portrait of a doctor in pristine white lab coat, 
        arms crossed confidently or standing with professional posture, 
        stethoscope draped around neck, 
        clean modern hospital or clinic background (softly blurred), 
        direct confident gaze at camera showing competence and trustworthiness, 
        perfect lighting highlighting facial features, 
        medical environment subtly visible, 
        executive professional portrait style,
        {photography_keywords}""",
        "focus": "portrait, professionalism, confidence"
    },
    
    "outdoor_professional": {
        "name": "Outdoor Professional Portrait",
        "prompt": """A doctor in professional attire outdoors in front of a modern medical building, 
        golden hour lighting creating warm glow, 
        wearing business casual or professional medical attire, 
        hospital or medical center visible in background (blurred), 
        natural confident smile, 
        greenery and modern architecture creating sophisticated backdrop, 
        professional outdoor corporate portrait style, 
        approachable yet authoritative presence,
        {photography_keywords}""",
        "focus": "outdoor, approachable, modern"
    },
    
    "research": {
        "name": "Medical Research",
        "prompt": """A doctor engaged in medical research in a high-tech laboratory, 
        examining samples or using advanced medical equipment, 
        wearing lab coat and safety equipment, 
        focused and scholarly expression, 
        sophisticated laboratory equipment visible (microscopes, computers, medical devices), 
        clean modern research facility with professional lighting, 
        demonstrating scientific expertise and innovation,
        {photography_keywords}""",
        "focus": "research, science, innovation"
    },
    
    "telehealth": {
        "name": "Telehealth Consultation",
        "prompt": """A doctor conducting a virtual telehealth consultation, 
        sitting at desk with computer or laptop showing video call interface, 
        professional home office or clinic setting, 
        warm and engaging expression while speaking to camera/screen, 
        wearing professional attire, 
        modern technology setup visible, 
        soft professional lighting, 
        contemporary and tech-savvy medical professional,
        {photography_keywords}""",
        "focus": "modern, technology, virtual care"
    },
    
    "surgery_prep": {
        "name": "Surgical Preparation",
        "prompt": """A surgeon in full surgical scrubs and cap preparing for surgery, 
        professional and focused demeanor, 
        surgical room or prep area visible in background, 
        hands being washed or reviewing surgical plans, 
        sterile medical environment with professional lighting, 
        demonstrating surgical expertise and seriousness, 
        confident yet concentrated expression,
        {photography_keywords}""",
        "focus": "surgery, precision, expertise"
    },
    
    "compassionate_care": {
        "name": "Compassionate Patient Care",
        "prompt": """A doctor showing compassionate care, sitting beside patient (blurred) in hospital room, 
        gentle and empathetic expression, 
        holding patient's hand or showing caring gesture, 
        warm hospital room lighting, 
        medical equipment subtly visible in background, 
        demonstrating human side of medicine and bedside manner, 
        soft emotional moment captured professionally,
        {photography_keywords}""",
        "focus": "compassion, empathy, human connection"
    }
}

# Negative prompts to avoid common AI artifacts
NEGATIVE_PROMPT = """low quality, blurry, distorted, deformed, disfigured, 
ugly, bad anatomy, bad proportions, duplicate, cloned face, 
watermark, signature, text, logo, jpeg artifacts, 
worst quality, low resolution, grainy, pixelated, 
cartoon, anime, illustration, painting, drawing, 
multiple heads, extra fingers, mutated hands, 
poorly drawn face, poorly drawn hands, 
bad composition, amateur, unprofessional"""

def build_prompt(scenario: str, photography_style: str = "cinematic", custom_additions: str = "") -> tuple:
    """
    Build a complete prompt for the selected scenario
    
    Args:
        scenario: The scenario key from SCENARIOS
        photography_style: Style of photography (cinematic, editorial, portrait, etc.)
        custom_additions: Additional custom prompt text to add
    
    Returns:
        tuple: (positive_prompt, negative_prompt)
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}. Available: {list(SCENARIOS.keys())}")
    
    # Get scenario template
    template = SCENARIOS[scenario]["prompt"]
    
    # Build photography keywords
    keywords = []
    keywords.extend(PHOTOGRAPHY_BASE["quality"][:3])  # Top 3 quality keywords
    keywords.extend(PHOTOGRAPHY_BASE["camera"][:2])   # Top 2 camera keywords
    keywords.extend(PHOTOGRAPHY_BASE["lighting"][:2]) # Top 2 lighting keywords
    keywords.append(PHOTOGRAPHY_BASE["style"][0])     # Primary style
    
    photography_keywords = ", ".join(keywords)
    
    # Format the prompt
    positive_prompt = template.format(photography_keywords=photography_keywords)
    
    # Add style prefix
    positive_prompt = f"{photography_style} photography style, {positive_prompt}"
    
    # Add custom additions if provided
    if custom_additions:
        positive_prompt = f"{positive_prompt}, {custom_additions}"
    
    # Clean up extra whitespace
    positive_prompt = " ".join(positive_prompt.split())
    
    return positive_prompt, NEGATIVE_PROMPT

def get_scenario_list() -> list:
    """Get list of all available scenarios"""
    return [
        {
            "key": key,
            "name": data["name"],
            "focus": data["focus"]
        }
        for key, data in SCENARIOS.items()
    ]

def get_random_scenario() -> str:
    """Get a random scenario key"""
    import random
    return random.choice(list(SCENARIOS.keys()))

# Example usage
if __name__ == "__main__":
    print("Available Scenarios:")
    print("=" * 60)
    for scenario in get_scenario_list():
        print(f"\n{scenario['key']:20} - {scenario['name']}")
        print(f"{'':20}   Focus: {scenario['focus']}")
    
    print("\n" + "=" * 60)
    print("\nExample Prompt (consultant scenario):")
    print("=" * 60)
    positive, negative = build_prompt("consultant")
    print("\nPositive Prompt:")
    print(positive)
    print("\nNegative Prompt:")
    print(negative)

