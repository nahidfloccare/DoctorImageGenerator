# Input Images Directory

Place your doctor's headshot photos here.

## Requirements for Best Results

### ✅ Ideal Input Photo

- **Resolution**: 1024x1024px or higher
- **Format**: JPG, PNG, or JPEG
- **Face Position**: Looking directly at camera
- **Angle**: Face-forward (not profile or 3/4 view)
- **Lighting**: Even, soft lighting without harsh shadows
- **Focus**: Sharp, in-focus facial features
- **Background**: Clean, uncluttered (will be replaced anyway)
- **Expression**: Neutral or slight smile
- **Accessories**: Minimal (no sunglasses covering eyes)
- **Style**: Passport photo or professional headshot style

### ❌ Avoid

- Low resolution (< 512px)
- Blurry or out of focus
- Extreme angles or profile shots
- Heavy shadows or backlighting
- Multiple people in frame
- Obscured face (masks, sunglasses)
- Heavy filters or editing
- Group photos

## Example File Naming

```
inputs/
├── doctor_john_doe.jpg
├── dr_smith_headshot.png
├── jane_physician.jpg
└── README.md  (this file)
```

## Photo Preparation Tips

### If you have a high-quality photo already:
Just copy it here!

```bash
cp /path/to/your/photo.jpg inputs/doctor.jpg
```

### If your photo needs cropping:
Use any image editor to:
1. Crop to square (1:1 aspect ratio)
2. Center the face
3. Include head and top of shoulders
4. Save at minimum 512x512px (1024x1024px preferred)

### If your photo is low quality:
Consider:
- Using an AI upscaler first (e.g., Real-ESRGAN)
- Taking a new photo with better lighting
- Using a professional headshot

## Quick Test

Once you have a photo here, test the setup:

```bash
python generate.py --doctor-image inputs/your_photo.jpg --scenario consultant
```

The first generation will take longer as models load into memory.

## Privacy Note

All images are processed locally. No photos are uploaded to any external service. You have complete control over your data.

