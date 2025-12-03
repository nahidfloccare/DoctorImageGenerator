#!/usr/bin/env python3
"""
Simple face integration using img2img approach
Takes a generated doctor image and blends your doctor's face
"""

from PIL import Image
import cv2
import numpy as np

def integrate_doctor_face(generated_image_path, doctor_face_path, output_path):
    """
    Simple face replacement using OpenCV
    """
    # Load images
    generated = cv2.imread(generated_image_path)
    doctor_face = cv2.imread(doctor_face_path)
    
    # Face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in generated image
    gray_gen = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)
    faces_gen = face_cascade.detectMultiScale(gray_gen, 1.1, 4)
    
    # Detect face in doctor image
    gray_doc = cv2.cvtColor(doctor_face, cv2.COLOR_BGR2GRAY)
    faces_doc = face_cascade.detectMultiScale(gray_doc, 1.1, 4)
    
    if len(faces_gen) > 0 and len(faces_doc) > 0:
        # Get first face from each
        (x_gen, y_gen, w_gen, h_gen) = faces_gen[0]
        (x_doc, y_doc, w_doc, h_doc) = faces_doc[0]
        
        # Extract and resize doctor's face
        doctor_face_cropped = doctor_face[y_doc:y_doc+h_doc, x_doc:x_doc+w_doc]
        doctor_face_resized = cv2.resize(doctor_face_cropped, (w_gen, h_gen))
        
        # Simple blend
        generated[y_gen:y_gen+h_gen, x_gen:x_gen+w_gen] = doctor_face_resized
        
        cv2.imwrite(output_path, generated)
        print(f"✅ Integrated face: {output_path}")
        return True
    else:
        print("❌ Face detection failed")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python3 integrate_face.py <generated_image> <doctor_face> <output>")
        sys.exit(1)
    
    integrate_doctor_face(sys.argv[1], sys.argv[2], sys.argv[3])
