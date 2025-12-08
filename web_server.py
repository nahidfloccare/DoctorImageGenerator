#!/usr/bin/env python3
"""
Web UI for Doctor Image Generator
Uses ComfyUI with PuLID + FaceDetailer architecture
"""

from flask import Flask, render_template, request, jsonify, send_file, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import time
import uuid
import json
import threading
from pathlib import Path
from doctor_image_gen import DoctorImageGenerator
import websocket
import queue

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads/temp'
app.config['OUTPUT_FOLDER'] = 'outputs'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Progress tracking
progress_queues = {}  # session_id -> queue
COMFYUI_WS_URL = "ws://127.0.0.1:8188/ws"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def track_progress(session_id, prompt_id, client_id):
    """Track ComfyUI progress via WebSocket"""
    try:
        progress_queue = queue.Queue()
        progress_queues[session_id] = progress_queue
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                msg_type = data.get('type')
                
                if msg_type == 'progress':
                    progress_data = data.get('data', {})
                    value = progress_data.get('value', 0)
                    max_val = progress_data.get('max', 100)
                    percentage = int((value / max_val) * 100) if max_val > 0 else 0
                    
                    progress_queue.put({
                        'type': 'progress',
                        'percentage': percentage,
                        'value': value,
                        'max': max_val
                    })
                
                elif msg_type == 'executing':
                    node = data.get('data', {}).get('node')
                    if node is None:
                        # Execution complete
                        progress_queue.put({'type': 'complete'})
                        ws.close()
                        
            except Exception as e:
                print(f"WebSocket message error: {e}")
        
        def on_error(ws, error):
            print(f"WebSocket error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            if session_id in progress_queues:
                progress_queues[session_id].put({'type': 'complete'})
        
        ws = websocket.WebSocketApp(
            f"{COMFYUI_WS_URL}?clientId={client_id}",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close
        )
        
        ws.run_forever()
    except Exception as e:
        print(f"Progress tracking error: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/progress/<session_id>')
def progress_stream(session_id):
    """Server-Sent Events endpoint for real-time progress updates"""
    def generate_progress():
        if session_id not in progress_queues:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Session not found'})}\n\n"
            return
        
        progress_queue = progress_queues[session_id]
        
        while True:
            try:
                # Get progress update with timeout
                progress_data = progress_queue.get(timeout=1)
                yield f"data: {json.dumps(progress_data)}\n\n"
                
                if progress_data.get('type') == 'complete':
                    break
            except queue.Empty:
                # Send keepalive
                yield f"data: {json.dumps({'type': 'keepalive'})}\n\n"
        
        # Cleanup
        if session_id in progress_queues:
            del progress_queues[session_id]
    
    return Response(generate_progress(), mimetype='text/event-stream')

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        # Validate request
        if 'images' not in request.files:
            return jsonify({'error': 'No images uploaded'}), 400
        
        files = request.files.getlist('images')
        if not files or len(files) == 0:
            return jsonify({'error': 'Please upload at least one image'}), 400
        
        # Save uploaded images
        session_id = str(uuid.uuid4())
        image_paths = []
        
        for idx, file in enumerate(files):
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{session_id}_{idx}_{filename}")
                file.save(filepath)
                image_paths.append(filepath)
        
        if not image_paths:
            return jsonify({'error': 'No valid images uploaded'}), 400
        
        # Get prompts from form data
        positive_prompt = request.form.get('positive_prompt', '').strip()
        negative_prompt = request.form.get('negative_prompt', '').strip()
        
        if not positive_prompt:
            return jsonify({'error': 'Please provide a positive prompt'}), 400
        
        # Log the request
        print(f"\n{'='*80}")
        print(f"🎨 New Generation Request")
        print(f"{'='*80}")
        print(f"📸 Images uploaded: {len(image_paths)}")
        print(f"📝 Positive prompt: {positive_prompt[:100]}...")
        if negative_prompt:
            print(f"🚫 Negative prompt: {negative_prompt[:100]}...")
        print(f"{'='*80}\n")
        
        # Initialize generator
        generator = DoctorImageGenerator()
        client_id = generator.client_id
        
        # Start progress tracking in background thread
        progress_thread = threading.Thread(
            target=track_progress,
            args=(session_id, None, client_id),
            daemon=True
        )
        progress_thread.start()
        
        # Generate image using the first uploaded image
        output_filename = f"generated_{session_id}.png"
        
        # Use 'consultant' as base scenario, custom_prompt will override it
        result = generator.generate(
            doctor_image_path=image_paths[0],
            scenario="consultant",
            photography_style="professional",
            custom_prompt=positive_prompt,
            custom_negative_prompt=negative_prompt if negative_prompt else None,
            output_filename=output_filename
        )
        
        # Clean up uploaded files
        for img_path in image_paths:
            try:
                if os.path.exists(img_path):
                    os.remove(img_path)
            except Exception as e:
                print(f"Warning: Could not delete temp file {img_path}: {e}")
        
        if result['status'] == 'success':
            print(f"\n✅ Generation successful!")
            print(f"📁 Output: {result['output_path']}\n")
            
            return jsonify({
                'status': 'success',
                'image_url': f"/api/output/{os.path.basename(result['output_path'])}",
                'message': 'Image generated successfully!',
                'session_id': session_id
            })
        else:
            print(f"\n❌ Generation failed: {result.get('message', 'Unknown error')}\n")
            return jsonify({
                'status': 'error',
                'error': result.get('message', 'Generation failed')
            }), 500
            
    except Exception as e:
        print(f"\n❌ Server error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # More detailed error message
        error_msg = str(e)
        if "status" in error_msg:
            error_msg = "Generation failed. Please check if ComfyUI is running and try again."
        
        return jsonify({
            'status': 'error',
            'error': error_msg
        }), 500

@app.route('/api/output/<filename>')
def output_file(filename):
    try:
        file_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='image/png')
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'service': 'Doctor Image Generator',
        'version': '2.0',
        'architecture': 'PuLID + FaceDetailer'
    })

if __name__ == '__main__':
    print("\n" + "="*80)
    print("🏥 Doctor Image Generator - Web UI")
    print("="*80)
    print("📋 Features:")
    print("   ✓ Upload 1-3 doctor reference images")
    print("   ✓ Custom positive and negative prompts")
    print("   ✓ PuLID for identity preservation")
    print("   ✓ FaceDetailer for hand and face refinement")
    print("   ✓ High-quality professional medical imagery")
    print("="*80)
    print("🌐 Access the UI at:")
    print(f"   Local: http://localhost:8000")
    print(f"   Network: http://0.0.0.0:8000")
    print("="*80)
    print("\n⚠️  Make sure ComfyUI is running on port 8188!")
    print("    Start it with: cd ComfyUI && python3 main.py --listen 0.0.0.0 --port 8188\n")
    
    app.run(host='0.0.0.0', port=8000, debug=False, threaded=True)

