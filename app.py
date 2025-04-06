from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import cv2
import uuid

app = Flask(__name__)

# Load both models
garbage_model = None
pothole_model = None
models_loaded = False

def load_models():
    global garbage_model, pothole_model, models_loaded
    
    try:
        if os.path.exists('garbage_detector_model.h5'):
            print("Loading garbage detection model...")
            garbage_model = load_model('garbage_detector_model.h5')
            if garbage_model is not None:
                models_loaded = True
        else:
            print("Warning: Garbage detection model not found!")
        
        if os.path.exists('pothole_detector_model.h5'):
            print("Loading pothole detection model...")
            pothole_model = load_model('pothole_detector_model.h5')
            if pothole_model is not None:
                models_loaded = True
        else:
            print("Warning: Pothole detection model not found!")
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        models_loaded = False

# Initialize model loading - but don't block app startup if models fail
try:
    load_models()
except Exception as e:
    print(f"Error during model loading: {str(e)}")

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def highlight_features(img_path, confidence, detection_type):
    """Apply visual highlighting to areas likely containing the detected feature"""
    try:
        img = cv2.imread(img_path)
        
        # Convert to HSV color space
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Create a blurred version for edge detection
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Apply highlight color based on detection type
        if confidence > 0.7:
            # Create a mask from edges
            mask = cv2.dilate(edges, None, iterations=2)
            
            if detection_type == 'garbage':
                # Red for garbage
                highlight_color = [0, 0, 255]
            else:
                # Orange for potholes
                highlight_color = [0, 165, 255]
                
            img[mask > 0] = highlight_color
        
        # Save highlighted image
        highlight_path = os.path.join(os.path.dirname(img_path), f"highlight_{os.path.basename(img_path)}")
        cv2.imwrite(highlight_path, img)
        
        return highlight_path
    except Exception as e:
        print(f"Error in highlight_features: {str(e)}")
        return None

def predict_image(img_path, detection_type):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_tensor = image.img_to_array(img)
        img_tensor = img_tensor / 255.0
        img_tensor = np.expand_dims(img_tensor, axis=0)

        if detection_type == 'garbage':
            if garbage_model is None:
                return "Error: Garbage model not loaded", 0, None
            prediction = garbage_model.predict(img_tensor)[0][0]
            positive_class = "Garbage Found 🗑️"
            negative_class = "No Garbage Found ✅"
        else:  # pothole
            if pothole_model is None:
                return "Error: Pothole model not loaded", 0, None
            prediction = pothole_model.predict(img_tensor)[0][0]
            positive_class = "Potholes Found 🚧"
            negative_class = "Road is Perfect ✅ No Potholes"
        
        confidence = float(prediction)
        
        # Threshold values can be adjusted based on model performance
        threshold = 0.6
        
        # Apply thresholds for classification
        if confidence > threshold:
            result = positive_class
            # Highlight the detected feature in the image
            highlight_path = highlight_features(img_path, confidence, detection_type)
        else:
            result = negative_class
            highlight_path = None
        
        # Calculate percentage for display
        if confidence > 0.5:
            percentage = confidence * 100
        else:
            percentage = (1 - confidence) * 100
            
        return result, percentage, highlight_path
    except Exception as e:
        print(f"Error in predict_image: {str(e)}")
        return f"Error processing image: {str(e)}", 0, None

@app.route('/')
def index():
    return render_template('index.html', models_loaded=models_loaded)

@app.route('/detect/<detection_type>', methods=['POST'])
def upload_and_predict(detection_type):
    if not models_loaded:
        return render_template('error.html', 
                              error_message="ML models are not loaded. Please contact the administrator."), 503
                              
    if detection_type not in ['garbage', 'pothole']:
        return redirect(url_for('index'))
        
    if 'image' not in request.files:
        return "No file uploaded.", 400

    file = request.files['image']
    if file.filename == '':
        return "No file selected.", 400
    
    try:
        # Generate unique filename to prevent overwriting
        filename = f"{uuid.uuid4()}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result, confidence, highlight_path = predict_image(filepath, detection_type)
        
        # Generate relative paths for template
        image_url = f"/static/uploads/{os.path.basename(filepath)}"
        highlight_url = f"/static/uploads/{os.path.basename(highlight_path)}" if highlight_path else None

        return render_template('result.html', 
                              image_path=image_url,
                              highlight_path=highlight_url,
                              result=result, 
                              confidence=confidence,
                              detection_type=detection_type)
    except Exception as e:
        return render_template('error.html', 
                              error_message=f"An error occurred: {str(e)}"), 500

@app.route('/health')
def health_check():
    """Health check endpoint for Render"""
    return jsonify({
        "status": "healthy",
        "models_loaded": models_loaded
    }), 200

@app.route('/status')
def status():
    """Status page showing app info"""
    return render_template('status.html', 
                          models_loaded=models_loaded,
                          garbage_model=garbage_model is not None,
                          pothole_model=pothole_model is not None)

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
