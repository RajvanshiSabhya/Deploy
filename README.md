# Garbage and Pothole Detection App

A Flask web application that uses machine learning models to detect garbage and potholes in images.

## Features

- Garbage detection in images
- Pothole detection in images
- Visual highlighting of detected features
- Confidence score display

## Technical Stack

- Flask
- TensorFlow/Keras
- OpenCV
- HTML/CSS/JavaScript

## Local Development

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `python app.py`
4. Open your browser and go to: `http://localhost:5000`

## Deployment on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
4. Add environment variables if needed
5. Deploy the application

## Models

The application uses two pretrained models:
- `garbage_detector_model.h5` - for garbage detection
- `pothole_detector_model.h5` - for pothole detection

Place these files in the root directory before deployment. 