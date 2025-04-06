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
3. Place the model files in the root directory:
   - `garbage_detector_model.h5`
   - `pothole_detector_model.h5`
4. Run the application: `python app.py`
5. Open your browser and go to: `http://localhost:5000`

## Deployment on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Use the following settings:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
4. Add environment variables if needed:
   - `FLASK_DEBUG`: Set to "True" for debugging (use "False" in production)

### Important: Deploying the ML Models

The application requires two model files that are not part of the repository:
- `garbage_detector_model.h5` - for garbage detection
- `pothole_detector_model.h5` - for pothole detection

**Option 1: Using Render Disk**
1. Create a Disk in your Render dashboard
2. Mount the disk to your web service
3. Upload the model files to the mounted disk
4. Update the file paths in `app.py` to point to the mounted disk location

**Option 2: Using a Model Hosting Service**
1. Upload your models to a service like AWS S3, Google Cloud Storage, or similar
2. Modify the code to download the models from the cloud storage at startup

**Option 3: For Testing/Development**
If your models are small enough:
1. Include the model files directly in your repository
2. Make sure they're not in .gitignore

## Checking Application Status

- Visit `/status` to check if the models are loaded correctly
- The application has been designed to work even if models aren't available, showing appropriate error messages

## Troubleshooting

- If you see a 502 error, check the Render logs to see if there are issues loading the models
- Verify that OpenCV dependencies are correctly installed (the Aptfile includes necessary libraries)
- If models fail to load, you can still access the application but detection functionality will be disabled 