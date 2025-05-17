import gradio as gr
import cv2
import numpy as np
import tensorflow as tf
from moviepy.editor import VideoFileClip # type: ignore
import os

# Load the trained deepfake detection model
MODEL_PATH = "models/final_model6.h5"  # Update with your actual model path
model = tf.keras.models.load_model(MODEL_PATH)

# Function to extract frames from the video
def extract_frames(video_path, num_frames=10):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for i in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, (224, 224))  # Adjust based on model input size
            frame = frame / 255.0  # Normalize
            frames.append(frame)
    
    cap.release()
    return np.array(frames)

# Deepfake detection function
def detect_deepfake(video_path):
    frames = extract_frames(video_path)
    
    if len(frames) == 0:
        return "Error: No valid frames extracted from video."
    
    frames = np.expand_dims(frames, axis=0)  # Add batch dimension
    prediction = model.predict(frames)  # Get predictions
    
    # Assuming the model outputs a probability (adjust threshold as needed)
    deepfake_score = np.mean(prediction)
    result = "Deepfake Detected" if deepfake_score > 0.5 else "Real Video"
    
    return result

# Gradio Interface
iface = gr.Interface(
    fn=detect_deepfake,
    inputs=gr.Video(label="Upload a Video"),
    outputs=gr.Textbox(label="Detection Result"),
    title="Deepfake Detection OSINT Tool",
    description="Upload a video to check if it's real or a deepfake."
)

# Run the app
iface.launch()

