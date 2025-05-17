from ultralytics import YOLO
from PIL import Image, ImageDraw

# Load YOLO model
model = YOLO("yolov8n.pt")

def detect_objects_yolo(image_path):
    """
    Detect objects in an image using YOLO.
    Args:
        image_path (str): Path to the input image.
    Returns:
        Tuple: Annotated image and text description of detections.
    """
    try:
        # Run YOLO on the input image
        results = model(image_path)

        # Load the original image
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        detections_text = []  # List to store detection descriptions

        # Process results and draw bounding boxes
        for result in results:
            for box in result.boxes:  # Loop through detected boxes
                # Get the class name, confidence score, and bounding box coordinates
                cls = result.names[box.cls[0].item()]  # Class name (e.g., "person")
                confidence = box.conf[0].item()  # Confidence score
                bbox = box.xyxy[0].tolist()  # Bounding box coordinates [x1, y1, x2, y2]

                # Draw bounding box
                x1, y1, x2, y2 = bbox
                draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

                # Prepare detection description
                detections_text.append(f"{cls} ({confidence:.2f})")

        # Combine detections into a single text
        detections_text = ", ".join(detections_text)

        return img, detections_text

    except Exception as e:
        print("Error in YOLO detection:", e)
        return None, "Error in detection"




import os
from google.cloud import texttospeech

def generate_gemini_text(detections):
    """
    Generate enhanced text using Google Text-to-Speech API (or another Gemini approach).
    """
    try:
        # Set credentials programmatically
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/service-account-key.json"

        # Initialize the Text-to-Speech client
        client = texttospeech.TextToSpeechClient()

        # Configure synthesis input
        synthesis_input = texttospeech.SynthesisInput(text=detections)

        # Select the voice
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )

        # Configure audio output
        audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3)

        # Perform text-to-speech synthesis
        response = client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )

        # Return the generated text as a fallback
        return detections

    except Exception as e:
        print("Error in Gemini text generation:", e)
        return "An error occurred while generating enhanced text."


def process_image_with_prompt(image_path, prompt):
    """
    Process the image and prompt by detecting objects with YOLO and enhancing text with Gemini.
    """
    # Detect objects with YOLO
    detection_result, detections_text = detect_objects_yolo(image_path)

    # Handle cases where no detections are made
    if not detections_text:
        detections_text = "No objects detected in the image."

    # Append the user prompt
    full_prompt = f"Detections: {detections_text}. User prompt: {prompt}"

    # Generate enhanced text using Gemini
    enhanced_text = generate_gemini_text(full_prompt)

    return detection_result, enhanced_text

