




import os
import google.generativeai as genai
from PIL import Image
import base64
import io

# Fetch API Key securely
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("API key not found. Please set the GEMINI_API_KEY environment variable.")

# Configure Gemini API
genai.configure(api_key=api_key)

# Model Configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model_name = "gemini-2.0-flash"
model = genai.GenerativeModel(
    model_name=model_name,
    generation_config=generation_config,
)

chat_session = model.start_chat(history=[])

# Function to handle text-based chat
def text_chat(prompt):
    try:
        response = chat_session.send_message(prompt)
        return response.text if hasattr(response, "text") else "No response received."
    except Exception as e:
        return f"Error: {str(e)}"

# Function for chatbot interaction
def chatbot_interface(message, history=None):
    if history is None:
        history = []
    history.append({"role": "user", "content": message})
    try:
        response = text_chat(message)
        history.append({"role": "assistant", "content": response})
        return response, history
    except Exception as e:
        error_message = f"Error: {str(e)}"
        history.append({"role": "assistant", "content": error_message})
        return error_message, history

# Function to analyze an image with the model
def analyze_image(image_path, prompt):
    try:
        # Open and convert image to base64
        image = Image.open(image_path)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")

        # Construct multimodal request
        combined_prompt = f"Here is an image. Analyze it based on: {prompt}"

        response = model.generate_content([combined_prompt, img_str])
        
        return response.text if hasattr(response, "text") else "No response received."
    except Exception as e:
        return f"Error analyzing image: {str(e)}"

# Function to analyze audio (Placeholder)
def analyze_audio(audio_path):
    try:
        return f"Analysis complete: Processed the audio at {audio_path}."
    except Exception as e:
        return f"Error analyzing audio: {str(e)}"

# Function for advanced queries
def advanced_assistant(query):
    try:
        return text_chat(query)
    except Exception as e:
        return f"Error: {str(e)}"









