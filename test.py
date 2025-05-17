import gradio as gr

def play_medical_video():
    return "12345.mp4"  # Replace with your actual video file

with gr.Blocks() as demo:
    gr.Markdown("# Medical Assistance in Disaster Management")
    
    # Video Section
    gr.Markdown("### Watch the Demonstration")
    video_player = gr.Video(value=play_medical_video, label="Medical Assistance Video")

    # Upload Video Feature (Optional)
    gr.Markdown("### Upload & Play Your Video")
    upload_video = gr.Video(label="Upload a Video")
    
    demo.launch()
