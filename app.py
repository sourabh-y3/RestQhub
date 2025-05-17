import gradio as gr
from backend import chatbot_interface, analyze_image

# Function to handle interactive chatbot messages
def interactive_chatbot(message, history=None):
    if history is None:
        history = []
    response, updated_history = chatbot_interface(message, history)
    return updated_history, updated_history

# Function to clear the chatbot
def clear_chat():
    return "", []

# Function to analyze the image with a prompt
def analyze_image_with_prompt(image_path, prompt):
    if not image_path or not prompt:
        return "Please upload an image and provide a prompt for analysis."
    try:
        image_analysis = analyze_image(image_path, prompt)  # Pass prompt to backend analyze_image
        return f"Image Analysis:\n{image_analysis}\n\nPrompt Analysis:\n{prompt}"
    except Exception as e:
        return f"Error analyzing image with prompt: {str(e)}"

# Function to clear the analysis results
def clear_analysis_history():
    return None, "Analysis history cleared."

# Main Gradio Interface
with gr.Blocks(css="styles.css") as app:
    # Tabs for different functionalities
    with gr.Tab("Text Chat"):
        gr.Markdown("# Chat With ResQHub")
        with gr.Row():
            with gr.Column():
                user_message = gr.Textbox(placeholder="Type your message here...", label="Your Message")
            with gr.Column():
                send_button = gr.Button("Send", variant="primary")
                clear_button = gr.Button("Clear Chat")

        chatbot_output = gr.Chatbot(label="VisionAI Chatbot", type="messages", show_label=True)

        send_button.click(interactive_chatbot, inputs=[user_message, gr.State()], outputs=[chatbot_output, gr.State()])
        user_message.submit(interactive_chatbot, inputs=[user_message, gr.State()], outputs=[chatbot_output, gr.State()])
        clear_button.click(clear_chat, inputs=None, outputs=[user_message, chatbot_output])

    with gr.Tab("Image Analysis"):
        gr.Markdown("## 🖼 Image Analysis with Prompt")
        image_input = gr.Image(type="filepath", label="Upload Image")
        prompt_input = gr.Textbox(placeholder="Describe what to analyze in the image...", label="Your Prompt")
        analyze_button = gr.Button("🔍 Analyze", variant="primary")
        clear_analysis = gr.Button("Clear Analysis History")
        result_output = gr.Textbox(label="Analysis Result")

        # Connect Gradio UI components
        analyze_button.click(analyze_image_with_prompt, inputs=[image_input, prompt_input], outputs=result_output)
        prompt_input.submit(analyze_image_with_prompt, inputs=[image_input, prompt_input], outputs=result_output)
        clear_analysis.click(clear_analysis_history, inputs=None, outputs=[image_input, result_output])

    gr.Markdown("Developed by Sangarsh 2.0", elem_id="footer")

if __name__ == "__main__":
    app.launch(share=True, debug=True)





