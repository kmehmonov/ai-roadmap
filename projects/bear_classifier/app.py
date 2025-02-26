import gradio as gr
from fastai.vision.all import *

# Load the trained model
learn = load_learner('models/bear_clf.pkl')

# Define the prediction function
def predict(img):
    pred, pred_idx, probs = learn.predict(img)
    return f"Prediction: {pred} (Confidence: {probs[pred_idx]:.4f})"

# Create Gradio Interface
demo = gr.Interface(
    fn=predict,                  # Function to run
    inputs=gr.Image(type="pil"),  # Input: Image (PIL format)
    outputs=gr.Textbox(),         # Output: Prediction text
    title="Image Classification with FastAI",
    description="Upload an image, and the model will predict its class.",
)

# Run the Gradio app
demo.launch()
