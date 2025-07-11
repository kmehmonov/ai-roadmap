import numpy as np
import gradio as gr
import joblib
from PIL import Image
from pathlib import Path

# Load model and scaler
path = Path(__file__).parent
model = joblib.load(path / 'svm_digit_classifier.pkl')
scaler = joblib.load(path / 'scaler.pkl')

# Prediction function
def predict_digit(img):
    if img is None:
        return "Please draw a digit."

    # Extract image if it's in a dict
    if isinstance(img, dict):
        for key in ['composite', 'image', 'data', 'layers']:
            if key in img and img[key] is not None:
                img = img[key]
                break
        else:
            return "No image data found."

    # Convert to NumPy array
    img = np.array(img, dtype=np.uint8)

    # Convert to grayscale if colored
    if img.ndim == 3 and img.shape[2] in [3, 4]:
        img = Image.fromarray(img).convert('L')
    else:
        img = Image.fromarray(img)

    # Resize and invert (MNIST format)
    img = img.resize((28, 28))
    img = np.array(img)
    img = 255 - img

    # Flatten and scale
    img_flat = img.reshape(1, -1)
    img_scaled = scaler.transform(img_flat)

    # Predict
    pred = model.predict(img_scaled)[0]
    return f"Predicted Digit: {pred}"

# Gradio interface
with gr.Blocks(title="Digit Predictor by Kamoliddin") as app:
    gr.Markdown("## Draw a digit (0-9) and click Predict.")
    canvas = gr.Sketchpad()
    btn = gr.Button("Predict")
    result = gr.Textbox()

    btn.click(predict_digit, inputs=canvas, outputs=result)

app.launch(share=True)
