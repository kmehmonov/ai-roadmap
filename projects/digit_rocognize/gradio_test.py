import gradio as gr
from PIL import Image

def predict(img, sketch):
    print(sketch)
    return "Prediction"


iface = gr.Interface(
    fn=predict,
    inputs=[gr.Image(), gr.Sketchpad()],
    outputs='text'
)

iface.launch()
