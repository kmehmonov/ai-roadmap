from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import io
import base64
import numpy as np
import tensorflow as tf
from pathlib import Path
import torch

# from models import AlexNet

app = Flask(__name__)
CORS(app)

models_path = Path(__file__).resolve().parent.parent / "models"
fnn = tf.keras.models.load_model(models_path / "first.keras") #type: ignore
# alexnet_checkpoint = torch.load(models_path / 'alexnet_mnist.pth')
# alexnet = AlexNet(num_classes=10, in_channels=1)
# alexnet.load_state_dict(alexnet_checkpoint)


def base64_to_img_data(base64_str: str):
    image_data = base64.b64decode(base64_str.split(",")[1])
    with Image.open(io.BytesIO(image_data)) as org_image:
        # NOTE: Image is in RGBA format
        image_np = np.array(org_image)

        # Extract the alpha channel
        alpha = image_np[:, :, 3]

        # Convert back to a PIL image
        gray_image = Image.fromarray(alpha).convert('L')

        # Resize to 28x28
        resized_image = gray_image.resize((28, 28))

        img_data = np.array(resized_image, dtype=np.uint8)

        # NOTE: This lines are for debugging
        org_image.save("images/org_image.png")
        gray_image.save("images/gray_image.png")
        resized_image.save("images/resized_image.png")
        np.save('images/arr.npy', img_data)

    return img_data

def image_transform(img_data):
    return (img_data / 255.0).round()



@app.route("/process-image", methods=["POST"])
def process_image():
    try:
        request_data = request.json
        if request_data:
            image_str = request_data.get("image")
        else:
            return jsonify({"error": "No image data provided"}), 400

        img_data = base64_to_img_data(image_str)
        img_data = image_transform(img_data)

        prediction = fnn.predict(img_data.reshape(1, 28, 28))
        digit = prediction.argmax()
        score = prediction.max()

        response = {
            "message": "Image analyzed successfully",
            "digit": int(digit),
            "score": float(score),
        }
        return jsonify(response)
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


@app.route('/', methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/digit", methods=["GET"])
def index_digit():
    models = {
        1: 'Logistic Regression',
        2: 'Feedforward NN',
        3: 'AlexNet'
    }
    return render_template("index_digit.html", models=models)


if __name__ == "__main__":
    app.run(debug=True)
