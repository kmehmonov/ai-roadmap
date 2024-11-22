from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image, ImageOps
import io
import base64
import numpy as np
import tensorflow as tf
from pathlib import Path

app = Flask(__name__)
CORS(app)

models_path = Path(__file__).resolve().parent.parent / "models"
model = tf.keras.models.load_model(models_path / "first.keras")


def base64_to_img_data(base64_str: str):
    image_data = base64.b64decode(base64_str.split(",")[1])
    with Image.open(io.BytesIO(image_data)) as org_image:
        org_image.save("images/org_image.png")

        gray_image = ImageOps.grayscale(org_image)
        gray_image.save("images/gray_image.png")

        resized_image = gray_image.resize((28, 28))
        resized_image.save("images/resized_image.png")

        img_data = np.array(resized_image)
        img_data = (img_data / 255).round()

    return img_data


@app.route("/process-image", methods=["POST"])
def process_image():
    try:
        image_str = request.json.get("image")
        if not image_str:
            return jsonify({"error": "No image data provided"}), 400

        img_data = base64_to_img_data(image_str)

        prediction = model.predict(img_data.reshape(1, 28, 28))
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


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
