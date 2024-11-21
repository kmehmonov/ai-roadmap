from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import io
import base64
import numpy as np
import tensorflow as tf
from pathlib import Path

app = Flask(__name__)
CORS(app)

models_path = Path(__file__).resolve().parent.parent / "models"

model = tf.keras.models.load_model(models_path / "mymodel.keras")


@app.route("/process-image", methods=["POST"])
def process_image():
    try:
        data = request.json.get("image")
        if not data:
            return jsonify({"error": "No image data provided"}), 400

        image_data = base64.b64decode(data.split(",")[1])
        image = Image.open(io.BytesIO(image_data))
        # image = image.convert('L') # 'L' mode is grayscale
        image.save("images/digit.png")

        image = image.resize((28, 28))
        image_array = np.array(image)

        np.save("img_data", image_array)

        image_array = (image_array @ np.array([0.2989, 0.5870, 0.1140, 0])).round()
        image_array = image_array / 255

        prediction = model.predict(image_array.reshape(1, 28, 28))
        digit = prediction.argmax()
        score = prediction.max()

        print("Prediction:", prediction)
        print(digit, score)

        response = {
            "message": "Image received successfully",
            "digit": int(digit),
            "score": float(score),
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
