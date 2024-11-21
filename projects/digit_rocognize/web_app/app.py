from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image
import io
import base64
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

model = load_model(r"D:\Python\Projects\ml\digit_rocognize\models\mymodel.keras")


@app.route("/process-image", methods=["POST"])
def process_image():
    try:
        data = request.json.get("image")
        if not data:
            return jsonify({"error": "No image data provided"}), 400

        image_data = base64.b64decode(data.split(",")[1])
        image = Image.open(io.BytesIO(image_data))
        # image = image.convert('LA') # 'LA' mode is grayscale with Aplha
        image_array = np.array(image)
        # print(image_array)
        np.save("img_data", image_array)

        image.save("myimage.png")
        height, widht, *_ = image_array.shape

        image_array = (image_array @ np.array([0.2989, 0.5870, 0.1140, 0])).round()
        image_array = image_array / 255
        prediction = model.predict(image_array.reshape(1, 28, 28))

        print("Prediction:", prediction)
        print(prediction.argmax(), prediction.max())

        response = {
            "message": "Image received successfully",
            "height": height,
            "width": widht,
        }
        return jsonify(response)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
