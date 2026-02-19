from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import traceback

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crop_model")

# Load SavedModel
try:
    model = tf.saved_model.load(MODEL_PATH)
    infer = model.signatures["serve"]
    print("✅ SavedModel Loaded Successfully")
except Exception as e:
    print("❌ Model Load Error:", e)
    model = None

class_names = [
    "Apple_scab",
    "Apple_black_rot",
    "Apple_cedar_rust",
    "Apple_healthy",
    "Corn_gray_leaf_spot",
    "Corn_common_rust",
    "Corn_healthy",
    "Potato_early_blight",
    "Potato_late_blight",
    "Potato_healthy",
    "Tomato_early_blight",
    "Tomato_late_blight",
    "Tomato_leaf_mold",
    "Tomato_septoria_leaf_spot",
    "Tomato_healthy"
]

@app.route("/")
def home():
    return "Krishibandh API Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        if "image" not in request.files:
            return jsonify({"error": "No image uploaded"}), 400

        file = request.files["image"]
        temp_path = os.path.join(BASE_DIR, "temp.jpg")
        file.save(temp_path)

        img = image.load_img(temp_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # 🔥 IMPORTANT FIX HERE
        prediction = infer(tf.constant(img_array))["output_0"].numpy()
        prediction = prediction.flatten()

        print("Prediction shape:", prediction.shape)

        index = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        return jsonify({
            "disease": class_names[index],
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
