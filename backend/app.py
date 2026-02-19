from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crop_model.h5")

# ===== LOAD MODEL SAFELY =====
model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model Loaded")
except Exception as e:
    print("❌ Model Load Error:", e)

# ===== UPDATE THIS IF MODEL HAS DIFFERENT ORDER =====
class_names = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy"
]

@app.route("/")
def home():
    return "Krishibandh Crop Analysis API Running 🚀"

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

        prediction = model.predict(img_array)

        # Convert to flat array safely
        prediction = np.array(prediction).flatten()

        print("Prediction raw:", prediction)

        if len(prediction) == 0:
            return jsonify({"error": "Empty prediction"}), 500

        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        # SAFETY CHECK
        if class_index >= len(class_names):
            return jsonify({
                "error": "Model class mismatch",
                "model_output_size": len(prediction),
                "class_names_size": len(class_names)
            }), 500

        disease = class_names[class_index]

        return jsonify({
            "disease": disease,
            "confidence": round(confidence, 2),
            "treatment": "Recommended fungicide spray and proper irrigation control."
        })

    except Exception as e:
        print("❌ Prediction Error:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
