from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import json

app = Flask(__name__)
CORS(app)

# ===============================
# SAFE MODEL LOAD
# ===============================

model = None
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "crop_model.h5")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model Loaded Successfully")
except Exception as e:
    print("❌ Model Load Error:", e)

# ===============================
# OPTIONAL AGRICULTURE DATA
# ===============================

agri_data = {}
DATA_PATH = os.path.join(BASE_DIR, "agriculture_data.json")

if os.path.exists(DATA_PATH):
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        agri_data = json.load(f)
    print("✅ Agriculture Data Loaded")
else:
    print("⚠ agriculture_data.json not found")

# ===============================
# CLASS NAMES (MODEL ORDER)
# ===============================

class_names = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy"
]

# ===============================
# ROUTES
# ===============================

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

        if isinstance(prediction, list):
            prediction = prediction[0]

        prediction = np.array(prediction).flatten()

        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        disease = class_names[class_index]

        details = agri_data.get(disease, {})

        return jsonify({
            "disease": disease,
            "confidence": round(confidence, 2),
            "causes": details.get("causes", []),
            "water_role": details.get("water_role", ""),
            "fertilizer": details.get("fertilizer_schedule", {}),
            "treatment": details.get("treatment", "उपाय उपलब्ध नाही")
        })

    except Exception as e:
        print("❌ Prediction Error:", e)
        return jsonify({"error": str(e)}), 500


# ===============================
# RUN
# ===============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
