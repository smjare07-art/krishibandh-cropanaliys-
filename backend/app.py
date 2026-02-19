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
# SAFE MODEL LOADING
# ===============================

model = None
model_path = os.path.join(os.path.dirname(__file__), "crop_model.h5")

try:
    model = tf.keras.models.load_model(model_path, compile=False)
    print("✅ Model loaded successfully")
except Exception as e:
    print("❌ Model loading failed:", e)

# ===============================
# LOAD AGRICULTURE DATA
# ===============================

agri_data = {}
data_path = os.path.join(os.path.dirname(__file__), "agriculture_data.json")

if os.path.exists(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        agri_data = json.load(f)
    print("✅ Agriculture data loaded")
else:
    print("⚠ agriculture_data.json not found")

# ===============================
# CLASS NAMES
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

        temp_path = os.path.join(os.path.dirname(__file__), "temp.jpg")
        file.save(temp_path)

        img = image.load_img(temp_path, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        disease = class_names[class_index]

        details = agri_data.get(disease, {})

        return jsonify({
            "disease": disease,
            "confidence": round(confidence, 2),
            "causes": details.get("causes", []),
            "water_role": details.get("water_role", "Information not available"),
            "fertilizer": details.get("fertilizer_schedule", {}),
            "treatment": details.get("treatment", "Information not available")
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
