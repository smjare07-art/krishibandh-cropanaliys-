from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
CORS(app)

# ===============================
# BASE DIRECTORY
# ===============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===============================
# LOAD EXPORTED MODEL
# ===============================

MODEL_PATH = os.path.join(BASE_DIR, "crop_model")

model = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model Loaded Successfully")
except Exception as e:
    print("❌ Model Load Error:", e)

# ===============================
# UPDATE THESE 15 CLASS NAMES
# ⚠ Replace with YOUR dataset folder names
# ===============================

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
        prediction = np.array(prediction).flatten()

        print("Prediction shape:", prediction.shape)

        if len(prediction) != len(class_names):
            return jsonify({
                "error": "Class count mismatch",
                "model_output": len(prediction),
                "class_names": len(class_names)
            }), 500

        class_index = int(np.argmax(prediction))
        confidence = float(np.max(prediction) * 100)

        disease = class_names[class_index]

        return jsonify({
            "disease": disease,
            "confidence": round(confidence, 2),
            "treatment": "Apply recommended fungicide and maintain proper irrigation."
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
