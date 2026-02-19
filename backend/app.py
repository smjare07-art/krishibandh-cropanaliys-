from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
CORS(app)

model = None  # Lazy load

class_names = [
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_healthy"
]

treatment = {
    "Tomato_Late_blight": "Use copper-based fungicide and remove infected leaves.",
    "Tomato_Early_blight": "Apply fungicide and improve air circulation.",
    "Tomato_healthy": "No disease detected. Keep monitoring."
}

def load_model():
    global model
    if model is None:
        model = tf.keras.models.load_model("crop_model.h5", compile=False)

@app.route("/")
def home():
    return "Crop Disease AI Backend Running 🚀"

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return "Use POST method with image upload"
def predict():
    load_model()  # Load only when needed

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    filepath = "temp.jpg"
    file.save(filepath)

    img = image.load_img(filepath, target_size=(128,128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    class_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction) * 100)

    disease = class_names[class_index]
    suggestion = treatment.get(disease, "No suggestion available")

    return jsonify({
        "disease": disease,
        "confidence": round(confidence, 2),
        "suggestion": suggestion
    })
