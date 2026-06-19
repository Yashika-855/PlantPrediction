import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ---- Config (must match training script) ----
IMG_SIZE = (96, 96)
MODEL_PATH = "plant_disease_fast.keras"
CLASS_NAMES_PATH = "class_names.json"
UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---- Load model and class names once at startup ----
print("Loading model...")
model = load_model(MODEL_PATH)

with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print(f"Model loaded. {len(class_names)} classes available.")


def predict_image(img_path):
    """Load an image, preprocess to match training pipeline, and predict."""
    img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32)
    # NOTE: no manual /255 here — the model has a built-in Rescaling(1./255) layer
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_idx = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_idx] * 100)
    class_name = class_names[predicted_idx]

    top3_idx = predictions[0].argsort()[-3:][::-1]
    top3 = [
        {"class": class_names[int(i)], "confidence": float(predictions[0][i] * 100)}
        for i in top3_idx
    ]

    return class_name, confidence, top3


def format_label(raw_name):
    """Turn 'Tomato___Late_blight' into 'Tomato - Late blight'."""
    return raw_name.replace("___", " - ").replace("_", " ")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(save_path)

    try:
        class_name, confidence, top3 = predict_image(save_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "prediction": format_label(class_name),
        "raw_class": class_name,
        "confidence": round(confidence, 2),
        "top3": [
            {"class": format_label(t["class"]), "confidence": round(t["confidence"], 2)}
            for t in top3
        ],
        "image_url": "/" + save_path
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
