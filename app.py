from flask import Flask, request, jsonify, render_template
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
import uuid  # To generate unique filenames

app = Flask(__name__)

# Ensure images folder exists
IMAGE_FOLDER = "images"
if not os.path.exists(IMAGE_FOLDER):
    os.makedirs(IMAGE_FOLDER)

# Load the trained model
model = load_model("/content/drive/MyDrive/path_to_your_model.h5")

# Define class labels for A-Z and a-z
class_labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    
    # Generate a unique filename and save the file
    filename = f"{uuid.uuid4().hex}.jpg"  
    filepath = os.path.join(IMAGE_FOLDER, filename)
    file.save(filepath)

    # Load image and preprocess it
    img = image.load_img(filepath, target_size=(64, 64))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction)

    return jsonify({
        "prediction": predicted_class,
        "confidence": float(confidence),
        "saved_image": filepath  # Optional: return image path if needed
    })

if __name__ == "__main__":
    app.run(debug=True)
