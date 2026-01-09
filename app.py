from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
try:
    from tensorflow import keras
except ImportError:
    import keras
import numpy as np
from PIL import Image
import os
import uuid

# ---------------------------
# Config
# ---------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

MODEL_PATHS = {
    "water": "water_best_model.h5",
    "air": "air_best_model.h5"
}

CLASS_NAMES = ["Clean", "Little Polluted", "Highly Polluted"]

# ---------------------------
# Helper functions
# ---------------------------
def load_model(model_path):
    if not os.path.exists(model_path):
        return None
    return keras.models.load_model(model_path)

def preprocess_image(image, img_size=(224, 224)):
    img = image.resize(img_size)
    img_array = keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
    return img_array

def analyze_water_rgb(image):
    img = np.array(image)
    avg_rgb = np.mean(img, axis=(0, 1))  
    r, g, b = avg_rgb
    analysis = []

    if g > r and g > b and g > 100:
        if g > 150:
             analysis.append("🟢 High green concentration: Critical algae bloom detected. Immediate treatment required.")
        else:
             analysis.append("🟢 Moderate green: Algae presence. Monitor oxygen levels.")
        
    if r > g and r > b and r > 100:
        if r > 160:
             analysis.append("🔴 Intense red/brown: Severe iron contamination or heavy sedimentation. Unsafe.")
        else:
             analysis.append("🟤 Brown turbidity: Possible suspended solids or iron. Filtration needed.")

    if b > r and b > g:
        if b > 140 and (r < 100 and g < 100):
             analysis.append("🔵 Clear blue: Water appears visually clean and free of turbidity.")
        elif b > 120:
             analysis.append("🔵 Blueish tint: Generally clean, but verify pH.")

    avg_val = np.mean(avg_rgb)
    if avg_val < 50:
        analysis.append("⚫ Extremely dark: High risk of industrial effluent, oil, or black water. Do not touch.")
    elif avg_val < 90:
        analysis.append("⚫ Dark/Turbid: Possible organic decay or waste. unsafe.")

    std_dev = np.std(img) # Check for texture/variance
    if std_dev < 20: 
        analysis.append("⚠️ Low texture variance: Water might be stagnant or consistent sludge.")

    if not analysis:
        analysis.append("⚪ Visual inspection is inconclusive. Chemical testing recommended.")
    return avg_rgb, analysis

def analyze_air_rgb(image):
    img = np.array(image)
    avg_rgb = np.mean(img, axis=(0, 1))  
    r, g, b = avg_rgb
    analysis = []

    if np.mean(avg_rgb) < 80:
        analysis.append("🌫️ Dark/gray tones: Possible smog or soot. Avoid exposure, use masks/air filters.")
    if r > 120 and r > g and r > b:
        analysis.append("🟤 Brown haze: Possible dust or industrial emissions. Reduce outdoor activity.")
    if g > 120 and g > r and g > b:
        analysis.append("🟢 Greenish tint: Could indicate chemical pollutants. Ventilation and monitoring advised.")
    if b > 130 and b > r and b > g:
        analysis.append("🔵 Clear sky with strong blue: Clean air likely.")
    if not analysis:
        analysis.append("No major harmful air constituents detected visually.")
    return avg_rgb, analysis

# ---------------------------
# Flask Routes
# ---------------------------
@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/select")
def select():
    return render_template("select.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    pollution_type = request.args.get("type", "water").lower()

    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400

        # Save uploaded file
        filename = str(uuid.uuid4()) + os.path.splitext(file.filename)[1]
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        file.save(filepath)

        # Open image
        image = Image.open(filepath).convert("RGB")

        # --- Gemini Integration ---
        from data.gemini_logic import analyze_image_with_gemini
        
        # Get result from Gemini
        gemini_result = analyze_image_with_gemini(filepath, pollution_type)
        
        prediction = gemini_result["prediction"]
        class_name = gemini_result["class_name"]
        probs = gemini_result["probs"]
        analysis = gemini_result["analysis"]

        # Calculate avg_rgb for compatibility (used in template or debug)
        # We can reuse the existing function just for RGB, ignoring its analysis
        if pollution_type == "water":
            avg_rgb, _ = analyze_water_rgb(image)
        else:
            avg_rgb, _ = analyze_air_rgb(image)

        return render_template(
            "result.html",
            filename=filename,
            prediction=prediction,
            class_name=class_name,
            probs=probs,
            analysis=analysis,
            avg_rgb=avg_rgb
        )

    return render_template("predict.html", type=pollution_type)

# ---------------------------
# Run App
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True)
