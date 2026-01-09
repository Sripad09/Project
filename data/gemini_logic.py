
import google.generativeai as genai
import os
import PIL.Image
import json
import ast

from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
# Using user provided key
API_KEY = os.getenv("GEMINI_API_KEY")

if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
    except Exception as e:
        print(f"Failed to configure Gemini: {e}")

CLASS_NAMES = ["Clean", "Little Polluted", "Highly Polluted"]

def analyze_image_with_gemini(image_path, pollution_type="water"):
    """
    Analyzes image using Gemini and maps to existing frontend structure.
    """
    prediction = ""
    class_name = "Medium"
    probs = {k: 0.0 for k in CLASS_NAMES}
    analysis = []

    if not API_KEY:
        return {
            "prediction": "Gemini Key Missing",
            "class_name": "Medium",
            "probs": probs,
            "analysis": ["Error: GEMINI_API_KEY not found."]
        }

    # Use a vision-capable model
    # User requested model
    model_name = "gemini-2.5-flash-image"
    
    try:
        model = genai.GenerativeModel(model_name)
    except:
        model = genai.GenerativeModel("gemini-pro-vision")

    prompt = f"""
    You are an expert environmental analyst.
    Analyze this {pollution_type} image for pollution levels.
    Classify the pollution level into exactly one of these categories: "Clean", "Little Polluted", "Highly Polluted".
    
    Requirements:
    1. Determine the Classification ("Clean", "Little Polluted", "Highly Polluted").
    2. Provide a confidence percentage (0-100) for that classification.
    3. Estimate probabilities for all three categories so they likely sum to 100.
    4. Provide 3-5 short, bullet-point style observations explaining the decision (e.g., color, turbidity, floating debris).

    Return ONLY valid JSON with this exact structure:
    {{
        "classification": "Clean",
        "confidence": 98.5,
        "probabilities": {{
            "Clean": 98.5,
            "Little Polluted": 1.0,
            "Highly Polluted": 0.5
        }},
        "analysis": [
            "Water appears clear blue with no visible debris.",
            "No signs of algae bloom or oil slick.",
            "Visual transparency suggests low turbidity."
        ]
    }}
    """
    
    try:
        img = PIL.Image.open(image_path)
        response = model.generate_content([prompt, img])
        
        content = response.text
        # Clean up code blocks if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
             content = content.split("```")[1].split("```")[0]
        
        content = content.strip()
        
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Fallback for malformed JSON, try ast literal eval if it looks like python dict
            try:
                data = ast.literal_eval(content)
            except:
                raise ValueError(f"Could not parse Gemini response: {content}")

        # Extract values
        pred_class = data.get("classification", "Little Polluted")
        # Validate class
        if pred_class not in CLASS_NAMES:
            # Simple fuzzy match or fallback
            if "clean" in pred_class.lower(): pred_class = "Clean"
            elif "little" in pred_class.lower(): pred_class = "Little Polluted"
            elif "high" in pred_class.lower(): pred_class = "Highly Polluted"
            else: pred_class = "Little Polluted"

        confidence = float(data.get("confidence", 0))
        
        raw_probs = data.get("probabilities", {})
        # Map raw probs to float and ensure keys
        for name in CLASS_NAMES:
            val = raw_probs.get(name, 0)
            probs[name] = float(val)
        
        analysis = data.get("analysis", ["No details provided."])
        
        # Determine CSS class
        if pred_class == "Clean":
            class_name = "Low"
        elif pred_class == "Little Polluted":
            class_name = "Medium"
        else:
            class_name = "High"

        # Format prediction string
        prediction = f"{pred_class} ({confidence:.2f}% confidence)"
        
        return {
            "prediction": prediction,
            "class_name": class_name,
            "probs": probs,
            "analysis": analysis
        }
        
    except Exception as e:
        print(f"Gemini Analysis Error: {e}")
        return {
            "prediction": "Analysis Failed",
            "class_name": "Medium",
            "probs": probs,
            "analysis": [f"Error occurred during AI analysis: {str(e)}"]
        }
