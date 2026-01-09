
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
            "prediction": "MODEL NOT FOUND",
            "class_name": "Medium",
            "probs": probs,
            "analysis": ["Error: MODEL NOT FOUND."]
        }

    # Use a vision-capable model
    # User requested model
    model_name = "gemini-2.5-flash-image"
    
    try:
        model = genai.GenerativeModel(model_name)
    except:
        model = genai.GenerativeModel("gemini-pro-vision")

    prompt = f"""
    You are an automated environmental sensor system.
    
    STEP 1: VALIDATION
    Determine if the image is relevant to "{pollution_type}".
    - If pollution_type is "water": The image MUST contain a water body (river, lake, ocean, pond, puddle, sewage, etc.).
    - If pollution_type is "air": The image MUST contain sky, atmosphere, industrial smoke, smog, or open outdoor views.
    
    IF THE IMAGE IS NOT OF WATER OR AIR:
    Return exactly this JSON and STOP:
    {{
        "classification": "Invalid",
        "confidence": 100,
        "probabilities": {{
            "Clean": 0,
            "Little Polluted": 0,
            "Highly Polluted": 0
        }},
        "analysis": ["The image is not a recognized {pollution_type} body.", "Analysis aborted."]
    }}

    STEP 2: POLLUTION ANALYSIS (Only if Valid)
    Analyze the ENTIRE provided {pollution_type} image to detect pollution levels.
    
    CRITICAL INSTRUCTIONS:
    - FOCUS ONLY ON POLLUTION INDICATORS (turbidity, unnatural color, waste, oil, smoke, smog).
    - DO NOT describe the artistic style, camera angle, or scenery unless it directly relates to pollution.
    - DO NOT use phrases like "The image shows", "I can see", or "Gemini".
    - Output must be purely technical and objective.

    Classify the pollution level into exactly one of these categories: "Clean", "Little Polluted", "Highly Polluted".
    
    Requirements:
    1. Determine the Classification ("Clean", "Little Polluted", "Highly Polluted").
    2. Provide a confidence percentage (0-100) for that classification.
    3. Estimate probabilities for all three categories so they likely sum to 100.
    4. Provide 3-5 short, direct, technical observations explaining the decision.

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
            "Water clarity is high with distinct visibility of riverbed details.",
            "Absence of surface debris or oily sheen.",
            "Natural water coloration with no signs of chemical turbidity."
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
                # If parsing fails, it's likely a refusal or plain text explanation
                return {
                    "prediction": "Try Again",
                    "class_name": "Medium",
                    "probs": probs,
                    "analysis": [content]
                }

        # Extract values
        pred_class = data.get("classification", "Little Polluted")
        
        # Handle Invalid Image Case
        if pred_class == "Invalid":
            return {
                "prediction": "Not a Water/Air Body",
                "class_name": "Medium",
                "probs": {k: 0.0 for k in CLASS_NAMES},
                "analysis": data.get("analysis", ["Image content does not match the selected pollution type."])
            }

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
