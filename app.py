from flask import Flask, request, jsonify
from pathlib import Path
import numpy as np
import pickle
import os
import json
import requests
import base64
import io
from flask_cors import CORS

try:
    import google.generativeai as genai
except Exception:
    genai = None


app = Flask(__name__)
# Enable CORS for all routes - Flask-CORS will handle all headers automatically
CORS(app, origins="*", methods=["GET", "POST", "OPTIONS"], allow_headers=["Content-Type", "Authorization"], supports_credentials=False)


"""Model auto-discovery: load the first usable .pkl under model/."""
MODEL = None
MODEL_PATH = None
model_dir = Path("model")
MODEL_LOAD_ERRORS = []
try:
    pkls = sorted(model_dir.glob("*.pkl")) if model_dir.exists() else []
    # Keep the old preferred names first if present
    preferred = [model_dir / "crop_recommendation_model.pkl", model_dir / "npk_crop_model.pkl"]
    ordered = [p for p in preferred if p in pkls] + [p for p in pkls if p not in preferred]
    for candidate in ordered:
        try:
            with open(candidate, "rb") as f:
                candidate_model = pickle.load(f)
            # Sanity check: must have predict
            if not hasattr(candidate_model, "predict"):
                MODEL_LOAD_ERRORS.append({
                    "path": str(candidate),
                    "reason": "Loaded object has no 'predict' attribute",
                })
                continue
            MODEL = candidate_model
            MODEL_PATH = str(candidate)
            break
        except Exception:
            # Try joblib as fallback
            try:
                from joblib import load as joblib_load  # type: ignore
                candidate_model = joblib_load(candidate)
                if not hasattr(candidate_model, "predict"):
                    MODEL_LOAD_ERRORS.append({
                        "path": str(candidate),
                        "reason": "Joblib-loaded object has no 'predict' attribute",
                    })
                    continue
                MODEL = candidate_model
                MODEL_PATH = str(candidate)
                break
            except Exception as e2:
                MODEL_LOAD_ERRORS.append({
                    "path": str(candidate),
                    "reason": f"Failed to load: {type(e2).__name__}: {str(e2)}",
                })
                continue
except Exception:
    MODEL = None
    MODEL_PATH = None


@app.route("/", methods=["GET"])  # Simple API health/info
def root_info():
    return jsonify({
        "name": "DT_project API",
        "modelLoaded": MODEL is not None,
        "modelPath": MODEL_PATH,
        "errors": MODEL_LOAD_ERRORS,
        "endpoints": [
            {"method": "POST", "path": "/predict"},
            {"method": "POST", "path": "/suggest"},
            {"method": "GET", "path": "/debug/models"},
            {"method": "GET", "path": "/weather"},
        ],
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)

        if MODEL is None:
            # Model not available; echo back the inputs so the frontend is wired up
            return jsonify({
                "modelLoaded": False,
                "message": "Model file not found or failed to load.",
                "inputs": payload,
            })

        # Determine how many features the loaded model expects
        expected_features = getattr(MODEL, "n_features_in_", None)
        if expected_features is None:
            # Try to infer from feature_names_in_
            feature_names_in = getattr(MODEL, "feature_names_in_", None)
            if feature_names_in is not None:
                expected_features = len(feature_names_in)

        # Full set we support from the frontend
        full_feature_order = [
            "N",
            "P",
            "K",
            "temperature",
            "humidity",
            "ph",
            "rainfall",
        ]

        # Choose subset based on model expectation
        if expected_features == 3:
            feature_order = ["N", "P", "K"]
        elif expected_features == 7:
            feature_order = full_feature_order
        else:
            # Fallback: if unknown, prefer full order; model may still handle internally (e.g., Pipeline)
            feature_order = full_feature_order

        # Validate and build feature vector in the chosen order
        features = []
        for key in feature_order:
            value = payload.get(key)
            if value is None:
                return jsonify({"error": f"Missing field: {key}", "expectedFields": feature_order}), 400
            try:
                features.append(float(value))
            except ValueError:
                return jsonify({"error": f"Invalid number for {key}", "expectedFields": feature_order}), 400

        X = np.array([features], dtype=float)

        # Run prediction
        try:
            y_pred = MODEL.predict(X)
        except Exception as e:
            return jsonify({
                "modelLoaded": True,
                "modelPath": MODEL_PATH,
                "error": f"Model prediction failed: {str(e)}",
                "usedFields": feature_order,
            }), 500

        # y_pred can be numpy array; take first element and convert to str for safety
        pred_value = y_pred[0]
        try:
            pred_serializable = pred_value.item() if hasattr(pred_value, "item") else pred_value
        except Exception:
            pred_serializable = str(pred_value)

        return jsonify({
            "modelLoaded": True,
            "modelPath": MODEL_PATH,
            "prediction": pred_serializable,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/suggest", methods=["POST"])
def suggest():
    try:
        payload = request.get_json(force=True)

        # Collect values (some may be missing if user only has NPK model)
        values = {
            "N": payload.get("N"),
            "P": payload.get("P"),
            "K": payload.get("K"),
            "temperature": payload.get("temperature"),
            "humidity": payload.get("humidity"),
            "ph": payload.get("ph"),
            "rainfall": payload.get("rainfall"),
            "prediction": payload.get("prediction"),
        }

        # Use Hugging Face Llama for suggestions
        text = None
        try:
            from openai import OpenAI
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                return jsonify({
                    "error": "HF_TOKEN not set in environment. In PowerShell: $env:HF_TOKEN = 'YOUR_KEY'"
                }), 400
            
            client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=hf_token,
            )

            # Ask for detailed paragraph format instead of JSON
            crop_name = values.get("prediction") or "the recommended crop"
            
            prompt = (
                "You are an agronomy assistant. Provide a detailed, well-structured paragraph explanation about crop cultivation. "
                "DO NOT output JSON format. Instead, write in clear, natural paragraphs covering all aspects.\n\n"
                f"Given these soil and climate conditions:\n"
                f"- Nitrogen (N): {values.get('N', 'N/A')} kg/ha\n"
                f"- Phosphorus (P): {values.get('P', 'N/A')} kg/ha\n"
                f"- Potassium (K): {values.get('K', 'N/A')} kg/ha\n"
                f"- Temperature: {values.get('temperature', 'N/A')}°C\n"
                f"- Humidity: {values.get('humidity', 'N/A')}%\n"
                f"- pH Level: {values.get('ph', 'N/A')}\n"
                f"- Rainfall: {values.get('rainfall', 'N/A')} mm\n"
                f"- Recommended Crop: {crop_name}\n\n"
                "Write a comprehensive paragraph explanation covering:\n"
                "1. Why this crop is recommended for these conditions\n"
                "2. How to cultivate it (soil preparation, planting, spacing)\n"
                "3. Duration/time to harvest\n"
                "4. Estimated cultivation cost\n"
                "5. Irrigation schedule and water requirements\n"
                "6. Fertilizer plan and nutrient management\n"
                "7. Pest and disease management\n"
                "8. Expected yield\n"
                "9. Market considerations\n\n"
                "Write in flowing paragraphs, not bullet points or JSON. Make it detailed and practical for farmers."
            )

            completion = client.chat.completions.create(
                model="meta-llama/Llama-3.2-3B-Instruct",
                messages=[{"role": "user", "content": prompt}],
            )
            
            text = completion.choices[0].message.content

        except ImportError:
            return jsonify({"error": "openai package not installed. Run: pip install openai"}), 500
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Invalid" in error_msg:
                return jsonify({
                    "error": "Invalid HF_TOKEN. Please check your Hugging Face API key.",
                }), 401
            elif "quota" in error_msg.lower() or "429" in error_msg:
                return jsonify({
                    "error": "Quota exceeded. Please wait a few minutes and try again.",
                }), 429
            else:
                return jsonify({
                    "error": f"Suggest error: {error_msg}",
                }), 500
        # Return paragraph format directly
        if text:
            # Clean up the text response
            suggestions_text = text.strip()
            
            # Return as paragraph format
            return jsonify({
                "suggestions": suggestions_text,
                "formatted": suggestions_text
            })
        else:
            return jsonify({"suggestions": "(No response generated)", "formatted": "(No response generated)"})
    except Exception as e:
        # Log server-side for debugging
        try:
            print("/suggest error:", repr(e))
        except Exception:
            pass
        return jsonify({
            "error": f"{type(e).__name__}: {str(e)}",
        }), 500


@app.route("/debug/models")
def debug_models():
    try:
        available = []
        try:
            available = [str(p) for p in (sorted(Path("model").glob("*.pkl")) if Path("model").exists() else [])]
        except Exception:
            available = []
        return jsonify({
            "modelLoaded": MODEL is not None,
            "modelPath": MODEL_PATH,
            "errors": MODEL_LOAD_ERRORS,
            "available": available,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/weather", methods=["GET"])
def weather():
    try:
        city = request.args.get("city", "Chennai")
        if not city:
            return jsonify({"error": "City parameter is required"}), 400

        api_key = os.getenv("WEATHERAPI_KEY")
        if not api_key:
            return jsonify({
                "error": "WEATHERAPI_KEY not set in environment. In PowerShell: $env:WEATHERAPI_KEY = 'YOUR_KEY'"
            }), 400

        # WeatherAPI.com current weather endpoint
        url = f"http://api.weatherapi.com/v1/current.json"
        params = {
            "key": api_key,
            "q": city,
            "aqi": "no"
        }

        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            error_data = response.json() if response.text else {}
            error_msg = error_data.get("error", {}).get("message", "Failed to fetch weather data")
            return jsonify({"error": error_msg}), response.status_code

        data = response.json()
        
        # Extract relevant weather data
        location_data = data.get("location", {})
        current_data = data.get("current", {})
        
        weather_result = {
            "location": location_data.get("name", city),
            "country": location_data.get("country", ""),
            "temperature": current_data.get("temp_c", 0),
            "feels_like": current_data.get("feelslike_c"),
            "humidity": current_data.get("humidity", 0),
            "pressure": current_data.get("pressure_mb", 0),
            "wind_kph": current_data.get("wind_kph", 0),
            "wind_dir": current_data.get("wind_dir", ""),
            "description": current_data.get("condition", {}).get("text", ""),
            "icon": current_data.get("condition", {}).get("icon", "")
        }

        return jsonify(weather_result)

    except requests.exceptions.Timeout:
        return jsonify({"error": "Weather API request timed out"}), 504
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Weather API request failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/fertilizer/calculate", methods=["POST"])
def fertilizer_calculate():
    try:
        payload = request.get_json(force=True)
        N = float(payload.get("N", 0))
        P = float(payload.get("P", 0))
        K = float(payload.get("K", 0))
        area = float(payload.get("area", 1))
        crop = payload.get("crop", "general")
        
        # Calculate fertilizer requirements (kg/ha to kg for area)
        n_required = max(0, (100 - N) * area / 10000)
        p_required = max(0, (50 - P) * area / 10000)
        k_required = max(0, (100 - K) * area / 10000)
        
        recommendations = []
        if n_required > 0:
            recommendations.append({"type": "Urea", "amount_kg": round(n_required * 2.17, 2), "purpose": "Nitrogen"})
        if p_required > 0:
            recommendations.append({"type": "DAP", "amount_kg": round(p_required * 2.17, 2), "purpose": "Phosphorus"})
        if k_required > 0:
            recommendations.append({"type": "Potash", "amount_kg": round(k_required * 1.67, 2), "purpose": "Potassium"})
        
        return jsonify({
            "recommendations": recommendations,
            "area_ha": round(area / 10000, 4),
            "crop": crop,
            "current_npk": {"N": N, "P": P, "K": K},
            "total_fertilizer_kg": round(sum(r["amount_kg"] for r in recommendations), 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/yield/predict", methods=["POST"])
def yield_predict():
    try:
        payload = request.get_json(force=True)
        crop = payload.get("crop", "general")
        area = float(payload.get("area", 1))
        soil_quality = payload.get("soil_quality", "medium")
        N = float(payload.get("N", 50))
        P = float(payload.get("P", 50))
        K = float(payload.get("K", 50))
        
        # Base yield per hectare (tons) based on crop and soil quality
        base_yields = {
            "rice": {"high": 6.5, "medium": 5.0, "low": 3.5},
            "wheat": {"high": 5.5, "medium": 4.0, "low": 2.5},
            "corn": {"high": 8.0, "medium": 6.0, "low": 4.0},
            "sugarcane": {"high": 100, "medium": 80, "low": 60},
            "cotton": {"high": 3.5, "medium": 2.5, "low": 1.5},
        }
        
        base = base_yields.get(crop.lower(), {"high": 5.0, "medium": 3.5, "low": 2.0}).get(soil_quality, 3.5)
        
        # Adjust based on NPK (average should be 50-100 for good yield)
        npk_avg = (N + P + K) / 3
        npk_factor = min(1.2, max(0.7, npk_avg / 75))
        
        yield_per_ha = base * npk_factor
        total_yield_tonnes = yield_per_ha * (area / 10000)
        area_hectares = round(area / 10000, 4)
        
        # Calculate confidence based on NPK factor and soil quality
        confidence = min(95, max(70, int(75 + (npk_factor - 1) * 20 + (0 if soil_quality == "medium" else (10 if soil_quality == "high" else -10)))))
        
        return jsonify({
            "crop": crop,
            "area_hectares": area_hectares,
            "yield_per_hectare": round(yield_per_ha, 2),
            "predicted_yield_tonnes": round(total_yield_tonnes, 2),
            "unit": "tons",
            "confidence": confidence,
            "factors": {
                "soil_quality": soil_quality,
                "npk_adjustment": round((npk_factor - 1) * 100, 1)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/irrigation/schedule", methods=["POST"])
def irrigation_schedule():
    try:
        payload = request.get_json(force=True)
        crop = payload.get("crop", "general")
        soil_type = payload.get("soil_type", "loam")
        rainfall = float(payload.get("rainfall", 0))
        
        # Water requirements (mm) based on crop and growth stage
        crop_requirements = {
            "rice": {"seedling": 50, "vegetative": 100, "reproductive": 150, "maturity": 50},
            "wheat": {"seedling": 30, "vegetative": 60, "reproductive": 80, "maturity": 40},
            "corn": {"seedling": 40, "vegetative": 80, "reproductive": 120, "maturity": 60},
            "cotton": {"seedling": 30, "vegetative": 70, "reproductive": 100, "maturity": 50},
        }
        
        requirements = crop_requirements.get(crop.lower(), {"seedling": 40, "vegetative": 70, "reproductive": 100, "maturity": 50})
        
        # Soil type retention factors
        soil_factors = {"clay": 1.2, "loam": 1.0, "sandy": 0.8}
        factor = soil_factors.get(soil_type, 1.0)
        
        schedule = []
        stages_obj = {}
        avg_frequency = 0
        avg_amount = 0
        
        for stage, base_mm in requirements.items():
            adjusted_mm = base_mm * factor - (rainfall / 4)  # Rainfall reduces irrigation need
            adjusted_mm = max(0, adjusted_mm)
            frequency_days = 7 if stage == "maturity" else (5 if stage == "reproductive" else (3 if stage == "vegetative" else 2))
            
            schedule.append({
                "stage": stage.capitalize(),
                "frequency_days": frequency_days,
                "water_mm": round(adjusted_mm, 1),
                "amount_mm": round(adjusted_mm, 1),
                "frequency": f"Every {frequency_days} days"
            })
            
            stages_obj[stage] = {
                "frequency_days": frequency_days,
                "amount_mm": round(adjusted_mm, 1)
            }
            
            avg_frequency += frequency_days
            avg_amount += adjusted_mm
        
        avg_frequency = round(avg_frequency / len(schedule)) if schedule else 0
        avg_amount = round(avg_amount / len(schedule), 1) if schedule else 0
        
        # Calculate rainfall impact description
        rainfall_impact = "High" if rainfall > 100 else ("Moderate" if rainfall > 50 else "Low")
        
        return jsonify({
            "crop": crop,
            "soil_type": soil_type,
            "schedule": {
                "frequency_days": avg_frequency,
                "amount_mm": avg_amount,
                "stages": stages_obj
            },
            "adjustments": {
                "soil_type": soil_type,
                "rainfall_impact": rainfall_impact
            },
            "total_water_mm": round(sum(s["water_mm"] for s in schedule), 1),
            "rainfall_adjustment": round(rainfall / 4, 1)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/calendar", methods=["POST"])
def calendar():
    try:
        payload = request.get_json(force=True)
        crop = payload.get("crop", "").strip().lower()
        
        # Planting calendar data - flattened structure for frontend compatibility
        calendars = {
            "rice": {
                "planting": ["June-July", "November-December"],
                "harvesting": ["October-November", "March-April"],
                "season": "Kharif & Rabi",
                "duration_days": 120
            },
            "wheat": {
                "planting": ["November-December"],
                "harvesting": ["March-April"],
                "season": "Rabi",
                "duration_days": 150
            },
            "corn": {
                "planting": ["June-July", "October-November"],
                "harvesting": ["September-October", "February-March"],
                "season": "Kharif & Rabi",
                "duration_days": 90
            },
            "cotton": {
                "planting": ["April-May"],
                "harvesting": ["October-December"],
                "season": "Kharif",
                "duration_days": 180
            },
            "sugarcane": {
                "planting": ["February-March"],
                "harvesting": ["December-March"],
                "season": "Annual",
                "duration_days": 365
            },
            "potato": {
                "planting": ["October-November"],
                "harvesting": ["January-February"],
                "season": "Rabi",
                "duration_days": 90
            },
            "tomato": {
                "planting": ["July-August", "November-December"],
                "harvesting": ["October-November", "February-March"],
                "season": "Kharif & Rabi",
                "duration_days": 90
            },
        }
        
        if crop and crop in calendars:
            return jsonify({"calendars": {crop: calendars[crop]}})
        elif crop:
            available = list(calendars.keys())
            return jsonify({
                "error": f"Calendar not available for '{crop}'. Available crops: {', '.join(available)}",
                "available_crops": available
            }), 400
        else:
            return jsonify({"calendars": calendars})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/market/prices", methods=["GET"])
def market_prices():
    try:
        # Mock market prices (in INR per kg, converted to per ton)
        # Price per ton = price per kg * 1000
        # Change percentage represents price change from previous period
        prices_data = {
            "rice": {"price_per_ton": 45000, "change": 0, "trend": "stable", "currency": "INR"},
            "wheat": {"price_per_ton": 28000, "change": 5.2, "trend": "up", "currency": "INR"},
            "corn": {"price_per_ton": 22000, "change": -1.5, "trend": "stable", "currency": "INR"},
            "cotton": {"price_per_ton": 85000, "change": -3.8, "trend": "down", "currency": "INR"},
            "sugarcane": {"price_per_ton": 3500, "change": 2.1, "trend": "up", "currency": "INR"},
            "potato": {"price_per_ton": 25000, "change": 0.5, "trend": "stable", "currency": "INR"},
            "tomato": {"price_per_ton": 40000, "change": 4.3, "trend": "up", "currency": "INR"},
        }
        
        return jsonify({
            "prices": prices_data,
            "last_updated": "2024-01-15",
            "currency": "INR"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    try:
        payload = request.get_json(force=True)
        message = payload.get("message", "")
        
        if not message:
            return jsonify({"error": "Message is required"}), 400
        
        # Try Hugging Face Llama
        try:
            from openai import OpenAI
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                return jsonify({
                    "error": "HF_TOKEN not set in environment. In PowerShell: $env:HF_TOKEN = 'YOUR_KEY'"
                }), 400
            
            client = OpenAI(
                base_url="https://router.huggingface.co/v1",
                api_key=hf_token,
            )
            
            completion = client.chat.completions.create(
                model="meta-llama/Llama-3.2-3B-Instruct",
                messages=[{"role": "user", "content": message}],
            )
            
            response_text = completion.choices[0].message.content
            return jsonify({"response": response_text})
            
        except ImportError:
            return jsonify({"error": "openai package not installed. Run: pip install openai"}), 500
        except Exception as e:
            error_msg = str(e)
            if "401" in error_msg or "Invalid" in error_msg:
                return jsonify({
                    "error": "Invalid HF_TOKEN. Please check your Hugging Face API key.",
                    "quota_error": False
                }), 401
            elif "quota" in error_msg.lower() or "429" in error_msg:
                return jsonify({
                    "error": "Quota exceeded. Please wait a few minutes and try again.",
                    "quota_error": True
                }), 429
            else:
                return jsonify({
                    "error": f"Chat error: {error_msg}",
                    "quota_error": False
                }), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/disease/detect", methods=["POST"])
def disease_detect():
    try:
        payload = request.get_json(force=True)
        image_base64 = payload.get("image", "")
        
        if not image_base64:
            return jsonify({"error": "Image is required"}), 400
        
        # Try to use Gemini Vision for disease detection
        if genai is None:
            return jsonify({
                "error": "google-generativeai not installed. Run: pip install google-generativeai"
            }), 500
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return jsonify({
                "error": "GEMINI_API_KEY not set in environment."
            }), 400
        
        genai.configure(api_key=api_key)
        
        try:
            from PIL import Image
            
            # Decode base64 image
            if image_base64.startswith("data:image"):
                image_base64 = image_base64.split(",")[1]
            
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            model = genai.GenerativeModel("gemini-1.5-flash")
            prompt = "Analyze this plant image and identify any diseases, pests, or health issues. Provide a detailed diagnosis."
            
            response = model.generate_content([prompt, image])
            analysis = response.text if hasattr(response, "text") else str(response)
            
            return jsonify({
                "disease": "See analysis below",
                "analysis": analysis,
                "confidence": "High" if len(analysis) > 100 else "Medium"
            })
        except ImportError:
            return jsonify({
                "error": "Pillow not installed. Run: pip install Pillow"
            }), 500
        except Exception as e:
            return jsonify({
                "error": f"Disease detection failed: {str(e)}"
            }), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# In-memory history storage (in production, use a database)
HISTORY_STORAGE = []


@app.route("/history", methods=["GET", "POST"])
def history():
    try:
        if request.method == "POST":
            # Save history entry
            payload = request.get_json(force=True)
            from datetime import datetime
            entry = {
                "id": len(HISTORY_STORAGE) + 1,
                "timestamp": payload.get("timestamp") or datetime.now().isoformat(),
                "input": payload.get("input", {}),
                "prediction": payload.get("prediction", ""),
                "confidence": payload.get("confidence", 0)
            }
            HISTORY_STORAGE.append(entry)
            return jsonify({"success": True, "id": entry["id"]})
        else:
            # Get history
            limit = int(request.args.get("limit", 50))
            history_list = HISTORY_STORAGE[-limit:] if len(HISTORY_STORAGE) > limit else HISTORY_STORAGE
            return jsonify({"history": history_list, "total": len(HISTORY_STORAGE)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/export", methods=["POST"])
def export():
    try:
        payload = request.get_json(force=True)
        format_type = request.args.get("format", "text")
        
        if format_type == "json":
            # JSON export is handled client-side
            return jsonify({"success": True})
        
        # Generate text report
        crop = payload.get("crop", payload.get("prediction", "Unknown"))
        input_data = payload.get("inputData", payload.get("input", {}))
        structured = payload.get("structured", {})
        
        report_lines = [
            "=" * 60,
            f"CROP RECOMMENDATION REPORT",
            "=" * 60,
            f"\nRecommended Crop: {crop}",
            f"\nInput Data:",
            f"  Nitrogen (N): {input_data.get('N', 'N/A')} kg/ha",
            f"  Phosphorus (P): {input_data.get('P', 'N/A')} kg/ha",
            f"  Potassium (K): {input_data.get('K', 'N/A')} kg/ha",
            f"  Temperature: {input_data.get('temperature', 'N/A')}°C",
            f"  Humidity: {input_data.get('humidity', 'N/A')}%",
            f"  pH Level: {input_data.get('ph', 'N/A')}",
            f"  Rainfall: {input_data.get('rainfall', 'N/A')} mm",
        ]
        
        if structured:
            report_lines.extend([
                f"\nCultivation Details:",
                f"  Reason: {structured.get('reason', 'N/A')}",
                f"  Duration: {structured.get('duration_weeks', 'N/A')} weeks",
                f"  Estimated Cost: {structured.get('estimated_cost', {}).get('amount', 'N/A')} {structured.get('estimated_cost', {}).get('currency', '')}",
                f"\nCultivation Steps:",
            ])
            for i, step in enumerate(structured.get("cultivation_steps", []), 1):
                report_lines.append(f"  {i}. {step}")
        
        report_lines.append("\n" + "=" * 60)
        report_text = "\n".join(report_lines)
        
        from flask import Response
        return Response(
            report_text,
            mimetype="text/plain",
            headers={"Content-Disposition": "attachment; filename=crop_report.txt"}
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/thingspeak/latest", methods=["GET"])
def thingspeak_latest():
    try:
        channel_id = request.args.get("channelId")
        api_key = request.args.get("apiKey")
        
        if not channel_id or not api_key:
            return jsonify({"error": "channelId and apiKey are required"}), 400
        
        # Fetch latest data from ThingSpeak
        url = f"https://api.thingspeak.com/channels/{channel_id}/feeds/last.json"
        params = {"api_key": api_key}
        
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code != 200:
            return jsonify({"error": f"ThingSpeak API error: {response.status_code}"}), response.status_code
        
        data = response.json()
        feed = data.get("feed", {})
        
        # Map ThingSpeak field names to our format
        result = {
            "N": feed.get("field1") or feed.get("field2") or feed.get("field3"),
            "P": feed.get("field2") or feed.get("field4") or feed.get("field5"),
            "K": feed.get("field3") or feed.get("field6") or feed.get("field7"),
            "temperature": feed.get("field4") or feed.get("field1"),
            "humidity": feed.get("field5") or feed.get("field2"),
            "ph": feed.get("field6") or feed.get("field3"),
            "rainfall": feed.get("field7") or feed.get("field4"),
        }
        
        # Try to extract numeric values
        for key in result:
            if result[key] is not None:
                try:
                    result[key] = float(result[key])
                except (ValueError, TypeError):
                    result[key] = None
        
        return jsonify(result)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"ThingSpeak request failed: {str(e)}"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/voice", methods=["POST"])
def voice():
    """Twilio IVR - Main voice menu"""
    if not twilio_available:
        return "Twilio not installed. Run: pip install twilio", 500
    
    resp = VoiceResponse()
    gather = Gather(num_digits=1, action="/menu", method="POST", timeout=10)
    gather.say(
        "Welcome to Smart Agriculture Service. "
        "Press 1 for crop advice. "
        "Press 2 for weather information. "
        "Press 3 for fertilizer recommendations. "
        "Press 4 for irrigation schedule. "
        "Press 5 to hear this menu again."
    )
    resp.append(gather)
    resp.redirect("/voice")
    return str(resp), 200, {"Content-Type": "text/xml"}


@app.route("/menu", methods=["POST"])
def menu():
    """Twilio IVR - Handle menu selection"""
    if not twilio_available:
        return "Twilio not installed. Run: pip install twilio", 500
    
    digit = request.form.get('Digits', '')
    resp = VoiceResponse()

    if digit == "1":
        # Crop advice - get from model or default recommendation
        if MODEL is not None:
            # Use default values for voice response
            default_input = {
                "N": 50, "P": 50, "K": 50,
                "temperature": 25, "humidity": 60,
                "ph": 6.5, "rainfall": 150
            }
            try:
                # Try to get prediction (simplified for voice)
                X = np.array([[default_input["N"], default_input["P"], default_input["K"]]], dtype=float)
                if hasattr(MODEL, "n_features_in_") and MODEL.n_features_in_ == 7:
                    X = np.array([[
                        default_input["N"], default_input["P"], default_input["K"],
                        default_input["temperature"], default_input["humidity"],
                        default_input["ph"], default_input["rainfall"]
                    ]], dtype=float)
                y_pred = MODEL.predict(X)
                crop = str(y_pred[0])
                resp.say(f"Based on soil conditions, the recommended crop is {crop}. "
                        f"For detailed cultivation advice, please visit our web portal.")
            except Exception:
                resp.say("Recommended crop is Rice for this season. "
                        "For detailed cultivation advice, please visit our web portal.")
        else:
            resp.say("Recommended crop is Rice for this season. "
                    "For detailed cultivation advice, please visit our web portal.")
    
    elif digit == "2":
        # Weather information
        try:
            # Try to get weather from API
            api_key = os.getenv("WEATHERAPI_KEY")
            if api_key:
                city = "Chennai"  # Default city
                url = f"http://api.weatherapi.com/v1/current.json"
                params = {"key": api_key, "q": city, "aqi": "no"}
                response = requests.get(url, params=params, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    current = data.get("current", {})
                    temp = current.get("temp_c", 25)
                    condition = current.get("condition", {}).get("text", "sunny")
                    resp.say(f"Weather today in {city} is {condition} with temperature {int(temp)} degrees Celsius. "
                            f"Have a great farming day!")
                else:
                    resp.say("Weather today is sunny with slight rain chances. "
                            "For detailed weather updates, please visit our web portal.")
            else:
                resp.say("Weather today is sunny with slight rain chances. "
                        "For detailed weather updates, please visit our web portal.")
        except Exception:
            resp.say("Weather today is sunny with slight rain chances. "
                    "For detailed weather updates, please visit our web portal.")
    
    elif digit == "3":
        # Fertilizer recommendations
        resp.say("For fertilizer recommendations, please provide your soil NPK values. "
                "Visit our web portal or contact our agricultural expert for detailed advice. "
                "Thank you for calling Smart Agriculture Service.")
    
    elif digit == "4":
        # Irrigation schedule
        resp.say("For irrigation scheduling, please provide your crop type, soil type, and rainfall data. "
                "Visit our web portal for personalized irrigation recommendations. "
                "Thank you for calling Smart Agriculture Service.")
    
    elif digit == "5":
        # Return to main menu
        resp.redirect("/voice")
    
    else:
        resp.say("Invalid option. Please try again.")
        resp.redirect("/voice")

    return str(resp), 200, {"Content-Type": "text/xml"}


if __name__ == "__main__":
    # Suitable for local development
    app.run(host="0.0.0.0", port=5000, debug=True)





