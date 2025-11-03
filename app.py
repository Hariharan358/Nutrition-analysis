from flask import Flask, request, jsonify
from pathlib import Path
import numpy as np
import pickle
import os
import json
from flask_cors import CORS
import requests
import time
from datetime import datetime, timedelta
import base64
import io
try:
    from openai import OpenAI
    openai_available = True
except ImportError:
    openai_available = False
    OpenAI = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import google.generativeai as genai
except Exception:
    genai = None


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes (adjust origins in production)


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
        # Use Hugging Face Llama for crop suggestions
        if not openai_available or OpenAI is None:
            return jsonify({
                "error": "openai package not installed. Run: pip install openai",
            }), 500

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

        # Configure Hugging Face
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            return jsonify({
                "error": "HF_TOKEN not set in environment. In PowerShell: $env:HF_TOKEN = 'YOUR_TOKEN'",
            }), 400

        # Initialize Hugging Face OpenAI-compatible client
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token,
        )

        # Ask strictly for JSON to enable structured UI rendering
        schema_example = {
            "crop": values.get("prediction") or "",
            "reason": "",
            "cultivation_steps": ["", ""],
            "duration_weeks": 0,
            "estimated_cost": {"amount": 0, "currency": "INR"},
            "soil_preparation": {
                "ph_adjustment": "",
                "bed_preparation": "",
                "organic_matter": ""
            },
            "spacing": {
                "row_spacing_cm": 0,
                "plant_spacing_cm": 0
            },
            "irrigation_schedule": [
                {"stage": "", "frequency_days": 0, "amount_mm": 0}
            ],
            "fertilizer_plan": [
                {"time": "", "type": "", "amount": ""}
            ],
            "pest_management": [
                {"pest": "", "monitoring": "", "control": ""}
            ],
            "expected_yield": {"amount": 0, "unit": "ton/ha"},
            "market_notes": ""
        }

        # Build system prompt for structured JSON output
        system_prompt = (
            "You are an expert agronomy assistant. You MUST respond ONLY with valid JSON. "
            "Do not include any explanatory text, markdown formatting, or code blocks - just raw JSON. "
            "Given soil N-P-K levels, climate data (temperature, humidity), pH, rainfall, and an optional predicted crop, "
            "provide detailed agricultural recommendations in JSON format."
        )

        user_prompt = (
            f"Given these inputs: {values}, provide a JSON response with these exact keys: "
            f"crop, reason, cultivation_steps (array of 3-8 steps), duration_weeks (number), "
            f"estimated_cost (object with amount and currency), soil_preparation (object with ph_adjustment, bed_preparation, organic_matter), "
            f"spacing (object with row_spacing_cm and plant_spacing_cm), irrigation_schedule (array of objects with stage, frequency_days, amount_mm), "
            f"fertilizer_plan (array of objects with time, type, amount), pest_management (array of objects with pest, monitoring, control), "
            f"expected_yield (object with amount and unit), market_notes (string).\n\n"
            f"Example structure: {json.dumps(schema_example)}\n\n"
            "Respond with ONLY the JSON object, no other text."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Retry with exponential backoff for quota errors
        max_retries = 3
        retry_delay = 5
        text = None
        last_error = None
        
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model="meta-llama/Llama-3.2-3B-Instruct",
                    messages=messages,
                    temperature=0.3,  # Lower temperature for more structured output
                    max_tokens=1500,
                )
                
                # Extract response text
                if completion.choices and len(completion.choices) > 0:
                    text = completion.choices[0].message.content
                    break
            except Exception as e:
                error_str = str(e)
                last_error = e
                # Check if it's a quota/rate limit error
                if "429" in error_str or "rate limit" in error_str.lower() or "quota" in error_str.lower():
                    retry_seconds = retry_delay * (2 ** attempt)
                    if attempt < max_retries - 1:
                        # Wait and retry
                        time.sleep(retry_seconds)
                        continue
                    else:
                        # Final attempt failed, return quota error
                        return jsonify({
                            "error": f"Quota/rate limit exceeded after {max_retries} attempts. Please wait a few minutes and try again.",
                            "quota_error": True,
                            "retry_after": retry_seconds
                        }), 429
                # Handle authentication errors
                elif "401" in error_str or "unauthorized" in error_str.lower() or "api key" in error_str.lower():
                    return jsonify({
                        "error": "Invalid HF_TOKEN. Please check your Hugging Face token.",
                    }), 401
                else:
                    # Non-quota error - return immediately (don't retry)
                    error_type = type(e).__name__
                    return jsonify({
                        "error": f"{error_type}: {error_str}",
                        "error_type": error_type
                    }), 500
        
        if text is None:
            if last_error:
                return jsonify({
                    "error": f"Failed to generate content: {str(last_error)}",
                }), 500
            return jsonify({
                "error": "Failed to generate content (unknown error)",
            }), 500

        # Clean the text - remove markdown code blocks if present
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

        structured = None
        if text:
            try:
                structured = json.loads(text)
            except Exception:
                structured = None

        # Return structured JSON if available; otherwise pass through text
        if structured:
            # Build a human-friendly formatted summary
            crop = structured.get("crop") or "-"
            reason = structured.get("reason") or "-"
            steps = structured.get("cultivation_steps") or []
            duration = structured.get("duration_weeks")
            cost = structured.get("estimated_cost") or {}
            amount = cost.get("amount")
            currency = cost.get("currency") or ""

            # Create a bullet-list style text block
            steps_lines = "\n".join([f"  {i+1}. {s}" for i, s in enumerate(steps)]) if steps else "  -"
            duration_text = f"{duration} weeks" if isinstance(duration, (int, float)) else "-"
            cost_text = (f"{amount} {currency}".strip()) if amount is not None else "-"

            # Extended formatted block with details
            soil = structured.get("soil_preparation") or {}
            spacing = structured.get("spacing") or {}
            irr = structured.get("irrigation_schedule") or []
            fert = structured.get("fertilizer_plan") or []
            pests = structured.get("pest_management") or []
            yield_obj = structured.get("expected_yield") or {}
            yield_text = f"{yield_obj.get('amount', '-') } {yield_obj.get('unit', '')}".strip()

            irr_lines = "\n".join([f"  - {i.get('stage','')}: every {i.get('frequency_days','-')} days, {i.get('amount_mm','-')} mm" for i in irr]) or "  -"
            fert_lines = "\n".join([f"  - {f.get('time','')}: {f.get('type','')} — {f.get('amount','')}" for f in fert]) or "  -"
            pest_lines = "\n".join([f"  - {p.get('pest','')}: monitor {p.get('monitoring','')}; control {p.get('control','')}" for p in pests]) or "  -"

            formatted = (
                f"- Crop: {crop}\n"
                f"- Why: {reason}\n"
                f"- How to cultivate:\n{steps_lines}\n"
                f"- Duration: {duration_text}\n"
                f"- Estimated cost: {cost_text}\n"
                f"- Soil preparation: pH — {soil.get('ph_adjustment','-')}; Bed — {soil.get('bed_preparation','-')}; OM — {soil.get('organic_matter','-')}\n"
                f"- Spacing: row {spacing.get('row_spacing_cm','-')} cm, plant {spacing.get('plant_spacing_cm','-')} cm\n"
                f"- Irrigation schedule:\n{irr_lines}\n"
                f"- Fertilizer plan:\n{fert_lines}\n"
                f"- Pest management:\n{pest_lines}\n"
                f"- Expected yield: {yield_text}\n"
                f"- Market notes: {structured.get('market_notes','-')}"
            )

            return jsonify({"structured": structured, "formatted": formatted})
        else:
            return jsonify({"suggestions": text or "(No text response)"})
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


@app.route("/thingspeak/latest", methods=["GET"])
def thingspeak_latest():
    try:
        # Expect query params: channelId, apiKey; optional: results (default 1)
        channel_id = request.args.get("channelId")
        api_key = request.args.get("apiKey")
        results = request.args.get("results", default="1")
        if not channel_id or not api_key:
            return jsonify({"error": "Missing channelId or apiKey"}), 400

        url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json"
        resp = requests.get(url, params={"api_key": api_key, "results": results}, timeout=10)
        if resp.status_code != 200:
            return jsonify({"error": f"ThingSpeak error {resp.status_code}", "body": resp.text}), 502
        data = resp.json()

        # Map ThingSpeak fields to our inputs (customize as needed)
        # Example: field1=N, field2=P, field3=K, field4=temperature, field5=humidity, field6=ph, field7=rainfall
        feeds = data.get("feeds", [])
        if not feeds:
            return jsonify({"error": "No feeds returned"}), 404
        latest = feeds[-1]

        mapped = {
            "N": _safe_float(latest.get("field1")),
            "P": _safe_float(latest.get("field2")),
            "K": _safe_float(latest.get("field3")),
            "temperature": _safe_float(latest.get("field4")),
            "humidity": _safe_float(latest.get("field5")),
            "ph": _safe_float(latest.get("field6")),
            "rainfall": _safe_float(latest.get("field7")),
            "raw": latest,
        }

        return jsonify(mapped)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _safe_float(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


# In-memory history store (use database in production)
PREDICTION_HISTORY = []


@app.route("/history", methods=["GET", "POST"])
def history():
    try:
        if request.method == "POST":
            # Save prediction to history
            data = request.get_json(force=True)
            entry = {
                "id": len(PREDICTION_HISTORY) + 1,
                "timestamp": datetime.now().isoformat(),
                "input": data.get("input"),
                "prediction": data.get("prediction"),
                "confidence": data.get("confidence", 95),
            }
            PREDICTION_HISTORY.append(entry)
            return jsonify({"success": True, "id": entry["id"]})
        else:
            # Get history
            limit = int(request.args.get("limit", 50))
            return jsonify({"history": PREDICTION_HISTORY[-limit:]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/weather", methods=["GET"])
def weather():
    try:
        # Get weather from WeatherAPI.com
        api_key = os.getenv("WEATHERAPI_KEY") or os.getenv("WEATHER_API_KEY") or request.args.get("apiKey")
        lat = request.args.get("lat")
        lon = request.args.get("lon")
        city = request.args.get("city")
        
        if not api_key:
            return jsonify({"error": "WeatherAPI.com API key required. Set WEATHERAPI_KEY or WEATHER_API_KEY or pass apiKey param"}), 400
        
        url = "https://api.weatherapi.com/v1/current.json"
        params = {"key": api_key}
        
        # WeatherAPI.com accepts city name, coordinates, or IP
        if lat and lon:
            params["q"] = f"{lat},{lon}"
        elif city:
            params["q"] = city
        else:
            return jsonify({"error": "Need lat/lon or city name"}), 400
        
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code != 200:
            return jsonify({"error": f"Weather API error {resp.status_code}", "body": resp.text}), 502
        
        data = resp.json()
        current = data.get("current", {})
        location_data = data.get("location", {})
        condition = current.get("condition", {})
        
        return jsonify({
            "temperature": current.get("temp_c"),  # Temperature in Celsius
            "humidity": current.get("humidity"),
            "pressure": current.get("pressure_mb"),  # Pressure in millibars
            "description": condition.get("text"),
            "location": location_data.get("name"),
            "country": location_data.get("country"),
            "icon": condition.get("icon"),  # Weather icon URL from WeatherAPI.com
            "wind_kph": current.get("wind_kph"),
            "wind_dir": current.get("wind_dir"),
            "feels_like": current.get("feelslike_c"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/compare", methods=["POST"])
def compare_crops():
    try:
        # Get top 3 crop recommendations by trying different combinations
        payload = request.get_json(force=True)
        
        if MODEL is None:
            return jsonify({
                "error": "Model not loaded",
                "alternatives": []
            }), 400
        
        # Get primary prediction
        primary_input = {
            "N": float(payload.get("N", 0)),
            "P": float(payload.get("P", 0)),
            "K": float(payload.get("K", 0)),
            "temperature": float(payload.get("temperature", 0)),
            "humidity": float(payload.get("humidity", 0)),
            "ph": float(payload.get("ph", 0)),
            "rainfall": float(payload.get("rainfall", 0)),
        }
        
        # For comparison, we'll return the primary prediction
        # In a real implementation, you'd query model probabilities or try variations
        expected_features = getattr(MODEL, "n_features_in_", None) or 7
        feature_order = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"] if expected_features == 7 else ["N", "P", "K"]
        
        features = [primary_input.get(k, 0) for k in feature_order]
        X = np.array([features], dtype=float)
        
        try:
            pred = MODEL.predict(X)[0]
            pred_str = str(pred)
        except Exception:
            pred_str = "unknown"
        
        # Mock alternatives (in production, use model probabilities)
        alternatives = [
            {"crop": pred_str, "confidence": 95, "reason": "Best match for your soil conditions"},
            {"crop": "rice", "confidence": 80, "reason": "Alternative option with good yield potential"},
            {"crop": "wheat", "confidence": 75, "reason": "Suitable for your climate zone"},
        ]
        
        return jsonify({
            "primary": {"crop": pred_str, "confidence": 95},
            "alternatives": alternatives[:3]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/export", methods=["POST"])
def export_data():
    try:
        # Export suggestions/predictions as text or JSON
        data = request.get_json(force=True)
        format_type = request.args.get("format", "text")
        
        if format_type == "json":
            return jsonify(data), 200, {"Content-Disposition": "attachment; filename=crop_data.json"}
        
        # Text format
        lines = []
        if data.get("crop"):
            lines.append(f"Recommended Crop: {data['crop']}")
        if data.get("prediction"):
            lines.append(f"Prediction: {data['prediction']}")
        if data.get("suggestions"):
            lines.append("\nSuggestions:")
            lines.append(data["suggestions"])
        if data.get("structured"):
            s = data["structured"]
            lines.append(f"\nCrop: {s.get('crop', '-')}")
            lines.append(f"Why: {s.get('reason', '-')}")
            lines.append("\nCultivation Steps:")
            for i, step in enumerate(s.get("cultivation_steps", []), 1):
                lines.append(f"{i}. {step}")
            lines.append(f"\nDuration: {s.get('duration_weeks', '-')} weeks")
            lines.append(f"Cost: {s.get('estimated_cost', {}).get('amount', '-')} {s.get('estimated_cost', {}).get('currency', '')}")
        
        text_content = "\n".join(lines)
        return text_content, 200, {"Content-Type": "text/plain", "Content-Disposition": "attachment; filename=crop_report.txt"}
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/disease/detect", methods=["POST"])
def detect_disease():
    try:
        # Disease detection using Gemini Vision
        if genai is None:
            return jsonify({"error": "google-generativeai not installed"}), 500
        
        if Image is None:
            return jsonify({"error": "Pillow not installed. Run: pip install Pillow"}), 500
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return jsonify({"error": "GEMINI_API_KEY not set"}), 400
        
        genai.configure(api_key=api_key)
        
        # Get image from request
        data = request.get_json(force=True)
        image_data = data.get("image")
        
        if not image_data:
            return jsonify({"error": "No image provided"}), 400
        
        # Decode base64 image
        try:
            if image_data.startswith("data:image"):
                image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            img = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            return jsonify({"error": f"Invalid image: {str(e)}"}), 400
        
        # Use Gemini Vision for disease detection
        model = genai.GenerativeModel("gemini-pro-vision")
        
        prompt = (
            "Analyze this crop/plant image and identify any diseases, pests, or health issues. "
            "Provide: 1) Disease/pest name (if any), 2) Confidence level (0-100%), "
            "3) Symptoms observed, 4) Recommended treatment, 5) Prevention tips. "
            "If the plant appears healthy, say so. Return as JSON: "
            '{"disease": "", "confidence": 0, "symptoms": [], "treatment": "", "prevention": "", "severity": ""}'
        )
        
        response = model.generate_content([prompt, img])
        text = getattr(response, "text", None)
        
        if not text:
            return jsonify({"error": "No response from AI"}), 500
        
        try:
            result = json.loads(text)
        except:
            result = {"analysis": text, "raw": True}
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/market/prices", methods=["GET"])
def market_prices():
    try:
        # Fetch crop prices (mock data for now - integrate with real API)
        crop = request.args.get("crop", "").lower()
        
        # Mock price data - replace with real API (e.g., USDA, commodity exchanges)
        price_data = {
            "rice": {"price_per_ton": 450, "currency": "USD", "trend": "up", "change": 5.2},
            "wheat": {"price_per_ton": 380, "currency": "USD", "trend": "down", "change": -2.1},
            "corn": {"price_per_ton": 320, "currency": "USD", "trend": "stable", "change": 0.5},
            "cotton": {"price_per_ton": 1800, "currency": "USD", "trend": "up", "change": 3.8},
            "sugarcane": {"price_per_ton": 280, "currency": "USD", "trend": "stable", "change": 1.2},
            "soybean": {"price_per_ton": 520, "currency": "USD", "trend": "up", "change": 4.5},
        }
        
        if crop and crop in price_data:
            return jsonify({crop: price_data[crop]})
        
        return jsonify({"prices": price_data, "last_updated": datetime.now().isoformat()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/calendar", methods=["POST"])
def planting_calendar():
    try:
        # Generate seasonal planting calendar
        payload = request.get_json(force=True)
        location = payload.get("location", "")
        crop = payload.get("crop", "").lower()
        
        # Mock calendar data - replace with real seasonal data
        calendars = {
            "rice": {
                "planting": ["March-April", "June-July"],
                "harvesting": ["August-September", "November-December"],
                "season": "Kharif, Rabi",
                "duration_days": 120
            },
            "wheat": {
                "planting": ["October-November"],
                "harvesting": ["March-April"],
                "season": "Rabi",
                "duration_days": 150
            },
            "corn": {
                "planting": ["May-June"],
                "harvesting": ["September-October"],
                "season": "Kharif",
                "duration_days": 90
            },
            "cotton": {
                "planting": ["May-June"],
                "harvesting": ["October-November"],
                "season": "Kharif",
                "duration_days": 180
            },
            "sugarcane": {
                "planting": ["February-March", "September-October"],
                "harvesting": ["December-January", "April-May"],
                "season": "Year-round",
                "duration_days": 365
            },
            "soybean": {
                "planting": ["June-July"],
                "harvesting": ["October-November"],
                "season": "Kharif",
                "duration_days": 100
            },
            "potato": {
                "planting": ["October-November"],
                "harvesting": ["January-February"],
                "season": "Rabi",
                "duration_days": 90
            },
        }
        
        if crop:
            crop_lower = crop.lower().strip()
            if crop_lower in calendars:
                return jsonify({crop_lower: calendars[crop_lower]})
            else:
                return jsonify({
                    "error": f"Crop '{crop}' not found. Available crops: {', '.join(calendars.keys())}",
                    "available_crops": list(calendars.keys())
                }), 404
        
        return jsonify({"calendars": calendars})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/fertilizer/calculate", methods=["POST"])
def calculate_fertilizer():
    try:
        # Fertilizer calculator based on soil NPK and target crop
        payload = request.get_json(force=True)
        current_n = float(payload.get("N", 0))
        current_p = float(payload.get("P", 0))
        current_k = float(payload.get("K", 0))
        area_hectares = float(payload.get("area", 1))
        crop = payload.get("crop", "").lower()
        
        # Target NPK levels by crop (kg/ha)
        targets = {
            "rice": {"N": 120, "P": 60, "K": 60},
            "wheat": {"N": 100, "P": 50, "K": 50},
            "corn": {"N": 150, "P": 70, "K": 70},
            "cotton": {"N": 80, "P": 40, "K": 40},
        }
        
        target = targets.get(crop, {"N": 100, "P": 50, "K": 50})
        
        # Calculate deficit
        deficit_n = max(0, target["N"] - current_n)
        deficit_p = max(0, target["P"] - current_p)
        deficit_k = max(0, target["K"] - current_k)
        
        # Fertilizer composition (typical)
        urea_n = 46  # %
        dap_n = 18   # %
        dap_p = 46   # %
        muriate_potash_k = 60  # %
        
        # Calculate requirements
        urea_needed = (deficit_n / urea_n) * 100 * area_hectares
        dap_needed = max((deficit_p / dap_p) * 100, (deficit_n / dap_n) * 100) * area_hectares
        potash_needed = (deficit_k / muriate_potash_k) * 100 * area_hectares
        
        return jsonify({
            "requirements_kg": {
                "urea": round(urea_needed, 2),
                "dap": round(dap_needed, 2),
                "muriate_potash": round(potash_needed, 2),
            },
            "target_npk": target,
            "current_npk": {"N": current_n, "P": current_p, "K": current_k},
            "deficit_npk": {"N": round(deficit_n, 2), "P": round(deficit_p, 2), "K": round(deficit_k, 2)},
            "area_hectares": area_hectares
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/yield/predict", methods=["POST"])
def predict_yield():
    try:
        # Yield prediction based on inputs
        payload = request.get_json(force=True)
        crop = payload.get("crop", "").lower()
        area = float(payload.get("area", 1))  # hectares
        soil_quality = payload.get("soil_quality", "medium")  # low, medium, high
        
        # Base yields (tonnes/hectare)
        base_yields = {
            "rice": {"low": 3, "medium": 5, "high": 7},
            "wheat": {"low": 2.5, "medium": 4, "high": 6},
            "corn": {"low": 4, "medium": 6, "high": 9},
            "cotton": {"low": 1.5, "medium": 2.5, "high": 4},
        }
        
        base = base_yields.get(crop, {"low": 3, "medium": 5, "high": 7})
        yield_per_ha = base.get(soil_quality, base["medium"])
        
        # Adjustments based on NPK
        n_adjust = min(float(payload.get("N", 50)) / 100, 1.2)
        p_adjust = min(float(payload.get("P", 50)) / 100, 1.2)
        k_adjust = min(float(payload.get("K", 50)) / 100, 1.2)
        
        adjustment = (n_adjust + p_adjust + k_adjust) / 3
        predicted_yield = yield_per_ha * adjustment * area
        
        return jsonify({
            "predicted_yield_tonnes": round(predicted_yield, 2),
            "yield_per_hectare": round(yield_per_ha * adjustment, 2),
            "area_hectares": area,
            "confidence": 85,
            "factors": {
                "soil_quality": soil_quality,
                "npk_adjustment": round(adjustment * 100, 1),
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/irrigation/schedule", methods=["POST"])
def irrigation_schedule():
    try:
        # Generate irrigation schedule
        payload = request.get_json(force=True)
        crop = payload.get("crop", "").lower()
        soil_type = payload.get("soil_type", "loam")  # sandy, loam, clay
        rainfall = float(payload.get("rainfall", 0))
        
        # Irrigation schedules (frequency in days, amount in mm)
        schedules = {
            "rice": {"frequency": 3, "amount": 50, "stage_adjustments": {
                "seedling": {"frequency": 2, "amount": 30},
                "vegetative": {"frequency": 3, "amount": 50},
                "flowering": {"frequency": 2, "amount": 60},
                "maturity": {"frequency": 4, "amount": 40}
            }},
            "wheat": {"frequency": 5, "amount": 40, "stage_adjustments": {
                "tillering": {"frequency": 5, "amount": 40},
                "jointing": {"frequency": 4, "amount": 50},
                "heading": {"frequency": 3, "amount": 40},
            }},
            "corn": {"frequency": 4, "amount": 45, "stage_adjustments": {
                "vegetative": {"frequency": 4, "amount": 45},
                "tasseling": {"frequency": 3, "amount": 50},
                "grain_fill": {"frequency": 5, "amount": 40},
            }},
        }
        
        base_schedule = schedules.get(crop, {"frequency": 4, "amount": 40, "stage_adjustments": {}})
        
        # Adjust for rainfall
        if rainfall > 100:
            base_schedule["frequency"] += 1  # Less irrigation needed
        
        # Adjust for soil type
        soil_multipliers = {"sandy": 0.7, "loam": 1.0, "clay": 1.3}
        amount_multiplier = soil_multipliers.get(soil_type, 1.0)
        
        return jsonify({
            "schedule": {
                "frequency_days": base_schedule["frequency"],
                "amount_mm": round(base_schedule["amount"] * amount_multiplier, 1),
                "stages": base_schedule.get("stage_adjustments", {}),
            },
            "adjustments": {
                "soil_type": soil_type,
                "rainfall_impact": "reduced" if rainfall > 100 else "normal",
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    try:
        # AI chatbot for agricultural queries using Hugging Face Llama
        if not openai_available or OpenAI is None:
            return jsonify({"error": "openai package not installed. Run: pip install openai"}), 500
        
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            return jsonify({"error": "HF_TOKEN not set. Set it in your environment: $env:HF_TOKEN='YOUR_TOKEN'"}), 400
        
        # Initialize Hugging Face OpenAI-compatible client
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=hf_token,
        )
        
        payload = request.get_json(force=True)
        message = payload.get("message", "")
        context = payload.get("context", {})
        
        if not message:
            return jsonify({"error": "No message provided"}), 400
        
        # Build context-aware system prompt
        context_str = f"\nContext: {json.dumps(context)}" if context else ""
        system_prompt = (
            "You are an expert agricultural assistant. Answer questions helpfully and concisely. "
            "If the user asks about crops, soil, farming techniques, or agricultural problems, provide accurate, practical advice."
        )
        
        # Build messages array for chat completion
        messages = [
            {"role": "system", "content": system_prompt + context_str},
            {"role": "user", "content": message}
        ]
        
        # Generate with error handling
        try:
            completion = client.chat.completions.create(
                model="meta-llama/Llama-3.2-3B-Instruct",
                messages=messages,
                temperature=0.7,
                max_tokens=500,
            )
            
            # Extract response text
            if completion.choices and len(completion.choices) > 0:
                text = completion.choices[0].message.content
            else:
                text = None
            
            if not text:
                text = "I'm sorry, I couldn't generate a response at this time."
        except Exception as e:
            error_str = str(e)
            # Handle rate limiting
            if "429" in error_str or "rate limit" in error_str.lower() or "quota" in error_str.lower():
                return jsonify({
                    "error": "API quota/rate limit exceeded. Please wait a few minutes and try again.",
                    "quota_error": True,
                }), 429
            # Handle authentication errors
            if "401" in error_str or "unauthorized" in error_str.lower() or "api key" in error_str.lower():
                return jsonify({
                    "error": "Invalid HF_TOKEN. Please check your Hugging Face token.",
                }), 401
            return jsonify({
                "error": f"Failed to generate response: {error_str}",
            }), 500
        
        return jsonify({
            "response": text,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        # Log server-side for debugging
        try:
            print("/chat error:", repr(e))
        except Exception:
            pass
        return jsonify({
            "error": f"{type(e).__name__}: {str(e)}",
        }), 500


@app.route("/translate", methods=["POST"])
def translate():
    try:
        # Optional: Programmatic translation endpoint
        # For Google Cloud Translation API, you'd need: pip install google-cloud-translate
        # For now, returns a note about using the widget or setting up the API
        
        payload = request.get_json(force=True)
        text = payload.get("text", "")
        target_lang = payload.get("target", "hi")  # Default to Hindi
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        # Simple note - in production, integrate Google Cloud Translation API here
        return jsonify({
            "message": "Translation API not configured. Use the Google Translate widget in the UI.",
            "note": "To enable API translation, install: pip install google-cloud-translate and configure credentials",
            "text": text,
            "target": target_lang,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Suitable for local development
    app.run(host="0.0.0.0", port=5000, debug=True)





