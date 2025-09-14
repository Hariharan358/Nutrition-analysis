from flask import Flask, request, jsonify
from pathlib import Path
import numpy as np
import pickle
import os
import json
from flask_cors import CORS

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
        if genai is None:
            return jsonify({
                "error": "google-generativeai not installed. Run: pip install google-generativeai",
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

        # Configure Gemini
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return jsonify({
                "error": "GEMINI_API_KEY not set in environment. In PowerShell: $env:GEMINI_API_KEY = 'YOUR_KEY'",
            }), 400

        genai.configure(api_key=api_key)

        # Try preferred model, fallback if unavailable
        model_name_candidates = ["gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-pro"]
        model = None
        last_error = None
        for name in model_name_candidates:
            try:
                model = genai.GenerativeModel(name)
                break
            except Exception as e:
                last_error = str(e)
                continue
        if model is None:
            return jsonify({
                "error": f"Failed to initialize Gemini model. Details: {last_error}",
            }), 500

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

        prompt = (
            "You are an agronomy assistant. Output STRICT JSON only (no prose). "
            "Given soil N-P-K, climate, pH, rainfall, optional predicted crop, return: "
            "crop, reason, cultivation_steps (3-8 steps), duration_weeks, "
            "estimated_cost {amount,currency}, soil_preparation {ph_adjustment,bed_preparation,organic_matter}, "
            "spacing {row_spacing_cm,plant_spacing_cm}, irrigation_schedule [{stage,frequency_days,amount_mm}], "
            "fertilizer_plan [{time,type,amount}], pest_management [{pest,monitoring,control}], expected_yield {amount,unit}, market_notes.\n\n"
            f"Inputs: {values}\n\n"
            f"Respond ONLY as minified JSON matching these keys and types: {json.dumps(schema_example)}"
        )

        response = model.generate_content(prompt)

        # Check for safety blocks or prompt feedback
        feedback = getattr(response, "prompt_feedback", None)
        if feedback and getattr(feedback, "block_reason", None):
            return jsonify({
                "error": f"Response blocked: {feedback.block_reason}",
            }), 400

        # Extract text robustly
        text = getattr(response, "text", None)
        if not text:
            # Attempt to assemble from candidates
            candidates = getattr(response, "candidates", None) or []
            collected = []
            try:
                for cand in candidates:
                    parts = getattr(cand, "content", None)
                    parts = getattr(parts, "parts", []) if parts else []
                    for p in parts:
                        if hasattr(p, "text") and p.text:
                            collected.append(p.text)
                text = "\n\n".join(collected) if collected else None
            except Exception:
                text = None

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


if __name__ == "__main__":
    # Suitable for local development
    app.run(host="0.0.0.0", port=5000, debug=True)





