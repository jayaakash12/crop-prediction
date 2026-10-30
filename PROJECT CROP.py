"""
app.py — Field Ledger backend
------------------------------
Flask API that serves the trained crop-recommendation model to the
dashboard over the network (default http://localhost:8000).

Endpoints (all match dashboard__1_.html exactly):
  GET  /api/health
  POST /api/predict
  POST /api/predict/top
  GET  /api/crops
  GET  /api/predictions/history?limit=50
  GET  /api/models/performance
  GET  /api/models/feature-importance
  GET  /api/models/current

Run:
  pip install -r requirements.txt --break-system-packages
  python train_model.py      # trains the model from data/Crop_recommendation.csv
  python app.py               # serves on :8000
"""
import json
import os
import sqlite3
import time
from datetime import datetime, timezone

import joblib
import numpy as np
from flask import Flask, request, jsonify

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
DB_PATH = os.path.join(HERE, "predictions.db")
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

ENABLE_DB = os.environ.get("ENABLE_DB", "true").lower() != "false"

# Short human-readable blurbs shown under a prediction ("explanation")
CROP_NOTES = {
    "rice": "Rice thrives in warm, humid conditions with standing water and high rainfall.",
    "maize": "Maize prefers moderate rainfall, warm days, and well-drained loamy soil.",
    "chickpea": "Chickpea is a cool-season legume that tolerates low rainfall and alkaline soil.",
    "kidneybeans": "Kidney beans favor mild temperatures and moderately acidic soil.",
    "pigeonpeas": "Pigeon peas are drought-tolerant and suited to warm, semi-arid climates.",
    "mothbeans": "Moth beans are highly drought-resistant, ideal for arid, low-rainfall regions.",
    "mungbean": "Mung bean needs warm temperatures and consistent moisture through flowering.",
    "blackgram": "Black gram grows best in warm climates with moderate, well-distributed rainfall.",
    "lentil": "Lentil is a cool-climate crop tolerant of low rainfall and neutral-to-alkaline soil.",
    "pomegranate": "Pomegranate favors hot, dry summers and cooler winters with well-drained soil.",
    "banana": "Banana needs consistent warmth, high humidity, and steady rainfall year-round.",
    "mango": "Mango prefers a hot, dry pre-flowering period followed by warm, humid fruiting.",
    "grapes": "Grapes need warm days, cool nights, and well-drained, moderately acidic soil.",
    "watermelon": "Watermelon thrives in warm temperatures with sandy, well-drained soil.",
    "muskmelon": "Muskmelon needs hot days, high humidity at fruiting, and light rainfall.",
    "apple": "Apple requires a cool climate with a winter chill period and moderate rainfall.",
    "orange": "Orange favors a subtropical climate with warm days and mild, humid nights.",
    "papaya": "Papaya needs hot, humid, frost-free conditions and steady rainfall.",
    "coconut": "Coconut requires a hot, humid coastal climate with high, well-distributed rainfall.",
    "cotton": "Cotton needs a long, hot growing season and moderate, well-timed rainfall.",
    "jute": "Jute favors hot, humid conditions with heavy monsoon rainfall.",
    "coffee": "Coffee grows best at moderate altitude with warm days, high humidity, and shade.",
}

app = Flask(__name__)


# ---------------------------------------------------------------- CORS ----
@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


@app.route("/api/<path:_any>", methods=["OPTIONS"])
def cors_preflight(_any):
    return ("", 204)


# ------------------------------------------------------------- model IO ----
_model = None
_label_encoder = None
_meta = None
_metrics = None
_feature_importance = None


def load_artifacts():
    global _model, _label_encoder, _meta, _metrics, _feature_importance
    try:
        _model = joblib.load(os.path.join(MODELS_DIR, "model.joblib"))
        _label_encoder = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
        with open(os.path.join(MODELS_DIR, "meta.json")) as f:
            _meta = json.load(f)
        with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
            _metrics = json.load(f)
        with open(os.path.join(MODELS_DIR, "feature_importance.json")) as f:
            _feature_importance = json.load(f)
        return True
    except FileNotFoundError:
        return False


MODEL_READY = load_artifacts()


# --------------------------------------------------------------- SQLite ----
def init_db():
    if not ENABLE_DB:
        return
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            crop TEXT NOT NULL,
            confidence REAL,
            model_name TEXT,
            N REAL, P REAL, K REAL,
            temperature REAL, humidity REAL, ph REAL, rainfall REAL
        )
    """)
    conn.commit()
    conn.close()


def log_prediction(crop, confidence, model_name, sample):
    if not ENABLE_DB:
        return
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """INSERT INTO predictions
           (timestamp, crop, confidence, model_name, N, P, K, temperature, humidity, ph, rainfall)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            crop, confidence, model_name,
            sample["N"], sample["P"], sample["K"],
            sample["temperature"], sample["humidity"], sample["ph"], sample["rainfall"],
        ),
    )
    conn.commit()
    conn.close()


init_db()


# ------------------------------------------------------------- helpers ----
def parse_sample(payload):
    missing = [f for f in FEATURES if f not in payload or payload[f] is None]
    if missing:
        raise ValueError(f"Missing field(s): {', '.join(missing)}")
    sample = {}
    for f in FEATURES:
        try:
            sample[f] = float(payload[f])
        except (TypeError, ValueError):
            raise ValueError(f"Field '{f}' must be a number")
    ranges = {
        "N": (0, 200), "P": (0, 200), "K": (0, 250),
        "temperature": (-10, 55), "humidity": (0, 100),
        "ph": (0, 14), "rainfall": (0, 500),
    }
    for f, (lo, hi) in ranges.items():
        if not (lo <= sample[f] <= hi):
            raise ValueError(f"Field '{f}' out of plausible range ({lo}-{hi})")
    return sample


def predict_proba(sample):
    x = np.array([[sample[f] for f in FEATURES]])
    proba = _model.predict_proba(x)[0]
    order = np.argsort(proba)[::-1]
    ranked = [
        {"crop": _label_encoder.inverse_transform([i])[0], "probability": round(float(proba[i]), 4)}
        for i in order
    ]
    return ranked


def error_response(message, status=400):
    return jsonify({"error": message}), status


def require_model():
    if not MODEL_READY:
        return error_response(
            "Model not trained yet. Run `python train_model.py` in the backend folder.", 503
        )
    return None


# ---------------------------------------------------------------- health ----
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": bool(MODEL_READY),
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })


# --------------------------------------------------------------- predict ----
@app.route("/api/predict", methods=["POST"])
def predict():
    err = require_model()
    if err:
        return err
    payload = request.get_json(silent=True) or {}
    try:
        sample = parse_sample(payload)
    except ValueError as e:
        return error_response(str(e))

    ranked = predict_proba(sample)
    top = ranked[0]
    model_name = _meta["model_name"]

    log_prediction(top["crop"], top["probability"], model_name, sample)

    return jsonify({
        "data": {
            "prediction": {"crop": top["crop"], "confidence": top["probability"]},
            "model": model_name,
            "explanation": CROP_NOTES.get(top["crop"], ""),
            "top_recommendations": ranked[1:4],
            "supplementary_crop_info": {
                "typical_N": round(sample["N"], 1),
                "typical_P": round(sample["P"], 1),
                "typical_K": round(sample["K"], 1),
            },
        }
    })


@app.route("/api/predict/top", methods=["POST"])
def predict_top():
    err = require_model()
    if err:
        return err
    payload = request.get_json(silent=True) or {}
    try:
        sample = parse_sample(payload)
    except ValueError as e:
        return error_response(str(e))

    top_n = payload.get("top_n", 3)
    try:
        top_n = max(1, min(int(top_n), len(_label_encoder.classes_)))
    except (TypeError, ValueError):
        top_n = 3

    ranked = predict_proba(sample)

    return jsonify({
        "data": {
            "model": _meta["model_name"],
            "recommendations": ranked[:top_n],
        }
    })


# ----------------------------------------------------------------- crops ----
@app.route("/api/crops", methods=["GET"])
def crops():
    err = require_model()
    if err:
        return err
    names = sorted(_label_encoder.classes_.tolist())
    return jsonify({"data": {"crops": names}})


# ---------------------------------------------------------------- history ----
@app.route("/api/predictions/history", methods=["GET"])
def history():
    if not ENABLE_DB:
        return jsonify({"data": {"enabled": False}})
    limit = request.args.get("limit", 50, type=int)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    out = []
    for r in rows:
        out.append({
            "timestamp": r["timestamp"],
            "predicted_crop": r["crop"],
            "confidence": r["confidence"],
            "model_name": r["model_name"],
            "N": r["N"], "P": r["P"], "K": r["K"],
            "temperature": r["temperature"], "humidity": r["humidity"],
            "ph": r["ph"], "rainfall": r["rainfall"],
        })
    return jsonify({"data": {"enabled": True, "history": out}})


# -------------------------------------------------------------- perf/meta ----
@app.route("/api/models/performance", methods=["GET"])
def performance():
    err = require_model()
    if err:
        return err
    return jsonify({"data": _metrics})


@app.route("/api/models/feature-importance", methods=["GET"])
def feature_importance():
    err = require_model()
    if err:
        return err
    return jsonify({"data": _feature_importance})


@app.route("/api/models/current", methods=["GET"])
def current_model():
    err = require_model()
    if err:
        return err
    return jsonify({"data": _meta})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    print(f"Field Ledger backend starting on http://localhost:{port}  (model_loaded={MODEL_READY})")
    app.run(host="0.0.0.0", port=port, debug=False)
