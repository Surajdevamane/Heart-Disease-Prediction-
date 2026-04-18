"""
app.py  –  FastAPI REST API for Heart Disease Prediction (Python 3.14 Compatible)
  /api/predict    → binary + severity prediction
  /api/train      → retrain model
  /api/health     → health check
  /api/model-info → model metadata & feature importances
"""

import os, json
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import uvicorn

# ──────────────────────────────────────────────────────────────
#  Paths
# ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "saved_model")
FRONTEND_DIR = os.path.join(os.path.dirname(BASE_DIR), "frontend")

# ──────────────────────────────────────────────────────────────
#  FastAPI App
# ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="CardioSense — Heart Disease Prediction API",
    version="2.0.0",
    description="Stacked Ensemble (XGBoost + DNN + RF → SVC) for heart disease prediction"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ──────────────────────────────────────────────────────────────
#  Load Models
# ──────────────────────────────────────────────────────────────
models = {"binary": {}, "multi": {}}


def load_all():
    for prefix in ("binary", "multi"):
        meta_path = os.path.join(SAVE_DIR, f"{prefix}_meta.json")
        ensemble_path = os.path.join(SAVE_DIR, f"{prefix}_ensemble.pkl")
        scaler_path = os.path.join(SAVE_DIR, f"{prefix}_scaler.pkl")

        if not os.path.exists(meta_path):
            print(f"[app] ⚠ {prefix} model not found – run train.py first")
            continue

        with open(meta_path) as f:
            meta = json.load(f)

        ensemble = joblib.load(ensemble_path)
        scaler = joblib.load(scaler_path)

        models[prefix] = {
            "meta": meta,
            "ensemble": ensemble,
            "scaler": scaler,
        }
        print(f"[app] ✓ Loaded {prefix} model (features={meta['features']})")


# ──────────────────────────────────────────────────────────────
#  Prediction helpers
# ──────────────────────────────────────────────────────────────

# All 13 input features the front-end sends
ALL_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]


def _predict_one(prefix, input_dict):
    """Run one variant (binary / multi) and return result dict."""
    m = models[prefix]
    meta = m["meta"]
    features = meta["features"]

    # Extract selected features in order
    x = np.array([[input_dict[f] for f in features]])
    x_scaled = m["scaler"].transform(x)

    # Ensemble prediction
    label = int(m["ensemble"].predict(x_scaled)[0])
    proba = m["ensemble"].predict_proba(x_scaled)[0].tolist()

    # Individual base model contributions
    contributions = m["ensemble"].individual_predictions(x_scaled)

    return label, proba, contributions


SEVERITY_LABELS = {
    0: "No Heart Disease",
    1: "Stage 1 — Mild",
    2: "Stage 2 — Moderate",
    3: "Stage 3 — Severe",
    4: "Stage 4 — Critical",
}


def _get_risk_level(confidence, is_positive):
    """Determine risk level for clinical interpretation."""
    if not is_positive:
        return "low"
    if confidence < 65:
        return "moderate"
    return "high"


# ──────────────────────────────────────────────────────────────
#  Pydantic Models
# ──────────────────────────────────────────────────────────────
class PatientData(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: float = Field(..., ge=0, le=1, description="Sex: 1=Male, 0=Female")
    cp: float = Field(..., ge=0, le=3, description="Chest pain type (0-3)")
    trestbps: float = Field(..., ge=50, le=250, description="Resting blood pressure (mmHg)")
    chol: float = Field(..., ge=100, le=600, description="Serum cholesterol (mg/dL)")
    fbs: float = Field(..., ge=0, le=1, description="Fasting blood sugar >120: 1=Yes, 0=No")
    restecg: float = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=60, le=220, description="Max heart rate achieved")
    exang: float = Field(..., ge=0, le=1, description="Exercise induced angina: 1=Yes, 0=No")
    oldpeak: float = Field(..., ge=0, le=7, description="ST depression")
    slope: float = Field(..., ge=0, le=2, description="Slope of ST segment (0-2)")
    ca: float = Field(..., ge=0, le=3, description="Major vessels colored (0-3)")
    thal: float = Field(..., ge=1, le=7, description="Thalassemia type")


# ──────────────────────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    load_all()


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "binary_loaded": bool(models["binary"]),
        "multi_loaded": bool(models["multi"]),
    }


@app.get("/api/model-info")
async def model_info():
    info = {}
    for prefix in ("binary", "multi"):
        if models[prefix]:
            meta = models[prefix]["meta"]
            info[prefix] = {
                "features": meta["features"],
                "metrics": meta.get("metrics", {}),
                "feature_importance": meta.get("feature_importance", []),
                "correlation_scores": meta.get("correlation_scores", {}),
                "num_classes": meta.get("num_classes", 2),
            }
    info["model_architecture"] = {
        "type": "Stacked Ensemble",
        "base_models": ["XGBoost", "DNN (MLPClassifier)", "Random Forest"],
        "meta_learner": "SVC (RBF kernel)",
        "dataset": "Cleveland Heart Disease (UCI)",
        "preprocessing": ["Label Encoding", "SMOTE (train only)", "StandardScaler", "PCC (≥0.2)"],
        "paper": "Sofi et al., Soft Computing, 2025",
    }
    return info


@app.post("/api/predict")
async def predict(patient: PatientData):
    try:
        input_dict = patient.model_dump()
        result = {}

        # Binary
        if models["binary"]:
            label_b, proba_b, contrib_b = _predict_one("binary", input_dict)
            is_positive = label_b == 1
            confidence = round(max(proba_b) * 100, 2)
            result["binary"] = {
                "prediction": label_b,
                "label": "Heart Disease" if is_positive else "No Heart Disease",
                "confidence": confidence,
                "risk_level": _get_risk_level(confidence, is_positive),
                "probabilities": {
                    "No Disease": round(proba_b[0] * 100, 2),
                    "Disease": round(proba_b[1] * 100, 2),
                },
                "model_contributions": contrib_b,
            }

        # Multiclass
        if models["multi"]:
            label_m, proba_m, contrib_m = _predict_one("multi", input_dict)
            result["severity"] = {
                "prediction": label_m,
                "label": SEVERITY_LABELS.get(label_m, f"Stage {label_m}"),
                "confidence": round(max(proba_m) * 100, 2),
                "probabilities": {
                    SEVERITY_LABELS.get(i, f"Stage {i}"): round(p * 100, 2)
                    for i, p in enumerate(proba_m)
                },
                "model_contributions": contrib_m,
            }

        # Feature importance
        if models["binary"]:
            result["feature_importance"] = (
                models["binary"]["meta"].get("feature_importance", [])
            )

        # Correlation scores
        if models["binary"]:
            result["correlation_scores"] = (
                models["binary"]["meta"].get("correlation_scores", {})
            )

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train")
async def train_endpoint():
    """Retrain models (long-running)."""
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(BASE_DIR)))
        from backend.train import train_and_save

        bin_res = train_and_save("binary")
        multi_res = train_and_save("multi")

        # Reload
        load_all()

        return {
            "status": "success",
            "binary_accuracy": bin_res["accuracy"],
            "multi_accuracy": multi_res["accuracy"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Serve frontend static files
@app.get("/")
async def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


# Mount static files AFTER API routes
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[app] Starting FastAPI server on http://localhost:8000")
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
