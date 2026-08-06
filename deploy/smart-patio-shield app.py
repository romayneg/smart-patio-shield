"""
Smart Patio Shield — Inference Service
============================================================

A minimal FastAPI service that loads the trained XGBoost model (Model 1) and
returns a wind-driven-rain probability plus a deploy-or-stow decision.

Why Model 1 alone:
    The project's central finding was that satellite imagery adds no discriminative
    value for this task beyond the free reanalysis weather features (Section 8-9 of
    the technical documentation). The deployed service therefore runs the tabular
    model only - a deliberate engineering decision that follows directly from the
    research result, and which keeps the live system lightweight (no satellite
    pipeline, no GPU).

How to run:
    pip install fastapi uvicorn xgboost pandas numpy
    python smart-patio-shield app.py
    # then open dashboard.html in a browser (it calls http://127.0.0.1:8000)

Endpoints:
    GET  /health          -> {"status": "ok", "model_loaded": true}
    POST /predict         -> takes intuitive weather inputs, returns decision
    GET  /sample_hours    -> a handful of real test-set hours for the demo picker
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "xgboost_baseline.json"
MANIFEST_PATH = HERE / "baseline_training.manifest.json"
SAMPLE_PATH = HERE / "sample_hours.json"             # generated demo hours from real test set

# The exact feature schema the model was trained on (from baseline.py)
NUMERIC_FEATURES = [
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "pressure_msl", "surface_pressure",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "wind_speed_10m",
    "et0_fao_evapotranspiration", "vapour_pressure_deficit", "sunshine_duration",
    "oni",
    "hour", "month", "hour_sin", "hour_cos", "month_sin", "month_cos",
    "precip_lag1", "precip_lag3", "precip_sum_3h",
    "gust_lag1", "gust_max_3h",
    "pressure_msl_trend_3h", "pressure_msl_trend_6h",
    "cloud_cover_low_lag1",
    "humidity_lag1", "humidity_lag3",
]
CATEGORICAL_FEATURES = ["location", "enso_phase"]
LOCATIONS = ["kingston", "montego_bay"]
ENSO_PHASES = ["nina", "neutral", "nino"]

# Decision threshold - the operating point for deploy/stow.
# 0.5 is the default; a real deployment would tune this to the cost of a
# missed event vs a false alarm (see technical doc Section 7.4).
DECISION_THRESHOLD = 0.5

app = FastAPI(title="Smart Patio Shield Inference Service")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Load model + the exact post-dummy column order the booster expects
# ----------------------------------------------------------------------
_model = None
_fit_columns = None


def _load():
    global _model, _fit_columns
    _model = xgb.XGBClassifier()
    _model.load_model(str(MODEL_PATH))
    if MANIFEST_PATH.exists():
        _fit_columns = json.load(open(MANIFEST_PATH))["feature_columns"]
    else:
        # Reconstruct the dummy-expanded column order deterministically
        cols = list(NUMERIC_FEATURES)
        for loc in LOCATIONS:
            cols.append(f"location_{loc}")
        for ph in ENSO_PHASES:
            cols.append(f"enso_phase_{ph}")
        _fit_columns = cols
    print(f"Model loaded. Expecting {len(_fit_columns)} columns.")


@app.on_event("startup")
def startup():
    _load()


# ----------------------------------------------------------------------
# Input model - Intuitive controls.
# The service derives the full 30+ feature vector from these, so the user manipulates human-understandable quantities.
# ----------------------------------------------------------------------
class WeatherInput(BaseModel):
    location: str = "montego_bay"          # kingston | montego_bay
    hour: int = 14                          # 0-23 local
    month: int = 10                         # 1-12
    relative_humidity: float = 85.0         # %
    cloud_cover_low: float = 70.0           # %
    cloud_cover_mid: float = 50.0           # %
    cloud_cover_high: float = 30.0          # %
    recent_rain_mm: float = 0.5             # mm in the previous hour (precip_lag1)
    recent_gust_kmh: float = 25.0           # max gust previous hour (gust_lag1)
    pressure_falling: bool = True           # is pressure trending down (storm sign)?
    enso_phase: str = "neutral"             # nina | neutral | nino
    threshold: float = DECISION_THRESHOLD   # allow the demo to move the operating point


def _derive_features(w: WeatherInput) -> pd.DataFrame:
    """Turn the ~11 intuitive inputs into the full model feature row.

    Non-exposed features are filled with physically reasonable values derived
    from the exposed ones (e.g. dew point from humidity and temperature).
    """
    # crude but reasonable temperature by hour (tropical coastal diurnal range)
    temp = 26.0 + 4.0 * math.sin((w.hour - 9) / 24 * 2 * math.pi)
    rh = float(np.clip(w.relative_humidity, 1, 100))
    # dew point via Magnus approximation
    a, b = 17.27, 237.7
    gamma = (a * temp) / (b + temp) + math.log(rh / 100.0)
    dew = (b * gamma) / (a - gamma)
    cloud_all = np.mean([w.cloud_cover_low, w.cloud_cover_mid, w.cloud_cover_high])
    # vapour pressure deficit (kPa), rough
    es = 0.6108 * math.exp((17.27 * temp) / (temp + 237.3))
    vpd = es * (1 - rh / 100.0)
    press_trend = -0.9 if w.pressure_falling else 0.4   # hPa over 3h, sign is what matters

    row = {
        "temperature_2m": temp,
        "relative_humidity_2m": rh,
        "dew_point_2m": dew,
        "apparent_temperature": temp - 1.0,
        "pressure_msl": 1013.0 + press_trend,
        "surface_pressure": 1011.0 + press_trend,
        "cloud_cover": cloud_all,
        "cloud_cover_low": w.cloud_cover_low,
        "cloud_cover_mid": w.cloud_cover_mid,
        "cloud_cover_high": w.cloud_cover_high,
        "wind_speed_10m": max(6.0, w.recent_gust_kmh * 0.6),
        "et0_fao_evapotranspiration": max(0.0, 0.15 * (1 - cloud_all / 100)),
        "vapour_pressure_deficit": max(0.0, vpd),
        "sunshine_duration": max(0.0, 3600 * (1 - cloud_all / 100) * (1 if 6 <= w.hour <= 18 else 0)),
        "oni": {"nina": -1.0, "neutral": 0.0, "nino": 1.0}[w.enso_phase],
        "hour": w.hour,
        "month": w.month,
        "hour_sin": math.sin(2 * math.pi * w.hour / 24),
        "hour_cos": math.cos(2 * math.pi * w.hour / 24),
        "month_sin": math.sin(2 * math.pi * w.month / 12),
        "month_cos": math.cos(2 * math.pi * w.month / 12),
        "precip_lag1": w.recent_rain_mm,
        "precip_lag3": w.recent_rain_mm * 1.5,
        "precip_sum_3h": w.recent_rain_mm * 2.0,
        "gust_lag1": w.recent_gust_kmh,
        "gust_max_3h": w.recent_gust_kmh * 1.1,
        "pressure_msl_trend_3h": press_trend,
        "pressure_msl_trend_6h": press_trend * 1.6,
        "cloud_cover_low_lag1": w.cloud_cover_low,
        "humidity_lag1": rh,
        "humidity_lag3": rh,
    }
    # one-hot the categoricals to match training
    for loc in LOCATIONS:
        row[f"location_{loc}"] = 1.0 if w.location == loc else 0.0
    for ph in ENSO_PHASES:
        row[f"enso_phase_{ph}"] = 1.0 if w.enso_phase == ph else 0.0

    X = pd.DataFrame([row])
    for c in _fit_columns:
        if c not in X.columns:
            X[c] = 0.0
    return X[_fit_columns]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None,
            "n_features": len(_fit_columns) if _fit_columns else 0}


@app.post("/predict")
def predict(w: WeatherInput):
    X = _derive_features(w)
    prob = float(_model.predict_proba(X)[:, 1][0])
    thr = float(w.threshold)
    decision = "DEPLOY" if prob >= thr else "STOW"
    # human-readable driver
    drivers = []
    if w.recent_rain_mm > 0.5:
        drivers.append("recent rain (strong persistence signal)")
    if w.cloud_cover_low > 60:
        drivers.append("heavy low cloud")
    if w.recent_gust_kmh > 25:
        drivers.append("gusty wind")
    if w.pressure_falling:
        drivers.append("falling pressure")
    if 12 <= w.hour <= 18:
        drivers.append("afternoon convection window")
    return {
        "probability": round(prob, 4),
        "threshold": thr,
        "decision": decision,
        "confidence_band": ("high" if abs(prob - thr) > 0.25 else "borderline"),
        "key_drivers": drivers or ["conditions generally dry"],
    }


@app.get("/sample_hours")
def sample_hours():
    if SAMPLE_PATH.exists():
        return json.load(open(SAMPLE_PATH))
    return {"hours": [], "note": "sample_hours.json not generated"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
