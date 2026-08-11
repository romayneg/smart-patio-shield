"""
Trains the tabular baseline models for Smart Patio Shield:
  - Logistic regression (with feature scaling): the linear reference.
  - XGBoost with early stopping: the strong baseline.

Loads processed features, applies the time-based train/val/test split,
trains both models on identical data (NaN rows dropped for clean comparison),
evaluates on train and val, and saves model artifacts + a training manifest.

The test set is not touched here. It is held out for final evaluation only.
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import time_based_split  # noqa: E402

PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "patio_features.parquet"
MODELS_DIR = PROJECT_ROOT / "models"


# ----- Feature schema -----
NUMERIC_FEATURES = [
    # Current-hour atmospheric state
    "temperature_2m", "relative_humidity_2m", "dew_point_2m", "apparent_temperature",
    "pressure_msl", "surface_pressure",
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "wind_speed_10m",
    "et0_fao_evapotranspiration", "vapour_pressure_deficit", "sunshine_duration",
    # ENSO
    "oni",
    # Temporal (raw + cyclical)
    "hour", "month", "hour_sin", "hour_cos", "month_sin", "month_cos",
    # Lag / trend features
    "precip_lag1", "precip_lag3", "precip_sum_3h",
    "gust_lag1", "gust_max_3h",
    "pressure_msl_trend_3h", "pressure_msl_trend_6h",
    "cloud_cover_low_lag1",
    "humidity_lag1", "humidity_lag3",
]
CATEGORICAL_FEATURES = ["location", "enso_phase"]
TARGET = "wet_veranda"


def prepare_xy(df_split: pd.DataFrame, fit_columns=None):
    """Build X, y from a split. One-hot encode categoricals, align to schema,
    drop rows with NaN from lag boundaries."""
    X = df_split[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df_split[TARGET].copy()
    X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=False)

    if fit_columns is not None:
        for col in fit_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[fit_columns]

    mask = ~X.isna().any(axis=1)
    return X[mask], y[mask]


def train_logistic_regression(X_train, y_train):
    """Train scaled LR with balanced class weights."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost with class imbalance weighting and early stopping."""
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=20,
    )
    return model


def evaluate(y_true, y_proba, name: str):
    """Compute metrics; return a dict (also prints a short summary)."""
    y_pred = (y_proba >= 0.5).astype(int)
    report = classification_report(
        y_true, y_pred, target_names=["dry", "wet"], digits=3, output_dict=True
    )
    metrics = {
        "name": name,
        "n_samples": int(len(y_true)),
        "positive_rate": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "precision_wet": float(report["wet"]["precision"]),
        "recall_wet": float(report["wet"]["recall"]),
        "f1_wet": float(report["wet"]["f1-score"]),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, target_names=["dry", "wet"], digits=3))
    print(f"ROC-AUC: {metrics['roc_auc']:.3f}  PR-AUC: {metrics['pr_auc']:.3f}")
    return metrics


def save_artifacts(lr_model, scaler, xgb_model, feature_columns, metrics, boundaries):
    """Save trained models, scaler, feature schema, and training manifest."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(lr_model, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(scaler, MODELS_DIR / "logistic_regression_scaler.joblib")
    xgb_model.save_model(str(MODELS_DIR / "xgboost_baseline.json"))

    with open(PROCESSED_FILE, "rb") as f:
        data_hash = hashlib.sha256(f.read()).hexdigest()

    manifest = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_data": {
            "file": PROCESSED_FILE.name,
            "sha256": data_hash,
        },
        "split_boundaries": {k: str(v) for k, v in boundaries.items()},
        "feature_columns": list(feature_columns),
        "models": {
            "logistic_regression": {
                "type": "sklearn.LogisticRegression",
                "model_file": "logistic_regression.joblib",
                "scaler_file": "logistic_regression_scaler.joblib",
                "hyperparameters": {
                    "class_weight": "balanced",
                    "max_iter": 1000,
                    "random_state": 42,
                },
                "metrics_train": metrics["lr_train"],
                "metrics_val": metrics["lr_val"],
            },
            "xgboost_baseline": {
                "type": "xgboost.XGBClassifier",
                "model_file": "xgboost_baseline.json",
                "hyperparameters": {
                    "n_estimators": 500,
                    "max_depth": 6,
                    "learning_rate": 0.1,
                    "scale_pos_weight": float(
                        metrics.get("xgb_scale_pos_weight", 0.0)
                    ),
                    "early_stopping_rounds": 30,
                    "eval_metric": "aucpr",
                    "random_state": 42,
                },
                "best_iteration": int(metrics.get("xgb_best_iteration", -1)),
                "metrics_train": metrics["xgb_train"],
                "metrics_val": metrics["xgb_val"],
            },
        },
    }

    manifest_path = MODELS_DIR / "baseline_training.manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"\nSaved models and manifest to {MODELS_DIR}")
    return manifest_path


if __name__ == "__main__":
    print("Loading data and applying split...")
    df = pd.read_parquet(PROCESSED_FILE)
    train, val, test, boundaries = time_based_split(df)
    print(f"  Train/Val/Test rows: {len(train):,} / {len(val):,} / {len(test):,}")

    print("Preparing X, y...")
    X_train, y_train = prepare_xy(train)
    X_val, y_val = prepare_xy(val, fit_columns=X_train.columns)
    print(f"  X_train: {X_train.shape}, positive rate {y_train.mean()*100:.2f}%")
    print(f"  X_val:   {X_val.shape}, positive rate {y_val.mean()*100:.2f}%")

    # ----- Logistic regression -----
    print("\nTraining logistic regression...")
    lr_model, scaler = train_logistic_regression(X_train, y_train)
    lr_train_proba = lr_model.predict_proba(scaler.transform(X_train))[:, 1]
    lr_val_proba = lr_model.predict_proba(scaler.transform(X_val))[:, 1]
    lr_train_metrics = evaluate(y_train, lr_train_proba, "LR on TRAIN")
    lr_val_metrics = evaluate(y_val, lr_val_proba, "LR on VAL")

    # ----- XGBoost -----
    print("\nTraining XGBoost with early stopping...")
    xgb_model = train_xgboost(X_train, y_train, X_val, y_val)
    print(f"\n  Best iteration: {xgb_model.best_iteration}")
    print(f"  Best val PR-AUC: {xgb_model.best_score:.4f}")
    xgb_train_proba = xgb_model.predict_proba(X_train)[:, 1]
    xgb_val_proba = xgb_model.predict_proba(X_val)[:, 1]
    xgb_train_metrics = evaluate(y_train, xgb_train_proba, "XGBoost on TRAIN")
    xgb_val_metrics = evaluate(y_val, xgb_val_proba, "XGBoost on VAL")

    # ----- Save -----
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    metrics = {
        "lr_train": lr_train_metrics,
        "lr_val": lr_val_metrics,
        "xgb_train": xgb_train_metrics,
        "xgb_val": xgb_val_metrics,
        "xgb_best_iteration": xgb_model.best_iteration,
        "xgb_scale_pos_weight": scale_pos_weight,
    }
    save_artifacts(lr_model, scaler, xgb_model, X_train.columns, metrics, boundaries)