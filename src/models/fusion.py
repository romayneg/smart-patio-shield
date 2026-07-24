"""
src/models/fusion.py

Model 3 - multimodal fusion of the tabular (Model 1) and vision (Model 2) branches.

Two designs, evaluated under one protocol:
  late fusion          : combine the two models' output probabilities with a
                         small meta-classifier.
  feature-level fusion : concatenate the CNN's 512-d embedding with the tabular
                         feature vector and train a joint head.

Leakage discipline: both combining stages are fitted on the base models'
validation predictions - never their training predictions, which are
unrealistically strong because the base models have already seen that data,
and the full stack is evaluated exactly once on the held-out test set.

Comparability: fusion can only score hours that have both a tabular row and a
satellite patch. All three models are therefore additionally reported on that
common intersection, so the final comparison is like-for-like.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed" / "patio_features.parquet"

from src.data.splits import time_based_split
from src.models.baseline import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
from src.models.cnn import build_model


# ---------------- tabular branch ----------------
def _tabular_xy(df_split, fit_columns):
    """X, y and join-keys for one split, aligned to the trained feature schema."""
    X = df_split[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df_split[TARGET].copy()
    keys = df_split["location"] + "|" + df_split["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    X = pd.get_dummies(X, columns=CATEGORICAL_FEATURES, drop_first=False)
    for col in fit_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[fit_columns]
    mask = ~X.isna().any(axis=1)
    return X[mask], y[mask].to_numpy(), keys[mask].to_numpy()


def tabular_branch(model_path, manifest_path):
    """Load Model 1 and return {split: DataFrame(key, p_tab, y, *features)}."""
    fit_columns = json.load(open(manifest_path))["feature_columns"]

    booster = xgb.XGBClassifier()
    booster.load_model(str(model_path))

    df = pd.read_parquet(PROCESSED)
    train, val, test, _ = time_based_split(df)

    out = {}
    for name, part in [("val", val), ("test", test)]:
        X, y, keys = _tabular_xy(part, fit_columns)
        p = booster.predict_proba(X)[:, 1]
        frame = pd.DataFrame({"key": keys, "p_tab": p, "y": y})
        out[name] = (frame, X.reset_index(drop=True))
    return out


# ---------------- vision branch ----------------
@torch.no_grad()
def vision_branch(model, dataset, device=None, batch_size=128):
    """Return DataFrame(key, p_img, y) and the 512-d embedding matrix."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # embedding extractor: same network with the classifier head removed
    head = model.fc
    model.fc = nn.Identity()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embs, ys = [], []
    for x, y in loader:
        embs.append(model(x.to(device)).cpu().numpy())
        ys.append(y.numpy())
    model.fc = head                                   # restore

    E = np.concatenate(embs)                          # (N, 512)
    ys = np.concatenate(ys)
    logits = head(torch.from_numpy(E).to(device)).squeeze(1)
    p = torch.sigmoid(logits).cpu().numpy()

    frame = pd.DataFrame({"key": dataset.keys, "p_img": p, "y": ys})
    return frame, E


# ---------------- alignment ----------------
def align(tab_frame, tab_X, img_frame, img_E):
    """Inner-join the branches on key; return probs, features, embeddings, labels."""
    tab = tab_frame.copy()
    tab["row"] = np.arange(len(tab))
    img = img_frame.copy()
    img["irow"] = np.arange(len(img))

    m = tab.merge(img[["key", "p_img", "irow"]], on="key", how="inner")
    assert (m["y"].to_numpy() == img_frame["y"].to_numpy()[m["irow"]]).all(), \
        "label mismatch between branches — check the key format"

    return {
        "keys": m["key"].to_numpy(),
        "p_tab": m["p_tab"].to_numpy(),
        "p_img": m["p_img"].to_numpy(),
        "y": m["y"].to_numpy(),
        "X_tab": tab_X.to_numpy()[m["row"].to_numpy()],
        "E_img": img_E[m["irow"].to_numpy()],
    }


# ---------------- fusion models ----------------
def late_fusion(val, test):
    """Meta-classifier over the two output probabilities."""
    Zv = np.column_stack([val["p_tab"], val["p_img"]])
    Zt = np.column_stack([test["p_tab"], test["p_img"]])
    meta = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    meta.fit(Zv, val["y"])
    return meta.predict_proba(Zt)[:, 1], meta


def feature_fusion(val, test):
    """Joint head over [512-d image embedding || tabular features]."""
    Fv = np.hstack([val["E_img"], val["X_tab"]]).astype("float32")
    Ft = np.hstack([test["E_img"], test["X_tab"]]).astype("float32")
    scaler = StandardScaler().fit(Fv)
    head = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42)
    head.fit(scaler.transform(Fv), val["y"])
    return head.predict_proba(scaler.transform(Ft))[:, 1], (scaler, head)


# ---------------- reporting ----------------
def report(name, y, p, threshold=0.5):
    ap = average_precision_score(y, p)
    cm = confusion_matrix(y, (p >= threshold).astype(int))
    tn, fp, fn, tp = cm.ravel()
    print(f"{name:34s} PR-AUC {ap:.4f}   "
          f"recall {tp/(tp+fn):.3f}  precision {tp/(tp+fp):.3f}")
    return {"name": name, "pr_auc": float(ap), "confusion_matrix": cm.tolist()}