"""
src/models/cnn_eval.py

Evaluation, interpretability, and ablation for Model 2 (vision).

  - evaluate_full : PR-AUC + PR curve + confusion matrix.
  - run_ablation  : trains IR-only vs IR+visible under identical settings and
                    tabulates the PR-AUC difference (the vision analogue of the
                    persistence experiment).
  - show_examples : highest/lowest-scored test patches (what the model is confident about).
  - saliency_map  : gradient-based highlight of which pixels drove a prediction.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, confusion_matrix,
)
from torch.utils.data import DataLoader

from src.models.cnn import build_model, train_model, evaluate


# ---------- 1. Full evaluation, same frame as Model 1 ----------
def evaluate_full(model, test_ds, device=None, threshold=0.5, batch_size=64):
    """PR-AUC, PR curve data, and confusion matrix on the held-out test set."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    out = evaluate(model, loader, device)
    probs, labels = out["probs"], out["labels"]

    pr_auc = average_precision_score(labels, probs)
    prec, rec, thr = precision_recall_curve(labels, probs)
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds)

    print(f"Test PR-AUC: {pr_auc:.4f}")
    print(f"Positive rate: {labels.mean():.4f}")
    print(f"Confusion matrix @ {threshold}:\n{cm}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(rec, prec, color="darkorange", lw=2, label=f"CNN (PR-AUC = {pr_auc:.3f})")
    ax.axhline(labels.mean(), ls="--", color="gray",
               label=f"Random ({labels.mean()*100:.1f}%)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Model 2 (vision) — Precision-Recall on test set")
    ax.legend(); plt.tight_layout()

    return {"pr_auc": float(pr_auc), "probs": probs, "labels": labels,
            "confusion_matrix": cm, "pr_curve": (prec, rec)}


# ---------- 2. Channel ablation: IR-only vs IR+visible ----------
def run_ablation(make_datasets, epochs=30, patience=5, device=None, **train_kw):
    """
    make_datasets(channels, norm_stats=None, patches=None) -> (train_ds, val_ds, test_ds)
    A factory the caller supplies, so this function stays agnostic about paths.
    Trains both channel configs under identical settings, returns a results dict.
    """
    from src.models.goes_dataset import IR_ONLY, ALL_BANDS, _load_all_patches
    patches = _load_all_patches()
    results = {}

    for label, channels in [("IR-only", IR_ONLY), ("IR+visible", ALL_BANDS)]:
        print(f"\n{'='*60}\n{label}  ({len(channels)} channels)\n{'='*60}")
        train_ds, val_ds, test_ds = make_datasets(channels, patches=patches)
        model, history = train_model(
            train_ds, val_ds, in_channels=len(channels),
            epochs=epochs, patience=patience, device=device, **train_kw,
        )
        test_eval = evaluate_full(model, test_ds, device=device)
        results[label] = {
            "channels": channels,
            "val_best_pr_auc": max(h["val_pr_auc"] for h in history),
            "test_pr_auc": test_eval["pr_auc"],
            "model": model,
        }

    print(f"\n{'='*60}\nABLATION SUMMARY\n{'='*60}")
    for label, r in results.items():
        print(f"{label:12s}  test PR-AUC {r['test_pr_auc']:.4f}")
    delta = results["IR+visible"]["test_pr_auc"] - results["IR-only"]["test_pr_auc"]
    print(f"\nVisible channel contribution: {delta:+.4f} PR-AUC")
    return results


# ---------- 3a. Confident examples ----------
@torch.no_grad()
def show_examples(model, test_ds, device=None, n=4):
    """Show the most confidently-correct and confidently-wrong test patches."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    probs, ys, idxs = [], [], []
    i = 0
    for x, y in loader:
        p = torch.sigmoid(model(x.to(device)).squeeze(1)).cpu().numpy()
        probs.append(p); ys.append(y.numpy())
        idxs.extend(range(i, i + len(y))); i += len(y)
    probs = np.concatenate(probs); ys = np.concatenate(ys)

    # confident-correct wet, confident-wrong (high prob but actually dry)
    conf_correct = np.where((ys == 1))[0][np.argsort(-probs[ys == 1])[:n]]
    conf_wrong = np.where((ys == 0))[0][np.argsort(-probs[ys == 0])[:n]]

    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    for row, (group, title) in enumerate(
            [(conf_correct, "Correct wet (high conf)"),
             (conf_wrong, "False alarm (high conf, actually dry)")]):
        for col, gi in enumerate(group):
            x, _ = test_ds[gi]
            axes[row, col].imshow(x[0].numpy(), cmap="gray_r")  # C13 channel
            axes[row, col].set_title(f"{title}\np={probs[gi]:.2f}", fontsize=9)
            axes[row, col].axis("off")
    plt.tight_layout()


# ---------- 3b. Saliency map ----------
def saliency_map(model, test_ds, idx, device=None):
    """Gradient-based saliency: which input pixels most affect the prediction.

    Computes |d(logit)/d(input)| — bright pixels are those a small change in
    which would most change the model's output. The vision analogue of feature
    importance, but per-pixel and per-image.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    x, y = test_ds[idx]
    x = x.unsqueeze(0).to(device).requires_grad_(True)   # (1, C, 64, 64)

    logit = model(x).squeeze()
    logit.backward()
    sal = x.grad.abs().squeeze(0).cpu().numpy()           # (C, 64, 64)
    sal = sal.max(axis=0)                                  # strongest channel per pixel
    prob = torch.sigmoid(logit).item()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].imshow(x.detach().cpu().squeeze(0)[0].numpy(), cmap="gray_r")
    axes[0].set_title(f"Input (C13)\nlabel={int(y)}, pred p={prob:.2f}")
    axes[1].imshow(sal, cmap="hot")
    axes[1].set_title("Saliency (|gradient|)")
    axes[2].imshow(x.detach().cpu().squeeze(0)[0].numpy(), cmap="gray_r")
    axes[2].imshow(sal, cmap="hot", alpha=0.5)
    axes[2].set_title("Overlay")
    for a in axes: a.axis("off")
    plt.tight_layout()
    return sal