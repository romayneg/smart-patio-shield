"""
src/models/cnn.py

CNN for Model 2 (vision) of Smart Patio Shield: predicts wet_veranda from a
stacked GOES satellite patch (IR-only 2-channel, or IR+visible 3-channel).

Transfer learning from a ResNet-18 pretrained on ImageNet:
  - The first conv layer is replaced to accept N input channels (2 or 3) instead
    of RGB's 3, since our channels are infrared/water-vapor/visible, not colours.
  - The final fully-connected layer is replaced with a single-logit head for
    binary classification.

Training mirrors the tabular discipline: class-weighted loss for the ~9% positive
rate, validation-PR-AUC early stopping, and PR-AUC as the headline metric.
"""

from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from torchvision import models


def build_model(in_channels: int) -> nn.Module:
    """
    ResNet-18 adapted for `in_channels` inputs and binary output.
    """
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Replace first conv to accept the channel count. Initialize the new conv by
    # averaging the pretrained RGB filters across the colour dimension and
    # repeating — preserves useful low-level edge/texture filters rather than
    # starting random.
    old = net.conv1
    new = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        mean_w = old.weight.mean(dim=1, keepdim=True)            # (64,1,7,7)
        new.weight.copy_(mean_w.repeat(1, in_channels, 1, 1))
    net.conv1 = new

    # Single-logit binary head
    net.fc = nn.Linear(net.fc.in_features, 1)
    return net


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """
    Return PR-AUC and the raw probabilities/labels for a loader.
    """
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        x = x.to(device)
        logit = model(x).squeeze(1)
        probs.append(torch.sigmoid(logit).cpu().numpy())
        ys.append(y.numpy())
    probs = np.concatenate(probs)
    ys = np.concatenate(ys)
    return {
        "pr_auc": float(average_precision_score(ys, probs)),
        "probs": probs,
        "labels": ys,
    }


def train_model(
    train_ds, val_ds, in_channels,
    epochs=30, batch_size=64, lr=1e-4, patience=5, device=None,
):
    """
    Train with class-weighted BCE + PR-AUC early stopping.
    Returns (best_model, history).
    """ 
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}, {in_channels}-channel input")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_model(in_channels).to(device)

    # Class weighting: positive class is ~9%, so up-weight positives in BCE.
    labels = np.array([train_ds.label_of[k] for k in train_ds.keys])
    pos_weight = torch.tensor([(labels == 0).sum() / max((labels == 1).sum(), 1)],
                              dtype=torch.float32, device=device)
    print(f"pos_weight: {pos_weight.item():.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_pr, best_state, best_epoch = -1.0, None, -1
    history = []

    for epoch in range(epochs):
        model.train()
        running = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logit = model(x).squeeze(1)
            loss = criterion(logit, y)
            loss.backward()
            optimizer.step()
            running += loss.item() * x.size(0)

        train_loss = running / len(train_ds)
        val_pr = evaluate(model, val_loader, device)["pr_auc"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_pr_auc": val_pr})
        print(f"epoch {epoch:2d}  train_loss {train_loss:.4f}  val_PR-AUC {val_pr:.4f}")

        if val_pr > best_pr:
            best_pr, best_state, best_epoch = val_pr, deepcopy(model.state_dict()), epoch
        elif epoch - best_epoch >= patience:
            print(f"Early stopping (no val improvement for {patience} epochs). "
                  f"Best PR-AUC {best_pr:.4f} at epoch {best_epoch}.")
            break

    model.load_state_dict(best_state)
    return model, history