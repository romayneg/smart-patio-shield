"""
ConvLSTM vision branch for Smart Patio Shield (Model 2 variant).

Motivation: the channel-stacking motion test (notebook 07) gave the network
consecutive frames as extra channels, which discards temporal ordering: a
convolution over stacked channels has no notion that frame t-1 precedes frame t.
A ConvLSTM keeps the spatial map and carries a recurrent hidden state across
frames, so cloud growth and displacement can in principle be represented.

Two things to be honest about when reporting results from this module:
  1. The network is trained from scratch. The ResNet-18 baseline is fine-tuned
     from ImageNet weights, so this comparison is not architecture-for-
     architecture equal; ConvLSTM starts at a disadvantage on limited data.
  2. Frame spacing is still hourly. If the earlier negative was caused by the
     sampling interval rather than by the architecture, ConvLSTM will not rescue
     it, and that outcome is itself informative.

Input convention: the temporal dataset yields (C * T, H, W) with frames in
contiguous blocks, oldest first. The classifier reshapes to (T, C, H, W)
internally. Notebook 08 verifies this ordering empirically before training.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader


class ConvLSTMCell(nn.Module):
    """Single ConvLSTM cell: gates are produced by one convolution over the
    concatenation of the current input and the previous hidden state."""

    def __init__(self, in_ch: int, hidden_ch: int, kernel: int = 3):
        super().__init__()
        self.hidden_ch = hidden_ch
        pad = kernel // 2
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel, padding=pad)

    def forward(self, x, state):
        h, c = state
        gates = self.conv(torch.cat([x, h], dim=1))
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
        g = torch.tanh(g)
        c = f * c + i * g
        h = o * torch.tanh(c)
        return h, c

    def init_state(self, batch, size, device):
        h = torch.zeros(batch, self.hidden_ch, *size, device=device)
        return h, h.clone()


class ConvLSTMClassifier(nn.Module):
    """ConvLSTM over T frames, then pool the final hidden state to one logit.

    Accepts (B, C*T, H, W) so it is a drop-in for the existing DataLoader, and
    reshapes to (B, T, C, H, W) internally.
    """

    def __init__(self, channels_per_frame: int, n_frames: int,
                 hidden: int = 64, kernel: int = 3, dropout: float = 0.3):
        super().__init__()
        self.cpf = channels_per_frame
        self.n_frames = n_frames
        self.cell = ConvLSTMCell(channels_per_frame, hidden, kernel)
        self.norm = nn.BatchNorm2d(hidden)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        b, ct, hgt, wid = x.shape
        assert ct == self.cpf * self.n_frames, (
            f"expected {self.cpf * self.n_frames} channels, got {ct}")
        seq = x.view(b, self.n_frames, self.cpf, hgt, wid)
        state = self.cell.init_state(b, (hgt, wid), x.device)
        for t in range(self.n_frames):
            state = self.cell(seq[:, t], state)
        return self.head(self.norm(state[0]))


def evaluate(model, loader, device) -> dict:
    """PR-AUC plus raw probabilities and labels, matching cnn.evaluate."""
    model.eval()
    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            logit = model(x.to(device)).squeeze(1)
            probs.append(torch.sigmoid(logit).cpu().numpy())
            ys.append(y.numpy())
    probs, ys = np.concatenate(probs), np.concatenate(ys)
    return {"pr_auc": float(average_precision_score(ys, probs)),
            "probs": probs, "labels": ys}


def train_convlstm(train_ds, val_ds, channels_per_frame, n_frames,
                   hidden=64, epochs=25, batch_size=64, lr=1e-3,
                   patience=4, device=None):
    """Same recipe as cnn.train_model: class-weighted BCE, Adam, PR-AUC early
    stopping. Learning rate defaults higher than the ResNet's 1e-4 because this
    network is trained from scratch rather than fine-tuned."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = ConvLSTMClassifier(channels_per_frame, n_frames, hidden=hidden).to(device)

    labels = np.array([train_ds.label_of[k] for k in train_ds.keys])
    pos_weight = torch.tensor([(labels == 0).sum() / max((labels == 1).sum(), 1)],
                              dtype=torch.float32, device=device)
    print(f"device {device} | {channels_per_frame}ch x {n_frames} frames | "
          f"pos_weight {pos_weight.item():.2f} | params "
          f"{sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_pr, best_state, best_epoch, history = -1.0, None, -1, []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x).squeeze(1), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * x.size(0)

        ev = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": total / len(train_ds),
                        "val_pr_auc": ev["pr_auc"]})
        print(f"  epoch {epoch:2d}  loss {total/len(train_ds):.4f}  "
              f"val PR-AUC {ev['pr_auc']:.4f}")

        if ev["pr_auc"] > best_pr:
            best_pr, best_epoch = ev["pr_auc"], epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        elif epoch - best_epoch >= patience:
            print(f"  early stop at epoch {epoch} (best {best_pr:.4f} @ {best_epoch})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history
