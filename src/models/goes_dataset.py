"""
PyTorch Dataset that serves GOES patches paired with wet_veranda labels.

Joins two sources by the "location|ISO-timestamp" key:
  - image patches inside data/raw/goes/goes_YYYY-MM-DD.npz   (3, 64, 64) float32
  - labels + split from data/processed/image_labels.parquet

Key design points:
  - Only indexes patches that exist on disk and have a label, for a given split,
    so it works correctly even while the download is still in progress.
  - Channel selection is a constructor arg, enabling the IR-only vs IR+visible
    ablation without re-reading or duplicating data.
  - Per-channel normalization stats are computed once on the TRAIN split and
    passed in, never recomputed on val/test (no leakage).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOES_DIR = PROJECT_ROOT / "data" / "raw" / "goes"
LABELS_FILE = PROJECT_ROOT / "data" / "processed" / "image_labels.parquet"

# Channel index map for the stored (3, 64, 64) arrays
CHANNELS = {"C13": 0, "C09": 1, "C02": 2}
IR_ONLY = ["C13", "C09"]
ALL_BANDS = ["C13", "C09", "C02"]


def _load_all_patches() -> dict:
    """
    Load every patch from every day-file into one {key: (3,64,64)} dict.
    8k-60k small patches fit comfortably in RAM; simpler and faster than
    re-opening files per access.
    """
    patches = {}
    for f in sorted(GOES_DIR.glob("goes_*.npz")):
        with np.load(f) as d:
            for k in d.files:
                patches[k] = d[k].astype("float32")
    return patches


class GoesPatchDataset(Dataset):
    def __init__(self, split: str, channels=IR_ONLY, norm_stats=None, patches=None):
        """
        split       : 'train' | 'val' | 'test'
        channels    : which bands to stack, e.g. IR_ONLY or ALL_BANDS
        norm_stats  : dict {band: (mean, std)} from the TRAIN split; required for
                      val/test, computed internally if None (train only).
        patches     : optional preloaded {key: array} dict (avoids reloading
                      from disk for each split).
        """
        self.split = split
        self.channels = channels
        self.ch_idx = [CHANNELS[c] for c in channels]

        labels = pd.read_parquet(LABELS_FILE)
        labels = labels[labels["split"] == split]
        self.label_of = dict(zip(labels["key"], labels["wet_veranda"]))

        all_patches = patches if patches is not None else _load_all_patches()
        # Keep only keys that have both a patch and a label for this split
        self.keys = [k for k in self.label_of if k in all_patches]
        self.patches = {k: all_patches[k] for k in self.keys}

        # NaN guard: a channel can be all-NaN if a scan was missing
        for k in self.keys:
            self.patches[k] = np.nan_to_num(self.patches[k], nan=0.0)

        self.norm_stats = norm_stats or self._compute_norm_stats()

    def _compute_norm_stats(self) -> dict:
        """Per-channel mean/std across this split (call on train only)."""
        stats = {}
        for c in self.channels:
            idx = CHANNELS[c]
            vals = np.stack([self.patches[k][idx] for k in self.keys])
            stats[c] = (float(vals.mean()), float(vals.std()) or 1.0)
        return stats

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        key = self.keys[i]
        arr = self.patches[key]
        chans = []
        for c in self.channels:
            idx = CHANNELS[c]
            mean, std = self.norm_stats[c]
            chans.append((arr[idx] - mean) / std)
        x = torch.from_numpy(np.stack(chans)).float()      # (C, 64, 64)
        y = torch.tensor(self.label_of[key], dtype=torch.float32)
        return x, y