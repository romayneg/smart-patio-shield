"""
Temporal variant of GoesPatchDataset: for each labelled hour T, stack the patch
at T together with the previous (n_frames - 1) hourly patches at T-1, T-2, ...
as additional channels, so a CNN can see how the cloud field moved into hour T.

This is a leakage-safe test of whether temporal context improves Model 2,
using only patches already on disk (no new download). Frames are hourly, which is
coarse for storm motion; if even hourly stacking helps, finer sub-hourly stacking
would likely help more (future work). If it does not help, single-frame is near
the ceiling for this data.

Channel layout for n_frames frames and bands B = [b1, b2, ...]:
    [ b1(T-(n-1)) ... bK(T-(n-1)),  ...,  b1(T) ... bK(T) ]
i.e. oldest frame first, current frame last; within each frame, band order = `channels`.
Total channels = len(channels) * n_frames.

Design notes:
  - Composes the same preloaded patch dict as GoesPatchDataset; no changes to the
    existing dataset/cnn modules.
  - A sample is included only if all n_frames patches exist for that location
    (so early-hour samples with no history are dropped). This shrinks the set
    slightly and equally across configs, keeping comparisons fair.
  - Normalization is per-(band) computed on TRAIN only and reused; the same band
    stat is applied to that band in every frame.
"""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELS_FILE = PROJECT_ROOT / "data" / "processed" / "image_labels.parquet"

CHANNELS = {"C13": 0, "C09": 1, "C02": 2}
IR_ONLY = ["C13", "C09"]
ALL_BANDS = ["C13", "C09", "C02"]

# Key format is "location|YYYY-MM-DDTHH:MM:SS"
_TS_FMT = "%Y-%m-%dT%H:%M:%S"


def _split_key(key):
    loc, ts = key.split("|", 1)
    return loc, datetime.strptime(ts, _TS_FMT)


def _make_key(loc, dt):
    return f"{loc}|{dt.strftime(_TS_FMT)}"


class GoesTemporalDataset(Dataset):
    def __init__(self, split, channels=IR_ONLY, n_frames=2,
                 norm_stats=None, patches=None):
        """
        split      : 'train' | 'val' | 'test'
        channels   : bands to stack per frame, e.g. IR_ONLY or ALL_BANDS
        n_frames   : number of consecutive hourly frames (>=1). n_frames=1 is
                     equivalent to the original single-frame dataset .
        norm_stats : {band: (mean, std)} from TRAIN; computed here if None.
        patches    : preloaded {key: (3,64,64)} dict.
        """
        assert n_frames >= 1
        assert patches is not None, "pass the preloaded patches dict"
        self.split = split
        self.channels = channels
        self.n_frames = n_frames

        labels = pd.read_parquet(LABELS_FILE)
        labels = labels[labels["split"] == split]
        self.label_of = dict(zip(labels["key"], labels["wet_veranda"]))

        self._patches = patches

        # Keep keys whose full temporal stack (T, T-1, ..., T-(n-1)) is available
        # and that have a label for this split.
        self.keys = []
        self._frame_keys = {}   # key -> [key_T-(n-1), ..., key_T]  (oldest first)
        for k in self.label_of:
            if k not in patches:
                continue
            loc, dt = _split_key(k)
            frame_ks = []
            ok = True
            for back in range(self.n_frames - 1, -1, -1):     # oldest -> current
                fk = _make_key(loc, dt - timedelta(hours=back))
                if fk not in patches:
                    ok = False
                    break
                frame_ks.append(fk)
            if ok:
                self.keys.append(k)
                self._frame_keys[k] = frame_ks

        self.norm_stats = norm_stats or self._compute_norm_stats()

    def _compute_norm_stats(self):
        """Per-band mean/std over the current-frame patches of this split (train only).

        Using the current frame's distribution is sufficient and keeps stats
        comparable to the single-frame baseline; the same band stat is applied to
        that band in every frame.
        """
        stats = {}
        for c in self.channels:
            idx = CHANNELS[c]
            vals = np.stack([
                np.nan_to_num(self._patches[self._frame_keys[k][-1]][idx], nan=0.0)
                for k in self.keys
            ])
            std = float(vals.std())
            stats[c] = (float(vals.mean()), std if std > 0 else 1.0)
        return stats

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i):
        key = self.keys[i]
        planes = []
        for fk in self._frame_keys[key]:                      # oldest -> current
            arr = np.nan_to_num(self._patches[fk], nan=0.0)
            for c in self.channels:
                idx = CHANNELS[c]
                mean, std = self.norm_stats[c]
                planes.append((arr[idx] - mean) / std)
        x = torch.from_numpy(np.stack(planes)).float()        # (C*n_frames, 64, 64)
        y = torch.tensor(self.label_of[key], dtype=torch.float32)
        return x, y