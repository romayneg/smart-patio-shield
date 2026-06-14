"""
Time-based train/validation/test splits for the Smart Patio Shield models.

For time-series data, randon splits leak future information into training via
temporally-adjacent rows. This module produces strictly chronological splits:
train comes first in time, then validation, then test - no temporal oerlap.

The same calendar boundaries apply to all locations, so each location contributes
rows to all three sets and the model learns spatial differences in every regime.
"""

from pathlib import Path

import pandas as pd

# Default split ratio. Train comes first chronologically, then val, then test.
DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15
# Test fraction is implied: 1 - train - val


def time_based_split(
        df: pd.DataFrame,
        time_col: str = "time",
        train_frac: float = DEFAULT_TRAIN_FRAC,
        val_frac: float = DEFAULT_VAL_FRAC,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """
    Split a time-indexes DataFrame chronologically into train/val/test.
    Boundaries are computed from the time range and applie to all rows
    uniformly, regardless of location. Returns four things: the three
    DataFrames plus a dict recording the boundary timestampes and split
    sizes.
    """

    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must leave room for a test set.")

    df = df.sort_values(time_col).reset_index(drop=True)
    t_min, t_max = df[time_col].min(), df[time_col].max()
    span = t_max - t_min

    train_end = t_min + span * train_frac
    val_end = t_min + span * (train_frac + val_frac)

    train = df[df[time_col] < train_end].copy().reset_index(drop=True)
    val = df[(df[time_col] >= train_end) & (df[time_col] < val_end)].copy().reset_index(drop=True)
    test = df[df[time_col] >= val_end].copy().reset_index(drop=True)

    boundaries = {
        "train_end": train_end,
        "val_end": val_end,
        "data_start": t_min,
        "data_end": t_max,
        "train_rows": len(train),
        "val_rows": len(val),
        "test_rows": len(test),
    }
    return train, val, test, boundaries