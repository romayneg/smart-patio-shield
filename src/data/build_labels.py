"""
Builds the label lookup table that bridges the tabular and image pipelines.

For every (location, timestamp) in the processed feature set, records:
  - wet_veranda  : the 0/1 target (identical to the tabular models' target)
  - split        : train / val / test, using the same chronological boundaries
                   as src/data/splits.py, so all models are compared on
                   identical date ranges.
  - key          : "location|ISO-timestamp", matching the keys used inside the
                   GOES .npz day-files, so the Dataset can join image <-> label.

Output: data/processed/image_labels.parquet (+ manifest)
"""

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import time_based_split  # noqa: E402

PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "patio_features.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "image_labels.parquet"


def build_label_table() -> pd.DataFrame:
    """Create the (key, location, time, wet_veranda, split) lookup table."""
    df = pd.read_parquet(PROCESSED_FILE)

    # Apply the SAME split as the tabular models.
    train, val, test, boundaries = time_based_split(df)
    split_of = {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        for idx in part.index:
            split_of[idx] = name
    # time_based_split resets indices, so re-derive split membership by
    # timestamp boundaries instead.
    train_end = boundaries["train_end"]
    val_end = boundaries["val_end"]

    def which_split(t):
        if t < train_end:
            return "train"
        elif t < val_end:
            return "val"
        return "test"

    out = df[["location", "time", "wet_veranda"]].copy()
    out["split"] = out["time"].apply(which_split)
    # Key must match the .npz keys exactly: "location|ISO-timestamp"
    out["key"] = out["location"] + "|" + out["time"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    return out, boundaries


def write_manifest(df: pd.DataFrame, boundaries: dict, path: Path):
    with open(path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
    manifest = {
        "data_file": path.name,
        "file_sha256": file_hash,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_file": PROCESSED_FILE.name,
        "row_count": len(df),
        "split_boundaries": {k: str(v) for k, v in boundaries.items()},
        "split_counts": df["split"].value_counts().to_dict(),
        "positive_rate_by_split": df.groupby("split")["wet_veranda"].mean().round(4).to_dict(),
        "note": "Keys match GOES .npz day-file keys: 'location|ISO-timestamp'.",
    }
    with open(path.with_suffix(".manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, default=str)


if __name__ == "__main__":
    out, boundaries = build_label_table()
    out.to_parquet(OUTPUT_FILE, index=False)
    write_manifest(out, boundaries, OUTPUT_FILE)

    print(f"Wrote {len(out):,} labels to {OUTPUT_FILE.name}")
    print(f"\nSplit counts:\n{out['split'].value_counts()}")
    print(f"\nPositive rate by split:")
    print(out.groupby('split')['wet_veranda'].mean().round(4))
    print(f"\nSample keys:")
    print(out['key'].head(3).to_list())