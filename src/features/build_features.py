"""
Builds the processed feature set for the Smart Patio Shield models.

Reads raw weather + ONI data, merges them, engineers features and
builds the target, then writes a single processed parquet that all
downstream modeling consumes.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
EXTERNAL_DIR = PROJECT_ROOT / "data" / "external"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

WEATHER_FILE = "weather_jamaica_2021-05-01_to_2026-05-01.parquet"
ONI_FILE = "noaa_oni_monthly.parquet"
OUTPUT_FILE = "patio_features.parquet"

# Wet-veranda target thresholds: provisional, see 01_weather_eda.ipynb Section 12.
# Rule: light rain combined with gusty wind blows rain sideways through the grille;
# heavier rain reaches the patio regardless of wind. The thresholds were physically
# reasoned but are not yet empirically calibrated. EDA threshold-sensitivity analysis
# identified the light-rain precip threshold as the dominant lever, and the gust
# threshold (currently low relative to typical rainy-hour gusts) as the priority for
# recalibration.

WET_VERANDA_LIGHT_PRECIP_MM = 0.1
WET_VERANDA_GUST_KMH = 15
WET_VERANDA_HEAVY_PRECIP_MM = 1.0


def load_raw_data():
    """
    Load raw weather and ONI datasets, applying EDA cleaning decisions.

    EDA decisions (see 01_weather_eda.ipynb, sections 4 and 6):
    - boundary_layer_height: 6-month coverage gap (Dec 2023 - Jun 2024), cannot be reliably imputed.
    - rain: a literal duplicate of precipitation (Jamaica has no snow, and the archive API folds shower
      into both). Correlation 1.0, zero diff.

    """

    weather = pd.read_parquet(RAW_DIR / WEATHER_FILE)
    oni = pd.read_parquet(EXTERNAL_DIR / ONI_FILE)

    drop_cols = ['boundary_layer_height', 'rain']
    weather = weather.drop(columns = [c for c in drop_cols if c in weather.columns])

    return weather, oni


def merge_oni(weather: pd.DataFrame, oni: pd.DataFrame) -> pd.DataFrame:
    """Merge monthly ONI into hourly weather; forward-fill trailing months."""
    weather = weather.copy()
    weather["year_month"] = weather["time"].dt.to_period("M").dt.to_timestamp()

    oni_cols = oni[["year_month", "oni", "enso_phase"]]
    df = weather.merge(oni_cols, on="year_month", how="left")

    # Forward-fill the trailing months (Apr/May 2026) not yet published.
    df = df.sort_values(["location", "time"]).reset_index(drop=True)
    df["oni"] = df.groupby("location")["oni"].ffill()
    df["enso_phase"] = df.groupby("location")["enso_phase"].ffill()

    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features derived from the timestamp.

    Includes both raw integer encodings and cyclical sin/cos
    encodings (which preserve wraparound adjacency: hour 23 sits next to
    hour 0, December next to January). Day-of-week is deliberately omitted:
    the EDA confirms rain rate is flat across weekdays, as it must be - the
    atmosphere has no concept of the work week.
    """

    df = df.copy()

    df['hour'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month

    # Cyclical encodings: map each periodic value onto the unit circle
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and trend features that capture recent atmosphere state.

    Lag features look backward in time, exposing momentum and trends that
    instantaneous snapshots miss (e.g., "pressure has been falling for 3 hours"
    is a classic storm precursor invisible in any single hour's reading).

    All operations are grouped by location so one city's history never leaks
    into another's. Rows at the start of each location's series will be NaN
    for the longer lags (up to 6 per location for the 6-hour pressure trend).
    I'll leave these as NaN rather than fill - XGBoost handles missing values
    natively, and downstream code can choose a fill strategy if needed.

    Leakage note: 'precipitation' and 'wind_gusts_10m' are components of the
    'wet_veranda' target. Their rolling features are computed on shifted
    values so the current hour is excluded - otherwise we'd be using the
    target's own ingredients as features. Non-targeted component variables
    (pressure, humidity, clouds) have no such constraints.
    """

    df = df.sort_values(['location','time']).reset_index(drop=True).copy()

    # Precipitation (target component) - past only, current hour excluded
    df['precip_lag1'] = df.groupby('location')['precipitation'].shift(1)
    df['precip_lag3'] = df.groupby('location')['precipitation'].shift(3)
    df['precip_sum_3h'] = df.groupby('location')['precipitation'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=3).sum()
    )

    # Wind gust (target component) - past only, current hour excluded
    df['gust_lag1'] = df.groupby('location')['wind_gusts_10m'].shift(1)
    df['gust_max_3h'] = df.groupby('location')['wind_gusts_10m'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=3).max()
    )

    # Pressure trends (not target component) - current minus past
    df['pressure_msl_trend_3h'] = (
        df['pressure_msl'] - df.groupby('location')['pressure_msl'].shift(3)
    )

    df['pressure_msl_trend_6h'] = (
            df['pressure_msl'] - df.groupby('location')['pressure_msl'].shift(6)
    )

    # Build up signals: humidity and low-cloud lags
    df['cloud_cover_low_lag1'] = df.groupby('location')['cloud_cover_low'].shift(1)
    df['humidity_lag1'] = df.groupby('location')['relative_humidity_2m'].shift(1)
    df['humidity_lag3'] = df.groupby('location')['relative_humidity_2m'].shift(3)

    return df


def build_target(
    df: pd.DataFrame,
    light_precip_mm: float = WET_VERANDA_LIGHT_PRECIP_MM,
    gust_kmh: float = WET_VERANDA_GUST_KMH,
    heavy_precip_mm: float = WET_VERANDA_HEAVY_PRECIP_MM,
) -> pd.DataFrame:
    """Construct the binary `wet_veranda` target.

    A wet-veranda event is defined as either:
      - light-or-heavier rain combined with meaningful wind gusts, OR
      - rainfall above the heavy-rain threshold regardless of wind.

    Defaults reflect the provisional thresholds documented in EDA Section 12.

    Returns a copy of df with an int-typed `wet_veranda` column appended.
    """
    df = df.copy()
    df["wet_veranda"] = (
        ((df["precipitation"] > light_precip_mm) & (df["wind_gusts_10m"] > gust_kmh))
        | (df["precipitation"] > heavy_precip_mm)
    ).astype(int)
    return df



def save_processed(df: pd.DataFrame, output_path: Path, stages: list) -> Path:
    """Write the processed dataset plus a reproducibility manifest."""
    df.to_parquet(output_path, index=False)

    with open(output_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    manifest = {
        "data_file": output_path.name,
        "file_sha256": file_hash,
        "file_size_bytes": output_path.stat().st_size,
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_files": [WEATHER_FILE, ONI_FILE],
        "row_count": len(df),
        "columns": list(df.columns),
        "pipeline_stages": stages,
        "target_thresholds": {
            "light_precip_mm": WET_VERANDA_LIGHT_PRECIP_MM,
            "gust_kmh": WET_VERANDA_GUST_KMH,
            "heavy_precip_mm": WET_VERANDA_HEAVY_PRECIP_MM,
        },
    }
    manifest_path = output_path.with_suffix(".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    stages = []

    print("Loading raw data...")
    weather, oni = load_raw_data()
    stages += ["load", "drop_boundary_layer_height", "drop_rain"]
    print(f"  Weather: {weather.shape}, ONI: {oni.shape}")

    print("Merging ONI...")
    df = merge_oni(weather, oni)
    stages.append("merge_oni")
    missing = df["oni"].isnull().sum()
    print(f"  Missing ONI after merge+ffill: {missing}")

    print("Adding temporal features...")
    df = add_temporal_features(df)
    stages.append("temporal_features")

    print("Adding lag features...")
    df = add_lag_features(df)
    stages.append("lag_features")

    print("Building target...")
    df = build_target(df)
    stages.append("build_target")
    pos_rate = df["wet_veranda"].mean()
    print(f"  Thresholds: light_precip={WET_VERANDA_LIGHT_PRECIP_MM} mm, "
          f"gust={WET_VERANDA_GUST_KMH} km/h, heavy_precip={WET_VERANDA_HEAVY_PRECIP_MM} mm")
    print(f"  Positive class rate: {pos_rate * 100:.2f}% ({int(df['wet_veranda'].sum()):,} events)")

    output_path = PROCESSED_DIR / OUTPUT_FILE
    manifest_path = save_processed(df, output_path, stages)

    print(f"\nSaved {len(df):,} rows to {output_path.name}")
    print(f"Manifest: {manifest_path.name}")
    print(f"Columns ({df.shape[1]}): {list(df.columns)}")