"""
Fetches the NOAA Oceanic Niño Index (ONI) — 3-month running mean of
sea surface temperature anomalies in the Niño 3.4 region.

The ONI is the standard measure of El Niño-Southern Oscillation (ENSO) state.
It is a naturally occurring, repeating climate cycle centered in the tropical
Pacific Ocean that changes weather and temperature patterns all around the world.

Negative values = La Niña (cool Pacific, generally wetter Caribbean),
Positive = El Niño (warm  Pacific, generally drier Caribbean).

The +/- 0.5°C threshold is the conventional boundary for declaring an El Niño or La Niña event.

Source: https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from io import StringIO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"

ONI_URL = "https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php"

# The 12 overlapping 3-month seasons used by NOAA. Each season is named by its three constituent months.
# The "center month" of each season is what we'll use to align with our weather data's monthly index.

SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]

# Mapping from calendar month to the season that's center on it.
# E.g. January is the middle of the Dec-Jan-Feb, so we map 1 -> DJF.
# This is the conventional way to assign a single ONI value to a month.
MONTH_TO_SEASON = {
    1: "DJF", 2: "JFM", 3: "FMA", 4: "MAM", 5: "AMJ", 6: "MJJ",
    7: "JJA", 8: "JAS", 9: "ASO", 10: "SON", 11: "OND", 12: "NDJ"
}


def fetch_oni_table() -> pd.DataFrame:
    """
    Scrape the ONI table from NOAA's CPC page.

    The table contains one large HTML table with year rows and 12 seasonal columns.
    pandas.read_html parses it directly, but we get repeated header rows every 10 years, so we filter those out
    """

    print(f"Fetching ONI table from {ONI_URL}")

    # User-Agent header because some government servers reject default Python requests
    headers = {"User-Agent": "Mozilla/5.0 (research; capstone-project)"}
    response = requests.get(ONI_URL, headers=headers, timeout=60)
    response.raise_for_status()

    # read_html returns a list of all tables on the page
    tables = pd.read_html(StringIO(response.text))

    # The ONI data is in the largest table by row count
    df = max(tables, key=len)

    # First row is the actual header; rename and drop the header row
    df.columns = ['year'] + SEASONS
    df = df.iloc[1:].copy()

    # Remove repeated header rows (every 10 years the header is repeated)
    df = df[df['year'] != "Year"]

    # Convert types
    df['year'] = pd.to_numeric(df['year'], errors="coerce")
    for season in SEASONS:
        df[season] = pd.to_numeric(df[season], errors="coerce")

    df = df.dropna(subset=['year']).reset_index(drop=True)
    df['year'] = df['year'].astype(int)

    return df


def reshape_to_monthly(wide_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the year x season wide format into a tidy monthly DataFrame with columns: year, month, season, oni

    This is the form we'll merge into our hourly weather data.
    """

    rows = []
    for _, row in wide_df.iterrows():
        year = int(row['year'])
        for month, season in MONTH_TO_SEASON.items():
            oni_value = row[season]
            if pd.notna(oni_value):
                rows.append({
                    "year": year,
                    "month": month,
                    "season": season,
                    "oni": float(oni_value),
                })

    df = pd.DataFrame(rows)
    # Construct a proper period index for the month
    df['year_month'] = pd.to_datetime(
        df['year'].astype(str) + "-" + df['month'].astype(str).str.zfill(2)
    )
    return df.sort_values("year_month").reset_index(drop=True)


def classify_enso_phase(oni: float) -> str:
    """ENSO phase based on the standard +/- 0.5°C threshold"""
    if oni >= 0.5:
        return "el_nino"
    elif oni <= -0.5:
        return "la_nina"
    else:
        return "neutral"


def write_manifest(df: pd.DataFrame, output_path: Path) -> Path:
    """Sidecar JSON manifest, same pattern as our weather data."""
    with open(output_path, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    manifest = {
        "data_file": output_path.name,
        "file_sha256": file_hash,
        "file_size_bytes": output_path.stat().st_size,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": ONI_URL,
        "row_count": len(df),
        "year_range": [int(df["year"].min()), int(df["year"].max())],
        "month_range": {
            "min": str(df["year_month"].min().date()),
            "max": str(df["year_month"].max().date()),
        },
        "description": (
            "Oceanic Niño Index (ONI): 3-month running mean of ERSST.v5 SST "
            "anomalies in the Niño 3.4 region (5°N-5°S, 120°-170°W). Each "
            "month is assigned the ONI value of the 3-month season centered "
            "on it."
        ),
    }

    manifest_path = output_path.with_suffix(".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path

if __name__ == "__main__":
    EXTERNAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    wide = fetch_oni_table()
    print(f"Fetched {len(wide)} years of ONI data ({wide['year'].min()} - {wide['year'].max()})")

    monthly = reshape_to_monthly(wide)
    monthly["enso_phase"] = monthly['oni'].apply(classify_enso_phase)

    print(f"\nReshaped to {len(monthly)} monthly observations")
    print(f"\nENSO phase distribution across all years:")
    print(monthly['enso_phase'].value_counts())

    output_path = EXTERNAL_DATA_DIR / "noaa_oni_monthly.parquet"
    monthly.to_parquet(output_path, index=False)

    manifest_path = write_manifest(monthly, output_path)

    print(f"\nSaved to {output_path.name}")
    print(f"\nManifest: {manifest_path.name}")
    print(f"\nMost recent 12 months:")
    print(monthly.tail(12)[["year_month", "season", "oni", "enso_phase"]].to_string(index=False))