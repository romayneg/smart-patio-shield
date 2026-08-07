"""
Fetches historical hourly weather data for Jamaican locations from Open-Meteo.

For project reproducibility, the date range is fixed in code. To extend the window
later, update SNAPSHOT_START and SNAPSHOT_END deliberately and re-fetch.

Open-Meteo's Historical Weather API serves ECMWF reanalysis data, a gridded
global dataset combining historical observations with a physical atmospheric
model. Dates from 2017 onward are served by ECMWF IFS at 9 km resolution, with
ERA5 (25 km) as fallback for earlier dates, so the 2021-2026 snapshot window is
almost entirely IFS. For Jamaica, the nearest grid cell is interpolated to our
requested latitude/longitude.

API docs: https://open-meteo.com/en/docs/historical-weather-api
"""

import hashlib
import json
import time
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

# Snapshot configuration (the project's frozen data window)
SNAPSHOT_END = date(2026, 5, 1)
SNAPSHOT_START = date(2021, 5, 1)  # 5 years prior

# Our two target locations. Coordinates from Google Maps
LOCATIONS = {
    "kingston": {"lat": 17.9714, "lon": -76.7945},
    "montego_bay": {"lat": 18.47298, "lon": -77.92134}
}

# The variables wanted from Open-Meteo.
# Note: in the Historical Archive API, 'rain' already includes convective
# showers (unlike the Forecast API where rain and showers are separate).
# Tropical precipitation is overwhelmingly convective; this is fine for
# training but means at inference we must fetch both rain and showers
# from the Forecast API and sum them.

HOURLY_VARS = [
    "temperature_2m",               # surface temperature
    "relative_humidity_2m",         # humidity is a strong rain predictor
    "dew_point_2m",                 # related to humidity; closer to temp = closer to saturation
    "apparent_temperature",         # "feels-like" temp
    "precipitation",                # our target variable (mm in the hour) / total: rain + snow (no snow in Jamaica)
    "rain",                         # liquid rain only (includes showers in archive API)
    "pressure_msl",                 # mean sea-level pressure; drops precede storms
    "surface_pressure",             # local pressure
    "cloud_cover",                  # total cloud cover %
    "cloud_cover_low",              # low clouds matter more for rain than high cirrus
    "cloud_cover_mid",
    "cloud_cover_high",
    "wind_speed_10m",               # wind at 10m above ground
    "wind_direction_10m",           # which way the wind is coming from
    "wind_gusts_10m",               # peak gusts
    "et0_fao_evapotranspiration",   # how much water is evaporating
    "vapour_pressure_deficit",      # atmospheric dryness; inverse correlates with rain
    "weather_code",                 # categorical WMO weather code (0 = clear, 95 = thunderstorm,
                                    # 80, 81, 82 = rain showers: slight, moderate, violent
                                    # 61, 63, 65 = rain: slight, moderate and heavy intensity
                                    # 51, 53, 55 = drizzle: light, moderate and dense intensity
    "sunshine_duration",            # seconds of sunshine in the hour
    "boundary_layer_height",        # PBL height; matters for convection
]


API_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_location(name:str, lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch hourly weather data for one location and date range.

    Open-Meteo accepts date ranges in YYYY-MM-DD format and returns JSON with parallel arrays:
    one timestamp array and one array per requested variable. We reshape that into a DataFrame.
    """

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "America/Jamaica",
    }

    print(f"Requesting {name}: {start_date} to {end_date}")
    response = requests.get(API_URL, params=params, timeout=120)
    response.raise_for_status()     # raise an exception if HTTP status != 200

    data =  response.json()

    # The 'hourly' key contains a dict where each variable is a list parallel to the 'time' list
    df = pd.DataFrame(data['hourly'])

    df['time'] = pd.to_datetime(df['time'])

    # Add location column so we can stack data from multiple locations
    df['location'] = name
    df['latitude'] = lat
    df['longitude'] = lon

    return df


def fetch_all_locations(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch all configured locations and concatenate into one DataFrame"""

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_data = []
    for name, coords in tqdm(LOCATIONS.items(), desc = "Locations"):
        df = fetch_location(name, coords['lat'], coords['lon'], start_date, end_date)
        all_data.append(df)
        time.sleep(1)

    return pd.concat(all_data, ignore_index=True)

def write_manifest(df: pd.DataFrame, output_path: Path, start: date, end: date) -> Path:
    with open(output_path,  "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()

    manifest = {
        "data_file": output_path.name,
        "file_sha256": file_hash,
        "file_size_bytes": output_path.stat().st_size,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "api_endpoint": API_URL,
        "snapshot_start_date": str(start),
        "snapshot_end_date": str(end),
        "locations": LOCATIONS,
        "hourly_variables": HOURLY_VARS,
        "row_count": len(df),
        "actual_time_range": {
            "min": str(df["time"].min()),
            "max": str(df["time"].max())
        },
        "row_count_per_location": df["location"].value_counts().to_dict(),
    }

    manifest_path = output_path.with_suffix(".manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path

if __name__ == "__main__":
    print(f"Fetching weather data from {SNAPSHOT_START} to {SNAPSHOT_END}")
    print(f"Locations: {list(LOCATIONS.keys())}")
    print(f"Variables per hour: {len(HOURLY_VARS)}")

    df = fetch_all_locations(str(SNAPSHOT_START), str(SNAPSHOT_END))

    # Filename encodes the snapshot for unambiguous identification
    output_name = f"weather_jamaica_{SNAPSHOT_START}_to_{SNAPSHOT_END}.parquet"
    output_path = RAW_DATA_DIR / output_name
    df.to_parquet(output_path, index = False)

    manifest_path = write_manifest(df, output_path, SNAPSHOT_START, SNAPSHOT_END)

    df.head(1000).to_csv(RAW_DATA_DIR / "weather_preview.csv", index = False)

    print(f"\nSaved {len(df):,} rows to {output_path.name}")
    print(f"Manifest: {manifest_path.name}")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")