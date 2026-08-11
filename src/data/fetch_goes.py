"""
Fetches GOES-East satellite imagery patches for the Smart Patio Shield project.

For each hour in the snapshot window, downloads three ABI bands (C13 clean IR,
C09 mid-level water vapour, C02 red visible) from the public NOAA archive on AWS,
crops a 64x64-pixel (~128x128 km) patch around each city, and saves stacked
patches to per-day .npz files.

Design notes:
  - Files are opened over S3 without downloading fully; only the
    HDF5 chunks covering the crop window are transferred (a few MB instead of
    25-200 MB per file).
  - A day whose output file already exists is skipped, so the script
    can be stopped and restarted freely.
  - Fetches newest days first, so the val/test periods
    (most recent) are covered even if the run is cut short.
  - Band 02 (0.5 km native) is mean-downsampled to the 2 km IR grid so all
    channels align at 64x64.
  - Missing/corrupt satellite files are logged and stored as NaN channels
    rather than crashing the run.
"""

import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import s3fs
import xarray as xr
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "data" / "raw" / "goes"

# Same frozen window as the weather snapshot
SNAPSHOT_START = pd.Timestamp("2021-05-01")
SNAPSHOT_END = pd.Timestamp("2026-05-01")

LOCATIONS = {
    "kingston": (17.9714, -76.7945),
    "montego_bay": (18.47298, -77.92134),
}
BANDS = ["C13", "C09", "C02"]   # clean IR, water vapor, red visible
PATCH = 64                      # pixels on the 2 km grid (~128 km square)
GOES19_CUTOVER = pd.Timestamp("2025-04-07", tz="UTC")  # GOES-East handover

fs = s3fs.S3FileSystem(anon=True, default_block_size=1024 * 1024)

# Crop indices are constant per (bucket, band): cache after first computation
_index_cache: dict = {}


def bucket_for(ts_utc: pd.Timestamp) -> str:
    return "noaa-goes19" if ts_utc >= GOES19_CUTOVER else "noaa-goes16"


def hour_prefix(ts_utc: pd.Timestamp, band: str) -> str:
    b = bucket_for(ts_utc)
    return f"{b}/ABI-L2-CMIPF/{ts_utc.year}/{ts_utc.dayofyear:03d}/{ts_utc.hour:02d}/", b


def crop_indices(ds, band_key):
    """Compute (and cache) the integer pixel windows for both cities."""
    if band_key in _index_cache:
        return _index_cache[band_key]
    gp = ds["goes_imager_projection"].attrs
    h = gp["perspective_point_height"]
    proj = pyproj.Proj(proj="geos", h=h,
                       lon_0=gp["longitude_of_projection_origin"],
                       sweep=gp["sweep_angle_axis"])
    # Band 02 is 0.5 km (4x denser grid): crop 4x the pixels, downsample later
    factor = 4 if band_key[1] == "C02" else 1
    half = (PATCH * factor) // 2
    xs, ys = ds.x.values, ds.y.values
    windows = {}
    for loc, (lat, lon) in LOCATIONS.items():
        x_m, y_m = proj(lon, lat)
        xi = int(np.abs(xs - x_m / h).argmin())
        yi = int(np.abs(ys - y_m / h).argmin())
        windows[loc] = (slice(yi - half, yi + half), slice(xi - half, xi + half))
    _index_cache[band_key] = windows
    return windows


def fetch_band_patches(ts_utc: pd.Timestamp, band: str) -> dict:
    """Return {location: 64x64 float32 patch} for one band-hour, NaN on failure."""
    nan_patch = np.full((PATCH, PATCH), np.nan, dtype="float32")
    try:
        prefix, b = hour_prefix(ts_utc, band)
        files = [f for f in fs.ls(prefix) if f"-M6{band}_" in f or f"{band}_" in f]
        files = [f for f in files if band in f]
        if not files:
            return {loc: nan_patch for loc in LOCATIONS}
        # Lazy open: only the chunks we slice get downloaded
        ds = xr.open_dataset(fs.open(files[0]), engine="h5netcdf")
        windows = crop_indices(ds, (b, band))
        out = {}
        for loc, (ys, xs) in windows.items():
            patch = ds["CMI"].isel(y=ys, x=xs).values.astype("float32")
            if band == "C02":  # 0.5 km -> 2 km mean downsample
                p = patch[: PATCH * 4, : PATCH * 4]
                patch = p.reshape(PATCH, 4, PATCH, 4).mean(axis=(1, 3))
            out[loc] = patch
        ds.close()
        return out
    except Exception as e:
        warnings.warn(f"{ts_utc} {band}: {e}")
        return {loc: nan_patch for loc in LOCATIONS}


def fetch_day(day_local: pd.Timestamp) -> dict:
    """
    Fetch all 24 hours x 3 bands x 2 cities for one day, parallelized across
    every (hour, band) job at once rather than hour-by-hour.
    """
    jobs = [(h, b) for h in range(24) for b in BANDS]      # 24 x 3 = 72 jobs
    results = {}                                            # (hour, band) -> {loc: patch}
    with ThreadPoolExecutor(max_workers=24) as ex:
        futures = {}
        for hour, band in jobs:
            ts_local = day_local + pd.Timedelta(hours=hour)
            ts_utc = (ts_local + pd.Timedelta(hours=5)).tz_localize("UTC")
            futures[ex.submit(fetch_band_patches, ts_utc, band)] = (hour, band)
        for fut in futures:
            results[futures[fut]] = fut.result()

    arrays, nan_channels = {}, 0
    for hour in range(24):
        ts_local = day_local + pd.Timedelta(hours=hour)
        for loc in LOCATIONS:
            stack = np.stack([results[(hour, b)][loc] for b in BANDS])   # (3, 64, 64)
            nan_channels += int(sum(np.isnan(s).all() for s in stack))
            arrays[f"{loc}|{ts_local.isoformat()}"] = stack
    total = 24 * len(LOCATIONS) * len(BANDS)
    if nan_channels > total * 0.5:
        raise RuntimeError(f"{nan_channels}/{total} channels empty — treating day as failed")
    return arrays


def write_manifest(days_done: int, days_failed: list):
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_start": str(SNAPSHOT_START.date()),
        "snapshot_end": str(SNAPSHOT_END.date()),
        "bands": BANDS,
        "patch_pixels": PATCH,
        "locations": LOCATIONS,
        "channel_order": BANDS,
        "day_files_present": days_done,
        "days_with_errors": days_failed[:50],
        "note": "C02 downsampled from 0.5km to 2km grid; missing scans stored as NaN.",
    }
    with open(OUT_DIR / "goes_patches.manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(SNAPSHOT_START, SNAPSHOT_END - pd.Timedelta(days=1), freq="D")
    days = days[::-1]   # newest first: val/test imagery lands before train history

    failed = []
    done = 0
    for day in tqdm(days, desc="Days"):
        out_path = OUT_DIR / f"goes_{day.date()}.npz"
        if out_path.exists():
            done += 1
            continue
        try:
            arrays = fetch_day(day)
            np.savez_compressed(out_path, **arrays)
            done += 1
        except KeyboardInterrupt:
            print("\nInterrupted — progress saved; re-run to resume.")
            break
        except Exception as e:
            failed.append(str(day.date()))
            warnings.warn(f"Day {day.date()} failed: {e}")

    write_manifest(done, failed)
    print(f"\nDay files on disk: {done}/{len(days)}  (errors: {len(failed)})")