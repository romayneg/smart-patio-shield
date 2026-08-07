"""
Generates real test-set hours for the dashboard's history picker.
"""


import json
from pathlib import Path
import pandas as pd

PROCESSED = Path("patio_features.parquet")
OUT = Path("sample_hours.json")

MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def main():
    df = pd.read_parquet(PROCESSED)
    # test period only (most recent 15%), same chronological split as the models
    df = df.sort_values("time")
    test = df.iloc[int(len(df)*0.85):].copy()

    # pick a spread: some wet, some dry
    picks = []
    wet = test[test.wet_veranda == 1].sample(min(4, (test.wet_veranda==1).sum()), random_state=1)
    dry = test[test.wet_veranda == 0].sample(4, random_state=2)
    chosen = pd.concat([wet, dry]).sort_values("time")

    for _, r in chosen.iterrows():
        loc = r["location"]
        t = pd.to_datetime(r["time"])
        label = f"{'MoBay' if loc=='montego_bay' else 'Kingston'} · {t.day} {MONTHS[t.month-1]} {t.year} {t.hour:02d}:00"
        picks.append({
            "label": label,
            "actual": int(r["wet_veranda"]),
            "actual_detail": f"{r.get('precipitation',0):.1f} mm, gust {r.get('wind_gusts_10m',0):.0f} km/h",
            "inputs": {
                "location": loc,
                "hour": int(r["hour"]),
                "month": int(r["month"]),
                "relative_humidity": float(r["relative_humidity_2m"]),
                "cloud_cover_low": float(r["cloud_cover_low"]),
                "cloud_cover_mid": float(r["cloud_cover_mid"]),
                "cloud_cover_high": float(r["cloud_cover_high"]),
                "recent_rain_mm": float(r["precip_lag1"]),
                "recent_gust_kmh": float(r["gust_lag1"]),
                "pressure_falling": bool(r.get("pressure_msl_trend_3h", 0) < 0),
                "enso_phase": r.get("enso_phase", "neutral"),
            },
        })

    json.dump({"hours": picks}, open(OUT, "w"), indent=2)
    print(f"wrote {OUT} with {len(picks)} hours")

if __name__ == "__main__":
    main()
