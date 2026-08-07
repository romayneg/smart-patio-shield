"""
Audits the downloaded GOES patch archive for completeness and integrity, and
optionally deletes incomplete day-files so they are re-fetched on the next
`python -m src.data.fetch_goes` run.

Two problems are detected:
  Missing days  : a day in the snapshot window with no .npz at all. (While the
                  main download is still running, "missing" just means "not yet
                  fetched"; this figure is only meaningful once it completes.)
  Partial days  : a .npz that exists but has blank (all-NaN) channels from scans
                  that failed to download, or fewer patches than expected (a
                  truncated write), or that won't open at all (corrupt).

Why a separate tool: fetch_goes skips any day whose file already exists, so it
will not re-fetch a partial day on its own. This script deletes partial days so
that the fetcher's normal resume logic re-downloads them.

Genuine gaps vs transient failures: some missing scans are permanent (the
archive truly lacks that band-hour) and re-fetching returns blank again; others
are transient (a dropped connection) and re-fetching recovers them. You cannot
tell which without trying.

Workflow: audit -> clean -> re-run fetcher ONCE -> audit again.
Whatever is still blank after one clean re-fetch is a genuine gap.

Usage (from project root, ideally AFTER the main download finishes):
  python -m src.data.audit_goes                      # report only
  python -m src.data.audit_goes --report audit.json  # + save full report
  python -m src.data.audit_goes --clean              # delete partials >= 10% affected
  python -m src.data.audit_goes --clean --min-blank-frac 0.0   # re-fetch EVERY partial once
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GOES_DIR = PROJECT_ROOT / "data" / "raw" / "goes"

# Must match fetch_goes.py
SNAPSHOT_START = pd.Timestamp("2021-05-01")
SNAPSHOT_END = pd.Timestamp("2026-05-01")
EXPECTED_PATCHES_PER_DAY = 48   # 24 hours x 2 cities


def audit():
    expected = pd.date_range(SNAPSHOT_START, SNAPSHOT_END - pd.Timedelta(days=1), freq="D")
    present, missing = [], []
    for day in expected:
        ((present if (GOES_DIR / f"goes_{day.date()}.npz").exists() else missing)).append(day)

    partial = {}
    tot = blank_any_tot = blank_all_tot = 0
    for day in present:
        path = GOES_DIR / f"goes_{day.date()}.npz"
        try:
            with np.load(path) as d:
                keys = d.files
                blank_any = blank_all = 0
                for k in keys:
                    arr = d[k]                                   # (3, 64, 64)
                    n = sum(np.isnan(arr[c]).all() for c in range(arr.shape[0]))
                    if n > 0: blank_any += 1
                    if n == arr.shape[0]: blank_all += 1
                tot += len(keys); blank_any_tot += blank_any; blank_all_tot += blank_all
                short = max(EXPECTED_PATCHES_PER_DAY - len(keys), 0)
                if blank_any > 0 or short > 0:
                    partial[str(day.date())] = {
                        "patches": len(keys), "blank_any_channel": blank_any,
                        "blank_all_channels": blank_all, "short_count": short,
                        "frac_affected": round(blank_any / max(len(keys), 1), 3),
                    }
        except Exception as e:
            partial[str(day.date())] = {"error": str(e)[:60], "frac_affected": 1.0,
                                        "patches": 0, "short_count": EXPECTED_PATCHES_PER_DAY,
                                        "blank_any_channel": 0, "blank_all_channels": 0}

    stats = {
        "expected_days": len(expected), "present_days": len(present),
        "missing_days": len(missing), "partial_days": len(partial),
        "total_patches": tot, "patches_blank_any": blank_any_tot,
        "patches_blank_all": blank_all_tot,
        "pct_blank_any": round(100 * blank_any_tot / max(tot, 1), 3),
        "pct_blank_all": round(100 * blank_all_tot / max(tot, 1), 3),
    }
    return missing, partial, stats


def print_report(missing, partial, stats):
    print("=" * 60); print("GOES ARCHIVE AUDIT"); print("=" * 60)
    print(f"Expected days : {stats['expected_days']}")
    print(f"Present       : {stats['present_days']}")
    print(f"Missing       : {stats['missing_days']}"
          + ("   (includes not-yet-downloaded days if fetch is still running)"
             if stats['missing_days'] else ""))
    print(f"Partial       : {stats['partial_days']}")
    print(f"\nPatches       : {stats['total_patches']:,}")
    print(f"  >=1 blank   : {stats['patches_blank_any']:,} ({stats['pct_blank_any']}%)")
    print(f"  fully blank : {stats['patches_blank_all']:,} ({stats['pct_blank_all']}%)")
    if partial:
        print("\nPartial days (worst first):")
        for day, i in sorted(partial.items(), key=lambda x: -x[1]['frac_affected'])[:25]:
            if "error" in i:
                print(f"  {day}: UNREADABLE ({i['error']})")
            else:
                extra = f", short {i['short_count']}" if i['short_count'] else ""
                print(f"  {day}: {i['blank_any_channel']}/{i['patches']} affected "
                      f"({i['frac_affected']*100:.0f}%{extra})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean", action="store_true")
    ap.add_argument("--min-blank-frac", type=float, default=0.10)
    ap.add_argument("--report", type=str, default=None)
    args = ap.parse_args()

    missing, partial, stats = audit()
    print_report(missing, partial, stats)

    if args.report:
        with open(args.report, "w") as f:
            json.dump({"stats": stats, "missing_days": [str(d.date()) for d in missing],
                       "partial_days": partial}, f, indent=2)
        print(f"\nWrote {args.report}")

    if args.clean:
        to_delete = [day for day, i in partial.items()
                     if i["frac_affected"] >= args.min_blank_frac
                     or i.get("short_count", 0) > 0 or "error" in i]
        print(f"\n--clean: deleting {len(to_delete)} partial day-files "
              f"(>= {args.min_blank_frac*100:.0f}% affected, or short/corrupt):")
        for day in to_delete:
            p = GOES_DIR / f"goes_{day}.npz"
            if p.exists(): p.unlink(); print(f"  deleted {p.name}")
        print("\nNow re-run:  python -m src.data.fetch_goes")
        print("Then re-run this audit; anything still blank is a genuine archive gap.")