"""Fit the regime clusterer on all available historical 4H OHLCV.

Uses ONLY structural features — no PnL, no trade outcomes. The fitted model
defines a fixed taxonomy of market states. Per-cluster bot performance is
tracked separately, from live regime-tagged trades only (not from this fit).

Run once to bootstrap. Refit periodically (monthly?) as more history accrues
or if the market regime distribution drifts meaningfully.
"""

import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.expanduser("~/MuriTrading")
sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.features.build_features import add_indicators
from src.jv2.regime_clusterer import REGIME_FEATURES, fit

ASSET_4H_FILES = [
    os.path.join(PROJECT_ROOT, "data", "raw", "XRP_USDT_4h.csv"),
    os.path.join(PROJECT_ROOT, "data", "raw", "BTC_USDT_4h.csv"),
    os.path.join(PROJECT_ROOT, "data", "raw", "ETH_USDT_4h.csv"),
]

K = 6  # number of clusters


def load_with_indicators(path: str):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    elif "timestamp" in df.columns:
        df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    else:
        return None
    df = df.set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df = add_indicators(df, "4h_")
    return df


def df_to_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pull the regime feature columns from an indicator-enriched 4H df."""
    out = pd.DataFrame(index=df.index)
    out["adx"] = df.get("4h_adx")
    out["rsi"] = df.get("4h_rsi_14")
    out["bb_pos"] = df.get("4h_bb_pos")
    out["bbw"] = df.get("4h_bb_width")
    # atr% relative to close
    atr = df.get("4h_atr_14")
    close = df["close"]
    out["atr_pct"] = (atr / close * 100) if atr is not None else np.nan
    out["chop"] = df.get("4h_chop")
    out["trend_consistency"] = df.get("4h_trend_consistency")
    out = out[REGIME_FEATURES]   # enforce order
    return out


def main():
    print("=" * 70)
    print("FIT REGIME CLUSTERS (k=%d)" % K)
    print("=" * 70)

    feature_frames = []
    fit_min_ts = None
    fit_max_ts = None

    for path in ASSET_4H_FILES:
        asset = os.path.basename(path).split("_")[0]
        df = load_with_indicators(path)
        if df is None:
            print(f"  {asset}: no data, skipped")
            continue
        feats = df_to_features(df).dropna()
        feature_frames.append(feats)
        ts_min = feats.index.min().isoformat()
        ts_max = feats.index.max().isoformat()
        if fit_min_ts is None or ts_min < fit_min_ts:
            fit_min_ts = ts_min
        if fit_max_ts is None or ts_max > fit_max_ts:
            fit_max_ts = ts_max
        print(f"  {asset}: {len(feats)} usable bars ({ts_min[:10]} - {ts_max[:10]})")

    if not feature_frames:
        print("\nNo data — nothing to fit.")
        sys.exit(1)

    full = pd.concat(feature_frames, axis=0)
    print(f"\nTotal training rows: {len(full)} (across {len(feature_frames)} assets)")

    arr = full.values
    rc = fit(arr, k=K, fit_range=(fit_min_ts, fit_max_ts))
    rc.save()

    print(f"\nFit inertia: {rc.meta.inertia:.0f}")
    print(f"\n=== Cluster taxonomy ===")
    for cid in range(K):
        c = rc.meta.centroids[cid]
        frac = rc.meta.fractions[cid]
        print(f"  [{cid}] {rc.meta.descriptions[cid]:>40s}  "
              f"({frac*100:5.1f}% of bars)")
        print(f"        adx={c[0]:5.1f}  rsi={c[1]:5.1f}  bb_pos={c[2]:.2f}  "
              f"bbw={c[3]:.4f}  atr%={c[4]:.2f}  chop={c[5]:.2f}  tc={c[6]:+.2f}")

    print(f"\nSaved model + scaler + meta to: ~/MuriTrading/models/regime/")


if __name__ == "__main__":
    main()
