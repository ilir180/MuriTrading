"""Backfill regime tags for existing trades.csv entries.

For each historical trade:
  1. Determine asset from bot_id suffix (_XRP / _BTC / _ETH / _SOL).
  2. Approximate entry time = exit_time - hold_candles * 4h.
  3. Find the 4H bar at-or-before entry time in raw OHLCV.
  4. Read precomputed indicators on that bar.
  5. Persist enriched trades.csv with regime_* columns.

Idempotent: if all regime_* columns already exist, exits early.
SOL trades get NaN regime values (no historical 4h CSV available yet).
Sentiment (fear_greed) is not available historically -> NaN for all backfilled rows.
"""

import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.expanduser("~/MuriTrading")
sys.path.insert(0, PROJECT_ROOT)

from src.features.build_features import add_indicators
from src.jv2.config import TRADES_CSV

ASSET_FILES = {
    "XRP": os.path.join(PROJECT_ROOT, "data", "raw", "XRP_USDT_4h.csv"),
    "BTC": os.path.join(PROJECT_ROOT, "data", "raw", "BTC_USDT_4h.csv"),
    "ETH": os.path.join(PROJECT_ROOT, "data", "raw", "ETH_USDT_4h.csv"),
    "SOL": os.path.join(PROJECT_ROOT, "data", "raw", "SOL_USDT_4h.csv"),
}

REGIME_COLS = [
    "regime_adx", "regime_rsi", "regime_bb_pos", "regime_bbw",
    "regime_atr_pct", "regime_chop", "regime_trend_consistency",
    "regime_fear_greed",
]


def asset_from_bot_id(bot_id: str) -> str:
    return bot_id.rsplit("_", 1)[1]


def load_indicators(path: str):
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    # Detect timestamp column
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


def lookup_regime(ind_df, entry_time):
    nan = float("nan")
    empty = {c: nan for c in REGIME_COLS}
    if ind_df is None or ind_df.empty:
        return empty
    idx = ind_df.index.searchsorted(entry_time, side="right") - 1
    if idx < 0 or idx >= len(ind_df):
        return empty
    row = ind_df.iloc[idx]
    price = float(row.get("close", 0) or 0)
    atr = float(row.get("4h_atr_14", 0) or 0)
    return {
        "regime_adx":               float(row.get("4h_adx", nan) or nan),
        "regime_rsi":               float(row.get("4h_rsi_14", nan) or nan),
        "regime_bb_pos":            float(row.get("4h_bb_pos", nan) or nan),
        "regime_bbw":               float(row.get("4h_bb_width", nan) or nan),
        "regime_atr_pct":           (atr / price * 100) if price > 0 else nan,
        "regime_chop":              float(row.get("4h_chop", nan) or nan),
        "regime_trend_consistency": float(row.get("4h_trend_consistency", nan) or nan),
        "regime_fear_greed":        nan,  # not retrievable historically
    }


def main():
    if not os.path.exists(TRADES_CSV):
        print(f"No trades file at {TRADES_CSV}, nothing to backfill.")
        return
    # Existing trades.csv may have non-utf-8 bytes (Mac → Windows handoff via signal reasoning text).
    trades = pd.read_csv(TRADES_CSV, encoding="utf-8", encoding_errors="replace")
    print(f"Loaded {len(trades)} trades from {TRADES_CSV}")

    if all(c in trades.columns for c in REGIME_COLS):
        print("All regime_* columns already present. Skipping (idempotent).")
        return

    cache = {}
    for asset, path in ASSET_FILES.items():
        ind = load_indicators(path)
        cache[asset] = ind
        if ind is not None:
            print(f"  {asset}: loaded {len(ind)} 4h bars, indicators ready")
        else:
            print(f"  {asset}: NO historical 4h data — regime will be NaN")

    enriched = []
    for _, t in trades.iterrows():
        bot_id = str(t["bot_id"])
        try:
            asset = asset_from_bot_id(bot_id)
        except IndexError:
            asset = None
        try:
            exit_time = pd.to_datetime(t["timestamp"], utc=True)
        except (ValueError, TypeError):
            enriched.append({c: float("nan") for c in REGIME_COLS})
            continue
        hold = int(t.get("hold_candles", 0) or 0)
        entry_time = exit_time - timedelta(hours=4 * hold)
        ind_df = cache.get(asset)
        enriched.append(lookup_regime(ind_df, entry_time))

    regime_df = pd.DataFrame(enriched)
    out = pd.concat([trades.reset_index(drop=True), regime_df], axis=1)

    backup = TRADES_CSV + ".pre_regime_backfill"
    if not os.path.exists(backup):
        # Backup the raw bytes, not the cleaned-up dataframe.
        with open(TRADES_CSV, "rb") as src, open(backup, "wb") as dst:
            dst.write(src.read())
        print(f"  Backed up original (raw bytes) to: {backup}")

    out.to_csv(TRADES_CSV, index=False, encoding="utf-8")
    print(f"\nWrote {len(out)} enriched trades. Added columns: {REGIME_COLS}")

    print("\n=== Coverage ===")
    for c in REGIME_COLS:
        non_null = out[c].notna().sum()
        print(f"  {c:30s} {non_null:>4d}/{len(out)} non-null")

    print("\n=== Sample (last 5 trades, key regime cols) ===")
    cols = ["timestamp", "bot_id", "reason", "pnl",
            "regime_adx", "regime_rsi", "regime_bb_pos", "regime_chop"]
    cols = [c for c in cols if c in out.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(out[cols].tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
