"""Per-bot, per-regime performance report.

Reads enriched trades.csv, buckets each trade by regime metrics, and reports
win-rate / avg-PnL per (bot, regime). This is the data that the future Coach
will use to pick lineups.

Buckets are coarse on purpose — refined automatically once enough trades
accumulate per cell.
"""

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.expanduser("~/MuriTrading")
sys.path.insert(0, PROJECT_ROOT)

from src.jv2.config import TRADES_CSV


def trend_bucket(adx):
    if pd.isna(adx):
        return None
    if adx >= 25:
        return "trend"
    if adx <= 18:
        return "chop"
    return "mid"


def vol_bucket(atr_pct):
    if pd.isna(atr_pct):
        return None
    if atr_pct >= 2.5:
        return "high_vol"
    if atr_pct <= 1.0:
        return "low_vol"
    return "mid_vol"


def rsi_bucket(rsi):
    if pd.isna(rsi):
        return None
    if rsi >= 65:
        return "overbought"
    if rsi <= 35:
        return "oversold"
    return "neutral"


def main():
    df = pd.read_csv(TRADES_CSV, encoding="utf-8", encoding_errors="replace")
    print(f"Total trades: {len(df)}")
    tagged = df.dropna(subset=["regime_adx"])
    print(f"Regime-tagged: {len(tagged)} ({len(tagged)/max(len(df),1):.0%})\n")

    df = tagged.copy()
    df["base_id"] = df["bot_id"].str.rsplit("_", n=1).str[0]
    df["asset"]   = df["bot_id"].str.rsplit("_", n=1).str[1]
    df["trend"]   = df["regime_adx"].apply(trend_bucket)
    df["vol"]     = df["regime_atr_pct"].apply(vol_bucket)
    df["rsi_b"]   = df["regime_rsi"].apply(rsi_bucket)
    df["win"]     = (df["pnl"] > 0).astype(int)

    # ── Overall by bot ──
    print("="*78)
    print("OVERALL by bot (regime-tagged trades only)")
    print("="*78)
    overall = df.groupby("base_id").agg(
        n=("pnl", "size"), wr=("win", "mean"),
        avg_pnl=("pnl", "mean"), total=("pnl", "sum"),
    ).sort_values("total", ascending=False)
    overall["wr"] = (overall["wr"] * 100).round(0).astype(int).astype(str) + "%"
    overall[["avg_pnl", "total"]] = overall[["avg_pnl", "total"]].round(2)
    print(overall.to_string())

    # ── By bot × trend regime ──
    print("\n" + "="*78)
    print("BOT × TREND REGIME (n = trades, wr = win-rate, $ = total PnL)")
    print("="*78)
    pivot = df.groupby(["base_id", "trend"]).agg(
        n=("pnl", "size"), wr=("win", "mean"), total=("pnl", "sum"),
    ).reset_index()
    pivot["cell"] = pivot.apply(
        lambda r: f"n={r['n']:>2d} wr={r['wr']*100:>3.0f}% ${r['total']:+.1f}", axis=1)
    table = pivot.pivot(index="base_id", columns="trend", values="cell").fillna("—")
    cols = [c for c in ["chop", "mid", "trend"] if c in table.columns]
    print(table[cols].to_string())

    # ── By bot × RSI regime ──
    print("\n" + "="*78)
    print("BOT × RSI REGIME")
    print("="*78)
    pivot2 = df.groupby(["base_id", "rsi_b"]).agg(
        n=("pnl", "size"), wr=("win", "mean"), total=("pnl", "sum"),
    ).reset_index()
    pivot2["cell"] = pivot2.apply(
        lambda r: f"n={r['n']:>2d} wr={r['wr']*100:>3.0f}% ${r['total']:+.1f}", axis=1)
    table2 = pivot2.pivot(index="base_id", columns="rsi_b", values="cell").fillna("—")
    cols2 = [c for c in ["oversold", "neutral", "overbought"] if c in table2.columns]
    print(table2[cols2].to_string())

    # ── Standout cells (n>=4, edge in either direction) ──
    print("\n" + "="*78)
    print("STANDOUT CELLS (n>=4, |total PnL| highest)")
    print("="*78)
    standouts = []
    for bot in df["base_id"].unique():
        for trend in ["chop", "mid", "trend"]:
            sub = df[(df["base_id"] == bot) & (df["trend"] == trend)]
            if len(sub) >= 4:
                wr = sub["win"].mean()
                total = sub["pnl"].sum()
                standouts.append((bot, trend, len(sub), wr, total))
    standouts.sort(key=lambda x: abs(x[4]), reverse=True)
    for bot, trend, n, wr, total in standouts[:12]:
        marker = "★" if wr >= 0.6 and total > 0 else ("✗" if wr <= 0.4 and total < 0 else " ")
        print(f"  {marker} {bot:>20s} | trend={trend:>5s} | n={n:>2d} wr={wr*100:>3.0f}% ${total:+.2f}")

    print("\nNote: 35 SOL trades have NaN regime (no historical 4h data) and are excluded.")


if __name__ == "__main__":
    main()
