"""Run counterfactual replay over all assets that have raw 4h OHLCV.

For each asset: reset all 8 bots' state, walk through historical 4h bars,
emit synthetic trades. Combine into one big counterfactual_trades.csv that
can be analysed with the same regime_report.py used on live trades.
"""

import os
import sys
import time
from datetime import timedelta

import pandas as pd

PROJECT_ROOT = os.path.expanduser("~/MuriTrading")
sys.path.insert(0, PROJECT_ROOT)

# Force UTF-8 for Windows console
import io
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.features.build_features import add_indicators
from src.jv2.bots import create_all_bots
from src.jv2.config import SYMBOLS, JV2_DIR
from src.jv2.models import TradeRecord
from src.jv2.replay import replay_asset

OUT_FILE = os.path.join(JV2_DIR, "counterfactual_trades.csv")

ASSET_FILES = {
    "XRP/USDT": (
        os.path.join(PROJECT_ROOT, "data", "raw", "XRP_USDT_1h.csv"),
        os.path.join(PROJECT_ROOT, "data", "raw", "XRP_USDT_4h.csv"),
        os.path.join(PROJECT_ROOT, "data", "raw", "XRP_USDT_1d.csv"),
    ),
    "BTC/USDT": (
        os.path.join(PROJECT_ROOT, "data", "raw", "BTC_USDT_1h.csv"),
        os.path.join(PROJECT_ROOT, "data", "raw", "BTC_USDT_4h.csv"),
        os.path.join(PROJECT_ROOT, "data", "raw", "BTC_USDT_1d.csv"),
    ),
    "ETH/USDT": (
        os.path.join(PROJECT_ROOT, "data", "raw", "ETH_USDT_1h.csv"),
        os.path.join(PROJECT_ROOT, "data", "raw", "ETH_USDT_4h.csv"),
        os.path.join(PROJECT_ROOT, "data", "raw", "ETH_USDT_1d.csv"),
    ),
}


def load_ohlcv(path: str):
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
    return df


def main():
    print("=" * 70)
    print("COUNTERFACTUAL REPLAY")
    print("=" * 70)

    all_bots = create_all_bots()
    bots_by_symbol = {}
    for b in all_bots:
        bots_by_symbol.setdefault(b.symbol, []).append(b)

    all_trades = []
    for symbol, (p1h, p4h, p1d) in ASSET_FILES.items():
        if symbol not in bots_by_symbol:
            continue
        print(f"\n--- {symbol} ---")
        df_1h = load_ohlcv(p1h)
        df_4h = load_ohlcv(p4h)
        df_1d = load_ohlcv(p1d)
        if df_4h is None:
            print(f"  No 4h data, skipping")
            continue
        print(f"  Loaded {len(df_4h)} 4h bars "
              f"({df_4h.index[0]} -> {df_4h.index[-1]})")

        t0 = time.time()
        df_1h = add_indicators(df_1h, "1h_") if df_1h is not None else None
        df_4h_ind = add_indicators(df_4h, "4h_")
        df_1d = add_indicators(df_1d, "1d_") if df_1d is not None else None
        print(f"  Indicators built in {time.time()-t0:.1f}s")

        t0 = time.time()
        sym_bots = bots_by_symbol[symbol]
        trades = replay_asset(sym_bots, symbol, df_1h, df_4h_ind, df_1d)
        print(f"  Replay produced {len(trades)} trades in {time.time()-t0:.1f}s")

        all_trades.extend(trades)

    print(f"\n=== TOTAL: {len(all_trades)} synthetic trades ===")

    if not all_trades:
        print("No trades produced. Check data and bot logic.")
        return

    rows = [_record_to_dict(t) for t in all_trades]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_FILE, index=False)
    print(f"Wrote {OUT_FILE}")

    # Summary
    print(f"\n=== Per-bot summary (counterfactual) ===")
    df["base_id"] = df["bot_id"].str.rsplit("_", n=1).str[0]
    df["asset"]   = df["bot_id"].str.rsplit("_", n=1).str[1]
    df["win"]     = (df["pnl"] > 0).astype(int)
    summary = df.groupby("base_id").agg(
        n=("pnl", "size"), wr=("win", "mean"),
        avg_pnl=("pnl", "mean"), total=("pnl", "sum"),
    ).sort_values("total", ascending=False)
    summary["wr"] = (summary["wr"] * 100).round(0).astype(int).astype(str) + "%"
    summary[["avg_pnl", "total"]] = summary[["avg_pnl", "total"]].round(2)
    print(summary.to_string())


def _record_to_dict(t: TradeRecord) -> dict:
    return {
        "timestamp": t.timestamp, "bot_id": t.bot_id, "direction": t.direction,
        "entry_price": t.entry_price, "exit_price": t.exit_price,
        "size_usd": t.size_usd, "pnl": t.pnl, "net_return_pct": t.net_return_pct,
        "reason": t.reason, "hold_candles": t.hold_candles,
        "bot_capital_after": t.bot_capital_after,
        "regime_adx": t.regime_adx, "regime_rsi": t.regime_rsi,
        "regime_bb_pos": t.regime_bb_pos, "regime_bbw": t.regime_bbw,
        "regime_atr_pct": t.regime_atr_pct, "regime_chop": t.regime_chop,
        "regime_trend_consistency": t.regime_trend_consistency,
        "regime_fear_greed": t.regime_fear_greed,
        "regime_cluster": t.regime_cluster,
    }


if __name__ == "__main__":
    main()
