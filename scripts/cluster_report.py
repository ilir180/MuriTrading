"""Per-cluster bot performance report from LIVE trades only.

Live regime-tagged trades are the single source of truth for promotion /
demotion decisions. Counterfactual replay numbers are deliberately not
included — they have lookahead bias and can't drive real deployment.

Output:
  1. Cluster taxonomy (model summary, no PnL).
  2. Per-(cluster, bot) cells from live trades.csv: n, win-rate, total PnL.
  3. Sample-size warnings: cells with n < 20 are not actionable yet.

Usage:
    python scripts/cluster_report.py
"""

import os
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.expanduser("~/MuriTrading")
sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.jv2.config import TRADES_CSV
from src.jv2.regime_clusterer import RegimeClusterer

PROMOTION_MIN_N = 20      # below this, the cell is statistically meaningless
PROMOTION_MIN_WR = 0.60   # threshold for "promote" recommendation
DEMOTION_MAX_WR = 0.35    # threshold for "demote" recommendation


def main():
    rc = RegimeClusterer.load()
    if rc.kmeans is None:
        print("ERROR: cluster model not found. Run scripts/fit_regime_clusters.py first.")
        sys.exit(1)

    print("=" * 78)
    print("REGIME CLUSTER TAXONOMY (structural — no PnL)")
    print("=" * 78)
    for cid in range(rc.meta.k):
        c = rc.meta.centroids[cid]
        print(f"  [{cid}] {rc.meta.descriptions[cid]:>40s}  "
              f"({rc.meta.fractions[cid]*100:5.1f}% of historical bars)")
        print(f"        adx={c[0]:5.1f}  rsi={c[1]:5.1f}  bb_pos={c[2]:.2f}  "
              f"bbw={c[3]:.4f}  atr%={c[4]:.2f}  chop={c[5]:.2f}  tc={c[6]:+.2f}")

    df = pd.read_csv(TRADES_CSV, encoding="utf-8", encoding_errors="replace")
    df = df.dropna(subset=["regime_cluster"])
    df = df[df["regime_cluster"] >= 0].copy()
    df["regime_cluster"] = df["regime_cluster"].astype(int)
    df["base_id"] = df["bot_id"].str.rsplit("_", n=1).str[0]
    df["win"] = (df["pnl"] > 0).astype(int)

    print()
    print("=" * 78)
    print(f"LIVE TRADES BY CLUSTER (source: trades.csv, {len(df)} regime+cluster-tagged)")
    print("=" * 78)
    if len(df) == 0:
        print("  No regime-clustered live trades yet. Bot needs to run.")
        return

    bots = sorted(df["base_id"].unique())
    clusters = sorted(df["regime_cluster"].unique())

    # Per-cluster summary
    print(f"\n{'cluster':>40s}     n   wr  total  unique-bots")
    for cid in clusters:
        sub = df[df["regime_cluster"] == cid]
        n = len(sub)
        wr = sub["win"].mean()
        total = sub["pnl"].sum()
        ubots = sub["base_id"].nunique()
        print(f"  [{cid}] {rc.describe(cid):>34s}  "
              f"{n:>4d} {wr*100:>4.0f}% ${total:+7.2f}  {ubots} bots")

    # The matrix
    print()
    print("=" * 78)
    print("BOT × CLUSTER (n / wr / total). '-' means no trades. n<20 is NOT actionable.")
    print("=" * 78)
    cell_rows = []
    for bot in bots:
        row = {"bot": bot}
        for cid in clusters:
            sub = df[(df["base_id"] == bot) & (df["regime_cluster"] == cid)]
            if len(sub) == 0:
                row[f"c{cid}"] = "-"
            else:
                n = len(sub)
                wr = sub["win"].mean() * 100
                total = sub["pnl"].sum()
                marker = ""
                if n >= PROMOTION_MIN_N:
                    if wr / 100 >= PROMOTION_MIN_WR:
                        marker = " ★"
                    elif wr / 100 <= DEMOTION_MAX_WR:
                        marker = " ✗"
                row[f"c{cid}"] = f"{n:>2d}/{wr:>2.0f}%/${total:+5.2f}{marker}"
        cell_rows.append(row)
    matrix = pd.DataFrame(cell_rows).set_index("bot")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(matrix.to_string())

    # Promotion / demotion recommendations
    print()
    print("=" * 78)
    print(f"RECOMMENDATIONS (n>={PROMOTION_MIN_N}, "
          f"promote@WR>={PROMOTION_MIN_WR*100:.0f}%, demote@WR<={DEMOTION_MAX_WR*100:.0f}%)")
    print("=" * 78)
    promote, demote, watching = [], [], []
    for bot in bots:
        for cid in clusters:
            sub = df[(df["base_id"] == bot) & (df["regime_cluster"] == cid)]
            n = len(sub)
            if n == 0:
                continue
            wr = sub["win"].mean()
            total = sub["pnl"].sum()
            if n >= PROMOTION_MIN_N and wr >= PROMOTION_MIN_WR:
                promote.append((bot, cid, n, wr, total))
            elif n >= PROMOTION_MIN_N and wr <= DEMOTION_MAX_WR:
                demote.append((bot, cid, n, wr, total))
            else:
                watching.append((bot, cid, n, wr, total))

    if promote:
        print("\n  ★ PROMOTE candidates:")
        for bot, cid, n, wr, total in sorted(promote, key=lambda x: -x[4]):
            print(f"      {bot} in cluster [{cid}] {rc.describe(cid)}: "
                  f"n={n} wr={wr*100:.0f}% total=${total:+.2f}")
    else:
        print("\n  ★ PROMOTE candidates: NONE — no cell has both n>=20 and WR>=60%.")

    if demote:
        print("\n  ✗ DEMOTE candidates:")
        for bot, cid, n, wr, total in sorted(demote, key=lambda x: x[4]):
            print(f"      {bot} in cluster [{cid}] {rc.describe(cid)}: "
                  f"n={n} wr={wr*100:.0f}% total=${total:+.2f}")
    else:
        print("\n  ✗ DEMOTE candidates: none with sufficient sample.")

    n_actionable = sum(1 for _, _, n, _, _ in (promote + demote))
    n_watching = len(watching)
    print(f"\n  Actionable cells: {n_actionable}.  Watching (n<{PROMOTION_MIN_N}): {n_watching}.")

    if n_actionable == 0:
        print("\n  Reading: not enough live data for any deployment decision yet.")
        print(f"  At ~5-15 trades/bot/week, expect first actionable cells in ~4-8 weeks.")


if __name__ == "__main__":
    main()
