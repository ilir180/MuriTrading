"""Tag every trade in a CSV with its regime_cluster (k-means cluster id).

Idempotent: re-running just recomputes the cluster column. No PnL involved.
Usage:
    python scripts/assign_clusters.py [path_to_trades_csv ...]

Defaults to live trades.csv + counterfactual_trades.csv.
"""

import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.expanduser("~/MuriTrading")
sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.jv2.config import TRADES_CSV, JV2_DIR
from src.jv2.regime_clusterer import RegimeClusterer, REGIME_FEATURES

DEFAULT_FILES = [
    TRADES_CSV,
    os.path.join(JV2_DIR, "counterfactual_trades.csv"),
]

# Map csv column names -> internal regime feature names
COL_MAP = {
    "regime_adx": "adx",
    "regime_rsi": "rsi",
    "regime_bb_pos": "bb_pos",
    "regime_bbw": "bbw",
    "regime_atr_pct": "atr_pct",
    "regime_chop": "chop",
    "regime_trend_consistency": "trend_consistency",
}


def assign_clusters_in_file(path: str, rc: RegimeClusterer):
    if not os.path.exists(path):
        print(f"  {path}: not found, skipped")
        return
    df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
    print(f"  {path}: {len(df)} rows")

    # Build regime dicts from the existing columns
    regimes = []
    for _, row in df.iterrows():
        r = {}
        for csv_col, feat_name in COL_MAP.items():
            v = row.get(csv_col)
            r[feat_name] = float(v) if pd.notna(v) else float("nan")
        regimes.append(r)
    cluster_ids = rc.assign_many(regimes)

    df["regime_cluster"] = cluster_ids
    df.to_csv(path, index=False, encoding="utf-8")

    # Coverage
    n_assigned = (df["regime_cluster"] >= 0).sum()
    print(f"    assigned cluster to {n_assigned}/{len(df)} trades")
    if n_assigned > 0:
        dist = df["regime_cluster"].value_counts().sort_index()
        for cid, n in dist.items():
            if cid < 0:
                continue
            label = rc.describe(int(cid))
            print(f"      [{cid}] {label}: {n} trades")


def main():
    rc = RegimeClusterer.load()
    if rc.kmeans is None:
        print("ERROR: regime cluster model not found. Run scripts/fit_regime_clusters.py first.")
        sys.exit(1)
    print(f"Loaded clusterer (k={rc.meta.k}, {rc.meta.fit_n} training rows)")

    files = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_FILES
    for f in files:
        assign_clusters_in_file(f, rc)

    print("\nDone.")


if __name__ == "__main__":
    main()
