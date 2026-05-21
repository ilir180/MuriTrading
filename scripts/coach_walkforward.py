"""Walk-forward validation of the Coach.

For every chronological fold of the counterfactual trade history, the Coach is
trained on prior folds and its decisions are applied to the held-out fold to
simulate "what would have happened with Coach in charge". We compare baseline
PnL (no Coach) vs coached PnL (Coach adjustments applied per trade).

Adjustments per trade in the held-out fold:
  - disable                  -> skip trade (PnL = 0)
  - invert                   -> flip PnL sign (signal was flipped, so outcome flips)
  - demote (cap_mult 0.6)    -> scale PnL by cap_mult * lev_mult
  - keep (cap_mult 1.0)      -> baseline
  - promote (cap_mult 1.3)   -> scale by cap_mult * lev_mult
  - champion (cap_mult 1.6)  -> scale by cap_mult * lev_mult (max impact)
  - regime_blacklist hit     -> skip trade (PnL = 0)

Parallelism: each fold is computed in its own process. The i7 has 8 cores —
this completes in seconds.

Usage:
    python scripts/coach_walkforward.py
    python scripts/coach_walkforward.py --folds 8
"""

import os
import sys
import csv
import argparse
from collections import defaultdict
from datetime import datetime
import multiprocessing as mp

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.expanduser("~/MuriTrading")
sys.path.insert(0, PROJECT_ROOT)

from src.jv2.coach import (
    _aggregate, _decide_cell, CellStats, CAP_MULT, LEV_MULT,
)


def _load(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _split_folds(trades, n_folds):
    """Sort by timestamp, split into chronological folds of ~equal size."""
    trades = sorted(trades, key=lambda r: r.get("timestamp", ""))
    n = len(trades)
    if n == 0:
        return []
    fold_size = n // n_folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end = (i + 1) * fold_size if i < n_folds - 1 else n
        folds.append(trades[start:end])
    return folds


def _apply_coach_to_trade(trade, decisions, action_lev_lookup):
    """Return (baseline_pnl, coached_pnl) for one trade."""
    pnl = float(trade.get("pnl", 0) or 0)
    bid = trade.get("bot_id")
    cluster = int(float(trade.get("regime_cluster", -1) or -1))

    d = decisions.get(bid)
    if d is None:
        return pnl, pnl  # no coach opinion -> baseline

    if d.action == "disable":
        return pnl, 0.0

    # regime blacklist drops the trade
    if cluster in d.regime_blacklist:
        return pnl, 0.0
    if d.regime_whitelist is not None and cluster not in d.regime_whitelist:
        return pnl, 0.0

    cap_mult = d.capital_multiplier
    lev_mult = d.leverage_multiplier
    # Confidence-scaled leverage (mirror of get_cell_directive logic)
    conf = d.confidence
    scaled_lev = 1.0 + (lev_mult - 1.0) * conf
    scaled_lev = max(0.5, min(2.5, scaled_lev))

    coached = pnl * cap_mult * scaled_lev

    # Invert means the bot would have taken the opposite trade.
    # Net outcome ≈ -pnl (under symmetric SL/TP — proxy).
    # However, if the bot was ALREADY inverted in this data window, flipping
    # invert again returns to original direction. Since stats came from the
    # post-inversion window, "invert: True" in the decision means "stay
    # inverted" if the data was inverted, which is no flip.
    # For the walk-forward we treat decision.invert as the absolute setting
    # and assume the trade pnl reflects "as it happened". So if the Coach
    # decided to flip mid-stream, expected outcome flips:
    was_inverted = bool(d.stats.get("was_inverted", False))
    if d.invert != was_inverted:
        # flip the outcome
        coached = -coached

    return pnl, coached


def _run_fold(args):
    fold_idx, training_trades, test_trades = args

    # Train Coach on prior trades
    train_stats = _aggregate(training_trades)
    bot_ids = set(train_stats.keys()) | set(t.get("bot_id") for t in test_trades)
    decisions = {}
    for bid in bot_ids:
        live = train_stats.get(bid, CellStats(bot_id=bid))
        decisions[bid] = _decide_cell(bid, live, None)

    # Apply to test fold
    base_total = 0.0
    coach_total = 0.0
    base_wr_w, base_wr_n = 0, 0
    coach_wr_w, coach_wr_n = 0, 0
    by_action = defaultdict(lambda: {"n": 0, "base": 0.0, "coach": 0.0})

    for t in test_trades:
        bp, cp = _apply_coach_to_trade(t, decisions, None)
        base_total += bp
        coach_total += cp
        if bp != 0:
            base_wr_n += 1
            if bp > 0:
                base_wr_w += 1
        if cp != 0:
            coach_wr_n += 1
            if cp > 0:
                coach_wr_w += 1
        act = decisions.get(t.get("bot_id"))
        a = act.action if act else "keep"
        by_action[a]["n"] += 1
        by_action[a]["base"] += bp
        by_action[a]["coach"] += cp

    return {
        "fold": fold_idx,
        "n_train": len(training_trades),
        "n_test": len(test_trades),
        "ts_start": test_trades[0]["timestamp"] if test_trades else "",
        "ts_end": test_trades[-1]["timestamp"] if test_trades else "",
        "base_total": round(base_total, 2),
        "coach_total": round(coach_total, 2),
        "delta": round(coach_total - base_total, 2),
        "base_wr": round(base_wr_w / base_wr_n, 3) if base_wr_n else 0,
        "coach_wr": round(coach_wr_w / coach_wr_n, 3) if coach_wr_n else 0,
        "by_action": {k: {"n": v["n"], "delta": round(v["coach"] - v["base"], 2)}
                      for k, v in by_action.items()},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--folds", type=int, default=8)
    p.add_argument("--data", default=os.path.join(
        PROJECT_ROOT, "data", "bot", "jv2", "counterfactual_trades.csv"))
    p.add_argument("--parallel", action="store_true", default=True)
    p.add_argument("--serial", action="store_true", help="disable multiprocessing")
    args = p.parse_args()

    trades = _load(args.data)
    print(f"Loaded {len(trades)} counterfactual trades")
    folds = _split_folds(trades, args.folds)
    print(f"Split into {len(folds)} chronological folds (~{len(folds[0]) if folds else 0} trades each)")

    # Build (fold_idx, training, test) jobs — first fold has no training, skip it
    jobs = []
    for i, fold in enumerate(folds):
        if i == 0:
            continue  # not enough training data
        training = []
        for j in range(i):
            training.extend(folds[j])
        jobs.append((i, training, fold))

    start = datetime.now()
    if args.serial or not args.parallel:
        results = [_run_fold(j) for j in jobs]
    else:
        # Use 1 worker per fold, capped at CPU count - 1
        n_workers = min(len(jobs), max(1, mp.cpu_count() - 1))
        print(f"Running {len(jobs)} folds in parallel on {n_workers} workers...")
        with mp.Pool(n_workers) as pool:
            results = pool.map(_run_fold, jobs)
    elapsed = (datetime.now() - start).total_seconds()
    print(f"Elapsed: {elapsed:.2f}s\n")

    # Print
    print(f"{'Fold':>4}  {'Period':<22} {'Trades':>6}  {'Base $':>9}  {'Coach $':>9}  {'Delta':>9}  {'BaseWR':>6}  {'CoachWR':>7}")
    total_base = 0.0
    total_coach = 0.0
    for r in results:
        period = r["ts_start"][:10] + "→" + r["ts_end"][:10]
        print(f"{r['fold']:>4}  {period:<22} {r['n_test']:>6}  "
              f"{r['base_total']:>+9.2f}  {r['coach_total']:>+9.2f}  "
              f"{r['delta']:>+9.2f}  {r['base_wr']:>6.1%}  {r['coach_wr']:>7.1%}")
        total_base += r["base_total"]
        total_coach += r["coach_total"]

    print("-" * 100)
    delta_total = total_coach - total_base
    pct_improve = (delta_total / abs(total_base) * 100) if total_base != 0 else 0
    print(f"{'TOTAL':>4}  {' ':<22} {sum(r['n_test'] for r in results):>6}  "
          f"{total_base:>+9.2f}  {total_coach:>+9.2f}  {delta_total:>+9.2f}  "
          f"({pct_improve:+.1f}% vs baseline)")

    print("\nAggregated impact by Coach action (over all folds):")
    agg = defaultdict(lambda: {"n": 0, "delta": 0.0})
    for r in results:
        for act, v in r["by_action"].items():
            agg[act]["n"] += v["n"]
            agg[act]["delta"] += v["delta"]
    for act in sorted(agg.keys(), key=lambda a: -agg[a]["delta"]):
        v = agg[act]
        print(f"  {act:9s}  trades={v['n']:5d}  delta=${v['delta']:+.2f}")


if __name__ == "__main__":
    main()
