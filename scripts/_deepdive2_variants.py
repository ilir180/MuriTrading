# Deep Dive 2 — Varianten-Tests:
#   A) trend_rider Thesis nur-ADX (EMA-Cross-Bedingung entfernt)
#   B) Drift-Gate nur bei starkem Drift (>1 ATR vom EMA50)
import runpy
from collections import defaultdict
from pathlib import Path

ctx = runpy.run_path(str(Path(__file__).with_name("_deepdive2_validate.py")))
bars4h = ctx["bars4h"]
idx_at = ctx["idx_at"]
trades = ctx["trades"]
f = ctx["f"]
SYM_MAP = ctx["SYM_MAP"]

def replay_adxonly(sym, direction, entry_ts, entry, atr):
    B = bars4h[sym]
    i0 = idx_at(sym, entry_ts) + 1
    sgn = 1 if direction == "long" else -1
    sl = entry - sgn * 4.0 * atr
    tp = entry + sgn * 5.0 * atr
    peak = entry
    trail = 0.0
    trail_on = False
    strikes = 0
    for k in range(i0, min(i0 + 24, len(B["ts"]))):
        h, l, c = B["H"][k], B["L"][k], B["C"][k]
        if direction == "long":
            if l <= sl:
                return (sl - entry) / entry - 0.0012, "SL"
            if h >= tp:
                return (tp - entry) / entry - 0.0012, "TP"
            peak = max(peak, c)
            if peak - entry >= 0.8 * atr:
                trail_on = True
                trail = max(trail, peak - 1.5 * atr, sl)
            if trail_on and c <= trail:
                return (trail - entry) / entry - 0.0012, "TRAIL"
        else:
            if h >= sl:
                return (entry - sl) / entry - 0.0012, "SL"
            if l <= tp:
                return (entry - tp) / entry - 0.0012, "TP"
            peak = min(peak, c)
            if entry - peak >= 0.8 * atr:
                trail_on = True
                nt = peak + 1.5 * atr
                trail = nt if trail == 0 else min(trail, nt)
                trail = min(trail, sl)
            if trail_on and c >= trail:
                return (entry - trail) / entry - 0.0012, "TRAIL"
        strikes = strikes + 1 if B["adx"][k] < 15 else 0
        if strikes >= 2:
            return sgn * (c - entry) / entry - 0.0012, "THESIS-ADX"
    last_c = B["C"][min(i0 + 23, len(B["C"]) - 1)]
    return sgn * (last_c - entry) / entry - 0.0012, "TIME"

tr = [t for t in trades if t["base"] == "trend_rider"]
tot = 0.0
n = 0
wins = 0
mix = defaultdict(int)
for t in tr:
    sym = t["sym"]
    entry = f(t["entry_price"])
    i = idx_at(sym, t["entry_ts"])
    if i < 1:
        continue
    ref = bars4h[sym]["O"][min(i + 1, len(bars4h[sym]["O"]) - 1)]
    if abs(ref - entry) / entry > 0.03:
        continue
    atr = bars4h[sym]["atr"][i] or entry * 0.02
    ret, reason = replay_adxonly(sym, t["direction"], t["entry_ts"], entry, atr)
    tot += f(t["size_usd"]) * ret
    n += 1
    wins += ret > 0
    mix[reason] += 1
print(f"A) trend_rider NUR-ADX-Thesis (2 Kerzen): n={n} PnL=${tot:+.2f} WR={wins/n*100:.1f}% {dict(mix)}")

ema50 = {}
for sym in SYM_MAP:
    B = bars4h[sym]
    k2 = 2 / 51
    e = None
    es = []
    for c in B["C"]:
        e = c if e is None else c * k2 + e * (1 - k2)
        es.append(e)
    ema50[sym] = es

def drift_strength(sym, ts):
    i = idx_at(sym, ts) - 1
    if i < 1:
        return 0, 0.0
    B = bars4h[sym]
    d = 1 if B["C"][i] > ema50[sym][i] else -1
    s = abs(B["C"][i] - ema50[sym][i]) / (B["atr"][i] or 1)
    return d, s

for half, lo, hi in [("H1", "", "2026-05-19"), ("H2", "2026-05-19", "9999"), ("GESAMT", "", "9999")]:
    weak = strong = 0.0
    nw = ns = 0
    for t in trades:
        if not (lo <= t["timestamp"] < hi):
            continue
        d, s = drift_strength(t["sym"], t["entry_ts"])
        if d == 0:
            continue
        sgn = 1 if t["direction"] == "long" else -1
        if sgn != d:
            if s > 1.0:
                strong += f(t["pnl"])
                ns += 1
            else:
                weak += f(t["pnl"])
                nw += 1
    print(f"B) {half}: counter bei STARKEM Drift (>1 ATR) n={ns} PnL=${strong:+.2f} | bei schwachem n={nw} PnL=${weak:+.2f}")
