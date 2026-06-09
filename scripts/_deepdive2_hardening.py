# Deep Dive 2 — Härtetests: Drift-Gate pro Bot, CF-Validität, v2-Threshold-Check
import bisect
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SNAP = Path(__file__).resolve().parents[1] / "data" / "_live_snapshot"
SYM_MAP = {"XRP": "XRPUSDT", "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}

def f(x, d=0.0):
    try:
        return float(x)
    except (ValueError, TypeError):
        return d

def stats(ts):
    n = len(ts)
    if n == 0:
        return dict(n=0, wr=0, pnl=0)
    return dict(n=n, wr=sum(1 for t in ts if f(t["pnl"]) > 0) / n,
                pnl=sum(f(t["pnl"]) for t in ts))

# 4H-EMA50 wie im Final-Script
ema50 = {}
h4closes = {}
for sym, pair in SYM_MAP.items():
    h4 = {}
    with open(SNAP / f"klines_5m_{pair}.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ts = int(r["open_time"]) / 1000
            h4[int(ts // 14400) * 14400] = f(r["close"])
    bts = sorted(h4)
    closes = [h4[b] for b in bts]
    k = 2 / 51
    e = None
    es = []
    for c in closes:
        e = c if e is None else c * k + e * (1 - k)
        es.append(e)
    ema50[sym] = (bts, es)
    h4closes[sym] = closes

def drift(sym, ts):
    bts, es = ema50[sym]
    i = bisect.bisect_right(bts, ts) - 2
    if i < 1:
        return 0
    return 1 if h4closes[sym][i] > es[i] else -1

trades = []
with open(SNAP / "trades.csv", encoding="utf-8", errors="replace") as fh:
    for t in csv.DictReader(fh):
        t["base"] = t["bot_id"].rsplit("_", 1)[0]
        t["sym"] = t["bot_id"].rsplit("_", 1)[1]
        exit_ts = datetime.fromisoformat(t["timestamp"]).timestamp()
        t["entry_ts"] = int((exit_ts - f(t["hold_candles"]) * 14400) // 14400) * 14400 + 120
        trades.append(t)

print("=" * 100)
print("X) DRIFT-GATE PRO BOT: schadet das Gate Mean-Reversion-Thesen?")
print("=" * 100)
per_bot = defaultdict(lambda: defaultdict(list))
for t in trades:
    d = drift(t["sym"], t["entry_ts"])
    if d == 0:
        continue
    sgn = 1 if t["direction"] == "long" else -1
    per_bot[t["base"]]["counter" if sgn != d else "aligned"].append(t)
print(f"  {'bot':18s} {'counter n/WR/PnL':>28s} {'aligned n/WR/PnL':>28s}  Gate-Wirkung (0.5x counter)")
for b in sorted(per_bot):
    c = stats(per_bot[b]["counter"])
    a = stats(per_bot[b]["aligned"])
    gate_delta = -c["pnl"] / 2  # halbe Size = halber PnL der counter-Trades
    flag = "HILFT" if gate_delta > 1 else ("SCHADET" if gate_delta < -1 else "neutral")
    print(f"  {b:18s} {c['n']:5d}/{c['wr']*100:4.1f}%/${c['pnl']:+8.2f} {a['n']:5d}/{a['wr']*100:4.1f}%/${a['pnl']:+8.2f}  {gate_delta:+7.2f} {flag}")

print()
print("=" * 100)
print("Y) CF-VALIDITÄT: Counterfactual vs Live pro Cell — korreliert das überhaupt?")
print("=" * 100)
cf_cell = defaultdict(list)
with open(SNAP / "counterfactual_trades.csv", encoding="utf-8", errors="replace") as fh:
    for t in csv.DictReader(fh):
        cf_cell[t["bot_id"]].append(t)
live_cell = defaultdict(list)
for t in trades:
    live_cell[t["bot_id"]].append(t)
pairs = []
for cell in set(cf_cell) & set(live_cell):
    sc, sl = stats(cf_cell[cell]), stats(live_cell[cell])
    if sc["n"] >= 30 and sl["n"] >= 10:
        pairs.append((cell, sc, sl))
print(f"  Cells mit n_cf>=30 und n_live>=10: {len(pairs)}")
print(f"  {'cell':24s} {'CF WR':>7s} {'Live WR':>8s} {'CF avgPnL':>10s} {'Live avgPnL':>12s}")
xs, ys = [], []
for cell, sc, sl in sorted(pairs, key=lambda p: -p[2]['wr']):
    print(f"  {cell:24s} {sc['wr']*100:6.1f}% {sl['wr']*100:7.1f}% {sc['pnl']/sc['n']:+10.3f} {sl['pnl']/sl['n']:+12.3f}")
    xs.append(sc["wr"])
    ys.append(sl["wr"])
n = len(xs)
if n >= 3:
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    r = cov / (vx * vy) if vx > 0 and vy > 0 else 0
    print(f"  Pearson-r (CF-WR vs Live-WR über {n} Cells): {r:+.3f}")

# CF-Zeitfenster: nur letzte 12 Monate des CF vs Live — passt es dann besser?
print()
print("  -- CF eingeschränkt auf 2025-06+ (letzte 12 Monate) --")
xs2, ys2 = [], []
for cell, _, sl in pairs:
    recent_cf = [t for t in cf_cell[cell] if t["timestamp"] >= "2025-06"]
    if len(recent_cf) >= 15:
        sc2 = stats(recent_cf)
        xs2.append(sc2["wr"])
        ys2.append(sl["wr"])
n2 = len(xs2)
if n2 >= 3:
    mx, my = sum(xs2) / n2, sum(ys2) / n2
    cov = sum((x - mx) * (y - my) for x, y in zip(xs2, ys2))
    vx = sum((x - mx) ** 2 for x in xs2) ** 0.5
    vy = sum((y - my) ** 2 for y in ys2) ** 0.5
    r2 = cov / (vx * vy) if vx > 0 and vy > 0 else 0
    print(f"  Pearson-r (recent-CF vs Live, {n2} Cells): {r2:+.3f}")

print()
print("=" * 100)
print("Z) REGIME-FEATURE-DRIFT: Live-Verteilung vs CF-Verteilung (ADX, ATR%, FearGreed)")
print("=" * 100)
def dist(rows, key):
    vals = sorted(f(t.get(key)) for t in rows if t.get(key) not in (None, ""))
    if not vals:
        return None
    def q(p):
        return vals[min(int(p * len(vals)), len(vals) - 1)]
    return q(0.25), q(0.5), q(0.75)
cf_all = [t for ts in cf_cell.values() for t in ts]
for key in ["regime_adx", "regime_atr_pct", "regime_fear_greed", "regime_chop"]:
    dl = dist(trades, key)
    dc = dist(cf_all, key)
    print(f"  {key:22s} live q25/50/75: {dl}   cf: {dc}")
