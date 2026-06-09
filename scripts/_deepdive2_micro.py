# Deep Dive 2 — Mikrostruktur: Kalibrierung, Long/Short, Zeit, Hold, Heat, Wilson, nicht-getradete Signale
import csv
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SNAP = Path(__file__).resolve().parents[1] / "data" / "_live_snapshot"

def load_csv(name):
    with open(SNAP / name, encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))

def f(x, d=0.0):
    try:
        return float(x)
    except (ValueError, TypeError):
        return d

def stats(ts):
    n = len(ts)
    if n == 0:
        return dict(n=0, wr=0, pnl=0, avg_ret=0, pf=0)
    wins = [t for t in ts if f(t["pnl"]) > 0]
    gw = sum(f(t["pnl"]) for t in wins)
    gl = -sum(f(t["pnl"]) for t in ts if f(t["pnl"]) <= 0)
    return dict(n=n, wr=len(wins) / n, pnl=sum(f(t["pnl"]) for t in ts),
                avg_ret=sum(f(t["net_return_pct"]) for t in ts) / n,
                pf=(gw / gl) if gl > 0 else float("inf"))

def fmt(s):
    return f"n={s['n']:4d} WR={s['wr']*100:5.1f}% PnL=${s['pnl']:+8.2f} avgRet={s['avg_ret']:+6.3f}% PF={s['pf']:5.2f}"

def wilson(k, n, z=1.96):
    if n == 0:
        return (0, 0)
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (center - half, center + half)

live = load_csv("trades.csv")
for t in live:
    t["base"] = t["bot_id"].rsplit("_", 1)[0]
    t["sym"] = t["bot_id"].rsplit("_", 1)[1]
    t["dt"] = datetime.fromisoformat(t["timestamp"])

print("=" * 100)
print("H) LONG vs SHORT pro Symbol (live)")
print("=" * 100)
for sym in ["XRP", "BTC", "ETH", "SOL"]:
    for d in ["long", "short"]:
        ts = [t for t in live if t["sym"] == sym and t["direction"] == d]
        if ts:
            print(f"  {sym} {d:5s} {fmt(stats(ts))}")
print("  -- gesamt --")
for d in ["long", "short"]:
    ts = [t for t in live if t["direction"] == d]
    print(f"  ALL {d:5s} {fmt(stats(ts))}")

print()
print("=" * 100)
print("I) LONG vs SHORT pro Bot (live) — wo ist die Richtungs-Edge?")
print("=" * 100)
by_bd = defaultdict(list)
for t in live:
    by_bd[(t["base"], t["direction"])].append(t)
for (b, d), ts in sorted(by_bd.items()):
    if len(ts) >= 8:
        print(f"  {b:18s} {d:5s} {fmt(stats(ts))}")

print()
print("=" * 100)
print("J) HOLD-DAUER: Winner vs Loser (Kerzen à 4h)")
print("=" * 100)
buckets = [(0, 1), (2, 3), (4, 6), (7, 12), (13, 999)]
for lo, hi in buckets:
    ts = [t for t in live if lo <= f(t["hold_candles"]) <= hi]
    if ts:
        print(f"  hold {lo:2d}-{hi:3d}  {fmt(stats(ts))}")
w = [f(t["hold_candles"]) for t in live if f(t["pnl"]) > 0]
l = [f(t["hold_candles"]) for t in live if f(t["pnl"]) <= 0]
print(f"  avg hold Winner: {sum(w)/len(w):.1f} Kerzen | Loser: {sum(l)/len(l):.1f} Kerzen")

print()
print("=" * 100)
print("K) EXIT-TAGESZEIT (UTC-Stunde des Exits) + WOCHENTAG")
print("=" * 100)
by_h = defaultdict(list)
for t in live:
    by_h[t["dt"].hour].append(t)
for h in sorted(by_h):
    s = stats(by_h[h])
    if s["n"] >= 10:
        print(f"  hour={h:02d}  {fmt(s)}")
by_wd = defaultdict(list)
for t in live:
    by_wd[t["dt"].strftime("%a")].append(t)
for wd in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]:
    if wd in by_wd:
        print(f"  {wd}  {fmt(stats(by_wd[wd]))}")

print()
print("=" * 100)
print("L) WILSON 95%-CI für Top/Flop-Cells (ist die Edge signifikant?)")
print("=" * 100)
by_cell = defaultdict(list)
for t in live:
    by_cell[t["bot_id"]].append(t)
rows = []
for c, ts in by_cell.items():
    k = sum(1 for t in ts if f(t["pnl"]) > 0)
    lo, hi = wilson(k, len(ts))
    rows.append((c, len(ts), k / len(ts), lo, hi, stats(ts)["pnl"]))
rows.sort(key=lambda r: -r[3])
for c, n, wr, lo, hi, pnl in rows:
    if n >= 10:
        sig = " ***" if lo > 0.5 else (" !!!" if hi < 0.35 else "")
        print(f"  {c:24s} n={n:3d} WR={wr*100:5.1f}%  CI=[{lo*100:4.1f}%, {hi*100:4.1f}%]  PnL=${pnl:+7.2f}{sig}")

print()
print("=" * 100)
print("M) CONFIDENCE-KALIBRIERUNG (signals.csv -> getradete + Insight-Outcomes)")
print("=" * 100)
sigs = load_csv("signals.csv")
print(f"  signals gesamt: {len(sigs)}")
ins = []
with open(SNAP / "insights.jsonl", encoding="utf-8", errors="replace") as fh:
    for line in fh:
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "outcome_pnl" in d and d.get("outcome_pnl") is not None and d.get("bot_id") != "test_bot":
            ins.append(d)
print(f"  insights mit Outcome: {len(ins)}")
cal = defaultdict(list)
for d in ins:
    c = f(d.get("confidence"))
    b = min(int(c * 10), 9) / 10
    cal[b].append(d)
for b in sorted(cal):
    ds = cal[b]
    wr = sum(1 for d in ds if f(d["outcome_pnl"]) > 0) / len(ds)
    avg = sum(f(d["outcome_pnl"]) for d in ds) / len(ds)
    print(f"  conf {b:.1f}-{b+0.1:.1f}  n={len(ds):4d}  WR={wr*100:5.1f}%  avgPnL=${avg:+6.2f}")

print()
print("=" * 100)
print("N) PORTFOLIO-HEAT: gleichzeitige offene Positionen (aus trades.csv rekonstruiert)")
print("=" * 100)
events = []
for t in live:
    exit_dt = t["dt"]
    entry_dt = None
    try:
        hold_h = f(t["hold_candles"]) * 4
        entry_dt = exit_dt.timestamp() - hold_h * 3600
    except Exception:
        continue
    events.append((entry_dt, 1, t))
    events.append((exit_dt.timestamp(), -1, t))
events.sort(key=lambda e: e[0])
open_now = 0
max_open = 0
samedir = defaultdict(int)
max_samedir = {}
cur = defaultdict(int)
for ts_, delta, t in events:
    open_now += delta
    max_open = max(max_open, open_now)
    key = (t["sym"], t["direction"])
    cur[key] += delta
    if cur[key] > samedir[key]:
        samedir[key] = cur[key]
print(f"  max gleichzeitig offen (gesamt): {max_open}")
print("  max gleichzeitig gleiche Richtung pro Symbol:")
for k, v in sorted(samedir.items(), key=lambda kv: -kv[1]):
    print(f"    {k[0]} {k[1]:5s}: {v}")

print()
print("=" * 100)
print("O) NICHT-GETRADETE SIGNALE: was liegt unter der Entry-Schwelle?")
print("=" * 100)
conf_traded = defaultdict(int)
for d in ins:
    conf_traded[round(f(d.get("confidence")), 2)] += 1
sig_conf = defaultdict(int)
for s in sigs:
    if s.get("direction") in ("long", "short"):
        sig_conf[round(f(s.get("confidence")), 2)] += 1
all_dir_sigs = sum(sig_conf.values())
print(f"  direktionale Signale: {all_dir_sigs} | mit Outcome verknüpfte Insights: {len(ins)}")
lo_band = [d for d in ins if f(d.get("confidence")) < 0.45]
hi_band = [d for d in ins if f(d.get("confidence")) >= 0.45]
for name, band in [("conf<0.45", lo_band), ("conf>=0.45", hi_band)]:
    if band:
        wr = sum(1 for d in band if f(d["outcome_pnl"]) > 0) / len(band)
        avg = sum(f(d["outcome_pnl"]) for d in band) / len(band)
        print(f"  {name:10s} n={len(band):4d} WR={wr*100:5.1f}% avgPnL=${avg:+6.2f}")
