# Deep Dive 2 — Finale Analysen: Drift-Gate, Coach-Counterfactual, Kalibrierung
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

# Klines + 4H-Closes + EMA50
ema50 = {}  # sym -> (boundary_list, ema_values)
for sym, pair in SYM_MAP.items():
    h4 = {}
    with open(SNAP / f"klines_5m_{pair}.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ts = int(r["open_time"]) / 1000
            b = int(ts // 14400) * 14400
            h4[b] = f(r["close"])  # letzter 5m-Close der 4H-Periode gewinnt
    bts = sorted(h4)
    k = 2 / 51
    e = None
    es = []
    for b in bts:
        c = h4[b]
        e = c if e is None else c * k + e * (1 - k)
        es.append(e)
    ema50[sym] = (bts, es)

def drift(sym, ts):
    """+1 wenn 4H-Close > EMA50 (letzte abgeschlossene Kerze), -1 sonst."""
    bts, es = ema50[sym]
    i = bisect.bisect_right(bts, ts) - 2  # letzte ABGESCHLOSSENE Kerze vor ts
    if i < 1:
        return 0
    # Preis = Close der Kerze i
    return 1 if (ema50[sym][0][i], es[i]) and _close(sym, i) > es[i] else -1

_h4_closes = {}
def _close(sym, i):
    if sym not in _h4_closes:
        bts, _ = ema50[sym]
        # rebuild closes
        h4 = {}
        with open(SNAP / f"klines_5m_{SYM_MAP[sym]}.csv", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                ts = int(r["open_time"]) / 1000
                b = int(ts // 14400) * 14400
                h4[b] = f(r["close"])
        _h4_closes[sym] = [h4[b] for b in sorted(h4)]
    return _h4_closes[sym][i]

trades = []
with open(SNAP / "trades.csv", encoding="utf-8", errors="replace") as fh:
    for t in csv.DictReader(fh):
        t["base"] = t["bot_id"].rsplit("_", 1)[0]
        t["sym"] = t["bot_id"].rsplit("_", 1)[1]
        exit_ts = datetime.fromisoformat(t["timestamp"]).timestamp()
        t["entry_ts"] = int((exit_ts - f(t["hold_candles"]) * 14400) // 14400) * 14400 + 120
        trades.append(t)

print("=" * 100)
print("U) DRIFT-GATE: Trades nach Drift-Alignment (4H-Close vs EMA50 am Entry)")
print("=" * 100)
groups = defaultdict(list)
for t in trades:
    d = drift(t["sym"], t["entry_ts"])
    if d == 0:
        continue
    aligned = (d == 1 and t["direction"] == "long") or (d == -1 and t["direction"] == "short")
    groups[("aligned" if aligned else "counter", t["direction"])].append(t)
tot_saved = 0.0
for key, ts in sorted(groups.items()):
    pnl = sum(f(t["pnl"]) for t in ts)
    wr = sum(1 for t in ts if f(t["pnl"]) > 0) / len(ts)
    print(f"  {key[0]:8s} {key[1]:5s}  n={len(ts):4d}  WR={wr*100:5.1f}%  PnL=${pnl:+8.2f}")
counter_pnl = sum(f(t["pnl"]) for k, ts in groups.items() if k[0] == "counter" for t in ts)
print(f"  -> Hard-Gate (alle counter-drift Trades skippen) hätte gespart: ${-counter_pnl:+.2f}")
print(f"  -> Soft-Gate (counter-drift mit halber Size) hätte gespart: ${-counter_pnl/2:+.2f}")

print()
print("=" * 100)
print("V) COACH-COUNTERFACTUAL: PnL seit 26.05. mit/ohne Coach-Size-Multiplier")
print("=" * 100)
coach = json.load(open(SNAP / "coach_state.json", encoding="utf-8"))
dec = coach["decisions"]
cut = "2026-05-26"
with_c = without_c = 0.0
for t in trades:
    if t["timestamp"] < cut:
        continue
    d = dec.get(t["bot_id"])
    pnl = f(t["pnl"])
    with_c += pnl
    mult = (d["capital_multiplier"] * d.get("dsr_multiplier", 1.0)) if d else 1.0
    without_c += pnl / mult if mult > 0 else pnl
print(f"  PnL seit {cut} MIT Coach-Sizing:  ${with_c:+.2f}")
print(f"  PnL seit {cut} OHNE Coach-Sizing: ${without_c:+.2f}  (Multiplier rausgerechnet)")
print(f"  -> Coach-Sizing-Wertbeitrag: ${with_c - without_c:+.2f}")
inv_cells = [c for c, d in dec.items() if d.get("invert")]
inv_pnl = sum(f(t["pnl"]) for t in trades if t["timestamp"] >= cut and t["bot_id"] in inv_cells)
print(f"  Invertierte Cells ({len(inv_cells)}): PnL seit {cut} = ${inv_pnl:+.2f}")
print(f"  (Ohne Inversion wären diese Trades grob gespiegelt ~= ${-inv_pnl:+.2f})")

print()
print("=" * 100)
print("W) CONFIDENCE-KALIBRIERUNG via signals->trades-Join (Entry-Zeit + bot_id + direction)")
print("=" * 100)
sigs = defaultdict(list)
with open(SNAP / "signals.csv", encoding="utf-8", errors="replace") as fh:
    for s in csv.DictReader(fh):
        if s.get("direction") in ("long", "short"):
            try:
                ts = datetime.fromisoformat(s["timestamp"]).timestamp()
            except ValueError:
                continue
            sigs[(s["bot_id"], s["direction"])].append((ts, f(s["confidence"])))
for v in sigs.values():
    v.sort()
matched = []
for t in trades:
    key = (t["bot_id"], t["direction"])
    cand = sigs.get(key, [])
    i = bisect.bisect_left(cand, (t["entry_ts"] - 600,))
    best = None
    for j in range(max(0, i - 1), min(len(cand), i + 3)):
        if abs(cand[j][0] - t["entry_ts"]) <= 3600:
            best = cand[j][1]
            break
    if best is not None:
        matched.append((best, f(t["pnl"]), f(t["net_return_pct"])))
print(f"  gematcht: {len(matched)}/{len(trades)}")
bands = defaultdict(list)
for conf, pnl, ret in matched:
    b = min(int(conf * 10) / 10, 0.9)
    bands[b].append((pnl, ret))
for b in sorted(bands):
    rows = bands[b]
    wr = sum(1 for p, _ in rows if p > 0) / len(rows)
    avg = sum(r for _, r in rows) / len(rows)
    print(f"  conf {b:.1f}-{b+0.1:.1f}  n={len(rows):4d}  WR={wr*100:5.1f}%  avgRet={avg:+6.3f}%")
