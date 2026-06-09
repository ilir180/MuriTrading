# Deep Dive 2 — Exit-Engine-Replay auf 5m-Kerzen
# Spielt jeden Live-Trade mit alternativen Exit-Regeln nach:
#   B = volle Engine OHNE Thesis-Exit (SL/TP/Trailing/Time)
#   C = nur SL/TP/Time (ohne Trailing)
#   D = MFE/MAE innerhalb max_hold
# Fees: ROUND_TRIP 0.12% wie live.
import bisect
import csv
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SNAP = Path(__file__).resolve().parents[1] / "data" / "_live_snapshot"
ROUND_TRIP = 0.0012
TRAIL_MULT = 1.5
TRAIL_ACT = 0.8

RP = {
    "trend_rider": dict(sl_atr=2.5, tp_atr=5.0, max_hold=24),
    "mean_reverter": dict(sl_atr=1.5, tp_atr=2.0, max_hold=12),
    "breakout_hunter": dict(sl_atr=1.5, tp_atr=3.0, max_hold=8),
    "contrarian": dict(sl_atr=2.5, tp_atr=3.5, max_hold=18),
    "flow_tracker": dict(sl_atr=1.5, tp_atr=2.5, max_hold=10),
    "momentum_surfer": dict(sl_atr=2.0, tp_atr=4.0, max_hold=16),
    "level_bouncer": dict(sl_atr=1.0, tp_atr=2.0, max_hold=12),
    "volatility_fader": dict(sl_atr=1.5, tp_atr=2.0, max_hold=6),
}
SYM_MAP = {"XRP": "XRPUSDT", "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}

def f(x, d=0.0):
    try:
        return float(x)
    except (ValueError, TypeError):
        return d

# ── Klines laden ──
klines = {}   # sym -> (ts_list, bars)
for sym, pair in SYM_MAP.items():
    ts_list, bars = [], []
    with open(SNAP / f"klines_5m_{pair}.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ts = int(r["open_time"]) / 1000
            ts_list.append(ts)
            bars.append((f(r["open"]), f(r["high"]), f(r["low"]), f(r["close"])))
    klines[sym] = (ts_list, bars)

# ── 4H-ATR(14) aus 5m aggregieren ──
atr4h = {}    # sym -> (boundary_ts_list, atr_values)
for sym, (ts_list, bars) in klines.items():
    h4 = {}
    for ts, (o, h, l, c) in zip(ts_list, bars):
        b = int(ts // 14400) * 14400
        if b not in h4:
            h4[b] = [o, h, l, c]
        else:
            agg = h4[b]
            agg[1] = max(agg[1], h)
            agg[2] = min(agg[2], l)
            agg[3] = c
    bts = sorted(h4)
    trs, atrs = [], []
    prev_c = None
    atr = None
    for b in bts:
        o, h, l, c = h4[b]
        tr = h - l if prev_c is None else max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)
        if atr is None:
            atr = sum(trs) / len(trs) if len(trs) < 14 else None
            if len(trs) == 14:
                atr = sum(trs) / 14
        else:
            atr = (atr * 13 + tr) / 14
        atrs.append(atr if atr else tr)
        prev_c = c
    atr4h[sym] = (bts, atrs)

def get_atr(sym, ts):
    bts, atrs = atr4h[sym]
    i = bisect.bisect_right(bts, ts) - 1
    return atrs[max(0, min(i, len(atrs) - 1))]

def replay(sym, direction, entry_ts, entry_price, atr, sl_mult, tp_mult, max_hold_h, use_trail):
    """Simuliert Exits auf 5m-Closes. Liefert (net_ret, reason, hold_h, mfe, mae)."""
    ts_list, bars = klines[sym]
    i0 = bisect.bisect_left(ts_list, entry_ts)
    sl_dist = sl_mult * atr
    tp_dist = tp_mult * atr
    sgn = 1 if direction == "long" else -1
    sl = entry_price - sgn * sl_dist
    tp = entry_price + sgn * tp_dist
    peak = entry_price
    trail = 0.0
    trail_active = False
    mfe = 0.0
    mae = 0.0
    end_ts = entry_ts + max_hold_h * 3600
    last_c = entry_price
    for i in range(i0, len(ts_list)):
        ts = ts_list[i]
        if ts > end_ts:
            break
        o, h, l, c = bars[i]
        last_c = c
        fav = sgn * (c - entry_price) / entry_price
        adv_extreme = (l if direction == "long" else h)
        fav_extreme = (h if direction == "long" else l)
        mfe = max(mfe, sgn * (fav_extreme - entry_price) / entry_price)
        mae = min(mae, sgn * (adv_extreme - entry_price) / entry_price)
        # SL/TP intrabar (high/low)
        if direction == "long":
            if l <= sl:
                return (sl - entry_price) / entry_price - ROUND_TRIP, "SL", (ts - entry_ts) / 3600, mfe, mae
            if h >= tp:
                return (tp - entry_price) / entry_price - ROUND_TRIP, "TP", (ts - entry_ts) / 3600, mfe, mae
        else:
            if h >= sl:
                return (entry_price - sl) / entry_price - ROUND_TRIP, "SL", (ts - entry_ts) / 3600, mfe, mae
            if l <= tp:
                return (entry_price - tp) / entry_price - ROUND_TRIP, "TP", (ts - entry_ts) / 3600, mfe, mae
        if use_trail:
            if direction == "long":
                peak = max(peak, c)
                if peak - entry_price >= TRAIL_ACT * atr:
                    trail_active = True
                    trail = max(trail, peak - TRAIL_MULT * atr, sl)
                if trail_active and c <= trail:
                    return (trail - entry_price) / entry_price - ROUND_TRIP, "TRAIL", (ts - entry_ts) / 3600, mfe, mae
            else:
                peak = min(peak, c)
                if entry_price - peak >= TRAIL_ACT * atr:
                    trail_active = True
                    nt = peak + TRAIL_MULT * atr
                    trail = nt if trail == 0 else min(trail, nt)
                    trail = min(trail, sl)
                if trail_active and c >= trail:
                    return (entry_price - trail) / entry_price - ROUND_TRIP, "TRAIL", (ts - entry_ts) / 3600, mfe, mae
    ret = sgn * (last_c - entry_price) / entry_price - ROUND_TRIP
    return ret, "TIME", max_hold_h, mfe, mae

# ── Trades laden + Entry-Zeit schätzen ──
trades = []
with open(SNAP / "trades.csv", encoding="utf-8", errors="replace") as fh:
    for t in csv.DictReader(fh):
        t["base"] = t["bot_id"].rsplit("_", 1)[0]
        t["sym"] = t["bot_id"].rsplit("_", 1)[1]
        exit_dt = datetime.fromisoformat(t["timestamp"]).timestamp()
        hold = f(t["hold_candles"])
        est = exit_dt - hold * 14400
        entry_ts = int(est // 14400) * 14400 + 120  # 4H-Boundary + 2 Min
        t["entry_ts"] = entry_ts
        trades.append(t)

skipped = 0
results = []
for t in trades:
    sym = t["sym"]
    rp = RP[t["base"]]
    entry = f(t["entry_price"])
    ts_list, bars = klines[sym]
    i0 = bisect.bisect_left(ts_list, t["entry_ts"])
    if i0 >= len(ts_list):
        skipped += 1
        continue
    ref_price = bars[min(i0, len(bars) - 1)][0]
    if abs(ref_price - entry) / entry > 0.02:
        # Entry-Schätzung passt nicht zur Preisrealität -> ±1 Kerze probieren
        ok = False
        for shift in (-14400, 14400, -28800):
            i1 = bisect.bisect_left(ts_list, t["entry_ts"] + shift)
            if i1 < len(ts_list) and abs(bars[i1][0] - entry) / entry <= 0.02:
                t["entry_ts"] += shift
                ok = True
                break
        if not ok:
            skipped += 1
            continue
    atr = get_atr(sym, t["entry_ts"])
    sl_mult = max(rp["sl_atr"], 4.0)  # Catastrophe-SL wie base_bot.py
    size = f(t["size_usd"])
    args = (sym, t["direction"], t["entry_ts"], entry, atr)
    rB = replay(*args, sl_mult, rp["tp_atr"], rp["max_hold"] * 4, True)
    rC = replay(*args, sl_mult, rp["tp_atr"], rp["max_hold"] * 4, False)
    results.append(dict(
        t=t, size=size,
        actual_ret=f(t["net_return_pct"]) / 100, actual_pnl=f(t["pnl"]),
        B_ret=rB[0], B_reason=rB[1], B_pnl=size * rB[0],
        C_ret=rC[0], C_reason=rC[1], C_pnl=size * rC[0],
        mfe=rB[3], mae=rB[4], atr_pct=atr / entry,
    ))

print(f"Trades repliziert: {len(results)}, übersprungen (Entry-Mismatch): {skipped}")
print()
print("=" * 100)
print("P) GESAMT: Actual vs B (ohne Thesis-Exit) vs C (nur SL/TP/Time)")
print("=" * 100)
for k, lbl in [("actual_pnl", "ACTUAL"), ("B_pnl", "B ohne Thesis"), ("C_pnl", "C ohne Thesis+Trail")]:
    tot = sum(r[k] for r in results)
    wr = sum(1 for r in results if r[k] > 0) / len(results)
    print(f"  {lbl:22s} PnL=${tot:+9.2f}  WR={wr*100:5.1f}%")

print()
print("=" * 100)
print("Q) NUR TRADES, DIE LIVE PER THESIS-EXIT GESCHLOSSEN WURDEN: was wäre ohne gewesen?")
print("=" * 100)
th = [r for r in results if r["t"]["reason"].startswith("THESIS")]
by_base = defaultdict(list)
for r in th:
    by_base[r["t"]["base"]].append(r)
print(f"  {'bot':18s} {'n':>4s} {'ACTUAL $':>10s} {'B (ohne Thesis) $':>18s} {'Delta $':>9s}  B-Exit-Mix")
tot_a = tot_b = 0.0
for b, rs in sorted(by_base.items(), key=lambda kv: -len(kv[1])):
    a = sum(r["actual_pnl"] for r in rs)
    bb = sum(r["B_pnl"] for r in rs)
    tot_a += a
    tot_b += bb
    mix = defaultdict(int)
    for r in rs:
        mix[r["B_reason"]] += 1
    print(f"  {b:18s} {len(rs):4d} {a:+10.2f} {bb:+18.2f} {bb-a:+9.2f}  {dict(mix)}")
print(f"  {'TOTAL':18s} {len(th):4d} {tot_a:+10.2f} {tot_b:+18.2f} {tot_b-tot_a:+9.2f}")

print()
print("=" * 100)
print("R) DASSELBE FÜR TRAILING-STOP-EXITS: hat der Trail Edge gerettet oder gekappt?")
print("=" * 100)
tr = [r for r in results if r["t"]["reason"].startswith("TRAILING")]
a = sum(r["actual_pnl"] for r in tr)
bb = sum(r["B_pnl"] for r in tr)
cc = sum(r["C_pnl"] for r in tr)
print(f"  n={len(tr)}  ACTUAL=${a:+.2f}  B(mit Trail,ohne Thesis)=${bb:+.2f}  C(ohne Trail)=${cc:+.2f}")

print()
print("=" * 100)
print("S) MFE/MAE: wieviel Gewinn lag im Trade, der nicht realisiert wurde?")
print("=" * 100)
by_base_all = defaultdict(list)
for r in results:
    by_base_all[r["t"]["base"]].append(r)
print(f"  {'bot':18s} {'avgMFE%':>8s} {'avgMAE%':>8s} {'avgMFE/ATR':>11s} {'TP-Distanz/ATR':>15s} {'%Trades MFE>=TP':>16s}")
for b, rs in sorted(by_base_all.items()):
    rp = RP[b]
    avg_mfe = sum(r["mfe"] for r in rs) / len(rs) * 100
    avg_mae = sum(r["mae"] for r in rs) / len(rs) * 100
    mfe_atr = sum(r["mfe"] / r["atr_pct"] for r in rs) / len(rs)
    hit = sum(1 for r in rs if r["mfe"] / r["atr_pct"] >= rp["tp_atr"]) / len(rs)
    print(f"  {b:18s} {avg_mfe:8.2f} {avg_mae:8.2f} {mfe_atr:11.2f} {rp['tp_atr']:15.1f} {hit*100:15.1f}%")

print()
print("=" * 100)
print("T) WAS-WÄRE-WENN-MATRIX pro Bot (alle Trades): Actual vs B vs C")
print("=" * 100)
print(f"  {'bot':18s} {'n':>4s} {'ACTUAL $':>10s} {'B $':>10s} {'C $':>10s}")
for b, rs in sorted(by_base_all.items(), key=lambda kv: -sum(r['actual_pnl'] for r in kv[1])):
    a = sum(r["actual_pnl"] for r in rs)
    bb = sum(r["B_pnl"] for r in rs)
    cc = sum(r["C_pnl"] for r in rs)
    print(f"  {b:18s} {len(rs):4d} {a:+10.2f} {bb:+10.2f} {cc:+10.2f}")
a = sum(r["actual_pnl"] for r in results)
bb = sum(r["B_pnl"] for r in results)
cc = sum(r["C_pnl"] for r in results)
print(f"  {'TOTAL':18s} {len(results):4d} {a:+10.2f} {bb:+10.2f} {cc:+10.2f}")
