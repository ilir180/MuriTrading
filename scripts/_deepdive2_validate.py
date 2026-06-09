# Deep Dive 2 — Validierung der DEPLOYTEN Regeln (nicht der Extremvarianten):
#   1) trend_rider Thesis-Hysterese (2 Kerzen) als echtes Replay mit ADX/EMA auf 4H
#   2) Drift-Gate Walk-Forward: hält der Befund in beiden Datenhälften?
import bisect
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

SNAP = Path(__file__).resolve().parents[1] / "data" / "_live_snapshot"
SYM_MAP = {"XRP": "XRPUSDT", "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
ROUND_TRIP = 0.0012

def f(x, d=0.0):
    try:
        return float(x)
    except (ValueError, TypeError):
        return d

# ── 4H-Bars aus 5m + Indikatoren (EMA9/21, ADX14, ATR14) ──
bars4h = {}   # sym -> dict(ts=[], o,h,l,c, ema9, ema21, adx, atr)
for sym, pair in SYM_MAP.items():
    h4 = {}
    with open(SNAP / f"klines_5m_{pair}.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ts = int(r["open_time"]) / 1000
            b = int(ts // 14400) * 14400
            o, h, l, c = f(r["open"]), f(r["high"]), f(r["low"]), f(r["close"])
            if b not in h4:
                h4[b] = [o, h, l, c]
            else:
                agg = h4[b]
                agg[1] = max(agg[1], h)
                agg[2] = min(agg[2], l)
                agg[3] = c
    bts = sorted(h4)
    O = [h4[b][0] for b in bts]
    H = [h4[b][1] for b in bts]
    L = [h4[b][2] for b in bts]
    C = [h4[b][3] for b in bts]
    def ema(vals, span):
        k = 2 / (span + 1)
        e = None
        out = []
        for v in vals:
            e = v if e is None else v * k + e * (1 - k)
            out.append(e)
        return out
    e9, e21 = ema(C, 9), ema(C, 21)
    # Wilder ADX(14)
    n = len(bts)
    tr = [0.0] * n
    pdm = [0.0] * n
    ndm = [0.0] * n
    for i in range(1, n):
        tr[i] = max(H[i] - L[i], abs(H[i] - C[i-1]), abs(L[i] - C[i-1]))
        up, dn = H[i] - H[i-1], L[i-1] - L[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        ndm[i] = dn if (dn > up and dn > 0) else 0.0
    atr = [0.0] * n
    spdm = [0.0] * n
    sndm = [0.0] * n
    adx = [0.0] * n
    a = sum(tr[1:15]) / 14 if n > 15 else (sum(tr) / max(1, n))
    sp = sum(pdm[1:15]) / 14 if n > 15 else 0
    sn = sum(ndm[1:15]) / 14 if n > 15 else 0
    dxs = []
    prev_adx = None
    for i in range(1, n):
        if i >= 15:
            a = (a * 13 + tr[i]) / 14
            sp = (sp * 13 + pdm[i]) / 14
            sn = (sn * 13 + ndm[i]) / 14
        atr[i] = a
        pdi = 100 * sp / a if a > 0 else 0
        ndi = 100 * sn / a if a > 0 else 0
        dx = 100 * abs(pdi - ndi) / (pdi + ndi) if (pdi + ndi) > 0 else 0
        dxs.append(dx)
        if len(dxs) < 14:
            adx[i] = sum(dxs) / len(dxs)
        else:
            adx[i] = (prev_adx * 13 + dx) / 14 if prev_adx else sum(dxs[-14:]) / 14
        prev_adx = adx[i]
    bars4h[sym] = dict(ts=bts, O=O, H=H, L=L, C=C, e9=e9, e21=e21, adx=adx, atr=atr)

def idx_at(sym, ts):
    return bisect.bisect_right(bars4h[sym]["ts"], ts) - 1

def replay_trendrider(sym, direction, entry_ts, entry, atr, hysterese: int):
    """Replay mit deployter Logik: Thesis-Check pro 4H-Kerze (ADX<15 oder
    EMA-Cross gegen Richtung, `hysterese` Kerzen in Folge) + SL/TP/Trail/Time.
    hysterese=1 = altes Verhalten, 2 = deployt."""
    B = bars4h[sym]
    i0 = idx_at(sym, entry_ts) + 1  # erste Kerze NACH Entry
    sl_mult, tp_mult, max_hold = 4.0, 5.0, 24  # trend_rider, Catastrophe-SL 4 ATR
    sgn = 1 if direction == "long" else -1
    sl = entry - sgn * sl_mult * atr
    tp = entry + sgn * tp_mult * atr
    peak = entry
    trail = 0.0
    trail_on = False
    strikes = 0
    for k in range(i0, min(i0 + max_hold, len(B["ts"]))):
        h, l, c = B["H"][k], B["L"][k], B["C"][k]
        # Intrabar SL/TP zuerst (wie live per Tick)
        if direction == "long":
            if l <= sl:
                return (sl - entry) / entry * sgn - ROUND_TRIP, "SL"
            if h >= tp:
                return (tp - entry) / entry * sgn - ROUND_TRIP, "TP"
        else:
            if h >= sl:
                return (entry - sl) / entry - ROUND_TRIP, "SL"
            if l <= tp:
                return (entry - tp) / entry - ROUND_TRIP, "TP"
        # Trailing auf Kerzen-Basis
        if direction == "long":
            peak = max(peak, c)
            if peak - entry >= 0.8 * atr:
                trail_on = True
                trail = max(trail, peak - 1.5 * atr, sl)
            if trail_on and c <= trail:
                return (trail - entry) / entry - ROUND_TRIP, "TRAIL"
        else:
            peak = min(peak, c)
            if entry - peak >= 0.8 * atr:
                trail_on = True
                nt = peak + 1.5 * atr
                trail = nt if trail == 0 else min(trail, nt)
                trail = min(trail, sl)
            if trail_on and c >= trail:
                return (entry - trail) / entry - ROUND_TRIP, "TRAIL"
        # Thesis-Check am Kerzenschluss
        ema_against = (B["e9"][k] < B["e21"][k]) if direction == "long" else (B["e9"][k] > B["e21"][k])
        if B["adx"][k] < 15 or ema_against:
            strikes += 1
        else:
            strikes = 0
        if strikes >= hysterese:
            ret = sgn * (c - entry) / entry - ROUND_TRIP
            return ret, "THESIS"
    last_c = B["C"][min(i0 + max_hold - 1, len(B["C"]) - 1)]
    return sgn * (last_c - entry) / entry - ROUND_TRIP, "TIME"

trades = []
with open(SNAP / "trades.csv", encoding="utf-8", errors="replace") as fh:
    for t in csv.DictReader(fh):
        t["base"] = t["bot_id"].rsplit("_", 1)[0]
        t["sym"] = t["bot_id"].rsplit("_", 1)[1]
        exit_ts = datetime.fromisoformat(t["timestamp"]).timestamp()
        t["entry_ts"] = int((exit_ts - f(t["hold_candles"]) * 14400) // 14400) * 14400 + 120
        trades.append(t)

print("=" * 100)
print("V1) TREND_RIDER: deployte Hysterese (2 Kerzen) vs alt (1 Kerze) vs ohne Thesis-Exit")
print("=" * 100)
tr_trades = [t for t in trades if t["base"] == "trend_rider"]
res = defaultdict(lambda: dict(pnl=0.0, n=0, wins=0, mix=defaultdict(int)))
used = 0
for t in tr_trades:
    sym = t["sym"]
    entry = f(t["entry_price"])
    i = idx_at(sym, t["entry_ts"])
    if i < 1:
        continue
    ref = bars4h[sym]["O"][min(i + 1, len(bars4h[sym]["O"]) - 1)]
    if abs(ref - entry) / entry > 0.03:
        continue
    atr = bars4h[sym]["atr"][i] or (entry * 0.02)
    size = f(t["size_usd"])
    used += 1
    for label, hyst in [("ALT (1 Kerze)", 1), ("DEPLOYT (2 Kerzen)", 2), ("OHNE Thesis (99)", 99)]:
        ret, reason = replay_trendrider(sym, t["direction"], t["entry_ts"], entry, atr, hyst)
        r = res[label]
        r["pnl"] += size * ret
        r["n"] += 1
        r["wins"] += 1 if ret > 0 else 0
        r["mix"][reason] += 1
print(f"  repliziert: {used}/{len(tr_trades)} trend_rider-Trades")
real = sum(f(t["pnl"]) for t in tr_trades)
print(f"  ACTUAL live: ${real:+.2f}")
for label, r in res.items():
    print(f"  {label:20s} PnL=${r['pnl']:+8.2f}  WR={r['wins']/max(1,r['n'])*100:5.1f}%  Exits={dict(r['mix'])}")

print()
print("=" * 100)
print("V2) DRIFT-GATE WALK-FORWARD: Befund in beiden Datenhälften?")
print("=" * 100)
ema50 = {}
for sym in SYM_MAP:
    B = bars4h[sym]
    k = 2 / 51
    e = None
    es = []
    for c in B["C"]:
        e = c if e is None else c * k + e * (1 - k)
        es.append(e)
    ema50[sym] = es

def drift(sym, ts):
    i = idx_at(sym, ts) - 1
    if i < 1:
        return 0
    return 1 if bars4h[sym]["C"][i] > ema50[sym][i] else -1

mid = "2026-05-19"
for half, lo, hi in [("H1 (27.04-18.05)", "", mid), ("H2 (19.05-09.06)", mid, "9999")]:
    grp = defaultdict(list)
    for t in trades:
        if not (lo <= t["timestamp"] < hi):
            continue
        d = drift(t["sym"], t["entry_ts"])
        if d == 0:
            continue
        sgn = 1 if t["direction"] == "long" else -1
        grp["counter" if sgn != d else "aligned"].append(t)
    for g in ["counter", "aligned"]:
        ts_ = grp[g]
        if ts_:
            pnl = sum(f(t["pnl"]) for t in ts_)
            wr = sum(1 for t in ts_ if f(t["pnl"]) > 0) / len(ts_)
            print(f"  {half} {g:8s} n={len(ts_):4d} WR={wr*100:5.1f}% PnL=${pnl:+8.2f}")
    cp = sum(f(t["pnl"]) for t in grp["counter"])
    print(f"  {half} -> Soft-Gate-Ersparnis: ${-cp/2:+.2f}")
