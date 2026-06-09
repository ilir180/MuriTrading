# Deep Dive 2 — Live-Realität vs Counterfactual, Coach-Audit, Challenger-Autopsie
# Datenbasis: data/_live_snapshot/ (Server-Pull 2026-06-10)
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SNAP = Path(__file__).resolve().parents[1] / "data" / "_live_snapshot"

def load_csv(name):
    with open(SNAP / name, encoding="utf-8", errors="replace") as f:
        return list(csv.DictReader(f))

def f(x, default=0.0):
    try:
        return float(x)
    except (ValueError, TypeError):
        return default

def stats(trades):
    n = len(trades)
    if n == 0:
        return dict(n=0, wr=0, pnl=0, avg_ret=0, pf=0)
    wins = [t for t in trades if f(t["pnl"]) > 0]
    gross_w = sum(f(t["pnl"]) for t in wins)
    gross_l = -sum(f(t["pnl"]) for t in trades if f(t["pnl"]) <= 0)
    return dict(
        n=n,
        wr=len(wins) / n,
        pnl=sum(f(t["pnl"]) for t in trades),
        avg_ret=sum(f(t["net_return_pct"]) for t in trades) / n,
        pf=(gross_w / gross_l) if gross_l > 0 else float("inf"),
    )

def fmt(s):
    return f"n={s['n']:4d} WR={s['wr']*100:5.1f}% PnL=${s['pnl']:+8.2f} avgRet={s['avg_ret']:+6.3f}% PF={s['pf']:5.2f}"

live = load_csv("trades.csv")
cf = load_csv("counterfactual_trades.csv")
for t in live + cf:
    t["base"] = t["bot_id"].rsplit("_", 1)[0]
    t["sym"] = t["bot_id"].rsplit("_", 1)[1]

print("=" * 100)
print("A) LIVE GESAMT + PRO BOT (base, alle Symbole) — 2026-04-27 bis 2026-06-09")
print("=" * 100)
print(f"  TOTAL: {fmt(stats(live))}")
by_base = defaultdict(list)
for t in live:
    by_base[t["base"]].append(t)
for b, ts in sorted(by_base.items(), key=lambda kv: -stats(kv[1])["pnl"]):
    print(f"  {b:18s} {fmt(stats(ts))}")

print()
print("=" * 100)
print("B) CELL-LEVEL (bot×symbol): LIVE vs COUNTERFACTUAL — wo ist die Edge real?")
print("=" * 100)
live_cell = defaultdict(list)
cf_cell = defaultdict(list)
for t in live:
    live_cell[t["bot_id"]].append(t)
for t in cf:
    cf_cell[t["bot_id"]].append(t)
rows = []
for cell in sorted(set(live_cell) | set(cf_cell)):
    sl, sc = stats(live_cell[cell]), stats(cf_cell[cell])
    rows.append((cell, sl, sc))
rows.sort(key=lambda r: -r[1]["pnl"])
print(f"  {'cell':24s} {'LIVE':>52s}   {'CF (WR/avgRet)':>20s}")
for cell, sl, sc in rows:
    cf_str = f"WR={sc['wr']*100:4.1f}% aR={sc['avg_ret']:+6.3f}%" if sc["n"] else "—"
    flag = ""
    if sl["n"] >= 8:
        if sl["wr"] >= 0.55 and sc["n"] and sc["wr"] < 0.45:
            flag = "  << LIVE>>CF"
        if sl["wr"] < 0.35 and sc["n"] and sc["wr"] > 0.5:
            flag = "  << CF-LÜGE"
    print(f"  {cell:24s} {fmt(sl)}   {cf_str}{flag}")

print()
print("=" * 100)
print("C) REGIME-CLUSTER × BOT (live, nur n>=5) — Market Map live")
print("=" * 100)
reg_cell = defaultdict(list)
for t in live:
    reg_cell[(t["base"], t["regime_cluster"])].append(t)
for (b, rc), ts in sorted(reg_cell.items()):
    if len(ts) >= 5:
        print(f"  {b:18s} cluster={rc:>2s}  {fmt(stats(ts))}")

print()
print("=" * 100)
print("D) EXIT-REASON-BREAKDOWN (live, pro Bot)")
print("=" * 100)
for b, ts in sorted(by_base.items()):
    reasons = defaultdict(list)
    for t in ts:
        r = t["reason"].split(":")[0].strip()
        reasons[r].append(t)
    parts = []
    for r, rts in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
        s = stats(rts)
        parts.append(f"{r}:{s['n']}({s['wr']*100:.0f}%W ${s['pnl']:+.0f})")
    print(f"  {b:18s} {' | '.join(parts)}")

print()
print("=" * 100)
print("E) COACH-AUDIT: aktuelle Decisions vs Live-Performance der Cells seit 14 Tagen")
print("=" * 100)
coach = json.load(open(SNAP / "coach_state.json", encoding="utf-8"))
cutoff = "2026-05-26"
recent_cell = defaultdict(list)
for t in live:
    if t["timestamp"] >= cutoff:
        recent_cell[t["bot_id"]].append(t)
by_action = defaultdict(list)
for cell, d in coach["decisions"].items():
    by_action[(d["action"], d.get("invert", False))].append(cell)
for (action, inv), cells in sorted(by_action.items()):
    agg = [t for c in cells for t in recent_cell.get(c, [])]
    s = stats(agg)
    print(f"  action={action:8s} invert={str(inv):5s} cells={len(cells):2d}  seit {cutoff}: {fmt(s)}")
    for c in cells:
        if recent_cell.get(c):
            print(f"      {c:24s} {fmt(stats(recent_cell[c]))}")

print()
print("=" * 100)
print("F) CHALLENGER-AUTOPSIE")
print("=" * 100)
for name, sfile, tfile in [
    ("v1 (additive boost)", "challenger_state.json", "challenger_trades.csv"),
    ("v2 (inverted boost)", "challenger_v2_state.json", "challenger_v2_trades.csv"),
]:
    st = json.load(open(SNAP / sfile, encoding="utf-8"))
    ct = load_csv(tfile)
    print(f"  {name}: capital=${st['capital']:.0f} totalPnL=${st['total_pnl']:+.0f} "
          f"W/L={st['wins']}/{st['losses']} taken={st['trades_taken']} openPos={len(st['positions'])}")
    if ct:
        sizes = [f(t["size_usd"]) for t in ct]
        print(f"    closed={len(ct)}  avgSize=${sum(sizes)/len(sizes):.0f}  maxSize=${max(sizes):.0f}")
        # Offene Positionen: unrealisierte Klumpenrisiken (identische Entries?)
        ent = defaultdict(list)
        for pid, p in st["positions"].items():
            ent[(round(f(p["entry_price"]), 2), p["direction"])].append(pid)
        dups = {k: v for k, v in ent.items() if len(v) > 1}
        if dups:
            print(f"    KLUMPEN (identische offene Entries): {dups}")
        # Signal-Validität: sagt funding_z überhaupt etwas voraus?
        for bucket, cond in [
            ("funding_z > +1", lambda t: f(t["funding_z"]) > 1),
            ("funding_z < -1", lambda t: f(t["funding_z"]) < -1),
            ("|funding_z| <= 1", lambda t: abs(f(t["funding_z"])) <= 1),
        ]:
            bt = [t for t in ct if cond(t)]
            if bt:
                print(f"    {bucket:18s} {fmt(stats(bt))}")
        # Boost-Wirkung: Trades mit hohem Boost-Delta vs ohne
        for bucket, cond in [
            ("boost>= +0.10", lambda t: f(t["boosted_confidence"]) - f(t["base_confidence"]) >= 0.10),
            ("boost ~ 0", lambda t: abs(f(t["boosted_confidence"]) - f(t["base_confidence"])) < 0.10),
        ]:
            bt = [t for t in ct if cond(t)]
            if bt:
                print(f"    {bucket:18s} {fmt(stats(bt))}")
    print()

print("=" * 100)
print("G) EQUITY-VERLAUF (Gesamtsystem, Wochenenden)")
print("=" * 100)
eq = load_csv("equity.csv")
if eq:
    cols = list(eq[0].keys())
    print("  cols:", cols)
    step = max(1, len(eq) // 12)
    for r in eq[::step] + [eq[-1]]:
        vals = [v for k, v in r.items() if k != "timestamp"]
        print("  ", r.get("timestamp", "?")[:16], vals[:6])
