# Smoke-Tests für die Deep-Dive-2-Fixes
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import src.jv2.insight_bus as ib

# ── 1) InsightBus: Rehydration aus Snapshot-JSONL ──
ib.INSIGHTS_JSONL = str(ROOT / "data" / "_live_snapshot" / "insights.jsonl")
ib.InsightBus._instance = None
bus = ib.InsightBus.get()
n_tails = sum(len(v) for v in bus.recent_by_bot.values())
linked = sum(1 for v in bus.recent_by_bot.values() for i in v if i.outcome_reason == "rehydrated-linked")
print(f"1) Rehydrate: {len(bus.recent_by_bot)} Bots, {n_tails} Insights in Tails, {linked} als verknüpft markiert")
assert n_tails > 100, "Rehydration leer"

# ── 2) link_outcome: zeitbasiertes Matching ──
ib.INSIGHTS_JSONL = str(ROOT / "data" / "_smoke_insights.jsonl")
now = datetime.now(timezone.utc)
bus.recent_by_bot["smoke_bot"] = []
for hours_ago, conf in [(24.1, 0.11), (12.0, 0.22), (0.1, 0.33)]:
    ins = ib.Insight(bot_id="smoke_bot", asset="XRP/USDT", direction="long",
                     confidence=conf, reasoning="t", price_at_signal=1.0,
                     generated_at=(now - timedelta(hours=hours_ago)).isoformat())
    bus.recent_by_bot["smoke_bot"].append(ins)
got = bus.link_outcome("smoke_bot", pnl=5.0, hold=3, reason="TP")  # 3*4h=12h zurück
assert got is not None and got.confidence == 0.22, f"Zeitmatch falsch: {got and got.confidence}"
print(f"2) link_outcome: hold=3 -> Insight von vor 12h gematcht (conf {got.confidence}) OK")
got2 = bus.link_outcome("smoke_bot", pnl=1.0, hold=18, reason="TIME")  # 72h zurück, keins da
assert got2 is not None, "Fallback griff nicht"
print(f"2b) Fallback: hold=18 ohne Kandidat im Fenster -> neuestes ({got2.confidence}) OK")

# ── 3) Challenger fair_size ──
from src.jv2 import challenger as ch
class FakeState:
    capital = 4000.0
    positions = {}
st = FakeState()
s = ch.fair_size(st, sl_pct=0.05)
cell = 4000 / 32
expected = min(cell * ch.CHALLENGER_RISK_PER_TRADE / 0.05, cell * ch.CHALLENGER_LEVERAGE)
assert s is not None and abs(s - expected) < 0.01, f"fair_size {s} != {expected}"
print(f"3) fair_size: ${s:.2f} (Cell-basiert, vorher wäre ${min(4000*ch.CHALLENGER_RISK_PER_TRADE/0.05, 4000*ch.CHALLENGER_LEVERAGE):.0f}) OK")
st.positions = {f"p{i}": {"size_usd": 1000} for i in range(8)}  # 8000 offen = 2x Kapital
assert ch.fair_size(st, 0.05) is None, "Exposure-Cap griff nicht"
print("3b) Exposure-Cap bei 2x Kapital greift OK")
st.positions = {}
st.capital = -10
assert ch.fair_size(st, 0.05) is None, "Bankruptcy-Guard griff nicht"
print("3c) Bankruptcy-Guard greift OK")

# ── 4) trend_rider Hysterese ──
from src.jv2.bots.trend_rider import TrendRider
tr = TrendRider("XRP/USDT")
class FakePos:
    direction = "long"
tr.state.position = FakePos()
md_bad = {"latest_4h": {"4h_adx": 30, "4h_ema_9_above_21": 0.0}}
ok1, _ = tr.check_thesis(md_bad)
ok2, why2 = tr.check_thesis(md_bad)
assert ok1 is True and ok2 is False, f"Hysterese falsch: {ok1}/{ok2}"
print(f"4) trend_rider: 1. Gegen-Kerze toleriert, 2. exitet ('{why2}') OK")
tr.state.position = FakePos()
ok3, _ = tr.check_thesis(md_bad)
md_good = {"latest_4h": {"4h_adx": 30, "4h_ema_9_above_21": 1.0}}
ok4, _ = tr.check_thesis(md_good)
ok5, _ = tr.check_thesis(md_bad)
assert ok3 and ok4 and ok5, "Strike-Reset bei gültiger Kerze fehlt"
print("4b) Strike-Reset nach gültiger Kerze OK")

# ── 5) Drift-Gate ──
import pandas as pd
closes_up = pd.DataFrame({"close": [1.0 + i * 0.01 for i in range(60)]})
closes_dn = pd.DataFrame({"close": [2.0 - i * 0.01 for i in range(60)]})
assert tr._market_drift({"df_4h": closes_up}) == 1
assert tr._market_drift({"df_4h": closes_dn}) == -1
assert tr._market_drift({}) == 0
print("5) _market_drift: up=+1, down=-1, fehlend=0 OK")

# size-Halbierung
sig_type = type("S", (), {"direction": "long", "confidence": 0.6})
tr.state.position = None
tr.state.capital = 125.0
tr.state.wins = tr.state.losses = 0
r_with = tr._open_position(sig_type(), 1.0, 0.02, {}, drift=-1)
size_counter = tr.state.position.size_usd
tr.state.position = None
r_align = tr._open_position(sig_type(), 1.0, 0.02, {}, drift=1)
size_aligned = tr.state.position.size_usd
assert abs(size_counter * 2 - size_aligned) < 0.05, f"{size_counter} vs {size_aligned}"
print(f"5b) Drift-Gate: counter ${size_counter} = 0.5x aligned ${size_aligned} OK")

print()
print("ALLE SMOKE-TESTS GRÜN")
