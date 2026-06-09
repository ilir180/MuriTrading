"""JV Boting v2 — Shadow Challenger v1.

Reads the Insight Bus, paper-trades each insight under an ALTERNATIVE
decision rule, persists its own equity series. Daily we compare Champion
(live Coach + bot path) vs Challenger.

v1 Challenger rule (what makes it different from Champion):
  - Champion decides direction from the bot signal alone.
  - Challenger BOOSTS confidence using Crypto-Native features:
      * funding_z aligned with signal direction  -> +0.10 conf
      * cvd_z aligned with signal direction      -> +0.10 conf
      * funding_z opposite to signal              -> -0.10 conf
      * cvd_z opposite to signal                  -> -0.10 conf
  - Trades only fire when effective confidence >= MIN_CONFIDENCE (0.20).
  - Position sizing: fixed 2% risk per trade, no Coach interaction.
  - Uses the same SL/TP/Trailing logic as Champion (so the test isolates
    the FEATURE-BOOST effect, not the execution-engine effect).

After 30+ days we compare:
  - If Challenger meaningfully outperforms Champion -> features matter,
    promote Challenger rules into Champion.
  - If Challenger underperforms -> the boost is noise, drop the feature
    integration.

State persisted to data/bot/jv2/challenger_state.json.
"""

import json
import os
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.jv2.config import JV2_DIR, MIN_CONFIDENCE, ROUND_TRIP


CHALLENGER_STATE = os.path.join(JV2_DIR, "challenger_state.json")
CHALLENGER_TRADES = os.path.join(JV2_DIR, "challenger_trades.csv")

# v2: inverted-boost variant. Same SL/TP/sizing, only the boost interpretation differs.
CHALLENGER_V2_STATE = os.path.join(JV2_DIR, "challenger_v2_state.json")
CHALLENGER_V2_TRADES = os.path.join(JV2_DIR, "challenger_v2_trades.csv")

# v2 thresholds for the inverted-boost rule.
# v1: 0.10/-0.10 (strict) required BOTH indicators to align — too rare in
# practice (0 trades over 24h after deploy). v2: 0.05/-0.05 means already
# one aligned indicator triggers, which generates a meaningful sample.
V2_BOOST_FLIP_THRESHOLD = 0.05    # boost > this -> flip direction (crowd aligned with bot -> bet against)
V2_BOOST_CONTRARIAN_THRESHOLD = -0.05  # boost < this -> take bot direction with magnitude as conf bonus

CHALLENGER_INITIAL_CAPITAL = 4000.0  # mirrors Champion
CHALLENGER_RISK_PER_TRADE = 0.02
CHALLENGER_LEVERAGE = 1.0            # un-leveraged for clean A/B
SL_ATR_MULT = 2.0
TP_ATR_MULT = 3.0


@dataclass
class ChallengerPosition:
    bot_id: str
    direction: str
    entry_price: float
    size_usd: float
    sl: float
    tp: float
    atr: float
    entry_time: str
    base_confidence: float
    boosted_confidence: float
    funding_z: float
    cvd_z: float


@dataclass
class ChallengerState:
    version: str = "challenger-1.0"
    last_update: str = ""
    capital: float = CHALLENGER_INITIAL_CAPITAL
    total_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    trades_taken: int = 0
    positions: Dict[str, dict] = field(default_factory=dict)  # bot_id -> position dict


# ── Persistence ──

def load_state(path: str = CHALLENGER_STATE,
               default_version: str = "challenger-1.0") -> ChallengerState:
    if not os.path.exists(path):
        s = ChallengerState()
        s.version = default_version
        return s
    try:
        with open(path) as f:
            data = json.load(f)
        s = ChallengerState()
        s.version = data.get("version", default_version)
        s.last_update = data.get("last_update", "")
        s.capital = float(data.get("capital", CHALLENGER_INITIAL_CAPITAL))
        s.total_pnl = float(data.get("total_pnl", 0.0))
        s.wins = int(data.get("wins", 0))
        s.losses = int(data.get("losses", 0))
        s.trades_taken = int(data.get("trades_taken", 0))
        s.positions = dict(data.get("positions", {}))
        return s
    except Exception:
        return ChallengerState()


def save_state(state: ChallengerState, path: str = CHALLENGER_STATE):
    state.last_update = datetime.now(timezone.utc).isoformat()
    payload = {
        "version": state.version,
        "last_update": state.last_update,
        "capital": round(state.capital, 4),
        "total_pnl": round(state.total_pnl, 4),
        "wins": state.wins,
        "losses": state.losses,
        "trades_taken": state.trades_taken,
        "positions": state.positions,
    }
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, path)
    except Exception:
        pass


def _append_trade(row: dict, path: str = CHALLENGER_TRADES):
    """Append-only CSV. Header is written on first call."""
    write_header = not os.path.exists(path)
    try:
        with open(path, "a", encoding="utf-8") as f:
            if write_header:
                f.write(",".join(row.keys()) + "\n")
            f.write(",".join(str(v) for v in row.values()) + "\n")
    except Exception:
        pass


# v2 wrappers
def load_state_v2() -> ChallengerState:
    return load_state(CHALLENGER_V2_STATE, default_version="challenger-2.0")


def save_state_v2(state: ChallengerState):
    save_state(state, CHALLENGER_V2_STATE)


# ── Boost logic ──

def boosted_confidence(base_confidence: float, direction: str,
                       funding_z: float, cvd_z: float) -> float:
    """Apply directional boost/cut based on crypto-native features.

    Bullish signal + bullish flow features = boost.
    Bullish signal + bearish flow features = cut.
    """
    sign = 1 if direction == "long" else -1
    boost = 0.0
    # Funding-Z aligned with direction (>+1 means strong crowd buy = bullish)
    if sign * funding_z > 1.0:
        boost += 0.10
    elif sign * funding_z < -1.0:
        boost -= 0.10
    # CVD-Z aligned with direction
    if sign * cvd_z > 1.0:
        boost += 0.10
    elif sign * cvd_z < -1.0:
        boost -= 0.10
    out = base_confidence + boost
    return max(0.0, min(1.0, out))


def raw_boost(direction: str, funding_z: float, cvd_z: float) -> float:
    """The boost value alone, not added to base confidence. Used by v2 logic."""
    sign = 1 if direction == "long" else -1
    boost = 0.0
    if sign * funding_z > 1.0:
        boost += 0.10
    elif sign * funding_z < -1.0:
        boost -= 0.10
    if sign * cvd_z > 1.0:
        boost += 0.10
    elif sign * cvd_z < -1.0:
        boost -= 0.10
    return boost


def v2_decision(base_direction: str, base_confidence: float,
                funding_z: float, cvd_z: float):
    """Returns (direction, effective_confidence, mode) or (None, 0, 'skip').

    Three branches:
      - boost > +0.10  : crowd CONFIRMS bot signal -> FLIP direction
                         (anti-crowd bet, conf += boost magnitude)
      - boost < -0.10  : crowd OPPOSES bot signal -> bot is contrarian, take
                         original direction with conf += |boost|
      - else           : signal too weak in boost dimension -> skip
    """
    boost = raw_boost(base_direction, funding_z, cvd_z)
    if boost > V2_BOOST_FLIP_THRESHOLD:
        flipped = "short" if base_direction == "long" else "long"
        conf = max(0.0, min(1.0, base_confidence + abs(boost)))
        return flipped, conf, "flip"
    if boost < V2_BOOST_CONTRARIAN_THRESHOLD:
        conf = max(0.0, min(1.0, base_confidence + abs(boost)))
        return base_direction, conf, "contrarian_confirm"
    return None, 0.0, "skip"


def fair_size(state, sl_pct: float):
    """Faires Sizing auf Cell-Basis + Exposure-Cap.

    Der alte Code nutzte den GESAMT-Pool ($4000) statt Cell-Kapital —
    16-64x oversized vs Baseline, dazu Klumpen aus bis zu 4 identischen
    Positionen pro Symbol. Resultat: v1 -64%, v2 0/13 — der A/B-Test war
    dadurch ungültig (Deep Dive 10.06.26). Jetzt: Cell-Kapital wie bei den
    Baseline-Bots (Pool/32) und Gesamt-Exposure max 2x Kapital.
    Returns size_usd oder None wenn kein Trade möglich.
    """
    if state.capital <= 0:
        return None
    cell_capital = state.capital / 32.0  # 8 Bots x 4 Assets
    risk_amount = cell_capital * CHALLENGER_RISK_PER_TRADE
    size_usd = risk_amount / sl_pct
    size_usd = min(size_usd, cell_capital * CHALLENGER_LEVERAGE)
    if size_usd < 5.0:
        return None
    open_exposure = sum(float(p.get("size_usd", 0)) for p in state.positions.values())
    if open_exposure + size_usd > state.capital * 2.0:
        return None
    return size_usd


# ── Trade lifecycle ──

def on_signal(state: ChallengerState, bot_id: str, asset: str,
              direction: str, base_confidence: float,
              price: float, atr: float, market_data: dict) -> bool:
    """A new bot signal arrived. Apply Challenger rules and maybe open paper trade.
    Returns True if a paper position was opened."""
    if direction == "neutral":
        return False
    if bot_id in state.positions:
        return False  # already has open paper position for this bot

    futures = market_data.get("futures", {}) or {}
    cvd = market_data.get("cvd", {}) or {}
    funding_z = float(futures.get("funding_z", 0.0))
    cvd_z = float(cvd.get("cvd_1h_z", 0.0))

    eff_conf = boosted_confidence(base_confidence, direction, funding_z, cvd_z)
    if eff_conf < MIN_CONFIDENCE:
        return False

    # Sizing: 2% risk / SL distance
    sl_dist = SL_ATR_MULT * atr
    tp_dist = TP_ATR_MULT * atr
    if direction == "long":
        sl = price - sl_dist
        tp = price + tp_dist
    else:
        sl = price + sl_dist
        tp = price - tp_dist
    sl_pct = abs(price - sl) / price
    if sl_pct < 0.001:
        sl_pct = 0.01
    size_usd = fair_size(state, sl_pct)
    if size_usd is None:
        return False

    pos = ChallengerPosition(
        bot_id=bot_id, direction=direction, entry_price=price,
        size_usd=round(size_usd, 2), sl=round(sl, 6), tp=round(tp, 6),
        atr=atr, entry_time=datetime.now(timezone.utc).isoformat(),
        base_confidence=base_confidence,
        boosted_confidence=eff_conf,
        funding_z=funding_z, cvd_z=cvd_z,
    )
    state.positions[bot_id] = pos.__dict__
    state.trades_taken += 1
    return True


def on_tick(state: ChallengerState, asset_prices: Dict[str, float],
            trades_path: str = CHALLENGER_TRADES) -> List[dict]:
    """Check SL/TP for all open paper positions. Returns list of closed trades."""
    closed = []
    for bot_id, pos in list(state.positions.items()):
        price = None
        # bot_id contains the asset short suffix (e.g. flow_tracker_BTC)
        for asset, p in asset_prices.items():
            short = asset.split("/")[0]
            if bot_id.endswith("_" + short):
                price = p
                break
        if price is None or price <= 0:
            continue
        direction = pos["direction"]
        sl = pos["sl"]
        tp = pos["tp"]
        exit_reason = None
        exit_price = None
        if direction == "long":
            if price <= sl:
                exit_reason, exit_price = "STOP-LOSS", sl
            elif price >= tp:
                exit_reason, exit_price = "TAKE-PROFIT", tp
        else:
            if price >= sl:
                exit_reason, exit_price = "STOP-LOSS", sl
            elif price <= tp:
                exit_reason, exit_price = "TAKE-PROFIT", tp
        if exit_reason:
            # Compute PnL
            size = pos["size_usd"]
            entry = pos["entry_price"]
            if direction == "long":
                ret = (exit_price - entry) / entry
            else:
                ret = (entry - exit_price) / entry
            ret_net = ret - ROUND_TRIP
            pnl = size * ret_net
            state.capital += pnl
            state.total_pnl += pnl
            if pnl > 0:
                state.wins += 1
            else:
                state.losses += 1
            closed.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "bot_id": bot_id,
                "direction": direction,
                "entry_price": entry,
                "exit_price": exit_price,
                "size_usd": size,
                "pnl": round(pnl, 4),
                "net_return_pct": round(ret_net * 100, 4),
                "reason": exit_reason,
                "boosted_confidence": pos["boosted_confidence"],
                "base_confidence": pos["base_confidence"],
                "funding_z": pos["funding_z"],
                "cvd_z": pos["cvd_z"],
            })
            _append_trade(closed[-1], path=trades_path)
            del state.positions[bot_id]
    return closed


def on_tick_v2(state: ChallengerState, asset_prices: Dict[str, float]) -> List[dict]:
    """v2 wrapper that writes to challenger_v2_trades.csv."""
    return on_tick(state, asset_prices, trades_path=CHALLENGER_V2_TRADES)


def on_signal_v2(state: ChallengerState, bot_id: str, asset: str,
                 direction: str, base_confidence: float,
                 price: float, atr: float, market_data: dict) -> bool:
    """Inverted-boost variant. Direction is FLIPPED when crowd aligns with bot
    (boost > +0.10). When crowd opposes bot (boost < -0.10), bot is contrarian
    and we take the same direction with boost magnitude added to confidence.
    Other cases: skip the signal entirely (no trade)."""
    if direction == "neutral":
        return False
    if bot_id in state.positions:
        return False

    futures = market_data.get("futures", {}) or {}
    cvd = market_data.get("cvd", {}) or {}
    funding_z = float(futures.get("funding_z", 0.0))
    cvd_z = float(cvd.get("cvd_1h_z", 0.0))

    v2_dir, v2_conf, mode = v2_decision(direction, base_confidence, funding_z, cvd_z)
    if v2_dir is None or v2_conf < MIN_CONFIDENCE:
        return False

    sl_dist = SL_ATR_MULT * atr
    tp_dist = TP_ATR_MULT * atr
    if v2_dir == "long":
        sl = price - sl_dist
        tp = price + tp_dist
    else:
        sl = price + sl_dist
        tp = price - tp_dist
    sl_pct = abs(price - sl) / price
    if sl_pct < 0.001:
        sl_pct = 0.01
    size_usd = fair_size(state, sl_pct)
    if size_usd is None:
        return False

    pos = ChallengerPosition(
        bot_id=bot_id, direction=v2_dir, entry_price=price,
        size_usd=round(size_usd, 2), sl=round(sl, 6), tp=round(tp, 6),
        atr=atr, entry_time=datetime.now(timezone.utc).isoformat(),
        base_confidence=base_confidence,
        boosted_confidence=v2_conf,
        funding_z=funding_z, cvd_z=cvd_z,
    )
    # Stash v2 mode in the dict for later analysis
    d = pos.__dict__.copy()
    d["v2_mode"] = mode
    d["v2_base_direction"] = direction
    state.positions[bot_id] = d
    state.trades_taken += 1
    return True


# ── Comparison ──

def compare_vs_champion(state: ChallengerState, champion_total_pnl: float,
                         champion_n_trades: int, champion_n_wins: int) -> dict:
    """Return a comparison snapshot for daily report."""
    n_trades = state.wins + state.losses
    wr = state.wins / n_trades if n_trades > 0 else 0.0
    champ_wr = champion_n_wins / champion_n_trades if champion_n_trades > 0 else 0.0
    return {
        "challenger_pnl": round(state.total_pnl, 2),
        "challenger_capital": round(state.capital, 2),
        "challenger_trades": n_trades,
        "challenger_wr": round(wr, 3),
        "champion_pnl": round(champion_total_pnl, 2),
        "champion_trades": champion_n_trades,
        "champion_wr": round(champ_wr, 3),
        "delta_pnl": round(state.total_pnl - champion_total_pnl, 2),
        "verdict": (
            "challenger_winning" if state.total_pnl > champion_total_pnl + 5
            else "champion_winning" if champion_total_pnl > state.total_pnl + 5
            else "no_decision"
        ),
    }
