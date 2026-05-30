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

def load_state() -> ChallengerState:
    if not os.path.exists(CHALLENGER_STATE):
        return ChallengerState()
    try:
        with open(CHALLENGER_STATE) as f:
            data = json.load(f)
        s = ChallengerState()
        s.version = data.get("version", "challenger-1.0")
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


def save_state(state: ChallengerState):
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
    tmp = CHALLENGER_STATE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, CHALLENGER_STATE)
    except Exception:
        pass


def _append_trade(row: dict):
    """Append-only CSV. Header is written on first call."""
    write_header = not os.path.exists(CHALLENGER_TRADES)
    try:
        with open(CHALLENGER_TRADES, "a", encoding="utf-8") as f:
            if write_header:
                f.write(",".join(row.keys()) + "\n")
            f.write(",".join(str(v) for v in row.values()) + "\n")
    except Exception:
        pass


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
    risk_amount = state.capital * CHALLENGER_RISK_PER_TRADE
    size_usd = risk_amount / sl_pct
    size_usd = min(size_usd, state.capital * CHALLENGER_LEVERAGE)
    if size_usd < 5.0:
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


def on_tick(state: ChallengerState, asset_prices: Dict[str, float]) -> List[dict]:
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
            _append_trade(closed[-1])
            del state.positions[bot_id]
    return closed


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
