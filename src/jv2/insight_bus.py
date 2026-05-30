"""JV Boting v2 — Insight Bus (Additive Foundation Layer).

Phase-1 implementation of the Renaissance-pattern architecture: bots publish
Insight objects to a central bus alongside (NOT instead of) their existing
trade path. The bus persists insights to a JSONL file for later analysis,
and exposes a snapshot API for Shadow-Challenger paper-trading and Meta-
Learner training.

This is the FOUNDATION — the existing Trader (Coach + base_bot) is
unaffected. Once we have N weeks of insight history we can:
  - Train a Meta-Learner on insight outcomes
  - Run alternate Coaches as Shadow-Challengers
  - Eventually refactor to bus-as-only-path

For now: insights are emitted, persisted, and queryable.
"""

import json
import os
import threading
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.jv2.config import JV2_DIR


INSIGHTS_JSONL = os.path.join(JV2_DIR, "insights.jsonl")
INSIGHT_BUFFER_SIZE = 5000   # in-memory ring buffer for fast queries


@dataclass
class Insight:
    """An atomic insight from a bot. Direction-only — sizing/risk lives in
    the Trader. Half-life is how many candles the insight is considered valid."""
    bot_id: str
    asset: str
    direction: str               # "long" | "short" | "neutral"
    confidence: float            # [0..1]
    reasoning: str
    price_at_signal: float
    regime_cluster: int = -1
    half_life_candles: int = 6
    generated_at: str = ""
    # Optional outcome fields — filled in by the bus when we close a trade
    # that was opened on this insight. Used for Meta-Learner training.
    outcome_pnl: Optional[float] = None
    outcome_hold: Optional[int] = None
    outcome_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


class InsightBus:
    """Process-wide singleton. Thread-safe ring buffer + JSONL append."""

    _instance: Optional["InsightBus"] = None
    _lock = threading.Lock()

    def __init__(self):
        self.buffer: deque = deque(maxlen=INSIGHT_BUFFER_SIZE)
        self.file_lock = threading.Lock()
        # Map signal-id -> insight ref for outcome linking
        self.recent_by_bot: Dict[str, List[Insight]] = {}

    @classmethod
    def get(cls) -> "InsightBus":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def publish(self, insight: Insight) -> Insight:
        """Add insight to ring buffer + append-only JSONL file."""
        if not insight.generated_at:
            insight.generated_at = datetime.now(timezone.utc).isoformat()
        self.buffer.append(insight)
        # Keep per-bot tail for outcome linking
        tail = self.recent_by_bot.setdefault(insight.bot_id, [])
        tail.append(insight)
        if len(tail) > 50:
            tail.pop(0)
        # Persist
        try:
            with self.file_lock:
                with open(INSIGHTS_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(insight.to_dict()) + "\n")
        except Exception:
            pass
        return insight

    def link_outcome(self, bot_id: str, pnl: float, hold: int, reason: str,
                     match_window_minutes: int = 240) -> Optional[Insight]:
        """Find the most recent non-neutral insight from `bot_id` whose
        outcome is unset, and attach the outcome. Returns the linked insight."""
        tail = self.recent_by_bot.get(bot_id, [])
        # Iterate newest-to-oldest
        for ins in reversed(tail):
            if ins.outcome_pnl is not None:
                continue
            if ins.direction == "neutral":
                continue
            ins.outcome_pnl = pnl
            ins.outcome_hold = hold
            ins.outcome_reason = reason
            # Re-persist a slim outcome line (not the full insight)
            try:
                with self.file_lock:
                    with open(INSIGHTS_JSONL, "a", encoding="utf-8") as f:
                        f.write(json.dumps({
                            "outcome": True,
                            "bot_id": bot_id,
                            "linked_at": datetime.now(timezone.utc).isoformat(),
                            "pnl": pnl,
                            "hold": hold,
                            "reason": reason,
                            "insight_generated_at": ins.generated_at,
                        }) + "\n")
            except Exception:
                pass
            return ins
        return None

    def snapshot(self, last_n: int = 200) -> List[Insight]:
        """Return the most recent N insights (newest last)."""
        return list(self.buffer)[-last_n:]

    def filter(self, bot_id: Optional[str] = None,
               direction: Optional[str] = None,
               since_iso: Optional[str] = None) -> List[Insight]:
        out = []
        for ins in self.buffer:
            if bot_id and ins.bot_id != bot_id:
                continue
            if direction and ins.direction != direction:
                continue
            if since_iso and ins.generated_at < since_iso:
                continue
            out.append(ins)
        return out


# Convenience module-level functions
def publish(bot_id: str, asset: str, direction: str, confidence: float,
            reasoning: str, price: float, regime_cluster: int = -1,
            half_life_candles: int = 6) -> Insight:
    ins = Insight(
        bot_id=bot_id, asset=asset, direction=direction,
        confidence=confidence, reasoning=reasoning,
        price_at_signal=price, regime_cluster=regime_cluster,
        half_life_candles=half_life_candles,
    )
    return InsightBus.get().publish(ins)


def link_outcome(bot_id: str, pnl: float, hold: int, reason: str) -> Optional[Insight]:
    return InsightBus.get().link_outcome(bot_id, pnl, hold, reason)


def snapshot(last_n: int = 200) -> List[Insight]:
    return InsightBus.get().snapshot(last_n)
