"""
Joint Venture Boting – Credit System
Jeder Bot verdient seinen Einfluss durch korrekte Signale.

Mechanik:
  - Richtig + hohe Confidence → grosse Belohnung
  - Falsch + hohe Confidence → grosse Strafe (Overconfidence kostet)
  - Neutral → kein Change, aber Decay läuft weiter
  - Credits verfallen: ×0.99 pro 4H-Kerze (~12 Tage Halbwertszeit)
  - Leader = höchste Credits → hat das Sagen
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional

from src.jv.config import (
    INITIAL_CREDITS, MAX_CREDITS, MIN_CREDITS, MIN_INFLUENCE,
    CREDIT_REWARD, CREDIT_PENALTY, MAGNITUDE_BONUS,
    DECAY_FACTOR, EVAL_MIN_MOVE,
)
from src.jv.signal_protocol import JVSignal


class CreditLedger:
    """Verwaltet Credits aller JV-Bots."""

    def __init__(self, bot_ids: list):
        self.bot_ids = bot_ids
        self.credits = {bid: INITIAL_CREDITS for bid in bot_ids}
        self.pending: list = []     # Signals die noch evaluiert werden müssen
        self.history: list = []     # Chronologische Credit-Änderungen
        self.eval_count = {bid: 0 for bid in bot_ids}
        self.correct_count = {bid: 0 for bid in bot_ids}

    # ── Credits ───────────────────────────────────────

    def get_credits(self, bot_id: str) -> float:
        return self.credits.get(bot_id, 0.0)

    def get_leader(self) -> str:
        """Bot mit den meisten Credits."""
        return max(self.credits, key=self.credits.get)

    def get_ranking(self) -> list:
        """Alle Bots sortiert nach Credits (absteigend)."""
        return sorted(self.credits.items(), key=lambda x: x[1], reverse=True)

    def get_weights(self) -> dict:
        """
        Normalisierte Gewichte für Prime-Aggregation.
        Bots unter MIN_INFLUENCE bekommen Gewicht 0.
        """
        eligible = {bid: cr for bid, cr in self.credits.items()
                    if cr >= MIN_INFLUENCE}
        if not eligible:
            return {bid: 0.0 for bid in self.bot_ids}

        total = sum(eligible.values())
        if total == 0:
            return {bid: 0.0 for bid in self.bot_ids}

        weights = {}
        for bid in self.bot_ids:
            if bid in eligible:
                weights[bid] = eligible[bid] / total
            else:
                weights[bid] = 0.0
        return weights

    # ── Signal Recording & Evaluation ─────────────────

    def record_signal(self, signal: JVSignal):
        """Speichert Signal für spätere Evaluation."""
        if signal.direction == "neutral":
            return  # Neutral wird nicht evaluiert

        self.pending.append({
            "signal": signal.to_dict(),
            "eval_after_candles": signal.ttl_candles,
            "candles_waited": 0,
        })

    def evaluate_pending(self, current_price: float, atr: float):
        """
        Evaluiert alle fälligen Pending-Signals.
        Wird bei jeder neuen 4H-Kerze aufgerufen.
        """
        still_pending = []
        results = []

        for entry in self.pending:
            entry["candles_waited"] += 1

            if entry["candles_waited"] < entry["eval_after_candles"]:
                still_pending.append(entry)
                continue

            # Evaluation!
            sig = entry["signal"]
            bot_id = sig["bot_id"]
            signal_price = sig["price_at_signal"]
            confidence = sig["confidence"]
            direction = sig["direction"]

            if signal_price == 0:
                continue

            # Tatsächliche Bewegung
            actual_move = (current_price - signal_price) / signal_price

            # Korrekt?
            if direction == "long":
                correct = actual_move > EVAL_MIN_MOVE
            else:  # short
                correct = actual_move < -EVAL_MIN_MOVE

            # Credit-Änderung berechnen
            if correct:
                delta = confidence * CREDIT_REWARD
                # Magnitude Bonus für grosse Moves
                if atr > 0:
                    move_magnitude = abs(actual_move * signal_price) / atr
                    bonus = min(move_magnitude, 2.0) * MAGNITUDE_BONUS
                    delta += bonus
            else:
                delta = confidence * CREDIT_PENALTY  # Negativ!

            # Credits anpassen
            old_credits = self.credits[bot_id]
            self.credits[bot_id] = max(MIN_CREDITS,
                                       min(MAX_CREDITS, old_credits + delta))

            # Statistik
            self.eval_count[bot_id] = self.eval_count.get(bot_id, 0) + 1
            if correct:
                self.correct_count[bot_id] = self.correct_count.get(bot_id, 0) + 1

            result = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "bot_id": bot_id,
                "direction": direction,
                "confidence": confidence,
                "signal_price": signal_price,
                "eval_price": current_price,
                "actual_move_pct": round(actual_move * 100, 4),
                "correct": correct,
                "delta": round(delta, 2),
                "credits_before": round(old_credits, 2),
                "credits_after": round(self.credits[bot_id], 2),
            }
            self.history.append(result)
            results.append(result)

        self.pending = still_pending
        return results

    # ── Decay ─────────────────────────────────────────

    def apply_decay(self):
        """Alle Credits × DECAY_FACTOR. Wird pro 4H-Kerze aufgerufen."""
        for bid in self.bot_ids:
            self.credits[bid] = max(MIN_CREDITS,
                                    self.credits[bid] * DECAY_FACTOR)

    # ── Bot Accuracy ──────────────────────────────────

    def get_accuracy(self, bot_id: str) -> Optional[float]:
        total = self.eval_count.get(bot_id, 0)
        if total == 0:
            return None
        return self.correct_count.get(bot_id, 0) / total

    # ── Persistence ───────────────────────────────────

    def save(self, path: str):
        state = {
            "credits": {k: round(v, 2) for k, v in self.credits.items()},
            "pending": self.pending,
            "history": self.history[-500:],  # Letzte 500 Einträge
            "eval_count": self.eval_count,
            "correct_count": self.correct_count,
            "leader": self.get_leader(),
            "last_update": datetime.now(timezone.utc).isoformat(),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path) as f:
            state = json.load(f)
        for bid in self.bot_ids:
            if bid in state.get("credits", {}):
                self.credits[bid] = state["credits"][bid]
        self.pending = state.get("pending", [])
        self.history = state.get("history", [])
        self.eval_count = state.get("eval_count", {bid: 0 for bid in self.bot_ids})
        self.correct_count = state.get("correct_count", {bid: 0 for bid in self.bot_ids})

    # ── Display ───────────────────────────────────────

    def leaderboard_text(self) -> str:
        """Formatierter Leaderboard-Text für Telegram/Log."""
        ranking = self.get_ranking()
        leader = ranking[0][0] if ranking else "?"
        lines = []
        for i, (bid, cr) in enumerate(ranking):
            acc = self.get_accuracy(bid)
            acc_str = f"{acc:.0%}" if acc is not None else "–"
            evals = self.eval_count.get(bid, 0)
            crown = " \U0001F451" if bid == leader else ""
            lines.append(f"  {i+1}. {bid:12s}  {cr:6.1f} Credits  "
                         f"Acc: {acc_str} ({evals} evals){crown}")
        return "\n".join(lines)
