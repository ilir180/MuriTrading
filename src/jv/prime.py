"""
Joint Venture Boting – Prime Bot
Der Execution-Spezialist. Aggregiert credit-gewichtete Signale und handelt.

Die JV-Bots sagen WAS und WANN.
Der Prime entscheidet WIE: Sizing, Stops, Trailing, Partial TPs.
"""

import math
import json
import os
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Optional

from src.jv.config import *
from src.jv.signal_protocol import JVSignal
from src.jv.credit_system import CreditLedger


# ── Position (aus v3 übernommen, angepasst) ───────────

class Position:
    def __init__(self, direction, strategy, entry_price, size, original_size,
                 stop_loss, take_profit, atr, entry_time, consensus_details=""):
        self.direction = direction
        self.strategy = strategy
        self.entry_price = entry_price
        self.size = size
        self.original_size = original_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.atr = atr
        self.entry_time = entry_time
        self.candles_held = 0
        self.peak_price = entry_price
        self.trough_price = entry_price
        self.trailing_active = False
        self.trailing_stop = 0.0
        self.partial_taken = False
        self.partial_pnl = 0.0
        self.consensus_details = consensus_details

    def update_trailing(self, current_price):
        if self.direction == "long":
            self.peak_price = max(self.peak_price, current_price)
            if self.peak_price - self.entry_price >= TRAILING_ACTIVATE * self.atr:
                self.trailing_active = True
                new_trail = self.peak_price - TRAILING_ATR_MULT * self.atr
                self.trailing_stop = max(self.trailing_stop, new_trail, self.stop_loss)
        else:
            self.trough_price = min(self.trough_price, current_price)
            if self.entry_price - self.trough_price >= TRAILING_ACTIVATE * self.atr:
                self.trailing_active = True
                new_trail = self.trough_price + TRAILING_ATR_MULT * self.atr
                if self.trailing_stop == 0:
                    self.trailing_stop = new_trail
                else:
                    self.trailing_stop = min(self.trailing_stop, new_trail)
                self.trailing_stop = min(self.trailing_stop, self.stop_loss)

    def check_partial_tp(self, current_price):
        if self.partial_taken:
            return False, 0.0
        sl_dist = abs(self.entry_price - self.stop_loss)
        partial_target = sl_dist * PARTIAL_TP_RR

        if self.direction == "long" and current_price >= self.entry_price + partial_target - 1e-9:
            partial_size = self.size * PARTIAL_SIZE_PCT
            raw_ret = (current_price - self.entry_price) / self.entry_price
            partial_pnl = partial_size * (raw_ret - ROUND_TRIP / 2)
            self.size -= partial_size
            self.partial_taken = True
            self.partial_pnl = partial_pnl
            self.stop_loss = max(self.stop_loss, self.entry_price * (1 + BREAKEVEN_BUFFER))
            if self.trailing_stop > 0:
                self.trailing_stop = max(self.trailing_stop, self.stop_loss)
            return True, partial_pnl

        if self.direction == "short" and current_price <= self.entry_price - partial_target + 1e-9:
            partial_size = self.size * PARTIAL_SIZE_PCT
            raw_ret = (self.entry_price - current_price) / self.entry_price
            partial_pnl = partial_size * (raw_ret - ROUND_TRIP / 2)
            self.size -= partial_size
            self.partial_taken = True
            self.partial_pnl = partial_pnl
            self.stop_loss = min(self.stop_loss, self.entry_price * (1 - BREAKEVEN_BUFFER))
            if self.trailing_stop > 0:
                self.trailing_stop = min(self.trailing_stop, self.stop_loss)
            return True, partial_pnl

        return False, 0.0

    def check_exit(self, current_price):
        if self.direction == "long" and current_price <= self.stop_loss:
            return True, "STOP-LOSS"
        if self.direction == "short" and current_price >= self.stop_loss:
            return True, "STOP-LOSS"
        if self.direction == "long" and current_price >= self.take_profit:
            return True, "TAKE-PROFIT"
        if self.direction == "short" and current_price <= self.take_profit:
            return True, "TAKE-PROFIT"
        if self.trailing_active:
            if self.direction == "long" and current_price <= self.trailing_stop:
                return True, "TRAILING-STOP"
            if self.direction == "short" and current_price >= self.trailing_stop:
                return True, "TRAILING-STOP"
        if self.candles_held >= MAX_HOLD_CANDLES:
            return True, "TIME-EXIT"
        return False, ""

    def calc_pnl(self, exit_price):
        if self.direction == "long":
            raw_ret = (exit_price - self.entry_price) / self.entry_price
        else:
            raw_ret = (self.entry_price - exit_price) / self.entry_price
        net_ret = raw_ret - ROUND_TRIP
        pnl = self.size * net_ret + self.partial_pnl
        return pnl, raw_ret, net_ret

    def to_dict(self):
        return self.__dict__.copy()

    @staticmethod
    def from_dict(d):
        p = Position(d["direction"], d["strategy"], d["entry_price"],
                     d["size"], d["original_size"], d["stop_loss"],
                     d["take_profit"], d["atr"], d["entry_time"],
                     d.get("consensus_details", ""))
        for k, v in d.items():
            if hasattr(p, k):
                setattr(p, k, v)
        return p


# ── Risk Manager ──────────────────────────────────────

class RiskManager:
    def __init__(self):
        self.month_start_capital = INITIAL_CAPITAL
        self.consecutive_losses = 0
        self.cooldown_until = None
        self.weekly_trades = {}
        self._current_month = None

    def calc_position_size(self, capital, entry_price, stop_loss, consensus_strength=1.0):
        sl_pct = abs(entry_price - stop_loss) / entry_price
        if sl_pct < 0.001:
            sl_pct = 0.01
        raw_size = (capital * RISK_PER_TRADE) / sl_pct
        max_size = capital * 0.20
        size = min(raw_size, max_size)
        # Skaliere mit Konsens-Stärke
        size *= min(1.0, consensus_strength / 0.5)
        return round(size, 2) if size >= 5.0 else 0.0

    def can_trade(self, capital):
        reasons = []
        monthly_dd = (self.month_start_capital - capital) / self.month_start_capital
        if monthly_dd >= MAX_MONTHLY_DD:
            reasons.append(f"Monthly DD {monthly_dd:.1%}")
        if self.cooldown_until:
            now = datetime.now(timezone.utc)
            if now < self.cooldown_until:
                remaining = (self.cooldown_until - now).total_seconds() / 3600
                reasons.append(f"Cooldown ({remaining:.1f}h)")
            else:
                self.cooldown_until = None
                self.consecutive_losses = 0
        week_key = datetime.now(timezone.utc).strftime("%Y-W%W")
        if self.weekly_trades.get(week_key, 0) >= MAX_TRADES_PER_WEEK:
            reasons.append(f"Weekly limit")
        return len(reasons) == 0, reasons

    def register_trade(self):
        week_key = datetime.now(timezone.utc).strftime("%Y-W%W")
        self.weekly_trades[week_key] = self.weekly_trades.get(week_key, 0) + 1

    def register_result(self, pnl):
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= CONSEC_LOSS_LIMIT:
                self.cooldown_until = datetime.now(timezone.utc) + timedelta(hours=COOLDOWN_HOURS)
                return True
        else:
            self.consecutive_losses = 0
        return False

    def new_month_check(self, capital):
        month_key = datetime.now(timezone.utc).strftime("%Y-%m")
        if self._current_month != month_key:
            self._current_month = month_key
            self.month_start_capital = capital

    def to_dict(self):
        return {
            "month_start_capital": self.month_start_capital,
            "consecutive_losses": self.consecutive_losses,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "weekly_trades": self.weekly_trades,
            "_current_month": self._current_month,
        }

    def load_from(self, d):
        self.month_start_capital = d.get("month_start_capital", INITIAL_CAPITAL)
        self.consecutive_losses = d.get("consecutive_losses", 0)
        cu = d.get("cooldown_until")
        self.cooldown_until = datetime.fromisoformat(cu) if cu else None
        self.weekly_trades = d.get("weekly_trades", {})
        self._current_month = d.get("_current_month")


# ── Prime Bot ─────────────────────────────────────────

class PrimeBot:
    """Der Execution-Spezialist."""

    def __init__(self):
        self.capital = INITIAL_CAPITAL
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0
        self.position: Optional[Position] = None
        self.risk_mgr = RiskManager()
        self.trades = []
        self.equity = []

    def aggregate_signals(self, signals: list, weights: dict,
                          leader_id: str) -> dict:
        """
        Berechnet gewichteten Konsens aller Bot-Signale.
        """
        long_score = 0.0
        short_score = 0.0
        details = []

        for sig in signals:
            w = weights.get(sig.bot_id, 0.0)
            if w == 0 or sig.direction == "neutral":
                details.append(f"{sig.bot_id}(w:{w:.0%}): {sig.direction}")
                continue

            vote = sig.confidence * w
            if sig.direction == "long":
                long_score += vote
            else:
                short_score += vote

            details.append(f"{sig.bot_id}(w:{w:.0%}): {sig.direction} @{sig.confidence:.0%}")

        total_vote = long_score + short_score
        if total_vote == 0:
            return {
                "direction": "neutral",
                "weighted_confidence": 0.0,
                "agreement_pct": 0.0,
                "leader_agrees": False,
                "leader_id": leader_id,
                "details": details,
            }

        net_score = long_score - short_score
        direction = "long" if net_score > 0 else "short"
        weighted_confidence = abs(net_score)

        agree_weight = long_score if direction == "long" else short_score
        agreement_pct = agree_weight / total_vote

        # Leader-Bestätigung
        leader_signal = next((s for s in signals if s.bot_id == leader_id), None)
        leader_agrees = (leader_signal is not None and
                         leader_signal.direction == direction)

        return {
            "direction": direction,
            "weighted_confidence": round(weighted_confidence, 4),
            "agreement_pct": round(agreement_pct, 4),
            "leader_agrees": leader_agrees,
            "leader_id": leader_id,
            "details": details,
            "long_score": round(long_score, 4),
            "short_score": round(short_score, 4),
        }

    def should_enter(self, consensus: dict) -> bool:
        """Prüft ob Entry-Bedingungen erfüllt sind."""
        if consensus["direction"] == "neutral":
            return False
        if consensus["weighted_confidence"] < MIN_CONSENSUS:
            return False
        if consensus["agreement_pct"] < MIN_AGREEMENT:
            return False
        if LEADER_MUST_AGREE and not consensus["leader_agrees"]:
            return False

        can_trade, reasons = self.risk_mgr.can_trade(self.capital)
        return can_trade

    def open_position(self, consensus: dict, price: float, atr: float):
        """Eröffnet Position basierend auf Konsens."""
        direction = consensus["direction"]
        sl_dist = SL_ATR_MULT * atr
        tp_dist = sl_dist * 1.5  # 1.5:1 R:R

        if direction == "long":
            sl = price - sl_dist
            tp = price + tp_dist
        else:
            sl = price + sl_dist
            tp = price - tp_dist

        size = self.risk_mgr.calc_position_size(
            self.capital, price, sl, consensus["weighted_confidence"])

        if size <= 0:
            return None

        self.position = Position(
            direction=direction,
            strategy=f"jv:{consensus['leader_id']}",
            entry_price=price,
            size=size,
            original_size=size,
            stop_loss=round(sl, 6),
            take_profit=round(tp, 6),
            atr=atr,
            entry_time=datetime.now(timezone.utc).isoformat(),
            consensus_details=str(consensus["details"]),
        )
        self.risk_mgr.register_trade()
        return self.position

    # ── State ─────────────────────────────────────────

    def save_state(self, path: str):
        state = {
            "version": "jv-1.0",
            "capital": round(self.capital, 2),
            "total_pnl": round(self.total_pnl, 4),
            "wins": self.wins,
            "losses": self.losses,
            "position": self.position.to_dict() if self.position else None,
            "risk_manager": self.risk_mgr.to_dict(),
            "last_update": datetime.now(timezone.utc).isoformat(),
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def load_state(self, path: str):
        if not os.path.exists(path):
            return
        with open(path) as f:
            state = json.load(f)
        if state.get("version") != "jv-1.0":
            return
        self.capital = state.get("capital", INITIAL_CAPITAL)
        self.total_pnl = state.get("total_pnl", 0.0)
        self.wins = state.get("wins", 0)
        self.losses = state.get("losses", 0)
        if state.get("position"):
            self.position = Position.from_dict(state["position"])
        if state.get("risk_manager"):
            self.risk_mgr.load_from(state["risk_manager"])
