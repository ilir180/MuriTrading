"""
JV Boting v2 – Datentypen
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional

from src.jv2.config import *


@dataclass
class JV2Signal:
    bot_id: str
    timestamp: str
    direction: str          # "long" | "short" | "neutral"
    confidence: float
    reasoning: str
    features: dict = field(default_factory=dict)
    price_at_signal: float = 0.0

    @staticmethod
    def neutral(bot_id, price, reason=""):
        return JV2Signal(
            bot_id=bot_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            direction="neutral",
            confidence=0.0,
            reasoning=reason,
            price_at_signal=price,
        )

    def to_dict(self):
        return {
            "bot_id": self.bot_id,
            "direction": self.direction,
            "confidence": round(self.confidence, 3),
            "reasoning": self.reasoning,
            "price": round(self.price_at_signal, 6),
        }


@dataclass
class BotPosition:
    bot_id: str
    direction: str
    entry_price: float
    size_usd: float
    stop_loss: float
    take_profit: float
    atr: float
    entry_time: str
    candles_held: int = 0
    peak_price: float = 0.0
    trough_price: float = 0.0
    trailing_active: bool = False
    trailing_stop: float = 0.0

    def __post_init__(self):
        if self.peak_price == 0.0:
            self.peak_price = self.entry_price
        if self.trough_price == 0.0:
            self.trough_price = self.entry_price

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

    def check_exit(self, current_price):
        if self.direction == "long":
            if current_price <= self.stop_loss:
                return True, "STOP-LOSS"
            if current_price >= self.take_profit:
                return True, "TAKE-PROFIT"
        else:
            if current_price >= self.stop_loss:
                return True, "STOP-LOSS"
            if current_price <= self.take_profit:
                return True, "TAKE-PROFIT"
        if self.trailing_active:
            if self.direction == "long" and current_price <= self.trailing_stop:
                return True, "TRAILING-STOP"
            if self.direction == "short" and current_price >= self.trailing_stop:
                return True, "TRAILING-STOP"
        max_hold = getattr(self, '_max_hold', MAX_HOLD_CANDLES)
        if self.candles_held >= max_hold:
            return True, "TIME-EXIT"
        return False, ""

    def calc_pnl(self, exit_price):
        if self.direction == "long":
            raw_ret = (exit_price - self.entry_price) / self.entry_price
        else:
            raw_ret = (self.entry_price - exit_price) / self.entry_price
        net_ret = raw_ret - ROUND_TRIP
        pnl = self.size_usd * net_ret
        return pnl, raw_ret, net_ret

    def unrealized_pnl(self, current_price):
        pnl, _, _ = self.calc_pnl(current_price)
        return pnl

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}

    @staticmethod
    def from_dict(d):
        extra = {}
        if '_max_hold' in d:
            extra['_max_hold'] = d.pop('_max_hold')
        pos = BotPosition(**d)
        for k, v in extra.items():
            setattr(pos, k, v)
        return pos


@dataclass
class TradeRecord:
    timestamp: str
    bot_id: str
    direction: str
    entry_price: float
    exit_price: float
    size_usd: float
    pnl: float
    net_return_pct: float
    reason: str
    hold_candles: int
    bot_capital_after: float

    def to_csv_row(self):
        return (f"{self.timestamp},{self.bot_id},{self.direction},"
                f"{self.entry_price:.6f},{self.exit_price:.6f},"
                f"{self.size_usd:.2f},{self.pnl:.4f},{self.net_return_pct:.4f},"
                f"{self.reason},{self.hold_candles},{self.bot_capital_after:.2f}")

    @staticmethod
    def csv_header():
        return "timestamp,bot_id,direction,entry_price,exit_price,size_usd,pnl,net_return_pct,reason,hold_candles,bot_capital_after"


@dataclass
class BotState:
    bot_id: str
    capital: float = INITIAL_ALLOC
    total_pnl: float = 0.0
    wins: int = 0
    losses: int = 0
    position: Optional[BotPosition] = None
    last_signal: Optional[JV2Signal] = None
    consecutive_losses: int = 0
    cooldown_until: Optional[str] = None
    trades_this_week: int = 0

    def to_dict(self):
        return {
            "bot_id": self.bot_id,
            "capital": round(self.capital, 2),
            "total_pnl": round(self.total_pnl, 4),
            "wins": self.wins,
            "losses": self.losses,
            "consecutive_losses": self.consecutive_losses,
            "cooldown_until": self.cooldown_until,
            "trades_this_week": self.trades_this_week,
            "position": self.position.to_dict() if self.position else None,
            "last_signal": self.last_signal.to_dict() if self.last_signal else None,
        }

    @staticmethod
    def from_dict(d):
        s = BotState(bot_id=d["bot_id"])
        s.capital = d.get("capital", INITIAL_ALLOC)
        s.total_pnl = d.get("total_pnl", 0.0)
        s.wins = d.get("wins", 0)
        s.losses = d.get("losses", 0)
        s.consecutive_losses = d.get("consecutive_losses", 0)
        s.cooldown_until = d.get("cooldown_until")
        s.trades_this_week = d.get("trades_this_week", 0)
        if d.get("position"):
            s.position = BotPosition.from_dict(d["position"])
        if d.get("last_signal"):
            ls = d["last_signal"]
            s.last_signal = JV2Signal(
                bot_id=ls.get("bot_id", d["bot_id"]),
                timestamp="",
                direction=ls.get("direction", "neutral"),
                confidence=ls.get("confidence", 0),
                reasoning=ls.get("reasoning", ""),
                price_at_signal=ls.get("price", 0),
            )
        return s
