"""
JV Boting v2 – Abstract Base Bot
Jeder Bot tradet selbst. Kein Konsens, kein Voting.
"""

import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

from src.jv2.config import *
from src.jv2.models import JV2Signal, BotPosition, BotState, TradeRecord


def _safe(val, default=0.0):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return float(val)


# Lazy-loaded singleton — clusterer model lives on disk.
_REGIME_CLUSTERER = None


def _get_clusterer():
    global _REGIME_CLUSTERER
    if _REGIME_CLUSTERER is None:
        from src.jv2.regime_clusterer import RegimeClusterer
        _REGIME_CLUSTERER = RegimeClusterer.load()
    return _REGIME_CLUSTERER


class JV2Bot(ABC):
    """Abstrakter JV2 Bot — jeder Spezialist erbt hiervon."""

    def __init__(self, bot_id: str, symbol: str = "XRP/USDT"):
        self.base_id = bot_id                     # z.B. "trend_rider"
        self.symbol = symbol
        sym_short = SYMBOLS.get(symbol, {}).get("short", symbol.split("/")[0])
        self.bot_id = f"{bot_id}_{sym_short}"     # z.B. "trend_rider_XRP"
        self.state = BotState(bot_id=self.bot_id)

        # Risikoprofil laden
        from src.jv2.config import BOT_RISK_PROFILES, BOT_OVERRIDES
        self.risk_profile = dict(BOT_RISK_PROFILES.get(bot_id, {
            "risk": RISK_PER_TRADE, "leverage": 1,
            "sl_atr": SL_ATR_MULT, "tp_atr": TP_ATR_MULT, "max_hold": MAX_HOLD_CANDLES,
        }))

        # Overrides pro Bot×Asset
        override = BOT_OVERRIDES.get(self.bot_id, {})
        self.invert_signal = override.get("invert", False)
        if override.get("exec_override"):
            self.risk_profile.update(override["exec_override"])

    # ── ABSTRAKT ──────────────────────────────────────

    @abstractmethod
    def generate_signal(self, market_data: dict, spy_intel: dict) -> JV2Signal:
        """Analysiere Markt und gib Signal zurück."""
        ...

    def check_thesis(self, market_data: dict) -> Tuple[bool, str]:
        """Prüft ob die These noch gilt. Override in Subklassen.
        Returns: (still_valid, reason)
        Default: These gilt immer (Fallback auf Stop/TP).
        """
        return True, ""

    # ── HELPERS ───────────────────────────────────────

    def neutral(self, price, reason=""):
        return JV2Signal.neutral(self.bot_id, price, reason)

    # ── TICK (alle 60s) ──────────────────────────────

    def tick(self, current_price: float) -> Optional[TradeRecord]:
        """Prüft Exits und Trailing. Gibt TradeRecord zurück wenn geschlossen."""
        if self.state.position is None:
            return None
        pos = self.state.position
        pos.update_trailing(current_price)
        should_exit, reason = pos.check_exit(current_price)
        if should_exit:
            return self._close_position(current_price, reason)
        return None

    # ── ON NEW CANDLE (alle 4H) ──────────────────────

    def on_new_candle(self, market_data: dict, spy_intel: dict) -> Tuple[JV2Signal, Optional[str]]:
        """Signal generieren, ggf. Position eröffnen. Gibt (Signal, entry_info, thesis_exit) zurück."""
        thesis_exit = None

        if self.state.position:
            self.state.position.candles_held += 1

            # These-Check: Stimmt der Grund für den Trade noch?
            still_valid, invalidation_reason = self.check_thesis(market_data)
            if not still_valid:
                thesis_exit = self._close_position(
                    market_data["price"], f"THESIS-EXIT: {invalidation_reason}")

        signal = self.generate_signal(market_data, spy_intel)

        # Invertierung: Signal umdrehen wenn Bot konsistent falsch liegt
        if self.invert_signal and signal.direction != "neutral":
            flipped = "short" if signal.direction == "long" else "long"
            signal = JV2Signal(
                bot_id=signal.bot_id,
                timestamp=signal.timestamp,
                direction=flipped,
                confidence=signal.confidence,
                reasoning=f"INV: {signal.reasoning}",
                features=signal.features,
                price_at_signal=signal.price_at_signal,
            )

        self.state.last_signal = signal

        entry_info = None
        if signal.direction != "neutral" and signal.confidence >= MIN_CONFIDENCE:
            if self.state.position is None and self._can_trade():
                regime = self._snapshot_regime(market_data)
                entry_info = self._open_position(
                    signal, market_data["price"], market_data["atr_4h"], regime)

        return signal, entry_info, thesis_exit

    # ── REGIME SNAPSHOT ──────────────────────────────
    # Captured at entry time, persisted on the position, copied to TradeRecord at close.
    # Drives per-regime stats for the Coach / promotion logic.
    def _snapshot_regime(self, market_data: dict) -> dict:
        r4 = market_data.get("latest_4h")
        sent = market_data.get("sentiment", {}) or {}
        price = market_data.get("price", 0.0)
        atr = market_data.get("atr_4h", 0.0)
        if r4 is None:
            return {}
        regime = {
            "adx": _safe(r4.get("4h_adx")),
            "rsi": _safe(r4.get("4h_rsi_14"), 50.0),
            "bb_pos": _safe(r4.get("4h_bb_pos"), 0.5),
            "bbw": _safe(r4.get("4h_bb_width")),
            "atr_pct": (atr / price * 100) if price > 0 else 0.0,
            "chop": _safe(r4.get("4h_chop")),
            "trend_consistency": _safe(r4.get("4h_trend_consistency")),
            "fear_greed": _safe(sent.get("fear_greed"), 50.0),
        }
        regime["cluster"] = _get_clusterer().assign(regime)
        return regime

    # ── RISK CHECK ───────────────────────────────────

    def _can_trade(self) -> bool:
        if self.state.capital < MIN_ALLOC:
            return False
        if self.state.cooldown_until:
            now = datetime.now(timezone.utc)
            try:
                cd = datetime.fromisoformat(self.state.cooldown_until)
                if now < cd:
                    return False
            except (ValueError, TypeError):
                pass
            self.state.cooldown_until = None
            self.state.consecutive_losses = 0
        return True

    # ── OPEN POSITION ────────────────────────────────

    def _open_position(self, signal, price, atr, regime: dict = None):
        rp = self.risk_profile
        # Katastrophen-Stop: weit weg (4 ATR), primärer Exit ist check_thesis()
        catastrophe_sl_atr = max(rp["sl_atr"], 4.0)
        sl_dist = catastrophe_sl_atr * atr
        tp_dist = rp["tp_atr"] * atr

        if signal.direction == "long":
            sl, tp = price - sl_dist, price + tp_dist
        else:
            sl, tp = price + sl_dist, price - tp_dist

        sl_pct = abs(price - sl) / price
        if sl_pct < 0.001:
            sl_pct = 0.01
        leverage = rp.get("leverage", 1)
        risk_amount = self.state.capital * rp["risk"]
        size_usd = risk_amount / sl_pct
        max_size = self.state.capital * leverage
        size_usd = min(size_usd, max_size)

        if size_usd < 5.0:
            return None

        self.state.position = BotPosition(
            bot_id=self.bot_id,
            direction=signal.direction,
            entry_price=price,
            size_usd=round(size_usd, 2),
            stop_loss=round(sl, 6),
            take_profit=round(tp, 6),
            atr=atr,
            entry_time=datetime.now(timezone.utc).isoformat(),
            regime=regime or {},
        )
        self.state.position._max_hold = rp["max_hold"]
        self.state.trades_this_week += 1
        return "opened"

    # ── CLOSE POSITION ───────────────────────────────

    def _close_position(self, exit_price, reason) -> TradeRecord:
        pos = self.state.position
        pnl, raw_ret, net_ret = pos.calc_pnl(exit_price)

        self.state.capital += pnl
        self.state.total_pnl += pnl

        if pnl > 0:
            self.state.wins += 1
            self.state.consecutive_losses = 0
        else:
            self.state.losses += 1
            self.state.consecutive_losses += 1
            if self.state.consecutive_losses >= CONSEC_LOSS_LIMIT:
                self.state.cooldown_until = (
                    datetime.now(timezone.utc) + timedelta(hours=COOLDOWN_HOURS)
                ).isoformat()

        rg = pos.regime or {}
        record = TradeRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            bot_id=self.bot_id,
            direction=pos.direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            size_usd=pos.size_usd,
            pnl=round(pnl, 4),
            net_return_pct=round(net_ret * 100, 4),
            reason=reason,
            hold_candles=pos.candles_held,
            bot_capital_after=round(self.state.capital, 2),
            regime_adx=float(rg.get("adx", 0.0)),
            regime_rsi=float(rg.get("rsi", 0.0)),
            regime_bb_pos=float(rg.get("bb_pos", 0.0)),
            regime_bbw=float(rg.get("bbw", 0.0)),
            regime_atr_pct=float(rg.get("atr_pct", 0.0)),
            regime_chop=float(rg.get("chop", 0.0)),
            regime_trend_consistency=float(rg.get("trend_consistency", 0.0)),
            regime_fear_greed=float(rg.get("fear_greed", 0.0)),
            regime_cluster=int(rg.get("cluster", -1)),
        )

        self.state.position = None
        return record
