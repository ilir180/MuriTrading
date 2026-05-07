"""
Joint Venture Boting – Momentum Bot
Spezialisiert auf Preis-Dynamik: Velocity, Acceleration, Multi-TF Momentum.

Sieht den Markt als Physik: Geschwindigkeit und Beschleunigung der Preisbewegung.
Wenn Velocity positiv UND Acceleration zunimmt → starkes Momentum.
Wenn Velocity und Acceleration divergieren → Momentum-Shift.
"""

import math
import pandas as pd
from src.jv.base_bot import JVBot
from src.jv.signal_protocol import JVSignal


def _safe(val, default=0.0):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return float(val)


class MomentumBot(JVBot):
    """
    Analysiert Preis-Dynamik über multiple Timeframes.

    Beobachtet:
      - Velocity: Rate of Change (wie schnell bewegt sich der Preis)
      - Acceleration: Änderung der Velocity (beschleunigt oder bremst?)
      - Multi-TF Alignment: stimmen 1H, 4H, 1D überein?
      - EMA Momentum: Abstand und Richtung zu EMAs
      - MACD Histogram Slope: steigt oder fällt das Momentum?
    """

    def __init__(self):
        super().__init__("momentum")

    def observe(self, market_data: dict) -> JVSignal:
        price = market_data["price"]
        row_1h = market_data.get("latest_1h")
        row_4h = market_data.get("latest_4h")
        row_1d = market_data.get("latest_1d")

        if row_4h is None:
            return self.neutral(price, "Keine 4H-Daten")

        # ── VELOCITY (Rate of Change) ─────────────────
        # Wie schnell bewegt sich der Preis?
        ret_1h_1 = _safe(row_1h.get("1h_return_1") if row_1h is not None else 0)
        ret_4h_1 = _safe(row_4h.get("4h_return_1"))
        ret_4h_3 = _safe(row_4h.get("4h_return_3"))
        ret_1d_1 = _safe(row_1d.get("1d_return_1") if row_1d is not None else 0)

        velocity_1h = ret_1h_1
        velocity_4h = ret_4h_1
        velocity_1d = ret_1d_1

        # ── ACCELERATION ──────────────────────────────
        # Wird die Bewegung schneller oder langsamer?
        ret_4h_prev = _safe(row_4h.get("4h_return_3")) / 3 if ret_4h_3 != 0 else 0
        acceleration_4h = velocity_4h - ret_4h_prev

        # ── MULTI-TF ALIGNMENT ────────────────────────
        # Stimmen alle Timeframes überein?
        bullish_count = 0
        bearish_count = 0

        # 1H EMA Alignment
        ema_1h = _safe(row_1h.get("1h_ema_9_above_21") if row_1h is not None else 0.5)
        if ema_1h > 0.5: bullish_count += 1
        else: bearish_count += 1

        # 4H EMA Alignment
        ema_4h_9_21 = _safe(row_4h.get("4h_ema_9_above_21", 0.5))
        ema_4h_21_50 = _safe(row_4h.get("4h_ema_21_above_50", 0.5))
        if ema_4h_9_21 > 0.5: bullish_count += 1
        else: bearish_count += 1
        if ema_4h_21_50 > 0.5: bullish_count += 1
        else: bearish_count += 1

        # 1D EMA Alignment
        ema_1d = _safe(row_1d.get("1d_ema_9_above_21") if row_1d is not None else 0.5)
        if ema_1d > 0.5: bullish_count += 1
        else: bearish_count += 1

        # ── MACD MOMENTUM ────────────────────────────
        macd_above_4h = _safe(row_4h.get("4h_macd_above", 0.5))
        macd_above_1h = _safe(row_1h.get("1h_macd_above") if row_1h is not None else 0.5)

        if macd_above_4h > 0.5: bullish_count += 1
        else: bearish_count += 1
        if macd_above_1h > 0.5: bullish_count += 1
        else: bearish_count += 1

        # ── STOCHASTIC RSI ────────────────────────────
        stoch_4h = _safe(row_4h.get("4h_stoch_rsi", 0.5))

        # ── ENTSCHEIDUNG ─────────────────────────────
        total_signals = bullish_count + bearish_count
        bull_pct = bullish_count / total_signals if total_signals > 0 else 0.5

        # Velocity-Richtung
        vel_direction = "long" if velocity_4h > 0 else "short" if velocity_4h < 0 else "neutral"

        # Confidence: Wie einig sind die Signale + wie stark ist die Velocity
        alignment = max(bull_pct, 1 - bull_pct)  # 0.5-1.0
        vel_strength = min(abs(velocity_4h) / 0.02, 1.0)  # Normalisiert auf ~2%
        accel_confirms = (acceleration_4h > 0 and velocity_4h > 0) or \
                         (acceleration_4h < 0 and velocity_4h < 0)

        confidence = 0.0
        direction = "neutral"
        reasons = []

        if bull_pct >= 0.65 and velocity_4h > 0.001:
            direction = "long"
            confidence = alignment * 0.5 + vel_strength * 0.3
            if accel_confirms:
                confidence += 0.15
                reasons.append("Accel+")
            if stoch_4h < 0.75:
                confidence += 0.05
            reasons.append(f"TF-Align:{bullish_count}/{total_signals}")
            reasons.append(f"Vel:{velocity_4h:+.3f}")

        elif bull_pct <= 0.35 and velocity_4h < -0.001:
            direction = "short"
            confidence = alignment * 0.5 + vel_strength * 0.3
            if accel_confirms:
                confidence += 0.15
                reasons.append("Accel-")
            if stoch_4h > 0.25:
                confidence += 0.05
            reasons.append(f"TF-Align:{bearish_count}/{total_signals}")
            reasons.append(f"Vel:{velocity_4h:+.3f}")

        confidence = min(confidence, 1.0)

        if direction == "neutral" or confidence < 0.15:
            return self.neutral(price, f"Mixed signals (Bull:{bull_pct:.0%} Vel:{velocity_4h:+.3f})")

        return JVSignal.create(
            bot_id=self.bot_id,
            direction=direction,
            confidence=round(confidence, 3),
            reasoning=f"Momentum {direction.upper()}: {', '.join(reasons)}",
            price=price,
            features={
                "velocity_1h": round(velocity_1h, 5),
                "velocity_4h": round(velocity_4h, 5),
                "velocity_1d": round(velocity_1d, 5),
                "acceleration_4h": round(acceleration_4h, 5),
                "bull_alignment": round(bull_pct, 2),
                "stoch_rsi_4h": round(stoch_4h, 3),
            },
            ttl_candles=2,
        )
