"""
Joint Venture Boting – Regime Bot
Spezialisiert auf Phasen-Erkennung: Trending, Ranging, Breakout-Imminent.

Sieht den Markt als Zustandsmaschine:
  TRENDING  → Trade die Richtung des Trends
  RANGING   → Neutral (oder Mean-Reversion bei Extremen)
  BREAKOUT  → Trade die wahrscheinliche Ausbruchsrichtung
"""

import math
from src.jv.base_bot import JVBot
from src.jv.signal_protocol import JVSignal


def _safe(val, default=0.0):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return float(val)


class RegimeBot(JVBot):
    def __init__(self):
        super().__init__("regime")

    def observe(self, market_data: dict) -> JVSignal:
        price = market_data["price"]
        row_4h = market_data.get("latest_4h")
        row_1d = market_data.get("latest_1d")

        if row_4h is None:
            return self.neutral(price, "Keine 4H-Daten")

        # ── REGIME INDICATORS ─────────────────────────
        adx = _safe(row_4h.get("4h_adx", 20))
        chop = _safe(row_4h.get("4h_chop", 0.5))
        bb_squeeze = _safe(row_4h.get("4h_bb_squeeze", 0))
        bb_width = _safe(row_4h.get("4h_bb_width", 0.05))
        trend_cons = _safe(row_4h.get("4h_trend_consistency", 0.5))
        regime_trend = _safe(row_4h.get("4h_regime_trend", 0))
        vol_regime = _safe(row_4h.get("4h_vol_regime", 0))

        # ── PHASE ERKENNEN ────────────────────────────
        # TRENDING: ADX > 28, Chop < 0.45, hohe Trend-Consistency
        is_trending = adx > 28 and chop < 0.45 and trend_cons > 0.55

        # BREAKOUT IMMINENT: BB Squeeze + ADX noch niedrig aber steigend
        is_breakout = (bb_squeeze > 0.5 or bb_width < 0.020) and adx < 25

        # RANGING: ADX < 20, hohe Choppiness
        is_ranging = adx < 20 and chop > 0.55

        # ── RICHTUNG BESTIMMEN ────────────────────────
        if is_trending:
            # Im Trend: Richtung aus EMA-Alignment
            ema9_21 = _safe(row_4h.get("4h_ema_9_above_21", 0.5))
            ema21_50 = _safe(row_4h.get("4h_ema_21_above_50", 0.5))
            rsi = _safe(row_4h.get("4h_rsi_14", 50))
            daily_ema = _safe(row_1d.get("1d_ema_9_above_21", 0.5) if row_1d is not None else 0.5)

            bullish = (ema9_21 > 0.5) + (ema21_50 > 0.5) + (daily_ema > 0.5)
            bearish = (ema9_21 < 0.5) + (ema21_50 < 0.5) + (daily_ema < 0.5)

            # Confidence basierend auf Trendstärke
            trend_strength = min((adx - 25) / 25, 1.0)  # 0-1, stärker bei hohem ADX
            consistency_bonus = max(0, (trend_cons - 0.5) * 2)  # 0-1

            if bullish >= 2:
                direction = "long"
                confidence = 0.3 + trend_strength * 0.4 + consistency_bonus * 0.2
                # Nicht überkauft?
                if 35 < rsi < 70:
                    confidence += 0.1
                reasoning = f"TREND-UP Phase (ADX:{adx:.0f} Chop:{chop:.2f} TC:{trend_cons:.2f})"

            elif bearish >= 2:
                direction = "short"
                confidence = 0.3 + trend_strength * 0.4 + consistency_bonus * 0.2
                if 30 < rsi < 65:
                    confidence += 0.1
                reasoning = f"TREND-DOWN Phase (ADX:{adx:.0f} Chop:{chop:.2f} TC:{trend_cons:.2f})"
            else:
                return self.neutral(price, f"Trend unklar (ADX:{adx:.0f})")

            return JVSignal.create(
                bot_id=self.bot_id,
                direction=direction,
                confidence=round(min(confidence, 1.0), 3),
                reasoning=reasoning,
                price=price,
                features={"adx": adx, "chop": chop, "trend_cons": trend_cons,
                           "phase": "trending", "bb_width": bb_width},
                ttl_candles=3,
            )

        elif is_breakout:
            # Breakout imminent: Richtung aus höherem TF
            daily_ema = _safe(row_1d.get("1d_ema_9_above_21", 0.5) if row_1d is not None else 0.5)
            daily_ema21_50 = _safe(row_1d.get("1d_ema_21_above_50", 0.5) if row_1d is not None else 0.5)
            vol_ratio = _safe(row_4h.get("4h_vol_ratio", 1.0))

            if daily_ema > 0.5 and daily_ema21_50 > 0.5:
                direction = "long"
            elif daily_ema < 0.5 and daily_ema21_50 < 0.5:
                direction = "short"
            else:
                return self.neutral(price, f"Breakout imminent, Richtung unklar (BBW:{bb_width:.3f})")

            # Confidence: niedrig weil Breakout noch nicht passiert
            confidence = 0.25
            if vol_ratio > 1.5:
                confidence += 0.15
            if bb_width < 0.015:
                confidence += 0.10

            return JVSignal.create(
                bot_id=self.bot_id,
                direction=direction,
                confidence=round(min(confidence, 1.0), 3),
                reasoning=f"BREAKOUT-IMMINENT ({direction.upper()}, BBW:{bb_width:.3f}, Vol:{vol_ratio:.1f}x)",
                price=price,
                features={"adx": adx, "bb_width": bb_width, "bb_squeeze": bb_squeeze,
                           "vol_ratio": vol_ratio, "phase": "breakout"},
                ttl_candles=4,
            )

        elif is_ranging:
            # Ranging: Neutral, es sei denn RSI ist extrem
            rsi = _safe(row_4h.get("4h_rsi_14", 50))
            bb_pos = _safe(row_4h.get("4h_bb_pos", 0.5))

            if rsi < 28 and bb_pos < 0.10:
                return JVSignal.create(
                    bot_id=self.bot_id,
                    direction="long",
                    confidence=0.35,
                    reasoning=f"RANGE-BODEN (RSI:{rsi:.0f} BB:{bb_pos:.2f} ADX:{adx:.0f})",
                    price=price,
                    features={"adx": adx, "rsi": rsi, "bb_pos": bb_pos, "phase": "ranging"},
                    ttl_candles=2,
                )
            elif rsi > 72 and bb_pos > 0.90:
                return JVSignal.create(
                    bot_id=self.bot_id,
                    direction="short",
                    confidence=0.35,
                    reasoning=f"RANGE-DECKE (RSI:{rsi:.0f} BB:{bb_pos:.2f} ADX:{adx:.0f})",
                    price=price,
                    features={"adx": adx, "rsi": rsi, "bb_pos": bb_pos, "phase": "ranging"},
                    ttl_candles=2,
                )

            return self.neutral(price, f"RANGING Phase (ADX:{adx:.0f} Chop:{chop:.2f}) – kein Extrem")

        # Übergang / Unklar
        return self.neutral(price, f"Regime unklar (ADX:{adx:.0f} Chop:{chop:.2f} BBW:{bb_width:.3f})")
