"""
JV Boting v2 – Breakout Hunter
These: "Kompression führt zur Explosion. Fang den Ausbruch."
"""

from src.jv2.base_bot import JV2Bot, _safe
from src.jv2.models import JV2Signal


class BreakoutHunter(JV2Bot):
    def __init__(self, symbol="XRP/USDT"):
        super().__init__("breakout_hunter", symbol)

    def check_thesis(self, market_data):
        """Breakout gilt solange Volume da ist und Preis nicht zurück in die Range fällt."""
        if not self.state.position:
            return True, ""
        r4 = market_data.get("latest_4h")
        if r4 is None:
            return True, ""
        vol_ratio = _safe(r4.get("4h_vol_ratio", 1.0))
        bb_pos = _safe(r4.get("4h_bb_pos", 0.5))
        # Breakout gescheitert: Preis zurück in die Mitte der BB
        if self.state.position.direction == "long" and bb_pos < 0.4:
            return False, f"Zurück in Range (BB:{bb_pos:.2f})"
        if self.state.position.direction == "short" and bb_pos > 0.6:
            return False, f"Zurück in Range (BB:{bb_pos:.2f})"
        # Volume versiegt = Breakout hat keinen Follow-Through
        if vol_ratio < 0.5 and self.state.position.candles_held >= 2:
            return False, f"Volume versiegt ({vol_ratio:.1f}x)"
        return True, ""

    def generate_signal(self, market_data, spy_intel):
        price = market_data["price"]
        r4 = market_data.get("latest_4h")
        r1d = market_data.get("latest_1d")
        if r4 is None:
            return self.neutral(price, "Keine 4H-Daten")

        bb_squeeze = _safe(r4.get("4h_bb_squeeze", 1.0))
        bb_width = _safe(r4.get("4h_bb_width", 0.05))
        bb_upper = _safe(r4.get("4h_bb_upper", price * 1.02))
        bb_lower = _safe(r4.get("4h_bb_lower", price * 0.98))
        vol_ratio = _safe(r4.get("4h_vol_ratio", 1.0))
        adx = _safe(r4.get("4h_adx", 20))
        close = _safe(r4.get("close", price))

        # Kompression erkennen
        is_compressed = bb_width < 0.025 or bb_squeeze < 0.7

        if not is_compressed:
            return self.neutral(price, f"Keine Kompression (BBW:{bb_width:.3f})")

        # OI Quadrant gives directional bias during compression.
        # OI rising in a squeeze = positioning building → directional breakout likely.
        futures = market_data.get("futures", {})
        oi_quad = int(_safe(futures.get("oi_quadrant", 0)))
        oi_score = _safe(futures.get("oi_quadrant_score", 0))
        # Quadrant 1 (OI↑ Price↑) -> bullish bias; Quadrant 2 (OI↑ Price↓) -> bearish bias
        oi_directional = oi_quad in (1, 2)

        # Expansion erkennen (Breakout passiert gerade): vol_ratio > 1.3 OR
        # OI is showing build-up at compression (positioning Pre-Breakout).
        is_expanding = vol_ratio > 1.3
        oi_buildup_signal = oi_directional and vol_ratio > 1.0

        if not is_expanding and not oi_buildup_signal:
            return self.neutral(
                price, f"Squeeze aber kein Volume/OI-Signal (Vol:{vol_ratio:.1f}x OI:Q{oi_quad})"
            )

        # Richtung — Priorität: BB-Break > OI-Quadrant > Daily-EMA
        if close > bb_upper:
            direction = "long"
            dir_reason = "BB-up"
        elif close < bb_lower:
            direction = "short"
            dir_reason = "BB-dn"
        elif oi_quad == 1:
            direction = "long"
            dir_reason = "OI-Q1"
        elif oi_quad == 2:
            direction = "short"
            dir_reason = "OI-Q2"
        else:
            if r1d is not None and _safe(r1d.get("1d_ema_9_above_21", 0.5)) > 0.5:
                direction = "long"
                dir_reason = "1D-EMA"
            elif r1d is not None:
                direction = "short"
                dir_reason = "1D-EMA"
            else:
                return self.neutral(price, "Breakout-Richtung unklar")

        # Confidence
        conf = 0.30
        if vol_ratio > 2.0: conf += 0.15
        if bb_width < 0.015: conf += 0.10
        if vol_ratio > 2.5: conf += 0.10
        if adx < 20: conf += 0.05
        # OI-Quadrant aligned with direction = directional conviction boost
        if direction == "long" and oi_quad == 1:
            conf += 0.10
        elif direction == "short" and oi_quad == 2:
            conf += 0.10
        # OI-Quadrant against direction = caution (potential fade)
        elif direction == "long" and oi_quad == 2:
            conf *= 0.7
        elif direction == "short" and oi_quad == 1:
            conf *= 0.7

        conf = min(conf, 1.0)

        return JV2Signal(
            bot_id=self.bot_id,
            timestamp=JV2Signal.neutral("", 0).timestamp,
            direction=direction,
            confidence=round(conf, 3),
            reasoning=f"BREAKOUT {direction.upper()}({dir_reason}): BBW:{bb_width:.3f} Vol:{vol_ratio:.1f}x OI:Q{oi_quad} ADX:{adx:.0f}",
            price_at_signal=price,
            features={"bb_width": bb_width, "bb_squeeze": bb_squeeze,
                      "vol_ratio": vol_ratio, "adx": adx,
                      "oi_quadrant": oi_quad, "oi_quadrant_score": oi_score},
        )
