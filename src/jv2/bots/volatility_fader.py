"""
JV Boting v2 – Volatility Fader
These: "Volatilität normalisiert sich immer. Fade den Spike."
"""

from src.jv2.base_bot import JV2Bot, _safe
from src.jv2.models import JV2Signal


class VolatilityFader(JV2Bot):
    def __init__(self, symbol="XRP/USDT"):
        super().__init__("volatility_fader", symbol)

    def check_thesis(self, market_data):
        """Vola-Fade gilt solange Volatilität noch erhöht ist. Wenn Vola normal → raus (These erfüllt)."""
        if not self.state.position:
            return True, ""
        r4 = market_data.get("latest_4h")
        if r4 is None:
            return True, ""
        vol_z = _safe(r4.get("4h_vol_regime", 0))
        adx = _safe(r4.get("4h_adx", 20))
        trend_cons = _safe(r4.get("4h_trend_consistency", 0.5))
        # Vola normalisiert = These erfüllt, Gewinn mitnehmen
        if vol_z < 0.5:
            return False, f"Vola normalisiert (z:{vol_z:.1f}) — These erfüllt"
        # Neuer Trend entsteht statt Vola-Normalisierung = These tot
        if adx > 30 and trend_cons > 0.6:
            return False, f"Trend statt Fade (ADX:{adx:.0f})"
        return True, ""

    def generate_signal(self, market_data, spy_intel):
        price = market_data["price"]
        r4 = market_data.get("latest_4h")
        if r4 is None:
            return self.neutral(price, "Keine 4H-Daten")

        vol_z = _safe(r4.get("4h_vol_regime", 0))
        bb_width = _safe(r4.get("4h_bb_width", 0.05))
        ret_3 = _safe(r4.get("4h_return_3", 0))
        adx = _safe(r4.get("4h_adx", 20))
        trend_cons = _safe(r4.get("4h_trend_consistency", 0.5))

        # Gate: Vola-Spike erkennen
        if vol_z < 1.2:
            return self.neutral(price, f"Normale Vola (z:{vol_z:.1f})")

        # Richtung: Fade den letzten Move
        if ret_3 > 0.015:
            direction = "short"
        elif ret_3 < -0.015:
            direction = "long"
        else:
            return self.neutral(price, f"Vola-Spike ohne Richtung (ret3:{ret_3:+.3f})")

        conf = 0.25
        if vol_z > 1.8: conf += 0.10
        if vol_z > 2.2: conf += 0.10
        if bb_width > 0.05: conf += 0.10
        if abs(ret_3) > 0.03: conf += 0.10

        # Sicherheit: Nicht faden wenn neuer Trend entsteht
        if adx > 30 and trend_cons > 0.6:
            conf *= 0.5

        conf = min(conf, 1.0)

        return JV2Signal(
            bot_id=self.bot_id,
            timestamp=JV2Signal.neutral("", 0).timestamp,
            direction=direction,
            confidence=round(conf, 3),
            reasoning=f"VOLFADE {direction.upper()}: z:{vol_z:.1f} ret3:{ret_3:+.3f} BBW:{bb_width:.3f}",
            price_at_signal=price,
            features={"vol_z": vol_z, "bb_width": bb_width, "ret_3": ret_3, "adx": adx},
        )
