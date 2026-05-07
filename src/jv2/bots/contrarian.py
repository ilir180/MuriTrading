"""
JV Boting v2 – Contrarian
These: "Die Masse liegt immer falsch bei Extremen."
"""

from src.jv2.base_bot import JV2Bot, _safe
from src.jv2.models import JV2Signal


class Contrarian(JV2Bot):
    def __init__(self, symbol="XRP/USDT"):
        super().__init__("contrarian", symbol)

    def check_thesis(self, market_data):
        """Contrarian hält bis Sentiment sich normalisiert. F&G 35-65 = neutral = raus."""
        if not self.state.position:
            return True, ""
        sent = market_data.get("sentiment", {})
        fng = _safe(sent.get("sent_fear_greed", 50))
        # Long-These: Fear → solange F&G unter 45 bleibt, ist noch Fear
        if self.state.position.direction == "long" and fng > 45:
            return False, f"Sentiment normalisiert (F&G:{fng:.0f})"
        # Short-These: Greed → solange F&G über 55 bleibt, ist noch Greed
        if self.state.position.direction == "short" and fng < 55:
            return False, f"Sentiment normalisiert (F&G:{fng:.0f})"
        return True, ""

    def generate_signal(self, market_data, spy_intel):
        price = market_data["price"]
        sent = market_data.get("sentiment", {})
        r4 = market_data.get("latest_4h")

        fng = _safe(sent.get("sent_fear_greed", 50))
        composite = _safe(sent.get("sent_composite", 0.5))
        cg_bull = _safe(sent.get("sent_cg_bullish_pct", 50))

        direction = "neutral"
        conf = 0.0
        reasons = []

        # Extreme Fear → LONG (contrarian)
        if fng <= 20:
            direction = "long"
            conf = 0.35
            if fng <= 10: conf += 0.15
            if composite < 0.3: conf += 0.10
            reasons.append(f"ExFear:{fng:.0f}")
        elif fng <= 35:
            direction = "long"
            conf = 0.22
            reasons.append(f"Fear:{fng:.0f}")

        # Extreme Greed → SHORT (contrarian)
        elif fng >= 80:
            direction = "short"
            conf = 0.35
            if fng >= 90: conf += 0.15
            if composite > 0.7: conf += 0.10
            reasons.append(f"ExGreed:{fng:.0f}")
        elif fng >= 65:
            direction = "short"
            conf = 0.22
            reasons.append(f"Greed:{fng:.0f}")

        if direction == "neutral":
            return self.neutral(price, f"Sentiment neutral (F&G:{fng:.0f})")

        # Preis schon bewegt? (late entry check)
        if r4 is not None:
            ret_3 = _safe(r4.get("4h_return_3", 0))
            if direction == "long" and ret_3 > 0.03:
                conf *= 0.6
                reasons.append("Late-entry")
            elif direction == "short" and ret_3 < -0.03:
                conf *= 0.6
                reasons.append("Late-entry")

        # Spy: Flow Tracker bestätigt
        if spy_intel.get("whale_direction") == direction:
            conf += 0.05
            reasons.append("Whale-confirms")

        conf = min(conf, 1.0)

        return JV2Signal(
            bot_id=self.bot_id,
            timestamp=JV2Signal.neutral("", 0).timestamp,
            direction=direction,
            confidence=round(conf, 3),
            reasoning=f"CONTRARIAN {direction.upper()}: {', '.join(reasons)}",
            price_at_signal=price,
            features={"fear_greed": fng, "composite": composite, "cg_bullish": cg_bull},
        )
