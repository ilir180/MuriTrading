"""
JV Boting v2 – Mean Reverter
These: "Alles kehrt zur Mitte zurück. Kaufe Angst, verkaufe Gier."
"""

from src.jv2.base_bot import JV2Bot, _safe
from src.jv2.models import JV2Signal


class MeanReverter(JV2Bot):
    def __init__(self, symbol="XRP/USDT"):
        super().__init__("mean_reverter", symbol)

    def check_thesis(self, market_data):
        """Mean-Reversion gilt bis RSI zurück in neutraler Zone ODER starker Trend entsteht."""
        if not self.state.position:
            return True, ""
        r4 = market_data.get("latest_4h")
        if r4 is None:
            return True, ""
        rsi = _safe(r4.get("4h_rsi_14", 50))
        adx = _safe(r4.get("4h_adx", 20))
        trend_cons = _safe(r4.get("4h_trend_consistency", 0.5))
        # Long-These: RSI war überverkauft, jetzt zurück zur Mitte = These erfüllt → raus
        if self.state.position.direction == "long" and rsi > 55:
            return False, f"RSI normalisiert ({rsi:.0f}) — These erfüllt"
        # Short-These: RSI war überkauft
        if self.state.position.direction == "short" and rsi < 45:
            return False, f"RSI normalisiert ({rsi:.0f}) — These erfüllt"
        # Starker Trend gegen Position = These tot
        if adx > 35 and trend_cons > 0.6:
            ema = _safe(r4.get("4h_ema_9_above_21", 0.5))
            if self.state.position.direction == "long" and ema < 0.5:
                return False, "Starker Abwärtstrend"
            if self.state.position.direction == "short" and ema > 0.5:
                return False, "Starker Aufwärtstrend"
        return True, ""

    def generate_signal(self, market_data, spy_intel):
        price = market_data["price"]
        r4 = market_data.get("latest_4h")
        if r4 is None:
            return self.neutral(price, "Keine 4H-Daten")

        rsi = _safe(r4.get("4h_rsi_14", 50))
        bb_pos = _safe(r4.get("4h_bb_pos", 0.5))
        ema50_dist = _safe(r4.get("4h_ema_50_dist", 0))
        stoch_rsi = _safe(r4.get("4h_stoch_rsi", 0.5))
        adx = _safe(r4.get("4h_adx", 20))
        trend_cons = _safe(r4.get("4h_trend_consistency", 0.5))

        # Liquidation features — capitulation = mechanical mean-rev trigger.
        liq = market_data.get("liquidations", {})
        long_capit = bool(liq.get("liq_long_capit_flag", 0))   # longs got liquidated -> mean-rev LONG bias
        short_capit = bool(liq.get("liq_short_capit_flag", 0)) # shorts got liquidated -> mean-rev SHORT bias
        liq_imb = _safe(liq.get("liq_imbalance", 0))

        direction = "neutral"
        conf = 0.0
        liq_reasons = []

        # LONG: Überverkauft (existing RSI/BB-based trigger)
        if rsi < 35 and bb_pos < 0.20:
            direction = "long"
            conf = 0.25
            if rsi < 25: conf += 0.15
            if bb_pos < 0.05: conf += 0.10
            if ema50_dist < -0.03: conf += 0.10
            if stoch_rsi < 0.15: conf += 0.10

        # SHORT: Überkauft (existing RSI/BB-based trigger)
        elif rsi > 65 and bb_pos > 0.80:
            direction = "short"
            conf = 0.25
            if rsi > 75: conf += 0.15
            if bb_pos > 0.95: conf += 0.10
            if ema50_dist > 0.03: conf += 0.10
            if stoch_rsi > 0.85: conf += 0.10

        # LIQUIDATION-CAPITULATION trigger — independent of RSI/BB.
        # Mechanic: forced liquidations create overshoots that mean-revert.
        # Only fires if existing trigger is neutral or aligned.
        if direction == "neutral":
            if long_capit and liq_imb > 0.5:
                direction = "long"
                conf = 0.30
                liq_reasons.append("LongLiqCapit!")
            elif short_capit and liq_imb < -0.5:
                direction = "short"
                conf = 0.30
                liq_reasons.append("ShortLiqCapit!")
        else:
            # Aligned-direction capitulation: confidence boost
            if direction == "long" and long_capit:
                conf += 0.20
                liq_reasons.append("LongLiqBoost")
            elif direction == "short" and short_capit:
                conf += 0.20
                liq_reasons.append("ShortLiqBoost")
            # Opposite-direction capitulation: confidence cut (regime shift risk)
            elif direction == "long" and short_capit:
                conf *= 0.5
                liq_reasons.append("ShortLiqWarn")
            elif direction == "short" and long_capit:
                conf *= 0.5
                liq_reasons.append("LongLiqWarn")

        if direction == "neutral":
            return self.neutral(price, f"Kein Extrem (RSI:{rsi:.0f} BB:{bb_pos:.2f})")

        # Sicherheit: Nicht gegen starken Trend
        if adx > 35 and trend_cons > 0.6:
            conf *= 0.5

        # Spy: Warnung von Trend-Bots
        trend_str = spy_intel.get("trend_strength", 0)
        if trend_str and trend_str > 0.6:
            conf *= 0.8

        conf = min(conf, 1.0)

        liq_tag = f" [{','.join(liq_reasons)}]" if liq_reasons else ""
        return JV2Signal(
            bot_id=self.bot_id,
            timestamp=JV2Signal.neutral("", 0).timestamp,
            direction=direction,
            confidence=round(conf, 3),
            reasoning=f"MEAN-REV {direction.upper()}: RSI:{rsi:.0f} BB:{bb_pos:.2f} EMA50d:{ema50_dist:.3f}{liq_tag}",
            price_at_signal=price,
            features={"rsi": rsi, "bb_pos": bb_pos, "ema50_dist": ema50_dist, "stoch_rsi": stoch_rsi},
        )
