"""
JV Boting v2 – Trend Rider
These: "Der Trend läuft weiter bis er bricht."
"""

from src.jv2.base_bot import JV2Bot, _safe
from src.jv2.models import JV2Signal


class TrendRider(JV2Bot):
    def __init__(self, symbol="XRP/USDT"):
        super().__init__("trend_rider", symbol)
        self._thesis_strikes = 0

    def check_thesis(self, market_data):
        """Trend gilt noch solange ADX > 15 (mit 2-Kerzen-Hysterese).

        Die frühere EMA-Cross-Bedingung ist bewusst ENTFERNT: Exit-Replay
        (Deep Dive 10.06.26, 146 trend_rider-Trades auf echten Kerzen) zeigte
        im selben Replay-Engine: mit EMA-Cross-Exit +$28, mit Hysterese +$38,
        NUR-ADX +$115 bei WR 62% — die EMA-Kreuzung ist 4H-Noise, echte
        Trendbrüche fängt der Trailing-Stop. ADX-Tod bleibt als Exit (rettet
        Trades vor dem TIME-Exit im toten Markt).
        """
        if not self.state.position:
            self._thesis_strikes = 0
            return True, ""
        r4 = market_data.get("latest_4h")
        if r4 is None:
            return True, ""
        adx = _safe(r4.get("4h_adx", 0))
        if adx >= 15:
            self._thesis_strikes = 0
            return True, ""
        self._thesis_strikes += 1
        if self._thesis_strikes >= 2:
            self._thesis_strikes = 0
            return False, f"Trend tot (ADX:{adx:.0f}, 2 Kerzen bestätigt)"
        return True, ""

    def generate_signal(self, market_data, spy_intel):
        price = market_data["price"]
        r4 = market_data.get("latest_4h")
        r1d = market_data.get("latest_1d")
        if r4 is None:
            return self.neutral(price, "Keine 4H-Daten")

        adx = _safe(r4.get("4h_adx", 0))
        trend_cons = _safe(r4.get("4h_trend_consistency", 0.5))
        chop = _safe(r4.get("4h_chop", 0.5))

        # EMA Alignment Score (0-4)
        ema_align = 0
        if _safe(r4.get("4h_ema_9_above_21", 0)) > 0.5: ema_align += 1
        else: ema_align -= 1
        if _safe(r4.get("4h_ema_21_above_50", 0)) > 0.5: ema_align += 1
        else: ema_align -= 1
        if r1d is not None:
            if _safe(r1d.get("1d_ema_9_above_21", 0)) > 0.5: ema_align += 1
            else: ema_align -= 1
            if _safe(r1d.get("1d_ema_21_above_50", 0)) > 0.5: ema_align += 1
            else: ema_align -= 1

        # Gate: ADX > 20
        if adx < 20:
            return self.neutral(price, f"Kein Trend (ADX:{adx:.0f})")

        # Richtung aus EMA Alignment
        if ema_align >= 2:
            direction = "long"
        elif ema_align <= -2:
            direction = "short"
        else:
            return self.neutral(price, f"EMA unklar (align:{ema_align})")

        # Confidence
        conf = 0.25
        conf += min((adx - 20) / 30, 0.25)
        if trend_cons > 0.5:
            conf += (trend_cons - 0.5) * 0.6
        if abs(ema_align) == 4:
            conf += 0.10
        if chop < 0.40:
            conf += 0.05

        # Spy Intel
        if spy_intel.get("whale_direction") == direction:
            conf += 0.05
        if spy_intel.get("momentum_confirms") == direction:
            conf += 0.03

        conf = min(conf, 1.0)

        return JV2Signal(
            bot_id=self.bot_id,
            timestamp=JV2Signal.neutral("", 0).timestamp,
            direction=direction,
            confidence=round(conf, 3),
            reasoning=f"TREND {direction.upper()}: ADX:{adx:.0f} EMA:{ema_align} TC:{trend_cons:.2f}",
            price_at_signal=price,
            features={"adx": adx, "ema_align": ema_align, "trend_cons": trend_cons, "chop": chop},
        )
