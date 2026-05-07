"""
JV Boting v2 – Momentum Surfer
These: "Stärke wird stärker. Ride the wave."
"""

from src.jv2.base_bot import JV2Bot, _safe
from src.jv2.models import JV2Signal


class MomentumSurfer(JV2Bot):
    def __init__(self, symbol="XRP/USDT"):
        super().__init__("momentum_surfer", symbol)

    def check_thesis(self, market_data):
        """Momentum gilt solange Velocity und Acceleration nicht drehen."""
        if not self.state.position:
            return True, ""
        r4 = market_data.get("latest_4h")
        if r4 is None:
            return True, ""
        vel = _safe(r4.get("4h_return_1", 0))
        ret3 = _safe(r4.get("4h_return_3", 0))
        accel = vel - (ret3 / 3 if ret3 != 0 else 0)
        # Long: Velocity oder Acceleration negativ = Momentum stirbt
        if self.state.position.direction == "long":
            if vel < -0.002 and accel < 0:
                return False, f"Momentum gedreht (Vel:{vel:+.3f} Acc:{accel:+.4f})"
        # Short: Velocity oder Acceleration positiv
        if self.state.position.direction == "short":
            if vel > 0.002 and accel > 0:
                return False, f"Momentum gedreht (Vel:{vel:+.3f} Acc:{accel:+.4f})"
        return True, ""

    def generate_signal(self, market_data, spy_intel):
        price = market_data["price"]
        r4 = market_data.get("latest_4h")
        r1h = market_data.get("latest_1h")
        r1d = market_data.get("latest_1d")
        if r4 is None:
            return self.neutral(price, "Keine 4H-Daten")

        # Velocity
        vel_4h = _safe(r4.get("4h_return_1", 0))
        vel_1h = _safe(r1h.get("1h_return_1", 0)) if r1h is not None else 0
        vel_1d = _safe(r1d.get("1d_return_1", 0)) if r1d is not None else 0

        # Acceleration
        ret_4h_3 = _safe(r4.get("4h_return_3", 0))
        avg_vel = ret_4h_3 / 3 if ret_4h_3 != 0 else 0
        accel = vel_4h - avg_vel

        # MACD Slope
        macd_hist = _safe(r4.get("4h_macd_hist", 0))
        df_4h = market_data.get("df_4h")
        macd_prev = 0
        if df_4h is not None and len(df_4h) >= 2:
            macd_prev = _safe(df_4h.iloc[-2].get("4h_macd_hist", 0))
        macd_slope = macd_hist - macd_prev

        # Multi-TF alignment
        tf_bull = sum([vel_1h > 0, vel_4h > 0, vel_1d > 0])

        direction = "neutral"
        conf = 0.0

        # LONG: velocity + acceleration + MACD alle positiv
        if vel_4h > 0.001 and accel > 0 and macd_slope > 0:
            direction = "long"
            conf = 0.25
            vel_str = min(abs(vel_4h) / 0.02, 1.0)
            conf += vel_str * 0.20
            if tf_bull >= 3: conf += 0.15
            elif tf_bull >= 2: conf += 0.08
            if accel > 0.001: conf += 0.10
            stoch = _safe(r4.get("4h_stoch_rsi", 0.5))
            if stoch < 0.80: conf += 0.05

        # SHORT: alles negativ
        elif vel_4h < -0.001 and accel < 0 and macd_slope < 0:
            direction = "short"
            conf = 0.25
            vel_str = min(abs(vel_4h) / 0.02, 1.0)
            conf += vel_str * 0.20
            tf_bear = 3 - tf_bull
            if tf_bear >= 3: conf += 0.15
            elif tf_bear >= 2: conf += 0.08
            if accel < -0.001: conf += 0.10
            stoch = _safe(r4.get("4h_stoch_rsi", 0.5))
            if stoch > 0.20: conf += 0.05

        if direction == "neutral":
            return self.neutral(price, f"Divergenz (Vel:{vel_4h:+.3f} Acc:{accel:+.4f} MACD:{macd_slope:+.4f})")

        conf = min(conf, 1.0)

        return JV2Signal(
            bot_id=self.bot_id,
            timestamp=JV2Signal.neutral("", 0).timestamp,
            direction=direction,
            confidence=round(conf, 3),
            reasoning=f"MOMENTUM {direction.upper()}: Vel:{vel_4h:+.3f} Acc:{accel:+.4f} TF:{tf_bull}/3",
            price_at_signal=price,
            features={"vel_4h": vel_4h, "accel": accel, "macd_slope": macd_slope, "tf_bull": tf_bull},
        )
