"""
JV Boting v2 – Flow Tracker
These: "Folge dem Smart Money."
"""

from src.jv2.base_bot import JV2Bot, _safe
from src.jv2.models import JV2Signal


class FlowTracker(JV2Bot):
    def __init__(self, symbol="XRP/USDT"):
        super().__init__("flow_tracker", symbol)

    def check_thesis(self, market_data):
        """Flow gilt solange Whale-Flow nicht dreht."""
        if not self.state.position:
            return True, ""
        whale = market_data.get("whale", {})
        imbalance = _safe(whale.get("whale_bid_ask_imbalance", 0.5))
        net_flow = _safe(whale.get("whale_net_flow_normalized", 0))
        # Long: Flow darf nicht bearish werden
        if self.state.position.direction == "long":
            if imbalance < 0.40 and net_flow < -0.3:
                return False, f"Flow gedreht bearish (Imb:{imbalance:.2f} Flow:{net_flow:.2f})"
        # Short: Flow darf nicht bullish werden
        if self.state.position.direction == "short":
            if imbalance > 0.60 and net_flow > 0.3:
                return False, f"Flow gedreht bullish (Imb:{imbalance:.2f} Flow:{net_flow:.2f})"
        return True, ""

    def generate_signal(self, market_data, spy_intel):
        price = market_data["price"]
        whale = market_data.get("whale", {})
        r4 = market_data.get("latest_4h")

        imbalance = _safe(whale.get("whale_bid_ask_imbalance", 0.5))
        net_flow = _safe(whale.get("whale_net_flow_normalized", 0))
        depth_1 = _safe(whale.get("whale_depth_ratio_1pct", 1.0))
        absorption_ask = whale.get("whale_absorption_ask", False)
        absorption_bid = whale.get("whale_absorption_bid", False)
        vol_ratio = _safe(r4.get("4h_vol_ratio", 1.0)) if r4 is not None else 1.0
        obv_norm = _safe(r4.get("4h_obv_norm", 0)) if r4 is not None else 0.0

        bull = 0.0
        bear = 0.0
        reasons = []

        # Bid/Ask Imbalance
        if imbalance > 0.58:
            bull += 0.25
            reasons.append(f"BidDom:{imbalance:.2f}")
        elif imbalance < 0.42:
            bear += 0.25
            reasons.append(f"AskDom:{imbalance:.2f}")

        # Net Flow
        if net_flow > 0.2:
            bull += 0.20
            reasons.append(f"Flow:+{net_flow:.2f}")
        elif net_flow < -0.2:
            bear += 0.20
            reasons.append(f"Flow:{net_flow:.2f}")

        # Depth Ratio
        if depth_1 > 1.3:
            bull += 0.15
            reasons.append(f"Depth:{depth_1:.1f}x")
        elif depth_1 < 0.7:
            bear += 0.15
            reasons.append(f"Depth:{depth_1:.1f}x")

        # Wall Absorption
        if absorption_ask:
            bull += 0.30
            reasons.append("AskAbsorbed!")
        if absorption_bid:
            bear += 0.30
            reasons.append("BidAbsorbed!")

        # Volume + OBV
        if vol_ratio > 1.5:
            if obv_norm > 0.3:
                bull += 0.10
                reasons.append("OBV+")
            elif obv_norm < -0.3:
                bear += 0.10
                reasons.append("OBV-")

        net = bull - bear
        if abs(net) < 0.12:
            return self.neutral(price, f"Flow gemischt (B:{bull:.2f} S:{bear:.2f})")

        direction = "long" if net > 0 else "short"
        conf = min(abs(net), 1.0)

        return JV2Signal(
            bot_id=self.bot_id,
            timestamp=JV2Signal.neutral("", 0).timestamp,
            direction=direction,
            confidence=round(conf, 3),
            reasoning=f"FLOW {direction.upper()}: {', '.join(reasons)}",
            price_at_signal=price,
            features={"imbalance": imbalance, "net_flow": net_flow, "depth": depth_1},
        )
