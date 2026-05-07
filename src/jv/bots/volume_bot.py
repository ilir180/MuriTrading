"""
Joint Venture Boting – Volume Bot
Spezialisiert auf Orderflow: Whale-Activity, Bid-Ask Imbalance, Absorption.

Sieht den Markt durch die Brille des Geldflusses:
  Wer kauft? Wer verkauft? Wo sind die grossen Walls? Werden sie absorbiert?
"""

import math
from src.jv.base_bot import JVBot
from src.jv.signal_protocol import JVSignal


def _safe(val, default=0.0):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return float(val)


class VolumeBot(JVBot):
    def __init__(self):
        super().__init__("volume")

    def observe(self, market_data: dict) -> JVSignal:
        price = market_data["price"]
        exchange = market_data.get("exchange")

        # Whale Features holen
        try:
            from src.features.whale_features import compute_whale_features
            whale = compute_whale_features()
        except Exception as e:
            return self.neutral(price, f"Whale-Daten nicht verfügbar: {e}")

        # Volumen-Indikatoren aus 4H-Daten
        row_4h = market_data.get("latest_4h")
        vol_ratio = _safe(row_4h.get("4h_vol_ratio", 1.0) if row_4h is not None else 1.0)
        obv_norm = _safe(row_4h.get("4h_obv_norm", 0) if row_4h is not None else 0)

        # ── WHALE SIGNALS ─────────────────────────────
        imbalance = _safe(whale.get("whale_bid_ask_imbalance", 0.5))
        net_flow = _safe(whale.get("whale_net_flow_normalized", 0))
        depth_1 = _safe(whale.get("whale_depth_ratio_1pct", 1.0))
        big_buys = _safe(whale.get("whale_big_buys_usd", 0))
        big_sells = _safe(whale.get("whale_big_sells_usd", 0))
        absorption_ask = whale.get("whale_absorption_ask", False)
        absorption_bid = whale.get("whale_absorption_bid", False)

        # ── SCORING ───────────────────────────────────
        bull_score = 0.0
        bear_score = 0.0
        reasons = []

        # Bid/Ask Imbalance
        if imbalance > 0.60:
            bull_score += 0.25
            reasons.append(f"BidDom:{imbalance:.2f}")
        elif imbalance < 0.40:
            bear_score += 0.25
            reasons.append(f"AskDom:{imbalance:.2f}")

        # Net Flow (grosse Trades)
        if net_flow > 0.3:
            bull_score += 0.20
            reasons.append(f"NetFlow:+{net_flow:.2f}")
        elif net_flow < -0.3:
            bear_score += 0.20
            reasons.append(f"NetFlow:{net_flow:.2f}")

        # Depth Ratio (mehr Bids als Asks nahe am Preis)
        if depth_1 > 1.5:
            bull_score += 0.15
            reasons.append(f"Depth:{depth_1:.1f}x")
        elif depth_1 < 0.67:
            bear_score += 0.15
            reasons.append(f"Depth:{depth_1:.1f}x")

        # Wall Absorption (sehr starkes Signal)
        if absorption_ask:
            bull_score += 0.25
            reasons.append("AskWall-Absorbed!")
        if absorption_bid:
            bear_score += 0.25
            reasons.append("BidWall-Absorbed!")

        # Volume Spike + OBV
        if vol_ratio > 1.5:
            if obv_norm > 0.5:
                bull_score += 0.10
                reasons.append(f"VolSpike+OBV:{obv_norm:.1f}")
            elif obv_norm < -0.5:
                bear_score += 0.10
                reasons.append(f"VolSpike+OBV:{obv_norm:.1f}")

        # ── ENTSCHEIDUNG ─────────────────────────────
        net_score = bull_score - bear_score

        if abs(net_score) < 0.15:
            return self.neutral(price,
                f"Orderflow gemischt (Bull:{bull_score:.2f} Bear:{bear_score:.2f})")

        direction = "long" if net_score > 0 else "short"
        confidence = min(abs(net_score), 1.0)

        return JVSignal.create(
            bot_id=self.bot_id,
            direction=direction,
            confidence=round(confidence, 3),
            reasoning=f"Orderflow {direction.upper()}: {', '.join(reasons)}",
            price=price,
            features={
                "imbalance": round(imbalance, 3),
                "net_flow": round(net_flow, 3),
                "depth_ratio": round(depth_1, 2),
                "vol_ratio": round(vol_ratio, 2),
                "obv_norm": round(obv_norm, 2),
                "absorption_ask": absorption_ask,
                "absorption_bid": absorption_bid,
            },
            ttl_candles=2,  # Orderflow ist kurzlebig
        )
