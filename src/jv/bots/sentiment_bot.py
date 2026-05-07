"""
Joint Venture Boting – Sentiment Bot
Spezialisiert auf externe Signale: Fear&Greed, Cross-Asset, BTC-Lead.

Sieht den Markt durch die Brille der Marktstimmung und Cross-Asset-Dynamik:
  Extreme Angst → contrarian bullish
  BTC bewegt sich, XRP nicht → Catch-Up Signal
  Alt-Season → XRP profitiert
"""

import math
from src.jv.base_bot import JVBot
from src.jv.signal_protocol import JVSignal


def _safe(val, default=0.0):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return float(val)


class SentimentBot(JVBot):
    def __init__(self):
        super().__init__("sentiment")

    def observe(self, market_data: dict) -> JVSignal:
        price = market_data["price"]
        exchange = market_data.get("exchange")

        # ── SENTIMENT FEATURES ────────────────────────
        try:
            from src.features.sentiment import compute_sentiment_features
            sent = compute_sentiment_features()
        except Exception:
            sent = {}

        # ── CROSS-ASSET FEATURES ──────────────────────
        try:
            from src.features.cross_asset import build_cross_asset_features
            cross = build_cross_asset_features(exchange)
        except Exception:
            cross = {}

        # ── SCORING ───────────────────────────────────
        bull_score = 0.0
        bear_score = 0.0
        reasons = []

        # 1. Fear & Greed (Contrarian)
        fng = _safe(sent.get("sent_fear_greed", 50))
        extreme = _safe(sent.get("sent_fear_greed_extreme", 0))

        if extreme == -1 or fng <= 20:
            # Extreme Fear → Contrarian Bullish
            bull_score += 0.30
            reasons.append(f"F&G:{fng:.0f}(ExFear)")
        elif extreme == 1 or fng >= 80:
            # Extreme Greed → Contrarian Bearish
            bear_score += 0.30
            reasons.append(f"F&G:{fng:.0f}(ExGreed)")
        elif fng < 35:
            bull_score += 0.10
            reasons.append(f"F&G:{fng:.0f}(Fear)")
        elif fng > 65:
            bear_score += 0.10
            reasons.append(f"F&G:{fng:.0f}(Greed)")

        # 2. BTC Lead-Lag (Catch-Up Signal)
        catchup = _safe(cross.get("ca_catchup_signal", 0))
        if catchup > 2.0:
            bull_score += 0.25
            reasons.append(f"BTC-Catchup:{catchup:.1f}")
        elif catchup < -2.0:
            bear_score += 0.25
            reasons.append(f"BTC-Catchup:{catchup:.1f}")

        # 3. Alt-Season
        alt_season = _safe(cross.get("ca_alt_season", 0))
        if alt_season >= 2:
            bull_score += 0.15
            reasons.append("AltSeason!")
        elif alt_season == 0:
            bear_score += 0.10
            reasons.append("BTC-Dom")

        # 4. BTC-Only Pump (bearish für Alts)
        if cross.get("ca_btc_only_pump", False):
            bear_score += 0.15
            reasons.append("BTC-Only-Pump")

        # 5. Alt-Only Pump (bullish für XRP)
        if cross.get("ca_alt_only_pump", False):
            bull_score += 0.15
            reasons.append("Alt-Only-Pump")

        # 6. Correlation Breakdown (Vorsicht)
        if cross.get("ca_corr_breakdown", False):
            # Reduziere beides – unsicheres Umfeld
            bull_score *= 0.7
            bear_score *= 0.7
            reasons.append("Corr-Breakdown")

        # ── ENTSCHEIDUNG ─────────────────────────────
        net_score = bull_score - bear_score

        if abs(net_score) < 0.10:
            return self.neutral(price,
                f"Sentiment neutral (Bull:{bull_score:.2f} Bear:{bear_score:.2f})")

        direction = "long" if net_score > 0 else "short"
        confidence = min(abs(net_score), 1.0)

        return JVSignal.create(
            bot_id=self.bot_id,
            direction=direction,
            confidence=round(confidence, 3),
            reasoning=f"Sentiment {direction.upper()}: {', '.join(reasons)}",
            price=price,
            features={
                "fear_greed": round(fng, 1),
                "catchup_signal": round(catchup, 2),
                "alt_season": alt_season,
                "composite": _safe(sent.get("sent_composite", 0.5)),
            },
            ttl_candles=4,  # Sentiment ist langlebiger
        )
