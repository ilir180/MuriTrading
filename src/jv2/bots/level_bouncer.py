"""
JV Boting v2 – Level Bouncer
These: "Support hält, Resistance hält. Trade den Bounce."
"""

from src.jv2.base_bot import JV2Bot, _safe
from src.jv2.models import JV2Signal


def _find_pivots(df, lookback=20, window=2):
    """Findet Pivot Highs und Lows in den letzten `lookback` Candles."""
    data = df.tail(lookback)
    highs, lows = [], []
    for i in range(window, len(data) - window):
        h = data.iloc[i]["high"]
        l = data.iloc[i]["low"]
        is_high = all(h >= data.iloc[i + j]["high"] for j in range(-window, window + 1) if j != 0)
        is_low = all(l <= data.iloc[i + j]["low"] for j in range(-window, window + 1) if j != 0)
        if is_high:
            highs.append(h)
        if is_low:
            lows.append(l)
    return highs, lows


def _cluster_levels(levels, tolerance=0.003):
    """Gruppiert nahe beieinander liegende Levels."""
    if not levels:
        return []
    levels = sorted(levels)
    clusters = [[levels[0]]]
    for lev in levels[1:]:
        if abs(lev - clusters[-1][-1]) / clusters[-1][-1] < tolerance:
            clusters[-1].append(lev)
        else:
            clusters.append([lev])
    return [sum(c) / len(c) for c in clusters]


def _count_touches(df, level, tolerance=0.003):
    """Zählt wie oft der Preis ein Level berührt hat."""
    count = 0
    for _, row in df.iterrows():
        if abs(row["low"] - level) / level < tolerance or abs(row["high"] - level) / level < tolerance:
            count += 1
    return count


class LevelBouncer(JV2Bot):
    def __init__(self, symbol="XRP/USDT"):
        super().__init__("level_bouncer", symbol)

    def check_thesis(self, market_data):
        """Level gilt solange das Support/Resistance-Level nicht gebrochen ist."""
        if not self.state.position:
            return True, ""
        price = market_data["price"]
        entry = self.state.position.entry_price
        df_4h = market_data.get("df_4h")
        if df_4h is None or len(df_4h) < 25:
            return True, ""

        # Rekonstruiere das nächste Level vom Entry
        highs, lows = _find_pivots(df_4h, lookback=30, window=2)
        supports = _cluster_levels(lows)
        resistances = _cluster_levels(highs)

        if self.state.position.direction == "long":
            # Finde Support unter Entry
            nearest_sup = None
            for s in sorted(supports, reverse=True):
                if s < entry * 1.01:
                    nearest_sup = s
                    break
            if nearest_sup and price < nearest_sup * 0.997:
                return False, f"Support ${nearest_sup:.4f} gebrochen"
        else:
            # Finde Resistance über Entry
            nearest_res = None
            for r in sorted(resistances):
                if r > entry * 0.99:
                    nearest_res = r
                    break
            if nearest_res and price > nearest_res * 1.003:
                return False, f"Resistance ${nearest_res:.4f} gebrochen"
        return True, ""

    def generate_signal(self, market_data, spy_intel):
        price = market_data["price"]
        df_4h = market_data.get("df_4h")
        r4 = market_data.get("latest_4h")
        if df_4h is None or len(df_4h) < 25:
            return self.neutral(price, "Zu wenig Daten")

        highs, lows = _find_pivots(df_4h, lookback=30, window=2)
        supports = _cluster_levels(lows)
        resistances = _cluster_levels(highs)

        # Nächste Levels finden
        nearest_sup = None
        nearest_res = None
        for s in sorted(supports, reverse=True):
            if s < price:
                nearest_sup = s
                break
        for r in sorted(resistances):
            if r > price:
                nearest_res = r
                break

        sup_dist = (price - nearest_sup) / price if nearest_sup else 1.0
        res_dist = (nearest_res - price) / price if nearest_res else 1.0

        direction = "neutral"
        conf = 0.0
        reasons = []

        # LONG: Bounce von Support
        if nearest_sup and sup_dist < 0.008 and sup_dist > 0:
            direction = "long"
            conf = 0.25
            touches = _count_touches(df_4h.tail(30), nearest_sup)
            if touches >= 3: conf += 0.15
            elif touches >= 2: conf += 0.08
            reasons.append(f"Sup:${nearest_sup:.4f}({touches}x)")

            vol_ratio = _safe(r4.get("4h_vol_ratio", 1.0)) if r4 is not None else 1.0
            if vol_ratio > 1.3: conf += 0.10

            lower_wick = _safe(r4.get("4h_lower_wick", 0)) if r4 is not None else 0
            if lower_wick > 0.5: conf += 0.10
            reasons.append(f"Wick:{lower_wick:.2f}")

        # SHORT: Rejection an Resistance
        elif nearest_res and res_dist < 0.008 and res_dist > 0:
            direction = "short"
            conf = 0.25
            touches = _count_touches(df_4h.tail(30), nearest_res)
            if touches >= 3: conf += 0.15
            elif touches >= 2: conf += 0.08
            reasons.append(f"Res:${nearest_res:.4f}({touches}x)")

            vol_ratio = _safe(r4.get("4h_vol_ratio", 1.0)) if r4 is not None else 1.0
            if vol_ratio > 1.3: conf += 0.10

            upper_wick = _safe(r4.get("4h_upper_wick", 0)) if r4 is not None else 0
            if upper_wick > 0.5: conf += 0.10
            reasons.append(f"Wick:{upper_wick:.2f}")

        if direction == "neutral":
            return self.neutral(price, f"Kein Level nahe (Sup:{sup_dist:.3f} Res:{res_dist:.3f})")

        conf = min(conf, 1.0)

        return JV2Signal(
            bot_id=self.bot_id,
            timestamp=JV2Signal.neutral("", 0).timestamp,
            direction=direction,
            confidence=round(conf, 3),
            reasoning=f"LEVEL {direction.upper()}: {', '.join(reasons)}",
            price_at_signal=price,
            features={"sup_dist": sup_dist, "res_dist": res_dist},
        )
