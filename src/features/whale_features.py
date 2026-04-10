"""
MuriTrading – Whale Order Detection
Erkennt grosse Orderbuch-Positionen und Whale-Trades in Echtzeit.

Features:
  - Bid/Ask Imbalance (Kauf- vs Verkaufsdruck)
  - Order Walls (grosse Einzelorders als Support/Resistance)
  - Big Trade Detection (Trades > $50k)
  - Net Whale Flow (Netto-Richtung der Wale)
  - Absorption (Wall wird aufgefressen = Breakout-Signal)
  - Depth Ratios bei 1%, 2%, 5% vom Midprice
"""

import time
import math
import requests

# ── Konfiguration ─────────────────────────────────────────────
BINANCE_URLS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
    "https://api.binance.us",
]
SYMBOL = "XRPUSDT"
BIG_TRADE_USD = 50_000
WALL_PCT_THRESHOLD = 0.05    # Einzelorder > 5% des gesamten Depths = Wall
WHALE_LOOKBACK_S = 600       # Letzte 10 Minuten

# ── State für Absorption-Detection ────────────────────────────
_previous_walls = {"bid_price": None, "bid_qty": None,
                   "ask_price": None, "ask_qty": None,
                   "mid_price": None, "timestamp": 0}


def _fetch_json(endpoint, params=None, timeout=10):
    """Fetcht JSON von Binance mit Fallback-URLs."""
    for base in BINANCE_URLS:
        try:
            r = requests.get(f"{base}{endpoint}", params=params, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            continue
    return None


def _parse_depth(raw):
    """Parsed Orderbuch in Listen von (price, qty, usd_value)."""
    bids, asks = [], []
    for p, q in raw.get("bids", []):
        price, qty = float(p), float(q)
        bids.append((price, qty, price * qty))
    for p, q in raw.get("asks", []):
        price, qty = float(p), float(q)
        asks.append((price, qty, price * qty))
    return bids, asks


def _depth_ratios(bids, asks, mid):
    """Berechnet Bid/Ask-Volumen-Ratio bei verschiedenen Distanzen vom Midprice."""
    ratios = {}
    for pct in [0.01, 0.02, 0.05]:
        lower = mid * (1 - pct)
        upper = mid * (1 + pct)
        bid_vol = sum(usd for p, q, usd in bids if p >= lower)
        ask_vol = sum(usd for p, q, usd in asks if p <= upper)
        ratios[f"whale_depth_ratio_{int(pct*100)}pct"] = (
            bid_vol / (ask_vol + 1e-10)
        )
    return ratios


def _detect_walls(entries, total_usd):
    """Findet die grösste Wall (Einzelorder > WALL_PCT_THRESHOLD des Gesamtvolumens)."""
    if total_usd == 0 or not entries:
        return False, 0.0, 0.0, 0.0
    best_price, best_qty, best_usd = 0, 0, 0
    for price, qty, usd in entries:
        if usd > best_usd:
            best_price, best_qty, best_usd = price, qty, usd
    is_wall = best_usd / total_usd > WALL_PCT_THRESHOLD
    return is_wall, best_price, best_qty, best_usd


def _compute_big_trades(trades, now_ms):
    """Analysiert grosse Trades in den letzten WHALE_LOOKBACK_S Sekunden."""
    cutoff = now_ms - WHALE_LOOKBACK_S * 1000
    big_buys_usd = 0.0
    big_sells_usd = 0.0
    big_count = 0
    total_volume_usd = 0.0

    for t in trades:
        ts = t.get("T", 0)
        if ts < cutoff:
            continue
        price = float(t["p"])
        qty = float(t["q"])
        usd = price * qty
        total_volume_usd += usd

        if usd >= BIG_TRADE_USD:
            big_count += 1
            if t.get("m", False):
                # isBuyerMaker=True → Seller ist Taker (aggressive sell)
                big_sells_usd += usd
            else:
                big_buys_usd += usd

    net_flow = big_buys_usd - big_sells_usd
    net_flow_norm = net_flow / (total_volume_usd + 1e-10)

    return {
        "whale_big_buys_usd": big_buys_usd,
        "whale_big_sells_usd": big_sells_usd,
        "whale_big_trade_count": big_count,
        "whale_net_flow": net_flow,
        "whale_net_flow_normalized": net_flow_norm,
        "whale_total_volume_usd": total_volume_usd,
    }


def _check_absorption(mid, wall_bid_price, wall_bid_qty, wall_ask_price, wall_ask_qty):
    """Prüft ob eine Wall vom letzten Zyklus aufgefressen wird (Absorption)."""
    global _previous_walls
    absorption_bid = 0
    absorption_ask = 0

    prev = _previous_walls
    if prev["timestamp"] > 0 and (time.time() - prev["timestamp"]) < 180:
        # Bid-Wall Absorption: gleicher Preis, aber Qty >50% geschrumpft
        if (prev["bid_price"] and wall_bid_price
                and abs(prev["bid_price"] - wall_bid_price) / (prev["bid_price"] + 1e-10) < 0.001
                and prev["bid_qty"] > 0):
            shrink = 1 - wall_bid_qty / prev["bid_qty"]
            if shrink > 0.5:
                absorption_bid = 1  # Bid-Wall wird aufgefressen → bearish

        # Ask-Wall Absorption: gleicher Preis, Qty >50% geschrumpft
        if (prev["ask_price"] and wall_ask_price
                and abs(prev["ask_price"] - wall_ask_price) / (prev["ask_price"] + 1e-10) < 0.001
                and prev["ask_qty"] > 0):
            shrink = 1 - wall_ask_qty / prev["ask_qty"]
            if shrink > 0.5:
                absorption_ask = 1  # Ask-Wall wird aufgefressen → bullish

    # State updaten
    _previous_walls.update({
        "bid_price": wall_bid_price, "bid_qty": wall_bid_qty,
        "ask_price": wall_ask_price, "ask_qty": wall_ask_qty,
        "mid_price": mid, "timestamp": time.time(),
    })

    return absorption_bid, absorption_ask


def compute_whale_features():
    """
    Hauptfunktion: Holt Orderbuch + Trades und berechnet alle Whale-Features.
    Gibt ein Dict zurück. Bei Fehler alle Werte NaN.
    """
    nan = float("nan")
    empty = {
        "whale_bid_ask_imbalance": nan,
        "whale_depth_ratio_1pct": nan,
        "whale_depth_ratio_2pct": nan,
        "whale_depth_ratio_5pct": nan,
        "whale_wall_bid": 0, "whale_wall_ask": 0,
        "whale_wall_bid_price": nan, "whale_wall_ask_price": nan,
        "whale_wall_bid_distance": nan, "whale_wall_ask_distance": nan,
        "whale_big_buys_usd": nan, "whale_big_sells_usd": nan,
        "whale_big_trade_count": 0, "whale_net_flow": nan,
        "whale_net_flow_normalized": nan, "whale_total_volume_usd": nan,
        "whale_absorption_bid": 0, "whale_absorption_ask": 0,
    }

    try:
        # 1. Orderbuch holen
        depth_raw = _fetch_json(f"/api/v3/depth", {"symbol": SYMBOL, "limit": 1000})
        if not depth_raw:
            return empty

        bids, asks = _parse_depth(depth_raw)
        if not bids or not asks:
            return empty

        mid = (bids[0][0] + asks[0][0]) / 2.0

        # 2. Bid/Ask Imbalance
        total_bid_usd = sum(usd for _, _, usd in bids)
        total_ask_usd = sum(usd for _, _, usd in asks)
        imbalance = total_bid_usd / (total_bid_usd + total_ask_usd + 1e-10)

        # 3. Depth Ratios
        ratios = _depth_ratios(bids, asks, mid)

        # 4. Walls
        wall_bid, wb_price, wb_qty, wb_usd = _detect_walls(bids, total_bid_usd)
        wall_ask, wa_price, wa_qty, wa_usd = _detect_walls(asks, total_ask_usd)

        wb_dist = (mid - wb_price) / mid if wb_price > 0 else nan
        wa_dist = (wa_price - mid) / mid if wa_price > 0 else nan

        # 5. Absorption
        abs_bid, abs_ask = _check_absorption(mid, wb_price, wb_qty, wa_price, wa_qty)

        # 6. Grosse Trades
        trades_raw = _fetch_json(f"/api/v3/aggTrades", {"symbol": SYMBOL, "limit": 1000})
        if trades_raw:
            now_ms = int(time.time() * 1000)
            trade_feats = _compute_big_trades(trades_raw, now_ms)
        else:
            trade_feats = {k: nan for k in [
                "whale_big_buys_usd", "whale_big_sells_usd",
                "whale_big_trade_count", "whale_net_flow",
                "whale_net_flow_normalized", "whale_total_volume_usd",
            ]}

        return {
            "whale_bid_ask_imbalance": round(imbalance, 4),
            **ratios,
            "whale_wall_bid": int(wall_bid),
            "whale_wall_ask": int(wall_ask),
            "whale_wall_bid_price": round(wb_price, 6) if wall_bid else nan,
            "whale_wall_ask_price": round(wa_price, 6) if wall_ask else nan,
            "whale_wall_bid_distance": round(wb_dist, 4) if wall_bid else nan,
            "whale_wall_ask_distance": round(wa_dist, 4) if wall_ask else nan,
            "whale_absorption_bid": abs_bid,
            "whale_absorption_ask": abs_ask,
            **trade_feats,
        }

    except Exception as e:
        return empty


def whale_signal_text(w):
    """Formatiert Whale-Features als kurzen Text-String für Logs/Telegram."""
    if math.isnan(w.get("whale_bid_ask_imbalance", float("nan"))):
        return "Whale: n/a"

    imb = w["whale_bid_ask_imbalance"]
    direction = "BUY" if imb > 0.55 else "SELL" if imb < 0.45 else "NEUTRAL"

    parts = [f"Whale: {direction} ({imb:.0%})"]

    net = w.get("whale_net_flow", 0)
    if not math.isnan(net) and abs(net) > 1000:
        parts.append(f"Flow:${net/1000:+.0f}k")

    if w.get("whale_wall_bid"):
        parts.append(f"BidWall@{w['whale_wall_bid_price']:.4f}")
    if w.get("whale_wall_ask"):
        parts.append(f"AskWall@{w['whale_wall_ask_price']:.4f}")

    if w.get("whale_absorption_ask"):
        parts.append("🔥ABSORPTION↑")
    if w.get("whale_absorption_bid"):
        parts.append("🔥ABSORPTION↓")

    return "  ".join(parts)
