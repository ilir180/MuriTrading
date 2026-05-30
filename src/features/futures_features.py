"""JV Boting v2 — Crypto-Native Futures Features.

Provides three mechanical edges identified in the Crypto-Native deep-dive:
  1. Funding Rate Z-Score (90d rolling) — sentiment extremum filter
  2. Open Interest Δ × Price Δ Quadrant — mean-reversion / continuation signal
  3. (Liquidation Stream lives in liquidation_stream.py — separate file)

All three pull from Binance Perpetual Futures public REST. No auth required.
Public endpoints: /fapi/v1/fundingRate, /fapi/v1/premiumIndex,
                  /fapi/v1/openInterest, /fapi/v1/openInterestHist

Designed to be resilient: any HTTP failure returns a "neutral" feature dict
so the bots don't crash. Features cache for 4 minutes (one tick worth) to
avoid hammering the endpoint.
"""

import math
import time
from collections import deque
from typing import Dict, List, Optional

import requests


BINANCE_FAPI = "https://fapi.binance.com"
CACHE_TTL_SEC = 240  # 4 minutes

# Per-symbol cache: {symbol: {"funding": (timestamp, dict), "oi": (timestamp, dict)}}
_CACHE: Dict[str, Dict[str, tuple]] = {}

# Rolling history for OI deltas. Persisted across ticks within a process.
_OI_HISTORY: Dict[str, deque] = {}


def _http_get_json(url: str, params: dict = None, timeout: float = 8.0) -> Optional[dict]:
    try:
        r = requests.get(url, params=params or {}, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def _cached(symbol: str, key: str):
    entry = _CACHE.get(symbol, {}).get(key)
    if entry is None:
        return None
    ts, value = entry
    if time.time() - ts > CACHE_TTL_SEC:
        return None
    return value


def _cache_set(symbol: str, key: str, value):
    _CACHE.setdefault(symbol, {})[key] = (time.time(), value)


# ── Funding Rate ───────────────────────────────────────────

def fetch_funding_rate_history(symbol: str, limit: int = 270) -> List[dict]:
    """Last `limit` funding events (each is 8h). 270 events ≈ 90 days."""
    cached = _cached(symbol, f"funding_hist_{limit}")
    if cached is not None:
        return cached
    data = _http_get_json(
        f"{BINANCE_FAPI}/fapi/v1/fundingRate",
        params={"symbol": symbol, "limit": limit},
    )
    if not data or not isinstance(data, list):
        return []
    _cache_set(symbol, f"funding_hist_{limit}", data)
    return data


def fetch_premium_index(symbol: str) -> Optional[dict]:
    """Current mark price, index price, current funding rate, next funding time."""
    cached = _cached(symbol, "premium")
    if cached is not None:
        return cached
    data = _http_get_json(
        f"{BINANCE_FAPI}/fapi/v1/premiumIndex",
        params={"symbol": symbol},
    )
    if not data or not isinstance(data, dict):
        return None
    _cache_set(symbol, "premium", data)
    return data


def compute_funding_features(symbol: str) -> Dict[str, float]:
    """Returns: {funding_current, funding_z, funding_pct_rank, funding_apr}.

    funding_z         — 90d rolling z-score of funding rate
    funding_pct_rank  — percentile of current within 90d history (0..1)
    funding_apr       — annualized estimate (3 events/day × 365)

    All zeros = neutral fallback (network failure, no data, etc.)
    """
    neutral = {
        "funding_current": 0.0,
        "funding_z": 0.0,
        "funding_pct_rank": 0.5,
        "funding_apr": 0.0,
    }
    history = fetch_funding_rate_history(symbol)
    if len(history) < 30:
        return neutral

    rates = []
    for item in history:
        try:
            rates.append(float(item.get("fundingRate", 0)))
        except (ValueError, TypeError):
            continue
    if len(rates) < 30:
        return neutral

    current = rates[-1]
    mean = sum(rates) / len(rates)
    var = sum((r - mean) ** 2 for r in rates) / len(rates)
    std = math.sqrt(var) if var > 0 else 0.0
    z = (current - mean) / std if std > 0 else 0.0
    pct_rank = sum(1 for r in rates if r < current) / len(rates)

    return {
        "funding_current": current,
        "funding_z": z,
        "funding_pct_rank": pct_rank,
        "funding_apr": current * 3 * 365,  # 3 events per day
    }


# ── Open Interest ──────────────────────────────────────────

def fetch_open_interest(symbol: str) -> Optional[float]:
    """Current OI in contracts (notional in base asset)."""
    cached = _cached(symbol, "oi_current")
    if cached is not None:
        return cached
    data = _http_get_json(
        f"{BINANCE_FAPI}/fapi/v1/openInterest",
        params={"symbol": symbol},
    )
    if not data or "openInterest" not in data:
        return None
    try:
        oi = float(data["openInterest"])
        _cache_set(symbol, "oi_current", oi)
        return oi
    except (ValueError, TypeError):
        return None


def fetch_oi_history(symbol: str, period: str = "4h", limit: int = 60) -> List[dict]:
    """OI history at given period. limit=60 at 4h = 10 days."""
    cached = _cached(symbol, f"oi_hist_{period}_{limit}")
    if cached is not None:
        return cached
    data = _http_get_json(
        f"{BINANCE_FAPI}/futures/data/openInterestHist",
        params={"symbol": symbol, "period": period, "limit": limit},
    )
    if not data or not isinstance(data, list):
        return []
    _cache_set(symbol, f"oi_hist_{period}_{limit}", data)
    return data


def compute_oi_features(symbol: str, current_price: float) -> Dict[str, float]:
    """Returns: {oi_current, oi_4h_delta, oi_4h_delta_pct, oi_quadrant, oi_quadrant_score}.

    oi_quadrant is a categorical code derived from sign(OI-Δ) × sign(Price-Δ):
      1 : OI↑ + Price↑  → new longs (bullish but liq risk rising)
      2 : OI↑ + Price↓  → new shorts (bearish conviction)
      3 : OI↓ + Price↑  → short squeeze (often short-lived)
      4 : OI↓ + Price↓  → long capitulation (often pre-bottom mean-rev)
      0 : insufficient data

    oi_quadrant_score is signed [−1, 1] proxy useful for direct mixing:
      +0.5 quad 4 (long-capit / mean-rev-long bias)
      +0.3 quad 1 (continuation-long)
      −0.3 quad 2 (continuation-short)
      −0.5 quad 3 (short-squeeze / mean-rev-short bias)
    """
    neutral = {
        "oi_current": 0.0, "oi_4h_delta": 0.0, "oi_4h_delta_pct": 0.0,
        "oi_quadrant": 0, "oi_quadrant_score": 0.0,
    }
    hist = fetch_oi_history(symbol, period="4h", limit=3)
    if len(hist) < 2:
        return neutral

    try:
        prev_oi = float(hist[-2].get("sumOpenInterest", 0))
        curr_oi = float(hist[-1].get("sumOpenInterest", 0))
        # Price history embedded in OI-history field "sumOpenInterestValue"
        prev_value = float(hist[-2].get("sumOpenInterestValue", 0))
        curr_value = float(hist[-1].get("sumOpenInterestValue", 0))
    except (ValueError, TypeError):
        return neutral

    if prev_oi <= 0:
        return neutral

    oi_delta = curr_oi - prev_oi
    oi_delta_pct = (oi_delta / prev_oi) if prev_oi > 0 else 0.0

    # Derive price-change from value/qty ratios (more robust than current ticker)
    prev_avg = prev_value / prev_oi if prev_oi > 0 else current_price
    curr_avg = curr_value / curr_oi if curr_oi > 0 else current_price
    price_delta_pct = (curr_avg - prev_avg) / prev_avg if prev_avg > 0 else 0.0

    THRESHOLD = 0.005  # 0.5% minimal move to count as directional
    oi_up = oi_delta_pct > THRESHOLD
    oi_dn = oi_delta_pct < -THRESHOLD
    px_up = price_delta_pct > THRESHOLD
    px_dn = price_delta_pct < -THRESHOLD

    quadrant = 0
    score = 0.0
    if oi_up and px_up:
        quadrant, score = 1, 0.3
    elif oi_up and px_dn:
        quadrant, score = 2, -0.3
    elif oi_dn and px_up:
        quadrant, score = 3, -0.5
    elif oi_dn and px_dn:
        quadrant, score = 4, 0.5

    return {
        "oi_current": curr_oi,
        "oi_4h_delta": oi_delta,
        "oi_4h_delta_pct": oi_delta_pct,
        "oi_quadrant": quadrant,
        "oi_quadrant_score": score,
    }


# ── Combined Features Helper ────────────────────────────────

# Map our 4 spot symbols to Binance perp symbols
SPOT_TO_PERP = {
    "XRP/USDT": "XRPUSDT",
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
    "SOL/USDT": "SOLUSDT",
}


def compute_futures_features(symbol: str, current_price: float) -> Dict[str, float]:
    """One-shot wrapper for all three feature groups.
    `symbol` is the spot symbol (e.g. "XRP/USDT")."""
    perp = SPOT_TO_PERP.get(symbol)
    if perp is None:
        return {}
    out = {}
    try:
        out.update(compute_funding_features(perp))
    except Exception:
        pass
    try:
        out.update(compute_oi_features(perp, current_price))
    except Exception:
        pass
    return out
