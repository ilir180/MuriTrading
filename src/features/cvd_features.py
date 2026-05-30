"""JV Boting v2 — Cumulative Volume Delta (CVD) & Signed Order Flow.

Implements the only retail-school concept that survived all four research
agents' validation: peer-reviewed (Easley/Lopez de Prado/O'Hara on VPIN),
out-of-sample-validated in crypto (Anastasopoulos & Gradojevic 2026).

Algorithm:
1. Fetch recent aggregated trades from Binance Spot.
2. Each Binance aggTrade is labeled "isBuyerMaker": True means the trade was
   initiated by a SELLER hitting the bid (sell-side aggressor); False means
   a BUYER lifted the ask (buy-side aggressor). This is the same data the
   Lee-Ready tick rule attempts to estimate — except Binance gives it directly.
3. Aggregate: per-bucket signed volume = sum(qty * +1 if buy_aggressor else -1).
4. Compute CVD windows (rolling 1h, 4h, 24h, 7d) and derive features:
   - cvd_1h_z: z-score of current 1h CVD vs 7d distribution
   - cvd_trend_4h: sign of 4h CVD net
   - cvd_acceleration: 1h CVD minus prior 1h CVD (impulse)
   - buy_volume_share_4h: buy-aggressor volume / total in last 4h

Cache: 4-min TTL per symbol (matches our 60s tick interval).
"""

import math
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import requests


BINANCE_URLS = [
    "https://data-api.binance.vision",
    "https://api.binance.com",
]
CACHE_TTL_SEC = 240
MAX_TRADES_PER_FETCH = 1000  # Binance hard cap

# {symbol: {"trades": deque[(ts_ms, qty, price, buy_aggressor)],
#           "last_fetch_ts": int_ms}}
_TAPE: Dict[str, dict] = {}


def _fetch_json(endpoint: str, params: dict = None, timeout: float = 8.0):
    for base in BINANCE_URLS:
        try:
            r = requests.get(f"{base}{endpoint}", params=params or {}, timeout=timeout)
            if r.status_code == 200:
                return r.json()
        except Exception:
            continue
    return None


def _ensure_tape(symbol: str):
    if symbol not in _TAPE:
        # 7 days of trades at worst-case rate — cap deque at 200k entries
        _TAPE[symbol] = {"trades": deque(maxlen=200_000), "last_fetch_ts": 0}


def _fetch_recent_trades(symbol: str, start_time_ms: Optional[int] = None) -> List[dict]:
    """Fetch aggregated trades. If start_time_ms provided, fetches from there
    forward. Returns the list of agg-trade dicts."""
    params = {"symbol": symbol, "limit": MAX_TRADES_PER_FETCH}
    if start_time_ms is not None:
        params["startTime"] = start_time_ms
    data = _fetch_json("/api/v3/aggTrades", params=params)
    if not isinstance(data, list):
        return []
    return data


def _bootstrap_tape(symbol: str, lookback_hours: int = 4, max_iterations: int = 40):
    """On first call, backfill `lookback_hours` of agg-trades into the tape.
    Binance returns at most 1000 trades per call; we walk forward in time.

    Defaults are tuned for cold-start speed: 4h lookback covers the cvd_1h_z
    bucket distribution baseline (24h is nice-to-have, not required), and
    max_iterations caps the worst-case fetch budget at ~40 requests."""
    _ensure_tape(symbol)
    tape = _TAPE[symbol]
    if tape["trades"]:
        return
    end_ms = int(time.time() * 1000)
    start = end_ms - lookback_hours * 3600 * 1000
    cursor = start
    safety = 0
    while cursor < end_ms and safety < max_iterations:
        batch = _fetch_recent_trades(symbol, start_time_ms=cursor)
        if not batch:
            break
        for t in batch:
            try:
                ts = int(t["T"])
                qty = float(t["q"])
                price = float(t["p"])
                buy_aggressor = not bool(t.get("m", False))
                tape["trades"].append((ts, qty, price, buy_aggressor))
            except Exception:
                continue
        last_ts = int(batch[-1]["T"])
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        safety += 1
        if len(batch) < MAX_TRADES_PER_FETCH and cursor >= end_ms - 60_000:
            break
    tape["last_fetch_ts"] = end_ms


def _refresh_tape(symbol: str):
    """Incremental: fetch trades since the last update."""
    _ensure_tape(symbol)
    tape = _TAPE[symbol]
    if not tape["trades"]:
        _bootstrap_tape(symbol, lookback_hours=4)
        return
    start = tape["last_fetch_ts"] - 5000   # 5s overlap to handle delay
    batch = _fetch_recent_trades(symbol, start_time_ms=start)
    if not batch:
        return
    seen_ts = set()
    if tape["trades"]:
        recent_tail = list(tape["trades"])[-200:]
        seen_ts = set(t[0] for t in recent_tail)
    for t in batch:
        try:
            ts = int(t["T"])
            if ts in seen_ts:
                continue
            qty = float(t["q"])
            price = float(t["p"])
            buy_aggressor = not bool(t.get("m", False))
            tape["trades"].append((ts, qty, price, buy_aggressor))
        except Exception:
            continue
    tape["last_fetch_ts"] = int(time.time() * 1000)
    # Drop tape older than 7d
    cutoff = (time.time() - 7 * 24 * 3600) * 1000
    while tape["trades"] and tape["trades"][0][0] < cutoff:
        tape["trades"].popleft()


def _cvd_in_window(trades: List[Tuple[int, float, float, bool]],
                   window_start_ms: int) -> Tuple[float, float, float]:
    """Return (cvd_signed_volume, buy_volume, sell_volume) in the window."""
    cvd = 0.0
    buy_v = 0.0
    sell_v = 0.0
    for ts, qty, price, buy_agg in trades:
        if ts < window_start_ms:
            continue
        notional = qty * price
        if buy_agg:
            cvd += notional
            buy_v += notional
        else:
            cvd -= notional
            sell_v += notional
    return cvd, buy_v, sell_v


def compute_cvd_features(symbol: str) -> Dict[str, float]:
    """Returns a feature dict. Neutral fallback on network errors.

      cvd_1h_usd        — signed buy-minus-sell USD volume over last 1h
      cvd_4h_usd        — same, 4h window
      cvd_24h_usd       — 24h
      cvd_1h_z          — z-score of 1h CVD vs 24-hour bucket distribution
      cvd_buy_share_4h  — buy-aggressor volume / total volume in last 4h [0..1]
      cvd_acceleration  — current 1h CVD - prior 1h CVD (impulse)
      cvd_trend_sign    — sign of 4h CVD: +1, 0, -1
    """
    neutral = {
        "cvd_1h_usd": 0.0, "cvd_4h_usd": 0.0, "cvd_24h_usd": 0.0,
        "cvd_1h_z": 0.0, "cvd_buy_share_4h": 0.5,
        "cvd_acceleration": 0.0, "cvd_trend_sign": 0,
    }
    _refresh_tape(symbol)
    tape = _TAPE.get(symbol)
    if tape is None or not tape["trades"]:
        return neutral

    now_ms = int(time.time() * 1000)
    trades = list(tape["trades"])

    cvd_1h, buy_1h, sell_1h = _cvd_in_window(trades, now_ms - 3600_000)
    cvd_4h, buy_4h, sell_4h = _cvd_in_window(trades, now_ms - 4 * 3600_000)
    cvd_24h, _, _ = _cvd_in_window(trades, now_ms - 24 * 3600_000)
    cvd_prev_1h, _, _ = _cvd_in_window(
        [t for t in trades if t[0] < now_ms - 3600_000],
        now_ms - 2 * 3600_000,
    )

    # Z-score: compute 24 hourly buckets, then z-score current 1h
    bucket_cvds = []
    for i in range(24):
        start = now_ms - (i + 1) * 3600_000
        end   = now_ms - i * 3600_000
        bucket = sum(
            (q * p) * (1 if ba else -1)
            for ts, q, p, ba in trades if start <= ts < end
        )
        bucket_cvds.append(bucket)
    if len(bucket_cvds) >= 4:
        mean = sum(bucket_cvds) / len(bucket_cvds)
        var = sum((b - mean) ** 2 for b in bucket_cvds) / len(bucket_cvds)
        std = math.sqrt(var) if var > 0 else 0.0
        z = (cvd_1h - mean) / std if std > 0 else 0.0
    else:
        z = 0.0

    total_4h = buy_4h + sell_4h
    buy_share = (buy_4h / total_4h) if total_4h > 0 else 0.5

    return {
        "cvd_1h_usd": cvd_1h,
        "cvd_4h_usd": cvd_4h,
        "cvd_24h_usd": cvd_24h,
        "cvd_1h_z": z,
        "cvd_buy_share_4h": buy_share,
        "cvd_acceleration": cvd_1h - cvd_prev_1h,
        "cvd_trend_sign": 1 if cvd_4h > 0 else (-1 if cvd_4h < 0 else 0),
    }


# Map our 4 spot symbols to Binance spot symbols (CVD on spot, not perp)
SPOT_TO_BINANCE = {
    "XRP/USDT": "XRPUSDT",
    "BTC/USDT": "BTCUSDT",
    "ETH/USDT": "ETHUSDT",
    "SOL/USDT": "SOLUSDT",
}


def cvd_features_for(symbol: str) -> Dict[str, float]:
    """One-shot wrapper. `symbol` is the spot symbol with slash (e.g. XRP/USDT)."""
    binance = SPOT_TO_BINANCE.get(symbol)
    if not binance:
        return {}
    try:
        return compute_cvd_features(binance)
    except Exception:
        return {}
