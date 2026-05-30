"""JV Boting v2 — Liquidation Stream (Binance Perp forceOrders).

Reads the public `@forceOrder` WebSocket per symbol and maintains a rolling
15-min buffer of liquidation volume per side. Exposes a snapshot feature
dict that the bots can poll.

WebSocket details:
  wss://fstream.binance.com/stream?streams=<symbol>@forceOrder

Per-event payload:
  {
    "e": "forceOrder",
    "o": {
      "s": "XRPUSDT", "S": "SELL"  (SELL = long was liquidated; BUY = short),
      "q": "12345",  (quantity in base asset)
      "p": "1.39",   (avg fill price)
      ...
    }
  }

Buffer: deque of (timestamp_ms, side, qty, price). 15-min rolling window.
Auto-reconnect with exponential backoff. Daemon thread (won't block shutdown).
"""

import json
import math
import time
import threading
from collections import deque
from typing import Dict, Optional

try:
    import websocket  # websocket-client
    HAVE_WS = True
except ImportError:
    HAVE_WS = False


# Per-symbol buffer: {perp_symbol: deque[(ts_ms, side, qty, price)]}
_BUFFERS: Dict[str, deque] = {}
_LOCKS: Dict[str, threading.Lock] = {}
_THREADS: Dict[str, threading.Thread] = {}
_STOP_FLAGS: Dict[str, threading.Event] = {}

BUFFER_WINDOW_SEC = 15 * 60   # 15 minutes
MAX_BUFFER_LEN    = 5000      # safety cap per symbol

# Trigger thresholds (USD notional in 15 min) — tunable per symbol
LIQUIDATION_THRESHOLDS = {
    "XRPUSDT": 200_000,
    "BTCUSDT": 5_000_000,
    "ETHUSDT": 1_000_000,
    "SOLUSDT": 300_000,
}


def _ensure_buffer(symbol: str):
    if symbol not in _BUFFERS:
        _BUFFERS[symbol] = deque(maxlen=MAX_BUFFER_LEN)
        _LOCKS[symbol] = threading.Lock()
        _STOP_FLAGS[symbol] = threading.Event()


def _on_message(symbol: str, msg: str):
    try:
        data = json.loads(msg)
        # The combined stream returns {"stream": ..., "data": {...}}
        o = data.get("data", {}).get("o") or data.get("o")
        if not o:
            return
        side = o.get("S", "")
        qty = float(o.get("q", 0) or 0)
        price = float(o.get("p", 0) or 0)
        ts_ms = int(o.get("T") or time.time() * 1000)
        if qty <= 0 or price <= 0:
            return
        with _LOCKS[symbol]:
            _BUFFERS[symbol].append((ts_ms, side, qty, price))
    except Exception:
        return


def _run_ws_for_symbol(symbol: str):
    """Daemon thread: open WS, listen, reconnect with backoff on disconnect."""
    if not HAVE_WS:
        return
    url = f"wss://fstream.binance.com/stream?streams={symbol.lower()}@forceOrder"
    backoff = 1.0
    stop = _STOP_FLAGS[symbol]
    while not stop.is_set():
        try:
            ws = websocket.create_connection(url, timeout=15)
            ws.settimeout(60)
            backoff = 1.0
            while not stop.is_set():
                try:
                    msg = ws.recv()
                    if msg:
                        _on_message(symbol, msg)
                except websocket.WebSocketTimeoutException:
                    # Send a ping-like idle
                    try:
                        ws.ping()
                    except Exception:
                        break
                except websocket.WebSocketException:
                    break
            try:
                ws.close()
            except Exception:
                pass
        except Exception:
            pass
        if stop.is_set():
            break
        time.sleep(backoff)
        backoff = min(backoff * 2, 60.0)


def start_stream(symbol: str):
    """Idempotent: start a daemon WS reader for `symbol` (Binance perp symbol
    like XRPUSDT). Safe to call multiple times — only one thread per symbol."""
    if not HAVE_WS:
        return False
    _ensure_buffer(symbol)
    existing = _THREADS.get(symbol)
    if existing is not None and existing.is_alive():
        return True
    t = threading.Thread(target=_run_ws_for_symbol, args=(symbol,),
                         daemon=True, name=f"liq_ws_{symbol}")
    _THREADS[symbol] = t
    t.start()
    return True


def stop_stream(symbol: str):
    if symbol in _STOP_FLAGS:
        _STOP_FLAGS[symbol].set()


def _prune(symbol: str):
    """Drop entries older than the buffer window."""
    if symbol not in _BUFFERS:
        return
    cutoff = (time.time() - BUFFER_WINDOW_SEC) * 1000
    with _LOCKS[symbol]:
        b = _BUFFERS[symbol]
        while b and b[0][0] < cutoff:
            b.popleft()


def compute_liquidation_features(perp_symbol: str) -> Dict[str, float]:
    """Returns a snapshot of liquidation activity over the last 15 minutes.

      liq_volume_15m_usd       — total notional liquidated in last 15 min
      liq_long_volume_15m_usd  — long-side (SELL events) notional
      liq_short_volume_15m_usd — short-side (BUY events) notional
      liq_imbalance            — (long - short) / (long + short) in [-1, 1]
      liq_event_count_15m      — number of liquidation events
      liq_capitulation_flag    — 1 if total notional > threshold, else 0
      liq_long_capit_flag      — 1 if long-side > 0.6 × threshold (mean-rev-long trigger)
      liq_short_capit_flag     — 1 if short-side > 0.6 × threshold (mean-rev-short trigger)
    """
    neutral = {
        "liq_volume_15m_usd": 0.0,
        "liq_long_volume_15m_usd": 0.0,
        "liq_short_volume_15m_usd": 0.0,
        "liq_imbalance": 0.0,
        "liq_event_count_15m": 0,
        "liq_capitulation_flag": 0,
        "liq_long_capit_flag": 0,
        "liq_short_capit_flag": 0,
    }
    if perp_symbol not in _BUFFERS:
        return neutral
    _prune(perp_symbol)
    with _LOCKS[perp_symbol]:
        snapshot = list(_BUFFERS[perp_symbol])

    long_usd = 0.0
    short_usd = 0.0
    for ts, side, qty, price in snapshot:
        notional = qty * price
        if side == "SELL":  # long liquidated
            long_usd += notional
        elif side == "BUY":  # short liquidated
            short_usd += notional

    total = long_usd + short_usd
    threshold = LIQUIDATION_THRESHOLDS.get(perp_symbol, 500_000)
    return {
        "liq_volume_15m_usd": total,
        "liq_long_volume_15m_usd": long_usd,
        "liq_short_volume_15m_usd": short_usd,
        "liq_imbalance": ((long_usd - short_usd) / total) if total > 0 else 0.0,
        "liq_event_count_15m": len(snapshot),
        "liq_capitulation_flag": 1 if total > threshold else 0,
        "liq_long_capit_flag":  1 if long_usd  > 0.6 * threshold else 0,
        "liq_short_capit_flag": 1 if short_usd > 0.6 * threshold else 0,
    }


def start_all(perp_symbols=None):
    """Convenience: start streams for the default 4 symbols (or a custom list)."""
    if not HAVE_WS:
        return False
    if perp_symbols is None:
        perp_symbols = ["XRPUSDT", "BTCUSDT", "ETHUSDT", "SOLUSDT"]
    started = 0
    for sym in perp_symbols:
        if start_stream(sym):
            started += 1
    return started
