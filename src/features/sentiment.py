"""
MuriTrading – Crypto Sentiment Features
Echtzeit-Sentiment als Trading-Signal aus mehreren Quellen.

Quellen:
  - Fear & Greed Index (alternative.me) — Markt-Stimmung 0-100
  - CoinGecko Community Sentiment — XRP-spezifische Up/Down Votes
  - CryptoPanic News Sentiment — Bullish/Bearish Nachrichtenratio (optional, braucht API-Key)

Alle Quellen sind gratis und brauchen keine Authentifizierung (ausser CryptoPanic).
"""

import time
import math
import requests

# ── Konfiguration ─────────────────────────────────────────────
CRYPTOPANIC_TOKEN = None  # Optional: registriere auf cryptopanic.com/developers/api/keys
CACHE_TTL_FNG = 3600      # Fear & Greed: cache 1h (updatet nur 1x/Tag)
CACHE_TTL_CG = 300        # CoinGecko: cache 5 min
CACHE_TTL_CP = 120        # CryptoPanic: cache 2 min

# ── Cache ─────────────────────────────────────────────────────
_cache = {}


def _cached_fetch(key, ttl, fetch_fn):
    """Einfacher TTL-Cache für API-Responses."""
    now = time.time()
    if key in _cache and (now - _cache[key]["ts"]) < ttl:
        return _cache[key]["data"]
    try:
        data = fetch_fn()
        _cache[key] = {"data": data, "ts": now}
        return data
    except Exception:
        # Bei Fehler: alten Cache zurückgeben falls vorhanden
        if key in _cache:
            return _cache[key]["data"]
        return None


# ── Fear & Greed Index ────────────────────────────────────────

def _fetch_fear_greed():
    """Holt den Crypto Fear & Greed Index (0=Extreme Fear, 100=Extreme Greed)."""
    r = requests.get("https://api.alternative.me/fng/?limit=2", timeout=10)
    if r.status_code != 200:
        return None
    data = r.json().get("data", [])
    if not data:
        return None

    current = data[0]
    previous = data[1] if len(data) > 1 else current

    return {
        "value": int(current["value"]),
        "classification": current["value_classification"],
        "previous": int(previous["value"]),
    }


# ── CoinGecko Community Sentiment ────────────────────────────

def _fetch_coingecko_sentiment():
    """Holt XRP Community Sentiment von CoinGecko."""
    r = requests.get(
        "https://api.coingecko.com/api/v3/coins/ripple",
        params={"localization": "false", "tickers": "false",
                "market_data": "false", "community_data": "true",
                "developer_data": "false"},
        timeout=10,
    )
    if r.status_code != 200:
        return None
    data = r.json()
    up = data.get("sentiment_votes_up_percentage", 50)
    down = data.get("sentiment_votes_down_percentage", 50)
    return {"up_pct": up or 50, "down_pct": down or 50}


# ── CryptoPanic News Sentiment ───────────────────────────────

def _fetch_cryptopanic():
    """Holt XRP News-Sentiment von CryptoPanic (braucht API-Key)."""
    if not CRYPTOPANIC_TOKEN:
        return None
    try:
        # Bullish news count
        r_bull = requests.get(
            "https://cryptopanic.com/api/v1/posts/",
            params={"auth_token": CRYPTOPANIC_TOKEN, "currencies": "XRP",
                    "filter": "bullish", "public": "true"},
            timeout=10,
        )
        # Bearish news count
        r_bear = requests.get(
            "https://cryptopanic.com/api/v1/posts/",
            params={"auth_token": CRYPTOPANIC_TOKEN, "currencies": "XRP",
                    "filter": "bearish", "public": "true"},
            timeout=10,
        )
        if r_bull.status_code != 200 or r_bear.status_code != 200:
            return None

        bull_count = len(r_bull.json().get("results", []))
        bear_count = len(r_bear.json().get("results", []))
        total = bull_count + bear_count
        ratio = bull_count / total if total > 0 else 0.5

        return {"bullish": bull_count, "bearish": bear_count, "ratio": ratio}
    except Exception:
        return None


# ── Hauptfunktion ─────────────────────────────────────────────

def compute_sentiment_features():
    """
    Berechnet alle Sentiment-Features. Gibt ein Dict zurück.
    Bei Fehler: NaN-Werte.
    """
    nan = float("nan")
    result = {
        "sent_fear_greed": nan,
        "sent_fear_greed_prev": nan,
        "sent_fear_greed_delta": nan,
        "sent_fear_greed_extreme": 0,
        "sent_cg_bullish_pct": nan,
        "sent_cg_sentiment_score": nan,
        "sent_cp_news_ratio": nan,
        "sent_composite": nan,
    }

    scores = []

    # 1. Fear & Greed Index
    fng = _cached_fetch("fng", CACHE_TTL_FNG, _fetch_fear_greed)
    if fng:
        val = fng["value"]
        prev = fng["previous"]
        result["sent_fear_greed"] = val
        result["sent_fear_greed_prev"] = prev
        result["sent_fear_greed_delta"] = val - prev

        # Extreme Fear (<20) oder Extreme Greed (>80)
        if val <= 20:
            result["sent_fear_greed_extreme"] = -1  # Extreme Fear (contrarian: bullish)
        elif val >= 80:
            result["sent_fear_greed_extreme"] = 1   # Extreme Greed (contrarian: bearish)

        # Normalisiert 0-1 für Composite
        scores.append(val / 100.0)

    # 2. CoinGecko Sentiment
    cg = _cached_fetch("coingecko", CACHE_TTL_CG, _fetch_coingecko_sentiment)
    if cg:
        result["sent_cg_bullish_pct"] = cg["up_pct"]
        # Score: 0.5 = neutral, >0.5 = bullish, <0.5 = bearish
        score = cg["up_pct"] / 100.0
        result["sent_cg_sentiment_score"] = score
        scores.append(score)

    # 3. CryptoPanic (optional)
    cp = _cached_fetch("cryptopanic", CACHE_TTL_CP, _fetch_cryptopanic)
    if cp:
        result["sent_cp_news_ratio"] = cp["ratio"]
        scores.append(cp["ratio"])

    # Composite Score (Durchschnitt aller verfügbaren Quellen)
    if scores:
        result["sent_composite"] = sum(scores) / len(scores)

    return result


def sentiment_signal_text(s):
    """Formatiert Sentiment als kurzen Text für Logs/Telegram."""
    fng = s.get("sent_fear_greed", float("nan"))
    if math.isnan(fng):
        return "Sentiment: n/a"

    # Fear & Greed Label
    if fng <= 20:
        label = "EXTREME FEAR"
    elif fng <= 40:
        label = "FEAR"
    elif fng <= 60:
        label = "NEUTRAL"
    elif fng <= 80:
        label = "GREED"
    else:
        label = "EXTREME GREED"

    parts = [f"Sentiment: {label} ({fng:.0f})"]

    delta = s.get("sent_fear_greed_delta", 0)
    if not math.isnan(delta) and abs(delta) >= 3:
        parts.append(f"Δ{delta:+.0f}")

    composite = s.get("sent_composite", float("nan"))
    if not math.isnan(composite):
        direction = "↑" if composite > 0.55 else "↓" if composite < 0.45 else "→"
        parts.append(f"Score:{composite:.0%}{direction}")

    return "  ".join(parts)
