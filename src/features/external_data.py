"""
MuriTrading – External Data Sources
Holt Sentiment- und On-Chain-Daten als zusätzliche Features.
Quellen:
  - Alternative.me Fear & Greed Index (kein Key nötig)
  - XRPScan On-Chain Metrics (kein Key nötig)
  - CoinGecko Market Data (kein Key nötig)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import time


# ═══════════════════════════════════════════════════════════════
#  FEAR & GREED INDEX
# ═══════════════════════════════════════════════════════════════

def fetch_fear_greed(days=365):
    """Holt Fear & Greed Index (0=Extreme Fear, 100=Extreme Greed)."""
    url = f"https://api.alternative.me/fng/?limit={days}&format=json"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()["data"]

    rows = []
    for d in data:
        rows.append({
            "datetime": pd.Timestamp(int(d["timestamp"]), unit="s", tz="UTC"),
            "fear_greed": int(d["value"]),
            "fear_greed_class": d["value_classification"],
        })

    df = pd.DataFrame(rows).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Abgeleitete Features
    df["fg_ma7"] = df["fear_greed"].rolling(7).mean()
    df["fg_ma30"] = df["fear_greed"].rolling(30).mean()
    df["fg_change"] = df["fear_greed"].diff()
    df["fg_extreme_fear"] = (df["fear_greed"] <= 20).astype(int)
    df["fg_extreme_greed"] = (df["fear_greed"] >= 80).astype(int)
    df["fg_trend"] = df["fear_greed"] - df["fg_ma7"]  # Über/unter kurzfristigem Schnitt

    return df


# ═══════════════════════════════════════════════════════════════
#  COINGECKO - XRP MARKET DATA
# ═══════════════════════════════════════════════════════════════

def fetch_xrp_market_data(days=365):
    """Holt XRP Marktdaten von CoinGecko (Market Cap, Volume, etc.)."""
    url = "https://api.coingecko.com/api/v3/coins/ripple/market_chart"
    # CoinGecko free tier limitiert auf 365 Tage
    chunk = min(days, 365)
    params = {"vs_currency": "usd", "days": chunk, "interval": "daily"}

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    # Preise
    df_price = pd.DataFrame(data["prices"], columns=["timestamp", "cg_price"])
    df_price["datetime"] = pd.to_datetime(df_price["timestamp"], unit="ms", utc=True)
    df_price = df_price.set_index("datetime").drop(columns=["timestamp"])

    # Market Cap
    df_mcap = pd.DataFrame(data["market_caps"], columns=["timestamp", "cg_market_cap"])
    df_mcap["datetime"] = pd.to_datetime(df_mcap["timestamp"], unit="ms", utc=True)
    df_mcap = df_mcap.set_index("datetime").drop(columns=["timestamp"])

    # Volume
    df_vol = pd.DataFrame(data["total_volumes"], columns=["timestamp", "cg_volume"])
    df_vol["datetime"] = pd.to_datetime(df_vol["timestamp"], unit="ms", utc=True)
    df_vol = df_vol.set_index("datetime").drop(columns=["timestamp"])

    df = df_price.join(df_mcap, how="outer").join(df_vol, how="outer")
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # Abgeleitete Features
    df["cg_vol_ma7"] = df["cg_volume"].rolling(7).mean()
    df["cg_vol_ratio"] = df["cg_volume"] / (df["cg_vol_ma7"] + 1e-10)
    df["cg_mcap_change"] = df["cg_market_cap"].pct_change()
    df["cg_vol_spike"] = (df["cg_vol_ratio"] > 2.0).astype(int)

    return df


# ═══════════════════════════════════════════════════════════════
#  BTC DOMINANCE & CORRELATION
# ═══════════════════════════════════════════════════════════════

def fetch_btc_data(days=365):
    """Holt BTC Daten für Korrelations-Features."""
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    chunk = min(days, 365)
    params = {"vs_currency": "usd", "days": chunk, "interval": "daily"}

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    df_price = pd.DataFrame(data["prices"], columns=["timestamp", "btc_price"])
    df_price["datetime"] = pd.to_datetime(df_price["timestamp"], unit="ms", utc=True)
    df_price = df_price.set_index("datetime").drop(columns=["timestamp"])

    df_price = df_price.sort_index()
    df_price = df_price[~df_price.index.duplicated(keep="last")]

    # BTC Returns
    df_price["btc_return_1d"] = df_price["btc_price"].pct_change()
    df_price["btc_return_7d"] = df_price["btc_price"].pct_change(7)
    df_price["btc_ma7"] = df_price["btc_price"].rolling(7).mean()
    df_price["btc_above_ma7"] = (df_price["btc_price"] > df_price["btc_ma7"]).astype(int)

    return df_price


def fetch_global_data():
    """Holt globale Krypto-Metriken (BTC Dominance etc.)."""
    url = "https://api.coingecko.com/api/v3/global"
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()["data"]

    return {
        "btc_dominance": data["market_cap_percentage"].get("btc", 0),
        "eth_dominance": data["market_cap_percentage"].get("eth", 0),
        "total_market_cap_usd": data["total_market_cap"].get("usd", 0),
        "total_volume_usd": data["total_volume"].get("usd", 0),
        "active_cryptocurrencies": data.get("active_cryptocurrencies", 0),
    }


# ═══════════════════════════════════════════════════════════════
#  MERGE ALL EXTERNAL DATA
# ═══════════════════════════════════════════════════════════════

def fetch_all_external(days=365):
    """
    Holt alle externen Datenquellen und merged sie auf Daily-Basis.
    Gibt DataFrame mit allen externen Features zurück.
    """
    print("  Lade Fear & Greed Index...", end="", flush=True)
    df_fg = fetch_fear_greed(days)
    print(f" {len(df_fg)} Tage")

    print("  Lade XRP Market Data...", end="", flush=True)
    time.sleep(1.5)  # CoinGecko Rate Limit
    df_xrp = fetch_xrp_market_data(days)
    print(f" {len(df_xrp)} Tage")

    print("  Lade BTC Korrelationsdaten...", end="", flush=True)
    time.sleep(1.5)
    df_btc = fetch_btc_data(days)
    print(f" {len(df_btc)} Tage")

    # Alle auf Daily-Index normalisieren
    df_fg.index = df_fg.index.normalize()
    df_xrp.index = df_xrp.index.normalize()
    df_btc.index = df_btc.index.normalize()

    # Merge
    df = df_fg.join(df_xrp, how="outer").join(df_btc, how="outer")
    df = df.sort_index().ffill()

    # Korrelation XRP/BTC (rolling 14 Tage)
    if "cg_price" in df.columns and "btc_price" in df.columns:
        xrp_ret = df["cg_price"].pct_change()
        btc_ret = df["btc_price"].pct_change()
        df["xrp_btc_corr_14d"] = xrp_ret.rolling(14).corr(btc_ret)

    # Nicht-numerische Spalten droppen
    df = df.drop(columns=["fear_greed_class"], errors="ignore")

    print(f"  External Features: {len(df.columns)} Spalten, {len(df)} Tage")
    return df


# ═══════════════════════════════════════════════════════════════
#  LIVE FETCH (für Dashboard/Bot - aktuellste Werte)
# ═══════════════════════════════════════════════════════════════

def fetch_live_external():
    """Holt aktuelle externe Daten als Dict (für live Prediction)."""
    features = {}

    try:
        # Fear & Greed (letzte 30 Tage für MA-Berechnung)
        fg = fetch_fear_greed(30)
        if not fg.empty:
            latest = fg.iloc[-1]
            features["fear_greed"] = latest["fear_greed"]
            features["fg_ma7"] = latest.get("fg_ma7", latest["fear_greed"])
            features["fg_ma30"] = latest.get("fg_ma30", latest["fear_greed"])
            features["fg_change"] = latest.get("fg_change", 0)
            features["fg_extreme_fear"] = latest.get("fg_extreme_fear", 0)
            features["fg_extreme_greed"] = latest.get("fg_extreme_greed", 0)
            features["fg_trend"] = latest.get("fg_trend", 0)
    except Exception as e:
        print(f"  Fear&Greed Fehler: {e}")

    try:
        time.sleep(1.5)
        # BTC Daten
        btc = fetch_btc_data(30)
        if not btc.empty:
            latest = btc.iloc[-1]
            features["btc_return_1d"] = latest.get("btc_return_1d", 0)
            features["btc_return_7d"] = latest.get("btc_return_7d", 0)
            features["btc_above_ma7"] = latest.get("btc_above_ma7", 0)
    except Exception as e:
        print(f"  BTC Fehler: {e}")

    try:
        time.sleep(1.5)
        # Global
        gd = fetch_global_data()
        features["btc_dominance"] = gd["btc_dominance"]
    except Exception as e:
        print(f"  Global Fehler: {e}")

    return features


if __name__ == "__main__":
    print("MuriTrading – External Data Test")
    print("=" * 40)
    df = fetch_all_external(days=30)
    print(f"\nSpalten: {list(df.columns)}")
    print(f"\nLetzte Zeile:")
    print(df.iloc[-1].to_string())
