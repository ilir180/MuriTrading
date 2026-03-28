"""
MuriTrading – Phase 1: Datenabruf
Holt XRP/USDT OHLCV Daten von Binance für alle 4 Timeframes
Speichert als CSV in data/raw/
"""

import ccxt
import pandas as pd
from datetime import datetime, timedelta
import time
import os

# ── Konfiguration ────────────────────────────────────────────────
SYMBOL     = "XRP/USDT"
TIMEFRAMES = ["1d", "4h", "1h", "15m"]
YEARS_BACK = 4
OUTPUT_DIR = os.path.expanduser("~/MuriTrading/data/raw")
# ────────────────────────────────────────────────────────────────

def fetch_ohlcv(exchange, symbol, timeframe, since_ms):
    """Holt alle OHLCV Daten seit 'since_ms' in Batches."""
    all_candles = []
    print(f"  Lade {timeframe}...", end="", flush=True)

    while True:
        candles = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since_ms,
            limit=1000
        )
        if not candles:
            break

        all_candles += candles
        last_ts = candles[-1][0]

        # Fertig wenn letzter Candle nahe Jetzt
        if last_ts >= exchange.milliseconds() - 2 * timeframe_to_ms(timeframe):
            break

        since_ms = last_ts + 1
        time.sleep(0.2)  # Rate-Limit respektieren
        print(".", end="", flush=True)

    print(f" {len(all_candles)} Kerzen")
    return all_candles


def timeframe_to_ms(tf):
    """Konvertiert Timeframe-String zu Millisekunden."""
    multipliers = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
    unit = tf[-1]
    value = int(tf[:-1])
    return value * multipliers[unit] * 1000


def candles_to_df(candles):
    """Konvertiert rohe Candle-Liste zu DataFrame."""
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("datetime")
    df = df.drop(columns=["timestamp"])
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"}
    })

    since_dt = datetime.utcnow() - timedelta(days=365 * YEARS_BACK)
    since_ms  = int(since_dt.timestamp() * 1000)

    print(f"\nMuriTrading – Datenabruf")
    print(f"Asset     : {SYMBOL}")
    print(f"Von       : {since_dt.strftime('%Y-%m-%d')} bis heute")
    print(f"Timeframes: {', '.join(TIMEFRAMES)}")
    print(f"Zielordner: {OUTPUT_DIR}\n")

    for tf in TIMEFRAMES:
        candles = fetch_ohlcv(exchange, SYMBOL, tf, since_ms)
        df = candles_to_df(candles)

        filename = f"XRP_USDT_{tf}.csv"
        filepath = os.path.join(OUTPUT_DIR, filename)
        df.to_csv(filepath)

        print(f"  Gespeichert: {filename}")
        print(f"  Zeitraum  : {df.index[0].date()} → {df.index[-1].date()}")
        print(f"  Zeilen    : {len(df):,}\n")

    print("Datenabruf abgeschlossen.")
    print(f"Alle Dateien in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
