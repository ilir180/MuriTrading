"""
MuriTrading – Phase 2: Feature Engineering
Berechnet Indikatoren, MTF Confluence und Labels (Min/Med/Max)
Input:  data/raw/XRP_USDT_*.csv
Output: data/processed/features_1h.csv  (Basis für ML-Modell)
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import timezone

DATA_DIR   = os.path.expanduser("~/MuriTrading/data/raw")
OUTPUT_DIR = os.path.expanduser("~/MuriTrading/data/processed")

# Prediction-Horizont: wie viele 1h-Kerzen voraus?
HORIZON_CANDLES = 2   # = ~2 Stunden

# Label-Perzentile für Min/Med/Max
LABEL_MIN = 0.10
LABEL_MED = 0.50
LABEL_MAX = 0.90


# ── 1. Laden ────────────────────────────────────────────────────

def load_csv(timeframe):
    path = os.path.join(DATA_DIR, f"XRP_USDT_{timeframe}.csv")
    df = pd.read_csv(path, index_col="datetime", parse_dates=True)
    df.index = df.index.tz_localize("UTC")
    return df


# ── 2. Indikatoren ──────────────────────────────────────────────

def add_indicators(df, prefix=""):
    """Berechnet alle technischen Indikatoren manuell – kein ta-lib nötig."""
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]
    p = prefix

    # EMA
    df[f"{p}ema_9"]   = c.ewm(span=9,   adjust=False).mean()
    df[f"{p}ema_21"]  = c.ewm(span=21,  adjust=False).mean()
    df[f"{p}ema_50"]  = c.ewm(span=50,  adjust=False).mean()
    df[f"{p}ema_200"] = c.ewm(span=200, adjust=False).mean()

    # EMA Abstände (normalisiert)
    df[f"{p}ema_9_dist"]  = (c - df[f"{p}ema_9"])  / c
    df[f"{p}ema_21_dist"] = (c - df[f"{p}ema_21"]) / c
    df[f"{p}ema_50_dist"] = (c - df[f"{p}ema_50"]) / c

    # EMA Kreuzungen
    df[f"{p}ema_9_above_21"]  = (df[f"{p}ema_9"]  > df[f"{p}ema_21"]).astype(int)
    df[f"{p}ema_21_above_50"] = (df[f"{p}ema_21"] > df[f"{p}ema_50"]).astype(int)
    df[f"{p}ema_50_above_200"]= (df[f"{p}ema_50"] > df[f"{p}ema_200"]).astype(int)

    # RSI (14)
    delta = c.diff()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df[f"{p}rsi_14"] = 100 - (100 / (1 + rs))
    df[f"{p}rsi_oversold"]  = (df[f"{p}rsi_14"] < 30).astype(int)
    df[f"{p}rsi_overbought"]= (df[f"{p}rsi_14"] > 70).astype(int)

    # MACD (12/26/9)
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df[f"{p}macd_line"]   = ema12 - ema26
    df[f"{p}macd_signal"] = df[f"{p}macd_line"].ewm(span=9, adjust=False).mean()
    df[f"{p}macd_hist"]   = df[f"{p}macd_line"] - df[f"{p}macd_signal"]
    df[f"{p}macd_above"]  = (df[f"{p}macd_line"] > df[f"{p}macd_signal"]).astype(int)

    # Bollinger Bands (20, 2σ)
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    df[f"{p}bb_upper"] = sma20 + 2 * std20
    df[f"{p}bb_lower"] = sma20 - 2 * std20
    df[f"{p}bb_width"] = (df[f"{p}bb_upper"] - df[f"{p}bb_lower"]) / sma20
    df[f"{p}bb_pos"]   = (c - df[f"{p}bb_lower"]) / (df[f"{p}bb_upper"] - df[f"{p}bb_lower"] + 1e-10)

    # ATR (14)
    tr = pd.concat([
        h - l,
        (h - c.shift()).abs(),
        (l - c.shift()).abs()
    ], axis=1).max(axis=1)
    df[f"{p}atr_14"]      = tr.ewm(alpha=1/14, adjust=False).mean()
    df[f"{p}atr_rel"]     = df[f"{p}atr_14"] / c   # normalisiert

    # Stochastic RSI (14)
    rsi = df[f"{p}rsi_14"]
    rsi_min = rsi.rolling(14).min()
    rsi_max = rsi.rolling(14).max()
    df[f"{p}stoch_rsi"] = (rsi - rsi_min) / (rsi_max - rsi_min + 1e-10)

    # Volumen
    vol_ma20 = v.rolling(20).mean()
    df[f"{p}vol_ratio"]   = v / (vol_ma20 + 1e-10)
    df[f"{p}vol_spike"]   = (df[f"{p}vol_ratio"] > 2.0).astype(int)

    # OBV
    obv = (np.sign(c.diff()) * v).fillna(0).cumsum()
    df[f"{p}obv_norm"] = (obv - obv.rolling(20).mean()) / (obv.rolling(20).std() + 1e-10)

    # Preis-Momentum
    df[f"{p}return_1"]  = c.pct_change(1)
    df[f"{p}return_3"]  = c.pct_change(3)
    df[f"{p}return_6"]  = c.pct_change(6)
    df[f"{p}return_12"] = c.pct_change(12)

    # Candlestick-Eigenschaften
    df[f"{p}candle_body"]  = (c - df["open"]).abs() / (h - l + 1e-10)
    df[f"{p}candle_bull"]  = (c > df["open"]).astype(int)
    df[f"{p}upper_wick"]   = (h - c.clip(lower=df["open"])) / (h - l + 1e-10)
    df[f"{p}lower_wick"]   = (c.clip(upper=df["open"]) - l) / (h - l + 1e-10)

    return df


# ── 3. MTF Confluence ───────────────────────────────────────────

def merge_higher_timeframes(df_1h, df_4h, df_1d):
    """
    Merged 4H und Daily Features auf den 1H-Index.
    Jede 1H-Kerze bekommt den Wert der aktuell laufenden 4H/Daily-Kerze.
    """
    # 4H Features auf 1H mergen (forward-fill = keine Look-ahead Bias)
    df_4h_sel = df_4h[[c for c in df_4h.columns if c.startswith("4h_")]].copy()
    df_1h = pd.merge_asof(
        df_1h.sort_index(),
        df_4h_sel.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward"
    )

    # Daily Features auf 1H mergen
    df_1d_sel = df_1d[[c for c in df_1d.columns if c.startswith("1d_")]].copy()
    df_1h = pd.merge_asof(
        df_1h.sort_index(),
        df_1d_sel.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward"
    )

    return df_1h


def add_confluence_score(df):
    """
    Berechnet einen Confluence-Score: wie viele Timeframes stimmen überein?
    +1 für bullisches Signal, -1 für bärisches Signal pro Indikator/TF.
    """
    bull_signals = [
        "1h_ema_9_above_21", "1h_ema_21_above_50", "1h_macd_above",
        "4h_ema_9_above_21", "4h_ema_21_above_50", "4h_macd_above",
        "1d_ema_9_above_21", "1d_ema_21_above_50", "1d_macd_above",
    ]
    bear_signals = [
        "1h_rsi_overbought", "4h_rsi_overbought", "1d_rsi_overbought",
    ]

    available_bull = [s for s in bull_signals if s in df.columns]
    available_bear = [s for s in bear_signals if s in df.columns]

    if available_bull:
        df["confluence_bull"] = df[available_bull].sum(axis=1)
    if available_bear:
        df["confluence_bear"] = df[available_bear].sum(axis=1)

    df["confluence_net"] = df.get("confluence_bull", 0) - df.get("confluence_bear", 0)

    return df


# ── 4. Labels (Min / Med / Max) ─────────────────────────────────

def add_labels(df, horizon=HORIZON_CANDLES):
    """
    Für jede 1H-Kerze: was passiert in den nächsten `horizon` Kerzen?
    label_min: 10. Perzentil der künftigen Highs/Lows  → Worst case
    label_med: 50. Perzentil                            → Erwarteter Wert
    label_max: 90. Perzentil                            → Best case
    label_dir: Richtung (1=bullisch, 0=bärisch)
    """
    future_returns = []

    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values

    for i in range(len(df)):
        end = min(i + horizon + 1, len(df))
        future_c = closes[i+1:end]
        future_h = highs[i+1:end]
        future_l = lows[i+1:end]

        if len(future_c) == 0:
            future_returns.append((np.nan, np.nan, np.nan, np.nan))
            continue

        current = closes[i]
        all_prices = np.concatenate([future_h, future_l, future_c])
        returns = (all_prices - current) / current

        future_returns.append((
            float(np.percentile(returns, LABEL_MIN * 100)),
            float(np.percentile(returns, LABEL_MED * 100)),
            float(np.percentile(returns, LABEL_MAX * 100)),
            int(future_c[-1] > current)
        ))

    labels_df = pd.DataFrame(
        future_returns,
        columns=["label_min", "label_med", "label_max", "label_dir"],
        index=df.index
    )

    return pd.concat([df, labels_df], axis=1)


# ── Main ─────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("MuriTrading – Feature Engineering")
    print("=" * 40)

    # Laden
    print("\n1. Daten laden...")
    df_15m = load_csv("15m")
    df_1h  = load_csv("1h")
    df_4h  = load_csv("4h")
    df_1d  = load_csv("1d")
    print(f"   15m: {len(df_15m):,} | 1h: {len(df_1h):,} | 4h: {len(df_4h):,} | 1d: {len(df_1d):,}")

    # Indikatoren pro Timeframe
    print("\n2. Indikatoren berechnen...")
    df_15m = add_indicators(df_15m, prefix="15m_")
    df_1h  = add_indicators(df_1h,  prefix="1h_")
    df_4h  = add_indicators(df_4h,  prefix="4h_")
    df_1d  = add_indicators(df_1d,  prefix="1d_")
    print(f"   15m: {len(df_15m.columns)} Features")
    print(f"   1h : {len(df_1h.columns)} Features")
    print(f"   4h : {len(df_4h.columns)} Features")
    print(f"   1d : {len(df_1d.columns)} Features")

    # 15m auf 1h mergen
    print("\n3. MTF Confluence aufbauen...")
    df_15m_sel = df_15m[[c for c in df_15m.columns if c.startswith("15m_")]].copy()
    df_base = pd.merge_asof(
        df_1h.sort_index(),
        df_15m_sel.sort_index(),
        left_index=True,
        right_index=True,
        direction="backward"
    )

    # 4h + 1d dazu
    df_base = merge_higher_timeframes(df_base, df_4h, df_1d)
    df_base = add_confluence_score(df_base)
    print(f"   Feature Matrix: {df_base.shape[1]} Spalten")

    # External Data (Sentiment, On-Chain, BTC Korrelation)
    print("\n3b. Externe Daten laden (Sentiment, BTC, Market)...")
    try:
        sys.path.insert(0, os.path.expanduser("~/MuriTrading"))
        from src.features.external_data import fetch_all_external
        df_ext = fetch_all_external(days=min(365 * 4, 2000))
        df_ext = df_ext[~df_ext.index.duplicated(keep="last")]
        ext_cols = [f"ext_{c}" for c in df_ext.columns]
        df_ext.columns = ext_cols
        # merge_asof: Daily external → 1h-Index
        df_base = pd.merge_asof(
            df_base.sort_index(),
            df_ext.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
        )
        # Externe Features mit ffill auffüllen (ältere Zeilen ohne Daten)
        for col in ext_cols:
            df_base[col] = df_base[col].ffill().bfill()
        print(f"   +{len(ext_cols)} externe Features hinzugefügt")
    except Exception as e:
        print(f"   Externe Daten übersprungen: {e}")

    print(f"   Feature Matrix: {df_base.shape[1]} Spalten")

    # Labels
    print("\n4. Labels berechnen (Min/Med/Max)...")
    df_base = add_labels(df_base, horizon=HORIZON_CANDLES)
    label_counts = df_base["label_dir"].value_counts()
    total = label_counts.sum()
    print(f"   Bullisch: {label_counts.get(1,0):,} ({label_counts.get(1,0)/total*100:.1f}%)")
    print(f"   Bärisch:  {label_counts.get(0,0):,} ({label_counts.get(0,0)/total*100:.1f}%)")

    # NaN-Zeilen entfernen (erste N Kerzen haben keine vollständigen Indikatoren)
    before = len(df_base)
    df_base = df_base.dropna()
    print(f"\n5. NaN-Bereinigung: {before:,} → {len(df_base):,} Zeilen")

    # Speichern
    output_path = os.path.join(OUTPUT_DIR, "features_1h.csv")
    df_base.to_csv(output_path)
    print(f"\n6. Gespeichert: {output_path}")
    print(f"   Shape: {df_base.shape}")
    print(f"   Zeitraum: {df_base.index[0].date()} → {df_base.index[-1].date()}")

    # Kurze Vorschau
    print("\n── Feature-Übersicht ──────────────────────")
    print(f"   Confluence Net (Mittel): {df_base['confluence_net'].mean():.2f}")
    print(f"   Label Med (Mittel):      {df_base['label_med'].mean()*100:.4f}%")
    print(f"   Label Min (Mittel):      {df_base['label_min'].mean()*100:.4f}%")
    print(f"   Label Max (Mittel):      {df_base['label_max'].mean()*100:.4f}%")
    print("\nPhase 2 abgeschlossen.")


if __name__ == "__main__":
    main()

