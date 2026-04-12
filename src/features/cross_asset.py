"""
MuriTrading – Cross-Asset Correlation Features
BTC und ETH als Frühwarnsystem für XRP.

Features:
  - BTC Momentum Lead (BTC pumpt → XRP folgt 5-15 Min später)
  - ETH/BTC Ratio (Alt-Season Indikator)
  - BTC Dominance Proxy (BTC allein vs. Alts mit)
  - Momentum Divergence (BTC bewegt sich, XRP noch nicht = Catch-Up)
  - Correlation Regime (Wenn BTC-XRP Korrelation bricht)
"""

import pandas as pd
import numpy as np


def _fetch_candles(exchange, symbol, timeframe, limit):
    """Holt Candles via ccxt (gleicher Exchange wie für XRP)."""
    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns=["timestamp"]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _align(*dfs):
    """Findet gemeinsamen Index aller DataFrames."""
    idx = dfs[0].index
    for df in dfs[1:]:
        idx = idx.intersection(df.index)
    return idx


# ═══════════════════════════════════════════════════════════════
#  BTC MOMENTUM LEAD
# ═══════════════════════════════════════════════════════════════

def _btc_lead_features(xrp, btc):
    common = _align(xrp, btc)
    f = pd.DataFrame(index=common)

    # BTC Returns (15m, 30m, 45m)
    f["ca_btc_ret_1"] = btc["close"].pct_change(1).loc[common]
    f["ca_btc_ret_2"] = btc["close"].pct_change(2).loc[common]
    f["ca_btc_ret_3"] = btc["close"].pct_change(3).loc[common]

    # Beschleunigung
    f["ca_btc_accel"] = f["ca_btc_ret_1"] - f["ca_btc_ret_1"].shift(1)

    # BTC RSI (schnell, 7 Perioden auf 15m)
    delta = btc["close"].diff().loc[common]
    gain = delta.clip(lower=0).ewm(alpha=1/7, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(alpha=1/7, adjust=False).mean()
    f["ca_btc_rsi_7"] = 100 - (100 / (1 + gain / loss.replace(0, np.nan)))

    # BTC Volume Surge
    vol = btc["volume"].loc[common]
    f["ca_btc_vol_surge"] = vol / (vol.rolling(20).mean() + 1e-10)

    # Lead Gap: BTC bewegt sich, XRP nicht
    xrp_ret = xrp["close"].pct_change().loc[common]
    f["ca_lead_gap_1"] = f["ca_btc_ret_1"] - xrp_ret
    f["ca_lead_gap_3"] = f["ca_btc_ret_3"] - xrp["close"].pct_change(3).loc[common]

    return f


# ═══════════════════════════════════════════════════════════════
#  ETH/BTC RATIO (ALT-SEASON)
# ═══════════════════════════════════════════════════════════════

def _ethbtc_ratio_features(eth_1h, btc_1h):
    common = _align(eth_1h, btc_1h)
    ratio = eth_1h["close"].loc[common] / btc_1h["close"].loc[common]
    f = pd.DataFrame(index=common)

    f["ca_ethbtc_ratio"] = ratio
    f["ca_ethbtc_ret_6"] = ratio.pct_change(6)
    f["ca_ethbtc_ret_24"] = ratio.pct_change(24)

    ema_24 = ratio.ewm(span=24, adjust=False).mean()
    f["ca_ethbtc_vs_ema"] = (ratio - ema_24) / (ema_24 + 1e-10)

    ema_72 = ratio.ewm(span=72, adjust=False).mean()
    f["ca_ethbtc_trend"] = (ratio > ema_72).astype(int)

    # Alt-Season Score: 0=BTC dominiert, 1=neutral, 2=alt-season
    f["ca_alt_season"] = (
        (f["ca_ethbtc_ret_24"] > 0).astype(int) + f["ca_ethbtc_trend"]
    )

    return f


# ═══════════════════════════════════════════════════════════════
#  BTC DOMINANCE PROXY
# ═══════════════════════════════════════════════════════════════

def _btc_dominance_features(xrp, eth, btc):
    common = _align(xrp, eth, btc)
    f = pd.DataFrame(index=common)

    for w in [6, 12]:
        btc_r = btc["close"].pct_change(w).loc[common]
        eth_r = eth["close"].pct_change(w).loc[common]
        xrp_r = xrp["close"].pct_change(w).loc[common]
        alt_avg = (eth_r + xrp_r) / 2

        f[f"ca_btc_dom_{w}"] = btc_r - alt_avg

    # BTC pumpt allein (bearish für Alts)
    btc_r6 = btc["close"].pct_change(6).loc[common]
    alt_avg6 = (eth["close"].pct_change(6).loc[common] + xrp["close"].pct_change(6).loc[common]) / 2
    f["ca_btc_only_pump"] = ((btc_r6 > 0.005) & (alt_avg6 < 0.001)).astype(int)
    f["ca_alt_only_pump"] = ((alt_avg6 > 0.005) & (btc_r6 < 0.001)).astype(int)

    return f


# ═══════════════════════════════════════════════════════════════
#  MOMENTUM DIVERGENCE (CATCH-UP SIGNAL)
# ═══════════════════════════════════════════════════════════════

def _divergence_features(xrp, btc):
    common = _align(xrp, btc)
    f = pd.DataFrame(index=common)

    for w in [4, 8, 16]:  # 1h, 2h, 4h in 15m-Bars
        btc_r = btc["close"].pct_change(w).loc[common]
        xrp_r = xrp["close"].pct_change(w).loc[common]
        div = btc_r - xrp_r
        f[f"ca_div_{w}"] = div
        div_std = div.rolling(48).std()
        f[f"ca_div_z_{w}"] = div / (div_std + 1e-10)

    # Catch-Up Signal (clipped z-score)
    f["ca_catchup_signal"] = f["ca_div_z_8"].clip(-3, 3)

    return f


# ═══════════════════════════════════════════════════════════════
#  CORRELATION REGIME
# ═══════════════════════════════════════════════════════════════

def _correlation_features(xrp, btc):
    common = _align(xrp, btc)
    btc_ret = btc["close"].pct_change().loc[common]
    xrp_ret = xrp["close"].pct_change().loc[common]
    f = pd.DataFrame(index=common)

    for w in [24, 48, 96]:
        f[f"ca_corr_{w}"] = btc_ret.rolling(w).corr(xrp_ret)

    # Correlation Change (schnell vs langsam)
    f["ca_corr_delta"] = f["ca_corr_24"] - f["ca_corr_96"]

    # Regime: 0=entkoppelt, 1=normal, 2=hoch korreliert
    f["ca_corr_regime"] = np.select(
        [f["ca_corr_48"] > 0.7, f["ca_corr_48"] < 0.3],
        [2, 0], default=1,
    )

    # Breakdown: Korrelation bricht ein
    f["ca_corr_breakdown"] = (f["ca_corr_delta"] < -0.3).astype(int)

    # Beta: XRP-Sensitivität auf BTC
    cov = btc_ret.rolling(48).cov(xrp_ret)
    var = btc_ret.rolling(48).var()
    f["ca_beta_48"] = cov / (var + 1e-10)

    beta_mean = f["ca_beta_48"].rolling(96).mean()
    beta_std = f["ca_beta_48"].rolling(96).std()
    f["ca_beta_z"] = (f["ca_beta_48"] - beta_mean) / (beta_std + 1e-10)

    return f


# ═══════════════════════════════════════════════════════════════
#  HAUPTFUNKTION
# ═══════════════════════════════════════════════════════════════

CROSS_ASSET_FEATURES = [
    "ca_btc_ret_1", "ca_btc_ret_2", "ca_btc_ret_3", "ca_btc_accel",
    "ca_btc_rsi_7", "ca_btc_vol_surge", "ca_lead_gap_1", "ca_lead_gap_3",
    "ca_ethbtc_ratio", "ca_ethbtc_ret_6", "ca_ethbtc_ret_24",
    "ca_ethbtc_vs_ema", "ca_ethbtc_trend", "ca_alt_season",
    "ca_btc_dom_6", "ca_btc_dom_12", "ca_btc_only_pump", "ca_alt_only_pump",
    "ca_div_4", "ca_div_8", "ca_div_16",
    "ca_div_z_4", "ca_div_z_8", "ca_div_z_16", "ca_catchup_signal",
    "ca_corr_24", "ca_corr_48", "ca_corr_96", "ca_corr_delta",
    "ca_corr_regime", "ca_corr_breakdown", "ca_beta_48", "ca_beta_z",
]


def build_cross_asset_features_batch(xrp_15m, btc_15m, eth_15m, btc_1h, eth_1h):
    """
    Batch-Version für historisches Training.
    Gibt kompletten DataFrame mit allen Cross-Asset Features zurück.

    Args:
        xrp_15m, btc_15m, eth_15m: 15m DataFrames (OHLCV)
        btc_1h, eth_1h: 1h DataFrames (OHLCV)

    Returns:
        DataFrame mit ca_* Spalten, indiziert auf 15m-Zeitstempel.
    """
    f1 = _btc_lead_features(xrp_15m, btc_15m)
    f2 = _ethbtc_ratio_features(eth_1h, btc_1h)
    f3 = _btc_dominance_features(xrp_15m, eth_15m, btc_15m)
    f4 = _divergence_features(xrp_15m, btc_15m)
    f5 = _correlation_features(xrp_15m, btc_15m)

    result = f1.copy()
    for fx in [f3, f4, f5]:
        result = result.join(fx, how="left")

    result = pd.merge_asof(
        result.sort_index(), f2.sort_index(),
        left_index=True, right_index=True, direction="backward",
    )

    # Forward-fill 1h Features
    for col in f2.columns:
        if col in result.columns:
            result[col] = result[col].ffill()

    return result


def build_cross_asset_features(exchange, xrp_15m=None):
    """
    Hauptfunktion: Holt BTC/ETH Daten und berechnet alle Cross-Asset Features.

    Args:
        exchange: ccxt.binance Instanz
        xrp_15m: optional XRP 15m DataFrame (vermeidet Doppel-Fetch)

    Returns:
        Dict mit allen Feature-Werten (letzte Zeile).
        Bei Fehler: alle Werte NaN.
    """
    nan = float("nan")
    empty = {col: nan for col in CROSS_ASSET_FEATURES}

    try:
        btc_15m = _fetch_candles(exchange, "BTC/USDT", "15m", 200)
        eth_15m = _fetch_candles(exchange, "ETH/USDT", "15m", 200)
        btc_1h = _fetch_candles(exchange, "BTC/USDT", "1h", 200)
        eth_1h = _fetch_candles(exchange, "ETH/USDT", "1h", 200)

        if xrp_15m is None:
            xrp_15m = _fetch_candles(exchange, "XRP/USDT", "15m", 200)

        # Feature-Gruppen berechnen
        f1 = _btc_lead_features(xrp_15m, btc_15m)
        f2 = _ethbtc_ratio_features(eth_1h, btc_1h)
        f3 = _btc_dominance_features(xrp_15m, eth_15m, btc_15m)
        f4 = _divergence_features(xrp_15m, btc_15m)
        f5 = _correlation_features(xrp_15m, btc_15m)

        # Zusammenführen auf 15m-Index
        result = f1.copy()
        for fx in [f3, f4, f5]:
            result = result.join(fx, how="left")

        # 1h Features (ETH/BTC) auf 15m mergen
        result = pd.merge_asof(
            result.sort_index(), f2.sort_index(),
            left_index=True, right_index=True, direction="backward",
        )

        # Letzte Zeile als Dict
        if len(result) == 0:
            return empty

        last = result.iloc[-1]
        return {col: float(last[col]) if col in last.index and not pd.isna(last[col]) else nan
                for col in CROSS_ASSET_FEATURES}

    except Exception:
        return empty


def cross_asset_signal_text(ca):
    """Formatiert Cross-Asset Features als kurzen Text."""
    import math
    btc_ret = ca.get("ca_btc_ret_1", float("nan"))
    if math.isnan(btc_ret):
        return "Cross: n/a"

    parts = [f"BTC:{btc_ret*100:+.2f}%"]

    gap = ca.get("ca_lead_gap_1", 0)
    if not math.isnan(gap) and abs(gap) > 0.002:
        direction = "XRP↑" if gap > 0 else "XRP↓"
        parts.append(f"Gap:{direction}")

    catchup = ca.get("ca_catchup_signal", 0)
    if not math.isnan(catchup) and abs(catchup) > 1.5:
        parts.append(f"CATCH-UP {'↑' if catchup > 0 else '↓'}")

    alt = ca.get("ca_alt_season", 0)
    if not math.isnan(alt):
        if alt >= 2:
            parts.append("ALT-SEASON")
        elif alt == 0:
            parts.append("BTC-DOM")

    corr = ca.get("ca_corr_48", 0)
    if not math.isnan(corr):
        parts.append(f"ρ:{corr:.2f}")

    if ca.get("ca_corr_breakdown"):
        parts.append("⚡DECOUPLE")

    if ca.get("ca_btc_only_pump"):
        parts.append("⚠BTC-ONLY")
    elif ca.get("ca_alt_only_pump"):
        parts.append("🚀ALT-PUMP")

    return "Cross: " + "  ".join(parts)
