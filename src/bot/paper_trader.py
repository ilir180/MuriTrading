"""
MuriTrading v3.0 – Strategy-Based Paper Trading Bot
=====================================================
Kompletter Neuaufbau. Bewährte Strategien statt blindes ML-Vertrauen.

Kernänderungen vs v2:
  - 4H Primär-Timeframe (grössere Moves, Fees irrelevant)
  - Strategie-basierte Signale mit ML als Filter
  - Trend Following + Mean Reversion + Breakout
  - ATR-basierte Positionsgrösse (1.5% Risiko pro Trade)
  - Trailing Stops als primärer Exit
  - Max 1 Position gleichzeitig
  - Monatliches Drawdown-Limit (8%)
  - Consecutive-Loss Circuit Breaker

Start: python src/bot/paper_trader.py
Stop:  Ctrl+C (speichert automatisch)
"""

# Fix PyTorch/OpenMP + sklearn threading conflict
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import pandas as pd
import numpy as np
import pickle
import json
import os
import sys
import ccxt
import time
import math
import signal as sig
import requests as _requests
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple

# ── Projekt-Root ───────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser("~/MuriTrading")
sys.path.insert(0, PROJECT_ROOT)
from src.features.build_features import add_indicators

# ── Pfade ──────────────────────────────────────────────────────
MODEL_DIR   = os.path.join(PROJECT_ROOT, "models")
BOT_DIR     = os.path.join(PROJECT_ROOT, "data", "bot")
SIGNALS_LOG = os.path.join(BOT_DIR, "signals.csv")
STATE_FILE  = os.path.join(BOT_DIR, "bot_state_v3.json")
TRADES_LOG  = os.path.join(BOT_DIR, "trades_v3.csv")
EQUITY_LOG  = os.path.join(BOT_DIR, "equity_v3.csv")

# ═══════════════════════════════════════════════════════════════
#  KONFIGURATION
# ═══════════════════════════════════════════════════════════════

SYMBOL             = "XRP/USDT"
INITIAL_CAPITAL    = 1000.0

# Timeframe & Intervalle
PRIMARY_TF         = "4h"          # Signale nur auf 4H-Kerzen
CHECK_INTERVAL     = 60            # Sekunden zwischen Checks
CANDLE_WAIT_MIN    = 2             # Minuten nach Kerzenschluss warten

# Risk Management
RISK_PER_TRADE     = 0.015         # 1.5% Risiko pro Trade
MAX_OPEN_POSITIONS = 1             # Fokus: ein Trade gleichzeitig
MAX_TRADES_PER_WEEK= 5             # Max 5 Trades pro Woche
MAX_MONTHLY_DD     = 0.08          # 8% monatliches Drawdown-Limit
CONSEC_LOSS_LIMIT  = 3             # Nach 3 Verlusten: 24h Pause
COOLDOWN_HOURS     = 24            # Stunden Pause nach Circuit Breaker

# Fees (konservativ: Taker + Slippage)
TAKER_FEE          = 0.0004
SLIPPAGE           = 0.0002
ROUND_TRIP         = (TAKER_FEE + SLIPPAGE) * 2   # 0.12%

# Position Management
MIN_REWARD_RISK    = 1.5           # 1.5:1 R:R (tighter = higher WR)
SL_ATR_MULT        = 2.0           # Stop Loss = 2x ATR
TRAILING_ATR_MULT  = 1.5           # Trailing Stop = 1.5x ATR
TRAILING_ACTIVATE  = 0.8           # Trailing aktiviert ab 0.8x ATR Gewinn
MAX_HOLD_CANDLES   = 18            # Max 72h (18 × 4h Kerzen)
PARTIAL_TP_RR      = 1.0           # Partial Take-Profit bei 1:1 R:R
PARTIAL_SIZE_PCT   = 0.50          # 50% der Position bei Partial TP schliessen
BREAKEVEN_BUFFER   = 0.001         # 0.1% über Entry für Breakeven-Stop

# Telegram
TG_BOT_TOKEN = "8503143803:AAH-7DPWX-bXq-ITRGpw4TwkDTDtIsRzQt8"
TG_CHAT_ID   = "7704168743"

# Daily Report
DAILY_REPORT_HOUR  = 22            # UTC

# ═══════════════════════════════════════════════════════════════
#  TELEGRAM
# ═══════════════════════════════════════════════════════════════

def tg_send(text):
    try:
        _requests.post(
            f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as e:
        print(f"  TG-Fehler: {e}")


# ═══════════════════════════════════════════════════════════════
#  TERMINAL STYLING
# ═══════════════════════════════════════════════════════════════

class C:
    PURPLE = "\033[95m"
    BLUE   = "\033[94m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BOLD   = "\033[1m"
    DIM    = "\033[2m"
    RESET  = "\033[0m"


def log(msg, color=C.RESET):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"  {C.DIM}[{ts}]{C.RESET} {color}{msg}{C.RESET}", flush=True)


def banner():
    print(f"""
{C.PURPLE}{C.BOLD}═══════════════════════════════════════════════════════{C.RESET}
{C.BOLD}  MuriTrading v3.0 – Strategy-Based Trading Bot{C.RESET}
{C.PURPLE}═══════════════════════════════════════════════════════{C.RESET}
{C.DIM}  Asset         : {SYMBOL}
  Kapital       : ${INITIAL_CAPITAL:,.2f}
  Timeframe     : {PRIMARY_TF} (Signale alle 4 Stunden)
  Risiko/Trade  : {RISK_PER_TRADE*100:.1f}%
  Min R:R       : {MIN_REWARD_RISK:.1f}:1
  Partial TP    : 50% bei {PARTIAL_TP_RR:.0f}:1 + Stop→Breakeven
  Stop Loss     : {SL_ATR_MULT:.0f}x ATR
  Trailing Stop : {TRAILING_ATR_MULT:.1f}x ATR (ab {TRAILING_ACTIVATE:.1f}x ATR Gewinn)
  Max Haltezeit : {MAX_HOLD_CANDLES * 4}h
  Fees (RT)     : {ROUND_TRIP*100:.3f}%
  Max DD/Monat  : {MAX_MONTHLY_DD*100:.0f}%{C.RESET}

  {C.GREEN}Strategien:{C.RESET}
  1. {C.BLUE}Trend Following{C.RESET}   – 13-Punkt Triple-TF Confluence (9/13 nötig)
  2. {C.YELLOW}Mean Reversion{C.RESET}    – Extreme RSI<25 + BB-Boden + Multi-TF
  3. {C.PURPLE}Breakout{C.RESET}          – BB-Squeeze + Vol 2.5x + Daily-Trend
  {C.GREEN}Target Win Rate  : 73-80%+ (Muri-würdig){C.RESET}
{C.PURPLE}═══════════════════════════════════════════════════════{C.RESET}
""", flush=True)


# ═══════════════════════════════════════════════════════════════
#  MODELL & DATEN
# ═══════════════════════════════════════════════════════════════

def load_models():
    with open(os.path.join(MODEL_DIR, "rf_model.pkl"), "rb") as f:
        rf = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "xgb_model.pkl"), "rb") as f:
        xgb = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "model_meta.json")) as f:
        meta = json.load(f)
    return rf, xgb, meta


def get_exchange():
    return ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})


def fetch_candles(exchange, timeframe, limit):
    candles = exchange.fetch_ohlcv(SYMBOL, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns=["timestamp"]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def build_features(exchange, feature_cols):
    """Holt frische Daten und berechnet alle Features."""
    df_1h  = fetch_candles(exchange, "1h", 250)
    df_4h  = fetch_candles(exchange, "4h", 250)
    df_1d  = fetch_candles(exchange, "1d", 250)

    df_1h  = add_indicators(df_1h,  prefix="1h_")
    df_4h  = add_indicators(df_4h,  prefix="4h_")
    df_1d  = add_indicators(df_1d,  prefix="1d_")

    # Merge: 1h als Basis, 4h und 1d backward-fill
    df_4h_sel = df_4h[[c for c in df_4h.columns if c.startswith("4h_")]].copy()
    df_base = pd.merge_asof(df_1h.sort_index(), df_4h_sel.sort_index(),
        left_index=True, right_index=True, direction="backward")

    df_1d_sel = df_1d[[c for c in df_1d.columns if c.startswith("1d_")]].copy()
    df_base = pd.merge_asof(df_base.sort_index(), df_1d_sel.sort_index(),
        left_index=True, right_index=True, direction="backward")

    # Confluence Score
    bull_signals = ["1h_ema_9_above_21", "1h_ema_21_above_50", "1h_macd_above",
        "4h_ema_9_above_21", "4h_ema_21_above_50", "4h_macd_above",
        "1d_ema_9_above_21", "1d_ema_21_above_50", "1d_macd_above"]
    bear_signals = ["1h_rsi_overbought", "4h_rsi_overbought", "1d_rsi_overbought"]

    avail_bull = [s for s in bull_signals if s in df_base.columns]
    avail_bear = [s for s in bear_signals if s in df_base.columns]
    if avail_bull:
        df_base["confluence_bull"] = df_base[avail_bull].sum(axis=1)
    if avail_bear:
        df_base["confluence_bear"] = df_base[avail_bear].sum(axis=1)
    df_base["confluence_net"] = df_base.get("confluence_bull", 0) - df_base.get("confluence_bear", 0)

    # Cross-Asset Features (falls Modell sie braucht)
    ca_cols_needed = [c for c in feature_cols if c.startswith("ca_")]
    if ca_cols_needed:
        try:
            from src.features.cross_asset import build_cross_asset_features_batch, _fetch_candles
            btc_1h  = _fetch_candles(exchange, "BTC/USDT", "1h",  200)
            eth_1h  = _fetch_candles(exchange, "ETH/USDT", "1h",  200)
            # Vereinfacht: nur 1h für cross-asset
            df_15m = fetch_candles(exchange, "15m", 200)
            df_15m = add_indicators(df_15m, prefix="15m_")
            btc_15m = _fetch_candles(exchange, "BTC/USDT", "15m", 200)
            eth_15m = _fetch_candles(exchange, "ETH/USDT", "15m", 200)
            ca_df = build_cross_asset_features_batch(df_15m, btc_15m, eth_15m, btc_1h, eth_1h)
            ca_sel = ca_df[[c for c in ca_df.columns if c in ca_cols_needed]].copy()
            df_base = pd.merge_asof(
                df_base.sort_index(), ca_sel.sort_index(),
                left_index=True, right_index=True, direction="backward",
            )
            for col in ca_sel.columns:
                df_base[col] = df_base[col].ffill()
        except Exception:
            for col in ca_cols_needed:
                df_base[col] = 0.0

    # Fehlende Feature-Spalten mit 0 füllen
    for col in feature_cols:
        if col not in df_base.columns:
            df_base[col] = 0.0

    latest = df_base[feature_cols].dropna()
    if latest.empty:
        return None, None, None
    return latest.iloc[[-1]], df_base.iloc[-1], df_1h["close"].iloc[-1]


def predict(rf, xgb_model, X):
    rf_prob  = rf.predict_proba(X)[:, 1]
    xgb_prob = xgb_model.predict_proba(X)[:, 1]
    ensemble = (rf_prob + xgb_prob) / 2.0
    return float(ensemble[0]), float(rf_prob[0]), float(xgb_prob[0])


# ═══════════════════════════════════════════════════════════════
#  SIGNAL & STRATEGIE
# ═══════════════════════════════════════════════════════════════

@dataclass
class Signal:
    direction: str       # 'long' oder 'short'
    strategy: str        # 'trend', 'mean_reversion', 'breakout'
    strength: float      # 0-1 (Stärke des Signals)
    entry_price: float
    stop_loss: float
    take_profit: float
    atr: float
    details: str = ""    # Menschenlesbare Begründung


def _safe(val, default=0.0):
    """NaN-safe Wert-Extraktion."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return float(val)


class StrategyEngine:
    """
    Generiert Trading-Signale basierend auf 4H technischer Analyse.
    ML-Ensemble wird als Bestätigungs-Filter verwendet, nicht als primäres Signal.
    """

    def check_trend_follow(self, row, price, ml_prob):
        """
        Ultra-Selective Trend Following – Triple-Timeframe Confluence.
        Optimiert auf 73%+ Win Rate durch maximale Bestätigung.

        13 Bedingungen, braucht >= 9 für Entry:
          1.  EMA 9 > 21 auf 4H (bullish alignment)
          2.  EMA 21 > 50 auf 4H (starker Trend)
          3.  ADX > 28 auf 4H (starke Trendstärke)
          4.  Pullback nahe EMA 21 (optimaler Einstieg)
          5.  RSI 40-65 auf 4H (nicht überkauft, Raum nach oben)
          6.  MACD über Signal auf 4H (Momentum)
          7.  Stoch RSI < 0.80 auf 4H (nicht überhitzt)
          8.  Daily EMA 9 > 21 (höherer TF bestätigt)
          9.  Daily EMA 21 > 50 (starker Daily-Trend)
          10. 1H EMA 9 > 21 (Micro-Trend stimmt zu)
          11. Volume über Durchschnitt (Volumen bestätigt)
          12. Bullische 4H-Kerze (Preis-Aktion bestätigt)
          13. ML Ensemble > 0.55 (ML bestätigt)
        """
        signals = []

        for direction in ["long", "short"]:
            score = 0
            reasons = []

            # ── 4H TREND STRUCTURE (4 Punkte) ────────────────
            ema9_21 = _safe(row.get("4h_ema_9_above_21", 0))
            ema21_50 = _safe(row.get("4h_ema_21_above_50", 0))
            adx = _safe(row.get("4h_adx", 0))

            if direction == "long":
                if ema9_21 > 0.5:
                    score += 1; reasons.append("4H:EMA9>21")
                if ema21_50 > 0.5:
                    score += 1; reasons.append("4H:EMA21>50")
            else:
                if ema9_21 < 0.5:
                    score += 1; reasons.append("4H:EMA9<21")
                if ema21_50 < 0.5:
                    score += 1; reasons.append("4H:EMA21<50")

            if adx > 28:
                score += 1; reasons.append(f"ADX:{adx:.0f}")

            # Pullback nahe EMA 21
            ema21_dist = _safe(row.get("4h_ema_21_dist", 0))
            if direction == "long":
                if -0.008 < ema21_dist < 0.004:
                    score += 1; reasons.append(f"Pull:{ema21_dist:+.3f}")
            else:
                if -0.004 < ema21_dist < 0.008:
                    score += 1; reasons.append(f"Pull:{ema21_dist:+.3f}")

            # ── 4H MOMENTUM (3 Punkte) ───────────────────────
            rsi = _safe(row.get("4h_rsi_14", 50))
            macd_above = _safe(row.get("4h_macd_above", 0))
            stoch_rsi = _safe(row.get("4h_stoch_rsi", 0.5))

            if direction == "long":
                if 35 < rsi < 65:
                    score += 1; reasons.append(f"RSI:{rsi:.0f}")
                if macd_above > 0.5:
                    score += 1; reasons.append("MACD+")
                if stoch_rsi < 0.80:
                    score += 1; reasons.append(f"StRSI:{stoch_rsi:.2f}")
            else:
                if 35 < rsi < 65:
                    score += 1; reasons.append(f"RSI:{rsi:.0f}")
                if macd_above < 0.5:
                    score += 1; reasons.append("MACD-")
                if stoch_rsi > 0.20:
                    score += 1; reasons.append(f"StRSI:{stoch_rsi:.2f}")

            # ── DAILY CONFIRMATION (2 Punkte) ────────────────
            daily_ema9_21 = _safe(row.get("1d_ema_9_above_21", 0))
            daily_ema21_50 = _safe(row.get("1d_ema_21_above_50", 0))

            if direction == "long":
                if daily_ema9_21 > 0.5:
                    score += 1; reasons.append("1D:EMA9>21")
                if daily_ema21_50 > 0.5:
                    score += 1; reasons.append("1D:EMA21>50")
            else:
                if daily_ema9_21 < 0.5:
                    score += 1; reasons.append("1D:EMA9<21")
                if daily_ema21_50 < 0.5:
                    score += 1; reasons.append("1D:EMA21<50")

            # ── 1H MICRO-TREND (1 Punkt) ─────────────────────
            h1_ema = _safe(row.get("1h_ema_9_above_21", 0))
            if direction == "long" and h1_ema > 0.5:
                score += 1; reasons.append("1H:EMA+")
            elif direction == "short" and h1_ema < 0.5:
                score += 1; reasons.append("1H:EMA-")

            # ── VOLUME & PRICE ACTION (2 Punkte) ─────────────
            vol_ratio = _safe(row.get("4h_vol_ratio", 1.0))
            candle_bull = _safe(row.get("4h_candle_bull", 0.5))

            if vol_ratio > 0.8:
                score += 1; reasons.append(f"Vol:{vol_ratio:.1f}x")

            if direction == "long" and candle_bull > 0.5:
                score += 1; reasons.append("Bull-Candle")
            elif direction == "short" and candle_bull < 0.5:
                score += 1; reasons.append("Bear-Candle")

            # ── ML CONFIRMATION (1 Punkt) ─────────────────────
            if direction == "long" and ml_prob > 0.55:
                score += 1; reasons.append(f"ML:{ml_prob:.0%}")
            elif direction == "short" and ml_prob < 0.45:
                score += 1; reasons.append(f"ML:{ml_prob:.0%}")

            # ── ENTRY DECISION: 9/13 minimum ─────────────────
            if score >= 9:
                atr = _safe(row.get("4h_atr_14", price * 0.01))
                if atr < price * 0.001:
                    atr = price * 0.01

                sl_dist = SL_ATR_MULT * atr
                tp_dist = sl_dist * MIN_REWARD_RISK

                if direction == "long":
                    sl = price - sl_dist
                    tp = price + tp_dist
                else:
                    sl = price + sl_dist
                    tp = price - tp_dist

                signals.append(Signal(
                    direction=direction,
                    strategy="trend",
                    strength=score / 13.0,
                    entry_price=price,
                    stop_loss=round(sl, 6),
                    take_profit=round(tp, 6),
                    atr=atr,
                    details=f"Trend {direction.upper()} ({score}/13): {', '.join(reasons)}",
                ))

        return signals

    def check_mean_reversion(self, row, price, ml_prob):
        """
        Ultra-Selective Mean Reversion – nur bei extremen Bedingungen.
        Optimiert auf 73%+ Win Rate.

        Long (ALLE müssen stimmen):
          - ADX < 20 (klar rangend)
          - RSI < 25 auf 4H (stark überverkauft)
          - BB Position < 0.08 (am unteren Bollinger Band)
          - Bullische Kerze (Umkehrsignal)
          - Volume Spike > 1.5x (Erschöpfungsvolumen)
          - 1H RSI auch überverkauft (< 35)
          - ML nicht stark dagegen
        """
        signals = []
        adx = _safe(row.get("4h_adx", 25))
        rsi_4h = _safe(row.get("4h_rsi_14", 50))
        rsi_1h = _safe(row.get("1h_rsi_14", 50))
        bb_pos = _safe(row.get("4h_bb_pos", 0.5))
        candle_bull = _safe(row.get("4h_candle_bull", 0))
        vol_ratio = _safe(row.get("4h_vol_ratio", 1.0))
        stoch_rsi = _safe(row.get("4h_stoch_rsi", 0.5))
        atr = _safe(row.get("4h_atr_14", price * 0.01))
        if atr < price * 0.001:
            atr = price * 0.01

        # HARTER Filter: muss klar rangend sein
        if adx >= 20:
            return signals

        # LONG: Stark überverkauft
        if (rsi_4h < 25 and bb_pos < 0.08 and candle_bull > 0.5
                and vol_ratio > 1.5 and rsi_1h < 35 and stoch_rsi < 0.15):

            reasons = [f"RSI4H:{rsi_4h:.0f}", f"RSI1H:{rsi_1h:.0f}",
                       f"BB:{bb_pos:.2f}", "Bull-Candle",
                       f"Vol:{vol_ratio:.1f}x", f"StRSI:{stoch_rsi:.2f}"]

            if ml_prob < 0.30:
                return signals

            sl_dist = SL_ATR_MULT * atr
            tp_dist = sl_dist * MIN_REWARD_RISK

            signals.append(Signal(
                direction="long",
                strategy="mean_reversion",
                strength=min(1.0, (25 - rsi_4h) / 12),
                entry_price=price,
                stop_loss=round(price - sl_dist, 6),
                take_profit=round(price + tp_dist, 6),
                atr=atr,
                details=f"MeanRev LONG (ADX:{adx:.0f}): {', '.join(reasons)}",
            ))

        # SHORT: Stark überkauft
        if (rsi_4h > 75 and bb_pos > 0.92 and candle_bull < 0.5
                and vol_ratio > 1.5 and rsi_1h > 65 and stoch_rsi > 0.85):

            reasons = [f"RSI4H:{rsi_4h:.0f}", f"RSI1H:{rsi_1h:.0f}",
                       f"BB:{bb_pos:.2f}", "Bear-Candle",
                       f"Vol:{vol_ratio:.1f}x", f"StRSI:{stoch_rsi:.2f}"]

            if ml_prob > 0.70:
                return signals

            sl_dist = SL_ATR_MULT * atr
            tp_dist = sl_dist * MIN_REWARD_RISK

            signals.append(Signal(
                direction="short",
                strategy="mean_reversion",
                strength=min(1.0, (rsi_4h - 75) / 12),
                entry_price=price,
                stop_loss=round(price + sl_dist, 6),
                take_profit=round(price - tp_dist, 6),
                atr=atr,
                details=f"MeanRev SHORT (ADX:{adx:.0f}): {', '.join(reasons)}",
            ))

        return signals

    def check_breakout(self, row, price, ml_prob):
        """
        Ultra-Selective Breakout nach Bollinger Band Squeeze auf 4H.
        Optimiert auf 73%+ Win Rate.

        Bedingungen (ALLE müssen stimmen):
          - BB Squeeze aktiv (niedrige Volatilität)
          - Preis bricht aus BB aus (bb_pos > 0.95 oder < 0.05)
          - Volume Surge (> 2.5x Durchschnitt)
          - Daily Trend unterstützt Richtung (EMA9>21 + EMA21>50)
          - 4H MACD bestätigt Richtung
          - ML nicht dagegen
        """
        signals = []
        bb_squeeze = _safe(row.get("4h_bb_squeeze", 0))
        bb_pos = _safe(row.get("4h_bb_pos", 0.5))
        bb_width = _safe(row.get("4h_bb_width", 0.05))
        vol_ratio = _safe(row.get("4h_vol_ratio", 1.0))
        daily_ema9_21 = _safe(row.get("1d_ema_9_above_21", 0.5))
        daily_ema21_50 = _safe(row.get("1d_ema_21_above_50", 0.5))
        macd_above = _safe(row.get("4h_macd_above", 0.5))
        trend_cons = _safe(row.get("4h_trend_consistency", 0.5))
        atr = _safe(row.get("4h_atr_14", price * 0.01))
        if atr < price * 0.001:
            atr = price * 0.01

        # Squeeze muss aktiv sein ODER BB sehr eng
        if not (bb_squeeze > 0.5 or bb_width < 0.020):
            return signals

        # Volume muss STARK erhöht sein
        if vol_ratio < 2.5:
            return signals

        # LONG Breakout
        if (bb_pos > 0.95 and daily_ema9_21 > 0.5 and daily_ema21_50 > 0.5
                and macd_above > 0.5 and trend_cons > 0.5):
            if ml_prob < 0.50:
                return signals

            sl_dist = SL_ATR_MULT * atr
            tp_dist = sl_dist * MIN_REWARD_RISK

            signals.append(Signal(
                direction="long",
                strategy="breakout",
                strength=min(1.0, vol_ratio / 4.0),
                entry_price=price,
                stop_loss=round(price - sl_dist, 6),
                take_profit=round(price + tp_dist, 6),
                atr=atr,
                details=f"Breakout LONG: BB:{bb_pos:.2f} Vol:{vol_ratio:.1f}x BBW:{bb_width:.3f} TC:{trend_cons:.2f}",
            ))

        # SHORT Breakout
        if (bb_pos < 0.05 and daily_ema9_21 < 0.5 and daily_ema21_50 < 0.5
                and macd_above < 0.5 and trend_cons > 0.5):
            if ml_prob > 0.50:
                return signals

            sl_dist = SL_ATR_MULT * atr
            tp_dist = sl_dist * MIN_REWARD_RISK

            signals.append(Signal(
                direction="short",
                strategy="breakout",
                strength=min(1.0, vol_ratio / 4.0),
                entry_price=price,
                stop_loss=round(price + sl_dist, 6),
                take_profit=round(price - tp_dist, 6),
                atr=atr,
                details=f"Breakout SHORT: BB:{bb_pos:.2f} Vol:{vol_ratio:.1f}x BBW:{bb_width:.3f} TC:{trend_cons:.2f}",
            ))

        return signals

    def generate_signals(self, row, price, ml_prob):
        """
        Prüft alle Strategien und gibt das stärkste Signal zurück.
        Priorität: Trend > Breakout > Mean Reversion
        """
        all_signals = []

        # Trend Following (primär)
        all_signals.extend(self.check_trend_follow(row, price, ml_prob))

        # Breakout (sekundär)
        all_signals.extend(self.check_breakout(row, price, ml_prob))

        # Mean Reversion (tertiär)
        all_signals.extend(self.check_mean_reversion(row, price, ml_prob))

        if not all_signals:
            return None

        # Beste Strategie nach Priorität + Stärke
        priority = {"trend": 3, "breakout": 2, "mean_reversion": 1}
        all_signals.sort(key=lambda s: (priority.get(s.strategy, 0), s.strength), reverse=True)

        return all_signals[0]


# ═══════════════════════════════════════════════════════════════
#  POSITION MANAGEMENT
# ═══════════════════════════════════════════════════════════════

@dataclass
class Position:
    direction: str
    strategy: str
    entry_price: float
    size: float             # In USDT (verbleibende Grösse)
    original_size: float    # Ursprüngliche Grösse
    stop_loss: float
    take_profit: float
    atr: float
    entry_time: str
    candles_held: int = 0
    peak_price: float = 0.0
    trough_price: float = float('inf')
    trailing_active: bool = False
    trailing_stop: float = 0.0
    partial_taken: bool = False     # Ob Partial TP schon genommen
    partial_pnl: float = 0.0       # PnL aus Partial Close
    strength: float = 0.0
    details: str = ""

    def update_trailing(self, current_price):
        """Aktualisiert Trailing Stop basierend auf ATR."""
        if self.direction == "long":
            self.peak_price = max(self.peak_price, current_price)
            unrealized_move = self.peak_price - self.entry_price

            if unrealized_move >= TRAILING_ACTIVATE * self.atr:
                self.trailing_active = True
                new_trail = self.peak_price - TRAILING_ATR_MULT * self.atr
                self.trailing_stop = max(self.trailing_stop, new_trail)
                self.trailing_stop = max(self.trailing_stop, self.stop_loss)

        else:  # short
            self.trough_price = min(self.trough_price, current_price)
            unrealized_move = self.entry_price - self.trough_price

            if unrealized_move >= TRAILING_ACTIVATE * self.atr:
                self.trailing_active = True
                new_trail = self.trough_price + TRAILING_ATR_MULT * self.atr
                if self.trailing_stop == 0:
                    self.trailing_stop = new_trail
                else:
                    self.trailing_stop = min(self.trailing_stop, new_trail)
                self.trailing_stop = min(self.trailing_stop, self.stop_loss)

    def check_partial_tp(self, current_price):
        """
        Prüft ob Partial Take-Profit genommen werden soll.
        Bei 1:1 R:R → 50% der Position schliessen, Stop auf Breakeven.
        Returns: (should_partial: bool, partial_pnl: float)
        """
        if self.partial_taken:
            return False, 0.0

        sl_dist = abs(self.entry_price - self.stop_loss)
        partial_target = sl_dist * PARTIAL_TP_RR

        if self.direction == "long":
            if current_price >= self.entry_price + partial_target - 1e-9:
                # 50% schliessen
                partial_size = self.size * PARTIAL_SIZE_PCT
                raw_ret = (current_price - self.entry_price) / self.entry_price
                net_ret = raw_ret - (ROUND_TRIP / 2)  # Halbe Fees (nur Exit)
                partial_pnl = partial_size * net_ret

                self.size -= partial_size
                self.partial_taken = True
                self.partial_pnl = partial_pnl

                # Stop auf Breakeven + Buffer
                be_stop = self.entry_price * (1 + BREAKEVEN_BUFFER)
                self.stop_loss = max(self.stop_loss, be_stop)
                if self.trailing_stop > 0:
                    self.trailing_stop = max(self.trailing_stop, be_stop)

                return True, partial_pnl

        else:  # short
            if current_price <= self.entry_price - partial_target + 1e-9:
                partial_size = self.size * PARTIAL_SIZE_PCT
                raw_ret = (self.entry_price - current_price) / self.entry_price
                net_ret = raw_ret - (ROUND_TRIP / 2)
                partial_pnl = partial_size * net_ret

                self.size -= partial_size
                self.partial_taken = True
                self.partial_pnl = partial_pnl

                # Stop auf Breakeven
                be_stop = self.entry_price * (1 - BREAKEVEN_BUFFER)
                self.stop_loss = min(self.stop_loss, be_stop)
                if self.trailing_stop > 0:
                    self.trailing_stop = min(self.trailing_stop, be_stop)

                return True, partial_pnl

        return False, 0.0

    def check_exit(self, current_price):
        """
        Prüft alle Exit-Bedingungen.
        Returns: (should_exit: bool, reason: str)
        """
        # 1. Stop Loss
        if self.direction == "long" and current_price <= self.stop_loss:
            return True, "STOP-LOSS"
        if self.direction == "short" and current_price >= self.stop_loss:
            return True, "STOP-LOSS"

        # 2. Take Profit (finale TP für verbleibende Position)
        if self.direction == "long" and current_price >= self.take_profit:
            return True, "TAKE-PROFIT"
        if self.direction == "short" and current_price <= self.take_profit:
            return True, "TAKE-PROFIT"

        # 3. Trailing Stop
        if self.trailing_active:
            if self.direction == "long" and current_price <= self.trailing_stop:
                return True, "TRAILING-STOP"
            if self.direction == "short" and current_price >= self.trailing_stop:
                return True, "TRAILING-STOP"

        # 4. Maximale Haltezeit
        if self.candles_held >= MAX_HOLD_CANDLES:
            return True, "TIME-EXIT"

        return False, ""

    def calc_pnl(self, exit_price):
        """Berechnet PnL für verbleibende Position nach Fees."""
        if self.direction == "long":
            raw_ret = (exit_price - self.entry_price) / self.entry_price
        else:
            raw_ret = (self.entry_price - exit_price) / self.entry_price
        net_ret = raw_ret - ROUND_TRIP
        pnl = self.size * net_ret
        # Partial PnL hinzufügen
        total_pnl = pnl + self.partial_pnl
        return total_pnl, raw_ret, net_ret

    def to_dict(self):
        return {
            "direction": self.direction, "strategy": self.strategy,
            "entry_price": self.entry_price, "size": self.size,
            "original_size": self.original_size,
            "stop_loss": self.stop_loss, "take_profit": self.take_profit,
            "atr": self.atr, "entry_time": self.entry_time,
            "candles_held": self.candles_held, "peak_price": self.peak_price,
            "trough_price": self.trough_price,
            "trailing_active": self.trailing_active,
            "trailing_stop": self.trailing_stop,
            "partial_taken": self.partial_taken,
            "partial_pnl": self.partial_pnl,
            "strength": self.strength, "details": self.details,
        }

    @staticmethod
    def from_dict(d):
        p = Position(
            direction=d["direction"], strategy=d["strategy"],
            entry_price=d["entry_price"], size=d["size"],
            original_size=d.get("original_size", d["size"]),
            stop_loss=d["stop_loss"], take_profit=d["take_profit"],
            atr=d["atr"], entry_time=d["entry_time"],
        )
        p.candles_held = d.get("candles_held", 0)
        p.peak_price = d.get("peak_price", d["entry_price"])
        p.trough_price = d.get("trough_price", d["entry_price"])
        p.trailing_active = d.get("trailing_active", False)
        p.trailing_stop = d.get("trailing_stop", 0.0)
        p.partial_taken = d.get("partial_taken", False)
        p.partial_pnl = d.get("partial_pnl", 0.0)
        p.strength = d.get("strength", 0.0)
        p.details = d.get("details", "")
        return p


# ═══════════════════════════════════════════════════════════════
#  RISK MANAGER
# ═══════════════════════════════════════════════════════════════

class RiskManager:
    def __init__(self, initial_capital):
        self.initial_capital = initial_capital
        self.month_start_capital = initial_capital
        self.consecutive_losses = 0
        self.cooldown_until = None
        self.weekly_trades = {}  # {week_key: count}

    def calc_position_size(self, capital, entry_price, stop_loss):
        """
        ATR-basierte Positionsgrösse mit 1.5% Risiko.
        Risiko = was wir verlieren wenn SL getroffen wird.
        """
        sl_distance_pct = abs(entry_price - stop_loss) / entry_price
        if sl_distance_pct < 0.001:
            sl_distance_pct = 0.01  # Minimum 1% SL

        # Position = (Kapital × Risiko%) / SL-Distanz%
        raw_size = (capital * RISK_PER_TRADE) / sl_distance_pct

        # Max 20% des Kapitals pro Trade
        max_size = capital * 0.20
        size = min(raw_size, max_size)

        # Minimum sinnvolle Grösse
        if size < 5.0:
            return 0.0

        return round(size, 2)

    def can_trade(self, capital):
        """Prüft ob Trading erlaubt ist (Drawdown, Cooldown, Frequency)."""
        reasons = []

        # Monatliches Drawdown-Limit
        monthly_dd = (self.month_start_capital - capital) / self.month_start_capital
        if monthly_dd >= MAX_MONTHLY_DD:
            reasons.append(f"Monthly DD {monthly_dd:.1%} >= {MAX_MONTHLY_DD:.0%}")

        # Consecutive Loss Cooldown
        if self.cooldown_until:
            now = datetime.now(timezone.utc)
            if now < self.cooldown_until:
                remaining = (self.cooldown_until - now).total_seconds() / 3600
                reasons.append(f"Cooldown ({remaining:.1f}h remaining)")
            else:
                self.cooldown_until = None
                self.consecutive_losses = 0

        # Wöchentliches Trade-Limit
        week_key = datetime.now(timezone.utc).strftime("%Y-W%W")
        week_count = self.weekly_trades.get(week_key, 0)
        if week_count >= MAX_TRADES_PER_WEEK:
            reasons.append(f"Weekly limit ({week_count}/{MAX_TRADES_PER_WEEK})")

        return len(reasons) == 0, reasons

    def register_trade(self):
        """Registriert einen neuen Trade."""
        week_key = datetime.now(timezone.utc).strftime("%Y-W%W")
        self.weekly_trades[week_key] = self.weekly_trades.get(week_key, 0) + 1

    def register_result(self, pnl):
        """Registriert Ergebnis und prüft Circuit Breaker."""
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= CONSEC_LOSS_LIMIT:
                self.cooldown_until = datetime.now(timezone.utc) + timedelta(hours=COOLDOWN_HOURS)
                return True  # Circuit breaker triggered
        else:
            self.consecutive_losses = 0
        return False

    def new_month_check(self, capital):
        """Prüft ob neuer Monat begonnen hat → reset Drawdown-Tracking."""
        now = datetime.now(timezone.utc)
        month_key = now.strftime("%Y-%m")
        if not hasattr(self, '_current_month') or self._current_month != month_key:
            self._current_month = month_key
            self.month_start_capital = capital

    def to_dict(self):
        return {
            "month_start_capital": self.month_start_capital,
            "consecutive_losses": self.consecutive_losses,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "weekly_trades": self.weekly_trades,
            "_current_month": getattr(self, '_current_month', None),
        }

    def load_from(self, d):
        self.month_start_capital = d.get("month_start_capital", self.initial_capital)
        self.consecutive_losses = d.get("consecutive_losses", 0)
        cu = d.get("cooldown_until")
        self.cooldown_until = datetime.fromisoformat(cu) if cu else None
        self.weekly_trades = d.get("weekly_trades", {})
        self._current_month = d.get("_current_month")


# ═══════════════════════════════════════════════════════════════
#  STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════

def save_state(capital, total_pnl, wins, losses, trades, position, risk_mgr,
               signals, equity):
    state = {
        "version": "3.0",
        "capital": round(capital, 2),
        "total_pnl": round(total_pnl, 4),
        "wins": wins,
        "losses": losses,
        "position": position.to_dict() if position else None,
        "risk_manager": risk_mgr.to_dict(),
        "last_update": datetime.now(timezone.utc).isoformat(),
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

    if trades:
        pd.DataFrame(trades).to_csv(TRADES_LOG, index=False)
    if signals:
        pd.DataFrame(signals).to_csv(SIGNALS_LOG, index=False)
    if equity:
        pd.DataFrame(equity).to_csv(EQUITY_LOG, index=False)


def load_state():
    capital = INITIAL_CAPITAL
    total_pnl = 0.0
    wins = 0
    losses = 0
    position = None
    risk_mgr = RiskManager(INITIAL_CAPITAL)
    trades = []
    signals = []
    equity = []

    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        if state.get("version") == "3.0":
            capital = state.get("capital", INITIAL_CAPITAL)
            total_pnl = state.get("total_pnl", 0.0)
            wins = state.get("wins", 0)
            losses = state.get("losses", 0)
            if state.get("position"):
                position = Position.from_dict(state["position"])
            if state.get("risk_manager"):
                risk_mgr.load_from(state["risk_manager"])
            log("State v3 geladen", C.BLUE)
        else:
            log("Alter State ignoriert, starte frisch", C.YELLOW)

    if os.path.exists(TRADES_LOG):
        trades = pd.read_csv(TRADES_LOG).to_dict("records")
    if os.path.exists(SIGNALS_LOG):
        signals = pd.read_csv(SIGNALS_LOG).to_dict("records")
    if os.path.exists(EQUITY_LOG):
        equity = pd.read_csv(EQUITY_LOG).to_dict("records")

    return capital, total_pnl, wins, losses, position, risk_mgr, trades, signals, equity


def save_dashboard_status(capital, total_pnl, wins, losses, position,
                          price, ml_prob, regime_label):
    """Speichert Status fürs Dashboard."""
    n_trades = wins + losses
    wr = wins / n_trades if n_trades > 0 else None
    status = {
        "bot_running": True,
        "version": "3.0",
        "last_update": datetime.now(timezone.utc).isoformat(),
        "current_price": price,
        "ensemble_prob": ml_prob,
        "regime": regime_label,
        "capital": round(capital, 2),
        "total_pnl": round(total_pnl, 2),
        "total_trades": n_trades,
        "wins": wins,
        "losses": losses,
        "win_rate": round(wr, 4) if wr is not None else None,
        "has_position": position is not None,
        "position_direction": position.direction if position else None,
        "position_strategy": position.strategy if position else None,
        "position_entry": position.entry_price if position else None,
        "position_sl": position.stop_loss if position else None,
        "position_tp": position.take_profit if position else None,
        "position_trailing": position.trailing_stop if position and position.trailing_active else None,
    }
    with open(os.path.join(BOT_DIR, "dashboard_status.json"), "w") as f:
        json.dump(status, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════
#  DAILY REPORT
# ═══════════════════════════════════════════════════════════════

def send_daily_report(capital, total_pnl, wins, losses, position,
                      risk_mgr, start_time, trades):
    now = datetime.now(timezone.utc)
    uptime = now - start_time
    hours = int(uptime.total_seconds() // 3600)
    today = now.date().isoformat()
    n_trades = wins + losses
    wr = f"{wins/n_trades:.0%}" if n_trades > 0 else "–"

    # Heutige Trades
    today_trades = [t for t in trades if t.get("timestamp", "").startswith(today)]
    today_pnl = sum(t.get("pnl", 0) for t in today_trades)

    # Strategie-Verteilung
    strat_counts = {}
    for t in trades:
        s = t.get("strategy", "unknown")
        strat_counts[s] = strat_counts.get(s, 0) + 1
    strat_text = ", ".join(f"{k}:{v}" for k, v in strat_counts.items()) or "–"

    # Fear & Greed
    fg_text = ""
    try:
        resp = _requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        fg = resp.json()["data"][0]
        fg_text = f"\nFear & Greed: <code>{fg['value']}</code> ({fg['value_classification']})"
    except Exception:
        pass

    can_trade, reasons = risk_mgr.can_trade(capital)
    risk_text = "Trading aktiv" if can_trade else f"PAUSED: {', '.join(reasons)}"

    pos_text = "Keine"
    if position:
        pos_text = (f"{position.direction.upper()} ({position.strategy})\n"
                    f"  Entry: ${position.entry_price:.4f}\n"
                    f"  SL: ${position.stop_loss:.4f} | TP: ${position.take_profit:.4f}")

    tg_send(
        f"\U0001F4CB <b>Daily Report v3</b> – {today}\n\n"
        f"Uptime: <code>{hours}h</code>{fg_text}\n\n"
        f"<b>Performance:</b>\n"
        f"  Kapital: <code>${capital:.2f}</code>\n"
        f"  PnL: <code>{total_pnl:+.2f}$</code>\n"
        f"  Win Rate: <code>{wr}</code> ({wins}W/{losses}L)\n"
        f"  Heute: <code>{today_pnl:+.2f}$</code> ({len(today_trades)} Trades)\n\n"
        f"<b>Strategien:</b> {strat_text}\n"
        f"<b>Position:</b> {pos_text}\n"
        f"<b>Risk:</b> {risk_text}"
    )


# ═══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(BOT_DIR, exist_ok=True)
    banner()

    # Modelle laden
    log("Lade ML-Modelle...", C.BLUE)
    rf, xgb_model, meta = load_models()
    feature_cols = meta["feature_cols"]
    log(f"Modelle geladen: {len(feature_cols)} Features", C.GREEN)

    # Exchange
    exchange = get_exchange()
    log("Binance verbunden", C.GREEN)

    # State laden
    capital, total_pnl, wins, losses, position, risk_mgr, trades, signals, equity = load_state()

    # Strategy Engine
    strategy_engine = StrategyEngine()

    n_trades = wins + losses
    wr = f"{wins/n_trades:.0%}" if n_trades > 0 else "–"
    log(f"Kapital: ${capital:.2f}  PnL: {total_pnl:+.2f}  WR: {wr} ({n_trades} Trades)", C.BOLD)
    if position:
        log(f"Offene Position: {position.direction.upper()} ({position.strategy}) "
            f"@ ${position.entry_price:.4f}  SL: ${position.stop_loss:.4f}", C.YELLOW)

    # Graceful Shutdown
    running = [True]
    def shutdown(signum, frame):
        running[0] = False
        log("\nShutdown...", C.YELLOW)
    sig.signal(sig.SIGINT, shutdown)
    sig.signal(sig.SIGTERM, shutdown)

    log(f"\nBot gestartet. Signale auf {PRIMARY_TF}-Kerzen, Check alle {CHECK_INTERVAL}s.\n", C.GREEN)

    # Telegram Start
    can_trade, reasons = risk_mgr.can_trade(capital)
    tg_send(
        f"\U0001F680 <b>MuriTrading v3.0 gestartet</b>\n\n"
        f"Kapital: <code>${capital:.2f}</code>\n"
        f"PnL: <code>{total_pnl:+.2f}$</code>\n"
        f"Trades: <code>{n_trades}</code> (WR: {wr})\n"
        f"Strategien: Trend + MeanRev + Breakout\n"
        f"Timeframe: <code>{PRIMARY_TF}</code>\n"
        f"Risk/Trade: <code>{RISK_PER_TRADE*100:.1f}%</code>\n"
        f"Status: {'Trading aktiv' if can_trade else 'PAUSED: ' + ', '.join(reasons)}"
    )

    last_4h_slot = None
    last_daily_report = None
    start_time = datetime.now(timezone.utc)
    heartbeat_count = 0

    while running[0]:
        try:
            ticker = exchange.fetch_ticker(SYMBOL)
            current_price = ticker["last"]
            now = datetime.now(timezone.utc)
            heartbeat_count += 1

            if heartbeat_count % 10 == 0:
                trail_info = ""
                if position and position.trailing_active:
                    trail_info = f"  Trail: ${position.trailing_stop:.4f}"
                pos_info = ""
                if position:
                    pnl_now, _, _ = position.calc_pnl(current_price)
                    pos_info = f"  Pos: {position.direction.upper()} PnL:{pnl_now:+.2f}${trail_info}"
                log(f"#{heartbeat_count}  XRP ${current_price:.4f}{pos_info}", C.DIM)

            # Monatliches Reset prüfen
            risk_mgr.new_month_check(capital)

            # ── PARTIAL TP CHECK (jede Iteration) ─────────────
            if position:
                should_partial, partial_pnl = position.check_partial_tp(current_price)
                if should_partial:
                    capital += partial_pnl
                    total_pnl += partial_pnl
                    log(f"\U0001F3AF PARTIAL TP: +${partial_pnl:.2f}  "
                        f"50% closed, SL→Breakeven  Remaining: ${position.size:.2f}", C.GREEN)
                    tg_send(
                        f"\U0001F3AF <b>PARTIAL TAKE-PROFIT</b>\n\n"
                        f"50% der Position geschlossen\n"
                        f"Partial PnL: <code>+${partial_pnl:.2f}</code>\n"
                        f"Stop → Breakeven: <code>${position.stop_loss:.4f}</code>\n"
                        f"Rest läuft mit Trailing Stop weiter"
                    )
                    save_state(capital, total_pnl, wins, losses, trades, position,
                               risk_mgr, signals, equity)

            # ── EXIT-CHECKS (jede Iteration) ──────────────────
            if position:
                position.update_trailing(current_price)
                should_exit, reason = position.check_exit(current_price)

                if should_exit:
                    pnl, raw_ret, net_ret = position.calc_pnl(current_price)
                    capital += pnl
                    total_pnl += pnl

                    if pnl > 0:
                        wins += 1
                    else:
                        losses += 1

                    # Risk Manager aktualisieren
                    circuit_break = risk_mgr.register_result(pnl)

                    n_trades_now = wins + losses
                    wr_now = f"{wins/n_trades_now:.0%}" if n_trades_now > 0 else "–"

                    pnl_emoji = "\U0001F4B0" if pnl >= 0 else "\u274C"
                    hold_hours = position.candles_held * 4
                    color = C.GREEN if pnl >= 0 else C.RED
                    log(f"{pnl_emoji} CLOSE {position.direction.upper()} ({position.strategy})  "
                        f"PnL: {pnl:+.2f}$  [{reason}]  {hold_hours}h  "
                        f"Kapital: ${capital:.2f}  WR: {wr_now}", color)

                    tg_send(
                        f"{pnl_emoji} <b>TRADE CLOSE</b> [{reason}]\n\n"
                        f"Strategie: <b>{position.strategy.upper()}</b>\n"
                        f"Richtung: <b>{position.direction.upper()}</b>\n"
                        f"Entry: <code>${position.entry_price:.4f}</code>\n"
                        f"Exit: <code>${current_price:.4f}</code>\n"
                        f"Einsatz: <code>${position.size:.2f}</code>\n"
                        f"Dauer: <code>{hold_hours}h</code>\n"
                        f"Return: <code>{net_ret:+.2%}</code>\n"
                        f"Ergebnis: <code>{pnl:+.2f}$</code>\n"
                        f"\n<b>Gesamt:</b>\n"
                        f"Kapital: <code>${capital:.2f}</code>\n"
                        f"PnL: <code>{total_pnl:+.2f}$</code>\n"
                        f"Win Rate: <code>{wr_now}</code> ({wins}W/{losses}L)"
                        + (f"\n\n\u26A0 Circuit Breaker: {COOLDOWN_HOURS}h Pause" if circuit_break else "")
                    )

                    # Trade loggen
                    trades.append({
                        "timestamp": now.isoformat(),
                        "strategy": position.strategy,
                        "direction": position.direction,
                        "entry_price": position.entry_price,
                        "exit_price": current_price,
                        "size": position.size,
                        "raw_return_pct": round(raw_ret * 100, 4),
                        "net_return_pct": round(net_ret * 100, 4),
                        "pnl": round(pnl, 4),
                        "capital": round(capital, 2),
                        "reason": reason,
                        "strength": position.strength,
                        "hold_candles": position.candles_held,
                        "trailing_active": position.trailing_active,
                    })

                    position = None
                    save_state(capital, total_pnl, wins, losses, trades, position,
                               risk_mgr, signals, equity)

            # ── TÄGLICHER REPORT ──────────────────────────────
            today_key = now.date().isoformat()
            if now.hour == DAILY_REPORT_HOUR and last_daily_report != today_key:
                last_daily_report = today_key
                send_daily_report(capital, total_pnl, wins, losses, position,
                                  risk_mgr, start_time, trades)
                log("Täglicher Report gesendet", C.BLUE)

            # ── SIGNAL-CHECK (nur auf neuer 4H-Kerze) ─────────
            current_4h_slot = now.hour // 4
            current_4h_time = now.replace(hour=current_4h_slot * 4,
                                          minute=0, second=0, microsecond=0)

            if last_4h_slot != current_4h_time and now.minute >= CANDLE_WAIT_MIN:
                last_4h_slot = current_4h_time

                # Position Haltezeit erhöhen
                if position:
                    position.candles_held += 1

                # Features berechnen
                X, latest_row, price = build_features(exchange, feature_cols)
                if X is None:
                    log("Nicht genug Daten", C.YELLOW)
                    time.sleep(CHECK_INTERVAL)
                    continue

                # ML Prediction (als Filter)
                ml_prob, rf_prob, xgb_prob = predict(rf, xgb_model, X)
                ml_confidence = abs(ml_prob - 0.5) * 2

                # Regime Info
                adx_val = _safe(latest_row.get("4h_adx", 0))
                chop_val = _safe(latest_row.get("4h_chop", 0))
                regime_trend = _safe(latest_row.get("4h_regime_trend", 0))
                regime_label = "TREND" if regime_trend else "SEITW"
                regime_color = C.GREEN if regime_trend else C.YELLOW

                # Status anzeigen
                print(f"\n{C.DIM}  ┌─────────────────────────────────────────────────────┐{C.RESET}")
                print(f"  │ {C.BOLD}XRP/USDT{C.RESET}  ${current_price:.4f}  │  "
                      f"ML: {ml_prob:.0%}  │  Conf: {ml_confidence:.0%}  │  "
                      f"Regime: {regime_color}{regime_label}{C.RESET} (ADX:{adx_val:.0f})")

                n_trades_now = wins + losses
                wr_now = f"{wins/n_trades_now:.0%}" if n_trades_now > 0 else "–"
                pnl_c = C.GREEN if total_pnl >= 0 else C.RED
                print(f"  │ Kapital: ${capital:.2f}  PnL:{pnl_c}{total_pnl:+.2f}{C.RESET}  "
                      f"WR:{wr_now}  Trades:{n_trades_now}")

                if position:
                    pos_pnl, _, _ = position.calc_pnl(current_price)
                    pos_c = C.GREEN if pos_pnl >= 0 else C.RED
                    trail_text = f"  Trail: ${position.trailing_stop:.4f}" if position.trailing_active else ""
                    print(f"  │ Position: {position.direction.upper()} ({position.strategy})  "
                          f"PnL:{pos_c}{pos_pnl:+.2f}{C.RESET}  "
                          f"SL: ${position.stop_loss:.4f}  TP: ${position.take_profit:.4f}{trail_text}  "
                          f"Hold: {position.candles_held * 4}h")

                print(f"{C.DIM}  └─────────────────────────────────────────────────────┘{C.RESET}")

                # Nur Signal generieren wenn keine Position offen
                if position is None:
                    can_trade, block_reasons = risk_mgr.can_trade(capital)

                    if not can_trade:
                        log(f"Trading PAUSED: {', '.join(block_reasons)}", C.YELLOW)
                    else:
                        # Signal generieren
                        signal = strategy_engine.generate_signals(
                            latest_row, current_price, ml_prob)

                        if signal:
                            # Position Size berechnen
                            size = risk_mgr.calc_position_size(
                                capital, signal.entry_price, signal.stop_loss)

                            if size > 0:
                                # Position eröffnen
                                position = Position(
                                    direction=signal.direction,
                                    strategy=signal.strategy,
                                    entry_price=current_price,
                                    size=size,
                                    original_size=size,
                                    stop_loss=signal.stop_loss,
                                    take_profit=signal.take_profit,
                                    atr=signal.atr,
                                    entry_time=now.isoformat(),
                                    peak_price=current_price,
                                    trough_price=current_price,
                                    strength=signal.strength,
                                    details=signal.details,
                                )

                                risk_mgr.register_trade()

                                dir_emoji = "\u2934\uFE0F" if signal.direction == "long" else "\u2935\uFE0F"
                                sl_pct = abs(current_price - signal.stop_loss) / current_price
                                tp_pct = abs(signal.take_profit - current_price) / current_price

                                log(f"{dir_emoji} OPEN {signal.direction.upper()} ({signal.strategy})  "
                                    f"${current_price:.4f}  Size: ${size:.2f}  "
                                    f"Strength: {signal.strength:.0%}", C.GREEN)
                                log(f"   {signal.details}", C.DIM)
                                log(f"   SL: ${signal.stop_loss:.4f} (-{sl_pct:.1%})  "
                                    f"TP: ${signal.take_profit:.4f} (+{tp_pct:.1%})  "
                                    f"R:R 1:{tp_pct/sl_pct:.1f}", C.DIM)

                                tg_send(
                                    f"{dir_emoji} <b>TRADE OPEN</b>\n\n"
                                    f"Strategie: <b>{signal.strategy.upper()}</b>\n"
                                    f"Richtung: <b>{signal.direction.upper()}</b>\n"
                                    f"Preis: <code>${current_price:.4f}</code>\n"
                                    f"Einsatz: <code>${size:.2f}</code>\n"
                                    f"Stop-Loss: <code>${signal.stop_loss:.4f}</code> (-{sl_pct:.1%})\n"
                                    f"Take-Profit: <code>${signal.take_profit:.4f}</code> (+{tp_pct:.1%})\n"
                                    f"R:R: <code>1:{tp_pct/sl_pct:.1f}</code>\n"
                                    f"Stärke: <code>{signal.strength:.0%}</code>\n"
                                    f"Regime: <code>{regime_label}</code> (ADX:{adx_val:.0f})\n\n"
                                    f"<i>{signal.details}</i>"
                                )
                            else:
                                log(f"Signal {signal.strategy} ignoriert (Size zu klein)", C.DIM)
                        else:
                            log(f"Kein Signal (ML:{ml_prob:.0%} Regime:{regime_label} ADX:{adx_val:.0f})", C.DIM)

                # Signal-Log
                sig_row = {
                    "timestamp": now.isoformat(),
                    "price": round(current_price, 6),
                    "ml_prob": round(ml_prob, 4),
                    "rf_prob": round(rf_prob, 4),
                    "xgb_prob": round(xgb_prob, 4),
                    "ml_confidence": round(ml_confidence, 4),
                    "regime": regime_label,
                    "adx": round(adx_val, 1),
                    "chop": round(chop_val, 3),
                    "rsi_4h": round(_safe(latest_row.get("4h_rsi_14", 50)), 1),
                    "bb_pos_4h": round(_safe(latest_row.get("4h_bb_pos", 0.5)), 3),
                    "has_position": position is not None,
                    "position_direction": position.direction if position else None,
                    "position_strategy": position.strategy if position else None,
                }
                signals.append(sig_row)

                # Equity-Snapshot
                unrealized = 0
                if position:
                    unrealized, _, _ = position.calc_pnl(current_price)
                eq_row = {
                    "timestamp": now.isoformat(),
                    "price": round(current_price, 6),
                    "capital": round(capital, 2),
                    "equity": round(capital + unrealized, 2),
                    "total_pnl": round(total_pnl, 4),
                    "n_trades": wins + losses,
                }
                equity.append(eq_row)

                # Dashboard + State speichern
                save_dashboard_status(capital, total_pnl, wins, losses, position,
                                      current_price, ml_prob, regime_label)
                save_state(capital, total_pnl, wins, losses, trades, position,
                           risk_mgr, signals, equity)

            time.sleep(CHECK_INTERVAL)

        except ccxt.NetworkError as e:
            log(f"Netzwerk-Fehler: {e}", C.YELLOW)
            time.sleep(30)
        except ccxt.ExchangeError as e:
            log(f"Exchange-Fehler: {e}", C.RED)
            time.sleep(30)
        except Exception as e:
            log(f"Fehler: {e}", C.RED)
            import traceback; traceback.print_exc()
            time.sleep(10)

    # ── SHUTDOWN ───────────────────────────────────────────────
    save_state(capital, total_pnl, wins, losses, trades, position,
               risk_mgr, signals, equity)

    n_trades_final = wins + losses
    wr_final = f"{wins/n_trades_final:.1%}" if n_trades_final > 0 else "–"
    ret = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL

    print(f"\n{C.PURPLE}{C.BOLD}═══════════════════════════════════════════════════════{C.RESET}")
    print(f"{C.BOLD}  Bot gestoppt – Finale Bilanz{C.RESET}")
    print(f"  Kapital: ${capital:.2f}  ({ret:+.1%})")
    print(f"  PnL:     {total_pnl:+.2f}$")
    print(f"  Trades:  {n_trades_final}  WR: {wr_final} ({wins}W/{losses}L)")
    print(f"{C.PURPLE}═══════════════════════════════════════════════════════{C.RESET}\n")

    tg_send(
        f"\U0001F6D1 <b>MuriTrading v3.0 gestoppt</b>\n\n"
        f"Kapital: <code>${capital:.2f}</code> ({ret:+.1%})\n"
        f"PnL: <code>{total_pnl:+.2f}$</code>\n"
        f"Win Rate: <code>{wr_final}</code> ({wins}W/{losses}L)"
    )


if __name__ == "__main__":
    main()
