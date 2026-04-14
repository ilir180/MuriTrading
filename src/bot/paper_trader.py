"""
MuriTrading – Phase 5: Paper-Trading Bot (Multi-Pool)
Drei Töpfe mit unterschiedlichen Confidence-Leveln laufen parallel:
  - Konservativ (75%): wenige Trades, höchste Accuracy
  - Standard (65%):    ausbalanciert
  - Aggressiv (50%):   viele Trades, niedrigere Accuracy

Start: python src/bot/paper_trader.py
Stop:  Ctrl+C (speichert automatisch)
"""

# Fix PyTorch/OpenMP + sklearn threading conflict (segfault prevention)
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
import signal as sig
import requests as _requests
from datetime import datetime, timezone, timedelta
# PPO wird lazy geladen um Speicher-Crash zu vermeiden

# ── Projekt-Root ───────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser("~/MuriTrading")
sys.path.insert(0, PROJECT_ROOT)
from src.features.build_features import add_indicators
from src.features.whale_features import compute_whale_features, whale_signal_text
from src.features.cross_asset import build_cross_asset_features, cross_asset_signal_text
from src.features.sentiment import compute_sentiment_features, sentiment_signal_text

# ── Pfade ──────────────────────────────────────────────────────
MODEL_DIR   = os.path.join(PROJECT_ROOT, "models")
BOT_DIR     = os.path.join(PROJECT_ROOT, "data", "bot")
SIGNALS_LOG = os.path.join(BOT_DIR, "signals.csv")

# ── Trading-Parameter ─────────────────────────────────────────
SYMBOL           = "XRP/USDT"
INITIAL_CAPITAL  = 1000.0       # Pro Topf
RISK_PER_TRADE   = 0.01
MAX_TRADES_PER_DAY = 4
TAKER_FEE        = 0.0004
SLIPPAGE         = 0.0002
ROUND_TRIP       = (TAKER_FEE + SLIPPAGE) * 2
STOP_LOSS_MULT   = 1.5
HOLD_CANDLES     = 2

# ── Hebel (simuliert, Confidence-gestaffelt) ─────────────────
# Regime-Trend → voller Hebel erlaubt, Seitwärts → max 2x
LEVERAGE_TIERS = [
    (0.80, 8),   # Conf ≥ 80% → 8x
    (0.70, 5),   # Conf ≥ 70% → 5x
    (0.60, 3),   # Conf ≥ 60% → 3x
    (0.00, 1),   # Darunter   → 1x (kein Hebel)
]
LEVERAGE_SIDEWAYS_CAP = 2  # Max Hebel im Seitwärtsmarkt
CHECK_INTERVAL   = 60
DAILY_REPORT_HOUR= 22          # UTC - täglicher Report
RETRAIN_WINDOW   = 50          # Letzte N Signale für Performance-Check
RETRAIN_THRESHOLD= 0.60        # Unter dieser Accuracy → Warnung

# ── Die vier Töpfe ────────────────────────────────────────────
POOLS = {
    "konservativ": {"confidence": 0.75, "emoji": "\U0001F9CA", "label": "Konservativ", "type": "ml"},
    "standard":    {"confidence": 0.65, "emoji": "\U0001F4CA", "label": "Standard",    "type": "ml"},
    "aggressiv":   {"confidence": 0.50, "emoji": "\U0001F525", "label": "Aggressiv",   "type": "ml"},
    "rl_agent":    {"confidence": 0.0,  "emoji": "\U0001F916", "label": "Predictlir RL", "type": "rl"},
}

# ── RL Modell ─────────────────────────────────────────────────
RL_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "rl", "ppo_xrp")

# ── Telegram ──────────────────────────────────────────────────
TG_BOT_TOKEN = "8503143803:AAH-7DPWX-bXq-ITRGpw4TwkDTDtIsRzQt8"
TG_CHAT_ID   = "7704168743"


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

def send_daily_report(pools, signals, start_time):
    """Sendet tägliche Zusammenfassung per Telegram."""
    now = datetime.now(timezone.utc)
    uptime = now - start_time
    hours = int(uptime.total_seconds() // 3600)
    today = now.date().isoformat()

    # Stärkstes Signal des Tages
    today_signals = [s for s in signals if s.get("timestamp", "").startswith(today)]
    max_conf = max((s.get("confidence", 0) for s in today_signals), default=0)
    n_signals = len(today_signals)

    # Fear & Greed live holen
    fg_text = ""
    try:
        resp = _requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
        fg = resp.json()["data"][0]
        fg_text = f"\nFear & Greed: <code>{fg['value']}</code> ({fg['value_classification']})"
    except Exception:
        pass

    # Pool-Status
    pool_lines = ""
    total_pnl = 0
    for name, pool in pools.items():
        cfg = POOLS[name]
        wr = f"{pool.win_rate:.0%}" if pool.n_trades > 0 else "–"
        today_trades = len([t for t in pool.trades if t.get("timestamp", "").startswith(today)])
        total_pnl += pool.total_pnl
        pool_lines += (
            f"{cfg['emoji']} <b>{cfg['label']}</b>: "
            f"<code>${pool.capital:.2f}</code> "
            f"({pool.total_pnl:+.2f}$) "
            f"WR:{wr} "
            f"Trades heute: {today_trades}\n"
        )

    tg_send(
        f"\U0001F4CB <b>Täglicher Report</b> – {today}\n\n"
        f"Uptime: <code>{hours}h</code>\n"
        f"Signale heute: <code>{n_signals}</code>\n"
        f"Stärkstes Signal: <code>{max_conf:.0%}</code>\n"
        f"{fg_text}\n\n"
        f"{pool_lines}\n"
        f"Gesamt PnL: <code>{total_pnl:+.2f}$</code>"
    )


def check_retrain_needed(signals, pools):
    """Prüft ob das Modell schlecht performt und Retrain nötig ist."""
    if len(signals) < RETRAIN_WINDOW:
        return

    recent = signals[-RETRAIN_WINDOW:]
    # Nur Signale mit Trade prüfen (aggressiv hat die meisten)
    traded = [s for s in recent if s.get("aggressiv_traded", False)]
    if len(traded) < 20:
        return

    # Vergleiche Signal-Richtung mit tatsächlicher Preisbewegung
    correct = 0
    total = 0
    for i, s in enumerate(traded):
        if i + 2 >= len(signals):
            break
        # Finde den Preis 2 Signale später
        future_idx = signals.index(s) + 2
        if future_idx >= len(signals):
            break
        future_price = signals[future_idx].get("price", 0)
        entry_price = s.get("price", 0)
        if entry_price == 0 or future_price == 0:
            continue

        actual_dir = "long" if future_price > entry_price else "short"
        predicted_dir = s.get("aggressiv_signal", "neutral")
        if predicted_dir == "neutral":
            continue

        total += 1
        if predicted_dir == actual_dir:
            correct += 1

    if total >= 15:
        accuracy = correct / total
        if accuracy < RETRAIN_THRESHOLD:
            tg_send(
                f"\u26A0\uFE0F <b>Retrain-Warnung</b>\n\n"
                f"Modell-Accuracy der letzten {total} Signale: "
                f"<code>{accuracy:.0%}</code>\n"
                f"Schwelle: <code>{RETRAIN_THRESHOLD:.0%}</code>\n\n"
                f"Empfehlung: Modell neu trainieren mit frischen Daten."
            )
            return True
    return False


def save_dashboard_status(pools, signals, ensemble_prob=None, confidence=None, price=None):
    """Speichert Bot-Status als JSON fürs Dashboard."""
    now = datetime.now(timezone.utc)
    today = now.date().isoformat()

    status = {
        "bot_running": True,
        "last_update": now.isoformat(),
        "current_price": price,
        "ensemble_prob": ensemble_prob,
        "confidence": confidence,
        "pools": {},
    }
    for name, pool in pools.items():
        cfg = POOLS[name]
        today_trades = [t for t in pool.trades if t.get("timestamp", "").startswith(today)]
        today_pnl = sum(t.get("pnl", 0) for t in today_trades)
        wr = pool.win_rate if pool.n_trades > 0 else None
        status["pools"][name] = {
            "label": cfg["label"],
            "emoji": cfg["emoji"],
            "capital": round(pool.capital, 2),
            "total_pnl": round(pool.total_pnl, 2),
            "today_pnl": round(today_pnl, 2),
            "today_trades": len(today_trades),
            "total_trades": pool.n_trades,
            "wins": pool.wins,
            "losses": pool.losses,
            "win_rate": round(wr, 4) if wr is not None else None,
            "max_dd": round(pool.max_dd, 4),
            "open_positions": len(pool.open_positions),
        }

    status_file = os.path.join(BOT_DIR, "dashboard_status.json")
    with open(status_file, "w") as f:
        json.dump(status, f, indent=2, default=str)


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
{C.BOLD}  MuriTrading – Paper-Trading Bot v2.0 (Multi-Pool){C.RESET}
{C.PURPLE}═══════════════════════════════════════════════════════{C.RESET}
{C.DIM}  Asset          : {SYMBOL}
  Kapital/Topf   : ${INITIAL_CAPITAL:,.2f} x 4 = ${INITIAL_CAPITAL*4:,.2f}
  Risiko/Trade   : {RISK_PER_TRADE*100:.1f}%
  Haltezeit      : {HOLD_CANDLES}h
  Fees (RT)      : {ROUND_TRIP*100:.3f}%{C.RESET}

  {C.BLUE}Konservativ{C.RESET}  ML Conf >75%  (wenige, sichere Trades)
  {C.GREEN}Standard{C.RESET}     ML Conf >65%  (ausbalanciert)
  {C.RED}Aggressiv{C.RESET}    ML Conf >50%  (viele Trades)
  {C.PURPLE}Predictlir{C.RESET}   RL Agent     (lernt selbst wann und wie viel)
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
    df_15m = fetch_candles(exchange, "15m", 200)
    df_1h  = fetch_candles(exchange, "1h", 250)
    df_4h  = fetch_candles(exchange, "4h", 250)
    df_1d  = fetch_candles(exchange, "1d", 250)
    df_15m = add_indicators(df_15m, prefix="15m_")
    df_1h  = add_indicators(df_1h,  prefix="1h_")
    df_4h  = add_indicators(df_4h,  prefix="4h_")
    df_1d  = add_indicators(df_1d,  prefix="1d_")

    df_15m_sel = df_15m[[c for c in df_15m.columns if c.startswith("15m_")]].copy()
    df_base = pd.merge_asof(df_1h.sort_index(), df_15m_sel.sort_index(),
        left_index=True, right_index=True, direction="backward")
    df_4h_sel = df_4h[[c for c in df_4h.columns if c.startswith("4h_")]].copy()
    df_base = pd.merge_asof(df_base.sort_index(), df_4h_sel.sort_index(),
        left_index=True, right_index=True, direction="backward")
    df_1d_sel = df_1d[[c for c in df_1d.columns if c.startswith("1d_")]].copy()
    df_base = pd.merge_asof(df_base.sort_index(), df_1d_sel.sort_index(),
        left_index=True, right_index=True, direction="backward")
    bull_signals = ["1h_ema_9_above_21","1h_ema_21_above_50","1h_macd_above",
        "4h_ema_9_above_21","4h_ema_21_above_50","4h_macd_above",
        "1d_ema_9_above_21","1d_ema_21_above_50","1d_macd_above"]
    bear_signals = ["1h_rsi_overbought","4h_rsi_overbought","1d_rsi_overbought"]
    avail_bull = [s for s in bull_signals if s in df_base.columns]
    avail_bear = [s for s in bear_signals if s in df_base.columns]
    if avail_bull: df_base["confluence_bull"] = df_base[avail_bull].sum(axis=1)
    if avail_bear: df_base["confluence_bear"] = df_base[avail_bear].sum(axis=1)
    df_base["confluence_net"] = df_base.get("confluence_bull", 0) - df_base.get("confluence_bear", 0)

    # Cross-Asset Features (BTC/ETH) live berechnen falls Modell sie braucht
    ca_cols_needed = [c for c in feature_cols if c.startswith("ca_")]
    if ca_cols_needed:
        try:
            from src.features.cross_asset import build_cross_asset_features_batch, _fetch_candles
            btc_15m = _fetch_candles(exchange, "BTC/USDT", "15m", 200)
            eth_15m = _fetch_candles(exchange, "ETH/USDT", "15m", 200)
            btc_1h  = _fetch_candles(exchange, "BTC/USDT", "1h",  200)
            eth_1h  = _fetch_candles(exchange, "ETH/USDT", "1h",  200)
            ca_df = build_cross_asset_features_batch(df_15m, btc_15m, eth_15m, btc_1h, eth_1h)
            ca_sel = ca_df[[c for c in ca_df.columns if c in ca_cols_needed]].copy()
            df_base = pd.merge_asof(
                df_base.sort_index(), ca_sel.sort_index(),
                left_index=True, right_index=True, direction="backward",
            )
            for col in ca_sel.columns:
                df_base[col] = df_base[col].ffill()
        except Exception as e:
            for col in ca_cols_needed:
                df_base[col] = 0.0

    latest = df_base[feature_cols].dropna()
    if latest.empty:
        return None, None, None
    return latest.iloc[[-1]], df_base.iloc[-1], df_1h["close"].iloc[-1]


def predict(rf, xgb, X):
    rf_prob  = rf.predict_proba(X)[:, 1]
    xgb_prob = xgb.predict_proba(X)[:, 1]
    ensemble = (rf_prob + xgb_prob) / 2.0
    return float(ensemble[0]), float(rf_prob[0]), float(xgb_prob[0])


def calc_leverage(confidence, is_trending, whale_confirmed=False):
    """
    Bestimmt den Hebel basierend auf Confidence, Regime und Whale-Bestätigung.

    Args:
        confidence: 0-1, ML Confidence
        is_trending: bool, ob der Markt im Trend ist
        whale_confirmed: bool, ob Whale-Flow die Richtung bestätigt

    Returns:
        int: Hebel (1-8x)
    """
    # Basis-Hebel nach Confidence
    leverage = 1
    for min_conf, lev in LEVERAGE_TIERS:
        if confidence >= min_conf:
            leverage = lev
            break

    # Seitwärtsmarkt → Hebel deckeln
    if not is_trending:
        leverage = min(leverage, LEVERAGE_SIDEWAYS_CAP)

    # Whale-Bestätigung → +1x Bonus (max 10x)
    if whale_confirmed and leverage >= 3:
        leverage = min(leverage + 1, 10)

    return leverage


# ═══════════════════════════════════════════════════════════════
#  POOL (ein Topf mit eigenem Kapital und Trades)
# ═══════════════════════════════════════════════════════════════

class Pool:
    def __init__(self, name, confidence_thresh):
        self.name = name
        self.conf = confidence_thresh
        self.capital = INITIAL_CAPITAL
        self.peak = INITIAL_CAPITAL
        self.max_dd = 0.0
        self.trades = []
        self.open_positions = []
        self.daily_trades = {}
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0

    @property
    def win_rate(self):
        t = self.wins + self.losses
        return self.wins / t if t > 0 else 0.0

    @property
    def drawdown(self):
        return (self.peak - self.capital) / self.peak if self.peak > 0 else 0.0

    @property
    def n_trades(self):
        return self.wins + self.losses

    def to_dict(self):
        return {
            "name": self.name, "confidence": self.conf,
            "capital": round(self.capital, 2), "peak": round(self.peak, 2),
            "max_dd": round(self.max_dd, 4), "total_pnl": round(self.total_pnl, 4),
            "wins": self.wins, "losses": self.losses,
            "open_positions": self.open_positions,
        }

    def load_from(self, d):
        self.capital = d.get("capital", INITIAL_CAPITAL)
        self.peak = d.get("peak", INITIAL_CAPITAL)
        self.max_dd = d.get("max_dd", 0.0)
        self.total_pnl = d.get("total_pnl", 0.0)
        self.wins = d.get("wins", 0)
        self.losses = d.get("losses", 0)
        self.open_positions = d.get("open_positions", [])

    def check_signal(self, ensemble_prob):
        """Prüft ob Signal stark genug für diesen Topf."""
        conf_prob = 0.5 + self.conf / 2
        is_long  = ensemble_prob >= conf_prob
        is_short = ensemble_prob <= (1 - conf_prob)
        return is_long, is_short

    def open_trade(self, direction, price, confidence, atr_rel, leverage=1):
        today = datetime.now(timezone.utc).date().isoformat()
        self.daily_trades[today] = self.daily_trades.get(today, 0)
        if self.daily_trades[today] >= MAX_TRADES_PER_DAY:
            return None

        stop_loss_pct = max(atr_rel * STOP_LOSS_MULT, 0.001)
        size = (self.capital * RISK_PER_TRADE) / stop_loss_pct
        size = min(size, self.capital * 0.20)

        # Hebel anwenden
        size *= leverage

        # Stop-Loss enger bei Hebel (SL-Distanz / Hebel damit Verlust gleich bleibt)
        effective_sl_pct = stop_loss_pct / leverage if leverage > 1 else stop_loss_pct
        sl = price * (1 - effective_sl_pct) if direction == "long" else price * (1 + effective_sl_pct)

        pos = {
            "direction": direction, "entry_price": price,
            "size": round(size, 2), "entry_time": datetime.now(timezone.utc).isoformat(),
            "candles_held": 0, "stop_loss": round(sl, 6),
            "confidence": round(confidence, 4), "pool": self.name,
            "leverage": leverage,
        }
        self.open_positions.append(pos)
        self.daily_trades[today] += 1
        return pos

    def check_exits(self, current_price):
        """Prüft Stop-Loss, Take-Profit und Haltezeit."""
        closed = []
        for pos in self.open_positions:
            pos["candles_held"] = pos.get("candles_held", 0) + 1

            hit_sl = (pos["direction"] == "long" and current_price <= pos["stop_loss"]) or \
                     (pos["direction"] == "short" and current_price >= pos["stop_loss"])

            # Take-Profit: 2x die SL-Distanz
            sl_dist = abs(pos["entry_price"] - pos["stop_loss"])
            if pos["direction"] == "long":
                tp = pos["entry_price"] + sl_dist * 2.0
                hit_tp = current_price >= tp
            else:
                tp = pos["entry_price"] - sl_dist * 2.0
                hit_tp = current_price <= tp

            if hit_sl:
                pnl = self._close(pos, current_price, "stop_loss")
                closed.append((pos, pnl, "STOP-LOSS"))
            elif hit_tp:
                pnl = self._close(pos, current_price, "take_profit")
                closed.append((pos, pnl, "TAKE-PROFIT \U0001F3AF"))
            elif pos["candles_held"] >= HOLD_CANDLES:
                pnl = self._close(pos, current_price, "time_exit")
                closed.append((pos, pnl, "TIME-EXIT"))

        for pos, _, _ in closed:
            if pos in self.open_positions:
                self.open_positions.remove(pos)
        return closed

    def _close(self, pos, exit_price, reason):
        if pos["direction"] == "long":
            raw_ret = (exit_price - pos["entry_price"]) / pos["entry_price"]
        else:
            raw_ret = (pos["entry_price"] - exit_price) / pos["entry_price"]

        net_ret = raw_ret - ROUND_TRIP
        pnl = pos["size"] * net_ret

        self.capital += pnl
        self.total_pnl += pnl
        if pnl > 0: self.wins += 1
        else: self.losses += 1

        if self.capital > self.peak: self.peak = self.capital
        dd = self.drawdown
        if dd > self.max_dd: self.max_dd = dd

        self.trades.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "pool": self.name, "direction": pos["direction"],
            "entry_price": pos["entry_price"], "exit_price": exit_price,
            "size": pos["size"],
            "raw_return_pct": round(raw_ret * 100, 4),
            "net_return_pct": round(net_ret * 100, 4),
            "pnl": round(pnl, 4), "capital": round(self.capital, 2),
            "reason": reason, "confidence": pos.get("confidence", 0),
        })
        return pnl


class RLPool(Pool):
    """RL Agent Pool v2 - entscheidet selbst, regime-aware."""

    # Regime-Feature-Spalten (gleich wie im Environment)
    REGIME_COLS = ["1h_adx", "1h_chop", "1h_vol_regime", "1h_bb_squeeze",
                   "1h_trend_consistency", "1h_regime_trend"]

    def __init__(self, name, model_path, feature_cols):
        super().__init__(name, confidence_thresh=0.0)
        from stable_baselines3 import PPO
        self.rl_model = PPO.load(model_path)
        self.feature_cols = feature_cols
        self.position_state = 0.0
        self.entry_price_rl = 0.0

        # Feature-Normalisierung aus vorberechneter JSON
        stats_path = os.path.join(os.path.dirname(model_path), "feature_stats.json")
        with open(stats_path) as f:
            stats = json.load(f)
        self.feat_mean = np.array(stats["mean"])
        self.feat_std = np.array(stats["std"])

        # Regime-Stats laden (v2)
        self.regime_cols = stats.get("regime_cols", [])
        if self.regime_cols:
            self.regime_mean = np.array(stats["regime_mean"])
            self.regime_std = np.array(stats["regime_std"])
            log(f"  Regime-Features geladen: {self.regime_cols}", C.PURPLE)
        else:
            self.regime_mean = None
            self.regime_std = None

    def get_rl_action(self, X_row, current_price, latest_row=None):
        """Fragt den RL Agent nach seiner Entscheidung (regime-aware)."""
        # Features normalisieren
        features = X_row[self.feature_cols].values.flatten()
        if self.feat_mean is not None:
            features = (features - self.feat_mean) / self.feat_std
        features = np.nan_to_num(features, 0.0).astype(np.float32)

        # Unrealisierter PnL
        unrealized = 0.0
        if abs(self.position_state) > 0.01 and self.entry_price_rl > 0:
            if self.position_state > 0:
                unrealized = (current_price - self.entry_price_rl) / self.entry_price_rl
            else:
                unrealized = (self.entry_price_rl - current_price) / self.entry_price_rl

        # Observation zusammenbauen (wie im Training)
        now = datetime.now(timezone.utc)
        extra = np.array([self.position_state, unrealized, now.hour / 23.0, now.weekday() / 6.0], dtype=np.float32)

        parts = [features, extra]

        # Regime-Features hinzufügen (v2)
        if self.regime_cols and latest_row is not None:
            regime_vals = []
            for col in self.regime_cols:
                val = latest_row.get(col, 0.0) if hasattr(latest_row, 'get') else 0.0
                if pd.isna(val):
                    val = 0.0
                regime_vals.append(float(val))
            regime_arr = np.array(regime_vals, dtype=np.float32)
            if self.regime_mean is not None:
                regime_arr = (regime_arr - self.regime_mean) / self.regime_std
            regime_arr = np.nan_to_num(regime_arr, 0.0).astype(np.float32)
            parts.append(regime_arr)

        obs = np.concatenate(parts).astype(np.float32)

        # Agent fragen + Confidence messen
        import torch
        obs_tensor = torch.as_tensor(obs).unsqueeze(0)
        with torch.no_grad():
            dist = self.rl_model.policy.get_distribution(obs_tensor)
            action_mean = dist.distribution.mean.cpu().numpy().flatten()
            action_std = dist.distribution.scale.cpu().numpy().flatten()

        action_val = float(np.clip(action_mean[0], -1.0, 1.0))
        # Confidence: niedrige Std = hohe Sicherheit
        confidence = max(0.0, 1.0 - action_std[0] / 0.5)
        return action_val, confidence

    def process_rl_signal(self, target_position, current_price, atr_rel):
        """Verarbeitet RL-Entscheidung und eröffnet/schliesst Trades."""
        position_change = target_position - self.position_state
        actions = []  # Liste von (action_type, details)

        # Nur handeln bei signifikanter Änderung (0.25 statt 0.10 = stärkere Überzeugung nötig)
        if abs(position_change) <= 0.25:
            return actions

        # Alte Position schliessen
        if abs(self.position_state) > 0.05 and len(self.open_positions) > 0:
            # Close über die normale Pool-Methode
            closed = self.check_exits_forced(current_price)
            actions.extend(closed)

        # Neue Position eröffnen
        if abs(target_position) > 0.25:
            direction = "long" if target_position > 0 else "short"
            # Position Size basiert auf RL Agent's Überzeugung
            size_factor = abs(target_position)  # 0-1
            size = size_factor * self.capital * self.max_position_pct

            today = datetime.now(timezone.utc).date().isoformat()
            self.daily_trades[today] = self.daily_trades.get(today, 0)
            rl_max_daily = 3  # RL Agent: max 3 Trades/Tag (konservativer)
            if self.daily_trades[today] < rl_max_daily:
                stop_loss_pct = max(atr_rel * STOP_LOSS_MULT, 0.001)

                # Hebel basierend auf Action-Stärke (|target_position| als Proxy für Confidence)
                rl_conf = abs(target_position)
                leverage = 1
                for min_conf, lev in LEVERAGE_TIERS:
                    if rl_conf >= min_conf:
                        leverage = lev
                        break

                size *= leverage
                effective_sl_pct = stop_loss_pct / leverage if leverage > 1 else stop_loss_pct
                sl = current_price * (1 - effective_sl_pct) if direction == "long" else current_price * (1 + effective_sl_pct)

                pos = {
                    "direction": direction, "entry_price": current_price,
                    "size": round(size, 2), "entry_time": datetime.now(timezone.utc).isoformat(),
                    "candles_held": 0, "stop_loss": round(sl, 6),
                    "confidence": round(abs(target_position), 4), "pool": self.name,
                    "leverage": leverage,
                }
                self.open_positions.append(pos)
                self.daily_trades[today] += 1
                self.position_state = target_position
                self.entry_price_rl = current_price
                actions.append(("OPEN", pos))
        else:
            self.position_state = 0.0
            self.entry_price_rl = 0.0

        return actions

    def check_exits_forced(self, current_price):
        """Schliesst alle offenen Positionen (RL will flat gehen)."""
        closed = []
        for pos in list(self.open_positions):
            pnl = self._close(pos, current_price, "rl_signal")
            closed.append(("CLOSE", pos, pnl, "RL-SIGNAL"))
            self.open_positions.remove(pos)
        self.position_state = 0.0
        self.entry_price_rl = 0.0
        return closed

    @property
    def max_position_pct(self):
        return 0.20


# ═══════════════════════════════════════════════════════════════
#  STATE MANAGEMENT
# ═══════════════════════════════════════════════════════════════

STATE_FILE  = os.path.join(BOT_DIR, "bot_state.json")
TRADES_LOG  = os.path.join(BOT_DIR, "trades.csv")
EQUITY_LOG  = os.path.join(BOT_DIR, "equity.csv")


def save_state(pools, signals, equity):
    state = {
        "pools": {name: p.to_dict() for name, p in pools.items()},
        "last_update": datetime.now(timezone.utc).isoformat(),
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)

    all_trades = []
    for p in pools.values():
        all_trades.extend(p.trades)
    if all_trades:
        pd.DataFrame(all_trades).to_csv(TRADES_LOG, index=False)
    if signals:
        pd.DataFrame(signals).to_csv(SIGNALS_LOG, index=False)
    if equity:
        pd.DataFrame(equity).to_csv(EQUITY_LOG, index=False)


def load_state(pools):
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            state = json.load(f)
        for name, p in pools.items():
            if name in state.get("pools", {}):
                p.load_from(state["pools"][name])
        log(f"State geladen", C.BLUE)

    signals, equity = [], []
    if os.path.exists(SIGNALS_LOG):
        signals = pd.read_csv(SIGNALS_LOG).to_dict("records")
    if os.path.exists(EQUITY_LOG):
        equity = pd.read_csv(EQUITY_LOG).to_dict("records")
    if os.path.exists(TRADES_LOG):
        df = pd.read_csv(TRADES_LOG)
        for name, p in pools.items():
            pool_trades = df[df["pool"] == name].to_dict("records") if "pool" in df.columns else []
            p.trades = pool_trades
    return signals, equity


# ═══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════

def main():
    os.makedirs(BOT_DIR, exist_ok=True)
    banner()

    # Modelle
    log("Lade Modelle...", C.BLUE)
    rf, xgb, meta = load_models()
    feature_cols = meta["feature_cols"]
    log(f"Modelle geladen: {len(feature_cols)} Features", C.GREEN)

    # Exchange
    exchange = get_exchange()
    log("Binance verbunden", C.GREEN)

    # Pools erstellen
    pools = {}
    for name, cfg in POOLS.items():
        if cfg["type"] == "rl":
            log("Lade RL Agent (Predictlir)...", C.PURPLE)
            rl_pool = RLPool(name, RL_MODEL_PATH, feature_cols)
            pools[name] = rl_pool
            log("RL Agent geladen", C.GREEN)
        else:
            pools[name] = Pool(name, cfg["confidence"])

    # State laden
    signals, equity = load_state(pools)

    for name, p in pools.items():
        cfg = POOLS[name]
        log(f"  {cfg['emoji']} {cfg['label']:14s}  ${p.capital:.2f}  |  {p.n_trades} Trades  |  Conf >{p.conf:.0%}", C.BOLD)

    # Graceful Shutdown
    running = [True]
    def shutdown(signum, frame):
        running[0] = False
        log("\nShutdown...", C.YELLOW)
    sig.signal(sig.SIGINT, shutdown)
    sig.signal(sig.SIGTERM, shutdown)

    log(f"\nBot gestartet. Check alle {CHECK_INTERVAL}s.\n", C.GREEN)

    # Telegram Start
    pool_lines = ""
    for name, p in pools.items():
        cfg = POOLS[name]
        pool_lines += f"{cfg['emoji']} {cfg['label']}: <code>${p.capital:.2f}</code> (Conf >{p.conf:.0%})\n"
    tg_send(
        f"\U0001F680 <b>MuriTrading Bot v2.0 gestartet</b>\n\n"
        f"{pool_lines}\n"
        f"Risiko/Trade: <code>{RISK_PER_TRADE*100:.0f}%</code>\n"
        f"Max Trades/Tag: <code>{MAX_TRADES_PER_DAY}</code>"
    )

    last_candle_time = None
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
                log(f"♻ Heartbeat #{heartbeat_count}  XRP ${current_price:.4f}  [{now.strftime('%H:%M')} UTC]", C.DIM)
            current_hour = now.replace(minute=0, second=0, microsecond=0)

            # Täglicher Report um 22:00 UTC
            today_key = now.date().isoformat()
            if now.hour == DAILY_REPORT_HOUR and last_daily_report != today_key:
                last_daily_report = today_key
                send_daily_report(pools, signals, start_time)
                check_retrain_needed(signals, pools)
                log("Täglicher Report gesendet", C.BLUE)

            # Exits prüfen (jede Iteration)
            for name, pool in pools.items():
                cfg = POOLS[name]
                closed = pool.check_exits(current_price)
                for pos, pnl, reason in closed:
                    pnl_emoji = "\U0001F4B0" if pnl >= 0 else "\u274C"  # 💰 Gewinn / ❌ Verlust
                    raw_ret = (current_price - pos["entry_price"]) / pos["entry_price"]
                    if pos["direction"] == "short": raw_ret = -raw_ret
                    net_ret = raw_ret - ROUND_TRIP
                    duration = HOLD_CANDLES if reason == "TIME-EXIT" else f"<{HOLD_CANDLES}"
                    wr = f"{pool.win_rate:.0%}" if pool.n_trades > 0 else "–"

                    log(f"{cfg['emoji']} {cfg['label']} CLOSE {pos['direction'].upper()}  "
                        f"PnL: {pnl:+.2f}$  [{reason}]",
                        C.GREEN if pnl >= 0 else C.RED)

                    tg_send(
                        f"{pnl_emoji} <b>TRADE CLOSE</b> – {cfg['emoji']} {cfg['label']}\n\n"
                        f"Richtung: <b>{pos['direction'].upper()}</b>\n"
                        f"Einsatz: <code>${pos['size']:.2f}</code>\n"
                        f"Dauer: <code>{duration}h</code>\n"
                        f"Ergebnis: <code>{pnl:+.2f}$</code> ({net_ret:+.2%})\n"
                        f"\n<b>{cfg['label']}:</b>\n"
                        f"Kapital: <code>${pool.capital:.2f}</code>\n"
                        f"PnL: <code>{pool.total_pnl:+.2f}$</code>\n"
                        f"Win Rate: <code>{wr}</code> ({pool.wins}W / {pool.losses}L)"
                    )

            # Neues Signal nur bei neuer Kerze
            if last_candle_time != current_hour and now.minute >= 1:
                last_candle_time = current_hour
                X, latest_row, price = build_features(exchange, feature_cols)
                if X is None:
                    log("Nicht genug Daten", C.YELLOW)
                    time.sleep(CHECK_INTERVAL)
                    continue

                ensemble_prob, rf_prob, xgb_prob = predict(rf, xgb, X)
                confidence = abs(ensemble_prob - 0.5) * 2

                # Whale-Features (Orderbuch + grosse Trades)
                whale = compute_whale_features()
                whale_txt = whale_signal_text(whale)
                import math
                whale_ok = not math.isnan(whale.get("whale_bid_ask_imbalance", float("nan")))

                # Whale-Adjustments auf ML Ensemble
                if whale_ok:
                    imb = whale["whale_bid_ask_imbalance"]
                    net_norm = whale.get("whale_net_flow_normalized", 0) or 0
                    dr1 = whale.get("whale_depth_ratio_1pct", 1.0) or 1.0

                    # Starker Whale-Kaufdruck → Ensemble leicht Richtung Long boosten
                    if net_norm > 0.3 and imb > 0.6 and dr1 > 1.5:
                        ensemble_prob = min(ensemble_prob + 0.05, 0.95)
                    elif net_norm < -0.3 and imb < 0.4 and dr1 < 0.67:
                        ensemble_prob = max(ensemble_prob - 0.05, 0.05)

                    # Ask-Wall Absorption → bullish breakout
                    if whale.get("whale_absorption_ask"):
                        ensemble_prob = min(ensemble_prob + 0.03, 0.95)
                    # Bid-Wall Absorption → bearish breakdown
                    if whale.get("whale_absorption_bid"):
                        ensemble_prob = max(ensemble_prob - 0.03, 0.05)

                    # Nahe Wall als Widerstand: hemmt Trades in diese Richtung
                    if whale.get("whale_wall_ask") and whale.get("whale_wall_ask_distance", 1) < 0.005:
                        confidence *= 0.8  # Resistance nahe → weniger Confidence
                    if whale.get("whale_wall_bid") and whale.get("whale_wall_bid_distance", 1) < 0.005:
                        confidence *= 1.1  # Support nahe → mehr Confidence

                confidence = min(confidence, 1.0)

                # Cross-Asset Features (BTC/ETH als Frühwarnung)
                cross = build_cross_asset_features(exchange)
                cross_txt = cross_asset_signal_text(cross)
                ca_ok = not math.isnan(cross.get("ca_btc_ret_1", float("nan")))

                # Cross-Asset Adjustments
                if ca_ok:
                    catchup = cross.get("ca_catchup_signal", 0) or 0
                    # Starke Divergenz: BTC bewegt sich, XRP noch nicht → Catch-Up
                    if catchup > 2.0:  # z-score > 2 = starkes Signal
                        ensemble_prob = min(ensemble_prob + 0.04, 0.95)
                    elif catchup < -2.0:
                        ensemble_prob = max(ensemble_prob - 0.04, 0.05)

                    # BTC-Only Pump = bearish für Alts
                    if cross.get("ca_btc_only_pump"):
                        confidence *= 0.7

                    # Alt-Pump ohne BTC = bullish für XRP
                    if cross.get("ca_alt_only_pump"):
                        ensemble_prob = min(ensemble_prob + 0.03, 0.95)

                    # Alt-Season Boost
                    if cross.get("ca_alt_season", 0) >= 2:
                        confidence *= 1.15

                    # Correlation Breakdown = Vorsicht
                    if cross.get("ca_corr_breakdown"):
                        confidence *= 0.8

                confidence = min(confidence, 1.0)

                # Sentiment Features
                sentiment = compute_sentiment_features()
                sentiment_txt = sentiment_signal_text(sentiment)
                sent_ok = not math.isnan(sentiment.get("sent_fear_greed", float("nan")))

                # Sentiment Adjustments (contrarian)
                if sent_ok:
                    fng = sentiment["sent_fear_greed"]
                    extreme = sentiment.get("sent_fear_greed_extreme", 0)
                    composite = sentiment.get("sent_composite", 0.5) or 0.5

                    # Extreme Fear = contrarian bullish (Markt überverkauft)
                    if extreme == -1:  # Fear & Greed ≤ 20
                        ensemble_prob = min(ensemble_prob + 0.03, 0.95)
                    # Extreme Greed = contrarian bearish
                    elif extreme == 1:  # Fear & Greed ≥ 80
                        ensemble_prob = max(ensemble_prob - 0.03, 0.05)

                confidence = min(confidence, 1.0)

                # Regime-Info
                adx_val = latest_row.get("1h_adx", 0) if latest_row is not None else 0
                chop_val = latest_row.get("1h_chop", 0) if latest_row is not None else 0
                regime_val = latest_row.get("1h_regime_trend", 0) if latest_row is not None else 0
                if pd.isna(adx_val): adx_val = 0
                if pd.isna(chop_val): chop_val = 0
                if pd.isna(regime_val): regime_val = 0
                regime_label = f"{C.GREEN}TREND{C.RESET}" if regime_val else f"{C.YELLOW}SEITW{C.RESET}"

                # Status
                print(f"\n{C.DIM}  ┌───────────────────────────────────────────────────────┐{C.RESET}", flush=True)
                print(f"  │ {C.BOLD}XRP/USDT{C.RESET}  ${current_price:.4f}  │  "
                      f"Ensemble: {ensemble_prob:.0%}  │  Conf: {confidence:.0%}  │  "
                      f"Regime: {regime_label} (ADX:{adx_val:.0f})", flush=True)
                print(f"  │ {C.PURPLE}{whale_txt}{C.RESET}", flush=True)
                print(f"  │ {C.BLUE}{cross_txt}{C.RESET}", flush=True)
                print(f"  │ {C.YELLOW}{sentiment_txt}{C.RESET}", flush=True)
                for name, pool in pools.items():
                    cfg = POOLS[name]
                    wr = f"{pool.win_rate:.0%}" if pool.n_trades > 0 else "–"
                    pnl_c = C.GREEN if pool.total_pnl >= 0 else C.RED
                    print(f"  │ {cfg['emoji']} {cfg['label']:12s}  "
                          f"${pool.capital:.2f}  PnL:{pnl_c}{pool.total_pnl:+.2f}{C.RESET}  "
                          f"WR:{wr}  T:{pool.n_trades}  Open:{len(pool.open_positions)}", flush=True)
                print(f"{C.DIM}  └───────────────────────────────────────────────────────┘{C.RESET}", flush=True)

                # Jeder Pool prüft unabhängig
                atr_rel = latest_row.get("1h_atr_rel", 0.005)
                if pd.isna(atr_rel): atr_rel = 0.005

                traded_pools = []
                for name, pool in pools.items():
                    cfg = POOLS[name]

                    # RL Agent entscheidet selbst (regime-aware)
                    if cfg["type"] == "rl" and isinstance(pool, RLPool):
                        rl_action, rl_confidence = pool.get_rl_action(X, current_price, latest_row=latest_row)

                        # Gate 1: Regime – kein neuer Trade im Seitwärtsmarkt
                        adx_live = float(latest_row.get("1h_adx", 0)) if latest_row is not None else 0
                        chop_live = float(latest_row.get("1h_chop", 0)) if latest_row is not None else 0
                        if pd.isna(adx_live): adx_live = 0
                        if pd.isna(chop_live): chop_live = 0
                        is_sideways = adx_live < 20 and chop_live > 0.6

                        if is_sideways and len(pool.open_positions) == 0 and abs(rl_action) > 0.10:
                            log(f"{cfg['emoji']} {cfg['label']} REGIME GATE: Sideways (ADX:{adx_live:.0f} Chop:{chop_live:.2f}) – blocked", C.YELLOW)
                            rl_action = 0.0

                        # Gate 2: Confidence – nur traden wenn PPO sicher ist
                        MIN_RL_CONFIDENCE = 0.7
                        if rl_confidence < MIN_RL_CONFIDENCE and len(pool.open_positions) == 0 and abs(rl_action) > 0.10:
                            log(f"{cfg['emoji']} {cfg['label']} LOW CONFIDENCE: {rl_confidence:.2f} < {MIN_RL_CONFIDENCE} – blocked", C.YELLOW)
                            rl_action = 0.0

                        actions = pool.process_rl_signal(rl_action, current_price, atr_rel)
                        for act in actions:
                            if act[0] == "OPEN":
                                pos = act[1]
                                traded_pools.append(name)
                                dir_emoji = "\u2934\uFE0F" if pos["direction"] == "long" else "\u2935\uFE0F"  # ⤴️ Long / ⤵️ Short
                                rl_lev = pos.get("leverage", 1)
                                rl_lev_txt = f"  {rl_lev}x" if rl_lev > 1 else ""
                                log(f"{cfg['emoji']} {cfg['label']} OPEN {pos['direction'].upper()}  "
                                    f"${current_price:.4f}  Size: ${pos['size']:.2f}{rl_lev_txt}  RL: {rl_action:+.2f} Conf: {rl_confidence:.2f}", C.PURPLE)
                                tg_send(
                                    f"{dir_emoji} <b>TRADE OPEN</b> – {cfg['emoji']} {cfg['label']}\n\n"
                                    f"Richtung: <b>{pos['direction'].upper()}</b>\n"
                                    f"Preis: <code>${current_price:.4f}</code>\n"
                                    f"Einsatz: <code>${pos['size']:.2f}</code>"
                                    f"{f' (<b>{rl_lev}x</b> Hebel)' if rl_lev > 1 else ''}\n"
                                    f"RL-Action: <code>{rl_action:+.2f}</code> Conf: <code>{rl_confidence:.2f}</code>\n"
                                    f"Stop-Loss: <code>${pos['stop_loss']:.4f}</code>\n"
                                    f"Haltezeit: {HOLD_CANDLES}h\n"
                                    f"🐋 {whale_txt}"
                                )
                            elif act[0] == "CLOSE":
                                _, pos, pnl, reason = act
                                pnl_emoji = "\U0001F4B0" if pnl >= 0 else "\u274C"  # 💰 Gewinn / ❌ Verlust
                                wr = f"{pool.win_rate:.0%}" if pool.n_trades > 0 else "–"
                                log(f"{cfg['emoji']} {cfg['label']} CLOSE {pos['direction'].upper()}  "
                                    f"PnL: {pnl:+.2f}$  [{reason}]",
                                    C.GREEN if pnl >= 0 else C.RED)
                                tg_send(
                                    f"{pnl_emoji} <b>TRADE CLOSE</b> – {cfg['emoji']} {cfg['label']}\n\n"
                                    f"Richtung: <b>{pos['direction'].upper()}</b>\n"
                                    f"Einsatz: <code>${pos['size']:.2f}</code>\n"
                                    f"Ergebnis: <code>{pnl:+.2f}$</code>\n"
                                    f"Kapital: <code>${pool.capital:.2f}</code>\n"
                                    f"Win Rate: <code>{wr}</code> ({pool.wins}W / {pool.losses}L)"
                                )
                        continue

                    # ML Pools: wie bisher
                    is_long, is_short = pool.check_signal(ensemble_prob)

                    if (is_long or is_short) and len(pool.open_positions) == 0:
                        direction = "long" if is_long else "short"

                        # Hebel berechnen
                        is_trend = bool(regime_val)
                        whale_dir_ok = (direction == "long" and whale.get("whale_bid_ask_imbalance", 0.5) > 0.6) or \
                                       (direction == "short" and whale.get("whale_bid_ask_imbalance", 0.5) < 0.4)
                        lev = calc_leverage(confidence, is_trend, whale_confirmed=whale_dir_ok)

                        pos = pool.open_trade(direction, current_price, confidence, atr_rel, leverage=lev)
                        if pos:
                            traded_pools.append(name)
                            dir_emoji = "\u2934\uFE0F" if direction == "long" else "\u2935\uFE0F"  # ⤴️ Long / ⤵️ Short
                            lev_txt = f"  {lev}x" if lev > 1 else ""
                            log(f"{cfg['emoji']} {cfg['label']} OPEN {direction.upper()}  "
                                f"${current_price:.4f}  Size: ${pos['size']:.2f}{lev_txt}  Conf: {confidence:.0%}", C.GREEN)

                            tg_send(
                                f"{dir_emoji} <b>TRADE OPEN</b> – {cfg['emoji']} {cfg['label']}\n\n"
                                f"Richtung: <b>{direction.upper()}</b>\n"
                                f"Preis: <code>${current_price:.4f}</code>\n"
                                f"Einsatz: <code>${pos['size']:.2f}</code>"
                                f"{f' (<b>{lev}x</b> Hebel)' if lev > 1 else ''}\n"
                                f"Confidence: <code>{confidence:.0%}</code>\n"
                                f"Stop-Loss: <code>${pos['stop_loss']:.4f}</code>\n"
                                f"Haltezeit: {HOLD_CANDLES}h\n"
                                f"🐋 {whale_txt}"
                            )

                if not traded_pools:
                    log(f"Kein Trade (ML-Conf: {confidence:.0%}, RL-Pos: {pools['rl_agent'].position_state:+.2f})", C.DIM)

                # Equity-Snapshot (alle Pools)
                eq_row = {
                    "timestamp": now.isoformat(),
                    "price": round(current_price, 6),
                    "ensemble_prob": round(ensemble_prob, 4),
                    "confidence": round(confidence, 4),
                }
                for name, pool in pools.items():
                    eq_row[f"{name}_capital"] = round(pool.capital, 2)
                    eq_row[f"{name}_pnl"] = round(pool.total_pnl, 4)
                    eq_row[f"{name}_trades"] = pool.n_trades
                equity.append(eq_row)

                # Signal-Log
                sig_row = {
                    "timestamp": now.isoformat(),
                    "price": round(current_price, 6),
                    "ensemble_prob": round(ensemble_prob, 4),
                    "rf_prob": round(rf_prob, 4),
                    "xgb_prob": round(xgb_prob, 4),
                    "confidence": round(confidence, 4),
                    "whale_imbalance": whale.get("whale_bid_ask_imbalance"),
                    "whale_net_flow": whale.get("whale_net_flow"),
                    "whale_big_trades": whale.get("whale_big_trade_count", 0),
                    "whale_wall_bid": whale.get("whale_wall_bid", 0),
                    "whale_wall_ask": whale.get("whale_wall_ask", 0),
                    "ca_btc_ret_1": cross.get("ca_btc_ret_1"),
                    "ca_catchup_signal": cross.get("ca_catchup_signal"),
                    "ca_corr_48": cross.get("ca_corr_48"),
                    "ca_alt_season": cross.get("ca_alt_season"),
                    "sent_fear_greed": sentiment.get("sent_fear_greed"),
                    "sent_composite": sentiment.get("sent_composite"),
                }
                for name in pools:
                    is_l, is_s = pools[name].check_signal(ensemble_prob)
                    sig_row[f"{name}_signal"] = "long" if is_l else "short" if is_s else "neutral"
                    sig_row[f"{name}_traded"] = name in traded_pools
                signals.append(sig_row)

                # Dashboard-Status + State speichern
                save_dashboard_status(pools, signals, ensemble_prob, confidence, current_price)
                save_state(pools, signals, equity)

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
    save_state(pools, signals, equity)

    print(f"\n{C.PURPLE}{C.BOLD}═══════════════════════════════════════════════════════{C.RESET}", flush=True)
    print(f"{C.BOLD}  Bot gestoppt – Finale Bilanz{C.RESET}", flush=True)
    print(f"{C.PURPLE}═══════════════════════════════════════════════════════{C.RESET}", flush=True)

    tg_lines = "\U0001F6D1 <b>MuriTrading Bot gestoppt</b>\n\n"
    for name, pool in pools.items():
        cfg = POOLS[name]
        wr = f"{pool.win_rate:.1%}" if pool.n_trades > 0 else "–"
        ret = (pool.capital - INITIAL_CAPITAL) / INITIAL_CAPITAL
        print(f"  {cfg['emoji']} {cfg['label']:14s}  "
              f"${pool.capital:.2f}  PnL: {pool.total_pnl:+.2f}  "
              f"WR: {wr}  Trades: {pool.n_trades}  DD: {pool.max_dd:.1%}", flush=True)
        tg_lines += (
            f"{cfg['emoji']} <b>{cfg['label']}</b>\n"
            f"  Kapital: <code>${pool.capital:.2f}</code> ({ret:+.1%})\n"
            f"  PnL: <code>{pool.total_pnl:+.2f}$</code>\n"
            f"  Win Rate: <code>{wr}</code> ({pool.wins}W / {pool.losses}L)\n"
            f"  Max DD: <code>{pool.max_dd:.1%}</code>\n\n"
        )

    print(f"{C.PURPLE}═══════════════════════════════════════════════════════{C.RESET}\n", flush=True)
    tg_send(tg_lines)


if __name__ == "__main__":
    main()
