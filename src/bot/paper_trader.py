"""
MuriTrading – Phase 5: Paper-Trading Bot (Multi-Pool)
Drei Töpfe mit unterschiedlichen Confidence-Leveln laufen parallel:
  - Konservativ (75%): wenige Trades, höchste Accuracy
  - Standard (65%):    ausbalanciert
  - Aggressiv (50%):   viele Trades, niedrigere Accuracy

Start: python src/bot/paper_trader.py
Stop:  Ctrl+C (speichert automatisch)
"""

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

# ── Projekt-Root ───────────────────────────────────────────────
PROJECT_ROOT = os.path.expanduser("~/MuriTrading")
sys.path.insert(0, PROJECT_ROOT)
from src.features.build_features import add_indicators

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
CHECK_INTERVAL   = 60

# ── Die drei Töpfe ────────────────────────────────────────────
POOLS = {
    "konservativ": {"confidence": 0.75, "emoji": "\U0001F9CA", "label": "Konservativ"},
    "standard":    {"confidence": 0.65, "emoji": "\U0001F4CA", "label": "Standard"},
    "aggressiv":   {"confidence": 0.50, "emoji": "\U0001F525", "label": "Aggressiv"},
}

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
  Kapital/Topf   : ${INITIAL_CAPITAL:,.2f} x 3 = ${INITIAL_CAPITAL*3:,.2f}
  Risiko/Trade   : {RISK_PER_TRADE*100:.1f}%
  Haltezeit      : {HOLD_CANDLES}h
  Fees (RT)      : {ROUND_TRIP*100:.3f}%{C.RESET}

  {C.BLUE}Konservativ{C.RESET}  Conf >75%  (wenige, sichere Trades)
  {C.GREEN}Standard{C.RESET}     Conf >65%  (ausbalanciert)
  {C.RED}Aggressiv{C.RESET}    Conf >50%  (viele Trades)
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

    latest = df_base[feature_cols].dropna()
    if latest.empty:
        return None, None, None
    return latest.iloc[[-1]], df_base.iloc[-1], df_1h["close"].iloc[-1]


def predict(rf, xgb, X):
    rf_prob  = rf.predict_proba(X)[:, 1]
    xgb_prob = xgb.predict_proba(X)[:, 1]
    ensemble = (rf_prob + xgb_prob) / 2.0
    return float(ensemble[0]), float(rf_prob[0]), float(xgb_prob[0])


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

    def open_trade(self, direction, price, confidence, atr_rel):
        today = datetime.now(timezone.utc).date().isoformat()
        self.daily_trades[today] = self.daily_trades.get(today, 0)
        if self.daily_trades[today] >= MAX_TRADES_PER_DAY:
            return None

        stop_loss_pct = max(atr_rel * STOP_LOSS_MULT, 0.001)
        size = (self.capital * RISK_PER_TRADE) / stop_loss_pct
        size = min(size, self.capital * 0.20)

        sl = price * (1 - stop_loss_pct) if direction == "long" else price * (1 + stop_loss_pct)

        pos = {
            "direction": direction, "entry_price": price,
            "size": round(size, 2), "entry_time": datetime.now(timezone.utc).isoformat(),
            "candles_held": 0, "stop_loss": round(sl, 6),
            "confidence": round(confidence, 4), "pool": self.name,
        }
        self.open_positions.append(pos)
        self.daily_trades[today] += 1
        return pos

    def check_exits(self, current_price):
        """Prüft Stop-Loss und Haltezeit."""
        closed = []
        for pos in self.open_positions:
            pos["candles_held"] = pos.get("candles_held", 0) + 1

            hit_sl = (pos["direction"] == "long" and current_price <= pos["stop_loss"]) or \
                     (pos["direction"] == "short" and current_price >= pos["stop_loss"])

            if hit_sl:
                pnl = self._close(pos, current_price, "stop_loss")
                closed.append((pos, pnl, "STOP-LOSS"))
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

    while running[0]:
        try:
            ticker = exchange.fetch_ticker(SYMBOL)
            current_price = ticker["last"]
            now = datetime.now(timezone.utc)
            current_hour = now.replace(minute=0, second=0, microsecond=0)

            # Exits prüfen (jede Iteration)
            for name, pool in pools.items():
                cfg = POOLS[name]
                closed = pool.check_exits(current_price)
                for pos, pnl, reason in closed:
                    pnl_emoji = "\u2705" if pnl >= 0 else "\u274C"
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

                # Status
                print(f"\n{C.DIM}  ┌───────────────────────────────────────────────────────┐{C.RESET}", flush=True)
                print(f"  │ {C.BOLD}XRP/USDT{C.RESET}  ${current_price:.4f}  │  "
                      f"Ensemble: {ensemble_prob:.0%}  │  Conf: {confidence:.0%}", flush=True)
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
                    is_long, is_short = pool.check_signal(ensemble_prob)

                    if (is_long or is_short) and len(pool.open_positions) == 0:
                        direction = "long" if is_long else "short"
                        pos = pool.open_trade(direction, current_price, confidence, atr_rel)
                        if pos:
                            traded_pools.append(name)
                            dir_emoji = "\U0001F7E2" if direction == "long" else "\U0001F534"
                            log(f"{cfg['emoji']} {cfg['label']} OPEN {direction.upper()}  "
                                f"${current_price:.4f}  Size: ${pos['size']:.2f}  Conf: {confidence:.0%}", C.GREEN)

                            tg_send(
                                f"{dir_emoji} <b>TRADE OPEN</b> – {cfg['emoji']} {cfg['label']}\n\n"
                                f"Richtung: <b>{direction.upper()}</b>\n"
                                f"Preis: <code>${current_price:.4f}</code>\n"
                                f"Einsatz: <code>${pos['size']:.2f}</code>\n"
                                f"Confidence: <code>{confidence:.0%}</code>\n"
                                f"Stop-Loss: <code>${pos['stop_loss']:.4f}</code>\n"
                                f"Haltezeit: {HOLD_CANDLES}h"
                            )

                if not traded_pools:
                    log(f"Kein Trade (Conf: {confidence:.0%})", C.DIM)

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
                }
                for name in pools:
                    is_l, is_s = pools[name].check_signal(ensemble_prob)
                    sig_row[f"{name}_signal"] = "long" if is_l else "short" if is_s else "neutral"
                    sig_row[f"{name}_traded"] = name in traded_pools
                signals.append(sig_row)

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
