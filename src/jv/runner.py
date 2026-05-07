"""
Joint Venture Boting – Runner
Orchestriert alle JV-Bots und den Prime in einem Prozess.

Ablauf:
  Alle 60s: Preis holen, Exits prüfen
  Auf neuer 4H-Kerze:
    1. Credit-Decay
    2. Pending Signals evaluieren
    3. Alle Bots → observe() → Signal
    4. Prime aggregiert → Trade-Entscheidung
    5. State speichern
"""

import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

import os
import sys
import time
import json
import math
import signal as sig
import ccxt
import pandas as pd
import requests as _requests
from datetime import datetime, timezone

PROJECT_ROOT = os.path.expanduser("~/MuriTrading")
sys.path.insert(0, PROJECT_ROOT)

from src.features.build_features import add_indicators
from src.jv.config import *
from src.jv.signal_protocol import JVSignal
from src.jv.credit_system import CreditLedger
from src.jv.prime import PrimeBot
from src.jv.bots.momentum_bot import MomentumBot
from src.jv.bots.regime_bot import RegimeBot
from src.jv.bots.volume_bot import VolumeBot
from src.jv.bots.sentiment_bot import SentimentBot


# ── Telegram ──────────────────────────────────────────

def tg_send(text):
    try:
        _requests.post(
            f"https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage",
            json={"chat_id": TG_CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        pass


# ── Terminal ──────────────────────────────────────────

class C:
    PURPLE = "\033[95m"; BLUE = "\033[94m"; GREEN = "\033[92m"
    YELLOW = "\033[93m"; RED = "\033[91m"; BOLD = "\033[1m"
    DIM = "\033[2m"; RESET = "\033[0m"

def log(msg, color=C.RESET):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"  {C.DIM}[{ts}]{C.RESET} {color}{msg}{C.RESET}", flush=True)


# ── Data ──────────────────────────────────────────────

def get_exchange():
    return ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})

def fetch_candles(exchange, timeframe, limit):
    candles = exchange.fetch_ohlcv(SYMBOL, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns=["timestamp"]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df

def _safe(val, default=0.0):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return float(val)


# ── Main ──────────────────────────────────────────────

def main():
    os.makedirs(SIGNALS_DIR, exist_ok=True)

    print(f"""
{C.PURPLE}{C.BOLD}═══════════════════════════════════════════════════════{C.RESET}
{C.BOLD}  MuriTrading – Joint Venture Boting{C.RESET}
{C.PURPLE}═══════════════════════════════════════════════════════{C.RESET}
{C.DIM}  Asset         : {SYMBOL}
  Timeframe     : {PRIMARY_TF}
  Kapital       : ${INITIAL_CAPITAL:,.2f}
  Risk/Trade    : {RISK_PER_TRADE*100:.1f}%
  Min Konsens   : {MIN_CONSENSUS:.0%}
  Min Agreement : {MIN_AGREEMENT:.0%}
  Credit Decay  : {DECAY_FACTOR}/4H-Kerze{C.RESET}

  {C.GREEN}JV Bots:{C.RESET}
  1. {C.BLUE}Momentum{C.RESET}   – Velocity, Acceleration, Multi-TF
  2. {C.YELLOW}Regime{C.RESET}     – Phasen-Erkennung, ADX, Volatilität
  3. {C.PURPLE}Volume{C.RESET}     – Whale-Flow, Orderbook, Absorption
  4. {C.RED}Sentiment{C.RESET}  – Fear&Greed, Cross-Asset, BTC-Lead
{C.PURPLE}═══════════════════════════════════════════════════════{C.RESET}
""", flush=True)

    # Exchange
    exchange = get_exchange()
    log("Binance verbunden", C.GREEN)

    # JV Bots – alle 4 Spezialisten
    bot_ids = ["momentum", "regime", "volume", "sentiment"]
    bots = [MomentumBot(), RegimeBot(), VolumeBot(), SentimentBot()]
    log(f"JV Bots geladen: {', '.join(bot_ids)}", C.GREEN)

    # Credit Ledger
    ledger = CreditLedger(bot_ids)
    ledger.load(LEDGER_FILE)
    log(f"Credits geladen. Leader: {ledger.get_leader()}", C.BLUE)

    # Prime
    prime = PrimeBot()
    prime.load_state(PRIME_STATE)

    n_trades = prime.wins + prime.losses
    wr = f"{prime.wins/n_trades:.0%}" if n_trades > 0 else "–"
    log(f"Prime: ${prime.capital:.2f}  PnL: {prime.total_pnl:+.2f}  WR: {wr}", C.BOLD)

    # Graceful Shutdown
    running = [True]
    def shutdown(signum, frame):
        running[0] = False
        log("\nShutdown...", C.YELLOW)
    sig.signal(sig.SIGINT, shutdown)
    sig.signal(sig.SIGTERM, shutdown)

    log(f"\nJV System gestartet. Check alle {CHECK_INTERVAL}s.\n", C.GREEN)

    tg_send(
        f"\U0001F91D <b>Joint Venture Boting gestartet</b>\n\n"
        f"Bots: {', '.join(bot_ids)}\n"
        f"Kapital: <code>${prime.capital:.2f}</code>\n"
        f"Leader: <code>{ledger.get_leader()}</code>"
    )

    last_4h_slot = None
    last_daily_report = None
    heartbeat_count = 0
    signals_history = []

    while running[0]:
        try:
            ticker = exchange.fetch_ticker(SYMBOL)
            current_price = ticker["last"]
            now = datetime.now(timezone.utc)
            heartbeat_count += 1

            if heartbeat_count % 10 == 0:
                pos_info = ""
                if prime.position:
                    pnl, _, _ = prime.position.calc_pnl(current_price)
                    pos_info = f"  Pos:{prime.position.direction.upper()} PnL:{pnl:+.2f}$"
                log(f"#{heartbeat_count}  XRP ${current_price:.4f}  "
                    f"Leader:{ledger.get_leader()}{pos_info}", C.DIM)

            prime.risk_mgr.new_month_check(prime.capital)

            # ── PARTIAL TP CHECK ──────────────────────
            if prime.position:
                should_partial, partial_pnl = prime.position.check_partial_tp(current_price)
                if should_partial:
                    prime.capital += partial_pnl
                    prime.total_pnl += partial_pnl
                    log(f"\U0001F3AF PARTIAL TP: +${partial_pnl:.2f}  SL→Breakeven", C.GREEN)
                    tg_send(f"\U0001F3AF <b>JV PARTIAL TP</b>\n+${partial_pnl:.2f}\nSL→Breakeven")

            # ── EXIT CHECK ────────────────────────────
            if prime.position:
                prime.position.update_trailing(current_price)
                should_exit, reason = prime.position.check_exit(current_price)

                if should_exit:
                    pnl, raw_ret, net_ret = prime.position.calc_pnl(current_price)
                    prime.capital += pnl
                    prime.total_pnl += pnl
                    if pnl > 0: prime.wins += 1
                    else: prime.losses += 1

                    circuit = prime.risk_mgr.register_result(pnl)
                    n_t = prime.wins + prime.losses
                    wr_now = f"{prime.wins/n_t:.0%}" if n_t > 0 else "–"

                    emoji = "\U0001F4B0" if pnl >= 0 else "\u274C"
                    color = C.GREEN if pnl >= 0 else C.RED
                    log(f"{emoji} CLOSE {prime.position.direction.upper()} [{reason}] "
                        f"PnL:{pnl:+.2f}$ Kapital:${prime.capital:.2f} WR:{wr_now}", color)

                    tg_send(
                        f"{emoji} <b>JV TRADE CLOSE</b> [{reason}]\n\n"
                        f"Richtung: <b>{prime.position.direction.upper()}</b>\n"
                        f"Strategy: <code>{prime.position.strategy}</code>\n"
                        f"PnL: <code>{pnl:+.2f}$</code> ({net_ret:+.2%})\n"
                        f"Kapital: <code>${prime.capital:.2f}</code>\n"
                        f"WR: <code>{wr_now}</code> ({prime.wins}W/{prime.losses}L)"
                    )

                    prime.trades.append({
                        "timestamp": now.isoformat(),
                        "strategy": prime.position.strategy,
                        "direction": prime.position.direction,
                        "entry_price": prime.position.entry_price,
                        "exit_price": current_price,
                        "size": prime.position.original_size,
                        "pnl": round(pnl, 4),
                        "net_return_pct": round(net_ret * 100, 4),
                        "reason": reason,
                        "hold_candles": prime.position.candles_held,
                    })
                    prime.position = None
                    prime.save_state(PRIME_STATE)

            # ── 4H CANDLE CHECK ───────────────────────
            current_4h_slot = now.hour // 4
            current_4h_time = now.replace(hour=current_4h_slot * 4,
                                          minute=0, second=0, microsecond=0)

            if last_4h_slot != current_4h_time and now.minute >= CANDLE_WAIT_MIN:
                last_4h_slot = current_4h_time

                if prime.position:
                    prime.position.candles_held += 1

                # Daten holen (einmal für alle Bots)
                df_1h = fetch_candles(exchange, "1h", 250)
                df_4h = fetch_candles(exchange, "4h", 250)
                df_1d = fetch_candles(exchange, "1d", 250)
                df_1h = add_indicators(df_1h, "1h_")
                df_4h = add_indicators(df_4h, "4h_")
                df_1d = add_indicators(df_1d, "1d_")

                latest_1h = df_1h.iloc[-1] if len(df_1h) > 0 else None
                latest_4h = df_4h.iloc[-1] if len(df_4h) > 0 else None
                latest_1d = df_1d.iloc[-1] if len(df_1d) > 0 else None

                atr_4h = _safe(latest_4h.get("4h_atr_14") if latest_4h is not None else None,
                               current_price * 0.01)
                if atr_4h < current_price * 0.001:
                    atr_4h = current_price * 0.01

                market_data = {
                    "price": current_price,
                    "df_1h": df_1h, "df_4h": df_4h, "df_1d": df_1d,
                    "latest_1h": latest_1h, "latest_4h": latest_4h, "latest_1d": latest_1d,
                    "exchange": exchange, "atr_4h": atr_4h,
                }

                # 1. Credit Decay
                ledger.apply_decay()

                # 2. Evaluate Pending Signals
                eval_results = ledger.evaluate_pending(current_price, atr_4h)
                for r in eval_results:
                    icon = "\u2705" if r["correct"] else "\u274C"
                    log(f"  {icon} {r['bot_id']}: {r['direction']} "
                        f"({r['actual_move_pct']:+.2f}%) → {r['delta']:+.1f} Credits "
                        f"[{r['credits_after']:.0f}]",
                        C.GREEN if r["correct"] else C.RED)

                # 3. Alle Bots → observe()
                current_signals = []
                for bot in bots:
                    try:
                        signal = bot.observe(market_data)
                        current_signals.append(signal)
                        signal.save(SIGNALS_DIR)

                        if signal.direction != "neutral":
                            ledger.record_signal(signal)

                        dir_icon = {
                            "long": f"{C.GREEN}\u2191",
                            "short": f"{C.RED}\u2193",
                            "neutral": f"{C.DIM}\u2500",
                        }
                        log(f"  {dir_icon.get(signal.direction, '')} {signal.bot_id}: "
                            f"{signal.direction.upper()} @{signal.confidence:.0%}  "
                            f"{signal.reasoning}{C.RESET}")
                    except Exception as e:
                        log(f"  Bot {bot.bot_id} Fehler: {e}", C.RED)

                # Credit Leaderboard
                leader = ledger.get_leader()
                ranking = ledger.get_ranking()
                print(f"\n{C.DIM}  ┌─── Credit Leaderboard ──────────────────────────┐{C.RESET}")
                for i, (bid, cr) in enumerate(ranking):
                    acc = ledger.get_accuracy(bid)
                    acc_str = f"{acc:.0%}" if acc is not None else "–"
                    crown = " \U0001F451" if bid == leader else ""
                    bar_len = int(cr / MAX_CREDITS * 20)
                    bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)
                    color = C.GREEN if cr >= MIN_INFLUENCE else C.RED
                    print(f"  │ {color}{i+1}. {bid:12s} {bar} {cr:6.1f} "
                          f"Acc:{acc_str}{crown}{C.RESET}")
                print(f"{C.DIM}  └──────────────────────────────────────────────────┘{C.RESET}")

                # 4. Prime Entscheidung
                if prime.position is None:
                    weights = ledger.get_weights()
                    consensus = prime.aggregate_signals(
                        current_signals, weights, leader)

                    leader_icon = "\u2705" if consensus["leader_agrees"] else "\u274C"
                    print(f"  {C.BOLD}Prime Konsens: {consensus['direction'].upper()} "
                          f"Conf:{consensus['weighted_confidence']:.0%} "
                          f"Agree:{consensus['agreement_pct']:.0%} "
                          f"Leader({leader}):{leader_icon}{C.RESET}")

                    if prime.should_enter(consensus):
                        pos = prime.open_position(consensus, current_price, atr_4h)
                        if pos:
                            sl_pct = abs(current_price - pos.stop_loss) / current_price
                            tp_pct = abs(pos.take_profit - current_price) / current_price
                            dir_emoji = "\u2934\uFE0F" if pos.direction == "long" else "\u2935\uFE0F"

                            log(f"{dir_emoji} OPEN {pos.direction.upper()} "
                                f"${current_price:.4f} Size:${pos.size:.2f} "
                                f"via {leader}", C.GREEN)

                            tg_send(
                                f"{dir_emoji} <b>JV TRADE OPEN</b>\n\n"
                                f"Leader: <code>{leader} \U0001F451</code>\n"
                                f"Richtung: <b>{pos.direction.upper()}</b>\n"
                                f"Preis: <code>${current_price:.4f}</code>\n"
                                f"Einsatz: <code>${pos.size:.2f}</code>\n"
                                f"SL: <code>${pos.stop_loss:.4f}</code> (-{sl_pct:.1%})\n"
                                f"TP: <code>${pos.take_profit:.4f}</code> (+{tp_pct:.1%})\n"
                                f"Konsens: <code>{consensus['weighted_confidence']:.0%}</code>\n\n"
                                f"<i>{' | '.join(consensus['details'])}</i>"
                            )
                    else:
                        log(f"  Kein Trade. Conf:{consensus['weighted_confidence']:.0%} "
                            f"Agree:{consensus['agreement_pct']:.0%}", C.DIM)

                # Signal History
                for s in current_signals:
                    signals_history.append({
                        "timestamp": now.isoformat(),
                        "bot_id": s.bot_id,
                        "direction": s.direction,
                        "confidence": s.confidence,
                        "price": current_price,
                        "credits": ledger.get_credits(s.bot_id),
                        "leader": leader,
                    })

                # Equity Snapshot
                unrealized = 0
                if prime.position:
                    unrealized, _, _ = prime.position.calc_pnl(current_price)
                prime.equity.append({
                    "timestamp": now.isoformat(),
                    "price": round(current_price, 6),
                    "capital": round(prime.capital, 2),
                    "equity": round(prime.capital + unrealized, 2),
                })

                # Save all
                ledger.save(LEDGER_FILE)
                prime.save_state(PRIME_STATE)
                if prime.trades:
                    pd.DataFrame(prime.trades).to_csv(JV_TRADES, index=False)
                if prime.equity:
                    pd.DataFrame(prime.equity).to_csv(JV_EQUITY, index=False)
                if signals_history:
                    pd.DataFrame(signals_history[-2000:]).to_csv(JV_SIG_HIST, index=False)

                # Dashboard Status
                n_t = prime.wins + prime.losses
                dashboard = {
                    "bot_running": True, "version": "jv-1.0",
                    "last_update": now.isoformat(),
                    "current_price": current_price,
                    "capital": round(prime.capital, 2),
                    "total_pnl": round(prime.total_pnl, 2),
                    "wins": prime.wins, "losses": prime.losses,
                    "win_rate": round(prime.wins / n_t, 4) if n_t > 0 else None,
                    "leader": leader,
                    "credits": {k: round(v, 1) for k, v in ledger.credits.items()},
                    "has_position": prime.position is not None,
                }
                with open(JV_DASHBOARD, "w") as f:
                    json.dump(dashboard, f, indent=2, default=str)

            # ── DAILY REPORT ──────────────────────────
            today_key = now.date().isoformat()
            if now.hour == DAILY_REPORT_HOUR and last_daily_report != today_key:
                last_daily_report = today_key
                n_t = prime.wins + prime.losses
                wr_now = f"{prime.wins/n_t:.0%}" if n_t > 0 else "–"

                tg_send(
                    f"\U0001F4CB <b>JV Daily Report</b>\n\n"
                    f"<b>Leaderboard:</b>\n"
                    f"<pre>{ledger.leaderboard_text()}</pre>\n\n"
                    f"<b>Prime:</b>\n"
                    f"Kapital: <code>${prime.capital:.2f}</code>\n"
                    f"PnL: <code>{prime.total_pnl:+.2f}$</code>\n"
                    f"WR: <code>{wr_now}</code> ({prime.wins}W/{prime.losses}L)\n"
                    f"Leader: <code>{ledger.get_leader()} \U0001F451</code>"
                )
                log("Daily Report gesendet", C.BLUE)

            time.sleep(CHECK_INTERVAL)

        except ccxt.NetworkError as e:
            log(f"Netzwerk: {e}", C.YELLOW)
            time.sleep(30)
        except ccxt.ExchangeError as e:
            log(f"Exchange: {e}", C.RED)
            time.sleep(30)
        except Exception as e:
            log(f"Fehler: {e}", C.RED)
            import traceback; traceback.print_exc()
            time.sleep(10)

    # ── SHUTDOWN ──────────────────────────────────────
    ledger.save(LEDGER_FILE)
    prime.save_state(PRIME_STATE)

    n_t = prime.wins + prime.losses
    wr_final = f"{prime.wins/n_t:.1%}" if n_t > 0 else "–"
    print(f"\n{C.PURPLE}{C.BOLD}═══ JV System gestoppt ═══{C.RESET}")
    print(f"  Kapital: ${prime.capital:.2f}  PnL: {prime.total_pnl:+.2f}  WR: {wr_final}")
    print(f"  Leader: {ledger.get_leader()}")
    print(ledger.leaderboard_text())

    tg_send(
        f"\U0001F6D1 <b>JV System gestoppt</b>\n\n"
        f"Kapital: <code>${prime.capital:.2f}</code>\n"
        f"WR: <code>{wr_final}</code>\n"
        f"Leader: <code>{ledger.get_leader()}</code>"
    )


if __name__ == "__main__":
    main()
