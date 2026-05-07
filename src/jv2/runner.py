"""
JV Boting v2 – Runner (Multi-Asset)
8 Bots × 4 Assets = 32 unabhängige Trader + 3 Agents

Ablauf:
  Alle 60s: Preis holen pro Asset, Exits prüfen
  Auf neuer 4H-Kerze:
    1. Daten holen pro Asset (1h/4h/1d + Whale + Sentiment)
    2. Scout: verpasste Moves analysieren
    3. Spy: Intel kompilieren
    4. Alle Bots: Signal + ggf. Trade
    5. State speichern
  Montag 00:00 UTC: Kapital-Rebalancing
  Täglich 22:00 UTC: Analyst HTML-Report
"""

import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

import os
import sys
import time
import math
import signal as sig
from datetime import datetime, timezone
from collections import defaultdict

import ccxt
import pandas as pd

PROJECT_ROOT = os.path.expanduser("~/MuriTrading")
sys.path.insert(0, PROJECT_ROOT)

from src.features.build_features import add_indicators
from src.jv2.config import *
from src.jv2.bots import create_all_bots
from src.jv2.agents.spy import SpyAgent
from src.jv2.agents.scout import ScoutAgent
from src.jv2.agents.analyst import AnalystAgent
from src.jv2.capital import rebalance
from src.jv2.persistence import (
    save_state, load_state, append_trade, append_equity,
    append_signal, append_spy_log,
)
from src.jv2.telegram import tg_send, fmt_trade_open, fmt_trade_close


# ── Terminal ──────────────────────────────────────────

class C:
    PURPLE = "\033[95m"; BLUE = "\033[94m"; GREEN = "\033[92m"
    YELLOW = "\033[93m"; RED = "\033[91m"; BOLD = "\033[1m"
    DIM = "\033[2m"; RESET = "\033[0m"

def log(msg, color=C.RESET):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"  {C.DIM}[{ts}]{C.RESET} {color}{msg}{C.RESET}", flush=True)


# ── Daten ──────────────────────────────────────────────

def get_exchange():
    return ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})


def fetch_candles(exchange, symbol, timeframe, limit):
    candles = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("datetime").drop(columns=["timestamp"]).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def _safe(val, default=0.0):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return float(val)


def fetch_whale(binance_id):
    """Whale Features für beliebiges Symbol."""
    try:
        import src.features.whale_features as wf
        old_sym = wf.SYMBOL
        wf.SYMBOL = binance_id
        result = wf.compute_whale_features()
        wf.SYMBOL = old_sym
        return result
    except Exception:
        return {}


def fetch_sentiment():
    """Sentiment ist global (nicht pro Asset)."""
    try:
        from src.features.sentiment import compute_sentiment_features
        return compute_sentiment_features()
    except Exception:
        return {}


# ── Main ──────────────────────────────────────────────

def main():
    exchange = get_exchange()
    bots = create_all_bots()
    spy = SpyAgent()
    scout = ScoutAgent()
    analyst = AnalystAgent()

    load_state(bots)

    # Bots nach Symbol gruppieren
    bots_by_symbol = defaultdict(list)
    for bot in bots:
        bots_by_symbol[bot.symbol].append(bot)

    running = [True]
    def shutdown(signum, frame):
        running[0] = False
        log("\nShutdown...", C.YELLOW)
    sig.signal(sig.SIGINT, shutdown)
    sig.signal(sig.SIGTERM, shutdown)

    # Banner
    n_bots = len(bots)
    n_symbols = len(SYMBOLS)
    print(f"\n{C.PURPLE}{'='*55}")
    print(f"  JV BOTING v2 — {n_bots} Bots ({NUM_BOTS}×{n_symbols} Assets)")
    print(f"  {', '.join(s.get('short','?') for s in SYMBOLS.values())} | 4H | Paper Trading")
    print(f"{'='*55}{C.RESET}\n")

    for symbol, sym_cfg in SYMBOLS.items():
        sym_short = sym_cfg["short"]
        sym_emoji = sym_cfg["emoji"]
        sym_bots = bots_by_symbol[symbol]
        total_cap = sum(b.state.capital for b in sym_bots)
        active = sum(1 for b in sym_bots if b.state.position)
        log(f"  {sym_emoji} {sym_short}: ${total_cap:.0f} | {active} Positionen", C.GREEN)

    total_eq = sum(b.state.capital for b in bots)
    log(f"\n  Total: ${total_eq:.0f} über {n_symbols} Assets", C.BOLD)
    log(f"Binance verbunden", C.GREEN)

    # TG Startup
    active_total = sum(1 for b in bots if b.state.position)
    asset_str = ", ".join(s["short"] for s in SYMBOLS.values())
    tg_send(f"\U0001F680 <b>JV Boting v2 Multi-Asset gestartet</b>\n"
            f"{n_bots} Bots | {n_symbols} Assets ({asset_str})\n"
            f"${total_eq:.0f} Total | {active_total} Positionen offen")

    # Loop State
    last_4h_slot = None
    last_daily_report = None
    last_rebalance_week = None
    heartbeat_count = 0
    spy_intel = {}
    # Cache für Daten zwischen Candles
    prices = {}
    market_data_cache = {}

    log(f"System läuft. Check alle {CHECK_INTERVAL}s.\n", C.GREEN)

    while running[0]:
        try:
            now = datetime.now(timezone.utc)
            heartbeat_count += 1

            # ── Preise holen für alle Assets ──
            for symbol in SYMBOLS:
                try:
                    ticker = exchange.fetch_ticker(symbol)
                    prices[symbol] = ticker["last"]
                except Exception as e:
                    if symbol not in prices:
                        prices[symbol] = 0

            # ── Heartbeat ──
            if heartbeat_count % 10 == 0:
                active = sum(1 for b in bots if b.state.position)
                total_pnl = sum(b.state.total_pnl for b in bots)
                price_str = "  ".join(
                    f"{SYMBOLS[s]['short']}:${prices.get(s,0):.1f}" if prices.get(s,0) > 100
                    else f"{SYMBOLS[s]['short']}:${prices.get(s,0):.4f}"
                    for s in SYMBOLS
                )
                log(f"#{heartbeat_count}  {price_str}  Active:{active}  PnL:${total_pnl:+.1f}", C.DIM)

            # ── Alle Bots: Exit-Check (60s) ──
            for bot in bots:
                price = prices.get(bot.symbol, 0)
                if price <= 0:
                    continue
                trade = bot.tick(price)
                if trade:
                    append_trade(trade)
                    sym_short = SYMBOLS.get(bot.symbol, {}).get("short", "?")
                    log(f"\u274C CLOSE {bot.bot_id} [{sym_short}] {trade.direction.upper()} "
                        f"[{trade.reason}] PnL:${trade.pnl:+.2f}",
                        C.GREEN if trade.pnl > 0 else C.RED)
                    tg_send(fmt_trade_close(trade))

            # ── Neue 4H-Kerze ──
            current_4h_slot = now.replace(hour=(now.hour // 4) * 4, minute=0, second=0, microsecond=0)
            if last_4h_slot != current_4h_slot and now.minute >= CANDLE_WAIT_MIN:
                last_4h_slot = current_4h_slot

                log(f"\n{'='*55}", C.DIM)
                log(f"Neue 4H Kerze: {current_4h_slot.strftime('%H:%M UTC')}", C.BLUE)

                # Sentiment (global, einmal für alle)
                sentiment = fetch_sentiment()

                # Pro Asset: Daten holen und Bots laufen lassen
                for symbol, sym_cfg in SYMBOLS.items():
                    sym_short = sym_cfg["short"]
                    sym_emoji = sym_cfg["emoji"]
                    price = prices.get(symbol, 0)
                    if price <= 0:
                        log(f"  {sym_emoji} {sym_short}: Kein Preis, übersprungen", C.YELLOW)
                        continue

                    log(f"\n  {sym_emoji} {sym_short} ${price:.4f}" if price < 100
                        else f"\n  {sym_emoji} {sym_short} ${price:.1f}", C.BOLD)

                    # Daten holen
                    try:
                        df_1h = fetch_candles(exchange, symbol, "1h", 250)
                        df_4h = fetch_candles(exchange, symbol, "4h", 250)
                        df_1d = fetch_candles(exchange, symbol, "1d", 100)
                        df_1h = add_indicators(df_1h, "1h_")
                        df_4h = add_indicators(df_4h, "4h_")
                        df_1d = add_indicators(df_1d, "1d_")
                    except Exception as e:
                        log(f"    Daten-Fehler {sym_short}: {e}", C.RED)
                        continue

                    # Whale Features
                    whale = fetch_whale(sym_cfg["binance_id"])

                    atr_4h = _safe(df_4h.iloc[-1].get("4h_atr_14"), price * 0.01)

                    market_data = {
                        "price": price,
                        "symbol": symbol,
                        "df_1h": df_1h, "df_4h": df_4h, "df_1d": df_1d,
                        "latest_1h": df_1h.iloc[-1],
                        "latest_4h": df_4h.iloc[-1],
                        "latest_1d": df_1d.iloc[-1],
                        "exchange": exchange,
                        "atr_4h": atr_4h,
                        "whale": whale,
                        "sentiment": sentiment,
                        "cross_asset": {},
                    }
                    market_data_cache[symbol] = market_data

                    # Scout: verpasste Moves für dieses Asset
                    missed = scout.analyze_missed_moves(df_4h, bots_by_symbol[symbol])
                    if missed:
                        log(f"    \U0001F50D Scout: {len(missed)} verpasste Moves", C.YELLOW)

                    # Spy Intel (nur für Bots dieses Symbols)
                    sym_bots = bots_by_symbol[symbol]
                    spy_intel = spy.compile_intel(sym_bots, price)

                    # Bots laufen lassen
                    for bot in sym_bots:
                        try:
                            signal, entry_info, thesis_exit = bot.on_new_candle(
                                market_data, spy_intel.get(bot.bot_id, {}))
                            append_signal(signal)

                            cfg = BOT_CONFIGS.get(bot.base_id, {})
                            emoji = cfg.get("emoji", "")

                            # Thesis-Exit loggen
                            if thesis_exit:
                                append_trade(thesis_exit)
                                log(f"    {emoji} \U0001F9E0 {bot.base_id}: {thesis_exit.reason} "
                                    f"PnL:${thesis_exit.pnl:+.2f}",
                                    C.GREEN if thesis_exit.pnl > 0 else C.RED)
                                tg_send(fmt_trade_close(thesis_exit))

                            if signal.direction != "neutral":
                                icon = "\U00002B06" if signal.direction == "long" else "\U00002B07"
                                log(f"    {emoji} {icon} {bot.base_id}: "
                                    f"{signal.direction.upper()} @{signal.confidence:.0%}  "
                                    f"{signal.reasoning}", C.GREEN if signal.direction == "long" else C.RED)
                            else:
                                log(f"    {emoji} \u2796 {bot.base_id}: {signal.reasoning}", C.DIM)

                            if entry_info:
                                pos = bot.state.position
                                log(f"      \U0001F4B0 OPEN {pos.direction.upper()} "
                                    f"Entry:${pos.entry_price:.4f} SL:${pos.stop_loss:.4f} "
                                    f"TP:${pos.take_profit:.4f} Size:${pos.size_usd:.0f}", C.BOLD)
                                tg_send(f"{sym_emoji} {sym_short} " + fmt_trade_open(bot.base_id, pos))

                        except Exception as e:
                            log(f"    {bot.bot_id} Fehler: {e}", C.RED)

                # State speichern
                save_state(bots)
                append_equity(bots, prices)

                # Summary
                active = sum(1 for b in bots if b.state.position)
                total_pnl = sum(b.state.total_pnl for b in bots)
                total_eq = sum(b.state.capital for b in bots)
                log(f"\n  Portfolio: ${total_eq:.0f} | PnL: ${total_pnl:+.2f} | Active: {active}/{len(bots)}", C.BOLD)

                # Per-Asset Summary
                for symbol, sym_cfg in SYMBOLS.items():
                    sym_bots = bots_by_symbol[symbol]
                    sym_eq = sum(b.state.capital for b in sym_bots)
                    sym_pnl = sum(b.state.total_pnl for b in sym_bots)
                    sym_active = sum(1 for b in sym_bots if b.state.position)
                    log(f"    {sym_cfg['emoji']} {sym_cfg['short']}: ${sym_eq:.0f} "
                        f"PnL:${sym_pnl:+.1f} Active:{sym_active}", C.DIM)

                log(f"{'='*55}\n", C.DIM)

            # ── Weekly Rebalance ──
            week_key = now.strftime("%Y-W%W")
            if now.weekday() == REBALANCE_DAY and now.hour == 0 and last_rebalance_week != week_key:
                last_rebalance_week = week_key
                # Rebalance per Asset-Gruppe
                for symbol in SYMBOLS:
                    sym_bots = bots_by_symbol[symbol]
                    rebalance(sym_bots)
                save_state(bots)
                log(f"\U0001F4CA Rebalancing durchgeführt", C.BLUE)
                tg_send("\U0001F4CA <b>Weekly Rebalance</b> durchgeführt")

            # ── Daily Report ──
            today_key = now.date().isoformat()
            if now.hour == DAILY_REPORT_HOUR and last_daily_report != today_key:
                last_daily_report = today_key

                market_info_parts = []
                for symbol, sym_cfg in SYMBOLS.items():
                    md = market_data_cache.get(symbol)
                    if md:
                        r4 = md.get("latest_4h")
                        if r4 is not None:
                            adx = _safe(r4.get("4h_adx", 0))
                            rsi = _safe(r4.get("4h_rsi_14", 50))
                            market_info_parts.append(f"{sym_cfg['short']}(ADX:{adx:.0f} RSI:{rsi:.0f})")
                market_info = " | ".join(market_info_parts) if market_info_parts else None

                # Evaluator: These/Execution/Edge berechnen
                eval_results = None
                try:
                    from src.jv2.agents.evaluator import Evaluator, build_signal_outcomes
                    df_4h_dict = {}
                    for symbol in SYMBOLS:
                        md = market_data_cache.get(symbol)
                        if md and "df_4h" in md:
                            df_4h_dict[symbol] = md["df_4h"]
                    if df_4h_dict:
                        outcomes = build_signal_outcomes(df_4h_dict, SIGNALS_CSV, TRADES_CSV)
                        evaluator = Evaluator()
                        eval_results = evaluator.evaluate_signals(outcomes)
                except Exception as e:
                    log(f"Evaluator: {e}", C.YELLOW)

                analyst.generate_daily_report(bots, scout, prices, market_info, eval_results)
                log(f"Daily Report gesendet", C.BLUE)

            time.sleep(CHECK_INTERVAL)

        except ccxt.NetworkError as e:
            log(f"Netzwerk: {e}", C.YELLOW)
            time.sleep(30)
        except ccxt.ExchangeError as e:
            log(f"Exchange: {e}", C.RED)
            time.sleep(60)
        except Exception as e:
            log(f"Fehler: {e}", C.RED)
            import traceback
            traceback.print_exc()
            time.sleep(10)

    # Shutdown
    save_state(bots)
    total_eq = sum(b.state.capital for b in bots)
    total_pnl = sum(b.state.total_pnl for b in bots)
    log(f"\nFinal: ${total_eq:.0f}  PnL: ${total_pnl:+.2f}", C.BOLD)
    tg_send(f"\U0001F6D1 <b>JV Boting v2 gestoppt</b>\nEquity: ${total_eq:.0f} | PnL: ${total_pnl:+.2f}")


if __name__ == "__main__":
    main()
