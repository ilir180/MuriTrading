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

# Windows: default stdout is cp1252 and crashes on emojis. Force UTF-8.
import sys as _sys
if hasattr(_sys.stdout, "reconfigure"):
    _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    _sys.stderr.reconfigure(encoding="utf-8", errors="replace")

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


def fetch_futures_features(symbol: str, price: float):
    """Funding Rate Z-Score + OI Quadrant from Binance Perp public API."""
    try:
        from src.features.futures_features import compute_futures_features
        return compute_futures_features(symbol, price)
    except Exception:
        return {}


def fetch_liquidation_features(symbol: str):
    """Rolling 15-min liquidation stream snapshot for this symbol."""
    try:
        from src.features.liquidation_stream import compute_liquidation_features
        from src.features.futures_features import SPOT_TO_PERP
        perp = SPOT_TO_PERP.get(symbol)
        if not perp:
            return {}
        return compute_liquidation_features(perp)
    except Exception:
        return {}


def fetch_cvd_features(symbol: str):
    """Cumulative Volume Delta over rolling 1h/4h/24h windows from spot tape."""
    try:
        from src.features.cvd_features import cvd_features_for
        return cvd_features_for(symbol)
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

    # Write an initial heartbeat ASAP so the watchdog doesn't kill us during
    # cold-start (CVD bootstrap can take ~70s across 4 symbols).
    try:
        with open(HEARTBEAT_FILE, "w") as _hb:
            _hb.write(datetime.now(timezone.utc).isoformat())
    except Exception:
        pass

    # Start a background heartbeat thread so the watchdog stays happy even
    # during long blocking operations (CVD bootstrap, 4H candle processing,
    # multi-symbol HTTP calls). Daemon: dies with the main process.
    import threading as _th
    def _heartbeat_thread():
        while True:
            try:
                with open(HEARTBEAT_FILE, "w") as _hb:
                    _hb.write(datetime.now(timezone.utc).isoformat())
            except Exception:
                pass
            time.sleep(45)
    _th.Thread(target=_heartbeat_thread, daemon=True, name="hb_writer").start()

    # Start liquidation WebSocket streams (one daemon thread per symbol).
    # Safe if module/install missing — returns False, continues.
    try:
        from src.features.liquidation_stream import start_all as _start_liq
        n_streams = _start_liq()
        if n_streams:
            log(f"Liquidation streams gestartet: {n_streams}", C.GREEN)
    except Exception as _e:
        log(f"Liquidation streams nicht verfügbar: {_e}", C.YELLOW)

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
    last_coach_run = None
    heartbeat_count = 0

    # Shadow Challenger state (paper-traded alternative Coach).
    # v1 = original boost (additive) — empirically anti-edge.
    # v2 = inverted boost (flip direction when crowd aligns with bot).
    from src.jv2.challenger import (
        load_state as _load_chal, save_state as _save_chal,
        on_signal as _chal_on_signal, on_tick as _chal_on_tick,
        load_state_v2 as _load_chal_v2, save_state_v2 as _save_chal_v2,
        on_signal_v2 as _chal_v2_on_signal, on_tick_v2 as _chal_v2_on_tick,
    )
    challenger_state = _load_chal()
    challenger_v2_state = _load_chal_v2()
    spy_intel = {}
    # Cache für Daten zwischen Candles
    prices = {}
    market_data_cache = {}

    log(f"System läuft. Check alle {CHECK_INTERVAL}s.\n", C.GREEN)

    while running[0]:
        try:
            now = datetime.now(timezone.utc)
            heartbeat_count += 1

            # Watchdog heartbeat — pro Iteration eine frische mtime hinterlassen.
            try:
                with open(HEARTBEAT_FILE, "w") as _hb:
                    _hb.write(now.isoformat())
            except Exception:
                pass

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
            had_close = False
            for bot in bots:
                price = prices.get(bot.symbol, 0)
                if price <= 0:
                    continue
                trade = bot.tick(price)
                if trade:
                    had_close = True
                    append_trade(trade)
                    sym_short = SYMBOLS.get(bot.symbol, {}).get("short", "?")
                    log(f"\u274C CLOSE {bot.bot_id} [{sym_short}] {trade.direction.upper()} "
                        f"[{trade.reason}] PnL:${trade.pnl:+.2f}",
                        C.GREEN if trade.pnl > 0 else C.RED)
                    tg_send(fmt_trade_close(trade))
            # Persist state immediately on any close — protects against
            # phantom re-closes if process dies mid-cycle before 4H save.
            if had_close:
                try:
                    save_state(bots)
                except Exception:
                    pass

            # Shadow Challenger: paper-position exit check every tick
            try:
                chal_closes = _chal_on_tick(challenger_state, prices)
                chal_v2_closes = _chal_v2_on_tick(challenger_v2_state, prices)
                if (chal_closes or chal_v2_closes) and heartbeat_count % 10 == 0:
                    log(f"\U0001F9EA v1: {len(chal_closes)} closes "
                        f"PnL ${challenger_state.total_pnl:+.2f} | "
                        f"v2: {len(chal_v2_closes)} closes "
                        f"PnL ${challenger_v2_state.total_pnl:+.2f}", C.PURPLE)
            except Exception:
                pass

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

                    # Crypto-Native Triade: Funding-Z + OI-Quadrant + Liquidations
                    futures = fetch_futures_features(symbol, price)
                    liq = fetch_liquidation_features(symbol)
                    cvd = fetch_cvd_features(symbol)

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
                        "futures": futures,        # funding_z, oi_quadrant, ...
                        "liquidations": liq,       # liq_volume_15m_usd, ...
                        "cvd": cvd,                # cvd_1h_z, buy_share_4h, ...
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
                    # Spy-Wirkung messbar machen: Intel pro Kerze loggen
                    # (war definiert, aber nie aufgerufen — Audit 10.06.26)
                    try:
                        non_empty = {b: i for b, i in spy_intel.items() if i}
                        if non_empty:
                            append_spy_log(
                                datetime.now(timezone.utc).isoformat(),
                                {"symbol": symbol, "intel": non_empty})
                    except Exception:
                        pass

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

                            # Shadow Challengers (v1 + v2) evaluate independently
                            try:
                                if signal.direction != "neutral":
                                    v1_opened = _chal_on_signal(
                                        challenger_state, bot.bot_id, symbol,
                                        signal.direction, float(signal.confidence),
                                        float(market_data["price"]),
                                        float(market_data["atr_4h"]),
                                        market_data,
                                    )
                                    if v1_opened:
                                        log(f"      \U0001F9EA v1 OPEN "
                                            f"{signal.direction.upper()} "
                                            f"conf={challenger_state.positions[bot.bot_id]['boosted_confidence']:.2f}",
                                            C.PURPLE)
                                    v2_opened = _chal_v2_on_signal(
                                        challenger_v2_state, bot.bot_id, symbol,
                                        signal.direction, float(signal.confidence),
                                        float(market_data["price"]),
                                        float(market_data["atr_4h"]),
                                        market_data,
                                    )
                                    if v2_opened:
                                        v2pos = challenger_v2_state.positions[bot.bot_id]
                                        log(f"      \U0001F9EA v2 OPEN "
                                            f"{v2pos['direction'].upper()} "
                                            f"[{v2pos.get('v2_mode','?')}] "
                                            f"conf={v2pos['boosted_confidence']:.2f}",
                                            C.PURPLE)
                            except Exception:
                                pass

                        except Exception as e:
                            log(f"    {bot.bot_id} Fehler: {e}", C.RED)

                # State speichern
                save_state(bots)
                append_equity(bots, prices)
                _save_chal(challenger_state)
                _save_chal_v2(challenger_v2_state)

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

            # ── Daily Coach Run (after daily report) ──
            # Runs once per day. If it fails, the runner does NOT crash —
            # bots keep operating on the previous coach_state.json.
            coach_hour = (DAILY_REPORT_HOUR + 1) % 24  # one hour after report
            if now.hour == coach_hour and last_coach_run != today_key:
                last_coach_run = today_key
                try:
                    from src.jv2.coach import Coach
                    from collections import defaultdict as _dd
                    all_ids = [b.bot_id for b in bots]
                    coach = Coach()
                    decisions = coach.evaluate(all_bot_ids=all_ids)
                    coach.write_state(decisions)

                    by_action = _dd(int)
                    for d in decisions.values():
                        by_action[d.action] += 1
                    summary_line = " | ".join(
                        f"{act}:{n}" for act, n in sorted(by_action.items()) if n)
                    log(f"\U0001F9E0 Coach: {summary_line}", C.PURPLE)
                    tg_send(f"\U0001F9E0 <b>Coach update</b>\n{summary_line}")

                    # Reload directives into live bots without restart.
                    from src.jv2.coach import get_cell_directive, load_coach_state
                    cs = load_coach_state()
                    for b in bots:
                        dr = get_cell_directive(b.bot_id, cs)
                        b.coach_action = dr["action"]
                        b.coach_lev_mult = dr["leverage_multiplier"]
                        b.coach_cap_mult = dr["capital_multiplier"]
                        b.regime_blacklist = set(dr["regime_blacklist"])
                        b.regime_whitelist = (set(dr["regime_whitelist"])
                                              if dr["regime_whitelist"] is not None else None)
                        b.invert_signal = dr["invert"]
                        if dr["exec_override"]:
                            b.risk_profile.update(dr["exec_override"])
                        b.coach_disabled = (dr["action"] == "disable"
                                            or dr["capital_multiplier"] == 0.0)
                except Exception as e:
                    log(f"Coach failed: {e} — bots continue with previous state", C.YELLOW)

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
