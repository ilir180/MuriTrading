"""JV Boting v2 — Counterfactual Replay Engine.

Walks historical 4H OHLCV bar-by-bar, evaluates every bot at every bar,
simulates trades (entry / SL / TP / trailing / time / thesis exits), and
emits TradeRecords with regime tags.

Purpose: turn one evening of historical data into the equivalent of months
of live trading. Output feeds the same regime_report.py used for live trades.

Limitations (v1):
- whale_features and sentiment are not historically retrievable -> empty dicts.
  Bots that depend on them (flow_tracker, contrarian) run with degraded info.
- spy_intel is empty (no historical Spy state).
- Intra-bar exit ordering: if both SL and TP could fire in the same bar,
  worst case wins (SL fires first for losing direction).
- 60-second tick granularity is collapsed to bar-level; trailing stops and
  exits are evaluated once per 4H bar using bar high/low.
"""

import math
from copy import deepcopy
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import pandas as pd

from src.jv2.config import (
    SYMBOLS, INITIAL_ALLOC, MIN_ALLOC, MIN_CONFIDENCE,
    CONSEC_LOSS_LIMIT, COOLDOWN_HOURS, ROUND_TRIP,
)
from src.jv2.models import JV2Signal, BotPosition, BotState, TradeRecord


WARMUP_BARS = 60   # need indicator history before any bot can produce sensible signals


def _safe(val, default=0.0):
    if val is None:
        return default
    if isinstance(val, float) and math.isnan(val):
        return default
    return float(val)


def _snapshot_regime_from_row(row, atr, price) -> dict:
    """Same shape as base_bot._snapshot_regime, computed from a historical bar."""
    return {
        "adx":               _safe(row.get("4h_adx")),
        "rsi":               _safe(row.get("4h_rsi_14"), 50.0),
        "bb_pos":            _safe(row.get("4h_bb_pos"), 0.5),
        "bbw":               _safe(row.get("4h_bb_width")),
        "atr_pct":           (atr / price * 100) if price > 0 else 0.0,
        "chop":              _safe(row.get("4h_chop")),
        "trend_consistency": _safe(row.get("4h_trend_consistency")),
        "fear_greed":        50.0,  # not available historically
    }


def _build_market_data(symbol, df_1h, df_4h, df_1d, i):
    """Synthesize the dict that bots expect, for bar i in df_4h."""
    bar_4h = df_4h.iloc[i]
    ts = df_4h.index[i]
    price = float(bar_4h["close"])
    atr_4h = _safe(bar_4h.get("4h_atr_14"), price * 0.01)

    latest_1h = None
    if df_1h is not None and not df_1h.empty:
        sub = df_1h.loc[df_1h.index <= ts]
        if len(sub) > 0:
            latest_1h = sub.iloc[-1]

    latest_1d = None
    if df_1d is not None and not df_1d.empty:
        sub = df_1d.loc[df_1d.index <= ts]
        if len(sub) > 0:
            latest_1d = sub.iloc[-1]

    return {
        "price": price,
        "symbol": symbol,
        "df_1h": df_1h,
        "df_4h": df_4h.iloc[: i + 1],   # only data up to current bar
        "df_1d": df_1d,
        "latest_1h": latest_1h,
        "latest_4h": bar_4h,
        "latest_1d": latest_1d,
        "exchange": None,
        "atr_4h": atr_4h,
        "whale": {},
        "sentiment": {},
        "cross_asset": {},
    }


def _open_position_from_signal(bot, signal, market_data, regime: dict) -> Optional[BotPosition]:
    """Mirrors base_bot._open_position but returns the position instead of mutating bot.state."""
    rp = bot.risk_profile
    price = market_data["price"]
    atr = market_data["atr_4h"]

    catastrophe_sl_atr = max(rp["sl_atr"], 4.0)
    sl_dist = catastrophe_sl_atr * atr
    tp_dist = rp["tp_atr"] * atr
    if signal.direction == "long":
        sl, tp = price - sl_dist, price + tp_dist
    else:
        sl, tp = price + sl_dist, price - tp_dist
    sl_pct = abs(price - sl) / price
    if sl_pct < 0.001:
        sl_pct = 0.01
    leverage = rp.get("leverage", 1)
    risk_amount = bot.state.capital * rp["risk"]
    size_usd = risk_amount / sl_pct
    size_usd = min(size_usd, bot.state.capital * leverage)
    if size_usd < 5.0:
        return None

    pos = BotPosition(
        bot_id=bot.bot_id,
        direction=signal.direction,
        entry_price=price,
        size_usd=round(size_usd, 2),
        stop_loss=round(sl, 6),
        take_profit=round(tp, 6),
        atr=atr,
        entry_time=market_data.get("_bar_time", ""),
        regime=regime,
    )
    pos._max_hold = rp["max_hold"]
    return pos


def _close_position(bot, exit_price: float, reason: str, exit_time: str) -> TradeRecord:
    """Build TradeRecord, update bot.state. Returns the record, clears position."""
    pos = bot.state.position
    pnl, _, net_ret = pos.calc_pnl(exit_price)
    bot.state.capital += pnl
    bot.state.total_pnl += pnl
    if pnl > 0:
        bot.state.wins += 1
        bot.state.consecutive_losses = 0
    else:
        bot.state.losses += 1
        bot.state.consecutive_losses += 1
        if bot.state.consecutive_losses >= CONSEC_LOSS_LIMIT:
            try:
                t = datetime.fromisoformat(exit_time)
            except (ValueError, TypeError):
                t = datetime.now(timezone.utc)
            bot.state.cooldown_until = (t + timedelta(hours=COOLDOWN_HOURS)).isoformat()
    rg = pos.regime or {}
    record = TradeRecord(
        timestamp=exit_time,
        bot_id=bot.bot_id,
        direction=pos.direction,
        entry_price=pos.entry_price,
        exit_price=exit_price,
        size_usd=pos.size_usd,
        pnl=round(pnl, 4),
        net_return_pct=round(net_ret * 100, 4),
        reason=reason,
        hold_candles=pos.candles_held,
        bot_capital_after=round(bot.state.capital, 2),
        regime_adx=float(rg.get("adx", 0.0)),
        regime_rsi=float(rg.get("rsi", 0.0)),
        regime_bb_pos=float(rg.get("bb_pos", 0.0)),
        regime_bbw=float(rg.get("bbw", 0.0)),
        regime_atr_pct=float(rg.get("atr_pct", 0.0)),
        regime_chop=float(rg.get("chop", 0.0)),
        regime_trend_consistency=float(rg.get("trend_consistency", 0.0)),
        regime_fear_greed=float(rg.get("fear_greed", 0.0)),
    )
    bot.state.position = None
    return record


def _check_intrabar_exit(pos: BotPosition, bar) -> Optional[str]:
    """Did SL or TP fire inside this bar? Returns reason string or None.
    Worst-case ordering: if both could fire, SL fires first."""
    high = float(bar["high"])
    low = float(bar["low"])
    if pos.direction == "long":
        if low <= pos.stop_loss:
            return "STOP-LOSS"
        if high >= pos.take_profit:
            return "TAKE-PROFIT"
    else:
        if high >= pos.stop_loss:
            return "STOP-LOSS"
        if low <= pos.take_profit:
            return "TAKE-PROFIT"
    return None


def _check_trailing_exit(pos: BotPosition, bar) -> Optional[str]:
    """Trailing stop hit during this bar?"""
    if not pos.trailing_active:
        return None
    high = float(bar["high"])
    low = float(bar["low"])
    if pos.direction == "long" and low <= pos.trailing_stop:
        return "TRAILING-STOP"
    if pos.direction == "short" and high >= pos.trailing_stop:
        return "TRAILING-STOP"
    return None


def _can_trade(bot, now_iso: str) -> bool:
    """Mirror of base_bot._can_trade but using the replay's clock."""
    if bot.state.capital < MIN_ALLOC:
        return False
    if bot.state.cooldown_until:
        try:
            cd = datetime.fromisoformat(bot.state.cooldown_until)
            now = datetime.fromisoformat(now_iso)
            if now < cd:
                return False
        except (ValueError, TypeError):
            pass
        bot.state.cooldown_until = None
        bot.state.consecutive_losses = 0
    return True


def replay_asset(bots: List, symbol: str, df_1h, df_4h, df_1d) -> List[TradeRecord]:
    """Replay all `bots` (already filtered to this symbol) over df_4h."""
    trades: List[TradeRecord] = []
    n = len(df_4h)
    if n < WARMUP_BARS + 5:
        return trades

    # Reset state for clean replay
    for bot in bots:
        bot.state = BotState(bot_id=bot.bot_id)
        bot.state.capital = INITIAL_ALLOC

    for i in range(WARMUP_BARS, n):
        bar = df_4h.iloc[i]
        bar_time = df_4h.index[i].isoformat()
        market_data = _build_market_data(symbol, df_1h, df_4h, df_1d, i)
        market_data["_bar_time"] = bar_time

        for bot in bots:
            # 1) intra-bar exits on existing position
            if bot.state.position is not None:
                bot.state.position.candles_held += 1
                # Trailing update using bar close
                bot.state.position.update_trailing(market_data["price"])

                exit_reason = _check_intrabar_exit(bot.state.position, bar)
                if exit_reason:
                    # SL/TP fired during this bar; price is sl/tp respectively.
                    pos = bot.state.position
                    exit_price = pos.stop_loss if exit_reason == "STOP-LOSS" else pos.take_profit
                    trades.append(_close_position(bot, exit_price, exit_reason, bar_time))
                    continue

                trail_reason = _check_trailing_exit(bot.state.position, bar)
                if trail_reason:
                    trades.append(_close_position(
                        bot, bot.state.position.trailing_stop, trail_reason, bar_time))
                    continue

                # Time exit
                pos = bot.state.position
                max_hold = getattr(pos, "_max_hold", 18)
                if pos.candles_held >= max_hold:
                    trades.append(_close_position(bot, market_data["price"], "TIME-EXIT", bar_time))
                    continue

                # Thesis exit at bar close
                still_valid, reason = bot.check_thesis(market_data)
                if not still_valid:
                    trades.append(_close_position(
                        bot, market_data["price"], f"THESIS-EXIT: {reason}", bar_time))
                    continue

            # 2) generate signal at bar close
            try:
                signal = bot.generate_signal(market_data, {})
            except Exception:
                continue
            if bot.invert_signal and signal.direction != "neutral":
                signal = JV2Signal(
                    bot_id=signal.bot_id,
                    timestamp=signal.timestamp,
                    direction="short" if signal.direction == "long" else "long",
                    confidence=signal.confidence,
                    reasoning=f"INV: {signal.reasoning}",
                    features=signal.features,
                    price_at_signal=signal.price_at_signal,
                )
            bot.state.last_signal = signal

            # 3) open new position if conditions met
            if (signal.direction != "neutral"
                    and signal.confidence >= MIN_CONFIDENCE
                    and bot.state.position is None
                    and _can_trade(bot, bar_time)):
                regime = _snapshot_regime_from_row(bar, market_data["atr_4h"], market_data["price"])
                pos = _open_position_from_signal(bot, signal, market_data, regime)
                if pos is not None:
                    bot.state.position = pos
                    bot.state.trades_this_week += 1

    # Close any positions still open at end of replay
    if n > 0:
        last_bar = df_4h.iloc[-1]
        last_time = df_4h.index[-1].isoformat()
        last_price = float(last_bar["close"])
        for bot in bots:
            if bot.state.position is not None:
                trades.append(_close_position(bot, last_price, "REPLAY-END", last_time))

    return trades
