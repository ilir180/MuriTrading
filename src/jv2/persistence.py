"""
JV Boting v2 – State Persistence
"""

import os
import json
import csv
from datetime import datetime, timezone

from src.jv2.config import *
from src.jv2.models import TradeRecord, BotState


def save_state(bots):
    state = {
        "version": "jv2-1.0",
        "last_update": datetime.now(timezone.utc).isoformat(),
        "bots": {bot.bot_id: bot.state.to_dict() for bot in bots},
    }
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


def load_state(bots):
    if not os.path.exists(STATE_FILE):
        return
    with open(STATE_FILE) as f:
        state = json.load(f)
    if state.get("version") != "jv2-1.0":
        return
    for bot in bots:
        if bot.bot_id in state.get("bots", {}):
            bot.state = BotState.from_dict(state["bots"][bot.bot_id])


def append_trade(record: TradeRecord):
    file_exists = os.path.exists(TRADES_CSV)
    with open(TRADES_CSV, "a") as f:
        if not file_exists:
            f.write(TradeRecord.csv_header() + "\n")
        f.write(record.to_csv_row() + "\n")


def append_equity(bots, prices):
    """prices: dict {symbol: price} oder float (backward compat)."""
    file_exists = os.path.exists(EQUITY_CSV)
    now = datetime.now(timezone.utc).isoformat()
    total_equity = sum(bot.state.capital for bot in bots)

    # Backward compat: prices kann ein dict oder ein float sein
    if isinstance(prices, (int, float)):
        prices = {"XRP/USDT": prices}

    with open(EQUITY_CSV, "a") as f:
        if not file_exists:
            cols = ["timestamp", "total_equity"] + [b.bot_id for b in bots]
            f.write(",".join(cols) + "\n")
        vals = [now, f"{total_equity:.2f}"]
        for bot in bots:
            eq = bot.state.capital
            if bot.state.position:
                p = prices.get(bot.symbol, 0)
                if p > 0:
                    eq += bot.state.position.unrealized_pnl(p)
            vals.append(f"{eq:.2f}")
        f.write(",".join(vals) + "\n")


def append_signal(signal):
    file_exists = os.path.exists(SIGNALS_CSV)
    with open(SIGNALS_CSV, "a") as f:
        if not file_exists:
            f.write("timestamp,bot_id,direction,confidence,price,reasoning\n")
        r = signal
        reasoning_clean = r.reasoning.replace(",", ";").replace("\n", " ")
        f.write(f"{r.timestamp},{r.bot_id},{r.direction},{r.confidence:.3f},"
                f"{r.price_at_signal:.6f},{reasoning_clean}\n")


def append_spy_log(timestamp, intel_summary):
    file_exists = os.path.exists(SPY_LOG_CSV)
    with open(SPY_LOG_CSV, "a") as f:
        if not file_exists:
            f.write("timestamp,intel\n")
        summary_clean = json.dumps(intel_summary, default=str).replace("\n", " ")
        f.write(f"{timestamp},{summary_clean}\n")
