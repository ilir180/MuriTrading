"""
JV Boting v2 – Capital Allocation & Weekly Rebalancing

HRP × Coach overlay:
  Base weight    = HRP allocation from rolling-60d returns covariance.
                   This is risk-balanced (inverse variance, correlation-aware).
  Coach overlay  = capital_multiplier per cell (champion 1.6, promote 1.3,
                   keep 1.0, demote 0.6, disable 0.0).
  Final weight   = HRP_base × coach_multiplier, renormalized.

This way HRP gives risk-aware diversification, Coach overlays performance
intelligence. Both contribute. Disabled cells go to MIN_ALLOC regardless.
"""

import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from src.jv2.config import INITIAL_ALLOC, MIN_ALLOC, MAX_ALLOC, TRADES_CSV
from src.jv2.coach import load_coach_state, get_cell_directive
from src.jv2.hrp import hierarchical_risk_parity


HRP_LOOKBACK_DAYS = 60


def _load_recent_returns(lookback_days: int = HRP_LOOKBACK_DAYS) -> dict:
    """Return {bot_id: [net_return_pct/100]} for trades in the lookback window."""
    out = defaultdict(list)
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).isoformat()
        with open(TRADES_CSV) as f:
            for row in csv.DictReader(f):
                if row.get("timestamp", "") < cutoff:
                    continue
                bid = row.get("bot_id")
                if not bid:
                    continue
                try:
                    ret = float(row.get("net_return_pct", 0)) / 100.0
                except (ValueError, TypeError):
                    continue
                out[bid].append(ret)
    except FileNotFoundError:
        pass
    return dict(out)


def rebalance(bots):
    """
    Wöchentliches Rebalancing nach HRP × Coach.
    Bots mit offener Position werden übersprungen.
    Returns dict {bot_id: {"old": float, "new": float, "skipped": bool,
                            "hrp_weight": float, "coach_mult": float}}.
    """
    total_equity = sum(bot.state.capital for bot in bots)
    coach_state = load_coach_state()
    returns_by_bot = _load_recent_returns()

    # Ensure every active bot is in the returns dict (even empty list);
    # HRP gives fallback weight to thin-data cells.
    for b in bots:
        returns_by_bot.setdefault(b.bot_id, [])

    hrp_weights = hierarchical_risk_parity(returns_by_bot)

    # Coach overlay
    overlay = {}
    coach_mults = {}
    for bot in bots:
        directive = get_cell_directive(bot.bot_id, coach_state)
        mult = directive["capital_multiplier"]
        coach_mults[bot.bot_id] = mult
        base = hrp_weights.get(bot.bot_id, 1.0 / max(1, len(bots)))
        overlay[bot.bot_id] = base * mult

    total_overlay = sum(overlay.values())
    if total_overlay == 0:
        # Fallback to equal-weight
        for b in bots:
            overlay[b.bot_id] = 1.0 / len(bots)
        total_overlay = 1.0

    changes = {}
    for bot in bots:
        old_cap = bot.state.capital
        hrp_w = hrp_weights.get(bot.bot_id, 0.0)
        c_mult = coach_mults.get(bot.bot_id, 1.0)

        if bot.state.position is not None:
            changes[bot.bot_id] = {
                "old": round(old_cap, 2), "new": round(old_cap, 2),
                "skipped": True, "hrp_weight": round(hrp_w, 4),
                "coach_mult": c_mult,
            }
            continue

        target = total_equity * (overlay[bot.bot_id] / total_overlay)
        if c_mult == 0.0:
            target = MIN_ALLOC
        else:
            target = max(MIN_ALLOC, min(MAX_ALLOC, target))
        bot.state.capital = round(target, 2)
        changes[bot.bot_id] = {
            "old": round(old_cap, 2), "new": round(target, 2),
            "skipped": False, "hrp_weight": round(hrp_w, 4),
            "coach_mult": c_mult,
        }

    # Round-off correction
    actual = sum(bot.state.capital for bot in bots)
    diff = total_equity - actual
    if abs(diff) > 0.01:
        eligible = [b for b in bots if b.state.position is None]
        if eligible:
            eligible.sort(key=lambda b: b.state.capital, reverse=True)
            eligible[0].state.capital += round(diff, 2)

    # Weekly trade counter reset
    for bot in bots:
        bot.state.trades_this_week = 0

    return changes
