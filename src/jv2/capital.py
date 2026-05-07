"""
JV Boting v2 – Capital Allocation & Weekly Rebalancing
"""

from src.jv2.config import INITIAL_ALLOC, MIN_ALLOC, MAX_ALLOC


def rebalance(bots):
    """
    Wöchentliches Rebalancing: Gewinner kriegen mehr, Verlierer weniger.
    Bots mit offener Position werden übersprungen.
    Returns dict mit alter/neuer Allokation.
    """
    total_equity = sum(bot.state.capital for bot in bots)

    # Performance-Score pro Bot
    scores = {}
    for bot in bots:
        n_trades = bot.state.wins + bot.state.losses
        if n_trades == 0:
            scores[bot.bot_id] = 0.0  # Neutral
            continue
        win_rate = bot.state.wins / n_trades
        pnl_score = bot.state.total_pnl / INITIAL_ALLOC
        scores[bot.bot_id] = 0.6 * pnl_score + 0.4 * (win_rate - 0.5) * 2

    # Normalisieren
    min_s = min(scores.values()) if scores else 0
    max_s = max(scores.values()) if scores else 0
    rng = max_s - min_s if max_s != min_s else 1.0

    weights = {}
    for bid, score in scores.items():
        weights[bid] = 0.5 + (score - min_s) / rng  # 0.5 - 1.5

    total_weight = sum(weights.values())

    changes = {}
    for bot in bots:
        old_cap = bot.state.capital
        if bot.state.position is not None:
            changes[bot.bot_id] = {"old": old_cap, "new": old_cap, "skipped": True}
            continue

        target = total_equity * (weights[bot.bot_id] / total_weight)
        target = max(MIN_ALLOC, min(MAX_ALLOC, target))
        bot.state.capital = round(target, 2)
        changes[bot.bot_id] = {"old": round(old_cap, 2), "new": round(target, 2), "skipped": False}

    # Rundungsfehler korrigieren
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
