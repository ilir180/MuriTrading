"""
JV Boting v2 – Capital Allocation & Weekly Rebalancing
"""

from src.jv2.config import INITIAL_ALLOC, MIN_ALLOC, MAX_ALLOC
from src.jv2.coach import load_coach_state, get_cell_directive


def rebalance(bots):
    """
    Wöchentliches Rebalancing: Gewinner kriegen mehr, Verlierer weniger.
    Coach-Multiplikator wird auf den Performance-Score angewandt — promotete
    Bots bekommen mehr, demoted weniger, disabled gehen auf MIN_ALLOC.
    Bots mit offener Position werden übersprungen.
    Returns dict mit alter/neuer Allokation.
    """
    total_equity = sum(bot.state.capital for bot in bots)
    coach_state = load_coach_state()

    # Performance-Score pro Bot, multipliziert mit Coach-Capital-Multiplier
    scores = {}
    coach_mults = {}
    for bot in bots:
        directive = get_cell_directive(bot.bot_id, coach_state)
        coach_mult = directive["capital_multiplier"]
        coach_mults[bot.bot_id] = coach_mult

        n_trades = bot.state.wins + bot.state.losses
        if n_trades == 0:
            raw_score = 0.0
        else:
            win_rate = bot.state.wins / n_trades
            pnl_score = bot.state.total_pnl / INITIAL_ALLOC
            raw_score = 0.6 * pnl_score + 0.4 * (win_rate - 0.5) * 2
        # Coach-Multiplier wirkt auf positive UND negative Scores symmetrisch:
        # promote (1.3) zieht den Score nach oben, demote (0.6) nach unten.
        scores[bot.bot_id] = raw_score * coach_mult + (coach_mult - 1.0) * 0.5

    # Normalisieren
    min_s = min(scores.values()) if scores else 0
    max_s = max(scores.values()) if scores else 0
    rng = max_s - min_s if max_s != min_s else 1.0

    weights = {}
    for bid, score in scores.items():
        # Disabled cells (coach_mult == 0) explicitly get the minimum weight,
        # so they get MIN_ALLOC and effectively park their capital.
        if coach_mults.get(bid, 1.0) == 0.0:
            weights[bid] = 0.01
        else:
            weights[bid] = 0.5 + (score - min_s) / rng  # 0.5 - 1.5

    total_weight = sum(weights.values())

    changes = {}
    for bot in bots:
        old_cap = bot.state.capital
        if bot.state.position is not None:
            changes[bot.bot_id] = {"old": old_cap, "new": old_cap, "skipped": True}
            continue

        target = total_equity * (weights[bot.bot_id] / total_weight)
        # Disabled cells: floor at MIN_ALLOC, never grow.
        if coach_mults.get(bot.bot_id, 1.0) == 0.0:
            target = MIN_ALLOC
        else:
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
