"""JV Boting v2 — Coach.

The Coach is the decision layer that turns regime-tagged trade history into
per-cell directives (one cell = one bot_id, e.g. trend_rider_SOL):

  - keep            : leave alone
  - promote         : winning consistently -> capital_multiplier > 1
  - demote          : losing but recoverable -> capital_multiplier < 1
  - invert          : these consistently wrong -> flip the signal (inversion principle)
  - exec_fix        : these correct but execution loses -> widen stops / extend hold
  - disable         : never triggers OR catastrophic bleeder -> capital_multiplier = 0
  - regime_gate     : winning in some regimes, losing in others -> only trade winning regimes

Inputs:
  - live trades.csv (truth, but small N)
  - counterfactual_trades.csv (large N, degraded features for whale/sentiment bots)
  - regime_cluster column on every trade

Output:
  - coach_state.json (one record per cell) with action, evidence, multipliers

Philosophy:
  - Coach decides intent. Runner / capital.py / base_bot enforce.
  - Conservative on thin data. Aggressive only with N >= MIN_AGGRESSIVE.
  - Every decision carries its evidence string so a human can audit it.
"""

import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from src.jv2.config import (
    TRADES_CSV, SIGNALS_CSV, JV2_DIR, BOT_OVERRIDES,
)
from src.jv2.coach_stats import (
    deflated_sharpe_from_returns, evaluate_cell as evaluate_cell_dsr,
)

COUNTERFACTUAL_CSV = os.path.join(JV2_DIR, "counterfactual_trades.csv")
COACH_STATE_FILE   = os.path.join(JV2_DIR, "coach_state.json")

# ── Thresholds ─────────────────────────────────────────────
# Conservative on small N, aggressive once N >= MIN_AGGRESSIVE.
MIN_TRADES_ANY_DECISION  = 5     # below this: always "keep"
MIN_TRADES_INVERT_LIVE   = 12    # invert needs at least this many LIVE trades
MIN_TRADES_INVERT_TOTAL  = 20    # OR this many live+cf combined
MIN_TRADES_DISABLE       = 30    # only disable with strong evidence
MIN_TRADES_REGIME_GATE   = 5     # per-regime sample size for gating

WR_INVERT_THRESHOLD      = 0.30
WR_DEMOTE_THRESHOLD      = 0.40
WR_PROMOTE_THRESHOLD     = 0.55
WR_REGIME_BAD_THRESHOLD  = 0.30
WR_REGIME_GOOD_THRESHOLD = 0.50

# Stricter regime gating for inverted cells: even 3 trades at 0% WR
# in a cluster is enough to blacklist. Live observation showed inversion
# fails catastrophically in trending regimes (cluster 0, 3).
MIN_TRADES_INVERT_REGIME_GATE = 3
WR_INVERT_REGIME_BAD          = 0.0001  # i.e. 0 wins

PF_PROMOTE_THRESHOLD     = 1.20  # profit factor

# Counterfactual is supporting evidence, not primary. Weight ratio:
CF_WEIGHT_RATIO          = 0.3   # blend = 0.7*live + 0.3*cf when both present

# Hysteresis: an existing action stays unless the new evaluation has at
# least this confidence. Prevents weekly whiplash (Champion -> Keep -> Champion).
HYSTERESIS_CONFIDENCE_THRESHOLD = 0.6

# DSR (Deflated Sharpe Ratio) integration. Used as a CONFIDENCE MODULATOR,
# not a hard gate — with our small N per cell (5-50 trades), absolute DSR
# is too punitive to use as a veto. Instead:
#   - Compute DSR per cell using returns (net_return_pct/100)
#   - Map DSR to a multiplier in [0.5, 1.5] that scales Coach confidence
#   - Top-quartile DSR cells get confidence boost, bottom-quartile get cut
# n_trials = number of cells we're effectively testing.
N_DSR_TRIALS = 32

# Capital multipliers applied at weekly rebalance time.
CAP_MULT = {
    "champion": 1.6,
    "promote":  1.3,
    "keep":     1.0,
    "demote":   0.6,
    "invert":   1.0,   # newly inverted: keep capital neutral until proven
    "exec_fix": 1.0,
    "disable":  0.0,
}

# Base leverage multipliers — these are scaled by Coach confidence
# at directive lookup time. Final multiplier is bounded [0.5, 2.5].
LEV_MULT = {
    "champion": 2.0,   # top performers — biggest punch (user accepts risk)
    "promote":  1.6,
    "keep":     1.0,
    "demote":   0.7,
    "invert":   1.1,   # inverted bots get slightly more leverage (high-confidence flip)
    "exec_fix": 1.0,
    "disable":  0.0,
}
LEV_MULT_MIN = 0.5
LEV_MULT_MAX = 2.5

# Champion threshold — best of the best
WR_CHAMPION_THRESHOLD = 0.65
PF_CHAMPION_THRESHOLD = 1.50


def _safe(val, default=0.0):
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


@dataclass
class CellStats:
    bot_id: str
    n: int = 0
    wins: int = 0
    pnl_sum: float = 0.0
    gross_win: float = 0.0
    gross_loss: float = 0.0
    by_regime: Dict[int, dict] = field(default_factory=lambda: defaultdict(
        lambda: {"n": 0, "wins": 0, "pnl": 0.0}))
    # Returns series (net_return_pct/100) used for DSR computation.
    returns: List[float] = field(default_factory=list)

    @property
    def wr(self) -> float:
        return self.wins / self.n if self.n > 0 else 0.0

    @property
    def profit_factor(self) -> float:
        if self.gross_loss <= 0:
            return float("inf") if self.gross_win > 0 else 0.0
        return self.gross_win / self.gross_loss

    @property
    def avg_pnl(self) -> float:
        return self.pnl_sum / self.n if self.n > 0 else 0.0


@dataclass
class CellDecision:
    bot_id: str
    action: str                              # keep|promote|demote|invert|exec_fix|disable
    capital_multiplier: float
    leverage_multiplier: float = 1.0         # 1.5x leverage for promoted cells
    invert: bool = False
    exec_override: Optional[dict] = None     # e.g. {"sl_atr": 3.5, "max_hold": 24}
    regime_blacklist: List[int] = field(default_factory=list)
    regime_whitelist: Optional[List[int]] = None  # if set, only trade these regimes
    confidence: float = 0.0
    evidence: str = ""
    stats: dict = field(default_factory=dict)
    dsr: Optional[float] = None              # Deflated Sharpe Ratio
    dsr_multiplier: float = 1.0              # confidence boost/cut from DSR rank


# ── Core ───────────────────────────────────────────────────

def _load_trades(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _aggregate(trades: List[dict]) -> Dict[str, CellStats]:
    stats: Dict[str, CellStats] = {}
    for t in trades:
        bid = t.get("bot_id")
        if not bid:
            continue
        pnl = _safe(t.get("pnl"))
        cluster = int(_safe(t.get("regime_cluster"), -1))
        ret = _safe(t.get("net_return_pct"), 0.0) / 100.0
        s = stats.setdefault(bid, CellStats(bot_id=bid))
        s.n += 1
        s.pnl_sum += pnl
        s.returns.append(ret)
        if pnl > 0:
            s.wins += 1
            s.gross_win += pnl
        else:
            s.gross_loss += abs(pnl)
        if cluster >= 0:
            r = s.by_regime[cluster]
            r["n"] += 1
            r["pnl"] += pnl
            if pnl > 0:
                r["wins"] += 1
    return stats


def _compute_cell_dsrs(stats: Dict[str, CellStats]) -> Dict[str, dict]:
    """Compute DSR for each cell with returns >= 5. Returns dict keyed by bot_id.
    Each entry also carries 'sharpe' (raw) for fallback ranking when DSRs
    cluster at zero due to severe multi-testing penalty on small samples."""
    out = {}
    for bid, s in stats.items():
        if len(s.returns) < 5:
            out[bid] = {"dsr": None, "sharpe": None,
                        "verdict": "thin_data", "n": len(s.returns)}
            continue
        try:
            info = evaluate_cell_dsr(s.returns, n_trials=N_DSR_TRIALS)
            out[bid] = info
        except Exception:
            out[bid] = {"dsr": None, "sharpe": None,
                        "verdict": "error", "n": len(s.returns)}
    return out


def _confidence_multiplier_from_ranking(
    score: Optional[float], all_scores: List[float]
) -> float:
    """Top-quartile -> 1.5, bottom-quartile -> 0.5, middle -> 1.0."""
    if score is None or not all_scores:
        return 1.0
    sorted_s = sorted(all_scores)
    n = len(sorted_s)
    if n < 4:
        return 1.0
    rank = sum(1 for d in sorted_s if d < score) / max(1, n - 1)
    if rank >= 0.75:
        return 1.0 + (rank - 0.75) * 2.0   # 1.0 -> 1.5
    if rank <= 0.25:
        return 0.5 + rank * 2.0            # 0.5 -> 1.0
    return 1.0


def _dsr_to_confidence_multiplier(dsr: Optional[float],
                                   all_dsrs: List[float],
                                   sharpe: Optional[float] = None,
                                   all_sharpes: Optional[List[float]] = None) -> float:
    """Map a cell's DSR rank to a confidence multiplier in [0.5, 1.5].

    Fallback: when DSR variance is degenerate (all cells stuck at 0 due to
    the multi-testing penalty being too harsh for our N), rank on raw Sharpe
    instead. This preserves discrimination even when absolute DSRs are flat.
    """
    if dsr is None or not all_dsrs:
        return 1.0
    # Detect degenerate DSR distribution (e.g. all values within 1e-6 of zero)
    dsr_range = max(all_dsrs) - min(all_dsrs)
    if dsr_range < 1e-4 and sharpe is not None and all_sharpes:
        return _confidence_multiplier_from_ranking(sharpe, all_sharpes)
    return _confidence_multiplier_from_ranking(dsr, all_dsrs)


def _blend_wr(live: CellStats, cf: Optional[CellStats]) -> Tuple[float, float, int]:
    """Returns (blended_wr, blended_pnl_per_trade, effective_n).
    Weights live more heavily; cf adds confidence when live is thin."""
    if live.n >= 15:
        return live.wr, live.avg_pnl, live.n
    if cf is None or cf.n == 0:
        return live.wr, live.avg_pnl, live.n
    # Blend
    live_w = (1 - CF_WEIGHT_RATIO) if live.n > 0 else 0.0
    cf_w = 1.0 - live_w
    blended_wr = live_w * live.wr + cf_w * cf.wr
    blended_avg = live_w * live.avg_pnl + cf_w * cf.avg_pnl
    return blended_wr, blended_avg, live.n + cf.n


def _decide_cell(bot_id: str, live: CellStats, cf: Optional[CellStats]) -> CellDecision:
    """Apply decision tree for one cell.

    State-aware: reads current BOT_OVERRIDES.invert to interpret trade data.
    All trade PnL reflects performance under the CURRENTLY ACTIVE config.
    So if a bot is currently inverted and WR = 45%, the inverted strategy is
    working OK — keep the inversion. If a bot is currently NOT inverted and
    WR = 25%, original thesis is broken — flip it.
    """
    current_override = BOT_OVERRIDES.get(bot_id, {})
    currently_inverted = bool(current_override.get("invert", False))

    n_live = live.n
    n_cf = cf.n if cf else 0
    n_total = n_live + n_cf

    blended_wr, blended_avg_pnl, eff_n = _blend_wr(live, cf)

    # ── DISABLE: never trades live, plenty of opportunity ──
    # Heuristic: 0 live trades AND no CF trades either -> probably broken thesis
    if n_live == 0 and n_cf == 0:
        # No data at all -> keep, but warn
        return CellDecision(
            bot_id=bot_id, action="keep", capital_multiplier=CAP_MULT["keep"],
            confidence=0.0,
            evidence="no trades — keep until signal data appears",
            stats={"n_live": 0, "n_cf": 0},
        )

    # ── Below threshold: keep ──
    if n_total < MIN_TRADES_ANY_DECISION:
        return CellDecision(
            bot_id=bot_id, action="keep", capital_multiplier=CAP_MULT["keep"],
            confidence=0.2,
            evidence=f"thin data N={n_total} (need {MIN_TRADES_ANY_DECISION})",
            stats={"n_live": n_live, "n_cf": n_cf, "wr": round(blended_wr, 3)},
        )

    # ── FLIP DECISION: data shows current setting is wrong ──
    # Requires meaningful LIVE evidence — counterfactual alone is too noisy
    # (degraded whale/sentiment features) to justify flipping.
    has_invert_evidence = (
        n_live >= MIN_TRADES_INVERT_LIVE
        or (n_live >= 5 and n_total >= MIN_TRADES_INVERT_TOTAL)
    )

    if blended_wr < WR_INVERT_THRESHOLD and has_invert_evidence:
        # Performance is bad under CURRENT config -> flip whatever it is now.
        new_invert = not currently_inverted
        verb = "flip ON inversion" if new_invert else "flip OFF inversion"
        decision = CellDecision(
            bot_id=bot_id, action="invert", invert=new_invert,
            capital_multiplier=CAP_MULT["invert"], leverage_multiplier=LEV_MULT["invert"],
            confidence=min(0.9, (WR_INVERT_THRESHOLD - blended_wr) * 3 + 0.3),
            evidence=(f"WR {blended_wr:.0%} over N={eff_n} under current "
                      f"setting (invert={currently_inverted}) "
                      f"(live {live.wr:.0%}/{n_live}, cf {cf.wr if cf else 0:.0%}/{n_cf}) "
                      f"-> {verb}"),
            stats={"n_live": n_live, "n_cf": n_cf, "wr": round(blended_wr, 3),
                   "avg_pnl": round(blended_avg_pnl, 3),
                   "was_inverted": currently_inverted},
        )
        _maybe_regime_gate(decision, live)
        return decision

    # ── DISABLE: catastrophic bleeder with lots of LIVE data ──
    # Never disable based on counterfactual alone — features differ.
    if (live.n >= MIN_TRADES_DISABLE
            and live.wr < WR_INVERT_THRESHOLD
            and live.avg_pnl < -0.2  # losing >$0.20 per trade on $125 base
            and live.profit_factor < 0.7):
        return CellDecision(
            bot_id=bot_id, action="disable",
            capital_multiplier=CAP_MULT["disable"],
            leverage_multiplier=LEV_MULT["disable"],
            invert=currently_inverted,
            confidence=0.85,
            evidence=(f"catastrophic: WR {live.wr:.0%}/{live.n}, "
                      f"avg PnL ${live.avg_pnl:+.2f}, PF {live.profit_factor:.2f}"),
            stats={"n_live": n_live, "n_cf": n_cf, "wr": round(live.wr, 3),
                   "pf": round(live.profit_factor, 2),
                   "was_inverted": currently_inverted},
        )

    # ── DEMOTE: losing but not crushed enough to flip ──
    if (n_total >= MIN_TRADES_ANY_DECISION
            and blended_wr < WR_DEMOTE_THRESHOLD
            and blended_avg_pnl < 0):
        decision = CellDecision(
            bot_id=bot_id, action="demote",
            capital_multiplier=CAP_MULT["demote"],
            leverage_multiplier=LEV_MULT["demote"],
            invert=currently_inverted,  # preserve manual setting
            confidence=0.5,
            evidence=(f"underperforming: WR {blended_wr:.0%}/{eff_n}, "
                      f"avg PnL ${blended_avg_pnl:+.2f}"),
            stats={"n_live": n_live, "n_cf": n_cf, "wr": round(blended_wr, 3),
                   "avg_pnl": round(blended_avg_pnl, 3),
                   "was_inverted": currently_inverted},
        )
        _maybe_regime_gate(decision, live)
        return decision

    # ── CHAMPION: top tier — winning hard, max leverage ──
    if (n_live >= 7 and live.wr >= WR_CHAMPION_THRESHOLD
            and live.profit_factor >= PF_CHAMPION_THRESHOLD):
        decision = CellDecision(
            bot_id=bot_id, action="champion",
            capital_multiplier=CAP_MULT["champion"],
            leverage_multiplier=LEV_MULT["champion"],
            invert=currently_inverted,
            confidence=min(0.95, (live.wr - WR_CHAMPION_THRESHOLD) * 5 + 0.55),
            evidence=(f"CHAMPION: WR {live.wr:.0%}/{live.n}, PF {live.profit_factor:.2f}, "
                      f"avg ${live.avg_pnl:+.2f} — max leverage authorized"),
            stats={"n_live": n_live, "n_cf": n_cf, "wr": round(live.wr, 3),
                   "pf": round(live.profit_factor, 2),
                   "was_inverted": currently_inverted},
        )
        _maybe_regime_gate(decision, live)
        return decision

    # ── PROMOTE: winning consistently under current setting ──
    if (n_live >= 8 and live.wr >= WR_PROMOTE_THRESHOLD
            and live.profit_factor >= PF_PROMOTE_THRESHOLD):
        decision = CellDecision(
            bot_id=bot_id, action="promote",
            capital_multiplier=CAP_MULT["promote"],
            leverage_multiplier=LEV_MULT["promote"],
            invert=currently_inverted,  # preserve current effective state
            confidence=min(0.9, (live.wr - WR_PROMOTE_THRESHOLD) * 5 + 0.4),
            evidence=(f"winning: WR {live.wr:.0%}/{live.n}, "
                      f"PF {live.profit_factor:.2f}, avg ${live.avg_pnl:+.2f}"),
            stats={"n_live": n_live, "n_cf": n_cf, "wr": round(live.wr, 3),
                   "pf": round(live.profit_factor, 2),
                   "was_inverted": currently_inverted},
        )
        _maybe_regime_gate(decision, live)
        return decision

    # ── KEEP (default) with possible regime gating — preserves current invert ──
    decision = CellDecision(
        bot_id=bot_id, action="keep", capital_multiplier=CAP_MULT["keep"],
        leverage_multiplier=LEV_MULT["keep"],
        invert=currently_inverted,  # do not touch what was set manually
        confidence=0.4,
        evidence=(f"mixed: WR {blended_wr:.0%}/{eff_n}, "
                  f"avg PnL ${blended_avg_pnl:+.2f}"
                  + (f" (currently inverted)" if currently_inverted else "")),
        stats={"n_live": n_live, "n_cf": n_cf, "wr": round(blended_wr, 3),
               "avg_pnl": round(blended_avg_pnl, 3),
               "was_inverted": currently_inverted},
    )
    _maybe_regime_gate(decision, live)
    return decision


def _maybe_regime_gate(decision: CellDecision, live: CellStats):
    """Look at per-regime live stats; gate regimes that are clearly bad.

    For inverted cells (or cells whose decision is to invert), use the stricter
    invert-specific thresholds: N>=3 with 0 wins is already a hard gate. This
    reflects live observation that inversion fails catastrophically in trend
    regimes — we want to gate those off after very few losses, not wait for a
    statistically clean sample.
    """
    is_inverted = decision.invert
    min_n_gate = MIN_TRADES_INVERT_REGIME_GATE if is_inverted else MIN_TRADES_REGIME_GATE
    wr_bad_threshold = (WR_INVERT_REGIME_BAD if is_inverted
                        else WR_REGIME_BAD_THRESHOLD)

    blacklist = []
    whitelist_candidates = []
    regime_summary = {}
    for cid, r in live.by_regime.items():
        if r["n"] < min_n_gate:
            continue
        wr = r["wins"] / r["n"]
        avg = r["pnl"] / r["n"]
        regime_summary[cid] = {"n": r["n"], "wr": round(wr, 3), "avg_pnl": round(avg, 3)}
        if wr <= wr_bad_threshold and avg < 0:
            blacklist.append(cid)
        elif wr >= WR_REGIME_GOOD_THRESHOLD:
            whitelist_candidates.append(cid)
    if blacklist:
        decision.regime_blacklist = sorted(blacklist)
        gate_tag = "INV-gate" if is_inverted else "gate"
        decision.evidence += f"; {gate_tag}-off regimes {blacklist}"
    if regime_summary:
        decision.stats["by_regime"] = regime_summary


def _apply_hysteresis(bot_id: str, new: CellDecision, prior_state: dict) -> CellDecision:
    """If a prior decision exists for this cell and the new decision changes the
    action with insufficient confidence, keep the prior action+invert flag but
    let everything else (stats, evidence, regime_blacklist) refresh.

    Always allow:
      - Disable (catastrophic, never delay)
      - Champion -> any (champion is the highest tier; if data no longer
        supports it, we want to drop fast)
      - Any -> any if new confidence >= HYSTERESIS_CONFIDENCE_THRESHOLD
    """
    prior_d = prior_state.get("decisions", {}).get(bot_id)
    if not prior_d:
        return new

    prior_action = prior_d.get("action", "keep")
    if prior_action == new.action:
        return new  # no action change, nothing to hold back
    if new.action == "disable":
        return new
    if prior_action == "champion":
        return new  # always allow champion downgrade
    if new.confidence >= HYSTERESIS_CONFIDENCE_THRESHOLD:
        return new

    # Hold previous action; carry fresh evidence/stats forward.
    held = CellDecision(
        bot_id=bot_id,
        action=prior_action,
        capital_multiplier=CAP_MULT.get(prior_action, 1.0),
        leverage_multiplier=LEV_MULT.get(prior_action, 1.0),
        invert=bool(prior_d.get("invert", False)),
        exec_override=prior_d.get("exec_override"),
        regime_blacklist=new.regime_blacklist,  # use fresh regime data
        regime_whitelist=new.regime_whitelist,
        confidence=new.confidence,
        evidence=(f"[HYSTERESIS held={prior_action}, new={new.action} "
                  f"conf={new.confidence:.2f} < {HYSTERESIS_CONFIDENCE_THRESHOLD}] "
                  + new.evidence),
        stats=new.stats,
    )
    return held


class Coach:
    """Top-level entry point. Reads trade data, emits per-cell decisions."""

    def __init__(self,
                 trades_path: str = TRADES_CSV,
                 counterfactual_path: str = COUNTERFACTUAL_CSV,
                 signals_path: str = SIGNALS_CSV):
        self.trades_path = trades_path
        self.cf_path = counterfactual_path
        self.signals_path = signals_path

    def evaluate(self, all_bot_ids: Optional[List[str]] = None,
                 apply_hysteresis: bool = True) -> Dict[str, CellDecision]:
        """Returns {bot_id: CellDecision}. If all_bot_ids given, ensures every
        cell has an entry (default keep) even with zero trades.

        Hysteresis: if a prior coach_state.json exists, a new action only
        replaces the prior action when the new confidence >= threshold.
        Otherwise the prior action+invert flag is preserved (but freshly-
        computed stats, evidence, and regime_blacklist still flow through).

        DSR: per-cell Deflated Sharpe Ratio is computed and used as a
        confidence multiplier. Cells in the top DSR quartile get a boost,
        bottom quartile cells get cut. Final confidence drives downstream
        Coach action thresholds (e.g. hysteresis).
        """
        live_trades = _load_trades(self.trades_path)
        cf_trades = _load_trades(self.cf_path)
        live_stats = _aggregate(live_trades)
        cf_stats = _aggregate(cf_trades)

        bot_ids = set(live_stats.keys()) | set(cf_stats.keys())
        if all_bot_ids:
            bot_ids |= set(all_bot_ids)

        # DSR pass over LIVE cells (use live returns only — cf is degraded
        # for whale/sentiment dependent bots).
        dsr_info = _compute_cell_dsrs(live_stats)
        all_dsrs = [info["dsr"] for info in dsr_info.values()
                    if info.get("dsr") is not None]
        all_sharpes = [info["sharpe"] for info in dsr_info.values()
                       if info.get("sharpe") is not None]

        prior = load_coach_state() if apply_hysteresis else None

        decisions = {}
        for bid in sorted(bot_ids):
            live = live_stats.get(bid, CellStats(bot_id=bid))
            cf = cf_stats.get(bid)
            new_decision = _decide_cell(bid, live, cf)

            # Attach DSR + multiplier, apply to confidence.
            cell_dsr_info = dsr_info.get(bid, {})
            dsr_val = cell_dsr_info.get("dsr")
            sharpe_val = cell_dsr_info.get("sharpe")
            mult = _dsr_to_confidence_multiplier(
                dsr_val, all_dsrs, sharpe_val, all_sharpes)
            new_decision.dsr = dsr_val
            new_decision.dsr_multiplier = round(mult, 3)
            new_decision.confidence = round(
                max(0.0, min(0.99, new_decision.confidence * mult)), 3)
            if cell_dsr_info.get("verdict"):
                new_decision.stats["dsr_verdict"] = cell_dsr_info["verdict"]
                new_decision.stats["sharpe"] = cell_dsr_info.get("sharpe")

            if prior is not None:
                new_decision = _apply_hysteresis(bid, new_decision, prior)
            decisions[bid] = new_decision
        return decisions

    def write_state(self, decisions: Dict[str, CellDecision],
                    out_path: str = COACH_STATE_FILE) -> dict:
        """Serialize decisions to coach_state.json. Returns the written payload."""
        payload = {
            "version": "coach-1.0",
            "last_update": datetime.now(timezone.utc).isoformat(),
            "thresholds": {
                "wr_invert": WR_INVERT_THRESHOLD,
                "wr_demote": WR_DEMOTE_THRESHOLD,
                "wr_promote": WR_PROMOTE_THRESHOLD,
                "wr_regime_bad": WR_REGIME_BAD_THRESHOLD,
                "min_trades_invert_live": MIN_TRADES_INVERT_LIVE,
                "min_trades_invert_total": MIN_TRADES_INVERT_TOTAL,
                "min_trades_invert_regime_gate": MIN_TRADES_INVERT_REGIME_GATE,
                "cf_weight_ratio": CF_WEIGHT_RATIO,
                "hysteresis_confidence": HYSTERESIS_CONFIDENCE_THRESHOLD,
            },
            "decisions": {bid: _decision_to_dict(d) for bid, d in decisions.items()},
        }
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, out_path)
        return payload

    def summary(self, decisions: Dict[str, CellDecision]) -> str:
        """Human-readable one-block summary for terminal / Telegram."""
        by_action = defaultdict(list)
        for bid, d in decisions.items():
            by_action[d.action].append(bid)
        order = ["champion", "promote", "invert", "exec_fix", "keep", "demote", "disable"]
        lines = []
        for act in order:
            cells = sorted(by_action.get(act, []))
            if cells:
                lines.append(f"  {act:8s} ({len(cells):2d}): {', '.join(cells)}")
        return "Coach decisions:\n" + "\n".join(lines)


def _decision_to_dict(d: CellDecision) -> dict:
    return {
        "action": d.action,
        "capital_multiplier": d.capital_multiplier,
        "leverage_multiplier": d.leverage_multiplier,
        "invert": d.invert,
        "exec_override": d.exec_override,
        "regime_blacklist": d.regime_blacklist,
        "regime_whitelist": d.regime_whitelist,
        "confidence": round(d.confidence, 3),
        "dsr": round(d.dsr, 4) if d.dsr is not None else None,
        "dsr_multiplier": d.dsr_multiplier,
        "evidence": d.evidence,
        "stats": d.stats,
    }


def load_coach_state(path: str = COACH_STATE_FILE) -> Optional[dict]:
    """Read coach_state.json. Returns None if missing or corrupt — never raises."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if data.get("version", "").startswith("coach-"):
            return data
    except (json.JSONDecodeError, OSError):
        return None
    return None


def get_cell_directive(bot_id: str, state: Optional[dict] = None) -> dict:
    """Lookup helper used by base_bot / capital. Always returns a safe dict.
    Falls back to {action=keep, capital_multiplier=1.0, invert=False, ...}.

    Leverage multiplier is confidence-scaled: high-confidence directives get
    closer to the action's base LEV_MULT, low-confidence ones blend toward 1.0
    (neutral). Final value is clamped to [LEV_MULT_MIN, LEV_MULT_MAX].
    """
    if state is None:
        state = load_coach_state()
    fallback = {
        "action": "keep", "capital_multiplier": 1.0, "leverage_multiplier": 1.0,
        "invert": False, "exec_override": None,
        "regime_blacklist": [], "regime_whitelist": None,
    }
    if state is None:
        return fallback
    d = state.get("decisions", {}).get(bot_id)
    if not d:
        return fallback

    # Confidence-scaled leverage: blend base lev with 1.0 by confidence.
    raw_lev = float(d.get("leverage_multiplier", 1.0))
    conf = float(d.get("confidence", 0.4))
    # Above 1.0: take a confidence-weighted average between raw and 1.0
    # Below 1.0: same (de-leverage is also confidence-weighted)
    scaled_lev = 1.0 + (raw_lev - 1.0) * conf
    scaled_lev = max(LEV_MULT_MIN, min(LEV_MULT_MAX, scaled_lev))

    return {
        "action": d.get("action", "keep"),
        "capital_multiplier": float(d.get("capital_multiplier", 1.0)),
        "leverage_multiplier": scaled_lev,
        "invert": bool(d.get("invert", False)),
        "exec_override": d.get("exec_override"),
        "regime_blacklist": d.get("regime_blacklist", []) or [],
        "regime_whitelist": d.get("regime_whitelist"),
    }
