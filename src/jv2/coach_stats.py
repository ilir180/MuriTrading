"""JV Boting v2 — Coach Statistics (Anti-Overfitting Layer).

Implements two non-negotiable filters from Bailey & Lopez de Prado:

1. **Deflated Sharpe Ratio (DSR)** — adjusts an observed Sharpe for the
   number of trials (multiple testing) and for non-normality (skew/kurtosis).
   The Coach has implicitly tested many strategies (cells × regimes × Coach
   variants); a raw Sharpe of 1.5 from one cell is much less significant
   when you've tested 200 cells than when you've tested 1. DSR returns the
   probability that the observed Sharpe is genuinely above 0 after all
   adjustments.

2. **Probability of Backtest Overfitting (PBO)** — for a set of strategies
   evaluated on multiple time-splits, computes the probability that the
   best in-sample strategy will be below median out-of-sample. PBO > 0.5
   means selection is statistically indistinguishable from random luck.

References:
- Bailey, López de Prado (2014): "The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting and Non-Normality"
- Bailey et al. (2017): "The Probability of Backtest Overfitting"
- Lopez de Prado (2018): "Advances in Financial Machine Learning" ch. 7-8

Used as a hard gate in coach.py for promote/champion decisions.
"""

import math
from typing import Sequence, Tuple, List, Optional

# Euler-Mascheroni constant — appears in expected-max-of-N-normals formula.
EULER_GAMMA = 0.5772156649015329


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via error function — no scipy dependency."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Inverse normal CDF (probit). Beasley-Springer-Moro approximation.
    Accurate to ~1e-9 across the full domain. No scipy."""
    if p <= 0.0 or p >= 1.0:
        if p == 0.0:
            return float("-inf")
        if p == 1.0:
            return float("inf")
        raise ValueError(f"p must be in (0,1), got {p}")

    # Coefficients (Beasley-Springer)
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]

    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)

    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _sample_moments(returns: Sequence[float]) -> Tuple[float, float, float, float]:
    """Return (mean, std, skew, excess_kurt) — biased/sample versions."""
    n = len(returns)
    if n < 2:
        return 0.0, 0.0, 0.0, 0.0
    m = sum(returns) / n
    var = sum((x - m) ** 2 for x in returns) / n
    std = math.sqrt(var) if var > 0 else 0.0
    if std == 0:
        return m, 0.0, 0.0, 0.0
    skew = sum((x - m) ** 3 for x in returns) / (n * std ** 3)
    kurt = sum((x - m) ** 4 for x in returns) / (n * std ** 4) - 3.0
    return m, std, skew, kurt


def sharpe_ratio(returns: Sequence[float], ann_factor: float = 1.0) -> float:
    """Plain Sharpe Ratio (no risk-free adjustment — returns assumed excess)."""
    if len(returns) < 2:
        return 0.0
    m, std, _, _ = _sample_moments(returns)
    if std == 0:
        return 0.0
    return m / std * math.sqrt(ann_factor)


def expected_max_sharpe(n_trials: int, std_sharpe_across_trials: float = 1.0) -> float:
    """Expected maximum Sharpe from N independent trials under H0 (true Sharpe=0).

    Closed-form approximation (Bailey/Lopez de Prado 2014, eq. 6):
        E[max_SR] ≈ V[SR_across_trials]^0.5 * (
            (1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e))
        )
    where γ is the Euler-Mascheroni constant.

    `std_sharpe_across_trials` is the variance of observed Sharpes across the
    N trials — if unknown, default to 1.0 (conservative).
    """
    if n_trials < 2:
        return 0.0
    e = math.e
    inner = (1 - EULER_GAMMA) * _norm_ppf(1 - 1.0 / n_trials) \
          + EULER_GAMMA * _norm_ppf(1 - 1.0 / (n_trials * e))
    return std_sharpe_across_trials * inner


def deflated_sharpe_ratio(
    observed_sharpe: float,
    n_observations: int,
    n_trials: int,
    skew: float = 0.0,
    excess_kurtosis: float = 0.0,
    std_sharpe_across_trials: float = 1.0,
) -> float:
    """Returns the probability that the observed Sharpe is genuinely > 0
    after adjusting for (a) multiple-testing selection bias (n_trials) and
    (b) non-normality of returns (skew, kurtosis).

    Output in [0, 1]. Convention used by Coach:
      DSR >= 0.95   strong evidence — promote OK
      DSR >= 0.80   moderate evidence — keep OK
      DSR <  0.80   weak/noise — do not promote
      DSR <  0.50   actively negative — consider disable

    Args:
      observed_sharpe: realized SR of the strategy (annualized or per-period
        — must match scale of n_observations)
      n_observations: number of return observations producing the SR
      n_trials: number of strategies effectively tested in selection
        (for us: ~32 cells, or more if we count Coach variants)
      skew, excess_kurtosis: from sample moments of the returns
      std_sharpe_across_trials: variance of Sharpes across trials (default 1.0)
    """
    if n_observations < 5:
        return 0.0
    sr_expected = expected_max_sharpe(n_trials, std_sharpe_across_trials)

    sr = observed_sharpe
    # Guard 1: infinite/huge SR (perfect return series with std≈0).
    # Treat as "obviously positive" if mean is positive (and the series is finite).
    if not math.isfinite(sr):
        return 0.999 if sr > 0 else 0.001
    # Clamp extreme SR to avoid the kurtosis term blowing the denominator.
    SR_CLAMP = 25.0
    if abs(sr) > SR_CLAMP:
        sr = math.copysign(SR_CLAMP, sr)

    # Standardized non-normality-adjusted Sharpe (Bailey/LdP 2014, eq. 9)
    denom_sq = 1.0 - skew * sr + (excess_kurtosis / 4.0) * (sr ** 2)
    if denom_sq <= 0:
        # Numerically unstable — return conservative low
        return 0.0
    denom = math.sqrt(denom_sq)
    numer = (sr - sr_expected) * math.sqrt(n_observations - 1)
    z = numer / denom
    return _norm_cdf(z)


def deflated_sharpe_from_returns(
    returns: Sequence[float],
    n_trials: int,
    std_sharpe_across_trials: float = 1.0,
) -> Tuple[float, dict]:
    """Convenience wrapper. Returns (DSR, info-dict)."""
    n = len(returns)
    m, std, skew, ek = _sample_moments(returns)
    sr = sharpe_ratio(returns)
    dsr = deflated_sharpe_ratio(
        observed_sharpe=sr,
        n_observations=n,
        n_trials=n_trials,
        skew=skew,
        excess_kurtosis=ek,
        std_sharpe_across_trials=std_sharpe_across_trials,
    )
    return dsr, {
        "n": n,
        "mean": round(m, 6),
        "std": round(std, 6),
        "skew": round(skew, 4),
        "excess_kurtosis": round(ek, 4),
        "sharpe": round(sr, 4),
        "expected_max_sharpe": round(expected_max_sharpe(n_trials, std_sharpe_across_trials), 4),
        "dsr": round(dsr, 4),
    }


# ── PBO ─────────────────────────────────────────────────────

def probability_backtest_overfitting(
    is_oos_pairs: List[Tuple[List[float], List[float]]],
) -> Tuple[float, dict]:
    """Probability of Backtest Overfitting (Bailey et al. 2017).

    Each element of `is_oos_pairs` is a tuple of (IS_performances, OOS_performances)
    where each list contains the performance of N candidate strategies. Across
    all pairs, we compute how often the IS-best strategy is OOS-below-median.

    PBO ≈ 0    selection is sound
    PBO ≈ 0.5  selection is no better than random
    PBO >  0.5 selection is actively anti-correlated with OOS (worst case)

    Implementation: rank-based. For each split, the relative rank of the
    IS-best strategy among the OOS performances determines its OOS quantile;
    if that quantile is < 0.5, we count it as overfit.
    """
    if not is_oos_pairs:
        return 1.0, {"n_splits": 0, "n_overfit": 0}

    n_overfit = 0
    n_splits = 0
    rank_records = []
    for is_perfs, oos_perfs in is_oos_pairs:
        n = len(is_perfs)
        if n < 2 or n != len(oos_perfs):
            continue
        # Best by IS
        best_idx = max(range(n), key=lambda i: is_perfs[i])
        # OOS rank of best (0-indexed, 0 = worst, n-1 = best)
        oos_best_value = oos_perfs[best_idx]
        oos_rank = sum(1 for v in oos_perfs if v < oos_best_value)
        # Logit: w = rank / (n - 1) in [0, 1]. <0.5 means below median.
        w = oos_rank / (n - 1) if n > 1 else 0.5
        rank_records.append(w)
        if w < 0.5:
            n_overfit += 1
        n_splits += 1

    if n_splits == 0:
        return 1.0, {"n_splits": 0, "n_overfit": 0}
    pbo = n_overfit / n_splits
    return pbo, {
        "n_splits": n_splits,
        "n_overfit": n_overfit,
        "mean_oos_quantile_of_is_best": (sum(rank_records) / len(rank_records))
                                        if rank_records else 0.5,
    }


# ── Convenience: per-cell evaluation ─────────────────────────

def evaluate_cell(
    pnl_series: Sequence[float],
    n_trials: int,
    promote_dsr_threshold: float = 0.80,
    champion_dsr_threshold: float = 0.95,
) -> dict:
    """Run the full DSR pipeline on a cell's PnL series and return a dict
    with the verdict the Coach can act on directly.

    Verdict values:
      'champion_ok'  DSR >= champion threshold (default 0.95)
      'promote_ok'   DSR in [promote, champion) — promote allowed
      'keep_only'    DSR in [0.5, promote)     — neither promote nor demote
      'demote_risk'  DSR < 0.5                — actively below noise level
    """
    if len(pnl_series) < 5:
        return {"verdict": "thin_data", "dsr": None, "n": len(pnl_series)}
    dsr, info = deflated_sharpe_from_returns(pnl_series, n_trials=n_trials)
    if dsr >= champion_dsr_threshold:
        verdict = "champion_ok"
    elif dsr >= promote_dsr_threshold:
        verdict = "promote_ok"
    elif dsr >= 0.5:
        verdict = "keep_only"
    else:
        verdict = "demote_risk"
    info["verdict"] = verdict
    return info
