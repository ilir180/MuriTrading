"""JV Boting v2 — Hierarchical Risk Parity allocator.

Lopez de Prado (2016): "Building Diversified Portfolios that Outperform Out-of-Sample."

HRP fixes two problems with classical Markowitz mean-variance:
1. Inverts the covariance matrix (numerically unstable when N >= 20 with
   short data history — exactly our regime).
2. Concentrates allocations on outliers.

The algorithm:
  Step 1 — Tree clustering: hierarchical clustering on distance matrix
           D = sqrt((1 - corr) / 2). Use single linkage for stability.
  Step 2 — Quasi-diagonalization: reorder rows/cols of covariance so
           highly-correlated assets are adjacent.
  Step 3 — Recursive bisection: split clusters in two, allocate inversely
           to cluster variance, recurse.

No scipy/sklearn dependency — we roll our own minimal HRP using only
the standard library + the small linkage primitive below.
"""

import math
from typing import Dict, List, Tuple, Optional


# ── Linear algebra mini-helpers (no numpy required) ─────────────

def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _cov(a: List[float], b: List[float]) -> float:
    n = min(len(a), len(b))
    if n < 2:
        return 0.0
    ma = sum(a[:n]) / n
    mb = sum(b[:n]) / n
    return sum((a[i] - ma) * (b[i] - mb) for i in range(n)) / (n - 1)


def _std(xs: List[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return math.sqrt(var) if var > 0 else 0.0


def correlation_matrix(returns_by_id: Dict[str, List[float]]) -> Tuple[List[str], List[List[float]]]:
    """Returns (ordered_ids, corr_matrix). Pairs are aligned by truncation
    to the shortest series. Cells with std=0 are returned with correlation 0
    to keep the matrix well-defined."""
    ids = sorted(returns_by_id.keys())
    n = len(ids)
    series = [returns_by_id[i] for i in ids]
    stds = [_std(s) for s in series]
    corr = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                corr[i][j] = 1.0
                continue
            if stds[i] == 0 or stds[j] == 0:
                corr[i][j] = 0.0
                continue
            c = _cov(series[i], series[j])
            corr[i][j] = max(-1.0, min(1.0, c / (stds[i] * stds[j])))
    return ids, corr


def covariance_matrix(returns_by_id: Dict[str, List[float]]) -> Tuple[List[str], List[List[float]]]:
    """Returns (ordered_ids, cov_matrix). Same alignment as correlation_matrix."""
    ids = sorted(returns_by_id.keys())
    n = len(ids)
    series = [returns_by_id[i] for i in ids]
    cov = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            cov[i][j] = _cov(series[i], series[j])
    return ids, cov


# ── Hierarchical clustering (single linkage) ────────────────────

def _distance(corr: List[List[float]]) -> List[List[float]]:
    """D_ij = sqrt((1 - rho_ij) / 2). Range [0, 1]."""
    n = len(corr)
    return [[math.sqrt(max(0.0, (1.0 - corr[i][j]) / 2.0)) for j in range(n)]
            for i in range(n)]


def _single_linkage(dist: List[List[float]]) -> List[Tuple[int, int, float]]:
    """Agglomerative single-linkage clustering. Returns list of (cluster_a,
    cluster_b, distance) — N-1 merges, indexed 0..N-1 for leaves and
    N..2N-2 for internal nodes (like scipy.cluster.hierarchy.linkage)."""
    n = len(dist)
    # Active cluster list: each holds list of leaf indices.
    clusters: Dict[int, List[int]] = {i: [i] for i in range(n)}
    next_id = n
    links: List[Tuple[int, int, float]] = []

    # Active distance map between current clusters
    def cluster_dist(a_ids: List[int], b_ids: List[int]) -> float:
        return min(dist[a][b] for a in a_ids for b in b_ids)

    while len(clusters) > 1:
        # Find closest pair
        ids_now = list(clusters.keys())
        best_d = float("inf")
        best_pair = (ids_now[0], ids_now[1])
        for i in range(len(ids_now)):
            for j in range(i + 1, len(ids_now)):
                a, b = ids_now[i], ids_now[j]
                d = cluster_dist(clusters[a], clusters[b])
                if d < best_d:
                    best_d = d
                    best_pair = (a, b)
        a, b = best_pair
        new_id = next_id
        next_id += 1
        clusters[new_id] = clusters[a] + clusters[b]
        del clusters[a]
        del clusters[b]
        links.append((a, b, best_d))
    return links


def _quasi_diag_order(links: List[Tuple[int, int, float]], n: int) -> List[int]:
    """From the linkage matrix, derive the quasi-diagonal ordering of leaves."""
    if not links:
        return list(range(n))
    # Build a list starting from the root and expand internal nodes to leaves.
    final_link_id = n + len(links) - 1
    order = [final_link_id]
    # Map from internal node id to (a, b)
    link_map = {n + i: (links[i][0], links[i][1]) for i in range(len(links))}

    def expand(node):
        if node < n:
            return [node]
        a, b = link_map[node]
        return expand(a) + expand(b)

    return expand(final_link_id)


# ── HRP recursive bisection ─────────────────────────────────────

def _inverse_variance_weights(diag_vars: List[float]) -> List[float]:
    """w_i ∝ 1/var_i, normalized to sum to 1."""
    invs = [1.0 / v if v > 0 else 0.0 for v in diag_vars]
    s = sum(invs)
    if s == 0:
        n = len(invs)
        return [1.0 / n] * n
    return [x / s for x in invs]


def _cluster_var(cov: List[List[float]], idxs: List[int]) -> float:
    """Variance of a cluster under inverse-variance allocation within."""
    if not idxs:
        return 0.0
    if len(idxs) == 1:
        return cov[idxs[0]][idxs[0]]
    diag = [cov[i][i] for i in idxs]
    w = _inverse_variance_weights(diag)
    # var = w' * Cov_sub * w
    total = 0.0
    for i, ai in enumerate(idxs):
        for j, aj in enumerate(idxs):
            total += w[i] * w[j] * cov[ai][aj]
    return total


def _recursive_bisection(cov: List[List[float]], order: List[int]) -> Dict[int, float]:
    """Allocate inversely to cluster variance using top-down recursive bisection.
    Returns {leaf_idx: weight}."""
    weights = {i: 1.0 for i in order}
    stack = [order]
    while stack:
        cluster = stack.pop()
        if len(cluster) <= 1:
            continue
        mid = len(cluster) // 2
        left = cluster[:mid]
        right = cluster[mid:]
        var_left = _cluster_var(cov, left)
        var_right = _cluster_var(cov, right)
        if var_left + var_right == 0:
            alpha = 0.5
        else:
            alpha = 1.0 - var_left / (var_left + var_right)
        for i in left:
            weights[i] *= alpha
        for i in right:
            weights[i] *= (1.0 - alpha)
        stack.append(left)
        stack.append(right)
    return weights


def hierarchical_risk_parity(returns_by_id: Dict[str, List[float]],
                              min_observations: int = 5) -> Dict[str, float]:
    """Top-level HRP entry point.

    Returns a dict {bot_id: weight}, weights summing to 1.0. Cells with
    fewer than min_observations returns are given equal-weight fallback
    among themselves and merged into the final allocation.

    The algorithm degrades gracefully:
    - If N < 4 active cells: equal weights
    - If all returns identical (no variance): equal weights
    - Otherwise: full HRP
    """
    # Filter to cells with enough data
    active = {bid: rs for bid, rs in returns_by_id.items()
              if len(rs) >= min_observations and _std(rs) > 0}
    if len(active) < 2:
        # Not enough data for HRP — equal weights
        all_ids = list(returns_by_id.keys())
        if not all_ids:
            return {}
        w = 1.0 / len(all_ids)
        return {i: w for i in all_ids}

    # Align all series to the shortest length (truncate from start)
    min_len = min(len(s) for s in active.values())
    aligned = {bid: s[-min_len:] for bid, s in active.items()}

    ids, corr = correlation_matrix(aligned)
    _, cov = covariance_matrix(aligned)
    dist = _distance(corr)
    links = _single_linkage(dist)
    order = _quasi_diag_order(links, len(ids))
    raw_weights = _recursive_bisection(cov, order)

    # Map back to bot_ids
    out = {ids[idx]: raw_weights[idx] for idx in raw_weights}

    # Cells that didn't qualify get tiny equal-weight share of 5% of total
    inactive = [bid for bid in returns_by_id if bid not in out]
    if inactive:
        # Normalize active to 95%, give inactive 5% split equally
        active_sum = sum(out.values())
        if active_sum > 0:
            scale = 0.95 / active_sum
            out = {b: w * scale for b, w in out.items()}
        share = 0.05 / len(inactive)
        for bid in inactive:
            out[bid] = share
    # Final normalization
    total = sum(out.values())
    if total > 0:
        out = {b: w / total for b, w in out.items()}
    return out


# ── Quarter-Kelly position sizing ───────────────────────────────

def quarter_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float,
                            kelly_fraction: float = 0.25) -> float:
    """Compute fractional Kelly bet size as fraction of capital at risk.

    Standard Kelly: f* = (p·b - q) / b, where:
      p = win rate
      q = 1 - p
      b = avg_win / avg_loss  (payoff ratio)

    Quarter-Kelly (default 0.25) cushions against:
      - Estimation error in win_rate/payoff (Backtest -> Live drift)
      - Non-stationary edge
      - Path-dependent drawdown
    Returns f in [0.0, 1.0] (clamped). 0 means "don't bet".
    """
    if avg_loss <= 0 or win_rate <= 0 or win_rate >= 1:
        return 0.0
    b = avg_win / avg_loss
    if b <= 0:
        return 0.0
    p = win_rate
    q = 1 - p
    full_kelly = (p * b - q) / b
    if full_kelly <= 0:
        return 0.0
    fractional = full_kelly * kelly_fraction
    return max(0.0, min(1.0, fractional))
