"""JV Boting v2 — Regime Clustering.

Defines the *structure* of market regimes via k-means on regime features.
This module does NOT compute or persist any PnL/performance information —
it only answers "what kind of market state is this bar?".

Per-cluster bot performance is tracked separately, exclusively from live
regime-tagged trades. The Coach reads that live performance to pick lineups;
this clusterer is the input layer (taxonomy of market states), not the
decision layer.

Usage:
    from src.jv2.regime_clusterer import RegimeClusterer
    rc = RegimeClusterer.load()           # load fitted model
    cid = rc.assign(regime_dict)           # int cluster id, or -1 if model missing
    desc = rc.describe(cid)                # human-readable cluster name
"""

import json
import os
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import numpy as np


# ── Feature set used for clustering ────────────────────────
# Order matters — used to vectorize regime dicts consistently.
REGIME_FEATURES = [
    "adx",
    "rsi",
    "bb_pos",
    "bbw",
    "atr_pct",
    "chop",
    "trend_consistency",
]

PROJECT_ROOT = os.path.expanduser("~/MuriTrading")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "regime")
KMEANS_PATH = os.path.join(MODEL_DIR, "kmeans.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
META_PATH   = os.path.join(MODEL_DIR, "meta.json")


@dataclass
class ClusterMeta:
    k: int
    features: List[str]
    centroids: List[List[float]]      # in feature units (de-standardized)
    centroids_z: List[List[float]]    # in z-score units (standardized)
    descriptions: List[str]
    fractions: List[float]            # share of training rows per cluster
    fit_n: int
    fit_range: List[str]              # [min_ts, max_ts]
    inertia: float


class RegimeClusterer:
    def __init__(self, kmeans=None, scaler=None, meta: Optional[ClusterMeta] = None):
        self.kmeans = kmeans
        self.scaler = scaler
        self.meta = meta

    # ── Vectorize a regime dict ────────────────────────────
    @staticmethod
    def to_vector(regime: Dict) -> np.ndarray:
        return np.array([float(regime.get(f, 0.0) or 0.0) for f in REGIME_FEATURES],
                        dtype=float)

    # ── Assignment ──────────────────────────────────────────
    def assign(self, regime: Dict) -> int:
        """Return cluster id for a regime dict, or -1 if model not loaded
        or any feature missing/NaN beyond a tolerable margin."""
        if self.kmeans is None or self.scaler is None:
            return -1
        v = self.to_vector(regime).reshape(1, -1)
        if np.any(np.isnan(v)):
            return -1
        z = self.scaler.transform(v)
        return int(self.kmeans.predict(z)[0])

    def assign_many(self, regimes: List[Dict]) -> List[int]:
        if self.kmeans is None or self.scaler is None:
            return [-1] * len(regimes)
        vs = np.array([self.to_vector(r) for r in regimes], dtype=float)
        out = np.full(len(regimes), -1, dtype=int)
        valid = ~np.any(np.isnan(vs), axis=1)
        if valid.any():
            z = self.scaler.transform(vs[valid])
            out[valid] = self.kmeans.predict(z)
        return out.tolist()

    def describe(self, cluster_id: int) -> str:
        if self.meta is None or cluster_id < 0 or cluster_id >= len(self.meta.descriptions):
            return f"cluster_{cluster_id}"
        return self.meta.descriptions[cluster_id]

    # ── Persistence ─────────────────────────────────────────
    @staticmethod
    def load() -> "RegimeClusterer":
        if not (os.path.exists(KMEANS_PATH)
                and os.path.exists(SCALER_PATH)
                and os.path.exists(META_PATH)):
            return RegimeClusterer()
        with open(KMEANS_PATH, "rb") as f:
            kmeans = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        with open(META_PATH) as f:
            d = json.load(f)
        meta = ClusterMeta(**d)
        return RegimeClusterer(kmeans, scaler, meta)

    def save(self):
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(KMEANS_PATH, "wb") as f:
            pickle.dump(self.kmeans, f)
        with open(SCALER_PATH, "wb") as f:
            pickle.dump(self.scaler, f)
        with open(META_PATH, "w") as f:
            json.dump(asdict(self.meta), f, indent=2)


# ── Fitting ────────────────────────────────────────────────
def _describe_centroid(c: np.ndarray) -> str:
    """Generate a short human-readable label from a centroid in feature units.
    No PnL involved — this is purely structural."""
    adx, rsi, bb_pos, bbw, atr_pct, chop, tc = c[:7]
    parts = []
    if adx >= 28:
        parts.append("Strong-Trend")
    elif adx <= 18:
        parts.append("Chop")
    else:
        parts.append("Mid-Trend")
    if rsi >= 60:
        parts.append("Overbought")
    elif rsi <= 40:
        parts.append("Oversold")
    if bb_pos >= 0.75:
        parts.append("Upper-Band")
    elif bb_pos <= 0.25:
        parts.append("Lower-Band")
    if atr_pct >= 2.0:
        parts.append("HighVol")
    elif atr_pct <= 0.8:
        parts.append("LowVol")
    return " | ".join(parts) if parts else "Generic"


def fit(features_array: np.ndarray, k: int = 6, fit_range=None,
        random_state: int = 42) -> RegimeClusterer:
    """Fit k-means on a (n, len(REGIME_FEATURES)) array of regime features."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    n = len(features_array)
    scaler = StandardScaler().fit(features_array)
    z = scaler.transform(features_array)

    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=20).fit(z)
    labels = kmeans.labels_

    centroids_z = kmeans.cluster_centers_
    centroids = scaler.inverse_transform(centroids_z)
    descriptions = [_describe_centroid(c) for c in centroids]

    fractions = []
    for cid in range(k):
        fractions.append(float((labels == cid).sum() / n))

    meta = ClusterMeta(
        k=k,
        features=list(REGIME_FEATURES),
        centroids=centroids.tolist(),
        centroids_z=centroids_z.tolist(),
        descriptions=descriptions,
        fractions=fractions,
        fit_n=int(n),
        fit_range=list(fit_range) if fit_range else ["", ""],
        inertia=float(kmeans.inertia_),
    )
    return RegimeClusterer(kmeans, scaler, meta)
