"""Takens delay-embedding representation.

Part C of the statistics-methods feature-representation work. Per Takens'
theorem, a scalar time series sampled from a deterministic dynamical system
can be embedded into a higher-dimensional space that reconstructs the
system's attractor topology, using delay vectors:

    y(t) = [x(t), x(t - tau), x(t - 2*tau), ..., x(t - (d-1)*tau)]

Applied here to two keypoint-layout-agnostic 1D signals that exist
regardless of Luna's 8-keypoint mouse layout vs Spence's 5-keypoint rat
layout: centroid speed, and PCA elongation (both computed fresh via
``ml/pose_utils.py``, mirroring the "universal, Layer-1-style" features in
the default extractor without touching it).

(tau, d) are dataset-dependent (Takens' theorem gives no closed-form
choice), so they are selected automatically from a calibration sample via:
    - tau: first local minimum of the average mutual information (AMI)
      between x(t) and x(t+tau), falling back to the first autocorrelation
      zero-crossing if no clear minimum exists.
    - d: false nearest neighbors (Kennel, Brown & Abarbanel 1992) — the
      smallest d at which the fraction of "false" neighbors (points that
      are only close because the embedding dimension is too low) drops
      below a threshold.

This calibration is a genuine fit step (unlike shape_space/topological) —
``fit()`` selects and caches (tau, d) per signal from a pooled sample so
every subsequent ``transform()`` call produces a consistent feature
dimensionality, required for pooled HDBSCAN clustering.
"""
from __future__ import annotations

import numpy as np

from ..pose_utils import compute_centroid, compute_pca_orientation, compute_speed, prepare_pose


def _average_mutual_information(x: np.ndarray, tau: int, n_bins: int = 16) -> float:
    """2D-histogram estimate of I(x(t); x(t+tau)) in nats."""
    if tau >= len(x):
        return 0.0
    a = x[:-tau]
    b = x[tau:]
    joint, _, _ = np.histogram2d(a, b, bins=n_bins)
    joint = joint / joint.sum()
    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = joint / (px * py)
        terms = joint * np.log(ratio)
    terms = np.nan_to_num(terms, nan=0.0, posinf=0.0, neginf=0.0)
    return float(terms.sum())


def select_tau_mutual_information(x: np.ndarray, max_tau: int = 50, n_bins: int = 16) -> int:
    """First local minimum of AMI(tau); falls back to first-zero-crossing of autocorrelation."""
    x = np.asarray(x, dtype=np.float64)
    max_tau = min(max_tau, len(x) // 4)
    if max_tau < 2:
        return 1

    ami = [_average_mutual_information(x, tau, n_bins) for tau in range(1, max_tau + 1)]
    for i in range(1, len(ami) - 1):
        if ami[i] < ami[i - 1] and ami[i] < ami[i + 1]:
            return i + 1  # tau values start at 1

    centered = x - x.mean()
    autocorr = np.correlate(centered, centered, mode="full")[len(centered) - 1:]
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]
        signs = np.sign(autocorr)
        crossings = np.where(np.diff(signs) < 0)[0]
        if len(crossings) > 0:
            return max(1, int(crossings[0]) + 1)
    return 1


def _false_nearest_neighbor_fraction(x: np.ndarray, tau: int, d: int, rtol: float = 15.0, atol: float = 2.0) -> float:
    """Fraction of false nearest neighbors when embedding at dimension d (Kennel et al. 1992)."""
    n = len(x) - d * tau
    if n < 10:
        return 0.0

    embed_d = np.array([x[i:i + d * tau:tau] for i in range(n)])
    embed_d1_extra = x[np.arange(n) + d * tau]  # the (d+1)-th coordinate, if it existed

    from scipy.spatial import cKDTree
    tree = cKDTree(embed_d)
    dists, idxs = tree.query(embed_d, k=2)  # nearest neighbor excluding self
    nn_dist = dists[:, 1]
    nn_idx = idxs[:, 1]

    nn_dist_safe = np.where(nn_dist < 1e-10, 1e-10, nn_dist)
    extra_diff = np.abs(embed_d1_extra - embed_d1_extra[nn_idx])
    ratio = extra_diff / nn_dist_safe

    sigma = np.std(x)
    is_false = (ratio > rtol) | (extra_diff / (sigma + 1e-12) > atol)
    return float(is_false.mean())


def select_embedding_dim_fnn(x: np.ndarray, tau: int, max_dim: int = 10, threshold: float = 0.01) -> int:
    """Smallest d at which the FNN fraction drops below ``threshold`` and stays low."""
    x = np.asarray(x, dtype=np.float64)
    for d in range(1, max_dim + 1):
        if len(x) - d * tau < 10:
            return max(1, d - 1)
        frac = _false_nearest_neighbor_fraction(x, tau, d)
        if frac < threshold:
            return d
    return max_dim


def _delay_embed(x: np.ndarray, tau: int, d: int) -> np.ndarray:
    """Delay-embed a 1D signal into (T, d), padding the first (d-1)*tau rows by edge-reflection."""
    T = len(x)
    pad = (d - 1) * tau
    x_padded = np.concatenate([np.full(pad, x[0]), x]) if pad > 0 else x
    embedded = np.empty((T, d))
    for i in range(d):
        offset = (d - 1 - i) * tau
        embedded[:, i] = x_padded[pad - offset: pad - offset + T]
    return embedded


class DelayEmbeddingExtractor:
    """Takens delay-embedding of centroid_speed + elongation. See module docstring."""

    SIGNALS = ("centroid_speed", "elongation")

    def __init__(self, fps: float = 30.0, smooth_window: int = 5, max_tau: int = 50, max_dim: int = 10):
        self.fps = fps
        self.smooth_window = smooth_window
        self.max_tau = max_tau
        self.max_dim = max_dim
        self.params: dict[str, dict[str, int]] = {}

    def _compute_signals(self, pose: np.ndarray) -> dict[str, np.ndarray]:
        pose_clean = prepare_pose(pose, self.smooth_window)
        centroid = compute_centroid(pose_clean)
        centroid_speed = compute_speed(centroid, self.fps)
        _, elongation = compute_pca_orientation(pose_clean)
        return {"centroid_speed": centroid_speed, "elongation": elongation}

    def fit(self, sample_poses: list[np.ndarray]) -> "DelayEmbeddingExtractor":
        pooled = {name: [] for name in self.SIGNALS}
        for pose in sample_poses:
            signals = self._compute_signals(pose)
            for name in self.SIGNALS:
                pooled[name].append(signals[name])

        for name in self.SIGNALS:
            x = np.concatenate(pooled[name]) if pooled[name] else np.zeros(0)
            if len(x) < 20:
                self.params[name] = {"tau": 1, "d": 2}
                continue
            tau = select_tau_mutual_information(x, self.max_tau)
            d = select_embedding_dim_fnn(x, tau, self.max_dim)
            self.params[name] = {"tau": tau, "d": d}
        return self

    def transform(self, pose: np.ndarray, confidence=None) -> tuple[np.ndarray, list[str]]:
        if not self.params:
            raise RuntimeError("DelayEmbeddingExtractor.fit() must be called before transform()")

        signals = self._compute_signals(pose)
        blocks = []
        names = []
        for name in self.SIGNALS:
            tau, d = self.params[name]["tau"], self.params[name]["d"]
            embedded = _delay_embed(signals[name], tau, d)
            blocks.append(embedded)
            names += [f"{name}_delay_{i}" for i in range(d)]

        features = np.concatenate(blocks, axis=1).astype(np.float32)
        return features, names

    def get_meta(self) -> dict:
        n_features = sum(p["d"] for p in self.params.values()) if self.params else None
        return {
            "mode": "delay_embedding",
            "fps": self.fps,
            "signals": list(self.SIGNALS),
            "params": self.params,
            "n_features": n_features,
        }

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "DelayEmbeddingExtractor":
        import joblib
        return joblib.load(path)
