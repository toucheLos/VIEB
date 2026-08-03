"""HDBSCAN on the delay-embedded coordinates.

Run directly on the embedding -- no UMAP. That is a deliberate asymmetry with
v1, which reduces with UMAP first; it is recorded in the benchmark output rather
than quietly equalised.

The -1 noise label is preserved and never force-assigned. Filling it in (by
nearest cluster, or by HMM smoothing) would destroy the one measurement that
tests the density-duration confound: whether unclustered frames are
systematically faster than clustered ones. Force-assignment would hide exactly
the detection bias the representation is meant to be evaluated against.
"""

from __future__ import annotations

import numpy as np

from .metrics import NOISE_LABEL, cluster_metrics, speed_diagnostics


def cluster(embedded, min_cluster_size=50, min_samples=None, metric="euclidean"):
    """Cluster delay-embedded coordinates. Returns (labels, probabilities).

    Labels are -1 for noise, preserved as-is.
    """
    import hdbscan

    embedded = np.asarray(embedded, dtype=np.float64)
    if embedded.ndim != 2:
        raise ValueError(f"embedded must be (N, D), got {embedded.shape}")
    if embedded.shape[0] == 0:
        return np.empty(0, dtype=int), np.empty(0)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
        metric=metric,
    )
    labels = clusterer.fit_predict(embedded)
    probs = getattr(clusterer, "probabilities_", np.ones(labels.shape))
    return labels, probs


def cluster_with_diagnostics(embedded, min_cluster_size=50, min_samples=None):
    """Cluster and report the metrics plus the confound diagnostics."""
    labels, probs = cluster(embedded, min_cluster_size, min_samples)
    if labels.size == 0:
        return {"labels": labels, "probabilities": probs, "metrics": None,
                "speed": None}
    return {
        "labels": labels,
        "probabilities": probs,
        "metrics": cluster_metrics(labels),
        "speed": speed_diagnostics(labels, embedded),
    }


def seed_stability(embedded, min_cluster_size=50, min_samples=None,
                   n_repeats=5, subsample_frac=0.8, seed=0):
    """Re-cluster on repeated random subsamples and report the spread.

    Instability is itself the signature of the confound. In simulation a real
    density peak was recovered on every seed (100% x6), while a marginal fast
    behavior swung between 15% and 100% recovery. So a large variance in
    `n_states` across subsamples is evidence that clusters are being found or
    missed by luck rather than by structure.
    """
    embedded = np.asarray(embedded, dtype=np.float64)
    n = embedded.shape[0]
    if n < 2:
        return {"n_states": [], "mean": None, "std": None}

    rng = np.random.default_rng(seed)
    counts = []
    for _ in range(n_repeats):
        idx = rng.choice(n, size=max(2, int(n * subsample_frac)), replace=False)
        labels, _ = cluster(embedded[idx], min_cluster_size, min_samples)
        counts.append(int(np.unique(labels[labels >= 0]).size))

    return {
        "n_states": counts,
        "mean": float(np.mean(counts)),
        "std": float(np.std(counts)),
        "n_repeats": n_repeats,
        "subsample_frac": subsample_frac,
    }


__all__ = ["cluster", "cluster_with_diagnostics", "seed_stability", "NOISE_LABEL"]
