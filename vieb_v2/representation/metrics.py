"""Cluster metrics, computed identically for both arms of the benchmark.

v1's stored diagnostics are not textbook definitions (compare.py:2110-2170):
`state_fracs` divides by `total_frames` *including* noise frames, so the
fractions sum to 1 - noise_frac rather than 1; and `state_entropy` is that
sub-normalised distribution's entropy divided by log(C), i.e. a normalised
entropy in nats.

That matters for the comparison. If v2 were scored with clean definitions and
v1's numbers read from disk, a definitional difference could masquerade as a v2
win. So both conventions are computed here, both arms are scored by this one
function, and v1's stored values are never read directly.
"""

from __future__ import annotations

import math

import numpy as np

NOISE_LABEL = -1


def cluster_metrics(labels):
    """Metrics in both conventions.

    labels : (N,) integer cluster labels, -1 for noise

    Returns a dict with `v1_convention` (fractions over all frames including
    noise, entropy normalised by log C) and `clustered_only` (fractions over
    clustered frames, so they sum to 1), plus shared counts.
    """
    labels = np.asarray(labels)
    total = int(labels.size)
    if total == 0:
        raise ValueError("no labels given")

    noise_count = int((labels == NOISE_LABEL).sum())
    clustered = labels[labels >= 0]
    ids = np.unique(clustered)
    counts = np.array([int((clustered == k).sum()) for k in ids], dtype=float)

    return {
        "n_states": int(ids.size),
        "n_frames": total,
        "n_noise": noise_count,
        "n_clustered": int(clustered.size),
        "noise_frac": noise_count / total,
        "v1_convention": _summarise(counts, total),
        "clustered_only": _summarise(counts, max(1, int(clustered.size))),
    }


def _summarise(counts, denominator):
    """Occupancy summary for a given denominator.

    Passing `total_frames` reproduces v1; passing the clustered count gives a
    proper probability distribution.
    """
    if counts.size == 0:
        return {"largest_state_frac": 0.0, "smallest_state_frac": 0.0,
                "state_entropy": 0.0, "state_fracs": []}

    fracs = counts / denominator
    positive = [f for f in fracs if f > 0]

    if len(positive) > 1:
        ent = -sum(f * math.log(f) for f in positive)
        max_ent = math.log(len(positive))
        entropy = ent / max_ent if max_ent > 0 else 1.0
    else:
        entropy = 0.0

    return {
        "largest_state_frac": float(max(fracs)),
        "smallest_state_frac": float(min(positive)) if positive else 0.0,
        "state_entropy": float(entropy),
        "state_fracs": [float(f) for f in fracs],
    }


def speed_diagnostics(labels, scores):
    """Test the density-duration confound rather than assuming it.

    Simulation showed the confound is a *detection* bias, not an inflation one:
    a large dominant state may simply be true occupancy, so
    `largest_state_frac` cannot diagnose it. What is predicted is that the
    *noise* label is speed-biased -- frames HDBSCAN failed to cluster should be
    systematically faster than clustered ones.

    If `noise_speed_ratio` is not meaningfully above 1, the density-duration
    story does not explain the dominant state and should stop being cited.

    labels : (N,) labels aligned with `scores`
    scores : (N, q) coordinates the labels were computed on
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.size != scores.shape[0]:
        raise ValueError(
            f"labels ({labels.size}) and scores ({scores.shape[0]}) disagree"
        )
    if labels.size < 2:
        return {"noise_speed_ratio": None, "per_cluster": {}}

    step = np.zeros(labels.size)
    step[1:] = np.linalg.norm(np.diff(scores, axis=0), axis=1)

    noise = labels == NOISE_LABEL
    noise_speed = float(step[noise].mean()) if noise.any() else None
    clustered_speed = float(step[~noise].mean()) if (~noise).any() else None
    ratio = (
        noise_speed / clustered_speed
        if noise_speed is not None and clustered_speed not in (None, 0.0)
        else None
    )

    per_cluster = {}
    for k in np.unique(labels[labels >= 0]):
        m = labels == k
        per_cluster[int(k)] = {
            "size": int(m.sum()),
            "mean_step": float(step[m].mean()),
            "mean_bout_frames": _mean_bout_length(m),
        }

    return {
        "noise_speed": noise_speed,
        "clustered_speed": clustered_speed,
        "noise_speed_ratio": ratio,
        "per_cluster": per_cluster,
        "size_speed_rank_corr": _rank_corr(
            [v["size"] for v in per_cluster.values()],
            [v["mean_step"] for v in per_cluster.values()],
        ),
    }


def _mean_bout_length(mask):
    """Mean run length of contiguous True stretches."""
    if not mask.any():
        return 0.0
    padded = np.concatenate([[False], mask, [False]])
    edges = np.diff(padded.astype(int))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return float((ends - starts).mean())


def _rank_corr(a, b):
    """Spearman correlation, without pulling in scipy."""
    if len(a) < 2:
        return None
    ra, rb = _ranks(a), _ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = math.sqrt(float((ra ** 2).sum()) * float((rb ** 2).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else None


def _ranks(values):
    arr = np.asarray(values, dtype=float)
    order = arr.argsort()
    ranks = np.empty(arr.size, dtype=float)
    ranks[order] = np.arange(arr.size, dtype=float)
    return ranks
