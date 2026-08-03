"""Subsampling arms for the density-duration confound.

Sampling density along a trajectory is 1/speed: a bout of n frames traces an
arclength of about n*v, so n/(n*v) = 1/v, independent of how long the bout
lasts. Slow behaviors pile samples into a tiny region; fast ones spread them
thinly. Duration then enters separately through cluster mass, and HDBSCAN's
stability is a sum over points, so long slow behaviors are rewarded twice.

Both functions here are arms to *evaluate*, not defaults to apply. Simulation
showed the confound is narrower than it first appears -- well-separated
behaviors are recovered regardless of speed -- so whether either mitigation
helps on real data is an empirical question, not a foregone conclusion.
"""

from __future__ import annotations

import numpy as np


def per_bout_indices(bout_labels, n_per_bout, rng=None):
    """Uniformly subsample a fixed number of frames from each bout.

    Each bout contributes `n_per_bout` frames regardless of duration, which
    removes the duration-to-mass advantage entirely.

    The circularity is real and worth stating: bout boundaries come from a
    segmentation, but segmentation is what clustering is for. This therefore
    needs a provisional labelling (a first-pass clustering, or a change-point
    detector) and inherits whatever biases that had.

    bout_labels : (N,) provisional per-frame labels; contiguous runs are bouts
    Returns sorted indices into the original frames.
    """
    labels = np.asarray(bout_labels)
    if labels.size == 0:
        return np.empty(0, dtype=int)
    rng = np.random.default_rng() if rng is None else rng

    keep = []
    for start, stop in _runs(labels):
        n = stop - start
        take = min(int(n_per_bout), n)
        if take >= n:
            keep.append(np.arange(start, stop))
        else:
            # Evenly spaced rather than random: preserves the bout's shape
            # instead of clumping, which matters when the point is to sample
            # a trajectory rather than a blob.
            offsets = np.linspace(0, n - 1, take).round().astype(int)
            keep.append(start + np.unique(offsets))
    return np.concatenate(keep) if keep else np.empty(0, dtype=int)


def arclength_indices(scores, delta):
    """Resample uniformly in arclength rather than in time.

    This attacks the confound at its root and needs no bout labels: sampling
    every `delta` of path length makes the linear density 1/delta for every
    behavior, exactly cancelling the 1/speed term.

    Its own failure mode is tracking jitter. A frozen animal still accumulates
    arclength from keypoint noise, so this would faithfully sample the noise --
    smooth the scores first, and treat the smoothing bandwidth as a parameter
    that interacts with the delay-embedding window.

    scores : (T, q) coordinates for one recording
    delta  : arclength between retained samples
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"scores must be (T, q), got {scores.shape}")
    if delta <= 0:
        raise ValueError("delta must be > 0")
    if scores.shape[0] == 0:
        return np.empty(0, dtype=int)

    step = np.zeros(scores.shape[0])
    step[1:] = np.linalg.norm(np.diff(scores, axis=0), axis=1)
    cumulative = np.cumsum(step)

    targets = np.arange(0.0, cumulative[-1] + delta, delta)
    idx = np.unique(np.searchsorted(cumulative, targets, side="left"))
    return idx[idx < scores.shape[0]]


def subsample_sessions(sessions, mode, **kwargs):
    """Apply a subsampling arm per recording.

    mode : "none" | "per_bout" | "arclength"

    Returns (subsampled sessions, list of index arrays). Cluster sizes stop
    encoding occupancy once anything but "none" is used, so `largest_state_frac`
    changes meaning -- map labels back onto all frames before reading durations.
    """
    if mode == "none":
        return list(sessions), [np.arange(len(s)) for s in sessions]

    out, indices = [], []
    for i, scores in enumerate(sessions):
        if mode == "arclength":
            idx = arclength_indices(scores, kwargs["delta"])
        elif mode == "per_bout":
            idx = per_bout_indices(
                kwargs["bout_labels"][i], kwargs["n_per_bout"],
                kwargs.get("rng"),
            )
        else:
            raise ValueError(f"unknown mode {mode!r}")
        out.append(np.asarray(scores)[idx])
        indices.append(idx)
    return out, indices


def _runs(labels):
    """Yield (start, stop) of contiguous equal-label runs."""
    if labels.size == 0:
        return
    change = np.flatnonzero(np.diff(labels)) + 1
    bounds = np.concatenate([[0], change, [labels.size]])
    for a, b in zip(bounds[:-1], bounds[1:]):
        yield int(a), int(b)
