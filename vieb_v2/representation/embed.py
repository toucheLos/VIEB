"""Delay embedding -- the temporal element.

    y_t = [p_t, p_{t-tau}, ..., p_{t-L*tau}]

HDBSCAN has no model of time. Delay embedding smuggles a fixed window of
temporal context into the coordinates themselves, so a static algorithm sees
short trajectories rather than instants. That is what lets it separate two
frames sharing a posture but differing in what is happening -- rear-up from
rear-down, mid-stance entering from mid-stance leaving. It is a partial fix
only: it still models neither transition structure nor duration.

Lags must never cross a recording boundary. Embedding a pre-concatenated array
would manufacture R*L*tau chimeric vectors splicing the end of one animal's
session onto the start of another's -- pure artifact. This module therefore only
accepts a list of per-recording arrays; there is no entry point that could
receive an already-concatenated array and silently do the wrong thing.
"""

from __future__ import annotations

import numpy as np


def embed_session(scores, n_lags, lag_stride=1, weights=None):
    """Delay-embed one recording.

    scores : (T, q) PC scores for a single recording
    Returns (T - n_lags*lag_stride, q*(n_lags+1)). Empty if the recording is
    shorter than the window.
    """
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError(f"scores must be (T, q), got {scores.shape}")
    if n_lags < 0:
        raise ValueError("n_lags must be >= 0")
    if lag_stride < 1:
        raise ValueError("lag_stride must be >= 1")

    T, q = scores.shape
    span = n_lags * lag_stride
    n_valid = T - span
    if n_valid <= 0:
        return np.empty((0, q * (n_lags + 1)), dtype=np.float64)

    w = _resolve_weights(weights, n_lags)
    blocks = [
        w[ell] * scores[span - ell * lag_stride: T - ell * lag_stride]
        for ell in range(n_lags + 1)
    ]
    return np.concatenate(blocks, axis=1)


def embed_all(sessions, n_lags, lag_stride=1, weights=None):
    """Delay-embed several recordings and stack them.

    sessions : list of (T_r, q) arrays -- one per recording, never concatenated

    Returns (embedded (N, q*(n_lags+1)), index (N, 2)) where index[i] is the
    (recording, frame) the row came from. The index is what makes the boundary
    guarantee checkable after the fact, and is needed to map labels back onto
    frames.
    """
    if not isinstance(sessions, (list, tuple)):
        raise TypeError(
            "sessions must be a list of per-recording arrays; passing one "
            "concatenated array would embed lags across recording boundaries"
        )

    span = n_lags * lag_stride
    rows, index = [], []
    for r, scores in enumerate(sessions):
        emb = embed_session(scores, n_lags, lag_stride, weights)
        if emb.shape[0] == 0:
            continue
        rows.append(emb)
        frames = np.arange(span, span + emb.shape[0])
        index.append(np.stack([np.full(emb.shape[0], r), frames], axis=1))

    if not rows:
        q = int(np.asarray(sessions[0]).shape[1]) if len(sessions) else 0
        return (np.empty((0, q * (n_lags + 1))), np.empty((0, 2), dtype=int))

    return np.concatenate(rows, axis=0), np.concatenate(index, axis=0).astype(int)


def valid_frames(n_frames, n_lags, lag_stride=1):
    """Frame indices that can be embedded in a recording of `n_frames`."""
    span = n_lags * lag_stride
    return np.arange(span, n_frames) if n_frames > span else np.empty(0, dtype=int)


def difference_matrix(n_lags):
    """The map from a delay stack to a finite-difference (derivative) stack.

    B[j, i] = (-1)^i * C(j, i) for i <= j. B is lower-triangular with diagonal
    (-1)^j, hence invertible: the delay vector and the
    (position, velocity, acceleration, ...) stack span the same space, so delay
    embedding is a superset of hand-built kinematic features.

    B is not orthogonal, so it does not preserve distances -- the two
    parametrisations give different clusterings. Conditioning also degrades fast
    (cond ~1.7e4 by n_lags=8), which is why few lags at wider stride beat many
    adjacent ones. Provided for testing that equivalence, not used in the
    pipeline.
    """
    from math import comb

    n = n_lags + 1
    b = np.zeros((n, n))
    for j in range(n):
        for i in range(j + 1):
            b[j, i] = ((-1) ** i) * comb(j, i)
    return b


def _resolve_weights(weights, n_lags):
    if weights is None:
        return np.ones(n_lags + 1)
    if np.isscalar(weights):
        # Geometric decay: recent frames dominate older ones.
        return np.asarray([float(weights) ** ell for ell in range(n_lags + 1)])
    w = np.asarray(weights, dtype=np.float64)
    if w.shape != (n_lags + 1,):
        raise ValueError(f"weights must have length {n_lags + 1}, got {w.shape}")
    return w
