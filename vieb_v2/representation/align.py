"""Weighted egocentric alignment.

Removes the SE(2) nuisance -- arena position and heading -- so that the same
posture performed in a different corner facing a different way maps to the same
coordinates. Scale is deliberately NOT removed: for 2D top-down mice, rearing is
an out-of-plane motion whose only signature is apparent foreshortening, so
normalising scale would delete it (docs/REPRESENTATION_MATH.md section 1.4c).

Keypoints are weighted by their DLC likelihoods, so a badly-tracked frame cannot
mis-rotate the whole body. That has a consequence worth knowing about: per-frame
weights break the exact rank-(2K-3) structure of the aligned covariance, because
the alignment's stationarity condition stops being a *fixed* linear functional.
`align_session` therefore re-centers its output on the unweighted centroid,
which restores the two translation nulls exactly. The rotational direction
retains a small variance that scales with how much the likelihoods fluctuate --
measured at ~0.4% of total variance for likelihoods in [0.9, 1]. `null_leakage`
reports it rather than letting it hide in the PCs.
"""

from __future__ import annotations

import numpy as np


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _weighted_centroid(pose, w):
    """Weighted centroid per frame. pose (T,K,2), w (T,K) -> (T,2)."""
    wsum = w.sum(axis=1, keepdims=True)
    return (w[:, :, None] * pose).sum(axis=1) / np.maximum(wsum, 1e-12)


def solve_rotation(centered, reference, w):
    """Optimal rotation angle per frame, aligning `centered` onto `reference`.

    Maximises tr(R M) with M = X^T diag(w) X_ref. The weights enter only through
    M, so the unweighted closed form carries over unchanged:

        theta* = atan2(m12 - m21, m11 + m22)

    centered  : (T, K, 2) translation-removed pose
    reference : (K, 2) reference shape
    w         : (T, K) per-frame keypoint weights
    Returns (T,) angles in radians.
    """
    # M[t] = centered[t].T @ diag(w[t]) @ reference   -> (T, 2, 2)
    m = np.einsum("tki,tk,kj->tij", centered, w, reference)
    return np.arctan2(m[:, 0, 1] - m[:, 1, 0], m[:, 0, 0] + m[:, 1, 1])


def _apply_rotation(centered, theta):
    """Rotate each frame by theta[t]: X @ R(theta)^T."""
    c, s = np.cos(theta), np.sin(theta)
    x, y = centered[..., 0], centered[..., 1]
    # Row-vector convention: [x y] @ R^T  =>  x' = x c - y s, y' = x s + y c
    return np.stack([x * c[:, None] - y * s[:, None],
                     x * s[:, None] + y * c[:, None]], axis=-1)


def align_session(pose, conf=None, reference=None, n_iter=5, tol=1e-9):
    """Egocentrically align one recording.

    pose      : (T, K, 2)
    conf      : (T, K) DLC likelihoods, or None for uniform weights
    reference : (K, 2) reference shape, or None to fit one by iterated
                generalized Procrustes (recommended -- it puts the reference at
                the data centroid, where the tangent-space linearisation that
                the downstream PCA relies on is most accurate)

    Returns (aligned (T,K,2), theta (T,), reference (K,2)).
    """
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim != 3 or pose.shape[2] != 2:
        raise ValueError(f"pose must be (T, K, 2), got {pose.shape}")
    T, K, _ = pose.shape

    if conf is None:
        w = np.ones((T, K), dtype=np.float64)
    else:
        w = np.asarray(conf, dtype=np.float64)
        if w.shape != (T, K):
            raise ValueError(f"conf must be (T, K)={T, K}, got {w.shape}")
        # Non-positive likelihoods would make the weighted centroid undefined.
        w = np.clip(w, 1e-6, None)

    centered = pose - _weighted_centroid(pose, w)[:, None, :]

    if reference is None:
        # Generalized Procrustes: align to the current mean, recompute, repeat.
        reference = centered[0].copy()
        for _ in range(n_iter):
            theta = solve_rotation(centered, reference, w)
            aligned = _apply_rotation(centered, theta)
            new_ref = aligned.mean(axis=0)
            new_ref = new_ref - new_ref.mean(axis=0)
            shift = float(np.abs(new_ref - reference).max())
            reference = new_ref
            if shift < tol:
                break
    else:
        reference = np.asarray(reference, dtype=np.float64)

    theta = solve_rotation(centered, reference, w)
    aligned = _apply_rotation(centered, theta)

    # Restore the two exact translation nulls that per-frame weighting broke.
    aligned = aligned - aligned.mean(axis=1, keepdims=True)
    return aligned, theta, reference


def align_all(sessions, n_iter=5):
    """Align a list of recordings against one shared reference.

    A shared reference matters for the same reason pooled PCA does: aligning
    each recording to its own reference would put different animals in
    differently-rotated frames, and no downstream coordinate would be
    comparable across them.

    sessions : list of (pose, conf) tuples, conf may be None
    Returns (list of aligned (T,K,2), reference (K,2)).
    """
    if not sessions:
        raise ValueError("no sessions given")

    # Fit the shared reference on the pooled frames.
    poses = [np.asarray(p, dtype=np.float64) for p, _ in sessions]
    confs = [c for _, c in sessions]
    pooled_pose = np.concatenate(poses, axis=0)
    pooled_conf = (
        None if any(c is None for c in confs)
        else np.concatenate([np.asarray(c, dtype=np.float64) for c in confs], axis=0)
    )
    _, _, reference = align_session(pooled_pose, pooled_conf, n_iter=n_iter)

    aligned = [align_session(p, c, reference=reference)[0]
               for p, c in zip(poses, confs)]
    return aligned, reference


def null_leakage(aligned_flat):
    """Variance that leaked into the three nominally-null directions.

    Under uniform weights the aligned covariance has exactly three zero
    eigenvalues (2 translation + 1 rotation). Per-frame confidence weighting
    partially refills them with likelihood fluctuation rather than pose. This
    returns that fraction of total variance so the pipeline can log it -- a
    large value means the PCs are partly tracking confidence noise.

    aligned_flat : (N, D) flattened aligned pose
    """
    x = np.asarray(aligned_flat, dtype=np.float64)
    if x.shape[0] < 2:
        return 0.0
    ev = np.linalg.eigvalsh(np.cov(x.T))
    total = float(ev.sum())
    if total <= 0:
        return 0.0
    return float(np.sort(ev)[:3].sum() / total)
