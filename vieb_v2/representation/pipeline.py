"""Composition of the v2 representation pipeline.

    pose -> drop noisy keypoints -> weighted egocentric alignment
         -> pooled PCA -> delay embedding -> HDBSCAN

Every parameter that changes the result is returned in the report rather than
left implicit, including the component count (required by the brief), the full
eigenvalue spectrum, and the variance that leaked into the nominally-null
directions because of confidence weighting.
"""

from __future__ import annotations

import numpy as np

from . import keypoints
from .align import align_all, null_leakage
from .cluster import cluster_with_diagnostics, seed_stability
from .embed import embed_all
from .pooled_pca import PooledPCA


def run(sessions, bodyparts=None, n_lags=4, lag_stride=2, var_threshold=0.95,
        min_cluster_size=50, min_samples=None, lag_weights=None,
        stability_repeats=0, use_gpu=False):
    """Run the full pipeline over a project's recordings.

    sessions : list of (pose (T,K,2), conf (T,K) or None) per recording --
               a list, never concatenated, so lags cannot cross a boundary
    bodyparts: keypoint names; defaults to the 8-point Luna model ordering

    Returns a dict with labels, the embedding index mapping rows back to
    (recording, frame), and a report of everything that shaped the result.
    """
    if not sessions:
        raise ValueError("no sessions given")

    if bodyparts is None:
        bodyparts = keypoints.DEFAULT_BODYPARTS

    # 1. Drop noisy keypoints (tail_tip) before anything else sees them.
    selected, dropped = [], None
    for pose, conf in sessions:
        pose, conf, kept = keypoints.select(
            np.asarray(pose, dtype=np.float64),
            None if conf is None else np.asarray(conf, dtype=np.float64),
            bodyparts,
        )
        dropped = [b for b in bodyparts if b not in kept]
        selected.append((pose, conf))

    # 2. Egocentric alignment against one shared reference.
    aligned, reference = align_all(selected)
    leakage = null_leakage(np.concatenate([a.reshape(a.shape[0], -1)
                                           for a in aligned], axis=0))

    # 3. Pooled PCA -- one basis for the whole project, never per recording.
    pca = PooledPCA(var_threshold=var_threshold).fit(aligned)
    scores = pca.transform_all(aligned)

    # 4. Delay embedding, boundary-safe by construction.
    embedded, index = embed_all(scores, n_lags, lag_stride, lag_weights)
    if embedded.shape[0] == 0:
        raise ValueError(
            f"no frames survived delay embedding: every recording is shorter "
            f"than the {n_lags * lag_stride + 1}-frame window"
        )

    # 5. HDBSCAN, keeping the -1 noise label.
    result = cluster_with_diagnostics(embedded, min_cluster_size, min_samples,
                                      use_gpu=use_gpu)

    stability = (
        seed_stability(embedded, min_cluster_size, min_samples,
                       n_repeats=stability_repeats)
        if stability_repeats else None
    )

    return {
        "labels": result["labels"],
        "probabilities": result["probabilities"],
        "index": index,
        "embedded": embedded,
        "scores": scores,
        "reference": reference,
        "metrics": result["metrics"],
        "speed_diagnostics": result["speed"],
        "seed_stability": stability,
        "report": {
            "n_recordings": len(sessions),
            "n_keypoints": aligned[0].shape[1],
            "dropped_keypoints": dropped,
            "pose_dim": aligned[0].shape[1] * 2,
            "expected_rank": aligned[0].shape[1] * 2 - 3,
            "null_direction_leakage": leakage,
            "pca": pca.spectrum_report(),
            "n_lags": n_lags,
            "lag_stride": lag_stride,
            "embedding_dim": int(embedded.shape[1]),
            "window_frames": n_lags * lag_stride + 1,
            "n_embedded_frames": int(embedded.shape[0]),
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "umap": False,
            "pca_backend": pca.backend_,
            "hdbscan_backend": result.get("backend", "cpu"),
        },
    }
