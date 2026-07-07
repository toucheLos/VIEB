"""Persistent-homology summary features.

Part D of the statistics-methods feature-representation work. For a
sliding window of frames, pools the window's keypoints across time into one
2D point cloud and computes persistent homology (Vietoris-Rips filtration,
via the ``ripser`` library — an existing, well-maintained implementation;
this module does not reimplement the algorithm) to summarize the
*topology* of movement within the window: e.g. a repetitive, looping
movement produces a persistent H1 (loop) feature; erratic or translating
movement does not.

Deliberately **not** per-frame: TDA on a point cloud is expensive, so this
computes one persistence diagram per stride-sized window (default stride of
5 frames) and broadcasts that window's summary statistics to every frame it
covers — bounding the number of ripser calls to T/stride rather than T, per
the task's explicit "keep this computationally bounded" instruction. Actual
runtime is measured and reported by ``benchmark_feature_modes.py`` (Part E)
rather than assumed.

Requires the optional ``ripser`` dependency (``pip install -e ".[topology]"``).
"""
from __future__ import annotations

import numpy as np

from ..pose_utils import interpolate_nans

try:
    import ripser as _ripser_module
    _RIPSER_AVAILABLE = True
except ImportError:
    _RIPSER_AVAILABLE = False


_SUMMARY_NAMES = [
    "topo_total_persistence_h0",
    "topo_total_persistence_h1",
    "topo_n_significant_h0",
    "topo_n_significant_h1",
    "topo_max_persistence_h1",
]


def _window_summary(point_cloud: np.ndarray, significance_threshold: float = 0.05) -> np.ndarray:
    """Run ripser on one window's pooled point cloud, return the 5 summary stats."""
    result = _ripser_module.ripser(point_cloud, maxdim=1)
    dgms = result["dgms"]

    h0 = dgms[0]
    h0_finite = h0[np.isfinite(h0[:, 1])]
    h0_pers = h0_finite[:, 1] - h0_finite[:, 0]

    h1 = dgms[1] if len(dgms) > 1 else np.zeros((0, 2))
    h1_pers = h1[:, 1] - h1[:, 0] if len(h1) else np.zeros(0)

    total_h0 = float(h0_pers.sum())
    total_h1 = float(h1_pers.sum())
    n_sig_h0 = int((h0_pers > significance_threshold).sum())
    n_sig_h1 = int((h1_pers > significance_threshold).sum())
    max_h1 = float(h1_pers.max()) if len(h1_pers) else 0.0

    return np.array([total_h0, total_h1, n_sig_h0, n_sig_h1, max_h1], dtype=np.float32)


class TopologicalExtractor:
    """Persistent-homology summary features over sliding pose windows. See module docstring."""

    def __init__(self, fps: float = 30.0, window_sec: float = 0.75, stride_frames: int = 5):
        if not _RIPSER_AVAILABLE:
            raise ImportError(
                "feature_mode='topological' requires the 'ripser' package. "
                "Install it with: pip install -e \".[topology]\""
            )
        self.fps = fps
        self.window_sec = window_sec
        self.stride_frames = stride_frames

    def fit(self, sample_poses: list[np.ndarray]) -> "TopologicalExtractor":
        """No calibration needed — each window is summarized independently."""
        return self

    def transform(self, pose: np.ndarray, confidence=None) -> tuple[np.ndarray, list[str]]:
        pose_clean = interpolate_nans(pose)
        T, K, D = pose_clean.shape
        window_frames = max(2, int(round(self.window_sec * self.fps)))
        stride = max(1, self.stride_frames)

        features = np.zeros((T, len(_SUMMARY_NAMES)), dtype=np.float32)
        for start in range(0, T, stride):
            end = min(start + window_frames, T)
            if end - start < 2:
                if start > 0:
                    features[start:] = features[start - 1]
                continue
            cloud = pose_clean[start:end].reshape(-1, D)
            summary = _window_summary(cloud)
            cover_end = min(start + stride, T)
            features[start:cover_end] = summary

        return features, list(_SUMMARY_NAMES)

    def get_meta(self) -> dict:
        return {
            "mode": "topological",
            "fps": self.fps,
            "window_sec": self.window_sec,
            "stride_frames": self.stride_frames,
            "n_features": len(_SUMMARY_NAMES),
        }

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "TopologicalExtractor":
        import joblib
        return joblib.load(path)
