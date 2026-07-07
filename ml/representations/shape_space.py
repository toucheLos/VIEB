"""Kendall shape-space representation (Procrustes superimposition).

Part B of the statistics-methods feature-representation work. Removes
translation, scale, and rotation from each frame's keypoint configuration,
leaving only "shape" — directly addressing the confound in the current
default representation (``ml/feature_extraction.py``), where raw pairwise
distances mix body size/position/camera angle with actual posture.

Math (see MATH.md for the full writeup):
    1. Translation: subtract the per-frame keypoint centroid.
    2. Scale: divide by centroid size (RMS distance of keypoints to centroid).
       Steps 1-2 produce the "pre-shape".
    3. Rotation: align every frame's pre-shape to a reference pose via the
       orthogonal Procrustes / Kabsch solution (SVD of the cross-covariance
       matrix), then iterate a small number of Generalized Procrustes
       Analysis (GPA) passes, recomputing the reference as the mean aligned
       shape each time, for convergence.

The output feature vector per frame is the flattened shape coordinates
(K*2, scale/rotation/translation-invariant) plus a small set of derived
dynamics (shape speed per keypoint, overall shape velocity/acceleration
norm) so downstream clustering still has access to motion, not just static
posture.
"""
from __future__ import annotations

import numpy as np

from ..pose_utils import prepare_pose


def _center_and_scale(pose: np.ndarray) -> np.ndarray:
    """Remove translation and scale per frame. pose: (T, K, D) -> pre-shape (T, K, D)."""
    centroid = pose.mean(axis=1, keepdims=True)  # (T, 1, D)
    centered = pose - centroid
    # Centroid size = RMS distance of keypoints to centroid (Kendall's convention)
    size = np.sqrt(np.mean(np.sum(centered ** 2, axis=-1), axis=-1))  # (T,)
    size = np.where(size < 1e-12, 1.0, size)
    return centered / size[:, None, None]


def _procrustes_rotate(pre_shape: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Rotate each frame's pre-shape onto ``reference`` via the 2D Kabsch/SVD solution.

    pre_shape: (T, K, D), reference: (K, D). Returns rotated (T, K, D).
    """
    T = pre_shape.shape[0]
    rotated = np.empty_like(pre_shape)
    for t in range(T):
        cross_cov = pre_shape[t].T @ reference  # (D, D)
        U, _, Vt = np.linalg.svd(cross_cov)
        # Ensure a proper rotation (no reflection): det(R) > 0
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        correction = np.eye(U.shape[0])
        correction[-1, -1] = d
        R = Vt.T @ correction @ U.T
        rotated[t] = pre_shape[t] @ R.T
    return rotated


def _generalized_procrustes(pre_shape: np.ndarray, n_iter: int = 3) -> np.ndarray:
    """Full GPA: iteratively align all frames to their mean shape."""
    reference = pre_shape[0].copy()
    aligned = pre_shape
    for _ in range(n_iter):
        aligned = _procrustes_rotate(pre_shape, reference)
        new_reference = aligned.mean(axis=0)
        norm = np.sqrt(np.sum(new_reference ** 2))
        reference = new_reference / norm if norm > 1e-12 else new_reference
    return aligned


class ShapeSpaceExtractor:
    """Procrustes shape-space feature representation. See module docstring."""

    def __init__(self, fps: float = 30.0, smooth_window: int = 5, gpa_iterations: int = 3):
        self.fps = fps
        self.smooth_window = smooth_window
        self.gpa_iterations = gpa_iterations
        self._n_keypoints: int | None = None

    def fit(self, sample_poses: list[np.ndarray]) -> "ShapeSpaceExtractor":
        """No calibration needed — shape space is defined per-frame/per-video."""
        return self

    def transform(self, pose: np.ndarray, confidence=None) -> tuple[np.ndarray, list[str]]:
        pose_clean = prepare_pose(pose, self.smooth_window)
        T, K, D = pose_clean.shape
        self._n_keypoints = K

        pre_shape = _center_and_scale(pose_clean)
        shape_coords = _generalized_procrustes(pre_shape, self.gpa_iterations)  # (T, K, D)

        flat_shape = shape_coords.reshape(T, K * D)

        # Derived dynamics: per-keypoint shape speed + overall velocity/accel norm
        delta = np.zeros_like(shape_coords)
        delta[1:] = shape_coords[1:] - shape_coords[:-1]
        shape_speed = np.linalg.norm(delta, axis=-1) * self.fps  # (T, K)

        overall_delta = shape_speed.mean(axis=1)  # (T,) velocity norm proxy
        shape_velocity_norm = overall_delta
        shape_accel_norm = np.zeros(T)
        shape_accel_norm[1:] = np.abs(np.diff(shape_velocity_norm)) * self.fps

        features = np.concatenate(
            [flat_shape, shape_speed, shape_velocity_norm[:, None], shape_accel_norm[:, None]],
            axis=1,
        ).astype(np.float32)

        names = self.get_feature_names(K, D)
        return features, names

    def get_feature_names(self, n_keypoints: int, n_dims: int = 2) -> list[str]:
        coord_names = ["x", "y", "z"][:n_dims]
        names = [f"shape_{c}_{k}" for k in range(n_keypoints) for c in coord_names]
        names += [f"shape_speed_{k}" for k in range(n_keypoints)]
        names += ["shape_velocity_norm", "shape_accel_norm"]
        return names

    def get_meta(self) -> dict:
        return {
            "mode": "shape_space",
            "fps": self.fps,
            "smooth_window": self.smooth_window,
            "gpa_iterations": self.gpa_iterations,
            "n_keypoints": self._n_keypoints,
            "n_features": (self._n_keypoints * 2 + self._n_keypoints + 2) if self._n_keypoints else None,
        }

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "ShapeSpaceExtractor":
        import joblib
        return joblib.load(path)
