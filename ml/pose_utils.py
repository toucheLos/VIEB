"""Shared low-level pose math for alternative feature representations.

Deliberately independent of ``ml/feature_extraction.py`` (frozen — see
docs/DECISIONS.md #2/#6): ``PoseFeatureExtractor`` must not change while new
representations (shape_space, delay_embedding, topological) are validated
alongside it. These functions duplicate a handful of small operations
(NaN interpolation, smoothing, centroid/orientation) rather than importing
from the frozen module, at the cost of some repetition.

All functions operate on ``pose: (T, K, D)`` arrays (T frames, K keypoints,
D=2 coordinates), the same convention used throughout ``ml/``.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def interpolate_nans(pose: np.ndarray) -> np.ndarray:
    """Linearly interpolate missing (NaN) values per keypoint trajectory.

    A trajectory that is entirely NaN is filled with zeros (matches
    ``PoseFeatureExtractor._interpolate_nans``'s convention).
    """
    pose_interp = pose.copy()
    T, K, D = pose.shape
    for k in range(K):
        for d in range(D):
            trajectory = pose[:, k, d]
            nans = np.isnan(trajectory)
            if nans.all():
                pose_interp[:, k, d] = 0
            elif nans.any():
                valid_idx = np.where(~nans)[0]
                pose_interp[nans, k, d] = np.interp(
                    np.where(nans)[0], valid_idx, trajectory[valid_idx]
                )
    return pose_interp


def smooth_pose(pose: np.ndarray, window: int = 5) -> np.ndarray:
    """Savitzky-Golay smoothing per keypoint trajectory (polyorder=2)."""
    if window < 3:
        return pose
    T, K, D = pose.shape
    win = min(window, T)
    if win % 2 == 0:
        win -= 1
    win = max(3, win)
    if T < win:
        return pose.copy()
    smoothed = np.zeros_like(pose)
    for k in range(K):
        for d in range(D):
            smoothed[:, k, d] = savgol_filter(pose[:, k, d], win, polyorder=2)
    return smoothed


def compute_centroid(pose: np.ndarray) -> np.ndarray:
    """Mean of all keypoints per frame. Shape (T, D)."""
    return np.mean(pose, axis=1)


def compute_pca_orientation(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame PCA principal-axis angle and elongation (sqrt(lambda2/lambda1)).

    Returns
    -------
    orientation : (T,) angle in radians
    elongation  : (T,) in [0, 1], 0 = a line, 1 = a circle
    """
    T = pose.shape[0]
    orientation = np.zeros(T)
    elongation = np.zeros(T)
    for t in range(T):
        points = pose[t]
        centered = points - points.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        principal_axis = eigenvectors[:, 0]
        orientation[t] = np.arctan2(principal_axis[1], principal_axis[0])
        lam1, lam2 = max(eigenvalues[0], 1e-12), max(eigenvalues[-1], 0.0)
        elongation[t] = float(np.sqrt(lam2 / lam1))
    return orientation, elongation


def compute_speed(signal_2d: np.ndarray, fps: float) -> np.ndarray:
    """Frame-to-frame speed (norm of central-difference velocity) of a (T, D) signal."""
    T = signal_2d.shape[0]
    velocity = np.zeros_like(signal_2d)
    if T >= 3:
        velocity[1:-1] = (signal_2d[2:] - signal_2d[:-2]) * (fps / 2.0)
        velocity[0] = (signal_2d[1] - signal_2d[0]) * fps
        velocity[-1] = (signal_2d[-1] - signal_2d[-2]) * fps
    elif T == 2:
        velocity[0] = velocity[1] = (signal_2d[1] - signal_2d[0]) * fps
    return np.linalg.norm(velocity, axis=-1)


def prepare_pose(pose: np.ndarray, smooth_window: int = 5) -> np.ndarray:
    """Standard prep pipeline shared by every alternative representation: interpolate then smooth."""
    return smooth_pose(interpolate_nans(pose), window=smooth_window)
