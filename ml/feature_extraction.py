"""
feature_extraction.py

Extract behavioral features from raw pose data for machine learning analysis.

This module transforms pose sequences (T, K, D) into feature vectors that capture:
- Spatial relationships between keypoints
- Kinematic properties (velocity, acceleration, angular velocity)
- Postural features (body orientation, elongation)
- Temporal dynamics (movement patterns over time windows)

These features enable detection of subtle behavioral patterns invisible to the naked eye.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import pdist, squareform
from scipy.signal import savgol_filter


class PoseFeatureExtractor:
    """
    Extract behavioral features from pose time series.

    Features include:
    - Velocities and accelerations
    - Inter-keypoint distances
    - Body angles and orientations
    - Movement statistics (speed, direction changes)
    - Postural configurations
    """

    def __init__(
        self,
        fps: float = 30.0,
        smooth_window: int = 5,
        feature_window: int = 30,
    ):
        """
        Parameters
        ----------
        fps : float
            Frames per second of the video (for velocity calculations).
        smooth_window : int
            Window size for Savitzky-Golay smoothing filter.
        feature_window : int
            Number of frames for temporal aggregation features.
        """
        self.fps = fps
        self.smooth_window = smooth_window
        self.feature_window = feature_window

    def extract_features(self, pose: np.ndarray, confidence: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Extract all behavioral features from pose data.

        Parameters
        ----------
        pose : np.ndarray
            Pose tensor of shape (T, K, D) where:
            T = number of frames
            K = number of keypoints (8 for mouse)
            D = spatial dimensions (2 for x,y)
        confidence : np.ndarray, optional
            Confidence scores of shape (T, K)

        Returns
        -------
        features : dict
            Dictionary containing extracted features:
            - "velocity": (T, K, D) - per-keypoint velocities
            - "acceleration": (T, K, D) - per-keypoint accelerations
            - "speed": (T, K) - per-keypoint scalar speeds
            - "distances": (T, K*(K-1)/2) - pairwise keypoint distances
            - "centroid": (T, D) - body centroid position
            - "centroid_velocity": (T, D) - centroid velocity
            - "body_orientation": (T,) - body angle
            - "elongation": (T,) - body elongation (aspect ratio)
            - "angular_velocity": (T,) - rate of rotation
            - "movement_entropy": (T,) - movement predictability
            - "temporal_features": (T, M) - aggregated temporal statistics
        """
        T, K, D = pose.shape

        # Handle NaN values (low confidence predictions)
        pose_clean = self._interpolate_nans(pose)

        # Smooth trajectories to reduce noise
        pose_smooth = self._smooth_pose(pose_clean)

        features = {}

        # --- Kinematic features ---
        features["velocity"] = self._compute_velocity(pose_smooth)
        features["acceleration"] = self._compute_acceleration(features["velocity"])
        features["speed"] = np.linalg.norm(features["velocity"], axis=2)

        # --- Spatial features ---
        features["distances"] = self._compute_pairwise_distances(pose_smooth)
        features["centroid"] = self._compute_centroid(pose_smooth)
        features["centroid_velocity"] = self._compute_velocity(features["centroid"][:, None, :]).squeeze(1)

        # --- Postural features ---
        features["body_orientation"] = self._compute_body_orientation(pose_smooth)
        features["elongation"] = self._compute_elongation(pose_smooth)
        features["angular_velocity"] = self._compute_angular_velocity(features["body_orientation"])

        # --- Temporal dynamics ---
        features["movement_entropy"] = self._compute_movement_entropy(features["speed"])
        features["temporal_features"] = self._compute_temporal_features(features)

        # Flatten features for ML (convert to 2D: samples x features)
        features["flattened"] = self._flatten_features(features)

        return features

    def _interpolate_nans(self, pose: np.ndarray) -> np.ndarray:
        """
        Interpolate missing (NaN) values in pose trajectories.
        """
        pose_interp = pose.copy()
        T, K, D = pose.shape

        for k in range(K):
            for d in range(D):
                trajectory = pose[:, k, d]
                nans = np.isnan(trajectory)

                if nans.all():
                    # If entire trajectory is NaN, fill with zeros
                    pose_interp[:, k, d] = 0
                elif nans.any():
                    # Linear interpolation for missing values
                    valid_idx = np.where(~nans)[0]
                    pose_interp[nans, k, d] = np.interp(
                        np.where(nans)[0],
                        valid_idx,
                        trajectory[valid_idx]
                    )

        return pose_interp

    def _smooth_pose(self, pose: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay filter to smooth trajectories.
        """
        if self.smooth_window < 3:
            return pose

        T, K, D = pose.shape
        pose_smooth = np.zeros_like(pose)

        # Ensure window size is odd and <= sequence length
        window = min(self.smooth_window, T)
        if window % 2 == 0:
            window -= 1
        window = max(3, window)

        for k in range(K):
            for d in range(D):
                if T >= window:
                    pose_smooth[:, k, d] = savgol_filter(pose[:, k, d], window, polyorder=2)
                else:
                    pose_smooth[:, k, d] = pose[:, k, d]

        return pose_smooth

    def _compute_velocity(self, pose: np.ndarray) -> np.ndarray:
        """
        Compute velocities using central differences.

        Returns
        -------
        velocity : np.ndarray
            Shape (T, K, D) in pixels per second
        """
        T = pose.shape[0]
        velocity = np.zeros_like(pose)

        # Central differences
        velocity[1:-1] = (pose[2:] - pose[:-2]) / (2 / self.fps)

        # Forward/backward differences at boundaries
        velocity[0] = (pose[1] - pose[0]) * self.fps
        velocity[-1] = (pose[-1] - pose[-2]) * self.fps

        return velocity

    def _compute_acceleration(self, velocity: np.ndarray) -> np.ndarray:
        """
        Compute accelerations from velocities.
        """
        return self._compute_velocity(velocity)

    def _compute_pairwise_distances(self, pose: np.ndarray) -> np.ndarray:
        """
        Compute pairwise Euclidean distances between all keypoints.

        Returns
        -------
        distances : np.ndarray
            Shape (T, K*(K-1)/2) - condensed distance matrix per frame
        """
        T, K, D = pose.shape
        n_pairs = K * (K - 1) // 2

        distances = np.zeros((T, n_pairs))

        for t in range(T):
            # Compute pairwise distances for frame t
            distances[t] = pdist(pose[t])

        return distances

    def _compute_centroid(self, pose: np.ndarray) -> np.ndarray:
        """
        Compute body centroid (mean position of all keypoints).

        Returns
        -------
        centroid : np.nd rray
            Shape (T, D)
        """
        return np.mean(pose, axis=1)

    def _compute_body_orientation(self, pose: np.ndarray) -> np.ndarray:
        """
        Compute body orientation angle using PCA of keypoint positions.

        Returns
        -------
        orientation : np.ndarray
            Shape (T,) - angle in radians
        """
        T = pose.shape[0]
        orientation = np.zeros(T)

        for t in range(T):
            points = pose[t]  # (K, 2)

            # Center the points
            centered = points - points.mean(axis=0)

            # Compute covariance and PCA
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)

            # Principal axis (largest eigenvalue)
            principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

            # Compute angle
            orientation[t] = np.arctan2(principal_axis[1], principal_axis[0])

        return orientation

    def _compute_elongation(self, pose: np.ndarray) -> np.ndarray:
        """
        Compute body elongation (ratio of major to minor axis from PCA).

        Returns
        -------
        elongation : np.ndarray
            Shape (T,)
        """
        T = pose.shape[0]
        elongation = np.zeros(T)

        for t in range(T):
            points = pose[t]
            centered = points - points.mean(axis=0)
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)

            # Avoid division by zero
            if eigenvalues[0] > 1e-6:
                elongation[t] = np.sqrt(eigenvalues[1] / eigenvalues[0])
            else:
                elongation[t] = 1.0

        return elongation

    def _compute_angular_velocity(self, orientation: np.ndarray) -> np.ndarray:
        """
        Compute angular velocity (rate of body rotation).

        Returns
        -------
        angular_velocity : np.ndarray
            Shape (T,) in radians per second
        """
        # Handle angle wrapping
        diff = np.diff(orientation)
        diff = np.arctan2(np.sin(diff), np.cos(diff))  # Wrap to [-pi, pi]

        angular_vel = np.zeros(len(orientation))
        angular_vel[1:] = diff * self.fps

        return angular_vel

    def _compute_movement_entropy(self, speed: np.ndarray) -> np.ndarray:
        """
        Compute local movement entropy (predictability measure).

        High entropy = erratic/unpredictable movement
        Low entropy = stereotyped/repetitive movement

        Returns
        -------
        entropy : np.ndarray
            Shape (T,)
        """
        T = speed.shape[0]
        entropy = np.zeros(T)
        window = self.feature_window

        for t in range(window, T):
            # Get speed distribution in local window
            local_speeds = speed[t-window:t].flatten()

            # Compute histogram entropy
            hist, _ = np.histogram(local_speeds, bins=10, density=True)
            hist = hist[hist > 0]  # Remove zero bins
            entropy[t] = -np.sum(hist * np.log2(hist + 1e-10))

        return entropy

    def _compute_temporal_features(self, features: Dict) -> np.ndarray:
        """
        Compute temporal aggregation features over sliding windows.

        Returns
        -------
        temporal_features : np.ndarray
            Shape (T, M) where M is number of temporal statistics
        """
        T = features["speed"].shape[0]
        window = self.feature_window

        # Features to aggregate
        feature_list = []

        for t in range(T):
            start = max(0, t - window)

            # Aggregate statistics over window
            stats = []

            # Speed statistics
            speed_window = features["speed"][start:t+1]
            stats.extend([
                np.mean(speed_window),
                np.std(speed_window),
                np.max(speed_window),
                np.percentile(speed_window, 90)
            ])

            # Distance change statistics
            dist_window = features["distances"][start:t+1]
            stats.extend([
                np.mean(dist_window),
                np.std(dist_window)
            ])

            # Orientation change
            if t > 0:
                orientation_change = np.abs(features["angular_velocity"][start:t+1])
                stats.extend([
                    np.mean(orientation_change),
                    np.max(orientation_change)
                ])
            else:
                stats.extend([0, 0])

            feature_list.append(stats)

        return np.array(feature_list)

    def _flatten_features(self, features: Dict) -> np.ndarray:
        """
        Flatten all features into a single 2D array for ML models.

        Returns
        -------
        flattened : np.ndarray
            Shape (T, F) where F is total number of features
        """
        T = features["speed"].shape[0]

        feature_arrays = []

        # Per-keypoint features (flatten spatial/keypoint dimensions)
        feature_arrays.append(features["speed"])  # (T, K)

        # Pairwise distances
        feature_arrays.append(features["distances"])  # (T, pairs)

        # Centroid velocity (scalar speed)
        centroid_speed = np.linalg.norm(features["centroid_velocity"], axis=1, keepdims=True)
        feature_arrays.append(centroid_speed)  # (T, 1)

        # Postural features
        feature_arrays.append(features["body_orientation"][:, None])  # (T, 1)
        feature_arrays.append(features["elongation"][:, None])  # (T, 1)
        feature_arrays.append(features["angular_velocity"][:, None])  # (T, 1)
        feature_arrays.append(features["movement_entropy"][:, None])  # (T, 1)

        # Temporal features
        feature_arrays.append(features["temporal_features"])  # (T, M)

        # Concatenate all features
        flattened = np.concatenate(feature_arrays, axis=1)

        return flattened

    def get_feature_names(self, n_keypoints: int = 8) -> List[str]:
        """
        Get human-readable names for all features in flattened array.

        Returns
        -------
        names : list of str
        """
        names = []

        # Speed per keypoint
        for k in range(n_keypoints):
            names.append(f"speed_kp{k}")

        # Pairwise distances
        n_pairs = n_keypoints * (n_keypoints - 1) // 2
        for p in range(n_pairs):
            names.append(f"dist_pair{p}")

        # Scalar features
        names.extend([
            "centroid_speed",
            "body_orientation",
            "elongation",
            "angular_velocity",
            "movement_entropy"
        ])

        # Temporal aggregation features
        names.extend([
            "speed_mean_window",
            "speed_std_window",
            "speed_max_window",
            "speed_p90_window",
            "dist_mean_window",
            "dist_std_window",
            "angular_vel_mean_window",
            "angular_vel_max_window"
        ])

        return names
