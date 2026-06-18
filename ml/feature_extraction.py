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


def resolve_feature_indices(feature_names: list) -> dict:
    """Build a {name: index} lookup from a feature_names list.

    Works with the list stored in index.json ``_meta.feature_names`` or
    returned by ``PoseFeatureExtractor.get_feature_names()``.
    """
    return {name: i for i, name in enumerate(feature_names)}


class PoseFeatureExtractor:
    """
    Extract behavioral features from pose time series.

    Features are organised in two layers:

    **Layer 1 — Universal** (always computed, no keypoint-name assumptions):
      per-keypoint speed, pairwise distances, centroid speed, body orientation
      (PCA fallback), elongation (PCA), angular velocity, movement entropy,
      temporal window statistics, Morlet wavelet amplitudes.

    **Layer 2 — Semantic** (computed only when required keypoint roles exist):
      rearing score, head angle.  Each entry in ``_SEMANTIC_FEATURES`` maps a
      feature name to the roles it needs; if any role is unresolved the feature
      is omitted and the vector is shorter.
    """

    # Semantic roles used for postural features
    _KNOWN_ROLES: tuple = ("nose", "left_ear", "right_ear", "center/centroid", "tail_base")

    # Hardcoded defaults for the 8-point Luna lab mouse model
    # (0=left_ear, 1=right_ear, 2=nose, 3=center, 4=left_hip, 5=right_hip, 6=tail_base, 7=tail_tip)
    _DEFAULT_ROLE_IDX: dict = {
        "nose": 2,
        "left_ear": 0,
        "right_ear": 1,
        "center/centroid": 3,
        "tail_base": 6,
    }

    # Layer 2 — semantic features and the roles each one requires.
    # To add a new semantic feature: add an entry here and a matching
    # ``_compute_<name>`` method.  The framework handles the rest.
    _SEMANTIC_FEATURES: dict = {
        "rearing_score": ("nose", "tail_base", "left_ear", "right_ear"),
        "head_angle":    ("nose", "tail_base", "left_ear", "right_ear"),
    }

    def __init__(
        self,
        fps: float = 30.0,
        smooth_window: int = 5,
        feature_window: int = 30,
        use_wavelets: bool = True,
        keypoint_roles: Optional[dict] = None,
        bodypart_names: Optional[List[str]] = None,
        object_keypoint_indices: Optional[List[int]] = None,
        object_keypoints: Optional[List[str]] = None,
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
        keypoint_roles : dict, optional
            Mapping from keypoint name → semantic role string (from config.json).
        bodypart_names : list of str, optional
            Ordered keypoint names from DLC config (defines index → name mapping).
            When provided, role indices are resolved from this list.
        object_keypoint_indices : list of int, optional
            Integer indices of keypoints that are object trackers, not body parts.
        object_keypoints : list of str, optional
            Names of keypoints that are object trackers (resolved via bodypart_names).
        """
        self.fps = fps
        self.smooth_window = smooth_window
        self.feature_window = feature_window
        self.use_wavelets = use_wavelets
        # Morlet wavelet frequencies (Hz)
        self._wavelet_freqs = np.array([1.0, 2.0, 4.0, 8.0, 16.0])

        # Start with hardcoded defaults; resolve from config when bodypart_names provided
        self._role_idx: dict = dict(self._DEFAULT_ROLE_IDX)
        self._roles_resolved: bool = False
        self._object_kp_indices: set = set(object_keypoint_indices or [])
        self._n_bodyparts: Optional[int] = None

        if bodypart_names is not None:
            resolved = self._resolve_keypoint_indices(bodypart_names, keypoint_roles or {})
            # When bodypart_names is explicitly provided, use only what was
            # resolved — don't fall back to hardcoded defaults which assume
            # the 8-point mouse model and may reference invalid indices.
            self._role_idx = resolved
            self._roles_resolved = bool(resolved)
            self._object_kp_indices = self._resolve_object_indices(
                bodypart_names, object_keypoints or []
            )
            self._n_bodyparts = len(bodypart_names)

        # Determine which Layer 2 (semantic) features can be computed
        self._available_semantic: set = set()
        for feat_name, required_roles in self._SEMANTIC_FEATURES.items():
            if all(self._get_role_idx(r) is not None for r in required_roles):
                self._available_semantic.add(feat_name)

    @classmethod
    def _resolve_keypoint_indices(
        cls,
        bodypart_names: List[str],
        keypoint_roles: dict,
    ) -> dict:
        """
        Build a role → index mapping from bodypart_names and keypoint_roles config.

        Parameters
        ----------
        bodypart_names : list of str
            Ordered keypoint names from the DLC config bodyparts list.
        keypoint_roles : dict
            Mapping from keypoint name → semantic role string (from config.json).

        Returns
        -------
        dict mapping role string → integer index into the pose keypoint axis.
        Empty dict if bodypart_names is empty.
        """
        if not bodypart_names:
            return {}

        name_exact_to_idx: dict = {name: i for i, name in enumerate(bodypart_names)}
        name_lower_to_idx: dict = {name.lower(): i for i, name in enumerate(bodypart_names)}

        # Invert keypoint_roles: role → list of keypoint names
        role_to_names: dict = {}
        for kp_name, role in (keypoint_roles or {}).items():
            role_to_names.setdefault(role, []).append(kp_name)

        result: dict = {}
        for role in cls._KNOWN_ROLES:
            # 1. Try explicit entry in keypoint_roles
            if role in role_to_names:
                for kp_name in role_to_names[role]:
                    if kp_name in name_exact_to_idx:
                        result[role] = name_exact_to_idx[kp_name]
                        break
                    if kp_name.lower() in name_lower_to_idx:
                        result[role] = name_lower_to_idx[kp_name.lower()]
                        break
                if role in result:
                    continue

            # 2. Case-insensitive match of role name (or aliases) against bodypart names
            candidates = [role.lower()]
            if "/" in role:
                candidates = [p.lower() for p in role.split("/")]
            for cand in candidates:
                if cand in name_lower_to_idx:
                    result[role] = name_lower_to_idx[cand]
                    break

        return result

    @classmethod
    def _resolve_object_indices(
        cls,
        bodypart_names: List[str],
        object_keypoints: List[str],
    ) -> set:
        """
        Return the set of integer indices for object keypoints.

        Parameters
        ----------
        bodypart_names : list of str
            Ordered keypoint names from the DLC config.
        object_keypoints : list of str
            Names of keypoints that track objects (from config.json "object_keypoints").
        """
        if not bodypart_names or not object_keypoints:
            return set()
        name_to_idx = {name: i for i, name in enumerate(bodypart_names)}
        name_lower_to_idx = {name.lower(): i for i, name in enumerate(bodypart_names)}
        indices: set = set()
        for kp in object_keypoints:
            if kp in name_to_idx:
                indices.add(name_to_idx[kp])
            elif kp.lower() in name_lower_to_idx:
                indices.add(name_lower_to_idx[kp.lower()])
        return indices

    @property
    def _body_kp_mask(self) -> Optional[np.ndarray]:
        """Boolean mask of shape (K,) where True = body keypoint (not an object).

        Returns None when no object keypoints are defined (backward-compatible path).
        """
        if not self._object_kp_indices:
            return None
        if self._n_bodyparts is None:
            return None
        mask = np.ones(self._n_bodyparts, dtype=bool)
        for idx in self._object_kp_indices:
            if 0 <= idx < self._n_bodyparts:
                mask[idx] = False
        return mask

    def _get_role_idx(self, role: str) -> Optional[int]:
        """Return the keypoint index for a semantic role, or None if unavailable."""
        return self._role_idx.get(role, None)

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
        self._last_n_keypoints = K

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

        # --- Layer 2: semantic postural indicators (conditional) ---
        if "rearing_score" in self._available_semantic:
            features["rearing_score"] = self._compute_rearing_score(pose_smooth)
        if "head_angle" in self._available_semantic:
            features["head_angle"] = self._compute_head_angle(pose_smooth)

        # --- Temporal dynamics ---
        features["movement_entropy"] = self._compute_movement_entropy(features["speed"])
        features["temporal_features"] = self._compute_temporal_features(features)

        # --- Morlet wavelet amplitudes (optional) ---
        if self.use_wavelets:
            features["wavelet_amplitudes"] = self._compute_wavelet_features(pose_smooth)

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
        Compute body centroid.

        Uses the "center/centroid" keypoint directly when roles were explicitly
        resolved from a DLC config; otherwise falls back to the mean of body
        keypoints only (object keypoints excluded when mask is defined).

        Returns
        -------
        centroid : np.ndarray
            Shape (T, D)
        """
        if self._roles_resolved:
            center_idx = self._get_role_idx("center/centroid")
            if center_idx is not None:
                return pose[:, center_idx, :]
        mask = self._body_kp_mask
        if mask is not None and mask.shape[0] == pose.shape[1]:
            return np.mean(pose[:, mask, :], axis=1)
        return np.mean(pose, axis=1)

    def _compute_body_orientation(self, pose: np.ndarray) -> np.ndarray:
        """
        Compute body orientation angle.

        Uses the tail_base → nose axis when roles were explicitly resolved from a
        DLC config. Falls back to PCA over all keypoints for backward compatibility
        (or when the required roles are unavailable).

        Returns
        -------
        orientation : np.ndarray
            Shape (T,) - angle in radians
        """
        if self._roles_resolved:
            nose_idx      = self._get_role_idx("nose")
            tail_base_idx = self._get_role_idx("tail_base")
            if nose_idx is not None and tail_base_idx is not None:
                body_vec = pose[:, nose_idx, :] - pose[:, tail_base_idx, :]
                return np.arctan2(body_vec[:, 1], body_vec[:, 0])

        # PCA fallback — filter to body keypoints when object mask is available
        mask = self._body_kp_mask
        use_mask = mask is not None and mask.shape[0] == pose.shape[1]
        T = pose.shape[0]
        orientation = np.zeros(T)
        for t in range(T):
            points = pose[t][mask] if use_mask else pose[t]
            centered = points - points.mean(axis=0)
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            principal_axis = eigenvectors[:, np.argmax(eigenvalues)]
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

    def _compute_rearing_score(self, pose: np.ndarray) -> np.ndarray:
        """
        Estimate rearing likelihood from a top-down camera.

        When the mouse rears, its 2D projection contracts: the nose moves
        closer to the tail base relative to the inter-ear span.
        High score = more contracted / likely rearing.

        Returns zeros if any required role (nose, tail_base, left_ear, right_ear)
        cannot be resolved for the current keypoint configuration.

        Returns
        -------
        rearing_score : np.ndarray
            Shape (T,) - ratio of inter-ear span to nose-tail distance.
        """
        nose_idx      = self._get_role_idx("nose")
        tail_base_idx = self._get_role_idx("tail_base")
        left_ear_idx  = self._get_role_idx("left_ear")
        right_ear_idx = self._get_role_idx("right_ear")

        if any(idx is None for idx in [nose_idx, tail_base_idx, left_ear_idx, right_ear_idx]):
            return np.zeros(len(pose))

        nose      = pose[:, nose_idx, :]
        tail_base = pose[:, tail_base_idx, :]
        left_ear  = pose[:, left_ear_idx, :]
        right_ear = pose[:, right_ear_idx, :]

        nose_tail_dist = np.linalg.norm(nose - tail_base, axis=1)
        ear_span       = np.linalg.norm(left_ear - right_ear, axis=1)

        # Avoid division by zero; large ratio means body is contracted
        score = ear_span / np.maximum(nose_tail_dist, 1e-6)
        return score

    def _compute_head_angle(self, pose: np.ndarray) -> np.ndarray:
        """
        Compute head angle relative to body axis.

        Body axis: tail_base → nose.
        Head direction: midpoint(ears) → nose.
        Returns the signed angle between them in radians.
        High absolute value = head turned sideways (exploration/investigation).
        Near zero = head aligned with body (locomotion/freezing).

        Returns zeros if any required role (nose, tail_base, left_ear, right_ear)
        cannot be resolved for the current keypoint configuration.

        Returns
        -------
        head_angle : np.ndarray
            Shape (T,) in radians, range [-pi, pi].
        """
        nose_idx      = self._get_role_idx("nose")
        tail_base_idx = self._get_role_idx("tail_base")
        left_ear_idx  = self._get_role_idx("left_ear")
        right_ear_idx = self._get_role_idx("right_ear")

        if any(idx is None for idx in [nose_idx, tail_base_idx, left_ear_idx, right_ear_idx]):
            return np.zeros(len(pose))

        nose      = pose[:, nose_idx, :]
        tail_base = pose[:, tail_base_idx, :]
        left_ear  = pose[:, left_ear_idx, :]
        right_ear = pose[:, right_ear_idx, :]

        ear_mid = (left_ear + right_ear) / 2.0

        body_vec = nose - tail_base
        head_vec = nose - ear_mid

        # Signed angle: arctan2 of 2D cross product / dot product
        cross = body_vec[:, 0] * head_vec[:, 1] - body_vec[:, 1] * head_vec[:, 0]
        dot   = body_vec[:, 0] * head_vec[:, 0] + body_vec[:, 1] * head_vec[:, 1]
        return np.arctan2(cross, dot)

    def _compute_wavelet_features(self, pose_smooth: np.ndarray) -> np.ndarray:
        """
        Morlet wavelet decomposition of per-keypoint speed.

        Computes instantaneous amplitude (envelope) at each frequency for each
        keypoint using a continuous wavelet transform (CWT) with Morlet2 wavelet.

        Parameters
        ----------
        pose_smooth : np.ndarray
            Smoothed pose array of shape (T, K, 2).

        Returns
        -------
        amplitudes : np.ndarray
            Shape (T, K * n_freqs) where n_freqs = len(self._wavelet_freqs).
        """
        T, K, D = pose_smooth.shape
        freqs = self._wavelet_freqs
        n_freqs = len(freqs)

        # Morlet2 central frequency in radians (default w=5.0)
        w = 5.0
        # Scale: s = w * fps / (2π * f) maps frequency f (Hz) → CWT width (samples)
        widths = (w * self.fps) / (2.0 * np.pi * freqs)

        # Compute per-keypoint speed (scalar, shape (T, K))
        grad = np.gradient(pose_smooth, axis=0) * self.fps  # (T, K, 2), pixels/s
        speed = np.linalg.norm(grad, axis=2)               # (T, K)

        amplitudes = np.zeros((T, K * n_freqs), dtype=np.float32)
        for k in range(K):
            sig = speed[:, k]
            coef = self._morlet_cwt(sig, widths)   # (n_freqs, T), complex
            amp = np.abs(coef).T                   # (T, n_freqs)
            amplitudes[:, k * n_freqs:(k + 1) * n_freqs] = amp.astype(np.float32)

        return amplitudes

    @staticmethod
    def _morlet_cwt(data: np.ndarray, widths: np.ndarray, w: float = 5.0) -> np.ndarray:
        """
        Continuous wavelet transform with complex Morlet wavelet.

        Implements the equivalent of the removed scipy.signal.cwt + morlet2
        using FFT convolution directly.

        Returns complex array of shape (len(widths), len(data)).
        """
        from scipy.signal import fftconvolve

        out = np.zeros((len(widths), len(data)), dtype=complex)
        for i, s in enumerate(widths):
            N = int(min(10.0 * s, len(data)))
            N = max(N, 1)
            # Morlet2 wavelet: normalised complex Morlet at scale s
            x = np.arange(N) - (N - 1) / 2.0
            x /= s
            wav = (np.exp(1j * w * x) * np.exp(-0.5 * x ** 2)
                   * np.pi ** (-0.25) / np.sqrt(s))
            out[i] = fftconvolve(data, np.conj(wav[::-1]), mode="same")
        return out

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

        # Layer 1: universal postural features
        feature_arrays.append(features["body_orientation"][:, None])  # (T, 1)
        feature_arrays.append(features["elongation"][:, None])  # (T, 1)
        feature_arrays.append(features["angular_velocity"][:, None])  # (T, 1)
        feature_arrays.append(features["movement_entropy"][:, None])  # (T, 1)

        # Layer 2: semantic postural features (only present when roles available)
        if "rearing_score" in features:
            feature_arrays.append(features["rearing_score"][:, None])    # (T, 1)
        if "head_angle" in features:
            feature_arrays.append(features["head_angle"][:, None])       # (T, 1)

        # Temporal features
        feature_arrays.append(features["temporal_features"])  # (T, M)

        # Morlet wavelet amplitudes (optional)
        if self.use_wavelets and "wavelet_amplitudes" in features:
            feature_arrays.append(features["wavelet_amplitudes"])  # (T, K*n_freqs)

        # Concatenate all features
        flattened = np.concatenate(feature_arrays, axis=1)

        return flattened

    def get_feature_names(self, n_keypoints: int = None) -> List[str]:
        """
        Get human-readable names for all features in flattened array.

        Returns
        -------
        names : list of str
        """
        if n_keypoints is None:
            n_keypoints = getattr(self, '_last_n_keypoints', 8)
        names = []

        # Speed per keypoint
        for k in range(n_keypoints):
            names.append(f"speed_kp{k}")

        # Pairwise distances
        n_pairs = n_keypoints * (n_keypoints - 1) // 2
        for p in range(n_pairs):
            names.append(f"dist_pair{p}")

        # Layer 1: universal scalar features
        names.extend([
            "centroid_speed",
            "body_orientation",
            "elongation",
            "angular_velocity",
            "movement_entropy",
        ])

        # Layer 2: semantic scalar features (only when roles are available)
        if "rearing_score" in self._available_semantic:
            names.append("rearing_score")
        if "head_angle" in self._available_semantic:
            names.append("head_angle")

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

        # Morlet wavelet amplitudes
        if self.use_wavelets:
            for k in range(n_keypoints):
                for f in self._wavelet_freqs:
                    names.append(f"wavelet_kp{k}_{int(f)}hz")

        return names

    def feature_index(self, name: str, n_keypoints: int = None) -> Optional[int]:
        """Return the column index of a named feature, or None if unavailable."""
        names = self.get_feature_names(n_keypoints)
        try:
            return names.index(name)
        except ValueError:
            return None

    def get_feature_meta(self, n_keypoints: int = None) -> dict:
        """Return metadata about the feature vector for serialization in index.json."""
        names = self.get_feature_names(n_keypoints)
        return {
            "feature_names": names,
            "n_features": len(names),
            "n_keypoints": n_keypoints or getattr(self, '_last_n_keypoints', 8),
            "use_wavelets": self.use_wavelets,
            "semantic_features": sorted(self._available_semantic),
        }
