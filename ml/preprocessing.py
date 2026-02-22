"""
preprocessing.py

Data preprocessing and normalization for behavioral ML models.

Handles:
- Feature normalization and scaling
- Train/test splitting with temporal awareness
- Data augmentation for behavioral sequences
- Handling missing values and low-confidence predictions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
import pickle


class BehaviorPreprocessor:
    """
    Preprocess behavioral features for machine learning.

    This class handles normalization, dimensionality reduction, and
    temporal data splitting to prepare features for downstream ML models.
    """

    def __init__(
        self,
        scaler_type: str = "standard",
        use_pca: bool = False,
        pca_variance: float = 0.95,
        remove_outliers: bool = True,
        outlier_threshold: float = 5.0
    ):
        """
        Parameters
        ----------
        scaler_type : str
            Type of scaler: "standard", "robust", or "minmax"
        use_pca : bool
            Whether to apply PCA for dimensionality reduction
        pca_variance : float
            Fraction of variance to retain if using PCA (0-1)
        remove_outliers : bool
            Whether to clip extreme outliers
        outlier_threshold : float
            Number of standard deviations for outlier clipping
        """
        self.scaler_type = scaler_type
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold

        # Initialize transformers
        self.scaler = None
        self.pca = None
        self.fitted = False

    def fit(self, features: np.ndarray) -> 'BehaviorPreprocessor':
        """
        Fit the preprocessor to training data.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (T, F)

        Returns
        -------
        self
        """
        # Initialize scaler
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "robust":
            self.scaler = RobustScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

        # Handle NaN and Inf values
        features_clean = self._handle_invalid_values(features)

        # Fit scaler
        self.scaler.fit(features_clean)

        # Fit PCA if enabled
        if self.use_pca:
            features_scaled = self.scaler.transform(features_clean)
            self.pca = PCA(n_components=self.pca_variance)
            self.pca.fit(features_scaled)

        self.fitted = True
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted preprocessor.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (T, F)

        Returns
        -------
        features_transformed : np.ndarray
            Preprocessed features
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before transform")

        # Handle invalid values
        features_clean = self._handle_invalid_values(features)

        # Scale features
        features_scaled = self.scaler.transform(features_clean)

        # Remove outliers if enabled
        if self.remove_outliers:
            features_scaled = self._clip_outliers(features_scaled)

        # Apply PCA if enabled
        if self.use_pca and self.pca is not None:
            features_scaled = self.pca.transform(features_scaled)

        return features_scaled

    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (T, F)

        Returns
        -------
        features_transformed : np.ndarray
        """
        self.fit(features)
        return self.transform(features)

    def inverse_transform(self, features_transformed: np.ndarray) -> np.ndarray:
        """
        Inverse transform features back to original scale.

        Useful for interpreting learned patterns.

        Parameters
        ----------
        features_transformed : np.ndarray
            Transformed features

        Returns
        -------
        features_original : np.ndarray
        """
        if not self.fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform")

        features = features_transformed

        # Inverse PCA if applied
        if self.use_pca and self.pca is not None:
            features = self.pca.inverse_transform(features)

        # Inverse scaling
        features = self.scaler.inverse_transform(features)

        return features

    def _handle_invalid_values(self, features: np.ndarray) -> np.ndarray:
        """
        Replace NaN and Inf values.
        """
        features_clean = features.copy()

        # Replace NaN with column mean
        col_mean = np.nanmean(features_clean, axis=0)
        nan_idx = np.isnan(features_clean)
        features_clean[nan_idx] = np.take(col_mean, np.where(nan_idx)[1])

        # Replace Inf with large finite values
        features_clean[np.isinf(features_clean)] = np.nan
        col_max = np.nanmax(np.abs(features_clean), axis=0)
        inf_idx = np.isnan(features_clean)
        features_clean[inf_idx] = np.take(col_max * 10, np.where(inf_idx)[1])

        # Final safety check - replace any remaining NaN with 0
        features_clean = np.nan_to_num(features_clean, nan=0.0, posinf=0.0, neginf=0.0)

        return features_clean

    def _clip_outliers(self, features: np.ndarray) -> np.ndarray:
        """
        Clip extreme outliers to threshold.
        """
        features_clipped = features.copy()

        # Clip to ±threshold standard deviations
        features_clipped = np.clip(
            features_clipped,
            -self.outlier_threshold,
            self.outlier_threshold
        )

        return features_clipped

    def save(self, filepath: str):
        """
        Save fitted preprocessor to disk.

        Parameters
        ----------
        filepath : str
            Path to save pickle file
        """
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted preprocessor")

        with open(filepath, 'wb') as f:
            pickle.dump({
                'scaler': self.scaler,
                'pca': self.pca,
                'scaler_type': self.scaler_type,
                'use_pca': self.use_pca,
                'pca_variance': self.pca_variance,
                'remove_outliers': self.remove_outliers,
                'outlier_threshold': self.outlier_threshold,
            }, f)

    @classmethod
    def load(cls, filepath: str) -> 'BehaviorPreprocessor':
        """
        Load fitted preprocessor from disk.

        Parameters
        ----------
        filepath : str
            Path to saved pickle file

        Returns
        -------
        preprocessor : BehaviorPreprocessor
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        preprocessor = cls(
            scaler_type=data['scaler_type'],
            use_pca=data['use_pca'],
            pca_variance=data['pca_variance'],
            remove_outliers=data['remove_outliers'],
            outlier_threshold=data['outlier_threshold']
        )

        preprocessor.scaler = data['scaler']
        preprocessor.pca = data['pca']
        preprocessor.fitted = True

        return preprocessor


class TemporalDataSplitter:
    """
    Split behavioral data while respecting temporal structure.

    Standard random train/test splits break temporal correlations.
    This class provides splits that preserve temporal ordering.
    """

    @staticmethod
    def temporal_train_test_split(
        features: np.ndarray,
        train_ratio: float = 0.8,
        gap_frames: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split time series data temporally (not randomly).

        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (T, F)
        train_ratio : float
            Fraction of data to use for training (0-1)
        gap_frames : int
            Number of frames to skip between train and test sets
            (prevents information leakage from temporal smoothing)

        Returns
        -------
        train_features : np.ndarray
        test_features : np.ndarray
        """
        T = features.shape[0]
        split_idx = int(T * train_ratio)

        train_features = features[:split_idx]
        test_features = features[split_idx + gap_frames:]

        return train_features, test_features

    @staticmethod
    def create_sequences(
        features: np.ndarray,
        sequence_length: int,
        stride: int = 1,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Create sliding window sequences for sequential models.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (T, F)
        sequence_length : int
            Length of each sequence
        stride : int
            Step size between consecutive sequences
        labels : np.ndarray, optional
            Optional labels for supervised learning

        Returns
        -------
        sequences : np.ndarray
            Shape (N, sequence_length, F)
        sequence_labels : np.ndarray, optional
            Shape (N,) or (N, ...) depending on label structure
        """
        T, F = features.shape
        sequences = []
        sequence_labels = [] if labels is not None else None

        for start_idx in range(0, T - sequence_length + 1, stride):
            end_idx = start_idx + sequence_length
            sequences.append(features[start_idx:end_idx])

            if labels is not None:
                # Use label from the last frame of the sequence
                sequence_labels.append(labels[end_idx - 1])

        sequences = np.array(sequences)

        if sequence_labels is not None:
            sequence_labels = np.array(sequence_labels)

        return sequences, sequence_labels

    @staticmethod
    def augment_sequences(
        sequences: np.ndarray,
        noise_level: float = 0.01,
        time_warp: bool = False
    ) -> np.ndarray:
        """
        Apply data augmentation to behavioral sequences.

        Parameters
        ----------
        sequences : np.ndarray
            Input sequences of shape (N, T, F)
        noise_level : float
            Standard deviation of Gaussian noise to add
        time_warp : bool
            Whether to apply temporal warping augmentation

        Returns
        -------
        augmented_sequences : np.ndarray
            Augmented sequences
        """
        augmented = sequences.copy()

        # Add Gaussian noise
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, augmented.shape)
            augmented = augmented + noise

        # Time warping (stretch/compress time axis slightly)
        if time_warp:
            N, T, F = augmented.shape
            for i in range(N):
                # Random time warp factor
                warp_factor = np.random.uniform(0.9, 1.1)
                new_T = int(T * warp_factor)

                # Resample sequence
                old_indices = np.linspace(0, T - 1, new_T)
                new_sequence = np.zeros((T, F))

                for f in range(F):
                    new_sequence[:, f] = np.interp(
                        np.arange(T),
                        old_indices,
                        augmented[i, :new_T, f]
                    )

                augmented[i] = new_sequence

        return augmented


def normalize_pose_to_body_frame(pose: np.ndarray) -> np.ndarray:
    """
    Normalize pose to body-centered reference frame.

    This makes features translation and rotation invariant.

    Parameters
    ----------
    pose : np.ndarray
        Pose tensor of shape (T, K, D)

    Returns
    -------
    pose_normalized : np.ndarray
        Body-centered pose coordinates
    """
    T, K, D = pose.shape
    pose_norm = np.zeros_like(pose)

    for t in range(T):
        points = pose[t]  # (K, D)

        # Center on centroid
        centroid = np.mean(points, axis=0)
        centered = points - centroid

        # Rotate to align principal axis with x-axis
        if D == 2:
            cov = np.cov(centered.T)
            eigenvalues, eigenvectors = np.linalg.eig(cov)
            principal_axis = eigenvectors[:, np.argmax(eigenvalues)]

            # Rotation angle
            angle = np.arctan2(principal_axis[1], principal_axis[0])

            # Rotation matrix
            cos_a, sin_a = np.cos(-angle), np.sin(-angle)
            rotation = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

            # Apply rotation
            pose_norm[t] = centered @ rotation.T

        else:
            # For 3D poses, use more complex alignment
            pose_norm[t] = centered

    return pose_norm
