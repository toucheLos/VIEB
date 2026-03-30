"""
preprocessing.py

Data preprocessing and normalization for behavioral ML models.
"""

import numpy as np
from typing import Optional
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


