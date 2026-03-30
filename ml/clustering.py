"""
clustering.py

KMeans-based clustering for discovering discrete behavioral states.
"""

import numpy as np
from typing import Dict, Optional
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt


class BehaviorClusterer:
    """
    Discover discrete behavioral states using unsupervised clustering.

    This class automatically detects patterns in mouse behavior by
    grouping similar feature vectors into behavioral states/motifs.
    """

    def __init__(
        self,
        method: str = "kmeans",
        n_clusters: Optional[int] = None,
        auto_tune: bool = True,
        min_clusters: int = 4,
        max_clusters: int = 20,
    ):
        """
        Parameters
        ----------
        method : str
            "kmeans" or "gmm".
        n_clusters : int, optional
            Number of clusters (if None and auto_tune=True, will be optimized)
        auto_tune : bool
            Whether to automatically determine optimal number of clusters
        min_clusters : int
            Minimum k to test when auto-tuning
        max_clusters : int
            Maximum number of clusters to consider when auto-tuning
        """
        self.method = method if method in ("kmeans", "gmm") else "kmeans"
        self.n_clusters = n_clusters
        self.auto_tune = auto_tune
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters

        self.model = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.fitted = False

    def fit(self, features: np.ndarray) -> 'BehaviorClusterer':
        """
        Fit clustering model to behavioral features.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (T, F)

        Returns
        -------
        self
        """
        # Auto-tune number of clusters if needed
        if self.auto_tune and self.n_clusters is None:
            self.n_clusters = self._find_optimal_clusters(features)
            print(f"Auto-tuned to {self.n_clusters} clusters")

        if self.method == "gmm":
            self._fit_gmm(features)
        else:
            self._fit_kmeans(features)

        self.fitted = True
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for new data.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (T, F)

        Returns
        -------
        labels : np.ndarray
            Cluster assignments of shape (T,)
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted before predict")

        return self.model.predict(features).astype(np.int32)

    def fit_predict(self, features: np.ndarray) -> np.ndarray:
        """
        Fit model and return cluster labels.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix of shape (T, F)

        Returns
        -------
        labels : np.ndarray
            Cluster assignments
        """
        self.fit(features)
        return self.labels_

    def _fit_kmeans(self, features: np.ndarray):
        """Fit K-Means clustering."""
        self.model = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            max_iter=300,
            random_state=42
        )
        self.labels_ = self.model.fit_predict(features)
        self.cluster_centers_ = self.model.cluster_centers_

    def _fit_gmm(self, features: np.ndarray):
        """Fit Gaussian Mixture Model clustering."""
        from sklearn.mixture import GaussianMixture
        self.model = GaussianMixture(
            n_components=self.n_clusters,
            covariance_type="full",
            n_init=5,
            random_state=42,
        )
        self.labels_ = self.model.fit_predict(features)
        # Expose cluster_centers_ as component means for compatibility
        self.cluster_centers_ = self.model.means_

    def _find_optimal_clusters(self, features: np.ndarray) -> int:
        """
        Find optimal number of clusters using silhouette score.

        Returns
        -------
        optimal_k : int
            Optimal number of clusters
        """
        min_k = max(2, self.min_clusters)
        max_k = min(self.max_clusters, len(features) // 10)

        if max_k < min_k:
            return min_k

        scores = []
        k_range = range(min_k, max_k + 1)

        for k in k_range:
            temp_model = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = temp_model.fit_predict(features)

            if len(set(labels)) < 2:
                scores.append(-1)
                continue

            score = silhouette_score(features, labels, sample_size=min(5000, len(features)))
            scores.append(score)
            print(f"    k={k}: silhouette={score:.4f}")

        optimal_k = k_range[np.argmax(scores)]
        return optimal_k

    def get_cluster_statistics(self, features: np.ndarray) -> Dict:
        """
        Compute statistics for each discovered cluster.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix used for clustering

        Returns
        -------
        stats : dict
            Dictionary containing cluster statistics
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted first")

        unique_labels = set(self.labels_)
        stats = {}

        for label in unique_labels:
            cluster_mask = self.labels_ == label
            cluster_features = features[cluster_mask]

            stats[f"cluster_{label}"] = {
                "size": np.sum(cluster_mask),
                "frequency": np.mean(cluster_mask),
                "center": self.cluster_centers_[label] if label < len(self.cluster_centers_) else None,
                "std": np.std(cluster_features, axis=0),
                "mean_duration": self._compute_mean_bout_duration(cluster_mask)
            }

        return stats

    def _compute_mean_bout_duration(self, cluster_mask: np.ndarray) -> float:
        """
        Compute mean duration of continuous bouts in a cluster.

        Parameters
        ----------
        cluster_mask : np.ndarray
            Boolean array indicating cluster membership

        Returns
        -------
        mean_duration : float
            Mean bout duration in frames
        """
        # Find continuous sequences
        diff = np.diff(np.concatenate([[0], cluster_mask.astype(int), [0]]))
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        if len(starts) == 0:
            return 0.0

        durations = ends - starts
        return np.mean(durations)

    def evaluate(self, features: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering quality using multiple metrics.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix

        Returns
        -------
        metrics : dict
            Dictionary of evaluation metrics
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted first")

        metrics = {}

        features_valid = features
        labels_valid = self.labels_

        if len(set(labels_valid)) < 2:
            return {"error": "Less than 2 clusters found"}

        # Silhouette score (higher is better, range [-1, 1])
        metrics["silhouette_score"] = silhouette_score(
            features_valid,
            labels_valid,
            sample_size=min(5000, len(features_valid))
        )

        # Davies-Bouldin index (lower is better)
        metrics["davies_bouldin_index"] = davies_bouldin_score(
            features_valid,
            labels_valid
        )

        # Calinski-Harabasz index (higher is better)
        metrics["calinski_harabasz_score"] = calinski_harabasz_score(
            features_valid,
            labels_valid
        )

        # Number of clusters
        metrics["n_clusters"] = len(set(labels_valid))

        return metrics

    def visualize_clusters(
        self,
        features: np.ndarray,
        method: str = "pca",
        save_path: Optional[str] = None
    ):
        """
        Visualize clusters in 2D space.

        Parameters
        ----------
        features : np.ndarray
            Feature matrix
        method : str
            Dimensionality reduction method: "pca" or "tsne"
        save_path : str, optional
            Path to save figure
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted first")

        # Reduce to 2D for visualization
        if method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            features_2d = reducer.fit_transform(features)
        elif method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            features_2d = reducer.fit_transform(features)
        else:
            raise ValueError(f"Unknown reduction method: {method}")

        # Plot
        plt.figure(figsize=(10, 8))

        unique_labels = set(self.labels_)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points in black
                color = 'k'
                marker = 'x'
                alpha = 0.3
            else:
                marker = 'o'
                alpha = 0.6

            mask = self.labels_ == label
            plt.scatter(
                features_2d[mask, 0],
                features_2d[mask, 1],
                c=[color],
                marker=marker,
                alpha=alpha,
                s=30,
                label=f'Cluster {label}' if label != -1 else 'Noise'
            )

        plt.xlabel(f'{method.upper()} Component 1')
        plt.ylabel(f'{method.upper()} Component 2')
        plt.title(f'Behavioral Clusters ({self.method.upper()})')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()

    def plot_temporal_states(
        self,
        fps: float = 30.0,
        save_path: Optional[str] = None
    ):
        """
        Plot behavioral states over time (ethogram).

        Parameters
        ----------
        fps : float
            Frames per second (for time axis)
        save_path : str, optional
            Path to save figure
        """
        if not self.fitted:
            raise RuntimeError("Model must be fitted first")

        time = np.arange(len(self.labels_)) / fps

        plt.figure(figsize=(15, 4))
        plt.plot(time, self.labels_, linewidth=0.5)
        plt.xlabel('Time (s)')
        plt.ylabel('Behavioral State')
        plt.title('Ethogram: Behavioral States Over Time')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
