"""The v1 arm: standardize, UMAP to 10-D, HDBSCAN, then HMM Viterbi smoothing.

Adapter over ``compare.py``'s ``cmd_cluster`` path. The helpers are **imported
from ``compare``, not reimplemented**, so this arm computes what it has always
computed — a reimplementation is exactly how a port silently changes a method.

Two ways it differs from the ``hdbscan`` segmenter, both preserved:

- **It reduces with UMAP first.** v2 clusters the delay embedding directly, and
  ``vieb_v2/representation/cluster.py`` calls that "a deliberate asymmetry with
  v1". Running v1's parameters through v2's path would be a different method.
- **It has a temporal prior.** HMM Viterbi smoothing runs on each contiguous
  non-noise segment, per recording. No v2 arm has this, and decision #65 lists
  restoring it as one of the two candidate explanations for MoSeq's win.

**A known defect is preserved here deliberately.** ``compare._fit_hmm`` estimates
its transition matrix from labels concatenated across every recording, with noise
frames deleted rather than segmented around — so it counts a transition across
each of the ~3,845 recording seams *and* across every excised noise run. Its own
comment asserts the opposite. This is the one boundary leak in the codebase that
reaches a published v1 number. Fixing it would change this arm's output and fail
its verification gate, so it is reported and left alone; see docs/DECISIONS.md.
Smoothing is *applied* per recording, so only the fitted matrix is contaminated.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ..data.dataset import PoseDataset
from ..registry import SEGMENTERS
from .base import Segmentation, make_segmentation

#: v1's UMAP settings, from compare.py. `random_state=42` is what makes the arm
#: reproducible at all, so it is a default rather than a derived value.
UMAP_NEIGHBORS = 30
UMAP_MIN_DIST = 0.0
UMAP_RANDOM_STATE = 42


@SEGMENTERS.register("vieb_v1")
class ViebV1Segmenter:
    """UMAP + HDBSCAN + HMM smoothing, as ``compare.py --cluster`` runs it.

    Parameters:
      ``min_cluster_size``  HDBSCAN's, default 50.
      ``min_samples``       None resolves to ``max(10, min(100, mcs // 10))``.
      ``umap_dims``         UMAP output dimension, default 10.
      ``hdbscan_sample``    frames HDBSCAN is *fitted* on, default 300k.
      ``smooth``            run the HMM Viterbi pass, default True.
    """

    name = "vieb_v1"
    version = "1.0.0"

    def __init__(
        self,
        min_cluster_size: int = 50,
        min_samples: int | None = None,
        umap_dims: int = 10,
        hdbscan_sample: int = 300_000,
        umap_sample: int = 200_000,
        smooth: bool = True,
    ):
        self.min_cluster_size = int(min_cluster_size)
        self.min_samples = min_samples
        self.umap_dims = int(umap_dims)
        self.hdbscan_sample = int(hdbscan_sample)
        self.umap_sample = int(umap_sample)
        self.smooth = bool(smooth)

        self._labels: np.ndarray | None = None
        self._probs: np.ndarray | None = None
        self._report: dict = {}
        self._seed: int | None = None

    def get_params(self) -> dict:
        return {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "umap_dims": self.umap_dims,
            "hdbscan_sample": self.hdbscan_sample,
            "umap_sample": self.umap_sample,
            "smooth": self.smooth,
            "umap_neighbors": UMAP_NEIGHBORS,
            "umap_min_dist": UMAP_MIN_DIST,
            "umap_random_state": UMAP_RANDOM_STATE,
        }

    @property
    def effective_min_samples(self) -> int:
        """v1's rule, at compare.py:1734. Not a tunable — a derived default."""
        if self.min_samples is not None:
            return int(self.min_samples)
        return max(10, min(100, self.min_cluster_size // 10))

    def fit(self, X: np.ndarray, data: PoseDataset, *, seed: int = 0) -> None:
        import hdbscan as hdbscan_lib
        import umap
        from sklearn.preprocessing import StandardScaler

        import compare as v1

        self._seed = int(seed)
        X = np.asarray(X, dtype=np.float64)

        # 1. Standardize. v1 sets use_pca=False literally at compare.py:1625 —
        #    UMAP handles the reduction, so there is no PCA in this path.
        scaler = StandardScaler()
        pooled = scaler.fit_transform(X)

        # 2. UMAP, fit on a sample when the project is large.
        rng = np.random.default_rng(seed)
        n = pooled.shape[0]
        if n > self.umap_sample:
            fit_idx = np.sort(rng.choice(n, self.umap_sample, replace=False))
        else:
            fit_idx = np.arange(n)

        reducer = umap.UMAP(
            n_components=self.umap_dims,
            n_neighbors=UMAP_NEIGHBORS,
            min_dist=UMAP_MIN_DIST,
            random_state=UMAP_RANDOM_STATE,
        )
        reducer.fit(pooled[fit_idx])
        embedded = reducer.transform(pooled)

        # 3. HDBSCAN, fit on a sample and approximate_predict the rest —
        #    v1's own helper, so the sampling behaviour is identical.
        if n > self.hdbscan_sample:
            h_fit = np.sort(rng.choice(n, self.hdbscan_sample, replace=False))
            h_pred = np.setdiff1d(np.arange(n), h_fit, assume_unique=False)
        else:
            h_fit, h_pred = np.arange(n), np.array([], dtype=int)

        model, raw_labels, raw_probs = v1._fit_cpu_hdbscan_with_assignment(
            hdbscan_lib.HDBSCAN, embedded, h_fit, h_pred,
            self.min_cluster_size, self.effective_min_samples,
        )

        n_found = int(raw_labels.max()) + 1 if (raw_labels >= 0).any() else 0
        self._report = {
            "n_clusters_raw": n_found,
            "noise_frac_raw": float((raw_labels < 0).mean()),
            "min_samples_resolved": self.effective_min_samples,
            "umap_fit_frames": int(fit_idx.size),
            "hdbscan_fit_frames": int(h_fit.size),
        }

        # 4. HMM Viterbi smoothing, per recording. See the module docstring on
        #    what _fit_hmm's transition counting does across seams.
        labels = raw_labels.astype(np.int32)
        if self.smooth and n_found > 1:
            per_recording = [labels[sl] for _, sl in data.slices()]
            valid = np.concatenate(per_recording)
            valid = valid[valid >= 0]
            if valid.size:
                params = v1._fit_hmm(valid, n_found)
                labels = np.concatenate(
                    [v1._smooth_with_noise(seg, params) for seg in per_recording]
                ).astype(np.int32)
                self._report["smoothed"] = True

        self._labels = labels
        self._probs = np.asarray(raw_probs, dtype=np.float32)

    def predict(self, X: np.ndarray, data: PoseDataset) -> Segmentation:
        if self._labels is None:
            raise RuntimeError("fit() must be called before predict()")
        if self._labels.shape[0] != data.n_frames:
            raise ValueError(
                f"segmenter was fit on {self._labels.shape[0]} frames but was handed "
                f"a dataset with {data.n_frames}; refit rather than reuse"
            )
        return make_segmentation(
            self._labels, data,
            probabilities=self._probs, report=self._report, seed=self._seed,
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {"name": self.name, "version": self.version, "params": self.get_params(),
                 "labels": self._labels, "probs": self._probs,
                 "report": self._report, "seed": self._seed},
                fh,
            )

    @classmethod
    def load(cls, path: Path) -> "ViebV1Segmenter":
        with Path(path).open("rb") as fh:
            blob = pickle.load(fh)
        obj = cls(**{k: v for k, v in blob.get("params", {}).items()
                     if k in {"min_cluster_size", "min_samples", "umap_dims",
                              "hdbscan_sample", "umap_sample", "smooth"}})
        obj._labels = blob.get("labels")
        obj._probs = blob.get("probs")
        obj._report = blob.get("report", {})
        obj._seed = blob.get("seed")
        return obj
