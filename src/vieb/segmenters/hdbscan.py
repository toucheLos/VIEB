"""HDBSCAN arm — a state is a density peak in the delay embedding.

Thin adapter over ``vieb_v2.representation.cluster``. The v2 implementation is
kept as-is (including its GPU/CPU backend handling and its subsample-then-
approximate-predict strategy for 22.4M frames); this file only maps it onto the
``Segmenter`` contract and strips the scoring it used to do for itself.

This is the arm decision #65 measured: on ``pca`` it returns 99.2% of clustered
frames in one state and a retrieval effect of exactly 0.000. It stays a scored
comparison arm — a measured null is part of the result — but it is no longer any
pipeline's default.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ..data.dataset import PoseDataset
from ..registry import SEGMENTERS
from ..representations.delay_embed import delay_embed, scatter_labels
from .base import Segmentation, make_segmentation


@SEGMENTERS.register("hdbscan")
class HDBSCANSegmenter:
    """Density clustering on a delay embedding.

    Parameters (all optional):
      ``min_cluster_size``  HDBSCAN's, default 50 — this sets the state count.
      ``min_samples``       None = HDBSCAN's own default.
      ``embed_k``           delay-embedding depth in frames, default 1 (none).
      ``embed_seconds``     depth in seconds; converted via fps, overrides embed_k.
      ``embed_stride``      spacing between stacked frames, default 1.
      ``fit_sample``        cap on points the clusterer is *fitted* on (300k).
      ``device``            auto | cpu | gpu.
    """

    name = "hdbscan"
    version = "2.0.0"

    def __init__(
        self,
        min_cluster_size: int = 50,
        min_samples: int | None = None,
        embed_k: int = 1,
        embed_seconds: float | None = None,
        embed_stride: int = 1,
        fit_sample: int = 300_000,
        device: str = "auto",
    ):
        self.min_cluster_size = int(min_cluster_size)
        self.min_samples = min_samples
        self.embed_k = int(embed_k)
        self.embed_seconds = embed_seconds
        self.embed_stride = int(embed_stride)
        self.fit_sample = fit_sample
        self.device = device

        self._labels: np.ndarray | None = None
        self._probs: np.ndarray | None = None
        self._backend: str | None = None
        self._seed: int | None = None

    # -- contract ---------------------------------------------------------

    def get_params(self) -> dict:
        return {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "embed_k": self.embed_k,
            "embed_seconds": self.embed_seconds,
            "embed_stride": self.embed_stride,
            "fit_sample": self.fit_sample,
            "device": self.device,
        }

    def fit(self, X: np.ndarray, data: PoseDataset, *, seed: int = 0) -> None:
        from vieb_v2.representation import cluster as v2cluster

        self._seed = int(seed)
        k, stride = self._embed_params(data)

        points, rec_idx, frm_idx = delay_embed(X, data, k, stride, return_index=True)

        labels, probs, backend = v2cluster.cluster(
            points,
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            use_gpu=(self.device == "gpu"),
            sample=self.fit_sample,
            seed=int(seed),
            return_backend=True,
        )
        self._backend = backend

        # v2 returns labels per delay vector; the contract is per frame.
        self._labels = scatter_labels(labels, rec_idx, frm_idx, data)
        self._probs = scatter_labels(
            (probs * 1000).astype(np.int32), rec_idx, frm_idx, data, fill=0
        ).astype(np.float32) / 1000.0

    def predict(self, X: np.ndarray, data: PoseDataset) -> Segmentation:
        """Return the fitted labels.

        HDBSCAN has no cheap out-of-sample rule beyond ``approximate_predict``,
        which v2 already applies inside ``fit`` to every point past the fit
        subsample. So ``fit`` labels everything and ``predict`` returns it; the
        split exists for methods like the HSMM that genuinely fit on a subsample.
        """
        if self._labels is None:
            raise RuntimeError("fit() must be called before predict()")
        if self._labels.shape[0] != data.n_frames:
            raise ValueError(
                f"segmenter was fit on {self._labels.shape[0]} frames but was handed "
                f"a dataset with {data.n_frames}; refit rather than reuse"
            )
        return make_segmentation(
            self._labels,
            data,
            probabilities=self._probs,
            backend=self._backend,
            seed=self._seed,
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "name": self.name,
                    "version": self.version,
                    "params": self.get_params(),
                    "labels": self._labels,
                    "probs": self._probs,
                    "backend": self._backend,
                    "seed": self._seed,
                },
                fh,
            )

    @classmethod
    def load(cls, path: Path) -> "HDBSCANSegmenter":
        with Path(path).open("rb") as fh:
            blob = pickle.load(fh)
        obj = cls(**blob.get("params", {}))
        obj._labels = blob.get("labels")
        obj._probs = blob.get("probs")
        obj._backend = blob.get("backend")
        obj._seed = blob.get("seed")
        return obj

    # -- internals --------------------------------------------------------

    def _embed_params(self, data: PoseDataset) -> tuple[int, int]:
        """Resolve the embedding depth, preferring the seconds-valued form.

        Every temporal parameter is specified in seconds and converted via fps, so
        the same config means the same thing on data recorded at a different frame
        rate — Luna is 30 fps and Spence is 250.
        """
        if self.embed_seconds is not None:
            k = data.seconds_to_frames(float(self.embed_seconds))
        else:
            k = int(self.embed_k)
        return k, int(self.embed_stride)
