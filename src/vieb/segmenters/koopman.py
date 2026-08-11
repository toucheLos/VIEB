"""Koopman arm — a state is a basin of attraction of local affine operators.

Thin adapter over ``vieb_v2.representation.koopman.extract_topology``. No
clustering is run: the states *are* the basins, found by fitting a local linear
operator in each of ``n_regions`` Voronoi cells, classifying its modes, and
merging cells whose flow leads to the same attractor.

Two things worth knowing when reading its output against HDBSCAN's:

- **``-1`` means something different here.** HDBSCAN's ``-1`` is "unclustered";
  Koopman's is "near a separatrix", i.e. a transition between basins (decision
  #57). Both are ``UNASSIGNED`` under the contract, but they are not the same
  quantity and a noise fraction is not comparable across the two families.
- **The state count is only partly an output.** Sweeping ``n_regions`` gives
  n proportional to r^1.04 in PCA space — one attractor per region, so the count
  *is* the parameter wearing a different name — and r^0.71 in diffusion space,
  where genuine merging happens. Neither is parameter-free.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ..data.dataset import PoseDataset
from ..registry import SEGMENTERS
from ..representations.delay_embed import delay_embed, scatter_labels
from .base import Segmentation, make_segmentation


@SEGMENTERS.register("koopman")
class KoopmanSegmenter:
    """Attractor-topology decomposition of local Koopman operators.

    Defaults reproduce decision #65's ``koopman_*`` runs: 48 regions, kNN 12,
    v_percentile 25, coherence_tol 0.5, seed 0, on a 4-lag delay embedding with
    stride 2.
    """

    name = "koopman"
    version = "2.0.0"

    def __init__(
        self,
        n_regions: int = 48,
        v_percentile: float = 25.0,
        unit_tol: float = 0.05,
        knn: int = 12,
        partition_method: str = "kmeans",
        min_edge_frac: float = 0.05,
        min_edge_count: int = 3,
        cycle_min_regions: int = 3,
        coherence_tol: float = 0.5,
        min_attractor_frac: float = 0.001,
        plausible_hz: tuple = (0.5, 12.0),
        knn_sample: int = 1_000_000,
        embed_k: int = 4,
        embed_seconds: float | None = None,
        embed_stride: int = 2,
    ):
        self.n_regions = int(n_regions)
        self.v_percentile = float(v_percentile)
        self.unit_tol = float(unit_tol)
        self.knn = int(knn)
        self.partition_method = partition_method
        self.min_edge_frac = float(min_edge_frac)
        self.min_edge_count = int(min_edge_count)
        self.cycle_min_regions = int(cycle_min_regions)
        self.coherence_tol = float(coherence_tol)
        self.min_attractor_frac = float(min_attractor_frac)
        self.plausible_hz = tuple(plausible_hz)
        self.knn_sample = int(knn_sample)
        self.embed_k = int(embed_k)
        self.embed_seconds = embed_seconds
        self.embed_stride = int(embed_stride)

        self._labels: np.ndarray | None = None
        self._report: dict = {}
        self._seed: int | None = None

    def get_params(self) -> dict:
        return {
            "n_regions": self.n_regions,
            "v_percentile": self.v_percentile,
            "unit_tol": self.unit_tol,
            "knn": self.knn,
            "partition_method": self.partition_method,
            "min_edge_frac": self.min_edge_frac,
            "min_edge_count": self.min_edge_count,
            "cycle_min_regions": self.cycle_min_regions,
            "coherence_tol": self.coherence_tol,
            "min_attractor_frac": self.min_attractor_frac,
            "plausible_hz": list(self.plausible_hz),
            "knn_sample": self.knn_sample,
            "embed_k": self.embed_k,
            "embed_seconds": self.embed_seconds,
            "embed_stride": self.embed_stride,
        }

    def fit(self, X: np.ndarray, data: PoseDataset, *, seed: int = 0) -> None:
        from vieb_v2.representation import koopman as v2koopman

        self._seed = int(seed)
        k = (
            data.seconds_to_frames(float(self.embed_seconds))
            if self.embed_seconds is not None else self.embed_k
        )
        points, rec_idx, frm_idx = delay_embed(
            X, data, k, self.embed_stride, return_index=True
        )

        # extract_topology needs a list of per-recording arrays — it forms
        # forward-difference snapshot pairs and raises on a concatenated array so
        # a pair cannot straddle a seam.
        sessions = [points[rec_idx == r] for r in range(data.n_recordings)
                    if np.any(rec_idx == r)]

        result = v2koopman.extract_topology(
            sessions,
            fps=data.fps,
            n_regions=self.n_regions,
            v_percentile=self.v_percentile,
            unit_tol=self.unit_tol,
            knn=self.knn,
            seed=int(seed),
            partition_method=self.partition_method,
            min_edge_frac=self.min_edge_frac,
            min_edge_count=self.min_edge_count,
            cycle_min_regions=self.cycle_min_regions,
            coherence_tol=self.coherence_tol,
            min_attractor_frac=self.min_attractor_frac,
            plausible_hz=self.plausible_hz,
            knn_sample=self.knn_sample,
        )
        self._report = result.get("report", {})
        self._labels = scatter_labels(
            np.asarray(result["labels"], dtype=np.int32), rec_idx, frm_idx, data
        )

    def predict(self, X: np.ndarray, data: PoseDataset) -> Segmentation:
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
            report=self._report,
            seed=self._seed,
            unassigned_means="separatrix",
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {"name": self.name, "version": self.version, "params": self.get_params(),
                 "labels": self._labels, "report": self._report, "seed": self._seed},
                fh,
            )

    @classmethod
    def load(cls, path: Path) -> "KoopmanSegmenter":
        with Path(path).open("rb") as fh:
            blob = pickle.load(fh)
        obj = cls(**blob.get("params", {}))
        obj._labels = blob.get("labels")
        obj._report = blob.get("report", {})
        obj._seed = blob.get("seed")
        return obj
