"""HDBSCAN arm — a state is a density peak in the delay embedding.

Thin adapter over ``vieb_v2.representation.cluster``. The v2 implementation is
kept as-is (including its GPU/CPU backend handling and its subsample-then-
approximate-predict strategy for 22.4M frames); this file only maps it onto the
``Model`` contract and strips the scoring it used to do for itself.

This is the arm §1h reproduces: pca-HDBSCAN, max |shift| ~= 0.000 at retrieval,
with 99.2% of frames in one state.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ..representation.embed import delay_embed, scatter_labels
from .base import RepresentationMeta, validate_labels
from .registry import register


@register("hdbscan")
class HDBSCANModel:
    """Density clustering on a delay embedding.

    Config keys (all optional):
      ``min_cluster_size``  HDBSCAN's, default 50 — this sets the state count.
      ``min_samples``       None = HDBSCAN's own default.
      ``embed_k``           delay-embedding depth in frames, default 1 (none).
      ``embed_seconds``     depth in seconds; converted via fps, overrides embed_k.
      ``embed_stride``      spacing between stacked frames, default 1.
      ``fit_sample``        cap on points the clusterer is *fitted* on (300k).
      ``device``            auto | cpu | gpu.
      ``seed``              default 0.
    """

    name = "hdbscan"
    version = "2.0.0"

    def __init__(self, **kwargs):
        self.cfg: dict = dict(kwargs)
        self._model = None
        self._backend: str | None = None
        self._labels: np.ndarray | None = None
        self._probs: np.ndarray | None = None
        self.n_states: int | None = None

    # -- contract ---------------------------------------------------------

    def fit(self, X: np.ndarray, meta: RepresentationMeta, cfg: dict) -> None:
        from vieb_v2.representation import cluster as v2cluster

        self.cfg = {**self.cfg, **(cfg or {})}
        k, stride = self._embed_params(meta)

        points, rec_idx, frm_idx = delay_embed(
            X, meta, k, stride, return_index=True
        )

        device = self.cfg.get("device", "auto")
        labels, probs, backend = v2cluster.cluster(
            points,
            min_cluster_size=int(self.cfg.get("min_cluster_size", 50)),
            min_samples=self.cfg.get("min_samples"),
            use_gpu=(device == "gpu"),
            sample=self.cfg.get("fit_sample", 300_000),
            seed=int(self.cfg.get("seed", 0)),
            return_backend=True,
        )
        self._backend = backend

        # v2 returns labels per delay vector; the contract is per frame.
        frame_labels = scatter_labels(labels, rec_idx, frm_idx, meta)
        self._labels = validate_labels(frame_labels, meta)
        self._probs = scatter_labels(
            (probs * 1000).astype(np.int32), rec_idx, frm_idx, meta, fill=0
        ).astype(np.float32) / 1000.0
        self.n_states = int(self._labels.max()) + 1 if (self._labels >= 0).any() else 0

    def label(self, X: np.ndarray, meta: RepresentationMeta) -> np.ndarray:
        """Return the fitted labels.

        HDBSCAN has no cheap out-of-sample rule beyond ``approximate_predict``,
        which v2 already applies inside ``fit`` to every point past the fit
        subsample. So ``fit`` labels everything and ``label`` returns it; the
        split exists for models like the HSMM that genuinely fit on a subsample.
        """
        if self._labels is None:
            raise RuntimeError("fit() must be called before label()")
        if self._labels.shape[0] != meta.n_frames:
            raise ValueError(
                f"model was fit on {self._labels.shape[0]} frames but was handed "
                f"a representation with {meta.n_frames}; refit rather than reuse"
            )
        return self._labels

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {
                    "name": self.name,
                    "version": self.version,
                    "cfg": self.cfg,
                    "labels": self._labels,
                    "probs": self._probs,
                    "backend": self._backend,
                    "n_states": self.n_states,
                },
                fh,
            )

    @classmethod
    def load(cls, path: Path) -> "HDBSCANModel":
        with Path(path).open("rb") as fh:
            blob = pickle.load(fh)
        obj = cls(**blob.get("cfg", {}))
        obj._labels = blob.get("labels")
        obj._probs = blob.get("probs")
        obj._backend = blob.get("backend")
        obj.n_states = blob.get("n_states")
        return obj

    # -- internals --------------------------------------------------------

    def _embed_params(self, meta: RepresentationMeta) -> tuple[int, int]:
        """Resolve the embedding depth, preferring the seconds-valued form.

        §2.4: all temporal parameters are specified in seconds and converted via
        fps, so the same config means the same thing on data recorded at a
        different frame rate.
        """
        if "embed_seconds" in self.cfg and self.cfg["embed_seconds"] is not None:
            k = meta.seconds_to_frames(float(self.cfg["embed_seconds"]))
        else:
            k = int(self.cfg.get("embed_k", 1))
        return k, int(self.cfg.get("embed_stride", 1))

    def diagnostics(self) -> dict:
        """Reported into the manifest, not used for ranking."""
        if self._labels is None:
            return {}
        assigned = self._labels[self._labels >= 0]
        if assigned.size == 0:
            return {"n_states": 0, "unassigned_frac": 1.0, "backend": self._backend}
        counts = np.bincount(assigned)
        return {
            "n_states": int(counts.size),
            "unassigned_frac": float((self._labels < 0).mean()),
            "largest_state_frac": float(counts.max() / self._labels.size),
            "backend": self._backend,
        }
