"""The postural representations: identity, pca, diffusion.

All three share the same first two stages — drop the noisy keypoints, then
egocentrically align every recording against **one shared reference** — and differ
only in what they do afterwards. That shared prefix is why they are one module:
aligning each recording to its own reference would put different animals in
differently-rotated frames and no downstream coordinate would be comparable
across them, and it is the kind of thing that gets re-derived subtly differently
when it is written out three times.

These are thin adapters over ``vieb_v2.representation``; the implementations are
kept as-is. What is new is only that they are named, hashed, and composable with
any segmenter.

**All three are purely postural.** ``align_session`` subtracts the per-frame
centroid and applies a per-frame rotation, so translation and heading are gone by
construction (decision #60) — and delay embedding cannot recover them, because
they were subtracted before measurement rather than merely differenced away.
Freezing is defined by near-zero locomotion, so this is the representation-side
half of why MoSeq beats every VIEB arm. It is preserved exactly, not repaired,
because repairing it here would change what the ported arms compute.
"""

from __future__ import annotations

import numpy as np

from ..data.dataset import PoseDataset
from ..registry import REPRESENTATIONS
from .base import BaseRepresentation


def _sessions(data: PoseDataset) -> list[tuple[np.ndarray, np.ndarray | None]]:
    """``PoseDataset`` -> the list-of-(pose, conf) shape v2 works in.

    A list, never a concatenated array: v2's own modules raise ``TypeError`` on
    an ndarray precisely so that no lag can cross a recording boundary.
    """
    out = []
    for _, sl in data.slices():
        conf = None if data.confidence is None else data.confidence[sl]
        out.append((np.asarray(data.keypoints[sl], dtype=np.float64), conf))
    return out


class _PosturalRepresentation(BaseRepresentation):
    """Shared prefix: select keypoints, then align against one shared reference."""

    #: Keypoints dropped before anything else sees them. ``tail_tip`` is the
    #: noisiest point in the Luna model and v2 has always dropped it; changing
    #: this changes the aligned space and therefore every arm built on it.
    drop_keypoints: tuple[str, ...] = ("tail_tip",)

    def __init__(self, *, align_iters: int = 5, drop: tuple[str, ...] | None = None):
        self.align_iters = int(align_iters)
        if drop is not None:
            self.drop_keypoints = tuple(drop)
        self.report_: dict = {}

    def _aligned(self, data: PoseDataset) -> list[np.ndarray]:
        from vieb_v2.representation import align as v2align
        from vieb_v2.representation import keypoints as v2keypoints

        sessions, kept_names = [], None
        for pose, conf in _sessions(data):
            pose, conf, kept = v2keypoints.select(
                pose,
                None if conf is None else np.asarray(conf, dtype=np.float64),
                list(data.keypoint_names),
            )
            kept_names = kept
            sessions.append((pose, conf))

        aligned, reference = v2align.align_all(sessions, n_iter=self.align_iters)
        self.report_["kept_keypoints"] = list(kept_names or [])
        self.report_["dropped_keypoints"] = [
            b for b in data.keypoint_names if b not in (kept_names or [])
        ]
        self.report_["reference_shape"] = list(np.shape(reference))
        return aligned

    def get_params(self) -> dict:
        return {
            "align_iters": self.align_iters,
            "drop_keypoints": list(self.drop_keypoints),
        }


@REPRESENTATIONS.register("identity")
class IdentityRepresentation(_PosturalRepresentation):
    """Aligned pose, flattened to ``(n_frames, n_kept_keypoints * 2)``.

    The ``pose`` representation of decision #65's table, and what "identity
    (aligned pose)" means for the MoSeq arm: no dimensionality reduction, so a
    segmenter sees the pose coordinates themselves.
    """

    name = "identity"

    def fit_transform(self, data: PoseDataset) -> np.ndarray:
        aligned = self._aligned(data)
        X = np.concatenate([a.reshape(a.shape[0], -1) for a in aligned], axis=0)
        return self._check_output(X, data)

    @property
    def channel_names(self) -> list[str]:
        kept = self.report_.get("kept_keypoints", [])
        return [f"{bp}_{axis}" for bp in kept for axis in ("x", "y")]


@REPRESENTATIONS.register("pca")
class PCARepresentation(_PosturalRepresentation):
    """Pooled PCA on aligned pose, 95% variance — 9 components on Luna.

    PCA maximizes *variance*: its coordinates are the directions in which posture
    varies most, and nothing about them is dynamical. Decision #65 attributes
    ``pca-HDBSCAN``'s exactly-zero effect to this — in a purely postural space the
    density has a single overwhelming mode.
    """

    name = "pca"

    def __init__(self, var_threshold: float = 0.95, max_components: int | None = None,
                 **kw):
        super().__init__(**kw)
        self.var_threshold = float(var_threshold)
        self.max_components = max_components

    def fit_transform(self, data: PoseDataset) -> np.ndarray:
        from vieb_v2.representation.pooled_pca import PooledPCA

        aligned = self._aligned(data)
        model = PooledPCA(
            var_threshold=self.var_threshold, max_components=self.max_components
        )
        scores = model.fit_transform(aligned)
        self.report_["latent"] = model.spectrum_report()
        self.model_ = model
        X = np.concatenate([np.asarray(s) for s in scores], axis=0)
        return self._check_output(X, data)

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            "var_threshold": self.var_threshold,
            "max_components": self.max_components,
        }


@REPRESENTATIONS.register("diffusion")
class DiffusionRepresentation(_PosturalRepresentation):
    """Landmark diffusion maps with Nystrom extension — 8 components on Luna.

    Diffusion map coordinates approximate the slowest-relaxing directions of a
    diffusion on the pose manifold: they are ordered by *relaxation time*, not
    variance. That difference is what decision #65 credits for
    ``diffusion-Koopman`` being the only VIEB arm within 2x of MoSeq.

    The defaults reproduce that run: alpha=1, 3,000 landmarks, and the
    ``min_neighbor_mass`` pruning of decision #54 — without which 20k frames
    collapse onto 29 distinct embedded points.
    """

    name = "diffusion"

    def __init__(self, n_components: int = 8, alpha: float = 1.0, epsilon="auto",
                 diffusion_time: int = 1, n_landmarks: int = 3000,
                 random_state: int = 0, use_gpu: bool = False,
                 min_neighbor_mass: float | None = None, **kw):
        super().__init__(**kw)
        self.n_components = int(n_components)
        self.alpha = float(alpha)
        self.epsilon = epsilon
        self.diffusion_time = diffusion_time
        self.n_landmarks = int(n_landmarks)
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.min_neighbor_mass = min_neighbor_mass

    def fit_transform(self, data: PoseDataset) -> np.ndarray:
        from vieb_v2.representation.diffusion import DiffusionMap

        aligned = self._aligned(data)
        model = DiffusionMap(
            n_components=self.n_components,
            alpha=self.alpha,
            epsilon=self.epsilon,
            diffusion_time=self.diffusion_time,
            n_landmarks=self.n_landmarks,
            random_state=self.random_state,
            use_gpu=self.use_gpu,
            min_neighbor_mass=self.min_neighbor_mass,
        )
        scores = model.fit_transform(aligned)
        self.report_["latent"] = model.spectrum_report()
        self.model_ = model
        X = np.concatenate([np.asarray(s) for s in scores], axis=0)
        return self._check_output(X, data)

    def get_params(self) -> dict:
        return {
            **super().get_params(),
            "n_components": self.n_components,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "diffusion_time": self.diffusion_time,
            "n_landmarks": self.n_landmarks,
            "random_state": self.random_state,
            "min_neighbor_mass": self.min_neighbor_mass,
        }
