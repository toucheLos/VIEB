"""The segmenter contract — one of the two slots the comparison varies.

Deliberately small. The harness owns loading, the representation, run-length
encoding into bouts, VUS-1 writing, and every metric. A segmenter contributes
exactly two things: how it is fit, and what state each frame is in.

**No segmenter gets its own scoring path.** Every arm that previously computed its
own statistics loses that code — that duplication is why the same question has been
answered differently in different places.

``fit`` and ``predict`` are separate so a segmenter can be fit on a subsample and
applied to everything; the HSMM needs this, and it is also how any arm gets a
train/test split without a bespoke code path.

Both take the ``PoseDataset`` even when the segmenter only needs ``X`` — some
methods need ``fps`` and the recording boundaries, and a uniform signature is worth
more than a minimal one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..data.dataset import UNASSIGNED, PoseDataset


@dataclass
class Segmentation:
    """What a segmenter returns: a state per frame, and how it got there."""

    frame_labels: np.ndarray
    """``(n_frames,)`` int32. ``-1`` means unassigned — HDBSCAN noise, a Koopman
    separatrix, a frame the method declined to label. It is the absence of a state,
    not a state."""

    n_states: int
    """Number of distinct assigned states. Excludes ``-1``."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Method-specific output that is not a per-frame label: HDBSCAN soft
    probabilities and backend, Koopman eigenvalues and separatrix labels, the slow
    states of a transfer operator. Recorded in the manifest, never used for ranking
    — ranking is on the metrics the harness computes identically for every arm."""

    def __post_init__(self) -> None:
        self.frame_labels = np.asarray(self.frame_labels)

    @property
    def unassigned_frac(self) -> float:
        if self.frame_labels.size == 0:
            return 0.0
        return float((self.frame_labels < 0).mean())

    def state_counts(self) -> np.ndarray:
        """Frames per state id, index-aligned to state id. Excludes ``-1``."""
        assigned = self.frame_labels[self.frame_labels >= 0]
        if assigned.size == 0:
            return np.zeros(0, dtype=np.int64)
        return np.bincount(assigned)


@runtime_checkable
class Segmenter(Protocol):
    """The whole contract."""

    name: str
    version: str

    def fit(self, X: np.ndarray, data: PoseDataset, *, seed: int) -> None:
        """Fit on ``X``, ``(n_frames, d)``, from a FROZEN representation."""
        ...

    def predict(self, X: np.ndarray, data: PoseDataset) -> Segmentation:
        """Return a ``Segmentation`` with one label per frame of ``data``."""
        ...

    def get_params(self) -> dict:
        """Every parameter that affects the result, for the config hash."""
        ...

    def save(self, path: Path) -> None:
        ...

    @classmethod
    def load(cls, path: Path) -> "Segmenter":
        ...


def validate_labels(labels: np.ndarray, data: PoseDataset) -> np.ndarray:
    """Check a segmenter's per-frame output against the contract.

    Called by the harness on every segmenter's output, so a violation surfaces at
    the boundary of the offending arm rather than three stages downstream in a
    metric that quietly produces a number.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {labels.shape}")
    if labels.shape[0] != data.n_frames:
        raise ValueError(
            f"labels has {labels.shape[0]} frames but the dataset has {data.n_frames}"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels must be an integer dtype, got {labels.dtype}")
    if labels.min(initial=0) < UNASSIGNED:
        raise ValueError(
            f"labels contain {int(labels.min())}; the only negative value "
            f"permitted is {UNASSIGNED} (unassigned)"
        )
    return labels.astype(np.int32, copy=False)


def make_segmentation(labels: np.ndarray, data: PoseDataset, **extra) -> Segmentation:
    """Validate labels and wrap them, deriving ``n_states`` the one same way.

    ``n_states`` counts *distinct assigned ids*, not ``max + 1``. A method that
    emits ids 0, 1, 4 has three states; counting five would inflate every
    per-state average by the two that never occur.
    """
    labels = validate_labels(labels, data)
    assigned = labels[labels >= 0]
    n_states = int(np.unique(assigned).size) if assigned.size else 0
    return Segmentation(frame_labels=labels, n_states=n_states, extra=dict(extra))
