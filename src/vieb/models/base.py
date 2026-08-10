"""The contract every model in the bakeoff implements.

This is deliberately small. The harness owns loading, the representation,
run-length encoding into bouts, VUS-1 writing, and every metric. A model
contributes exactly two things: how it is fit, and what state each frame is in.

No model gets its own scoring path. Every arm that previously computed its own
statistics loses that code — that duplication is why the same question has been
answered differently in different places.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

UNASSIGNED = -1
"""Frame label meaning "no state". HDBSCAN noise, unfilled gaps, masked frames."""


@dataclass(frozen=True)
class RepresentationMeta:
    """Everything a model may know about ``X`` beyond the array itself.

    Frozen because a model must never be able to reshape the representation it
    was handed. If a model needs something not on here, that is a signal the
    thing belongs in the representation build, not in the model.
    """

    name: str
    """Representation name, e.g. ``pose_plus_locomotor``."""

    repr_hash: str
    """``sha256:...`` of the representation artifact. Becomes a results column."""

    fps: float
    """Frames per second. Every temporal parameter is specified in seconds and
    converted through this — never in raw frames."""

    recording_ids: list[str]
    """One id per recording, in the order the recordings appear in ``X``.
    Already normalized (trailing ``DLC_*`` and extension stripped)."""

    boundaries: np.ndarray
    """``(n_recordings + 1,)`` int64 index array such that recording ``k``
    occupies ``X[boundaries[k]:boundaries[k + 1]]``. This is the authority on
    where recordings begin and end."""

    channel_names: list[str] = field(default_factory=list)
    """Column names of ``X``, length ``n_features``."""

    dataset: str = ""
    """Dataset name, e.g. ``luna``."""

    def __post_init__(self) -> None:
        b = np.asarray(self.boundaries)
        if b.ndim != 1 or b.size < 2:
            raise ValueError(f"boundaries must be 1-D with >= 2 entries, got shape {b.shape}")
        if b[0] != 0:
            raise ValueError(f"boundaries must start at 0, got {b[0]}")
        if np.any(np.diff(b) <= 0):
            bad = int(np.argmin(np.diff(b)))
            raise ValueError(
                f"boundaries must be strictly increasing; recording {bad} "
                f"({self.recording_ids[bad] if bad < len(self.recording_ids) else '?'}) is empty or negative-length"
            )
        if len(self.recording_ids) != b.size - 1:
            raise ValueError(
                f"{len(self.recording_ids)} recording_ids but boundaries imply {b.size - 1} recordings"
            )

    @property
    def n_recordings(self) -> int:
        return len(self.recording_ids)

    @property
    def n_frames(self) -> int:
        return int(self.boundaries[-1])

    def slices(self):
        """Yield ``(recording_id, slice)`` per recording.

        Use this for anything that must not cross a recording boundary —
        derivatives, lags, delay embeddings, transition counts. A window
        straddling two recordings is silently garbage, and this is the only
        supported way to avoid it.
        """
        b = np.asarray(self.boundaries)
        for k, rid in enumerate(self.recording_ids):
            yield rid, slice(int(b[k]), int(b[k + 1]))

    def seconds_to_frames(self, seconds: float) -> int:
        """Convert a duration in seconds to whole frames, rounding to nearest.

        Always at least 1 frame, so a sub-frame parameter degrades to the
        smallest representable lag rather than to zero.
        """
        return max(1, int(round(seconds * self.fps)))

    def frames_to_seconds(self, frames: int | np.ndarray):
        return np.asarray(frames) / self.fps


@runtime_checkable
class Model(Protocol):
    """The whole contract.

    ``fit`` and ``label`` are separate so a model can be fit on a subsample and
    applied to everything — the HSMM needs this, and it is also how any arm gets
    a train/test split without a bespoke code path.
    """

    name: str
    version: str

    def fit(self, X: np.ndarray, meta: RepresentationMeta, cfg: dict) -> None:
        """Fit on ``X``, ``(n_frames, n_features)``, from a FROZEN artifact."""
        ...

    def label(self, X: np.ndarray, meta: RepresentationMeta) -> np.ndarray:
        """Return ``(n_frames,)`` int32 state ids. ``-1`` means unassigned."""
        ...

    def save(self, path: Path) -> None:
        ...

    @classmethod
    def load(cls, path: Path) -> "Model":
        ...


def validate_labels(labels: np.ndarray, meta: RepresentationMeta) -> np.ndarray:
    """Check a ``label()`` return value against the contract.

    Called by the harness on every model's output, so a violation surfaces at
    the boundary of the offending model rather than three stages downstream in a
    metric that quietly produces a number.
    """
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError(f"labels must be 1-D, got shape {labels.shape}")
    if labels.shape[0] != meta.n_frames:
        raise ValueError(
            f"labels has {labels.shape[0]} frames but representation "
            f"{meta.name} has {meta.n_frames}"
        )
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"labels must be an integer dtype, got {labels.dtype}")
    if labels.min(initial=0) < UNASSIGNED:
        raise ValueError(
            f"labels contain {int(labels.min())}; the only negative value "
            f"permitted is {UNASSIGNED} (unassigned)"
        )
    return labels.astype(np.int32, copy=False)
