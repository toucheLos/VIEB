"""The representation contract — the other of the two slots.

Keeping representation as a separate, frozen axis is what makes the comparison
fair *and* what makes the results table multiply: a new representation adds a
column across every segmenter rather than one more unattributable row. It is also
the fix for the known confound — MoSeq's win is currently unattributable because it
differs from every VIEB arm in representation *and* algorithm at once.

A representation turns a ``PoseDataset`` into ``(n_frames, d)``. It must produce
one row per frame, so that a label is a label for a frame and every arm's output
joins on the same index. Methods that consume windows (delay embedding, wavelets)
do the windowing themselves and map back — via ``PoseDataset.valid_windows`` and
``delay_embed``/``scatter_labels``, never by hand.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ..data.dataset import PoseDataset
from ..paths import config_hash


@runtime_checkable
class Representation(Protocol):
    """The whole contract."""

    name: str

    def fit_transform(self, data: PoseDataset) -> np.ndarray:
        """Return ``(n_frames, d)`` float32/float64, one row per frame of ``data``."""
        ...

    def get_params(self) -> dict:
        """Every parameter that affects the result, for ``repr_hash``."""
        ...


class BaseRepresentation:
    """Optional base supplying the bookkeeping every representation needs.

    Subclasses set ``name``, implement ``fit_transform``, and override
    ``get_params``. Nothing requires inheriting from this — ``Representation`` is a
    Protocol — but the hash and the output check should not be reimplemented per arm.
    """

    name: str = "base"

    def get_params(self) -> dict:
        return {}

    @property
    def repr_hash(self) -> str:
        """``sha256:...`` over name and params.

        This is the column that lets the comparison table flag a row whose
        representation disagrees with its config, rather than silently comparing
        two arms that were built on different inputs.
        """
        return config_hash({"representation": self.name, **self.get_params()})

    @property
    def channel_names(self) -> list[str]:
        """Column names of the output. Empty when the representation's dimensions
        are not individually meaningful (PCA components, diffusion coordinates)."""
        return []

    def _check_output(self, X: np.ndarray, data: PoseDataset) -> np.ndarray:
        """One row per frame, 2-D, finite-shaped. Called by the harness.

        A representation that silently returns fewer rows than frames — because it
        dropped a lead-in window, say — would shift every later label onto the wrong
        frame, and the resulting per-animal statistics would still look plausible.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(
                f"representation {self.name!r} must return (n_frames, d), got "
                f"shape {X.shape}"
            )
        if X.shape[0] != data.n_frames:
            raise ValueError(
                f"representation {self.name!r} returned {X.shape[0]} rows but the "
                f"dataset has {data.n_frames} frames. A representation must emit one "
                f"row per frame; map windowed output back with scatter_labels rather "
                f"than returning a shorter array."
            )
        return X
