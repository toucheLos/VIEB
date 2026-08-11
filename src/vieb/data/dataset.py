"""``PoseDataset`` — the one thing every representation and segmenter is handed.

This is the authority on where recordings begin and end. Every derivative, lag,
delay embedding and transition count in the codebase goes through ``slices()`` or
``valid_windows()``; a window that straddles two recordings is silently garbage —
it produces a point, not an error — so there is exactly one place that knows the
seams and everything else asks it.

It also owns ``fps``. Every temporal parameter in the codebase is specified in
seconds and converted here, because Luna is 30 fps and Spence is 250 fps and a
hardcoded frame count is an 8x different real-world window between the two rigs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np
import pandas as pd

UNASSIGNED = -1
"""Frame label meaning "no state". HDBSCAN noise, unfilled gaps, masked frames."""


@dataclass
class PoseDataset:
    """Pose keypoints for a set of recordings, concatenated along the frame axis.

    ``keypoints`` is one array for all recordings; ``recording_index`` says which
    recording each frame belongs to. The frames of a recording are contiguous and
    recordings appear in ascending order — that is validated, not assumed, because
    every boundary-safe operation below relies on it.
    """

    keypoints: np.ndarray
    """``(n_frames, n_keypoints, 2)`` float. May contain NaN for missing keypoints."""

    recording_index: np.ndarray
    """``(n_frames,)`` int. The recording each frame belongs to, as an index into
    ``recording_ids``. Must be non-decreasing and contiguous."""

    recording_ids: list[str]
    """One id per recording, index-aligned with ``recording_index``. Already
    normalized (trailing ``DLC_*`` and extension stripped) — see
    ``vieb.io.vus1.normalize_recording_id``."""

    keypoint_names: list[str]
    """Length ``n_keypoints``, in the order they appear in ``keypoints``."""

    fps: float
    """Frames per second. Every temporal parameter is specified in seconds and
    converted through this — never in raw frames."""

    metadata: pd.DataFrame = field(default_factory=pd.DataFrame)
    """One row per recording: ``recording_id, animal_id, context, day``. May be
    empty; representations and segmenters must not require it."""

    confidence: np.ndarray | None = None
    """``(n_frames, n_keypoints)`` float, or None when the source carried none.
    Not part of §3's contract but several arms weight by it, and dropping it here
    would force them to re-read the pose files."""

    dataset: str = ""
    """Dataset name, e.g. ``luna``. Becomes a results column."""

    _boundaries: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        kp = np.asarray(self.keypoints)
        if kp.ndim != 3 or kp.shape[2] != 2:
            raise ValueError(
                f"keypoints must be (n_frames, n_keypoints, 2), got shape {kp.shape}"
            )
        if len(self.keypoint_names) != kp.shape[1]:
            raise ValueError(
                f"{len(self.keypoint_names)} keypoint_names but keypoints has "
                f"{kp.shape[1]} keypoints"
            )

        idx = np.asarray(self.recording_index)
        if idx.ndim != 1:
            raise ValueError(f"recording_index must be 1-D, got shape {idx.shape}")
        if idx.shape[0] != kp.shape[0]:
            raise ValueError(
                f"recording_index has {idx.shape[0]} entries but keypoints has "
                f"{kp.shape[0]} frames"
            )
        if idx.size == 0:
            raise ValueError("PoseDataset must contain at least one frame")
        if not np.issubdtype(idx.dtype, np.integer):
            raise ValueError(f"recording_index must be an integer dtype, got {idx.dtype}")

        # Non-decreasing is what makes a recording a contiguous slice. Interleaved
        # frames would make every slice() consumer silently wrong.
        if np.any(np.diff(idx) < 0):
            bad = int(np.argmin(np.diff(idx)))
            raise ValueError(
                f"recording_index must be non-decreasing; it drops at frame {bad + 1} "
                f"({int(idx[bad])} -> {int(idx[bad + 1])}). Recordings must be "
                f"contiguous blocks, not interleaved."
            )
        if idx[0] != 0:
            raise ValueError(f"recording_index must start at 0, got {int(idx[0])}")

        n_rec = int(idx[-1]) + 1
        if len(self.recording_ids) != n_rec:
            raise ValueError(
                f"{len(self.recording_ids)} recording_ids but recording_index "
                f"implies {n_rec} recordings"
            )
        if len(set(self.recording_ids)) != len(self.recording_ids):
            dupes = sorted({r for r in self.recording_ids
                            if self.recording_ids.count(r) > 1})
            raise ValueError(
                f"recording_ids must be unique; duplicates: {dupes[:5]}"
                f"{' ...' if len(dupes) > 5 else ''}. Duplicate ids are how a "
                f"cross-method join silently double-counts."
            )

        # Every recording must actually appear, or an id maps to zero frames and
        # boundaries stops being strictly increasing.
        starts = np.searchsorted(idx, np.arange(n_rec), side="left")
        ends = np.searchsorted(idx, np.arange(n_rec), side="right")
        empty = np.flatnonzero(ends - starts == 0)
        if empty.size:
            raise ValueError(
                f"recording {int(empty[0])} ({self.recording_ids[int(empty[0])]}) "
                f"has zero frames"
            )

        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError(f"fps must be positive and finite, got {self.fps}")

        if self.confidence is not None:
            conf = np.asarray(self.confidence)
            if conf.shape != kp.shape[:2]:
                raise ValueError(
                    f"confidence must be (n_frames, n_keypoints) = {kp.shape[:2]}, "
                    f"got {conf.shape}"
                )

        object.__setattr__(
            self, "_boundaries",
            np.concatenate(([0], ends)).astype(np.int64),
        )

    # -- shape ------------------------------------------------------------

    @property
    def n_frames(self) -> int:
        return int(self.keypoints.shape[0])

    @property
    def n_keypoints(self) -> int:
        return int(self.keypoints.shape[1])

    @property
    def n_recordings(self) -> int:
        return len(self.recording_ids)

    # -- boundaries -------------------------------------------------------

    def boundaries(self) -> np.ndarray:
        """``(n_recordings + 1,)`` int64 such that recording ``k`` occupies
        ``[boundaries[k]:boundaries[k + 1]]``.

        This is the authority on where recordings begin and end.
        """
        return self._boundaries

    def slices(self) -> Iterator[tuple[str, slice]]:
        """Yield ``(recording_id, slice)`` per recording.

        Use this for anything that must not cross a recording boundary —
        derivatives, lags, delay embeddings, transition counts.
        """
        b = self._boundaries
        for k, rid in enumerate(self.recording_ids):
            yield rid, slice(int(b[k]), int(b[k + 1]))

    def valid_windows(self, k: int, stride: int = 1) -> np.ndarray:
        """``(n_frames,)`` bool mask of ``k``-frame windows that stay in one recording.

        Element ``i`` is True when the window *ending* at frame ``i`` — that is,
        frames ``i - (k-1)*stride ... i`` — lies entirely within one recording.
        The window is identified by its last frame because that is the convention
        ``delay_embed`` uses, so a mask from here indexes its output directly.

        ``k=1`` is every frame: a one-frame window cannot straddle anything.
        """
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")

        span = (k - 1) * stride
        mask = np.zeros(self.n_frames, dtype=bool)
        b = self._boundaries
        for j in range(self.n_recordings):
            lo, hi = int(b[j]), int(b[j + 1])
            # The first `span` frames of each recording have no room behind them.
            if hi - lo > span:
                mask[lo + span : hi] = True
        return mask

    # -- time -------------------------------------------------------------

    def seconds_to_frames(self, seconds: float) -> int:
        """Convert a duration in seconds to whole frames, rounding to nearest.

        Always at least 1 frame, so a sub-frame parameter degrades to the smallest
        representable lag rather than to zero — a zero lag turns a transition count
        into a self-count.
        """
        return max(1, int(round(float(seconds) * self.fps)))

    def frames_to_seconds(self, frames):
        return np.asarray(frames) / self.fps

    # -- construction -----------------------------------------------------

    @classmethod
    def from_sessions(
        cls,
        sessions: list[np.ndarray],
        recording_ids: list[str],
        keypoint_names: list[str],
        fps: float,
        *,
        confidences: list[np.ndarray] | None = None,
        metadata: pd.DataFrame | None = None,
        dataset: str = "",
    ) -> "PoseDataset":
        """Build from a list of per-recording ``(n_frames, n_keypoints, 2)`` arrays.

        This is the shape every loader produces and the shape ``vieb_v2`` passes
        around internally, so it is the seam where per-recording data becomes one
        boundary-aware dataset.
        """
        if len(sessions) != len(recording_ids):
            raise ValueError(
                f"{len(sessions)} sessions but {len(recording_ids)} recording_ids"
            )
        if not sessions:
            raise ValueError("cannot build a PoseDataset from zero sessions")

        lengths = [int(np.asarray(s).shape[0]) for s in sessions]
        index = np.repeat(np.arange(len(sessions), dtype=np.int64), lengths)
        conf = None
        if confidences is not None:
            if len(confidences) != len(sessions):
                raise ValueError(
                    f"{len(confidences)} confidences but {len(sessions)} sessions"
                )
            conf = np.concatenate([np.asarray(c) for c in confidences], axis=0)

        return cls(
            keypoints=np.concatenate([np.asarray(s) for s in sessions], axis=0),
            recording_index=index,
            recording_ids=list(recording_ids),
            keypoint_names=list(keypoint_names),
            fps=fps,
            metadata=metadata if metadata is not None else pd.DataFrame(),
            confidence=conf,
            dataset=dataset,
        )

    def subset(self, recording_ids: list[str]) -> "PoseDataset":
        """A new dataset containing only the named recordings, in the given order.

        Used by the verification gate to run an arm on a small Luna subset, and by
        the train/test split. Metadata is filtered to match.
        """
        wanted = {rid: i for i, rid in enumerate(self.recording_ids)}
        missing = [r for r in recording_ids if r not in wanted]
        if missing:
            raise KeyError(
                f"{len(missing)} recording_ids not in this dataset: {missing[:5]}"
                f"{' ...' if len(missing) > 5 else ''}"
            )

        b = self._boundaries
        sessions, confs = [], []
        for rid in recording_ids:
            k = wanted[rid]
            sl = slice(int(b[k]), int(b[k + 1]))
            sessions.append(self.keypoints[sl])
            if self.confidence is not None:
                confs.append(self.confidence[sl])

        meta = self.metadata
        if not meta.empty and "recording_id" in meta.columns:
            meta = meta[meta["recording_id"].isin(recording_ids)].reset_index(drop=True)

        return PoseDataset.from_sessions(
            sessions,
            list(recording_ids),
            self.keypoint_names,
            self.fps,
            confidences=confs if self.confidence is not None else None,
            metadata=meta,
            dataset=self.dataset,
        )
