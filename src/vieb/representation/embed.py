"""Boundary-safe delay embedding, shared by every model that needs one.

Delay embedding lives here rather than in each model because the failure mode is
silent: a K-frame window straddling two recordings produces a point that is
half one animal and half another, and nothing downstream can tell. There is one
implementation, it takes ``RepresentationMeta``, and it never crosses a seam.

K stays a *model* hyperparameter rather than being baked into the frozen
representation, because §4a picks K by sweeping it — but every model that
embeds does so through this function, so the boundary handling is identical
across arms and cannot be a source of difference between them.
"""

from __future__ import annotations

import numpy as np

from ..models.base import RepresentationMeta


def embedded_length(n: int, k: int, stride: int = 1) -> int:
    """How many delay vectors a recording of ``n`` frames yields."""
    span = (k - 1) * stride
    return max(0, n - span)


def delay_embed(
    X: np.ndarray,
    meta: RepresentationMeta,
    k: int,
    stride: int = 1,
    *,
    return_index: bool = False,
    report: dict | None = None,
):
    """Stack ``k`` frames spaced ``stride`` apart into one point, per recording.

    Returns ``(n_points, k * n_features)``. With ``k=1`` this is ``X`` unchanged,
    which is the K=1 control §4d asks for.

    Recordings shorter than the embedding span contribute **nothing** and are
    dropped rather than padded — padding would invent frames, and a short
    recording silently contributing a zero-padded point is exactly the kind of
    thing that shows up later as a spurious rare state.

    ``return_index`` additionally returns ``(rec_index, frame_index)`` locating
    each delay vector by the recording it came from and the index of its
    **last** frame within that recording, so labels can be mapped back to frames
    without reconstructing the offsets by hand.
    """
    X = np.asarray(X)
    if X.shape[0] != meta.n_frames:
        raise ValueError(f"X has {X.shape[0]} frames, meta has {meta.n_frames}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    span = (k - 1) * stride
    blocks: list[np.ndarray] = []
    rec_idx: list[np.ndarray] = []
    frm_idx: list[np.ndarray] = []
    dropped: list[str] = []

    for r, (rid, sl) in enumerate(meta.slices()):
        seg = X[sl]
        n_out = embedded_length(seg.shape[0], k, stride)
        if n_out <= 0:
            dropped.append(rid)
            continue
        # Column j is the frame at offset j*stride within each window; the
        # window's last frame is at index span + i for output row i.
        cols = [seg[j * stride : j * stride + n_out] for j in range(k)]
        blocks.append(np.concatenate(cols, axis=1))
        if return_index:
            rec_idx.append(np.full(n_out, r, dtype=np.int64))
            frm_idx.append(np.arange(span, span + n_out, dtype=np.int64))

    if not blocks:
        raise ValueError(
            f"delay embedding with k={k}, stride={stride} (span {span} frames) "
            f"leaves no points: every one of {meta.n_recordings} recordings is "
            f"shorter than the window"
        )

    out = np.concatenate(blocks, axis=0)
    if dropped:
        out_meta_note = (
            f"{len(dropped)} of {meta.n_recordings} recordings shorter than the "
            f"{span + 1}-frame window and dropped"
        )
        out = np.asarray(out)
        # Attached rather than warned so the caller can record it in the manifest.
        setattr(out, "_dropped_note", out_meta_note)

    if return_index:
        return out, np.concatenate(rec_idx), np.concatenate(frm_idx)
    return out


def scatter_labels(
    point_labels: np.ndarray,
    rec_index: np.ndarray,
    frame_index: np.ndarray,
    meta: RepresentationMeta,
    *,
    fill: int = -1,
    backfill_window: bool = True,
) -> np.ndarray:
    """Map per-delay-vector labels back onto per-frame labels.

    A delay vector describes the whole window it spans. ``backfill_window``
    assigns its label to every frame of the window that has no label yet, which
    keeps the first ``span`` frames of each recording from being unassigned
    purely as an artifact of the embedding. Set it False to label only each
    window's last frame and leave the rest ``fill``.
    """
    labels = np.full(meta.n_frames, fill, dtype=np.int32)
    b = np.asarray(meta.boundaries)
    offsets = b[rec_index]
    labels[offsets + frame_index] = point_labels

    if backfill_window:
        # Walk backwards from each recording's first labelled frame, filling the
        # lead-in with that frame's label. Cheap and per-recording, so it can
        # never pull a label across a seam.
        for k in range(meta.n_recordings):
            lo, hi = int(b[k]), int(b[k + 1])
            seg = labels[lo:hi]
            assigned = np.flatnonzero(seg != fill)
            if assigned.size:
                seg[: assigned[0]] = seg[assigned[0]]
    return labels
