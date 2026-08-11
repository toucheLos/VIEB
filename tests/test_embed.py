"""Tests for boundary-safe delay embedding.

The property that matters: no delay vector ever mixes frames from two
recordings. That failure is silent — it produces a point, not an error — so it
is tested by construction rather than by inspection.
"""

from __future__ import annotations

import numpy as np
import pytest

from vieb.models.base import RepresentationMeta
from vieb.representation.embed import delay_embed, embedded_length, scatter_labels


def make_meta(lengths, fps=30.0):
    bounds = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    return RepresentationMeta(
        name="t", repr_hash="sha256:0", fps=fps,
        recording_ids=[f"rec{i}" for i in range(len(lengths))],
        boundaries=bounds, channel_names=["c0"],
    )


class TestDelayEmbed:
    def test_k1_is_identity(self):
        meta = make_meta([5, 4])
        X = np.arange(9, dtype=float).reshape(9, 1)
        out = delay_embed(X, meta, k=1)
        assert np.array_equal(out, X), "k=1 is the control and must not alter X"

    def test_never_mixes_recordings(self):
        # Recording 0 holds values 0..4, recording 1 holds 100..103. Any delay
        # vector containing both a small and a large value crossed the seam.
        meta = make_meta([5, 4])
        X = np.concatenate([np.arange(5.0), 100 + np.arange(4.0)]).reshape(-1, 1)
        out = delay_embed(X, meta, k=3)
        for row in out:
            assert (row < 50).all() or (row >= 50).all(), (
                f"delay vector {row} spans a recording boundary"
            )

    def test_output_length_per_recording(self):
        meta = make_meta([5, 4])
        X = np.zeros((9, 2))
        out = delay_embed(X, meta, k=3)
        # 5 -> 3 windows, 4 -> 2 windows
        assert out.shape == (5, 6)

    def test_stride_respected(self):
        meta = make_meta([7])
        X = np.arange(7.0).reshape(7, 1)
        out = delay_embed(X, meta, k=3, stride=2)
        # span = 4, so 3 windows: [0,2,4], [1,3,5], [2,4,6]
        assert out.shape == (3, 3)
        assert np.array_equal(out[0], [0.0, 2.0, 4.0])
        assert np.array_equal(out[2], [2.0, 4.0, 6.0])

    def test_short_recordings_dropped_not_padded(self):
        meta = make_meta([10, 2])
        X = np.zeros((12, 1))
        out = delay_embed(X, meta, k=5)
        # only recording 0 contributes: 10 - 4 = 6
        assert out.shape[0] == 6

    def test_raises_when_nothing_survives(self):
        meta = make_meta([3, 2])
        with pytest.raises(ValueError, match="no points"):
            delay_embed(np.zeros((5, 1)), meta, k=10)

    def test_index_locates_last_frame_of_window(self):
        meta = make_meta([5, 4])
        X = np.zeros((9, 1))
        out, rec, frm = delay_embed(X, meta, k=3, return_index=True)
        assert out.shape[0] == rec.size == frm.size == 5
        assert rec.tolist() == [0, 0, 0, 1, 1]
        # window last-frame indices are recording-local and start at span=2
        assert frm.tolist() == [2, 3, 4, 2, 3]

    @pytest.mark.parametrize("n,k,stride,expected", [
        (10, 1, 1, 10), (10, 3, 1, 8), (10, 3, 2, 6), (3, 5, 1, 0), (10, 5, 3, 0),
    ])
    def test_embedded_length(self, n, k, stride, expected):
        assert embedded_length(n, k, stride) == expected


class TestScatterLabels:
    def test_round_trips_to_frames(self):
        meta = make_meta([5, 4])
        X = np.zeros((9, 1))
        _, rec, frm = delay_embed(X, meta, k=3, return_index=True)
        point_labels = np.array([0, 0, 1, 2, 2], dtype=np.int32)
        frames = scatter_labels(point_labels, rec, frm, meta)
        assert frames.shape == (9,)
        # rec0 windows end at local 2,3,4 -> global 2,3,4; lead-in 0,1 backfilled
        assert frames[:5].tolist() == [0, 0, 0, 0, 1]
        # rec1 windows end at local 2,3 -> global 7,8; lead-in 5,6 backfilled
        assert frames[5:].tolist() == [2, 2, 2, 2]

    def test_backfill_never_crosses_a_seam(self):
        meta = make_meta([4, 4])
        X = np.zeros((8, 1))
        _, rec, frm = delay_embed(X, meta, k=3, return_index=True)
        # rec0 -> labels 5,5 ; rec1 -> labels 9,9
        frames = scatter_labels(np.array([5, 5, 9, 9], dtype=np.int32), rec, frm, meta)
        assert set(frames[:4].tolist()) == {5}
        assert set(frames[4:].tolist()) == {9}

    def test_without_backfill_leaves_lead_in_unassigned(self):
        meta = make_meta([5])
        X = np.zeros((5, 1))
        _, rec, frm = delay_embed(X, meta, k=3, return_index=True)
        frames = scatter_labels(
            np.array([1, 1, 1], dtype=np.int32), rec, frm, meta, backfill_window=False
        )
        assert frames.tolist() == [-1, -1, 1, 1, 1]
