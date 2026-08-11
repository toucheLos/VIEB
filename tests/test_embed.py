"""Tests for boundary-safe delay embedding.

The property that matters: no delay vector ever mixes frames from two recordings.
That failure is silent — it produces a point, not an error — so it is tested by
construction rather than by inspection.
"""

from __future__ import annotations

import numpy as np
import pytest

from vieb.data.dataset import PoseDataset
from vieb.representations.delay_embed import delay_embed, embedded_length, scatter_labels


def make_data(lengths, fps=30.0):
    index = np.repeat(np.arange(len(lengths)), lengths).astype(np.int64)
    return PoseDataset(
        keypoints=np.zeros((int(sum(lengths)), 1, 2)),
        recording_index=index,
        recording_ids=[f"rec{i}" for i in range(len(lengths))],
        keypoint_names=["kp"],
        fps=fps,
    )


class TestDelayEmbed:
    def test_k1_is_identity(self):
        data = make_data([5, 4])
        X = np.arange(9, dtype=float).reshape(9, 1)
        out = delay_embed(X, data, k=1)
        assert np.array_equal(out, X), "k=1 is the control and must not alter X"

    def test_never_mixes_recordings(self):
        # Recording 0 holds values 0..4, recording 1 holds 100..103. Any delay
        # vector containing both a small and a large value crossed the seam.
        data = make_data([5, 4])
        X = np.concatenate([np.arange(5.0), 100 + np.arange(4.0)]).reshape(-1, 1)
        out = delay_embed(X, data, k=3)
        for row in out:
            assert (row < 50).all() or (row >= 50).all(), (
                f"delay vector {row} spans a recording boundary"
            )

    def test_output_length_per_recording(self):
        data = make_data([5, 4])
        X = np.zeros((9, 2))
        out = delay_embed(X, data, k=3)
        # 5 -> 3 windows, 4 -> 2 windows
        assert out.shape == (5, 6)

    def test_stride_respected(self):
        data = make_data([7])
        X = np.arange(7.0).reshape(7, 1)
        out = delay_embed(X, data, k=3, stride=2)
        # span = 4, so 3 windows: [0,2,4], [1,3,5], [2,4,6]
        assert out.shape == (3, 3)
        assert np.array_equal(out[0], [0.0, 2.0, 4.0])
        assert np.array_equal(out[2], [2.0, 4.0, 6.0])

    def test_short_recordings_dropped_not_padded(self):
        data = make_data([10, 2])
        X = np.zeros((12, 1))
        out = delay_embed(X, data, k=5)
        # only recording 0 contributes: 10 - 4 = 6
        assert out.shape[0] == 6

    def test_dropped_recordings_are_reported(self):
        # A run that silently dropped short recordings and one that dropped none
        # must not produce indistinguishable provenance.
        data = make_data([10, 2])
        report: dict = {}
        delay_embed(np.zeros((12, 1)), data, k=5, report=report)
        assert report["dropped_recordings"] == ["rec1"]
        assert "1 of 2 recordings" in report["dropped_note"]

    def test_no_report_key_when_nothing_dropped(self):
        data = make_data([10, 10])
        report: dict = {}
        delay_embed(np.zeros((20, 1)), data, k=5, report=report)
        assert report == {}

    def test_raises_when_nothing_survives(self):
        data = make_data([3, 2])
        with pytest.raises(ValueError, match="no points"):
            delay_embed(np.zeros((5, 1)), data, k=10)

    def test_index_locates_last_frame_of_window(self):
        data = make_data([5, 4])
        X = np.zeros((9, 1))
        out, rec, frm = delay_embed(X, data, k=3, return_index=True)
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
        data = make_data([5, 4])
        X = np.zeros((9, 1))
        _, rec, frm = delay_embed(X, data, k=3, return_index=True)
        point_labels = np.array([0, 0, 1, 2, 2], dtype=np.int32)
        frames = scatter_labels(point_labels, rec, frm, data)
        assert frames.shape == (9,)
        # rec0 windows end at local 2,3,4 -> global 2,3,4; lead-in 0,1 backfilled
        assert frames[:5].tolist() == [0, 0, 0, 0, 1]
        # rec1 windows end at local 2,3 -> global 7,8; lead-in 5,6 backfilled
        assert frames[5:].tolist() == [2, 2, 2, 2]

    def test_backfill_never_crosses_a_seam(self):
        data = make_data([4, 4])
        X = np.zeros((8, 1))
        _, rec, frm = delay_embed(X, data, k=3, return_index=True)
        # rec0 -> labels 5,5 ; rec1 -> labels 9,9
        frames = scatter_labels(np.array([5, 5, 9, 9], dtype=np.int32), rec, frm, data)
        assert set(frames[:4].tolist()) == {5}
        assert set(frames[4:].tolist()) == {9}

    def test_without_backfill_leaves_lead_in_unassigned(self):
        data = make_data([5])
        X = np.zeros((5, 1))
        _, rec, frm = delay_embed(X, data, k=3, return_index=True)
        frames = scatter_labels(
            np.array([1, 1, 1], dtype=np.int32), rec, frm, data, backfill_window=False
        )
        assert frames.tolist() == [-1, -1, 1, 1, 1]


class TestBoundaryLeakage:
    """§7.4's gate: frame 0 of recording 2 must inherit nothing from recording 1.

    Parametrized over embedding depths so it covers every arm's configuration,
    not just the default. Any representation that takes a derivative or lag is
    asserted through this same fixture.
    """

    @pytest.mark.parametrize("k,stride", [(2, 1), (3, 1), (4, 2), (5, 3)])
    def test_first_frame_of_second_recording_is_uncontaminated(self, k, stride):
        # Recording 1 is all zeros; recording 2 is all ones. A delay vector that
        # leaked would contain both values. 20 frames each so the widest window
        # under test (k=5, stride=3, span=12) still yields points.
        data = make_data([20, 20])
        X = np.concatenate([np.zeros(20), np.ones(20)]).reshape(-1, 1)
        out, rec, _ = delay_embed(X, data, k=k, stride=stride, return_index=True)
        for row, r in zip(out, rec):
            expected = 0.0 if r == 0 else 1.0
            assert (row == expected).all(), (
                f"delay vector from recording {int(r)} contains {set(row.tolist())}; "
                f"the window crossed the seam at frame 20"
            )

    def test_derivative_over_slices_never_crosses(self):
        # The generic pattern every representation must follow: differentiate
        # per recording, never over the pooled array.
        data = make_data([6, 6])
        X = np.concatenate([np.arange(6.0), np.arange(6.0) + 1000]).reshape(-1, 1)

        pooled = np.diff(X, axis=0, prepend=X[:1])
        per_recording = np.zeros_like(X)
        for _, sl in data.slices():
            seg = X[sl]
            per_recording[sl] = np.diff(seg, axis=0, prepend=seg[:1])

        # The pooled version invents a 995-unit jump at the seam.
        assert pooled[6, 0] == pytest.approx(995.0)
        assert per_recording[6, 0] == 0.0, (
            "frame 0 of recording 2 must not inherit recording 1's last frame"
        )
