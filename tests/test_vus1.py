"""Tests for the model contract and the VUS-1 schema.

The recording-boundary behaviour is the point. A bout that spans two recordings,
or a delay window straddling a seam, is silently garbage — it produces a number
rather than an error. These tests are what make that loud.
"""

from __future__ import annotations

import numpy as np
import pytest

from vieb.models.base import UNASSIGNED, RepresentationMeta, validate_labels
from vieb.paths import config_hash, run_dir, short
from vieb.schema.vus1 import encode_bouts, normalize_recording_id


def make_meta(lengths=(5, 4), fps=30.0):
    bounds = np.concatenate(([0], np.cumsum(lengths))).astype(np.int64)
    return RepresentationMeta(
        name="test",
        repr_hash="sha256:" + "0" * 64,
        fps=fps,
        recording_ids=[f"rec{i}" for i in range(len(lengths))],
        boundaries=bounds,
        channel_names=["a", "b"],
        dataset="fixture",
    )


class TestRepresentationMeta:
    def test_slices_partition_all_frames(self):
        meta = make_meta((5, 4, 3))
        seen = np.zeros(meta.n_frames, dtype=int)
        for _, sl in meta.slices():
            seen[sl] += 1
        assert (seen == 1).all(), "slices must partition the frames exactly once"

    def test_rejects_empty_recording(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            RepresentationMeta(
                name="t", repr_hash="sha256:x", fps=30.0,
                recording_ids=["a", "b"],
                boundaries=np.array([0, 5, 5]),  # b is empty
            )

    def test_rejects_id_count_mismatch(self):
        with pytest.raises(ValueError, match="recording_ids"):
            RepresentationMeta(
                name="t", repr_hash="sha256:x", fps=30.0,
                recording_ids=["only_one"],
                boundaries=np.array([0, 5, 9]),
            )

    def test_seconds_to_frames_never_zero(self):
        meta = make_meta(fps=30.0)
        assert meta.seconds_to_frames(1.0) == 30
        assert meta.seconds_to_frames(1 / 30) == 1
        # A sub-frame parameter must degrade to the smallest representable lag,
        # not to zero — a zero lag turns a transition count into a self-count.
        assert meta.seconds_to_frames(1e-9) == 1


class TestEncodeBouts:
    def test_never_spans_a_recording_boundary(self):
        # Same label continuous across the seam between rec0 and rec1.
        meta = make_meta((5, 4))
        labels = np.zeros(9, dtype=np.int32)
        bouts = encode_bouts(labels, meta)
        assert len(bouts) == 2, "one continuous label must still yield one bout per recording"
        assert bouts["recording_id"].tolist() == ["rec0", "rec1"]
        assert bouts["start_frame"].tolist() == [0, 0], "frame indices are recording-local"
        assert bouts["end_frame"].tolist() == [5, 4]

    def test_drops_unassigned_by_default(self):
        meta = make_meta((6,))
        labels = np.array([0, 0, UNASSIGNED, UNASSIGNED, 1, 1], dtype=np.int32)
        bouts = encode_bouts(labels, meta)
        assert bouts["state"].tolist() == [0, 1]
        assert bouts["start_frame"].tolist() == [0, 4]

    def test_keeps_unassigned_when_asked(self):
        meta = make_meta((6,))
        labels = np.array([0, 0, UNASSIGNED, UNASSIGNED, 1, 1], dtype=np.int32)
        bouts = encode_bouts(labels, meta, keep_unassigned=True)
        assert bouts["state"].tolist() == [0, UNASSIGNED, 1]

    def test_durations_sum_to_assigned_frames(self):
        meta = make_meta((7, 5))
        rng = np.random.default_rng(0)
        labels = rng.integers(-1, 3, size=12).astype(np.int32)
        bouts = encode_bouts(labels, meta)
        total = (bouts["end_frame"] - bouts["start_frame"]).sum()
        assert total == int((labels != UNASSIGNED).sum())

    def test_alternating_labels_give_unit_bouts(self):
        meta = make_meta((6,))
        labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
        bouts = encode_bouts(labels, meta)
        assert len(bouts) == 6
        assert ((bouts["end_frame"] - bouts["start_frame"]) == 1).all()

    def test_rejects_wrong_length(self):
        meta = make_meta((5, 4))
        with pytest.raises(ValueError, match="frames"):
            encode_bouts(np.zeros(8, dtype=np.int32), meta)


class TestValidateLabels:
    def test_accepts_valid(self):
        meta = make_meta((5, 4))
        out = validate_labels(np.zeros(9, dtype=np.int64), meta)
        assert out.dtype == np.int32

    def test_rejects_negative_other_than_unassigned(self):
        meta = make_meta((5, 4))
        labels = np.zeros(9, dtype=np.int32)
        labels[3] = -7
        with pytest.raises(ValueError, match="unassigned"):
            validate_labels(labels, meta)

    def test_rejects_float(self):
        meta = make_meta((5, 4))
        with pytest.raises(ValueError, match="integer"):
            validate_labels(np.zeros(9, dtype=np.float32), meta)

    def test_rejects_length_mismatch(self):
        meta = make_meta((5, 4))
        with pytest.raises(ValueError, match="frames"):
            validate_labels(np.zeros(10, dtype=np.int32), meta)


class TestNormalizeRecordingId:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            (
                "20241016_Box_1_CFC_Day_0_(Context_A)_308DLC_Resnet50_VIEBFeb11shuffle2_snapshot_best-30.h5",
                "20241016_Box_1_CFC_Day_0_(Context_A)_308",
            ),
            ("plain_name.csv", "plain_name"),
            ("plain_name", "plain_name"),
            ("/some/dir/nested_308DLC_Resnet50_x.h5", "nested_308"),
            ("Coord_3D.rat142.baseline.2020-02-26_14_01_48.csv",
             "Coord_3D.rat142.baseline.2020-02-26_14_01_48"),
        ],
    )
    def test_normalizes(self, raw, expected):
        assert normalize_recording_id(raw) == expected

    def test_is_idempotent(self):
        raw = "x_308DLC_Resnet50_y.h5"
        once = normalize_recording_id(raw)
        assert normalize_recording_id(once) == once


class TestPaths:
    def test_config_hash_is_order_independent(self):
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})

    def test_config_hash_separates_kappa(self):
        # The failure this prevents: two kappa values overwriting one run dir.
        a = config_hash({"model": "moseq", "kappa": 1e4})
        b = config_hash({"model": "moseq", "kappa": 1e6})
        assert a != b
        assert short(a) != short(b)

    def test_run_dir_layout(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VIEB_RUN_STORE", str(tmp_path))
        h = config_hash({"kappa": 1.0})
        d = run_dir("luna", "sha256:" + "a" * 64, "moseq", h)
        assert d.relative_to(tmp_path).parts == ("luna", "a" * 12, "moseq", short(h))
