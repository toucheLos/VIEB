"""Tests for PoseDataset, the segmenter contract, and the VUS-1 schema.

The recording-boundary behaviour is the point. A bout that spans two recordings,
or a delay window straddling a seam, is silently garbage — it produces a number
rather than an error. These tests are what make that loud.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vieb.data.dataset import UNASSIGNED, PoseDataset
from vieb.io.vus1 import (
    RunManifest,
    encode_bouts,
    normalize_recording_id,
    read_bouts,
    read_run,
    write_run,
)
from vieb.paths import config_hash, run_dir, short
from vieb.segmenters.base import make_segmentation, validate_labels


def make_data(lengths=(5, 4), fps=30.0, n_keypoints=2):
    """A dataset with the given per-recording frame counts."""
    index = np.repeat(np.arange(len(lengths)), lengths).astype(np.int64)
    total = int(sum(lengths))
    return PoseDataset(
        keypoints=np.zeros((total, n_keypoints, 2)),
        recording_index=index,
        recording_ids=[f"rec{i}" for i in range(len(lengths))],
        keypoint_names=[f"kp{i}" for i in range(n_keypoints)],
        fps=fps,
        dataset="fixture",
    )


class TestPoseDataset:
    def test_slices_partition_all_frames(self):
        data = make_data((5, 4, 3))
        seen = np.zeros(data.n_frames, dtype=int)
        for _, sl in data.slices():
            seen[sl] += 1
        assert (seen == 1).all(), "slices must partition the frames exactly once"

    def test_boundaries_match_lengths(self):
        data = make_data((5, 4, 3))
        assert data.boundaries().tolist() == [0, 5, 9, 12]

    def test_rejects_empty_recording(self):
        # recording 1 never appears in the index
        with pytest.raises(ValueError, match="zero frames"):
            PoseDataset(
                keypoints=np.zeros((5, 2, 2)),
                recording_index=np.array([0, 0, 0, 2, 2]),
                recording_ids=["a", "b", "c"],
                keypoint_names=["x", "y"],
                fps=30.0,
            )

    def test_rejects_id_count_mismatch(self):
        with pytest.raises(ValueError, match="recording_ids"):
            PoseDataset(
                keypoints=np.zeros((9, 2, 2)),
                recording_index=np.repeat([0, 1], [5, 4]),
                recording_ids=["only_one"],
                keypoint_names=["x", "y"],
                fps=30.0,
            )

    def test_rejects_interleaved_recordings(self):
        # Interleaving would make every slice() consumer silently wrong.
        with pytest.raises(ValueError, match="non-decreasing"):
            PoseDataset(
                keypoints=np.zeros((4, 2, 2)),
                recording_index=np.array([0, 1, 0, 1]),
                recording_ids=["a", "b"],
                keypoint_names=["x", "y"],
                fps=30.0,
            )

    def test_rejects_duplicate_recording_ids(self):
        # Duplicate ids are how a cross-method join silently double-counts.
        with pytest.raises(ValueError, match="unique"):
            PoseDataset(
                keypoints=np.zeros((4, 2, 2)),
                recording_index=np.repeat([0, 1], [2, 2]),
                recording_ids=["same", "same"],
                keypoint_names=["x", "y"],
                fps=30.0,
            )

    def test_seconds_to_frames_never_zero(self):
        data = make_data(fps=30.0)
        assert data.seconds_to_frames(1.0) == 30
        assert data.seconds_to_frames(1 / 30) == 1
        # A sub-frame parameter must degrade to the smallest representable lag,
        # not to zero — a zero lag turns a transition count into a self-count.
        assert data.seconds_to_frames(1e-9) == 1

    def test_seconds_to_frames_tracks_fps(self):
        # The whole point of storing parameters in seconds: one config, two rigs.
        assert make_data(fps=30.0).seconds_to_frames(0.5) == 15
        assert make_data(fps=250.0).seconds_to_frames(0.5) == 125

    def test_from_sessions_round_trips(self):
        a = np.arange(6.0).reshape(3, 1, 2)
        b = np.arange(4.0).reshape(2, 1, 2) + 100
        data = PoseDataset.from_sessions([a, b], ["ra", "rb"], ["nose"], 30.0)
        assert data.n_frames == 5
        assert data.boundaries().tolist() == [0, 3, 5]
        assert np.array_equal(data.keypoints[:3], a)
        assert np.array_equal(data.keypoints[3:], b)

    def test_subset_selects_and_reindexes(self):
        data = make_data((5, 4, 3))
        sub = data.subset(["rec2", "rec0"])
        assert sub.recording_ids == ["rec2", "rec0"]
        assert sub.boundaries().tolist() == [0, 3, 8]

    def test_subset_rejects_unknown_id(self):
        with pytest.raises(KeyError, match="not in this dataset"):
            make_data((5, 4)).subset(["rec0", "nope"])


class TestValidWindows:
    def test_k1_is_every_frame(self):
        data = make_data((5, 4))
        assert data.valid_windows(1).all(), "a one-frame window cannot straddle"

    def test_masks_lead_in_of_every_recording(self):
        data = make_data((5, 4))
        mask = data.valid_windows(3)
        # rec0 spans frames 0..4, rec1 spans 5..8; span = 2
        assert mask.tolist() == [False, False, True, True, True,
                                 False, False, True, True]

    def test_never_true_across_a_seam(self):
        # The property that matters: for every True frame, the whole window
        # behind it belongs to the same recording.
        data = make_data((7, 6, 5))
        k, stride = 4, 2
        span = (k - 1) * stride
        mask = data.valid_windows(k, stride)
        idx = np.asarray(data.recording_index)
        for i in np.flatnonzero(mask):
            window = idx[i - span : i + 1]
            assert (window == window[0]).all(), (
                f"window ending at frame {i} spans recordings {set(window.tolist())}"
            )

    def test_short_recording_contributes_nothing(self):
        data = make_data((10, 2))
        mask = data.valid_windows(5)
        assert not mask[10:].any(), "a recording shorter than the window has no valid frames"

    def test_agrees_with_delay_embed_count(self):
        from vieb.representations.delay_embed import delay_embed

        data = make_data((7, 6, 5))
        X = np.zeros((data.n_frames, 1))
        out = delay_embed(X, data, k=3, stride=2)
        assert out.shape[0] == int(data.valid_windows(3, 2).sum()), (
            "valid_windows must count exactly the frames delay_embed emits"
        )


class TestEncodeBouts:
    def test_never_spans_a_recording_boundary(self):
        # Same label continuous across the seam between rec0 and rec1.
        data = make_data((5, 4))
        labels = np.zeros(9, dtype=np.int32)
        bouts = encode_bouts(labels, data)
        assert len(bouts) == 2, "one continuous label must still yield one bout per recording"
        assert bouts["recording_id"].tolist() == ["rec0", "rec1"]
        assert bouts["start_frame"].tolist() == [0, 0], "frame indices are recording-local"
        assert bouts["end_frame"].tolist() == [5, 4]

    def test_drops_unassigned_by_default(self):
        data = make_data((6,))
        labels = np.array([0, 0, UNASSIGNED, UNASSIGNED, 1, 1], dtype=np.int32)
        bouts = encode_bouts(labels, data)
        assert bouts["state"].tolist() == [0, 1]
        assert bouts["start_frame"].tolist() == [0, 4]

    def test_keeps_unassigned_when_asked(self):
        data = make_data((6,))
        labels = np.array([0, 0, UNASSIGNED, UNASSIGNED, 1, 1], dtype=np.int32)
        bouts = encode_bouts(labels, data, keep_unassigned=True)
        assert bouts["state"].tolist() == [0, UNASSIGNED, 1]

    def test_durations_sum_to_assigned_frames(self):
        data = make_data((7, 5))
        rng = np.random.default_rng(0)
        labels = rng.integers(-1, 3, size=12).astype(np.int32)
        bouts = encode_bouts(labels, data)
        total = (bouts["end_frame"] - bouts["start_frame"]).sum()
        assert total == int((labels != UNASSIGNED).sum())

    def test_alternating_labels_give_unit_bouts(self):
        data = make_data((6,))
        labels = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
        bouts = encode_bouts(labels, data)
        assert len(bouts) == 6
        assert ((bouts["end_frame"] - bouts["start_frame"]) == 1).all()

    def test_rejects_wrong_length(self):
        data = make_data((5, 4))
        with pytest.raises(ValueError, match="frames"):
            encode_bouts(np.zeros(8, dtype=np.int32), data)


class TestValidateLabels:
    def test_accepts_valid(self):
        data = make_data((5, 4))
        out = validate_labels(np.zeros(9, dtype=np.int64), data)
        assert out.dtype == np.int32

    def test_rejects_negative_other_than_unassigned(self):
        data = make_data((5, 4))
        labels = np.zeros(9, dtype=np.int32)
        labels[3] = -7
        with pytest.raises(ValueError, match="unassigned"):
            validate_labels(labels, data)

    def test_rejects_float(self):
        data = make_data((5, 4))
        with pytest.raises(ValueError, match="integer"):
            validate_labels(np.zeros(9, dtype=np.float32), data)

    def test_rejects_length_mismatch(self):
        data = make_data((5, 4))
        with pytest.raises(ValueError, match="frames"):
            validate_labels(np.zeros(10, dtype=np.int32), data)


class TestSegmentation:
    def test_n_states_counts_distinct_not_max(self):
        # A method emitting ids 0, 1, 4 has three states. Counting five would
        # inflate every per-state average by the two that never occur.
        data = make_data((5,))
        seg = make_segmentation(np.array([0, 1, 4, 4, 1], dtype=np.int32), data)
        assert seg.n_states == 3

    def test_unassigned_excluded_from_state_count(self):
        data = make_data((4,))
        seg = make_segmentation(np.array([-1, -1, 0, 0], dtype=np.int32), data)
        assert seg.n_states == 1
        assert seg.unassigned_frac == 0.5

    def test_all_unassigned_is_zero_states(self):
        # The exbias result: n_states 0, noise_frac 1.0. It must survive the
        # contract rather than raising, because a measured null is a result.
        data = make_data((4,))
        seg = make_segmentation(np.full(4, -1, dtype=np.int32), data)
        assert seg.n_states == 0
        assert seg.unassigned_frac == 1.0

    def test_extra_carries_method_specific_output(self):
        data = make_data((4,))
        seg = make_segmentation(np.zeros(4, dtype=np.int32), data, backend="cpu")
        assert seg.extra["backend"] == "cpu"


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
            # The exbias segment files, which are already normalized.
            ("20241016_Box_1_CFC_Day_0_(Context_A)_308.npz",
             "20241016_Box_1_CFC_Day_0_(Context_A)_308"),
        ],
    )
    def test_normalizes(self, raw, expected):
        assert normalize_recording_id(raw) == expected

    def test_is_idempotent(self):
        raw = "x_308DLC_Resnet50_y.h5"
        once = normalize_recording_id(raw)
        assert normalize_recording_id(once) == once

    def test_csv_and_h5_of_one_recording_agree(self):
        # This is the join that silently breaks comparisons: the same recording
        # arrives as .csv from one arm and .h5 from another.
        stem = "20241016_Box_1_CFC_Day_0_(Context_A)_308"
        suffix = "DLC_Resnet50_VIEBFeb11shuffle2_snapshot_best-30"
        assert (normalize_recording_id(stem + suffix + ".csv")
                == normalize_recording_id(stem + suffix + ".h5")
                == stem)


class TestVUS1RoundTrip:
    def _manifest(self, **kw):
        base = dict(
            representation="pca", segmenter="hdbscan", segmenter_version="2.0.0",
            config={"min_cluster_size": 50}, config_hash="sha256:c",
            repr_hash="sha256:r", dataset="luna", fps=30.0,
            git_sha="abc123", git_dirty=False, seed=0, device="cpu",
            wall_clock_s=1.5,
        )
        base.update(kw)
        return RunManifest(**base)

    def test_write_then_read(self, tmp_path):
        data = make_data((5, 4))
        labels = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32)
        bouts = encode_bouts(labels, data)
        write_run(tmp_path / "run", bouts, self._manifest())

        manifest, got = read_run(tmp_path / "run")
        assert manifest.representation == "pca"
        assert manifest.segmenter == "hdbscan"
        assert manifest.schema_version == "VUS-1"
        pd.testing.assert_frame_equal(
            got.reset_index(drop=True), bouts.reset_index(drop=True)
        )

    def test_manifest_records_both_slots(self):
        # §5: an artifact that records only a fused arm name cannot be grouped
        # along either axis.
        m = self._manifest(representation="diffusion", segmenter="koopman")
        assert m.representation == "diffusion" and m.segmenter == "koopman"

    def test_reads_exbias_style_manifest(self, tmp_path):
        # ExBias shipped method_name/method_version and no representation.
        import json

        run = tmp_path / "exbias_002"
        run.mkdir()
        (run / "run_manifest.json").write_text(json.dumps({
            "schema_version": "VUS-1",
            "method_name": "exbias",
            "method_version": "1.0-axiomatic",
            "seed": 0, "fps": 30.0,
            "n_recordings": 3846, "n_frames_total": 22355989,
            "parameters": {"degree": 3, "L": 20},
        }))
        m = RunManifest.read(run / "run_manifest.json")
        assert m.segmenter == "exbias"
        assert m.segmenter_version == "1.0-axiomatic"
        assert m.representation == "identity"
        assert m.config == {"degree": 3, "L": 20}
        assert m.n_frames == 22355989

    def test_reads_exbias_style_bouts(self, tmp_path):
        # ExBias used state_id plus derived duration columns.
        path = tmp_path / "bouts.parquet"
        pd.DataFrame({
            "recording_id": pd.Series(["a", "b"], dtype="string"),
            "state_id": np.array([0, 1], dtype="int16"),
            "start_frame": np.array([0, 0], dtype="int64"),
            "end_frame": np.array([5, 7], dtype="int64"),
            "duration_frames": np.array([5, 7], dtype="int32"),
        }).to_parquet(path, index=False)

        got = read_bouts(path)
        assert list(got.columns) == ["recording_id", "state", "start_frame", "end_frame"]
        assert got["state"].tolist() == [0, 1]

    def test_manifest_written_last(self, tmp_path):
        # A run interrupted mid-write must leave no manifest, so the comparison
        # table skips it rather than reading truncated bouts as complete.
        data = make_data((4,))
        bouts = encode_bouts(np.zeros(4, dtype=np.int32), data)
        run = tmp_path / "run"
        write_run(run, bouts, self._manifest())
        assert (run / "bouts.parquet").exists()
        assert (run / "run_manifest.json").exists()


class TestPaths:
    def test_config_hash_is_order_independent(self):
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})

    def test_config_hash_separates_kappa(self):
        # The failure this prevents: two kappa values overwriting one run dir.
        a = config_hash({"segmenter": "moseq", "kappa": 1e4})
        b = config_hash({"segmenter": "moseq", "kappa": 1e6})
        assert a != b
        assert short(a) != short(b)

    def test_run_dir_layout(self, monkeypatch, tmp_path):
        monkeypatch.setenv("VIEB_RUN_STORE", str(tmp_path))
        h = config_hash({"kappa": 1.0})
        d = run_dir("luna", "sha256:" + "a" * 64, "moseq", h)
        assert d.relative_to(tmp_path).parts == ("luna", "a" * 12, "moseq", short(h))
