"""Tests for the harness that composes the two slots.

The runner is what makes the comparison a grid rather than a pile of scripts, so
what is tested here is the composition itself: that arms are named by string,
that the manifest records both slots, and that the label comparison the
verification gate depends on distinguishes "identical" from "same partition".
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vieb.compare.runner import (
    ArmSpec,
    compare_labels,
    run_arm,
    run_grid,
    specs_from_config,
)
from vieb.data.dataset import PoseDataset
from vieb.registry import REPRESENTATIONS, SEGMENTERS
from vieb.representations.base import BaseRepresentation
from vieb.segmenters.base import make_segmentation


@pytest.fixture
def data():
    lengths = [40, 35, 30]
    rng = np.random.default_rng(0)
    return PoseDataset(
        keypoints=rng.random((sum(lengths), 3, 2)),
        recording_index=np.repeat(np.arange(3), lengths).astype(np.int64),
        recording_ids=["rec_a", "rec_b", "rec_c"],
        keypoint_names=["nose", "center", "tail_base"],
        fps=30.0,
        dataset="fixture",
    )


@pytest.fixture
def toy_arm():
    """Register a trivial representation and segmenter, then clean up."""

    class ToyRepr(BaseRepresentation):
        name = "toy_repr"

        def __init__(self, scale=1.0):
            self.scale = scale

        def fit_transform(self, data):
            X = data.keypoints.reshape(data.n_frames, -1) * self.scale
            return self._check_output(X, data)

        def get_params(self):
            return {"scale": self.scale}

    class ToySeg:
        name = "toy_seg"
        version = "1.0"

        def __init__(self, n_states=3):
            self.n_states = n_states
            self._labels = None

        def fit(self, X, data, *, seed=0):
            rng = np.random.default_rng(seed)
            self._labels = rng.integers(0, self.n_states, size=data.n_frames).astype(np.int32)

        def predict(self, X, data):
            return make_segmentation(self._labels, data, note="toy")

        def get_params(self):
            return {"n_states": self.n_states}

    REPRESENTATIONS._resolved["toy_repr"] = ToyRepr
    SEGMENTERS._resolved["toy_seg"] = ToySeg
    yield
    REPRESENTATIONS._resolved.pop("toy_repr", None)
    SEGMENTERS._resolved.pop("toy_seg", None)


class TestArmSpec:
    def test_default_name_joins_both_slots(self):
        assert ArmSpec("pca", "hdbscan").name == "pca_hdbscan"

    def test_unknown_representation_fails_at_construction(self):
        # Config typos must surface here, not three stages later.
        with pytest.raises(KeyError, match="unknown representation"):
            ArmSpec("nope", "hdbscan")

    def test_unknown_segmenter_fails_at_construction(self):
        with pytest.raises(KeyError, match="unknown segmenter"):
            ArmSpec("pca", "nope")

    @pytest.mark.parametrize("dead", ["ticc", "flow_field"])
    def test_dropped_methods_are_rejected_by_name(self, dead):
        with pytest.raises(KeyError, match="unknown segmenter"):
            ArmSpec("pca", dead)


class TestRunArm:
    def test_runs_and_writes_vus1(self, data, toy_arm, tmp_path):
        res = run_arm(ArmSpec("toy_repr", "toy_seg"), data, store=tmp_path)
        assert (res.run_dir / "run_manifest.json").exists()
        assert (res.run_dir / "bouts.parquet").exists()
        assert res.segmentation.frame_labels.shape == (data.n_frames,)

    def test_manifest_records_both_slots(self, data, toy_arm, tmp_path):
        res = run_arm(ArmSpec("toy_repr", "toy_seg"), data, store=tmp_path)
        assert res.manifest.representation == "toy_repr"
        assert res.manifest.segmenter == "toy_seg"
        assert res.manifest.repr_hash.startswith("sha256:")
        assert res.manifest.config_hash.startswith("sha256:")

    def test_manifest_records_shape_and_result(self, data, toy_arm, tmp_path):
        res = run_arm(ArmSpec("toy_repr", "toy_seg"), data, store=tmp_path)
        assert res.manifest.n_recordings == 3
        assert res.manifest.n_frames == data.n_frames
        assert res.manifest.n_states == 3
        assert res.manifest.fps == 30.0

    def test_params_reach_the_components(self, data, toy_arm, tmp_path):
        res = run_arm(
            ArmSpec("toy_repr", "toy_seg",
                    representation_params={"scale": 2.0},
                    segmenter_params={"n_states": 5}),
            data, store=tmp_path,
        )
        assert res.manifest.config["representation_params"]["scale"] == 2.0
        assert res.manifest.segmenter == "toy_seg"
        assert res.segmentation.n_states == 5

    def test_different_params_get_different_config_hashes(self, data, toy_arm, tmp_path):
        # Two runs of one arm with different settings must not overwrite each other.
        a = run_arm(ArmSpec("toy_repr", "toy_seg", segmenter_params={"n_states": 3}),
                    data, store=tmp_path / "a")
        b = run_arm(ArmSpec("toy_repr", "toy_seg", segmenter_params={"n_states": 7}),
                    data, store=tmp_path / "b")
        assert a.manifest.config_hash != b.manifest.config_hash

    def test_write_false_leaves_no_run(self, data, toy_arm, tmp_path):
        res = run_arm(ArmSpec("toy_repr", "toy_seg"), data, store=tmp_path, write=False)
        assert not (res.run_dir / "run_manifest.json").exists()
        assert res.segmentation.n_states == 3

    def test_bouts_respect_recording_boundaries(self, data, toy_arm, tmp_path):
        res = run_arm(ArmSpec("toy_repr", "toy_seg"), data, store=tmp_path)
        bouts = pd.read_parquet(res.run_dir / "bouts.parquet")
        assert set(bouts["recording_id"]) == {"rec_a", "rec_b", "rec_c"}
        for rid, grp in bouts.groupby("recording_id"):
            span = data.boundaries()[data.recording_ids.index(rid) + 1] - \
                   data.boundaries()[data.recording_ids.index(rid)]
            assert grp["end_frame"].max() <= span, "bout runs past its recording"

    def test_seed_is_honoured(self, data, toy_arm, tmp_path):
        a = run_arm(ArmSpec("toy_repr", "toy_seg", seed=1), data, store=tmp_path / "a")
        b = run_arm(ArmSpec("toy_repr", "toy_seg", seed=1), data, store=tmp_path / "b")
        c = run_arm(ArmSpec("toy_repr", "toy_seg", seed=2), data, store=tmp_path / "c")
        assert np.array_equal(a.segmentation.frame_labels, b.segmentation.frame_labels)
        assert not np.array_equal(a.segmentation.frame_labels, c.segmentation.frame_labels)


class TestRunGrid:
    def test_runs_every_arm(self, data, toy_arm, tmp_path):
        specs = [
            ArmSpec("toy_repr", "toy_seg", name="a"),
            ArmSpec("toy_repr", "toy_seg", segmenter_params={"n_states": 5}, name="b"),
        ]
        results, failures = run_grid(specs, data, store=tmp_path)
        assert len(results) == 2 and failures == []

    def test_collect_lets_the_grid_finish(self, data, toy_arm, tmp_path):
        # A missing optional dependency in one arm must not cost the others.
        class Exploding:
            name, version = "boom", "1.0"

            def __init__(self):
                raise ImportError("no keypoint_moseq here")

        SEGMENTERS._resolved["boom"] = Exploding
        try:
            specs = [ArmSpec("toy_repr", "toy_seg"), ArmSpec("toy_repr", "boom")]
            results, failures = run_grid(specs, data, store=tmp_path, on_error="collect")
            assert len(results) == 1
            assert failures[0][0] == "toy_repr_boom"
            assert "keypoint_moseq" in failures[0][1]
        finally:
            SEGMENTERS._resolved.pop("boom", None)


class TestSpecsFromConfig:
    def test_crosses_representations_and_segmenters(self):
        specs = specs_from_config(
            {"representations": ["pca", "diffusion"], "segmenters": ["hdbscan", "koopman"]}
        )
        assert len(specs) == 4
        assert {s.name for s in specs} == {
            "pca_hdbscan", "pca_koopman", "diffusion_hdbscan", "diffusion_koopman"
        }

    def test_explicit_arms_list(self):
        specs = specs_from_config(
            {"arms": [{"representation": "engineered91", "segmenter": "vieb_v1"}]}
        )
        assert len(specs) == 1 and specs[0].name == "engineered91_vieb_v1"

    def test_per_component_params_are_applied(self):
        specs = specs_from_config({
            "representations": ["pca"],
            "segmenters": ["hdbscan"],
            "segmenter_params": {"hdbscan": {"min_cluster_size": 200}},
        })
        assert specs[0].segmenter_params["min_cluster_size"] == 200

    def test_requires_something_to_run(self):
        with pytest.raises(ValueError, match="must define either"):
            specs_from_config({})


class TestCompareLabels:
    def test_identical_is_exact(self):
        a = np.array([0, 0, 1, 1, 2], dtype=np.int32)
        out = compare_labels(a, a.copy())
        assert out["exact"] is True and out["n_differing"] == 0
        assert out["ari_all_frames"] == pytest.approx(1.0)

    def test_permuted_ids_are_the_same_partition_but_not_exact(self):
        # v2 records that the GPU and CPU HDBSCAN backends agree at ARI 1.0 with
        # permuted integer labels, so the gate must distinguish these.
        a = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        b = np.array([2, 2, 0, 0, 1, 1], dtype=np.int32)
        out = compare_labels(a, b)
        assert out["exact"] is False
        assert out["ari_all_frames"] == pytest.approx(1.0)

    def test_different_partitions_score_low(self):
        rng = np.random.default_rng(0)
        a = rng.integers(0, 4, 500)
        b = rng.integers(0, 4, 500)
        assert compare_labels(a, b)["ari_all_frames"] < 0.1

    def test_length_mismatch_is_not_comparable(self):
        out = compare_labels(np.zeros(10), np.zeros(12))
        assert out["comparable"] is False
        assert "length mismatch" in out["reason"]

    def test_reports_unassigned_fractions(self):
        a = np.array([-1, -1, 0, 0], dtype=np.int32)
        b = np.array([-1, 0, 0, 0], dtype=np.int32)
        out = compare_labels(a, b)
        assert out["unassigned_a"] == 0.5
        assert out["unassigned_b"] == 0.25
