"""Tests for the segmenter adapters.

These check the contract and the adapter's bookkeeping, not the algorithms —
the algorithms are ported as-is and are verified against their pre-port outputs
by the verification gate, not here.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from vieb.data.dataset import PoseDataset
from vieb.registry import SEGMENTERS
from vieb.segmenters.external import ExBiasSegmenter, MoSeqSegmenter
from vieb.segmenters.vieb_v1 import ViebV1Segmenter

SUFFIX = "DLC_Resnet50_VIEBFeb11shuffle2_snapshot_best-30"


@pytest.fixture
def data():
    lengths = [20, 15]
    return PoseDataset(
        keypoints=np.zeros((sum(lengths), 3, 2)),
        recording_index=np.repeat([0, 1], lengths).astype(np.int64),
        recording_ids=["rec_308", "rec_311"],
        keypoint_names=["nose", "center", "tail_base"],
        fps=30.0,
    )


class TestMoSeqSegmenter:
    def _write(self, root, name, syllables):
        pd.DataFrame({"syllable": syllables, "centroid x": np.zeros(len(syllables))}).to_csv(
            root / f"{name}{SUFFIX}.csv", index=False
        )

    def test_reads_syllables_and_joins_on_normalized_id(self, tmp_path, data):
        self._write(tmp_path, "rec_308", [0] * 10 + [1] * 10)
        self._write(tmp_path, "rec_311", [2] * 15)
        seg = MoSeqSegmenter(source=tmp_path)
        seg.fit(None, data, seed=0)
        s = seg.predict(None, data)
        assert s.n_states == 3
        assert s.frame_labels[:10].tolist() == [0] * 10
        assert s.frame_labels[20:].tolist() == [2] * 15

    def test_never_refits(self, tmp_path, data):
        # The reference arm's AR-HMM is Gibbs-sampled; a refit would not
        # reproduce the syllables decision #65 scored.
        self._write(tmp_path, "rec_308", [1] * 20)
        self._write(tmp_path, "rec_311", [1] * 15)
        seg = MoSeqSegmenter(source=tmp_path)
        seg.fit(np.random.default_rng(0).random((35, 9)), data, seed=0)
        a = seg.predict(None, data).frame_labels
        seg.fit(np.random.default_rng(1).random((35, 4)), data, seed=99)
        assert np.array_equal(a, seg.predict(None, data).frame_labels)

    def test_length_mismatch_raises(self, tmp_path, data):
        # A frame offset attributes each recording's behavior to a neighbour.
        self._write(tmp_path, "rec_308", [0] * 19)
        self._write(tmp_path, "rec_311", [0] * 15)
        seg = MoSeqSegmenter(source=tmp_path)
        with pytest.raises(ValueError, match="disagree in length"):
            seg.fit(None, data, seed=0)

    def test_total_id_drift_raises(self, tmp_path, data):
        self._write(tmp_path, "totally_different", [0] * 20)
        seg = MoSeqSegmenter(source=tmp_path)
        with pytest.raises(ValueError, match="normalization drift"):
            seg.fit(None, data, seed=0)

    def test_missing_source_is_a_clear_error(self, data):
        with pytest.raises(ValueError, match="needs `source`"):
            MoSeqSegmenter().fit(None, data, seed=0)

    def test_missing_syllable_column(self, tmp_path, data):
        pd.DataFrame({"nope": [1, 2]}).to_csv(tmp_path / f"rec_308{SUFFIX}.csv", index=False)
        with pytest.raises(ValueError, match="no syllable column"):
            MoSeqSegmenter(source=tmp_path).fit(None, data, seed=0)


class TestExBiasSegmenter:
    def _write_run(self, root, rows, n_states=0):
        root.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=["recording_id", "state_id", "start_frame", "end_frame"]) \
          .astype({"state_id": "int16", "start_frame": "int64", "end_frame": "int64"}) \
          .to_parquet(root / "bouts.parquet", index=False)
        (root / "run_manifest.json").write_text(json.dumps({
            "schema_version": "VUS-1", "method_name": "exbias",
            "method_version": "1.0-axiomatic", "seed": 0, "fps": 30.0,
        }))

    def test_zero_states_survives_the_contract(self, tmp_path, data):
        # Both real exbias runs produced n_states 0 / noise_frac 1.0. A measured
        # null is part of the result, so this must not raise.
        self._write_run(tmp_path / "run", [])
        seg = ExBiasSegmenter(source=tmp_path / "run", strict=False)
        seg.fit(None, data, seed=0)
        s = seg.predict(None, data)
        assert s.n_states == 0
        assert s.unassigned_frac == 1.0

    def test_reads_state_id_column(self, tmp_path, data):
        self._write_run(tmp_path / "run", [
            ("rec_308", 0, 0, 10), ("rec_308", 1, 10, 20), ("rec_311", 0, 0, 15),
        ])
        seg = ExBiasSegmenter(source=tmp_path / "run")
        seg.fit(None, data, seed=0)
        s = seg.predict(None, data)
        assert s.n_states == 2
        assert s.frame_labels[:10].tolist() == [0] * 10
        assert s.frame_labels[10:20].tolist() == [1] * 10

    def test_missing_bouts_is_a_clear_error(self, tmp_path, data):
        (tmp_path / "empty").mkdir()
        with pytest.raises(FileNotFoundError, match="no bouts.parquet"):
            ExBiasSegmenter(source=tmp_path / "empty").fit(None, data, seed=0)


class TestViebV1Segmenter:
    def test_min_samples_rule_matches_v1(self):
        # compare.py:1734 — a derived default, not a tunable.
        assert ViebV1Segmenter(min_cluster_size=50).effective_min_samples == 10
        assert ViebV1Segmenter(min_cluster_size=30).effective_min_samples == 10
        assert ViebV1Segmenter(min_cluster_size=500).effective_min_samples == 50
        assert ViebV1Segmenter(min_cluster_size=5000).effective_min_samples == 100

    def test_explicit_min_samples_wins(self):
        assert ViebV1Segmenter(min_cluster_size=50, min_samples=7).effective_min_samples == 7

    def test_params_record_the_umap_constants(self):
        # random_state=42 is what makes this arm reproducible at all.
        p = ViebV1Segmenter().get_params()
        assert p["umap_random_state"] == 42
        assert p["umap_neighbors"] == 30
        assert p["umap_min_dist"] == 0.0

    def test_predict_before_fit_raises(self, data):
        with pytest.raises(RuntimeError, match="fit\\(\\) must be called"):
            ViebV1Segmenter().predict(None, data)


class TestRegisteredSegmenters:
    @pytest.mark.parametrize(
        "name,expected",
        [("hdbscan", "hdbscan"), ("koopman", "koopman"), ("moseq", "moseq"),
         ("exbias", "exbias"), ("vieb_v1", "vieb_v1")],
    )
    def test_name_attribute_matches_registry_key(self, name, expected):
        assert SEGMENTERS[name].name == expected

    @pytest.mark.parametrize(
        "name", ["hdbscan", "koopman", "moseq", "exbias", "vieb_v1"]
    )
    def test_get_params_is_json_serializable(self, name):
        # get_params feeds the config hash, which is json-dumped.
        params = SEGMENTERS.build(name).get_params()
        json.dumps(params, default=str)

    @pytest.mark.parametrize(
        "name", ["hdbscan", "koopman", "moseq", "exbias", "vieb_v1"]
    )
    def test_predict_before_fit_raises(self, name, data):
        seg = SEGMENTERS.build(name)
        with pytest.raises(RuntimeError):
            seg.predict(None, data)
