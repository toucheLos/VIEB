"""Tests for the ``moseq_latent`` representation.

Runs in the repo venv — it is pure pandas, with no jax dependency, unlike the
``hsmm`` segmenter that consumes it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vieb.data.dataset import PoseDataset
from vieb.registry import REPRESENTATIONS
from vieb.representations.moseq_latent import (
    MoSeqLatentRepresentation,
    find_results_dir,
    index_results,
)

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


def write_results(root, name, n, latent_dim=10, locomotor=True):
    root.mkdir(parents=True, exist_ok=True)
    cols = {"syllable": np.zeros(n, dtype=int)}
    if locomotor:
        cols.update({"centroid x": np.arange(n, dtype=float),
                     "centroid y": np.arange(n, dtype=float),
                     "heading": np.zeros(n)})
    for i in range(latent_dim):
        cols[f"latent_state {i}"] = np.full(n, float(i))
    pd.DataFrame(cols).to_csv(root / f"{name}{SUFFIX}.csv", index=False)


class TestFindAndIndex:
    def test_index_normalizes_recording_ids(self, tmp_path):
        write_results(tmp_path, "rec_308", 20)
        assert list(index_results(tmp_path)) == ["rec_308"]

    def test_find_results_dir_accepts_a_project_root(self, tmp_path):
        write_results(tmp_path / "2026_07_26-19_54_24" / "results", "rec_308", 20)
        assert find_results_dir(tmp_path).name == "results"

    def test_find_results_dir_accepts_the_results_dir_itself(self, tmp_path):
        write_results(tmp_path / "results", "rec_308", 20)
        assert find_results_dir(tmp_path / "results").name == "results"

    def test_missing_results_is_a_clear_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="does not fit PCA"):
            find_results_dir(tmp_path)


class TestFitTransform:
    def test_reads_latents_in_dataset_order(self, tmp_path, data):
        write_results(tmp_path / "results", "rec_308", 20)
        write_results(tmp_path / "results", "rec_311", 15)
        X = MoSeqLatentRepresentation(source=tmp_path).fit_transform(data)
        assert X.shape == (35, 10)
        # every row is [0,1,...,9] by construction, for both recordings
        assert np.array_equal(X[0], np.arange(10, dtype=float))
        assert np.array_equal(X[34], np.arange(10, dtype=float))

    def test_frame_count_mismatch_raises(self, tmp_path, data):
        # A one-frame offset would attribute every state to the wrong frame.
        write_results(tmp_path / "results", "rec_308", 19)
        write_results(tmp_path / "results", "rec_311", 15)
        with pytest.raises(ValueError, match="frame offset"):
            MoSeqLatentRepresentation(source=tmp_path).fit_transform(data)

    def test_id_drift_reports_overlap(self, tmp_path, data):
        write_results(tmp_path / "results", "totally_different", 20)
        with pytest.raises(ValueError, match="normalization drift"):
            MoSeqLatentRepresentation(source=tmp_path).fit_transform(data)

    def test_too_few_latent_columns_raises(self, tmp_path, data):
        write_results(tmp_path / "results", "rec_308", 20, latent_dim=4)
        write_results(tmp_path / "results", "rec_311", 15, latent_dim=4)
        with pytest.raises(ValueError, match="latent columns"):
            MoSeqLatentRepresentation(source=tmp_path).fit_transform(data)

    def test_locomotor_columns_are_reported_but_not_returned(self, tmp_path, data):
        write_results(tmp_path / "results", "rec_308", 20)
        write_results(tmp_path / "results", "rec_311", 15)
        rep = MoSeqLatentRepresentation(source=tmp_path)
        X = rep.fit_transform(data)
        assert X.shape[1] == 10
        assert rep.report_["locomotor_columns_available"] == [
            "centroid x", "centroid y", "heading"
        ]
        assert rep.report_["locomotor_channels_used"] == []


class TestContract:
    def test_registered_under_its_own_name(self):
        assert REPRESENTATIONS["moseq_latent"].name == "moseq_latent"

    def test_repr_hash_tracks_latent_dim(self):
        a = MoSeqLatentRepresentation(latent_dim=10).repr_hash
        b = MoSeqLatentRepresentation(latent_dim=8).repr_hash
        assert a != b and a.startswith("sha256:")

    def test_channel_names_match_the_csv_columns(self):
        names = MoSeqLatentRepresentation(latent_dim=3).channel_names
        assert names == ["latent_state 0", "latent_state 1", "latent_state 2"]

    def test_locomotor_channels_refuses_rather_than_guessing(self):
        # representation-repair has not reported; inventing the channel definition
        # here would make the hsmm arm conditional on a definition it never sanctioned.
        with pytest.raises(NotImplementedError, match="representation-repair"):
            MoSeqLatentRepresentation(locomotor_channels=True)
