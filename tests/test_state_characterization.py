"""Tests for state_characterizer.py — backend characterization outputs."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import state_characterizer as sc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cluster_info(n_states: int = 3, n_features: int = 10) -> dict:
    rng = np.random.default_rng(42)
    centers = rng.normal(0, 1, (n_states, n_features)).tolist()
    return {
        "n_clusters": n_states,
        "cluster_centers": centers,
        "method": "umap+hdbscan",
    }


def _make_feature_index(n_features: int = 10) -> dict:
    names = [f"feat_{i}" for i in range(n_features)]
    return {
        "_meta": {
            "n_keypoints": 8,
            "n_features": n_features,
            "use_wavelets": False,
            "feature_names": names,
            "semantic_features": [],
        }
    }


def _setup_dirs(tmp_path: Path, n_states=3, n_features=10) -> tuple[Path, Path, Path]:
    results = tmp_path / "results"
    shared = results / "shared"
    features = results / "features"
    char = results / "characterization"
    for d in (shared, features, char):
        d.mkdir(parents=True)

    ci = _make_cluster_info(n_states, n_features)
    (shared / "cluster_info.json").write_text(json.dumps(ci), encoding="utf-8")

    idx = _make_feature_index(n_features)
    (features / "index.json").write_text(json.dumps(idx), encoding="utf-8")

    return results, shared, features


def _make_bouts(n_states: int = 3) -> pd.DataFrame:
    rows = []
    for sid in range(n_states):
        for i in range(10):
            rows.append({
                "stem": f"vid{i}",
                "state": sid,
                "start_frame": i * 100,
                "end_frame": i * 100 + 60,
                "start_sec": i * 3.0,
                "end_sec": i * 3.0 + 2.0,
                "duration_sec": 2.0 + (i % 3) * 0.5,
                "context": ["A", "B", "C"][i % 3],
                "animal_id": f"animal{i % 5}",
                "day": str((i % 3) + 1),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_run_saves_expected_files(tmp_path):
    results, shared, features = _setup_dirs(tmp_path)
    out = sc.run(results, shared, features)
    assert out["n_states"] == 3
    assert not out["errors"]
    char = results / "characterization"
    for fname in [
        "state_feature_profiles.csv",
        "state_feature_zscores.csv",
        "state_duration_summary.csv",
        "state_group_enrichment.csv",
        "state_characterization.json",
    ]:
        assert (char / fname).exists(), f"Missing: {fname}"


def test_zscores_have_one_row_per_state(tmp_path):
    results, shared, features = _setup_dirs(tmp_path, n_states=4)
    sc.run(results, shared, features)
    df = pd.read_csv(results / "characterization" / "state_feature_zscores.csv")
    assert len(df) == 4
    assert "state_id" in df.columns
    assert list(df["state_id"]) == [0, 1, 2, 3]


def test_feature_profiles_columns_match_feature_names(tmp_path):
    n_features = 12
    results, shared, features = _setup_dirs(tmp_path, n_features=n_features)
    sc.run(results, shared, features)
    df = pd.read_csv(results / "characterization" / "state_feature_profiles.csv")
    expected_names = [f"feat_{i}" for i in range(n_features)]
    for name in expected_names:
        assert name in df.columns


def test_duration_summary_has_one_row_per_state(tmp_path):
    results, shared, features = _setup_dirs(tmp_path, n_states=3)
    bouts = _make_bouts(3)
    bouts.to_csv(results / "characterization" / "bouts.csv", index=False)
    sc.run(results, shared, features)
    df = pd.read_csv(results / "characterization" / "state_duration_summary.csv")
    assert len(df) == 3
    assert set(df.columns).issuperset({"state_id", "n_bouts", "mean_sec", "median_sec"})
    for sid in range(3):
        row = df[df["state_id"] == sid].iloc[0]
        assert int(row["n_bouts"]) == 10


def test_state_with_no_bouts_does_not_crash(tmp_path):
    """A state with zero bouts should appear in duration_summary with NaN stats."""
    results, shared, features = _setup_dirs(tmp_path, n_states=3)
    # Only write bouts for state 0 and 1 — state 2 gets none
    bouts = _make_bouts(3)
    bouts = bouts[bouts["state"] != 2]
    bouts.to_csv(results / "characterization" / "bouts.csv", index=False)
    sc.run(results, shared, features)
    df = pd.read_csv(results / "characterization" / "state_duration_summary.csv")
    row2 = df[df["state_id"] == 2].iloc[0]
    assert int(row2["n_bouts"]) == 0
    assert pd.isna(row2["mean_sec"])


def test_missing_metadata_produces_empty_group_enrichment(tmp_path):
    """When no metadata is available and bouts have no group columns, enrichment is empty."""
    results, shared, features = _setup_dirs(tmp_path, n_states=2)
    # Bouts with no group columns
    bouts = pd.DataFrame({
        "stem": ["vid1", "vid2"],
        "state": [0, 1],
        "duration_sec": [1.0, 2.0],
    })
    bouts.to_csv(results / "characterization" / "bouts.csv", index=False)
    sc.run(results, shared, features)
    df = pd.read_csv(results / "characterization" / "state_group_enrichment.csv")
    assert len(df) == 0 or "state_id" in df.columns


def test_group_enrichment_populated_when_bouts_have_context(tmp_path):
    results, shared, features = _setup_dirs(tmp_path, n_states=3)
    bouts = _make_bouts(3)
    bouts.to_csv(results / "characterization" / "bouts.csv", index=False)
    sc.run(results, shared, features)
    df = pd.read_csv(results / "characterization" / "state_group_enrichment.csv")
    assert not df.empty
    assert "context" in df["group_variable"].values


def test_json_summary_has_correct_n_states(tmp_path):
    results, shared, features = _setup_dirs(tmp_path, n_states=5)
    sc.run(results, shared, features)
    data = json.loads((results / "characterization" / "state_characterization.json").read_text())
    assert data["n_states"] == 5
    assert len(data["states"]) == 5


def test_missing_cluster_info_returns_error(tmp_path):
    results = tmp_path / "results"
    shared = results / "shared"
    features = results / "features"
    shared.mkdir(parents=True)
    features.mkdir(parents=True)
    out = sc.run(results, shared, features)
    assert out["n_states"] == 0
    assert any("cluster_info" in e for e in out["errors"])


def test_feature_names_fallback_when_index_missing(tmp_path):
    """When index.json is missing, feature names fall back to feat_0, feat_1, ..."""
    results, shared, features = _setup_dirs(tmp_path, n_states=2, n_features=5)
    (features / "index.json").unlink()
    sc.run(results, shared, features)
    df = pd.read_csv(results / "characterization" / "state_feature_zscores.csv")
    assert len(df) == 2
    # Columns should be generic names
    feat_cols = [c for c in df.columns if c != "state_id"]
    assert all(c.startswith("feature_") for c in feat_cols)


def test_load_outputs_returns_empty_dataframes_when_missing(tmp_path):
    results = tmp_path / "results"
    out = sc.load_outputs(results)
    for key in ("feature_profiles", "feature_zscores", "duration_summary", "group_enrichment"):
        assert isinstance(out[key], pd.DataFrame)
        assert out[key].empty
    assert out["characterization"] == {}


def test_active_project_results_dir_is_respected(tmp_path):
    """run() writes to the results_dir passed in, not to a hardcoded path."""
    results, shared, features = _setup_dirs(tmp_path, n_states=2)
    other = tmp_path / "other_project" / "results"
    other.mkdir(parents=True)
    sc.run(results, shared, features)
    # Files appear in the correct directory
    assert (results / "characterization" / "state_feature_zscores.csv").exists()
    # They do NOT appear in an unrelated directory
    assert not (other / "characterization" / "state_feature_zscores.csv").exists()
