"""Tests for single-key concatenated H5 extraction in compare.py.

A single-key H5 file has one key (e.g. /coords) containing all sessions
concatenated, with a source_file column identifying which session each
row belongs to.  This test creates a synthetic 3-session file and verifies
that all 3 sessions are extracted (not just the first one via ordinal
fallback).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BODYPARTS = ["nose", "left_ear", "right_ear"]
SOURCE_COL = "source_file"
SOURCES = [
    "Coord_3D.rat142.baseline.2020-02-26_14_01_48.csv",
    "Coord_3D.rat143.test.2020-02-27_10_30_00.csv",
    "Coord_3D.rat144.recall.2020-02-28_09_15_22.csv",
]
FRAMES_PER_SOURCE = [40, 55, 30]


@pytest.fixture()
def single_key_h5(tmp_path):
    """Create a single-key concatenated H5 with 3 sessions."""
    h5_path = tmp_path / "concat_pose.h5"
    rng = np.random.default_rng(42)

    rows = []
    for src, n_frames in zip(SOURCES, FRAMES_PER_SOURCE):
        for _ in range(n_frames):
            row = {SOURCE_COL: src}
            for bp in BODYPARTS:
                row[f"{bp}_x"] = rng.random() * 100
                row[f"{bp}_y"] = rng.random() * 100
            rows.append(row)

    df = pd.DataFrame(rows)

    with pd.HDFStore(str(h5_path), mode="w") as store:
        store.put("coords", df, format="table")

    return str(h5_path)


def test_single_key_h5_all_sessions_extracted(tmp_path, single_key_h5, monkeypatch):
    """All 3 sessions in a single-key concatenated H5 should be extracted."""
    h5_path = single_key_h5

    project_dir = tmp_path / "project"
    results_dir = project_dir / "results"
    project_dir.mkdir()

    meta_df = pd.DataFrame({
        "filename": ["dummy.mp4"],
        "date": ["20200226"],
        "box": [1],
        "experiment": ["CFC"],
        "day": [0],
        "context": ["A"],
        "no_shock": ["no"],
        "animal_id": ["dummy"],
        "fear": [""],
    })
    meta_path = project_dir / "metadata.csv"
    meta_df.to_csv(meta_path, index=False)

    config = {
        "pose_source": "h5",
        "h5_path": h5_path,
        "h5_manifest_path": "",
        "h5_source_col": SOURCE_COL,
        "results_dir": str(results_dir),
        "raw_videos_dir": str(project_dir / "raw_videos"),
        "metadata_csv_path": str(meta_path),
    }
    (project_dir / "config.json").write_text(json.dumps(config))

    app_config = {"active_project": str(project_dir)}
    app_config_path = tmp_path / "app_config.json"
    app_config_path.write_text(json.dumps(app_config))

    import vieb_config as vc
    monkeypatch.setattr(vc, "_APP_CONFIG_PATH", str(app_config_path))
    monkeypatch.setattr(vc, "_CONFIG_PATH", str(project_dir / "config.json"))

    import compare
    compare.cmd_extract(fps=30.0, use_wavelets=False)

    index_path = results_dir / "features" / "index.json"
    assert index_path.exists(), "index.json was not created"
    with open(index_path) as f:
        index = json.load(f)

    assert index["_meta"]["pose_source"] == "h5"

    expected_stems = [os.path.splitext(s)[0] for s in SOURCES]
    for stem in expected_stems:
        assert stem in index, f"Session {stem!r} missing from index"
        entry = index[stem]
        assert entry["video_path"] is None
        assert entry["h5_path"] == h5_path
        assert entry["h5_key"] == "coords"
        assert entry["h5_source"] in SOURCES
        assert entry["n_keypoints"] == len(BODYPARTS)

        feat_path = results_dir / "features" / f"{stem}_features.npy"
        assert feat_path.exists(), f"Feature file missing: {feat_path}"
        arr = np.load(feat_path)
        assert arr.ndim == 2
        assert arr.shape[1] == entry["n_features"]

    # Verify frame counts match
    for stem, expected_frames in zip(expected_stems, FRAMES_PER_SOURCE):
        assert index[stem]["n_frames"] == expected_frames


def test_single_key_h5_skips_existing(tmp_path, single_key_h5, monkeypatch):
    """Already-extracted sessions should be skipped on re-run."""
    h5_path = single_key_h5

    project_dir = tmp_path / "project"
    results_dir = project_dir / "results"
    project_dir.mkdir()

    meta_df = pd.DataFrame({
        "filename": ["dummy.mp4"],
        "date": ["20200226"],
        "box": [1],
        "experiment": ["CFC"],
        "day": [0],
        "context": ["A"],
        "no_shock": ["no"],
        "animal_id": ["dummy"],
        "fear": [""],
    })
    meta_path = project_dir / "metadata.csv"
    meta_df.to_csv(meta_path, index=False)

    config = {
        "pose_source": "h5",
        "h5_path": h5_path,
        "h5_manifest_path": "",
        "h5_source_col": SOURCE_COL,
        "results_dir": str(results_dir),
        "raw_videos_dir": str(project_dir / "raw_videos"),
        "metadata_csv_path": str(meta_path),
    }
    (project_dir / "config.json").write_text(json.dumps(config))

    app_config = {"active_project": str(project_dir)}
    app_config_path = tmp_path / "app_config.json"
    app_config_path.write_text(json.dumps(app_config))

    import vieb_config as vc
    monkeypatch.setattr(vc, "_APP_CONFIG_PATH", str(app_config_path))
    monkeypatch.setattr(vc, "_CONFIG_PATH", str(project_dir / "config.json"))

    import compare
    # First run
    compare.cmd_extract(fps=30.0, use_wavelets=False)
    # Second run — should skip all 3
    compare.cmd_extract(fps=30.0, use_wavelets=False)

    with open(results_dir / "features" / "index.json") as f:
        index = json.load(f)

    expected_stems = [os.path.splitext(s)[0] for s in SOURCES]
    for stem in expected_stems:
        assert stem in index


def test_multi_key_h5_still_works(tmp_path, monkeypatch):
    """Multi-key H5 (one key per session) continues to work as before."""
    h5_path = tmp_path / "multi_key_pose.h5"
    rng = np.random.default_rng(99)

    keys = {"mouse_A": 30, "mouse_B": 25}
    with pd.HDFStore(str(h5_path), mode="w") as store:
        for i, (key, n_frames) in enumerate(keys.items()):
            data = {}
            for bp in BODYPARTS:
                data[f"{bp}_x"] = rng.random(n_frames) * 100
                data[f"{bp}_y"] = rng.random(n_frames) * 100
            store.put(key, pd.DataFrame(data), format="table")

    project_dir = tmp_path / "project"
    results_dir = project_dir / "results"
    project_dir.mkdir()

    meta_df = pd.DataFrame({
        "filename": ["mouse_A.mp4", "mouse_B.mp4"],
        "date": ["20200101", "20200101"],
        "box": [1, 1],
        "experiment": ["CFC", "CFC"],
        "day": [0, 0],
        "context": ["A", "A"],
        "no_shock": ["no", "no"],
        "animal_id": ["A", "B"],
        "fear": ["", ""],
    })
    meta_path = project_dir / "metadata.csv"
    meta_df.to_csv(meta_path, index=False)

    config = {
        "pose_source": "h5",
        "h5_path": str(h5_path),
        "h5_manifest_path": "",
        "h5_source_col": "",
        "results_dir": str(results_dir),
        "raw_videos_dir": str(project_dir / "raw_videos"),
        "metadata_csv_path": str(meta_path),
    }
    (project_dir / "config.json").write_text(json.dumps(config))

    app_config = {"active_project": str(project_dir)}
    app_config_path = tmp_path / "app_config.json"
    app_config_path.write_text(json.dumps(app_config))

    import vieb_config as vc
    monkeypatch.setattr(vc, "_APP_CONFIG_PATH", str(app_config_path))
    monkeypatch.setattr(vc, "_CONFIG_PATH", str(project_dir / "config.json"))

    import compare
    compare.cmd_extract(fps=30.0, use_wavelets=False)

    with open(results_dir / "features" / "index.json") as f:
        index = json.load(f)

    for stem in ("mouse_A", "mouse_B"):
        assert stem in index, f"Multi-key session {stem!r} missing"
        assert index[stem]["n_keypoints"] == len(BODYPARTS)
