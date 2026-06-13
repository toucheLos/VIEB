"""Tests for the H5 pose loading path (pose_io.inspect_h5/load_pose_h5,
h5_manifest.resolve_h5_key) and the "virtual H5" extract mode in compare.py.

Uses a small synthetic H5 file written via pandas.HDFStore with DLC-style
MultiIndex columns, since no real H5 pose file is available in this
environment.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pose_io  # noqa: E402
from h5_manifest import load_manifest, resolve_h5_key  # noqa: E402

BODYPARTS = ["nose", "left_ear", "right_ear"]
SCORER = "synthetic"


def _make_dlc_df(n_frames: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = pd.MultiIndex.from_product(
        [[SCORER], BODYPARTS, ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    data = rng.random((n_frames, len(BODYPARTS) * 3))
    return pd.DataFrame(data, columns=cols)


@pytest.fixture()
def synthetic_h5(tmp_path):
    h5_path = tmp_path / "synthetic_pose.h5"
    keys = {"mouse_9001": 40, "mouse_9002": 55}
    with pd.HDFStore(h5_path, mode="w") as store:
        for i, (key, n_frames) in enumerate(keys.items()):
            store.put(key, _make_dlc_df(n_frames, seed=i), format="table")
    return str(h5_path), keys


def test_inspect_h5(synthetic_h5):
    h5_path, keys = synthetic_h5
    info = pose_io.inspect_h5(h5_path)
    assert set(info["keys"]) == set(keys.keys())
    for key, n_frames in keys.items():
        details = info["details"][key]
        assert details["n_frames"] == n_frames
        assert len(details["columns"]) == len(BODYPARTS) * 3


def test_load_pose_h5_exact_key(synthetic_h5):
    h5_path, keys = synthetic_h5
    for key, n_frames in keys.items():
        pose, conf, bodyparts = pose_io.load_pose_h5(h5_path, key=key)
        assert pose.shape == (n_frames, len(BODYPARTS), 2)
        assert conf.shape == (n_frames, len(BODYPARTS))
        assert bodyparts == BODYPARTS


def test_load_pose_h5_requires_key_when_ambiguous(synthetic_h5):
    h5_path, _ = synthetic_h5
    with pytest.raises(ValueError):
        pose_io.load_pose_h5(h5_path)


def test_resolve_h5_key_exact_match(synthetic_h5):
    h5_path, keys = synthetic_h5
    h5_keys = list(keys.keys())
    row = {"filename": "mouse_9001.mp4", "animal_id": "9001"}
    key, strategy = resolve_h5_key(row, h5_keys, manifest=None, ordinal_index=0)
    assert key == "mouse_9001"
    assert strategy == "exact"


def test_resolve_h5_key_manifest_match(tmp_path, synthetic_h5):
    h5_path, keys = synthetic_h5
    h5_keys = list(keys.keys())

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {"animal_id": ["7777"], "h5_key": ["mouse_9002"]}
    ).to_csv(manifest_path, index=False)
    manifest = load_manifest(str(manifest_path))

    row = {"filename": "", "animal_id": "7777"}
    key, strategy = resolve_h5_key(row, h5_keys, manifest=manifest, ordinal_index=0)
    assert key == "mouse_9002"
    assert strategy == "manifest"


def test_resolve_h5_key_ordinal_fallback(synthetic_h5):
    h5_path, keys = synthetic_h5
    h5_keys = list(keys.keys())
    row = {"filename": "totally_unrelated.mp4", "animal_id": "unknown"}
    key, strategy = resolve_h5_key(row, h5_keys, manifest=None, ordinal_index=1)
    assert key == h5_keys[1]
    assert strategy == "ordinal"


def test_resolve_h5_key_no_match_raises(synthetic_h5):
    h5_path, keys = synthetic_h5
    h5_keys = list(keys.keys())
    row = {"filename": "totally_unrelated.mp4", "animal_id": "unknown"}
    with pytest.raises(ValueError):
        resolve_h5_key(row, h5_keys, manifest=None, ordinal_index=99)


def test_extract_h5_mode_end_to_end(tmp_path, synthetic_h5, monkeypatch):
    """End-to-end: compare.py --extract with pose_source='h5' against a
    synthetic project produces index.json entries with video_path=None."""
    h5_path, keys = synthetic_h5

    project_dir = tmp_path / "project"
    results_dir = project_dir / "results"
    project_dir.mkdir()

    meta_df = pd.DataFrame(
        {
            "filename": ["mouse_9001.mp4", "mouse_9002.mp4"],
            "date": ["20250101", "20250101"],
            "box": [1, 1],
            "experiment": ["CFC", "CFC"],
            "day": [0, 0],
            "context": ["A", "A"],
            "no_shock": ["no", "no"],
            "animal_id": ["9001", "9002"],
            "fear": ["", ""],
        }
    )
    meta_path = project_dir / "metadata.csv"
    meta_df.to_csv(meta_path, index=False)

    config = {
        "pose_source": "h5",
        "h5_path": h5_path,
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

    index_path = results_dir / "features" / "index.json"
    assert index_path.exists()
    with open(index_path) as f:
        index = json.load(f)

    assert index["_meta"]["pose_source"] == "h5"
    for stem in ("mouse_9001", "mouse_9002"):
        entry = index[stem]
        assert entry["video_path"] is None
        assert entry["h5_path"] == h5_path
        assert entry["n_keypoints"] == len(BODYPARTS)
        features_path = results_dir / "features" / f"{stem}_features.npy"
        assert features_path.exists()
        arr = np.load(features_path)
        assert arr.shape[1] == entry["n_features"]
