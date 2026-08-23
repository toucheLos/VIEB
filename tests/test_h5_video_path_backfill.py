"""Tests for H5 extraction populating video_path (h5_manifest.load_video_paths,
compare.py's _cmd_extract_h5 wiring) and the --backfill-video-paths repair
command (compare.cmd_backfill_video_paths).
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from h5_manifest import load_video_paths  # noqa: E402

BODYPARTS = ["nose", "left_ear", "right_ear"]


def _make_flat_pose_df(n_frames: int, seed: int, source_file: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, np.ndarray] = {"source_file": np.array([source_file] * n_frames)}
    for bp in BODYPARTS:
        data[f"{bp}_x"] = rng.random(n_frames)
        data[f"{bp}_y"] = rng.random(n_frames)
        data[f"{bp}_likelihood"] = rng.random(n_frames)
    return pd.DataFrame(data)


def _make_dlc_df(n_frames: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = pd.MultiIndex.from_product(
        [["synthetic"], BODYPARTS, ["x", "y", "likelihood"]],
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


@pytest.fixture()
def synthetic_concat_h5(tmp_path):
    h5_path = tmp_path / "synthetic_concat_pose.h5"
    sessions = {
        "rat142.baseline.csv": 18,
        "rat143.baseline.csv": 21,
    }
    frames = [
        _make_flat_pose_df(n_frames, seed=i, source_file=source_file)
        for i, (source_file, n_frames) in enumerate(sessions.items())
    ]
    df = pd.concat(frames, ignore_index=True)
    with pd.HDFStore(h5_path, mode="w") as store:
        store.put("coords", df, format="table")
    return str(h5_path), sessions


def _write_project(tmp_path, monkeypatch, config: dict, meta_df: pd.DataFrame, project_name="project"):
    project_dir = tmp_path / project_name
    results_dir = project_dir / "results"
    project_dir.mkdir()

    meta_path = project_dir / "metadata.csv"
    meta_df.to_csv(meta_path, index=False)

    config = dict(config)
    config.setdefault("results_dir", str(results_dir))
    config.setdefault("raw_videos_dir", str(project_dir / "raw_videos"))
    config.setdefault("metadata_csv_path", str(meta_path))
    (project_dir / "config.json").write_text(json.dumps(config))

    app_config_path = tmp_path / f"app_config_{project_name}.json"
    app_config_path.write_text(json.dumps({"active_project": str(project_dir)}))

    import vieb_config as vc
    monkeypatch.setattr(vc, "_APP_CONFIG_PATH", str(app_config_path))
    monkeypatch.setattr(vc, "_CONFIG_PATH", str(project_dir / "config.json"))

    return project_dir, results_dir


# ---------------------------------------------------------------------------
# h5_manifest.load_video_paths()
# ---------------------------------------------------------------------------

def test_load_video_paths_finds_first_matching_column(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({
        "animal_id": ["142", "143"],
        "source_file": ["rat142.baseline.csv", "rat143.baseline.csv"],
        "source_path": ["/mnt/videos/rat142.mp4", "/mnt/videos/rat143.mp4"],
    }).to_csv(manifest_path, index=False)

    mapping = load_video_paths(str(manifest_path), value_col="source_file")

    from h5_manifest import _normalize
    assert mapping[_normalize("142")] == "/mnt/videos/rat142.mp4"
    assert mapping[_normalize("rat142.baseline.csv")] == "/mnt/videos/rat142.mp4"


def test_load_video_paths_no_matching_column_returns_empty(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({"animal_id": ["142"], "h5_key": ["mouse_9001"]}).to_csv(manifest_path, index=False)

    assert load_video_paths(str(manifest_path)) == {}


def test_load_video_paths_missing_manifest_returns_empty():
    assert load_video_paths("/does/not/exist.csv") == {}
    assert load_video_paths("") == {}


# ---------------------------------------------------------------------------
# compare.py::_cmd_extract_h5 — video_path resolution
# ---------------------------------------------------------------------------

def test_extract_h5_standard_populates_video_path_from_manifest(tmp_path, synthetic_h5, monkeypatch):
    h5_path, keys = synthetic_h5

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({
        "animal_id": ["9001", "9002"],
        "h5_key": ["mouse_9001", "mouse_9002"],
        "video_path": ["/remote/videos/mouse_9001.mp4", "/remote/videos/mouse_9002.mp4"],
    }).to_csv(manifest_path, index=False)

    meta_df = pd.DataFrame({
        "filename": ["mouse_9001.mp4", "mouse_9002.mp4"],
        "date": ["20250101", "20250101"], "box": [1, 1], "experiment": ["CFC", "CFC"],
        "day": [0, 0], "context": ["A", "A"], "no_shock": ["no", "no"],
        "animal_id": ["9001", "9002"], "fear": ["", ""],
    })
    config = {
        "pose_source": "h5", "h5_path": h5_path,
        "h5_manifest_path": str(manifest_path), "h5_source_col": "",
    }
    project_dir, results_dir = _write_project(tmp_path, monkeypatch, config, meta_df, "std_manifest")

    import compare
    compare.cmd_extract(fps=30.0, use_wavelets=False)

    with open(results_dir / "features" / "index.json") as f:
        index = json.load(f)

    assert index["mouse_9001"]["video_path"] == "/remote/videos/mouse_9001.mp4"
    assert index["mouse_9002"]["video_path"] == "/remote/videos/mouse_9002.mp4"


def test_extract_h5_concatenated_populates_video_path_from_manifest(tmp_path, synthetic_concat_h5, monkeypatch):
    h5_path, sessions = synthetic_concat_h5
    source_files = list(sessions.keys())

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({
        "source_file": source_files,
        "video_path": [
            f"/remote/videos/{os.path.splitext(sf)[0]}.mp4" for sf in source_files
        ],
    }).to_csv(manifest_path, index=False)

    meta_df = pd.DataFrame({
        "filename": ["rat142.mp4", "rat143.mp4"],
        "date": ["20250101", "20250101"], "box": [1, 1], "experiment": ["CFC", "CFC"],
        "day": [0, 0], "context": ["A", "A"], "no_shock": ["no", "no"],
        "animal_id": ["142", "143"], "fear": ["", ""],
        "source_file": source_files,
    })
    config = {
        "pose_source": "h5", "h5_path": h5_path,
        "h5_manifest_path": str(manifest_path), "h5_source_col": "source_file",
    }
    project_dir, results_dir = _write_project(tmp_path, monkeypatch, config, meta_df, "concat_manifest")

    import compare
    compare.cmd_extract(fps=30.0, use_wavelets=False)

    with open(results_dir / "features" / "index.json") as f:
        index = json.load(f)

    for source_file in source_files:
        stem = os.path.splitext(os.path.basename(source_file))[0]
        expected = f"/remote/videos/{stem}.mp4"
        assert index[stem]["video_path"] == expected


def test_extract_h5_falls_back_to_raw_videos_dir_extension_match(tmp_path, synthetic_h5, monkeypatch):
    h5_path, keys = synthetic_h5

    meta_df = pd.DataFrame({
        "filename": ["mouse_9001.mp4", "mouse_9002.mp4"],
        "date": ["20250101", "20250101"], "box": [1, 1], "experiment": ["CFC", "CFC"],
        "day": [0, 0], "context": ["A", "A"], "no_shock": ["no", "no"],
        "animal_id": ["9001", "9002"], "fear": ["", ""],
    })
    config = {
        "pose_source": "h5", "h5_path": h5_path,
        "h5_manifest_path": "", "h5_source_col": "",
    }
    project_dir, results_dir = _write_project(tmp_path, monkeypatch, config, meta_df, "extension_fallback")

    raw_dir = project_dir / "raw_videos"
    raw_dir.mkdir(parents=True)
    (raw_dir / "mouse_9001.mp4").touch()
    # mouse_9002 has no matching file — should stay None.

    import compare
    compare.cmd_extract(fps=30.0, use_wavelets=False)

    with open(results_dir / "features" / "index.json") as f:
        index = json.load(f)

    assert index["mouse_9001"]["video_path"] == str(raw_dir / "mouse_9001.mp4")
    assert index["mouse_9002"]["video_path"] is None


def test_extract_h5_video_path_none_when_unresolvable(tmp_path, synthetic_h5, monkeypatch):
    """No manifest, no matching file in raw_videos_dir -> video_path stays
    None, and extraction completes without crashing (expected/valid outcome)."""
    h5_path, keys = synthetic_h5
    meta_df = pd.DataFrame({
        "filename": ["mouse_9001.mp4", "mouse_9002.mp4"],
        "date": ["20250101", "20250101"], "box": [1, 1], "experiment": ["CFC", "CFC"],
        "day": [0, 0], "context": ["A", "A"], "no_shock": ["no", "no"],
        "animal_id": ["9001", "9002"], "fear": ["", ""],
    })
    config = {
        "pose_source": "h5", "h5_path": h5_path,
        "h5_manifest_path": "", "h5_source_col": "",
    }
    project_dir, results_dir = _write_project(tmp_path, monkeypatch, config, meta_df, "unresolvable")

    import compare
    compare.cmd_extract(fps=30.0, use_wavelets=False)

    with open(results_dir / "features" / "index.json") as f:
        index = json.load(f)

    assert index["mouse_9001"]["video_path"] is None
    assert index["mouse_9002"]["video_path"] is None
    # Extraction itself still succeeded for both sessions.
    assert (results_dir / "features" / "mouse_9001_features.npy").exists()
    assert (results_dir / "features" / "mouse_9002_features.npy").exists()


# ---------------------------------------------------------------------------
# compare.py::cmd_backfill_video_paths
# ---------------------------------------------------------------------------

def test_backfill_video_paths_updates_only_missing_entries_preserves_rest(tmp_path, monkeypatch):
    project_dir = tmp_path / "backfill_project"
    results_dir = project_dir / "results"
    features_dir = results_dir / "features"
    features_dir.mkdir(parents=True)
    raw_dir = project_dir / "raw_videos"
    raw_dir.mkdir(parents=True)
    (raw_dir / "sess_resolvable.mp4").touch()

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame({
        "animal_id": ["7"],
        "h5_key": ["h5key_7"],
        "video_path": ["/remote/videos/sess_manifest.mp4"],
    }).to_csv(manifest_path, index=False)

    (project_dir / "metadata.csv").write_text("filename\n")
    config = {
        "pose_source": "h5",
        "h5_path": str(tmp_path / "does_not_matter.h5"),
        "h5_manifest_path": str(manifest_path),
        "h5_source_col": "",
        "results_dir": str(results_dir),
        "raw_videos_dir": str(raw_dir),
        "metadata_csv_path": str(project_dir / "metadata.csv"),
    }
    (project_dir / "config.json").write_text(json.dumps(config))
    app_config_path = tmp_path / "app_config_backfill.json"
    app_config_path.write_text(json.dumps({"active_project": str(project_dir)}))

    import vieb_config as vc
    monkeypatch.setattr(vc, "_APP_CONFIG_PATH", str(app_config_path))
    monkeypatch.setattr(vc, "_CONFIG_PATH", str(project_dir / "config.json"))

    index = {
        "sess_already_had": {
            "video_path": "/already/set.mp4", "csv_path": None,
            "h5_path": str(tmp_path / "does_not_matter.h5"), "h5_key": "h5key_already",
            "n_frames": 10, "n_keypoints": 3, "n_features": 51,
            "features_path": str(features_dir / "sess_already_had_features.npy"),
        },
        "sess_resolvable": {
            "video_path": None, "csv_path": None,
            "h5_path": str(tmp_path / "does_not_matter.h5"), "h5_key": "h5key_resolvable",
            "n_frames": 20, "n_keypoints": 3, "n_features": 51,
            "features_path": str(features_dir / "sess_resolvable_features.npy"),
        },
        "h5key_7": {
            "video_path": None, "csv_path": None,
            "h5_path": str(tmp_path / "does_not_matter.h5"), "h5_key": "h5key_7",
            "n_frames": 15, "n_keypoints": 3, "n_features": 51,
            "features_path": str(features_dir / "h5key_7_features.npy"),
        },
        "sess_still_unresolvable": {
            "video_path": None, "csv_path": None,
            "h5_path": str(tmp_path / "does_not_matter.h5"), "h5_key": "h5key_nope",
            "n_frames": 5, "n_keypoints": 3, "n_features": 51,
            "features_path": str(features_dir / "sess_still_unresolvable_features.npy"),
        },
        "sess_no_h5": {
            "video_path": None, "csv_path": "/some/path.csv",
            "h5_path": None, "n_frames": 8, "n_keypoints": 3, "n_features": 51,
            "features_path": str(features_dir / "sess_no_h5_features.npy"),
        },
        "_meta": {"n_keypoints": 3, "n_features": 51, "use_wavelets": False,
                  "vieb_version": "1.0", "pose_source": "h5"},
    }
    index_path = features_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    import compare
    compare.cmd_backfill_video_paths()

    with open(index_path) as f:
        new_index = json.load(f)

    # Resolved via manifest's h5_key match.
    assert new_index["h5key_7"]["video_path"] == "/remote/videos/sess_manifest.mp4"
    # Resolved via raw_videos_dir extension fallback (stem == "sess_resolvable").
    assert new_index["sess_resolvable"]["video_path"] == str(raw_dir / "sess_resolvable.mp4")
    # Untouched: already had a video_path.
    assert new_index["sess_already_had"]["video_path"] == "/already/set.mp4"
    # Still unresolvable: stays None, no crash.
    assert new_index["sess_still_unresolvable"]["video_path"] is None
    # No h5_path -> out of scope, untouched.
    assert new_index["sess_no_h5"]["video_path"] is None

    # Every other field on every entry survives untouched.
    for stem in ("sess_already_had", "sess_still_unresolvable", "sess_no_h5"):
        for k, v in index[stem].items():
            if k == "video_path":
                continue
            assert new_index[stem][k] == v
    assert new_index["_meta"] == index["_meta"]
