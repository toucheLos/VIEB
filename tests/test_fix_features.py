"""Tests for compare.py --fix-features: re-extracting only the videos whose
feature dimension doesn't match the current config.json settings."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BODYPARTS = ["nose", "left_ear", "right_ear", "tail_base", "center",
              "left_hip", "right_hip", "tail_tip"]
SCORER = "synthetic"


def _make_pose_df(n_frames: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = pd.MultiIndex.from_product(
        [[SCORER], BODYPARTS, ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    data = rng.random((n_frames, len(BODYPARTS) * 3))
    return pd.DataFrame(data, columns=cols)


def _setup_project(tmp_path, monkeypatch, use_wavelets: bool):
    project_dir = tmp_path / "project"
    results_dir = project_dir / "results"
    raw_dir = project_dir / "raw_videos"
    features_dir = results_dir / "features"
    features_dir.mkdir(parents=True)
    raw_dir.mkdir(parents=True)

    meta_path = project_dir / "metadata.csv"
    pd.DataFrame({
        "filename": ["vidA.mp4", "vidB.mp4"],
        "date": ["20250101", "20250101"],
        "box": [1, 1],
        "experiment": ["CFC", "CFC"],
        "day": [0, 0],
        "context": ["A", "A"],
        "no_shock": ["no", "no"],
        "animal_id": ["1", "2"],
        "fear": ["", ""],
    }).to_csv(meta_path, index=False)

    config = {
        "pose_source": "csv",
        "results_dir": str(results_dir),
        "raw_videos_dir": str(raw_dir),
        "metadata_csv_path": str(meta_path),
        "use_wavelets": use_wavelets,
    }
    config_path = project_dir / "config.json"
    config_path.write_text(json.dumps(config))

    app_config_path = tmp_path / "app_config.json"
    app_config_path.write_text(json.dumps({"active_project": str(project_dir)}))

    import vieb_config as vc
    monkeypatch.setattr(vc, "_APP_CONFIG_PATH", str(app_config_path))
    monkeypatch.setattr(vc, "_CONFIG_PATH", str(config_path))

    import compare
    monkeypatch.setattr(compare, "_load_extractor_config", lambda: ({}, [], BODYPARTS))

    # Write CSV pose data + dummy "video" files for both stems.
    for stem, seed in (("vidA", 1), ("vidB", 2)):
        (raw_dir / f"{stem}.mp4").touch()
        df = _make_pose_df(n_frames=60, seed=seed)
        csv_path = raw_dir / f"{stem}DLC_resnet50.csv"
        df.to_csv(csv_path)

    return compare, project_dir, results_dir, features_dir


def _extract_features_array(compare, results_dir, stem, video_path, csv_path, use_wavelets):
    from ml import PoseFeatureExtractor
    from pose_io import load_pose

    pose, conf, _ = load_pose(str(csv_path))
    extractor = PoseFeatureExtractor(
        fps=30.0, use_wavelets=use_wavelets,
        keypoint_roles={}, bodypart_names=BODYPARTS, object_keypoints=[],
    )
    feats = extractor._flatten_features(extractor.extract_features(pose, confidence=conf))
    out_path = results_dir / "features" / f"{stem}_features.npy"
    np.save(out_path, feats.astype(np.float32))
    return feats, out_path, pose


def test_fix_features_reextracts_mismatched_videos(tmp_path, monkeypatch):
    compare, project_dir, results_dir, features_dir = _setup_project(
        tmp_path, monkeypatch, use_wavelets=True
    )

    raw_dir = project_dir / "raw_videos"

    # vidA: extracted WITHOUT wavelets (mismatched, 51D)
    feats_a, path_a, pose_a = _extract_features_array(
        compare, results_dir, "vidA",
        raw_dir / "vidA.mp4", raw_dir / "vidADLC_resnet50.csv", use_wavelets=False,
    )
    # vidB: extracted WITH wavelets (matches current config, 91D)
    feats_b, path_b, pose_b = _extract_features_array(
        compare, results_dir, "vidB",
        raw_dir / "vidB.mp4", raw_dir / "vidBDLC_resnet50.csv", use_wavelets=True,
    )

    assert feats_a.shape[1] != feats_b.shape[1]

    index = {
        "vidA": {
            "video_path": str(raw_dir / "vidA.mp4"),
            "csv_path": str(raw_dir / "vidADLC_resnet50.csv"),
            "n_frames": int(pose_a.shape[0]),
            "n_keypoints": int(pose_a.shape[1]),
            "n_features": int(feats_a.shape[1]),
            "features_path": str(path_a),
        },
        "vidB": {
            "video_path": str(raw_dir / "vidB.mp4"),
            "csv_path": str(raw_dir / "vidBDLC_resnet50.csv"),
            "n_frames": int(pose_b.shape[0]),
            "n_keypoints": int(pose_b.shape[1]),
            "n_features": int(feats_b.shape[1]),
            "features_path": str(path_b),
        },
        "_meta": {
            "n_keypoints": int(pose_a.shape[1]),
            "n_features": int(feats_a.shape[1]),
            "use_wavelets": False,
            "vieb_version": "1.0",
        },
    }
    index_path = features_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    compare.cmd_fix_features(fps=30.0)

    with open(index_path) as f:
        new_index = json.load(f)

    # Both videos should now agree on the wavelet-enabled (91D) dimension.
    assert new_index["vidA"]["n_features"] == feats_b.shape[1]
    assert new_index["vidB"]["n_features"] == feats_b.shape[1]
    assert new_index["_meta"]["n_features"] == feats_b.shape[1]
    assert new_index["_meta"]["use_wavelets"] is True

    reloaded_a = np.load(path_a)
    assert reloaded_a.shape[1] == feats_b.shape[1]


def test_fix_features_noop_when_consistent(tmp_path, monkeypatch, capsys):
    compare, project_dir, results_dir, features_dir = _setup_project(
        tmp_path, monkeypatch, use_wavelets=False
    )
    raw_dir = project_dir / "raw_videos"

    feats_a, path_a, pose_a = _extract_features_array(
        compare, results_dir, "vidA",
        raw_dir / "vidA.mp4", raw_dir / "vidADLC_resnet50.csv", use_wavelets=False,
    )
    feats_b, path_b, pose_b = _extract_features_array(
        compare, results_dir, "vidB",
        raw_dir / "vidB.mp4", raw_dir / "vidBDLC_resnet50.csv", use_wavelets=False,
    )

    index = {
        "vidA": {
            "video_path": str(raw_dir / "vidA.mp4"),
            "csv_path": str(raw_dir / "vidADLC_resnet50.csv"),
            "n_frames": int(pose_a.shape[0]),
            "n_keypoints": int(pose_a.shape[1]),
            "n_features": int(feats_a.shape[1]),
            "features_path": str(path_a),
        },
        "vidB": {
            "video_path": str(raw_dir / "vidB.mp4"),
            "csv_path": str(raw_dir / "vidBDLC_resnet50.csv"),
            "n_frames": int(pose_b.shape[0]),
            "n_keypoints": int(pose_b.shape[1]),
            "n_features": int(feats_b.shape[1]),
            "features_path": str(path_b),
        },
        "_meta": {
            "n_keypoints": int(pose_a.shape[1]),
            "n_features": int(feats_a.shape[1]),
            "use_wavelets": False,
            "vieb_version": "1.0",
        },
    }
    index_path = features_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    compare.cmd_fix_features(fps=30.0)
    captured = capsys.readouterr()
    assert "Nothing to fix" in captured.out


def test_fix_features_preserves_and_reconstructs_meta_fields(tmp_path, monkeypatch):
    """cmd_fix_features must not drop existing _meta fields and must rebuild feature_names."""
    compare, project_dir, results_dir, features_dir = _setup_project(
        tmp_path, monkeypatch, use_wavelets=True
    )
    raw_dir = project_dir / "raw_videos"

    feats_a, path_a, pose_a = _extract_features_array(
        compare, results_dir, "vidA",
        raw_dir / "vidA.mp4", raw_dir / "vidADLC_resnet50.csv", use_wavelets=False,
    )
    feats_b, path_b, pose_b = _extract_features_array(
        compare, results_dir, "vidB",
        raw_dir / "vidB.mp4", raw_dir / "vidBDLC_resnet50.csv", use_wavelets=True,
    )

    index = {
        "vidA": {
            "video_path": str(raw_dir / "vidA.mp4"),
            "csv_path": str(raw_dir / "vidADLC_resnet50.csv"),
            "n_frames": int(pose_a.shape[0]),
            "n_keypoints": int(pose_a.shape[1]),
            "n_features": int(feats_a.shape[1]),
            "features_path": str(path_a),
        },
        "vidB": {
            "video_path": str(raw_dir / "vidB.mp4"),
            "csv_path": str(raw_dir / "vidBDLC_resnet50.csv"),
            "n_frames": int(pose_b.shape[0]),
            "n_keypoints": int(pose_b.shape[1]),
            "n_features": int(feats_b.shape[1]),
            "features_path": str(path_b),
        },
        "_meta": {
            "n_keypoints": int(pose_a.shape[1]),
            "n_features": int(feats_a.shape[1]),
            "use_wavelets": False,
            "vieb_version": "1.0",
            "pose_source": "csv",
            "project_name": "test_project",  # custom field that must survive
        },
    }
    index_path = features_dir / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    compare.cmd_fix_features(fps=30.0)

    with open(index_path) as f:
        new_index = json.load(f)

    meta = new_index["_meta"]
    # Updated fields
    assert meta["use_wavelets"] is True
    assert meta["n_features"] == feats_b.shape[1]
    # Preserved fields must not be dropped
    assert meta.get("vieb_version") == "1.0"
    assert meta.get("pose_source") == "csv"
    assert meta.get("project_name") == "test_project"
    # feature_names must be rebuilt (non-empty list of strings)
    assert isinstance(meta.get("feature_names"), list) and len(meta["feature_names"]) > 0
    assert meta["feature_names"][0].startswith("speed_")
