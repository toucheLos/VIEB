<<<<<<< HEAD
"""Tests for _find_pose_file() routing: when pose_source='h5' and h5_path
is set, _find_pose_file() should skip CSV lookup and return an
(h5_path, h5_key) tuple.

Mocks inspect_h5 and h5_manifest since pytables/h5py are not installed
in this test environment.
"""
=======
"""Verification tests for pose_io._find_pose_file() H5 routing."""
>>>>>>> 81e1fbe94c256750a3bc620286ae81ebb1673e42

from __future__ import annotations

import os
import sys
<<<<<<< HEAD
from unittest.mock import patch, MagicMock

=======

import pandas as pd
>>>>>>> 81e1fbe94c256750a3bc620286ae81ebb1673e42
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

<<<<<<< HEAD
import pose_io  # noqa: E402


@pytest.fixture()
def fake_h5(tmp_path):
    """Create a dummy .h5 file on disk (just needs to exist for os.path.exists)."""
    h5_path = tmp_path / "all_trials.h5"
    h5_path.write_bytes(b"fake")
    return str(h5_path)


def _mock_inspect_h5(h5_path):
    return {
        "keys": ["mouse_9001", "mouse_9002"],
        "details": {
            "mouse_9001": {"columns": ["x", "y"], "n_frames": 40},
            "mouse_9002": {"columns": ["x", "y"], "n_frames": 55},
        },
    }


def test_find_pose_file_h5_returns_tuple(tmp_path, fake_h5):
    """When pose_source='h5' and h5_path is valid, _find_pose_file returns
    an (h5_path, key) tuple for a matching video stem."""
    video_path = str(tmp_path / "mouse_9001.mp4")

    cfg = {
        "pose_source": "h5",
        "h5_path": fake_h5,
        "h5_key": "",
        "h5_manifest_path": "",
        "h5_source_col": "",
    }

    with patch.object(pose_io, "inspect_h5", side_effect=_mock_inspect_h5):
        result = pose_io._find_pose_file(video_path, cfg=cfg)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}: {result}"
    assert result[0] == fake_h5
    assert result[1] == "mouse_9001"


def test_find_pose_file_h5_no_match_uses_ordinal_fallback(tmp_path, fake_h5):
    """When the video stem doesn't match any H5 key by name, resolve_h5_key
    falls back to ordinal matching (index 0 → first key)."""
    video_path = str(tmp_path / "totally_unknown_video.mp4")

    cfg = {
        "pose_source": "h5",
        "h5_path": fake_h5,
        "h5_key": "",
        "h5_manifest_path": "",
        "h5_source_col": "",
    }

    with patch.object(pose_io, "inspect_h5", side_effect=_mock_inspect_h5):
        result = pose_io._find_pose_file(video_path, cfg=cfg)

    # ordinal fallback at index 0 returns the first key
    assert isinstance(result, tuple)
    assert result[0] == fake_h5
    assert result[1] == "mouse_9001"


def test_find_pose_file_csv_mode_falls_through(tmp_path):
    """When pose_source='csv', _find_pose_file falls through to _find_dlc_csv."""
    video_path = tmp_path / "test_video.mp4"
    video_path.touch()
    csv_path = tmp_path / "test_videoDLC_resnet50.csv"
    csv_path.touch()

    cfg = {"pose_source": "csv"}

    result = pose_io._find_pose_file(str(video_path), cfg=cfg)
    assert result is not None
    assert result.endswith(".csv")


def test_find_pose_file_empty_cfg_falls_through(tmp_path):
    """When cfg is empty (no pose_source), falls through to _find_dlc_csv."""
    video_path = tmp_path / "my_video.mp4"
    video_path.touch()

    result = pose_io._find_pose_file(str(video_path), cfg={})
    assert result is None  # no CSV exists either


def test_find_pose_file_h5_with_manifest(tmp_path, fake_h5):
    """When a manifest maps an unrelated video stem to an H5 key, it resolves."""
    import pandas as pd

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {"animal_id": ["custom_animal"], "h5_key": ["mouse_9002"]}
    ).to_csv(manifest_path, index=False)

    video_path = str(tmp_path / "custom_animal.mp4")

    cfg = {
        "pose_source": "h5",
        "h5_path": fake_h5,
        "h5_key": "",
        "manifest_path": str(manifest_path),
        "h5_source_col": "",
    }

    with patch.object(pose_io, "inspect_h5", side_effect=_mock_inspect_h5):
        result = pose_io._find_pose_file(video_path, cfg=cfg)

    assert isinstance(result, tuple)
    assert result[0] == fake_h5
    assert result[1] == "mouse_9002"


def test_find_pose_file_h5_missing_file_returns_none(tmp_path):
    """When h5_path doesn't exist on disk, returns None."""
    video_path = str(tmp_path / "mouse_9001.mp4")

    cfg = {
        "pose_source": "h5",
        "h5_path": str(tmp_path / "nonexistent.h5"),
    }

    result = pose_io._find_pose_file(video_path, cfg=cfg)
    assert result is None
=======
import pose_io


def test_h5_source_returns_tuple(tmp_path):
    """pose_source='h5' + valid h5_path → (h5_path, key) tuple, no CSV search."""
    fake_h5 = tmp_path / "pose.h5"
    fake_h5.write_bytes(b"")

    cfg = {
        "pose_source": "h5",
        "h5_path": str(fake_h5),
        "h5_key": "/coords",
        "h5_source_col": "source_file",
        "h5_frame_col": "Frame Number",
        "manifest_path": "",
        "h5_manifest_path": "",
    }
    result = pose_io._find_pose_file("/data/video001.mp4", cfg=cfg)

    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    h5_out, key_out = result
    assert h5_out == str(fake_h5)
    assert key_out == "/coords"


def test_h5_source_empty_h5_path_falls_back():
    """pose_source='h5' but h5_path='' falls back to per-video CSV search."""
    cfg = {"pose_source": "h5", "h5_path": ""}
    result = pose_io._find_pose_file("/nonexistent/video001.mp4", cfg=cfg)
    assert result is None or isinstance(result, str)


def test_csv_source_ignores_h5_path():
    """pose_source='csv' always falls back to _find_dlc_csv regardless of h5_path."""
    cfg = {"pose_source": "csv", "h5_path": "/some/file.h5"}
    result = pose_io._find_pose_file("/nonexistent/video001.mp4", cfg=cfg)
    assert result is None or isinstance(result, str)


def test_no_cfg_falls_back():
    """cfg=None delegates to _find_dlc_csv."""
    result = pose_io._find_pose_file("/nonexistent/video001.mp4", cfg=None)
    assert result is None or isinstance(result, str)


def test_manifest_lookup_overrides_default_key(tmp_path):
    """Manifest CSV entry for the video stem wins over cfg['h5_key']."""
    fake_h5 = tmp_path / "pose.h5"
    fake_h5.write_bytes(b"")

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        {"filename": ["session42.mp4"], "h5_key": ["/session/42"]}
    ).to_csv(manifest_path, index=False)

    cfg = {
        "pose_source": "h5",
        "h5_path": str(fake_h5),
        "h5_key": "/default",
        "manifest_path": str(manifest_path),
        "h5_manifest_path": "",
    }
    result = pose_io._find_pose_file("/data/session42.mp4", cfg=cfg)

    assert isinstance(result, tuple)
    assert result[0] == str(fake_h5)
    assert result[1] == "/session/42"
>>>>>>> 81e1fbe94c256750a3bc620286ae81ebb1673e42
