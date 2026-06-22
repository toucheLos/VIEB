"""Tests for pose_io._find_pose_file() routing.

These tests avoid opening real H5 contents so they do not require pytables.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pose_io  # noqa: E402


def test_h5_source_returns_tuple_with_configured_key(tmp_path):
    """pose_source='h5' + valid h5_path returns (h5_path, configured key)."""
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

    assert result == (str(fake_h5), "/coords")


def test_h5_source_empty_h5_path_falls_back_to_csv_search():
    cfg = {"pose_source": "h5", "h5_path": ""}

    result = pose_io._find_pose_file("/nonexistent/video001.mp4", cfg=cfg)

    assert result is None or isinstance(result, str)


def test_csv_source_ignores_h5_path():
    cfg = {"pose_source": "csv", "h5_path": "/some/file.h5", "h5_key": "/coords"}

    result = pose_io._find_pose_file("/nonexistent/video001.mp4", cfg=cfg)

    assert result is None or isinstance(result, str)


def test_no_cfg_falls_back_to_csv_search():
    result = pose_io._find_pose_file("/nonexistent/video001.mp4", cfg=None)

    assert result is None or isinstance(result, str)


def test_csv_mode_finds_matching_dlc_csv(tmp_path):
    video_path = tmp_path / "test_video.mp4"
    video_path.touch()
    csv_path = tmp_path / "test_videoDLC_resnet50.csv"
    csv_path.touch()

    result = pose_io._find_pose_file(str(video_path), cfg={"pose_source": "csv"})

    assert result == str(csv_path)


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

    assert result == (str(fake_h5), "/session/42")


def test_h5_source_missing_file_still_returns_configured_tuple(tmp_path):
    """_find_pose_file routes by config; actual H5 readability is checked later."""
    h5_path = str(tmp_path / "nonexistent.h5")
    cfg = {"pose_source": "h5", "h5_path": h5_path, "h5_key": "/coords"}

    result = pose_io._find_pose_file(str(tmp_path / "mouse_9001.mp4"), cfg=cfg)

    assert result == (h5_path, "/coords")
