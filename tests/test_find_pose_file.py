"""Verification tests for pose_io._find_pose_file() H5 routing."""

from __future__ import annotations

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
