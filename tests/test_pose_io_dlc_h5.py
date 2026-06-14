"""Tests for per-video DLC pose file discovery and loading, covering both
.csv and .h5 output formats (DLC writes .h5 by default, .csv when
save_as_csv=True is passed to analyze_videos).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pose_io  # noqa: E402

BODYPARTS = ["nose", "left_ear", "right_ear", "tail_base"]
SCORER = "DLC_resnet50_VIEBshuffle2_100000"


def _make_dlc_df(n_frames: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = pd.MultiIndex.from_product(
        [[SCORER], BODYPARTS, ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    data = rng.random((n_frames, len(BODYPARTS) * 3))
    return pd.DataFrame(data, columns=cols)


def test_find_dlc_csv_finds_h5_when_no_csv(tmp_path):
    video_path = tmp_path / "mouse_0001.mp4"
    video_path.touch()

    df = _make_dlc_df(n_frames=20)
    h5_path = tmp_path / f"mouse_0001{SCORER}.h5"
    df.to_hdf(h5_path, key="df_with_missing", format="table")

    found = pose_io._find_dlc_csv(str(video_path))
    assert found == str(h5_path)


def test_find_dlc_csv_prefers_csv_over_h5(tmp_path):
    video_path = tmp_path / "mouse_0002.mp4"
    video_path.touch()

    df = _make_dlc_df(n_frames=10)
    csv_path = tmp_path / f"mouse_0002{SCORER}.csv"
    h5_path = tmp_path / f"mouse_0002{SCORER}.h5"
    df.to_csv(csv_path)
    df.to_hdf(h5_path, key="df_with_missing", format="table")

    found = pose_io._find_dlc_csv(str(video_path))
    assert found == str(csv_path)


def test_find_dlc_csv_excludes_full_h5_sidecar(tmp_path):
    video_path = tmp_path / "mouse_0003.mp4"
    video_path.touch()

    df = _make_dlc_df(n_frames=15)
    full_h5_path = tmp_path / f"mouse_0003{SCORER}_full.h5"
    df.to_hdf(full_h5_path, key="df_with_missing", format="table")

    # Only the "_full.h5" sidecar exists — should not be picked up.
    found = pose_io._find_dlc_csv(str(video_path))
    assert found is None

    # Now add the real pose .h5; it should be found instead of "_full.h5".
    h5_path = tmp_path / f"mouse_0003{SCORER}.h5"
    df.to_hdf(h5_path, key="df_with_missing", format="table")
    found = pose_io._find_dlc_csv(str(video_path))
    assert found == str(h5_path)


def test_load_pose_from_h5_returns_correct_shape(tmp_path):
    n_frames = 30
    df = _make_dlc_df(n_frames=n_frames)
    h5_path = tmp_path / f"mouse_0004{SCORER}.h5"
    df.to_hdf(h5_path, key="df_with_missing", format="table")

    pose, conf, bodyparts = pose_io.load_pose(str(h5_path))

    assert pose.shape == (n_frames, len(BODYPARTS), 2)
    assert conf.shape == (n_frames, len(BODYPARTS))
    assert bodyparts == BODYPARTS

    expected_pose, expected_conf, _ = pose_io._pose_from_dlc_df(df)
    np.testing.assert_allclose(pose, expected_pose)
    np.testing.assert_allclose(conf, expected_conf)
