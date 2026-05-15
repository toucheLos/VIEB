"""
pose_io.py — DLC pose loading utilities for VIEB.

All pipeline scripts import load_pose and _find_dlc_csv from here.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd


def _find_dlc_csv(video_path: str) -> str | None:
    """Return the DLC-generated CSV for a given video file, or None."""
    stem = os.path.splitext(video_path)[0]
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    patterns = [
        f"{stem}*.csv",
        os.path.join(video_dir, f"{video_name}*.csv"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        matches = [m for m in matches if "metadata" not in os.path.basename(m).lower()]
        if matches:
            return matches[0]
    return None


def load_pose(csv_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load DLC CSV output into a pose array.

    Returns
    -------
    pose      : np.ndarray  shape (T, K, 2)  — x, y per keypoint per frame
    conf      : np.ndarray  shape (T, K)     — likelihood per keypoint per frame
    bodyparts : list[str]
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    bodyparts = df.columns.get_level_values(1).unique().tolist()
    scorer = df.columns.get_level_values(0)[0]

    T = len(df)
    K = len(bodyparts)
    pose = np.zeros((T, K, 2))
    conf = np.zeros((T, K))

    for k, bp in enumerate(bodyparts):
        pose[:, k, 0] = df[(scorer, bp, "x")].values
        pose[:, k, 1] = df[(scorer, bp, "y")].values
        if (scorer, bp, "likelihood") in df.columns:
            conf[:, k] = df[(scorer, bp, "likelihood")].values
        else:
            conf[:, k] = 1.0

    return pose, conf, bodyparts
