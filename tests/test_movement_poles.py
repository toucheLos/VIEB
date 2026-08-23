"""Tests for movement pole computation in state_characterizer.py."""

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
# Unit tests: compute_movement_poles
# ---------------------------------------------------------------------------

def _make_centers(speed_vals: list[float], n_features: int = 50, n_kp: int = 8) -> np.ndarray:
    """Build fake cluster centers with centroid_speed at the positional index."""
    speed_idx = n_kp + n_kp * (n_kp - 1) // 2
    centers = np.zeros((len(speed_vals), n_features))
    for i, v in enumerate(speed_vals):
        centers[i, speed_idx] = v
    return centers


def _feature_names(n_features: int = 50) -> list[str]:
    return [f"feature_{i}" for i in range(n_features)]


def _feature_names_with_speed(n_features: int = 20, speed_idx: int = 36) -> list[str]:
    names = [f"feature_{i}" for i in range(n_features)]
    if speed_idx < n_features:
        names[speed_idx] = "centroid_speed"
    return names


def test_low_high_poles_from_clear_centers():
    """Three states with clearly separated speeds → correct pole state IDs."""
    centers = _make_centers([-2.0, 0.0, 3.0])
    names = _feature_names_with_speed(n_features=50, speed_idx=36)
    poles = sc.compute_movement_poles(centers, names, pd.DataFrame(), n_keypoints=8)
    assert poles["low_motion"]["state_id"] == 0
    assert poles["high_motion"]["state_id"] == 2
    assert poles["low_motion"]["type"] == "state"
    assert poles["high_motion"]["type"] == "state"


def test_poles_speed_zscore_values():
    """speed_zscore stored in poles matches the center value at the speed index."""
    centers = _make_centers([-1.5, 0.0, 2.5])
    names = _feature_names_with_speed(n_features=50, speed_idx=36)
    poles = sc.compute_movement_poles(centers, names, pd.DataFrame(), n_keypoints=8)
    assert abs(poles["low_motion"]["speed_zscore"] - (-1.5)) < 1e-4
    assert abs(poles["high_motion"]["speed_zscore"] - 2.5) < 1e-4


def test_poles_ambiguity_warning():
    """Two states with nearly identical low speeds → low_motion warning non-empty."""
    # States 0 and 1 differ by only 0.05 std in speed
    centers = _make_centers([-2.0, -1.98, 3.0])
    names = _feature_names_with_speed(n_features=50, speed_idx=36)
    poles = sc.compute_movement_poles(centers, names, pd.DataFrame(), n_keypoints=8)
    assert poles["low_motion"]["warning"] != ""


def test_poles_no_warning_when_clear_separation():
    """Clear separation → no ambiguity warning."""
    centers = _make_centers([-3.0, 0.0, 3.0])
    names = _feature_names_with_speed(n_features=50, speed_idx=36)
    poles = sc.compute_movement_poles(centers, names, pd.DataFrame(), n_keypoints=8)
    assert poles["low_motion"]["warning"] == ""
    assert poles["high_motion"]["warning"] == ""


def test_poles_missing_speed_feature():
    """No 'centroid_speed' in feature_names and n_keypoints=0 → unavailable."""
    centers = np.zeros((3, 10))
    names = [f"feat_{i}" for i in range(10)]
    poles = sc.compute_movement_poles(centers, names, pd.DataFrame(), n_keypoints=0)
    assert poles["low_motion"]["type"] == "unavailable"
    assert poles["high_motion"]["type"] == "unavailable"
    assert poles["low_motion"]["state_id"] is None


def test_poles_empty_centers():
    """Zero states → unavailable poles, no crash."""
    centers = np.zeros((0, 20))
    poles = sc.compute_movement_poles(centers, [], pd.DataFrame(), n_keypoints=8)
    assert poles["low_motion"]["type"] == "unavailable"
    assert poles["high_motion"]["type"] == "unavailable"


def test_poles_bout_count_from_bouts_df():
    """n_bouts populated from bouts_df when available."""
    centers = _make_centers([-1.0, 0.0, 2.0])
    names = _feature_names_with_speed(n_features=50, speed_idx=36)
    bouts = pd.DataFrame({"state": [0, 0, 0, 2, 2]})
    poles = sc.compute_movement_poles(centers, names, bouts, n_keypoints=8)
    assert poles["low_motion"]["n_bouts"] == 3
    assert poles["high_motion"]["n_bouts"] == 2


def test_poles_feature_name_lookup_takes_priority():
    """If 'centroid_speed' appears at an explicit index, that overrides positional formula."""
    n_features = 50
    # Put centroid_speed at index 5 (not the positional index 36)
    names = [f"feat_{i}" for i in range(n_features)]
    names[5] = "centroid_speed"
    centers = np.zeros((3, n_features))
    centers[0, 5] = -2.0  # state 0: lowest at index 5
    centers[2, 5] = 3.0   # state 2: highest at index 5
    poles = sc.compute_movement_poles(centers, names, pd.DataFrame(), n_keypoints=8)
    assert poles["low_motion"]["state_id"] == 0
    assert poles["high_motion"]["state_id"] == 2


# ---------------------------------------------------------------------------
# Integration tests: run() writes files
# ---------------------------------------------------------------------------

def _setup_run(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    results = tmp_path / "results"
    shared = results / "shared"
    features = results / "features"
    char = results / "characterization"
    for d in (shared, features, char):
        d.mkdir(parents=True)

    n_kp = 8
    n_features = 50
    speed_idx = n_kp + n_kp * (n_kp - 1) // 2  # 36

    centers = np.zeros((3, n_features))
    centers[0, speed_idx] = -2.0
    centers[1, speed_idx] = 0.0
    centers[2, speed_idx] = 3.0

    cluster_info = {"n_clusters": 3, "cluster_centers": centers.tolist()}
    (shared / "cluster_info.json").write_text(json.dumps(cluster_info))

    feature_names = [f"feat_{i}" for i in range(n_features)]
    feature_names[speed_idx] = "centroid_speed"
    index_meta = {"n_features": n_features, "n_keypoints": n_kp, "feature_names": feature_names}
    (features / "index.json").write_text(json.dumps({"_meta": index_meta}))

    return results, shared, features, char


def test_poles_json_written_by_run(tmp_path):
    results, shared, features, char = _setup_run(tmp_path)
    out = sc.run(str(results), str(shared), str(features))
    assert out["errors"] == []
    poles_path = char / "poles.json"
    assert poles_path.exists()
    poles = json.loads(poles_path.read_text())
    assert poles["low_motion"]["state_id"] == 0
    assert poles["high_motion"]["state_id"] == 2


def test_movement_poles_csv_written_by_run(tmp_path):
    results, shared, features, char = _setup_run(tmp_path)
    sc.run(str(results), str(shared), str(features))
    csv_path = char / "movement_poles.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert "pole_name" in df.columns
    assert "state_id" in df.columns
    assert "speed_zscore" in df.columns
    assert set(df["pole_name"]) == {"low_motion", "high_motion"}


def test_load_outputs_includes_poles(tmp_path):
    results, shared, features, char = _setup_run(tmp_path)
    sc.run(str(results), str(shared), str(features))
    out = sc.load_outputs(str(results))
    assert "poles" in out
    assert out["poles"]["low_motion"]["state_id"] == 0
