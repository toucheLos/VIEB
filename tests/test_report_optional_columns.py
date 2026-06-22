"""Tests for optional metadata columns in compare.py --report."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _setup_report_project(tmp_path, monkeypatch, metadata: pd.DataFrame):
    project_dir = tmp_path / "project"
    results_dir = project_dir / "results"
    features_dir = results_dir / "features"
    shared_dir = results_dir / "shared"
    features_dir.mkdir(parents=True)
    shared_dir.mkdir(parents=True)

    stems = metadata["filename"].str.replace(r"\.mp4$", "", regex=True).tolist()
    index = {"_meta": {"n_keypoints": 2, "n_features": 10, "use_wavelets": False}}
    label_sets = [
        np.array([0, 0, 1, 1, 0, 1], dtype=np.int32),
        np.array([1, 1, 1, 0, 0, 0], dtype=np.int32),
        np.array([0, 1, 0, 1, 0, 1], dtype=np.int32),
        np.array([1, 0, 1, 0, 1, 0], dtype=np.int32),
    ]
    for i, stem in enumerate(stems):
        np.save(shared_dir / f"{stem}_labels.npy", label_sets[i % len(label_sets)])
        np.save(shared_dir / f"{stem}_probs.npy", np.full(6, 0.95, dtype=np.float32))
        index[stem] = {
            "video_path": None,
            "csv_path": None,
            "n_frames": 6,
            "n_keypoints": 2,
            "n_features": 10,
            "features_path": str(features_dir / f"{stem}_features.npy"),
        }

    with open(features_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    cluster_info = {
        "n_clusters": 2,
        "cluster_centers": [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        ],
    }
    with open(shared_dir / "cluster_info.json", "w") as f:
        json.dump(cluster_info, f, indent=2)

    meta_path = project_dir / "metadata.csv"
    metadata.to_csv(meta_path, index=False)

    config_path = project_dir / "config.json"
    config_path.write_text(json.dumps({
        "results_dir": str(results_dir),
        "metadata_csv_path": str(meta_path),
        "optional_report_columns": ["fear"],
    }))
    app_config_path = tmp_path / "app_config.json"
    app_config_path.write_text(json.dumps({"active_project": str(project_dir)}))

    import vieb_config as vc
    monkeypatch.setattr(vc, "_APP_CONFIG_PATH", str(app_config_path))
    monkeypatch.setattr(vc, "_CONFIG_PATH", str(config_path))

    import compare
    return compare, results_dir


def _base_metadata(include_fear: bool) -> pd.DataFrame:
    data = {
        "filename": ["vidA.mp4", "vidB.mp4", "vidC.mp4", "vidD.mp4"],
        "date": ["20250101"] * 4,
        "box": [1, 1, 2, 2],
        "experiment": ["spence"] * 4,
        "day": [0, 0, 1, 1],
        "context": ["A", "B", "A", "B"],
        "animal_id": ["101", "101", "102", "102"],
    }
    if include_fear:
        data["fear"] = ["low", "high", "low", "high"]
    return pd.DataFrame(data)


def test_cmd_report_skips_missing_fear_column(tmp_path, monkeypatch, capsys):
    compare, results_dir = _setup_report_project(
        tmp_path, monkeypatch, _base_metadata(include_fear=False)
    )

    compare.cmd_report(fps=30.0, min_confidence=0.8)

    out = capsys.readouterr().out
    assert "[info] No fear column found; skipping fear-specific report." in out
    assert (results_dir / "comparison" / "summary_table.csv").exists()
    assert (results_dir / "characterization" / "bouts.csv").exists()
    assert (results_dir / "characterization" / "state_summary.csv").exists()
    assert (results_dir / "comparison" / "transition_table.csv").exists()
    assert (results_dir / "comparison" / "transition_by_context.png").exists()
    assert (results_dir / "comparison" / "state_by_context.png").exists()
    assert (results_dir / "comparison" / "state_by_animal.png").exists()
    assert (results_dir / "comparison" / "motifs.csv").exists()
    assert not (results_dir / "comparison" / "state_by_fear.png").exists()


def test_cmd_report_keeps_fear_specific_output_when_present(tmp_path, monkeypatch, capsys):
    compare, results_dir = _setup_report_project(
        tmp_path, monkeypatch, _base_metadata(include_fear=True)
    )

    compare.cmd_report(fps=30.0, min_confidence=0.8)

    out = capsys.readouterr().out
    assert "No fear column found" not in out
    assert (results_dir / "comparison" / "state_by_fear.png").exists()
    summary = pd.read_csv(results_dir / "comparison" / "summary_table.csv")
    assert "fear" in summary.columns
