"""Tests for optional metadata columns in compare.py --report."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import project_manager as pm


def _setup_report_project(tmp_path, monkeypatch, metadata: pd.DataFrame, config_extra: dict | None = None):
    project_dir = tmp_path / "project"
    results_dir = project_dir / "results"
    features_dir = results_dir / "features"
    shared_dir = results_dir / "shared"
    features_dir.mkdir(parents=True)
    shared_dir.mkdir(parents=True)

    import metadata_schema as ms
    schema_cfg = config_extra or {}
    normalized = ms.normalize_metadata_columns(metadata, schema_cfg)
    stems = normalized["stem"].tolist()
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
    config = {
        "results_dir": str(results_dir),
        "metadata_csv_path": str(meta_path),
        "optional_report_columns": ["fear"],
    }
    if config_extra:
        config.update(config_extra)
    config_path.write_text(json.dumps(config))
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


def test_cmd_report_spence_style_source_file_mapping(tmp_path, monkeypatch, capsys):
    metadata = pd.DataFrame({
        "source_file": ["rat1_tp0.csv", "rat1_tp1.h5", "rat2_tp0.mp4", "rat2_tp1.avi"],
        "rat": ["r1", "r1", "r2", "r2"],
        "timepoint": ["baseline", "drug", "baseline", "drug"],
        "treatment": ["vehicle", "drug", "vehicle", "drug"],
        "protein_A": [1.0, 2.0, 1.5, 2.5],
    })
    cfg = {
        "metadata_schema": {
            "id_column": "source_file",
            "column_map": {
                "session_id": "source_file",
                "animal_id": "rat",
                "day": "timepoint",
            },
            "optional_columns": {
                "treatment": "treatment",
                "protein_A": "protein_A",
            },
            "analysis_groups": [
                {
                    "name": "Timepoint",
                    "column": "day",
                    "enabled": True,
                    "plots": ["state_fraction"],
                },
                {
                    "name": "Treatment",
                    "column": "treatment",
                    "enabled": True,
                    "plots": ["state_fraction", "motif_enrichment"],
                },
            ],
            "correlations": [
                {
                    "name": "Protein correlations",
                    "columns": ["protein_A"],
                    "targets": ["state_fraction"],
                    "enabled": True,
                }
            ],
        }
    }
    compare, results_dir = _setup_report_project(tmp_path, monkeypatch, metadata, cfg)

    compare.cmd_report(fps=30.0, min_confidence=0.8)

    out = capsys.readouterr().out
    assert "No fear column found" not in out
    summary = pd.read_csv(results_dir / "comparison" / "summary_table.csv")
    assert {"session_id", "stem", "animal_id", "day", "treatment", "protein_A"}.issubset(summary.columns)
    assert summary["stem"].tolist() == ["rat1_tp0", "rat1_tp1", "rat2_tp0", "rat2_tp1"]
    assert (results_dir / "comparison" / "state_by_treatment.png").exists()
    assert (results_dir / "comparison" / "motifs_by_treatment.csv").exists()
    assert (results_dir / "comparison" / "correlations.csv").exists()
    assert (results_dir / "metadata_schema_report.json").exists()


def test_cmd_report_without_context_skips_context_outputs(tmp_path, monkeypatch, capsys):
    metadata = pd.DataFrame({
        "filename": ["vidA.mp4", "vidB.mp4", "vidC.mp4", "vidD.mp4"],
        "animal_id": ["101", "101", "102", "102"],
        "day": [0, 1, 0, 1],
    })
    compare, results_dir = _setup_report_project(
        tmp_path, monkeypatch, metadata, {"optional_report_columns": ["fear"]}
    )

    compare.cmd_report(fps=30.0, min_confidence=0.8)

    out = capsys.readouterr().out
    assert "context" in out
    assert (results_dir / "comparison" / "summary_table.csv").exists()
    assert (results_dir / "comparison" / "transition_table.csv").exists()
    assert not (results_dir / "comparison" / "transition_by_context.png").exists()
    assert not (results_dir / "comparison" / "motifs.csv").exists()


def test_metadata_schema_missing_session_identifier_validation():
    import metadata_schema as ms

    report = ms.validate_metadata_schema(pd.DataFrame({"rat": ["r1"], "timepoint": ["t0"]}), {})

    assert not report["valid"]
    assert "session_id" in report["missing_required_fields"]


def test_stem_derivation_common_extensions():
    import metadata_schema as ms

    values = ["a/b/session1.mp4", "session2.csv", "session3.h5", "session4.mov", "session5"]

    assert [ms.derive_stem(v) for v in values] == [
        "session1", "session2", "session3", "session4", "session5"
    ]


def test_report_cli_diagnostics_use_active_project_paths(tmp_path, capsys):
    project = pm.create_project(tmp_path / "projects" / "report_project", "Report Project", repo_root=tmp_path)
    app = tmp_path / "app_config.json"
    app.write_text(json.dumps({"active_project": str(project)}), encoding="utf-8")

    import compare

    paths = compare._print_project_path_diagnostics(str(tmp_path), str(app))

    out = capsys.readouterr().out
    assert f"Active project: {project}" in out
    assert f"Metadata path: {project / 'metadata.csv'} (origin: project_config)" in out
    assert f"Results dir: {project / 'results'} (origin: project_config)" in out
    assert f"Raw videos dir: {project / 'raw_videos'} (origin: project_config)" in out
    assert f"Config path: {project / 'config.json'}" in out
    assert paths["metadata"].path == (project / "metadata.csv").resolve()
