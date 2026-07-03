"""Tests for compare.py::_print_project_path_diagnostics()'s loud warning +
gated --repair-paths behavior when a resolved path (e.g. metadata.csv) is a
doubled, nonexistent path from a pre-refactor config.json."""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import compare  # noqa: E402


def _setup_doubled_project(tmp_path):
    repo_root = tmp_path
    project = repo_root / "projects" / "spence_lab"
    raw_videos = project / "raw_videos"
    results = project / "results"
    raw_videos.mkdir(parents=True)
    results.mkdir(parents=True)

    # The real metadata.csv lives at the correct, non-doubled location.
    real_meta = project / "metadata.csv"
    real_meta.write_text("session_id,stem\nv1,v1\n", encoding="utf-8")

    # config.json stores a doubled path for "metadata" — as if it were still
    # relative to repo_root under the pre-refactor resolution scheme.
    unit = project.relative_to(repo_root).parts
    doubled_meta = os.path.join(str(project), *unit, "metadata.csv")

    cfg = {
        "project_name": "spence_lab",
        "paths": {
            "raw_videos": str(raw_videos),
            "pose_files": "",
            "pose_h5": None,
            "metadata": doubled_meta,
            "results": str(results),
        },
    }
    (project / "config.json").write_text(json.dumps(cfg), encoding="utf-8")

    app_config_path = tmp_path / "app_config.json"
    app_config_path.write_text(json.dumps({"active_project": str(project)}), encoding="utf-8")

    return repo_root, app_config_path, project, real_meta


def test_print_project_path_diagnostics_warns_without_repair(tmp_path, capsys):
    repo_root, app_config_path, project, real_meta = _setup_doubled_project(tmp_path)

    compare._print_project_path_diagnostics(str(repo_root), str(app_config_path), repair=False)

    captured = capsys.readouterr()
    assert "[WARN] Resolved metadata path does not exist" in captured.out
    assert "doubled path from a pre-refactor config.json" in captured.out
    assert str(real_meta) in captured.out
    assert "Re-run with --repair-paths" in captured.out

    cfg_after = json.loads((project / "config.json").read_text(encoding="utf-8"))
    assert cfg_after["paths"]["metadata"] != str(real_meta)


def test_print_project_path_diagnostics_repairs_when_flagged(tmp_path, capsys):
    repo_root, app_config_path, project, real_meta = _setup_doubled_project(tmp_path)

    compare._print_project_path_diagnostics(str(repo_root), str(app_config_path), repair=True)

    captured = capsys.readouterr()
    assert "[FIX]    config.json updated: paths.metadata ->" in captured.out

    cfg_after = json.loads((project / "config.json").read_text(encoding="utf-8"))
    assert cfg_after["paths"]["metadata"] == str(real_meta.resolve())
