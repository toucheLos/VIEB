import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("PyQt5")
import user_interface as ui


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_project(root: Path, name: str = "proj") -> tuple[Path, Path, Path]:
    project = root / "projects" / name
    raw = project / "raw_videos"
    results = project / "results"
    raw.mkdir(parents=True)
    results.mkdir(parents=True)
    metadata = project / "metadata.csv"
    metadata.write_text("session_id,stem\nvideo1,video1\n", encoding="utf-8")
    _write_json(project / "config.json", {
        "project_name": name,
        "paths": {
            "raw_videos": str(raw),
            "pose_files": "",
            "pose_h5": None,
            "metadata": str(metadata),
            "results": str(results),
        },
        "metadata_schema": {"column_map": {"session_id": "session_id"}},
        "analysis_groups": [],
        "ui_panels": {},
        "pipeline_settings": {},
    })
    app = root / "app_config.json"
    _write_json(app, {"active_project": str(project), "recent_projects": []})
    return project, results, app


def _patch_project(monkeypatch, root: Path, app: Path, results: Path):
    monkeypatch.setattr(ui, "ROOT", root)
    monkeypatch.setattr(ui, "APP_CONFIG_PATH", app)
    monkeypatch.setattr(ui, "RESULTS", results)
    monkeypatch.setattr(ui, "CLIPS", results.parent / "clips")


def _run_loader() -> dict:
    out = {}
    errors = []
    loader = ui.DataLoader(lightweight=True)
    loader.loaded.connect(lambda data: out.update(data))
    loader.error.connect(errors.append)
    loader.run()
    assert not errors
    return out


def test_lightweight_loader_autoloads_overview_summary(tmp_path, monkeypatch):
    _, results, app = _make_project(tmp_path)
    _patch_project(monkeypatch, tmp_path, app, results)
    _write_json(results / "shared" / "cluster_info.json", {"n_clusters": 3})
    _write_json(results / "shared" / "run_manifest.json", {"run_id": "run-a"})
    _write_json(results / "shared" / "overview_summary.json", {
        "total_videos": 2,
        "total_frames": 120,
        "n_states": 3,
        "noise_fraction": 0.125,
        "last_run_time": "2026-06-28 14:46",
        "active_run_id": "run-a",
        "state_means": {"0": 0.5, "1": 0.25, "2": 0.125},
    })

    data = _run_loader()

    assert data["overview_summary"]["total_videos"] == 2
    assert data["cluster_info"]["n_clusters"] == 3
    assert data["_lightweight"] is True
    assert data.get("summary") is None


def test_lightweight_loader_does_not_read_large_csvs(tmp_path, monkeypatch):
    _, results, app = _make_project(tmp_path)
    _patch_project(monkeypatch, tmp_path, app, results)
    (results / "comparison").mkdir(parents=True)
    (results / "characterization").mkdir(parents=True)
    (results / "comparison" / "summary_table.csv").write_text("state_0_frac\n1\n", encoding="utf-8")
    (results / "characterization" / "labels_per_frame.csv").write_text("frame,state\n0,1\n", encoding="utf-8")
    _write_json(results / "shared" / "overview_summary.json", {"total_videos": 1})

    def fail_read_csv(path, *args, **kwargs):
        text = str(path)
        if "summary_table.csv" in text or "labels_per_frame.csv" in text:
            raise AssertionError(f"large CSV read during lightweight load: {text}")
        raise AssertionError(f"unexpected CSV read during lightweight load: {text}")

    monkeypatch.setattr(ui.pd, "read_csv", fail_read_csv)

    data = _run_loader()

    assert data["overview_summary"]["total_videos"] == 1
    assert data["markers"]["summary"] is True


def test_invalid_active_project_does_not_load_repo_root_results(tmp_path, monkeypatch):
    project, results, app = _make_project(tmp_path)
    repo_results = tmp_path / "results"
    _write_json(repo_results / "shared" / "overview_summary.json", {"total_videos": 99})
    cfg = json.loads((project / "config.json").read_text(encoding="utf-8"))
    cfg["paths"]["results"] = str(repo_results)
    _write_json(project / "config.json", cfg)
    _patch_project(monkeypatch, tmp_path, app, results)

    data = _run_loader()

    assert data.get("overview_summary") is None
    assert data["markers"]["summary"] is False
    assert "project_warning" in data


def test_overview_cards_populate_from_summary_without_click(qtbot):
    view = ui.OverviewView()
    qtbot.addWidget(view)

    view.update_data({
        "overview_summary": {
            "total_videos": 4,
            "total_frames": 2000,
            "n_states": 5,
            "noise_fraction": 0.1,
            "last_run_time": "2026-06-28 14:46",
            "state_means": {"0": 0.6, "1": 0.3},
        },
        "cluster_info": {"n_clusters": 5},
    })

    assert view._c_videos._value.text() == "4"
    assert view._c_frames._value.text() == "2,000"
    assert view._c_states._value.text() == "5"
    assert view._c_noise._value.text() == "10.0%"
    assert "2026-06-28 14:46" in view._run_lbl.text()


def test_overview_has_no_previous_session_alert(qtbot):
    view = ui.OverviewView()
    qtbot.addWidget(view)

    assert not hasattr(view, "_prev_banner")
    assert not hasattr(view, "_prev_load_btn")
    assert not hasattr(view, "load_previous_requested")
