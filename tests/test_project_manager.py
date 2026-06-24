import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import project_manager as pm


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_project(root: Path, name: str = "proj", *, external: dict | None = None) -> Path:
    project = root / "projects" / name
    project.mkdir(parents=True)
    results = external.get("results") if external else project / "results"
    raw = external.get("raw_videos") if external else project / "raw_videos"
    meta = external.get("metadata") if external else project / "metadata.csv"
    Path(results).mkdir(parents=True, exist_ok=True)
    Path(raw).mkdir(parents=True, exist_ok=True)
    Path(meta).parent.mkdir(parents=True, exist_ok=True)
    Path(meta).write_text("session_id,stem\n", encoding="utf-8")
    _write_json(project / "config.json", {
        "project_name": name,
        "paths": {
            "raw_videos": str(raw),
            "pose_files": "",
            "pose_h5": None,
            "metadata": str(meta),
            "results": str(results),
        },
        "metadata_schema": {"column_map": {"session_id": "session_id"}},
        "analysis_groups": [{"name": "Session", "column": "session_id", "enabled": True}],
        "ui_panels": {},
        "pipeline_settings": {},
    })
    return project


def test_valid_active_project(tmp_path):
    project = _make_project(tmp_path)
    app = tmp_path / "app_config.json"
    _write_json(app, {"active_project": str(project), "recent_projects": []})

    selected = pm.select_startup_project(tmp_path, app)

    assert selected.action == "use_active"
    assert selected.active_project == project.resolve()


def test_missing_active_project_no_projects_opens_onboarding(tmp_path):
    app = tmp_path / "app_config.json"
    _write_json(app, {"active_project": "", "recent_projects": []})

    selected = pm.select_startup_project(tmp_path, app)

    assert selected.action == "onboarding_required"
    assert selected.active_project is None


def test_deleted_active_project_with_one_detected_project(tmp_path):
    deleted = tmp_path / "projects" / "deleted"
    project = _make_project(tmp_path, "detected")
    app = tmp_path / "app_config.json"
    _write_json(app, {"active_project": str(deleted), "recent_projects": []})

    selected = pm.select_startup_project(tmp_path, app)

    assert selected.action == "auto_selected"
    assert selected.active_project == project.resolve()
    assert json.loads(app.read_text())["active_project"] == str(project.resolve())


def test_multiple_projects_require_picker(tmp_path):
    _make_project(tmp_path, "one")
    _make_project(tmp_path, "two")
    app = tmp_path / "app_config.json"
    _write_json(app, {"active_project": "", "recent_projects": []})

    selected = pm.select_startup_project(tmp_path, app)

    assert selected.action == "picker_required"
    assert len(selected.candidates) == 2


def test_legacy_repo_root_detected_but_not_selected(tmp_path):
    (tmp_path / "metadata.csv").write_text("session_id\n", encoding="utf-8")
    (tmp_path / "raw_videos").mkdir()
    app = tmp_path / "app_config.json"
    _write_json(app, {"active_project": "", "recent_projects": []})

    selected = pm.select_startup_project(tmp_path, app)

    assert selected.action == "onboarding_required"
    assert selected.legacy_detected is True
    assert json.loads(app.read_text())["active_project"] == ""


def test_legacy_project_can_be_registered(tmp_path):
    (tmp_path / "metadata.csv").write_text("session_id\n", encoding="utf-8")
    (tmp_path / "raw_videos").mkdir()
    (tmp_path / "results").mkdir()
    app = tmp_path / "app_config.json"

    project = pm.register_legacy_project(tmp_path, app)

    assert project == tmp_path.resolve()
    assert pm.select_startup_project(tmp_path, app).active_project == tmp_path.resolve()
    cfg = json.loads((tmp_path / "config.json").read_text())
    assert cfg["project_name"] == "legacy_project"


def test_create_new_project_writes_scaffold(tmp_path):
    app = tmp_path / "app_config.json"
    project = pm.create_project(tmp_path / "projects" / "new_project", "New Project", app_config_path=app, repo_root=tmp_path)

    assert (project / "config.json").exists()
    assert (project / "results").is_dir()
    assert (project / "logs").is_dir()
    assert (project / "metadata.csv").exists()
    assert json.loads(app.read_text())["active_project"] == str(project)


def test_onboarding_generates_missing_metadata_from_raw_videos(tmp_path):
    project = tmp_path / "projects" / "raw_project"
    raw = project / "raw_videos"
    results = project / "results"
    raw.mkdir(parents=True)
    results.mkdir()
    (raw / "session_001.mp4").write_bytes(b"")
    _write_json(project / "config.json", {
        "project_name": "raw_project",
        "paths": {
            "raw_videos": str(raw),
            "pose_files": "",
            "pose_h5": None,
            "metadata": str(project / "metadata.csv"),
            "results": str(results),
        },
    })
    app = tmp_path / "app_config.json"
    _write_json(app, {"active_project": str(project), "recent_projects": []})

    selected = pm.onboard_project(tmp_path, app)

    assert selected.active_project == project.resolve()
    assert (project / "metadata.csv").exists()
    assert "session_001.mp4" in (project / "metadata.csv").read_text(encoding="utf-8")


def test_onboarding_without_sources_does_not_require_advanced_settings(tmp_path):
    project = tmp_path / "projects" / "empty_project"
    project.mkdir(parents=True)
    (project / "results").mkdir()
    _write_json(project / "config.json", {
        "project_name": "empty_project",
        "paths": {
            "raw_videos": str(project / "raw_videos"),
            "pose_files": "",
            "pose_h5": None,
            "metadata": str(project / "metadata.csv"),
            "results": str(project / "results"),
        },
    })
    app = tmp_path / "app_config.json"
    _write_json(app, {"active_project": str(project), "recent_projects": []})

    selected = pm.onboard_project(tmp_path, app)

    assert selected.active_project == project.resolve()
    assert "Stage 0 requires raw videos" in selected.message
    assert pm.onboarding_complete(project) is False


def test_import_pose_csv_source_generates_metadata_and_skips_dlc(tmp_path):
    app = tmp_path / "app_config.json"
    project = pm.create_project(tmp_path / "projects" / "pose_project", "Pose Project", app_config_path=app, repo_root=tmp_path)
    pose_dir = tmp_path / "pose"
    pose_dir.mkdir()
    (pose_dir / "session_A_DLC.csv").write_text("x,y\n1,2\n", encoding="utf-8")

    result = pm.import_data_source(project, "pose_csvs", pose_dir)
    cfg = json.loads((project / "config.json").read_text(encoding="utf-8"))

    assert result["valid"] is True
    assert cfg["pose_source"] == "csv"
    assert cfg["paths"]["pose_files"] == str(pose_dir.resolve())
    assert "session_A_DLC.csv" in (project / "metadata.csv").read_text(encoding="utf-8")
    assert pm.onboarding_complete(project) is True


def test_import_existing_metadata_csv_normalizes_into_project(tmp_path):
    app = tmp_path / "app_config.json"
    project = pm.create_project(tmp_path / "projects" / "manifest_project", "Manifest Project", app_config_path=app, repo_root=tmp_path)
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("source_file,animal\nvid1.mp4,a1\nvid2.mp4,a2\n", encoding="utf-8")

    result = pm.import_data_source(project, "metadata", manifest)

    assert result["valid"] is True
    text = (project / "metadata.csv").read_text(encoding="utf-8")
    assert "session_id" in text
    assert "vid1.mp4" in text
    assert pm.onboarding_complete(project) is True


def test_external_metadata_and_raw_video_paths(tmp_path):
    external = tmp_path / "external"
    raw = external / "videos"
    meta = external / "metadata.csv"
    raw.mkdir(parents=True)
    meta.write_text("session_id\n", encoding="utf-8")
    project = _make_project(tmp_path, "external_paths", external={"raw_videos": raw, "metadata": meta, "results": external / "results"})
    app = tmp_path / "app_config.json"
    _write_json(app, {"active_project": str(project), "recent_projects": []})

    assert pm.resolve_project_path("metadata", tmp_path, app) == meta.resolve()
    assert pm.resolve_project_path("raw_videos", tmp_path, app) == raw.resolve()


def test_existing_results_mark_stages_resumable(tmp_path):
    project = _make_project(tmp_path)
    results = project / "results"
    (results / "features").mkdir(exist_ok=True)
    (results / "features" / "index.json").write_text("{}", encoding="utf-8")
    (results / "shared").mkdir(exist_ok=True)
    (results / "shared" / "cluster_info.json").write_text("{}", encoding="utf-8")
    (results / "comparison").mkdir(exist_ok=True)
    (results / "comparison" / "summary_table.csv").write_text("session_id\n", encoding="utf-8")

    status = pm.resume_status(project)

    assert status["features"] is True
    assert status["clusters"] is True
    assert status["reports"] is True


def test_pipeline_refuses_without_valid_project(tmp_path):
    app = tmp_path / "app_config.json"
    _write_json(app, {"active_project": "", "recent_projects": []})

    with pytest.raises(pm.ProjectSelectionError):
        pm.get_active_project(tmp_path, app)


def test_path_resolver_refuses_repo_root_paths_unless_root_is_active(tmp_path):
    project = _make_project(tmp_path, "bad_paths")
    cfg = json.loads((project / "config.json").read_text())
    cfg["paths"]["metadata"] = str(tmp_path / "metadata.csv")
    cfg["paths"]["raw_videos"] = str(tmp_path / "raw_videos")
    cfg["paths"]["results"] = str(tmp_path / "results")
    _write_json(project / "config.json", cfg)
    app = tmp_path / "app_config.json"
    _write_json(app, {"active_project": str(project), "recent_projects": []})

    with pytest.raises(pm.ProjectSelectionError):
        pm.resolve_project_path("metadata", tmp_path, app)

    (tmp_path / "metadata.csv").write_text("session_id\n", encoding="utf-8")
    (tmp_path / "raw_videos").mkdir(exist_ok=True)
    (tmp_path / "results").mkdir(exist_ok=True)
    pm.register_legacy_project(tmp_path, app)
    assert pm.resolve_project_path("metadata", tmp_path, app) == (tmp_path / "metadata.csv").resolve()
