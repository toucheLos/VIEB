import json
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import metadata_generator
import project_manager as pm
import vieb_config


@pytest.fixture
def isolate_ambient_config(monkeypatch):
    """Stop metadata normalization from re-resolving the ambient active project.

    validate_metadata_csv -> vieb_config.normalize_metadata_columns -> _load_config
    otherwise loads whatever project the repo's real app_config.json points at,
    which pollutes call counts/timing with unrelated (and re-entrant) work. Tests
    of project_manager's own memoization must not depend on that global state.
    """
    monkeypatch.setattr(vieb_config, "_load_config", lambda: {})


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_project(root: Path, name: str = "proj", *, external: dict | None = None, external_paths: list[str] | None = None) -> Path:
    project = root / "projects" / name
    project.mkdir(parents=True)
    external = external or {}
    results = external.get("results", project / "results")
    raw = external.get("raw_videos", project / "raw_videos")
    meta = external.get("metadata", project / "metadata.csv")
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
        "external_paths": external_paths or [],
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
    project = _make_project(
        tmp_path,
        "external_paths",
        external={"raw_videos": raw, "metadata": meta, "results": external / "results"},
        external_paths=["raw_videos", "metadata", "results"],
    )
    app = tmp_path / "app_config.json"
    _write_json(app, {"active_project": str(project), "recent_projects": []})

    assert pm.resolve_project_path("metadata", tmp_path, app) == meta.resolve()
    assert pm.resolve_project_path("raw_videos", tmp_path, app) == raw.resolve()


def test_missing_project_paths_default_to_project_local(tmp_path):
    project = tmp_path / "projects" / "defaults"
    project.mkdir(parents=True)
    _write_json(project / "config.json", {"project_name": "defaults"})
    paths = pm.resolve_project_paths_for_project(project, {}, tmp_path)

    assert paths["metadata"].path == (project / "metadata.csv").resolve()
    assert paths["metadata"].origin == "project_default"
    assert paths["results"].path == (project / "results").resolve()
    assert paths["results"].origin == "project_default"
    assert paths["raw_videos"].path == (project / "raw_videos").resolve()
    assert paths["raw_videos"].origin == "project_default"


def test_stage0_rejects_accidental_repo_root_metadata_and_results(tmp_path):
    project = _make_project(tmp_path, "bad_stage0")
    (tmp_path / "metadata.csv").write_text("session_id,stem\nvideo1,video1\n", encoding="utf-8")
    (tmp_path / "results").mkdir(exist_ok=True)
    cfg = json.loads((project / "config.json").read_text(encoding="utf-8"))
    cfg["paths"]["metadata"] = str(tmp_path / "metadata.csv")
    cfg["paths"]["results"] = str(tmp_path / "results")
    _write_json(project / "config.json", cfg)
    app = tmp_path / "app_config.json"
    _write_json(app, {"active_project": str(project), "recent_projects": []})

    validation = pm.validate_project(project, tmp_path)
    checks = {check.key: check for check in validation.checks}

    assert checks["metadata"].status == "red"
    assert checks["results"].status == "red"
    assert pm.onboarding_complete(project, tmp_path) is False
    with pytest.raises(pm.ProjectSelectionError):
        pm.resolve_project_path("results", tmp_path, app)


def test_unmarked_external_results_are_refused(tmp_path):
    external = tmp_path / "external"
    project = _make_project(
        tmp_path,
        "external_results",
        external={"results": external / "results"},
        external_paths=[],
    )
    app = tmp_path / "app_config.json"
    _write_json(app, {"active_project": str(project), "recent_projects": []})

    paths = pm.resolve_project_paths_for_project(project, json.loads((project / "config.json").read_text()), tmp_path)

    assert paths["results"].valid is False
    with pytest.raises(pm.ProjectSelectionError):
        pm.resolve_project_path("results", tmp_path, app)


def test_explicit_external_results_are_allowed(tmp_path):
    external = tmp_path / "external"
    project = _make_project(
        tmp_path,
        "external_results_ok",
        external={"results": external / "results"},
        external_paths=["results"],
    )
    app = tmp_path / "app_config.json"
    _write_json(app, {"active_project": str(project), "recent_projects": []})

    paths = pm.resolve_project_paths_for_project(project, json.loads((project / "config.json").read_text()), tmp_path)

    assert paths["results"].origin == "external_explicit"
    assert pm.resolve_project_path("results", tmp_path, app) == (external / "results").resolve()


def test_gui_readiness_and_cli_resolver_agree_on_paths(tmp_path):
    project = _make_project(tmp_path, "agreement")
    raw = project / "raw_videos"
    (raw / "video1.mp4").write_bytes(b"")
    meta = project / "metadata.csv"
    meta.write_text("session_id,stem,animal_id,context\nvideo1,video1,a,A\n", encoding="utf-8")
    app = tmp_path / "app_config.json"
    _write_json(app, {"active_project": str(project), "recent_projects": []})

    assert pm.onboarding_complete(project, tmp_path) is True
    assert pm.resolve_project_path("metadata", tmp_path, app) == meta.resolve()
    assert pm.resolve_project_path("results", tmp_path, app) == (project / "results").resolve()


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


# --------------------------------------------------------------------------
# Memoization of the expensive validation work (Part A / B)
# --------------------------------------------------------------------------

def _make_large_project(root: Path, name: str = "big", n: int = 3700):
    """A project with n raw videos and an n-row metadata.csv (Spence-lab scale)."""
    project = _make_project(root, name)
    raw = project / "raw_videos"
    for i in range(n):
        (raw / f"session_{i:05d}.mp4").write_bytes(b"")
    lines = ["session_id,stem"] + [f"session_{i:05d},session_{i:05d}" for i in range(n)]
    (project / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    app = root / "app_config.json"
    _write_json(app, {"active_project": str(project), "recent_projects": []})
    return project, app


class _NoCache(dict):
    """Dict that never retains entries — used to simulate the pre-memoization code."""

    def get(self, *args, **kwargs):
        return None

    def __setitem__(self, *args, **kwargs):
        pass


def _count_validate_calls(monkeypatch):
    calls = {"n": 0}
    real = metadata_generator.validate_metadata_csv

    def counting(path):
        calls["n"] += 1
        return real(path)

    monkeypatch.setattr(metadata_generator, "validate_metadata_csv", counting)
    return calls


def test_repeated_validation_memoized(tmp_path, monkeypatch, isolate_ambient_config):
    """The redundant validation passes onboard performs collapse to one real pass.

    validate_project() -> session_source_status() is what onboard_project runs
    2-3x per call; with memoization the expensive validate_metadata_csv pass runs
    exactly once when nothing on disk changes between calls.
    """
    project, _ = _make_large_project(tmp_path, "memo", n=200)
    pm.invalidate_project_cache()
    calls = _count_validate_calls(monkeypatch)

    results = [pm.validate_project(project) for _ in range(3)]

    assert calls["n"] == 1
    assert {r.status for r in results} == {results[0].status}  # identical each pass


def test_source_status_cache_invalidates_on_metadata_change(tmp_path, isolate_ambient_config):
    """A genuine metadata edit (mtime bump) refreshes the memoized result."""
    project, _ = _make_large_project(tmp_path, "invalidate", n=50)
    pm.invalidate_project_cache()

    first = pm.session_source_status(project)
    assert first["metadata_rows"] == 50

    meta = project / "metadata.csv"
    lines = ["session_id,stem"] + [f"session_{i:05d},session_{i:05d}" for i in range(70)]
    meta.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Force a distinct mtime so the test does not depend on filesystem resolution.
    bumped = os.stat(meta).st_mtime + 100
    os.utime(meta, (bumped, bumped))

    second = pm.session_source_status(project)
    assert second["metadata_rows"] == 70


def test_validate_project_attaches_source_and_paths(tmp_path, isolate_ambient_config):
    """validate_project exposes the paths/source it already computed (Part B)."""
    project = _make_project(tmp_path, "attach")
    (project / "raw_videos" / "v1.mp4").write_bytes(b"")
    (project / "metadata.csv").write_text(
        "session_id,stem\nv1,v1\n", encoding="utf-8"
    )
    pm.invalidate_project_cache()

    validation = pm.validate_project(project, tmp_path)

    assert validation.source is not None
    assert validation.paths is not None
    assert validation.source == pm.session_source_status(project, validation.config, tmp_path)
    direct_paths = pm.resolve_project_paths_for_project(project, validation.config, tmp_path)
    assert set(validation.paths.keys()) == set(direct_paths.keys())
    assert validation.paths["metadata"].path == direct_paths["metadata"].path


def test_onboard_project_benchmark_large_project(tmp_path, monkeypatch, isolate_ambient_config):
    """Onboarding a 3,700-file/-row project does less work and is faster with caching."""
    project, app = _make_large_project(tmp_path, "bench", n=3700)
    calls = _count_validate_calls(monkeypatch)

    # BEFORE: disable the new memoization caches -> heavy work repeats every call.
    monkeypatch.setattr(pm, "_source_status_cache", _NoCache())
    monkeypatch.setattr(pm, "_metadata_validation_cache", _NoCache())
    monkeypatch.setattr(pm, "_count_files_cache", _NoCache())
    pm.invalidate_project_cache()
    calls["n"] = 0
    t0 = time.perf_counter()
    sel_before = pm.onboard_project(tmp_path, app)
    before = time.perf_counter() - t0
    before_calls = calls["n"]

    # AFTER: real caches -> heavy work runs once, the rest are cache hits.
    monkeypatch.setattr(pm, "_source_status_cache", {})
    monkeypatch.setattr(pm, "_metadata_validation_cache", {})
    monkeypatch.setattr(pm, "_count_files_cache", {})
    pm.invalidate_project_cache()
    calls["n"] = 0
    t0 = time.perf_counter()
    sel_after = pm.onboard_project(tmp_path, app)
    after = time.perf_counter() - t0
    after_calls = calls["n"]

    print(
        f"\n[benchmark] onboard_project before={before*1000:.1f}ms/{before_calls} passes "
        f"after={after*1000:.1f}ms/{after_calls} passes"
    )
    assert sel_before.active_project == sel_after.active_project == project.resolve()
    assert after_calls < before_calls  # deterministic: fewer full validation passes
    assert after < before              # and measurably faster
