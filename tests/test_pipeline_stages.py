import ast
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import project_manager as pm


# ── helpers ──────────────────────────────────────────────────────────────────

def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _make_project(root: Path, name: str = "proj", *, with_metadata: bool = True) -> Path:
    project = root / "projects" / name
    project.mkdir(parents=True)
    raw = project / "raw_videos"
    raw.mkdir()
    results = project / "results"
    results.mkdir()
    meta = project / "metadata.csv"
    if with_metadata:
        meta.write_text("session_id,stem\nvideo1,video1\n", encoding="utf-8")
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


def _load_stages():
    src = Path(__file__).resolve().parents[1] / "user_interface.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "STAGES":
                    return ast.literal_eval(node.value)
    raise AssertionError("STAGES assignment not found")


# ── existing stage-ordering tests ─────────────────────────────────────────────

def test_stage_zero_is_onboarding_not_project_onboarding():
    stages = _load_stages()

    assert stages[0]["id"] == 0
    assert stages[0]["name"] == "Onboarding"
    assert "Project Onboarding" not in stages[0]["name"]


def test_pipeline_stage_ids_are_in_product_order():
    stages = _load_stages()

    assert [(s["id"], s["name"]) for s in stages] == [
        (0, "Onboarding"),
        (1, "Pose Estimation / DLC Analysis"),
        (2, "Feature Extraction"),
        (3, "Preprocessing · UMAP · Clustering · Smoothing"),
        (4, "State Collapsing (optional)"),
        (5, "Report Generation"),
        (6, "Per-Animal Scalars"),
        (7, "Motif Discovery"),
        (8, "Generate Clips"),
        (9, "Add Videos"),
    ]


# ── Stage 0 readiness tests ───────────────────────────────────────────────────

def test_stage0_panel_is_stage0_readiness_panel_not_project_onboarding():
    """Stage 0 uses Stage0ReadinessPanel, not the old ProjectOnboardingPanel."""
    src = Path(__file__).resolve().parents[1] / "user_interface.py"
    text = src.read_text(encoding="utf-8")
    assert "Stage0ReadinessPanel" in text
    assert "ProjectOnboardingPanel" not in text


def test_stage0_readiness_complete_when_valid_project(tmp_path):
    """onboarding_complete returns True for a project with metadata and a raw-videos source."""
    project = _make_project(tmp_path, with_metadata=True)
    app = tmp_path / "app_config.json"
    _write_json(app, {"active_project": str(project), "recent_projects": []})

    assert pm.onboarding_complete(project) is True


def test_stage0_readiness_incomplete_when_metadata_missing(tmp_path):
    """onboarding_complete returns False when metadata.csv does not exist."""
    project = _make_project(tmp_path, with_metadata=False)
    assert pm.onboarding_complete(project) is False


def test_stage0_no_repo_root_fallback(tmp_path):
    """get_active_project raises ProjectSelectionError when no active project is configured,
    even if a metadata.csv exists at the repo root (no implicit fallback)."""
    (tmp_path / "metadata.csv").write_text("session_id\n", encoding="utf-8")
    app = tmp_path / "app_config.json"
    _write_json(app, {"active_project": "", "recent_projects": []})

    with pytest.raises(pm.ProjectSelectionError):
        pm.get_active_project(tmp_path, app)


def test_stage0_blocks_pipeline_error_message():
    """The workers module contains the exact blocking message for Stage 0."""
    src = Path(__file__).resolve().parents[1] / "_workers.py"
    text = src.read_text(encoding="utf-8")
    assert "No valid project selected. Complete Stage 0: Onboarding before running the pipeline." in text


def test_stage0_session_source_status_is_lightweight():
    """session_source_status must not use os.walk or Path.rglob (recursive scanning)."""
    src = Path(__file__).resolve().parents[1] / "project_manager.py"
    text = src.read_text(encoding="utf-8")
    fn_start = text.find("def session_source_status(")
    assert fn_start != -1, "session_source_status not found"
    # Find the end of the function by looking for the next top-level def/class
    fn_body = text[fn_start:]
    next_def = fn_body.find("\ndef ", 1)
    if next_def == -1:
        next_def = len(fn_body)
    fn_body = fn_body[:next_def]
    assert "os.walk" not in fn_body, "session_source_status must not use os.walk"
    assert ".rglob(" not in fn_body, "session_source_status must not use Path.rglob"
    assert 'glob("**/' not in fn_body, "session_source_status must not use recursive glob"


def test_stage0_does_not_import_heavy_libraries():
    """Stage 0 panel source must not import GPU/DLC/heavy libraries at module level."""
    src = Path(__file__).resolve().parents[1] / "user_interface.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    heavy = {"deeplabcut", "torch", "tensorflow", "cuml", "rapids", "h5py", "hdbscan", "umap"}
    # Collect top-level imports (not inside function/class bodies)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for name in names:
                root_pkg = (name or "").split(".")[0].lower()
                assert root_pkg not in heavy, f"user_interface.py imports heavy library at top level: {name}"
