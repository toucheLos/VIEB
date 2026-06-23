"""Project detection, validation, and path resolution for VIEB."""

from __future__ import annotations

import csv
import json
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
APP_CONFIG_PATH = REPO_ROOT / "app_config.json"
PROJECTS_DIR = REPO_ROOT / "projects"
PROJECT_CONFIG_NAME = "config.json"

KNOWN_RESULT_FILES = (
    Path("results/features/index.json"),
    Path("results/shared/cluster_info.json"),
    Path("results/comparison/summary_table.csv"),
)

LEGACY_MARKERS = (
    Path("metadata.csv"),
    Path("raw_videos"),
    Path("results/features/index.json"),
    Path("results/shared/cluster_info.json"),
    Path("results/comparison/summary_table.csv"),
)


class ProjectSelectionError(RuntimeError):
    """Raised when VIEB cannot resolve a valid active project."""


@dataclass
class Check:
    key: str
    label: str
    status: str
    message: str = ""


@dataclass
class ProjectValidation:
    path: Path
    valid: bool
    checks: list[Check] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def project_name(self) -> str:
        return str(self.config.get("project_name") or self.path.name)

    @property
    def status(self) -> str:
        if self.valid:
            return "valid"
        if any(c.status == "yellow" for c in self.checks):
            return "needs_attention"
        return "invalid"

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "valid": self.valid,
            "status": self.status,
            "project_name": self.project_name,
            "checks": [c.__dict__ for c in self.checks],
        }


@dataclass
class StartupSelection:
    action: str
    active_project: Path | None = None
    candidates: list[Path] = field(default_factory=list)
    validation: ProjectValidation | None = None
    legacy_detected: bool = False
    message: str = ""


def load_app_config(app_config_path: Path | str = APP_CONFIG_PATH) -> dict[str, Any]:
    path = Path(app_config_path)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("active_project", "")
                data.setdefault("recent_projects", [])
                data.setdefault("projects", [])
                return data
        except Exception:
            pass
    return {"active_project": "", "recent_projects": [], "projects": []}


def save_app_config(data: dict[str, Any], app_config_path: Path | str = APP_CONFIG_PATH) -> None:
    path = Path(app_config_path)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _abs(path: Path | str | None, base: Path) -> Path | None:
    if path in (None, ""):
        return None
    p = Path(str(path)).expanduser()
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _read_project_config(project_path: Path) -> dict[str, Any]:
    cfg_path = project_path / PROJECT_CONFIG_NAME
    if not cfg_path.exists():
        return {}
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def normalize_project_config(config: dict[str, Any], project_path: Path) -> dict[str, Any]:
    """Return a flat-compatible config with canonical nested paths populated."""
    cfg = json.loads(json.dumps(config or {}))
    paths = dict(cfg.get("paths") or {})

    aliases = {
        "raw_videos": "raw_videos_dir",
        "pose_files": "pose_files_dir",
        "pose_h5": "h5_path",
        "metadata": "metadata_csv_path",
        "results": "results_dir",
        "external_data_root": "external_data_root",
    }
    for nested_key, flat_key in aliases.items():
        if not paths.get(nested_key) and cfg.get(flat_key):
            paths[nested_key] = cfg.get(flat_key)

    paths.setdefault("raw_videos", str(project_path / "raw_videos"))
    paths.setdefault("pose_files", "")
    paths.setdefault("pose_h5", cfg.get("h5_path") or None)
    paths.setdefault("metadata", str(project_path / "metadata.csv"))
    paths.setdefault("results", str(project_path / "results"))

    cfg["paths"] = paths
    cfg.setdefault("project_name", project_path.name)
    cfg.setdefault("metadata_schema", cfg.get("metadata_schema") or {})
    cfg.setdefault("analysis_groups", cfg.get("analysis_groups") or [])
    cfg.setdefault("ui_panels", cfg.get("ui_panels") or {})
    cfg.setdefault("pipeline_settings", cfg.get("pipeline_settings") or {})

    cfg["raw_videos_dir"] = str(_abs(paths.get("raw_videos"), project_path) or "")
    cfg["pose_files_dir"] = str(_abs(paths.get("pose_files"), project_path) or "")
    cfg["h5_path"] = str(_abs(paths.get("pose_h5"), project_path) or "")
    cfg["metadata_csv_path"] = str(_abs(paths.get("metadata"), project_path) or "")
    cfg["results_dir"] = str(_abs(paths.get("results"), project_path) or "")
    cfg["project_path"] = str(project_path)
    return cfg


def write_project_config(project_path: Path | str, config: dict[str, Any]) -> None:
    project = Path(project_path).resolve()
    cfg = normalize_project_config(config, project)
    (project / PROJECT_CONFIG_NAME).write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def project_indicators(path: Path | str) -> list[str]:
    project = Path(path)
    indicators: list[str] = []
    for rel in (Path("config.json"), Path("metadata.csv"), Path("results"), Path("raw_videos"), *KNOWN_RESULT_FILES):
        if (project / rel).exists():
            indicators.append(str(rel))
    return indicators


def is_valid_project(path: Path | str) -> bool:
    return validate_project(path).valid


def validate_project(path: Path | str) -> ProjectValidation:
    project = Path(path).expanduser().resolve()
    checks: list[Check] = []

    exists = project.exists() and project.is_dir()
    checks.append(Check("project_dir", "project directory exists", "green" if exists else "red", str(project)))
    if not exists:
        return ProjectValidation(project, False, checks, {})

    cfg_path = project / PROJECT_CONFIG_NAME
    cfg = _read_project_config(project)
    if cfg_path.exists():
        writable = os.access(cfg_path, os.W_OK)
        checks.append(Check("config", "project config exists/writable", "green" if writable else "yellow", str(cfg_path)))
    else:
        checks.append(Check("config", "project config exists/writable", "yellow", "config.json can be created"))

    cfg = normalize_project_config(cfg, project)
    meta = Path(cfg["metadata_csv_path"])
    meta_parent_ok = meta.exists() or meta.parent.exists()
    checks.append(Check("metadata", "metadata path exists or can be created", "green" if meta.exists() else ("yellow" if meta_parent_ok else "red"), str(meta)))

    raw = Path(cfg["raw_videos_dir"]) if cfg.get("raw_videos_dir") else None
    pose_files = Path(cfg["pose_files_dir"]) if cfg.get("pose_files_dir") else None
    pose_h5 = Path(cfg["h5_path"]) if cfg.get("h5_path") else None
    source_ok = any(p and p.exists() for p in (raw, pose_files, pose_h5))
    checks.append(Check("pose_source", "raw video / pose source exists", "green" if source_ok else "yellow", "set raw_videos, pose_files, or pose_h5"))

    results = Path(cfg["results_dir"])
    results_ok = results.exists() and os.access(results, os.W_OK)
    results_parent_ok = results.exists() or results.parent.exists()
    checks.append(Check("results", "results directory exists/writable", "green" if results_ok else ("yellow" if results_parent_ok else "red"), str(results)))

    feature = results / "features" / "index.json"
    cluster = results / "shared" / "cluster_info.json"
    report = results / "comparison" / "summary_table.csv"
    motif = (results / "comparison" / "motifs.csv").exists() or (results / "motifs" / "motif_summary.csv").exists()
    checks.append(Check("features", "previous feature results detected", "green" if feature.exists() else "yellow", str(feature)))
    checks.append(Check("clusters", "previous cluster results detected", "green" if cluster.exists() else "yellow", str(cluster)))
    checks.append(Check("reports", "previous report outputs detected", "green" if report.exists() else "yellow", str(report)))
    checks.append(Check("motifs", "motif outputs detected", "green" if motif else "yellow", "optional"))
    checks.append(Check("metadata_schema", "metadata schema status", "green" if cfg.get("metadata_schema") or cfg.get("column_map") else "yellow", "configure canonical column mapping"))
    groups = cfg.get("analysis_groups") or cfg.get("metadata_schema", {}).get("analysis_groups") or []
    checks.append(Check("analysis_groups", "configured analysis groups", "green" if groups else "yellow", f"{len(groups)} group(s)"))

    valid = bool(project_indicators(project))
    return ProjectValidation(project, valid, checks, cfg)


def _candidate_paths(app_cfg: dict[str, Any], repo_root: Path) -> list[Path]:
    paths: list[Path] = []
    projects_dir = repo_root / "projects"
    if projects_dir.exists():
        paths.extend(p for p in projects_dir.iterdir() if p.is_dir())
    for raw in app_cfg.get("recent_projects", []):
        if raw:
            paths.append(Path(raw))
    for entry in app_cfg.get("projects", []):
        raw = entry.get("path") if isinstance(entry, dict) else ""
        if raw:
            paths.append(Path(raw))
    if app_cfg.get("active_project"):
        paths.append(Path(app_cfg["active_project"]))
    seen: set[Path] = set()
    uniq: list[Path] = []
    for p in paths:
        q = p.expanduser()
        if not q.is_absolute():
            q = repo_root / q
        q = q.resolve()
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    return uniq


def detect_projects(
    search_roots: list[str] | None = None,
    repo_root: Path | str = REPO_ROOT,
    app_config_path: Path | str | None = None,
) -> list[ProjectValidation]:
    root = Path(repo_root).resolve()
    app_cfg = load_app_config(app_config_path or root / "app_config.json")
    candidates = _candidate_paths(app_cfg, root)
    for raw in search_roots or []:
        p = Path(raw)
        if not p.is_absolute():
            p = root / p
        p = p.resolve()
        if p.exists() and p.is_dir():
            if p.name == "projects":
                candidates.extend(child for child in p.iterdir() if child.is_dir())
            else:
                candidates.append(p)
    seen: set[Path] = set()
    out: list[ProjectValidation] = []
    for candidate in candidates:
        c = candidate.resolve()
        if c in seen or c == root:
            continue
        seen.add(c)
        validation = validate_project(c)
        if validation.valid:
            out.append(validation)
    return out


def detect_legacy_project(repo_root: Path | str = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    markers = [str(rel) for rel in LEGACY_MARKERS if (root / rel).exists()]
    return {"detected": bool(markers), "path": str(root), "markers": markers}


def _remember_project(app_cfg: dict[str, Any], project_path: Path, name: str) -> dict[str, Any]:
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    project_s = str(project_path)
    app_cfg["active_project"] = project_s
    recent = [p for p in app_cfg.get("recent_projects", []) if str(Path(p).resolve()) != project_s]
    app_cfg["recent_projects"] = [project_s, *recent][:20]
    projects = [p for p in app_cfg.get("projects", []) if str(Path(p.get("path", "")).resolve()) != project_s]
    projects.insert(0, {"name": name, "path": project_s, "last_opened": now})
    app_cfg["projects"] = projects[:20]
    return app_cfg


def select_startup_project(repo_root: Path | str = REPO_ROOT, app_config_path: Path | str | None = None) -> StartupSelection:
    root = Path(repo_root).resolve()
    app_path = Path(app_config_path) if app_config_path else root / "app_config.json"
    app_cfg = load_app_config(app_path)
    legacy = detect_legacy_project(root)["detected"]

    active_raw = app_cfg.get("active_project") or ""
    if active_raw:
        active = Path(active_raw)
        if not active.is_absolute():
            active = root / active
        active = active.resolve()
        validation = validate_project(active)
        if validation.valid:
            _remember_project(app_cfg, active, validation.project_name)
            save_app_config(app_cfg, app_path)
            return StartupSelection("use_active", active, [active], validation, legacy)

    candidates = detect_projects(repo_root=root, app_config_path=app_path)
    paths = [c.path for c in candidates]
    if len(candidates) == 1:
        validation = candidates[0]
        _remember_project(app_cfg, validation.path, validation.project_name)
        save_app_config(app_cfg, app_path)
        return StartupSelection("auto_selected", validation.path, paths, validation, legacy)
    if len(candidates) > 1:
        return StartupSelection("picker_required", None, paths, None, legacy, "Multiple valid projects found.")
    return StartupSelection("onboarding_required", None, [], None, legacy, "No valid project selected. Complete Project Onboarding before running the pipeline.")


def get_active_project(repo_root: Path | str = REPO_ROOT, app_config_path: Path | str | None = None) -> Path:
    selected = select_startup_project(repo_root, app_config_path)
    if selected.active_project is None:
        raise ProjectSelectionError(selected.message or "No valid project selected. Complete Project Onboarding before running the pipeline.")
    return selected.active_project


def load_active_project_config(repo_root: Path | str = REPO_ROOT, app_config_path: Path | str | None = None) -> dict[str, Any]:
    project = get_active_project(repo_root, app_config_path)
    return normalize_project_config(_read_project_config(project), project)


def resolve_project_path(key: str, repo_root: Path | str = REPO_ROOT, app_config_path: Path | str | None = None) -> Path:
    project = get_active_project(repo_root, app_config_path)
    cfg = load_active_project_config(repo_root, app_config_path)
    paths = cfg.get("paths", {})
    value = {
        "raw_videos": cfg.get("raw_videos_dir") or paths.get("raw_videos"),
        "pose_files": cfg.get("pose_files_dir") or paths.get("pose_files"),
        "pose_h5": cfg.get("h5_path") or paths.get("pose_h5"),
        "metadata": cfg.get("metadata_csv_path") or paths.get("metadata"),
        "results": cfg.get("results_dir") or paths.get("results"),
        "config": str(project / "config.json"),
        "project": str(project),
    }.get(key)
    if not value:
        raise ProjectSelectionError(f"Project path '{key}' is not configured.")
    resolved = _abs(value, project)
    if resolved is None:
        raise ProjectSelectionError(f"Project path '{key}' is not configured.")
    root = Path(repo_root).resolve()
    if resolved in (root / "metadata.csv", root / "raw_videos", root / "results") and project != root:
        raise ProjectSelectionError(
            f"Refusing repo-root fallback for {key}. Complete Project Onboarding before running the pipeline."
        )
    return resolved


def create_project(project_path: Path | str, project_name: str, paths: dict[str, Any] | None = None, repo_root: Path | str = REPO_ROOT, app_config_path: Path | str | None = None) -> Path:
    project = Path(project_path)
    if not project.is_absolute():
        project = Path(repo_root).resolve() / project
    project = project.resolve()
    project.mkdir(parents=True, exist_ok=True)
    (project / "raw_videos").mkdir(exist_ok=True)
    (project / "results").mkdir(exist_ok=True)
    (project / "logs").mkdir(exist_ok=True)
    template = project / "metadata_template.csv"
    if not template.exists():
        with template.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["session_id", "stem", "subject_id", "animal_id", "context", "condition", "day", "timepoint"])
    cfg = {
        "project_name": project_name,
        "paths": {
            "raw_videos": str((project / "raw_videos").resolve()),
            "pose_files": "",
            "pose_h5": None,
            "metadata": str(template.resolve()),
            "results": str((project / "results").resolve()),
            **(paths or {}),
        },
        "metadata_schema": {},
        "analysis_groups": [],
        "ui_panels": {},
        "pipeline_settings": {
            "fps": 30,
            "confidence_threshold": 0.7,
            "use_wavelets": True,
            "umap_dims": 10,
            "hdbscan_min_cluster_size": 2000,
            "hdbscan_min_samples": 0,
            "hdbscan_sample_size": 100000,
        },
    }
    write_project_config(project, cfg)
    app_path = Path(app_config_path) if app_config_path else Path(repo_root).resolve() / "app_config.json"
    app_cfg = load_app_config(app_path)
    _remember_project(app_cfg, project, project_name)
    save_app_config(app_cfg, app_path)
    return project


def register_legacy_project(repo_root: Path | str = REPO_ROOT, app_config_path: Path | str | None = None, name: str = "legacy_project") -> Path:
    root = Path(repo_root).resolve()
    paths = {
        "raw_videos": str(root / "raw_videos"),
        "pose_files": "",
        "pose_h5": None,
        "metadata": str(root / "metadata.csv"),
        "results": str(root / "results"),
    }
    cfg = normalize_project_config({
        "project_name": name,
        "paths": paths,
        "metadata_schema": {},
        "analysis_groups": [],
        "ui_panels": {},
        "pipeline_settings": {},
    }, root)
    write_project_config(root, cfg)
    app_path = Path(app_config_path) if app_config_path else root / "app_config.json"
    app_cfg = load_app_config(app_path)
    _remember_project(app_cfg, root, name)
    save_app_config(app_cfg, app_path)
    return root


def migrate_legacy_project(repo_root: Path | str = REPO_ROOT, app_config_path: Path | str | None = None, name: str = "legacy_project") -> Path:
    root = Path(repo_root).resolve()
    project = root / "projects" / name
    paths = {
        "raw_videos": str(root / "raw_videos"),
        "pose_files": "",
        "pose_h5": None,
        "metadata": str(root / "metadata.csv"),
        "results": str(root / "results"),
    }
    create_project(project, name, paths=paths, repo_root=root, app_config_path=app_config_path)
    if (root / "config.json").exists() and not (project / "legacy_root_config.json").exists():
        shutil.copy2(root / "config.json", project / "legacy_root_config.json")
    return project


def resume_status(project_path: Path | str) -> dict[str, bool]:
    validation = validate_project(project_path)
    results = Path(validation.config.get("results_dir") or Path(project_path) / "results")
    return {
        "features": (results / "features" / "index.json").exists(),
        "clusters": (results / "shared" / "cluster_info.json").exists(),
        "reports": (results / "comparison" / "summary_table.csv").exists(),
        "motifs": (results / "comparison" / "motifs.csv").exists() or (results / "motifs" / "motif_summary.csv").exists(),
    }
