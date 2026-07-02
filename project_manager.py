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
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

LEGACY_MARKERS = (
    Path("metadata.csv"),
    Path("raw_videos"),
    Path("results/features/index.json"),
    Path("results/shared/cluster_info.json"),
    Path("results/comparison/summary_table.csv"),
)

_startup_cache: dict = {}

# Per-process memoization of the expensive validation work
# (session_source_status / validate_metadata_csv / _count_files). Keyed on
# (path, mtime) so a genuine filesystem change refreshes the entry, while
# repeated calls within one onboarding/startup pass reuse the result. Cleared
# by invalidate_project_cache() alongside _startup_cache.
_source_status_cache: dict = {}
_metadata_validation_cache: dict = {}
_count_files_cache: dict = {}


def _mtime(path: Path | str | None) -> float:
    """Return the mtime of ``path`` (or -1.0 when it is missing/unreadable)."""
    if not path:
        return -1.0
    try:
        return Path(path).stat().st_mtime
    except Exception:
        return -1.0


def _startup_cache_key(app_path: Path) -> str:
    try:
        if not app_path.exists():
            return ""
        mtime = app_path.stat().st_mtime
        active = json.loads(app_path.read_text(encoding="utf-8")).get("active_project", "")
        return f"{active}|{mtime:.6f}"
    except Exception:
        return ""


def invalidate_project_cache() -> None:
    _startup_cache.clear()
    _source_status_cache.clear()
    _metadata_validation_cache.clear()
    _count_files_cache.clear()


def _validate_metadata_csv_cached(meta_path: Path | str) -> dict:
    """Memoized wrapper around metadata_generator.validate_metadata_csv.

    Keyed on (path, mtime) so the expensive pandas pass runs once per file
    version instead of on every validation call.
    """
    meta_str = str(meta_path)
    key = (meta_str, _mtime(meta_path))
    cached = _metadata_validation_cache.get(key)
    if cached is not None:
        return cached
    from metadata_generator import validate_metadata_csv

    report = validate_metadata_csv(meta_str)
    _metadata_validation_cache[key] = report
    return report


class ProjectSelectionError(RuntimeError):
    """Raised when VIEB cannot resolve a valid active project."""


@dataclass
class Check:
    key: str
    label: str
    status: str
    message: str = ""


@dataclass
class ResolvedProjectPath:
    key: str
    path: Path
    origin: str
    valid: bool
    message: str = ""


@dataclass
class ProjectValidation:
    path: Path
    valid: bool
    checks: list[Check] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    # Populated by validate_project() so callers can reuse the resolved paths
    # and session-source status it already computes instead of recomputing
    # them. None when validation returned early (project missing).
    source: dict[str, Any] | None = None
    paths: dict[str, "ResolvedProjectPath"] | None = None

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
    invalidate_project_cache()
    path = Path(app_config_path)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _abs(path: Path | str | None, base: Path) -> Path | None:
    if path in (None, ""):
        return None
    p = Path(str(path)).expanduser()
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def _path_is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _repo_root_for_project(project_path: Path, repo_root: Path | str | None = None) -> Path:
    if repo_root is not None:
        return Path(repo_root).resolve()
    project = project_path.resolve()
    if project.parent.name == "projects":
        return project.parent.parent.resolve()
    return REPO_ROOT.resolve()


def _external_path_keys(config: dict[str, Any]) -> set[str]:
    raw = config.get("external_paths", [])
    if isinstance(raw, dict):
        return {str(k) for k, v in raw.items() if v}
    if isinstance(raw, (list, tuple, set)):
        return {str(item) for item in raw}
    return set()


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


def _configured_path_value(config: dict[str, Any], key: str, project_path: Path) -> tuple[Any, bool]:
    paths = config.get("paths") if isinstance(config.get("paths"), dict) else {}
    flat_keys = {
        "raw_videos": "raw_videos_dir",
        "pose_files": "pose_files_dir",
        "pose_h5": "h5_path",
        "metadata": "metadata_csv_path",
        "results": "results_dir",
    }
    if key in paths and paths.get(key) not in (None, ""):
        return paths.get(key), True
    flat_key = flat_keys.get(key)
    if flat_key and config.get(flat_key) not in (None, ""):
        return config.get(flat_key), True
    defaults = {
        "raw_videos": project_path / "raw_videos",
        "metadata": project_path / "metadata.csv",
        "results": project_path / "results",
        "config": project_path / PROJECT_CONFIG_NAME,
        "project": project_path,
    }
    return defaults.get(key), False


def _classify_project_path(
    key: str,
    value: Any,
    *,
    configured: bool,
    project_path: Path,
    repo_root: Path,
    external_keys: set[str],
) -> ResolvedProjectPath:
    resolved = _abs(value, project_path)
    if resolved is None:
        return ResolvedProjectPath(key, project_path / key, "project_config", False, f"{key} path is not configured.")

    if key in {"config", "project", "pose_files", "pose_h5"}:
        origin = "project_config" if configured else "project_default"
        return ResolvedProjectPath(key, resolved, origin, True, f"{resolved} ({origin})")

    defaults = {
        "raw_videos": project_path / "raw_videos",
        "metadata": project_path / "metadata.csv",
        "results": project_path / "results",
    }
    repo_root_paths = {
        "raw_videos": repo_root / "raw_videos",
        "metadata": repo_root / "metadata.csv",
        "results": repo_root / "results",
    }
    project_default = defaults[key].resolve()
    repo_root_path = repo_root_paths[key].resolve()
    project_is_repo_root = project_path.resolve() == repo_root.resolve()
    project_local = resolved == project_default or _path_is_relative_to(resolved, project_path)

    if project_is_repo_root and resolved == repo_root_path:
        origin = "legacy_repo_root" if configured else "project_default"
        return ResolvedProjectPath(key, resolved, origin, True, f"{resolved} ({origin})")

    if resolved == repo_root_path and not project_is_repo_root:
        if key in external_keys:
            return ResolvedProjectPath(
                key,
                resolved,
                "external_explicit",
                True,
                f"{resolved} (explicit external repo-root path)",
            )
        return ResolvedProjectPath(
            key,
            resolved,
            "invalid_repo_root_fallback",
            False,
            f"{resolved} is a repo-root legacy path; migrate it into the project or mark it external.",
        )

    if project_local:
        origin = "project_config" if configured else "project_default"
        return ResolvedProjectPath(key, resolved, origin, True, f"{resolved} ({origin})")

    if key in external_keys:
        return ResolvedProjectPath(key, resolved, "external_explicit", True, f"{resolved} (explicit external path)")

    return ResolvedProjectPath(
        key,
        resolved,
        "project_config",
        False,
        f"{resolved} is outside the project; add '{key}' to external_paths if intentional.",
    )


def resolve_project_paths_for_project(
    project_path: Path | str,
    config: dict[str, Any] | None = None,
    repo_root: Path | str | None = None,
) -> dict[str, ResolvedProjectPath]:
    project = Path(project_path).expanduser().resolve()
    root = _repo_root_for_project(project, repo_root)
    raw_cfg = config if config is not None else _read_project_config(project)
    external_keys = _external_path_keys(raw_cfg)
    out: dict[str, ResolvedProjectPath] = {}
    for key in ("project", "config", "raw_videos", "metadata", "results", "pose_files", "pose_h5"):
        value, configured = _configured_path_value(raw_cfg, key, project)
        if value in (None, "") and key in {"pose_files", "pose_h5"}:
            continue
        out[key] = _classify_project_path(
            key,
            value,
            configured=configured,
            project_path=project,
            repo_root=root,
            external_keys=external_keys,
        )
    return out


def resolve_project_paths(
    repo_root: Path | str = REPO_ROOT,
    app_config_path: Path | str | None = None,
) -> dict[str, ResolvedProjectPath]:
    root = Path(repo_root).resolve()
    app_path = Path(app_config_path) if app_config_path else root / "app_config.json"
    app_cfg = load_app_config(app_path)
    active_raw = app_cfg.get("active_project") or ""
    if not active_raw:
        raise ProjectSelectionError(
            "No active project. Complete Stage 0: Onboarding before running the pipeline."
        )
    project = Path(active_raw)
    if not project.is_absolute():
        project = root / project
    project = project.resolve()
    if not project.is_dir():
        raise ProjectSelectionError(
            f"Active project directory does not exist: {project}. "
            "Complete Stage 0: Onboarding before running the pipeline."
        )
    cfg = _read_project_config(project)
    return resolve_project_paths_for_project(project, cfg, root)


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


def validate_project(path: Path | str, repo_root: Path | str | None = None) -> ProjectValidation:
    project = Path(path).expanduser().resolve()
    root = _repo_root_for_project(project, repo_root)
    checks: list[Check] = []

    exists = project.exists() and project.is_dir()
    checks.append(Check("active_project", "active project exists", "green" if exists else "red", str(project)))
    if not exists:
        return ProjectValidation(project, False, checks, {})

    cfg_path = project / PROJECT_CONFIG_NAME
    cfg = _read_project_config(project)
    if cfg_path.exists():
        writable = os.access(cfg_path, os.W_OK)
        checks.append(Check("config", "project config exists/writable", "green" if writable else "yellow", str(cfg_path)))
    else:
        checks.append(Check("config", "project config exists/writable", "yellow", "config.json can be created"))

    paths = resolve_project_paths_for_project(project, cfg, root)
    cfg = normalize_project_config(cfg, project)
    meta_info = paths["metadata"]
    meta = meta_info.path
    if not meta_info.valid:
        checks.append(Check("metadata", "metadata exists or can be generated", "red", meta_info.message))
    else:
        meta_parent_ok = meta.exists() or meta.parent.exists()
        meta_status = "green" if meta.exists() else ("yellow" if meta_parent_ok else "red")
        checks.append(Check("metadata", "metadata exists or can be generated", meta_status, meta_info.message))

    source = session_source_status(project, cfg, root)
    source_msg = source["message"] if source["valid"] else "import raw videos, pose CSVs, H5 pose data, or metadata CSV"
    checks.append(Check("pose_source", "session-defining data source detected", "green" if source["valid"] else "yellow", source_msg))

    results_info = paths["results"]
    results = results_info.path
    if not results_info.valid:
        checks.append(Check("results", "results directory exists/writable", "red", results_info.message))
    else:
        results_ok = results.exists() and os.access(results, os.W_OK)
        results_parent_ok = results.exists() or results.parent.exists()
        results_status = "green" if results_ok else ("yellow" if results_parent_ok else "red")
        checks.append(Check("results", "results directory exists/writable", results_status, results_info.message))

    feature = results / "features" / "index.json"
    cluster = results / "shared" / "cluster_info.json"
    report = results / "comparison" / "summary_table.csv"
    motif = (results / "comparison" / "motifs.csv").exists() or (results / "motifs" / "motif_summary.csv").exists()
    checks.append(Check("features", "previous feature outputs detected", "green" if feature.exists() else "yellow", str(feature)))
    checks.append(Check("clusters", "previous clustering outputs detected", "green" if cluster.exists() else "yellow", str(cluster)))
    checks.append(Check("reports", "previous report outputs detected", "green" if report.exists() else "yellow", str(report)))
    checks.append(Check("motifs", "motif outputs detected", "green" if motif else "yellow", "optional"))
    valid = bool(project_indicators(project))
    return ProjectValidation(project, valid, checks, cfg, source=source, paths=paths)


def _count_files(folder: Path | str | None, suffixes: set[str]) -> int:
    if not folder:
        return 0
    path = Path(folder)
    if not path.exists() or not path.is_dir():
        return 0
    # Cache the count per (directory, mtime, suffixes): the dir mtime changes
    # when files are added/removed, so a stale count self-invalidates. Avoids a
    # full iterdir() on every repeated call within one validation pass.
    key = (str(path.resolve()), _mtime(path), tuple(sorted(suffixes)))
    cached = _count_files_cache.get(key)
    if cached is not None:
        return cached
    count = sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in suffixes)
    _count_files_cache[key] = count
    return count


def _metadata_row_count(path: Path | str | None) -> int:
    if not path:
        return 0
    meta = Path(path)
    if not meta.exists() or not meta.is_file():
        return 0
    try:
        with meta.open(newline="", encoding="utf-8") as f:
            return max(0, sum(1 for row in csv.reader(f) if any(cell.strip() for cell in row)) - 1)
    except Exception:
        return 0


def session_source_status(
    project_path: Path | str,
    config: dict[str, Any] | None = None,
    repo_root: Path | str | None = None,
) -> dict[str, Any]:
    """Return whether the project has at least one source that defines sessions."""
    project = Path(project_path).resolve()
    raw_cfg = config if config is not None else _read_project_config(project)
    cfg = normalize_project_config(raw_cfg, project)
    root = _repo_root_for_project(project, repo_root)

    # Memoize the whole result: every dependency below is captured by an mtime
    # in the key, so adding/removing files or editing config/metadata refreshes
    # the entry while repeated calls in one pass reuse it.
    cache_key = (
        str(project), str(root),
        _mtime(project / PROJECT_CONFIG_NAME),
        _mtime(cfg.get("metadata_csv_path")),
        _mtime(cfg.get("raw_videos_dir")),
        _mtime(cfg.get("pose_files_dir")),
        _mtime(cfg.get("h5_path")),
    )
    cached = _source_status_cache.get(cache_key)
    if cached is not None:
        return cached

    paths = resolve_project_paths_for_project(project, raw_cfg, root)
    raw_info = paths["raw_videos"]
    meta_info = paths["metadata"]
    raw_count = _count_files(raw_info.path if raw_info.valid else None, VIDEO_EXTENSIONS)
    pose_csv_count = _count_files(cfg.get("pose_files_dir"), {".csv"})
    h5_path = Path(cfg["h5_path"]) if cfg.get("h5_path") else None
    h5_exists = bool(h5_path and h5_path.exists() and h5_path.is_file())
    metadata_rows = _metadata_row_count(meta_info.path if meta_info.valid else None)
    metadata_valid = False
    if metadata_rows:
        try:
            metadata_valid = bool(_validate_metadata_csv_cached(meta_info.path).get("valid"))
        except Exception:
            metadata_valid = False

    parts = []
    if raw_count:
        parts.append(f"{raw_count} raw video(s)")
    if pose_csv_count:
        parts.append(f"{pose_csv_count} pose CSV(s)")
    if h5_exists:
        parts.append("H5 pose file")
    if metadata_valid:
        parts.append(f"{metadata_rows} metadata row(s)")

    result = {
        "valid": bool(raw_count or pose_csv_count or h5_exists or metadata_valid),
        "raw_videos": raw_count,
        "pose_csvs": pose_csv_count,
        "h5": h5_exists,
        "metadata_rows": metadata_rows,
        "metadata_valid": metadata_valid,
        "message": ", ".join(parts) if parts else "no session-defining source detected",
    }
    _source_status_cache[cache_key] = result
    return result


def column_mapping_status(project_path: Path | str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return whether metadata has a resolvable session identifier column."""
    project = Path(project_path).resolve()
    raw_cfg = config if config is not None else _read_project_config(project)
    cfg = normalize_project_config(raw_cfg, project)
    paths = resolve_project_paths_for_project(project, raw_cfg, _repo_root_for_project(project))
    meta_info = paths["metadata"]
    meta_path = meta_info.path
    if not meta_info.valid or not meta_path.exists() or not meta_path.is_file():
        return {"mapped": False, "reason": "no_metadata"}

    try:
        with meta_path.open(newline="", encoding="utf-8-sig") as f:
            headers = next(csv.reader(f), [])
    except Exception:
        return {"mapped": False, "reason": "unreadable"}

    lower_headers = {str(h).lower(): str(h) for h in headers}
    mapped_col = (
        cfg.get("metadata_schema", {})
        .get("column_map", {})
        .get("session_id")
    )
    if mapped_col:
        original = lower_headers.get(str(mapped_col).lower())
        if original is not None:
            return {
                "mapped": True,
                "session_id_column": original,
                "reason": "explicit",
            }

    from metadata_schema import SESSION_ALIASES

    for alias in SESSION_ALIASES:
        original = lower_headers.get(str(alias).lower())
        if original is not None:
            return {
                "mapped": True,
                "session_id_column": original,
                "auto_detected": True,
                "reason": "alias",
            }
    return {"mapped": False, "reason": "not_found", "headers": headers}


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
        validation = validate_project(c, root)
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

    cache_key = _startup_cache_key(app_path)
    if cache_key and _startup_cache.get("key") == cache_key:
        return _startup_cache["result"]

    app_cfg = load_app_config(app_path)
    legacy = detect_legacy_project(root)["detected"]

    active_raw = app_cfg.get("active_project") or ""
    if active_raw:
        active = Path(active_raw)
        if not active.is_absolute():
            active = root / active
        active = active.resolve()
        validation = validate_project(active, root)
        if validation.valid:
            _remember_project(app_cfg, active, validation.project_name)
            save_app_config(app_cfg, app_path)
            result = StartupSelection("use_active", active, [active], validation, legacy)
            _startup_cache["key"] = _startup_cache_key(app_path)
            _startup_cache["result"] = result
            return result

    candidates = detect_projects(repo_root=root, app_config_path=app_path)
    paths = [c.path for c in candidates]
    if len(candidates) == 1:
        validation = candidates[0]
        _remember_project(app_cfg, validation.path, validation.project_name)
        save_app_config(app_cfg, app_path)
        result = StartupSelection("auto_selected", validation.path, paths, validation, legacy)
        _startup_cache["key"] = _startup_cache_key(app_path)
        _startup_cache["result"] = result
        return result
    if len(candidates) > 1:
        result = StartupSelection("picker_required", None, paths, None, legacy, "Multiple valid projects found.")
    else:
        result = StartupSelection("onboarding_required", None, [], None, legacy, "No valid project selected. Complete Stage 0: Onboarding before running the pipeline.")
    _startup_cache["key"] = cache_key
    _startup_cache["result"] = result
    return result


def get_active_project(repo_root: Path | str = REPO_ROOT, app_config_path: Path | str | None = None) -> Path:
    selected = select_startup_project(repo_root, app_config_path)
    if selected.active_project is None:
        raise ProjectSelectionError(selected.message or "No valid project selected. Complete Stage 0: Onboarding before running the pipeline.")
    return selected.active_project


def load_active_project_config(repo_root: Path | str = REPO_ROOT, app_config_path: Path | str | None = None) -> dict[str, Any]:
    project = get_active_project(repo_root, app_config_path)
    return normalize_project_config(_read_project_config(project), project)


def resolve_project_path(key: str, repo_root: Path | str = REPO_ROOT, app_config_path: Path | str | None = None) -> Path:
    paths = resolve_project_paths(repo_root, app_config_path)
    resolved = paths.get(key)
    if resolved is None:
        raise ProjectSelectionError(f"Project path '{key}' is not configured.")
    if not resolved.valid:
        raise ProjectSelectionError(
            f"Refusing project path for {key}: {resolved.message} Complete Stage 0: Onboarding before running the pipeline."
        )
    return resolved.path


def set_active_project(
    project_path: Path | str,
    repo_root: Path | str = REPO_ROOT,
    app_config_path: Path | str | None = None,
) -> ProjectValidation:
    """Validate, remember, and return the active project."""
    invalidate_project_cache()
    root = Path(repo_root).resolve()
    project = Path(project_path).expanduser()
    if not project.is_absolute():
        project = root / project
    validation = validate_project(project, root)
    if not validation.valid:
        raise ProjectSelectionError(f"Not a valid VIEB project: {project}")
    app_path = Path(app_config_path) if app_config_path else root / "app_config.json"
    app_cfg = load_app_config(app_path)
    _remember_project(app_cfg, validation.path, validation.project_name)
    save_app_config(app_cfg, app_path)
    return validation


def ensure_project_metadata(project_path: Path | str, *, overwrite: bool = False) -> dict[str, Any]:
    """Validate existing metadata or create metadata.csv from detected sources."""
    project = Path(project_path).resolve()
    raw_cfg = _read_project_config(project)
    paths = resolve_project_paths_for_project(project, raw_cfg)
    meta_info = paths["metadata"]
    if not meta_info.valid:
        return {"created": False, "path": meta_info.path, "valid": False, "messages": [meta_info.message]}
    cfg = normalize_project_config(raw_cfg, project)
    meta = meta_info.path
    source = session_source_status(project, cfg)
    if not source["valid"]:
        return {"created": False, "path": meta, "valid": False, "messages": ["Stage 0 requires raw videos, pose CSVs, an H5 pose file, or an existing metadata CSV."]}
    if meta.exists() and not overwrite:
        report = _validate_metadata_csv_cached(meta)
        if report.get("valid") and source["valid"]:
            return {"created": False, "path": meta, "valid": True, "messages": report.get("messages", [])}

    raw = cfg.get("raw_videos_dir") or ""
    pose_files = cfg.get("pose_files_dir") or ""
    h5 = cfg.get("h5_path") or ""
    from metadata_generator import generate_metadata_template, validate_metadata, write_metadata_csv

    df = generate_metadata_template(raw_videos_dir=raw, pose_files_dir=pose_files, h5_path=h5)
    if df.empty:
        return {
            "created": False,
            "path": meta,
            "valid": False,
            "messages": ["Could not generate metadata automatically; import raw videos, pose CSVs, an H5 pose file, or an existing metadata CSV."],
        }
    meta.parent.mkdir(parents=True, exist_ok=True)
    write_metadata_csv(df, str(meta))
    report = validate_metadata(df)
    return {"created": True, "path": meta, "valid": bool(report.get("valid")), "messages": report.get("messages", [])}


def import_data_source(
    project_path: Path | str,
    source_type: str,
    source_path: Path | str,
) -> dict[str, Any]:
    """Save a session-defining source path and create/validate project metadata."""
    project = Path(project_path).resolve()
    cfg = normalize_project_config(_read_project_config(project), project)
    paths = cfg.setdefault("paths", {})
    source = Path(source_path).expanduser()
    if not source.is_absolute():
        source = (project / source).resolve()
    else:
        source = source.resolve()

    if source_type == "raw_videos":
        if not source.is_dir():
            raise ProjectSelectionError("Raw videos source must be a folder.")
        paths["raw_videos"] = str(source)
        if not _path_is_relative_to(source, project):
            cfg.setdefault("external_paths", [])
            if "raw_videos" not in cfg["external_paths"]:
                cfg["external_paths"].append("raw_videos")
        cfg["pose_source"] = "none"
        write_project_config(project, cfg)
        report = ensure_project_metadata(project, overwrite=True)
    elif source_type == "pose_csvs":
        if not source.is_dir():
            raise ProjectSelectionError("Pose CSV source must be a folder.")
        paths["pose_files"] = str(source)
        cfg["pose_source"] = "csv"
        write_project_config(project, cfg)
        report = ensure_project_metadata(project, overwrite=True)
    elif source_type == "pose_h5":
        if not source.is_file():
            raise ProjectSelectionError("H5 pose source must be a file.")
        paths["pose_h5"] = str(source)
        cfg["pose_source"] = "h5"
        write_project_config(project, cfg)
        report = ensure_project_metadata(project, overwrite=True)
    elif source_type == "metadata":
        if not source.is_file():
            raise ProjectSelectionError("Metadata source must be a CSV file.")
        target = project / "metadata.csv"
        from metadata_generator import generate_metadata_from_manifest, validate_metadata

        normalized = generate_metadata_from_manifest(str(source), cfg, str(target))
        paths["metadata"] = str(target)
        report = {"created": source.resolve() != target.resolve(), "path": target, **validate_metadata(normalized)}
    else:
        raise ProjectSelectionError(f"Unknown data source type: {source_type}")

    if source_type == "metadata":
        write_project_config(project, cfg)
    status = session_source_status(project)
    ok = bool(report.get("valid")) and status["valid"]
    return {
        "valid": ok,
        "metadata": report,
        "source": status,
        "messages": report.get("messages", []),
    }


def onboarding_complete(project_path: Path | str, repo_root: Path | str | None = None) -> bool:
    """Return True when metadata and at least one session source are valid."""
    project = Path(project_path).resolve()
    root = _repo_root_for_project(project, repo_root)
    raw_cfg = _read_project_config(project)
    paths = resolve_project_paths_for_project(project, raw_cfg, root)
    if not paths["metadata"].valid or not paths["results"].valid or not paths["raw_videos"].valid:
        return False
    cfg = normalize_project_config(raw_cfg, project)
    source = session_source_status(project, cfg, root)
    meta = paths["metadata"].path
    if not source["valid"] or not meta.exists():
        return False
    try:
        return bool(_validate_metadata_csv_cached(meta).get("valid"))
    except Exception:
        return False


def onboard_project(
    repo_root: Path | str = REPO_ROOT,
    app_config_path: Path | str | None = None,
) -> StartupSelection:
    """Resolve the startup project and validate/create metadata when possible."""
    selected = select_startup_project(repo_root, app_config_path)
    if selected.active_project is not None:
        meta_report = ensure_project_metadata(selected.active_project)
        selected.validation = validate_project(selected.active_project)
        if not meta_report.get("valid"):
            selected.message = "; ".join(meta_report.get("messages", []))
    return selected


def create_project(project_path: Path | str, project_name: str, paths: dict[str, Any] | None = None, repo_root: Path | str = REPO_ROOT, app_config_path: Path | str | None = None) -> Path:
    project = Path(project_path)
    if not project.is_absolute():
        project = Path(repo_root).resolve() / project
    project = project.resolve()
    project.mkdir(parents=True, exist_ok=True)
    (project / "raw_videos").mkdir(exist_ok=True)
    (project / "results").mkdir(exist_ok=True)
    (project / "logs").mkdir(exist_ok=True)
    metadata = project / "metadata.csv"
    if not metadata.exists():
        with metadata.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(["session_id", "stem", "subject_id", "animal_id", "context", "condition", "day", "timepoint"])
    supplied_paths = paths or {}
    external_paths = [
        key
        for key, raw in supplied_paths.items()
        if key in {"raw_videos", "metadata", "results"} and _abs(raw, project) and not _path_is_relative_to(_abs(raw, project), project)
    ]
    cfg = {
        "project_name": project_name,
        "paths": {
            "raw_videos": str((project / "raw_videos").resolve()),
            "pose_files": "",
            "pose_h5": None,
            "metadata": str(metadata.resolve()),
            "results": str((project / "results").resolve()),
            **(paths or {}),
        },
        "external_paths": external_paths,
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
    paths = resolve_project_paths_for_project(validation.path, validation.config)
    results = paths["results"].path if paths["results"].valid else Path(project_path) / "results"
    return {
        "features": (results / "features" / "index.json").exists(),
        "clusters": (results / "shared" / "cluster_info.json").exists(),
        "reports": (results / "comparison" / "summary_table.csv").exists(),
        "motifs": (results / "comparison" / "motifs.csv").exists() or (results / "motifs" / "motif_summary.csv").exists(),
    }
