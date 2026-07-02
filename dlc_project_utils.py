from __future__ import annotations

import re
from datetime import date
from pathlib import Path


def normalize_dlc_project_path(path: str | Path | None) -> Path | None:
    """Return the DLC project directory for either a folder or config.yaml path."""
    if not path:
        return None
    p = Path(path).expanduser()
    if p.name == "config.yaml":
        p = p.parent
    return p.resolve()


def dlc_config_path(path: str | Path | None) -> Path | None:
    project = normalize_dlc_project_path(path)
    return project / "config.yaml" if project else None


def is_valid_dlc_project(path: str | Path | None) -> bool:
    config = dlc_config_path(path)
    return bool(config and config.exists())


def find_trained_dlc_snapshots(path: str | Path | None) -> list[Path]:
    """Return trained DLC snapshot files for TensorFlow or PyTorch projects."""
    project = normalize_dlc_project_path(path)
    if not project or not project.exists():
        return []
    patterns = (
        "dlc-models/**/train/snapshot-*.index",
        "dlc-models-pytorch/**/train/snapshot-*.pt",
    )
    snapshots: list[Path] = []
    for pattern in patterns:
        snapshots.extend(project.glob(pattern))
    return sorted(p for p in snapshots if p.is_file())


def has_trained_dlc_model(path: str | Path | None) -> bool:
    """Return True when a DLC project has at least one trained snapshot."""
    return bool(find_trained_dlc_snapshots(path))


def default_dlc_task_name(project_name: str | None, today: date | None = None) -> str:
    """Build VIEB's editable default DLC task name."""
    today = today or date.today()
    base = (project_name or "VIEB Project").strip() or "VIEB Project"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", base).strip("-._")
    safe = re.sub(r"-{2,}", "-", safe) or "VIEB"
    return f"DLC-{safe}-{today:%Y-%m-%d}"


def expected_dlc_project_dir(
    root: str | Path,
    task: str,
    scorer: str,
    today: date | None = None,
) -> Path:
    today = today or date.today()
    return Path(root).expanduser().resolve() / f"{task}-{scorer}-{today:%Y-%m-%d}"


def discover_dlc_projects(root: str | Path) -> list[Path]:
    """Find valid DLC projects, preferring new DLC names and keeping legacy support."""
    root_path = Path(root)
    patterns = ("DLC*/config.yaml", "VIEB-*/config.yaml", "*DLC*/config.yaml")
    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for config in sorted(root_path.glob(pattern)):
            project = normalize_dlc_project_path(config)
            if project and project not in seen and (project / "config.yaml").exists():
                seen.add(project)
                found.append(project)
    return found
