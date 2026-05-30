"""
vieb_config.py — Project-wide path resolution for VIEB.

All code that needs the DLC project directory should call get_dlc_project_path()
rather than hardcoding any directory name.  The DLC project path is resolved in
this priority order:

  1. config.json key "dlc_project_path" (explicit override)
  2. Auto-discovery: any directory in the project root matching VIEB-*-20YY-MM-DD
     that also contains a config.yaml
  3. None  — DLC not yet configured; callers decide what to do
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import Optional

PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH: str = os.path.join(PROJECT_ROOT, "config.json")
_APP_CONFIG_PATH: str = os.path.join(PROJECT_ROOT, "app_config.json")

# Standard DLC project directory naming: VIEB-<anything>-YYYY-MM-DD
_DLC_NAME_RE = re.compile(r"^VIEB-.+-20\d{2}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# config.json helpers (thin wrappers — gui.py is the authoritative writer)
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    # Check app_config.json for the active project first
    if os.path.exists(_APP_CONFIG_PATH):
        try:
            with open(_APP_CONFIG_PATH, encoding="utf-8") as f:
                app_cfg = json.load(f)
            active = app_cfg.get("active_project", "")
            if active:
                project_cfg = os.path.join(active, "config.json")
                if os.path.exists(project_cfg):
                    with open(project_cfg, encoding="utf-8") as f:
                        return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    # Fallback: ROOT/config.json
    if os.path.exists(_CONFIG_PATH):
        try:
            with open(_CONFIG_PATH, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_config(data: dict) -> None:
    config_path = _CONFIG_PATH
    if os.path.exists(_APP_CONFIG_PATH):
        try:
            with open(_APP_CONFIG_PATH, encoding="utf-8") as f:
                app_cfg = json.load(f)
            active = app_cfg.get("active_project", "")
            if active and os.path.isdir(active):
                config_path = os.path.join(active, "config.json")
        except Exception:
            pass
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# DLC project path resolution
# ---------------------------------------------------------------------------

def get_dlc_project_path() -> Optional[str]:
    """
    Resolve the DLC project directory for this installation.

    Returns None if no DLC project is found — callers should handle that case
    by showing a helpful message rather than crashing.
    """
    cfg = _load_config()

    # 1. Explicit entry in config.json
    explicit = cfg.get("dlc_project_path")
    if explicit and os.path.isdir(explicit):
        if os.path.exists(os.path.join(explicit, "config.yaml")):
            return explicit
        # Registered but config.yaml missing — fall through to discovery

    # 2. Auto-discovery: scan project root for any matching directory
    try:
        entries = sorted(os.listdir(PROJECT_ROOT))
    except OSError:
        entries = []

    for entry in entries:
        if _DLC_NAME_RE.match(entry):
            candidate = os.path.join(PROJECT_ROOT, entry)
            if os.path.isdir(candidate) and os.path.exists(
                os.path.join(candidate, "config.yaml")
            ):
                return candidate

    return None


def get_dlc_config_path() -> Optional[str]:
    """Return the path to config.yaml inside the DLC project, or None."""
    project = get_dlc_project_path()
    return os.path.join(project, "config.yaml") if project else None


def set_dlc_project_path(path: str) -> None:
    """Persist a DLC project path to config.json so future calls find it immediately."""
    cfg = _load_config()
    cfg["dlc_project_path"] = os.path.abspath(path)
    _save_config(cfg)


def get_raw_videos_dir() -> str:
    """Return the raw-videos directory, reading 'raw_videos_dir' from config.json.
    Falls back to PROJECT_ROOT/raw_videos when the key is absent or empty."""
    cfg = _load_config()
    val = cfg.get("raw_videos_dir")
    if val and isinstance(val, str) and val.strip():
        return val.strip()
    return os.path.join(PROJECT_ROOT, "raw_videos")


def get_results_dir() -> str:
    """Return the results directory, reading 'results_dir' from config.json.
    Falls back to PROJECT_ROOT/results when the key is absent or empty."""
    cfg = _load_config()
    val = cfg.get("results_dir")
    if val and isinstance(val, str) and val.strip():
        return val.strip()
    return os.path.join(PROJECT_ROOT, "results")


def get_metadata_path() -> str:
    """Return the metadata CSV path, reading 'metadata_csv_path' from config.json.
    Falls back to PROJECT_ROOT/metadata.csv when the key is absent or empty."""
    cfg = _load_config()
    val = cfg.get("metadata_csv_path")
    if val and isinstance(val, str) and val.strip():
        return val.strip()
    return os.path.join(PROJECT_ROOT, "metadata.csv")


def get_clips_dir() -> str:
    """Return the clips directory.
    Derived as the sibling of get_results_dir() named 'clips', which reproduces
    the default PROJECT_ROOT/clips when results_dir is PROJECT_ROOT/results."""
    from pathlib import Path as _Path
    return str(_Path(get_results_dir()).parent / "clips")


def require_dlc_project_path() -> str:
    """
    Like get_dlc_project_path() but exits with a clear, actionable message if
    no project is found.  Use this inside CLI commands that genuinely need DLC.
    """
    path = get_dlc_project_path()
    if path is None:
        print(
            "\n[VIEB] Error: No DLC project directory found.\n"
            "Expected: A directory named VIEB-<name>-<YYYY-MM-DD>/ in the project root,\n"
            "          OR 'dlc_project_path' key set in config.json.\n"
            "\nFix (choose one):\n"
            "  python setup_dlc_training.py                     # create & label a new project\n"
            "  python setup_dlc_training.py --use-pretrained mouse_8kp_v1  # use pretrained model\n"
        )
        sys.exit(1)
    return path
