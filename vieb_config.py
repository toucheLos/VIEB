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
from pathlib import Path

import project_manager as _pm

PROJECT_ROOT: str = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH: str = os.path.join(PROJECT_ROOT, "config.json")
_APP_CONFIG_PATH: str = os.path.join(PROJECT_ROOT, "app_config.json")

# Standard DLC project directory naming: VIEB-<anything>-YYYY-MM-DD
_DLC_NAME_RE = re.compile(r"^VIEB-.+-20\d{2}-\d{2}-\d{2}$")

_path_cache: dict[str, str] = {}


def invalidate_path_cache() -> None:
    _path_cache.clear()


# ---------------------------------------------------------------------------
# config.json helpers (thin wrappers — gui.py is the authoritative writer)
# ---------------------------------------------------------------------------

def _load_config() -> dict:
    try:
        return _pm.load_active_project_config(PROJECT_ROOT, _APP_CONFIG_PATH)
    except _pm.ProjectSelectionError:
        return {}


def _require_config() -> dict:
    try:
        return _pm.load_active_project_config(PROJECT_ROOT, _APP_CONFIG_PATH)
    except _pm.ProjectSelectionError as exc:
        raise RuntimeError(
            "No valid project selected. Complete Project Onboarding before running the pipeline."
        ) from exc


def _save_config(data: dict) -> None:
    _pm.invalidate_project_cache()
    invalidate_path_cache()
    project = _pm.get_active_project(PROJECT_ROOT, _APP_CONFIG_PATH)
    _pm.write_project_config(project, data)


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


def get_column_map() -> dict:
    """Return column_map from active project config.json.

    Falls back to identity mapping if column_map is absent or empty.
    Values of '' or '— not mapped —' are treated as unmapped (no rename).
    """
    cfg = _load_config()
    try:
        import metadata_schema as _ms
        return dict(_ms.get_metadata_schema(cfg).get("column_map", {}))
    except Exception:
        defaults = {
            "session_id": "filename",
            "animal_id":  "animal_id",
            "day":        "day",
            "context":    "context",
            "experiment": "experiment",
            "cohort":     "",
            "event":      "",
        }
        stored = cfg.get("column_map", {})
        result = dict(defaults)
        for k, v in stored.items():
            if v and v != "— not mapped —":
                result[k] = v
        return result


def get_condition_labels() -> tuple[str, str]:
    """Return (condition_A_label, condition_B_label) from config.

    If condition_a_label / condition_b_label are empty, auto-detects from the
    first two sorted unique context values in results/comparison/summary_table.csv.
    Falls back to ("Condition A", "Condition B") when no data is available.
    """
    cfg = _load_config()
    label_a = cfg.get("condition_a_label", "").strip()
    label_b = cfg.get("condition_b_label", "").strip()
    if label_a and label_b:
        return label_a, label_b

    # Auto-detect from summary_table.csv
    results_dir = cfg.get("results_dir", "").strip()
    summary_path = os.path.join(results_dir, "comparison", "summary_table.csv")
    if os.path.exists(summary_path):
        try:
            import pandas as _pd
            df = _pd.read_csv(summary_path, usecols=["context"], nrows=5000)
            unique_vals = sorted(df["context"].dropna().astype(str).unique().tolist())
            if len(unique_vals) >= 2:
                return label_a or unique_vals[0], label_b or unique_vals[1]
        except Exception:
            pass

    return label_a or "Condition A", label_b or "Condition B"


def get_condition_a_label() -> str:
    return get_condition_labels()[0]


def get_condition_b_label() -> str:
    return get_condition_labels()[1]


def get_primary_metric_label() -> str:
    """Return the primary scalar metric label.
    Reads cfg["primary_metric_label"].
    Falls back to a generic label if not set."""
    cfg = _load_config()
    val = cfg.get("primary_metric_label", "").strip()
    return val or "Primary Metric"


def get_optional_report_columns() -> list[str]:
    """Return optional metadata columns to plot when present.

    These are experiment-specific report targets. Missing columns are expected
    for many projects and should never make the core report fail.
    """
    cfg = _load_config()
    val = cfg.get("optional_report_columns", cfg.get("analysis_columns", []))
    if isinstance(val, str):
        items = [x.strip() for x in val.split(",")]
    elif isinstance(val, (list, tuple)):
        items = [str(x).strip() for x in val]
    else:
        items = []
    return [x for x in items if x]


def get_metadata_schema() -> dict:
    """Return the active project metadata schema."""
    try:
        import metadata_schema as _ms
        return _ms.get_metadata_schema(_load_config())
    except Exception:
        return {}


def get_enabled_analysis_groups(df=None) -> list[dict]:
    """Return enabled metadata analysis groups for the active project."""
    try:
        import metadata_schema as _ms
        return _ms.get_enabled_analysis_groups(_load_config(), df)
    except Exception:
        return []


def resolve_session_id_column(df, warn: bool = False) -> str | None:
    try:
        import metadata_schema as _ms
        return _ms.resolve_session_id_column(df, _load_config(), warn=warn)
    except Exception:
        return None


def normalize_metadata_columns(df) -> "pd.DataFrame":
    """Rename user CSV columns to VIEB standard names using the project column_map.

    Only renames columns where the mapped name differs from the concept name
    and the mapped column exists in the DataFrame.  Leaves unmapped columns
    (empty string or '— not mapped —') untouched.
    """
    try:
        import metadata_schema as _ms
        return _ms.normalize_metadata_columns(df, _load_config(), warn=True)
    except Exception:
        cm = get_column_map()
        rename = {v: k for k, v in cm.items() if v and v != k and v in df.columns}
        if rename:
            return df.rename(columns=rename)
        return df


def get_raw_videos_dir() -> str:
    """Return the active project's raw-videos directory."""
    if "raw_videos_dir" not in _path_cache:
        _path_cache["raw_videos_dir"] = str(_pm.resolve_project_path("raw_videos", PROJECT_ROOT, _APP_CONFIG_PATH))
    return _path_cache["raw_videos_dir"]


def get_results_dir() -> str:
    """Return the active project's results directory."""
    if "results_dir" not in _path_cache:
        _path_cache["results_dir"] = str(_pm.resolve_project_path("results", PROJECT_ROOT, _APP_CONFIG_PATH))
    return _path_cache["results_dir"]


def get_metadata_path() -> str:
    """Return the active project's metadata CSV path."""
    if "metadata_path" not in _path_cache:
        _path_cache["metadata_path"] = str(_pm.resolve_project_path("metadata", PROJECT_ROOT, _APP_CONFIG_PATH))
    return _path_cache["metadata_path"]


def get_pose_source() -> str:
    """Return 'csv' or 'h5', reading 'pose_source' from config.json.
    Falls back to 'csv' when absent or invalid."""
    cfg = _load_config()
    val = cfg.get("pose_source", "csv")
    return val if val in ("csv", "h5") else "csv"


def get_use_wavelets() -> bool:
    """Return the 'use_wavelets' feature-extraction setting from config.json.
    Falls back to True when absent."""
    cfg = _load_config()
    return bool(cfg.get("use_wavelets", True))


def get_h5_path() -> str:
    """Return the configured H5 pose file path, or '' if unset."""
    try:
        return str(_pm.resolve_project_path("pose_h5", PROJECT_ROOT, _APP_CONFIG_PATH))
    except Exception:
        cfg = _load_config()
        val = cfg.get("h5_path", "")
        return val.strip() if isinstance(val, str) else ""


def get_h5_key() -> str:
    """Return the configured default H5 key, or '' if unset."""
    cfg = _load_config()
    val = cfg.get("h5_key", "")
    return val.strip() if isinstance(val, str) else ""


def get_h5_manifest_path() -> str:
    """Return the configured H5 manifest CSV path, or '' if unset."""
    cfg = _load_config()
    val = cfg.get("h5_manifest_path", "")
    return val.strip() if isinstance(val, str) else ""


def get_h5_source_col() -> str:
    """Return the configured H5 source column/dataset name, or '' if unset."""
    cfg = _load_config()
    val = cfg.get("h5_source_col", "")
    return val.strip() if isinstance(val, str) else ""


def get_clips_dir() -> str:
    """Return the clips directory.
    Derived as the sibling of get_results_dir() named 'clips', which reproduces
    the default PROJECT_ROOT/clips when results_dir is PROJECT_ROOT/results."""
    if "clips_dir" not in _path_cache:
        from pathlib import Path as _Path
        _path_cache["clips_dir"] = str(_Path(get_results_dir()).parent / "clips")
    return _path_cache["clips_dir"]


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
