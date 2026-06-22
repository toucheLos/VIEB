"""Project metadata schema helpers for VIEB.

The schema layer lets labs keep their own metadata column names while VIEB
uses stable canonical names internally.
"""

from __future__ import annotations

import os
import re
from copy import deepcopy
from typing import Any

import pandas as pd

CANONICAL_FIELDS = ["session_id", "animal_id", "context", "day", "experiment"]
SESSION_ALIASES = [
    "session_id", "filename", "file", "source_file", "video", "video_file",
    "csv", "h5_key", "recording", "recording_file",
]
OPTIONAL_DEFAULTS = {
    "fear": "fear",
    "no_shock": "no_shock",
    "sex": "sex",
    "genotype": "genotype",
    "treatment": "treatment",
    "cohort": "cohort",
    "timepoint": "timepoint",
}
EXT_RE = re.compile(r"\.(mp4|avi|mov|mkv|csv|h5|hdf5)$", re.IGNORECASE)


DEFAULT_SCHEMA: dict[str, Any] = {
    "id_column": "filename",
    "column_map": {
        "session_id": "filename",
        "animal_id": "animal_id",
        "context": "context",
        "day": "day",
        "experiment": "experiment",
    },
    "optional_columns": {"fear": "fear"},
    "analysis_groups": [
        {
            "name": "Context",
            "column": "context",
            "enabled": True,
            "plots": ["state_fraction", "transition_matrix", "motif_enrichment"],
        },
        {
            "name": "Animal",
            "column": "animal_id",
            "enabled": True,
            "plots": ["state_fraction", "trajectory"],
        },
        {
            "name": "Day",
            "column": "day",
            "enabled": True,
            "plots": ["state_fraction"],
        },
        {
            "name": "Experiment",
            "column": "experiment",
            "enabled": True,
            "plots": ["state_fraction"],
            "optional": True,
        },
        {
            "name": "Fear",
            "column": "fear",
            "enabled": True,
            "plots": ["state_fraction"],
            "optional": True,
        },
    ],
    "correlations": [],
}


def _clean_mapping(mapping: dict | None) -> dict[str, str]:
    if not isinstance(mapping, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in mapping.items():
        if value and value != "— not mapped —":
            result[str(key)] = str(value)
    return result


def get_metadata_schema(config: dict | None = None) -> dict:
    """Return a normalized schema dict, accepting old and new config shapes."""
    cfg = config or {}
    schema = deepcopy(DEFAULT_SCHEMA)
    stored = cfg.get("metadata_schema")
    if isinstance(stored, dict):
        for key, value in stored.items():
            if key in ("column_map", "optional_columns"):
                merged = dict(schema.get(key, {}))
                merged.update(_clean_mapping(value))
                schema[key] = merged
            elif key == "analysis_groups" and isinstance(value, list):
                schema[key] = value
            elif key == "correlations" and isinstance(value, list):
                schema[key] = value
            else:
                schema[key] = value

    # Backward compatibility with the flat config used by older VIEB projects.
    flat_map = _clean_mapping(cfg.get("column_map"))
    if flat_map:
        schema["column_map"].update(flat_map)
    if "filename" not in schema["column_map"].values():
        id_col = cfg.get("id_column") or schema.get("id_column") or "filename"
        schema["column_map"].setdefault("session_id", str(id_col))

    optional = dict(schema.get("optional_columns", {}))
    for col in cfg.get("optional_report_columns", cfg.get("analysis_columns", [])) or []:
        optional.setdefault(str(col), str(col))
    schema["optional_columns"] = optional
    return schema


def resolve_session_id_column(
    df: pd.DataFrame,
    config: dict | None = None,
    *,
    warn: bool = False,
) -> str | None:
    """Find the column that identifies a recording/session."""
    schema = get_metadata_schema(config)
    candidates = [
        schema.get("column_map", {}).get("session_id", ""),
        schema.get("id_column", ""),
        "session_id",
        "filename",
        "source_file",
    ]
    candidates.extend([c for c in df.columns if str(c).lower().endswith("_file")])
    candidates.extend(SESSION_ALIASES)

    lower_to_original = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if not candidate:
            continue
        if candidate in df.columns:
            if warn and candidate not in (
                schema.get("column_map", {}).get("session_id"),
                schema.get("id_column"),
            ):
                print(f"[info] Auto-detected session identifier column: {candidate}")
            return str(candidate)
        original = lower_to_original.get(str(candidate).lower())
        if original is not None:
            if warn:
                print(f"[info] Auto-detected session identifier column: {original}")
            return str(original)
    return None


def derive_stem(value: object) -> str:
    """Derive a VIEB stem from a filename, source_file, H5 key, or session id."""
    text = "" if pd.isna(value) else str(value).strip()
    if not text:
        return ""
    text = text.replace("\\", "/").rstrip("/")
    base = os.path.basename(text)
    return EXT_RE.sub("", base)


def normalize_metadata_columns(
    df: pd.DataFrame,
    config: dict | None = None,
    *,
    warn: bool = False,
) -> pd.DataFrame:
    """Rename/alias project metadata to canonical VIEB columns."""
    schema = get_metadata_schema(config)
    out = df.copy()

    for logical, source in _clean_mapping(schema.get("column_map")).items():
        if source in out.columns and logical not in out.columns:
            out[logical] = out[source]
    for logical, source in _clean_mapping(schema.get("optional_columns")).items():
        if source in out.columns and logical not in out.columns:
            out[logical] = out[source]

    session_col = "session_id" if "session_id" in out.columns else resolve_session_id_column(out, config, warn=warn)
    if session_col and session_col in out.columns:
        out["session_id"] = out[session_col].astype(str)
        if "filename" not in out.columns:
            out["filename"] = out["session_id"]
        out["stem"] = out["session_id"].map(derive_stem)
    return out


def has_optional_column(
    df: pd.DataFrame,
    logical_name: str,
    config: dict | None = None,
) -> bool:
    schema = get_metadata_schema(config)
    col = logical_name
    mapped = schema.get("optional_columns", {}).get(logical_name)
    return col in df.columns or bool(mapped and mapped in df.columns)


def get_enabled_analysis_groups(config: dict | None, df: pd.DataFrame | None = None) -> list[dict]:
    """Return configured analysis groups that are enabled and, if df is given, valid."""
    schema = get_metadata_schema(config)
    groups = []
    seen: set[str] = set()
    for group in schema.get("analysis_groups", []):
        if not isinstance(group, dict) or not group.get("enabled", True):
            continue
        col = str(group.get("column", "")).strip()
        if not col or col in seen:
            continue
        item = dict(group)
        item["column"] = col
        item.setdefault("name", col.replace("_", " ").title())
        item.setdefault("plots", ["state_fraction"])
        if df is not None:
            if col not in df.columns:
                item["available"] = False
                item["skip_reason"] = f"column '{col}' not found"
            elif not df[col].notna().any():
                item["available"] = False
                item["skip_reason"] = f"column '{col}' has no values"
            else:
                item["available"] = True
        groups.append(item)
        seen.add(col)
    return groups


def validate_metadata_schema(df: pd.DataFrame, config: dict | None = None) -> dict:
    """Validate metadata against the project schema."""
    schema = get_metadata_schema(config)
    normalized = normalize_metadata_columns(df, config)
    session_col = resolve_session_id_column(df, config)
    missing_required = []
    if not session_col or "session_id" not in normalized.columns:
        missing_required.append("session_id")
    elif normalized["session_id"].isna().all() or (normalized["session_id"].astype(str).str.strip() == "").all():
        missing_required.append("session_id")

    optional_missing = []
    for logical in schema.get("optional_columns", {}):
        if logical not in normalized.columns:
            optional_missing.append(logical)

    groups = get_enabled_analysis_groups(config, normalized)
    skipped = []
    for group in groups:
        if not group.get("available", True):
            skipped.append({"name": group.get("name"), "column": group.get("column"), "reason": group.get("skip_reason")})

    messages = []
    if missing_required:
        messages.append("metadata is missing required field(s): " + ", ".join(missing_required))
    for item in skipped:
        messages.append(f"analysis group '{item['name']}' skipped: {item['reason']}")

    return {
        "valid": not missing_required,
        "detected_columns": list(df.columns),
        "mapped_canonical_fields": {
            field: field for field in CANONICAL_FIELDS if field in normalized.columns
        },
        "session_id_column": session_col,
        "missing_required_fields": missing_required,
        "missing_optional_fields": optional_missing,
        "enabled_analysis_groups": groups,
        "skipped_analyses": skipped,
        "messages": messages,
    }


def metadata_schema_report(df: pd.DataFrame, config: dict | None = None) -> dict:
    return validate_metadata_schema(df, config)
