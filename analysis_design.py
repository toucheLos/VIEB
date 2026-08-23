"""Detect analysis design from project metadata.

This module keeps report generation independent of any one lab's metadata
vocabulary. It prefers explicit metadata_schema/column_map configuration and
falls back to conservative heuristics.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import metadata_schema as _ms
except Exception:  # pragma: no cover - import failure fallback for standalone use
    _ms = None


MODE_VALUES = {
    "time_only",
    "condition_only",
    "time_and_condition",
    "group_only",
    "time_and_group",
    "condition_and_group",
    "minimal",
}

SUBJECT_ALIASES = (
    "animal_id", "subject_id", "subject", "animal", "mouse", "rat",
    "mouse_id", "rat_id", "id",
)
TIME_ALIASES = ("day", "week", "timepoint", "time_point", "session_day", "phase")
CONDITION_ALIASES = ("condition", "context", "trial_type", "environment", "stimulus")
GROUP_ALIASES = ("genotype", "sex", "treatment", "cohort", "group", "strain", "line")
SESSION_ALIASES = ("stem", "session_id", "filename", "source_file", "file", "video")

TEMPORAL_WORDS = {
    "baseline": -100,
    "base": -100,
    "pre": -90,
    "before": -90,
    "day0": 0,
    "d0": 0,
    "post": 10_000,
    "after": 10_000,
    "endpoint": 20_000,
}
TEMPORAL_RE = re.compile(r"^(?:day|d|week|wk|w|tp|timepoint)[_\-\s]*(\d+)$", re.I)


def _clean_mapping(mapping: dict | None) -> dict[str, str]:
    if not isinstance(mapping, dict):
        return {}
    out = {}
    for key, value in mapping.items():
        if value and value != "-- not mapped --" and value != "— not mapped —":
            out[str(key)] = str(value)
    return out


def _schema(config: dict | None) -> dict:
    if _ms is not None:
        try:
            return _ms.get_metadata_schema(config or {})
        except Exception:
            pass
    cfg = config or {}
    schema = dict(cfg.get("metadata_schema") or {})
    if cfg.get("column_map"):
        merged = dict(schema.get("column_map") or {})
        merged.update(_clean_mapping(cfg.get("column_map")))
        schema["column_map"] = merged
    return schema


def _canonicalize(df: pd.DataFrame, config: dict | None) -> pd.DataFrame:
    if _ms is not None:
        try:
            return _ms.normalize_metadata_columns(df, config or {})
        except Exception:
            pass
    return df.copy()


def _find_column(df: pd.DataFrame, names: list[str] | tuple[str, ...]) -> str | None:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for name in names:
        if name in df.columns:
            return str(name)
        original = lower.get(str(name).lower())
        if original is not None:
            return original
    return None


def _mapped_column(df: pd.DataFrame, config: dict | None, logical_names: list[str]) -> str | None:
    schema = _schema(config)
    mapping = _clean_mapping(schema.get("column_map"))
    lower = {str(c).lower(): str(c) for c in df.columns}
    for logical in logical_names:
        if logical in df.columns:
            return logical
        original = lower.get(logical.lower())
        if original is not None:
            return original
        source = mapping.get(logical)
        if source in df.columns:
            return str(source)
        original = lower.get(str(source).lower()) if source else None
        if original is not None:
            return original
    return None


def _is_repeated_categorical(series: pd.Series, *, min_values: int = 2, max_values: int = 12) -> bool:
    clean = series.dropna()
    if clean.empty:
        return False
    values = clean.astype(str).str.strip()
    values = values[values != ""]
    if values.empty:
        return False
    n_unique = values.nunique()
    if n_unique < min_values or n_unique > max_values:
        return False
    return len(values) / max(1, n_unique) >= 2


def _time_key(value: object) -> tuple[int, float, str]:
    text = "" if pd.isna(value) else str(value).strip()
    low = text.lower().replace(" ", "")
    if low in TEMPORAL_WORDS:
        return (0, float(TEMPORAL_WORDS[low]), text)
    try:
        return (1, float(text), text)
    except ValueError:
        pass
    match = TEMPORAL_RE.match(low)
    if match:
        return (2, float(match.group(1)), text)
    digits = re.findall(r"\d+", low)
    if digits and any(word in low for word in ("day", "week", "wk", "time", "tp")):
        return (3, float(digits[0]), text)
    return (4, 0.0, text)


def _looks_temporal_name(name: str) -> bool:
    low = name.lower()
    return any(token in low for token in ("day", "week", "time", "phase", "session"))


def _looks_temporal_values(series: pd.Series) -> bool:
    clean = series.dropna().astype(str).str.strip()
    clean = clean[clean != ""]
    if clean.nunique() < 2:
        return False
    sample = clean.drop_duplicates().head(20).tolist()
    hits = 0
    for value in sample:
        low = value.lower().replace(" ", "")
        if low in TEMPORAL_WORDS or TEMPORAL_RE.match(low):
            hits += 1
        elif any(word in low for word in ("baseline", "pre", "post", "day", "week", "wk", "timepoint")):
            hits += 1
        else:
            try:
                float(low)
                hits += 1
            except ValueError:
                pass
    return hits / max(1, len(sample)) >= 0.6


def _time_order(series: pd.Series | None) -> list[Any] | None:
    if series is None:
        return None
    values = [v for v in series.dropna().unique().tolist() if str(v).strip() != ""]
    if not values:
        return None
    return sorted(values, key=_time_key)


def _numeric_columns(df: pd.DataFrame, excluded: set[str]) -> list[str]:
    cols = []
    for col in df.columns:
        name = str(col)
        if name in excluded or name.startswith("state_") or name.startswith("trans_"):
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() >= max(3, int(len(df) * 0.5)):
            cols.append(name)
    return cols


def detect_analysis_design(metadata_df: pd.DataFrame, config: dict | None = None) -> dict[str, Any]:
    """Return a JSON-serializable project analysis design description."""
    df = _canonicalize(metadata_df, config)
    mapping = _clean_mapping(_schema(config).get("column_map"))
    role_source_cols = {v for k, v in mapping.items() if k in {"session_id", "animal_id", "subject_id", "subject", "day", "week", "timepoint"}}

    subject_col = (
        _mapped_column(df, config, ["animal_id", "subject_id", "subject"])
        or _find_column(df, SUBJECT_ALIASES)
    )

    time_col = _mapped_column(df, config, ["day", "week", "timepoint", "time_col"])
    if time_col is None:
        for col in df.columns:
            name = str(col)
            if name == subject_col or name.lower() in SESSION_ALIASES:
                continue
            if _looks_temporal_name(name) and _looks_temporal_values(df[col]):
                time_col = name
                break
    if time_col is None:
        for col in df.columns:
            name = str(col)
            if name == subject_col or name.lower() in SESSION_ALIASES:
                continue
            if _looks_temporal_values(df[col]):
                time_col = name
                break

    mapped_condition = _mapped_column(df, config, ["condition", "context"])
    condition_cols: list[str] = []
    if mapped_condition and mapped_condition != time_col:
        condition_cols.append(mapped_condition)
    for alias in CONDITION_ALIASES:
        col = _find_column(df, (alias,))
        if col and col not in condition_cols and col != time_col:
            if _is_repeated_categorical(df[col], max_values=6):
                condition_cols.append(col)
    for col in df.columns:
        name = str(col)
        if name in condition_cols or name in {subject_col, time_col} or name in role_source_cols:
            continue
        if name.lower() in SESSION_ALIASES or name.lower() in GROUP_ALIASES:
            continue
        if _is_repeated_categorical(df[col], max_values=3) and not _looks_temporal_values(df[col]):
            condition_cols.append(name)

    group_cols: list[str] = []
    schema = _schema(config)
    for group in schema.get("analysis_groups", []) or []:
        if not isinstance(group, dict) or not group.get("enabled", True):
            continue
        col = str(group.get("column") or "").strip()
        if col and col in df.columns and col not in {subject_col, time_col} and col not in condition_cols and col not in role_source_cols:
            if _is_repeated_categorical(df[col], max_values=12):
                group_cols.append(col)
    for alias in GROUP_ALIASES:
        col = _find_column(df, (alias,))
        if col and col not in group_cols and col not in {subject_col, time_col} and col not in condition_cols and col not in role_source_cols:
            if _is_repeated_categorical(df[col], max_values=12):
                group_cols.append(col)

    excluded = {c for c in [subject_col, time_col] if c}
    excluded.update(condition_cols)
    excluded.update(group_cols)
    excluded.update(str(c) for c in df.columns if str(c).lower() in SESSION_ALIASES)
    continuous_cols = _numeric_columns(df, excluded)

    has_time = bool(time_col)
    has_condition = bool(condition_cols)
    has_group = bool(group_cols)
    if has_time and has_condition:
        mode = "time_and_condition"
    elif has_time and has_group:
        mode = "time_and_group"
    elif has_condition and has_group:
        mode = "condition_and_group"
    elif has_time:
        mode = "time_only"
    elif has_condition:
        mode = "condition_only"
    elif has_group:
        mode = "group_only"
    else:
        mode = "minimal"

    return {
        "subject_col": subject_col,
        "time_col": time_col,
        "time_order": _time_order(df[time_col]) if time_col else None,
        "condition_cols": condition_cols,
        "group_cols": group_cols,
        "continuous_cols": continuous_cols,
        "detected_mode": mode if mode in MODE_VALUES else "minimal",
    }


def write_analysis_design(
    metadata_df: pd.DataFrame,
    results_dir: str | Path,
    config: dict | None = None,
) -> dict[str, Any]:
    design = detect_analysis_design(metadata_df, config)
    out_path = Path(results_dir) / "analysis_design.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(design, indent=2, default=str), encoding="utf-8")
    print(f"Analysis design saved: results/analysis_design.json ({design['detected_mode']})")
    return design
