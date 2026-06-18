"""
h5_manifest.py — Resolve metadata.csv rows to keys inside a shared H5 pose file.

When `pose_source == "h5"`, each metadata.csv row (one per animal/session)
needs to be matched to a key/group inside the shared H5 file. Three
strategies are tried in order:

  1. exact/substring match of filename stem or animal_id against H5 keys
  2. explicit manifest CSV mapping (animal_id/filename -> h5_key)
  3. ordinal fallback — nth metadata row maps to the nth H5 key, with a
     warning since this is fragile if ordering differs.
"""

from __future__ import annotations

import os
import re

import pandas as pd


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def detect_concatenated_table(
    h5_info: dict,
    h5_source_col: str = "source_file",
) -> str | None:
    """
    Return the single top-level H5 key when the file is a concatenated table.

    A concatenated table is defined as an H5 file with exactly one top-level
    key whose stored DataFrame contains the source/session identifier column.
    """
    keys = h5_info.get("keys", [])
    if len(keys) != 1:
        return None

    only_key = keys[0]
    details = h5_info.get("details", {}).get(only_key, {})
    columns = details.get("columns", []) or []
    if h5_source_col in columns:
        return only_key
    return None


def load_manifest(
    manifest_path: str,
    value_col: str = "h5_key",
) -> dict[str, str]:
    """
    Load a manifest CSV mapping animal_id/filename -> a target H5 selector.

    Expects columns including `value_col` and at least one of
    `animal_id` or `filename`. Returns a dict keyed by the normalized
    animal_id, filename stem, and exact selector value, mapping to the raw
    selector string.
    """
    if not manifest_path or not os.path.exists(manifest_path):
        return {}

    df = pd.read_csv(manifest_path, dtype=str).fillna("")
    if value_col not in df.columns:
        raise ValueError(
            f"Manifest {manifest_path} must contain a '{value_col}' column"
        )

    mapping: dict[str, str] = {}
    for _, row in df.iterrows():
        value = row[value_col]
        if not value:
            continue
        if "animal_id" in df.columns and row.get("animal_id"):
            mapping[_normalize(row["animal_id"])] = value
        if "filename" in df.columns and row.get("filename"):
            stem = os.path.splitext(row["filename"])[0]
            mapping[_normalize(stem)] = value
        mapping[_normalize(value)] = value
    return mapping


def resolve_h5_key(
    row: dict,
    h5_keys: list[str],
    manifest: dict[str, str] | None,
    ordinal_index: int,
    *,
    concatenated_key: str | None = None,
    h5_source_col: str = "source_file",
) -> tuple[str, str]:
    """
    Resolve the H5 key for one metadata row.

    Parameters
    ----------
    row           : metadata.csv row as a dict (normalized column names:
                    "filename", "animal_id", etc.)
    h5_keys       : list of available top-level keys in the H5 file
    manifest      : optional dict from load_manifest(); may be None
    ordinal_index : 0-based row index, used as the strategy-3 fallback

    Returns
    -------
    (h5_key, strategy) where strategy is one of "exact", "manifest", "ordinal".
    For concatenated-table H5 files, returns the per-session source value
    instead of a top-level H5 key.

    Raises
    ------
    ValueError if no strategy can resolve a key (e.g. ordinal_index out of range).
    """
    if concatenated_key is not None:
        source_value = str(row.get(h5_source_col, "") or "").strip()
        if source_value:
            return source_value, "source_file"

        if manifest:
            candidates = []
            filename = row.get("filename")
            if filename:
                candidates.append(_normalize(os.path.splitext(str(filename))[0]))
            animal_id = row.get("animal_id")
            if animal_id:
                candidates.append(_normalize(animal_id))
            for cand in candidates:
                if cand and cand in manifest:
                    return manifest[cand], "manifest"

        raise ValueError(
            f"Could not resolve a concatenated H5 row for metadata row {ordinal_index} "
            f"using column {h5_source_col!r}"
        )

    norm_keys = {_normalize(k): k for k in h5_keys}

    # Strategy 1: exact/substring match on filename stem or animal_id
    candidates = []
    filename = row.get("filename")
    if filename:
        candidates.append(_normalize(os.path.splitext(str(filename))[0]))
    animal_id = row.get("animal_id")
    if animal_id:
        candidates.append(_normalize(animal_id))

    for cand in candidates:
        if not cand:
            continue
        if cand in norm_keys:
            return norm_keys[cand], "exact"
        for norm_key, raw_key in norm_keys.items():
            if cand in norm_key or norm_key in cand:
                return raw_key, "exact"

    # Strategy 2: manifest lookup
    if manifest:
        for cand in candidates:
            if cand and cand in manifest:
                manifest_key = manifest[cand]
                if manifest_key in h5_keys:
                    return manifest_key, "manifest"
                if _normalize(manifest_key) in norm_keys:
                    return norm_keys[_normalize(manifest_key)], "manifest"

    # Strategy 3: ordinal fallback
    if 0 <= ordinal_index < len(h5_keys):
        print(
            f"  [h5_manifest] WARNING: no name/manifest match for row "
            f"{ordinal_index} (filename={filename!r}, animal_id={animal_id!r}); "
            f"falling back to ordinal H5 key '{h5_keys[ordinal_index]}'"
        )
        return h5_keys[ordinal_index], "ordinal"

    raise ValueError(
        f"Could not resolve an H5 key for row {ordinal_index} "
        f"(filename={filename!r}, animal_id={animal_id!r}); "
        f"{len(h5_keys)} keys available"
    )
