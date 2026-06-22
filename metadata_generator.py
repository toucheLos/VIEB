"""
metadata_generator.py — Build a metadata.csv template from raw videos/CSVs
or from a shared H5 pose file.

Standard VIEB metadata columns remain Luna-compatible by default:
filename, date, box, experiment, day, context, no_shock, animal_id, fear.
Projects can also normalize an existing manifest CSV through metadata_schema.

Also provides validate_metadata() / validate_metadata_csv(), used by
compare.py and the GUI to warn the user about incomplete rows before
feature extraction runs.
"""

from __future__ import annotations

import glob
import os
import re

import pandas as pd

import metadata_schema as _ms

META_COLUMNS = [
    "filename", "date", "box", "experiment", "day", "context",
    "no_shock", "animal_id", "fear",
]

# Only a session identifier is universally required. animal_id/context/day are
# optional but unlock additional reports when present.
REQUIRED_COLUMNS = ["session_id"]

# e.g. "20241113_Box_1_CFC_Day_0_(Context_A)_9001.mp4"
_FULL_RE = re.compile(
    r"(?P<date>\d{8})_Box_(?P<box>\d+)_(?P<experiment>\w+?)_Day_(?P<day>\d+)"
    r"_\(Context_(?P<context>\w+)(?:,[^)]*)?\)_(?P<animal_id>\d+)",
    re.IGNORECASE,
)

# Looser fallback patterns, applied independently so partial matches still help.
# Each is tried in order; the first match for a given field wins.

# 8-digit date, e.g. 20241113 (YYYYMMDD)
_DATE_RE = re.compile(r"(?<!\d)(?P<date>\d{8})(?!\d)")
# 6-digit date, e.g. 241113 (YYMMDD) or 113024 (MMDDYY)
_DATE_RE_SHORT = re.compile(r"(?<!\d)(?P<date>\d{6})(?!\d)")
# Hyphenated date, e.g. 2024-11-13 or 11-13-2024
_DATE_RE_HYPHEN = re.compile(r"(?P<date>\d{4}-\d{2}-\d{2}|\d{2}-\d{2}-\d{4})")

_BOX_RE = re.compile(r"Box[_-]?(?P<box>\d+)", re.IGNORECASE)
_BOX_RE_SHORT = re.compile(r"(?:^|[_-])B(?P<box>\d{1,2})(?=[_-]|\.|$)", re.IGNORECASE)

_DAY_RE = re.compile(r"Day[_-]?(?P<day>\d+)", re.IGNORECASE)
_DAY_RE_SHORT = re.compile(r"(?:^|[_-])D(?P<day>\d{1,2})(?=[_-]|\.|$)", re.IGNORECASE)

_CONTEXT_RE = re.compile(r"Context[_-]?\(?(?P<context>[A-Za-z0-9]+)\)?", re.IGNORECASE)
_CONTEXT_RE_SHORT = re.compile(r"(?:^|[_-])Ctx[_-]?(?P<context>[A-Za-z0-9]+?)(?=[_-]|\.|$)", re.IGNORECASE)

# Trailing run of 3+ digits, e.g. "..._9001.mp4"
_ANIMAL_RE = re.compile(r"_(?P<animal_id>\d{3,})(?:\.\w+)?$")
# "Mouse_9001", "Animal-9001", "M9001"
_ANIMAL_RE_LABELED = re.compile(
    r"(?:Mouse|Animal|Subject|M)[_-]?(?P<animal_id>\d{2,})", re.IGNORECASE
)

# Experiment label preceding "Day": "..._CFC_Day_0..."
_EXPERIMENT_RE = re.compile(r"_(?P<experiment>[A-Za-z]+)_Day[_-]?\d+", re.IGNORECASE)
# Experiment label right after a date/box prefix: "20241113_Box_1_CFC_..."
_EXPERIMENT_RE_AFTER_BOX = re.compile(
    r"Box[_-]?\d+_(?P<experiment>[A-Za-z]+)", re.IGNORECASE
)

# Additional fallback patterns for animal_id / day, tried when the patterns
# above find nothing.
_ANIMAL_FALLBACK_RES = [
    re.compile(r"rat[_-]?(?P<animal_id>\d+)", re.IGNORECASE),
    re.compile(r"mouse[_-]?(?P<animal_id>\d+)", re.IGNORECASE),
    re.compile(r"animal[_-]?(?P<animal_id>\d+)", re.IGNORECASE),
    re.compile(r"(?P<animal_id>\d{4})"),
]
_DAY_FALLBACK_RES = [
    re.compile(r"session[_-]?(?P<day>\d+)", re.IGNORECASE),
    re.compile(r"[ds][_-]?(?P<day>\d+)", re.IGNORECASE),
]


def infer_fields_from_name(name: str) -> dict:
    """Infer metadata fields from a filename or H5-key stem.

    Returns a dict with whatever subset of date/box/experiment/day/context/
    animal_id could be inferred. Missing fields are absent from the dict
    (callers should fill blanks rather than guess).
    """
    result: dict = {}

    m = _FULL_RE.search(name)
    if m:
        result.update(m.groupdict())
        return result

    for rx, field in (
        (_DATE_RE, "date"),
        (_DATE_RE_HYPHEN, "date"),
        (_DATE_RE_SHORT, "date"),
        (_BOX_RE, "box"),
        (_BOX_RE_SHORT, "box"),
        (_DAY_RE, "day"),
        (_DAY_RE_SHORT, "day"),
        (_CONTEXT_RE, "context"),
        (_CONTEXT_RE_SHORT, "context"),
        (_EXPERIMENT_RE, "experiment"),
        (_EXPERIMENT_RE_AFTER_BOX, "experiment"),
        (_ANIMAL_RE_LABELED, "animal_id"),
        (_ANIMAL_RE, "animal_id"),
    ):
        if field in result:
            continue
        m = rx.search(name)
        if m:
            result[field] = m.group(field)

    if "animal_id" not in result:
        for rx in _ANIMAL_FALLBACK_RES:
            m = rx.search(name)
            if m:
                result["animal_id"] = m.group("animal_id")
                break

    if "day" not in result:
        for rx in _DAY_FALLBACK_RES:
            m = rx.search(name)
            if m:
                result["day"] = m.group("day")
                break

    return result


def _row_from_name(name: str, filename: str = "") -> dict:
    row = {col: "" for col in META_COLUMNS}
    row["filename"] = filename
    fields = infer_fields_from_name(name)
    for k, v in fields.items():
        if k in row:
            row[k] = v
    # animal_id is left blank when it can't be inferred — the researcher
    # fills it in. Do not guess from the filename stem.
    return row


def scan_raw_videos(raw_videos_dir: str) -> list[dict]:
    """Scan raw_videos_dir for .mp4 files and infer metadata fields
    from each filename. Returns a list of row dicts (one per video)."""
    if not raw_videos_dir or not os.path.isdir(raw_videos_dir):
        return []

    videos = sorted(glob.glob(os.path.join(raw_videos_dir, "*.mp4")))
    rows = []
    for video_path in videos:
        filename = os.path.basename(video_path)
        rows.append(_row_from_name(filename, filename=filename))
    return rows


def scan_h5_keys(h5_path: str) -> list[dict]:
    """Inspect an H5 pose file and infer metadata fields from each key name.
    Returns a list of row dicts (one per key); `filename` is left blank since
    H5-sourced rows are matched to keys at extract time."""
    if not h5_path or not os.path.exists(h5_path):
        return []

    from pose_io import inspect_h5

    info = inspect_h5(h5_path)
    rows = []
    for key in info["keys"]:
        rows.append(_row_from_name(key, filename=""))
    return rows


def generate_metadata_template(
    raw_videos_dir: str | None = None,
    h5_path: str | None = None,
    filename_regex: str | None = None,
) -> pd.DataFrame:
    """Build a metadata.csv template DataFrame from raw_videos_dir and/or an
    H5 pose file. If both are given, raw_videos_dir rows take priority and
    h5 rows are appended for keys without an obviously matching filename."""
    rows: list[dict] = []
    if raw_videos_dir:
        if filename_regex:
            rows.extend(scan_raw_videos_with_regex(raw_videos_dir, filename_regex))
        else:
            rows.extend(scan_raw_videos(raw_videos_dir))
    if h5_path:
        existing_ids = {r["animal_id"] for r in rows if r["animal_id"]}
        for row in scan_h5_keys(h5_path):
            if not row["animal_id"] or row["animal_id"] not in existing_ids:
                rows.append(row)

    if not rows:
        return pd.DataFrame(columns=META_COLUMNS)

    return pd.DataFrame(rows, columns=META_COLUMNS)


def scan_raw_videos_with_regex(raw_videos_dir: str, filename_regex: str) -> list[dict]:
    """Scan videos and infer fields with a user-provided named-group regex."""
    if not raw_videos_dir or not os.path.isdir(raw_videos_dir):
        return []
    rx = re.compile(filename_regex)
    rows = []
    for video_path in sorted(glob.glob(os.path.join(raw_videos_dir, "*.mp4"))):
        filename = os.path.basename(video_path)
        row = {col: "" for col in META_COLUMNS}
        row["filename"] = filename
        m = rx.search(filename)
        if m:
            for key, value in m.groupdict().items():
                if key in row:
                    row[key] = value
                elif key == "session_id":
                    row["filename"] = value
        rows.append(row)
    return rows


def generate_metadata_from_manifest(
    manifest_path: str,
    config: dict | None = None,
    out_path: str | None = None,
) -> pd.DataFrame:
    """Normalize a user-supplied manifest CSV to canonical VIEB metadata."""
    df = pd.read_csv(manifest_path, dtype=str).fillna("")
    normalized = _ms.normalize_metadata_columns(df, config or {}, warn=True)
    if out_path:
        normalized.to_csv(out_path, index=False)
    return normalized


def write_metadata_csv(df: pd.DataFrame, out_path: str) -> None:
    """Write a metadata template DataFrame to CSV.

    Plain UTF-8 CSV with a header row — opens cleanly in Excel and Google
    Sheets so the researcher can fill in the blanks.
    """
    df.to_csv(out_path, index=False)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_metadata(df: pd.DataFrame) -> dict:
    """Check a metadata DataFrame for blanks in required columns.

    Returns a dict:
      {
        "valid": bool,
        "missing_columns": [...],            # required columns absent entirely
        "missing_animal_id": [{"row": csv_row, "filename": ...}, ...],
        "missing_context": [{"row": csv_row, "filename": ...}, ...],
        "messages": [human-readable summary strings],
      }

    "row" is the 1-based row number as it would appear in Excel/Google
    Sheets (header = row 1, first data row = row 2).
    """
    schema_report = _ms.validate_metadata_schema(df)
    result: dict = {
        "valid": schema_report["valid"],
        "missing_columns": schema_report.get("missing_required_fields", []),
        "missing_session_id": [],
        "missing_animal_id": [],
        "missing_context": [],
        "messages": list(schema_report.get("messages", [])),
        "schema_report": schema_report,
    }

    normalized = _ms.normalize_metadata_columns(df)
    if "session_id" in normalized.columns:
        blank_mask = normalized["session_id"].isna() | (normalized["session_id"].astype(str).str.strip() == "")
        for idx in normalized.index[blank_mask]:
            csv_row = int(idx) + 2
            filename = str(normalized.at[idx, "filename"]) if "filename" in normalized.columns else ""
            result["missing_session_id"].append({"row": csv_row, "filename": filename})

    if result["missing_session_id"]:
        result["valid"] = False
        shown = result["missing_session_id"][:20]
        descr = ", ".join(
            f"row {r['row']} ({r['filename']})" if r["filename"] else f"row {r['row']}"
            for r in shown
        )
        extra = f", and {len(result['missing_session_id']) - 20} more" if len(result["missing_session_id"]) > 20 else ""
        result["messages"].append(f"{len(result['missing_session_id'])} row(s) missing 'session_id': {descr}{extra}")

    return result


def validate_metadata_csv(path: str) -> dict:
    """Read a metadata CSV and run validate_metadata() on it.

    Returns the same dict shape as validate_metadata(), with an additional
    "messages" entry if the file is missing or unreadable.
    """
    if not path or not os.path.exists(path):
        return {
            "valid": False,
            "missing_columns": [],
            "missing_animal_id": [],
            "missing_context": [],
            "messages": [f"Metadata CSV not found: {path}"],
        }

    try:
        df = pd.read_csv(path, dtype=str).fillna("")
    except Exception as e:
        return {
            "valid": False,
            "missing_columns": [],
            "missing_animal_id": [],
            "missing_context": [],
            "messages": [f"Could not read metadata CSV: {e}"],
        }

    try:
        import vieb_config as _vc
        df = _vc.normalize_metadata_columns(df)
    except Exception:
        pass

    return validate_metadata(df)
