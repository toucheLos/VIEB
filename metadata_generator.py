"""
metadata_generator.py — Build a metadata.csv template from raw videos/CSVs
or from a shared H5 pose file.

Standard VIEB metadata columns: filename, date, box, experiment, day,
context, no_shock, animal_id, fear. Fields that can't be inferred are left
blank for the user to fill in (no_shock and fear are always left blank).

Also provides validate_metadata() / validate_metadata_csv(), used by
compare.py and the GUI to warn the user about incomplete rows before
feature extraction runs.
"""

from __future__ import annotations

import glob
import os
import re

import pandas as pd

META_COLUMNS = [
    "filename", "date", "box", "experiment", "day", "context",
    "no_shock", "animal_id", "fear",
]

# Columns that must be filled in for downstream analysis to make sense.
REQUIRED_COLUMNS = ["animal_id", "context"]

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
) -> pd.DataFrame:
    """Build a metadata.csv template DataFrame from raw_videos_dir and/or an
    H5 pose file. If both are given, raw_videos_dir rows take priority and
    h5 rows are appended for keys without an obviously matching filename."""
    rows: list[dict] = []
    if raw_videos_dir:
        rows.extend(scan_raw_videos(raw_videos_dir))
    if h5_path:
        existing_ids = {r["animal_id"] for r in rows if r["animal_id"]}
        for row in scan_h5_keys(h5_path):
            if not row["animal_id"] or row["animal_id"] not in existing_ids:
                rows.append(row)

    if not rows:
        return pd.DataFrame(columns=META_COLUMNS)

    return pd.DataFrame(rows, columns=META_COLUMNS)


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
    result: dict = {
        "valid": True,
        "missing_columns": [],
        "missing_animal_id": [],
        "missing_context": [],
        "messages": [],
    }

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            result["missing_columns"].append(col)

    if result["missing_columns"]:
        result["valid"] = False
        result["messages"].append(
            "metadata.csv is missing required column(s): "
            + ", ".join(result["missing_columns"])
        )

    key_map = {"animal_id": "missing_animal_id", "context": "missing_context"}
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            continue
        blank_mask = df[col].isna() | (df[col].astype(str).str.strip() == "")
        for idx in df.index[blank_mask]:
            csv_row = int(idx) + 2  # +1 for 0-based index, +1 for header row
            filename = ""
            if "filename" in df.columns:
                filename = str(df.at[idx, "filename"])
            result[key_map[col]].append({"row": csv_row, "filename": filename})

    for col, key in key_map.items():
        rows = result[key]
        if not rows:
            continue
        result["valid"] = False
        shown = rows[:20]
        descr = ", ".join(
            f"row {r['row']} ({r['filename']})" if r["filename"] else f"row {r['row']}"
            for r in shown
        )
        extra = f", and {len(rows) - 20} more" if len(rows) > 20 else ""
        result["messages"].append(
            f"{len(rows)} row(s) missing '{col}': {descr}{extra}"
        )

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
