"""
metadata_generator.py — Build a metadata.csv template from raw videos/CSVs
or from a shared H5 pose file.

Standard VIEB metadata columns: filename, date, box, experiment, day,
context, no_shock, animal_id, fear. Fields that can't be inferred are left
blank for the user to fill in (no_shock and fear are always left blank).
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

# e.g. "20241113_Box_1_CFC_Day_0_(Context_A)_9001.mp4"
_FULL_RE = re.compile(
    r"(?P<date>\d{8})_Box_(?P<box>\d+)_(?P<experiment>\w+?)_Day_(?P<day>\d+)"
    r"_\(Context_(?P<context>\w+)(?:,[^)]*)?\)_(?P<animal_id>\d+)",
    re.IGNORECASE,
)

# Looser fallback patterns, applied independently so partial matches still help
_DATE_RE = re.compile(r"(?P<date>\d{8})")
_BOX_RE = re.compile(r"Box[_-]?(?P<box>\d+)", re.IGNORECASE)
_DAY_RE = re.compile(r"Day[_-]?(?P<day>\d+)", re.IGNORECASE)
_CONTEXT_RE = re.compile(r"Context[_-]?\(?(?P<context>[A-Za-z0-9]+)\)?", re.IGNORECASE)
_ANIMAL_RE = re.compile(r"_(?P<animal_id>\d{3,})(?:\.\w+)?$")


def infer_fields_from_name(name: str) -> dict:
    """Infer metadata fields from a filename or H5-key stem.

    Returns a dict with whatever subset of date/box/experiment/day/context/
    animal_id could be inferred. Missing fields are absent from the dict
    (callers should fill blanks).
    """
    result: dict = {}

    m = _FULL_RE.search(name)
    if m:
        result.update(m.groupdict())
        return result

    for rx, field in (
        (_DATE_RE, "date"),
        (_BOX_RE, "box"),
        (_DAY_RE, "day"),
        (_CONTEXT_RE, "context"),
        (_ANIMAL_RE, "animal_id"),
    ):
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
    if not row["animal_id"]:
        row["animal_id"] = os.path.splitext(name)[0]
    return row


def scan_raw_videos(raw_videos_dir: str) -> list[dict]:
    """Scan raw_videos_dir for .mp4/.csv files and infer metadata fields
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
        existing_ids = {r["animal_id"] for r in rows}
        for row in scan_h5_keys(h5_path):
            if row["animal_id"] not in existing_ids:
                rows.append(row)

    if not rows:
        return pd.DataFrame(columns=META_COLUMNS)

    return pd.DataFrame(rows, columns=META_COLUMNS)


def write_metadata_csv(df: pd.DataFrame, out_path: str) -> None:
    """Write a metadata template DataFrame to CSV."""
    df.to_csv(out_path, index=False)
