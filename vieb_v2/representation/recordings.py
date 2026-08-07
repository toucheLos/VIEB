"""Recording identity: normalized ids, filename metadata, and deduplication.

Nothing in v2 has ever persisted *which* recording a row came from. `pack`
stores a `lengths` array, `embed_all` stores an `(recording_idx, frame_idx)`
index, and both are positional -- the mapping back to a file depends on
re-running `find_pose_files` with an identical glob and trusting `sorted()`.
`load_sessions` drops unreadable files into a `skipped` list that shifts every
later index and is never written down. So the map is not merely absent, it is
not reconstructible after the fact.

That gap is why the pose directory's duplication went unnoticed. Of the 4,925
files under `raw_videos`, 3,846 are `.h5` and 1,079 are `.csv` -- and every one
of those 1,079 csv files is the *same recording* as one of the h5 files, under
DLC's other export format. v2 has been treating them as 4,925 independent
recordings and 28,626,107 frames, against a true 3,846 and 22,355,989.

For a transfer operator that is not cosmetic. `pi` is the stationary measure of
the chain, estimated by counting how long the animal spends in each state, so
1,079 recordings counted twice contribute twice the mass. Deduplication has to
happen before any operator is built on top of it.

The id rule matches ExBias's (`exbias.py:295`) exactly, because a cross-method
comparison keyed on differently-normalized ids silently joins nothing.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

# Extensions stripped before the DLC suffix is cut. Order is irrelevant --
# at most one can match.
_EXTENSIONS = (".csv", ".h5", ".npy", ".mp4")

# Preference order when the same recording exists in several formats. h5 first:
# it is the format all 3,846 recordings share, it carries the likelihoods
# without a text round-trip, and its frame count is readable from the node
# shape without parsing the body.
DEFAULT_PREFERENCE = (".h5", ".csv", ".npy")

# Luna filenames encode the whole experimental design. The `,_No_Shock` group is
# optional and load-bearing: without it the pattern matches 3,548 of 3,846 and
# silently drops exactly the 298 retrieval-test recordings, which are the
# post-shock condition the analysis is about.
RECORDING_RE = re.compile(
    r"^(?P<date>\d{8})"
    r"_Box_(?P<box>\d+)"
    r"_(?P<experiment>[A-Za-z]+)"
    r"_Day_(?P<day>\d+)"
    r"_\(Context_(?P<context>[A-Za-z]+)(?P<no_shock>,_No_Shock)?\)"
    r"_(?P<animal>\w+)$"
)


def normalize_id(path):
    """File path -> canonical recording id.

    Strips a known extension, then truncates at the first `DLC_`. Matches
    `exbias.py:295` so VUS-1 runs from either project join on the same key.

        20241016_Box_1_CFC_Day_0_(Context_A)_308DLC_Resnet50_...h5
        -> 20241016_Box_1_CFC_Day_0_(Context_A)_308
    """
    base = os.path.basename(str(path))
    for ext in _EXTENSIONS:
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    marker = base.find("DLC_")
    return base[:marker] if marker > 0 else base


def parse_id(recording_id):
    """Recording id -> experimental design fields, or None if it does not match.

    `day` and `box` come back as ints and `no_shock` as a bool, so callers can
    filter on them without re-parsing strings.
    """
    match = RECORDING_RE.match(str(recording_id))
    if match is None:
        return None
    fields = match.groupdict()
    return {
        "date": fields["date"],
        "box": int(fields["box"]),
        "experiment": fields["experiment"],
        "day": int(fields["day"]),
        "context": fields["context"],
        "no_shock": fields["no_shock"] is not None,
        "animal": fields["animal"],
    }


def dedupe(paths, prefer=DEFAULT_PREFERENCE):
    """Keep one file per recording id.

    Returns (kept, dropped, ambiguous) where `kept` is sorted by recording id
    for a stable order independent of glob order, `dropped` maps each id to the
    losing paths, and `ambiguous` lists ids whose winner was chosen among
    several files sharing the *same* extension -- a genuine collision rather
    than a format duplicate, and worth surfacing rather than resolving quietly.
    """
    rank = {ext: i for i, ext in enumerate(prefer)}
    by_id = {}
    for path in paths:
        by_id.setdefault(normalize_id(path), []).append(str(path))

    kept, dropped, ambiguous = [], {}, []
    for rid in sorted(by_id):
        candidates = sorted(
            by_id[rid],
            key=lambda p: (rank.get(os.path.splitext(p)[1].lower(), len(rank)), p),
        )
        winner = candidates[0]
        kept.append(winner)
        if len(candidates) > 1:
            dropped[rid] = candidates[1:]
            win_ext = os.path.splitext(winner)[1].lower()
            if any(os.path.splitext(p)[1].lower() == win_ext for p in candidates[1:]):
                ambiguous.append(rid)
    return kept, dropped, ambiguous


def frame_count(path):
    """Frames in a pose file without reading the pose itself.

    h5py is not installed in the project venv; PyTables is, and it is what
    `pandas.read_hdf` uses underneath, so the h5 branch goes through it. Returns
    None if the count cannot be determined cheaply -- callers should treat that
    as unknown, not as zero.
    """
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".h5":
        try:
            import tables
        except ImportError:
            return None
        try:
            with tables.open_file(path, mode="r") as handle:
                best = None
                for node in handle.walk_nodes("/"):
                    shape = getattr(node, "shape", None)
                    if shape and len(shape) >= 1:
                        best = max(best or 0, int(shape[0]))
                return best
        except Exception:
            return None
    if ext == ".csv":
        try:
            with open(path, "rb") as handle:
                lines = sum(1 for _ in handle)
            # DLC csv carries a 3-row MultiIndex header plus the index column.
            return max(0, lines - 3)
        except OSError:
            return None
    return None


def build_manifest(paths, prefer=DEFAULT_PREFERENCE, count_frames=True):
    """Deduplicated recordings with their parsed design fields.

    Returns (rows, report). `rows` is a list of dicts in the kept order --
    plain dicts rather than a DataFrame so this module stays importable
    without pandas, matching the rest of `representation/`.
    """
    kept, dropped, ambiguous = dedupe(paths, prefer=prefer)

    rows, unparsed = [], []
    for idx, path in enumerate(kept):
        rid = normalize_id(path)
        fields = parse_id(rid)
        if fields is None:
            unparsed.append(rid)
        row = {"idx": idx, "recording_id": rid, "path": path,
               "parse_ok": fields is not None}
        row.update(fields or {k: None for k in
                              ("date", "box", "experiment", "day",
                               "context", "no_shock", "animal")})
        if count_frames:
            row["n_frames_file"] = frame_count(path)
        rows.append(row)

    report = {
        "n_input_files": len(paths),
        "n_recordings": len(kept),
        "n_duplicates_dropped": sum(len(v) for v in dropped.values()),
        "n_ids_with_duplicates": len(dropped),
        "n_ambiguous": len(ambiguous),
        "ambiguous_ids": ambiguous[:20],
        "n_parsed": len(kept) - len(unparsed),
        "n_unparsed": len(unparsed),
        "unparsed_ids": unparsed[:20],
    }
    if count_frames:
        counts = [r["n_frames_file"] for r in rows if r["n_frames_file"]]
        report["n_frames_total"] = int(sum(counts))
        report["n_frames_unknown"] = len(rows) - len(counts)
    return rows, report


def validate_against_metadata(rows, metadata_path):
    """Cross-check parsed fields against the hand-maintained metadata.csv.

    metadata.csv covers 222 of 3,846 recordings, so it cannot be the source of
    truth -- but where it overlaps it is independent of the filename parser,
    and a disagreement means one of the two is wrong. Returns a report; the
    caller decides whether a mismatch is fatal.
    """
    import csv

    by_id = {r["recording_id"]: r for r in rows}
    checked, mismatches, missing = 0, [], 0
    with open(metadata_path, newline="", encoding="utf-8") as handle:
        for meta in csv.DictReader(handle):
            rid = normalize_id(meta.get("filename", ""))
            row = by_id.get(rid)
            if row is None:
                missing += 1
                continue
            checked += 1
            for meta_key, row_key, cast in (("experiment", "experiment", str),
                                            ("day", "day", int),
                                            ("context", "context", str),
                                            ("animal_id", "animal", str)):
                want = meta.get(meta_key, "")
                if want in ("", None):
                    continue
                try:
                    want_cast = cast(want)
                except (TypeError, ValueError):
                    continue
                got = row.get(row_key)
                if cast is str:
                    got, want_cast = str(got), str(want_cast)
                if got != want_cast:
                    mismatches.append(
                        {"recording_id": rid, "field": meta_key,
                         "metadata": want_cast, "parsed": got})
    return {"n_checked": checked, "n_not_in_manifest": missing,
            "n_mismatches": len(mismatches), "mismatches": mismatches[:20]}


def attach_lengths(rows, lengths):
    """Record the post-load frame count beside the pre-load one.

    This is the consistency assertion that a positional artifact cannot provide
    on its own: if `lengths` does not line up with the manifest one-for-one, the
    checkpoint was built from a different file set and every index into it is
    meaningless.
    """
    lengths = np.asarray(lengths)
    if len(rows) != len(lengths):
        raise ValueError(
            f"manifest has {len(rows)} recordings but the checkpoint has "
            f"{len(lengths)} -- these were built from different file sets")
    for row, length in zip(rows, lengths):
        row["n_frames"] = int(length)
    return rows


def save_manifest(out_dir, rows, report=None, name="recordings"):
    """Write `<name>.csv` plus `<name>_report.json`."""
    import csv

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"{name}.csv")
    if rows:
        fields = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    json_path = os.path.join(out_dir, f"{name}_report.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(report or {}, handle, indent=2)
    return csv_path


def load_manifest(out_dir, name="recordings"):
    """Read back `save_manifest`'s csv, restoring int/bool columns."""
    import csv

    path = os.path.join(out_dir, f"{name}.csv")
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key in ("idx", "box", "day", "n_frames", "n_frames_file"):
                if row.get(key) not in (None, ""):
                    row[key] = int(row[key])
            for key in ("no_shock", "parse_ok"):
                if row.get(key) not in (None, ""):
                    row[key] = row[key] in ("True", "true", "1")
            rows.append(row)
    return rows


__all__ = [
    "RECORDING_RE", "DEFAULT_PREFERENCE", "normalize_id", "parse_id", "dedupe",
    "frame_count", "build_manifest", "validate_against_metadata",
    "attach_lengths", "save_manifest", "load_manifest",
]
