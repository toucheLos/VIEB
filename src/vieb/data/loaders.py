"""The one pose loader.

Six independent implementations of "turn a DLC file into ``(pose, conf,
bodyparts)``" existed before this: ``pose_io.load_pose``, ``main.load_pose``,
``compare._cmd_extract_h5``'s inlined copy, ``vieb_v2.representation.pose_loader``,
``tracking.deeplabcut_backend``, and a stale copy under ``build/lib``. They
disagreed on four things that change numbers:

- **csv-vs-h5 preference** — v1 preferred csv, v2 preferred h5.
- **dispatch** — v1 branched on the file extension and assumed a DLC MultiIndex
  inside every h5, so a flat-column h5 crashed; v2 branched on the column
  structure after loading, which handles both.
- **missing-value fill** — v1 filled absent coordinates with ``0.0`` and absent
  likelihoods with ``1.0``, i.e. "at the origin, with full confidence"; v2 filled
  ``NaN``, i.e. "unknown". That is a semantic difference, not a formatting one,
  and it propagates into every downstream feature.
- **skip lists** — neither covered the other's cases.

This module resolves each choice once and takes ``fill`` as an explicit parameter
rather than baking in one project's answer, because the ported arms must each be
run under the convention their reference output was produced with.
"""

from __future__ import annotations

import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..io.vus1 import normalize_recording_id
from .dataset import PoseDataset

POSE_PATTERNS = ("*.h5", "*.csv")

# Union of v1's and v2's skip lists — neither covered the other's cases.
# Matched against the *basename*, never the full path: both originals tested the
# whole path, so a project living under a directory called `metadata` or
# `..._meta/` silently found zero pose files.
SKIP_FILE_MARKERS = (
    "CollectedData", "machinelabels", "metadata",
    "_meta.", "_full.h5", "_full.pickle",
)

# DLC's labelling directories, matched as whole path components.
SKIP_DIRS = ("labeled-data", "training-datasets", "dlc-models")

# Preference when the same recording exists in several formats. h5 first: it is
# the format all 3,846 Luna recordings share, it carries the likelihoods without
# a text round-trip, and its frame count is readable from the node shape.
DEFAULT_PREFERENCE = (".h5", ".hdf5", ".csv", ".npy")

_FLAT_RE = re.compile(r"^(?P<bp>.+?)[_/](x|y|likelihood)$", re.IGNORECASE)

# DLC writes scorer/bodyparts/coords as three header rows.
_DLC_HEADER_TOKENS = ("scorer", "bodyparts", "coords")


class FrameCountMismatch(ValueError):
    """A file's cheap frame count disagrees with the pose actually parsed from it.

    Raised, never warned. A silent off-by-one here shifts every later index and
    attributes each recording's behavior to a neighbouring animal, while still
    producing entirely plausible p-values.
    """


def find_pose_files(*roots) -> list[str]:
    """Locate pose files under ``roots``, skipping DLC's bookkeeping outputs.

    Returned in ``sorted()`` order, which is the order every existing checkpoint's
    positional index was built in.
    """
    found: list[str] = []
    for root in roots:
        if not root or not os.path.isdir(str(root)):
            continue
        for pattern in POSE_PATTERNS:
            found.extend(
                glob.glob(os.path.join(str(root), "**", pattern), recursive=True)
            )
    return sorted({f for f in found if _is_pose_file(f, root)})


def _is_pose_file(path: str, root) -> bool:
    """Whether a globbed file is a pose file rather than DLC bookkeeping.

    File markers are matched against the basename and directory markers against
    the path components *below the root*, so nothing about where the project
    happens to live can exclude its own data.
    """
    p = Path(path)
    name = p.name
    if any(marker in name for marker in SKIP_FILE_MARKERS):
        return False
    try:
        parts = p.relative_to(Path(str(root))).parts[:-1]
    except ValueError:  # pragma: no cover - path not under root
        parts = p.parts[:-1]
    return not any(part in SKIP_DIRS for part in parts)


def dedupe(paths, prefer=DEFAULT_PREFERENCE) -> tuple[list[str], dict, list[str]]:
    """Keep one file per recording id.

    Of the 4,925 pose files under Luna's ``raw_videos``, 3,846 are ``.h5`` and
    1,079 are ``.csv`` — and each of those csv files is the *same recording* as one
    of the h5 files, under DLC's other export format. Treating them as 4,925
    independent recordings inflates the frame total to 28,626,107 against a true
    22,355,989, and for anything that estimates an occupancy or a stationary
    measure those 1,079 recordings contribute twice the mass.

    Returns ``(kept, dropped, ambiguous)``. ``kept`` is sorted by recording id, so
    the order does not depend on glob order. ``ambiguous`` lists ids whose winner
    was chosen among several files sharing the *same* extension — a genuine
    collision rather than a format duplicate, surfaced rather than resolved quietly.
    """
    rank = {ext: i for i, ext in enumerate(prefer)}
    by_id: dict[str, list[str]] = {}
    for path in paths:
        by_id.setdefault(normalize_recording_id(path), []).append(str(path))

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


def _csv_header_rows(path: str) -> int:
    """How many header rows this csv has: 3 for DLC's layout, else 1."""
    with open(path, "r", errors="replace") as handle:
        first = [handle.readline().split(",", 1)[0].strip().strip('"').lower()
                 for _ in range(3)]
    return 3 if first == list(_DLC_HEADER_TOKENS) else 1


def frame_count(path) -> int:
    """Frames in a pose file, without parsing the pose.

    Raises rather than returning None. The previous implementation returned None
    on any failure and its caller then treated None as "unknown" and excluded it
    from the total — which also silently dropped legitimate zeros, and meant a
    corrupt file reduced the frame count instead of stopping the run.
    """
    path = str(path)
    ext = os.path.splitext(path)[1].lower()

    if ext in (".h5", ".hdf5"):
        try:
            import tables
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise FrameCountMismatch(
                f"cannot count frames in {path}: PyTables is not installed"
            ) from exc
        with tables.open_file(path, mode="r") as handle:
            best = None
            for node in handle.walk_nodes("/"):
                shape = getattr(node, "shape", None)
                if shape and len(shape) >= 1:
                    best = max(best or 0, int(shape[0]))
        if best is None:
            raise FrameCountMismatch(f"{path} contains no array node to count")
        return int(best)

    if ext == ".csv":
        header = _csv_header_rows(path)
        with open(path, "rb") as handle:
            lines = sum(1 for _ in handle)
        return max(0, lines - header)

    raise FrameCountMismatch(f"cannot count frames in {path}: unsupported extension {ext!r}")


def load_pose_file(path, *, fill: str = "nan"):
    """Load one pose file. -> ``(pose (T,K,2), conf (T,K) or None, bodyparts)``.

    Dispatch is on the *column structure* after loading, not on the file
    extension, so a flat-column h5 works — v1's extension branch assumed a DLC
    MultiIndex inside every h5 and crashed on those.

    ``fill`` decides what an absent coordinate becomes:
      ``"nan"``   unknown (v2's convention; ``interpolate_gaps`` fills them later)
      ``"zero"``  at the origin with likelihood 1.0 (v1's convention)

    There is no default that is right for both, so the caller states which
    reference output it is reproducing.
    """
    if fill not in ("nan", "zero"):
        raise ValueError(f"fill must be 'nan' or 'zero', got {fill!r}")

    path = str(path)
    if path.lower().endswith((".h5", ".hdf5")):
        df = pd.read_hdf(path)
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"{path}: unexpected HDF5 payload {type(df).__name__}")
    else:
        df = _read_dlc_csv(path)

    if isinstance(df.columns, pd.MultiIndex):
        pose, conf, bodyparts = _from_multiindex(df)
    else:
        pose, conf, bodyparts = _from_flat(df)

    if fill == "zero":
        # v1's convention: a missing keypoint is "at the origin, fully confident".
        pose = np.nan_to_num(pose, nan=0.0)
        conf = np.ones(pose.shape[:2]) if conf is None else np.nan_to_num(conf, nan=1.0)

    return pose, conf, bodyparts


def _read_dlc_csv(path: str) -> pd.DataFrame:
    """Read a csv with the right number of header rows for its actual layout.

    The inherited implementation tried ``header=[0, 1, 2]`` first and accepted the
    result whenever the columns came back as a MultiIndex — but pandas *always*
    builds a MultiIndex from three header rows, so the check never failed and a
    flat one-header-row csv was silently parsed with its first two data rows
    consumed as headers. The header layout is detected instead.
    """
    if _csv_header_rows(path) == 3:
        return pd.read_csv(path, header=[0, 1, 2], index_col=0)
    return pd.read_csv(path)


def _from_multiindex(df: pd.DataFrame):
    # The bodypart level is second in DLC's (scorer, bodypart, coord) layout.
    level = 1 if df.columns.nlevels >= 3 else 0
    bodyparts = list(dict.fromkeys(df.columns.get_level_values(level)))
    bodyparts = [b for b in bodyparts if not str(b).startswith("Unnamed")]

    n = len(df)
    pose = np.full((n, len(bodyparts), 2), np.nan)
    conf = np.full((n, len(bodyparts)), np.nan)

    for k, bp in enumerate(bodyparts):
        block = df.xs(bp, axis=1, level=level)
        coords = {str(c).lower(): c for c in block.columns.get_level_values(-1)}
        for j, axis in enumerate(("x", "y")):
            if axis in coords:
                pose[:, k, j] = block.xs(coords[axis], axis=1, level=-1).values[:, 0]
        if "likelihood" in coords:
            conf[:, k] = block.xs(coords["likelihood"], axis=1, level=-1).values[:, 0]

    return pose, (None if np.isnan(conf).all() else conf), bodyparts


def _from_flat(df: pd.DataFrame):
    bodyparts, columns = [], {}
    for col in df.columns:
        m = _FLAT_RE.match(str(col))
        if not m:
            continue
        bp = m.group("bp")
        if bp not in columns:
            columns[bp] = {}
            bodyparts.append(bp)
        columns[bp][str(col).rsplit("_", 1)[-1].lower()] = col

    if not bodyparts:
        raise ValueError("no recognisable pose columns found")

    n = len(df)
    pose = np.full((n, len(bodyparts), 2), np.nan)
    conf = np.full((n, len(bodyparts)), np.nan)
    for k, bp in enumerate(bodyparts):
        for j, axis in enumerate(("x", "y")):
            if axis in columns[bp]:
                pose[:, k, j] = df[columns[bp][axis]].values
        if "likelihood" in columns[bp]:
            conf[:, k] = df[columns[bp]["likelihood"]].values

    return pose, (None if np.isnan(conf).all() else conf), bodyparts


def interpolate_gaps(pose, conf=None, max_gap=None):
    """Fill NaN keypoints by linear interpolation over time.

    Kept separate from confidence weighting on purpose: a *missing* point has to
    be filled to compute anything at all, whereas a merely *low-confidence* point
    is better down-weighted than replaced, since interpolation invents pose.
    """
    pose = np.array(pose, dtype=np.float64, copy=True)
    T, K, _ = pose.shape
    t = np.arange(T)

    for k in range(K):
        for j in range(2):
            column = pose[:, k, j]
            bad = np.isnan(column)
            if not bad.any() or bad.all():
                continue
            if max_gap is not None and _longest_run(bad) > max_gap:
                continue
            column[bad] = np.interp(t[bad], t[~bad], column[~bad])
            pose[:, k, j] = column

    if conf is not None:
        conf = np.nan_to_num(np.asarray(conf, dtype=np.float64), nan=0.0)
    return pose, conf


def _longest_run(mask) -> int:
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def load_dataset(
    root,
    *,
    fps: float,
    fill: str = "nan",
    deduplicate: bool = True,
    verify_frame_counts: bool = True,
    max_gap: int | None = None,
    min_frames: int = 2,
    metadata: pd.DataFrame | None = None,
    dataset: str = "",
    limit: int | None = None,
    paths: list | None = None,
) -> tuple[PoseDataset, dict]:
    """Load a directory of DLC output into one ``PoseDataset``.

    ``deduplicate=False`` reproduces the **pre-dedup** file set — all 4,925 Luna
    files in ``sorted()`` order — which is what the ``koopman_*`` runs were fit on.
    The verification gate needs it: an arm can only be shown to match its
    reference output if it is handed the same input bytes. It is not a mode any
    new run should use.

    ``verify_frame_counts`` compares each file's cheap frame count against the
    pose actually parsed from it and raises on any mismatch.

    Returns ``(dataset, report)``.
    """
    all_paths = list(paths) if paths is not None else find_pose_files(root)
    if not all_paths:
        raise FileNotFoundError(f"no pose files found under {root!r}")

    if deduplicate:
        kept, dropped, ambiguous = dedupe(all_paths)
    else:
        kept, dropped, ambiguous = list(all_paths), {}, []

    if limit is not None:
        kept = kept[:limit]

    sessions, confs, ids, skipped = [], [], [], []
    canonical: list[str] = []
    bodyparts: list[str] | None = None
    mismatches: list[str] = []
    seen: dict[str, int] = {}

    for path in kept:
        rid = base_rid = normalize_recording_id(path)
        # Without deduplication the same recording appears once per export
        # format, so ids collide by construction. Unique ids are load-bearing
        # (a duplicate is how a join double-counts), so later occurrences are
        # disambiguated by their source extension and the true id is kept in
        # `canonical` for anything that needs to join back.
        if not deduplicate:
            n_seen = seen.get(base_rid, 0)
            seen[base_rid] = n_seen + 1
            if n_seen:
                rid = f"{base_rid}#{os.path.splitext(path)[1].lstrip('.').lower()}"
        try:
            pose, conf, names = load_pose_file(path, fill=fill)
        except Exception as exc:
            skipped.append((path, f"{type(exc).__name__}: {exc}"))
            continue

        if verify_frame_counts:
            declared = frame_count(path)
            if declared != pose.shape[0]:
                mismatches.append(
                    f"{path}: file declares {declared} frames, parsed {pose.shape[0]}"
                )
                continue

        if pose.shape[0] < min_frames:
            skipped.append((path, f"only {pose.shape[0]} frames"))
            continue

        pose, conf = interpolate_gaps(pose, conf, max_gap)
        if np.isnan(pose).any():
            skipped.append((path, "unfillable gaps remain"))
            continue

        if bodyparts is None:
            bodyparts = names
        elif list(names) != list(bodyparts):
            raise ValueError(
                f"{path} has bodyparts {names} but earlier files had {bodyparts}; "
                f"a dataset must be uniform in its keypoints"
            )

        sessions.append(pose)
        confs.append(conf if conf is not None else np.ones(pose.shape[:2]))
        ids.append(rid)
        canonical.append(base_rid)

    if mismatches:
        raise FrameCountMismatch(
            f"{len(mismatches)} of {len(kept)} files disagree with their own frame "
            f"count. A file was truncated or has changed on disk, so every later "
            f"index is shifted.\n  " + "\n  ".join(mismatches[:10])
            + (f"\n  ... and {len(mismatches) - 10} more" if len(mismatches) > 10 else "")
        )

    if not sessions:
        raise ValueError(
            f"every one of {len(kept)} pose files was skipped; "
            f"first reasons: {skipped[:3]}"
        )

    data = PoseDataset.from_sessions(
        sessions, ids, bodyparts or [], fps,
        confidences=confs,
        metadata=metadata if metadata is not None else pd.DataFrame(),
        dataset=dataset,
    )

    report = {
        "n_input_files": len(all_paths),
        "n_selected": len(kept),
        "n_loaded": len(sessions),
        "n_skipped": len(skipped),
        "skipped": skipped[:20],
        "n_duplicates_dropped": sum(len(v) for v in dropped.values()),
        "n_ids_with_duplicates": len(dropped),
        "n_ambiguous": len(ambiguous),
        "ambiguous_ids": ambiguous[:20],
        "deduplicated": bool(deduplicate),
        "fill": fill,
        "n_frames_total": int(data.n_frames),
        "n_recordings": int(data.n_recordings),
        "canonical_recording_ids": canonical,
        "n_disambiguated_ids": sum(1 for a, b in zip(ids, canonical) if a != b),
    }
    return data, report


def assert_id_overlap(a: list[str], b: list[str], *, minimum: float = 0.90,
                      names: tuple[str, str] = ("a", "b")) -> float:
    """Assert two arms' recording ids overlap enough to be compared.

    Ids that disagree between arms are how a join silently drops rows and still
    produces a table. This is checked *before* any comparison claims to have run,
    because afterwards the only symptom is a smaller n.
    """
    sa, sb = set(a), set(b)
    if not sa or not sb:
        raise ValueError(f"cannot compare {names[0]} and {names[1]}: one side is empty")
    overlap = len(sa & sb) / max(len(sa), len(sb))
    if overlap < minimum:
        only_a = sorted(sa - sb)[:5]
        only_b = sorted(sb - sa)[:5]
        raise ValueError(
            f"{names[0]} and {names[1]} share only {overlap:.1%} of their recording "
            f"ids ({len(sa & sb)} of {max(len(sa), len(sb))}); need {minimum:.0%}. "
            f"This is normalization drift, not missing data.\n"
            f"  only in {names[0]}: {only_a}\n  only in {names[1]}: {only_b}"
        )
    return overlap
