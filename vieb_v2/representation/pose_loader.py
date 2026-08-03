"""Read DLC pose output.

Self-contained on purpose: v2 does not import v1's `pose_io`, so the backend
runs on a cluster node that only has this checkout. The formats handled are the
two DLC actually emits -- a three-level-header CSV and the matching HDF5 -- plus
flat `<bodypart>_x` style columns, which is what most hand-exported files look
like.
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np

POSE_PATTERNS = ("*.h5", "*.csv")
_FLAT_RE = re.compile(r"^(?P<bp>.+?)[_/](x|y|likelihood)$", re.IGNORECASE)


def find_pose_files(*roots):
    """Locate pose files, skipping DLC's own bookkeeping outputs."""
    found = []
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for pattern in POSE_PATTERNS:
            found.extend(glob.glob(os.path.join(root, "**", pattern),
                                   recursive=True))
    skip = ("CollectedData", "_meta", "machinelabels", "labeled-data")
    return sorted({f for f in found if not any(s in f for s in skip)})


def load_pose(path):
    """Load one pose file. -> (pose (T,K,2), conf (T,K) or None, bodyparts)."""
    import pandas as pd

    if path.endswith(".h5"):
        df = pd.read_hdf(path)
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"{path}: unexpected HDF5 payload")
    else:
        df = _read_dlc_csv(path)

    if isinstance(df.columns, pd.MultiIndex):
        return _from_multiindex(df)
    return _from_flat(df)


def _read_dlc_csv(path):
    import pandas as pd

    # DLC writes scorer/bodypart/coord as three header rows.
    try:
        df = pd.read_csv(path, header=[0, 1, 2], index_col=0)
        if isinstance(df.columns, pd.MultiIndex):
            return df
    except Exception:
        pass
    return pd.read_csv(path)


def _from_multiindex(df):
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
                pose[:, k, j] = block.xs(coords[axis], axis=1,
                                         level=-1).values[:, 0]
        if "likelihood" in coords:
            conf[:, k] = block.xs(coords["likelihood"], axis=1,
                                  level=-1).values[:, 0]

    return pose, (None if np.isnan(conf).all() else conf), bodyparts


def _from_flat(df):
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

    Alignment cannot proceed through NaNs. This is deliberately kept separate
    from the confidence weighting in `align`: a *missing* point has to be
    filled to compute anything at all, whereas a merely *low-confidence* point
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


def _longest_run(mask):
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def load_sessions(paths, max_gap=None, min_frames=2):
    """Load several recordings, dropping any that are unusable.

    Returns (sessions, bodyparts, skipped) where sessions is the
    list-of-recordings the rest of the pipeline expects -- never concatenated,
    so delay embedding cannot cross a boundary.
    """
    sessions, bodyparts, skipped = [], None, []
    for path in paths:
        try:
            pose, conf, names = load_pose(path)
        except Exception as exc:
            skipped.append((path, f"{type(exc).__name__}: {exc}"))
            continue
        if pose.shape[0] < min_frames:
            skipped.append((path, f"only {pose.shape[0]} frames"))
            continue
        pose, conf = interpolate_gaps(pose, conf, max_gap)
        if np.isnan(pose).any():
            skipped.append((path, "unfillable gaps remain"))
            continue
        bodyparts = bodyparts or names
        sessions.append((pose, conf))
    return sessions, bodyparts, skipped
