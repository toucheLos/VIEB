"""
pose_io.py — DLC pose loading utilities for VIEB.

All pipeline scripts import load_pose and _find_dlc_csv from here.
H5-based pose sources (single shared H5 file containing multiple
animals/trials) are loaded via inspect_h5() / load_pose_h5().
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd


def _find_dlc_csv(video_path: str) -> str | None:
    """Return the DLC-generated CSV for a given video file, or None."""
    stem = os.path.splitext(video_path)[0]
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    patterns = [
        f"{stem}*.csv",
        os.path.join(video_dir, f"{video_name}*.csv"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        matches = [m for m in matches if "metadata" not in os.path.basename(m).lower()]
        if matches:
            return matches[0]
    return None


def _pose_from_dlc_df(df: "pd.DataFrame") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert a DLC-style DataFrame (3-level MultiIndex columns: scorer,
    bodypart, x/y/likelihood) into (pose, conf, bodyparts)."""
    bodyparts = df.columns.get_level_values(1).unique().tolist()
    scorer = df.columns.get_level_values(0)[0]

    T = len(df)
    K = len(bodyparts)
    pose = np.zeros((T, K, 2))
    conf = np.zeros((T, K))

    for k, bp in enumerate(bodyparts):
        pose[:, k, 0] = df[(scorer, bp, "x")].values
        pose[:, k, 1] = df[(scorer, bp, "y")].values
        if (scorer, bp, "likelihood") in df.columns:
            conf[:, k] = df[(scorer, bp, "likelihood")].values
        else:
            conf[:, k] = 1.0

    return pose, conf, bodyparts


def load_pose(csv_path: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load DLC CSV output into a pose array.

    Returns
    -------
    pose      : np.ndarray  shape (T, K, 2)  — x, y per keypoint per frame
    conf      : np.ndarray  shape (T, K)     — likelihood per keypoint per frame
    bodyparts : list[str]
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    return _pose_from_dlc_df(df)


# ---------------------------------------------------------------------------
# H5 pose source (single shared file containing multiple animals/trials)
# ---------------------------------------------------------------------------

_FLAT_COL_RE = re.compile(r"^(?P<bp>.+?)[_/](x|y|likelihood)$", re.IGNORECASE)


def _pose_from_flat_df(df: "pd.DataFrame") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Convert a flat-column DataFrame (e.g. nose_x, nose_y, nose_likelihood)
    into (pose, conf, bodyparts)."""
    bodyparts: list[str] = []
    for col in df.columns:
        m = _FLAT_COL_RE.match(str(col))
        if m:
            bp = m.group("bp")
            if bp not in bodyparts:
                bodyparts.append(bp)

    if not bodyparts:
        raise ValueError(
            "Could not infer bodyparts from flat H5 columns: "
            f"{list(df.columns)[:10]}..."
        )

    T = len(df)
    K = len(bodyparts)
    pose = np.zeros((T, K, 2))
    conf = np.ones((T, K))

    for k, bp in enumerate(bodyparts):
        for axis, idx in (("x", 0), ("y", 1)):
            for sep in ("_", "/"):
                col = f"{bp}{sep}{axis}"
                if col in df.columns:
                    pose[:, k, idx] = df[col].values
                    break
        for sep in ("_", "/"):
            col = f"{bp}{sep}likelihood"
            if col in df.columns:
                conf[:, k] = df[col].values
                break

    return pose, conf, bodyparts


def inspect_h5(h5_path: str) -> dict:
    """
    Inspect an H5 pose file and report its available keys/groups.

    Returns
    -------
    dict with:
      "keys"    : list[str]  — top-level keys/groups in the file
      "details" : dict[str, dict] — per-key info:
          for pandas-stored frames: {"columns": [...], "n_frames": int}
          for raw h5py groups/datasets: {"datasets": {name: shape}, "n_frames": int|None}
    """
    info: dict = {"keys": [], "details": {}}

    try:
        with pd.HDFStore(h5_path, mode="r") as store:
            keys = [k.lstrip("/") for k in store.keys()]
            for key, raw_key in zip(keys, store.keys()):
                try:
                    storer = store.get_storer(raw_key)
                    n_frames = int(getattr(storer, "nrows", 0)) or None
                    sample = store.select(raw_key, start=0, stop=1)
                    columns = [
                        "/".join(str(c) for c in col) if isinstance(col, tuple) else str(col)
                        for col in sample.columns
                    ]
                except Exception:
                    n_frames = None
                    columns = []
                info["keys"].append(key)
                info["details"][key] = {"columns": columns, "n_frames": n_frames}
        if info["keys"]:
            return info
    except Exception:
        pass

    # Fall back to raw h5py group/dataset layout
    import h5py

    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            item = f[key]
            details: dict = {}
            if isinstance(item, h5py.Dataset):
                details["shape"] = tuple(item.shape)
                details["n_frames"] = item.shape[0] if item.ndim > 0 else None
            else:
                datasets = {
                    name: tuple(item[name].shape)
                    for name in item.keys()
                    if isinstance(item[name], h5py.Dataset)
                }
                details["datasets"] = datasets
                n_frames = None
                for shape in datasets.values():
                    if shape:
                        n_frames = shape[0]
                        break
                details["n_frames"] = n_frames
            info["keys"].append(key)
            info["details"][key] = details

    return info


def load_pose_h5(
    h5_path: str,
    key: str | None = None,
    source_col: str | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load a single animal/trial's pose data from a shared H5 pose file.

    Parameters
    ----------
    h5_path    : path to the .h5 file
    key        : top-level key/group identifying the animal/trial. Required
                 unless the file contains exactly one key.
    source_col : for raw h5py group layouts, the dataset name within the
                 group holding a (T, K, 2 or 3) coordinate array.

    Returns
    -------
    pose, conf, bodyparts — same contract as load_pose().
    """
    info = inspect_h5(h5_path)
    keys = info["keys"]
    if not keys:
        raise ValueError(f"No keys found in H5 file: {h5_path}")

    if key is None:
        if len(keys) == 1:
            key = keys[0]
        else:
            raise ValueError(
                f"H5 file {h5_path} contains multiple keys; specify `key`. "
                f"Available keys: {keys}"
            )
    if key not in keys:
        raise ValueError(f"Key '{key}' not found in {h5_path}. Available keys: {keys}")

    details = info["details"].get(key, {})

    # Pandas-stored frame (DLC-style MultiIndex or flat columns)
    if "columns" in details:
        with pd.HDFStore(h5_path, mode="r") as store:
            raw_key = key if key.startswith("/") else f"/{key}"
            df = store[raw_key]
        if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels >= 3:
            return _pose_from_dlc_df(df)
        return _pose_from_flat_df(df)

    # Raw h5py group/dataset layout
    import h5py

    with h5py.File(h5_path, "r") as f:
        item = f[key]
        if isinstance(item, h5py.Dataset):
            arr = np.asarray(item)
        else:
            if source_col is None:
                ds_names = list(details.get("datasets", {}).keys())
                if len(ds_names) == 1:
                    source_col = ds_names[0]
                else:
                    raise ValueError(
                        f"Group '{key}' in {h5_path} has multiple datasets; "
                        f"specify `source_col`. Available: {ds_names}"
                    )
            arr = np.asarray(item[source_col])

    if arr.ndim != 3 or arr.shape[-1] not in (2, 3):
        raise ValueError(
            f"Expected a (T, K, 2 or 3) coordinate array for key '{key}' "
            f"(source_col={source_col!r}), got shape {arr.shape}"
        )

    T, K = arr.shape[0], arr.shape[1]
    pose = arr[:, :, :2].astype(float)
    if arr.shape[-1] >= 3:
        conf = arr[:, :, 2].astype(float)
    else:
        conf = np.ones((T, K))

    bodyparts = [f"kp{i}" for i in range(K)]
    return pose, conf, bodyparts
