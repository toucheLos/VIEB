"""Loading v1 feature files, with the mixed-dimension hazard handled explicitly.

`results/features/` holds output from more than one extraction run: 3526 files
are 51-D (`--no-wavelets`) and 320 are 91-D (with Morlet wavelets). Only the
51-D layout matches the `feature_names` list in `index.json`, which is the
authoritative description of the column positions.

This matters because every consumer indexes features *by column position*. A
naive glob either crashes on the concatenate -- which is the lucky case -- or,
if it slices a fixed range, silently reads columns from two different feature
spaces and averages them into one result. There is no in-band signal that it
happened.

So the dimension is resolved from `index.json` rather than guessed, files that
disagree are excluded rather than coerced, and the count of exclusions is
returned so it appears in the report instead of being discovered later.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np


def declared_dimension(features_dir):
    """Feature count declared by index.json, or None if it cannot be read."""
    path = os.path.join(features_dir, "index.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh).get("_meta", {})
        names = meta.get("feature_names")
        return len(names) if names else None
    except (ValueError, OSError):
        return None


def feature_files(features_dir, dim=None):
    """Files whose column count matches `dim` (default: whatever index.json says).

    Returns (paths, report). The report carries the excluded counts by
    dimension so a run can state what it ignored.
    """
    paths = sorted(glob.glob(os.path.join(features_dir, "*_features.npy")))
    if not paths:
        raise SystemExit(f"no *_features.npy under {features_dir!r}")

    if dim is None:
        dim = declared_dimension(features_dir)

    by_dim = {}
    for path in paths:
        width = int(np.load(path, mmap_mode="r").shape[1])
        by_dim.setdefault(width, []).append(path)

    if dim is None:
        dim = max(by_dim, key=lambda k: len(by_dim[k]))

    kept = by_dim.get(dim, [])
    if not kept:
        raise SystemExit(
            f"no feature file has {dim} columns; found {sorted(by_dim)}")

    return kept, {
        "feature_dim_used": int(dim),
        "dim_source": ("index.json" if declared_dimension(features_dir) == dim
                       else "majority vote"),
        "n_files_kept": len(kept),
        "n_files_excluded": sum(len(v) for k, v in by_dim.items() if k != dim),
        "excluded_by_dim": {str(k): len(v) for k, v in sorted(by_dim.items())
                            if k != dim},
    }


__all__ = ["declared_dimension", "feature_files"]
