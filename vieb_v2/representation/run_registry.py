"""Append-only record of pipeline runs.

Written by both the CLI and the GUI so the Cluster Runs page shows every run
regardless of where it was launched from. Deliberately a plain JSON list rather
than a database: the file is small, human-readable, diffable, and survives being
copied off a cluster with the results.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime

REGISTRY_NAME = "runs.json"


def registry_path(out_dir):
    return os.path.join(out_dir, REGISTRY_NAME)


def new_run_id():
    return uuid.uuid4().hex[:8]


def record(out_dir, latent_method, params, metrics=None, report=None,
           run_id=None, source="cli"):
    """Append one run and return its record.

    `latent_method` is stored at the top level rather than buried in params
    because it is the column the Cluster Runs page exists to show.
    """
    entry = {
        "run_id": run_id or new_run_id(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "latent_method": latent_method,
        "source": source,
        "params": _jsonable(params or {}),
        "metrics": _jsonable(metrics or {}),
        "report": _jsonable(report or {}),
    }

    runs = load(out_dir)
    runs.append(entry)

    os.makedirs(out_dir, exist_ok=True)
    tmp = registry_path(out_dir) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(runs, fh, indent=2)
    # Atomic replace, so a crash mid-write cannot leave a truncated registry
    # that would lose every earlier run.
    os.replace(tmp, registry_path(out_dir))
    return entry


def load(out_dir):
    """Return all recorded runs, newest last. Missing or corrupt -> []."""
    path = registry_path(out_dir)
    if not os.path.exists(path):
        return []
    try:
        with open(path) as fh:
            runs = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []
    return runs if isinstance(runs, list) else []


def summarise(runs):
    """Flatten runs into the rows the Cluster Runs table renders."""
    rows = []
    for run in runs:
        metrics = run.get("metrics") or {}
        clean = metrics.get("clustered_only") or {}
        rows.append({
            "run_id": run.get("run_id", ""),
            "timestamp": run.get("timestamp", ""),
            "latent_method": run.get("latent_method", ""),
            "source": run.get("source", ""),
            "n_states": metrics.get("n_states"),
            "noise_frac": metrics.get("noise_frac"),
            "largest_state_frac": clean.get("largest_state_frac"),
            "state_entropy": clean.get("state_entropy"),
        })
    return rows


def _jsonable(obj):
    import numpy as np

    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj
