"""Stage checkpoints shared by the CLI and the GUI.

Both paths write the same four `.npz` files into the output directory, so a run
started in the GUI is indistinguishable on disk from one started with `cli run`
-- and the Pipeline and Artifacts pages report the same thing either way.

Ragged per-recording arrays are stored as a concatenated block plus the
per-recording lengths. The lengths are what make the recording boundaries
recoverable; a flattened blob alone would let a later stage delay-embed across
two animals.
"""

from __future__ import annotations

import json
import os

import numpy as np

ALIGNED = "aligned.npz"
SCORES = "scores.npz"
EMBEDDED = "embedded.npz"
LABELS = "labels.npz"
# Basin labels sit beside HDBSCAN's rather than overwriting them: the two are
# built from different frame sets (no delay window is dropped here), so both
# must survive for the comparison to be possible at all.
KOOPMAN_LABELS = "koopman_labels.npz"

STAGE_FILES = (ALIGNED, SCORES, EMBEDDED, LABELS)


def pack(sessions):
    """Ragged list of per-recording arrays -> flat block + lengths."""
    return {"stacked": np.concatenate(sessions, axis=0),
            "lengths": np.array([len(s) for s in sessions])}


def unpack(data):
    """Inverse of `pack`, restoring the per-recording split."""
    return np.split(data["stacked"], np.cumsum(data["lengths"])[:-1])


def save(path, arrays, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(jsonable(meta)), **arrays)
    return path


def load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    return data, json.loads(str(data["meta"]))


def save_run(out_dir, aligned, scores, embedded, index, labels, probs, meta):
    """Write all four stage checkpoints for a completed in-memory run."""
    os.makedirs(out_dir, exist_ok=True)
    save(os.path.join(out_dir, ALIGNED),
         pack([a.reshape(a.shape[0], -1) for a in aligned]), meta)
    save(os.path.join(out_dir, SCORES), pack(scores), meta)
    save(os.path.join(out_dir, EMBEDDED),
         {"embedded": embedded, "index": index}, meta)
    save(os.path.join(out_dir, LABELS),
         {"labels": labels, "probabilities": probs, "index": index}, meta)
    return out_dir


def completed_stages(out_dir):
    """Which stage checkpoints exist. Filesystem-driven, so a CLI or batch run
    is visible to the GUI without it having observed the run."""
    return [name for name in STAGE_FILES
            if os.path.exists(os.path.join(out_dir, name))]


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    return obj
