"""VUS-1: the one output schema every model emits, written by the harness.

Two files per run:

  - ``run_manifest.json``  provenance; a row with ``git_dirty: true`` is not
    reproducible and is refused by the comparison table.
  - ``bouts.parquet``      ``recording_id, state, start_frame, end_frame``

Models never write these. They return per-frame labels and the harness does the
rest, so that bout construction is identical across arms — otherwise a
difference in run-length encoding shows up as a difference in bout duration,
which is one of the things being measured.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..models.base import UNASSIGNED, RepresentationMeta

BOUT_COLUMNS = ["recording_id", "state", "start_frame", "end_frame"]


def encode_bouts(
    labels: np.ndarray,
    meta: RepresentationMeta,
    *,
    keep_unassigned: bool = False,
) -> pd.DataFrame:
    """Run-length encode per-frame labels into bouts, per recording.

    ``end_frame`` is exclusive and frame indices are recording-local, so a bout
    is ``labels[start:end]`` within its own recording. Encoding restarts at every
    recording boundary — a bout never spans two recordings even when the label
    happens to be continuous across the seam.

    ``-1`` runs are dropped by default. They are absence of a state, not a state;
    including them would let an arm that abstains on 40% of frames report a
    tighter duration distribution than one that commits.
    """
    labels = np.asarray(labels)
    if labels.shape[0] != meta.n_frames:
        raise ValueError(
            f"labels has {labels.shape[0]} frames, representation has {meta.n_frames}"
        )

    rec_ids: list[str] = []
    states: list[int] = []
    starts: list[int] = []
    ends: list[int] = []

    for rid, sl in meta.slices():
        seg = labels[sl]
        if seg.size == 0:
            continue
        # Boundaries where the label changes, in recording-local coordinates.
        change = np.flatnonzero(np.diff(seg)) + 1
        bounds = np.concatenate(([0], change, [seg.size]))
        for a, b in zip(bounds[:-1], bounds[1:]):
            state = int(seg[a])
            if state == UNASSIGNED and not keep_unassigned:
                continue
            rec_ids.append(rid)
            states.append(state)
            starts.append(int(a))
            ends.append(int(b))

    return pd.DataFrame(
        {
            "recording_id": pd.Series(rec_ids, dtype="string"),
            "state": pd.Series(states, dtype="int32"),
            "start_frame": pd.Series(starts, dtype="int64"),
            "end_frame": pd.Series(ends, dtype="int64"),
        },
        columns=BOUT_COLUMNS,
    )


def normalize_recording_id(name: str) -> str:
    """Strip a trailing ``DLC_*`` suffix and any file extension.

    ``20241016_Box_1_CFC_Day_0_(Context_A)_308DLC_Resnet50_VIEBFeb11shuffle2_snapshot_best-30.h5``
    becomes ``20241016_Box_1_CFC_Day_0_(Context_A)_308``.

    Every loader routes through this. Recording ids that disagree between arms
    are how a join silently drops rows.
    """
    stem = Path(str(name)).name
    for ext in (".h5", ".hdf5", ".csv", ".mp4", ".avi", ".npy", ".parquet"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    idx = stem.find("DLC_")
    if idx > 0:
        stem = stem[:idx]
    return stem


@dataclass
class RunManifest:
    """Provenance for one run. Written to ``run_manifest.json``."""

    model: str
    model_version: str
    config: dict[str, Any]
    config_hash: str
    repr_name: str
    repr_hash: str
    dataset: str
    fps: float
    git_sha: str
    git_dirty: bool
    seed: int
    device: str
    wall_clock_s: float
    library_versions: dict[str, str] = field(default_factory=dict)
    # Resource accounting, so the next request is calibrated rather than doubled
    # on faith. Filled in post-hoc from `sacct -o MaxRSS` where available.
    mem_requested_gb: float | None = None
    mem_peak_gb: float | None = None
    slurm_job_id: str | None = None
    n_states: int | None = None
    unassigned_frac: float | None = None
    notes: str = ""

    def write(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: Path) -> "RunManifest":
        return cls(**json.loads(Path(path).read_text()))


def git_state(repo: Path | None = None) -> tuple[str, bool]:
    """Return ``(sha, dirty)`` for the working tree.

    ``dirty`` is load-bearing: a dirty row cannot be reproduced from the recorded
    sha, so the comparison script refuses to publish it.
    """
    cwd = str(repo) if repo else None
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, cwd=cwd,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, cwd=cwd,
        ).stdout.strip()
        return sha, bool(porcelain)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown", True
