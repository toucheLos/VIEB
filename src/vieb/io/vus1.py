"""VUS-1: the one output schema every arm emits, written by the harness.

Two files per run:

  - ``run_manifest.json``  provenance; a row with ``git_dirty: true`` is not
    reproducible and is refused by the comparison table.
  - ``bouts.parquet``      ``recording_id, state, start_frame, end_frame``

Segmenters never write these. They return per-frame labels and the harness does
the rest, so that bout construction is identical across arms — otherwise a
difference in run-length encoding shows up as a difference in bout duration, which
is one of the things being measured.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..data.dataset import UNASSIGNED, PoseDataset

SCHEMA_VERSION = "VUS-1"

BOUT_COLUMNS = ["recording_id", "state", "start_frame", "end_frame"]

MANIFEST_NAME = "run_manifest.json"
BOUTS_NAME = "bouts.parquet"


def encode_bouts(
    labels: np.ndarray,
    data: PoseDataset,
    *,
    keep_unassigned: bool = False,
) -> pd.DataFrame:
    """Run-length encode per-frame labels into bouts, per recording.

    ``end_frame`` is exclusive and frame indices are recording-local, so a bout is
    ``labels[start:end]`` within its own recording. Encoding restarts at every
    recording boundary — a bout never spans two recordings even when the label
    happens to be continuous across the seam.

    ``-1`` runs are dropped by default. They are absence of a state, not a state;
    including them would let an arm that abstains on 40% of frames report a tighter
    duration distribution than one that commits.
    """
    labels = np.asarray(labels)
    if labels.shape[0] != data.n_frames:
        raise ValueError(
            f"labels has {labels.shape[0]} frames, dataset has {data.n_frames}"
        )

    rec_ids: list[str] = []
    states: list[int] = []
    starts: list[int] = []
    ends: list[int] = []

    for rid, sl in data.slices():
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

    Every loader routes through this, at both read and write. Recording ids that
    disagree between arms are how a join silently drops rows — it is the single
    thing most likely to make a comparison wrong while still producing a table.
    """
    stem = Path(str(name)).name
    for ext in (".h5", ".hdf5", ".csv", ".mp4", ".avi", ".npy", ".npz", ".parquet"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
            break
    idx = stem.find("DLC_")
    if idx > 0:
        stem = stem[:idx]
    return stem


@dataclass
class RunManifest:
    """Provenance for one run. Written to ``run_manifest.json``.

    ``representation`` and ``segmenter`` are separate fields because they are
    separate slots: the central experiment holds one fixed and varies the other, so
    an artifact that records only a fused arm name ("pca-HDBSCAN") cannot be grouped
    along either axis. Every saved label file in this project predates that and is
    attributable only by directory name.
    """

    representation: str
    segmenter: str
    segmenter_version: str
    config: dict[str, Any]
    config_hash: str
    repr_hash: str
    dataset: str
    fps: float
    git_sha: str
    git_dirty: bool
    seed: int
    device: str
    wall_clock_s: float
    n_recordings: int | None = None
    n_frames: int | None = None
    schema_version: str = SCHEMA_VERSION
    library_versions: dict[str, str] = field(default_factory=dict)
    # Resource accounting, so the next request is calibrated rather than doubled on
    # faith. Filled in post-hoc from `sacct -o MaxRSS` where available.
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
        return cls(**_migrate_manifest(json.loads(Path(path).read_text())))


def _migrate_manifest(blob: dict) -> dict:
    """Accept manifests written before the two-slot schema.

    ExBias wrote ``method_name``/``method_version`` and no representation field;
    the first cut of this module wrote ``model``/``model_version``/``repr_name``.
    Both are real files on disk, so both are read rather than orphaned. Unknown
    keys are dropped instead of raising — an old manifest carrying an extra
    diagnostic is still valid provenance.
    """
    blob = dict(blob)
    if "segmenter" not in blob:
        blob["segmenter"] = blob.pop("model", None) or blob.pop("method_name", "unknown")
    if "representation" not in blob:
        blob["representation"] = blob.pop("repr_name", "") or "identity"
    if "segmenter_version" not in blob:
        blob["segmenter_version"] = (
            blob.pop("model_version", None) or blob.pop("method_version", "unknown")
        )
    blob.pop("model", None)
    blob.pop("model_version", None)
    blob.pop("repr_name", None)
    blob.pop("method_name", None)
    blob.pop("method_version", None)

    blob.setdefault("config", blob.pop("parameters", {}) or {})
    blob.setdefault("config_hash", "")
    blob.setdefault("repr_hash", "")
    blob.setdefault("dataset", "")
    blob.setdefault("fps", 30.0)
    blob.setdefault("git_sha", "unknown")
    blob.setdefault("git_dirty", True)
    blob.setdefault("seed", 0)
    blob.setdefault("device", "unknown")
    blob.setdefault("wall_clock_s", 0.0)
    blob.setdefault("n_recordings", blob.pop("n_recordings", None))
    if "n_frames" not in blob:
        blob["n_frames"] = blob.pop("n_frames_total", None)

    known = set(RunManifest.__dataclass_fields__)
    return {k: v for k, v in blob.items() if k in known}


def write_run(
    run_dir: Path,
    bouts: pd.DataFrame,
    manifest: RunManifest,
) -> tuple[Path, Path]:
    """Write the VUS-1 pair. Returns ``(manifest_path, bouts_path)``.

    The manifest is written *last*, so a run interrupted mid-write leaves a
    directory without a manifest rather than a manifest that claims a complete
    run. The comparison table keys on the manifest, so a partial run is skipped
    instead of silently contributing truncated bouts.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    missing = [c for c in BOUT_COLUMNS if c not in bouts.columns]
    if missing:
        raise ValueError(f"bouts is missing required columns {missing}")

    bouts_path = run_dir / BOUTS_NAME
    bouts.to_parquet(bouts_path, index=False)

    manifest_path = run_dir / MANIFEST_NAME
    manifest.write(manifest_path)
    return manifest_path, bouts_path


def read_run(run_dir: Path) -> tuple[RunManifest, pd.DataFrame]:
    """Read a VUS-1 run directory written by this module or by ExBias."""
    run_dir = Path(run_dir)
    manifest = RunManifest.read(run_dir / MANIFEST_NAME)
    return manifest, read_bouts(run_dir / BOUTS_NAME)


def read_bouts(path: Path) -> pd.DataFrame:
    """Read ``bouts.parquet``, accepting ExBias's column names.

    ExBias shipped ``state_id`` plus derived ``duration_frames``/``duration_s``
    columns. Renaming on read keeps its two existing runs comparable without
    rewriting them, and the derived columns are dropped because the harness
    recomputes durations identically for every arm.
    """
    df = pd.read_parquet(path)
    if "state" not in df.columns and "state_id" in df.columns:
        df = df.rename(columns={"state_id": "state"})
    missing = [c for c in BOUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns {missing}")
    return df[BOUT_COLUMNS]


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
