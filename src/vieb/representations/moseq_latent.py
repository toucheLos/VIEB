"""The latent space keypoint-MoSeq actually fitted, read back from its own output.

§5 requires the HSMM to run at the "same ``latent_dim``, same K, same seed as the
MoSeq baseline so the comparison isolates the duration model". The strongest form
of that is not to refit kpms's PCA with matching settings but to **read the
trajectories kpms already wrote**: ``~/moseq/luna_demo/<model>/results/*.csv``
carries ``latent_state 0..9`` per frame for all 3,846 recordings. Then the two
arms are not merely configured alike, they are on the same numbers.

A refit would not be. kpms whitens, fits PCA on a 1e6-frame subsample, and
initializes heading and centroid from the data; reproducing that exactly is a
verification problem of its own, and any residual difference would land in the
one comparison this branch exists to make.

Locomotion
----------
The CSVs also carry ``centroid x``, ``centroid y`` and ``heading``. Those are read
and reported but **not** returned as channels, because kpms models them outside
the AR process that generates syllables — which is the representation-side reason
every arm here struggles to separate freezing from locomotion (decision #60).
Turning them into channels is ``representation-repair``'s decision, not this
branch's; ``locomotor_channels`` is where it will attach when that reports.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..data.dataset import PoseDataset
from ..ids import normalize_recording_id
from ..registry import REPRESENTATIONS
from .base import BaseRepresentation

#: Written by ``kpms.apply_model``; the count is ``latent_dim``.
LATENT_PREFIX = "latent_state "
LOCOMOTOR_COLUMNS = ("centroid x", "centroid y", "heading")


def find_results_dir(root: Path) -> Path:
    """The newest kpms model directory under ``root`` that has a ``results/``."""
    root = Path(root).expanduser()
    if (root / "results").is_dir():
        return root / "results"
    # ``root`` may already be the results directory — accept it if it holds csvs,
    # so both `--moseq <project>` and `--moseq <project>/<model>/results` work.
    if root.is_dir() and any(root.glob("*.csv")):
        return root
    candidates = sorted(p for p in root.glob("*/results") if p.is_dir())
    if not candidates:
        raise FileNotFoundError(
            f"no keypoint-MoSeq results/ directory under {root}. This "
            f"representation reads what kpms already wrote; it does not fit PCA."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def index_results(results_dir: Path) -> dict[str, Path]:
    """``normalize_recording_id`` -> csv path, so this joins like every other arm."""
    out: dict[str, Path] = {}
    for p in sorted(Path(results_dir).glob("*.csv")):
        out[normalize_recording_id(p.name)] = p
    return out


@REPRESENTATIONS.register("moseq_latent")
class MoSeqLatentRepresentation(BaseRepresentation):
    """``latent_state 0..d-1`` per frame, straight from the kpms result CSVs."""

    name = "moseq_latent"

    def __init__(self, *, source="~/moseq/luna_demo", latent_dim: int = 10,
                 locomotor_channels: bool = False):
        self.source = str(source)
        self.latent_dim = int(latent_dim)
        if locomotor_channels:
            raise NotImplementedError(
                "locomotor_channels is the deliverable of `representation-repair`, "
                "which has not reported. Enabling it here would mean inventing the "
                "channel definition this arm is supposed to be evaluated against."
            )
        self.locomotor_channels = False
        self.report_: dict = {}

    def fit_transform(self, data: PoseDataset) -> np.ndarray:
        results = find_results_dir(Path(self.source))
        index = index_results(results)

        wanted = [normalize_recording_id(r) for r in data.recording_ids]
        missing = [r for r in wanted if r not in index]
        if missing:
            overlap = 1.0 - len(missing) / max(1, len(wanted))
            raise ValueError(
                f"{len(missing)} of {len(wanted)} recordings have no kpms result csv "
                f"under {results} (overlap {overlap:.4f}). If the overlap is near "
                f"zero this is recording-id normalization drift, not missing data; "
                f"first missing: {missing[:3]}"
            )

        cols = [f"{LATENT_PREFIX}{i}" for i in range(self.latent_dim)]
        blocks, locomotor = [], {}
        for rid, sl in data.slices():
            key = normalize_recording_id(rid)
            df = pd.read_csv(index[key])
            n = sl.stop - sl.start
            if len(df) != n:
                raise ValueError(
                    f"recording {rid!r}: kpms wrote {len(df)} frames but the dataset "
                    f"has {n}. A frame offset would attribute every state to the "
                    f"wrong frame, so this is fatal rather than trimmed."
                )
            have = [c for c in cols if c in df.columns]
            if len(have) != len(cols):
                raise ValueError(
                    f"{index[key].name} has {len([c for c in df.columns if c.startswith(LATENT_PREFIX)])} "
                    f"latent columns, need {self.latent_dim}"
                )
            blocks.append(df[cols].to_numpy(dtype=np.float64))
            present = [c for c in LOCOMOTOR_COLUMNS if c in df.columns]
            if present:
                locomotor[key] = present

        self.report_ = {
            "results_dir": str(results),
            "n_recordings": len(blocks),
            "latent_dim": self.latent_dim,
            "locomotor_columns_available": sorted(
                {c for v in locomotor.values() for c in v}
            ),
            "locomotor_channels_used": [],
        }
        return self._check_output(np.concatenate(blocks, axis=0), data)

    @property
    def channel_names(self) -> list[str]:
        return [f"{LATENT_PREFIX}{i}" for i in range(self.latent_dim)]

    def get_params(self) -> dict:
        return {
            "source": self.source,
            "latent_dim": self.latent_dim,
            "locomotor_channels": self.locomotor_channels,
        }
