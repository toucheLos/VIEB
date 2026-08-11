"""Arms whose labels are produced outside this codebase and read back in.

MoSeq and ExBias are both real, scored arms and neither is refit here.

For **MoSeq** that is a correctness requirement, not a convenience. Its AR-HMM is
Gibbs-sampled, so a refit would not reproduce the saved syllables at any
tolerance the verification gate could accept — and MoSeq is the *reference* arm,
the one every VIEB arm is judged against, so it has to be the same syllables
decision #65 scored. Refitting is a separate, explicit operation.

For **ExBias** it is because the method inverts the usual order: it segments
first (at discontinuities in acceleration) and clusters the segments afterwards,
so it never consumes a per-frame representation at all. Wrapping its 768-line
standalone script behind ``fit(X, ...)`` would be a fiction. Its result is read
from the VUS-1 it already writes.

Both adapters key on the normalized recording id, so a label file named with
DLC's suffix joins the same dataset a ``.h5`` did.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.dataset import UNASSIGNED, PoseDataset
from ..io.vus1 import normalize_recording_id
from ..registry import SEGMENTERS
from .base import Segmentation, make_segmentation


class _ExternalLabels:
    """Shared machinery: map ``{recording_id: labels}`` onto a dataset's frames.

    A recording present in the dataset but absent from the external run is left
    entirely ``-1`` and counted, rather than silently shifting later recordings —
    which is the failure mode that makes a positional index unrecoverable.
    """

    name = "external"
    version = "1.0.0"

    def __init__(self, source: str | Path | None = None, strict: bool = True):
        self.source = str(source) if source is not None else None
        self.strict = bool(strict)
        self._by_id: dict[str, np.ndarray] = {}
        self._labels: np.ndarray | None = None
        self._report: dict = {}

    def get_params(self) -> dict:
        return {"source": self.source, "strict": self.strict}

    def _load_by_id(self) -> dict[str, np.ndarray]:  # pragma: no cover - subclass
        raise NotImplementedError

    def fit(self, X: np.ndarray, data: PoseDataset, *, seed: int = 0) -> None:
        """Read the saved labels. ``X`` is unused — these arms were not fit here."""
        self._by_id = self._load_by_id()
        self._labels = self._assemble(data)

    def _assemble(self, data: PoseDataset) -> np.ndarray:
        out = np.full(data.n_frames, UNASSIGNED, dtype=np.int32)
        missing, length_mismatch = [], []

        for rid, sl in data.slices():
            labels = self._by_id.get(rid)
            if labels is None:
                missing.append(rid)
                continue
            n = sl.stop - sl.start
            if labels.shape[0] != n:
                length_mismatch.append(f"{rid}: external has {labels.shape[0]}, dataset has {n}")
                continue
            out[sl] = labels

        if length_mismatch and self.strict:
            raise ValueError(
                f"{len(length_mismatch)} recordings disagree in length with the "
                f"external labels in {self.source!r}. A frame offset here attributes "
                f"each recording's behavior to a neighbouring one.\n  "
                + "\n  ".join(length_mismatch[:10])
            )
        if missing and self.strict and len(missing) == data.n_recordings:
            raise ValueError(
                f"none of the {data.n_recordings} recordings were found in "
                f"{self.source!r}; first looked for {data.recording_ids[0]!r}. "
                f"This is normalization drift, not missing data."
            )

        self._report = {
            "n_external_recordings": len(self._by_id),
            "n_matched": data.n_recordings - len(missing) - len(length_mismatch),
            "n_missing": len(missing),
            "missing": missing[:20],
            "n_length_mismatch": len(length_mismatch),
            "length_mismatch": length_mismatch[:20],
        }
        return out

    def predict(self, X: np.ndarray, data: PoseDataset) -> Segmentation:
        if self._labels is None:
            raise RuntimeError("fit() must be called before predict()")
        return make_segmentation(self._labels, data, report=self._report,
                                 source=self.source)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump({"name": self.name, "version": self.version,
                         "params": self.get_params(), "labels": self._labels,
                         "report": self._report}, fh)

    @classmethod
    def load(cls, path: Path):
        with Path(path).open("rb") as fh:
            blob = pickle.load(fh)
        obj = cls(**blob.get("params", {}))
        obj._labels = blob.get("labels")
        obj._report = blob.get("report", {})
        return obj


@SEGMENTERS.register("moseq")
class MoSeqSegmenter(_ExternalLabels):
    """Keypoint-MoSeq AR-HMM syllables, read from ``apply_model``'s output.

    ``source`` is the ``results/`` directory of a fitted model, e.g.
    ``~/moseq/luna_demo/2026_07_26-19_54_24/results``. That run wrote one csv per
    recording and no ``results.h5``, so the csv path is the supported one.

    This is the reference arm — 48 syllables, retrieval effect 0.361, the number
    every VIEB arm is compared against.
    """

    name = "moseq"
    version = "1.0.0"

    SYLLABLE_COLUMNS = ("syllable", "syllables", "syllables_reindexed", "state")

    def _load_by_id(self) -> dict[str, np.ndarray]:
        if not self.source:
            raise ValueError(
                "moseq needs `source`: the results/ directory of a fitted model"
            )
        root = Path(self.source).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"moseq results directory not found: {root}")

        by_id: dict[str, np.ndarray] = {}
        for path in sorted(root.glob("*.csv")):
            df = pd.read_csv(path)
            col = next((c for c in self.SYLLABLE_COLUMNS if c in df.columns), None)
            if col is None:
                lowered = {str(c).lower(): c for c in df.columns}
                col = next(
                    (lowered[c] for c in self.SYLLABLE_COLUMNS if c in lowered), None
                )
            if col is None:
                raise ValueError(
                    f"{path} has no syllable column; saw {list(df.columns)[:6]}"
                )
            by_id[normalize_recording_id(path.name)] = (
                df[col].to_numpy().astype(np.int32)
            )
        if not by_id:
            raise FileNotFoundError(f"no result csvs under {root}")
        return by_id


@SEGMENTERS.register("exbias")
class ExBiasSegmenter(_ExternalLabels):
    """ExBias segments, read from the VUS-1 run it already writes.

    ``source`` is a run directory such as ``~/exbias/runs/exbias_002``, holding
    ``bouts.parquet`` and ``run_manifest.json``.

    Both of its runs produced ``n_states: 0`` with ``noise_frac: 1.0`` — every
    micro-centroid was labelled noise by HDBSCAN. That is preserved, not
    repaired: a measured null is part of the result, and the contract admits a
    segmentation with zero states rather than raising on one.
    """

    name = "exbias"
    version = "1.0-axiomatic"

    def _load_by_id(self) -> dict[str, np.ndarray]:
        if not self.source:
            raise ValueError("exbias needs `source`: a run directory with bouts.parquet")
        root = Path(self.source).expanduser()
        bouts_path = root / "bouts.parquet"
        if not bouts_path.exists():
            raise FileNotFoundError(f"no bouts.parquet under {root}")

        from ..io.vus1 import read_bouts

        bouts = read_bouts(bouts_path)
        manifest = root / "run_manifest.json"
        if manifest.exists():
            self._manifest = json.loads(manifest.read_text())

        by_id: dict[str, np.ndarray] = {}
        for rid, grp in bouts.groupby("recording_id"):
            rid = normalize_recording_id(str(rid))
            n = int(grp["end_frame"].max())
            labels = np.full(n, UNASSIGNED, dtype=np.int32)
            for _, row in grp.iterrows():
                labels[int(row["start_frame"]):int(row["end_frame"])] = int(row["state"])
            by_id[rid] = labels
        return by_id

    def _assemble(self, data: PoseDataset) -> np.ndarray:
        # ExBias's bouts.parquet records only assigned spans, and its two runs
        # assigned nothing, so `end_frame.max()` under-runs the true recording
        # length. Length is therefore not checked against it — the dataset's own
        # frame count is authoritative and unassigned frames stay -1.
        out = np.full(data.n_frames, UNASSIGNED, dtype=np.int32)
        matched = 0
        for rid, sl in data.slices():
            labels = self._by_id.get(rid)
            if labels is None:
                continue
            matched += 1
            n = min(labels.shape[0], sl.stop - sl.start)
            out[sl.start:sl.start + n] = labels[:n]

        if matched == 0 and self.strict:
            raise ValueError(
                f"none of the {data.n_recordings} recordings were found in "
                f"{self.source!r}. This is normalization drift, not missing data."
            )
        self._report = {
            "n_external_recordings": len(self._by_id),
            "n_matched": matched,
            "n_missing": data.n_recordings - matched,
        }
        return out
