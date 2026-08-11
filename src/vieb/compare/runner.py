"""The harness: compose one representation with one segmenter, emit VUS-1.

This file imports no concrete representation and no concrete segmenter. Both are
resolved by name from config, which is what makes the comparison a grid rather
than a pile of scripts — and what stops "which arm produced this?" from being
answerable only by reading a directory name.

The harness owns loading, the representation, run-length encoding into bouts,
VUS-1 writing, and every metric. **No arm gets its own scoring path**; that
duplication is why the same question has been answered differently in different
places.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..data.dataset import PoseDataset
from ..io.vus1 import RunManifest, encode_bouts, git_state, write_run
from ..paths import config_hash, run_dir
from ..registry import REPRESENTATIONS, SEGMENTERS
from ..segmenters.base import Segmentation


@dataclass
class ArmSpec:
    """One cell of the comparison grid: a representation crossed with a segmenter."""

    representation: str
    segmenter: str
    representation_params: dict[str, Any] = field(default_factory=dict)
    segmenter_params: dict[str, Any] = field(default_factory=dict)
    seed: int = 0
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.representation}_{self.segmenter}"
        if self.representation not in REPRESENTATIONS:
            raise KeyError(
                f"arm {self.name!r}: unknown representation "
                f"{self.representation!r}; known: {', '.join(REPRESENTATIONS.names())}"
            )
        if self.segmenter not in SEGMENTERS:
            raise KeyError(
                f"arm {self.name!r}: unknown segmenter {self.segmenter!r}; "
                f"known: {', '.join(SEGMENTERS.names())}"
            )


@dataclass
class ArmResult:
    spec: ArmSpec
    segmentation: Segmentation
    manifest: RunManifest
    run_dir: Path
    representation_report: dict = field(default_factory=dict)
    wall_clock_s: float = 0.0


def run_arm(
    spec: ArmSpec,
    data: PoseDataset,
    *,
    store: Path | None = None,
    write: bool = True,
    device: str = "cpu",
) -> ArmResult:
    """Run one arm end to end and emit its VUS-1 pair.

    Returns the result even when ``write`` is False, so the verification gate can
    compare frame labels against a reference without depositing a run.
    """
    started = time.time()

    representation = REPRESENTATIONS.build(spec.representation, spec.representation_params)
    X = representation.fit_transform(data)
    repr_hash = getattr(representation, "repr_hash", "") or config_hash(
        {"representation": spec.representation, **spec.representation_params}
    )

    segmenter = SEGMENTERS.build(spec.segmenter, spec.segmenter_params)
    segmenter.fit(X, data, seed=spec.seed)
    segmentation = segmenter.predict(X, data)

    elapsed = time.time() - started

    cfg = {
        "representation": spec.representation,
        "representation_params": getattr(representation, "get_params", dict)(),
        "segmenter": spec.segmenter,
        "segmenter_params": getattr(segmenter, "get_params", dict)(),
        "seed": spec.seed,
    }
    cfg_hash = config_hash(cfg)
    sha, dirty = git_state()

    manifest = RunManifest(
        representation=spec.representation,
        segmenter=spec.segmenter,
        segmenter_version=getattr(segmenter, "version", "unknown"),
        config=cfg,
        config_hash=cfg_hash,
        repr_hash=repr_hash,
        dataset=data.dataset,
        fps=data.fps,
        git_sha=sha,
        git_dirty=dirty,
        seed=spec.seed,
        device=device,
        wall_clock_s=round(elapsed, 3),
        n_recordings=data.n_recordings,
        n_frames=data.n_frames,
        n_states=segmentation.n_states,
        unassigned_frac=round(segmentation.unassigned_frac, 6),
        library_versions=_library_versions(),
    )

    target = (
        Path(store) / spec.name
        if store is not None
        else run_dir(data.dataset or "unnamed", repr_hash, spec.segmenter, cfg_hash)
    )
    if write:
        bouts = encode_bouts(segmentation.frame_labels, data)
        write_run(target, bouts, manifest)

    return ArmResult(
        spec=spec,
        segmentation=segmentation,
        manifest=manifest,
        run_dir=target,
        representation_report=dict(getattr(representation, "report_", {})),
        wall_clock_s=elapsed,
    )


def run_grid(
    specs: list[ArmSpec],
    data: PoseDataset,
    *,
    store: Path | None = None,
    write: bool = True,
    device: str = "cpu",
    on_error: str = "raise",
) -> tuple[list[ArmResult], list[tuple[str, str]]]:
    """Run every arm against one dataset.

    ``on_error="collect"`` lets a grid finish when one arm's optional dependency
    is missing — the point of a bakeoff is the arms that did run, and a missing
    ``keypoint-moseq`` should not cost the other six.
    """
    if on_error not in ("raise", "collect"):
        raise ValueError(f"on_error must be 'raise' or 'collect', got {on_error!r}")

    results, failures = [], []
    for spec in specs:
        try:
            results.append(
                run_arm(spec, data, store=store, write=write, device=device)
            )
        except Exception as exc:
            if on_error == "raise":
                raise
            failures.append((spec.name, f"{type(exc).__name__}: {exc}"))
    return results, failures


def specs_from_config(cfg: dict) -> list[ArmSpec]:
    """Build the grid from a config dict.

    Either an explicit ``arms`` list, or ``representations`` x ``segmenters``
    crossed — the cross is the point, since holding one slot fixed and varying
    the other is the experiment.
    """
    seed = int(cfg.get("seed", 0))
    repr_params = cfg.get("representation_params", {}) or {}
    seg_params = cfg.get("segmenter_params", {}) or {}

    if cfg.get("arms"):
        return [
            ArmSpec(
                representation=a["representation"],
                segmenter=a["segmenter"],
                representation_params={
                    **repr_params.get(a["representation"], {}),
                    **a.get("representation_params", {}),
                },
                segmenter_params={
                    **seg_params.get(a["segmenter"], {}),
                    **a.get("segmenter_params", {}),
                },
                seed=int(a.get("seed", seed)),
                name=a.get("name", ""),
            )
            for a in cfg["arms"]
        ]

    reps = cfg.get("representations") or []
    segs = cfg.get("segmenters") or []
    if not reps or not segs:
        raise ValueError(
            "config must define either 'arms', or both 'representations' and "
            "'segmenters' to cross"
        )
    return [
        ArmSpec(
            representation=r,
            segmenter=s,
            representation_params=dict(repr_params.get(r, {})),
            segmenter_params=dict(seg_params.get(s, {})),
            seed=seed,
        )
        for r in reps
        for s in segs
    ]


def compare_labels(a: np.ndarray, b: np.ndarray) -> dict:
    """Compare two label vectors the way the verification gate needs.

    Reports exact equality *and* adjusted Rand, because the two answer different
    questions. State ids are only meaningful within one run — v2's own note
    records that the GPU and CPU HDBSCAN backends produce the same partition
    (ARI 1.0) with permuted integer labels — so an arm that is deterministic
    should match exactly, and one that is not is judged on the partition.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return {
            "comparable": False,
            "reason": f"length mismatch: {a.shape[0]} vs {b.shape[0]}",
        }

    exact = bool(np.array_equal(a, b))
    out = {
        "comparable": True,
        "exact": exact,
        "n_frames": int(a.size),
        "n_differing": int((a != b).sum()),
        "unassigned_a": float((a < 0).mean()),
        "unassigned_b": float((b < 0).mean()),
    }
    try:
        from sklearn.metrics import adjusted_rand_score

        both = (a >= 0) & (b >= 0)
        out["ari_assigned_only"] = (
            float(adjusted_rand_score(a[both], b[both])) if both.any() else float("nan")
        )
        out["ari_all_frames"] = float(adjusted_rand_score(a, b))
    except ImportError:  # pragma: no cover - sklearn is a hard dependency
        out["ari_all_frames"] = float("nan")
    return out


def _library_versions() -> dict[str, str]:
    """Versions of the libraries that can change a result.

    Recorded because a run is only reproducible from its manifest if the things
    that compute the numbers are pinned in it.
    """
    versions = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "hdbscan", "umap"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            continue
    return versions
