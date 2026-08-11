"""Where runs live.

§1e of the plan specifies ``$SCRATCH/vieb-runs/``. **This cluster has no scratch
filesystem** — ``$SCRATCH`` is unset, ``/scratch/$USER`` does not exist, and the
existing sbatch scripts write ``%x-%j.out`` relative to the submit directory,
which is exactly how 36 SLURM logs ended up committed next to the sbatch files.

Home is the large filesystem here (630T, ~500T free), so the default run store is
``~/vieb-runs``. The resolution order below keeps the plan's intent — runs never
live in the repo — while staying portable to a machine that does have scratch.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ENV_VAR = "VIEB_RUN_STORE"


def run_store() -> Path:
    """Root of the content-addressed run store.

    Order: ``$VIEB_RUN_STORE`` → ``$SCRATCH/vieb-runs`` → ``~/vieb-runs``.
    """
    explicit = os.environ.get(ENV_VAR)
    if explicit:
        return Path(explicit).expanduser().resolve()
    scratch = os.environ.get("SCRATCH")
    if scratch and Path(scratch).is_dir():
        return Path(scratch).expanduser().resolve() / "vieb-runs"
    return Path.home() / "vieb-runs"


def config_hash(cfg: dict[str, Any]) -> str:
    """Stable ``sha256:`` digest of a config dict.

    This is load-bearing in the run path. Without it two runs of the same model
    with different kappa overwrite each other, which is how a table row ends up
    with no traceable origin.
    """
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def short(h: str, n: int = 12) -> str:
    """``sha256:abcdef...`` → ``abcdef...`` truncated, for use in a path."""
    return h.split(":", 1)[-1][:n]


def run_dir(
    dataset: str,
    repr_hash: str,
    segmenter: str,
    cfg_hash: str,
    *,
    create: bool = False,
) -> Path:
    """``<store>/<dataset>/<repr_hash>/<segmenter>/<config_hash>/``.

    Representation and segmenter are separate path components because they are
    separate slots — this is what lets the run store be listed along either axis.
    """
    path = run_store() / dataset / short(repr_hash) / segmenter / short(cfg_hash)
    if create:
        (path / "logs").mkdir(parents=True, exist_ok=True)
        (path / "figures").mkdir(parents=True, exist_ok=True)
    return path


def representation_dir(create: bool = False) -> Path:
    path = run_store() / "representations"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path
