"""Model registry — the mechanism that makes models plugins selected by config.

A model is named in a YAML file under ``configs/models/``; nothing else in the
codebase branches on which arm is running. Adding an arm is registering a class,
not editing the harness.
"""

from __future__ import annotations

import importlib
from typing import Callable

from .base import Model

_REGISTRY: dict[str, Callable[[], type]] = {}

# Lazily imported so that a missing optional dependency (ssm, pyhsmm, deeptime,
# keypoint-moseq) breaks only the arm that needs it, not `vieb submit --help`.
# The value is the module path; the class is resolved on first use.
_LAZY: dict[str, tuple[str, str]] = {
    "moseq": ("vieb.models.moseq", "MoSeqModel"),
    "ticc": ("vieb.models.ticc", "TICCModel"),
    "vieb_v1": ("vieb.models.vieb_v1", "ViebV1Model"),
    "hdbscan": ("vieb.models.hdbscan_arm", "HDBSCANModel"),
    "koopman": ("vieb.models.koopman_arm", "KoopmanModel"),
    "exbias": ("vieb.models.exbias", "ExBiasModel"),
    "flow_field": ("vieb.models.flow_field", "FlowFieldModel"),
    "hsmm": ("vieb.models.hsmm", "HSMMModel"),
    "ulam_msm": ("vieb.models.ulam_msm", "UlamMSMModel"),
}


def register(name: str):
    """Decorator registering a model class under ``name``."""

    def _inner(cls: type) -> type:
        if name in _REGISTRY:
            raise ValueError(f"model {name!r} is already registered")
        _REGISTRY[name] = lambda: cls
        return cls

    return _inner


def get(name: str) -> type:
    """Resolve a model name to its class, importing the module if needed."""
    if name in _REGISTRY:
        return _REGISTRY[name]()
    if name not in _LAZY:
        raise KeyError(f"unknown model {name!r}; known: {', '.join(sorted(available()))}")
    module_path, cls_name = _LAZY[name]
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise ImportError(
            f"model {name!r} needs an optional dependency that is not installed "
            f"in this environment ({exc}). Install it on the LOGIN NODE — compute "
            f"nodes have no outbound internet and pip will fail there in a way "
            f"that looks like a code error."
        ) from exc
    cls = getattr(module, cls_name)
    _REGISTRY[name] = lambda: cls
    return cls


def build(name: str, cfg: dict | None = None) -> Model:
    """Instantiate a model by name."""
    return get(name)(**(cfg or {}))


def available() -> list[str]:
    return sorted(set(_REGISTRY) | set(_LAZY))
