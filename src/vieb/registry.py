"""Name → class lookup, so the comparison runner instantiates by string from config.

Two registries, because the central experiment has always been holding one slot
fixed and varying the other. Representation and segmenter are named independently
and composed by the runner, which imports no concrete class.

Resolution is lazy. A missing optional dependency (``keypoint-moseq``, ``cuml``,
``deeptime``) must break only the arm that needs it, not ``--help`` and not the six
arms that do not use it.
"""

from __future__ import annotations

import importlib
from typing import Iterator


class Registry:
    """Dict-like name → class map with deferred imports."""

    def __init__(self, kind: str, lazy: dict[str, tuple[str, str]] | None = None):
        self.kind = kind
        self._lazy: dict[str, tuple[str, str]] = dict(lazy or {})
        self._resolved: dict[str, type] = {}

    def register(self, name: str):
        """Decorator registering a class under ``name``."""

        def _inner(cls: type) -> type:
            if name in self._resolved:
                raise ValueError(f"{self.kind} {name!r} is already registered")
            self._resolved[name] = cls
            return cls

        return _inner

    def __getitem__(self, name: str) -> type:
        if name in self._resolved:
            return self._resolved[name]
        if name not in self._lazy:
            raise KeyError(
                f"unknown {self.kind} {name!r}; known: {', '.join(self.names())}"
            )
        module_path, cls_name = self._lazy[name]
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"{self.kind} {name!r} needs an optional dependency that is not "
                f"installed in this environment ({exc}). Install it on the LOGIN "
                f"NODE — compute nodes have no outbound internet and pip will fail "
                f"there in a way that looks like a code error."
            ) from exc
        cls = getattr(module, cls_name)
        self._resolved[name] = cls
        return cls

    def build(self, name: str, params: dict | None = None):
        """Instantiate by name with keyword params from config."""
        return self[name](**(params or {}))

    def names(self) -> list[str]:
        return sorted(set(self._resolved) | set(self._lazy))

    def __contains__(self, name: object) -> bool:
        return name in self._resolved or name in self._lazy

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __len__(self) -> int:
        return len(set(self._resolved) | set(self._lazy))

    def keys(self) -> list[str]:
        return self.names()


# ``ticc`` and ``flow_field`` are deliberately absent. Neither has an
# implementation anywhere in this repo or in any sibling tree — the previous
# registry listed both, pointing at modules that were never written, so selecting
# either produced an ImportError that read like a broken environment. The
# comparison list closes at the methods that exist; see docs/DECISIONS.md.
REPRESENTATIONS = Registry(
    "representation",
    {
        "identity": ("vieb.representations.postural", "IdentityRepresentation"),
        "pca": ("vieb.representations.postural", "PCARepresentation"),
        "diffusion": ("vieb.representations.postural", "DiffusionRepresentation"),
        "engineered91": ("vieb.representations.engineered91", "Engineered91Representation"),
    },
)

SEGMENTERS = Registry(
    "segmenter",
    {
        "hdbscan": ("vieb.segmenters.hdbscan", "HDBSCANSegmenter"),
        "koopman": ("vieb.segmenters.koopman", "KoopmanSegmenter"),
        "moseq": ("vieb.segmenters.external", "MoSeqSegmenter"),
        "exbias": ("vieb.segmenters.external", "ExBiasSegmenter"),
        "vieb_v1": ("vieb.segmenters.vieb_v1", "ViebV1Segmenter"),
        "hsmm": ("vieb.segmenters.hsmm", "HSMMSegmenter"),
        "ulam": ("vieb.segmenters.ulam", "UlamSegmenter"),
    },
)
