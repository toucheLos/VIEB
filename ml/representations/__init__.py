"""Pluggable alternative pose feature representations.

Selectable via ``compare.py --feature-mode {shape_space,delay_embedding,topological}``.
The default representation (``PoseFeatureExtractor`` in
``ml/feature_extraction.py``) is NOT wrapped here — ``compare.py`` uses it
directly and unchanged when ``--feature-mode`` is omitted or ``"default"``,
per docs/DECISIONS.md #2/#6 (frozen file, additive-only new modes).

Each representation implements the same minimal contract:

    .fit(sample_poses: list[np.ndarray]) -> self   # calibration; no-op if unneeded
    .transform(pose, confidence=None) -> (features: (T, F) ndarray, names: list[str])
    .get_meta() -> dict
    .save(path) / .load(path)                       # pickle, mirrors BehaviorPreprocessor

``fit()`` exists uniformly so ``compare.py``'s extraction loop can always
call it, even though only ``delay_embedding`` needs a real calibration pass
(picking (tau, d) from a data sample) — ``shape_space`` and ``topological``
implement it as a no-op returning ``self`` immediately, rather than forcing
every mode to fake a fitting step it doesn't need.
"""
from __future__ import annotations

AVAILABLE_MODES = ("shape_space", "delay_embedding", "topological")


def get_representation(mode: str, **kwargs):
    """Factory: construct a new (unfit) representation instance for ``mode``."""
    if mode == "shape_space":
        from .shape_space import ShapeSpaceExtractor
        return ShapeSpaceExtractor(**kwargs)
    if mode == "delay_embedding":
        from .delay_embedding import DelayEmbeddingExtractor
        return DelayEmbeddingExtractor(**kwargs)
    if mode == "topological":
        from .topological import TopologicalExtractor
        return TopologicalExtractor(**kwargs)
    raise ValueError(f"Unknown feature_mode: {mode!r}. Available: {AVAILABLE_MODES}")
