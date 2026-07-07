"""Tests for ml/representations/ (statistics-methods branch): shape_space,
delay_embedding, topological. Covers both Luna-like (8-keypoint mouse) and
Spence-like (5-keypoint rat) layouts to confirm the representations are
keypoint-layout-agnostic, per docs/DECISIONS.md #2."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

LUNA_BODYPARTS = ["nose", "left_ear", "right_ear", "tail_base", "center",
                  "left_hip", "right_hip", "tail_tip"]
SPENCE_BODYPARTS = ["nose", "center", "left_hip", "right_hip", "tail_base"]


def _make_pose(n_frames: int, n_keypoints: int, seed: int) -> np.ndarray:
    """Synthetic (T, K, 2) pose: a slow oscillation + random-walk drift, not
    uniform noise, so AMI/FNN/TDA have real temporal structure to find."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 20, n_frames)
    base = rng.random((n_keypoints, 2)) * 10
    pose = base[None, :, :] + 0.5 * np.sin(t)[:, None, None]
    pose = pose + rng.normal(0, 0.2, (n_frames, n_keypoints, 2))
    pose = pose + np.cumsum(rng.normal(0, 0.1, (n_frames, 1, 2)), axis=0)
    return pose


@pytest.fixture(params=[("luna", LUNA_BODYPARTS), ("spence", SPENCE_BODYPARTS)])
def layout(request):
    return request.param


# ---------------------------------------------------------------------------
# shape_space (Part B)
# ---------------------------------------------------------------------------

def test_shape_space_output_shape_and_finite(layout):
    _, bodyparts = layout
    from ml.representations.shape_space import ShapeSpaceExtractor

    pose = _make_pose(200, len(bodyparts), seed=1)
    ext = ShapeSpaceExtractor(fps=30.0)
    ext.fit([pose])
    features, names = ext.transform(pose)

    assert features.shape[0] == 200
    assert features.shape[1] == len(names)
    assert np.isfinite(features).all()
    meta = ext.get_meta()
    assert meta["mode"] == "shape_space"
    assert meta["n_features"] == features.shape[1]


def test_shape_space_invariant_to_scale_and_rotation():
    """Removing translation/scale/rotation should make otherwise-identical
    postures (up to a similarity transform) collapse to (near-)identical
    shape coordinates, while the *default* extractor's raw pairwise
    distances remain scale-dependent — the explicit contrast this
    representation is supposed to fix."""
    from ml.representations.shape_space import ShapeSpaceExtractor
    from scipy.spatial.distance import pdist

    K = 8
    rng = np.random.default_rng(1)
    base = rng.random((K, 2)) * 10
    theta = np.pi / 2
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

    frames = np.stack([
        base,                                    # original
        base * 2.0,                               # scaled 2x
        base @ R.T + np.array([50.0, 50.0]),       # rotated + translated
        (base * 2.0) @ R.T + np.array([-30.0, 10.0]),  # scaled + rotated + translated
    ])
    # Repeat each frame a few times so Savitzky-Golay smoothing has enough points.
    pose_seq = np.repeat(frames, 6, axis=0)

    ext = ShapeSpaceExtractor(fps=30.0, smooth_window=3)
    ext.fit([pose_seq])
    features, _ = ext.transform(pose_seq)
    shape_only = features[:, :K * 2]

    idxs = [0, 6, 12, 18]  # first frame of each of the 4 configurations
    max_pairwise_diff = np.max(np.abs(shape_only[idxs][:, None, :] - shape_only[idxs][None, :, :]))
    assert max_pairwise_diff < 1e-6, "shape_space must be invariant to scale/rotation/translation"

    # Contrast: the default extractor's raw pairwise distances are NOT invariant.
    d_orig = pdist(frames[0])
    d_scaled = pdist(frames[1])
    ratio = (d_scaled / d_orig).mean()
    assert abs(ratio - 2.0) < 1e-6, "raw pairwise distances should scale with the posture (not invariant)"


def test_shape_space_no_calibration_needed():
    from ml.representations.shape_space import ShapeSpaceExtractor
    ext = ShapeSpaceExtractor()
    result = ext.fit([])  # empty sample — must be a no-op, not an error
    assert result is ext


# ---------------------------------------------------------------------------
# delay_embedding (Part C)
# ---------------------------------------------------------------------------

def test_delay_embedding_output_shape_and_finite(layout):
    _, bodyparts = layout
    from ml.representations.delay_embedding import DelayEmbeddingExtractor

    pose = _make_pose(300, len(bodyparts), seed=2)
    ext = DelayEmbeddingExtractor(fps=30.0)
    ext.fit([pose])
    features, names = ext.transform(pose)

    assert features.shape[0] == 300
    assert features.shape[1] == len(names)
    assert np.isfinite(features).all()
    meta = ext.get_meta()
    assert meta["mode"] == "delay_embedding"
    assert set(meta["params"].keys()) == {"centroid_speed", "elongation"}


def test_delay_embedding_requires_fit_before_transform():
    from ml.representations.delay_embedding import DelayEmbeddingExtractor
    pose = _make_pose(100, 8, seed=3)
    ext = DelayEmbeddingExtractor(fps=30.0)
    with pytest.raises(RuntimeError):
        ext.transform(pose)


def test_select_tau_and_dim_on_known_signal():
    """A noisy sine wave has real periodic structure — tau/d selection
    should land in a sane, non-degenerate range."""
    from ml.representations.delay_embedding import select_tau_mutual_information, select_embedding_dim_fnn

    rng = np.random.default_rng(4)
    t = np.linspace(0, 40, 2000)
    x = np.sin(t) + rng.normal(0, 0.05, len(t))

    tau = select_tau_mutual_information(x, max_tau=50)
    assert 1 <= tau <= 50

    d = select_embedding_dim_fnn(x, tau, max_dim=10)
    assert 1 <= d <= 10


# ---------------------------------------------------------------------------
# topological (Part D)
# ---------------------------------------------------------------------------

def test_topological_output_shape_and_finite(layout):
    pytest.importorskip("ripser")
    _, bodyparts = layout
    from ml.representations.topological import TopologicalExtractor

    pose = _make_pose(150, len(bodyparts), seed=5)
    ext = TopologicalExtractor(fps=30.0, window_sec=0.5, stride_frames=5)
    ext.fit([pose])
    features, names = ext.transform(pose)

    assert features.shape == (150, 5)
    assert features.shape[1] == len(names)
    assert np.isfinite(features).all()
    meta = ext.get_meta()
    assert meta["mode"] == "topological"


def test_topological_runtime_bounded():
    """Benchmark-style sanity check: a modest synthetic video should not take
    an unreasonable amount of time (the task explicitly calls for measuring
    this before using TDA dataset-wide — see benchmark_feature_modes.py for
    the real per-project measurement)."""
    pytest.importorskip("ripser")
    import time
    from ml.representations.topological import TopologicalExtractor

    pose = _make_pose(500, 8, seed=6)
    ext = TopologicalExtractor(fps=30.0)
    t0 = time.perf_counter()
    ext.transform(pose)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, f"topological extraction took {elapsed:.1f}s on 500 frames — investigate before dataset-wide use"


def test_topological_missing_dependency_raises(monkeypatch):
    import ml.representations.topological as topo_mod
    monkeypatch.setattr(topo_mod, "_RIPSER_AVAILABLE", False)
    with pytest.raises(ImportError, match="topology"):
        topo_mod.TopologicalExtractor()


# ---------------------------------------------------------------------------
# Factory (ml/representations/__init__.py)
# ---------------------------------------------------------------------------

def test_get_representation_factory():
    from ml.representations import get_representation, AVAILABLE_MODES

    assert AVAILABLE_MODES == ("shape_space", "delay_embedding", "topological")
    for mode in AVAILABLE_MODES:
        if mode == "topological":
            pytest.importorskip("ripser")
        rep = get_representation(mode, fps=30.0)
        assert rep is not None


def test_get_representation_unknown_mode_raises():
    from ml.representations import get_representation
    with pytest.raises(ValueError):
        get_representation("not_a_real_mode")
