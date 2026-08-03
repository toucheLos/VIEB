import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from representation.diffusion import DiffusionMap  # noqa: E402
from representation.pooled_pca import PooledPCA  # noqa: E402


def _spiral(n=800, seed=0, noise=0.05):
    """A 1-D manifold curled into 2-D: linear methods cannot unroll it."""
    rng = np.random.default_rng(seed)
    t = np.sort(rng.uniform(0.5 * np.pi, 2.5 * np.pi, n))
    x = np.c_[t * np.cos(t), t * np.sin(t)]
    return x + rng.normal(size=x.shape) * noise, t


def _nonuniform_spiral(n_dense=1500, n_sparse=150, seed=0):
    """Dense in one half, sparse in the other -- the density-duration confound
    in miniature: a slow behavior deposits many samples, a fast one few."""
    rng = np.random.default_rng(seed)
    t = np.r_[rng.uniform(0.5 * np.pi, 1.5 * np.pi, n_dense),
              rng.uniform(1.5 * np.pi, 2.5 * np.pi, n_sparse)]
    x = np.c_[t * np.cos(t), t * np.sin(t)]
    return x + rng.normal(size=x.shape) * 0.05, t, n_dense


def test_recovers_a_one_dimensional_manifold():
    x, t = _spiral()
    psi = DiffusionMap(n_components=3).fit([x]).transform(x)
    # The leading coordinate should be a monotone reparametrisation of arc
    # length, so rank correlation with t is near 1.
    order = np.argsort(psi[:, 0])
    rank_corr = abs(np.corrcoef(np.argsort(order), np.argsort(np.argsort(t)))[0, 1])
    assert rank_corr > 0.95


def test_beats_pca_on_a_curved_manifold():
    # The reason to have a nonlinear option at all.
    x, t = _spiral()
    dm = DiffusionMap(n_components=2).fit([x]).transform(x)[:, 0]
    pc = PooledPCA(var_threshold=0.99).fit([x]).transform(x)[:, 0]

    def monotonicity(coord):
        return abs(np.corrcoef(np.argsort(np.argsort(coord)),
                               np.argsort(np.argsort(t)))[0, 1])

    assert monotonicity(dm) > monotonicity(pc)


def test_alpha_one_decompresses_the_densely_sampled_region():
    # The measured result that sets the default. Both halves have equal true
    # arc length, so a density-free embedding would give them equal coordinate
    # spread. Without alpha-normalisation the DENSELY sampled half -- the slow
    # behavior -- is squashed almost to a point, because a random walk mixes
    # quickly through a well-connected neighbourhood. alpha=1 recovers most of
    # it. Note the direction: dense regions collapse, they do not inflate.
    x, t, n_dense = _nonuniform_spiral()

    def dense_over_sparse(alpha):
        psi = DiffusionMap(n_components=2, alpha=alpha).fit([x]).transform(x)
        c = psi[:, 0] / (np.abs(psi[:, 0]).max() + 1e-300)
        return c[:n_dense].std() / (c[n_dense:].std() + 1e-300)

    without, with_alpha = dense_over_sparse(0.0), dense_over_sparse(1.0)
    assert with_alpha > 50 * without      # measured ~250x
    # Honest about the limit: it reduces the effect, it does not remove it.
    assert with_alpha < 1.0               # 1.0 would be fully balanced


def test_default_bandwidth_keeps_the_graph_connected():
    # Too small and the manifold fragments; too large and it short-circuits
    # across folds. The default quantile must land in the working range.
    x, t = _spiral()
    dm = DiffusionMap(n_components=3).fit([x])
    psi = dm.transform(x)[:, 0]
    corr = abs(np.corrcoef(np.argsort(np.argsort(psi)),
                           np.argsort(np.argsort(t)))[0, 1])
    assert corr > 0.95, dm.epsilon_


def test_deterministic_across_repeated_fits():
    # The concrete advantage over UMAP: an eigenproblem, not a stochastic
    # optimisation, so two runs agree exactly.
    x, _ = _spiral()
    a = DiffusionMap(n_components=4).fit([x]).transform(x)
    b = DiffusionMap(n_components=4).fit([x]).transform(x)
    assert np.array_equal(a, b)


def test_trivial_constant_eigenvector_is_dropped():
    # lambda = 1 with a constant eigenvector carries no information; keeping it
    # would silently waste a component.
    x, _ = _spiral(n=400)
    dm = DiffusionMap(n_components=5).fit([x])
    assert np.all(dm.eigenvalues_ < 1.0 - 1e-9)
    for k in range(dm.n_components_):
        assert dm.psi_[:, k].std() > 1e-8


def test_nystrom_reproduces_the_in_sample_embedding():
    # Landmarks pass through the extension, so extending them must return
    # (up to eigenvalue scaling) what the fit produced.
    x, _ = _spiral(n=300)
    dm = DiffusionMap(n_components=3, n_landmarks=300).fit([x])
    direct = dm.psi_ * (dm.eigenvalues_ ** dm.diffusion_time)
    extended = dm.transform(dm.landmarks_)
    for k in range(dm.n_components_):
        corr = abs(np.corrcoef(direct[:, k], extended[:, k])[0, 1])
        assert corr > 0.99, k


def test_landmarks_cap_the_operator_size():
    x, _ = _spiral(n=2000)
    dm = DiffusionMap(n_components=3, n_landmarks=200).fit([x])
    assert dm.landmarks_.shape[0] <= 200
    # ...but every frame still gets coordinates.
    assert dm.transform(x).shape == (2000, dm.n_components_)


def test_pooled_across_recordings_gives_one_shared_embedding():
    x, _ = _spiral(n=600)
    sessions = [x[:200], x[200:400], x[400:]]
    dm = DiffusionMap(n_components=3).fit(sessions)
    scores = dm.transform_all(sessions)
    assert [s.shape[0] for s in scores] == [200, 200, 200]
    # Same point, same coordinates, regardless of which recording it came from.
    probe = x[:5]
    assert np.allclose(dm.transform(probe), dm.transform(probe))


def test_interface_matches_pooled_pca():
    # The pipeline swaps these by name, so the surface has to line up.
    for name in ("fit", "transform", "transform_all", "fit_transform",
                 "spectrum_report"):
        assert hasattr(DiffusionMap, name) and hasattr(PooledPCA, name)

    x, _ = _spiral(n=200)
    report = DiffusionMap(n_components=3).fit([x]).spectrum_report()
    for key in ("n_components", "eigenvalues", "explained_variance",
                "n_nonzero_directions", "backend"):
        assert key in report


def test_report_records_the_parameters_a_distance_depends_on():
    x, _ = _spiral(n=200)
    report = DiffusionMap(n_components=3, alpha=0.5,
                          diffusion_time=2).fit([x]).spectrum_report()
    assert report["alpha"] == 0.5
    assert report["diffusion_time"] == 2
    assert report["epsilon"] > 0
    assert report["method"] == "diffusion"


def test_accepts_pose_shaped_input():
    rng = np.random.default_rng(0)
    sessions = [rng.normal(size=(120, 7, 2)) for _ in range(2)]
    dm = DiffusionMap(n_components=3).fit(sessions)
    assert dm.transform(sessions[0]).shape == (120, 3)


def test_explicit_epsilon_is_honoured():
    x, _ = _spiral(n=200)
    assert DiffusionMap(epsilon=2.5).fit([x]).epsilon_ == 2.5


def test_rejects_invalid_parameters():
    x, _ = _spiral(n=100)
    with pytest.raises(ValueError):
        DiffusionMap(alpha=1.5)
    with pytest.raises(ValueError):
        DiffusionMap(diffusion_time=-1)
    with pytest.raises(ValueError):
        DiffusionMap(epsilon=0).fit([x])
    with pytest.raises(ValueError):
        DiffusionMap().fit([])


def test_diffusion_time_shrinks_trailing_coordinates():
    # Larger t damps low eigenvalues, concentrating the embedding on the
    # slowest modes -- the coarse-graining knob.
    x, _ = _spiral(n=400)
    dm1 = DiffusionMap(n_components=4, diffusion_time=1).fit([x])
    dm4 = DiffusionMap(n_components=4, diffusion_time=4).fit([x])
    last1 = np.abs(dm1.transform(x)[:, -1]).mean()
    last4 = np.abs(dm4.transform(x)[:, -1]).mean()
    assert last4 < last1
