import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from representation.pooled_pca import PooledPCA  # noqa: E402


def test_identical_pose_maps_to_identical_coordinates_across_recordings():
    # The reason pooling is mandatory: one global chart means equality of pose
    # implies equality of coordinates, so a state means the same thing in every
    # recording.
    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 14))
    b = rng.normal(size=(200, 14)) + 5.0  # different animal, different offset
    probe = rng.normal(size=(1, 14))

    pca = PooledPCA(var_threshold=0.99).fit([a, b])
    in_a = pca.transform(np.vstack([a, probe]))[-1]
    in_b = pca.transform(np.vstack([b, probe]))[-1]
    assert np.allclose(in_a, in_b, atol=1e-12)


def test_per_recording_fit_would_not_be_comparable():
    # Demonstrates the failure the pooled API prevents.
    rng = np.random.default_rng(1)
    a = rng.normal(size=(200, 14))
    b = rng.normal(size=(200, 14)) * 3 + 5.0
    probe = rng.normal(size=(1, 14))

    per_a = PooledPCA(var_threshold=0.99).fit([a]).transform(probe)
    per_b = PooledPCA(var_threshold=0.99).fit([b]).transform(probe)
    assert not np.allclose(per_a, per_b, atol=1e-6)


def test_accepts_pose_shaped_input():
    rng = np.random.default_rng(2)
    sessions = [rng.normal(size=(100, 7, 2)) for _ in range(3)]
    pca = PooledPCA().fit(sessions)
    scores = pca.transform_all(sessions)
    assert all(s.shape[0] == 100 for s in scores)
    assert all(s.shape[1] == pca.n_components_ for s in scores)


def test_component_count_respects_variance_threshold():
    rng = np.random.default_rng(3)
    # Three dominant directions, the rest tiny.
    latent = rng.normal(size=(2000, 3)) * np.array([10.0, 5.0, 2.0])
    basis = np.linalg.qr(rng.normal(size=(14, 14)))[0]
    data = latent @ basis[:3] + rng.normal(size=(2000, 14)) * 1e-3

    assert PooledPCA(var_threshold=0.99).fit([data]).n_components_ == 3
    assert PooledPCA(var_threshold=0.60).fit([data]).n_components_ == 1


def test_null_directions_are_never_retained():
    # A rank-deficient input must not yield components along numerically-null
    # directions, whatever the variance threshold asks for.
    rng = np.random.default_rng(4)
    latent = rng.normal(size=(500, 4))
    basis = np.linalg.qr(rng.normal(size=(14, 14)))[0][:4]
    data = latent @ basis  # rank 4 in a 14-dim ambient space

    pca = PooledPCA(var_threshold=1.0).fit([data])
    assert pca.n_components_ == 4
    assert pca.spectrum_report()["n_nonzero_directions"] == 4


def test_spectrum_report_logs_component_count():
    rng = np.random.default_rng(5)
    pca = PooledPCA().fit([rng.normal(size=(300, 14))])
    report = pca.spectrum_report()
    assert report["n_components"] == pca.n_components_
    assert len(report["eigenvalues"]) == 14
    assert 0.0 < report["explained_variance"] <= 1.0


def test_max_components_caps_selection():
    rng = np.random.default_rng(6)
    pca = PooledPCA(var_threshold=1.0, max_components=3)
    pca.fit([rng.normal(size=(300, 14))])
    assert pca.n_components_ == 3


def test_rejects_empty_input():
    with pytest.raises(ValueError):
        PooledPCA().fit([])
