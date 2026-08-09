import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from representation import cluster as cluster_mod  # noqa: E402
from representation import gpu  # noqa: E402
from representation.cluster import GPUFallbackWarning, cluster  # noqa: E402


def _two_blobs(n=120, seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(size=(n, 3)),
                      rng.normal(size=(n, 3)) + 10.0])


@pytest.fixture
def gpu_available(monkeypatch):
    """Pretend the capability probe passed, so only fit-time behaviour is under
    test."""
    monkeypatch.setattr(gpu, "resolve", lambda use_gpu: True)


@pytest.fixture
def gpu_fails_at_fit(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("cuml exploded")

    monkeypatch.setattr(cluster_mod, "_fit_gpu", boom)


def test_explicit_gpu_raises_when_the_fit_fails(gpu_available, gpu_fails_at_fit):
    # The whole point of "--gpu on": a job that asked for a GPU must not
    # quietly spend its allocation on CPU. resolve() already guarantees this
    # for an unusable backend; this covers the backend that fails mid-fit.
    with pytest.raises(RuntimeError, match="cuml exploded"):
        cluster(_two_blobs(), min_cluster_size=5, use_gpu="on")


def test_explicit_gpu_error_names_the_original_exception(gpu_available,
                                                         gpu_fails_at_fit):
    # The 18h silent-CPU incident was undiagnosable because the cuml error was
    # discarded; keeping it is the fix.
    with pytest.raises(RuntimeError) as excinfo:
        cluster(_two_blobs(), min_cluster_size=5, use_gpu=True)
    assert "--gpu on" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, RuntimeError)


def test_auto_falls_back_to_cpu_but_says_so(gpu_available, gpu_fails_at_fit):
    # "auto" is best-effort, so the fallback stays -- but silence is what let a
    # CPU run masquerade as a GPU run for 18 hours.
    with pytest.warns(GPUFallbackWarning, match="cuml exploded"):
        labels, probs, backend = cluster(_two_blobs(), min_cluster_size=5,
                                         use_gpu="auto", return_backend=True)
    assert backend == "cpu"
    assert labels.shape[0] == 240


def test_cpu_path_is_untouched_by_the_fallback_logic():
    labels, probs, backend = cluster(_two_blobs(), min_cluster_size=5,
                                     use_gpu="off", return_backend=True)
    assert backend == "cpu"
    assert np.unique(labels[labels >= 0]).size == 2


def test_successful_gpu_fit_reports_the_gpu_backend(gpu_available, monkeypatch):
    def fake_fit(fit_data, all_data, *args, **kwargs):
        n = all_data.shape[0]
        return np.zeros(n, dtype=int), np.ones(n)

    monkeypatch.setattr(cluster_mod, "_fit_gpu", fake_fit)
    labels, probs, backend = cluster(_two_blobs(), min_cluster_size=5,
                                     use_gpu="on", return_backend=True)
    assert backend == "gpu"


def test_explicitly_requested_matches_resolves_on_spellings():
    # One source of truth: anything resolve() treats as a demand for GPU must
    # also make a fit-time failure fatal.
    for value in (True, "on", "yes", "true"):
        assert gpu.explicitly_requested(value) is True
    for value in (False, "off", "no", "false", None, "auto"):
        assert gpu.explicitly_requested(value) is False
