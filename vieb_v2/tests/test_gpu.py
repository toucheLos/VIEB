import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from representation import gpu  # noqa: E402


@pytest.fixture(autouse=True)
def clean_cache():
    gpu.reset_cache()
    yield
    gpu.reset_cache()


def test_force_cpu_disables_every_gpu_path(monkeypatch):
    monkeypatch.setenv("VIEB_FORCE_CPU", "1")
    gpu.reset_cache()
    ok, reason = gpu.hdbscan_backend()
    assert ok is False
    assert "VIEB_FORCE_CPU" in reason
    assert gpu.resolve("auto") is False


def test_resolve_off_is_always_false():
    assert gpu.resolve("off") is False
    assert gpu.resolve(False) is False


def test_resolve_on_raises_when_unavailable(monkeypatch):
    # Failing in the first second beats discovering it an hour into a job.
    monkeypatch.setenv("VIEB_FORCE_CPU", "1")
    gpu.reset_cache()
    with pytest.raises(RuntimeError, match="unusable"):
        gpu.resolve("on")


def test_resolve_rejects_nonsense():
    with pytest.raises(ValueError):
        gpu.resolve("maybe")


def test_probe_result_is_cached(monkeypatch):
    calls = []

    def counting_probe():
        calls.append(1)
        return False, "stub"

    monkeypatch.setattr(gpu, "_probe_hdbscan", counting_probe)
    gpu.reset_cache()
    gpu.hdbscan_backend()
    gpu.hdbscan_backend()
    gpu.hdbscan_backend()
    # A probe fits a real model; doing it per call would be wasteful.
    assert len(calls) == 1


def test_report_has_the_fields_doctor_prints():
    report = gpu.report()
    for key in ("device", "n_devices", "hdbscan_gpu", "hdbscan_gpu_reason",
                "cupy_linalg", "cupy_linalg_reason", "forced_cpu",
                "loader_hint"):
        assert key in report


def test_cupy_probe_is_isolated_from_import_order():
    """cupy's linalg is probed in a subprocess deliberately.

    Importing cuml side-loads CUDA shared libraries, after which cupy's linalg
    starts working in that process even when it fails standalone. Probing
    in-process would therefore report whatever the import order produced. This
    asserts the probe does not consult an already-imported cupy.
    """
    import inspect

    source = inspect.getsource(gpu._probe_cupy)
    assert "subprocess" in source
