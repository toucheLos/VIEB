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
                "loader_hint", "driver_ok", "driver", "driver_cuda",
                "recommended_stack_id", "recommended_stack_label",
                "recommended_stack_packages", "stack_message"):
        assert key in report


# --------------------------------------------------- driver -> stack selection

def test_no_driver_selects_no_stack():
    assert gpu.select_gpu_stack(None) is None
    assert gpu.select_gpu_stack(()) is None


def test_driver_below_every_minimum_selects_no_stack():
    # One version short of the oldest stack: pip would install wheels the
    # driver cannot load, so the honest answer is "none".
    assert gpu.select_gpu_stack((525, 60, 12)) is None
    assert gpu.select_gpu_stack((470, 0, 0)) is None


def test_driver_exactly_at_a_minimum_qualifies():
    # min_driver is inclusive.
    assert gpu.select_gpu_stack((525, 60, 13))["id"] == "rapids-24.12-cuda12.2"
    assert gpu.select_gpu_stack((575, 51, 3))["id"] == "rapids-26.04-cuda12.9"


def test_newest_matching_stack_wins():
    # A driver satisfying both minimums gets the newer stack, not the first.
    assert gpu.select_gpu_stack((580, 0, 0))["id"] == "rapids-26.04-cuda12.9"
    # ...and one between the two minimums falls back to the older stack.
    assert gpu.select_gpu_stack((550, 0, 0))["id"] == "rapids-24.12-cuda12.2"


def test_every_stack_pins_cuml_since_that_is_what_hdbscan_needs():
    for stack in gpu.GPU_STACKS:
        assert any(p.startswith("cuml-cu12==") for p in stack["packages"])


def test_version_compare_pads_unequal_lengths():
    assert gpu._version_gte((525, 60), (525, 60, 13)) is False
    assert gpu._version_gte((525, 61), (525, 60, 13)) is True


# ------------------------------------------------------------ driver detection

class _Proc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


_SMI = (
    "Fri Aug  4 10:00:00 2026\n"
    "+---------------------------------------------------------------+\n"
    "| NVIDIA-SMI 575.57.08    Driver Version: 575.57.08"
    "    CUDA Version: 12.9  |\n"
)


def test_detect_driver_parses_version_and_name(monkeypatch):
    def fake_run(cmd, **kw):
        if "--query-gpu=name" in cmd:
            return _Proc(stdout="NVIDIA A100-SXM4-80GB\n")
        return _Proc(stdout=_SMI)

    monkeypatch.setattr(gpu.subprocess, "run", fake_run)
    info = gpu.detect_nvidia_driver()
    assert info["ok"] is True
    assert info["driver"] == "575.57.08"
    assert info["driver_tuple"] == (575, 57, 8)
    assert info["cuda"] == "12.9"
    assert info["gpu_name"] == "NVIDIA A100-SXM4-80GB"
    # This driver is what the newer stack exists for.
    assert gpu.select_gpu_stack(info["driver_tuple"])["id"] == "rapids-26.04-cuda12.9"


def test_detect_driver_handles_missing_nvidia_smi(monkeypatch):
    # The login-node case: no driver, no crash, and a message that says so.
    def fake_run(cmd, **kw):
        raise FileNotFoundError("No such file or directory: 'nvidia-smi'")

    monkeypatch.setattr(gpu.subprocess, "run", fake_run)
    info = gpu.detect_nvidia_driver()
    assert info["ok"] is False
    assert info["driver_tuple"] is None
    assert "nvidia-smi" in info["error"]
    assert "gpu partition" in gpu.stack_message(info)


def test_detect_driver_handles_nonzero_exit(monkeypatch):
    monkeypatch.setattr(gpu.subprocess, "run", lambda cmd, **kw: _Proc(
        returncode=9, stderr="couldn't communicate with the NVIDIA driver"))
    info = gpu.detect_nvidia_driver()
    assert info["ok"] is False
    assert "NVIDIA driver" in info["error"]


def test_stack_message_names_a_too_old_driver_explicitly():
    msg = gpu.stack_message({"ok": True, "driver": "470.1", "cuda": "11.4",
                             "driver_tuple": (470, 1)})
    assert "470.1" in msg
    assert "525.60.13" in msg


def test_stack_message_recommends_the_matched_stack():
    msg = gpu.stack_message({"ok": True, "driver": "575.57.08", "cuda": "12.9",
                             "driver_tuple": (575, 57, 8)})
    assert "RAPIDS 26.04" in msg


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
