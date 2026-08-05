"""GPU capability detection.

Only one stage of the pipeline is worth putting on a GPU: HDBSCAN. After
dropping tail_tip the pose dimension is D = 14, so the pooled PCA is a 14x14
eigenproblem -- accelerating that would be pointless. Clustering, by contrast,
runs on millions of delay-embedded points and is the actual bottleneck.

Capability is probed by *running* something, not by importing something. On this
hardware cupy imports fine and reports a visible device while every linear
algebra call fails, because libcublas/libcusolver are not on the loader path. An
import-based check would report "GPU available" and then die partway through a
job -- on a cluster that costs the whole allocation. So each backend is
exercised once with a tiny problem and the answer cached.
"""

from __future__ import annotations

import glob
import os
import sys

_PROBE_CACHE = {}


def _cached(key, fn):
    if key not in _PROBE_CACHE:
        _PROBE_CACHE[key] = fn()
    return _PROBE_CACHE[key]


def reset_cache():
    """Forget probe results (used by tests)."""
    _PROBE_CACHE.clear()


def forced_cpu():
    return bool(os.environ.get("VIEB_FORCE_CPU"))


def hdbscan_backend():
    """Probe whether cuml's GPU HDBSCAN actually runs. -> (bool, reason)."""
    return _cached("hdbscan", _probe_hdbscan)


def _probe_hdbscan():
    if forced_cpu():
        return False, "VIEB_FORCE_CPU is set"
    try:
        import numpy as np
        from cuml.cluster import hdbscan as cuh
    except Exception as exc:
        return False, f"cuml unavailable ({type(exc).__name__})"

    try:
        rng = np.random.default_rng(0)
        x = np.vstack([rng.normal(size=(60, 3)),
                       rng.normal(size=(60, 3)) + 8]).astype("float32")
        cuh.HDBSCAN(min_cluster_size=5).fit_predict(x)
        return True, None
    except Exception as exc:
        return False, f"cuml HDBSCAN failed to run ({type(exc).__name__})"


def cupy_linalg():
    """Probe cupy's linear algebra, in a clean subprocess.

    The subprocess is not paranoia. Importing cuml side-loads CUDA shared
    libraries into the process, after which cupy's linalg starts working even
    though it fails on its own -- so probing in-process would report whatever
    the import order happened to produce. A diagnostic that depends on probe
    order is worse than none, so this answers the question a *fresh* process
    would get.

    The pipeline itself does not use cupy; this is reported because a broken
    cupy is a strong signal that the CUDA environment is misconfigured.
    """
    return _cached("cupy_linalg", _probe_cupy)


_CUPY_PROBE = (
    "import warnings; warnings.filterwarnings('ignore')\n"
    "import numpy as np, cupy as cp\n"
    "cp.linalg.eigh(cp.asarray(np.eye(8)))\n"
    "print('ok')\n"
)


def _probe_cupy():
    if forced_cpu():
        return False, "VIEB_FORCE_CPU is set"

    import subprocess

    try:
        proc = subprocess.run(
            [sys.executable, "-c", _CUPY_PROBE],
            capture_output=True, text=True, timeout=120,
        )
    except Exception as exc:
        return False, f"probe failed ({type(exc).__name__})"

    if proc.returncode == 0 and "ok" in proc.stdout:
        return True, None

    stderr = (proc.stderr or "").strip().splitlines()
    last = stderr[-1] if stderr else "unknown error"
    return False, last[:100]


def device_info():
    """Best-effort description of the visible GPU."""
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        if n < 1:
            return {"n_devices": 0, "device": None}
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        return {
            "n_devices": int(n),
            "device": name.decode() if isinstance(name, bytes) else str(name),
        }
    except Exception:
        return {"n_devices": 0, "device": None}


def loader_path_hint():
    """Suggest an LD_LIBRARY_PATH fix when CUDA libs are installed but hidden.

    pip's `nvidia-*` wheels drop libcublas/libcusolver inside site-packages
    where the dynamic loader will not look. That is a one-line fix, but only if
    you know to look for it -- which is the entire reason this exists.
    """
    ok, reason = cupy_linalg()
    if ok or not reason or "so." not in reason:
        return None

    roots = [os.path.join(p, "nvidia") for p in sys.path if p]
    libdirs = []
    for root in roots:
        if os.path.isdir(root):
            libdirs.extend(sorted(glob.glob(os.path.join(root, "*", "lib"))))
    if not libdirs:
        return None

    return (
        f"CUDA libraries are installed but not on the loader path "
        f"({len(libdirs)} lib dirs found under site-packages/nvidia).\n"
        f"Fix for this session:\n"
        f"    export LD_LIBRARY_PATH={':'.join(libdirs)}:$LD_LIBRARY_PATH\n"
        f"Not required by the pipeline -- HDBSCAN on GPU works without it."
    )


# Driver-gated RAPIDS stacks, ported from v1's _utils.GPU_STACKS. Bare
# `pip install cuml cupy` is wrong twice over: `cuml` on plain PyPI is an
# unrelated abandoned package, and bare `cupy` has no prebuilt wheel so pip
# falls back to a source build needing a full CUDA dev toolkit. The suffixed
# wheels from pypi.nvidia.com are prebuilt and bundle their own runtime libs.
RAPIDS_STACKS = [
    {
        "id": "rapids-26.04-cuda12.9",
        "label": "RAPIDS 26.04 / CUDA 12.9",
        "min_driver": (575, 51, 3),
        "packages": ["cuml-cu12==26.4.0", "cupy-cuda12x==14.1.1"],
    },
    {
        "id": "rapids-24.12-cuda12.2",
        "label": "RAPIDS 24.12 / CUDA 12.2",
        "min_driver": (525, 60, 13),
        "packages": ["cuml-cu12==24.12.0", "cupy-cuda12x==12.2.0"],
    },
]


def driver_version():
    """NVIDIA driver version from nvidia-smi. -> (text, tuple) or (None, None)."""
    import re
    import subprocess

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30,
        )
    except Exception:
        return None, None
    if proc.returncode != 0:
        return None, None

    text = (proc.stdout or "").strip().splitlines()
    if not text:
        return None, None
    match = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", text[0])
    if not match:
        return text[0], None
    return text[0], tuple(int(p) for p in match.groups(default="0"))


def recommended_stack(driver_tuple=None):
    """Highest RAPIDS stack this driver supports, or None."""
    if driver_tuple is None:
        _text, driver_tuple = driver_version()
    if driver_tuple is None:
        return None
    for stack in RAPIDS_STACKS:            # ordered newest-first
        if driver_tuple >= stack["min_driver"]:
            return stack
    return None


def install_command(stack=None):
    """The exact pip command for this machine, or a reason it can't be given.

    `--only-binary=:all:` is deliberate: without it a missing wheel silently
    becomes a source build that fails minutes later on a missing cublas header,
    which is the failure this whole table exists to prevent.
    """
    stack = stack or recommended_stack()
    if stack is None:
        return None
    packages = " ".join(stack["packages"])
    return (f"pip install --only-binary=:all: "
            f"--extra-index-url=https://pypi.nvidia.com {packages}")


def resolve(use_gpu):
    """Turn auto|on|off into a bool for the HDBSCAN backend.

    "on" raises rather than silently falling back, because quietly spending
    hours on CPU in a GPU job is worse than failing in the first second.
    """
    if use_gpu in (True, "on", "yes", "true"):
        ok, reason = hdbscan_backend()
        if not ok:
            raise RuntimeError(
                f"--gpu on was requested but the GPU HDBSCAN backend is "
                f"unusable: {reason}"
            )
        return True
    if use_gpu in (False, "off", "no", "false"):
        return False
    if use_gpu in (None, "auto"):
        return hdbscan_backend()[0]
    raise ValueError(f"--gpu must be auto, on or off; got {use_gpu!r}")


def report():
    """Everything `doctor` prints."""
    gpu_ok, gpu_reason = hdbscan_backend()
    cupy_ok, cupy_reason = cupy_linalg()
    info = device_info()
    driver_text, driver_tuple = driver_version()
    stack = recommended_stack(driver_tuple)
    return {
        **info,
        "driver": driver_text,
        "hdbscan_gpu": gpu_ok,
        "hdbscan_gpu_reason": gpu_reason,
        "cupy_linalg": cupy_ok,
        "cupy_linalg_reason": cupy_reason,
        "forced_cpu": forced_cpu(),
        "loader_hint": loader_path_hint(),
        "recommended_stack": stack["label"] if stack else None,
        "install_command": (None if gpu_ok else install_command(stack)),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
    }
