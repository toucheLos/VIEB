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
    return {
        **info,
        "hdbscan_gpu": gpu_ok,
        "hdbscan_gpu_reason": gpu_reason,
        "cupy_linalg": cupy_ok,
        "cupy_linalg_reason": cupy_reason,
        "forced_cpu": forced_cpu(),
        "loader_hint": loader_path_hint(),
    }
