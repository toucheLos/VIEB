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

Probing answers "does the GPU work here"; it cannot answer "what should I
install". That needs the driver version, because RAPIDS wheels are pinned
against a minimum driver and pip will happily install a stack the driver
cannot load. GPU_STACKS below is the driver -> stack table, ported from v1's
_utils.py (its WSL2 half is dropped -- irrelevant on a Linux cluster).
"""

from __future__ import annotations

import glob
import os
import re
import subprocess
import sys

_PROBE_CACHE = {}

# Pinned RAPIDS stacks, oldest first. Each entry is the *minimum* NVIDIA driver
# that can load it, so a newer driver qualifies for more than one and the newest
# match wins. Keep in sync with v1's _utils.GPU_STACKS and pyproject's [gpu].
GPU_STACKS = [
    {
        "id": "rapids-24.12-cuda12.2",
        "label": "RAPIDS 24.12 / CUDA 12.2",
        "min_driver": (525, 60, 13),
        "packages": [
            "cuml-cu12==24.12.0",
            "cudf-cu12==24.12.0",
            "cupy-cuda12x==12.2.0",
            "cuda-python==12.2.1",
            "cuda-toolkit[cublas,cufft,curand,cusolver,cusparse]==12.2.2",
            "nvidia-cuda-runtime-cu12==12.2.140",
            "nvidia-cuda-nvrtc-cu12==12.2.140",
            "nvidia-nvjitlink-cu12==12.2.140",
            "nvidia-cublas-cu12==12.2.5.6",
            "nvidia-cufft-cu12==11.0.8.103",
            "nvidia-curand-cu12==10.3.3.141",
            "nvidia-cusolver-cu12==11.5.2.141",
            "nvidia-cusparse-cu12==12.1.2.141",
        ],
    },
    {
        "id": "rapids-26.04-cuda12.9",
        "label": "RAPIDS 26.04 / CUDA 12.9",
        "min_driver": (575, 51, 3),
        "packages": [
            "cudf-cu12==26.4.0",
            "cuml-cu12==26.4.0",
            "cupy-cuda12x==14.1.1",
        ],
    },
]


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


def _parse_version_tuple(text):
    m = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", text)
    if not m:
        return None
    return tuple(int(part) for part in m.groups(default="0"))


def _version_gte(found, required):
    size = max(len(found), len(required))
    found = found + (0,) * (size - len(found))
    required = required + (0,) * (size - len(required))
    return found >= required


def detect_nvidia_driver():
    """Return NVIDIA driver/GPU details from nvidia-smi.

    Not cached: unlike the probes this costs a subprocess, not a fitted model,
    and on a cluster the same checkout runs on nodes with different hardware.
    """
    info = {
        "ok": False,
        "gpu_name": None,
        "driver": None,
        "driver_tuple": None,
        "cuda": None,
        "error": None,
    }
    try:
        proc = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=8)
    except Exception as exc:
        # FileNotFoundError on a login node without the driver -- expected, and
        # the reason `doctor` has to run on a GPU node to say anything useful.
        info["error"] = str(exc)
        return info
    if proc.returncode != 0:
        info["error"] = (proc.stderr or proc.stdout).strip()
        return info

    text = proc.stdout
    driver = re.search(r"Driver Version:\s*([0-9.]+)", text)
    cuda = re.search(r"CUDA Version:\s*([0-9.]+)", text)
    if driver:
        info["driver"] = driver.group(1)
        info["driver_tuple"] = _parse_version_tuple(driver.group(1))
    if cuda:
        info["cuda"] = cuda.group(1)

    try:
        gpu_proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5)
        info["gpu_name"] = gpu_proc.stdout.strip().splitlines()[0].strip() or None
    except Exception:
        pass

    info["ok"] = bool(info["driver_tuple"])
    return info


def select_gpu_stack(driver_tuple):
    """Newest pinned RAPIDS stack this driver can load, or None."""
    if not driver_tuple:
        return None
    for stack in sorted(GPU_STACKS, key=lambda s: s["min_driver"], reverse=True):
        if _version_gte(driver_tuple, stack["min_driver"]):
            return stack
    return None


def stack_message(driver_info):
    """One line of installation advice for whatever the driver turned out to be."""
    if not driver_info.get("ok"):
        return ("No working NVIDIA driver was detected. On a cluster this is "
                "expected on a login node -- run doctor on the gpu partition.")

    driver = driver_info.get("driver") or "unknown"
    cuda = driver_info.get("cuda") or "unknown"
    stack = select_gpu_stack(driver_info.get("driver_tuple"))
    if stack:
        return (f"Detected NVIDIA driver {driver} (CUDA {cuda}). "
                f"Recommended install: {stack['label']}.")

    minimum = min(s["min_driver"] for s in GPU_STACKS)
    min_text = ".".join(str(part) for part in minimum)
    return (f"Detected NVIDIA driver {driver} (CUDA {cuda}). The pinned RAPIDS "
            f"stacks require driver {min_text} or newer -- no GPU stack can be "
            f"installed against this driver.")


_EXPLICIT_ON = (True, "on", "yes", "true")


def explicitly_requested(use_gpu):
    """True when the caller demanded a GPU rather than accepting whatever is
    there.

    `resolve` only answers "should this run on GPU", which is the same bool for
    "on" and for an "auto" that happened to find a device. The difference
    matters once a *runtime* failure happens: under "on" the caller has said a
    CPU fallback is not an acceptable outcome, so the same tuple that drives
    resolve's raise-early behaviour is shared here rather than restated.
    """
    return use_gpu in _EXPLICIT_ON


def resolve(use_gpu):
    """Turn auto|on|off into a bool for the HDBSCAN backend.

    "on" raises rather than silently falling back, because quietly spending
    hours on CPU in a GPU job is worse than failing in the first second.
    """
    if explicitly_requested(use_gpu):
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
    driver = detect_nvidia_driver()
    stack = select_gpu_stack(driver.get("driver_tuple"))
    return {
        **info,
        "hdbscan_gpu": gpu_ok,
        "hdbscan_gpu_reason": gpu_reason,
        "cupy_linalg": cupy_ok,
        "cupy_linalg_reason": cupy_reason,
        "forced_cpu": forced_cpu(),
        "loader_hint": loader_path_hint(),
        # What is installed (above) versus what should be (below) -- a GPU can
        # be idle here because nothing is installed or because the wrong
        # RAPIDS pin is, and those need different fixes.
        "driver_ok": driver["ok"],
        "driver": driver["driver"],
        "driver_cuda": driver["cuda"],
        "driver_gpu_name": driver["gpu_name"],
        "driver_error": driver["error"],
        "recommended_stack_id": stack["id"] if stack else None,
        "recommended_stack_label": stack["label"] if stack else None,
        "recommended_stack_packages": list(stack["packages"]) if stack else [],
        "stack_message": stack_message(driver),
    }
