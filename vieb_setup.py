"""
VIEB First-Time Setup
---------------------
Get a fresh clone of VIEB to a running installation with a single command,
on Linux, macOS (Intel or Apple Silicon), Windows, or Windows+WSL2.

Usage:
    python vieb_setup.py
    python3 vieb_setup.py
    py vieb_setup.py          (Windows)

Requirements: Python 3 stdlib only — no pip installs needed to run this script.
"""
from __future__ import print_function

import json
import os
import platform
import re
import shutil
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")

GPU_STACKS = [
    {
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
        "label": "RAPIDS 26.04 / CUDA 12.9",
        "min_driver": (575, 51, 3),
        "packages": [
            "cudf-cu12==26.4.0",
            "cuml-cu12==26.4.0",
            "cupy-cuda12x==14.1.1",
        ],
    },
]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def run_streaming(cmd, description):
    """Run a command, stream its output line by line. Return exit code."""
    print("\n{0}".format(description))
    print("-" * len(description))
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    for raw in iter(proc.stdout.readline, b""):
        line = raw.decode("utf-8", errors="replace").rstrip("\n\r")
        print(line)
    proc.wait()
    return proc.returncode


def ask(prompt, default=""):
    """Prompt the user and return their input, falling back to default."""
    try:
        answer = raw_input(prompt)  # Python 2
    except NameError:
        answer = input(prompt)      # Python 3
    answer = answer.strip()
    return answer if answer else default


def get_version(cmd_parts):
    """
    Run `<cmd_parts> --version` and return a (major, minor, patch, version_string)
    tuple, or None if the command fails or the version is not parseable.
    """
    try:
        proc = subprocess.Popen(
            cmd_parts + ["--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = proc.communicate()
        text = (out + err).decode("utf-8", errors="replace")
        m = re.search(r"Python\s+(\d+)\.(\d+)\.(\d+)", text, re.IGNORECASE)
        if m:
            major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
            version_str = "{0}.{1}.{2}".format(major, minor, patch)
            return (major, minor, patch, version_str)
    except (OSError, IOError):
        pass
    return None


# ---------------------------------------------------------------------------
# Part 0 — Platform detection
# ---------------------------------------------------------------------------

def detect_platform():
    """
    Return one of: 'linux', 'wsl2', 'mac_intel', 'mac_arm', 'windows'.
    Every later decision (Python candidates, GPU handling, torch wheels,
    launch script format) branches off this value.
    """
    system = platform.system()

    if system == "Darwin":
        machine = platform.machine().lower()
        return "mac_arm" if machine in ("arm64", "aarch64") else "mac_intel"

    if system == "Windows":
        return "windows"

    if system == "Linux":
        try:
            with open("/proc/version") as f:
                text = f.read().lower()
            if "microsoft" in text:
                return "wsl2"
        except (OSError, IOError):
            pass
        return "linux"

    # Unknown platform: treat like Linux (closest stdlib behavior) but warn.
    print("WARNING: Unrecognized platform '{0}'. Proceeding as Linux.".format(system))
    return "linux"


PLATFORM_LABELS = {
    "linux": "Linux",
    "wsl2": "Windows (WSL2)",
    "mac_intel": "macOS (Intel)",
    "mac_arm": "macOS (Apple Silicon)",
    "windows": "Windows (native)",
}


# ---------------------------------------------------------------------------
# Part 1 & 2 — Python detection and selection
# ---------------------------------------------------------------------------

def find_python_interpreters(plat):
    """
    Probe candidate Python commands and return a sorted list of
    (version_string, command_string) for Python 3.10-3.12 only.
    Sorted descending by version (newest first).
    """
    candidates = ["python", "python3", "python3.10", "python3.11", "python3.12"]

    if plat == "windows":
        candidates += ["py -3.10", "py -3.11", "py -3.12"]
    elif plat in ("mac_intel", "mac_arm"):
        candidates += [
            "/usr/bin/python3.11",
            "/usr/local/bin/python3.10",
            "/usr/local/bin/python3.11",
            "/usr/local/bin/python3.12",
            "/opt/homebrew/bin/python3.10",
            "/opt/homebrew/bin/python3.11",
            "/opt/homebrew/bin/python3.12",
        ]
    else:  # linux / wsl2
        candidates += [
            "/usr/bin/python3.10",
            "/usr/bin/python3.11",
            "/usr/bin/python3.12",
            "/usr/local/bin/python3.10",
            "/usr/local/bin/python3.11",
            "/usr/local/bin/python3.12",
        ]

    seen_paths = set()
    found = []

    for candidate in candidates:
        parts = candidate.split()

        result = get_version(parts)
        if result is None:
            continue

        major, minor, patch, version_str = result
        if major != 3 or minor < 10 or minor > 12:
            continue

        try:
            proc = subprocess.Popen(
                parts + ["-c", "import sys; print(sys.executable)"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            out, _ = proc.communicate()
            resolved = out.decode("utf-8", errors="replace").strip()
        except (OSError, IOError):
            resolved = candidate

        real = os.path.realpath(resolved) if os.path.isfile(resolved) else resolved
        if real in seen_paths:
            continue
        seen_paths.add(real)

        found.append((major, minor, patch, version_str, candidate))

    if not found:
        print(
            "\nNo compatible Python found (need 3.10, 3.11, or 3.12).\n"
            "DeepLabCut and several other VIEB dependencies do not yet support 3.13+.\n"
            "\nDownload a compatible Python here:\n"
            "  https://www.python.org/downloads/release/python-3119/\n"
            "\nInstall it, then re-run this script:\n"
            "  python3 vieb_setup.py"
        )
        sys.exit(1)

    found.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
    return [(t[3], t[4]) for t in found]


def select_interpreter(interpreters):
    """
    If one interpreter found, use it automatically.
    If multiple, let the user pick. Returns (version_string, command_string).
    """
    if len(interpreters) == 1:
        version_str, cmd = interpreters[0]
        print("Using Python {0} ({1})".format(version_str, cmd))
        return interpreters[0]

    print("\nFound multiple compatible Python versions:")
    for i, (version_str, cmd) in enumerate(interpreters, 1):
        print("  {0}. Python {1} ({2})".format(i, version_str, cmd))

    choice = ask("Enter number [1]: ", "1")
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(interpreters):
            raise ValueError
    except ValueError:
        print("Invalid choice, defaulting to 1.")
        idx = 0

    version_str, cmd = interpreters[idx]
    print("Using Python {0} ({1})".format(version_str, cmd))
    return interpreters[idx]


# ---------------------------------------------------------------------------
# Part 3 — Venv creation (shared by core venv and venv-dlc)
# ---------------------------------------------------------------------------

def venv_python_path(venv_dir, plat):
    if plat == "windows":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def create_or_reuse_venv(python_cmd, venv_name, plat, label):
    """
    Create (or reuse, by user choice) ./<venv_name>.
    Returns the path to that venv's Python executable, or None if the user
    aborted entirely.
    """
    venv_dir = os.path.join(PROJECT_ROOT, venv_name)

    if os.path.isdir(venv_dir):
        print(
            "\n{0} environment already exists at ./{1}\n"
            "[r] Reuse existing  [d] Delete and recreate  [s] Skip  [q] Quit".format(
                label, venv_name
            )
        )
        choice = ask("Choice [r]: ", "r").lower()

        if choice == "q":
            print("Aborted.")
            sys.exit(0)
        elif choice == "s":
            return None
        elif choice == "d":
            print("Removing existing {0}...".format(venv_name))
            shutil.rmtree(venv_dir)
        else:
            print("Reusing existing {0}.".format(venv_name))

    if not os.path.isdir(venv_dir):
        python_parts = python_cmd.split()
        ret = run_streaming(
            python_parts + ["-m", "venv", venv_name],
            "Creating {0} virtual environment ({1})...".format(label, venv_name),
        )
        if ret != 0:
            print(
                "\nFailed to create {0}. Try:\n"
                "  {1} -m pip install virtualenv\n"
                "  {1} -m virtualenv {2}".format(venv_name, python_cmd, venv_name)
            )
            sys.exit(1)

    venv_python = venv_python_path(venv_dir, plat)
    if not os.path.isfile(venv_python):
        print("ERROR: Could not find venv Python at: {0}".format(venv_python))
        sys.exit(1)

    print("Upgrading pip and setuptools in ./{0}...".format(venv_name))
    subprocess.call([venv_python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "--quiet"])

    return venv_python


# ---------------------------------------------------------------------------
# Part 4 — Core dependencies + optional GPU (cuML/RAPIDS)
# ---------------------------------------------------------------------------

def detect_cuda_version():
    """Run nvidia-smi and parse CUDA version. Returns (major, minor) or None."""
    try:
        proc = subprocess.Popen(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, _ = proc.communicate()
        text = out.decode("utf-8", errors="replace")
        m = re.search(r"CUDA\s+Version:\s+(\d+)\.(\d+)", text)
        if m:
            return (int(m.group(1)), int(m.group(2)))
    except (OSError, IOError):
        pass
    return None


def _parse_version_tuple(text):
    m = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", text or "")
    if not m:
        return None
    return tuple(int(part) for part in m.groups(default="0"))


def _version_gte(found, required):
    size = max(len(found), len(required))
    found = found + (0,) * (size - len(found))
    required = required + (0,) * (size - len(required))
    return found >= required


def detect_nvidia_driver_version():
    """Run nvidia-smi and parse driver version. Returns tuple or None."""
    try:
        proc = subprocess.Popen(
            ["nvidia-smi"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, _ = proc.communicate()
        text = out.decode("utf-8", errors="replace")
        m = re.search(r"Driver Version:\s*([0-9.]+)", text)
        if m:
            return _parse_version_tuple(m.group(1))
    except (OSError, IOError):
        pass
    return None


def select_gpu_stack(driver_version):
    if not driver_version:
        return None
    for stack in sorted(GPU_STACKS, key=lambda item: item["min_driver"], reverse=True):
        if _version_gte(driver_version, stack["min_driver"]):
            return stack
    return None


def install_core(venv_python, plat):
    """
    Install VIEB's core dependencies. On Linux/WSL2, attempt GPU (RAPIDS)
    extras and verify they actually import; fall back gracefully otherwise.
    Returns a status dict for the summary.
    """
    status = {"core": False, "gpu": None, "gpu_reason": None}

    ret = run_streaming(
        [venv_python, "-m", "pip", "install", "--no-build-isolation", "-e", "."],
        "Installing core VIEB dependencies...",
    )
    if ret != 0:
        print(
            "\nCore installation failed.\n"
            "This usually means a system build tool is missing (e.g. a C compiler\n"
            "for scipy/hdbscan, or a missing Qt5 system library for PyQt5).\n"
            "Check the output above for the failing package, install its system\n"
            "dependency, then re-run: python vieb_setup.py"
        )
        sys.exit(1)
    status["core"] = True

    if plat not in ("linux", "wsl2"):
        status["gpu"] = None
        status["gpu_reason"] = {
            "mac_intel": "GPU acceleration (RAPIDS/cuML) requires an NVIDIA GPU on Linux; not available on macOS.",
            "mac_arm":   "GPU acceleration (RAPIDS/cuML) requires an NVIDIA GPU on Linux; not available on Apple Silicon.",
            "windows":   "RAPIDS/cuML does not support Windows natively. Run this script inside WSL2 for GPU acceleration.",
        }.get(plat, "GPU acceleration is only supported on Linux/WSL2.")
        return status

    cuda = detect_cuda_version()
    if cuda is None:
        status["gpu"] = False
        status["gpu_reason"] = (
            "No NVIDIA GPU detected (nvidia-smi not found or failed). "
            "Continuing without GPU acceleration — this is optional."
        )
        return status

    cuda_major, cuda_minor = cuda
    if cuda_major < 12:
        status["gpu"] = False
        status["gpu_reason"] = (
            "Detected CUDA {0}.{1}, but RAPIDS requires CUDA 12.0+. "
            "Update your NVIDIA driver to enable GPU acceleration. "
            "Continuing without it.".format(cuda_major, cuda_minor)
        )
        return status

    driver_version = detect_nvidia_driver_version()
    stack = select_gpu_stack(driver_version)
    if stack is None:
        min_driver = ".".join(str(part) for part in min(s["min_driver"] for s in GPU_STACKS))
        status["gpu"] = False
        status["gpu_reason"] = (
            "Detected CUDA {0}.{1}, but VIEB's pinned RAPIDS stack requires "
            "NVIDIA driver {2} or newer. Upgrade the NVIDIA driver, or run "
            "CPU mode for now.".format(cuda_major, cuda_minor, min_driver)
        )
        return status

    ret = run_streaming(
        [venv_python, "-m", "pip", "install",
         "--extra-index-url", "https://pypi.nvidia.com"] + stack["packages"],
        "Installing GPU (RAPIDS/cuML) extras: {0}...".format(stack["label"]),
    )
    if ret != 0:
        status["gpu"] = False
        status["gpu_reason"] = (
            "RAPIDS installation failed (see output above). "
            "GPU acceleration is optional — continuing without it."
        )
        return status

    # Verify cuML actually imports before declaring success.
    verify = subprocess.run(
        [venv_python, "-c", "import cuml"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if verify.returncode != 0:
        status["gpu"] = False
        status["gpu_reason"] = (
            "RAPIDS packages installed but 'import cuml' failed:\n"
            "  {0}\n"
            "This is usually a CUDA driver/library mismatch. "
            "GPU acceleration is optional — continuing without it.".format(
                verify.stderr.decode("utf-8", errors="replace").strip().splitlines()[-1]
                if verify.stderr else "unknown error"
            )
        )
        return status

    status["gpu"] = True
    status["gpu_cuda"] = "{0}.{1}".format(cuda_major, cuda_minor)
    return status


# ---------------------------------------------------------------------------
# Part 5 — venv-dlc (DeepLabCut) with platform-correct torch wheels
# ---------------------------------------------------------------------------

# Known-working combination per the project notes: torch 2.1.0 (cu121 wheel)
# + torchvision 0.16.0 (cu118 wheel). The mismatched CUDA tags are intentional
# and reproduce the configuration that has been verified to work with DLC 3.0.
_TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu121"
_TORCHVISION_CUDA_INDEX = "https://download.pytorch.org/whl/cu118"
_TORCH_VERSION = "2.1.0"
_TORCHVISION_VERSION = "0.16.0"


def install_dlc(venv_dlc_python, plat):
    """
    Install torch/torchvision (platform-appropriate) plus DeepLabCut into
    venv-dlc, then verify `import deeplabcut` succeeds.
    Returns a status dict for the summary.
    """
    status = {"dlc": False, "dlc_reason": None}

    if plat in ("mac_intel", "mac_arm"):
        ret = run_streaming(
            [
                venv_dlc_python, "-m", "pip", "install",
                "torch=={0}".format(_TORCH_VERSION),
                "torchvision=={0}".format(_TORCHVISION_VERSION),
            ],
            "Installing PyTorch {0} / torchvision {1} (CPU, macOS)...".format(
                _TORCH_VERSION, _TORCHVISION_VERSION
            ),
        )
        torch_label = "CPU"
    else:
        # linux, wsl2, windows: known-working CUDA wheel combination.
        ret = run_streaming(
            [
                venv_dlc_python, "-m", "pip", "install",
                "torch=={0}".format(_TORCH_VERSION),
                "--index-url", _TORCH_CUDA_INDEX,
            ],
            "Installing PyTorch {0} (CUDA 12.1 wheel)...".format(_TORCH_VERSION),
        )
        if ret == 0:
            ret = run_streaming(
                [
                    venv_dlc_python, "-m", "pip", "install",
                    "torchvision=={0}".format(_TORCHVISION_VERSION),
                    "--index-url", _TORCHVISION_CUDA_INDEX,
                ],
                "Installing torchvision {0} (CUDA 11.8 wheel)...".format(_TORCHVISION_VERSION),
            )
        torch_label = "CUDA 12.1/11.8 wheels"

    if ret != 0:
        status["dlc_reason"] = (
            "Failed to install PyTorch/torchvision ({0}). "
            "Check your network connection and that pip can reach "
            "download.pytorch.org. DLC setup aborted; re-run "
            "python vieb_setup.py to retry just this step "
            "(the core venv is unaffected).".format(torch_label)
        )
        return status

    ret = run_streaming(
        [venv_dlc_python, "-m", "pip", "install", "--no-build-isolation", "-e", ".[deeplabcut]"],
        "Installing DeepLabCut and its dependencies (this can take a while)...",
    )
    if ret != 0:
        status["dlc_reason"] = (
            "DeepLabCut package installation failed (see output above). "
            "Common causes: missing system Qt libraries for PySide6/napari, "
            "or a transient PyPI timeout. Re-run python vieb_setup.py to retry."
        )
        return status

    print("\nVerifying DeepLabCut import...")
    verify = subprocess.run(
        [venv_dlc_python, "-c", "import deeplabcut"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if verify.returncode != 0:
        err_text = verify.stderr.decode("utf-8", errors="replace").strip()
        err_last = err_text.splitlines()[-1] if err_text else "unknown error"
        status["dlc_reason"] = (
            "DeepLabCut installed but 'import deeplabcut' failed:\n"
            "  {0}\n"
            "This usually means a torch/torchvision version mismatch. "
            "Try deleting venv-dlc and re-running python vieb_setup.py.".format(err_last)
        )
        return status

    print("DeepLabCut import OK.")
    status["dlc"] = True
    return status


# ---------------------------------------------------------------------------
# Part 6 — Write dlc_python into config.json
# ---------------------------------------------------------------------------

def write_dlc_python_config(venv_dlc_python):
    """
    Persist the venv-dlc interpreter path into config.json's 'dlc_python' key
    so VIEB uses venv-dlc for DLC operations automatically.
    """
    cfg = {}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                cfg = json.load(f)
        except (json.JSONDecodeError, OSError):
            cfg = {}

    cfg["dlc_python"] = os.path.abspath(venv_dlc_python)

    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    print("\nWrote dlc_python = {0} to config.json".format(cfg["dlc_python"]))


# ---------------------------------------------------------------------------
# Part 7 — Launch script generation
# ---------------------------------------------------------------------------

def write_launch_script(plat):
    """
    Write run_vieb.sh (Linux/macOS/WSL2) or run_vieb.bat (Windows native).
    Returns (script_path, alias_command_string).
    """
    if plat == "windows":
        bat_path = os.path.join(PROJECT_ROOT, "run_vieb.bat")
        with open(bat_path, "w") as f:
            f.write("@echo off\r\n")
            f.write('cd /d "%~dp0"\r\n')
            f.write("call venv\\Scripts\\activate\r\n")
            f.write("python user_interface.py %*\r\n")
        print("Created: run_vieb.bat")
        alias = '  doskey vieb="{0}"'.format(bat_path)
        return bat_path, alias

    sh_path = os.path.join(PROJECT_ROOT, "run_vieb.sh")
    with open(sh_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write('cd "$(dirname "$0")"\n')
        f.write("source venv/bin/activate\n")
        f.write('python user_interface.py "$@"\n')
    try:
        os.chmod(sh_path, 0o755)
    except OSError:
        pass
    print("Created: run_vieb.sh")
    alias = '  alias vieb="{0}"'.format(sh_path)
    return sh_path, alias


# ---------------------------------------------------------------------------
# Part 8 — Summary
# ---------------------------------------------------------------------------

def print_summary(plat, version_str, core_status, dlc_status, launch_path, alias_cmd):
    print("\n" + "=" * 60)
    print(" VIEB Setup Complete")
    print("=" * 60)
    print("Platform:    {0}".format(PLATFORM_LABELS.get(plat, plat)))
    print("Python:      {0}".format(version_str))

    # Core
    if core_status["core"]:
        print("Core deps:   installed (./venv)")
    else:
        print("Core deps:   FAILED")

    # GPU
    gpu = core_status.get("gpu")
    if gpu is True:
        print("GPU (RAPIDS): installed and verified (CUDA {0})".format(core_status.get("gpu_cuda", "?")))
    elif gpu is False:
        print("GPU (RAPIDS): skipped — {0}".format(core_status.get("gpu_reason", "")))
    else:
        print("GPU (RAPIDS): not applicable — {0}".format(core_status.get("gpu_reason", "")))

    # DLC
    if dlc_status is None:
        print("DeepLabCut:  skipped (venv-dlc step not run)")
    elif dlc_status.get("dlc"):
        print("DeepLabCut:  installed and verified (./venv-dlc)")
    else:
        print("DeepLabCut:  FAILED — {0}".format(dlc_status.get("dlc_reason", "unknown error")))

    print("-" * 60)
    print("To launch VIEB:")
    if plat == "windows":
        print("  run_vieb.bat")
    else:
        print("  ./run_vieb.sh")
    print("\nOptional shell alias (add to your shell profile):")
    print(alias_cmd)
    print("\nOr manually:")
    if plat == "windows":
        print("  venv\\Scripts\\activate")
    else:
        print("  source venv/bin/activate")
    print("  python user_interface.py")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Commands that pip/setuptools passes when using this file as a build backend
# fallback. Delegate to setuptools so our interactive bootstrap does not
# re-run inside a pip subprocess and cause infinite recursion.
_LEGACY_BUILD_COMMANDS = frozenset([
    "egg_info", "dist_info", "develop", "install", "bdist_wheel",
    "sdist", "build", "build_ext", "build_py", "build_clib",
    "build_scripts", "clean", "check", "config", "install_lib",
    "install_headers", "install_scripts", "install_data",
    "install_egg_info", "rotate", "saveopts", "setopt",
    "upload", "register", "alias", "easy_install", "test",
])


def main():
    if len(sys.argv) > 1 and sys.argv[1] in _LEGACY_BUILD_COMMANDS:
        try:
            import setuptools
            setuptools.setup()
        except Exception as exc:
            sys.stderr.write("setuptools error: {0}\n".format(exc))
            sys.exit(1)
        return

    print("=" * 60)
    print(" VIEB First-Time Setup")
    print("=" * 60)

    plat = detect_platform()
    print("Detected platform: {0}".format(PLATFORM_LABELS.get(plat, plat)))

    # Part 1 & 2: detect and select interpreter
    interpreters = find_python_interpreters(plat)
    version_str, python_cmd = select_interpreter(interpreters)

    # Part 3: core venv
    venv_python = create_or_reuse_venv(python_cmd, "venv", plat, "Core")
    if venv_python is None:
        print("\nCore venv skipped — nothing more to do without it. Exiting.")
        sys.exit(0)

    # Part 4: core deps + optional GPU
    core_status = install_core(venv_python, plat)

    # Part 5: venv-dlc (optional)
    dlc_status = None
    print(
        "\nSet up DeepLabCut (pose estimation) in a separate environment "
        "(venv-dlc)? (y/n) [y]: ", end=""
    )
    dlc_answer = ask("", "y").lower()
    if dlc_answer != "n":
        venv_dlc_python = create_or_reuse_venv(python_cmd, "venv-dlc", plat, "DLC")
        if venv_dlc_python is not None:
            dlc_status = install_dlc(venv_dlc_python, plat)
            if dlc_status["dlc"]:
                # Part 6: write dlc_python into config.json
                write_dlc_python_config(venv_dlc_python)

    # Part 7: launch script
    launch_path, alias_cmd = write_launch_script(plat)

    # Part 8: summary
    print_summary(plat, version_str, core_status, dlc_status, launch_path, alias_cmd)


if __name__ == "__main__":
    main()
