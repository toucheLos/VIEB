"""
VIEB Setup Script
-----------------
Bootstrap a working VIEB environment on Windows, Linux, or macOS.

Usage:
    python setup.py
    python3 setup.py
    py setup.py          (Windows)

Requirements: Python 2.7+ stdlib only (runs on any system Python).
"""
from __future__ import print_function

import os
import re
import subprocess
import sys
import shutil

# ---------------------------------------------------------------------------
# Helpers
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
    Run `<cmd_parts> --version` and return a (major, minor, patch, version_string) tuple,
    or None if the command fails or the version is not parseable.
    """
    try:
        proc = subprocess.Popen(
            cmd_parts + ["--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        out, err = proc.communicate()
        text = (out + err).decode("utf-8", errors="replace")
        # Match "Python 3.11.8" anywhere in output
        m = re.search(r"Python\s+(\d+)\.(\d+)\.(\d+)", text, re.IGNORECASE)
        if m:
            major, minor, patch = int(m.group(1)), int(m.group(2)), int(m.group(3))
            version_str = "{0}.{1}.{2}".format(major, minor, patch)
            return (major, minor, patch, version_str)
    except (OSError, IOError):
        pass
    return None


# ---------------------------------------------------------------------------
# Part 1 & 2 — Python detection and selection
# ---------------------------------------------------------------------------

def find_python_interpreters():
    """
    Probe candidate Python commands and return a sorted list of
    (version_string, command_string) for Python 3.10-3.12 only.
    Sorted descending by version (newest first).
    """
    candidates = ["python", "python3", "python3.10", "python3.11", "python3.12"]

    if sys.platform == "win32":
        candidates += ["py -3.10", "py -3.11", "py -3.12"]
    else:
        extra_paths = [
            "/usr/bin/python3.11",
            "/usr/local/bin/python3.11",
            "/opt/homebrew/bin/python3.11",
            "/opt/homebrew/bin/python3.12",
        ]
        candidates += extra_paths

    seen_paths = set()
    found = []

    for candidate in candidates:
        # Split "py -3.11" style strings into a list
        parts = candidate.split()

        result = get_version(parts)
        if result is None:
            continue

        major, minor, patch, version_str = result
        if major != 3 or minor < 10 or minor > 12:
            continue

        # Deduplicate by resolved path when possible
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
            "\nNo compatible Python found (need 3.10-3.12).\n"
            "\nInstall Python from https://www.python.org/downloads/\n"
            "Then re-run this script."
        )
        sys.exit(1)

    # Sort descending by (major, minor, patch)
    found.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)

    # Return (version_string, command_string)
    return [(t[3], t[4]) for t in found]


def select_interpreter(interpreters):
    """
    If one interpreter found, use it automatically.
    If multiple, let the user pick.
    Returns (version_string, command_string).
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
# Part 3 — Venv creation
# ---------------------------------------------------------------------------

def create_venv(python_cmd):
    """
    Create (or reuse) ./venv.  Returns the path to the venv Python executable.
    """
    venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")

    if os.path.isdir(venv_dir):
        print(
            "\nVirtual environment already exists at ./venv\n"
            "[r] Reuse existing  [d] Delete and recreate  [q] Quit"
        )
        choice = ask("Choice [r]: ", "r").lower()

        if choice == "q":
            print("Aborted.")
            sys.exit(0)
        elif choice == "d":
            print("Removing existing venv...")
            shutil.rmtree(venv_dir)
        else:
            print("Reusing existing venv.")

    if not os.path.isdir(venv_dir):
        python_parts = python_cmd.split()
        ret = run_streaming(
            python_parts + ["-m", "venv", "venv"],
            "Creating virtual environment",
        )
        if ret != 0:
            print(
                "\nFailed to create virtual environment. Try:\n"
                "  {0} -m pip install virtualenv\n"
                "  {0} -m virtualenv venv".format(python_cmd)
            )
            sys.exit(1)

    # Determine venv Python path
    if sys.platform == "win32":
        venv_python = os.path.join(venv_dir, "Scripts", "python.exe")
    else:
        venv_python = os.path.join(venv_dir, "bin", "python")

    if not os.path.isfile(venv_python):
        print("ERROR: Could not find venv Python at: {0}".format(venv_python))
        sys.exit(1)

    # Upgrade pip quietly
    print("Upgrading pip...")
    subprocess.call(
        [venv_python, "-m", "pip", "install", "--upgrade", "pip", "--quiet"]
    )

    return venv_python


# ---------------------------------------------------------------------------
# Part 4 — Dependency installation
# ---------------------------------------------------------------------------

def detect_cuda_version():
    """
    Run nvidia-smi and parse CUDA version. Returns (major, minor) or None.
    """
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


def install_dependencies(venv_python):
    """
    Install core, optional GPU, and optional DLC dependencies.
    Returns a dict with status info for the summary.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    status = {
        "core": False,
        "gpu": None,       # None = skipped, True = ok, string = CUDA version
        "gpu_cuda": None,
        "dlc": None,
    }

    # ── Step 1: Core ────────────────────────────────────────────────────────
    # --no-build-isolation: use the venv's already-installed setuptools so pip
    # does not spawn a fresh build env that would re-invoke this setup.py.
    ret = run_streaming(
        [venv_python, "-m", "pip", "install", "--no-build-isolation", "-e", "."],
        "Installing core dependencies...",
    )
    if ret != 0:
        print(
            "\nCore installation failed.\n"
            "Check the output above for details and re-run setup.py."
        )
        sys.exit(1)
    status["core"] = True

    # ── Step 2: GPU (optional) ───────────────────────────────────────────────
    gpu_answer = ask("\nDo you have an NVIDIA GPU? (y/n) [n]: ", "n").lower()

    if gpu_answer == "y":
        if sys.platform == "win32":
            print(
                "\nGPU acceleration on Windows requires WSL2.\n"
                "Install WSL2 first, then run this script inside WSL2.\n"
                "Proceeding with CPU-only installation."
            )
            status["gpu"] = False
        else:
            cuda = detect_cuda_version()
            if cuda is None:
                print(
                    "nvidia-smi not found. Cannot detect CUDA version.\n"
                    "Install NVIDIA drivers first."
                )
                status["gpu"] = False
            else:
                cuda_major, cuda_minor = cuda
                cuda_str = "{0}.{1}".format(cuda_major, cuda_minor)
                if cuda_major < 12:
                    print(
                        "\nCUDA {0} detected. RAPIDS requires CUDA 12.0+.\n"
                        "Please update your NVIDIA driver.\n"
                        "GPU acceleration will not be available.".format(cuda_str)
                    )
                    status["gpu"] = False
                else:
                    # RAPIDS
                    ret = run_streaming(
                        [venv_python, "-m", "pip", "install", "--no-build-isolation", "-e", ".[gpu]"],
                        "Installing GPU (RAPIDS) extras...",
                    )
                    if ret != 0:
                        print("WARNING: GPU extras installation failed. Continuing.")
                        status["gpu"] = False
                    else:
                        # PyTorch with matching CUDA wheel
                        if cuda_major == 12 and cuda_minor <= 2:
                            torch_index = "https://download.pytorch.org/whl/cu122"
                        else:
                            torch_index = "https://download.pytorch.org/whl/cu124"

                        ret = run_streaming(
                            [
                                venv_python, "-m", "pip", "install",
                                "torch", "--index-url", torch_index,
                            ],
                            "Installing PyTorch (CUDA {0})...".format(cuda_str),
                        )
                        if ret != 0:
                            print("WARNING: PyTorch installation failed. Continuing.")
                            status["gpu"] = False
                        else:
                            status["gpu"] = True
                            status["gpu_cuda"] = cuda_str

    # ── Step 3: DeepLabCut (optional) ───────────────────────────────────────
    print(
        "\nInstall DeepLabCut for pose estimation? (y/n) [n]:\n"
        "(Only needed if you have not already run pose tracking.\n"
        " Recommended to install in a separate environment.) ",
        end=""
    )
    dlc_answer = ask("", "n").lower()

    if dlc_answer == "y":
        if status["gpu"]:
            print(
                "\nWARNING: DeepLabCut conflicts with the GPU (RAPIDS) stack.\n"
                "If you installed GPU extras, consider using a separate venv for DLC.\n"
                "Continue anyway? (y/n) [n]: ",
                end=""
            )
            confirm = ask("", "n").lower()
            if confirm != "y":
                print("Skipping DeepLabCut installation.")
                status["dlc"] = False
                return status

        ret = run_streaming(
            [venv_python, "-m", "pip", "install", "--no-build-isolation", "-e", ".[deeplabcut]"],
            "Installing DeepLabCut...",
        )
        if ret != 0:
            print("WARNING: DeepLabCut installation failed. Continuing.")
            status["dlc"] = False
        else:
            status["dlc"] = True

    return status


# ---------------------------------------------------------------------------
# Part 5 — Launch script generation
# ---------------------------------------------------------------------------

def write_launch_script():
    """Write run_vieb.sh (Linux/Mac) or run_vieb.bat (Windows)."""
    project_root = os.path.dirname(os.path.abspath(__file__))

    if sys.platform == "win32":
        bat_path = os.path.join(project_root, "run_vieb.bat")
        with open(bat_path, "w") as f:
            f.write("@echo off\r\n")
            f.write('cd /d "%~dp0"\r\n')
            f.write("call venv\\Scripts\\activate\r\n")
            f.write("python user_interface.py %*\r\n")
        print("Created: run_vieb.bat")
    else:
        sh_path = os.path.join(project_root, "run_vieb.sh")
        with open(sh_path, "w") as f:
            f.write('#!/bin/bash\n')
            f.write('cd "$(dirname "$0")"\n')
            f.write("source venv/bin/activate\n")
            f.write('python user_interface.py "$@"\n')
        try:
            os.chmod(sh_path, 0o755)
        except OSError:
            pass
        print("Created: run_vieb.sh")


# ---------------------------------------------------------------------------
# Part 6 — Summary
# ---------------------------------------------------------------------------

def print_summary(version_str, status):
    """Print the final setup summary box."""

    def status_str(val, ok_label="installed", skip_label="skipped"):
        if val is True:
            return "✓ {0}".format(ok_label)
        elif val is False:
            return "✗ {0}".format(skip_label)
        else:
            return "✗ {0}".format(skip_label)

    core_s  = status_str(status["core"])
    dlc_s   = status_str(status["dlc"])

    if status["gpu"] is True:
        gpu_s = "✓ installed (CUDA {0})".format(status["gpu_cuda"])
    elif status["gpu"] is False:
        gpu_s = "✗ skipped"
    else:
        gpu_s = "✗ skipped"

    # Pad each field to a fixed width for the box
    width = 42

    def row(label, value):
        content = "  {0:<14}{1}".format(label, value)
        pad = width - len(content) - 2
        if pad < 0:
            pad = 0
        return "║{0}{1}║".format(content, " " * pad)

    border_top    = "╔{0}╗".format("═" * width)
    border_mid    = "╠{0}╣".format("═" * width)
    border_bot    = "╚{0}╝".format("═" * width)
    title_text    = "VIEB Setup Complete"
    title_pad_l   = (width - len(title_text)) // 2
    title_pad_r   = width - len(title_text) - title_pad_l
    title_row     = "║{0}{1}{2}║".format(" " * title_pad_l, title_text, " " * title_pad_r)

    blank = "║{0}║".format(" " * width)

    def fixed(text):
        pad = width - len(text) - 2
        if pad < 0:
            pad = 0
        return "║  {0}{1}║".format(text, " " * pad)

    print("")
    print(border_top)
    print(title_row)
    print(border_mid)
    print(row("Python:", version_str))
    print(row("Core:", core_s))
    print(row("GPU:", gpu_s))
    print(row("DeepLabCut:", dlc_s))
    print(border_mid)
    print(fixed("To launch VIEB:"))
    if sys.platform == "win32":
        print(fixed("  Windows:    run_vieb.bat"))
    else:
        print(fixed("  Linux/Mac:  ./run_vieb.sh"))
    print(blank)
    print(fixed("Or manually:"))
    if sys.platform == "win32":
        print(fixed("  venv\\Scripts\\activate"))
    else:
        print(fixed("  source venv/bin/activate"))
    print(fixed("  python user_interface.py"))
    print(border_bot)
    print("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Commands that pip/setuptools passes when using setup.py as a build backend.
# Detect these and delegate to setuptools so our interactive bootstrap does
# not re-run inside a pip subprocess and cause infinite recursion.
_LEGACY_BUILD_COMMANDS = frozenset([
    "egg_info", "dist_info", "develop", "install", "bdist_wheel",
    "sdist", "build", "build_ext", "build_py", "build_clib",
    "build_scripts", "clean", "check", "config", "install_lib",
    "install_headers", "install_scripts", "install_data",
    "install_egg_info", "rotate", "saveopts", "setopt",
    "upload", "register", "alias", "easy_install", "test",
])


def main():
    # When pip/setuptools calls `python setup.py egg_info` (legacy build
    # fallback), delegate to setuptools rather than running our bootstrap.
    if len(sys.argv) > 1 and sys.argv[1] in _LEGACY_BUILD_COMMANDS:
        try:
            import setuptools
            setuptools.setup()
        except Exception as exc:
            sys.stderr.write("setuptools error: {0}\n".format(exc))
            sys.exit(1)
        return

    print("=" * 50)
    print(" VIEB Environment Setup")
    print("=" * 50)

    # Part 1 & 2: detect and select interpreter
    interpreters = find_python_interpreters()
    version_str, python_cmd = select_interpreter(interpreters)

    # Part 3: create venv
    venv_python = create_venv(python_cmd)

    # Part 4: install dependencies
    status = install_dependencies(venv_python)

    # Part 5: write launch script
    write_launch_script()

    # Part 6: print summary
    print_summary(version_str, status)


if __name__ == "__main__":
    main()
