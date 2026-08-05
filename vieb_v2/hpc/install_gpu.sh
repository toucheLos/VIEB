#!/bin/bash
# =============================================================================
# Install the RAPIDS stack matching this machine's NVIDIA driver.
#
# Run on a GPU node, so the driver being queried is the one the jobs will use:
#     srun --partition=gpu --gres=gpu:1 --pty bash
#     ./install_gpu.sh
#
# Why not just `pip install cuml cupy`: that fails twice over. `cuml` on plain
# PyPI is an unrelated abandoned package (0.6.1.post1), and bare `cupy` has no
# prebuilt wheel, so pip falls back to a source build that needs a full CUDA
# dev toolkit (cublas_v2.h, nvcc) which compute nodes generally lack. The
# suffixed wheels from pypi.nvidia.com are prebuilt and bundle their runtimes.
# =============================================================================
set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/vieb}"

module purge
module load python/3.11.4
source "$HOME/vieb/venv/bin/activate"

cd "$REPO_DIR/vieb_v2"

py_version=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "venv python: $py_version"

case "$py_version" in
    3.10|3.11|3.12) ;;
    *)
        echo
        echo "WARNING: RAPIDS wheels may not exist for Python $py_version."
        echo "A venv is bound to the Python that created it, so 'module load"
        echo "python/3.11.4' does not change an existing venv. To rebuild:"
        echo
        echo "    module purge && module load python/3.11.4"
        echo "    rm -rf $HOME/vieb/venv"
        echo "    python -m venv $HOME/vieb/venv"
        echo "    source $HOME/vieb/venv/bin/activate"
        echo "    pip install numpy pandas tables hdbscan"
        echo
        read -r -p "Continue anyway? [y/N] " reply
        [ "$reply" = "y" ] || exit 1
        ;;
esac

# Ask doctor what this driver supports rather than hardcoding a guess.
cmd=$(python - <<'PY'
import sys
sys.path.insert(0, ".")
from representation import gpu
print(gpu.install_command() or "")
PY
)

if [ -z "$cmd" ]; then
    echo "No NVIDIA driver detected -- are you on a GPU node?"
    echo "  srun --partition=gpu --gres=gpu:1 --pty bash"
    exit 2
fi

echo
echo "running: $cmd"
eval "$cmd"

echo
echo "verifying..."
python -m cli doctor
