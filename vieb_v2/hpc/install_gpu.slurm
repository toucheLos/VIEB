#!/bin/bash
#SBATCH --job-name=vieb2-install-gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
# --account=<FILL IN IF REQUIRED>
#
# One-time GPU environment setup. Run this once (and again only if the node's
# NVIDIA driver changes), NOT as part of every pipeline submission -- that is
# why it is not chained into submit.sh.
#
#     sbatch hpc/install_gpu.slurm
#
# It builds a SEPARATE venv from the default ~/vieb/venv. The default one is
# Python 3.13, and the older of the two pinned RAPIDS stacks publishes no 3.13
# wheels at all (cuml-cu12 24.12.0 is a source tarball only), so which stack the
# driver resolves to would decide whether the install worked. A dedicated
# python/3.11.4 venv makes that irrelevant: both stacks have 3.11 wheels.
#
# Runs on the gpu partition because nvidia-smi -- and therefore the driver
# version the stack is chosen from -- is not visible from a login node.

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/vieb}"
GPU_VENV="${GPU_VENV:-$HOME/vieb/venv-gpu}"
HPC_DIR="$REPO_DIR/vieb_v2/hpc"

module purge
module load python/3.11.4

echo "=== building $GPU_VENV on $(python --version 2>&1) ==="
[ -d "$GPU_VENV" ] || python -m venv "$GPU_VENV"
source "$GPU_VENV/bin/activate"
python -m pip install --upgrade pip

# Base pipeline deps. pyproject pins requires-python <3.13, which this venv
# satisfies and the default 3.13 one does not.
python -m pip install -e "$REPO_DIR"

cd "$REPO_DIR/vieb_v2"

# Ask the CLI which RAPIDS stack this node's driver can load. Keeping the
# driver table in gpu.py (one place) rather than duplicating it in bash is the
# whole point of `doctor --print-packages`; it needs only the stdlib, so it
# answers correctly before cuml/cupy exist.
echo "=== selecting RAPIDS stack for the detected driver ==="
# Via a temp file, not a pipe: `mapfile < <(...)` would hide a non-zero exit
# from the process substitution and quietly leave PACKAGES empty.
PACKAGES_FILE="$(mktemp)"
trap 'rm -f "$PACKAGES_FILE"' EXIT
python -m cli doctor --print-packages > "$PACKAGES_FILE" || {
    echo "no RAPIDS stack matches the detected driver -- see the message above" >&2
    exit 1
}
mapfile -t PACKAGES < "$PACKAGES_FILE"

if [ ${#PACKAGES[@]} -eq 0 ]; then
    echo "doctor --print-packages returned nothing" >&2
    exit 1
fi

printf '  %s\n' "${PACKAGES[@]}"
python -m pip install "${PACKAGES[@]}"

# Marker read by 02_compare_latents.slurm and checked by submit.sh, so a GPU
# job fails at submit time rather than after waiting in the queue.
echo "$GPU_VENV" > "$HPC_DIR/.gpu_venv"
echo "=== wrote $HPC_DIR/.gpu_venv ==="

# Confirm the install actually runs on this hardware. gpu.py probes by fitting
# a tiny model, so "hdbscan_gpu": true here means the real job will work.
echo "=== verifying ==="
python -m cli doctor --json
