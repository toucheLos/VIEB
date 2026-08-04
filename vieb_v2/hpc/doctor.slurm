#!/bin/bash
#SBATCH --job-name=vieb2-doctor
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=%x-%j.out
# --account=<FILL IN IF REQUIRED>
#
# Runs on the gpu partition on purpose: nvidia-smi is not present on a login
# node, so a doctor run anywhere else cannot report a driver version and
# therefore cannot recommend a RAPIDS stack.
#
# Deliberately uses the DEFAULT venv, not the GPU one -- this reports what is
# currently installed against what the driver could support, which is the
# question you ask before running install_gpu.slurm.

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/vieb}"
module purge
source ~/vieb/venv/bin/activate
cd "$REPO_DIR/vieb_v2"
# --json prints the human-readable report as well, then the machine-readable one.
python -m cli doctor --json
