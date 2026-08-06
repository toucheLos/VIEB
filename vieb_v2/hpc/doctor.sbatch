#!/bin/bash
# =============================================================================
# VIEB v2 -- environment preflight. Run this before any real job.
#
#   sbatch doctor.sbatch                          # CPU node
#   sbatch --partition=gpu --gres=gpu:1 doctor.sbatch   # GPU node
#
# GPU capability is probed on the node it runs on, by actually fitting a tiny
# cuml model rather than checking whether an import succeeds -- on this project
# cupy has imported fine on a machine where every linear-algebra call failed.
# So the GPU-node run is the one that tells you whether full_pipeline.sbatch
# will work.
#
# If GPU clustering is unavailable, doctor prints the exact pip command for
# this machine's driver.
# =============================================================================
#SBATCH --job-name=vieb2-doctor
#SBATCH --partition=normal
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# #SBATCH --account=<FILL IN IF YOUR SITE REQUIRES ONE>

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/vieb}"

module purge
module load python/3.11.4
VENV="${VENV:-$HOME/vieb/venv}"
source "$VENV/bin/activate"

cd "$REPO_DIR/vieb_v2"

echo "node   = $(hostname)"
echo "python = $(python -c 'import sys; print(sys.executable)')"
echo "version= $(python -c 'import sys; print(sys.version.split()[0])')"
echo

python -m cli doctor
