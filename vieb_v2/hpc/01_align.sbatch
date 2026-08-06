#!/bin/bash
# =============================================================================
# VIEB v2 -- stage 1 of 2: alignment (CPU only).
#
# Pose loading, dropping tail_tip, and weighted Procrustes alignment. No GPU
# benefit, so this runs on a CPU partition rather than holding a GPU
# allocation idle. Writes aligned.npz, which stage 2 consumes.
#
# Submit via submit.sh -- OUT_DIR must be exported, and submit.sh is the one
# place it is decided.
# =============================================================================
#SBATCH --job-name=vieb2-align
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# #SBATCH --account=<FILL IN IF YOUR SITE REQUIRES ONE>

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/vieb}"
POSE_DIR="${POSE_DIR:-$HOME/dlc-training/raw_videos}"
: "${OUT_DIR:?OUT_DIR must be set -- submit via ./submit.sh}"

module purge
module load python/3.11.4
VENV="${VENV:-$HOME/vieb/venv}"
source "$VENV/bin/activate"

cd "$REPO_DIR/vieb_v2"
mkdir -p "$OUT_DIR"

# Echoed so the value is recoverable from this job's own log, whatever way
# the job was submitted.
echo "OUT_DIR=$OUT_DIR"
echo "POSE_DIR=$POSE_DIR"
echo "node=$(hostname)  job=${SLURM_JOB_ID:-interactive}"

python -m cli align --pose "$POSE_DIR" --out "$OUT_DIR" --gpu off
