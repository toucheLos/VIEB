#!/bin/bash
# =============================================================================
# transfer-operator -- stage 1: alignment, deduplicated, keeping the SE(2) part.
#
# Differs from 01_align.sbatch in three ways that matter:
#
#   * the pose directory is deduplicated by normalized recording id. DLC exports
#     the same recording as both .h5 and .csv, and all 1,079 csv files under
#     raw_videos are duplicates of h5 files. Counting them twice double-weights
#     those recordings in the stationary measure of the transfer operator, which
#     is literally an occupancy count. Expect 4,925 files -> 3,846 recordings.
#   * it writes pose_frame.npz -- arena position and heading, which align_all
#     computes and discards. Nothing downstream can recover them by delay
#     embedding, because they were subtracted before measurement.
#   * it writes recordings.csv, so a row of any later checkpoint can be traced
#     back to a video. Until now the only identifier was position in a sorted
#     glob.
#
# CPU only. Alignment itself is ~90 s; the wall time is entirely reading 405 GB
# of pose off shared storage.
#
# Submit via to_submit.sh -- OUT_DIR must be exported, and that is the one place
# it is decided.
# =============================================================================
#SBATCH --job-name=vieb2-to-align
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# #SBATCH --account=<FILL IN IF YOUR SITE REQUIRES ONE>

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/vieb}"
POSE_DIR="${POSE_DIR:-$HOME/dlc-training/raw_videos}"
: "${OUT_DIR:?OUT_DIR must be set -- submit via ./to_submit.sh}"

module purge
module load python/3.11.4
VENV="${VENV:-$HOME/vieb/venv}"
source "$VENV/bin/activate"

cd "$REPO_DIR/vieb_v2"
mkdir -p "$OUT_DIR"

echo "OUT_DIR=$OUT_DIR"
echo "POSE_DIR=$POSE_DIR"
echo "node=$(hostname)  job=${SLURM_JOB_ID:-interactive}"
echo "python=$(which python3)"

python -m cli align --pose "$POSE_DIR" --out "$OUT_DIR" --gpu off
