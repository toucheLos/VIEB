#!/bin/bash
# =============================================================================
# transfer-operator -- stage 2: implied timescales. The falsification gate.
#
# Builds the Ulam matrix on a Voronoi partition of the observation space and
# sweeps the lag. The decision rule is fixed before the run:
#
#   plateau in t_imp over some lag window -> timescale separation exists, the
#       Chapman-Kolmogorov condition is approximately satisfied, continue.
#   t_imp grows linearly in tau from the start with no plateau anywhere ->
#       there is no Markovian coarse-graining at any resolution on this data.
#       The branch dies here. Report it and stop; do not tune.
#
# Two arms, so the Stage 0 channel restoration is itself under test:
#   channels   pose PCs + centroid_speed + angular_velocity  (the branch)
#   pose-only  pose PCs alone                                (the control)
#
# If the two arms give the same answer, restoring the channels did not matter
# for Markovianity and that is worth knowing. If only the channels arm
# plateaus, the v2 representation was the binding constraint.
#
# CPU only. Exit 1 means "ran, but no plateau" -- a finding, not a failure.
#
# Submit via to_submit_timescales.sh -- OUT_DIR must be exported.
# =============================================================================
#SBATCH --job-name=vieb2-to-timescales
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# #SBATCH --account=<FILL IN IF YOUR SITE REQUIRES ONE>

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/vieb}"
: "${OUT_DIR:?OUT_DIR must be set -- submit via ./to_submit_timescales.sh}"

FPS="${FPS:-30}"
N_STATES="${N_STATES:-500}"
N_BOOT="${N_BOOT:-200}"
N_TAU="${N_TAU:-28}"

module purge
module load python/3.11.4
VENV="${VENV:-$HOME/vieb/venv}"
source "$VENV/bin/activate"

cd "$REPO_DIR/vieb_v2"

echo "OUT_DIR=$OUT_DIR"
echo "node=$(hostname)  job=${SLURM_JOB_ID:-interactive}"
echo "python=$(which python3)"
echo "N_STATES=$N_STATES  N_BOOT=$N_BOOT  N_TAU=$N_TAU  FPS=$FPS"
echo

# `set -e` would abandon the control arm if the main arm exits 1, and exit 1
# here means "no plateau", which is exactly the result worth comparing across
# arms. Capture each status instead and decide at the end.
rc_channels=0
echo "=== arm 1/2: pose PCs + restored channels ==="
python -m cli timescales --out "$OUT_DIR" --fps "$FPS" \
    --n-states "$N_STATES" --n-boot "$N_BOOT" --n-tau "$N_TAU" \
    --tag channels || rc_channels=$?

rc_pose=0
echo
echo "=== arm 2/2: pose PCs only (control) ==="
python -m cli timescales --out "$OUT_DIR" --fps "$FPS" \
    --n-states "$N_STATES" --n-boot "$N_BOOT" --n-tau "$N_TAU" \
    --pose-only --tag pose_only || rc_pose=$?

echo
echo "exit: channels=$rc_channels pose_only=$rc_pose  (1 = ran, no plateau)"
[ "$rc_channels" -le 1 ] && [ "$rc_pose" -le 1 ]
