#!/bin/bash
# =============================================================================
# Submit the transfer-operator falsification gate.
#
#     OUT_DIR=$HOME/vieb2-results/to_align_20260807_203030 ./to_submit_timescales.sh
#     N_STATES=1000 OUT_DIR=... ./to_submit_timescales.sh
#
# OUT_DIR must be a directory that already holds aligned.npz and pose_frame.npz
# from to_01_align.sbatch -- this stage reads both and writes its results back
# beside them.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

export REPO_DIR="${REPO_DIR:-$HOME/vieb}"
export VENV="${VENV:-$HOME/vieb/venv}"
: "${OUT_DIR:?OUT_DIR must be set to the directory holding aligned.npz}"
export OUT_DIR

export FPS="${FPS:-30}"
export N_STATES="${N_STATES:-500}"
export N_BOOT="${N_BOOT:-200}"
export N_TAU="${N_TAU:-28}"

for f in aligned.npz pose_frame.npz; do
    if [ ! -f "$OUT_DIR/$f" ]; then
        echo "error: $OUT_DIR/$f not found -- run to_01_align.sbatch first" >&2
        exit 2
    fi
done

echo "OUT_DIR   = $OUT_DIR"
echo "N_STATES  = $N_STATES   N_BOOT = $N_BOOT   N_TAU = $N_TAU   FPS = $FPS"
echo

jid=$(sbatch --parsable --export=ALL "$HERE/to_02_timescales.sbatch")
echo "job  to-timescales  $jid  (partition normal)"

echo
echo "watch:   squeue -u \$USER"
echo "log:     vieb2-to-timescales-$jid.out"
echo "results: $OUT_DIR/timescales_channels.json"
echo "         $OUT_DIR/timescales_pose_only.json"
