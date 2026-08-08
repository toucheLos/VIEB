#!/bin/bash
# =============================================================================
# Submit the transfer-operator alignment stage.
#
#     ./to_submit.sh
#     OUT_DIR=$HOME/vieb2-results/to-run ./to_submit.sh
#
# One CPU job, no GPU dependency -- unlike submit.sh, which chains a gpu-partition
# stage after alignment. Nothing in this branch's core path uses a GPU, by
# design: it has to run in any lab.
#
# OUT_DIR is computed once here and forwarded with `sbatch --export=ALL`, the
# same contract submit.sh uses, so two concurrent runs cannot race over a
# path written to a file.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

export REPO_DIR="${REPO_DIR:-$HOME/vieb}"
export POSE_DIR="${POSE_DIR:-$HOME/dlc-training/raw_videos}"
export OUT_DIR="${OUT_DIR:-$HOME/vieb2-results/to_$(date +%Y%m%d_%H%M%S)}"
export VENV="${VENV:-$HOME/vieb/venv}"

if [ ! -d "$POSE_DIR" ]; then
    echo "error: pose directory not found: $POSE_DIR" >&2
    echo "  set POSE_DIR, or run DLC first: python setup_dlc_training.py --analyze" >&2
    exit 2
fi

# This is a CPU job and the CPU venv is the only one it may use. venv-gpu is
# built against a different interpreter and is not interchangeable.
if [ ! -x "$VENV/bin/python3" ]; then
    echo "error: venv not found: $VENV" >&2
    exit 2
fi

n_h5=$(find "$POSE_DIR" -maxdepth 2 -name '*.h5' 2>/dev/null | wc -l)
n_csv=$(find "$POSE_DIR" -maxdepth 2 -name '*.csv' 2>/dev/null | wc -l)
mkdir -p "$OUT_DIR"

echo "OUT_DIR   = $OUT_DIR"
echo "POSE_DIR  = $POSE_DIR"
echo "            $n_h5 h5 + $n_csv csv candidates"
echo "            every csv is expected to be a duplicate of an h5 recording;"
echo "            the job deduplicates and should report 3,846 recordings."
echo

jid=$(sbatch --parsable --export=ALL "$HERE/to_01_align.sbatch")
echo "job  to-align  $jid  (partition normal)"

echo
echo "watch:   squeue -u \$USER"
echo "log:     vieb2-to-align-$jid.out"
echo "results: $OUT_DIR/aligned.npz"
echo "         $OUT_DIR/pose_frame.npz"
echo "         $OUT_DIR/recordings.csv"
