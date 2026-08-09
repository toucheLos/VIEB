#!/bin/bash
# =============================================================================
# Submit the two-stage pipeline: CPU alignment, then GPU clustering.
#
#     ./submit.sh
#     OUT_DIR=$HOME/vieb2-results/my-run ./submit.sh
#     MIN_CLUSTER_SIZE=100 ./submit.sh
#
# OUT_DIR is computed once here and forwarded to both jobs with
# `sbatch --export=ALL`. An earlier design had stage 1 write the path to a file
# for stage 2 to read back, which broke whenever stage 2 ran before stage 1 had
# finished writing it -- and would have raced between two concurrent runs
# submitted from this directory.
#
# For a single job that does everything instead, use full_pipeline.sbatch.
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

export REPO_DIR="${REPO_DIR:-$HOME/vieb}"
export POSE_DIR="${POSE_DIR:-$HOME/dlc-training/raw_videos}"
export OUT_DIR="${OUT_DIR:-$HOME/vieb2-results/run_$(date +%Y%m%d_%H%M%S)}"

export MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-50}"
export N_LAGS="${N_LAGS:-4}"
export LAG_STRIDE="${LAG_STRIDE:-2}"
export ALPHA="${ALPHA:-1.0}"
export N_LANDMARKS="${N_LANDMARKS:-3000}"
export HDBSCAN_SAMPLE="${HDBSCAN_SAMPLE:-300000}"
export GPU="${GPU:-on}"

if [ ! -d "$POSE_DIR" ]; then
    echo "error: pose directory not found: $POSE_DIR" >&2
    echo "  set POSE_DIR, or run DLC first: python setup_dlc_training.py --analyze" >&2
    exit 2
fi

# Stage 2 is a gpu-partition job with GPU=on, so a missing GPU venv makes it
# fail on start -- after stage 1 has already burned an alignment. Check before
# queuing anything (#53). Deliberately not exported: stage 1 is a CPU job and
# must keep using the default venv.
GPU_VENV="${VENV:-$HOME/vieb/venv-gpu}"
if [ "$GPU" != "off" ] && [ ! -x "$GPU_VENV/bin/python" ]; then
    echo "error: GPU venv not found: $GPU_VENV" >&2
    echo "  build it on a GPU node -- see hpc/README.md 'Quick start' step 2" >&2
    echo "  or submit CPU-only: GPU=off ./submit.sh" >&2
    exit 2
fi

n_pose=$(find "$POSE_DIR" -maxdepth 2 \( -name '*.h5' -o -name '*.csv' \) 2>/dev/null | wc -l)
mkdir -p "$OUT_DIR"

echo "OUT_DIR   = $OUT_DIR"
echo "POSE_DIR  = $POSE_DIR  ($n_pose candidate pose files)"
echo

jid1=$(sbatch --parsable --export=ALL "$HERE/01_align.sbatch")
echo "job 1  align            $jid1  (partition normal)"

jid2=$(sbatch --parsable --export=ALL --dependency=afterok:"$jid1" \
       "$HERE/02_compare_latents.sbatch")
echo "job 2  compare-latents  $jid2  (partition gpu, waits on $jid1)"

echo
echo "watch:   squeue -u \$USER"
echo "results: $OUT_DIR/latent_comparison.json"
