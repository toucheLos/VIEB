#!/bin/bash
# =============================================================================
# VIEB v2 -- Koopman attractor topology: behavioral states as basins.
#
# Reads scores.npz from OUT_DIR (write it with latent.sbatch first) and emits
# koopman_labels.npz beside it, in the same three-array shape labels.npz uses.
# No clustering runs: the state count is an output, not a parameter.
#
#     OUT_DIR=~/vieb2-results/koopman_pca sbatch --export=ALL koopman.sbatch
#
# N_REGIONS is the one free parameter that could manufacture the state count,
# so sweep it and check whether the counts move. Give each arm its own OUT_DIR
# with scores.npz symlinked in -- koopman_labels.npz has a fixed name, so arms
# sharing one OUT_DIR would overwrite each other's labels and registry:
#     for n in 96 192; do
#         d=~/vieb2-results/koopman_pca_r$n
#         mkdir -p "$d" && ln -sf ~/vieb2-results/koopman_pca/scores.npz "$d/"
#         OUT_DIR=$d N_REGIONS=$n sbatch --export=ALL koopman.sbatch
#     done
#
# No --gres: the GPU accelerates HDBSCAN only, and none is run here. The
# memory ask is for the global DMD, which holds X, X' and Vh at 9 x 28.6M
# float64 and peaks in the residual step.
# =============================================================================
#SBATCH --job-name=vieb2-koopman
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# #SBATCH --account=<FILL IN IF YOUR SITE REQUIRES ONE>

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/vieb}"
N_REGIONS="${N_REGIONS:-48}"
FPS="${FPS:-30.0}"
KNN="${KNN:-12}"
KNN_SAMPLE="${KNN_SAMPLE:-1000000}"
V_PERCENTILE="${V_PERCENTILE:-25.0}"
COHERENCE_TOL="${COHERENCE_TOL:-0.5}"
MIN_EDGE_FRAC="${MIN_EDGE_FRAC:-0.05}"
MIN_EDGE_COUNT="${MIN_EDGE_COUNT:-3}"
SEED="${SEED:-0}"
: "${OUT_DIR:?OUT_DIR must be set -- see the header of this file}"
JSON_OUT="${JSON_OUT:-$OUT_DIR/koopman_report_r${N_REGIONS}.json}"

module purge
module load python/3.11.4
VENV="${VENV:-$HOME/vieb/venv}"
source "$VENV/bin/activate"

cd "$REPO_DIR/vieb_v2"

# Echoed so the value is recoverable from this job's own log, whatever way
# the job was submitted.
echo "OUT_DIR=$OUT_DIR"
echo "N_REGIONS=$N_REGIONS"
echo "JSON_OUT=$JSON_OUT"
echo "node=$(hostname)  job=${SLURM_JOB_ID:-interactive}"

python -m cli koopman \
    --out "$OUT_DIR" \
    --fps "$FPS" \
    --n-regions "$N_REGIONS" \
    --v-percentile "$V_PERCENTILE" \
    --coherence-tol "$COHERENCE_TOL" \
    --min-edge-frac "$MIN_EDGE_FRAC" \
    --min-edge-count "$MIN_EDGE_COUNT" \
    --knn "$KNN" \
    --knn-sample "$KNN_SAMPLE" \
    --seed "$SEED" \
    --json "$JSON_OUT"
