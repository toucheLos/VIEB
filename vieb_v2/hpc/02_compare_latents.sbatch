#!/bin/bash
# =============================================================================
# VIEB v2 -- stage 2 of 2: both latent spaces + HDBSCAN, on a GPU.
#
# Reads aligned.npz from stage 1, then for each of PCA and diffusion maps:
# reduce -> delay embed -> HDBSCAN, and prints the two side by side.
#
# The GPU accelerates HDBSCAN only. PCA is a 14x14 eigenproblem and the
# diffusion map runs on CPU, so the speedup is in the two clustering runs.
#
# Re-runnable on its own against an existing OUT_DIR, e.g. to try a different
# min_cluster_size without repeating alignment:
#     OUT_DIR=<existing> sbatch --export=ALL 02_compare_latents.sbatch
# =============================================================================
#SBATCH --job-name=vieb2-compare-latents
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# #SBATCH --account=<FILL IN IF YOUR SITE REQUIRES ONE>

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/vieb}"
: "${OUT_DIR:?OUT_DIR must be set -- submit via ./submit.sh}"

MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-50}"
N_LAGS="${N_LAGS:-4}"
LAG_STRIDE="${LAG_STRIDE:-2}"
ALPHA="${ALPHA:-1.0}"
N_LANDMARKS="${N_LANDMARKS:-3000}"
HDBSCAN_SAMPLE="${HDBSCAN_SAMPLE:-300000}"
GPU="${GPU:-on}"

module purge
module load python/3.11.4
source "$HOME/vieb/venv/bin/activate"

cd "$REPO_DIR/vieb_v2"

echo "OUT_DIR=$OUT_DIR"
echo "node=$(hostname)  job=${SLURM_JOB_ID:-interactive}"

python -m cli doctor

python -m cli compare-latents \
    --out              "$OUT_DIR" \
    --n-lags           "$N_LAGS" \
    --lag-stride       "$LAG_STRIDE" \
    --alpha            "$ALPHA" \
    --n-landmarks      "$N_LANDMARKS" \
    --min-cluster-size "$MIN_CLUSTER_SIZE" \
    --hdbscan-sample   "$HDBSCAN_SAMPLE" \
    --gpu              "$GPU" \
    --json             "$OUT_DIR/latent_comparison.json"

echo "done -- artifacts in $OUT_DIR"
ls -lh "$OUT_DIR"
