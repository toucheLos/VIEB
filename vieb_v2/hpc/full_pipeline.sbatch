#!/bin/bash
# =============================================================================
# VIEB v2 -- entire pipeline in one GPU job, comparing both latent spaces.
#
#   pose -> align -> [PCA | diffusion] -> delay embed -> HDBSCAN -> comparison
#
# Submit:
#     sbatch full_pipeline.sbatch
#     OUT_DIR=$HOME/vieb2-results/my-run sbatch full_pipeline.sbatch
#
# GPU applies to HDBSCAN only. PCA is a 14x14 eigenproblem and the diffusion
# map's kernel/eigh/Nystrom run on CPU, so the GPU accelerates the two
# clustering runs, not the latent construction. Run doctor.sbatch first if you
# have not confirmed cuml works on this cluster.
# =============================================================================
#SBATCH --job-name=vieb2-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# #SBATCH --account=<FILL IN IF YOUR SITE REQUIRES ONE>
# #SBATCH --qos=<FILL IN IF YOUR SITE REQUIRES ONE>

set -euo pipefail

# ── Paths (override by exporting before sbatch) ─────────────────────────────
REPO_DIR="${REPO_DIR:-$HOME/vieb}"
POSE_DIR="${POSE_DIR:-$HOME/dlc-training/raw_videos}"
OUT_DIR="${OUT_DIR:-$HOME/vieb2-results/run_$(date +%Y%m%d_%H%M%S)}"

# ── Pipeline parameters ─────────────────────────────────────────────────────
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-50}"
N_LAGS="${N_LAGS:-4}"
LAG_STRIDE="${LAG_STRIDE:-2}"
ALPHA="${ALPHA:-1.0}"
N_LANDMARKS="${N_LANDMARKS:-3000}"
HDBSCAN_SAMPLE="${HDBSCAN_SAMPLE:-300000}"
# Which method gets a full set of checkpoints written in step 3.
LATENT_METHOD="${LATENT_METHOD:-pca}"
# 'on' fails immediately if the GPU backend is unusable, rather than quietly
# spending a GPU allocation on CPU work. Set GPU=auto to allow CPU fallback.
GPU="${GPU:-on}"

# ── Environment ─────────────────────────────────────────────────────────────
module purge
module load python/3.11.4
# The RAPIDS stack lives in a dedicated python/3.11.4 venv (#53); the default
# venv is 3.13 and CPU-only, so sourcing it turns this GPU allocation into a
# single-core CPU run (#58). --gpu on catches that in the first second, but a
# missing venv deserves a better message than an import error.
VENV="${VENV:-$HOME/vieb/venv-gpu}"
if [ ! -x "$VENV/bin/python" ]; then
    echo "error: GPU venv not found: $VENV" >&2
    echo "  build it on a GPU node -- see hpc/README.md 'Quick start' step 2" >&2
    exit 2
fi
source "$VENV/bin/activate"

cd "$REPO_DIR/vieb_v2"
mkdir -p "$OUT_DIR"

echo "======================================================================"
echo "OUT_DIR   = $OUT_DIR"
echo "POSE_DIR  = $POSE_DIR"
echo "REPO      = $REPO_DIR  ($(git -C "$REPO_DIR" rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout'))"
echo "node      = $(hostname)   job = ${SLURM_JOB_ID:-interactive}"
echo "params    = min_cluster_size=$MIN_CLUSTER_SIZE n_lags=$N_LAGS"
echo "            lag_stride=$LAG_STRIDE alpha=$ALPHA landmarks=$N_LANDMARKS"
echo "            latent_method=$LATENT_METHOD (gets full checkpoints)"
echo "======================================================================"

# ── Preflight ───────────────────────────────────────────────────────────────
# Reports the driver, whether cuml actually runs, and -- if not -- the exact
# pip command for this driver. Cheap, and it runs on the GPU node itself,
# which is the only node whose answer matters.
echo; echo "--- doctor -----------------------------------------------------"
python -m cli doctor

# ── 1. Align (CPU-bound: pose load, drop tail_tip, weighted Procrustes) ─────
echo; echo "--- 1/2  align -------------------------------------------------"
python -m cli align \
    --pose "$POSE_DIR" \
    --out  "$OUT_DIR" \
    --gpu  off

# ── 2. Both latents from that one alignment, then HDBSCAN on each ───────────
# Sharing a single alignment is the point: the two arms then differ only in
# the reducer, so any metric difference is attributable to that.
echo; echo "--- 2/3  compare-latents (PCA vs diffusion) ---------------------"
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

# ── 3. Materialise a complete run for one method ────────────────────────────
# compare-latents holds both arms in memory and persists only the comparison --
# it cannot write scores/embedded/labels without ambiguity about which arm they
# belong to. So finish the pipeline for LATENT_METHOD, reusing the alignment
# from step 1 rather than starting over. This is what leaves a full set of
# checkpoints the GUI and `cli sweep` can pick up.
echo; echo "--- 3/3  full run [$LATENT_METHOD] -----------------------------"
python -m cli latent \
    --out              "$OUT_DIR" \
    --latent-method    "$LATENT_METHOD" \
    --alpha            "$ALPHA" \
    --n-landmarks      "$N_LANDMARKS"

python -m cli embed \
    --out              "$OUT_DIR" \
    --n-lags           "$N_LAGS" \
    --lag-stride       "$LAG_STRIDE"

python -m cli cluster \
    --out              "$OUT_DIR" \
    --min-cluster-size "$MIN_CLUSTER_SIZE" \
    --hdbscan-sample   "$HDBSCAN_SAMPLE" \
    --gpu              "$GPU" \
    --json             "$OUT_DIR/metrics.json"

echo
echo "======================================================================"
echo "done -- artifacts in $OUT_DIR"
ls -lh "$OUT_DIR"
echo "======================================================================"
