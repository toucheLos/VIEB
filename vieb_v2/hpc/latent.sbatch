#!/bin/bash
# =============================================================================
# VIEB v2 -- one latent space, checkpointed to disk.
#
# `compare-latents` builds both latents in memory and never writes scores.npz;
# only this stage does. Koopman reads scores.npz, so it needs the checkpoint to
# exist -- hence a latent-only job rather than a rerun of 02_compare_latents.
#
# Reads aligned.npz from OUT_DIR, writes scores.npz beside it. Submit once per
# latent, into a separate OUT_DIR each time so both checkpoints survive:
#     OUT_DIR=~/vieb2-results/koopman_pca       LATENT_METHOD=pca \
#         sbatch --export=ALL latent.sbatch
#     OUT_DIR=~/vieb2-results/koopman_diffusion LATENT_METHOD=diffusion \
#         sbatch --export=ALL latent.sbatch
#
# No --gres: the GPU accelerates HDBSCAN only. PCA is a 14x14 eigenproblem and
# the diffusion map runs on CPU, so this stage gains nothing from the gpu queue.
# =============================================================================
#SBATCH --job-name=vieb2-latent
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# #SBATCH --account=<FILL IN IF YOUR SITE REQUIRES ONE>

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/vieb}"
LATENT_METHOD="${LATENT_METHOD:-pca}"
ALPHA="${ALPHA:-1.0}"
N_LANDMARKS="${N_LANDMARKS:-3000}"
VAR_THRESHOLD="${VAR_THRESHOLD:-0.95}"
: "${OUT_DIR:?OUT_DIR must be set -- see the header of this file}"

module purge
module load python/3.11.4
VENV="${VENV:-$HOME/vieb/venv}"
source "$VENV/bin/activate"

cd "$REPO_DIR/vieb_v2"

# Echoed so the value is recoverable from this job's own log, whatever way
# the job was submitted.
echo "OUT_DIR=$OUT_DIR"
echo "LATENT_METHOD=$LATENT_METHOD"
echo "node=$(hostname)  job=${SLURM_JOB_ID:-interactive}"

python -m cli latent \
    --out "$OUT_DIR" \
    --latent-method "$LATENT_METHOD" \
    --var-threshold "$VAR_THRESHOLD" \
    --alpha "$ALPHA" \
    --n-landmarks "$N_LANDMARKS" \
    --gpu off
