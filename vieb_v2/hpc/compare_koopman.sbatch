#!/bin/bash
# =============================================================================
# VIEB v2 -- HDBSCAN states beside Koopman basins, both latents.
#
# Four arms from two out-dirs, every arm scored with the *same* metrics, and
# the two label sets joined on their shared `index` array. Nothing is
# special-cased for the method that produced the labels.
#
#     PCA_OUT=~/vieb2-results/koopman_pca \
#     DIFFUSION_OUT=~/vieb2-results/koopman_diffusion \
#     sbatch --export=ALL compare_koopman.sbatch
#
# Each OUT_DIR must already hold all four checkpoints:
#     labels.npz + embedded.npz   (embed_cluster.sbatch)
#     koopman_labels.npz + scores.npz  (latent.sbatch then koopman.sbatch)
# Arms with a missing checkpoint are skipped with a note, not an error, so a
# half-finished pair still reports what it can.
#
# The join is on `index`, never position: Koopman labels every frame of
# scores.npz while HDBSCAN labels the delay-embedded frames, which are fewer by
# one window per recording. See README.md and decision #57.
#
# No --gres: no clustering runs here. Memory is dominated by the point arrays
# the speed diagnostics need -- embedded.npz is ~10 GB at 28.6M x 45 float64 --
# but only one arm is held at a time, so the peak is one embedding plus slack.
# =============================================================================
#SBATCH --job-name=vieb2-compare-koopman
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# #SBATCH --account=<FILL IN IF YOUR SITE REQUIRES ONE>

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/vieb}"
PCA_OUT="${PCA_OUT:-$HOME/vieb2-results/koopman_pca}"
DIFFUSION_OUT="${DIFFUSION_OUT:-$HOME/vieb2-results/koopman_diffusion}"
JSON_OUT="${JSON_OUT:-$HOME/vieb2-results/koopman_comparison.json}"

module purge
module load python/3.11.4
VENV="${VENV:-$HOME/vieb/venv}"
source "$VENV/bin/activate"

cd "$REPO_DIR/vieb_v2"

# Echoed so the values are recoverable from this job's own log, whatever way
# the job was submitted.
echo "PCA_OUT=$PCA_OUT"
echo "DIFFUSION_OUT=$DIFFUSION_OUT"
echo "JSON_OUT=$JSON_OUT"
echo "node=$(hostname)  job=${SLURM_JOB_ID:-interactive}"

python -m cli compare-koopman \
    --pca-out       "$PCA_OUT" \
    --diffusion-out "$DIFFUSION_OUT" \
    --json          "$JSON_OUT"
