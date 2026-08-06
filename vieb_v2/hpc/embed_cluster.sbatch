#!/bin/bash
# =============================================================================
# VIEB v2 -- delay embed + HDBSCAN, checkpointed.
#
# `compare-latents` clusters in memory and keeps only the aggregate metrics, so
# no per-frame HDBSCAN labels survive it -- the same reason it never writes
# scores.npz. Comparing basins against states frame by frame needs labels.npz
# on disk, which only `embed` and `cluster` write. Hence this job.
#
#     OUT_DIR=~/vieb2-results/koopman_pca sbatch --export=ALL embed_cluster.sbatch
#
# Reads scores.npz from OUT_DIR (write it with latent.sbatch first) and emits
# embedded.npz + labels.npz beside it.
#
# labels.npz is NOT positionally comparable with koopman_labels.npz: the delay
# window is dropped per recording here, so this is shorter by n_recordings *
# (n_lags * lag_stride). Join the two on their shared `index` array.
# =============================================================================
#SBATCH --job-name=vieb2-embed-cluster
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
# 200G, not 256G: the gpu nodes have 256895 MB total, so a 256G ask is
# unschedulable there. The delay embedding is ~10 GB at 28.6M x 45 float64.
#SBATCH --mem=200G
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
# #SBATCH --account=<FILL IN IF YOUR SITE REQUIRES ONE>

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/vieb}"
N_LAGS="${N_LAGS:-4}"
LAG_STRIDE="${LAG_STRIDE:-2}"
FPS="${FPS:-30.0}"
MIN_CLUSTER_SIZE="${MIN_CLUSTER_SIZE:-50}"
HDBSCAN_SAMPLE="${HDBSCAN_SAMPLE:-300000}"
GPU="${GPU:-auto}"
: "${OUT_DIR:?OUT_DIR must be set -- see the header of this file}"

module purge
module load python/3.11.4
VENV="${VENV:-$HOME/vieb/venv}"
source "$VENV/bin/activate"

cd "$REPO_DIR/vieb_v2"

# Echoed so the value is recoverable from this job's own log, whatever way
# the job was submitted.
echo "OUT_DIR=$OUT_DIR"
echo "node=$(hostname)  job=${SLURM_JOB_ID:-interactive}"

python -m cli embed \
    --out "$OUT_DIR" \
    --n-lags "$N_LAGS" \
    --lag-stride "$LAG_STRIDE" \
    --fps "$FPS" \
    --gpu off

python -m cli cluster \
    --out "$OUT_DIR" \
    --min-cluster-size "$MIN_CLUSTER_SIZE" \
    --hdbscan-sample "$HDBSCAN_SAMPLE" \
    --gpu "$GPU" \
    --json "$OUT_DIR/hdbscan_report.json"
