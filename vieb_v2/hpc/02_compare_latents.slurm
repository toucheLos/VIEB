#!/bin/bash
#SBATCH --job-name=vieb2-compare-latents
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=%x-%j.out
# --account=<FILL IN IF REQUIRED>
#
# Both latents from one alignment: PCA and diffusion maps, each delay-embedded
# and clustered with identical HDBSCAN settings, printed side by side.
#
# This is the stage worth a GPU. HDBSCAN fits on a subsample and then labels
# every frame -- 28.6M of them on the current project -- while the latent step
# itself is a 14x14 eigenproblem that gains nothing from a device.

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/vieb}"
: "${OUT_DIR:?OUT_DIR must be set -- submit via submit.sh}"

GPU_VENV_MARKER="$REPO_DIR/vieb_v2/hpc/.gpu_venv"
if [ ! -f "$GPU_VENV_MARKER" ]; then
    echo "no GPU venv recorded at $GPU_VENV_MARKER" >&2
    echo "run 'sbatch hpc/install_gpu.slurm' once before submitting" >&2
    exit 1
fi
GPU_VENV="$(cat "$GPU_VENV_MARKER")"

module purge
module load python/3.11.4
source "$GPU_VENV/bin/activate"
cd "$REPO_DIR/vieb_v2"

nvidia-smi || true

# --gpu on, not auto: a silent CPU fallback would spend this whole 36h GPU
# allocation doing what the normal partition could have done. gpu.resolve()
# raises in the first second instead.
python -m cli compare-latents --out "$OUT_DIR" \
    --min-cluster-size 50 --gpu on \
    --json "$OUT_DIR/latent_comparison.json"
