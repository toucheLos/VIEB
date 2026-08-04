#!/bin/bash
#SBATCH --job-name=vieb2-align
#SBATCH --partition=normal
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
# --account=<FILL IN IF REQUIRED>

set -euo pipefail
REPO_DIR="${REPO_DIR:-$HOME/vieb}"
POSE_DIR="${POSE_DIR:-$HOME/dlc-training/raw_videos}"
: "${OUT_DIR:?OUT_DIR must be set -- submit via submit.sh, or pass "
    "--export=ALL,OUT_DIR=/path yourself}"

# No GPU here by design: alignment and the pooled PCA are cheap enough on CPU,
# so this stays on the normal partition and uses the default venv.
module purge
source ~/vieb/venv/bin/activate
cd "$REPO_DIR/vieb_v2"
mkdir -p "$OUT_DIR"

python -m cli align --pose "$POSE_DIR" --out "$OUT_DIR" --gpu off
