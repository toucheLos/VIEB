#!/bin/bash
#SBATCH --job-name=vieb_hsmm
#SBATCH --output=/home/tul26194/vieb-logs/hsmm_%j.out
#SBATCH --error=/home/tul26194/vieb-logs/hsmm_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

# The explicit-duration AR-HSMM arm.
#
# BLOCKED at time of writing: this runs the model on the MoSeq latent space as
# kpms fitted it, which carries no locomotion (decision #60). Until
# `representation-repair` reports and locomotor channels exist, a beautifully
# calibrated dwell-time model here may be describing one undifferentiated state.
# The §4 synthetic gate is what is verified; this is the harness for §5, staged
# and ready, not a result.
#
#   sbatch scripts/hsmm.slurm            # full fit
#   sbatch scripts/hsmm.slurm 6 50       # smoke: 6 recordings, 50 iterations

set -euo pipefail

module purge
module load python/3.11.4

VENV="/home/tul26194/moseq/venv-moseq"
REPO="/home/tul26194/vieb"

# jaxlib finds its CUDA libraries through the nvidia wheels, not the module tree.
LD_LIBRARY_PATH="$(find "$VENV/lib/python3.11/site-packages/nvidia" -maxdepth 2 -type d -name lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH
# shellcheck disable=SC1091
source "$VENV/bin/activate"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

# Verify the interpreter before trusting anything downstream: `module load` after
# `source activate` silently wins, and the failure surfaces much later as a
# missing package.
echo "python3: $(which python3)"
python3 -c "import sys; assert sys.prefix == '$VENV', f'wrong interpreter: {sys.prefix}'"
python3 -c "import jax; print('JAX devices:', jax.devices())"

mkdir -p /home/tul26194/vieb-logs /home/tul26194/vieb-runs

LIMIT="${1:-}"
ITERS="${2:-500}"

ARGS=(--num-iters "$ITERS" --store "/home/tul26194/vieb-runs" --device gpu)
if [ -n "$LIMIT" ]; then
    ARGS+=(--limit "$LIMIT")
fi

python3 "$REPO/scripts/run_hsmm.py" "${ARGS[@]}"

echo
echo "=== the comparison table, read back from VUS-1 only ==="
python3 "$REPO/compare_methods.py" --runs "/home/tul26194/vieb-runs"
