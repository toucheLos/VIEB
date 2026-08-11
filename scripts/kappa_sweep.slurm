#!/bin/bash
#SBATCH --job-name=vieb_kappa
#SBATCH --output=/home/tul26194/vieb-logs/kappa_%j.out
#SBATCH --error=/home/tul26194/vieb-logs/kappa_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

# Sweep the sticky HMM's kappa on the existing baseline.
#
# Unlike the HSMM arm this is NOT blocked on `representation-repair`: it refits
# the baseline exactly as it already runs and varies one hyperparameter. It is
# the cheapest possible improvement to the current best method, and it tests
# whether the 42.35% dominant syllable is a duration artifact -- which is this
# branch's hypothesis, approached from the other side.
#
#   sbatch scripts/kappa_sweep.slurm
#   sbatch scripts/kappa_sweep.slurm "1e5,1e6,1e7,1e8"

set -euo pipefail

module purge
module load python/3.11.4

VENV="/home/tul26194/moseq/venv-moseq"
REPO="/home/tul26194/vieb"

LD_LIBRARY_PATH="$(find "$VENV/lib/python3.11/site-packages/nvidia" -maxdepth 2 -type d -name lib | tr '\n' ':')${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH
# shellcheck disable=SC1091
source "$VENV/bin/activate"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}"

echo "python3: $(which python3)"
python3 -c "import sys; assert sys.prefix == '$VENV', f'wrong interpreter: {sys.prefix}'"
python3 -c "import jax; print('JAX devices:', jax.devices())"

mkdir -p /home/tul26194/vieb-logs /home/tul26194/vieb-runs

KAPPAS="${1:-1e5,1e6,1e7,1e8}"

python3 "$REPO/scripts/kappa_sweep.py" \
    --project "/home/tul26194/moseq/luna_demo" \
    --results "/home/tul26194/moseq/luna_demo/2026_07_26-19_54_24/results" \
    --kappa "$KAPPAS" \
    --num-iters 200 \
    --seed 0 \
    --fit-sample-size 100 \
    --out "/home/tul26194/vieb-runs/kappa_sweep.json"
