#!/bin/bash
#SBATCH --job-name=moseq_control
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output="logs/moseq_control_%j.out"
#SBATCH --error="logs/moseq_control_%j.err"
#
# Section 8 positive control: does any MoSeq syllable shift in Context A
# post-shock? CPU only -- no GPU is requested and none is needed.
#
# h5py must already be installed. Install it on the LOGIN node; compute nodes
# have no outbound internet:
#     module load python/3.11.4
#     source "${VIEB_VENV}/bin/activate"
#     pip install h5py
#
# Every path below is quoted: recording filenames contain '(', ')' and ','.

set -euo pipefail

# module load FIRST, then activate. venv activation does not survive a module
# change on this cluster, and the failure is silent -- you get the system
# python with none of your packages.
module load python/3.11.4
source "${VIEB_VENV:?set VIEB_VENV to the virtualenv root}/bin/activate"

# Verify before trusting anything.
echo "which python3 : $(which python3)"
echo "version       : $(python3 --version)"
python3 - <<'PY'
import sys
try:
    import h5py, scipy, numpy
except ImportError as exc:
    sys.exit(f"missing dependency: {exc}. pip install it on the LOGIN node.")
print(f"h5py {h5py.__version__} / scipy {scipy.__version__} / numpy {numpy.__version__}")
PY

MOSEQ_RESULTS="${MOSEQ_RESULTS:?set MOSEQ_RESULTS to the results.h5 path}"
METADATA="${METADATA:?set METADATA to metadata.csv}"
OUT="${OUT:-results/transfer/moseq_control.json}"

mkdir -p "$(dirname "${OUT}")" logs

python3 "$(dirname "${BASH_SOURCE[0]}")/../transfer/stage8_moseq_control.py" \
    --results  "${MOSEQ_RESULTS}" \
    --metadata "${METADATA}" \
    --context  "${CONTEXT:-A}" \
    --shock-day "${SHOCK_DAY:-0}" \
    --out      "${OUT}"

echo "wrote ${OUT}"
