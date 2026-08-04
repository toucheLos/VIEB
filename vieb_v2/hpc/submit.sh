#!/bin/bash
#
# Submit the full v2 pipeline: align (CPU) -> compare-latents (GPU).
#
#     hpc/submit.sh
#
# Run 'sbatch hpc/install_gpu.slurm' once first -- job 2 needs the GPU venv it
# builds. That install is deliberately not chained in here: it would put a
# gpu-partition job on the critical path of every submission to redo work that
# only changes when the driver does.

set -euo pipefail
HPC_DIR="$(cd "$(dirname "$0")" && pwd)"

export REPO_DIR="${REPO_DIR:-$HOME/vieb}"
export POSE_DIR="${POSE_DIR:-$HOME/dlc-training/raw_videos}"
export OUT_DIR="${OUT_DIR:-$HOME/vieb2-results/run_$(date +%Y%m%d_%H%M%S)}"

# Fail here rather than letting job 2 sit in the gpu queue only to exit
# immediately once it starts.
if [ ! -f "$HPC_DIR/.gpu_venv" ]; then
    echo "no GPU venv recorded at $HPC_DIR/.gpu_venv" >&2
    echo "run 'sbatch hpc/install_gpu.slurm' once before submitting" >&2
    exit 1
fi

echo "OUT_DIR=$OUT_DIR"
echo "GPU_VENV=$(cat "$HPC_DIR/.gpu_venv")"
mkdir -p "$OUT_DIR"

jid1=$(sbatch --parsable --export=ALL "$HPC_DIR/01_align.slurm")
echo "job 1 (align, normal):             $jid1"

jid2=$(sbatch --parsable --export=ALL --dependency=afterok:$jid1 \
    "$HPC_DIR/02_compare_latents.slurm")
echo "job 2 (compare-latents, gpu):      $jid2  (waits on $jid1)"
