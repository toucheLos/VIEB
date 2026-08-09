# Running VIEB v2 on the cluster

## Quick start

```bash
cd ~/vieb/vieb_v2/hpc

# 1. one-time: CPU packages, in the default venv (used by the `normal` jobs)
source "${VENV:-~/vieb/venv}/bin/activate"
pip install numpy pandas tables hdbscan

# 2. one-time: GPU packages, on a GPU node so the right driver is queried.
# This builds a *separate* python/3.11.4 venv at ~/vieb/venv-gpu -- the default
# venv is 3.13, which RAPIDS has no wheels for. Every gpu-partition job uses it.
srun --partition=gpu --gres=gpu:1 --pty bash
./install_gpu.sh
exit

# 3. confirm the environment, on the partition the job will use
sbatch --partition=gpu --gres=gpu:1 doctor.sbatch

# 4. run everything
sbatch full_pipeline.sbatch
```

## What each script does

| script | partition | purpose |
|---|---|---|
| `doctor.sbatch` | either | preflight: packages, driver, whether cuml actually runs |
| `full_pipeline.sbatch` | `gpu` | **everything in one job**: align → both latents → embed → HDBSCAN → comparison |
| `01_align.sbatch` | `normal` | stage 1 only: alignment (no GPU benefit) |
| `02_compare_latents.sbatch` | `gpu` | stage 2 only: both latents + HDBSCAN |
| `submit.sh` | — | chains 01 → 02 with `--dependency=afterok` |
| `latent.sbatch` | `normal` | one latent, **checkpointed** to `scores.npz` (`compare-latents` never writes it) |
| `embed_cluster.sbatch` | `gpu` | delay embed + HDBSCAN, **checkpointed** to `labels.npz` (`compare-latents` never writes it either) |
| `koopman.sbatch` | `normal` | attractor topology: states as basins, no clustering |
| `compare_koopman.sbatch` | `normal` | HDBSCAN states vs Koopman basins, all four arms, joined on `index` |
| `install_gpu.sh` | — | installs the RAPIDS stack matching this driver |

**`full_pipeline.sbatch`** is the simplest: one job, one log, everything done.

**`submit.sh`** splits it so alignment runs on a CPU node instead of holding a
GPU allocation idle. Worth it if your GPU queue is contended or alignment is
slow on your dataset; otherwise the single job is easier to manage.

## What the GPU actually accelerates

**HDBSCAN, and nothing else.** Specifically:

- **PCA** — a 14×14 eigenproblem after `tail_tip` is dropped (D = 2K = 14).
  A GPU would add transfer overhead and no speedup.
- **Diffusion maps** — the landmark eigendecomposition is a dense 3000×3000
  `eigh` and the Nyström extension is batched matmul. Neither is offloaded, so
  this runs on CPU.
- **Alignment** — CPU-bound pose loading and Procrustes.
- **HDBSCAN** — millions of delay-embedded points. This is the bottleneck and
  the reason to request a GPU.

So `compare-latents --gpu on` speeds up the two clustering runs, not the latent
construction. If the GPU queue is long, `--gpu off` gives identical results
more slowly — the science does not change.

`--gpu on` fails immediately if the GPU backend is unusable rather than
silently spending the allocation on CPU. Use `GPU=auto` to allow fallback.

## Configuration

Every parameter is an environment variable, so nothing needs editing:

```bash
OUT_DIR=$HOME/vieb2-results/my-run sbatch full_pipeline.sbatch
MIN_CLUSTER_SIZE=100 N_LAGS=6 sbatch full_pipeline.sbatch
GPU=auto sbatch full_pipeline.sbatch
POSE_DIR=/scratch/$USER/pose sbatch full_pipeline.sbatch
```

| variable | default | meaning |
|---|---|---|
| `REPO_DIR` | `$HOME/vieb` | checkout containing `vieb_v2/` |
| `POSE_DIR` | `$HOME/dlc-training/raw_videos` | DLC `.h5`/`.csv` output |
| `OUT_DIR` | `$HOME/vieb2-results/run_<timestamp>` | artifacts |
| `MIN_CLUSTER_SIZE` | 50 | HDBSCAN minimum cluster size |
| `N_LAGS` / `LAG_STRIDE` | 4 / 2 | delay window (9 frames = 0.30 s at 30 fps) |
| `ALPHA` | 1.0 | diffusion density normalisation |
| `N_LANDMARKS` | 3000 | points the diffusion operator is built on |
| `HDBSCAN_SAMPLE` | 300000 | fit on this many points, label the rest by `approximate_predict` |
| `GPU` | `on` | `on` \| `auto` \| `off` |
| `VENV` | `$HOME/vieb/venv-gpu` on `gpu`, `$HOME/vieb/venv` on `normal` | python env to activate |

**Two venvs, on purpose.** `normal`-partition jobs (`01_align`, `latent`,
`koopman`, `doctor`) use `~/vieb/venv` (Python 3.13, CPU-only). Every
`gpu`-partition job (`full_pipeline`, `02_compare_latents`, `embed_cluster`)
uses `~/vieb/venv-gpu` (Python 3.11.4 + RAPIDS), because RAPIDS publishes no
3.13 wheels. `doctor` stays on the default venv deliberately — its job is to
report what is installed there against what the driver wants.

Getting this wrong is silent and expensive: a gpu job on the CPU venv finds no
`cuml`, and `--gpu auto` will spend the whole allocation on one core without
logging anything. That is why every gpu script defaults to `GPU=on`, which
raises in the first second instead.

Site-specific `#SBATCH --account` / `--qos` lines are present but commented out
at the top of each script — uncomment if your cluster requires them.

## Results

```
$OUT_DIR/
  aligned.npz              aligned pose
  scores.npz               latent coordinates
  embedded.npz             delay embedding
  labels.npz               cluster labels (-1 = noise, never force-assigned)
  koopman_labels.npz       basin labels (-1 = near a separatrix, NOT noise)
  koopman_report_r<N>.json attractor topology at --n-regions N
  runs.json                run registry -- also read by the GUI
  latent_comparison.json   PCA vs diffusion, side by side
```

`koopman_labels.npz` carries the same three arrays as `labels.npz`, so it
drops into the same slot. It is **not** positionally comparable with it:
Koopman labels every frame of `scores.npz`, while HDBSCAN labels the
delay-embedded frames, which are fewer by one window per recording. Join the
two on the `index` array (`recording`, `frame`) that both checkpoints carry.
Its `-1` means *near a separatrix* -- a transition -- not HDBSCAN noise; the
meaning is recorded in the checkpoint's `noise_label_means`.

The comparison is also printed in the job log. It reports `n_states`,
`noise_frac`, `largest_state_frac`, `state_entropy` and `noise_speed_ratio`
for both arms and **declares no winner** — interpretation is yours.

Two things to read carefully:

- `largest_state_frac` and `state_entropy` are reported in two conventions.
  v1's counts fractions over *all* frames including noise (so they sum to
  1 − `noise_frac`); the "clean" ones normalise over clustered frames. They are
  not interchangeable.
- `noise_speed_ratio` above 1 means the frames HDBSCAN failed to cluster are
  the *fast* ones — the signature of density-based clustering under-detecting
  brief behaviors. Near 1 means that explanation does not apply here.

A large dominant state is **not** by itself evidence of a problem; it may
simply be how long the animal spent in that behavior.

## Resuming and re-running

Each stage checkpoints, so clustering can be re-run without repeating
alignment:

```bash
OUT_DIR=<existing run> sbatch --export=ALL 02_compare_latents.sbatch
```

To sweep the clustering parameter against one existing embedding:

```bash
python -m cli sweep --out <existing run> --min-cluster-sizes 25,50,100,200 --gpu on
```

To test the Koopman arm end to end against an existing `scores.npz`, sweep
`--n-regions` into separate out-dirs (the state count is only an output if the
parameter that could fake it has been varied — #55, #57), then compare:

```bash
for n in 12 24 48 96 192; do
    d=~/vieb2-results/koopman_pca_r$n
    mkdir -p "$d" && ln -sf ~/vieb2-results/koopman_pca/scores.npz "$d/"
    sbatch --export=ALL,OUT_DIR=$d,N_REGIONS=$n koopman.sbatch
done
sbatch --export=ALL compare_koopman.sbatch   # needs labels.npz in both base dirs
```

Runs launched here also appear in the GUI's Cluster Runs and Overview pages —
the checkpoint and registry formats are shared.

## Troubleshooting

**`No module named cli`** — the checkout is on the wrong branch or predates the
CLI. `git -C ~/vieb pull origin v2`, then confirm `vieb_v2/cli.py` exists.

**`OUT_DIR must be set`** — `01`/`02` are meant to be submitted through
`submit.sh`, which is the single place `OUT_DIR` is decided. To run one alone,
export it first: `OUT_DIR=<dir> sbatch --export=ALL 02_compare_latents.sbatch`.

**`--gpu on ... unusable`** — run `doctor` on the GPU partition; it prints the
exact pip command for that node's driver.

**`pip install cuml` pulled version 0.6.1** — that is an unrelated package
squatting the name. Use `install_gpu.sh`, which installs `cuml-cu12` from
`pypi.nvidia.com`.

**cupy compiled from source and failed on `cublas_v2.h`** — bare `cupy` has no
wheel for this platform. `install_gpu.sh` uses `--only-binary=:all:` so a
missing wheel fails in seconds instead of starting a doomed build.

**RAPIDS has no wheel for this Python** — a venv is permanently bound to the
Python that created it; `module load python/3.11.4` does not change an existing
venv. `install_gpu.sh` prints the rebuild steps.

**Exit codes** — `0` completed, `1` needs attention (no pose found, only one
state), `2` hard failure (`--gpu on` with no usable GPU).
