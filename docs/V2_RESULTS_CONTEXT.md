# v2 results — working context

Written on branch `v2_results` (off `transfer-operator`, 2026-08-09). This is the
map of what has actually been run, where the artifacts live, and which facts
were verified on disk rather than assumed. It exists so the next session does
not have to re-derive any of it.

Companion documents: `TRANSFER_OPERATOR_FINDINGS.md` (the §3 gate),
`DECISIONS.md` #53–#64, `vieb_v2/hpc/README.md` (how to launch),
`V2_MODEL_COMPARISON.md` (the report this context feeds).

---

## 1. What "the models" are

Six things get called a model in these runs. They live at three different
layers, and comparing them requires saying which layer a claim is about.

### Layer 1 — representation (what the state space is)

| id | what | dim | where |
|---|---|---|---|
| `pose` | aligned pose, centroid + heading subtracted | 14 (7 kp × 2) | `align_all` |
| `pca` | pooled PCA on aligned pose, 95% var | 9 | `representation/pooled_pca.py` |
| `diffusion` | landmark diffusion maps + Nyström, α=1 | 8 | `representation/diffusion.py` |
| `obs` | 9 pose PCs **+** restored `centroid_speed`, `angular_velocity` | 11 | `representation/observations.py` |

`pose`/`pca`/`diffusion` are **postural only** — `align_session` subtracts the
per-frame centroid and applies a per-frame rotation, so translation and heading
are gone by construction (#60). `obs` is the repair.

### Layer 2 — state discovery (how the space is cut up)

| id | algorithm | state count is | module |
|---|---|---|---|
| `hdbscan` | density clustering on the delay embedding | a parameter's consequence (`min_cluster_size`) | `representation/cluster.py` |
| `koopman` | basins of attraction of local affine Koopman operators | claimed to be an *output* | `representation/koopman.py` |
| `ulam` | Voronoi microstates + transfer operator, spectral | 500 microstates, macrostates from the spectrum | `representation/transfer_operator.py` |
| `moseq` | Keypoint-MoSeq AR-HMM syllables (external) | a prior's consequence (κ) | `~/moseq/luna_demo/` |

### Layer 3 — evaluation

`representation/metrics.py` (`cluster_metrics`, `speed_diagnostics`),
`scripts/moseq_control.py` (the context-discrimination control),
`representation/transfer_operator.py` (implied timescales, the §3 gate).

---

## 2. Where the artifacts are

Root: `~/vieb2-results/`.

| dir | arm | what is in it |
|---|---|---|
| `run_20260804_160351/` | base alignment (**pre-dedup**, 4,925 sessions) | `aligned.npz`, `latent_comparison.json` |
| `koopman_pca/` | pca | `scores.npz`, `embedded.npz`, `labels.npz` (HDBSCAN), `koopman_labels.npz`, `hdbscan_report.json`, `koopman_report_r48.json` |
| `koopman_diffusion/` | diffusion | same set |
| `koopman_{pca,diffusion}_r{12,24,96,192}/` | `--n-regions` sweep | `koopman_report_r<N>.json` only |
| `koopman_comparison.json` | all four arms joined on `index` | the head-to-head |
| `to_align_20260807_203030/` | transfer operator (**deduped**, 3,846) | `aligned.npz`, `pose_frame.npz`, `recordings.csv`, `degeneracy.json`, `timescales_channels.json`, `timescales_pose_only.json`, `*.png` |
| `transfer_operator/moseq_control{,_trunc}/` | MoSeq positive control | `moseq_control.json`, `moseq_syllable_contrasts.csv` |

**r48 lives in the base dirs, not in `koopman_*_r48/`** — those directories do
not exist. `koopman_pca/koopman_report_r48.json` is the r=48 point.

### The dedup split, which is the single biggest trap here

The `koopman_*` runs were built on `run_20260804_160351/aligned.npz`, which
predates the h5/csv deduplication (#59). They see **4,925 sessions /
28,626,107 frames**. The transfer-operator run sees **3,846 / 22,355,989**.
Every count from the two families is on a different denominator. Any
cross-family comparison has to dedupe first.

---

## 3. Verified on disk (2026-08-09)

- **The `index` array of `labels.npz` is reconstructible.**
  `find_pose_files(~/dlc-training/raw_videos)` returns exactly 4,925 paths in
  `sorted()` order; `aligned.npz["lengths"]` has 4,925 entries;
  `recordings.frame_count(path)` matches `lengths[i]` **for all 4,925, exact,
  zero mismatches**. So `index[:,0]` → path → `parse_id` → design metadata is
  sound, and nothing was dropped by `load_sessions` in that run. This is what
  makes the MoSeq-style contrast runnable on VIEB's own labels; without it the
  labels have no experimental design attached and can only be scored on
  geometry. `recordings.py:1` warns the map is "not reconstructible after the
  fact" — it is, for *this* run, because the skip list came back empty.
- All 4,925 filenames parse (`parse_id` 4925/4925).
- MoSeq emits 48 distinct syllables, 35 pass the frequency floor and are tested
  (#59's retraction is the correct version).
- `to_align_20260807_203030/` has **no `microstates.npz` / `transfer_operator.npz`**,
  so the Ulam arm has no per-frame state labels on disk. It can be scored on
  implied timescales and nothing else. Re-running `cli timescales` with the
  checkpoint written is what would change that.

---

## 4. Headline numbers already on disk

Latents (`latent_comparison.json`): PCA 9 comps / 95.34% var / 206 s;
diffusion 8 comps / spectral gap 0.0034 / 1,189 s, 142 of 3,000 landmarks
pruned.

Four arms (`koopman_comparison.json`, 28.59M joined frames):

| arm | states | noise | largest state | entropy (norm.) | noise speed ratio |
|---|---|---|---|---|---|
| pca-HDBSCAN | 6 | 0.154 | **0.839** | 0.106 | **9.67** |
| pca-Koopman | 43 | 0.477 | 0.031 | 0.587 | 1.34 |
| diffusion-HDBSCAN | 37 | 0.108 | **0.860** | 0.094 | **18.96** |
| diffusion-Koopman | 16 | 0.139 | 0.303 | 0.695 | 0.60 |

Agreement between the two families is near zero: adjusted Rand **0.0021** (pca)
and **0.0197** (diffusion). They are not finding the same partition.

`--n-regions` sweep — **the state count tracks the parameter**:

| n_regions | 12 | 24 | 48 | 96 | 192 |
|---|---|---|---|---|---|
| pca attractors | 9 | 22 | 43 | 79 | 162 |
| diffusion attractors | 5 | 8 | 16 | 22 | 36 |

0 limit cycles at every resolution in both arms.

Transfer operator: t₂ grows 0.611 s → 65.4 s over a 1,080× lag range, exponent
d log t₂ / d log τ ≈ 0.67, no plateau, both arms. Degeneracy: AUC 0.693 linear /
0.790 boosted.

MoSeq control: 33/35 syllables shift at retrieval, null 0/100, syllable 1
0.095 → 0.463.

---

## 5. Open questions / what is not answerable from these artifacts

1. **No per-frame Ulam labels** — see above.
2. **No `v_coherence` / flow-field results.** `~/vieb-flow-field` is a worktree
   of branch `flow-field`, which is identical to `v2` at `4dc824f`. It carries
   the v1 tree, not a flow-field model. #55's gate is still unmet.
3. **HDBSCAN `min_cluster_size` was never swept** the way `--n-regions` was, so
   its state count has had exactly the scrutiny that #55/#57 demanded of
   Koopman's and did not get.
4. **The Koopman arms have no seed-stability estimate** (`seed_stability: null`
   in every checkpoint meta).
5. **Delay embedding before the timescale gate was never run** (§5a). This is
   the concern §5 of `TRANSFER_OPERATOR_FINDINGS.md` raises about the gate.

---

## 6. Reproduce this analysis

```bash
cd vieb_v2
python -m results_analysis.collect        --results-root ~/vieb2-results
python -m results_analysis.discriminate   --results-root ~/vieb2-results
python -m results_analysis.discriminate   --max-frames 5381 \
                                          --name discrimination_trunc5381
python -m results_analysis.rank           --report ~/vieb2-results/_report
python -m results_analysis.plots          --report ~/vieb2-results/_report
python -m results_analysis.report_html    --report ~/vieb2-results/_report
```

Everything lands in `~/vieb2-results/_report/`. The findings are written up in
`V2_MODEL_COMPARISON.md`; `DECISIONS.md` #65 is the short version.

`discriminate` re-globs the pose directory to rebuild `index[:,0] -> path` and
**raises** if the frame counts disagree with the checkpoint's `lengths`. If the
pose directory has changed since the `koopman_*` runs, that check will fail and
the comparison is genuinely not reconstructible — do not weaken it.
