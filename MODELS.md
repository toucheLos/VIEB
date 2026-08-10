# Models

Every model is a plugin in `src/vieb/models/`, selected by name. Nothing else in
the codebase branches on which arm is running.

```bash
vieb models                                    # this table, live
vieb run --model koopman --repr diffusion --dataset luna
vieb run --model all --repr all --dataset luna # the full grid
vieb compare                                   # one table, every run
```

## The factorization

The v2 "four arms" were never four models. They are **two algorithms crossed
with two representations**:

|                | `pca` (9-D) | `diffusion` (8-D) |
|----------------|-------------|-------------------|
| **`hdbscan`**  | pca-HDBSCAN | diffusion-HDBSCAN |
| **`koopman`**  | pca-Koopman | diffusion-Koopman |

Keeping representation as a separate, frozen axis (§1d) is what makes the
comparison fair *and* what makes the table multiply: a new representation adds a
column across every model rather than one more unattributable row. It is also
the fix for the known confound — MoSeq's win is currently unattributable because
it differs from every VIEB arm in representation *and* algorithm at once.

## Model status

| model | what a state is | status | last scored | notes |
|---|---|---|---|---|
| `hdbscan` | a density peak in the delay embedding | **runs** | #65 | state count set by `min_cluster_size` (50) |
| `koopman` | a basin of local affine Koopman operators | **runs** | #65 | best VIEB arm on `diffusion` (0.187) |
| `moseq` | an AR-HMM syllable (external Keypoint-MoSeq) | **runs** | #65 | reference arm, 0.361. Pinned external dep, thin adapter |
| `ulam` | Voronoi microstates + transfer-operator spectrum | **gate failed at K=1** | #64 | no plateau at any lag; see below |
| `exbias` | — | code exists, **never scored** | — | `~/exbias/exbias.py`, 768 lines, standalone |
| `vieb_v1` | HDBSCAN on 91 engineered features → UMAP | code exists, **never scored on this axis** | — | the flat `~/vieb` tree |
| `hsmm` | an explicit-duration AR-HSMM state | **not built** | — | Stage 2 |
| `ulam_msm` | a macrostate from the coarse-grained spectrum | **not built** | — | Stage 2; resumes `ulam` with delay embedding |

### Does not exist

Two entries in the original plan's adapter list have no implementation anywhere
in this repo, and are not merely unrun:

- **`ticc`** (Toeplitz inverse covariance) — no code, no mention in any doc.
- **`flow_field`** — `docs/V2_RESULTS_CONTEXT.md` records that
  `~/vieb-flow-field` is a worktree of a branch identical to `v2` at `4dc824f`
  and "carries the v1 tree, not a flow-field model". Decision #55's gate is
  unmet.

Both are dropped from the bakeoff unless someone decides to write them. The
comparison list closes at the models that exist.

### `ulam` — a failure with an asterisk

Decision #64 recorded the pre-registered kill condition firing: no implied
timescale plateau for t2, t3 or t4, at any lag from 0.033 s to 36 s, in both
arms, on all 3,846 recordings. The estimate was not broken — 500/500 microstates
retained, one connected component, `leak_frac` 0, 18.2–22.4M pairs per lag.

**But it was measured at K=1, with no delay embedding**, and #64 flags the
caveat itself: Costa et al. predict exactly this at K=1, because the
instantaneous observable is not the full state; delay embedding is their remedy.
The growth is scale-free (local exponent drifting 0.529 → 0.847), which is
neither a plateau nor the trivial large-tau artifact — long memory, no timescale
separation.

`ulam_msm` (§4) is therefore not a fresh build. It resumes this run with the
delay embedding that was never applied, and #64 supplies the **K=1 point of
§4d's K ∈ {1, 3, 6, 12, 24} sweep as an already-measured negative control**. If
a plateau appears as K grows, that is a measurement of how much memory the
representation carries. If it does not appear at any K, §4d's kill condition is
met honestly and twice.

## Representations

Frozen, hashed artifacts (§1d). `repr_hash` is a column in every results row;
a row whose hash disagrees with its config is flagged, never silently compared.

| repr | what | dim | build cost |
|---|---|---|---|
| `pca` | pooled PCA on aligned pose, 95% variance | 9 | 206 s |
| `diffusion` | landmark diffusion maps + Nyström, α=1, 3,000 landmarks | 8 | 1,189 s |
| `obs` | 9 pose PCs **+** restored `centroid_speed`, `angular_velocity` | 11 | — |

`pca` vs `obs` is the repaired-representation contrast the plan asks to run the
full bakeoff on. The difference between those two tables is itself a result.

## How models are judged

Not by clustering quality. Silhouette, DBCV, UMAP separation and "largest state
%" all reward the duration confound — `pca-HDBSCAN` produces six tight,
well-separated clusters and a positive-control effect of exactly 0.000. Ranking
is on the `summary.json` block, identical for every model (§6):
`retrieval_context_A__z`, `duration_occupancy_corr`, `occupancy_entropy`,
`transition_entropy`, `ck_residual_n2..n4`, bout duration mode / CV. CIs
bootstrap over **animals**. Cross-method agreement uses variation of
information, not ARI.

## Adding a model

1. Implement `Model` (`src/vieb/models/base.py`): `fit`, `label`, `save`, `load`.
2. Register it in `src/vieb/models/registry.py`.
3. Add `configs/models/<name>.yaml`.

That is all. The harness owns loading, the representation, run-length encoding
into bouts, VUS-1 output, and every metric. **No model gets its own scoring
path** — that duplication is why the same question has been answered differently
in different places.
