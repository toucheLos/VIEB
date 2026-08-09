# VIEB v2 — how the behavior-detection models differ, and which one works

Branch `v2_results`. Luna: 3,846 recordings, 298 animals, 22,355,989 frames,
30 fps. Every number below is reproducible from
`vieb_v2/results_analysis/` against `~/vieb2-results/`.

Context map: `V2_RESULTS_CONTEXT.md` (where the artifacts are, what was
verified). Prior findings: `TRANSFER_OPERATOR_FINDINGS.md`, `DECISIONS.md`
#53–#64.

---

## The headline

Four state-discovery algorithms have been run on Luna over two latent spaces.
Until now they were compared only on **geometry** — state count, entropy, noise
fraction — which describes a partition and says nothing about whether it tracks
behavior. This report scores all four on the axis the MoSeq positive control
established: *does the state set separate Context A after conditioning?*

The result reorders them, and it contradicts the geometric reading:

| rank | arm | states | largest paired shift at retrieval | verdict |
|---|---|---|---|---|
| — | **MoSeq** (reference) | 48 | **0.361** | the standard |
| 1 | **diffusion-Koopman** | 16 | **0.187** | the only VIEB arm within 2× of MoSeq |
| 2 | pca-Koopman | 43 | 0.036 | fragmented; real but small effects |
| 3 | diffusion-HDBSCAN | 37 | 0.020 | 35/37 "significant" — a power artifact |
| 4 | pca-HDBSCAN | 6 | **0.000** | a non-detector |

Three things to take away:

1. **`pca-HDBSCAN` — the v2 default pipeline — does not detect the effect at
   all.** Its largest median paired difference across every state is *exactly*
   0.0. Truncated to MoSeq's common session length it finds **0 of 4** testable
   states significant, and its own sign-flip null reaches that count **100% of
   the time**. This is not a weak detector; it is a null one.
2. **Significance count is not effect size.** `diffusion-HDBSCAN` scores 35 of
   37 states significant — a higher hit rate than MoSeq's 33 of 35 — on a top
   state that moves from 0.951 to 0.971. With 298 animals paired within-subject,
   a two-point shift in a state occupying 96% of every session clears q < 1e-31.
   Ranking by significant-state count puts this arm first and puts the arm with
   a 0.55 → 0.39 shift third.
3. **One VIEB arm beats MoSeq on one axis.** `diffusion-Koopman`'s retrieval
   effect is **9.95×** its novel-context effect; MoSeq's is 5.92×. Its shift is
   more specific to the *conditioned* context, rather than to any context
   change. It does not beat MoSeq on effect size, and nothing here does.

---

## 1. What the models are

"Model" names three different layers in these runs. Claims are only comparable
within a layer.

### Layer 1 — representation

| id | what | dim | cost |
|---|---|---|---|
| `pca` | pooled PCA on aligned pose, 95% variance | 9 | 206 s |
| `diffusion` | landmark diffusion maps + Nyström, α=1, 3,000 landmarks (142 pruned) | 8 | 1,189 s |
| `obs` | 9 pose PCs **+** restored `centroid_speed`, `angular_velocity` | 11 | — |

All three start from `align_session`, which subtracts the per-frame centroid and
applies a per-frame rotation. **The v2 aligned space is purely postural** —
translation and heading are gone by construction (#60), and delay embedding
cannot recover them because they were subtracted before measurement.

The two latents differ in what they optimize, and this turns out to drive the
entire result:

- **PCA maximizes variance.** Its coordinates are the directions in which
  posture varies most. Nothing about them is dynamical.
- **Diffusion maps approximate the slow eigenfunctions of a diffusion on the
  pose manifold.** Its coordinates are ordered by *relaxation time*, not
  variance — eigenvalues 0.988 … 0.935, spectral gap 0.0034.

### Layer 2 — state discovery

| id | what a state is | state count is set by |
|---|---|---|
| `hdbscan` | a density peak in the delay embedding | `min_cluster_size` (=50) |
| `koopman` | a basin of attraction of local affine Koopman operators | claimed to be an *output*; measured below |
| `ulam` | 500 Voronoi microstates + a transfer operator; macrostates from the spectrum | the spectrum (if a plateau exists) |
| `moseq` | an AR-HMM syllable (external, Keypoint-MoSeq) | the stickiness prior κ |

### Layer 3 — what MoSeq does differently

MoSeq is not a VIEB arm and beats all of them, so it is worth being precise
about why. Two structural differences, both of which VIEB v2 gave up:

- **It never subtracts the locomotor channels.** MoSeq models centroid velocity
  and heading change as modeled dimensions. VIEB's `align_all` removes exactly
  the signal that carries freezing.
- **It has a temporal prior.** The AR-HMM's stickiness makes syllables *bouts*
  by construction. HDBSCAN and Koopman both assign per frame with no temporal
  model in this path (v1's HMM Viterbi smoothing is not in the v2 pipeline).

---

## 2. How they were compared

### The common metric

`results_analysis/discriminate.py` reuses `scripts/moseq_control.py`'s
statistics **verbatim** — per-animal means, paired Wilcoxon signed-rank,
BH-FDR, the same within-animal sign-flip null, the same 0.1%/50-recording
frequency floor. Only the state labels change. The three contrasts:

| contrast | what it isolates |
|---|---|
| day 1 Context A (no shock) vs day 0 Context A | the post-shock effect |
| day 2 Context C vs day 0 Context A | does *any* context change do this? |
| CFD Context A vs Context B, days 3–7 | day-matched discrimination |

### The join that made it possible

`labels.npz` stores an `index` of `(recording_idx, frame_idx)` and nothing else.
`recordings.py:1` states the map back to a file "is not merely absent, it is not
reconstructible after the fact" — `load_sessions` drops unreadable files into a
`skipped` list that shifts every later index and is never persisted.

That is true in general and **verifiably false for this run**:
`find_pose_files` returns 4,925 paths, `aligned.npz["lengths"]` has 4,925
entries, and `frame_count(paths[i]) == lengths[i]` for **all 4,925, exactly, zero
mismatches**. A skipped file would break that correspondence. `verify_index`
performs the check and raises rather than warning — a silent off-by-one would
attribute each recording's behavior to a neighbouring animal and still produce
plausible p-values.

### Two corrections applied to the scoring

- **Deduplication.** The `koopman_*` runs predate the h5/csv dedup (#59) and see
  4,925 sessions. 1,079 duplicate rows are collapsed (h5 preferred) before
  per-animal averaging, giving 3,846 recordings / 298 animals in every arm.
- **Truncation.** Session length is confounded with arm — Context A runs ~6,302
  frames against ~5,392 for B and C, because the shock protocol needs the time.
  Every contrast was re-run truncated to MoSeq's common 5,381 frames.

---

## 3. Results

### 3.1 Discrimination — the metric that is about behavior

Full sessions, retrieval contrast:

| arm | states | tested | significant | hit rate | **max \|median shift\|** | top state A → B | null ≥ obs |
|---|---|---|---|---|---|---|---|
| **MoSeq** | 48 | 35 | 33 | 0.943 | **0.3605** | 0.463 ← 0.095 | 0.00 |
| diffusion-Koopman | 16 | 16 | 14 | 0.875 | **0.1867** | 0.392 ← 0.551 | 0.00 |
| pca-Koopman | 43 | 43 | 33 | 0.767 | 0.0364 | 0.059 ← 0.118 | 0.00 |
| diffusion-HDBSCAN | 37 | 37 | 35 | 0.946 | 0.0199 | 0.971 ← 0.951 | 0.00 |
| pca-HDBSCAN | 6 | 4 | 2 | 0.500 | **0.0000** | 0.0009 ← 0.0009 | 0.01 |

Truncated to a common 5,381 frames — the same treatment MoSeq's control used:

| arm | significant | hit rate | max \|median shift\| | change vs full |
|---|---|---|---|---|
| diffusion-Koopman | **16/16** | 1.000 | **0.2107** | effect **grows** |
| diffusion-HDBSCAN | 34/37 | 0.919 | 0.0193 | ~flat |
| pca-Koopman | 31/43 | 0.721 | 0.0391 | ~flat |
| pca-HDBSCAN | **0/4** | 0.000 | 0.0000 | collapses; null ≥ obs = **1.00** |

Truncation is the decisive test. It **strengthens** `diffusion-Koopman`
(0.187 → 0.211) — the same direction MoSeq moved — so session length was
diluting its effect, not manufacturing it. And it removes `pca-HDBSCAN`'s two
"significant" states entirely, leaving a result its own null reproduces every
single time.

→ `figures/1_effect_vs_power.png`, `figures/2_retrieval_timecourse.png`

**The time course** is what makes an effect interpretable. MoSeq's syllable 1:
0.095 naive → 0.463 at retrieval → 0.170 in a novel context, then A-vs-B
separating monotonically across days 3–7. `diffusion-Koopman`'s state 13 is the
mirror image — 0.551 → 0.392, staying *below* Context B on every discrimination
day. A state suppressed by fear, where MoSeq found one elevated by it. Both are
telling the same story with opposite sign. Neither HDBSCAN arm's top state
moves visibly at all.

### 3.2 Partition geometry — and why it misleads

| arm | states | largest state (of clustered) | noise | entropy (clean) | **noise speed ratio** | size↔speed rank corr |
|---|---|---|---|---|---|---|
| pca-HDBSCAN | 6 | **0.992** | 0.154 | 0.032 | **9.67** | +0.09 |
| pca-Koopman | 43 | 0.060 | 0.477 | 0.949 | 1.34 | −0.83 |
| diffusion-HDBSCAN | 37 | **0.964** | 0.108 | 0.074 | **18.96** | −0.51 |
| diffusion-Koopman | 16 | 0.352 | 0.139 | 0.753 | 0.60 | −0.57 |

→ `figures/4_partition_geometry.png`

**`noise_speed_ratio` is the diagnostic that predicts the discrimination
result.** Above 1 means the frames the method failed to label are the *fast*
ones — the documented signature of density-based clustering under-detecting
brief behaviors (`hpc/README.md:138`). Both HDBSCAN arms discard frames moving
**10× and 19× faster** than what they keep. Both Koopman arms are near or below
1. The two arms that throw away the fast frames are the two arms with no
effect.

Note the noise columns are not the same quantity: HDBSCAN's `-1` means
unclustered, Koopman's means *near a separatrix* — a transition (#57).

### 3.3 Do the two families agree? No.

Joined on `index` over 28,586,707 frames:

| latent | both assigned | noise agreement | **adjusted Rand** |
|---|---|---|---|
| pca | 0.450 | 0.531 | **0.0021** |
| diffusion | 0.765 | 0.777 | **0.0197** |

Adjusted Rand of 0.002 is indistinguishable from independent partitions. Run on
the same frames, in the same latent space, HDBSCAN and Koopman are not finding
the same structure — they are not two estimates of one thing.

### 3.4 Is the Koopman state count an output? Partly — and not for PCA.

#55 and #57 made the case that a state count is only an "output" if the
parameter that could fake it has been varied. Sweeping `--n-regions`:

| n_regions | 12 | 24 | 48 | 96 | 192 | scaling |
|---|---|---|---|---|---|---|
| pca attractors | 9 | 22 | 43 | 79 | 162 | **n ∝ r^1.04** |
| diffusion attractors | 5 | 8 | 16 | 22 | 36 | n ∝ r^0.71 |

→ `figures/3_state_count_sweep.png`

**In PCA space the claim fails outright.** One attractor per region, across a
16× sweep — the state count *is* the parameter, wearing a different name. In
diffusion space the exponent is 0.71, so genuine merging is happening and the
count is partly an output, but it is not parameter-free either. **0 limit cycles
at every resolution in both arms** — no oscillatory behavior (gait, grooming
rhythm) was recovered anywhere.

### 3.5 The transfer operator — the gate still fails

500 Voronoi microstates on the 11-D observation space, lag swept 0.033 s → 36 s.

| arm | dim | t₂ at 0.033 s | t₂ at 36 s | growth | global exponent | min t₂/τ |
|---|---|---|---|---|---|---|
| pose PCs + locomotor channels | 11D | 0.611 s | 65.4 s | 76× | 0.670 | 1.82 |
| pose PCs only (control) | 9D | 0.546 s | 71.9 s | 91× | 0.707 | 2.00 |

→ `figures/5_implied_timescales.png`

No plateau at any lag, in both arms. The local exponent drifts monotonically
from 0.53 to 0.86 — strictly between 0 (a plateau) and 1 (the trivial large-τ
artifact) everywhere. That is **long-memory, multi-scale behavior with no
timescale separation**, a stronger statement than "no plateau."

The estimator is not broken, and this matters for reading the result: at every
one of the 26 lags, **all 500 microstates retained, 0 dropped,
`dropped_frame_frac` = 0, 1 connected component, `leak_frac` = 0,
`near_reducible` = False**, 18.2–22.4M pairs. The absence of a plateau is a
property of the data.

**The standing concern (§5 of `TRANSFER_OPERATOR_FINDINGS.md`) is unchanged and
is now more pointed.** This negative result is exactly what the branch's own
cited reference (Costa, Ahamed, Jordan & Stephens) predicts at K=1, where delay
embedding is the remedy. The gate was specified to run *before* any delay
embedding, so it may be falsifying K=1 rather than the branch. The comparison in
this report adds a reason to take that seriously: the arms that **did** find a
real effect (both Koopman arms) run on the delay embedding, and the transfer
operator does not.

### 3.6 What alignment actually cost

Classifying top-vs-bottom tercile of raw centroid speed from **aligned pose
alone**, held out by recording:

| model | AUC | 95% CI |
|---|---|---|
| logistic | 0.693 | [0.687, 0.698] |
| gradient boosting | **0.790** | [0.784, 0.795] |

The two terciles differ by **55×** in real speed (median 0.77 vs 43.8 px/s), and
posture recovers that at 0.79, not 1.0. So the locomotor signature is *present
but not linearly decodable* — a partial loss, +0.097 of it non-linear. This is
the quantitative version of "alignment cost real information but did not destroy
it," and it bounds how much any purely postural method can recover.

---

## 4. Ranking

→ `figures/6_ranking.png`

Six named axes, min-max scaled across arms, weighted. Arms missing an axis are
renormalized over the axes they have, so "not measured" never reads as "measured
badly." Min-max means the worst arm on an axis contributes exactly 0 to it.

| axis | weight | what it measures |
|---|---|---|
| effect | 0.350 | largest paired occupancy shift at retrieval |
| specificity | 0.200 | retrieval effect ÷ novel-context effect |
| coverage | 0.150 | significant ÷ tested states |
| resolution | 0.150 | 1 − largest state fraction |
| parsimony | 0.075 | 1 − d log(states)/d log(parameter) |
| cleanliness | 0.075 | 1 − noise fraction |

| # | arm | composite | effect | specificity | coverage | resolution | parsimony | cleanliness |
|---|---|---|---|---|---|---|---|---|
| 1 | **MoSeq (reference)** | **0.778** | 0.3605 | 5.92 | 0.943 | — | 0.00 | — |
| 2 | **diffusion-Koopman** | **0.754** | 0.1867 | **9.95** | 0.875 | 0.648 | 0.288 | 0.861 |
| 3 | diffusion-HDBSCAN | 0.399 | 0.0199 | 6.62 | 0.946 | 0.036 | — | 0.892 |
| 4 | pca-Koopman | 0.275 | 0.0364 | 1.57 | 0.767 | 0.940 | 0.000 | 0.523 |
| 5 | pca-HDBSCAN | 0.090 | 0.0000 | — | 0.500 | 0.008 | — | 0.846 |

Effect is weighted above coverage deliberately, and that single choice is what
separates this ranking from the geometric one. Ranking by significant-state
count alone puts `diffusion-HDBSCAN` first.

`parsimony` is scored `—` for both HDBSCAN arms because **`min_cluster_size` was
never swept**. That is untested, not passed — the scrutiny #55/#57 demanded of
Koopman's state count has never been applied to HDBSCAN's.

---

## 5. Why each model behaved as it did

Hypotheses, labelled by how much the data here supports them.

**`pca-HDBSCAN` finds one state because there is one density mode.**
*(well supported)* Alignment removes translation and heading; PCA then keeps the
directions of largest postural variance. In that space the density has a single
overwhelming mode — with `min_cluster_size=50` against 28.6M points a cluster
needs only 50 members, and it still returned 99.2% of clustered frames in one
state. The 15.4% it called noise moves 9.7× faster. It clustered "the animal is
in a typical posture" and discarded the rest. Since freezing and locomotion
differ 55× in speed and speed was subtracted, the surviving postural variation
genuinely has one mode. Effect exactly 0.0 is the honest consequence.

**`diffusion-HDBSCAN`'s 37 states are one core plus 36 slivers.**
*(well supported)* The diffusion spectrum is nearly degenerate — eigenvalues
0.988 … 0.935, spectral gap **0.0034**. No gap means no natural cluster count,
so HDBSCAN shaves satellites off a continuum: 96.4% of clustered frames in one
state, entropy 0.074, and the worst noise-speed ratio of any arm at 19.0. Its
35/37 significant states are a **power** result on a 96% state moving two
points, not an effect result.

**`pca-Koopman` fragments because PCA space has no basin structure.**
*(well supported)* n ∝ r^1.04: each Voronoi region's local operator finds its own
fixed point and the graph pruning never merges them, because there are no real
basins to merge into. 47.7% of frames land near a separatrix — a partition that
is mostly boundary. Its effects are real but small (0.036) because no state is
larger than 3.1%; a genuine behavioral regime split across 40 basins cannot show
a large shift in any one of them.

**`diffusion-Koopman` wins because its coordinates are dynamical, not
variance-based.** *(supported, and the most interesting claim here)* Diffusion
map coordinates approximate the slowest-relaxing directions of a diffusion on
the pose manifold — they are ordered by *timescale*. A basin decomposition
computed in slow coordinates is far more likely to align with sustained
behavioral regimes than the same decomposition in variance-maximizing
coordinates. Three independent signals agree: the sublinear state scaling
(r^0.71 — genuine merging), the low transition fraction (13.9% vs pca's 47.7% —
basins with real interiors), and a noise-speed ratio *below* 1 (0.60 — its
separatrix frames are slower than basin interiors, i.e. postural pauses between
regimes rather than fast transitions). This is the one arm where the state
count, the geometry, and the behavioral effect all point the same way.

**MoSeq wins because it kept the channel VIEB subtracted, and has a temporal
prior.** *(well supported, but not isolated)* Freezing is defined by near-zero
locomotion; `align_all` subtracts locomotion; the degeneracy test says posture
alone recovers speed tercile at AUC 0.79, not 1.0. On top of that, MoSeq's
stickiness prior makes syllables bouts by construction, where these VIEB arms
assign per frame with no temporal model. **The confound:** MoSeq differs from
VIEB in both representation *and* algorithm at once, so this comparison cannot
attribute its win between the two. Section 6 says how to separate them.

---

## 6. Does anything surpass MoSeq?

**Overall, no.** MoSeq's retrieval effect is 0.361; the best VIEB arm reaches
0.187 (0.211 truncated) — within a factor of 2, but behind.

**On one axis, yes.** `diffusion-Koopman`'s retrieval effect is **9.95×** its
novel-context effect, against MoSeq's **5.92×**. Its shift is more specific to
the conditioned context than to context change in general — which is the
property a fear assay actually wants, and it is invisible in any
single-contrast summary.

**One apparent win that is not one.** `diffusion-HDBSCAN`'s hit rate (0.946)
edges MoSeq's (0.943). This is a power artifact and should not be reported as a
win; see §3.1.

Two caveats that bound all of the above:

- MoSeq's comparison is not clean: it differs in representation *and* algorithm.
- MoSeq's numbers come from truncated sessions; the VIEB full-session numbers
  are the ones in the headline table. The truncated re-run (§3.1) is the
  matched comparison, and it moves the ranking in `diffusion-Koopman`'s favour.

---

## 7. Limits of this comparison

Stated rather than worked around.

1. **The `koopman_*` models were fit on double-counted data.** They used the
   pre-dedup alignment: 1,079 recordings entered the fit twice. Deduplication
   here fixes the *scoring*, not the fit. HDBSCAN's density estimate and
   Koopman's basins both saw 28% of recordings at double weight. Every arm is
   equally affected, so the ranking is probably safe, but no arm's absolute
   numbers are.
2. **The transfer-operator arm has no per-frame labels on disk.** No
   `microstates.npz` was written, so `ulam` could not be scored on
   discrimination at all. It appears in §3.5 and nowhere else.
3. **`min_cluster_size` was never swept.** HDBSCAN's state count has had none of
   the scrutiny Koopman's `--n-regions` got.
4. **One partition resolution for the transfer operator** (N=500), as
   `TRANSFER_OPERATOR_FINDINGS.md` §6 already flagged.
5. **No seed-stability estimate anywhere** — `seed_stability` is `null` in every
   checkpoint.
6. **Reversibilization gives an upper bound** on relaxation times for
   irreversible dynamics, and behavior is irreversible.
7. **The composite weighting is a judgement**, stated in §4 so it can be
   disagreed with. The `effect` > `coverage` ordering is the load-bearing part;
   everything else moves the ranking very little.
8. **A near-decomposable pooled chain is metastable mathematically.** Two
   sub-populations never observed transitioning produce a real slow eigenvalue,
   and no pooled diagnostic distinguishes "states are behaviors" from "states
   are animals" (`TRANSFER_OPERATOR_FINDINGS.md` §2). Nothing here tests that.

---

## 8. Next steps

Ordered by information gained per node-hour.

### Immediate — cheap, and each one settles a live question

1. **Re-run `diffusion-Koopman` on the deduplicated alignment.** The best arm is
   the one whose fit is most worth trusting, and right now it was fit on
   double-counted data. `~/vieb2-results/to_align_20260807_203030/aligned.npz`
   is the clean one. This is the single highest-value run on this list.
2. **Sweep `min_cluster_size` for both HDBSCAN arms** (25/50/100/200/500), the
   way `--n-regions` was swept, and score each with `discriminate`. It is the
   only way to know whether HDBSCAN's collapse is a parameter artifact or a
   property of the latent. `cli sweep --out <dir> --min-cluster-sizes ...`
   already exists.
3. **Sweep `--n-regions` for `diffusion-Koopman` under the discrimination
   metric,** not just the geometric one. The r^0.71 scaling says the state count
   is partly an output; the question is whether the *effect* is stable across
   it. If the 0.19 shift survives r=12→192, that is a strong result.
4. **Write `microstates.npz` from `cli timescales`** so the Ulam arm can be
   scored on the same axis as everything else.

### Next — separates the confounds

5. **Run Koopman on the 11-D `obs` space** (pose PCs + restored locomotor
   channels). This is the clean test of "MoSeq wins because it kept the speed
   channel": if `diffusion-Koopman`'s effect moves toward 0.36 when speed is
   restored, representation is the explanation; if it does not, the AR-HMM's
   temporal prior is.
6. **Add a temporal prior to the VIEB arms.** v1's HMM Viterbi smoothing exists
   (`compare.py`) and is not in the v2 path. This is the other half of the same
   question, and it is cheap.
7. **Delay-embed before the timescale gate (§5a).** The gate's own stated
   concern is that it falsified K=1 rather than the branch. Both arms that found
   an effect run on the delay embedding; the transfer operator does not. Running
   §5a is what turns that observation into a test.

### Then — makes the result publishable

8. **Seed stability** on the winning arm: refit with 5 seeds, report the
   adjusted Rand between runs. Currently `null` everywhere.
9. **Watch clips of `diffusion-Koopman` state 13.** The whole argument is that it
   is a behaviorally real, fear-suppressed state. `generate_clips.py` exists.
   Nobody has looked at it.
10. **Test the near-decomposability confound** — split by animal and check
    whether the slow structure survives within-animal.

### Open questions for the user

- **Is `pca-HDBSCAN` still the production default?** It is the v2 pipeline's
  default path (`compare.py --cluster`) and it is a null detector on this
  dataset. If anything downstream is consuming its labels, that should stop.
- **Should the transfer-operator branch resume at §5a?** The §3 gate says stop;
  §5 says the gate may have tested the wrong thing; this report adds evidence
  for §5. That call was explicitly left to you.
- **Is there a ground-truth freezing annotation anywhere?** Every metric here is
  a proxy contrast. Even a few hundred hand-scored frames would convert this
  from a relative ranking into an absolute one.

---

## Reproduce

```bash
cd vieb_v2
python -m results_analysis.collect        --results-root ~/vieb2-results
python -m results_analysis.discriminate   --results-root ~/vieb2-results
python -m results_analysis.discriminate   --max-frames 5381 \
                                          --name discrimination_trunc5381
python -m results_analysis.rank           --report ~/vieb2-results/_report
python -m results_analysis.plots          --report ~/vieb2-results/_report
```

Outputs land in `~/vieb2-results/_report/`: `model_comparison.json`,
`discrimination.json`, `discrimination_trunc5381.json`, `ranking.json`,
`figures/*.png`.
