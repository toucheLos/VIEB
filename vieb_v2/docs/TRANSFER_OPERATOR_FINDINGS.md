# `transfer-operator` — findings

**Result: the branch dies at Stage 1 (§3). Reported, not tuned.**

The transfer operator was built, verified against four synthetic systems with
known answers, and run on Luna. The implied timescales do not plateau at any
lag or any partition resolution tested, and the Chapman–Kolmogorov error is
well above threshold. Per §3's stated decision rule and §9's "no tuning of a
stage that failed its gate", Stages 3–4 (§5, §6) and the VUS-1 emit (§7) were
**not** built.

---

## Stage 0 (§2) — representation

### §2a degeneracy — **AUC 0.7065, 95% CI [0.661, 0.729]**

Logistic regression separating top vs bottom `centroid_speed` tercile from
**posture only**; 300 recordings, 240,000 frames, 31 predictors, bootstrap over
recordings.

This falls **between** the brief's two decision thresholds. The CI excludes 0.5
(posture is not blind to locomotion) and also excludes 0.8 (posture is far from
sufficient). Read as: alignment leaves a real but partial locomotor signature,
so §2b is closer to a rescue than an improvement — but it is not the total
degeneracy the brief's first branch describes.

**This is a proxy.** The stated test wants raw pre-alignment centroid speed
against the v2 aligned pose. No per-recording raw pose exists in this project —
only DLC training artifacts (`labeled-data/CollectedData_*.h5`, evaluation
CSVs). Both sides come from v1's engineered features instead: target is v1's
`centroid_speed` (computed pre-alignment, so the right quantity); predictors are
the translation- and heading-invariant subset only (28 pairwise distances,
elongation, `rearing_score`, `head_angle`). `body_orientation` is excluded
despite being available — it is a heading, and heading is what alignment
removes; including it would flatter the result.

### §2c keypoint discrepancy — **resolved: v1 uses 8, v2 uses 7**

`index.json` `_meta.feature_names` contains `speed_kp0`…`speed_kp7`. v2's
`representation/pipeline.py` explicitly drops `tail_tip` before anything else
sees it. So v1 carries tail tip and v2 does not — and tail tip is the
highest-amplitude moving marker on the animal.

### Data-integrity finding — **mixed feature dimensions in one directory**

`results/features/` holds **3,526 files at 51-D and 320 at 91-D** (the wavelet
variant), pooled from two extraction runs. `index.json` declares 51.

Every consumer indexes features by *column position*. A naive glob either
crashes on the concatenate — the lucky case, and what happened here — or slices
a fixed range and silently mixes two feature spaces with no in-band signal. The
first §2a run was contaminated this way and scored 0.685; restricted to the
declared 51-D layout it scores 0.7065. `transfer/featureio.py` resolves the
dimension from `index.json`, excludes mismatches, and reports the count.

### §2b — not performed

`centroid_speed` and `angular_velocity` already exist as named channels
(columns 36 and 39) in every v1 feature file, so the channels §2b would restore
are on disk for all recordings. Restoring them into the *v2* path was not done,
because §3 failed and §9 forbids continuing past a failed gate.

---

## Stage 2 (§4) — synthetic verification gate: **4/4 PASS**

Run before any new code touched real data. 14 tests, all passing.

| system | result |
|---|---|
| **1. Underdamped double well** | Two metastable wells split along `phi_2` (chi 0.85+). `t_imp` tracks the independently-counted hop rate across an **8× range** of timescales: β=2 → 14.60 s vs 1/(2k)=11.32; β=3 → 44.46 vs 34.95; β=4 → 117.73 vs 87.80. Ratio 1.27–1.34, i.e. a **constant** offset in the direction reversibilization predicts (upper bound), not a breakdown. Equipartition recovered from `x` alone to within 2% at all three temperatures (⟨KE⟩ 0.2489/0.1631/0.1242 vs kT/2 0.2500/0.1667/0.1250) — the delay coordinate genuinely carries the unobserved velocity. |
| **2. Limit cycle** | Stuart–Landau, true period 1.0. Spectral revival at **τ = 0.985** — 1.5% error. |
| **3. Duration control** | 20:1 dwell ratio recovered in `pi` as **20.9:1**. The rare fast state (1/40th the dwell of the dominant one) survives as **its own metastable set** carrying `pi` = 0.0198 at purity 0.90, alongside the dominant state (`pi` 0.935, purity 1.00) and the short state (`pi` 0.045, purity 0.95). **Occupancy and identity are genuinely separated** — this is the branch's central claim, and it holds. |
| **4. Null** | i.i.d. Gaussian: `lambda_2` < 0.25 at every lag, no plateau, best split chi < 0.6. Zero false positives. |

One simulator bug found and fixed during this stage: at `theta=4` the rare
state's OU relaxation took ~10 frames while its dwell *was* 10 frames, so its
frames sat in transit near other centres and the test measured that transit
rather than the operator. Raised to `theta=12` so approach (~3 frames) is fast
relative to the shortest dwell, isolating the duration variable the control
exists to test.

---

## Stage 1 (§3) — falsification gate: **FAIL**

150 recordings, 868,976 frames (8.0 h), 51-D features → PCA (70.7% variance at
10 components), Voronoi microstates, τ swept 0.033 → 60 s log-spaced, bootstrap
over recordings.

`t_imp` for the slowest mode rises **monotonically across the entire 1800×
sweep** and never flattens:

| N | τ = 0.033 s | τ = 60 s | overall d log t / d log τ | flattest qualifying window | CK mean TV, n=2..5 |
|---|---|---|---|---|---|
| 100 | 1.34 s | 130.8 s | 0.646 | 0.486 over 4.2× | 0.146, 0.231, 0.287, 0.327 |
| 200 | 1.48 s | 140.6 s | 0.604 | 0.365 over 4.2× | 0.268, 0.344, 0.391, 0.421 |

Criteria, fixed in advance: a plateau requires `|d log t / d log τ| < 0.15` over
a window spanning ≥3× in τ, **and** Chapman–Kolmogorov mean TV < 0.10 at n=2
(a genuine Markov chain scores <0.05 — see the synthetic gate). Points with
`lambda_2 > 0.95` (near-identity) and `t_imp < 2τ` (unresolvable) are excluded
first, as §3 requires. Neither partition meets either criterion.

**Robustness.** The negative holds across N ∈ {50, 100, 200, 400}, 10 and 20 PCA
components, 150 and 250 recordings, and two seeds. Minimum window slope observed
anywhere: 0.365–0.411. It is not an artifact of a parameter choice.

### A false positive I generated and then corrected

The first version of `plateau_score` reported **"PLATEAU FOUND"**. It was wrong
three ways: it never excluded the near-identity small-τ region (its "plateau"
for N=100 was τ = 0.033–0.1 s, exactly the artifact §3 warns about), its window
search could prefer narrow windows on a monotonically rising curve, and it
ignored the CK test despite a plateau being necessary-but-not-sufficient. Worth
recording because the failure mode is generic: a flatness search on a rising
curve will always return *something*.

### Deviation from the brief

§3 says k-means microstates already exist from a coarse-then-refine path and
should not be rebuilt. **They do not exist.** `results/shared/` contains 404
HDBSCAN `_labels.npy` — density-based, and therefore exactly what must not
partition an Ulam operator. Microstates were built fresh and geometrically, and
the sweep repeated at multiple N so the conclusion does not rest on one choice.

---

## §8 positive control — written, not run

`transfer/stage8_moseq_control.py` + `hpc/stage8_moseq_control.sbatch` test
whether any MoSeq syllable shifts in Context A post-shock, per-animal with BH
FDR, asserting recording-id overlap before claiming a comparison ran.

Not run here: `results.h5` needs h5py, which is absent from the dev machine,
blocked by PEP-668 (including `--user`), and in none of six local virtualenvs.
It is for the cluster. Note the MoSeq run has **21** syllable grid movies, not
the 48 the brief cites — worth confirming against `results.h5` when it runs.

**This control now matters more, not less.** §3's failure says no Markovian
coarse-graining exists on *this representation*. Whether that is a property of
behavior or of the representation is exactly what §8 discriminates, and it is
the cheapest remaining question.

---

## What was not built, and why

Stages 3 and 4 (§5, §6) and the VUS-1 emit (§7) were not built: §3 failed and
§9 forbids continuing past a failed gate. `compare_methods.py` and the VUS-1
format do not exist in the repository in any case.

## Honest limits

- Reversibilizing gives an **upper bound** on relaxation times for irreversible
  dynamics, and behavior is irreversible. Every `t_imp` above is a bound.
- `chi` is `pi`-weighted, so a set carrying little stationary mass contributes
  little to global coherence. The framework removes the *systematic* bias
  toward slow states; it does not solve sampling of rare fast ones.
- Ulam partitioning degrades in high dimension. Reduction to 10 PCs retains only
  70.7% of variance; the negative result also holds at 20 components, but the
  cost is real and belongs in the writeup.
- §3's negative is a statement about **this representation** (v1's 51-D
  engineered features), not about behavior in general. A different observable —
  in particular one with the locomotor channels restored per §2b — could give a
  different answer, and §8 bounds how likely that is.
- deeptime/PyEMMA were unavailable, so the spectral machinery is hand-rolled.
  It is verified against four known-answer systems (§4), which is the mitigation,
  not a substitute for the reference implementations.
