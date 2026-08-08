# Transfer operator branch — findings

Branch `transfer-operator`, off `koopman`. Luna, 3,846 recordings, 22,355,989
frames, 298 animals, 30 fps.

Every stage below was pre-registered in `prompts/` before it ran. The headline
is a **negative result at the §3 gate**, reported rather than tuned around, plus
one substantive concern about whether that gate tests what it was meant to.

---

## Summary

| stage | result |
|---|---|
| §8 positive control (MoSeq) | **passes emphatically** — 33/35 syllables shift at retrieval, null 0/100 |
| §2a representation degeneracy | **AUC 0.693 linear / 0.790 boosted** — partial, neither anticipated outcome |
| §4 synthetic verification | **all pass** (30 tests), and corrected two of my own defaults |
| §3 falsification gate | **no plateau at any lag** — the pre-registered death condition |

---

## 0. Corrections to the brief's premises

Four premises were checked against the repo and data before any code was written.

| brief says | actual |
|---|---|
| "You already have k-means microstates — do not rebuild" | `koopman.partition()` computes `region_ids`, but `save_topology` (`koopman.py:896`) writes only `labels`/`probabilities`/`index`. They never reach disk and had to be rebuilt. |
| h5/csv duplication should be fixed | All 1,079 CSVs duplicate H5 recordings. v2 was ingesting **4,925 sessions for 3,846 recordings** — 28.6M vs 22.36M frames. π *is* an occupancy count, so this double-weighted 28% of recordings. |
| §0c: 8 keypoints vs 7 | Not a discrepancy. DLC exports 8 including `tail_tip`; v2 drops it at `keypoints.py:32` → K=7, D=14, rank ceiling 11. Both were right at different stages. |
| MoSeq's 48 syllables | Correct as written. An earlier claim of 42 in this repo was wrong and is retracted in `DECISIONS.md` #59. |

**Independent confirmation of the dedup:** MoSeq's own frame total across all
3,846 result CSVs is **22,355,989** — identical to the deduplicated alignment,
from a tool that never saw the dedup logic.

`compare_methods.py` and any VUS-1 consumer **do not exist** anywhere in the
repo, so "so `compare_methods.py` works unchanged" (§7) is not achievable as
written.

---

## 1. Stage 0 — representation repair

`align_session` subtracts the per-frame centroid and applies a per-frame
rotation, so the v2 aligned space is purely postural. Delay embedding recovers
derivatives of what was *measured*; it cannot recover what was subtracted before
measurement.

### 0a. Tercile separability (`degeneracy.json`)

Top vs bottom tercile of raw centroid speed, classified from **aligned pose
alone**, held out **by recording**.

| model | AUC | 95% CI |
|---|---|---|
| logistic regression | 0.693 | [0.687, 0.698] |
| gradient boosting | **0.790** | [0.784, 0.795] |
| restored channels (circular) | 1.000 | wiring check only |

Neither outcome §2a anticipated (≈0.5 or ≳0.8). **+0.097 of the signature is
present but not linearly decodable**, and the boosted CI stops just short of
0.8 — so the logistic number alone would have overstated the degeneracy claim.

Worth carrying into the writeup: the two terciles differ by **55×** in real
speed (median 0.8 vs 43.8 px/s), and posture recovers that only at 0.79.

Two protocol choices carry this number. The split is **by recording** — adjacent
frames are near-copies, so a frame-wise split reports autocorrelation. The CI
bootstraps over **recordings** for the same reason.

### 0b. Restored channels

`centroid_speed` and `angular_velocity` added as explicit channels, each
standardised to unit variance **before** concatenation (measured scale ratio
2.1×, so raw concatenation would have let px/s set the geometry). All windows in
**seconds**, converted through fps at the boundary. Derivatives via
`np.gradient` **per recording**, so none crosses a boundary.

Validated against MoSeq's independent centroid estimate: positions agree to a
**0.6 px** mean offset; raw per-frame speed agrees at r=0.61, rising to 0.76 at
5-frame and 0.82 at 15-frame smoothing. The disagreement is high-frequency
keypoint jitter differentiated into the speed estimate, which is why smoothing
precedes differentiation. Heading confirmed as `-theta` to 6e-15.

---

## 2. Stage §4 — synthetic verification gate

All four required systems pass, plus a fifth. 30 tests. The gate did its job by
**correcting two of my own defaults**:

- `lag_margin` was rejecting the very regime where the estimate is best. On an
  OU process the plateau is flat to within 5% across a 30× lag range, and the
  original margin marked nearly all of it "unresolved."
- At 60 microstates, k-means gave the rare state **no centre at all** — 313
  points absorbed into a neighbour, the state gone before any operator was
  built. 120 resolves it. A discretization limit on rare states that nothing
  announces on real data.

**Duration control (§4.3), the one that matters most:** two states of identical
dynamical character with a 20:1 dwell ratio. π recovers **20.0**, stable at
every lag, and the rare third state survives. Entry counts are equal to within
**0.3%** (607 vs 605) while occupancy differs 19.4× — a sharper statement of the
confound being dissolved than the analytic model predicted.

**Stated plainly:** a near-decomposable pooled chain *is* metastable
mathematically. Two sub-populations never observed transitioning produce a real
slow eigenvalue and a convincing plateau, and no pooled diagnostic distinguishes
"states are behaviors" from "states are animals."

---

## 3. §8 positive control — passes

Tests whether any MoSeq syllable shifts in Context A post-shock. Paired
within-animal, 298 animals, BH-FDR.

- **Syllable 1: 0.095 → 0.463** at retrieval (d=+0.361, q=8e-47)
- **33 of 35** testable syllables shift significantly
- Sign-flip null: median 0, 95% of repeats give zero, **0/100 reach 33**
- Session lengths differ by arm, so re-run truncated to a common 5,381 frames:
  the effect **grows** (0.095 → 0.463), so length is not the explanation
- Time course is coherent: 0.095 conditioning → 0.463 retrieval → 0.170 novel
  context; A−B discrimination gap widens monotonically across days
  (+0.078 → +0.199)

Per §8, a context-discriminating state is already present, so §2's degeneracy
claim is weaker than the brief stated — consistent with the measured 0.790.

---

## 4. §3 falsification gate — **NO PLATEAU**

500 Voronoi microstates on the 11-D observation space (9 pose PCs at 95.5%
variance + 2 restored channels). Lag swept 0.033 s → 36 s in 26 log-spaced
steps. Horizon 36 s = 0.2 × median recording.

**The brief's requested 60 s sweep is not answerable by this data.** Recordings
are 3-minute sessions (median 5,402 frames, 179–211 s). CK at n=5, τ=60 s needs
a 9,000-frame lag against a 6,321-frame maximum — the count matrix comes back
empty. The sweep is capped at the horizon.

### The result

t₂ grows **monotonically from 0.611 s to 65.4 s** across a 1,080× lag range and
never flattens. Over *any* 4× lag window it changes by ≥2×, against a 1.2×
flatness criterion. Same for t₃ and t₄.

```
 tau_s      t2      t3      t4     t2/tau
 0.033   0.611   0.597   0.492     18.33
 0.333   2.106   1.907   1.694      6.32
 3.500   9.554   6.888   6.280      2.73
36.000  65.424  46.030  39.128      1.82
```

### Neither artifact region explains it

The brief names two artifacts that must not be read as findings. Both are ruled
out:

- **Not the near-identity artifact.** t₂/τ ≥ 1.82 everywhere — the timescales sit
  above the resolution floor across the entire sweep.
- **Not the large-τ noise artifact.** λ₂ is still 0.575 at τ=36 s and still
  declining, on 18.2M pairs.

### The estimate is not broken

At every one of the 26 lags: **all 500 microstates retained**, 0 dropped,
`dropped_frame_frac` = 0, **1 connected component**, `leak_frac` = 0,
`near_reducible` = False, 18.2–22.4M pairs. The operator is well-conditioned and
fully ergodic. The absence of a plateau is a property of the data.

### What the failure actually looks like

The growth is **scale-free**, not linear:

| lag range | d log t₂ / d log τ |
|---|---|
| 0.03–0.13 s | 0.529 |
| 0.43–0.97 s | 0.605 |
| 3.50–7.63 s | 0.763 |
| 9.87–21.47 s | 0.847 |

The exponent drifts monotonically upward and is strictly between 0 (a plateau)
and 1 (the trivial large-τ artifact) everywhere.

The leading eigenvalue violates the semigroup property in a consistent
direction — **λ₂(2τ) exceeds λ₂(τ)² at every lag**, by +0.028 to +0.061, with
the excess growing:

```
tau 0.033->0.067s   observed 0.9240  markov 0.8956  excess +0.0284
tau 0.100->0.200s   observed 0.8811  markov 0.8268  excess +0.0543
tau 0.167->0.333s   observed 0.8526  markov 0.7915  excess +0.0611
```

**Correlations decay more slowly than any single exponential, at every scale
from 33 ms to 36 s.** This is long-memory, multi-scale behaviour with no
timescale separation — a stronger and more specific statement than "no plateau."

---

## 5. The concern with the gate itself

Reported because it would change the decision, not to argue around the result.

**This negative result is what the branch's own cited reference predicts at
K=1.** Costa, Ahamed, Jordan & Stephens (§11) build their method on exactly this
observation: the instantaneous observable is not the full state, implied
timescales therefore grow with τ, and **delay embedding to K\* is the remedy** —
that is what §5a is for. Their C. elegans analysis reaches a plateau only after
embedding.

§3 was specified to run *before* any delay embedding ("run before writing any
new representation code", on the existing non-embedded microstates), and its
decision rule kills the branch on that basis. So the gate as written may be
falsifying K=1 rather than falsifying the branch.

Per §3 and §9 I have **stopped and not tuned**. Whether to proceed to §5a's
delay embedding is the user's call, not mine.

---

## 6. Honest limits

- **One partition resolution.** The claim "no Markovian coarse-graining at *any*
  resolution" was tested at N=500 only. MSM theory says coarse partitions bias
  timescales *low*, so a finer partition raises the curve — but a power law does
  not become a plateau by rescaling. Untested, and named as untested.
- **Reversibilization gives an upper bound** on relaxation times for irreversible
  dynamics. Behavior is irreversible. The reported t_imp are bounds, not exact.
- **No CK test, no bootstrap CIs, no π distribution, no χ(S), no escape rates.**
  All of these are downstream of a τ\* that only exists if a plateau exists. §7's
  report table is therefore only partly fillable, and the empty rows are empty
  because the gate failed, not because they were skipped.
- **Partition cost.** `koopman._nearest` is chunked broadcast distance math:
  2,046 s for 22.4M × 11D at k=500. Memory is bounded, but this dominates
  runtime and would matter for §5b's larger dK.

---

## Reproduce

```bash
cd vieb_v2/hpc
OUT_DIR=$HOME/vieb2-results/to_align_$(date +%Y%m%d_%H%M%S) ./to_submit.sh
OUT_DIR=<that dir> ./to_submit_timescales.sh
python -m cli degeneracy --out <that dir>
python scripts/plot_timescales.py --out <that dir>
```

**Related:** `DECISIONS.md` #59–#64,
`vieb_v2/representation/{observations,transfer_operator,recordings}.py`,
`vieb_v2/scripts/{moseq_control,plot_timescales}.py`
