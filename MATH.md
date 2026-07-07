# VIEB Mathematics Reference

This document summarizes every non-trivial mathematical method used in the
VIEB pipeline: what it computes, where it lives, why it was chosen, and the
math behind it. It exists so that a reader (or a future contributor) can
understand the quantitative reasoning of the pipeline without reverse
engineering it from code. It covers both the long-standing pipeline and the
statistics-methods branch additions (alternative feature representations +
validation framework); the new material is marked **(new)**.

For architectural/product rationale (why a method was chosen over
alternatives), see `docs/DECISIONS.md`. This document is about the math
itself.

---

## 1. Feature extraction (`ml/feature_extraction.py`)

`PoseFeatureExtractor` turns a `(T, K, 2)` pose array (T frames, K
keypoints, x/y) into a `(T, F)` feature matrix. It is split into a
universal **Layer 1** (works for any keypoint layout) and a conditional
**Layer 2** (semantic features that need specific keypoints resolved).

### 1.1 Savitzky-Golay smoothing
**Where:** `_smooth_pose()`. **Why:** raw DLC keypoint tracking is noisy
frame-to-frame; a polynomial local-fit filter removes jitter while
preserving the shape of genuine movement (unlike a moving average, which
flattens peaks).
**Math:** for each keypoint coordinate trajectory, fits a degree-2
polynomial to a sliding window of `smooth_window` frames (least squares)
and evaluates it at the window center — `scipy.signal.savgol_filter(x,
window, polyorder=2)`.

### 1.2 Velocity, acceleration, speed
**Where:** `_compute_velocity()`. **Why:** raw position alone doesn't
capture movement dynamics.
**Math:** central difference for interior frames,
$$v_t = \frac{x_{t+1} - x_{t-1}}{2/\text{fps}}$$
forward/backward difference at the boundaries. Acceleration is the
velocity of the velocity signal (same operator applied twice). Speed is
the Euclidean norm of velocity per keypoint.

### 1.3 Pairwise distances
**Where:** `_compute_pairwise_distances()`, `scipy.spatial.distance.pdist`.
**Why:** the relative configuration of keypoints (how far the nose is from
the tail, etc.) is a simple, interpretable posture signal. **Confound
(addressed by the new `shape_space` mode, §7.1):** these distances mix
body size, position, and orientation with actual posture — a mouse twice
as large produces roughly 2x the distances for an identical posture.
**Math:** condensed pairwise Euclidean distance matrix per frame, $\binom{K}{2}$ values.

### 1.4 Body orientation and elongation (PCA)
**Where:** `_compute_body_orientation()`, `_compute_elongation()`.
**Why:** a single scalar "which way is the animal facing" and "how
stretched out is it" — cheap proxies for posture that don't depend on
which keypoints are semantically resolved.
**Math:** per frame, mean-center the keypoint cloud, compute the 2x2
covariance matrix $C = \frac{1}{K}\sum_k (p_k - \bar p)(p_k - \bar p)^T$,
eigendecompose it. Orientation is the angle of the dominant eigenvector
($\arg\max$ eigenvalue $\lambda_1$); elongation is $\sqrt{\lambda_2/\lambda_1}$
— 0 for a perfect line, 1 for a circle (isotropic spread). When named
roles are resolved (nose/tail_base), orientation instead uses the direct
tail→nose vector angle, which is more semantically precise.

### 1.5 Angular velocity, movement entropy, temporal stats
**Where:** `_compute_angular_velocity` (rate of change of orientation,
angle-unwrapped to avoid ±π discontinuities), `_compute_movement_entropy`
(windowed Shannon entropy of the speed histogram — see §5.1 for the
entropy formula — high entropy = erratic/unpredictable speed, low = steady
movement or steady stillness), and `_compute_temporal_features` (sliding-window
mean/std/max/p90 of speed and distance, mean/max-abs of angular velocity —
8 summary columns capturing *how* a feature is behaving over the recent
past, not just its instantaneous value).

### 1.6 Morlet continuous wavelet transform
**Where:** `_compute_wavelet_features()`, `_morlet_cwt()`.
**Why:** speed alone tells you *how fast*; a wavelet transform tells you
*at what timescale* the speed is oscillating — e.g. a grooming bout
produces energy at a different frequency band than a locomotion bout,
even if mean speed is similar.
**Math:** complex Morlet wavelet
$$\psi(t) = e^{i\omega t} e^{-t^2/2}, \quad \omega = 5$$
convolved (via FFT, `scipy.signal.fftconvolve`) with each keypoint's speed
signal at 5 fixed frequencies (1, 2, 4, 8, 16 Hz), with wavelet width
$s = \frac{\omega \cdot \text{fps}}{2\pi f}$ per frequency $f$. The
amplitude envelope $|\text{CWT}(t, f)|$ at each frequency becomes a
feature — 5 columns per keypoint.

### 1.7 Semantic (Layer 2) features
**Where:** `_compute_rearing_score()` (ear-span / nose-tail-distance
ratio — rears produce a larger apparent ear span relative to body length
when viewed from certain angles), `_compute_head_angle()` (signed angle
between the body axis and the head axis). Both require specific keypoint
roles to resolve and are silently omitted (not zero-filled) when they
don't — see `docs/DECISIONS.md` #2.

---

## 2. Preprocessing (`ml/preprocessing.py`)

`BehaviorPreprocessor(use_pca=False)` — standardization only in the
production pipeline (UMAP handles dimensionality reduction separately).
**Math:** z-score per feature, $z = \frac{x - \mu}{\sigma}$
(`sklearn.preprocessing.StandardScaler`), fit on training data only, with
optional outlier clipping at ±5σ applied only at `.transform()` time (not
during `.fit()`, so the scaler's own statistics aren't distorted by the
clipping it will apply to new data).

---

## 3. Dimensionality reduction — UMAP

**Where:** `compare.py cmd_cluster()`, via `umap-learn` (CPU) or
`cuml.manifold.UMAP` (GPU). **Why:** HDBSCAN needs the "curse of
dimensionality" reduced — density-based clustering degrades in 51-91D raw
feature space; UMAP preserves local neighborhood structure in ~10D, which
HDBSCAN can then cluster meaningfully.
**Math (brief — this is an existing, well-documented library, not
reimplemented here):** UMAP builds a fuzzy simplicial set representing the
high-dimensional neighborhood graph (via a locally-adaptive exponential
kernel calibrated so each point has a fixed effective number of neighbors,
`n_neighbors=30`), then optimizes a low-dimensional embedding via
stochastic gradient descent to minimize cross-entropy between the
high-D and low-D fuzzy simplicial sets. Fit on a ≤200k-frame subsample for
memory, then `.transform()` on every frame.

---

## 4. Clustering — HDBSCAN

**Where:** `compare.py cmd_cluster()`, via `hdbscan` (CPU) or
`cuml.cluster.HDBSCAN` (GPU). **Why:** unlike K-means, HDBSCAN doesn't
require a pre-specified number of clusters, finds clusters of varying
density/shape, and explicitly labels ambiguous points as noise (`-1`)
rather than forcing them into the nearest cluster — appropriate for
behavior, where many frames are transitional and don't belong cleanly to
any discrete state.
**Math (brief):** builds a minimum spanning tree over mutual-reachability
distances (a density-corrected distance metric), extracts a cluster
hierarchy from it, then condenses that hierarchy by selecting the split
that maximizes cluster *stability* (persistence across a range of density
thresholds) — the Excess of Mass method (`cluster_selection_method="eom"`).
Soft cluster-membership probabilities come from each point's mutual
reachability distance to its assigned cluster's persistent core. Fit on a
capped subsample (`--hdbscan-sample`, default 300k frames) for memory,
remaining frames assigned via `approximate_predict` (nearest-exemplar
matching in the fit space).

### 4.1 Gini coefficient (state-size imbalance)
**Where:** `compare.py _gini()`, used in `_generate_diagnostics()`.
**Why:** flags a degenerate clustering where one state dominates and the
rest are near-empty (a red flag that `--min-cluster-size` may be
miscalibrated).
**Math:** for sorted non-negative values $v_1 \le \dots \le v_n$,
$$G = \frac{\sum_{i=1}^n (2i - n - 1)\,v_i}{n \sum_i v_i}$$
0 = perfectly equal state sizes, 1 = maximal inequality (all frames in one
state).

---

## 5. Temporal smoothing — Hidden Markov Model (Viterbi)

**Where:** `compare.py _fit_hmm()`, `_hmm_viterbi()`, `_smooth_with_noise()`.
**Why:** HDBSCAN labels each frame independently, so isolated single-frame
misclassifications ("flicker") are common even within an otherwise stable
bout. An HMM uses the *sequence* — the fact that behavioral states persist
across many consecutive frames — to correct these.
**Math:** parameters estimated directly from the raw HDBSCAN labels
(empirical, not EM-fit):
- **Prior** $\pi_s$ = fraction of frames in state $s$.
- **Transition matrix** $A_{ij}$ = row-normalized empirical count of
  consecutive-frame pairs $(s_t = i, s_{t+1} = j)$.
- **Emission matrix** $B$ = a "soft identity": $B_{ss} = 1-\varepsilon$,
  $B_{ij} = \varepsilon/(n-1)$ for $i \ne j$ ($\varepsilon = 0.05$) — i.e.
  the observed (raw HDBSCAN) label is assumed correct 95% of the time,
  giving Viterbi just enough slack to overrule a single flickered frame
  when the surrounding context strongly disagrees.

Viterbi decoding (log-space, to avoid numerical underflow over long
sequences) finds the most likely *sequence* of true states given the
observed (raw) label sequence:
$$\delta_t(s) = \max_{s'} \big[\delta_{t-1}(s') + \log A_{s's}\big] + \log B_{s,\,o_t}$$
with backpointers `psi` recording the argmax at each step, then a
backward pass reconstructs the optimal path. Run independently on each
contiguous non-noise segment of each video (noise frames, label `-1`, are
never smoothed — they're preserved exactly, since HDBSCAN's noise
designation is itself meaningful, not something to correct away).

---

## 6. Cohort-level and quantification statistics

### 6.1 Shannon entropy (behavioral diversity, movement entropy)
**Where:** `quantify.py` (`behavioral_diversity`, `transition_entropy_A/B`),
`ml/feature_extraction.py` (`_compute_movement_entropy`).
**Why:** a single scalar for "how spread out" a distribution is — high
entropy means an animal/window visits many states/speeds roughly equally;
low entropy means it's dominated by one.
**Math:** for a probability distribution $p$ (state-occupancy fractions,
or a speed histogram),
$$H(p) = -\sum_i p_i \log p_i$$
(natural log; $0\log 0 := 0$).

### 6.2 Mann-Whitney U test + effect size
**Where:** `cohort_analysis.py` (pairwise cohort comparisons per state),
`compare.py`/`quantify.py` (ad hoc pairwise comparisons).
**Why:** a non-parametric test for "are these two groups' distributions
different" that doesn't assume normality — appropriate for behavioral
state fractions, which are bounded in [0,1] and often skewed.
**Math:** `scipy.stats.mannwhitneyu` (rank-sum based U statistic);
effect size reported as
$$r = \frac{|z|}{\sqrt{n_a+n_b}} \cdot \text{sign}(\bar x_a - \bar x_b)$$
where $z$ is the U statistic's normal-approximation z-score — a
scale-free effect size comparable across states/tests, signed so the
direction of the effect (which group has the higher mean) is preserved.

### 6.3 Benjamini-Hochberg FDR correction
**Where:** `cohort_analysis.py _bh_correct()` (canonical implementation;
`quantify.py` has a duplicate — see `docs/DECISIONS.md` #45 for the
"one shared helper" precedent this should eventually follow).
**Why:** testing every state pairwise inflates the false-positive rate;
FDR correction controls the *expected proportion of false discoveries*
among rejected hypotheses, which is less conservative (more statistical
power) than a Bonferroni family-wise-error correction while still
correcting for multiple comparisons.
**Math:** sort p-values ascending, $p_{(1)} \le \dots \le p_{(n)}$; adjusted
value $\tilde p_{(i)} = \min_{j \ge i} \left( p_{(j)} \cdot \frac{n}{j} \right)$
(the running-minimum-from-the-top ensures monotonicity). Uses
`statsmodels.stats.multitest.multipletests` when available, with a
hand-rolled fallback (identical formula) when it isn't.

### 6.4 Linear regression (learning rates)
**Where:** `quantify.py compute_state_learning_rates()`.
**Why:** "is this animal's freezing increasing over days" is naturally a
slope — ordinary least-squares gives both the rate (slope) and a fit
quality check ($r^2$) in one step.
**Math:** `scipy.stats.linregress(day, state_occupancy)` — standard OLS,
$\hat\beta = \frac{\text{Cov}(x,y)}{\text{Var}(x)}$, reported alongside
$r^2$ (rejects the slope as meaningless if the fit is poor).

### 6.5 Contrast vectors
**Where:** `quantify.py compute_contrast_vector()`.
**Why:** a single vector (and scalar magnitude) summarizing how an
animal's behavioral profile shifts between two conditions (e.g.
context A vs B), excluding whichever state is most dominant overall (so
the comparison isn't swamped by one common, uninformative behavior).
**Math:** per animal, $\Delta_s = \bar p_{A,s} - \bar p_{B,s}$ for each
non-dominant state $s$ (mean state-occupancy fraction per context, then
differenced); the contrast vector is $(\Delta_s)_{s}$, and
`contrast_magnitude` is its Euclidean norm $\|\Delta\|_2$.

---

## 7. Alternative pose feature representations **(new)**

Selected via `compare.py --feature-mode`; see `ml/representations/` and
`docs/DECISIONS.md` #51. All three address a specific limitation of the
default representation (§1) — see each subsection.

### 7.1 Procrustes / Kendall shape space (`shape_space` mode)

**Where:** `ml/representations/shape_space.py`.
**Addresses:** §1.3's confound — raw pairwise distances and orientation
mix body size, position, and camera angle with actual posture.
**Why this math:** Kendall's shape space is the standard mathematical
formalization of "shape" as *everything about a configuration of points
except its position, scale, and rotation* — exactly the invariance this
problem needs, and (per `docs/DECISIONS.md` #6) it's closed-form and
interpretable rather than learned.

**Math**, per frame, on the $(K, 2)$ keypoint configuration $X$:
1. **Remove translation:** center on the centroid,
   $\tilde X = X - \bar X$.
2. **Remove scale:** divide by centroid size (Kendall's convention — the
   RMS distance to the centroid),
   $$X' = \tilde X \Big/ \sqrt{\tfrac{1}{K}\textstyle\sum_k \|\tilde x_k\|^2}$$
   The result of steps 1-2 is the "pre-shape."
3. **Remove rotation:** align to a reference pre-shape $R$ via the
   orthogonal Procrustes solution (2D Kabsch algorithm) — find the
   rotation $\hat R$ minimizing $\|X'\hat R - R\|_F^2$, given by the SVD
   of the cross-covariance $M = X'^T R = U\Sigma V^T$:
   $$\hat R = V \, \text{diag}(1,\dots,1,\det(VU^T)) \, U^T$$
   (the `diag(...,det(VU^T))` term flips the last singular vector's sign
   when needed to exclude reflections — a valid rotation must have
   determinant +1, and without this correction the Kabsch solution can
   return a mirror-flip instead of a rotation).
4. **Generalized Procrustes Analysis (GPA):** since step 3 needs a
   reference, and no single frame's pose should be privileged as "the"
   reference, iterate: align all frames to the current reference, set the
   new reference to the mean of the aligned shapes (renormalized to unit
   size), repeat (3 iterations by default) — this converges to the mean
   shape of the whole video, so no frame is arbitrarily privileged.

The final per-frame feature vector is the flattened invariant shape
coordinates ($K \times 2$ values) plus a few derived dynamics (per-keypoint
"shape speed" — the frame-to-frame rate of change of shape coordinates,
and an overall shape velocity/acceleration norm) so clustering retains
access to movement dynamics, not just static invariant posture.

**Validated in `tests/test_feature_representations.py`:** a synthetic
posture at 4 different scales/rotations collapses to (near-)identical
shape coordinates, while the *default* extractor's raw pairwise distances
scale linearly with size (unaffected — confirming the fix actually
changes the representation, not just adding invariant-sounding math that
happens not to matter).

### 7.2 Takens delay embedding (`delay_embedding` mode)

**Where:** `ml/representations/delay_embedding.py`.
**Addresses:** the default representation only looks at instantaneous
(or short-sliding-window) statistics; it doesn't reconstruct the
underlying low-dimensional dynamical attractor a repetitive behavior
(grooming, locomotion gait) actually traces out over time.
**Why this math:** Takens' theorem (1981) states that for a
deterministic dynamical system, a scalar time series sampled from it —
delay-embedded into a sufficiently high-dimensional space — is (generically)
diffeomorphic to the system's true (possibly higher-dimensional)
attractor. In practice, this means a single 1D signal (here: centroid
speed, and PCA elongation — both keypoint-layout-agnostic) can be turned
into a feature vector that captures the *dynamics*, not just the
instantaneous value.

**Math:** the delay-embedding vector at time $t$, for delay $\tau$ and
embedding dimension $d$:
$$y(t) = \big[x(t),\, x(t-\tau),\, x(t-2\tau),\, \dots,\, x(t-(d-1)\tau)\big]$$

Takens' theorem doesn't specify $\tau$ or $d$ — they must be chosen per
signal/dataset:

- **$\tau$ selection — average mutual information (AMI):** choose the
  first local minimum of
  $$I(\tau) = \sum_{a,b} p(x_t{=}a,\, x_{t+\tau}{=}b) \log\frac{p(a,b)}{p(a)p(b)}$$
  (estimated via a 2D histogram over the signal's value range). The first
  local minimum is the standard heuristic (Fraser & Swinney 1986): too
  small a $\tau$ makes consecutive embedding coordinates near-redundant
  (near-1 correlation, not adding information); too large decorrelates
  them entirely (adding only noise). The minimum of shared information is
  the sweet spot. Falls back to the first autocorrelation zero-crossing
  if no clear AMI minimum exists.
- **$d$ selection — false nearest neighbors (FNN, Kennel, Brown & Abarbanel 1992):**
  for increasing $d$, embed the signal and find each point's nearest
  neighbor; check whether that neighbor is still close after adding one
  more embedding dimension. A neighbor that becomes far apart in $(d{+}1)$-D
  was only a "false" neighbor — an artifact of projecting the true
  attractor down into too few dimensions, causing distant points to
  spuriously overlap. The smallest $d$ at which the false-neighbor
  fraction drops below a threshold (1%) is taken as the true embedding
  dimension.

Both selections run once per signal on a calibration sample (`fit()`),
then are held fixed for every video in the run (`transform()`), since
pooled clustering requires a consistent feature dimensionality across all
videos — analogous to how UMAP is fit on a sample and then applied to all
frames.

### 7.3 Persistent homology (`topological` mode)

**Where:** `ml/representations/topological.py`, using the `ripser`
library (Vietoris-Rips persistent homology; not reimplemented here per
the task's explicit instruction to use an existing library).
**Addresses:** none of the other representations directly summarize the
*shape* (in the topological sense — loops, connected components) that a
short burst of movement traces out; a genuinely circular/repetitive
movement (e.g., grooming) has different topology than translating,
erratic, or still movement, even if speed/posture statistics look similar.
**Why this math:** persistent homology is the standard tool for
quantifying "shape" of a point cloud in a way that is robust to noise and
doesn't require choosing a single distance-threshold — it tracks
topological features (connected components = $H_0$, loops = $H_1$) across
*all* thresholds simultaneously and reports how long each one persists.

**Math:** for a window of $W$ frames, pool the $K$ keypoints across all
$W$ frames into one point cloud ($W \times K$ points in $\mathbb{R}^2$).
Build the Vietoris-Rips filtration: at each distance threshold $\varepsilon$,
connect all point pairs within $\varepsilon$ of each other, forming a
simplicial complex; as $\varepsilon$ grows, topological features (connected
components, loops) appear ("birth") and later merge/fill in ("death").
The **persistence diagram** records each feature's $(\text{birth}, \text{death})$
pair; its **persistence** is $\text{death} - \text{birth}$ — long-persisting
features are genuine topological structure, short-lived ones are noise.
Five summary statistics are extracted per window: total persistence in
$H_0$ and $H_1$ (sum of all persistences — a measure of overall
topological complexity), counts of "significant" features in each
dimension (persistence above a threshold), and max $H_1$ persistence (the
most prominent loop, if any).

Computed on non-overlapping strided windows (default: 0.75s, stride 5
frames) rather than every frame, since a Vietoris-Rips computation is
$O(n^3)$-ish in point-cloud size and per-frame computation would be
prohibitively expensive — each window's summary is broadcast to every
frame it covers. Runtime is measured directly (not assumed) by
`benchmark_feature_modes.py` and in
`tests/test_feature_representations.py::test_topological_runtime_bounded`.

---

## 8. Validation framework **(new)**

`ml/validation_stats.py` — the quantitative bar for comparing feature
representations against each other and against the default (see
`docs/DECISIONS.md` #52): a representation is "better" if it produces
more repeatable, less-conflated behavioral states, not if it produces a
visually cleaner UMAP embedding.

### 8.1 Nakagawa & Schielzeth adjusted repeatability (R)

**Where:** `compute_repeatability_R()`, wired into `compare.py cmd_report()`.
**Why this math:** repeatability (an intraclass correlation) is the
standard behavioral-ecology metric for "how much of the variation in a
trait is between individuals versus noise within an individual across
repeated measurements" (Nakagawa & Schielzeth, 2010, *Biological Reviews*).
Applied here to state occupancy: if an animal's fraction-of-time-in-state-k
is a real, stable trait, it should be more similar across that animal's own
repeated sessions than across different animals' sessions. High R means
the state is capturing something real and animal-specific; low R means
it's dominated by session-to-session noise (or the state isn't behaviorally
meaningful).

**Math:** one-way ANOVA across individuals (animals), computed per state
column:
$$MS_{\text{among}} = \frac{\sum_i n_i (\bar y_i - \bar y)^2}{a-1}, \qquad
MS_{\text{within}} = \frac{\sum_i \sum_j (y_{ij} - \bar y_i)^2}{\sum_i (n_i - 1)}$$
where $a$ = number of animals, $n_i$ = number of sessions for animal $i$,
$y_{ij}$ = state-occupancy fraction for animal $i$'s session $j$. The
unequal-sample-size correction factor (needed when animals have different
numbers of sessions):
$$n_0 = \frac{1}{a-1}\left(\sum_i n_i - \frac{\sum_i n_i^2}{\sum_i n_i}\right)$$
Between-individual and within-individual variance components:
$$V_{\text{among}} = \max\!\left(0, \frac{MS_{\text{among}} - MS_{\text{within}}}{n_0}\right), \qquad
V_{\text{within}} = MS_{\text{within}}$$
(clipped to 0 because a variance component can't be negative — a
negative point estimate just means the true value is close to 0).
Adjusted repeatability:
$$R = \frac{V_{\text{among}}}{V_{\text{among}} + V_{\text{within}}} \in [0,1]$$
Computed purely with numpy/scipy (no `statsmodels`/mixed-model dependency
needed) — requires ≥2 animals each with ≥2 sessions; otherwise skips
gracefully rather than reporting a meaningless number.

### 8.2 Transition-graph modularity / bridge-state detection

**Where:** `compute_transition_modularity()`, wired into `compare.py cmd_report()`.
**Why this math:** if a discovered "state" actually straddles two
functionally distinct behavioral regimes, its transitions should split
roughly evenly between the two regimes' other states, rather than
belonging clearly to one neighborhood — the same signature graph
community detection is built to find (a "bridge" node connecting two
otherwise well-separated communities).

**Math:**
1. **Build the graph:** node per state; edge weight = symmetrized
   transition count, $w_{ij} = c_{ij} + c_{ji}$ (aggregated across all
   videos), where $c_{ij}$ is the raw count of frame-to-frame transitions
   from state $i$ to state $j$.
2. **Community detection — Louvain algorithm:** greedily and iteratively
   moves nodes between communities to maximize **modularity**,
   $$Q = \frac{1}{2m}\sum_{ij} \left(w_{ij} - \frac{k_i k_j}{2m}\right)\delta(c_i, c_j)$$
   where $m = \frac12\sum_{ij} w_{ij}$, $k_i = \sum_j w_{ij}$ is node $i$'s
   weighted degree, and $\delta(c_i,c_j)=1$ iff nodes $i,j$ are in the same
   community — i.e., modularity rewards partitions where within-community
   edge weight exceeds what you'd expect from each node's degree alone
   under a random-rewiring null model. (`networkx.algorithms.community.louvain_communities`.)
3. **Bridge score per state:** the fraction of a state's total transition
   weight that crosses into a *different* community than its own,
   $$\text{bridge}(s) = \frac{\sum_{j:\, c_j \ne c_s} w_{sj}}{\sum_j w_{sj}}$$
   States with bridge score above a threshold (default 0.4 — i.e. 40%+ of
   transitions leave the state's assigned community) are flagged as
   `possible_split_states` — a signal (not a certainty) that HDBSCAN may
   have merged two distinct behaviors into one cluster.

Requires ≥3 states (community detection is not meaningful below that) and
skips gracefully when there are no cross-state transitions at all.

---

## Summary table

| Method | File | Library | New? |
|---|---|---|---|
| Savitzky-Golay smoothing | `ml/feature_extraction.py`, `ml/pose_utils.py` | scipy | existing + new (shared helper) |
| PCA orientation/elongation | `ml/feature_extraction.py`, `ml/pose_utils.py` | numpy | existing + new (shared helper) |
| Morlet CWT | `ml/feature_extraction.py` | scipy (hand-rolled conv) | existing |
| Shannon entropy | `ml/feature_extraction.py`, `quantify.py` | numpy | existing |
| Standardization | `ml/preprocessing.py` | scikit-learn | existing |
| UMAP | `compare.py` | umap-learn / cuml | existing |
| HDBSCAN | `compare.py` | hdbscan / cuml | existing |
| Gini coefficient | `compare.py` | numpy | existing |
| HMM Viterbi smoothing | `compare.py` | numpy (hand-rolled) | existing |
| Mann-Whitney U + effect size | `cohort_analysis.py` | scipy | existing |
| Benjamini-Hochberg FDR | `cohort_analysis.py` | statsmodels / hand-rolled | existing |
| OLS regression (learning rates) | `quantify.py` | scipy | existing |
| Contrast vectors | `quantify.py` | numpy | existing |
| **Procrustes / Kendall shape space** | `ml/representations/shape_space.py` | numpy (SVD) | **new** |
| **Takens delay embedding + AMI/FNN** | `ml/representations/delay_embedding.py` | numpy/scipy | **new** |
| **Persistent homology** | `ml/representations/topological.py` | ripser | **new** |
| **Nakagawa-Schielzeth repeatability (R)** | `ml/validation_stats.py` | numpy/scipy (ANOVA) | **new** |
| **Transition-graph modularity (Louvain)** | `ml/validation_stats.py` | networkx | **new** |
