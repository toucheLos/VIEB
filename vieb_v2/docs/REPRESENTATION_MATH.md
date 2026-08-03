# VIEB v2 — Representation Backend: Mathematics

**Status: DRAFT FOR REVIEW. No code written. Awaiting sign-off on the math.**

Pipeline under consideration:

> egocentric alignment → pooled PCA → delay embedding → HDBSCAN

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| $r \in \{1..R\}$ | recording (session); $T_r$ frames |
| $K$ | keypoints per frame (VIEB: 8) |
| $d$ | spatial dims (VIEB: 2, top-down) |
| $X_t^{(r)} \in \mathbb{R}^{K\times d}$ | raw pose at frame $t$ |
| $D = Kd$ | raw pose dim (VIEB: 16) |
| $q$ | retained PCs |
| $L$, $\tau$ | number of lags, lag spacing (frames) |
| $f$ | frame rate (fps) |
| $N=\sum_r T_r$ | total frames |

---

## 1. Egocentric alignment

### 1.1 What we are quotienting by

Arena position and heading are nuisances: the same behavior performed in a
different corner facing a different way must map to the same point. The nuisance
group is $SE(2) = \mathbb{R}^2 \rtimes SO(2)$, acting as
$X \mapsto XR^\top + \mathbf{1}b^\top$.

We want $\Phi$ with $\Phi(XR^\top + \mathbf 1 b^\top) = \Phi(X)$ for all
$R\in SO(2),\, b\in\mathbb{R}^2$.

### 1.2 Translation

Centering matrix $C = I_K - \tfrac1K \mathbf 1\mathbf 1^\top$, giving
$\tilde X_t = C X_t$.

Alternative: anchor on one keypoint, $\tilde X_t = X_t - \mathbf 1 x_{t,\text{anc}}^\top$.

Trade-off, stated honestly: centroid centering averages tracking noise down by
$1/\sqrt K$ but makes *every* coordinate depend on *every* keypoint — a jittery
tail tip translates the whole body. Anchor centering localizes the noise but
injects the anchor's own jitter into all coordinates. **Recommendation:**
centroid over a stable subset (spine/shoulder/hip), excluding tail tip and nose.

### 1.3 Rotation — orthogonal Procrustes

Align each frame to a reference shape $\tilde X_{\rm ref}$ by solving

$$R_t^\star = \arg\min_{R\in SO(2)} \big\| \tilde X_t R^\top - \tilde X_{\rm ref} \big\|_F^2 .$$

Expanding, with $R^\top R = I$:

$$\|\tilde X_t R^\top - \tilde X_{\rm ref}\|_F^2
= \|\tilde X_t\|_F^2 + \|\tilde X_{\rm ref}\|_F^2 - 2\operatorname{tr}\!\big(R\, M\big),
\qquad M := \tilde X_t^\top \tilde X_{\rm ref} \in \mathbb{R}^{2\times2}.$$

So minimizing distance $\equiv$ maximizing $\operatorname{tr}(RM)$.

**General solution (Kabsch/Umeyama).** With SVD $M = U\Sigma V^\top$,
$\operatorname{tr}(RM) = \operatorname{tr}(Z\Sigma)$ for orthogonal
$Z = V^\top R U$, and $\operatorname{tr}(Z\Sigma)=\sum_i Z_{ii}\sigma_i \le \sum_i\sigma_i$
with equality at $Z=I$. Hence

$$R^\star = V \operatorname{diag}\!\big(1,\ \det(VU^\top)\big)\, U^\top,$$

the $\det$ factor forbidding reflections (a reflected mouse is not a mouse).

**Closed form in 2D.** With $R(\theta)=\begin{pmatrix}\cos\theta&-\sin\theta\\ \sin\theta&\cos\theta\end{pmatrix}$,

$$\operatorname{tr}(R(\theta)M) = (m_{11}+m_{22})\cos\theta + (m_{12}-m_{21})\sin\theta$$

$$\boxed{\ \theta_t^\star = \operatorname{atan2}\big(m_{12}-m_{21},\ \ m_{11}+m_{22}\big)\ }$$

*Verified numerically:* over 2000 random trials the closed form agrees with the
SVD/Kabsch solution and with brute-force maximization over a 200k-point $\theta$
grid, and recovers $-\varphi$ exactly — **0 mismatches**.

*Check.* Let $\tilde X_t$ be the reference rotated by $+\varphi$, i.e.
$\tilde X_t = \tilde X_{\rm ref}R(\varphi)^\top$, and let the reference be
isotropic, $\tilde X_{\rm ref}^\top \tilde X_{\rm ref} = gI$. Then
$M = R(\varphi)\,gI$, so $m_{11}+m_{22}=2g\cos\varphi$ and
$m_{12}-m_{21} = -g\sin\varphi - g\sin\varphi = -2g\sin\varphi$, giving
$\theta^\star = \operatorname{atan2}(-2g\sin\varphi, 2g\cos\varphi) = -\varphi$ —
exactly the rotation that undoes the applied one. ✓

Cheaper alternative: a two-point heading
$\theta_t = \operatorname{atan2}(u_y,u_x)$ from a tail→head vector $u_t$. Faster
and more interpretable, but the estimate rests on two keypoints, so its noise is
not averaged. Procrustes uses all $K$ and is the better default.

### 1.4 Geometry of the result — three consequences that matter downstream

Write $\hat X_t = \tilde X_t R_t^{\star\top}$, $z_t = \operatorname{vec}(\hat X_t)\in\mathbb{R}^D$.

**(a) The quotient is a manifold, not a vector space.** Dimension
$2K - 3$ (subtract 2 translation + 1 rotation); for $K=8$, that is **13** of 16
raw coordinates. PCA in §2 is a *linear* method applied to data living on a
curved quotient. This is exactly Procrustes tangent-space analysis: accurate
near the reference, increasingly distorted for poses far from it.
**Practical consequence:** iterate the reference (generalized Procrustes — align
all, recompute mean, repeat to convergence) so the reference sits near the data
centroid and the linearization is at its best.

**(b) The pooled covariance has exactly three zero eigenvalues; rank is
$2K-3$.** Two come from centering, which removes the *fixed* 2-dimensional
subspace $\{\mathbf 1 b^\top\}$, identical for every frame.

The third is less obvious and is worth deriving, because the naive argument gets
it backwards. One is tempted to say: the rotational direction is the group-orbit
tangent $\frac{d}{d\theta}(XR(\theta)^\top)|_0 = XJ^\top$ with
$J=\left(\begin{smallmatrix}0&-1\\1&0\end{smallmatrix}\right)$, which is
*configuration-dependent*, hence not a fixed linear subspace, hence no exact
zero. **That reasoning is wrong.** What pins the data is not the orbit tangent
but the *stationarity condition of the alignment*, and the reference is fixed.
At the optimum ($\theta=0$ residual), $\frac{d}{d\theta}\operatorname{tr}(R(\theta)M)=0$
requires $m_{12}-m_{21}=0$, i.e.

$$\sum_k \Big(\hat X_{k,1}\,X_{{\rm ref},k,2} \;-\; \hat X_{k,2}\,X_{{\rm ref},k,1}\Big) = 0
\quad\Longleftrightarrow\quad \big\langle \hat X,\ W\big\rangle = 0,
\qquad W := \big[\,X_{{\rm ref}}^{(y)},\ -X_{{\rm ref}}^{(x)}\,\big].$$

Because $X_{\rm ref}$ is **fixed**, this is a *linear* functional of $\hat X$.
Every aligned frame lies exactly in that hyperplane, so it contributes a third
exact zero eigenvalue with null direction $W$.

*Verified numerically:* over 5000 aligned frames $|\langle\hat X,W\rangle|_{\max}
= 2.7\times10^{-15}$, and the covariance null space coincides with
$\operatorname{span}\{\mathbf 1^{(x)},\mathbf 1^{(y)},W\}$ to all three principal
angles (singular values $1,1,1$). Numerical rank $13 = 2K-3$ for $K=8$. ✓

*Scope of the claim:* this exactness requires aligning every frame to the **same
fixed reference** (true also for generalized Procrustes, whose reference is fixed
at convergence). Align each frame to its *predecessor* instead and the constraint
is no longer a single fixed hyperplane, and the rank statement fails.

**Hard consequence:** cap $q \le 2K-3$ (=13 here). Components beyond the rank are
pure roundoff. This is also the sharpest argument against whitening (§2.3):
whitening divides by $\sqrt{\lambda_i}$, so near-zero eigenvalues amplify
numerical noise without bound.

**(c) Do not remove scale — for mice.** Kendall shape space would also quotient
by scale. For 2D top-down mice, **rearing is an out-of-plane motion whose only
2D signature is apparent foreshortening** — the body gets shorter. Removing
scale deletes the rearing signal. (For faces the opposite holds; see §7.2.)

---

## 2. Pooled PCA

$$\mu = \tfrac1N\textstyle\sum_{r,t} z_t^{(r)}, \qquad
S = \tfrac1N \textstyle\sum_{r,t}(z_t^{(r)}-\mu)(z_t^{(r)}-\mu)^\top = U\Lambda U^\top,$$

$$p_t = U_q^\top (z_t - \mu) \in \mathbb{R}^q .$$

### 2.1 Why pooled, never per-recording — stated precisely

Fit per recording and you get $(\mu^{(r)}, U^{(r)})$. Two frames from different
recordings with **identical aligned pose** $z$ then map to **different**
coordinates:

$$U_q^{(r)\top}(z-\mu^{(r)}) \ \ne\ U_q^{(r')\top}(z-\mu^{(r')})
\quad\text{unless } U^{(r)}=U^{(r')} \text{ and } \mu^{(r)}=\mu^{(r')}.$$

Cluster labels would then be incomparable across animals — "state 3" would name
a different region of pose space in every video, and *any* cross-group
comparison would be meaningless. Pooling fixes a single global chart, so
equality of pose implies equality of coordinates. This is the formal content of
*"a state must mean the same thing everywhere."*

Corollary worth stating: pooled PCA is only valid if the recordings are
commensurable — same keypoint set, same camera geometry, comparable pixel scale.
If not, pose must be normalized to a common frame *before* pooling, or the top
PCs will encode rig differences rather than behavior.

### 2.2 Component count

Select smallest $q$ with $\sum_{i\le q}\lambda_i / \sum_i \lambda_i \ge \tau$.
Per the brief, **$q$ must be logged** — and alongside it: $\tau$, the full
eigenvalue spectrum, and per-component explained variance, so the choice is
auditable rather than a magic number.

### 2.3 Whitening — a real fork, not a detail

Raw scores carry variance $\lambda_i$, so Euclidean distance is dominated by
PC1. Whitening ($p_i/\sqrt{\lambda_i}$) equalizes them — and thereby promotes
low-variance noise directions to full weight in the distance HDBSCAN consumes.
**Recommendation: do not whiten.** Keep the natural variance ordering; the
variance ranking is signal, since the leading PCs are the large postural
deformations. This should be a logged, flippable setting, not hard-coded.

---

## 3. Delay embedding — the temporal element

$$\boxed{\ y_t = \big[p_t^\top,\ p_{t-\tau}^\top,\ p_{t-2\tau}^\top,\ \dots,\ p_{t-L\tau}^\top\big]^\top \in \mathbb{R}^{q(L+1)}\ }$$

The window spans $L\tau + 1$ frames $= L\tau/f$ seconds.

### 3.1 The boundary constraint, formally

Valid indices are

$$\mathcal{V} = \big\{(r,t)\ :\ L\tau \le t < T_r \big\},
\qquad N_{\rm emb} = \sum_r (T_r - L\tau).$$

Naively delay-embedding a concatenated array manufactures $R \cdot L\tau$
**chimeric vectors** splicing the end of one animal's session onto the start of
another's. These are pure artifact and — being unlike anything real — would
likely form their own spurious cluster or inflate the noise count. This must be
enforced structurally and tested, not left to convention.

### 3.2 What this buys — and what it does not

**The honest version of the Takens argument.** Takens (1981) says: for a smooth,
*deterministic, autonomous, noise-free* dynamical system with an $m$-dimensional
attractor, a generic delay map with $\ge 2m+1$ delays is an embedding —
diffeomorphic to the attractor. Mouse behavior is stochastic, non-autonomous
(tone, shock, novelty) and observed through noisy keypoints, so **Takens does
not apply as a theorem here.** It is motivation, not a guarantee.

What we actually rely on is weaker and defensible: $y_t$ is a **finite-window
trajectory descriptor**, so proximity in $y$-space means *similar recent
movement history*, not merely similar instantaneous posture. HDBSCAN has no
model of time; delay embedding smuggles a fixed quantum of temporal context into
the coordinates so that a static algorithm sees short trajectories. It is a
*partial* fix — it still models neither transition structure nor duration. An
HMM/HSMM does that, and remains the honest comparison point.

**The phase-disambiguation property (the core win).** Two frames can share a
posture but differ in what is happening. On a periodic or reversible movement,
$p_t$ alone is ambiguous — each posture recurs twice per cycle, once on the way
out, once on the way back. The lag block $p_{t-\tau}$ resolves it:

$$p_t^{(\rm A)} = p_t^{(\rm B)}\ \text{ but }\ p_{t-\tau}^{(\rm A)} \ne p_{t-\tau}^{(\rm B)}
\ \Longrightarrow\ y_t^{(\rm A)} \ne y_t^{(\rm B)}.$$

Rear-up vs rear-down; mid-stance entering vs leaving; smile onset vs offset.

**Exact relation to velocity/acceleration features.** Let
$\Delta^j p_t = \sum_{i=0}^j (-1)^i\binom{j}{i}p_{t-i}$. Stacking,

$$\big[p_t;\ \Delta p_t;\ \dots;\ \Delta^L p_t\big] = (B\otimes I_q)\big[p_t;\ p_{t-1};\ \dots;\ p_{t-L}\big],
\qquad B_{ji} = (-1)^i\tbinom{j}{i}\ (i\le j).$$

$B$ is lower-triangular with diagonal $B_{jj}=(-1)^j\neq0$, so
$\det B = \prod_{j}(-1)^j \ne 0$ and **$B$ is invertible**. Therefore the delay
vector and the (position, velocity, acceleration, …) stack **span the same
linear space** — delay embedding is a superset of v1's hand-built kinematic
features up to linear reparametrization, obtained without hand-designing
anything.

**But** $B$ is *not orthogonal*, so it does **not** preserve Euclidean distances.
Since HDBSCAN consumes distances, the two parametrizations give *different*
clusterings. The spaces are equivalent; the metrics are not. Worth saying out
loud, because "delay embedding ≈ adding velocities" is true as linear algebra
and false as clustering.

*Verified numerically:* $\det B = \pm1$ exactly as predicted for $L=1..8$, and
$BB^\top \ne I$ in every case. But the conditioning degrades fast —
$\kappa(B) = 2.6,\,7.9,\,26,\,92,\,333,\,1.2\!\times\!10^3,\,4.5\!\times\!10^3,\,1.7\!\times\!10^4$
for $L=1..8$. So the equivalence, while exact in theory, is numerically fragile
for large $L$: high-order finite differences are dominated by noise. Another
reason (§3.3) to prefer few lags at wider spacing $\tau$ over many adjacent ones.

### 3.3 Choosing $L$ and $\tau$

Distances decompose as

$$\|y_t - y_s\|^2 = \sum_{\ell=0}^{L}\|p_{t-\ell\tau} - p_{s-\ell\tau}\|^2 .$$

Pose at 30 fps is strongly autocorrelated, so with $\tau=1$ consecutive blocks
are near-duplicates: ambient dimension grows by $(L{+}1)\times$ while
*information* grows far less. Two consequences:

1. Prefer $\tau > 1$ (subsample the lags) over many adjacent lags — same window,
   less redundancy.
2. Choose $\tau$ near the first $1/e$ crossing of the PC autocorrelation, or the
   first minimum of time-delayed mutual information (Fraser–Swinney).
   **Avoid $\tau$ commensurate with a periodic behavior's period**, where
   $p_{t-\tau}\approx p_t$ and the extra block adds nothing.

Behavioral prior at 30 fps: stride cycle ≈ 0.1–0.2 s (3–6 frames), rear ≈ 0.5 s
(≈15 frames), freeze bouts seconds-long. A window $L\tau \in [5,15]$ frames
covers strides and sub-second actions; much longer and the window straddles
behavior boundaries, blurring the very transitions we want sharp.

Optional geometric down-weighting $w_\ell = \rho^\ell$ ($\rho\lesssim1$) so
recent frames dominate. Default $w_\ell = 1$; expose, log, do not bury.

---

## 4. HDBSCAN — definitions used below

- **Core distance** $d_{\rm core}^{(k)}(a)$: distance to $a$'s $k$-th nearest
  neighbour ($k=$ `min_samples`).
- **Mutual reachability:**
  $d_{\rm mreach}(a,b)=\max\{d_{\rm core}^{(k)}(a),\,d_{\rm core}^{(k)}(b),\,d(a,b)\}$.
- Build the MST of the mutual-reachability graph, cut edges in decreasing weight
  to get a hierarchy, condense by `min_cluster_size`, and select clusters
  maximizing **stability**, with $\lambda = 1/\text{distance}$:

$$\mathrm{Stab}(C) = \sum_{p\in C}\big(\lambda_p - \lambda_{C,\rm birth}\big)$$

where $\lambda_p$ is where point $p$ falls out of $C$. Unselected points get
label $-1$ (noise).

Two properties drive everything in §5: **stability is a sum over points**
(extensive in cluster size), and **core distance is a $k$-NN radius**
(set by local density).

---

## 5. The density–duration confound

This is the known limitation to design around, not hide. Here is the mechanism
in full.

### 5.1 Sampling density is the reciprocal of speed

A bout of behavior $b$ lasting $n_b$ frames traces a path in embedding space of
arclength $\ell_b = \sum_t\|y_{t+1}-y_t\| \approx n_b \bar v_b$, where
$\bar v_b$ is the mean per-frame step length. The **linear** sampling density
along that path is

$$\rho^{\rm line}_b = \frac{n_b}{\ell_b} = \frac{1}{\bar v_b}.$$

**Sampling density along the trajectory is $1/\text{speed}$ — set entirely by how
fast the behavior is, and independent of how long the bout lasts.** Freezing
($\bar v\to0$) piles unboundedly many samples into a vanishing region.

### 5.2 Duration enters separately, through cluster mass

What HDBSCAN sees is *volume* density in the $m$-dimensional region behavior $b$
occupies across all its bouts:

$$\rho^{\rm vol}_b \;\approx\; \frac{N_b\,\bar n_b}{\mathrm{Vol}(b)},$$

$N_b$ = number of bouts, $\bar n_b$ = mean frames per bout. Duration inflates the
**numerator**; speed inflates the **denominator** (a fast behavior sweeps a
larger region). Both push slow, long behaviors to high density.

For a locally-uniform density $\rho$ in $m$ effective dimensions, the $k$-NN
radius obeys $\rho\, c_m r^m \approx k$, so

$$d_{\rm core}^{(k)} \;\sim\; \Big(\tfrac{k}{\rho\,c_m}\Big)^{1/m} \;\propto\; \rho^{-1/m}.$$

High density → tiny core distance → the cluster is born early (large
$\lambda_{\rm birth}$), survives long, and — because
$\mathrm{Stab}(C)=\sum_{p\in C}(\lambda_p-\lambda_{\rm birth})$ is a **sum over
points** — its stability is further multiplied by its sheer frame count.

**So duration is rewarded twice: once through density, once through the
extensive sum.** Slow behaviors are structurally advantaged in cluster
selection. Nothing about this is a bug in HDBSCAN; it is what density clustering
of time-uniform samples *means*.

### 5.3 Illustrative magnitude

Freeze 10 s @30 fps = 300 frames; rear 0.5 s = 15 frames → **20× the samples**.
If freeze step length is ~10× smaller, the occupied volume ratio is $\sim10^{-m}$,
so for $m=3$ the density ratio is $\approx 20\times10^3 = 2\times10^4$.

### 5.4 What simulation actually shows — three corrections

The algebra above is necessary but **not sufficient**, and testing it changed the
claim. Simulations with `hdbscan` (`min_cluster_size=50`, `min_samples=10`,
$m=3$), 6 seeds each:

**(i) Large cluster mass is not by itself a bias.** Slow behavior (40 bouts ×
300 frames, tight) vs fast (40 bouts × 15 frames, long sweep), well separated:
HDBSCAN returned `n_states=2`, `noise_frac=0.0%`,
`largest_state_frac=95.2%` — which is *exactly* the true occupancy
$12000/12600$. Both behaviors were fully recovered. **A dominant state is the
correct answer when the animal genuinely spent that long in one behavior.**
`largest_state_frac` alone therefore cannot diagnose the confound; a high value
may simply be true.

**(ii) With well-separated behaviors there is no speed bias at all.** Holding
frame counts *equal* (4000 each, 40 bouts each) and varying only spatial spread
by 13×: both behaviors recovered **100.0% on 6/6 seeds**. Isolated density peaks
are found regardless of how fast they are traversed.

**(iii) The bias is a *detection* bias, and it needs a surrounding continuum.**
Repeating (ii) with both behaviors **embedded in a background continuum** of
8000 transition frames — the realistic geometry, since pose space is continuous
and behaviors are not isolated islands:

| seed | slow recovered | fast recovered |
|---|---|---|
| 0 | 100.0% | 100.0% |
| 1 | 100.0% | 100.0% |
| 2 | 100.0% | **14.7%** |
| 3 | 100.0% | **17.1%** |
| 4 | 100.0% | 95.5% |
| 5 | 100.0% | 95.9% |
| **mean** | **100.0%** | **70.5%** |

With frame count, bout count and separation all held equal, the slow behavior is
recovered perfectly every time; the fast behavior is recovered erratically and
sometimes collapses to ~15%. **This is the confound, stated correctly:** it is
not that slow behaviors produce big clusters (they legitimately do), but that
**fast behaviors fail to be detected at all when they must be distinguished from
surrounding density** — and the failure is unstable across seeds, so it will not
announce itself as an obvious artifact.

**Revised claim about the ~42% dominant state.** The mechanism is real but
narrower than "density clustering inflates slow behaviors." A large dominant
state may be entirely correct (finding i). What the simulations support is that
*fast, brief behaviors embedded in a continuum are unreliably detected*, which
inflates the dominant state's **relative** share by removing competitors rather
than by padding the winner. This is **consistent with** the ~42% seen across
VIEB, MoSeq-on-frames and TICC — all three cluster time-uniform samples — but
remains unproven for real data. §8.3 gives the diagnostic that would test it.

### 5.5 Mitigations

**(a) Per-bout uniform subsampling** (named in the brief). Each bout contributes
a fixed $n^\star$ points regardless of duration, removing the duration→mass
advantage in §5.2. **Circularity to flag:** it needs bout boundaries *before*
clustering, but bouts come *from* clustering. Requires a provisional
segmentation (first-pass clustering, or a change-point detector) — and the
result then inherits that segmentation's biases. This must be stated, not
glossed.

**(b) Arclength (uniform-in-space) resampling** — no bout labels needed.
Reparametrize each recording's trajectory by arclength $s$ rather than time $t$
and sample every $\delta$ of arclength. Then $\rho^{\rm line} = 1/\delta$ for
*all* behaviors, **exactly cancelling the $1/\bar v_b$ term** in §5.1. This
attacks the confound at its root and sidesteps the chicken-and-egg problem.

*Its own failure mode:* tracking jitter inflates arclength. A frozen mouse has
$\bar v \neq 0$ purely from keypoint noise, so arclength resampling would
faithfully sample the noise. Requires denoising (temporal smoothing of $p_t$)
first, and the smoothing bandwidth then becomes a parameter that itself
interacts with §3.3.

*A harder ceiling, found while implementing.* Resampling can only **discard**
frames, never invent them. The achievable linear density is

$$\rho^{\rm line} = \min\!\Big(\tfrac1\delta,\ \tfrac1{\bar v_b}\Big),$$

so once $\delta < \bar v_b$ every frame is already kept and the density is
capped at the frame rate. **Arclength resampling can therefore bring a slow
behavior *down* to a fast one's density, but can never bring a fast one *up*.**
The fast behavior is under-sampled at acquisition time, and no reweighting fixes
that — only a higher frame rate does. This bounds what any subsampling
mitigation can achieve, and it applies equally to per-bout subsampling (a) when
$n^\star$ exceeds a short bout's frame count.

*Shared cost of both:* cluster sizes no longer encode occupancy, so
`largest_state_frac` changes meaning between subsampled and unsubsampled runs.
Durations must be recovered afterwards by mapping labels back to all frames.

Per the brief these are **to evaluate, not to assume** — each is a run in the
§8 benchmark grid, not a default.

---

## 6. Summary of the transform chain

$$X_t \;\xrightarrow[\text{centroid}]{\text{center}}\; \tilde X_t
\;\xrightarrow[\text{Procrustes}]{\text{rotate}}\; \hat X_t
\;\xrightarrow[\text{pooled}]{\text{PCA}}\; p_t \in \mathbb{R}^q
\;\xrightarrow[\text{no crossing}]{\text{delay}}\; y_t \in \mathbb{R}^{q(L+1)}
\;\xrightarrow{\text{HDBSCAN}}\; c_t$$

Dimension audit for VIEB ($K=8,d=2$): $16 \to$ **rank exactly 13** (3 exact
zeros: 2 centering + 1 alignment, §1.4b) $\to q \le 13$, realistically 6–10
$\to q(L{+}1)\approx 24$–48.

Flag: at $q(L{+}1)\approx48$, ambient dimension is high enough for distance
concentration to blunt $k$-NN density estimates. The saving grace is that
*intrinsic* dimension stays low (the data is a trajectory, not an iid cloud) and
$k$-NN behaves according to intrinsic dimension — but estimation quality still
degrades. Prefer small $q$ and larger $\tau$ over large $q$ and many lags.

---

## 7. Why this is sensible across domains

### 7.1 Mice (VIEB's case)

- Egocentric alignment is the single biggest win: without it, clustering
  partitions by **arena location and heading**, not behavior. Same rear in two
  corners = two states. This is the failure the whole pipeline exists to prevent.
- Pooled PCA is what makes a state comparable across the 222 videos — a
  precondition for any group contrast.
- Delay embedding covers stride and postural transitions at the 5–15 frame scale.
- **Do not remove scale** (§1.4c): rearing is out-of-plane and survives in 2D
  only as foreshortening.
- The confound bites hardest exactly here: **freezing is both the slowest
  behavior and the dependent variable** in fear conditioning. The bias inflates
  the very state the experiment is about — which is why this must be measured
  (§8.3) rather than assumed benign.

### 7.2 Faces

- Landmarks (e.g. 68) in place of keypoints; alignment removes head translation
  and in-plane roll. Steps 1–2 are then **exactly the Active Shape Model**
  construction (Cootes et al.): Procrustes + PCA on landmarks. Well-trodden
  ground, which is reassurance that the first half of the pipeline is sound.
- **Where it breaks:** out-of-plane head rotation (yaw/pitch) is *not* in
  $SE(2)$ and survives 2D alignment. PCA will then spend its leading components
  on head pose rather than expression. Needs 3D landmarks, an explicit pose
  regression, or a 3D morphable model. This is a genuine limitation, not a
  tuning issue.
- **Scale: normalize here** (e.g. by interocular distance) — camera distance is
  pure nuisance. Note this is the *opposite* of the mouse recommendation,
  which is the point: the choice follows from what carries signal in each
  domain, and should not be copied across.
- **Temporal element is essential:** expression is a trajectory
  (onset → apex → offset), and genuine vs posed smiles differ mainly in
  *temporal profile*, not peak shape. A static clustering cannot represent that
  distinction in principle; a delay-embedded one can.
- Confound: a long neutral face is a vast dense cluster; a micro-expression
  (≈1/25–1/5 s) contributes a handful of widely-spread points and is likely
  labeled noise — again, the thing of interest is the thing discarded.

### 7.3 Gait

- Gait is quasi-periodic: the trajectory is approximately a **limit cycle**.
  This is where delay embedding is most strongly justified, because the
  Takens picture genuinely applies to (near-)periodic deterministic motion —
  far more so than to spontaneous behavior.
- On a closed orbit, instantaneous pose is **provably ambiguous**: each pose
  value recurs twice per cycle. Mid-swing forward and mid-stance backward can
  have near-identical joint angles and opposite velocity. $p_t$ cannot separate
  them; $y_t$ can (§3.2). Phase is recoverable only from the lags.
- Set $L\tau$ to a fraction (~½–¾) of the stride period; avoid $\tau$ at a
  multiple of the period, where the lag block degenerates to a copy.
- Pooled PCA across subjects is what makes a gait state comparable between
  individuals or patients — the same argument as across recordings, and the
  basis of any clinical contrast.
- Confound, very concretely: **stance is slow, swing is fast.** Density
  clustering will over-segment stance and under-detect swing, systematically
  distorting duty-factor estimates — a quantity gait analysis actually reports.

**Cross-cutting:** in all three domains the pipeline's strength is removing a
known nuisance group and giving trajectories rather than instants; and in all
three the *same* weakness recurs — the fast, brief, informative event
(rear, micro-expression, swing) is the one density clustering is worst at.

---

## 8. Benchmark protocol

Required: v2 vs **v1's default feature set**, same project, same HDBSCAN
parameters, reporting `largest_state_frac`, `n_states`, `state_entropy` side by
side. **No winner declared in code.**

### 8.1 A definitional catch that must be settled first

v1's metrics (`compare.py:2110-2170`) are **not** textbook definitions:

```python
state_fracs.append(float((all_labels == k).sum()) / max(1, total_frames))
...
ent = -sum(f * math.log(f) for f in pos_fracs if f > 0)
max_ent = math.log(len(pos_fracs))
state_entropy = round(ent / max_ent, 4)
```

Two things follow:

1. `state_fracs` divides by `total_frames`, which **includes noise frames**
   (`compare.py:2090`). So $\sum_k \pi_k = 1 - \text{noise\_frac} \neq 1$, and
   `largest_state_frac` is a fraction of *all* frames, not of clustered frames.
2. `state_entropy` is therefore $-\sum \pi_k\log\pi_k$ over a **sub-normalized**
   distribution, then divided by $\log(\#\text{states})$ — a *normalized* entropy
   in nats, not a plain one.

**Consequence:** if v2 computes "clean" metrics (normalize over non-noise) the
two columns are **not comparable**, and any apparent v2 advantage could be an
artifact of the definition alone. Recommendation: **recompute both arms with one
shared metric function** rather than reading v1's stored diagnostics, report
`noise_frac` alongside always, and state the convention in the output header.

Note also that normalization by $\log C$ is load-bearing: raw entropy grows
mechanically with the number of states, so comparing two methods that produce
different $n_{\rm states}$ via unnormalized $H$ is confounded by construction.

### 8.2 Fairness caveat on "same HDBSCAN parameters"

Identical `min_cluster_size` / `min_samples` across two representations of
different dimensionality and scale is a *defensible convention*, not a neutral
one — $k$-NN radii mean different things in $\mathbb{R}^{91}$ and
$\mathbb{R}^{48}$. Report the parameters, and ideally sweep them, so the
comparison is a curve rather than a single point.

### 8.3 The diagnostic that actually tests §5

Per §5.4(i), **`largest_state_frac` cannot diagnose the confound** — a large
dominant state may simply be true occupancy. The falsifiable prediction is about
*detection*, and the diagnostic must target that:

> **Predicted:** it is the **noise** label, not the dominant cluster, that is
> speed-biased. Frames labelled $-1$ should be systematically *faster* than
> clustered frames, and fast behavior should be recovered less reliably than
> slow behavior at matched frame counts.

Report:

1. **Speed of noise vs clustered frames.** Mean $\|p_t - p_{t-1}\|$ for
   $c_t = -1$ against $c_t \ge 0$. If noise frames are not faster, §5 is not
   operating and we should stop citing it.
2. **Per-cluster table:** $|C|$, mean step length, mean bout duration; rank
   correlation between size and speed.
3. **Seed stability.** Per §5.4(iii) the failure was erratic across seeds
   (100%, 100%, 15%, 17%, 96%, 96%). Re-run clustering under several
   subsample seeds and report the *variance* of `n_states` and of small-cluster
   recovery. **Instability across seeds is itself the signature** — a real
   density peak is found every time; a marginal fast behavior is not.
4. **Held-out fast behavior, if any labels exist.** Where hand-scored rears or
   other brief events are available, report their recovery rate directly. This
   is the only unambiguous test.

If (1) and (3) come back negative, the density–duration story does not explain
the ~42% and the document should be corrected again.

---

## 9. Open questions before implementation

1. **Reference shape** for Procrustes — first frame, per-project mean, or
   iterated generalized Procrustes (recommended)?
2. **$\tau$ and $L$** — fix from autocorrelation/MI per project, or expose as
   swept parameters?
3. **Subsampling** — implement per-bout (as briefed), arclength (§5.5b), or
   both as comparable arms? Per-bout needs a provisional segmentation; whose?
4. **Metric convention** — adopt v1's definitions verbatim for comparability, or
   compute both arms cleanly and re-derive v1's numbers? (Recommend the latter.)
5. **Confidence weighting** — DLC per-keypoint likelihoods are available; should
   low-confidence keypoints be down-weighted in the Procrustes fit (weighted
   Procrustes is a trivial extension) rather than trusted equally?
