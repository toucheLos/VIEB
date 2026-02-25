# VIEB Pipeline: Mathematics, Rationale, and Pattern Detection

A complete technical breakdown of how 222 mouse videos become behavioral science.

---

## Table of Contents

1. [The Problem Being Solved](#1-the-problem-being-solved)
2. [Stage 0 — Raw Data: What a Pose Is](#2-stage-0--raw-data-what-a-pose-is)
3. [Stage 1 — Feature Extraction](#3-stage-1--feature-extraction)
4. [Stage 2 — Preprocessing](#4-stage-2--preprocessing)
5. [Stage 3 — Clustering](#5-stage-3--clustering)
6. [Stage 4 — Anomaly Detection](#6-stage-4--anomaly-detection)
7. [Stage 5 — Cross-Video Shared Model](#7-stage-5--cross-video-shared-model)
8. [How Patterns Are Actually Detected](#8-how-patterns-are-actually-detected)
9. [Reading the Results: Your Data Interpreted](#9-reading-the-results-your-data-interpreted)

---

## 1. The Problem Being Solved

Traditional behavioral scoring asks a human to watch a video and annotate: "was the mouse freezing at second 14?" This is:

- **Subjective** — two scorers will disagree on ambiguous moments
- **Low resolution** — humans annotate seconds; the real behavior lives at 30 frames/second
- **Biased toward visible behaviors** — if you don't already know what to look for, you won't look for it
- **Slow** — one human, one video, many hours

VIEB inverts this. It asks: *given only the positions of 8 body points over time, what natural groupings of posture+motion exist in this animal's behavioral repertoire?* No human decides in advance what the categories are. The mathematics discovers them.

---

## 2. Stage 0 — Raw Data: What a Pose Is

DeepLabCut outputs a CSV file. After loading, the raw data is:

```
pose  shape: (T, K, 2)
conf  shape: (T, K)
```

Where:
- **T** = number of frames (e.g., 6,302 for a ~3.5 minute video at 30 fps)
- **K** = 8 keypoints: `left_ear, right_ear, nose, center, left_hip, right_hip, tail_base, tail_tip`
- **2** = pixel coordinates (x, y) in the camera frame
- **conf** = DeepLabCut's confidence that the keypoint was correctly placed (0–1)

At frame *t*, keypoint *k* is a 2D point:

```
p_{t,k} = (x_{t,k}, y_{t,k})
```

This is all the pipeline ever sees. There is no video — only numbers describing where each body part was in each frame.

**Why 8 keypoints?**
They were chosen to fully characterize:
- **Head direction**: nose, left_ear, right_ear
- **Body axis**: center (neck-level), tail_base
- **Lateral body shape**: left_hip, right_hip
- **Tail posture**: tail_tip

Together, these 8 points constrain the animal's heading, body elongation, postural symmetry, and limb configuration — enough information to distinguish freezing from locomotion, rearing from grooming, and stretched-attend from curled resting.

---

## 3. Stage 1 — Feature Extraction

**Why not use raw coordinates directly?**

Raw pixel coordinates are camera-dependent (a mouse at the top of the frame vs. the bottom of the frame has different x,y values but identical behavior). The features transform the raw coordinates into quantities that are **translation-invariant** (independent of where in the cage the mouse is) and **biologically meaningful**.

### 3.1 Velocity

The instantaneous speed of each keypoint between consecutive frames:

```
v_{t,k} = p_{t+1,k} − p_{t,k}

speed_{t,k} = ||v_{t,k}|| = sqrt((x_{t+1,k} − x_{t,k})² + (y_{t+1,k} − y_{t,k})²)
```

Units: pixels/frame. At 30 fps, 1 pixel/frame = 30 pixels/second.

This gives **K=8 speed values per frame**, capturing which body parts are moving and how fast.

### 3.2 Acceleration

The rate of change of velocity:

```
a_{t,k} = v_{t+1,k} − v_{t,k}
```

Acceleration distinguishes **sustained movement** (low acceleration, constant velocity) from **initiating/stopping movement** (high acceleration). A mouse beginning a freeze has a brief high-acceleration event before settling to zero velocity.

### 3.3 Pairwise Distances

Every pair of keypoints has an Euclidean distance:

```
d_{t,(j,k)} = ||p_{t,j} − p_{t,k}||
```

For K=8 keypoints, there are K(K−1)/2 = **28 pairwise distances**. These encode **body shape** independent of position:

- `dist(nose, tail_base)` → body elongation (stretched vs. crouched)
- `dist(left_hip, right_hip)` → lateral body width
- `dist(left_ear, right_ear)` → head width (stable reference; large change = rearing or head turn)
- `dist(center, tail_tip)` → total body length including tail posture

### 3.4 Centroid

The mean position of all keypoints:

```
c_t = (1/K) Σ_k p_{t,k}
```

The **centroid velocity** is the translational speed of the whole animal, separating whole-body locomotion from internal postural changes (e.g., a mouse that shifts its weight without moving).

### 3.5 Body Orientation

The angle of the body's main axis, estimated from the nose-to-tail_base vector:

```
θ_t = atan2(y_nose − y_tailbase, x_nose − x_tailbase)
```

Range: [−π, π]. This captures **heading direction**.

### 3.6 Angular Velocity

The rate of turning:

```
ω_t = θ_{t+1} − θ_t
```

(with wraparound correction for the ±π discontinuity). Distinguishes straight-line locomotion (ω ≈ 0) from circling, turning, or head-scanning.

### 3.7 Body Elongation

How stretched vs. compact the body is. Estimated by fitting an ellipse to the 8 keypoints at each frame using PCA on the point cloud:

```
eigenvalues λ₁ ≥ λ₂ of Cov(p_{t,1:K})
elongation_t = λ₁ / λ₂
```

A high ratio = elongated (running, stretched-attend). Ratio near 1 = compact (rearing upright, frozen in a ball).

### 3.8 Movement Entropy

Over a sliding window of W frames, compute the probability distribution of speeds, then take Shannon entropy:

```
H_t = −Σ_i p_i * log(p_i)
```

where p_i = fraction of frames in speed bin i.

Low entropy = stereotyped, consistent behavior (sustained freezing, steady locomotion).
High entropy = variable, mixed movement (exploratory sniffing, grooming transitions).

### 3.9 The Full Feature Vector

After extraction, each frame becomes a vector of **49 features** capturing:

| Feature group | Count | Captures |
|---------------|-------|---------|
| Keypoint speeds | 8 | Which body parts are moving |
| Keypoint accelerations | 8 | Starting/stopping movement |
| Pairwise distances | 28 | Body shape (posture) |
| Centroid speed | 1 | Whole-animal locomotion |
| Angular velocity | 1 | Turning rate |
| Body elongation | 1 | Stretched vs. crouched |
| Movement entropy | 1 | Behavioral consistency |
| Body orientation | 1 | Heading direction |

**Total: 49 features per frame**

The key insight: a mouse that is **freezing** will have near-zero values for speeds/accelerations, stable pairwise distances, and low entropy — regardless of *where* it is in the cage. Two mice freezing in opposite corners of the cage will produce nearly identical 49-dimensional feature vectors. That's the point.

---

## 4. Stage 2 — Preprocessing

### 4.1 Standardization (Z-score normalization)

The 49 features have wildly different scales. Pixel distances can be in the hundreds; entropy in the range [0, 3]. If we fed raw features into a clustering algorithm, features with large values would dominate the distance metric.

Solution: transform each feature to have mean 0 and standard deviation 1:

```
x'_{t,f} = (x_{t,f} − μ_f) / σ_f
```

where μ_f and σ_f are the mean and standard deviation of feature f across **all frames in all 222 videos** (for the shared model). After standardization, every feature contributes equally.

### 4.2 PCA — Principal Component Analysis

**The problem**: the 49 features are highly correlated. Speed of the nose and speed of the left_ear are both large when the mouse runs; they carry redundant information. Correlated features don't add information — they add noise.

**The math**: PCA finds a new coordinate system where the axes (principal components) are ordered by how much variance they explain, and are mutually uncorrelated (orthogonal).

Step 1: Compute the covariance matrix of the standardized features:
```
Σ = (1/N) X^T X      where X is the (N × 49) feature matrix
```

Step 2: Eigendecompose:
```
Σ = V Λ V^T
```
where V is the matrix of eigenvectors (principal components) and Λ is the diagonal matrix of eigenvalues (variances explained).

Step 3: Project data onto the top components:
```
Z = X V_k
```
where V_k contains only the eigenvectors explaining the top 95% of total variance.

**Result**: 49 correlated features → **~20 uncorrelated components** (the exact number depends on the data; for your 1.28M frames it landed at 20).

**Why 95% variance?** Keeping 95% means we discard only noise (the remaining 5%). The 20 components are the "true degrees of freedom" in mouse behavior given these 8 keypoints.

**Critical**: the preprocessing is fit on **pooled** data across all 222 videos. If you fit separately per-video, a "high speed" in video A might mean something different than "high speed" in video B. The shared fit ensures a common reference frame.

---

## 5. Stage 3 — Clustering

### 5.1 K-Means: What It Optimizes

K-Means partitions N data points into K clusters by minimizing the **within-cluster sum of squared distances** (WCSS):

```
J = Σ_{k=1}^{K} Σ_{x ∈ C_k} ||x − μ_k||²
```

where μ_k is the centroid (mean) of cluster k, and C_k is the set of points assigned to cluster k.

**Lloyd's Algorithm** (what actually runs):
1. Initialize K centroids randomly (or with K-means++ for better initialization)
2. **Assignment step**: assign each point to its nearest centroid:
   ```
   label(x) = argmin_k ||x − μ_k||²
   ```
3. **Update step**: recompute each centroid as the mean of its assigned points:
   ```
   μ_k = (1/|C_k|) Σ_{x ∈ C_k} x
   ```
4. Repeat until assignments stop changing.

**Guarantee**: J is non-increasing at each iteration; the algorithm converges. **Not guaranteed**: finding the global minimum (it finds a local minimum). Solution: run 10 restarts (`n_init=10`) with different initializations, take the best.

### 5.2 Auto-tuning K: The Silhouette Score

How many clusters? The algorithm tries K = 4, 5, 6, ..., 12 and picks the best by **silhouette score**.

For each data point i:
- **a(i)** = mean distance from i to all other points in the same cluster (intra-cluster cohesion)
- **b(i)** = mean distance from i to all points in the nearest different cluster (nearest-cluster separation)

```
s(i) = (b(i) − a(i)) / max(a(i), b(i))
```

Range: [−1, 1]
- s(i) near +1 → point is well within its cluster, far from others → good
- s(i) near 0 → point is near the boundary between two clusters
- s(i) near −1 → point is probably in the wrong cluster

The **mean silhouette score** across all points measures overall cluster quality. The K that maximizes this is selected.

**Why start at K=4?** At K=2, the silhouette score almost always peaks because "moving vs. still" is an easy binary split with perfect separation. But K=2 is scientifically useless — you already knew mice sometimes move and sometimes don't. Starting at K=4 forces the algorithm to find finer behavioral structure.

**Result for your data**: K=5 was selected, meaning 5 clusters maximized the average separation quality across 1.28M frames.

### 5.3 Why K-Means for Behavior?

Alternatives considered:
- **DBSCAN**: density-based, great for arbitrary shapes but requires tuning ε (neighborhood radius) which is dataset-specific and fails for high-dimensional data (curse of dimensionality)
- **GMM** (Gaussian Mixture Models): assumes each cluster is a multivariate Gaussian — biologically reasonable, but computationally heavier and prone to degenerate solutions with high-dimensional data
- **Hierarchical**: no predict() method, can't apply to new data — useless for cross-video generalization

K-Means' assumption (spherical clusters in Euclidean space) is reasonable after PCA because the PCA components are uncorrelated and the behavioral manifold in this space tends to be roughly spherical per-state.

---

## 6. Stage 4 — Anomaly Detection

### 6.1 The Autoencoder Architecture

An autoencoder is a neural network trained to reconstruct its own input. The architecture has two halves:

**Encoder**: compresses the input into a lower-dimensional representation (bottleneck):
```
h = f_enc(x) = σ(W_2 σ(W_1 x + b_1) + b_2)
```

**Decoder**: reconstructs the original input from the bottleneck:
```
x̂ = f_dec(h) = σ(W_4 σ(W_3 h + b_3) + b_4)
```

For your data: input dimension = 20 (PCA components), bottleneck = ~8 dimensions.

**Training objective**: minimize reconstruction loss across all training frames:
```
L = (1/N) Σ_t (1/F) Σ_f (x_{t,f} − x̂_{t,f})²
```

This is mean squared error (MSE) between the original and reconstructed feature vectors.

### 6.2 Why Reconstruction Error Detects Anomalies

The autoencoder is trained on the full dataset, which is dominated by **common behaviors** (locomotion, freezing, grooming). It learns to reconstruct these well. Its bottleneck representation has learned the "grammar" of normal mouse behavior.

When a truly unusual behavioral frame is fed in:
- The encoder maps it into the bottleneck space, but it doesn't fit the learned structure
- The decoder, which only knows how to reconstruct normal patterns, produces a poor approximation
- **Reconstruction error is high**

Formally: let E_t = MSE(x_t, x̂_t) for frame t. The **anomaly threshold** τ is the 95th percentile of E across all frames:

```
τ = percentile(E, 95)
```

Frames where E_t > τ are flagged as anomalous — they are the 5% of frames that the model finds hardest to reconstruct.

**Why 95th percentile?** It means we expect ~5% of frames to be anomalous by definition. This is a hyperparameter — you could use 99th percentile for stricter detection.

### 6.3 What Anomalies Mean for Behavior

An anomalous frame is not necessarily "wrong" — it means the frame represents a posture/motion pattern that is statistically unusual given the animal's behavioral repertoire. For fear conditioning research, anomalies are:

- Rare transitional postures (mid-rear, falling, stumbling)
- Unusual velocity signatures (sudden acceleration after long freeze)
- Postural asymmetries (limping, tilting — health indicators)
- Artifacts (keypoint tracking failure producing implausible positions)

The 3.7–4.2% anomaly rate you observed is reasonable for healthy mice in a standard conditioning box.

---

## 7. Stage 5 — Cross-Video Shared Model

### 7.1 The Problem With Per-Video Clustering

If you fit a separate K-Means on each video:
- Video A: cluster 0 = fast locomotion, cluster 1 = freezing, ...
- Video B: cluster 0 = freezing, cluster 1 = slow locomotion, ...

Cluster 0 in video A and cluster 0 in video B are **completely different behaviors**. You cannot aggregate or compare.

More subtly: the *scale* of features differs per video. A mouse in video A that moves at 5 pixels/frame might be "fast" in a cage full of slow mice. In a pool of 222 videos, 5 pixels/frame might be average speed. The meaning of "fast" is defined relative to the reference population.

### 7.2 The Solution: Pool First, Then Fit Once

**Feature pooling**: concatenate all 222 feature matrices:
```
X_pool = [X_1; X_2; ...; X_222]   shape: (1,288,650 × 49)
```

**Fit once** on X_pool: the StandardScaler's μ_f and σ_f are computed across all 1.28M frames. The PCA eigenvectors are computed from the joint covariance matrix. The K-Means centroids represent the mean feature vector of each behavioral state in the global population.

**Apply to each video individually** using the saved model:
```
Z_i = X_i @ V_k               # project video i's features into shared PCA space
labels_i = KMeans.predict(Z_i) # assign each frame to its nearest global centroid
```

Now cluster 3 in video A and cluster 3 in video B are **the same behavioral state** — defined by the same centroid in the same feature space. The fraction of frames in cluster 3 is directly comparable across all 222 videos, across all conditions, across all animals.

### 7.3 Mathematical Guarantee

The prediction step is just nearest-centroid assignment:
```
label(x) = argmin_k ||x − μ_k||²
```

This is deterministic given the trained centroids μ_k. No re-fitting, no randomness. Two frames with identical feature vectors will always receive identical labels, regardless of which video they came from.

---

## 8. How Patterns Are Actually Detected

### 8.1 The Geometry of Behavioral States

In the 20-dimensional PCA space, each frame is a point. Frames that are behaviorally similar cluster together; frames that are behaviorally different are far apart.

Why does this work? Because similar behaviors produce similar feature vectors:
- Two freeze frames, even from different videos, will have near-zero speeds, stable pairwise distances, and low entropy → their 49-feature vectors will be numerically close → they will be nearby in PCA space → they will be assigned to the same cluster.

**State 3 (likely freezing)** is the tightest cluster because freezing is the most stereotyped behavior: near-zero velocity across all 8 keypoints simultaneously. There are very few ways to freeze; there are many ways to move. This is why K-Means captures freezing well — it's a compact, well-separated sphere in feature space.

**State 4 (Context B-specific)** is a cluster that is spatially separated from the others in PCA space, but is sparse (6.1% of all frames). Its centroid represents a posture+motion combination that is qualitatively different from any of the other 4 states. The fact that it occurs predominantly in Context B is discovered *after* clustering by joining with metadata — the clustering itself is blind to experimental conditions.

### 8.2 Why These Patterns Are Invisible to Human Scoring

Human observers score behavior at the resolution of **behavioral episodes**: "the mouse was freezing for 8 seconds, then moved for 2 seconds." This collapses all within-episode variation.

The pipeline operates at **frame resolution** (1/30 second). Consider what happens in the 8 seconds before a freeze ends:
- Seconds 7–8: The mouse shows microsecond ear twitches, subtle weight shifts, a slight forward lean — all while remaining "frozen" by human standards
- The feature vector for these frames has tiny but nonzero velocities for ear/nose keypoints, a small change in body elongation, a slight increase in entropy
- These frames may cluster into State 4 rather than State 3

A human scores the entire episode as "freeze." The algorithm resolves it into a freeze-to-movement transition sequence. The transition itself — how the freeze ends — may carry information about fear memory retrieval that sustained freezing duration alone cannot capture.

### 8.3 The Discrimination Signal in State 4

Here is the concrete geometry of what you observed:

In Context A (shock context), mice spend 50.6% of time in State 3 (freeze) and 0.1% in State 4.

In Context B (safe context), mice spend 39.5% in State 3 and 17.6% in State 4.

This means **State 4 is a behavior that mice actively perform when they recognize they are safe**, and almost completely suppress when they believe they are in danger. The K-Means centroid of State 4 defines the average postural/kinematic signature of this behavior. To identify it:

1. Find frames where label = 4
2. Look at the feature values at those frames
3. The features with highest absolute deviation from the global mean define the behavior (e.g., if elongation and nose speed are both high but ear speed is low, this is stretch-attend posture with the nose extended for risk assessment while the body is still)

The behavior was always there in the videos. Human scorers weren't looking for it because they didn't know it existed. The algorithm found it because it found a dense region in feature space that is geometrically distinct from all other regions.

---

## 9. Reading the Results: Your Data Interpreted

### 9.1 The Five States (Hypotheses)

| State | Global % | Hypothesis | Evidence |
|-------|----------|------------|---------|
| 0 | 19.8% | Moderate locomotion | Consistent ~20% across all days/contexts; locomotion is always present |
| 1 | 7.0% | Novel context exploration | Spikes to 39.6% on Day 2/Context C only; normally <2% |
| 2 | 23.9% | Slow locomotion / idle | Drops from 46% (Day 0) to 17% (Day 5+); decreases as fear consolidates |
| 3 | 43.3% | Freezing | Jumps from 31% (Day 0) to 57% (Day 1) after fear acquisition; highest in Context A |
| 4 | 6.1% | Safe-context behavior | 17.6% in Context B, 0.1% in Context A; emerges Days 4–7 |

### 9.2 The Day-by-Day Fear Story

```
Day 0: Habituation     — State 3 = 31%  (baseline, no fear)
Day 1: After CFC       — State 3 = 57%  (+26%) — FEAR ACQUIRED
Day 2: Context C       — State 3 = 13%, State 1 = 40% — novel context, low fear
Day 3: Re-exposure     — State 3 = 56%  — fear reinstated by context
Day 4: CFD begins      — State 3 = 39%, State 4 = 13% — discrimination emerging
Day 5–7: CFD ongoing   — State 3 = 44–49%, State 4 = 12–13% — stable discrimination
```

Day 0 → Day 1 is classical fear conditioning. The 26% jump in State 3 is fear memory.
Day 4+ shows both high State 3 (still afraid of Context A) and rising State 4 (recognizing Context B is safe). This is fear **discrimination** — the animal has learned that the world has dangerous and safe zones.

### 9.3 What to Do With This

**Verify State 4**: run this to get representative frame numbers from a Context B video, then watch those exact frames:
```python
import numpy as np
import json

with open('results/features/index.json') as f:
    index = json.load(f)

# pick a CFD Context B video
stem = '20241113_Box_1_CFD_Day_4_(Context_B)_9001'  # example
labels = np.load(f'results/shared/{stem}_labels.npy')
state4_frames = np.where(labels == 4)[0]
sample = np.random.choice(state4_frames, 10, replace=False)
print(sorted(sample.tolist()))
```

**Fill in the `fear` column** in `metadata.csv` using the experimental design (which conditions involved shock), re-run `python compare.py --report`, and you will get Mann-Whitney U p-values testing whether fear vs. no-fear conditions differ in each behavioral state.

**The publishable claim, if State 4 is confirmed**: mice in fear-conditioned contextual discrimination experiments express a distinct behavioral state selectively in the safe context that increases with discrimination training and is not captured by traditional freezing-based scoring.

---

*This document describes the pipeline as implemented in `ml/feature_extraction.py`, `ml/preprocessing.py`, `ml/clustering.py`, `ml/anomaly_detection.py`, and `compare.py`.*
