# Feature Ablation & Dimensionality Study — Findings

**Status: TEMPLATE — results tables are empty until run per project.**

This document records the evidence for a human decision about VIEB's default
feature set. It does **not** change the production default. Fill in the
per-project tables by running `feature_ablation.py` against each project
(see "How to run"), then write the interpretation in the prose slots.

## Hypothesis under test

VIEB's clustering conflates behaviorally distinct states (e.g. grooming vs.
freezing) **not because it lacks features, but because it has too many**. At
91 features the curse of dimensionality flattens density contrast in the
distance metric UMAP/HDBSCAN rely on, so the few features that actually
separate two states (per-keypoint paw/nose speeds — high in grooming,
near-zero in freezing) get drowned out by dozens of low-value dimensions.
The study cuts features down to a minimal high-signal set; it never adds new
feature families.

## Method

`feature_ablation.py` re-runs the **unchanged** clustering pipeline
(`BehaviorPreprocessor` standardize → UMAP → HDBSCAN) on column-masked
subsets of the already-extracted feature matrix — never re-extracting from
pose, never pooling two projects. Feature families (per §1 of `MATH.md`):

| Family | Cols (K=8) | Note |
|---|---|---|
| per_keypoint_speed | 8 | hypothesized high-signal |
| pairwise_distances | 28 | |
| centroid_speed | 1 | |
| body_orientation | 1 | removal candidate |
| elongation | 1 | |
| angular_velocity | 1 | |
| movement_entropy | 1 | removal candidate |
| semantic (rearing/head_angle) | 0–2 | conditional on keypoint roles |
| temporal_window_stats | 8 | |
| wavelets | 40 | prime removal candidate |

Each subset is scored on: **DBCV** (density-based cluster validity — the
direct measure of the hypothesis), **repeatability R** with a bootstrap CI,
**ARI stability** (partition robustness across resamples), **modularity Q**,
noise fraction, and number of states. See `MATH.md` §8/§10 for the math.

## How to run

Run once per project — switch the active project between runs. Luna and
Spence are analyzed **independently**; different optimal sets are a valid,
expected outcome.

```bash
# 1. Ensure features are extracted for the active project:
python compare.py --extract
# 2. (Optional, for the Part D shape-space rows):
python compare.py --extract --feature-mode shape_space
# 3. Run the ablation study:
python feature_ablation.py                 # full study (baseline + LOO + cumulative + shape-space)
python feature_ablation.py --study leave_one_out   # just one study
```

Output: `results/ablation/feature_ablation_<project>.csv` (upserted by
subset name on re-run).

---

## Part A — Standardization audit

**Result: PASS (no bug).** The clustering path applies
`BehaviorPreprocessor(use_pca=False)` → `StandardScaler` (per-column
z-score) to the entire pooled feature matrix, fit on the exact same data
that is then clustered; no column bypasses standardization and nothing is
concatenated after it before UMAP (verified in `compare.py cmd_cluster`, and
enforced at runtime by the audit assertion in
`feature_ablation._standardize`). Full write-up: `MATH.md` §9.

A single unstandardized high-variance feature would dominate the Euclidean
distance metric and could alone explain poor clustering — this is confirmed
**not** to be happening.

---

## Project: Luna (mouse, 8-keypoint)

> Fill in after running `feature_ablation.py` on the Luna project.

### Comparison table
Paste `results/ablation/feature_ablation_luna*.csv` here.

| subset | n_features | DBCV | R [CI] | ARI stability | Q | noise% | n_states |
|---|---|---|---|---|---|---|---|
| all_features | 91 | | | | | | |
| minus_wavelets | 51 | | | | | | |
| minus_… | | | | | | | |
| only_per_keypoint_speed | 8 | | | | | | |
| cumulative_… (minimal set) | | | | | | | |
| shape_space | | | | | | | |
| shape_space+per_keypoint_speed | | | | | | | |

### Interpretation (fill in)
- **Leave-one-family-out impact:** _which families, when removed, do NOT
  degrade (or improve) DBCV/stability/R? Those are dilutive dead weight._
- **Recommended minimal set:** _the smallest subset matching/beating the
  91-feature baseline._
- **Shape-space replacement:** _won / lost / tied vs the current set?_
- **Grooming/freezing separation:** _did the previously-conflated states
  separate under the minimal / speed-emphasizing set? (Requires
  `results/annotations/annotations.csv`; skipped if absent.)_

---

## Project: Spence (rat, 5-keypoint leg-only)

> Fill in after running `feature_ablation.py` on the Spence project.

### Comparison table
Paste `results/ablation/feature_ablation_spence*.csv` here.

| subset | n_features | DBCV | R [CI] | ARI stability | Q | noise% | n_states |
|---|---|---|---|---|---|---|---|
| all_features | | | | | | | |
| … | | | | | | | |

### Interpretation (fill in)
- **Leave-one-family-out impact:** …
- **Recommended minimal set:** …
- **Shape-space replacement:** …
- **⚠ Clustering-sufficiency flag:** Spence's data is stride/gait-periodic.
  If no feature subset yields well-separated, repeatable states (low DBCV /
  low R across the board), that is evidence that **density clustering alone
  may be insufficient** for Spence's periodicity structure — a finding to
  raise for a future decision. **Do not build stride-specific machinery off
  this study** (confirm-before-building); just record the observation here.

---

## Cross-project summary & proposed decision

> Fill in after both projects are run.

- Luna recommended set: …
- Spence recommended set: …
- Do the projects agree on which families are dilutive? …

The proposed default-feature-set change (if any) is logged as a **proposed**
entry in `docs/DECISIONS.md` for a human decision — the production default is
**not** changed by this study.
