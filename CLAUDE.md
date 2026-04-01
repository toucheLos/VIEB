# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VIEB (Video Interpreter for Experimental Behavior) analyzes mouse fear-conditioning videos. It takes DeepLabCut pose-tracking output (CSV files with 8 keypoints per frame) and runs an unsupervised ML pipeline to discover and compare behavioral states across 222 videos.

## Installation

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -e ".[tracking]"      # Full install including DeepLabCut
pip install -e .                  # Core ML pipeline only (no DLC)
```

Dependencies are in `pyproject.toml`. There is no `requirements.txt`. Python 3.10–3.12 required; 3.13+ has compatibility issues with DLC.

## Key Commands

### Pose estimation (DeepLabCut — run once per project)
```bash
python setup_dlc_training.py               # Add videos, extract frames, open labeling GUI
python setup_dlc_training.py --label       # Open next unlabeled video (numbered queue)
python setup_dlc_training.py --label 39    # Jump to video #39/222 in the queue
python setup_dlc_training.py --train       # Create dataset and train (after labeling)
python setup_dlc_training.py --evaluate    # Evaluate trained model (check mAP)
python setup_dlc_training.py --analyze     # Run pose estimation on all 222 videos → CSV
```

### Per-video behavioral analysis
```bash
python main.py --all                       # Analyze all videos that have a DLC CSV
python main.py --video raw_videos/X.mp4   # Single video
python main.py --all --n-clusters 6       # Fix number of behavioral states
python main.py --all --no-anomaly         # Skip autoencoder (faster)
```

### Cross-video comparison pipeline (run in order)
```bash
python compare.py --extract              # Extract features → results/features/  (51 features)
python compare.py --extract --no-wavelets  # Skip Morlet wavelets (faster, 51 features instead of 91)
python compare.py --cluster              # UMAP+HDBSCAN, HMM-smooth → results/shared/
python compare.py --cluster --min-cluster-size 30  # Tune HDBSCAN (default 50)
python compare.py --report               # Comparison plots + CSVs → results/comparison/
python compare.py --summarize            # Per-animal AUC + discrimination ratio
python characterize.py                   # Behavioral state profiles + t-SNE → results/characterization/
python characterize.py --clips           # Also export exemplar video clips → clips/state_<id>/
python characterize.py --n-clips 10      # Change clips per category (default 15)
```

## Architecture

### Two-tier pipeline

**Tier 1 — Pose tracking** (`setup_dlc_training.py` + `tracking/`):
DeepLabCut trains a ResNet50 model on manually labeled frames (8 keypoints: `left_ear, right_ear, nose, center, left_hip, right_hip, tail_base, tail_tip`). Outputs one CSV per video alongside each `.mp4` in `raw_videos/`.

**Tier 2 — Behavioral ML** (`ml/` package):
Stateless pipeline that transforms `(T, K, 2)` pose arrays into behavioral labels and reports. All five modules are imported via `ml/__init__.py`.

### `ml/` module responsibilities

| Module | Class | Role |
|--------|-------|------|
| `feature_extraction.py` | `PoseFeatureExtractor` | Pose → kinematic/spatial/postural + Morlet wavelet features `(T, 91)` |
| `preprocessing.py` | `BehaviorPreprocessor` | Standardize only (`use_pca=False` in `compare.py`); has `fit()` / `transform()` |
| `clustering.py` | `BehaviorClusterer` | K-Means/GMM; used only in per-video `main.py` now |
| `anomaly_detection.py` | `AnomalyDetector` | PyTorch autoencoder; flags unusual frames |
| `analysis.py` | `BehaviorAnalyzer` | Statistics, plots, JSON/CSV/text export |

### Per-video vs. cross-video analysis

`main.py --all` fits an independent model per video — cluster IDs are **not** comparable across videos.

`compare.py` fits **one shared pipeline** on all 1.28M frames pooled together:
1. Standardize with `BehaviorPreprocessor(use_pca=False)` → 91D (or 51D with `--no-wavelets`)
2. Reduce with **UMAP** (fit on ≤200k-frame sample, transform all) → 10D
3. Cluster with **HDBSCAN** (min_cluster_size=50 by default) — noise frames labeled `-1`
4. HMM Viterbi smoothing on each contiguous non-noise segment; `-1` frames preserved

Shared models saved in `results/shared/`: `preprocessor.pkl`, `umap_reducer.pkl`, `clusterer.pkl`, `cluster_info.json`. Per-video labels in `results/shared/<stem>_labels.npy` (int32, -1 = noise).

### Important API notes

- `BehaviorPreprocessor` has `fit()`, `transform()`, and `fit_transform()` — use `transform()` on new data without refitting. In `compare.py` it is used with `use_pca=False` (standardization only); UMAP handles reduction separately.
- `BehaviorClusterer` has `fit()` and `predict()` — but **no `save()`/`load()`**. Persist via `joblib.dump(clusterer.model, path)`. Used only in `main.py` (per-video); cross-video now uses HDBSCAN directly.
- `BehaviorClusterer.visualize_clusters()` only supports `method="pca"` or `method="tsne"` (not `"umap"`).
- `AnomalyDetector.trained` must be `True` before calling `compute_reconstruction_error()`. The flag is set before `_compute_threshold()` is called.
- `BehaviorAnalyzer.generate_report()` writes UTF-8 — open with `encoding='utf-8'` or the `→` character will crash on Windows cp1252.
- `analysis.py`'s `export_results()` must convert numpy types to Python before `json.dump()`.
- `PoseFeatureExtractor` now accepts `use_wavelets=True` (default). Feature vector is 91D with wavelets, 51D without. All downstream code that loads `_features.npy` must use the same setting used during `--extract`.
- `_labels.npy` files contain `int32` with `-1` meaning noise (HDBSCAN). All downstream code must skip `-1` frames (e.g., `labels[labels >= 0]`). State fractions in `summary_table.csv` do NOT sum to 1 when noise is present — that is correct.
- `cluster_info.json` now includes `"method": "umap+hdbscan"` and `"min_cluster_size"`. `cluster_centers` are in standardized (51D or 91D) feature space for `characterize.py` compatibility.

### DLC project structure

The DLC project lives in `VIEB-Carlos-2026-02-11/`. Config at `VIEB-Carlos-2026-02-11/config.yaml`. Labeled frames in `VIEB-Carlos-2026-02-11/labeled-data/<video_name>/`. Labeling queue (random order of all 222 videos) persisted in `VIEB-Carlos-2026-02-11/labeling_queue.txt`.

### Metadata

`metadata.csv` has one row per video: `filename, date, box, experiment, day, context, no_shock, animal_id, fear`. The `fear` column is blank — the user will fill it in manually. Video stems match by stripping `.mp4` from `filename`.

## Output structure

```
results/
  <video_stem>/          # Per-video outputs from main.py
    clusters_pca.png
    ethogram.png
    behavior_summary.png
    preprocessor.pkl
    anomaly_detector.pt
    analysis_report.txt    (UTF-8)
    behavioral_states.csv
    transition_matrix.csv
  features/              # From compare.py --extract
    <stem>_features.npy  # float32 (T, 51) per video
    index.json
  shared/                # From compare.py --cluster
    preprocessor.pkl          # joblib BehaviorPreprocessor (standardizer, no PCA)
    umap_reducer.pkl          # joblib UMAP reducer (10 components)
    clusterer.pkl             # joblib HDBSCAN model
    cluster_info.json         # n_clusters, cluster_centers, method, min_cluster_size
    <stem>_labels.npy         # int32 (T,) per video; -1 = noise frame
  comparison/            # From compare.py --report and --summarize
    summary_table.csv         # 222 rows: state fracs + metadata (fracs may not sum to 1)
    transition_table.csv      # summary_table + flattened per-video transition matrices
    transition_by_context.png # side-by-side mean transition heatmaps per context
    state_by_day.png
    state_by_context.png
    state_by_experiment.png
    state_by_fear.png         # only once fear column is filled in
    state_by_animal.png
    animal_trajectories.png   # per-animal state occupancy across days
    animal_scalars.csv        # freeze AUC + discrimination ratio per animal
  characterization/      # From characterize.py
    state_summary.csv         # kinematic profiles + heuristic labels per state
    context_report.csv        # A/B/C enrichment, effect sizes, bootstrap CIs
    hidden_behaviors.csv      # rare states enriched in a context + anomaly bouts
    bouts.csv                 # all bouts with metadata (smoothed labels, 0.5s window)
    labels_per_frame.csv      # per-frame state + context for every video
    context_fractions.png     # bar plot of state occupancy by context
    cluster_tsne.png          # t-SNE scatter of sampled frames colored by cluster
clips/                   # From characterize.py --clips
  state_<id>/
    longest_NN.mp4            # longest bouts per state
    typical_NN.mp4            # bouts closest to cluster centroid
    context_<X>_NN.mp4        # bouts from most-enriched context
videos/                  # Manually curated or exported video files
```
