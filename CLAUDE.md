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
python compare.py --extract    # Extract features from all 222 videos → results/features/
python compare.py --cluster    # Fit shared model, label all videos → results/shared/
python compare.py --report     # Comparison plots + summary_table.csv → results/comparison/
```

## Architecture

### Two-tier pipeline

**Tier 1 — Pose tracking** (`setup_dlc_training.py` + `tracking/`):
DeepLabCut trains a ResNet50 model on manually labeled frames (8 keypoints: `left_ear, right_ear, nose, center, left_hip, right_hip, tail_base, tail_tip`). Outputs one CSV per video alongside each `.mp4` in `raw_videos/`.

**Tier 2 — Behavioral ML** (`ml/` package):
Stateless pipeline that transforms `(T, K, 2)` pose arrays into behavioral labels and reports. All six modules are imported via `ml/__init__.py`.

### `ml/` module responsibilities

| Module | Class | Role |
|--------|-------|------|
| `feature_extraction.py` | `PoseFeatureExtractor` | Pose → kinematic/spatial/postural features `(T, 49)` |
| `preprocessing.py` | `BehaviorPreprocessor` | Standardize + PCA; has separate `fit()` / `transform()` |
| `clustering.py` | `BehaviorClusterer` | K-Means/GMM/DBSCAN; auto-tunes from k=4–12 using silhouette |
| `anomaly_detection.py` | `AnomalyDetector` | PyTorch autoencoder; flags unusual frames |
| `sequence_models.py` | `TemporalBehaviorModel` | Bidirectional LSTM for temporal dynamics |
| `analysis.py` | `BehaviorAnalyzer` | Statistics, plots, JSON/CSV/text export |

### Per-video vs. cross-video analysis

`main.py --all` fits an independent model per video — cluster IDs are **not** comparable across videos.

`compare.py` fits **one shared preprocessor + clusterer** on all 1.28M frames pooled together, then predicts labels for each video using that shared model. This is the correct path for cross-condition comparisons. The shared models are saved in `results/shared/` (preprocessor as `.pkl` via `BehaviorPreprocessor.save()`, clusterer via `joblib`).

### Important API notes

- `BehaviorPreprocessor` has `fit()`, `transform()`, and `fit_transform()` — use `transform()` on new data without refitting.
- `BehaviorClusterer` has `fit()` and `predict()` — but **no `save()`/`load()`**. Persist via `joblib.dump(clusterer.model, path)`.
- `BehaviorClusterer.visualize_clusters()` only supports `method="pca"` or `method="tsne"` (not `"umap"`).
- `AnomalyDetector.trained` must be `True` before calling `compute_reconstruction_error()`. The flag is set before `_compute_threshold()` is called.
- `BehaviorAnalyzer.generate_report()` writes UTF-8 — open with `encoding='utf-8'` or the `→` character will crash on Windows cp1252.
- `analysis.py`'s `export_results()` must convert numpy types to Python before `json.dump()`.

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
    <stem>_features.npy  # float32 (T, 49) per video
    index.json
  shared/                # From compare.py --cluster
    preprocessor.pkl
    clusterer.pkl        # joblib-serialized sklearn KMeans
    cluster_info.json
    <stem>_labels.npy    # int32 (T,) per video
  comparison/            # From compare.py --report
    summary_table.csv    # 222 rows: state fracs + metadata
    state_by_day.png
    state_by_context.png
    state_by_fear.png    # only once fear column is filled in
```
