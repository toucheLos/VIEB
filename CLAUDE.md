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
python compare.py --cluster --validate   # Train/test split validation (80/20, seed=42)
python compare.py --report               # Comparison plots + CSVs → results/comparison/
python compare.py --report --min-confidence 0.7  # Exclude low-confidence frames
python compare.py --summarize            # Per-animal AUC + discrimination ratio
python compare.py --quantify             # Full master_table.csv + contrast vectors + learning rates
python compare.py --quantify --min-confidence 0.7  # Quantify with confidence filtering
python feature_reduction_test.py         # Feature subset experiment → results/feature_reduction/
python quantify.py --build               # Build master_table.csv only
python quantify.py --contrast            # Per-animal contrast vectors → results/quantification/contrast_vectors.csv
python quantify.py --contrast --cohort cohort_normalized.csv  # + cohort-level contrast stats
python generate_clips.py                 # Export exemplar video clips → clips/state_<id>/
python generate_clips.py --n-clips 10   # Limit clips per category per state (default 15)
python generate_clips.py --clip-purity 0.95  # Min label purity for clip expansion (default 0.95)
python generate_clips.py --output clips/ # Override output directory
```

### Cohort-level analysis (run after compare.py)
```bash
python cohort_analysis.py --cohort cohort_normalized.csv --output results/cohort/
python cohort_analysis.py --cohort cohort.xlsx --groupby genotype_treatment
python cohort_analysis.py --cohort cohort_normalized.csv --dry-run  # preview without writing
```
`--groupby` options: `age | treatment | sex | genotype | age_treatment (default) | genotype_treatment | age_sex | full`

## Architecture

### Two-tier pipeline

**Tier 1 — Pose tracking** (`setup_dlc_training.py` + `tracking/`):
DeepLabCut trains a ResNet50 model on manually labeled frames (8 keypoints: `left_ear, right_ear, nose, center, left_hip, right_hip, tail_base, tail_tip`). Outputs one CSV per video alongside each `.mp4` in `raw_videos/`.

**Tier 2 — Behavioral ML** (`ml/` package):
Stateless pipeline that transforms `(T, K, 2)` pose arrays into behavioral labels and reports. All five modules are imported via `ml/__init__.py`.

### `ml/` module responsibilities

| Module | Class | Role |
|--------|-------|------|
| `feature_extraction.py` | `PoseFeatureExtractor` | Two-layer feature extraction: Layer 1 (universal) + Layer 2 (semantic, conditional on keypoint roles). Feature count depends on available roles; 51/91 for standard 8-keypoint mouse model. |
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

Shared models saved in `results/shared/`: `preprocessor.pkl`, `umap_reducer.pkl`, `clusterer.pkl`, `cluster_info.json`. Per-video labels in `results/shared/<stem>_labels.npy` (int32, -1 = noise). Per-video soft probabilities in `results/shared/<stem>_probs.npy` (float32, 0–1; 0 for noise frames).

### characterize.py — Clip Reviewer data layer

`characterize.py` is **not a pipeline script** — it is a pure Python module providing the backend API for the Clip Reviewer GUI. It has no CLI entry point. Do not try to run it directly.

Public API:

| Function | Signature | Returns |
|----------|-----------|---------|
| `load_clips` | `(clips_dir)` | `{state_id: [clip_path, ...]}` |
| `load_annotations` | `(annotations_path)` | `{clip_path: label_string}` |
| `save_annotations` | `(annotations, annotations_path)` | writes/updates annotations.csv (never overwrites) |
| `get_clip_distribution` | `(annotations, all_clips, predictions=None)` | distribution dict (see below) |
| `shuffle_clips` | `(all_clips, seed=None)` | shuffled flat list |
| `train_classifier` | `(annotations_path, features_index, shared_dir, output_path)` | training report dict |
| `predict_clips` | `(classifier_path, shared_dir, all_clips, annotations_path, output_path)` | predictions DataFrame |

`get_clip_distribution` returns:
```python
{
  "total": int,
  "annotated": int,
  "unannotated": int,
  "by_label": {"Success": 12, "Failure": 8},
  "by_label_pct": {"Success": 0.20, "Failure": 0.16, "unannotated": 0.64},
  "by_label_predicted": {...}  # only when predictions= is passed
}
```

`train_classifier` uses cluster_centers from `cluster_info.json` (transformed through `umap_reducer.pkl`) as the per-state feature vector. Returns `{"trained": False, "reason": "..."}` when fewer than 2 categories or fewer than 5 clips per category.

`predict_clips` never modifies `annotations.csv` — predictions go to `predictions.csv` only.

### generate_clips.py — Clip generation pipeline stage

`generate_clips.py` is a standalone CLI script (pipeline stage 11). It is the extracted clip-generation logic, copied verbatim from the old `characterize.py --clips` functionality.

For each state 0..N-1:
- Longest bouts of consecutive frames labeled as that state → `longest_NN.mp4`
- Bouts nearest to the cluster centroid → `typical_NN.mp4`
- Bouts from the most context-enriched context → `context_{X}_NN.mp4`

Requires `results/characterization/bouts.csv` (or builds it on the fly) and `results/characterization/context_report.csv` (optional, for context-specific clips).

### Important API notes

- `BehaviorPreprocessor` has `fit()`, `transform()`, and `fit_transform()` — use `transform()` on new data without refitting. In `compare.py` it is used with `use_pca=False` (standardization only); UMAP handles reduction separately.
- `BehaviorClusterer` has `fit()` and `predict()` — but **no `save()`/`load()`**. Persist via `joblib.dump(clusterer.model, path)`. Used only in `main.py` (per-video); cross-video now uses HDBSCAN directly.
- `BehaviorClusterer.visualize_clusters()` only supports `method="pca"` or `method="tsne"` (not `"umap"`).
- `AnomalyDetector.trained` must be `True` before calling `compute_reconstruction_error()`. The flag is set before `_compute_threshold()` is called.
- `BehaviorAnalyzer.generate_report()` writes UTF-8 — open with `encoding='utf-8'` or the `→` character will crash on Windows cp1252.
- `analysis.py`'s `export_results()` must convert numpy types to Python before `json.dump()`.
- `PoseFeatureExtractor` uses a two-layer architecture: Layer 1 (universal, always computed: speeds, distances, PCA elongation, centroid speed, angular velocity, temporal stats, wavelets) and Layer 2 (semantic, conditional: rearing_score, head_angle — only computed when required keypoint roles are resolved). Feature vector length depends on available roles and wavelet setting; for the standard 8-keypoint mouse model: 51D without wavelets, 91D with. When keypoints are missing, semantic features are omitted entirely (not filled with zeros) and the vector is shorter. `index.json` `_meta.feature_names` stores the authoritative feature name list; use `resolve_feature_indices(names)` to look up column positions by name instead of hardcoding indices. `_meta.semantic_features` lists which Layer 2 features were included. `_meta.keypoint_groups` records which anatomical groups were resolved (e.g. `"head"`, `"tail"`, `"forepaws"`).
- `PoseFeatureExtractor` supports config-driven anatomical keypoint groups via `config.json "keypoint_roles"`. New format: `{"head": ["nose", "left_ear"], "tail": ["tail_base"]}` (group → list of keypoint names). Old format (`{"nose": "nose"}`, name → role) is auto-detected and converted. When `keypoint_roles` is omitted, groups are auto-resolved from `_KNOWN_GROUPS` defaults by matching against DLC bodypart names. `_SEMANTIC_FEATURE_GROUPS` maps each Layer 2 feature to required groups. `get_feature_availability_report()` returns which groups resolved and which features were skipped.
- `compare.py --motifs` discovers context-enriched bigram/trigram motifs. Primary output: `results/comparison/motifs.csv` (columns: `type`, `motif`, `enrichment_ratio`, `flagged` — consumed by `behavioral_fingerprint.py`). Motif column uses Python tuple strings like `"(2, 4)"` parsed via `ast.literal_eval`. Supplementary bout-based outputs in `results/motifs/`: `bouts.csv` (with prev/next state), `motif_sequences.csv`, `motif_summary.csv`, `motif_context_enrichment.csv`.
- `_labels.npy` files contain `int32` with `-1` meaning noise (HDBSCAN). All downstream code must skip `-1` frames (e.g., `labels[labels >= 0]`). State fractions in `summary_table.csv` do NOT sum to 1 when noise is present — that is correct.
- `cluster_info.json` now includes `"method": "umap+hdbscan"`, `"min_cluster_size"`, `"mean_confidence"`, and `"low_confidence_frac"`. `cluster_centers` are in standardized (51D or 91D) feature space.
- `_probs.npy` files (float32) contain HDBSCAN soft assignment probabilities (0–1) parallel to `_labels.npy`. Noise frames have prob=0. Use `valid = (labels >= 0) & (probs >= threshold)` for confidence-filtered fractions.
- `--cluster --validate` does an 80/20 video-level train/test split (seed=42), fits on train only, predicts test via `approximate_predict`, and saves `results/shared/validation_report.json` with the generalization score.
- `compare.py --report --min-confidence N` and `--quantify --min-confidence N` exclude frames with soft probability < N from all state fraction calculations.
- `quantify.compute_state_learning_rates()` computes linear regression slopes of state occupancy vs day (Context A only, ≥3 days required). Called automatically by `compare.py --quantify`; saves `results/quantification/learning_rates.csv` and adds `fear_learning_rate`/`fear_learning_r2` to `master_table.csv`.
- `feature_reduction_test.py` tests 5 feature subsets (4→91 features) on a 50k-frame sample. Saves UMAP scatter plots to `results/feature_reduction/` and a comparison CSV. Completes in under 5 minutes.
- `KinematicsPanel` widget (`_widgets.py`) shows centroid_speed/angular_velocity/rearing_score time series alongside video clips in Browse States and Validation views. A QTimer fires at 33ms to update the cursor position without blocking the main thread.
- `cohort_analysis.py` computes per-animal means (not per-session) as the unit of analysis before any cohort-level statistics. All cohort grouping is driven by `cohort_loader.load_cohort_excel()`. Dominant state is excluded from all state-level comparisons. FDR correction uses BH method via statsmodels if available, manual fallback otherwise.
- `quantify.compute_contrast_vector()` excludes the dominant state dynamically, detects Context A/B case-insensitively, stores vectors as JSON strings in CSV. Parse with `json.loads(row["contrast_vector_json"])`. `contrast_magnitude` is NaN (not 0) when an animal has no sessions in a context. `contrast_magnitude` is automatically added to `master_table.csv` by `compare.py --quantify` and is therefore included in Jess correlations without any special casing.
- `plot_cohort.py --contrast` requires `contrast_vectors.csv` and `cohort_contrast_vectors.csv` to already exist (run `quantify.py --contrast` first). Pass `--jess FILE` for the scatter plot.
- `characterize.save_annotations()` never overwrites — it appends or updates rows by `clip_path` key. Always safe to call multiple times.
- Annotations (human labels) live in `results/annotations/annotations.csv`. Predictions (model labels) live in `results/annotations/predictions.csv`. These are **never mixed** — `annotations.csv` contains only human labels.
- `characterize.train_classifier()` requires `results/shared/cluster_info.json` and optionally `results/shared/umap_reducer.pkl`. If the UMAP reducer is missing it falls back to standardized cluster centers.
- `classifier.pkl` in `results/annotations/` stores `{"clf": RandomForestClassifier, "state_features": dict, "classes": list}`. Check `saved["classes"]` to detect category drift between sessions.

### GUI architecture (`user_interface.py`)

The main GUI is `user_interface.py` (standalone, run directly). The sidebar order is:

1. Overview — dashboard with stat cards
2. Pipeline — staged pipeline runner
3. Browse States — state explorer with clips
4. **Analysis** — consolidated analysis hub (`views/analysis.py`)
5. **Validation** — frame labeling + Clip Reviewer (`views/validation.py`)
6. Settings — config editor

**Stage 0 — Onboarding** (`Stage0ReadinessPanel` in `user_interface.py`): A compact pipeline card that performs lightweight readiness checks only — active project detection, config existence, metadata existence, session source detection, results-dir creation. It routes the user to existing workflows (project selector, data import) rather than duplicating them. Does not scan large directories, does not expose settings parameters. If metadata is missing but a source is detected, generation runs in a `QThread`. Clicking "Check Project Readiness" is the only action needed; later stages block until this passes.

**Artifacts view** (`views/artifacts.py`): Scans use `ArtifactScanWorker` (a `QThread`) so the UI never blocks during directory scanning. The `refresh()` method accepts an optional pre-loaded data dict; the worker calls `_on_scan_done` / `_on_scan_failed` on completion. Row insertion is batched via `_insert_next_rows`.

`views/analysis.py` contains `AnalysisView` with a vertical tab bar (ten tabs) split into two labeled sections:

**CORE ANALYSIS**

| Tab | Command | Data source |
|-----|---------|-------------|
| State Characterization | *(no longer from characterize.py)* | `results/characterization/state_summary.csv` + clips/ |
| State Comparison | `compare.py --report` | `results/comparison/summary_table.csv` |
| Transitions & Motifs | `compare.py --motifs` | `results/comparison/motifs.csv` |
| Diagnostics | *(inline)* | `results/shared/cluster_info.json` |

**OPTIONAL ANALYSIS**

| Tab | Command | Data source |
|-----|---------|-------------|
| Cohort Analysis | `cohort_analysis.py --groupby X` | `results/cohort/` |
| Quantification | `compare.py --quantify` | `results/quantification/master_table.csv` |
| [metric label] | `fear_index.py --cohort X` | `results/quantification/fear_index.csv` |
| Jess Correlation | `compare.py --jess` | `results/quantification/jess_correlations.csv` |
| Event Alignment | `compare.py --event-align` | `results/comparison/` |
| Column Mapping | *(inline)* | project `config.json` |

The tab list uses section headers (non-selectable rows) and a `_row_to_stack` dict mapping list row → stack index, since header rows do not correspond to stack pages.

`views/validation.py` — `ValidationView` has three tabs:

| Tab | Purpose |
|-----|---------|
| **Clip Reviewer** | Session-based clip annotation with user-defined categories, distribution tracking, and supervised classifier (first tab, default) |
| Video Watching | Random clip sampling + freeze/walk/groom/rear/other labeling per state |
| Frame Sampling (Advanced) | Frame-level labeling for paper figures (requires characterization outputs) |

**Clip Reviewer session flow:**
1. Define categories (tag chips, Enter/comma to add, × to remove; up to 10)
2. Set shuffle seed (0 = random each session)
3. Click "Start Session" — requires ≥2 categories and clips to exist
4. Review clips one by one: colored category buttons (keyboard 1–9), Skip, Back, End Session
5. Distribution bars update after every annotation
6. "Train Classifier" appears when ≥2 categories have annotations; enabled when each has ≥5
7. After training, optionally apply predictions to unannotated clips (shown in lighter bar colors)
8. Annotations persist across sessions; previous session progress is resumed automatically

Config keys for Clip Reviewer (in `config.json`):
- `reviewer_categories`: list of category name strings (restored on app load)
- `reviewer_seed`: int (0 = random)

`AnalysisView` emits `worker_running(bool)` connected to `MainWindow._set_running()` so the status-bar pulse indicator reflects running analysis commands.

### Pipeline stages (in `_utils.py` and `user_interface.py` STAGES list)

| ID | Name | Script |
|----|------|--------|
| 1 | Pose Estimation (DLC) | `setup_dlc_training.py --analyze` |
| 2 | Feature Extraction | `compare.py --extract` |
| 3–6 | Preprocessing → HMM Smoothing | `compare.py --cluster` |
| 7 | State Collapsing (optional) | `compare.py --collapse` |
| 8 | Report Generation | `compare.py --report` |
| 9 | Per-Animal Scalars | `compare.py --summarize` |
| 10 | Motif Discovery | `compare.py --motifs` |
| 11 | **Generate Clips** | `generate_clips.py` |
| 12 | Event Alignment (optional) | `compare.py --event-align` |

Stage 11 completion is detected by checking whether the `clips/` directory exists and is non-empty.

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
    cluster_info.json         # n_clusters, cluster_centers, method, min_cluster_size, mean_confidence, low_confidence_frac
    <stem>_labels.npy         # int32 (T,) per video; -1 = noise frame
    <stem>_probs.npy          # float32 (T,) per video; HDBSCAN soft probabilities 0-1
    validation_report.json    # from --cluster --validate: generalization score, train/test split
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
    motifs.csv                # bigram/trigram enrichment between contexts (consumed by behavioral_fingerprint.py)
    motif_heatmap.png         # top enriched motifs heatmap
  motifs/                  # From compare.py --motifs (bout-based supplementary)
    bouts.csv                 # enriched bouts with prev_state/next_state columns
    motif_sequences.csv       # every n-gram occurrence per video
    motif_summary.csv         # global motif frequency table
    motif_context_enrichment.csv  # enrichment ratio per motif per context
  characterization/      # Legacy outputs (no longer regenerated by characterize.py)
    bouts.csv                 # all bouts — still used by generate_clips.py as input
    context_report.csv        # still used by generate_clips.py for context-specific clips
clips/                   # From generate_clips.py (pipeline stage 11)
  state_<id>/
    longest_NN.mp4            # longest bouts per state
    typical_NN.mp4            # bouts closest to cluster centroid
    context_<X>_NN.mp4        # bouts from most-enriched context
annotations/             # From Clip Reviewer (views/validation.py)
  annotations.csv             # human labels: clip_path, state_id, assigned_label, timestamp
  predictions.csv             # model predictions: clip_path, state_id, predicted_label, confidence
  classifier.pkl              # trained RandomForest: {"clf", "state_features", "classes"}
  training_report.json        # accuracy, confusion_matrix, feature_importances, classes
videos/                  # Manually curated or exported video files
cohort/                  # From cohort_analysis.py
  cohort_state_profiles.csv     # per-cohort mean ± SE per non-dominant state
  cohort_behavioral_metrics.csv # fear_AUC, disc_ratio, etc. per cohort
  cohort_statistics.csv         # pairwise Mann-Whitney U + BH FDR for all states
  cohort_significant_states.csv # filtered to FDR p < 0.05, sorted by fold-change
  cohort_state_profiles.png     # bar charts per cohort (one subplot per cohort)
  cohort_comparison.png         # top-20-state grouped bar chart across cohorts
  cohort_metrics.png            # behavioral scalar means per cohort
feature_reduction/       # From feature_reduction_test.py
  subset_N_umap.png             # UMAP scatter plots per feature subset (5 files)
  subset_comparison.csv         # comparison: n_features, n_clusters, sil, noise%, locomotion cluster
quantification/          # From quantify.py / compare.py --quantify
  master_table.csv              # one row per animal: all behavioral scalars + contrast_magnitude + fear_learning_rate
  learning_rates.csv            # per-animal linear regression slopes of state occupancy vs day
  contrast_vectors.csv          # per-animal contrast vector, magnitude, dominant states (JSON columns)
  cohort_contrast_vectors.csv   # mean contrast vector + 95% CI per cohort
  cohort_contrast_stats.csv     # pairwise Mann-Whitney U + BH FDR on contrast_magnitude
  contrast_bars.png             # diverging bar chart per cohort (plot_cohort.py --contrast)
  contrast_heatmap.png          # per-animal contrast vector heatmap
  contrast_magnitude.png        # cohort bar chart + individual dots
  contrast_scatter.png          # contrast_magnitude vs Jess protein (if jess data available)
```
