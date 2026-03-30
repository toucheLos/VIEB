# VIEB — Video Interpreter for Experimental Behavior

VIEB is an unsupervised machine learning pipeline for analyzing mouse fear-conditioning behavior from top-down video. It takes DeepLabCut pose-tracking output and discovers, labels, and compares behavioral states across hundreds of videos without requiring manual annotation of behavior.

## Overview

The pipeline has two tiers:

1. **Pose tracking** (DeepLabCut) — tracks 8 keypoints per frame across 222 videos → one CSV per video
2. **Behavioral ML** (`ml/` package) — extracts 51 kinematic/postural features, fits a shared clustering model across all videos, and generates comparison plots grouped by day, context, experiment, and animal

Key outputs: ethograms, t-SNE cluster plots, per-animal behavioral trajectories across days, and group-level statistics linking behavioral states to fear conditioning.

## Installation

Python 3.10–3.12 required (3.13+ has compatibility issues with DeepLabCut).

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e .                # Core ML pipeline (no DLC)
pip install -e ".[tracking]"    # Full install including DeepLabCut
```

## Data Setup

DLC pose-tracking CSVs are not included in the repository due to size (~430 MB). Download them from the GitHub Release and extract into `raw_videos/`:

```
raw_videos/
  <video_name>.mp4
  <video_name>DLC_Resnet50_VIEBFeb11shuffle2_snapshot_best-30.csv
  ...
```

If you need to re-run pose tracking from scratch, see [Pose Tracking](#pose-tracking-dlc) below.

## Running the Pipeline

### 1. Extract features and cluster (run in order)

```bash
python compare.py --extract    # Pose CSVs → feature matrices → results/features/
python compare.py --cluster    # Fit shared model, HMM-smooth labels → results/shared/
python compare.py --report     # Group comparison plots → results/comparison/
```

### 2. Characterize behavioral states

```bash
python characterize.py         # Feature profiles, t-SNE, context contrasts → results/characterize/
```

### 3. Per-video analysis (optional)

```bash
python main.py --all                     # Independent model per video
python main.py --video raw_videos/X.mp4  # Single video
```

## Output Structure

```
results/
  features/         # compare.py --extract: (T, 51) feature arrays per video
  shared/           # compare.py --cluster: shared preprocessor, clusterer, per-video labels
  comparison/       # compare.py --report: boxplots by day/context/fear/animal, trajectories
  characterize/     # characterize.py: feature heatmap, t-SNE, ethograms
  <video_stem>/     # main.py: per-video ethogram, cluster plot, report
```

Key plots in `results/comparison/`:
- `state_by_day.png` — behavioral state occupancy across training days
- `state_by_context.png` — context A vs B vs C
- `state_by_fear.png` — fear vs no-fear animals (requires `fear` column in `metadata.csv`)
- `animal_trajectories.png` — per-animal state occupancy across days

## Behavioral Features

51 features extracted per frame:
- Per-keypoint speeds (8)
- Pairwise keypoint distances (28)
- Centroid speed, body orientation, elongation, angular velocity, movement entropy
- **Rearing score** — ear span / nose-tail distance (high = body contracted = rearing)
- **Head angle** — signed angle of head relative to body axis (high = head turned = exploration/grooming)
- **Relative head speed** — nose speed relative to body (high = head active, body still = grooming)
- **Head angle variability** — std of head angle in 1s window (high = rhythmic head motion = grooming)
- Temporal window statistics (speed mean/std/max/p90, distance mean/std, angular velocity mean/max)

Clustering uses KMeans or GMM (k=4–15, auto-tuned by silhouette score), followed by HMM Viterbi smoothing to correct single-frame label jitter.

## Pose Tracking (DLC)

Only needed if re-training or re-running inference:

```bash
python setup_dlc_training.py               # Add videos, extract frames
python setup_dlc_training.py --label       # Open labeling GUI (sequential queue)
python setup_dlc_training.py --train       # Train ResNet50 model
python setup_dlc_training.py --evaluate    # Check mAP
python setup_dlc_training.py --analyze     # Run inference on all 222 videos → CSV
```

Keypoints: `left_ear, right_ear, nose, center, left_hip, right_hip, tail_base, tail_tip`

## Metadata

`metadata.csv` contains one row per video: `filename, date, box, experiment, day, context, no_shock, animal_id, fear`. The `fear` column is left blank — fill it in manually to enable fear-group comparisons.
