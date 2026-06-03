# VIEB Pipeline Commands

Commands listed in execution order.

---

## 1. Pose Estimation (DeepLabCut — run once per project)

```bash
python setup_dlc_training.py               # Add videos, extract frames, open labeling GUI
python setup_dlc_training.py --label       # Open next unlabeled video (numbered queue)
python setup_dlc_training.py --label 39    # Jump to video #39/222 in the queue
python setup_dlc_training.py --train       # Create dataset and train (after labeling)
python setup_dlc_training.py --evaluate    # Evaluate trained model (check mAP)
python setup_dlc_training.py --analyze     # Run pose estimation on all 222 videos → CSV
```

---

## 2. Per-Video Behavioral Analysis

```bash
python main.py --all                       # Analyze all videos that have a DLC CSV
python main.py --video raw_videos/X.mp4   # Single video
python main.py --all --n-clusters 6       # Fix number of behavioral states
python main.py --all --no-anomaly         # Skip autoencoder (faster)
```

---

## 3. Cross-Video Comparison Pipeline

Run these stages **in order**.

### 3a. Extract features

```bash
python compare.py --extract              # Extract features → results/features/  (91 features with wavelets)
python compare.py --extract --no-wavelets  # Skip Morlet wavelets (faster, 51 features)
```

### 3b. Cluster (shared model)

```bash
python compare.py --cluster              # UMAP + HDBSCAN + HMM-smooth → results/shared/
python compare.py --cluster --min-cluster-size 30  # Tune HDBSCAN (default 50)
python compare.py --cluster --validate   # 80/20 train/test split validation (seed=42)
```

### 3c. Comparison report

```bash
python compare.py --report               # Comparison plots + CSVs → results/comparison/
python compare.py --report --min-confidence 0.7  # Exclude low-confidence frames
```

### 3d. Per-animal summary

```bash
python compare.py --summarize            # Per-animal AUC + discrimination ratio
```

### 3e. Quantification

```bash
python compare.py --quantify             # master_table.csv + contrast vectors + learning rates
python compare.py --quantify --min-confidence 0.7  # Quantify with confidence filtering
```

---

## 4. Behavioral Characterization

```bash
python characterize.py                   # Behavioral state profiles + t-SNE → results/characterization/
python characterize.py --clips           # Also export exemplar video clips → clips/state_<id>/
python characterize.py --n-clips 10      # Change clips per category (default 15)
```

---

## 5. Cohort-Level Analysis

```bash
python cohort_analysis.py --cohort cohort_normalized.csv --output results/cohort/
python cohort_analysis.py --cohort cohort.xlsx --groupby genotype_treatment
python cohort_analysis.py --cohort cohort_normalized.csv --dry-run  # Preview without writing
```

`--groupby` options: `age | treatment | sex | genotype | age_treatment (default) | genotype_treatment | age_sex | full`

---

## Optional / Standalone Tools

```bash
# Feature subset experiment
python feature_reduction_test.py         # Feature subset experiment → results/feature_reduction/

# Quantification (individual steps)
python quantify.py --build               # Build master_table.csv only
python quantify.py --contrast            # Per-animal contrast vectors → results/quantification/contrast_vectors.csv
python quantify.py --contrast --cohort cohort_normalized.csv  # + cohort-level contrast stats

# Contrast plots (requires contrast_vectors.csv from quantify.py --contrast)
python plot_cohort.py --contrast
python plot_cohort.py --contrast --jess FILE  # Include Jess protein scatter plot
```
