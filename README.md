# VIEB — Video Interpreter for Experimental Behavior

> **Discover behavioral states you didn't know to look for.**

Standard behavioral analysis tools require researchers to define behaviors before they can measure them. You draw a zone, set a threshold, or label example clips — and the software counts what you told it to count. This works for behaviors you already understand. It fails for the ones you don't.

VIEB takes a different approach. Given any rodent behavioral video with pose-tracking output, VIEB uses unsupervised machine learning to discover the full repertoire of behavioral states present in your data — without any prior labels, thresholds, or human assumptions. The states it finds are grounded entirely in the kinematics of the animal across every frame of every video.

This matters because fear, memory, and neurological disease do not express themselves through a single behavior. They reorganize the entire behavioral landscape. VIEB captures that reorganization.

---

## The Science

### Why unsupervised discovery

Classical ethology defines behaviors top-down: the researcher observes, categorizes, and then quantifies. This works at the level of coarse categories (freezing, locomotion, grooming) but misses the fine-grained structure within and between those categories. An animal that freezes with its head low is not behaving identically to one that freezes upright. An animal that transitions rapidly between micro-movements may be expressing a fundamentally different internal state than one that sustains them.

VIEB discovers these distinctions without being told they exist.

### How it works

Each video frame is represented as a 51-dimensional feature vector derived from DeepLabCut pose-tracking output (8 keypoints):

- **Kinematic features** — centroid speed, angular velocity, acceleration
- **Postural features** — body elongation, rearing score, head angle, body orientation
- **Pairwise distances** — all keypoint-to-keypoint distances normalized to body length
- **Temporal window statistics** — mean and variance of kinematics over 500ms windows

These features are pooled across all videos and animals, then reduced to a low-dimensional manifold using **UMAP** and clustered using **HDBSCAN**. The result is a set of shared behavioral states that are directly comparable across animals, sessions, and experimental conditions — without any per-video fitting.

States are then characterized by their kinematic profiles, their enrichment in specific experimental contexts, and their temporal dynamics. The key quantitative output is the **contrast vector**: the difference in state occupancy between the fear context (Context A) and the safe context (Context B) for each animal. The L2 norm of this vector — the **contrast magnitude** — captures how strongly an animal reorganizes its behavior between contexts. Animals with stronger fear memory show higher contrast magnitude.

This framework is grounded in Gründemann et al. (2019, *Nature*), who showed that population vector distance between contexts predicts fear memory strength at the neural level. VIEB applies the same logic at the behavioral level.

### What VIEB finds that standard tools miss

- **Sub-freezing states** — distinct postural configurations within what EthoVision would label as a single "freeze" bout
- **Locomotion variants** — slow exploratory movement vs. escape locomotion vs. rearing-associated movement
- **Transition structure** — which states tend to follow which, and how that structure changes with fear learning
- **Context-specific behavioral reorganization** — which states are selectively enriched in the fear context vs. the safe context, independent of any assumption about which behaviors are "fear behaviors"

---

## Installation

**Requirements:** Python 3.10–3.12, pip, a GPU is strongly recommended for clustering (NVIDIA with CUDA 12.x)

```bash
git clone https://github.com/toucheLos/VIEB.git
cd VIEB
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -e .                # Core pipeline (no DLC)
pip install -e ".[tracking]"    # Full install including DeepLabCut
```

**GPU acceleration (Linux):** Use the app's **Set up GPU acceleration** button or run
`python vieb_setup.py`. The setup flow checks your NVIDIA driver and installs a pinned
RAPIDS/cuML stack compatible with that driver.
```bash
python vieb_setup.py
```

**Launch the GUI:**
```bash
python user_interface.py
```

---

## Workflow

VIEB is organized as a staged pipeline. Each stage builds on the previous one. The GUI guides you through all stages with a live terminal output and status indicators.

### Stage 0 — Onboarding

Before the pipeline can run, VIEB verifies that an active project exists and has enough information to proceed. Click **Check Project Readiness** in the Stage 0 card. If anything is missing, the card points you to the relevant existing workflow (project selector, data import). All checks are lightweight — no directory scanning, no heavy library loading.

### Stage 1 — Pose Estimation (DeepLabCut)

VIEB takes DeepLabCut CSV output as input. If you have already run DLC on your videos, point VIEB to your `config.yaml` via the DLC Setup view and your raw videos directory via Settings. VIEB will find the pose CSVs automatically.

If you are starting from scratch, VIEB includes a full DLC training pipeline: add videos, extract frames, label keypoints, train, evaluate, and analyze — all from within the GUI.

**Required keypoints (default):** `nose, left_ear, right_ear, center, left_hip, right_hip, tail_base, tail_tip`

Custom keypoint configurations are supported via the keypoint mapping panel in DLC Setup.

### Stage 2 — Feature Extraction

```bash
python compare.py --extract
python compare.py --extract --no-wavelets   # faster, 51 features instead of 91
```

Extracts a feature vector for every frame of every video. Results saved to `results/features/`.

### Stages 2–4 — UMAP + HDBSCAN Clustering

```bash
python compare.py --cluster --umap-dims 3 --min-cluster-size 3000 --hdbscan-min-samples 5
```

Fits a single shared embedding and clustering model across all videos. Key parameters:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `--umap-dims` | 10 | UMAP output dimensions. Lower (3–5) = faster, coarser. Higher (10–15) = more structure. |
| `--min-cluster-size` | 2000 | Minimum frames per behavioral state. Lower = more states, more noise sensitivity. |
| `--hdbscan-min-samples` | None | Controls cluster border conservatism. Defaults to `min_cluster_size`. |

Shared models saved to `results/shared/`. Per-video state labels saved as `<stem>_labels.npy`.

### Stage 5 — State Characterization

```bash
python characterize.py
python generate_clips.py         # export exemplar video clips per state
```

Generates kinematic profiles, context enrichment statistics, and behavioral bout structure for each discovered state. Outputs `results/characterization/state_summary.csv` and `context_report.csv`.

### Stage 6 — Quantification

```bash
python compare.py --quantify --cohort cohort_normalized.csv
```

Computes per-animal behavioral scalars:

- **Contrast vector** — difference in state occupancy between Context A and Context B
- **Contrast magnitude** — L2 norm of the contrast vector, normalized to [0, 1]
- **Fear AUC** — area under the fear-state occupancy curve across training days
- **Discrimination ratio** — (occ_A − occ_B) / (occ_A + occ_B)
- **Learning rate** — slope of contrast magnitude across days

### Stage 7 — Cohort Analysis

```bash
python cohort_analysis.py --cohort cohort_normalized.csv --groupby age_treatment
```

Compares behavioral profiles across experimental groups using Mann-Whitney U tests with Benjamini-Hochberg FDR correction.

---

## GUI Overview

The VIEB GUI provides a complete interface for running and inspecting the pipeline without touching the terminal.

### Overview
Dashboard showing dataset statistics, number of discovered behavioral states, and mean state occupancy broken down by cohort or individual animal.

### Pipeline
Staged pipeline runner. Each stage shows its command, status, and live terminal output. Run stages individually or in sequence.

### Browse States
Visual explorer for discovered behavioral states. Displays exemplar video clips, kinematic profiles, and context enrichment for each state. Paginated across all animals.

### Analysis
Six-tab analysis hub:
- **Comparison Report** — state occupancy by day, context, experiment, and animal
- **State Characterization** — kinematic profiles and heuristic labels per state
- **Cohort Analysis** — group-level comparisons with statistical outputs
- **Quantification** — per-animal behavioral scalars and master table
- **Fear Index** — leave-one-out normalized fear expression index
- **Jess Correlation** — correlation between behavioral scalars and protein expression data

### Validation
Manual frame labeling interface for ground-truth validation of discovered states.

### Settings
Configure all paths (raw videos, results, metadata, DLC project), arena bounds, FPS, and clustering parameters.

---

## Output Files

| File | Description |
|------|-------------|
| `results/features/<stem>_features.npy` | Per-video feature arrays (T × 51 or T × 91) |
| `results/shared/cluster_info.json` | Cluster metadata, method, confidence |
| `results/shared/<stem>_labels.npy` | Per-frame state labels (int32, −1 = noise) |
| `results/shared/<stem>_probs.npy` | HDBSCAN soft assignment probabilities |
| `results/comparison/summary_table.csv` | Per-video state fractions + metadata |
| `results/characterization/state_summary.csv` | Kinematic profiles per state |
| `results/characterization/context_report.csv` | Context enrichment + effect sizes |
| `results/quantification/master_table.csv` | Per-animal behavioral scalars |
| `results/quantification/contrast_vectors.csv` | Per-animal contrast vectors |
| `clips/state_<id>/` | Exemplar video clips per behavioral state |

---

## Metadata Format

VIEB requires a `metadata.csv` with one row per video:

| Column | Description |
|--------|-------------|
| `filename` | Video filename (e.g. `animal9001_day1_A.mp4`) |
| `animal_id` | Unique animal identifier |
| `day` | Training day number |
| `context` | Experimental context (`A`, `B`, or `C`) |
| `experiment` | Experiment type (`CFC` or `CFD`) |
| `box` | Recording chamber ID |
| `date` | Recording date |
| `fear` | Fear condition label (fill in manually after scoring) |

---

## Citation

If you use VIEB in your research, please cite:

> Eckert C. et al. (in preparation). VIEB: unsupervised discovery of behavioral states from pose-tracked rodent videos.

**Key references:**
- McInnes et al. (2018). UMAP: Uniform Manifold Approximation and Projection. *arXiv*.
- Campello et al. (2013). Density-based clustering based on hierarchical density estimates. *PAKDD*.
- Gründemann et al. (2019). Amygdala ensembles encode behavioral states. *Nature*, 562, 210–215.
- Mathis et al. (2018). DeepLabCut: markerless pose estimation of user-defined body parts. *Nature Neuroscience*.

---

## License

MIT License. See `LICENSE` for details.

---

*VIEB is developed at Temple University in collaboration with the Luna Lab. For questions, open an issue or contact the repository maintainer.*
