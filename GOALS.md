# VIEB — Vision and Success Criteria

## What VIEB is

VIEB is a domain-agnostic latent behavioral state
discovery system. In its current instantiation it
analyzes mouse fear conditioning videos. In its
general form it is a framework for discovering
recurring behavioral primitives in any agent whose
behavior can be tracked and represented as temporal
feature vectors.

The pipeline has six layers:
1. Perception — raw observations to trajectories
2. Representation — trajectories to feature vectors
3. Latent state discovery — feature vectors to states
4. Temporal dynamics — state sequences to behavioral grammar
5. Contextual interpretation — states to biological meaning
6. Cross-group comparison and molecular correlation

VIEB's scientific claim is that unsupervised behavioral
state discovery reveals structure in behavior that
manual scoring cannot detect, and that this structure
correlates with molecular substrates of memory and
disease.

---

## The finished product

VIEB is finished when it can do the following without
human intervention beyond pressing a button:

1. Take a folder of raw .mp4 videos and a metadata CSV
2. Run pose estimation using a pretrained model
3. Extract behavioral features
4. Discover behavioral states
5. Characterize those states with video clips and
   kinematic profiles
6. Produce a quantification table with one row per
   animal and 20+ behavioral scalar columns
7. Accept a Jess/IHC protein data file and produce
   a correlation report showing which behavioral
   metrics predict which proteins and vice versa
8. Output paper-ready figures and a summary report

A new lab member with no coding experience should
be able to run the full pipeline in one afternoon
using only the GUI and the README.

---

## The paper

The paper makes three claims:

Claim 1 — Methodological:
VIEB discovers behaviorally meaningful states from
pose data without supervision. States are validated
against manual labels (Cohen's kappa > 0.8) and
are stable across pipeline runs (R² > 0.9).

Claim 2 — Biological:
In Het3 Alzheimer's model mice, behavioral state
dynamics differ from controls in ways that are
treatment-sensitive. TMZ-treated animals show
[specific state] differences relative to vehicle
controls. Young animals show different behavioral
diversity profiles than aged animals.

Claim 3 — Translational:
Behavioral quantification metrics correlate with
synaptic protein expression (GluA1, GluA2, NMDA1,
GFAP) in ways that are biologically interpretable.
The direction of these correlations suggests
[specific molecular mechanism] underlies the
observed behavioral differences.

---

## Necessary ingredients for success

Read CLAUDE.md before starting. This prompt defines
the complete requirements for VIEB to be a finished,
publishable, and usable system. Use this as the
master checklist. For each ingredient, check whether
it currently exists, partially exists, or is missing.
Report the status of each item before writing any code.

---

### Ingredient 1 — Stable clustering

REQUIRED: 8-25 behavioral states where no single
state captures more than 40% of valid frames.

Current status: UNKNOWN — awaiting 3D UMAP mcs=500
run with collapse.

Success criteria:
- dominant_state_frac < 0.40
- n_clusters between 8 and 25 after collapse
- silhouette score > 0.5
- cluster stability R² > 0.85 across two independent
  runs with different random seeds

What needs to be built:
- Cluster stability test: run clustering twice with
  different random seeds, match clusters by centroid
  similarity, compute R² of cluster size correlations
  Add as: python diagnose_clusters.py --stability-test

---

### Ingredient 2 — Validated behavioral states

REQUIRED: every discovered state has a human-assigned
behavioral label with inter-rater reliability reported.

Current status: MISSING — no manual labeling done.

Success criteria:
- Minimum 200 frames labeled per state (or 50 per
  state if n_states > 15)
- Two independent raters label the same frames
- Cohen's kappa > 0.6 (substantial agreement)
- Confusion matrix showing cluster assignment vs
  manual label agreement > 70% per state

What needs to be built:
- Validation view in GUI must be functional
- Frame sampler must correctly load video frames
  with keypoint overlay
- Labels must export to results/validation/
  frame_labels.csv with rater_id column
- Kappa computation must be implemented

---

### Ingredient 3 — Correct behavioral quantification

REQUIRED: master_table.csv with non-zero, biologically
meaningful values for all 22 animals.

Current status: BROKEN — freeze_AUC = 0 for 20/22
animals, disc_ratio = NaN for 19/22 animals.

Success criteria:
- fear_AUC > 0 for at least 18/22 animals
- disc_ratio between -1 and 1 for all 22 animals
  with variance across animals (not all same value)
- behavioral_diversity values between 0 and 1
- No column with >30% NaN values
- Cohort means differ visibly between
  18-24Mo Vehicle vs 18-24Mo TMZ

What needs to be built:
- Fix cmd_summarize() freeze state detection
  (use fear-relevant states from context_report.csv
  not lowest-speed microcluster)
- Fix discrimination ratio to not require same-day
  pairing (compute per animal across all sessions)
- Add entropy measures directly to animal_scalars
- Rebuild master_table after fixes

---

### Ingredient 4 — Jess correlation infrastructure

REQUIRED: given a Jess protein CSV, produce a
correlation report and method comparison automatically.

Current status: PARTIAL — run_jess_correlation()
exists in quantify.py but the six-method comparison
framework and reverse regression are not built.

Success criteria:
- run_jess_correlation() produces correlation heatmap
  and ranked table
- compare_quantification_methods() ranks all six
  quantification families by predictive power
- jess_predicts_behavior() runs Ridge LOO-CV regression
- All three produce output with real Jess data
- Bonferroni and FDR thresholds both reported

What needs to be built:
- build_state_day_table()
- fit_learning_curves()
- compute_bout_statistics()
- compute_entropy_measures()
- extract_transition_scalars()
- compute_fear_index()
- build_master_table()
- compare_quantification_methods()
- jess_predicts_behavior()
- compare_methods_permutation()

---

### Ingredient 5 — Paper-ready figures

REQUIRED: every figure in the paper can be generated
by a single command and requires no manual editing.

Current status: PARTIAL — some plots exist in
results/comparison/ but are not paper-ready quality.

Success criteria:
- All figures 300 DPI, white background, no titles
  that duplicate axis labels
- Figure 1: pipeline schematic (can be made in
  Illustrator/PowerPoint — not code)
- Figure 2: state discovery — UMAP scatter colored
  by state, representative video stills per state,
  kinematic profile heatmap
- Figure 3: behavioral dynamics — state occupancy
  by context, transition matrix A vs B,
  fear-enriched states sorted bar chart
- Figure 4: cohort comparison — grouped bar chart
  by Age × Treatment, learning curves per cohort,
  discrimination ratio distributions
- Figure 5: molecular correlation — heatmap of
  behavioral metrics × Jess proteins, top
  correlations scatter plots with regression lines

What needs to be built:
- plot_paper_figures.py — single script that
  generates all figures in results/figures/
  at 300 DPI with consistent styling

---

### Ingredient 6 — Pretrained model and clean install

REQUIRED: a new user can clone the repo and run
the full pipeline in one afternoon.

Current status: PARTIAL — pretrained model package
infrastructure exists but model has not been packaged
and README is incomplete.

Success criteria:
- python setup_dlc_training.py --use-pretrained
  mouse_8kp_v1 works from a fresh clone
- python compare.py --extract runs without error
  on a new machine
- All dependencies install with pip install -e .
- README Quick Start works end-to-end

What needs to be built:
- Package pretrained DLC model weights
- Upload to GitHub Releases
- Update README with correct Quick Start
- Test on a clean Python environment

---

### Ingredient 7 — Robust error handling

REQUIRED: every pipeline step fails gracefully with
a clear message, not a Python traceback.

Current status: PARTIAL — some error handling exists,
many edge cases still crash.

Success criteria:
- Every missing file produces a clear message with
  the exact command to fix it
- Every NaN-producing computation is caught and
  reported with the animal_id and column name
- GUI never shows a Python traceback to the user
- All file paths work on Windows, macOS, and Linux

---

### Ingredient 8 — VIEB generalizes beyond mice

REQUIRED FOR LONG-TERM VISION ONLY — not needed
for the first paper but needed for VIEB to be a
platform rather than a one-time tool.

The architecture already supports this. What is
needed is documentation of the abstraction layer:

- A CONTRIBUTING.md explaining how to add a new
  species or agent type (new DLC model + new
  feature extractor = new VIEB application)
- One worked example beyond mice
  (recommended: rat, since paradigm is identical)

---

## Priority order for completion

Given the PI meeting is imminent and the paper
needs to be written, work in this order:

WEEK 1 — Foundation
  Day 1: Fix clustering (Ingredient 1)
  Day 2: Fix quantification (Ingredient 3)
  Day 3: Manual validation of states (Ingredient 2)

WEEK 2 — Science
  Day 4-5: Run full pipeline with fixed clustering,
            generate cohort comparison figures
  Day 6-7: Jess data integration (Ingredient 4)

WEEK 3 — Paper
  Day 8-9: Generate all paper figures (Ingredient 5)
  Day 10: Write Methods section (pipeline description)
  Day 11-12: Write Results section
  Day 13-14: Write Discussion and Introduction

WEEK 4 — Polish
  Pretrained model packaging (Ingredient 6)
  Error handling sweep (Ingredient 7)
  README update
  Submit

---

## Definition of done

VIEB is done when:

A researcher with no coding experience can:
1. Clone the repo
2. Follow the README
3. Run the GUI
4. Load their videos
5. Press "Run All"
6. Import their Jess data
7. See a correlation report

AND a neuroscience reviewer can read the paper and:
1. Understand the methodology without ambiguity
2. Reproduce the results from the code
3. Apply VIEB to their own data

Both conditions must be true simultaneously.

gui.py is cluttered and filled with unnecessary metrics, that is why user_interface exists. 
# Layout for gui.py (exclude unnecessary metrics):
in design_and_cohorts folder