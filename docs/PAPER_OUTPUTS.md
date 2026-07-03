# Paper Outputs Specification

This is the single source of truth for what the Luna and Spence papers need
out of VIEB — every figure, table, and clip category, with an exact status
against the current codebase. **Do not let scope grow past this list without
an explicit update to this file first.**

Status legend:
- **DONE** — exists, wired into the pipeline/GUI, and correct.
- **PARTIAL** — exists in some form but is broken, incomplete, unwired, not
  persisted, or fragmented across multiple non-canonical implementations.
- **MISSING** — does not exist anywhere in the codebase.

Every PARTIAL/MISSING entry below was verified directly against the code
(grep + read), not assumed. Where a `docs/DECISIONS.md` entry is relevant it
is cited by number.

---

## State Summary (per project)

Per-state fields needed for paper tables/figures.

| Field | Status | Evidence |
|---|---|---|
| **state id** | DONE | `state_summary.csv` (`state` col); `state_feature_zscores.csv` / `state_feature_profiles.csv` / `state_duration_summary.csv` / `state_group_enrichment.csv` (`state_id` col) — all from `state_characterizer.py` |
| **occupancy** | PARTIAL | Computed as a global mean fraction and rendered to `results/characterization/state_occupancy.png` (`compare.py:_save_state_occupancy_plot`, `compare.py:2747`), but never written back as a numeric column into `state_summary.csv` or `state_characterization.json`. No single table has occupancy sitting next to bout count/duration for a paper export. |
| **bout count** | DONE | `n_bouts` in `state_summary.csv` and `state_duration_summary.csv` (`state_characterizer.py:_compute_duration_summary`, lines 286–311) |
| **mean/median duration** | DONE | `mean_bout_dur_sec` / `median_bout_dur_sec` in `state_summary.csv`; full percentile set (p5/p25/p75/p95/min/max) in `state_duration_summary.csv` |
| **movement intensity** | DONE | Kinematic values from cluster centers via `_extract_kinematic_values` (`compare.py:259`), folded into `state_summary.csv`; raw/z-scored full feature profiles in `state_feature_profiles.csv` / `state_feature_zscores.csv` |
| **posture/shape signature** | PARTIAL | Layer-2 semantic features — elongation, rearing_score, head_angle (Decision #2) — appear in the generic per-feature z-score/profile CSVs when keypoint roles resolve, but there's no dedicated posture-signature summary or plot; it's buried among ~50–90 undifferentiated feature columns. |
| **transition profile** | PARTIAL | Raw ingredients exist (`prev_state` / `next_state` on bouts via `compare.py:3414-3423`; full matrices in `transition_table.csv` and `transition_by_context.png`) but nothing aggregates this into a per-state "top preceding/following states" summary field, unlike `_compute_top_features` for kinematics. |
| **enriched timepoints/conditions** | DONE | `state_group_enrichment.csv` via `_compute_group_enrichment` (`state_characterizer.py:314-386`) — context/day/animal_id/experiment fractional enrichment |
| **representative clips** | DONE | `generate_clips.py` (pipeline stage 8) → `clips/state_<id>/{longest,typical,context}_*.mp4`; State Characterization view browses the full folder per Decision #42 |
| **user label/notes** | MISSING | No per-state persisted label/notes exists anywhere. `characterize.py:291-304` derives a `state_label` dict internally purely to train the clip classifier — never saved to disk or exposed as an editable note. No `state_annotations.json` or equivalent exists. |

---

## Generalized plots (any project)

- **state_summary_plot** — PARTIAL. Fragmented across non-canonical cousins:
  `compare.py._save_state_occupancy_plot` → `state_occupancy.png`;
  `plot_cohort.plot_cohort_state_profiles` **and** an independently
  implemented `cohort_analysis.py` Task 6 both write `cohort_state_profiles.png`
  via two separate code paths (worth deduplicating); a real per-state
  kinematic-profile heatmap exists only in the unwired standalone script
  `make_plots.py`. No single canonical "state summary" figure exists.

- **umap_by_state** — DONE. `compare.py._save_umap_embedding_plot`
  (`compare.py:2238-2276`) → `results/diagnostics/umap_embedding_by_state.png`.
  Full functional match; filename differs slightly from the spec name.

- **umap_by_time_or_condition** — MISSING. `plot_cohort.plot_animal_umap`
  only colors by genotype/treatment/sex/age_group; nothing colors a UMAP
  embedding by day/context/time anywhere in the codebase.

- **state_transition_matrix** — PARTIAL. The wired pipeline path only
  produces *grouped* heatmaps (`compare.py._plot_transition_heatmaps` →
  `transition_by_context.png` / `transition_by_<group>.png`). A true single
  aggregate-matrix plotter exists (`ml/analysis.py
  BehaviorAnalyzer.plot_transition_matrix`, lines 297-335) but is dead code,
  never called from `main.py`.

- **state_duration_distribution** — PARTIAL. `state_duration_summary.csv`
  has percentile stats only, no plot in the wired pipeline. A real
  distribution plot (boxplot) exists only in the unwired, hardcoded
  `make_plots.py` (`5_bout_duration.png`).

---

## Luna-specific plots (condition_and_time mode)

- **state_occupancy_by_time_and_condition** — MISSING. Only hardcoded
  single-state (`state_9`) cousins exist in unwired `make_plots.py`
  (`6_state9_by_day_context.png`; `8_state_context_heatmap_full.png` has
  context but no day axis). No general (all-states × day × context) plot
  exists in the wired pipeline.

- **condition_contrast_over_time** — PARTIAL. `user_interface.py:_render_learning_curves`
  (lines 4796-4900, backing the "Learning Curves" GUI panel, configurable via
  `views/settings.py` — `baseline_group`, `comparison_group`, `order_column`)
  computes and renders exactly this live, but **never saves a PNG artifact**.
  This violates the Decision #30 expectation that key paper figures persist
  to Artifacts, not just render in-GUI.

- **context_enriched_states** — PARTIAL. A CSV exists at state level
  (`state_group_enrichment.csv`) with no plot in the wired pipeline; a plot
  cousin exists only in unwired `make_plots.py`. A *motif*-level (not
  state-level) enrichment bar chart renders live in the GUI
  (`views/analysis.py:1128-1160`) but also isn't persisted.

- **transition_by_condition** — DONE. Closest exact match of the entire
  list: `transition_by_context.png` via
  `compare.py._plot_transition_heatmaps` / `cmd_report`. The output is named
  "context" rather than "condition," but the concept is identical.

- **condition_state_trajectories** — PARTIAL. `compare.py._plot_animal_trajectories`
  (`compare.py:2697-2744`) → `animal_trajectories.png` colors lines by
  animal identity, not by condition. Unwired `make_plots.py` has a
  single-state, condition-colored cousin only.

---

## Spence-specific plots (time_only mode)

- **state_occupancy_over_time** — MISSING from the wired pipeline; exists
  only in unwired `make_plots.py` (`4_state_by_day.png`).

- **state_duration_over_time** — MISSING. No per-day/per-time duration
  trend is computed anywhere in the codebase; `state_characterizer.py` only
  computes overall (not per-day) duration percentiles.

- **transition_entropy_over_time** — MISSING as a time series, and flagged
  as actively **Luna-only, not just missing a time axis**:
  `quantify.compute_transition_entropy()` (`quantify.py:221-249`) only
  computes a static Context-A-vs-B scalar (`transition_entropy_A` /
  `transition_entropy_B` columns in `master_table.csv`), and — same as
  `fear_index.py` — this module family hard-requires a context column.
  Spence data (no context column) would need this generalized before a
  time-only entropy trend could even be built.

- **per_subject_state_trajectories** — DONE. `compare.py._plot_animal_trajectories`
  → `animal_trajectories.png` is exactly this concept (per-animal state
  occupancy across days), just filed under a Luna-flavored name.

- **change_from_baseline** — PARTIAL. Existing "baseline" concepts
  (`fear_index.cohort_normalize`, `plot_cohort.plot_deviation_distributions`)
  compute change from a **cohort-mean** baseline, not a **first-day/time**
  baseline. `_render_learning_curves` could produce a genuine time-baseline
  version if repointed at a time column instead of context, but isn't wired
  that way today, and isn't persisted regardless (same artifact-persistence
  gap as `condition_contrast_over_time` above).

- **story_distance_from_baseline** — MISSING entirely. Confirmed zero
  occurrences of "story"/"stories" anywhere in the codebase (case-insensitive
  grep across the whole repo). This is a new concept, not a rename of
  something existing.

---

## Tables

- **video_stories.csv** — MISSING. Would be a per-video narrative/sequence
  summary: the state sequence over time for one video, condensed into a
  "story" of what the animal did during the session. Needed because raters
  and reviewers need a compact, readable per-video summary rather than a
  raw frame-by-frame or bout-by-bout table. No such concept exists in the
  codebase today; the closest adjacent data is raw `bouts.csv`
  (frame-level bout list, no narrative structure) and `motif_sequences.csv`
  (every n-gram occurrence per video, not a condensed per-video summary).

- **video_story_bouts.csv** — MISSING. Would be the bout-level detail table
  backing `video_stories.csv` — which specific bouts compose each video's
  story, so the narrative summary can be traced back to source frames.
  Needed alongside `video_stories.csv` for reproducibility/auditability of
  any narrative claims made in the paper. Depends on `video_stories.csv`
  existing first; nothing beyond the existing generic `bouts.csv` has been
  built toward this yet.

- **subject_journeys.csv** — MISSING. Would be a per-animal, per-timepoint
  trajectory table — the tabular backing for `per_subject_state_trajectories`
  and `condition_state_trajectories`, letting the paper report exact
  occupancy values per animal per day rather than only reading them off a
  plot. Closest existing cousins are `animal_trajectories.png` (a plot, not
  an exportable table) and `master_table.csv` / `learning_rates.csv`
  (per-animal scalars/slopes, not a full time-series-per-subject table).

- **state_annotations.json** — MISSING. Would be the persisted form of
  "user label/notes" per state from the State Summary section above —
  letting a curator attach a human-readable label and free-text notes to
  each discovered state, durably, for use in paper figures/tables (e.g.
  state axis labels). Closest existing cousin is per-clip
  `annotations.csv` / `predictions.csv` / `classifier.pkl` (clip-level
  human/model labels, aggregated only in-memory via
  `characterize.py:291-304`) — none of this is state-level or persisted to
  disk.

---

## Cross-cutting gap: no formal project-mode system

No `condition_and_time` vs `time_only` mode detection exists in code today
— confirmed zero hits for those terms, `project_type`, or any equivalent
mode-branching field/enum anywhere in the repo. "Luna" and "Spence" appear
only as branding/history text (e.g. `views/help.py`, `projects.json`), never
as behavior-switching code.

Generic degradation (missing context/condition column → skip output, not
crash) already works correctly in `compare.py --report`,
`metadata_schema.py get_enabled_analysis_groups()`, and the UI
`PANEL_REGISTRY` (tested in `tests/test_report_optional_columns.py`).

However, `quantify.py compute_contrast_vector()` (`_ctx_col` used in 8
places, hard `sys.exit` if no context column found) and all of
`fear_index.py` (`cohort_normalize`, `compute_fear_index`,
`cohort_fear_profiles`) **hard-require** a context column. These two files
are the concrete Luna-only chokepoints that would break outright on
Spence-style time-only data. Any future work enabling the Spence-specific
MISSING/PARTIAL plots above will need to touch these two files first. This
is a finding for future scoping — it is not addressed by this document.
