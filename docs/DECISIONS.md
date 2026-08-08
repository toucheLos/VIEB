# VIEB Decision Log

Numbered, newest at bottom. One entry per meaningful decision, root-cause
finding, or deferred idea. Keep entries short — link to a commit hash,
branch, or prompts/ file instead of re-explaining details that live
elsewhere.

Format:
```
## N — Short title
**Decision/finding:** one or two sentences.
**Why:** the reasoning, briefly.
**Related:** commit hash, branch, or prompts/ file.
```

---

## 1 — Ludovick access gated until preprint
**Decision:** VIEB will not be shared with Ludovick (Inscopix) before the
bioRxiv preprint is submitted.
**Why:** Preprint establishes timestamped public prior art, protecting
against independent reproduction of the idea. Commit history alone is
weaker protection.

## 2 — Two-layer feature extractor architecture
**Decision:** Feature extraction split into Layer 1 (universal — speeds,
distances, orientation, wavelets; works on any keypoint layout) and Layer 2
(semantic — rearing_score, head_angle; only computed when required named
keypoint roles exist). Missing roles skip gracefully instead of crashing.
**Why:** Original extractor hardcoded Luna's 8-keypoint mouse layout and
threw IndexError on Spence's 5-keypoint leg-only rat layout.
**Related:** `ml/feature_extraction.py`

## 3 — cuML HDBSCAN memory/parameter ceilings
**Decision:** Use CPU HDBSCAN with subsampling + batched
`approximate_predict()` whenever min_samples needs to go below 1023 or
frame count exceeds VRAM budget.
**Why:** cuML clamps min_samples to ≤1023 and OOMs on large frame counts
(7M+ frames needs ~30GB VRAM, exceeds 24GB on the 3090 Ti).
**Related:** `compare.py cmd_cluster()`, `_fit_cpu_hdbscan_with_assignment()`

## 4 — H5 concatenated-table support
**Decision:** Auto-detect two H5 patterns: multi-key (one key per session,
DLC default) vs single-key concatenated table with a source_file column
(Spence lab format), and route extraction accordingly.
**Why:** Spence's H5 has one key with 3,735 sessions concatenated; original
manifest logic assumed one key per session and skipped 3,734/3,735 rows.
**Related:** `h5_manifest.py`, `pose_io.py`, `compare.py _cmd_extract_h5()`

## 5 — Agent features deferred to experimental worktree
**Decision:** Diagnostic/tuning agent, onboarding/config agent, and
bias-audit agent are real but will not be built on dev. Work begins on a
separate `experimental` branch/worktree after the Luna paper is submitted.
**Why:** Paper submission is the current priority; agent work risks
destabilizing dev mid-deadline.

## 6 — Graph-based feature extraction: rule-based, not learned
**Decision:** Full learned graph-network feature extraction (ST-GCN style)
is out of scope. Rule-based joint-angle graph traversal (read skeleton
from DLC config.yaml, auto-derive joint angles/symmetry from edges) is the
interim plan, built after the paper ships.
**Why:** Learned embeddings are hard to defend in a methods paper claiming
to exclude ethogram bias; rule-based traversal stays interpretable while
still generalizing across labs' keypoint layouts.

## 7 — Analysis should be question-driven, not script-driven
**Decision:** Reorganize analysis around user questions — what states were
found, how they differ across groups/time, what sequences matter, whether
clustering is trustworthy — rather than a fragmented, historically-scripted
panel layout.
**Why:** Current panels (Quantification, Primary Metric, Jess Correlation,
Cohort Analysis) are fragmented and partly Luna-specific.

## 8 — Proposed Analysis structure: State Explorer, Group & Time Effects, Sequences, Quality Control
**Decision:** Replace the current Analysis sidebar with four core panels:
`State Explorer`, `Group & Time Effects`, `Sequences`, `Quality Control`.
Advanced/optional analyses nest inside these instead of competing as
top-level panels.
**Why:** Matches the real workflow (interpret states → compare
groups/timepoints → inspect sequences → validate quality) and removes
Luna-specific naming from the core product.
**Related:** `views/analysis.py`

## 9 — Column Mapping belongs in Onboarding/Settings, not Analysis
**Decision:** Move Column Mapping out of Analysis into setup/configuration.
Analysis consumes canonical metadata; it should not ask users to fix
metadata mapping mid-interpretation.
**Related:** `metadata_schema.py`, `metadata_generator.py`, `views/analysis.py`

## 10 — Remove Luna-specific panel names from core UI
**Decision:** Replace names like `Jess Correlation`, `Primary Metric`, and
fear/cohort-specific terminology with generic equivalents: `External
Variable Correlation`, `Custom Metric`, `Group Comparison`, `Condition`,
`Subject`, `Timepoint`.
**Why:** VIEB should work for any lab without encoding one experiment's
vocabulary into the main UI. Project-specific terms map through metadata
config instead.

## 11 — State Characterization is not renamed to Clips
**Decision:** State Characterization remains the main interpretive view.
Clips are evidence inside it, not the panel's identity.
**Why:** Renaming to "Clips" would make VIEB feel like a video browser
rather than a behavioral-state interpretation platform.
**Related:** `views/state_characterization.py`
**Supersedes:** earlier rename-to-"Clips" prompt from this session.

## 12 — Browse States rebuilt inside State Characterization / future State Explorer
**Decision:** The old unwired Browse States view is not kept as a separate
large view. Useful functionality (state browsing, summaries, exemplar
clips) gets rebuilt into one coherent state-interpretation workflow.
**Related:** `views/browse_states.py`

## 13 — Validation returns later as State Quality / Human Review
**Decision:** The old broad Validation view is not preserved as-is. The
concept returns focused: review exemplar clips, label good/bad examples,
flag bad clusters, validate state meanings, add notes — living inside
Quality Control / State Explorer.
**Related:** `views/validation.py`

## 14 — Stage 0 Onboarding exposes one primary action
**Decision:** Stage 0 shows one dominant action (e.g. `Continue
Onboarding` / `Resolve Setup`), with secondary actions (Create Project,
Open Project, Add Data Source, Check Readiness, Migrate Legacy Paths)
tucked under details/more options.
**Why:** The app can auto-detect most setup state; showing every option
at once forces users to understand internal implementation categories.
**Related:** `user_interface.py`, `project_manager.py`

## 15 — Onboarding auto-detects before asking questions
**Decision:** Stage 0 inspects the active project/config/filesystem first
and only asks the user when detection can't resolve the next step — it
does not open by asking whether a project exists or where data lives.
**Related:** `project_manager.py`, `metadata_generator.py`, `user_interface.py`

## 16 — Previous session summaries auto-load
**Decision:** If valid previous results exist for the active project, VIEB
loads the lightweight session summary automatically on startup/project
selection, instead of requiring a manual `Load Previous Session` click.
**Why:** Opening an existing project should populate Overview immediately;
manual loading is for refresh, not normal startup.
**Related:** `user_interface.py`, DataLoader

## 17 — Reload Data must not claim to run compare.py unless it does
**Decision:** `Reload Data` refreshes GUI data from disk only. It must not
display messages like "Running compare.py --report" unless a real
subprocess is actually launched.
**Why:** Misleading status messages make debugging impossible and erode
trust. Compute actions and lightweight UI refresh must be visibly distinct.
**Related:** `user_interface.py`

## 18 — Staged warm-loading instead of cold lazy-loading every tab
**Decision:** Show the main window quickly, then warm-load likely next
views in the background in order: Overview → Pipeline → Cluster Runs →
Analysis shell → Artifacts shell.
**Why:** Full eager loading blocks startup; full cold lazy-loading makes
every first tab-click feel slow. Staged warm-loading is the middle ground.
**Related:** `user_interface.py`, view manager/preloader

## 19 — Heavy data stays out of startup and warm-load paths
**Decision:** Startup/warm-loading may construct lightweight shells and
read small JSON/marker files, but must never read large CSVs, H5 files,
clips, binary artifacts, or trigger GPU/DLC imports or recursive artifact
scans.
**Why:** VIEB projects can contain millions of frames and thousands of
videos; loading heavy data during startup causes unacceptable latency.
**Related:** DataLoader, Artifacts scanner
**Directly relevant to:** startup slowdown investigation (see #33).

## 20 — Cluster Runs is a top-level workspace panel
**Decision:** Restore `Cluster Runs` / `Explore Clusters` as a top-level
panel for browsing previous runs, comparing diagnostics, and selecting the
active run.
**Why:** Cluster selection is central to VIEB; users need a dedicated place
to switch/compare runs, separate from interpreting the active one.
**Related:** `user_interface.py`, future run manifest model

## 21 — Analysis reads the active cluster, does not manage all clusters
**Decision:** Analysis displays results from the active cluster run and
links out to Cluster Runs for switching/comparison, rather than duplicating
run-selection logic.
**Related:** `views/analysis.py`, run manifest handling

## 22 — features/index.json is the canonical feature-metadata source
**Decision:** `features/index.json["_meta"]` is canonical for feature
count and wavelet usage. Missing wavelet metadata displays as `unknown`,
never `no`.
**Why:** Older `cluster_diagnostics.json` files stored feature_count but
not use_wavelets, producing impossible displays like "Features: 91,
Wavelets: no."
**Related:** `compare.py`, `features/index.json`

## 23 — cmd_fix_features must preserve feature metadata
**Decision:** `cmd_fix_features()` must not rewrite index.json while
dropping use_wavelets, n_features, feature_count, feature names, or
extraction settings.
**Why:** Repair commands must not corrupt the metadata foundation used by
diagnostics, Analysis, and Cluster Runs.
**Related:** `compare.py`

## 24 — Movement poles are general low/high-motion references
**Decision:** Add `Low-Motion Pole` / `High-Motion Pole` as generic
concepts; UI labels may interpret them as freezing/locomotion contextually,
but the underlying concept is not hardcoded to fear conditioning.
**Related:** `compare.py`, `views/state_characterization.py`

## 25 — Movement poles are saved as artifacts
**Decision:** Pole selections save to `results/characterization/poles.json`
(optionally CSV) including selected state/bout range, mean speed, variance,
bout count, and representative clip paths.
**Why:** Pole selection must be reproducible and exportable, not
memory-only.

## 26 — Degenerate repeated-state motifs are excluded from motif rankings
**Decision:** Motifs like `(48,48)` or `(46,46,46)` are excluded from Top
Context-Enriched Motifs — they represent bout duration, not sequence
structure.
**Why:** Repeated-state tuples dominated rankings and obscured real
transitions like `(12,47,8)`.
**Related:** `compare.py`, motif enrichment outputs

## 27 — Bout duration is a separate analysis from motifs
**Decision:** Repeated-state duration effects report under `Bout Duration
by Context`, not as sequence motifs.
**Related:** Transitions & Motifs redesign

## 28 — Motif clips must be generated and indexed consistently
**Decision:** The GUI must not show motif clip tables/buttons unless
`Generate Motif Clips` actually writes playable clips to the project's
clips/results path and produces a usable exemplar index, visible in
Artifacts.
**Related:** `generate_clips.py`, `compare.py`, `views/analysis.py`

## 29 — Long Analysis pages must be scrollable
**Decision:** Pages with matplotlib charts, long tables, or stacked panels
must be wrapped in correct scrollable containers.
**Why:** Transitions & Motifs and similar pages were unusable — content
clipped with no way to scroll.
**Related:** `views/analysis.py`

## 30 — Key paper/export figures must persist to Artifacts
**Decision:** State occupancy, contrast vector comparison, UMAP embedding
visualization, pole clips, state exemplars, and motif clips all save to
disk, not just render in-GUI.
**Why:** Needed for reliable export/reproduction in paper workflows.
**Related:** `views/artifacts.py`, `compare.py`

## 31 — Artifacts is the file cabinet, Analysis is interpretation
**Decision:** Generated files, plots, clips, manifests, diagnostics, and
exports live in Artifacts. Analysis shows curated, contextual views of the
same results, not every generated file.
**Related:** `views/artifacts.py`, `views/analysis.py`

## 32 — cuML/DLC dependency separation
**Decision:** cuML is unrelated to DeepLabCut. DLC needs its own
deep-learning/GPU stack; cuML only accelerates UMAP/HDBSCAN clustering.
cuML being installed does not imply DLC works, and vice versa.
**Related:** DLC vs clustering setup docs

## 33 — DLC environment: Python headers + Python 3.11 preferred
**Decision:** `vispy` build failures (missing Python.h) require Python dev
headers (`python3.11-devel` etc). Prefer a dedicated Python 3.11 `venv-dlc`
over 3.12 if package builds get painful.
**Related:** `venv-dlc` setup

## 34 — HDBSCAN sampling must be enforced, not just requested
**Decision:** `--hdbscan-sample` must be genuinely respected on both GPU
and CPU paths — fit only on the sample, then assign remaining frames via
`approximate_predict()`. Add a hard guard that aborts rather than silently
attempting full-frame HDBSCAN when frame count exceeds hdbscan_sample.
**Why:** A run with `--hdbscan-sample 300000` still attempted fitting on
all 7,468,010 frames, causing GPU OOM, then CPU fallback got SIGKILLed —
sampling was printed but not enforced in the cuML path, and CPU fallback
had no bound either.
**Related:** `compare.py cmd_cluster()`, `_fit_cpu_hdbscan_with_assignment()`

## 35 — Codebase cleanup: separate dead-code removal from feature integration
**Decision:** First cleanup pass only removes confirmed dead/unwired code
and deduplicates safe constants — it does not wire large unused views
(Browse States, Validation) into the app, since that changes runtime
behavior and startup cost.
**Related:** `user_interface.py`, `views/`

## 36 — Delete dead files, keep the product concepts in the decision log
**Decision:** Unused Browse States/Validation files may be deleted if
unwired, as long as the underlying product ideas survive here as explicit
future work rather than as stale hidden modules.

## 37 — Commit messages should capture product/architecture decisions
**Decision:** Commits affecting startup, clustering UX, onboarding,
metadata, motif, clip, artifact, or analysis-structure direction should
summarize the "why," not just list changed files.

## 38 — Startup slowdown root-caused to commit d585791 (onboarding foundation)
**Decision/finding:** `git bisect` isolated the startup regression to
d585791, "Implemented project onboarding foundation and wired it into
startup, path resolution, and pipeline execution." `git show d585791
--stat` shows the real shape of the change: `gui.py` deleted (-3054
lines, dead code per #35/#36), `project_manager.py` added (+470 lines,
new onboarding logic), `user_interface.py` heavily modified (627 lines
changed), plus smaller changes to `vieb_config.py`, `_utils.py`,
`_workers.py`, `views/pipeline.py`, `views/project_selector.py`.
**Why suspected:** The commit message itself says onboarding was "wired
into startup" — the new `project_manager.py` logic (470 lines) is the
prime suspect for doing synchronous filesystem/path-resolution work on
every launch rather than only during first-time setup, in likely
violation of #19 (heavy data / scans must stay out of the startup path).
**Caveat:** Later commits after d585791 currently test as "instant" — this
is expected to be an artifact of empty/no project data in those test
checkouts, not evidence the regression was fixed. Valid comparison
requires testing every commit against the *same* real project (same
config.json, same populated results/features directories), not a fresh
project with nothing to scan.
**Next step:** Profile `project_manager.py` and the startup path in
d585791 directly (`python -X importtime`, `cProfile`) against a populated
real project, rather than continuing to bisect blindly.
**Related:** commit d585791, `project_manager.py`, #19

---

<!-- Add new entries below this line, incrementing the number -->

## 39 — DLC setup accepts PyTorch trained snapshots
**Decision/finding:** Existing-model validation must accept DLC 3/PyTorch
snapshots in `dlc-models-pytorch/**/train/snapshot-*.pt`, not only
TensorFlow `.index` snapshots under `dlc-models/`.
**Why:** Imported trained DLC projects can be valid PyTorch projects; the
old TensorFlow-only guard falsely blocked pose estimation with "No trained
model found yet."
**Related:** `dlc_project_utils.py`, `views/dlc_setup.py`

## 40 — DLC progress is reported by VIEB markers
**Decision/finding:** DLC Setup progress uses explicit `[VIEB_PROGRESS]`
markers emitted by VIEB's DLC wrapper, rather than parsing DeepLabCut's
internal progress text.
**Why:** Stable app-owned markers make bottom-status video counts reliable
and keep the GUI independent of DLC progress-bar formatting changes.
**Related:** `setup_dlc_training.py`, `views/dlc_setup.py`, `user_interface.py`

## 41 — DLC analysis stays as one bulk DeepLabCut call
**Decision/finding:** Restore DLC pose estimation to one bulk
`deeplabcut.analyze_videos()` call and remove bottom-status video-remaining
updates.
**Why:** Splitting inference into per-video calls caused DLC 3 to report
existing `_full.pickle` outputs with no new `.h5` files, destabilizing
analysis for already-processed videos.
**Related:** `setup_dlc_training.py`, `pretrained_manager.py`, `views/dlc_setup.py`
**Supersedes:** #40

## 42 — State Characterization browses the full clips folder; curated exemplars removed
**Decision/finding:** The State Characterization view lists every clip in
`clips/state_<id>/` (longest/typical/context) instead of only 3 curated
exemplars. The curated 3-cap selection (`select_state_exemplars`,
`state_exemplars.csv`) was removed from `generate_clips.py` as redundant.
Movement Poles panel replaced by a per-metric highest/lowest-state summary;
technical ("More") categories removed; clip playback auto-loads/auto-advances.
**Why:** `generate_clips.py` already exports all per-state clips; capping the UI
at 3 curated clips hid most of them and duplicated the clips directory.
**Related:** `views/state_characterization.py`, `generate_clips.py`,
`artifact_scanner.py`, `characterize.load_clips`

## 43 — H5 video_path resolution: manifest column + extension fallback, repair via CLI flag
**Decision/finding:** `_cmd_extract_h5()` now resolves `video_path` per
session from the H5 manifest's video-path column (first match among
`video_path`/`source_path`/`video_file`/`source_video`, via
`h5_manifest.load_video_paths()`) or, failing that, a `raw_videos_dir/<stem><ext>`
match. `video_path` staying `None` when nothing resolves is expected, not an
error. Existing projects extracted before this fix use the new
`compare.py --backfill-video-paths` command to re-resolve without a full
re-extraction, following `cmd_fix_features()`'s in-place-repair discipline
(never drops unrelated `index.json` fields, per #23).
**Why:** H5-extracted entries always wrote `video_path: None`, unconditionally
— generate_clips.py had no way to locate the source video for H5-only labs.
**Related:** `h5_manifest.py`, `compare.py _cmd_extract_h5()`,
`compare.py cmd_backfill_video_paths()`

## 44 — generate_clips.py treats a missing video as a skip, not a crash
**Decision/finding:** `_resolve_video_path()` returns `None` (not the
original unresolved path string) when a video can't be found anywhere.
`cmd_clips()` filters bouts down to resolvable videos before attempting any
export, and prints one availability summary ("N/M sessions have a usable
local video, X missing video_path, Y unresolvable locally") regardless of
whether bouts came from an existing `bouts.csv` or a fresh `_build_bouts_df()`
pass. The "all videos missing" `RuntimeError` guard is preserved for the
case where nothing at all is usable.
**Why:** A `None`/unresolvable `video_path` (now possible after #43) crashed
`os.path.exists()` with a `TypeError` instead of producing clips for whatever
videos a lab with partial local data (e.g. Spence's ~1,638/3,735 local
videos) actually has.
**Related:** `generate_clips.py`

## 45 — animal_id merges always coerce to str via one shared helper
**Decision/finding:** `quantify.py` exposes a public `coerce_id_column(df,
col="animal_id")` (returns a `.astype(str)` copy) and applies it at every
merge/join on `animal_id`, including two previously-unguarded sites in
`build_master_table()` (the cohort merge and the fear_index merge).
Already-safe inline `df["animal_id"] = df["animal_id"].astype(str)` casts
elsewhere in `quantify.py` and in `compare.py::cmd_quantify()` were refactored
to call the same helper instead of repeating the cast inline.
**Why:** `cohort_loader.load_cohort_excel()` casts `animal_id` to `int64`;
`summary_table.csv`'s `animal_id` (via plain `pd.read_csv`) is whatever
pandas infers from the column — a lab whose IDs are alphanumeric (forcing
str) crashes the merge against a cohort file with purely-numeric IDs. This
class of bug recurs wherever IDs get merged across sources with different
origins, hence one reusable helper rather than fixing sites ad hoc.
**Related:** `quantify.py coerce_id_column()`, `quantify.py build_master_table()`

## 46 — Doubled-path detection warns loudly; repair requires an explicit flag
**Decision/finding:** `project_manager.py`'s `detect_doubled_project_segment()`
detects a resolved metadata/results/raw_videos path whose project-relative
segment (`project_path.relative_to(repo_root)`, e.g. `("projects",
"spence_lab")`) appears twice back-to-back — the symptom of a pre-refactor
repo-root-relative `config.json` value being re-resolved under the current
project-relative scheme. `compare.py`'s startup path-diagnostics
(`_print_project_path_diagnostics()`) always prints a loud `[WARN]` with the
exact broken path and, if a working repair candidate exists on disk, prints
it too — but only rewrites `config.json` (via `repair_project_config_path()`)
when the new `--repair-paths` CLI flag is passed.
**Why:** The old resolver classified path *origin*, never *existence* — a
doubled path silently produced an empty-looking dataset (0 animals, "no
values") instead of an error, which is more dangerous than a crash. Repair
is opt-in rather than automatic so a routine pipeline run never silently
rewrites a user's `config.json`.
**Related:** `project_manager.py`, `compare.py _print_project_path_diagnostics()`

## 47 — Settings must maintain `external_paths` (parity with import_data_source)
**Decision/finding:** `views/settings.py:_save()` now keeps `external_paths` in
sync with the results/raw_videos/metadata directory fields — a path resolving
outside the project folder is added to the whitelist, an inside path is removed
— and also writes the nested `paths["metadata"]` key. Previously Settings wrote
only the flat/nested path keys, so pointing raw videos (or results/metadata) at
a folder outside the project (e.g. an external drive) left the path
un-whitelisted; `_classify_project_path()` then rejected it and the value
appeared to "revert" on the next resolve/restart. Metadata additionally reverted
because only the flat `metadata_csv_path` was written while the stale nested
`paths.metadata` won on `normalize_project_config`.
**Why:** The onboarding import flow (`project_manager.import_data_source`,
lines ~901-904) already whitelists external sources; Settings was the only
directory-editing path that didn't, making it the one that silently reverted.
Keeping the logic in `_save()` mirrors the existing pattern with no new module.
**Related:** `views/settings.py:_save()`, `project_manager.import_data_source`,
`project_manager._classify_project_path`

## 48 — State categories are per-video/clip metadata, not a per-state overwrite
**Decision/finding:** In State Characterization, selecting a category chip
and clicking Save no longer writes a single clobbering value into
`results/validation/state_labels.csv` (keyed only by `state_id`). Instead it
appends a vote to `results/characterization/video_state_categories.csv`
(`video, state_id, clip_path, category, timestamp`, keyed by
`(video, state_id, clip_path)`, never-overwrite update-by-key like
`characterize.save_annotations`), resolved to a source video via a new
`results/characterization/clip_video_index.csv` manifest written by
`generate_clips.py:cmd_clips` (mirroring the existing `motif_exemplars.csv`
pattern, Decision #28). The state list's chip pre-highlight now shows the
most recent vote for that state as a lightweight preview.
**Why:** The old mechanism always overwrote whichever single state row was
selected in the left-hand list, regardless of which clip was actually being
reviewed — repeated browse+save cycles just clobbered one value. Per-video
metadata preserves the full history so a state's overall "character" can
later be derived from the distribution of categories seen across its clips.
**Deferred:** Computing that average/highest-weighted aggregate per state
and using it to drive the state's displayed category is explicitly not
built yet — only the metadata logging. Do not assume aggregation exists
without checking `_latest_category_for_state()` in
`views/state_characterization.py`, which today only returns the single
most recent vote.
**Related:** `views/state_characterization.py:_save_state_label`,
`_save_video_category_vote`, `_latest_category_for_state`,
`generate_clips.py:cmd_clips`

## 49 — Video Stories added as a single new Analysis tab, not the full Sequences reorg
**Decision/finding:** Added a "Video Stories" tab to Analysis' CORE ANALYSIS
section (`views/analysis.py`, new `views/video_stories.py`), reading
`results/sequences/video_stories.csv` + `video_story_bouts.csv` (falling back
to `results/characterization/bouts.csv` + `results/comparison/summary_table.csv`
when the sequence artifacts haven't been generated yet). Per-video state
timeline (matplotlib `broken_barh`), story summary card, state legend, and
click-to-play (fixed-window clip generation reusing `generate_clips._export_clip`,
deterministic path under `clips/stories/<video_id>/`, skips regeneration if
that path already exists). Does not add the "Motifs" / "Transitions" /
"Bout Duration" tabs or a nested "Sequences" section — the user explicitly
scoped this pass to Video Stories only; those remain future work under #8.
**Why:** #8 already decided on a `Sequences` panel as part of the broader
Analysis reorg, but building all four sub-tabs in one pass risked touching
the working "Transitions & Motifs" tab unnecessarily. State labels are read
from `results/validation/state_labels.csv` (the real persistence layer used
by State Characterization, per `_load_saved_state_labels`), not a
`state_annotations.json` that doesn't exist anywhere in the codebase — the
original task spec named a file that was never built. Similarly, the
empty-state message points at `python compare.py --report` (which actually
produces these files via `sequence_artifacts.build_sequence_artifacts`),
not a nonexistent `--stories` flag.
**Related:** `views/video_stories.py`, `views/analysis.py`,
`tests/test_video_stories.py`, #7, #8

## 50 — Video Stories Part B: Journey comparison layer, "Video Stories" Artifacts category
**Decision/finding:** Added a collapsed-by-default "Compare across time"
section to the Video Stories panel: compact per-timepoint mini-timelines
for the selected subject (x-axis normalized to fraction of session
duration so sessions of different lengths compare proportionally; visual
only, no click-to-play — confirmed with the user), plus two small plots
of `transition_rate`/`state_entropy`/`distance_from_baseline` read
directly from `results/sequences/subject_journeys.csv` columns (never
recomputed from bouts), with dominant-state shown by coloring each
timepoint's x-tick label rather than a third subplot. Both the main
Part A timeline and the new comparison strips share one drawing routine
(`draw_bout_strip`) so there's exactly one place that renders the
behavioral-barcode look, and both route through a new
`load_possible_split_states()` hook that hatches a state's segments if a
`possible_split_states` key ever appears in `cluster_info.json` — that
key does not exist anywhere in the codebase today (no transition-graph
modularity check has been built), so this is a no-op today by design,
per the task's "skip this item silently" instruction.

Also renamed `artifact_scanner.py`'s `("sequences/", "Sequences")`
category rule to `("sequences/", "Video Stories")`, and split the
previously-uniform `clips/` → `"Clips"` categorization so
`clips/stories/<video_id>/` also reports category "Video Stories" (state
clips under `clips/state_<id>/` are unchanged, still "Clips"; motif clips
are unchanged, still "Motifs") — giving stories a dedicated Artifacts
home instead of a shared generic "Sequences" bucket, confirmed with the
user. `views/artifacts.py` gained a public `select_category(name)` +
pending-category mechanism (handles the case where Artifacts' async scan
hasn't finished yet) so other views can navigate in with a category
preselected, mirroring the existing `_navigate_to_help(section_id)`
pattern. The Video Stories panel's new "Related artifacts" row links
"Video Stories data"/"Story Clips" → category "Video Stories", "Motif
Clips" → existing "Motifs", "State Clips" → existing "Clips".
**Why:** Comparison strips exist to answer "does this subject's state
progression look more baseline-like over time" at a glance — a dense,
literally-scaled full timeline per row would defeat that at the sizes
this section renders at. Reusing `subject_journeys.csv` rather than
recomputing keeps the panel from duplicating the aggregation logic
already in `sequence_artifacts.py`. The `possible_split` hook exists so a
future diagnostic doesn't require new UI plumbing, not because that
diagnostic is being built now.
**Related:** `views/video_stories.py`, `artifact_scanner.py`,
`views/artifacts.py`, `views/analysis.py`, `user_interface.py`,
`tests/test_video_stories.py`, `tests/test_artifact_scanner.py`, #49

## 51 — Artifacts restyled to mirror Analysis's sidebar/card conventions; category is now primary navigation
**Decision/finding:** `views/artifacts.py` was restructured to visually match
`AnalysisView`: a left `QListWidget` category sub-nav (same stylesheet as
Analysis's vertical tab bar) replaces the old `Category` combobox as the
primary navigation mechanism; Search and Type remain as secondary in-category
refinement. A summary card (`"{Category} — N files, size"`, styled with
`state_characterization.py`'s `_CARD_STYLE`) sits above the table, and the
preview pane is wrapped in a titled card (`_section_title("Preview")`,
imported from `views.analysis`) instead of a bare empty-state label. Category
clicks only call `_apply_filters()` over the already-scanned in-memory list —
never `_scan()` — preserving #19 (no rescan triggered by routine navigation
against the 31k+-file dataset). Also added a `("runs/", "Cluster Runs")` rule
to `artifact_scanner.py`'s `_CATEGORY_RULES`, so the 213 files under
`results/runs/<run_id>/` (cluster snapshots) get their own category instead
of falling into the generic `"Raw Tables"` bucket.
**Why:** Artifacts was a flat dense table with combobox filters and plain
buttons, visually disconnected from Analysis's sidebar/card idiom, making the
app feel like two products. Per #31, this was scoped as a layout/navigation
pass only — no change to what data is shown, how scanning/filtering/export
works, or the Artifacts-vs-Analysis conceptual split.
**Related:** `views/artifacts.py`, `artifact_scanner.py`,
`tests/test_artifact_scanner.py`, #19, #31

## 52 — `onboard.py`: headless CLI port of Stage 0 for HPC use
**Decision/finding:** Added a standalone, Qt-free `onboard.py` at the repo root
so VIEB can be onboarded on a headless HPC cluster with one command. It is a
thin wrapper over the already-Qt-free `project_manager` functions the GUI's
`Stage0ReadinessPanel` uses (`create_project` / `register_legacy_project` /
`set_active_project` / `ensure_project_metadata` / `validate_project`); the
panel's `_determine_state` state machine and its status/action wording are
ported verbatim, split into a target-centric `determine_state` + `_classify`.
Two deliberate deviations from the GUI: (1) target selection — an explicit
`--path X` operates strictly on X (creates it when there is no active project;
`--check --path X` reports only X, never a discovered project), while a bare
invocation prefers the active/detected project like the GUI and falls back to
the current directory only when none exists, so a fresh no-active-project setup
can be created in place; (2) it adds a blank-cell scan of all `metadata.csv` columns (not
just validation-required ones) and exits `1` when any are blank — per the user,
"treat as done, note the blanks, exit 1, and print a fill-in command." Exit
codes: `0` ready, `1` needs manual attention (blanks / incomplete), `2` hard
failure (no data source / invalid project). Stage 0 only — runs nothing from
Stage 1+.
**Why:** Onboarding was GUI-only (`user_interface.py`), unreachable on a
cluster; the underlying logic was already headless, so a wrapper avoided
re-implementing it.
**Related:** `onboard.py`, `project_manager.py`, `metadata_generator.py`,
`user_interface.py` (`Stage0ReadinessPanel`)

## 53 — v2 HPC GPU pipeline: driver-matched RAPIDS stack + a dedicated python/3.11.4 GPU venv
**Decision/finding:** Ported v1's `_utils.py:90-177` driver→stack table into
`vieb_v2/representation/gpu.py` (`GPU_STACKS`, `detect_nvidia_driver`,
`select_gpu_stack`, `stack_message`), dropping the WSL2 half as irrelevant on a
Linux cluster. `gpu.report()` now carries both halves of the question — what is
installed (the existing run-it-don't-import-it probes) and what *should* be
(driver-matched recommendation) — since an idle GPU can mean "nothing
installed" or "wrong RAPIDS pin", which need different fixes. `doctor` prints
the recommendation and gains `--print-packages`, which emits only the pinned
pip arguments on stdout (reason goes to stderr, exit 1 when no stack matches),
so `install_gpu.slurm` consumes the table without duplicating it in bash;
it needs only the stdlib, so it answers before cuml/cupy exist.

The long-open "wheels on the existing 3.13 venv vs. rebuild on 3.11.4"
question is resolved, and the reason it was stuck is that it had no fixed
answer: `~/vieb/venv` is Python 3.13.12 (already outside `pyproject.toml`'s
own `requires-python = ">=3.10,<3.13"`), and per PyPI, RAPIDS 24.12 /
cupy 12.2.0 publishes **no** 3.13 wheels at all (`cuml-cu12==24.12.0` is a
source tarball only) while RAPIDS 26.4 / cupy 14.1.1 does (abi3 + cp313). So
`--only-binary=:all:` would have succeeded or failed depending on which stack
the driver resolved to. Chosen resolution: GPU installs always build a
dedicated `python/3.11.4` venv (`hpc/install_gpu.slurm`, `$HOME/vieb/venv-gpu`),
which both stacks have wheels for; the default 3.13 venv stays CPU-only and
untouched. The venv path is recorded in `hpc/.gpu_venv`, checked by
`submit.sh` before queuing and by `02_compare_latents.slurm` on start.

`install_gpu.slurm` is a one-time preflight, deliberately **not** chained into
`submit.sh` — it would put a gpu-partition job on the critical path of every
submission to redo work that only changes when the driver does. `doctor.slurm`
moved to `partition=gpu` because `nvidia-smi` does not exist on a login node
(verified), so a driver recommendation is impossible anywhere else; it keeps
using the default venv on purpose, since its job is to compare installed
against recommended. `02_compare_latents.slurm` moved to `partition=gpu
--gres=gpu:1` with `--gpu on` (not `auto`): a silent CPU fallback would spend
the whole 36h GPU allocation, and `gpu.resolve()` already raises in the first
second instead. `01_align.slurm` stays CPU on `normal` — `gpu.py`'s own
reasoning is that the pooled PCA is a 14×14 eigenproblem; only HDBSCAN, which
labels all 28.6M frames of the current project, is worth a device. Also
removed the vestigial `module load python/3.11.4` from `01_align.slurm` and
`doctor.slurm`, where it was silently shadowed by the 3.13 venv activated on
the next line.
**Why:** GPU support was half-decided: the CLI had `--gpu auto|on|off` and
capability probes, but the HPC scripts ran `--gpu off` on `partition=normal`
and `doctor` could only report what was already installed, never what to
install — leaving no path from "GPU is idle" to "GPU is working."
**Related:** `vieb_v2/representation/gpu.py`, `vieb_v2/cli.py`,
`vieb_v2/hpc/install_gpu.slurm` (new), `vieb_v2/hpc/doctor.slurm`,
`vieb_v2/hpc/02_compare_latents.slurm`, `vieb_v2/hpc/submit.sh`,
`vieb_v2/hpc/01_align.slurm`, `vieb_v2/tests/test_gpu.py`,
`vieb_v2/tests/test_cli.py`, `_utils.py:90-177`, #3, #32, #52

## 54 — Diffusion maps: isolated landmarks must be pruned, and every λ=1 eigenvector dropped
**Decision/finding:** Root-caused the v2 GPU job that ran 18h on CPU with a
silent log (job 46264, `--gpu on`). Two independent defects, one data-shaped
and one control-flow-shaped, compounded:

*The embedding.* `DiffusionMap.fit` built its operator on 3000 landmarks
sampled evenly in time. On the 28.6M-frame project ~142 of those are outlier
poses that no neighbourhood reaches, so each forms its own connected component.
A disconnected component contributes an eigenvalue of **exactly 1.0**, and
since 1.0 is the largest eigenvalue the operator admits, those spikes sort to
the *top* of the spectrum and evict every genuine mode from the retained set —
measured participation ratio ≈ 1 (each "coordinate" was a spike on one
landmark). The embedding degenerated into component indicators: **20k frames
collapsed onto 29 distinct points**. Fixed by `_prune_isolated`, which drops
landmarks whose off-diagonal kernel mass is below `MIN_NEIGHBOR_MASS = 1.0`
("one effective neighbour"), iteratively, re-resolving epsilon each round.
Measured on the real checkpoint: 142/3000 pruned (4.7%), epsilon essentially
unchanged (478.6 vs 501.4), spectrum decays properly (λ₁ 0.988 → λ₈ 0.935),
exactly one trivial eigenvector, **49,869/50,000 distinct embedded points**.

Rejected: connecting the same graph by *raising* epsilon instead. Binary search
puts the smallest connecting bandwidth at the 90.3rd percentile of pairwise
distances — 12.7× the default — which is deep in the short-circuit regime
`EPSILON_PERCENTILE` was tuned to avoid (the module's own measurements put p50
at rank-corr 0.09). Dropping a handful of unreachable outlier poses is both
cheaper and more honest than over-smoothing the whole manifold to accommodate
them.

*The eigenvector mask.* `_non_trivial` looked for the first λ=1 eigenvector
that was *constant* and `break`ed. On a connected operator λ=1 has multiplicity
1 and is constant, so that read correctly. On a disconnected one the
multiplicity equals the component count and `eigh` returns an arbitrary basis
of that eigenspace — component indicators, not constants — so the constancy
test recognised none of them and let all but one through. Now keyed on the
eigenvalue alone, with `n_trivial_eigenvectors > 1` warning loudly
(`DegenerateOperatorWarning`) since that is the signature of an operator that
is still fragmented.

*The silent fallback.* `representation/cluster.py:cluster()` caught every
`_fit_gpu` exception with a bare `except Exception: result = None` and no log,
so cuML dying on the degenerate embedding was invisible and CPU HDBSCAN
inherited 28.6M points. `--gpu on` now re-raises with the original exception
chained (`gpu.explicitly_requested()` shares `resolve()`'s spelling tuple, so
there is one source of truth for what counts as a demand for GPU); `auto`
still falls back, but via `GPUFallbackWarning` rather than silence. This is
the same principle as #53 — fail in the first second rather than spend the
allocation — extended from capability-probe time to fit time, which is where
it actually bit.
**Why:** `--gpu on` existed precisely so a GPU job could not quietly become a
CPU job, but the guarantee stopped at the capability probe; a runtime failure
walked straight past it. And the degenerate embedding would have produced
meaningless states even if the clustering had finished.
**Related:** `vieb_v2/representation/diffusion.py`,
`vieb_v2/representation/cluster.py`, `vieb_v2/representation/gpu.py`,
`vieb_v2/cli.py` (`_report_operator_health`),
`vieb_v2/tests/test_diffusion.py`, `vieb_v2/tests/test_cluster.py` (new), #53,
#34

## 55 — Koopman attractor-topology decomposition built and synthetically verified; its real-data half deferred on an unmet `flow-field` prerequisite
**Decision/finding:** Added `vieb_v2/representation/koopman.py` (+
`vieb_v2/tests/test_koopman.py`) on branch `koopman`: global SVD-based DMD,
per-region affine Koopman operators, and topology extraction (fixed points,
limit cycles, basins, separatrices) where a behavioral state is a **basin of
attraction**, not a density peak — so the state count is an output, not a
parameter. No clustering runs in this path.

The prompt specifying this work gated it on the `flow-field` branch (Prompt A)
having run on real Luna data with an acceptable `v_coherence` distribution.
That gate is **unmet and cannot be evaluated here**: no `flow-field` branch
exists locally or on `origin`, `v_coherence` appears nowhere in the tree or
history, and `/home/carlos/vieb` (the Luna project in `projects.json`) is not
present on this machine. `VUS-1` and `compare_methods.py`, named as the output
format and comparison target, also do not exist. So the topology machinery plus
its known-answer verification — which the prompt itself ordered first — was
built, and the Luna run, the VUS-1 emit and the confound *report* were not.
Basin labels use `-1` = near-separatrix, matching `metrics.NOISE_LABEL`, so
`cluster_metrics` / `speed_diagnostics` already score them unchanged and
`noise_speed_ratio` / `size_speed_rank_corr` need no new code when the gate
clears. On-disk shape reuses `checkpoints.save` (`labels`/`probabilities`/
`index`), identical to `labels.npz`; the `VUS-1` name is deferred until a spec
exists.

Four findings worth not rediscovering:
1. **Local operators must be affine.** `p -> A p` cannot represent circulation
   about any point but the origin, so a limit-cycle arc reports contraction
   instead of rotation. Implemented by centering both snapshot matrices, which
   makes the least-squares `A` the affine one and lets global and local fits
   share one code path.
2. **Region pairs are selected by origin frame, not both endpoints.** A local
   operator is the flow map *on* a region; its image is not the region. On a
   cycle the per-frame step routinely exceeds a cell's width, so the
   both-endpoints rule produced *zero* pairs for precisely the fast regions
   whose rotation matters, and the cycle went undetected.
3. **Graph edges need an absolute count floor, not only a share floor.** A
   sparse transient region owns a large Voronoi cell, so one stray frame can be
   >2% of its outgoing mass. Unpruned, a single noise excursion welded both
   basins into one recurrent set and every attractor vanished. Defaults are now
   `min_edge_frac=0.05` **and** `min_edge_count=3`.
4. **Eigenvalues cannot separate a freeze from a gait.** A stable spiral and a
   stable limit cycle both have complex eigenvalues of modulus near 1, and a
   `||v||` percentile threshold is no better because a fixed point sits inside
   the population defining it. The discriminator is **direction coherence**
   (mean cosine between consecutive step vectors): ~1 on a cycle, ~0 at a
   noise-driven fixed point, and scale-free so it needs no calibration to the
   latent's arbitrary units.

**Why:** Extracting attractor topology from a flow that was never validated
would read structure out of noise — the prompt's own stated reason for the
gate. The synthetic half has no dependence on the flow field, so it was
deliverable in full and is what makes the real-data half trustworthy later.
The verification system deviates from the suggested "damped oscillator + Van
der Pol in separate regions": two disjoint regions share no boundary, so the
separatrix check would pass vacuously, and Van der Pol's period has no closed
form outside `mu -> 0`. Used instead is
`r' = -k r (r-a)(r-b), theta' = omega`, whose fixed point, separatrix (the
unstable cycle at `r=a`), stable cycle (period exactly `2*pi/omega`) and basins
are all known analytically. Verified over 6 seeds: 1 fixed point + 1 limit
cycle every time, recovered period 1.004–1.030 s (eigenvalue) and 1.000 s
(return time) against a true 1.000 s, basin accuracy 0.997–1.000.
**Related:** `vieb_v2/representation/koopman.py`,
`vieb_v2/tests/test_koopman.py`, `vieb_v2/representation/metrics.py`,
`vieb_v2/representation/checkpoints.py`, `vieb_v2/representation/embed.py`
(the boundary guard this mirrors)

## 56 — Merging `v2` into `koopman`: v2's GPU/driver stack wins wholesale, and the `.sbatch` suite replaces the `.slurm` one

**Decision:** `v2` and `koopman` solved the same GPU-preflight problem in
parallel from a shared base (`dd67c6d`). The merge resolves toward `v2` in
every overlapping file rather than splicing the two:

1. **`representation/gpu.py` and `representation/diffusion.py` take `v2`
   wholesale.** `v2`'s driver detection (`GPU_STACKS` / `detect_nvidia_driver`
   / `select_gpu_stack` / `stack_message`) is a strict superset of koopman's
   `RAPIDS_STACKS` path, and its `diffusion.py` carries the lookup-table-collapse
   and Nyström-NaN fixes that koopman's copy predates. `use_gpu` defaults to
   `False`, so koopman-side callers that dropped the kwarg still work.
2. **`cmd_doctor` keeps only `v2`'s body.** The automatic merge appended
   koopman's tail block to `v2`'s, and that block reads `install_command`,
   `recommended_stack` and `python_version` — keys `v2`'s `report()` does not
   emit, so `doctor` raised `KeyError` on every invocation. The block was
   deleted rather than aliased into `report()`: `v2`'s lines above it already
   print the recommended stack, the `stack_message`, and the 3.11-vs-3.13 venv
   caveat, so aliasing would only have duplicated the advice.
3. **`hpc/submit.sh` and the `.sbatch` suite are koopman's**; the four
   re-added `.slurm` files (`01_align`, `02_compare_latents`, `doctor`,
   `install_gpu`) were deleted. `hpc/README.md` documents the `.sbatch` suite,
   and `install_gpu` is a plain `./install_gpu.sh`, not a batch job — the three
   stale `hpc/install_gpu.slurm` references in `cli.py` were repointed.

**Also:** every `.sbatch` (and `install_gpu.sh`) hardcoded
`source "$HOME/vieb/venv/bin/activate"`. That venv is Python 3.13, which
`install_gpu.sh` itself warns cannot host RAPIDS; the GPU stack lives in a
separate 3.11 venv. All five now use `VENV="${VENV:-$HOME/vieb/venv}"`, so the
venv is selectable per-submission without editing tracked files. This
generalizes the local path hack that was sitting in `git stash`, which is
therefore obsolete — it patches files this merge deleted.

**Why:** Two parallel solutions to one problem cannot both survive without a
rule for which wins; picking per-file by supersededness keeps one driver-
detection code path instead of two that drift. The `KeyError` is the concrete
cost of splicing instead of choosing.
**Related:** `vieb_v2/representation/gpu.py`, `vieb_v2/representation/diffusion.py`,
`vieb_v2/cli.py`, `vieb_v2/hpc/*.sbatch`, `vieb_v2/hpc/install_gpu.sh`,
`vieb_v2/hpc/README.md`, decisions #53, #54, #55

## 57 — Koopman wired to the CLI; the separatrix kNN is subsampled, and its `-1` is joined by index, never position

**Decision:** `koopman.py` had no caller — no CLI subcommand, no sbatch, no
importer (decision #55 deferred the real-data half). It is now reachable as
`python -m cli koopman`, with three changes forced by the jump from
verification scale (14k frames) to Luna (4925 sessions, 28,626,107 frames --
12x the "~2.3M frames" `_knn_indices` was written against):

1. **The separatrix neighbour index is fitted on a subsample.** A
   fit-and-query over all pairs is what does not scale, not the query. Above
   `--knn-sample` (default 1,000,000) frames the index is fitted on a seeded
   subsample and *every* frame is queried against it, so every frame still gets
   a label — only the pool of possible neighbours is sampled. This mirrors what
   `--hdbscan-sample` already does for clustering. `knn_sample` and
   `knn_subsampled` are written into the report so an approximate result is
   visible rather than implied; `--knn-sample 0` forces the exact index.
2. **`n_jobs=-1` on the neighbour query.** sklearn defaults to one core, and
   the query is the whole cost at this scale: 28.6M frames against a 1M-point
   index in 9-D measures ~150 min single-threaded.
3. **The per-row `np.unique` loop is gone.** Counting distinct neighbour
   labels row by row is fine at 14k rows and hopeless at 28.6M. It is now a
   sort-and-count. The obvious sentinel implementation is a **bug** under
   NumPy 2: `np.where(labels >= 0, labels, np.iinfo(np.int64).max)` on an
   `int32` array silently wraps the sentinel to `-1` under weak promotion
   rather than upcasting, so noise reads as a distinct state and frames are
   over-flagged as transitions. Sorting alone separates the classes — every
   negative sorts ahead of every valid label — so no sentinel is needed.

**Index alignment.** `koopman_labels.npz` carries the same three arrays as
`labels.npz` and drops into the same slot, but the two are **not** positionally
comparable: Koopman labels every frame of `scores.npz` (28,626,107) while
HDBSCAN labels the delay-embedded frames (28,586,707), fewer by one window per
recording (4925 x 8 = 39,400). They must be joined on the `index` array
(`recording`, `frame`) both checkpoints carry. Its `-1` is `metrics.NOISE_LABEL`
by value but means *near a separatrix*, not HDBSCAN noise.

**Also:** `compare-latents` computes latents in memory and never writes
`scores.npz`; only `latent` does. Since Koopman reads `scores.npz`, a
`latent.sbatch` was added to checkpoint one latent per `OUT_DIR`.
`--n-regions` is the one free parameter that could manufacture the state
count, so `koopman.sbatch` documents sweeping it into *separate* out-dirs —
`koopman_labels.npz` has a fixed name and arms sharing an out-dir would
overwrite each other's labels and registry.

**Why:** The state count is supposed to be an output. That claim is only worth
anything if the parameter that could fake it has been varied, and if the
approximations taken to make the run feasible are recorded where a reader will
see them.
**Related:** `vieb_v2/representation/koopman.py`, `vieb_v2/cli.py`,
`vieb_v2/hpc/koopman.sbatch`, `vieb_v2/hpc/latent.sbatch`,
`vieb_v2/representation/checkpoints.py`, decisions #53, #55

## 58 — The `.sbatch` rewrite dropped #53's GPU venv wiring; every gpu job silently ran on the CPU venv
**Decision/finding:** Jobs 46982/46986 (`embed_cluster`, `--gres=gpu:1`) held
two GPUs at 0% utilisation for 4.5h while one CPU core did HDBSCAN on
28.6M×45 points. Root cause is a regression from #56: the `.slurm` → `.sbatch`
port carried #53's *venv* forward but not the *wiring to it*. #53 established
`$HOME/vieb/venv-gpu` (python/3.11.4 + RAPIDS) with the path recorded in
`hpc/.gpu_venv` and "checked by `submit.sh` before queuing and by
`02_compare_latents` on start". After the port, no script read the marker, the
marker itself had been renamed to `.venv-gpu` (so `.gitignore` no longer
matched it and it showed up untracked), and every gpu-partition script sourced
`$HOME/vieb/venv`  — Python 3.13, CPU-only, no `cuml`. The `module load
python/3.11.4` line #53 removed as "silently shadowed by the 3.13 venv
activated on the next line" had also come back in all of them.

`embed_cluster.sbatch` (added later, in e76c67c) compounded it by defaulting
`GPU=auto` where #53 had ruled `on` for gpu-partition jobs. **`auto` is the one
setting with no diagnostic at all**: #54 added `GPUFallbackWarning` for a
*fit-time* cuml failure, but when `gpu.resolve()` simply never selects the GPU —
because cuml was never importable — there is no warning anywhere. The 4.5h log
is three lines, none of them about the GPU. This is the third time this failure
class has appeared (#53 the capability probe, #54 fit-time, now import-time).

**Fixed by** restoring #53's arrangement rather than reinventing it: `VENV`
defaults to `$HOME/vieb/venv-gpu` in `full_pipeline`, `02_compare_latents` and
`embed_cluster`, each failing with a pointed message if it is absent;
`embed_cluster` now defaults `GPU=on`; `submit.sh` re-checks the venv before
queuing anything, since stage 2 failing on start otherwise wastes stage 1's
alignment. `install_gpu.sh` also targeted the default venv, which would have
installed RAPIDS into the 3.13 env #53 specifically ruled out — it now targets
and, if needed, builds `venv-gpu`. `doctor.sbatch` deliberately keeps the
default venv, per #53's reasoning that it compares installed against
recommended. `.gitignore` updated to the marker's actual name.

Dropped the marker file as a source of truth: nothing read it, and a plain
default plus an existence check gives the same protection without a second
place for the path to drift. Kept it gitignored as local state.

**Measured, for calibration:** CPU HDBSCAN fitted on 20k points at 45-D takes
18.3s to `approximate_predict` one 25k batch; the full job is 1,144 batches.
Progress inferred from RSS growth (`_predict_batched` fills `labels`/`probs`
sequentially) put the 45-D arm at ~19h of prediction remaining against a 12h
wall — it would have died with nothing written.
**Why:** #53 and #54 both concluded "fail in the first second rather than spend
the allocation," and both were correct; the guarantee was lost not in the logic
but in a file rename and a default. A capability that is only enforced by a
default in six separate scripts will be lost again, so the check now lives at
submission time as well as job start.
**Related:** `vieb_v2/hpc/embed_cluster.sbatch`,
`vieb_v2/hpc/02_compare_latents.sbatch`, `vieb_v2/hpc/full_pipeline.sbatch`,
`vieb_v2/hpc/install_gpu.sh`, `vieb_v2/hpc/submit.sh`,
`vieb_v2/hpc/.gitignore`, `vieb_v2/hpc/README.md`, jobs 46982/46986, #53, #54,
#56

## 59 — `transfer-operator` branches off `koopman`, not `main`, and four of Prompt C's premises were wrong

**Decision:** the `transfer-operator` branch is cut from `koopman`, not from
`main` as the brief specified. `main` contains **zero** `vieb_v2/` files — no
`representation/`, no `koopman.py`, no `.sbatch` suite. `koopman` is 56 commits
ahead of `main` and 0 behind, so it strictly contains it. Branching off `main`
would have discarded the very module the brief's §6 says to repurpose.

**Findings that changed the plan**, each verified on disk before any code:

- **No k-means microstate assignments exist.** The brief says "you already have
  them, do not rebuild." `koopman.partition()` computes `region_ids` and
  `save_topology` (`koopman.py:896`) writes only `labels`/`probabilities`/
  `index` — the assignments never reach disk. They must be rebuilt.
- ~~**MoSeq produced 42 distinct syllables** (39 above the frequency floor), not
  the 48 the brief cites.~~ **Wrong — retracted.** Counted over all 3,846 result
  CSVs, MoSeq emits exactly **48** distinct syllables, of which 28 clear a 0.1%
  frequency floor. The brief's 48 was correct and this entry's correction of it
  was not. (`N_REGIONS=48` in `koopman.sbatch:37` is a coincidence, not the
  source.) The same count confirms something more useful: MoSeq's total is
  **22,355,989 frames**, identical to the deduplicated alignment, which is an
  independent check on the h5/csv dedup from a tool that never saw it.
- **`compare_methods.py` and any VUS-1 consumer do not exist**, so "so
  `compare_methods.py` works unchanged" is unachievable. ExBias is the only
  producer and its one run has `n_states: 0`.
- **`behavior_metrics.py`'s pooled-index bug is already fixed**
  (`behavior_metrics.py:269-288` masks by `recording_id`). A residual defect
  remains and is *not* fixed here: `load_pose_aligned` omits the
  confidence-weighted smoothing `exbias.prepare` applies, so `C_shape`/`C_pred`
  are measured on a different signal than the bounds were derived from.
- **§0c is already resolved**: K=7, `tail_tip` dropped at `keypoints.py:32`.
  Only the docstring at `keypoints.py:5` was wrong — it listed a keypoint order
  contradicting `DEFAULT_BODYPARTS` at L19. Fixed.

**Why:** the brief is a structural argument from prior measurements, and it says
so (§10: "none of this has been tested on this data"). Checking its premises
against the repo before building on them cost an hour and removed four
dependencies on things that were not true.
**Related:** `vieb_v2/representation/{align,checkpoints,keypoints,pose_loader,
koopman}.py`, `vieb_v2/cli.py`, #55, #57

## 60 — Alignment discarded arena position and heading; `align_all_full` keeps them

**Decision/finding:** `align_all` (`align.py:114`) returns `align_session(...)[0]`
and drops `theta`; the weighted centroid is computed at L89 and never surfaced.
The v2 aligned space is therefore purely postural — translation and heading are
gone by construction, so freezing and steady locomotion are degenerate in it.
v1's `PoseFeatureExtractor` computed `centroid_speed` and `angular_velocity` in
Layer 1; the v2 path dropped both.

Delay embedding cannot repair this. It recovers derivatives of what was
*measured*, and these were subtracted before measurement.

Added `align_all_full()` returning `{aligned, reference, theta, centroid}`
alongside the unchanged `align_all`, plus public `weighted_centroid()` and
`frame_weights()` so a caller reconstructing the centroid uses exactly the
weights the alignment used. **Verified against a synthetic rigid body under a
known time-varying rotation: `heading = -theta` to 6.2e-15**, and the centroid
reproduces arena position exactly under uniform weights.

**Why:** additive rather than a change to `align_all`, so `cmd_align` and
`test_align.py` are untouched and the 206-test suite stays green. The locomotor
channels land in a new `latent_plus.npz` beside `scores.npz` rather than
replacing it, which is what makes "did restoring them change the answer?"
answerable instead of assumed.
**Related:** `vieb_v2/representation/align.py`, `checkpoints.py` `EXTRA_FILES`,
#59

## 61 — The positive control passes emphatically: MoSeq already separates Context A after conditioning

**Finding:** Prompt C's §8 asked whether any Keypoint-MoSeq syllable shifts in
Context A post-shock, on the grounds that it is free and bounds how broken the
representation is. It does, overwhelmingly.

Syllable 1, across the whole design (mean occupancy, 298 animals, every animal
in every phase, all recordings truncated to a common 5,381 frames):

| phase | occupancy |
|---|---|
| CFC d0 Context A — conditioning | 0.095 |
| CFC d1 Context A No Shock — **retrieval** | **0.463** |
| CFC d2 Context C — novel context | 0.170 |
| CFD d3–d7 Context A | 0.429 → 0.552 → 0.597 → 0.602 → 0.597 |
| CFD d3–d7 Context B | 0.351 → 0.435 → 0.419 → 0.414 → 0.398 |

Paired Wilcoxon per syllable on per-animal means, BH FDR: **33/35 syllables at
q < 0.05** for retrieval vs conditioning; syllable 1 at q = 8.4e-47, rank-biserial
+0.98. The A−B discrimination gap widens monotonically across the CFD days
(+0.078, +0.117, +0.178, +0.188, +0.199).

The profile is what freezing should look like: near-absent while the animal is
naive, quadrupled on re-exposure to the conditioned context, only mildly raised
by a *novel* context, and increasingly context-selective as discrimination is
learned. We have not confirmed the label against the grid movies, so it is a
candidate, not an identification.

**Two controls, both required before believing it:**
- *Session length.* Context A sessions run ~6,302 frames against ~5,392 for
  Context B/C — the shock protocol needs the time — so a syllable whose rate
  drifts within a session would separate the arms for free. Truncating every
  recording to a common 5,381 frames *increases* the effect (0.119→0.463
  becomes 0.095→0.463). Not a length artifact.
- *Sign-flip null.* Swapping an animal's two arms negates its difference vector,
  which is the exact randomization null for a paired signed-rank test. Over 100
  repeats: median 0 significant, mean 0.15, **97% of repeats found none**, and
  **0% reached the observed 33**. The occasional burst to 13 is compositional
  dependence — occupancies sum to 1, so the 35 tests are strongly dependent and
  the rejection count is overdispersed — which is why the whole distribution is
  reported rather than a mean.

**What this does and does not change.** The brief anticipated that a passing
control would weaken §2's degeneracy claim. It does not, because it is not a
test of the v2 representation: MoSeq runs its own egocentric alignment and fits
an AR-HMM to pose *dynamics*, with centroid and heading jointly inferred rather
than discarded. What it establishes is that the effect is present in this pose
data and is findable — which converts every later negative result from ambiguous
into interpretable. If the transfer operator cannot recover a state like this,
the loss happened in the representation or the operator, not in the animals.
§0a remains the decisive test of the aligned space.

**Also fixed here:** the rank-biserial effect size was derived from scipy's
two-sided `statistic`, which is `min(W+, W-)` and therefore signless — every
syllable reported a positive effect, including ones whose occupancy had more
than halved. Now computed from W+ and W- directly, and pinned by a test.

**Related:** `vieb_v2/scripts/moseq_control.py`,
`vieb_v2/tests/test_moseq_control.py`,
`/home/tul26194/vieb2-results/transfer_operator/moseq_control/`, #59

## 62 — The transfer operator clears its synthetic gate, and the gate corrected two of its own defaults

**Decision/finding:** `representation/transfer_operator.py` passes five synthetic
systems with analytically known answers, in 8 s. No real data goes near it until
they pass. Three of the five caught something.

**The duration control — the branch's central claim — holds.** A 3-state chain
with geometric dwells: A and B entered essentially equally often (measured 607
against 605, a 0.3% difference) while A occupies **19.4x** the time, plus a rare
fast state C at 313 frames in 360,000. `pi` recovers the occupancy to within 1%
and C survives the connected set at its correct measure. This is the confound
dissolved rather than relabelled: a density-based clusterer cannot separate
"where the animal spends time" from "what the animal is doing", because for it
those are the same number.

**Two defaults were wrong and the gate is what found them:**

- `lag_margin` was 5.0, on the reasoning that a timescale only a few lags long
  is fitted from a handful of eigenvalue digits. The Ornstein-Uhlenbeck system
  measured that: OU's timescale is flat within 5% of its analytic 1/theta across
  a **thirtyfold** lag range, and a 5x margin rejected all but the two shortest
  lags of it — excluding exactly the regime where the estimate is best. Now 1.0,
  the standard `y = tau` line.
- `min_spectral_gap` was 1.2. OU's eigenvalues are exp(-n·theta·t), so its
  consecutive timescale ratios are exactly (n+1):n and **t2/t3 = 2.00**. A gap
  threshold at or below 2 cannot distinguish one-dimensional relaxation from
  metastability at any tuning. Raised to 2.0, and — more importantly — the
  verdict now requires eigenvector **sign structure** as a separate condition,
  since that is the criterion that actually separates the two. OU passes plateau
  and gap and is rejected only by sign structure, which is asserted directly.

**Three test premises were wrong, not the code:**

- *Limit cycle.* Aliasing lives in the eigenvalues, not the timescales. At half
  an orbit the reversible operator is a shift by B/2, giving 12 two-cycles and
  eigenvalues of exactly ±1 — the period-2 mode makes `t_imp` NaN *by design*,
  since clipping a negative eigenvalue would report a fast process where there
  is a rhythmic one. At a full orbit P is literally the identity (diagonal
  fraction 1.000, 24 singleton components) and `operator_at_lag` correctly
  refuses to report anything.
- *i.i.d. noise.* Timescales **grow linearly** in tau on noise — lambda_2 sits at
  the sampling floor and does not move, so t = -tau·dt/log(lambda_2) is
  proportional to tau by construction. That linear growth is the null signature
  the brief names as the falsification condition; asserting flatness would have
  been asserting the opposite of the truth.
- *Double well.* Read at tau=5 the second timescale is 171 s against an empirical
  111 s; it converges downward through 119, 98, 90 to 87 as the lag grows. The
  short-lag inflation is the near-identity artifact — with velocity hidden, a
  particle that has barely moved looks like it stayed. Measured at a converged
  lag the agreement is 0.81–0.85 across all three temperatures and the Arrhenius
  slope comes out at -0.995 against a true barrier of -1.0.

**One limit recorded rather than fixed:** at 60 microstates, k-means gives state
C no centre at all — 313 points absorbed into a neighbour, the state gone before
any operator is built. 120 resolves it. This is a discretization limit on rare
states that the operator cannot repair and that nothing announces on real data.
Pinned by its own test so the k choice reads as a measured requirement.

**And one thing the operator cannot do, stated plainly:** a near-decomposable
pooled chain *is* metastable, mathematically. Two sub-populations never observed
moving between each other produce a real slow eigenvalue and a convincing
plateau, and no pooled diagnostic distinguishes "states are behaviors" from
"states are animals". The test demonstrates a pooled t2 more than 50x anything
present within either group. Refitting within strata is the only thing that
detects it, which is why `--stratify` is mandatory in the gate rather than
advisory.

**Related:** `vieb_v2/representation/transfer_operator.py`,
`vieb_v2/tests/test_transfer_operator.py` (34 tests), #59, #61

---

## 63 — Alignment costs real locomotor information, but does not destroy it: AUC 0.79, not 0.5

Prompt C §2a set up a two-way read: AUC ~0.5 means alignment destroyed the
freeze/locomote distinction, AUC >~0.8 means it left a postural signature and
restoring the channels is a refinement rather than a rescue. Measured on all
3,846 recordings, the answer is **neither**, and the gap between the two models
is the finding.

| model | AUC | 95% CI (bootstrap over recordings) |
|---|---|---|
| logistic regression | 0.693 | [0.687, 0.698] |
| gradient boosting | **0.790** | [0.784, 0.795] |
| restored channels (circular) | 1.000 | wiring check only |

Read off the logistic number alone, this looks like a strong degeneracy claim.
It is not: **+0.097 of the signature is present but not linearly decodable**, and
the boosted CI stops just short of 0.8 — close enough that the linear number
alone would have overstated the case. `--model both` is therefore the default;
a linear probe on its own cannot separate "the information is absent" from "the
information is curved", and that distinction is the whole question here.

The magnitude worth carrying into the writeup: the slow and fast terciles differ
by **55x** in real speed (median 0.8 vs 43.8 px/s), and posture still recovers
that only at 0.79. A 55-fold kinematic difference is not fully visible in the
aligned representation, which is why §0b restores the channels rather than
trusting delay embedding to reconstruct them — delay embedding recovers
derivatives of what was measured, and centroid translation was subtracted
before measurement.

**Two protocol choices that carry the number.** The held-out split is **by
recording**, not by frame: adjacent frames are near-copies, so a random frame
split reports autocorrelation rather than generalization. The CI is a bootstrap
over **recordings** for the same reason — the frame count is arbitrary, the
session count is not.

**Related:** `vieb_v2/representation/observations.py`,
`vieb_v2/tests/test_observations.py` (27 tests), #60, Prompt C §2

---

## 64 — The §3 gate fails: no plateau at any lag, in both arms. The branch stops here.

Prompt C §3 pre-registered the death condition: *"t_imp grows linearly in tau
from the start, no plateau at any tau -> there is no Markovian coarse-graining
at any resolution on this data. The branch dies here. Report it, stop, do not
tune."* Measured on all 3,846 recordings at 500 Voronoi microstates, over lags
from 0.033 s to 36 s, **there is no plateau** for t2, t3 or t4 — in either arm.

| arm | dim | t2 at 0.033s | t2 at 36s | d log t2 / d log tau | verdict |
|---|---|---|---|---|---|
| pose PCs + restored channels | 11D | 0.611 s | 65.4 s | 0.670 | no plateau |
| pose PCs only (control) | 9D | 0.546 s | 71.9 s | 0.707 | no plateau |

**Neither artifact region the brief warns about explains it.** t2/tau >= 1.82
everywhere, so this is not the small-lag near-identity region; lambda2 is still
0.575 and declining at tau=36 s on 18.2M pairs, so it is not the large-lag noise
region. **And the estimate is not broken**: at every one of the 26 lags all 500
microstates are retained, `dropped_frame_frac` is 0, there is exactly one
connected component, `leak_frac` is 0, `near_reducible` is False, and every lag
carries 18.2–22.4M transition pairs.

**How it fails is worth more than the fact that it fails.** The growth is
scale-free rather than linear — the local exponent drifts monotonically from
0.529 to 0.847, strictly between a plateau (0) and the trivial large-tau
artifact (1). The leading eigenvalue violates the semigroup property in one
consistent direction, **lambda2(2 tau) > lambda2(tau)^2 at every lag** (excess
+0.028 to +0.061, growing). Correlations decay more slowly than any single
exponential at every scale from 33 ms to 36 s: long memory, no timescale
separation.

**Three confounds ruled out.** Reversibilization is not the cause — the
counts-symmetrized estimator agrees within 2.2% (exponent 0.667 vs 0.670). The
restored channels are not the cause and do not repair it — the pose-only control
is marginally worse. Smoothing is not the cause — the control applies none and
behaves identically.

**The concern with the gate, recorded rather than acted on.** This is precisely
what Costa et al. (the branch's own reference, §11) predict at K=1: the
instantaneous observable is not the full state, implied timescales therefore
grow with tau, and delay embedding to K* is their remedy — which is what §5a
exists to do. §3 was specified to run *before* any delay embedding, so the gate
as written may be falsifying K=1 rather than the branch. Per §3 and §9 the work
stopped and nothing was tuned; whether to run §5a is the user's call.

**Untested, and named as such:** one partition resolution (N=500). Coarse
partitions bias timescales low, so a finer one raises the curve, but a power law
does not become a plateau by rescaling.

**Related:** `docs/TRANSFER_OPERATOR_FINDINGS.md`,
`vieb_v2/hpc/to_02_timescales.sbatch`, job 47423, #60, #62, #63
