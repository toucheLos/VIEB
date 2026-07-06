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
