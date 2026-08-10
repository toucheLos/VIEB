# Branch triage

Performed 2026-08-10 as §1g of the `bakeoff` plan. Every branch below has an
`archive/<name>` tag pointing at its tip, created **before** any deletion. To
recover any branch: `git checkout -b <name> archive/<name>`.

Rule going forward: **the only long-lived branch is `main`.** Datasets and
models are never branches — `luna` is a config, `koopman` is a plugin.

## Starting state

6 local branches, 20 remote (excluding `origin/HEAD`). All 6 local branches were
fully contained in `origin/v2_results` and required no salvage.

`main` was fast-forwarded to `origin/v2_results` (`1deb112`) as step 1, since it
held the most recent committed results and everything else is judged relative to
it. The merge was a clean fast-forward — `main` was a strict ancestor.

## Disposition

| branch | ahead of main | disposition | rationale |
|---|---|---|---|
| `v2_results` | — | **merged → archive** | Fast-forwarded into `main`. Now the baseline. |
| `experimental` | 29 | **archive** | Fully contained in `v2_results`. |
| `koopman` | 45 | **archive** | Fully contained in `v2_results`. Code becomes a plugin under `src/vieb/models/koopman_arm.py`. |
| `transfer-operator` | 61 | **archive** | Fully contained in `v2_results`. |
| `v2` | 44 | **archive** | Fully contained in `v2_results`. |
| `luna` | 0 | **archive** | Already merged into `main`. A dataset is a config, not a branch. |
| `claude/feature-extractor-architecture-mzgscx` | 0 | **archive** | Already merged into `main`. |
| `claude/keypoint-bounds-checks-pqhrp2` | 0 | **archive** | Already merged into `main`. |
| `claude/metadata-generator-robustness-a5ljwb` | 0 | **archive** | Already merged into `main`. |
| `claude/vieb-clustering-video-paths-j5qtgy` | 0 | **archive** | Already merged into `main`. |
| `backup-local-pc` | 46 | **salvaged → archive** | See "Salvage" below. Broken merge. |
| `claude/hardcore-mayer-cf7f01` | 1 | **archive** | 51 commits behind. GUI rework superseded by the `views/` tree in `main`. |
| `claude/vieb-h5-csv-integration-4lw72y` | 2 | **archive** | 37 behind. Superseded. |
| `claude/vieb-h5-single-key-0dxtfl` | 1 | **archive** | Superseded — see "Superseded, not lost". |
| `compare_upgrade` | 2 | **archive** | Superseded — feature groups + motifs both shipped in `main`. |
| `ml_shift` | 2 | **archive** | Shares `af5b25c` with `compare_upgrade`. Same disposition. |
| `feature/step0-onboarding` | 1 | **archive** | Single 178-line `user_interface.py` hook. Platform work, out of scope (§8). |
| `dev` | 39 | **archive** | 2 unique commits, both Video Stories UX. Platform work, out of scope (§8). |
| `sample` | 41 | **archive, one bugfix noted** | Bundled 5-video demo project (~30k lines of data). Carries one real unmerged bugfix — see below. |
| `worktree-statistics-methods` | 33 | **salvage deferred → archive** | Genuinely unmerged, and relevant. See below. |

## Salvage

### `backup-local-pc` — the spec's expectation was wrong

The plan predicted "a backup, not a branch. Tag it, delete it." That is not what
it is. It holds the `vieb_v2/transfer/` stage package, which is **absent from
`main`** and is direct groundwork for `ulam-msm` (§4a/§4d):

- `transfer/stage0_degeneracy.py`
- `transfer/stage1_timescales.py` — implied-timescale machinery
- `transfer/stage8_moseq_control.py`
- `transfer/featureio.py`
- `docs/TRANSFER_OPERATOR_FINDINGS.md`, `hpc/stage8_moseq_control.sbatch`

**But it is also a broken merge.** Commit `306f483` ("Merge remote v2_results
into transfer-operator accepting remote changes") was committed with unresolved
`<<<<<<< HEAD` conflict markers in two files. This made the branch appear, to
`git diff --shortstat`, as a clean `+2062 / -0` superset of `v2_results` when it
is nothing of the sort — the "insertions" were largely conflict blocks.

Resolution (commit `ac250ec` on `bakeoff`): took the 7 conflict-free files;
kept `main`'s side of `transfer_operator.py` and `test_transfer_operator.py`,
which is the newer side the merge had intended to accept. 34 existing
transfer-operator tests pass after the salvage.

*Lesson worth keeping: `--shortstat` on a three-dot diff is not evidence a
branch is a superset. Check for conflict markers before trusting an all-additions
diffstat.*

### `worktree-statistics-methods` — salvage deferred, not discarded

Two unmerged commits with no equivalent in `main`:

- `ae12d62` pluggable alternative pose feature representations + validation framework
- `489e47f` feature ablation & dimensionality study harness

with 4 test files, none present in `main`: `test_benchmark_feature_modes.py`,
`test_feature_ablation.py`, `test_feature_representations.py`,
`test_validation_stats.py`.

"Pluggable alternative pose feature representations" is close to what §1d needs
for the two frozen representation artifacts. Salvage is deferred to the
representation task rather than done blind here, so the code lands shaped to the
`RepresentationMeta` contract instead of being dropped in and reworked twice.
**Do not delete `archive/worktree-statistics-methods` until that task closes.**

### `sample` — one unmerged bugfix, out of scope

`15a52ba` "Fix cluster runs not saving: config.json writes hit repo root, not
active project" is a genuine bug, not in `main`. It is platform/GUI code, which
§8 puts out of scope for `bakeoff`. Recorded here so it is not lost:
`git cherry-pick 15a52ba` onto `main` when the platform is next touched.

## Superseded, not lost

The plan flagged four `claude/*` branches as "several look like real fixes". Three
were already merged. The remaining ones were checked by pulling their tests onto
`bakeoff` and running them against `main`. All three failed — **on API shape, not
on missing behaviour**:

| test | expected | `main` actually does |
|---|---|---|
| `test_h5_single_key.py` | `entry["h5_key"] == "coords"` | writes the source-file name (`Coord_3D.rat142...csv`) |
| `test_feature_groups.py` | `report["groups"]["head"]["keypoints"]` | no `keypoints` key in the report schema |
| `test_motifs.py` | — | passes given a populated project; the test is not hermetic |

`main` has all three features; the branches encode an older contract. In the h5
case `main`'s behaviour is the one §1a actually wants, since `concat_h5.py` is
keyed by `source_file`. The tests were dropped rather than carried, because a
test asserting a superseded API is worse than no test. They remain in the tags.

`test_motifs.py` failing for want of a populated `results/` dir is itself a data
point for §1f: the current test suite is not hermetic. The `tests/` tree on
`bakeoff` should run against the small committed fixture instead.

## Environment findings (feed into §1f)

Discovered while trying to run the salvaged tests:

- `venv/` is **Python 3.13** — which `CLAUDE.md` itself declares unsupported
  ("3.10–3.12 required; 3.13+ has compatibility issues") — and is missing
  `seaborn`. It is not a usable reference environment for `requirements.lock`.
- `venv-gpu/` is Python 3.11.4 and healthy (numpy 2.2.6, sklearn 1.9.0), but
  fails with `libpython3.11.so.1.0: cannot open shared object file` unless
  `module load python/3.11.4` precedes activation. This is exactly the failure
  mode §1f describes, reproduced on the login node.
- Available modules: `python/3.11.4`, `python/3.13.0`, `python/3.14.3` (default).
  The default is **not** the version the venv was built against.

`requirements.lock` should be generated from `venv-gpu`, not `venv`.

## Blocked: the remote half

The archive tags exist locally and `main` is merged locally. Pushing is blocked —
this environment has no credentials for `https://github.com/touchelos/vieb.git`
(no credential helper, no `gh`). Run these from an authenticated shell:

```bash
# 1. publish the merged main and every archive tag FIRST
git push origin main
git push origin --tags

# 2. only after confirming the tags are on the remote, delete the branches
for b in backup-local-pc claude/feature-extractor-architecture-mzgscx \
         claude/hardcore-mayer-cf7f01 claude/keypoint-bounds-checks-pqhrp2 \
         claude/metadata-generator-robustness-a5ljwb \
         claude/vieb-clustering-video-paths-j5qtgy \
         claude/vieb-h5-csv-integration-4lw72y claude/vieb-h5-single-key-0dxtfl \
         compare_upgrade dev experimental feature/step0-onboarding koopman luna \
         ml_shift sample transfer-operator v2 v2_results \
         worktree-statistics-methods; do
  git push origin --delete "$b"
done
```

Verify with `git ls-remote --tags origin 'refs/tags/archive/*' | wc -l` → 20
before running the deletion loop.

## Still to do

- Enable branch protection on `main`; set default merge to squash (§1g step 6).
  Requires repo admin access — not doable from this environment.
