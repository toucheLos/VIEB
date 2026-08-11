# Provenance audit of existing label artifacts

Required by §5 of the bakeoff consolidation: audit every saved label file and
`bouts.parquet` for representation/segmenter provenance, and list which downstream
artifacts consumed the demoted null arms. **This is a report. Nothing is acted on.**

Run 2026-08-11 against `~/vieb2-results`, `~/exbias`, `~/moseq`.

---

## 1. What provenance exists

| store | label artifacts | `run_manifest.json` | `bouts.parquet` |
|---|---|---|---|
| `~/vieb2-results` | 13 `*labels.npz` | **0** | **0** |
| `~/exbias/runs` | 2 `segments.npz` | 2 | 2 |
| `~/moseq/luna_demo` | 2 x 3,846 csv | 0 | 0 |

Only ExBias ever wrote VUS-1, and it predates the two-slot schema
(`method_name`/`method_version`, no representation field). `vieb.io.vus1` reads
that spelling, so its two runs stay comparable without being rewritten.

## 2. Arm identity of every saved label file

`frames` is the label array length. Note the two families are on **different
denominators**: HDBSCAN loses ~39,400 frames to the delay-embedding span, and both
predate the h5/csv dedup (#59), so neither is on the true 22,355,989.

| artifact | representation | segmenter | frames | recoverable from |
|---|---|---|---|---|
| `koopman_pca/labels.npz` | `pca` | `hdbscan` | 28,586,707 | dir name **+ `meta.latent_method`** |
| `koopman_diffusion/labels.npz` | `diffusion` | `hdbscan` | 28,586,707 | dir name **+ `meta.latent_method`** |
| `koopman_pca/koopman_labels.npz` | `pca` | `koopman` | 28,626,107 | **dir name only** |
| `koopman_diffusion/koopman_labels.npz` | `diffusion` | `koopman` | 28,626,107 | **dir name only** |
| `koopman_{pca,diffusion}_r{12,24,96,192}/koopman_labels.npz` (8 files) | per dir | `koopman` | 28,626,107 | **dir name only** |
| `koopman_smoke/koopman_labels.npz` | ? | `koopman` | 115,196 | **dir name only** |

**The sharpest finding: the Koopman files do not record their representation at
all.** Only the HDBSCAN `labels.npz` carry `meta.latent_method`. For the ten
`koopman_labels.npz` files, the *only* thing that says whether a run was `pca` or
`diffusion` is the directory it sits in — and the directory is also the only thing
that says the segmenter was Koopman rather than HDBSCAN, since both live in the
same directory under different filenames.

Renaming a directory silently rewrites the provenance of 10 artifacts.

## 3. Downstream artifacts that consumed the demoted arms

`pca-HDBSCAN` (effect exactly 0.000, 99.2% of clustered frames in one state) and
`diffusion-HDBSCAN` (35/37 "significant" states on a 96% state moving two points —
a power artifact) appear in:

| artifact | what it is |
|---|---|
| `~/vieb2-results/koopman_comparison.json` | the four-arm geometric head-to-head |
| `~/vieb2-results/_report/model_comparison.json` | collected per-arm metrics |
| `~/vieb2-results/_report/discrimination.json` | the MoSeq-axis contrasts |
| `~/vieb2-results/_report/discrimination_trunc5381.json` | truncated re-run |
| `~/vieb2-results/_report/ranking.json` | **the published ranking** |
| `~/vieb2-results/_report/figures/*.png` (6) | every published figure |
| `~/vieb2-results/_report/v2_model_comparison.html` | the rendered report |
| `docs/V2_MODEL_COMPARISON.md` | the write-up, and decision #65 |

Both arms are *scored* in all of these, which is correct and should stay: a
measured null is part of the result, and `pca-HDBSCAN`'s 0.000 is one of the more
informative numbers in the comparison. Nothing downstream treats either arm's
labels as ground truth, and no per-animal scalar, cohort table or clip set was
built from them.

**So there is nothing to retract.** The demotion is about defaults, not artifacts.

## 4. What was demoted

`pca` is no longer any pipeline's default representation. Both HDBSCAN arms remain
registered and runnable — `REPRESENTATIONS["pca"]` x `SEGMENTERS["hdbscan"]` — and
are expected to appear in the comparison as measured nulls.

One correction to the §5 premise: `pca-HDBSCAN` was **never** `compare.py
--cluster`'s default. `compare.py:1625` sets `use_pca=False` literally; v1's
default path is standardize -> UMAP-10 -> HDBSCAN on 91 engineered features, which
is the `vieb_v1` arm. PCA is `vieb_v2`'s default (`pipeline.make_latent`). The
claim in `V2_MODEL_COMPARISON.md` §8 that it "is the v2 pipeline's default path
(`compare.py --cluster`)" conflates the two trees.

## 5. What this changes going forward

`vieb.io.vus1.RunManifest` records `representation` and `segmenter` as separate
fields, plus `repr_hash`, `config_hash`, `git_sha` and `git_dirty`. An artifact
written through `vieb.compare.run_arm` is attributable without reading its path.
The 13 files above are not, and cannot be made so retroactively — the
directory-name mapping in §2 is the record.
