"""feature_ablation.py — Feature ablation & dimensionality study.

Tests the hypothesis that VIEB's clustering conflates behaviorally distinct
states because it has TOO MANY features (curse of dimensionality flattens
density contrast), not too few. Finds the smallest high-signal feature
subset per project by re-running the FULL clustering pipeline
(standardize -> UMAP -> HDBSCAN, unchanged) on column-masked subsets of the
already-extracted feature matrix — NEVER re-extracting from raw pose, NEVER
adding new feature families, NEVER pooling two projects together.

Runs against whichever project is currently active (project_manager's
active-project mechanism, same as benchmark_feature_modes.py). Run it once
per project (switch the active project between runs) — Luna and Spence may
have different optimal feature sets, which is a valid, expected outcome.

Usage
-----
    python feature_ablation.py                    # full study on active project
    python feature_ablation.py --study leave_one_out
    python feature_ablation.py --max-frames 200000 --n-boot 500

Output: results/ablation/feature_ablation_<project>.csv — one row per
feature subset, columns: subset, n_features, dbcv, repeatability_R,
R_ci_low, R_ci_high, ari_stability, modularity_Q, noise_frac, n_states.

This script does NOT declare a winner and does NOT change the production
default feature set. It produces evidence for a human decision (see
docs/FEATURE_ABLATION_FINDINGS.md).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import vieb_config as _vc


# ---------------------------------------------------------------------------
# Feature family classification (drives off index.json _meta.feature_names)
# ---------------------------------------------------------------------------

# Ablatable families. Names that removal-candidate per the study hypothesis
# are noted in docs/FEATURE_ABLATION_FINDINGS.md, not encoded here.
FAMILY_ORDER = [
    "per_keypoint_speed",
    "pairwise_distances",
    "centroid_speed",
    "body_orientation",
    "elongation",
    "angular_velocity",
    "movement_entropy",
    "semantic",
    "temporal_window_stats",
    "wavelets",
]

_LITERAL_FAMILY = {
    "centroid_speed": "centroid_speed",
    "body_orientation": "body_orientation",
    "elongation": "elongation",
    "angular_velocity": "angular_velocity",
    "movement_entropy": "movement_entropy",
    "rearing_score": "semantic",
    "head_angle": "semantic",
}


def classify_families(feature_names: list[str]) -> dict[str, list[int]]:
    """Map each feature column index to its family.

    Uses the ``_window``-suffix-first rule to avoid the naming collisions in
    ``ml/feature_extraction.py``'s ``get_feature_names()``:
      - ``speed_kp*`` (per-keypoint) vs ``speed_*_window`` (temporal)
      - ``dist_pair*`` (pairwise) vs ``dist_*_window`` (temporal)
      - ``angular_velocity`` (scalar) vs ``angular_vel_*_window`` (temporal)
    All 8 temporal names uniquely end in ``_window`` and no other family
    does, so classifying the temporal family first is unambiguous.
    """
    families: dict[str, list[int]] = {f: [] for f in FAMILY_ORDER}
    for i, name in enumerate(feature_names):
        if name.endswith("_window"):
            families["temporal_window_stats"].append(i)
        elif name.startswith("speed_kp"):
            families["per_keypoint_speed"].append(i)
        elif name.startswith("dist_pair"):
            families["pairwise_distances"].append(i)
        elif name.startswith("wavelet_kp"):
            families["wavelets"].append(i)
        elif name in _LITERAL_FAMILY:
            families[_LITERAL_FAMILY[name]].append(i)
        else:
            raise ValueError(
                f"Unclassifiable feature name {name!r} at column {i}. "
                f"Update classify_families() if a new family was added."
            )
    return {f: idx for f, idx in families.items() if idx}


# ---------------------------------------------------------------------------
# Data loading (per-project, never pooled across projects)
# ---------------------------------------------------------------------------

def _features_dir(feature_mode: str = "default") -> str:
    base = os.path.join(_vc.get_results_dir(), "features")
    return base if feature_mode == "default" else os.path.join(base, feature_mode)


def load_project_features(feature_mode: str = "default"):
    """Load per-video feature matrices for the active project.

    Returns (pooled, boundaries, feature_names, stems) where pooled is
    (total_frames, F), boundaries maps stem -> (start, end) row slice, and
    feature_names comes from index.json _meta (authoritative column order).
    """
    features_dir = _features_dir(feature_mode)
    index_path = os.path.join(features_dir, "index.json")
    if not os.path.exists(index_path):
        sys.exit(f"[ERROR] {index_path} not found. Run: python compare.py --extract"
                 + ("" if feature_mode == "default" else f" --feature-mode {feature_mode}"))
    with open(index_path) as f:
        index = json.load(f)

    meta = index.get("_meta", {})
    feature_names = meta.get("feature_names")

    stems = sorted(k for k in index.keys() if k != "_meta")
    blocks, boundaries, cursor = [], {}, 0
    for stem in stems:
        fp = index[stem].get("features_path", "")
        if not fp or not os.path.exists(fp):
            # try relative to features_dir
            alt = os.path.join(features_dir, f"{stem}_features.npy")
            fp = alt if os.path.exists(alt) else fp
        if not fp or not os.path.exists(fp):
            continue
        arr = np.load(fp).astype(np.float64)
        blocks.append(arr)
        boundaries[stem] = (cursor, cursor + len(arr))
        cursor += len(arr)

    if not blocks:
        sys.exit("[ERROR] No feature files could be loaded.")
    pooled = np.vstack(blocks)

    if not feature_names or len(feature_names) != pooled.shape[1]:
        # Fall back to positional names only if _meta is missing/mismatched.
        print(f"[warn] index.json _meta.feature_names missing or mismatched "
              f"({len(feature_names) if feature_names else 0} names vs "
              f"{pooled.shape[1]} columns) — family masking may be unreliable.")
        feature_names = [f"feat_{i}" for i in range(pooled.shape[1])]

    return pooled, boundaries, list(feature_names), list(boundaries.keys())


def load_metadata(stems: list[str]) -> pd.DataFrame:
    """Load metadata.csv (animal_id, day, ...) keyed by stem, for repeatability R."""
    meta_path = _vc.get_metadata_path()
    if not os.path.exists(meta_path):
        return pd.DataFrame({"stem": stems})
    meta = pd.read_csv(meta_path)
    meta = _vc.normalize_metadata_columns(meta)
    return meta


# ---------------------------------------------------------------------------
# Clustering a subset (mirrors compare.py cmd_cluster, in-memory)
# ---------------------------------------------------------------------------

class AblationConfig:
    def __init__(self, umap_dims=10, min_cluster_size=50, min_samples=None,
                 fit_sample=200_000, dbcv_sample=20_000, seed=42):
        self.umap_dims = umap_dims
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples if min_samples else max(10, min(100, min_cluster_size // 10))
        self.fit_sample = fit_sample
        self.dbcv_sample = dbcv_sample
        self.seed = seed


def _standardize(matrix: np.ndarray, audit: bool = True) -> np.ndarray:
    """Standardize with the SAME preprocessor the production pipeline uses,
    and (Part A) assert every column was genuinely standardized."""
    from ml import BehaviorPreprocessor
    pre = BehaviorPreprocessor(use_pca=False)
    scaled = pre.fit_transform(matrix)
    if audit:
        # Part A standardization audit, enforced in code: every column must be
        # ~zero-mean / unit-variance, or a handled zero-variance column (all 0).
        means = np.abs(scaled.mean(axis=0))
        stds = scaled.std(axis=0)
        ok = (means < 0.5) & ((np.abs(stds - 1.0) < 0.5) | (stds < 1e-6))
        if not ok.all():
            bad = np.where(~ok)[0][:5]
            raise AssertionError(
                f"Standardization audit FAILED for columns {bad.tolist()} "
                f"(mean={means[bad].round(3).tolist()}, std={stds[bad].round(3).tolist()}). "
                f"A feature bypassed standardization — clustering would be dominated by it."
            )
    return scaled


def _fit_umap_hdbscan(scaled: np.ndarray, config: AblationConfig):
    """Fit UMAP + HDBSCAN on a subset matrix; return (embedding_all, labels_all)."""
    import umap as umap_lib
    import hdbscan as hdbscan_lib

    rng = np.random.default_rng(config.seed)
    n = len(scaled)

    reducer = umap_lib.UMAP(n_components=config.umap_dims, n_neighbors=30,
                            min_dist=0.0, random_state=config.seed, low_memory=True,
                            verbose=False)
    if n > config.fit_sample:
        fit_idx = np.sort(rng.choice(n, config.fit_sample, replace=False))
        reducer.fit(scaled[fit_idx])
    else:
        reducer.fit(scaled)
    embedding = reducer.transform(scaled)

    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=config.min_cluster_size,
        min_samples=config.min_samples,
        cluster_selection_method="eom",
        prediction_data=True,
    )
    if n > config.fit_sample:
        fit_idx = np.sort(rng.choice(n, config.fit_sample, replace=False))
        clusterer.fit(embedding[fit_idx])
        labels, _ = hdbscan_lib.approximate_predict(clusterer, embedding)
    else:
        clusterer.fit(embedding)
        labels = clusterer.labels_

    return embedding, np.asarray(labels)


# ---------------------------------------------------------------------------
# Metrics for one subset
# ---------------------------------------------------------------------------

def _occupancy_df(labels_all, boundaries, metadata, n_states):
    """Per-video state-occupancy fractions joined with metadata (for R)."""
    state_cols = [f"state_{k}_frac" for k in range(n_states)]
    rows = []
    for stem, (a, b) in boundaries.items():
        lbl = labels_all[a:b]
        row = {"stem": stem}
        for k in range(n_states):
            row[f"state_{k}_frac"] = float((lbl == k).mean())
        rows.append(row)
    occ = pd.DataFrame(rows)
    if "stem" in metadata.columns:
        occ = occ.merge(metadata, on="stem", how="left")
    return occ, state_cols


def _transition_counts(labels_all, boundaries, n_states):
    """Aggregate raw transition counts across all videos (for modularity)."""
    counts = np.zeros((n_states, n_states), dtype=np.float64)
    for stem, (a, b) in boundaries.items():
        lbl = labels_all[a:b]
        x, y = lbl[:-1], lbl[1:]
        valid = (x >= 0) & (y >= 0)
        for xi, yi in zip(x[valid], y[valid]):
            counts[xi, yi] += 1
    return counts


def evaluate_subset(name, col_indices, pooled, boundaries, metadata, config, n_boot=200):
    """Run the full pipeline on a column subset and return a metric row."""
    from ml.validation_stats import (
        compute_dbcv, compute_ari_stability,
        compute_repeatability_R, compute_transition_modularity,
    )

    subset = pooled[:, col_indices]
    scaled = _standardize(subset, audit=True)
    embedding, labels = _fit_umap_hdbscan(scaled, config)

    n_states = int(len(np.unique(labels[labels >= 0])))
    noise_frac = float((labels == -1).mean())

    # DBCV on a capped sample of the embedding (validity_index is ~O(n^2)).
    rng = np.random.default_rng(config.seed)
    if len(embedding) > config.dbcv_sample:
        di = np.sort(rng.choice(len(embedding), config.dbcv_sample, replace=False))
        dbcv = compute_dbcv(embedding[di], labels[di])
    else:
        dbcv = compute_dbcv(embedding, labels)

    # Repeatability R (+ bootstrap CI) from per-video occupancy.
    occ, state_cols = _occupancy_df(labels, boundaries, metadata, max(n_states, 1))
    rep = compute_repeatability_R(occ, state_cols, n_boot=n_boot)

    # Transition modularity from sequential per-video labels.
    mod = compute_transition_modularity(
        _transition_counts(labels, boundaries, max(n_states, 1)),
        state_ids=list(range(max(n_states, 1))),
    )

    # ARI stability: re-cluster row subsamples of the standardized subset.
    def _recluster(idx):
        _, lab = _fit_umap_hdbscan(scaled[idx], config)
        return lab
    ari = compute_ari_stability(_recluster, n_samples=len(scaled),
                                n_runs=5, subsample=0.8, seed=config.seed)

    return {
        "subset": name,
        "n_features": len(col_indices),
        "dbcv": dbcv.get("dbcv"),
        "repeatability_R": rep.get("mean_R"),
        "R_ci_low": rep.get("R_ci_low"),
        "R_ci_high": rep.get("R_ci_high"),
        "ari_stability": ari.get("ari_stability"),
        "modularity_Q": mod.get("modularity_Q"),
        "noise_frac": round(noise_frac, 4),
        "n_states": n_states,
    }


# ---------------------------------------------------------------------------
# Studies (C1 baseline, C2 leave-one-out, C3 cumulative greedy)
# ---------------------------------------------------------------------------

def _all_indices(families: dict[str, list[int]]) -> list[int]:
    return sorted(i for idx in families.values() for i in idx)


def _score(row: dict) -> float:
    """Aggregate quality score for greedy selection: reward DBCV, R, ARI
    stability; penalize noise. All optional metrics default to 0 when they
    skipped. Deliberately simple and transparent — the report interprets;
    this only orders the greedy search."""
    dbcv = row.get("dbcv") or 0.0
    r = row.get("repeatability_R") or 0.0
    ari = row.get("ari_stability") or 0.0
    noise = row.get("noise_frac") or 0.0
    return dbcv + r + ari - noise


def study_baseline(families, pooled, boundaries, metadata, config, n_boot):
    cols = _all_indices(families)
    return [evaluate_subset("all_features", cols, pooled, boundaries, metadata, config, n_boot)]


def study_leave_one_out(families, pooled, boundaries, metadata, config, n_boot):
    rows = []
    all_cols = _all_indices(families)
    rows.append(evaluate_subset("all_features", all_cols, pooled, boundaries, metadata, config, n_boot))
    for fam in [f for f in FAMILY_ORDER if f in families]:
        remaining = sorted(set(all_cols) - set(families[fam]))
        if not remaining:
            continue
        rows.append(evaluate_subset(f"minus_{fam}", remaining, pooled, boundaries, metadata, config, n_boot))
    return rows


def study_cumulative(families, pooled, boundaries, metadata, config, n_boot):
    """Greedy build-up: start from the single strongest family, add families
    one at a time, keeping an addition only if it improves the score."""
    rows = []
    present = [f for f in FAMILY_ORDER if f in families]

    # Score each single family alone to pick the strongest starting point.
    singles = {}
    for fam in present:
        r = evaluate_subset(f"only_{fam}", families[fam], pooled, boundaries, metadata, config, n_boot)
        rows.append(r)
        singles[fam] = _score(r)
    chosen = [max(singles, key=singles.get)]
    best_score = singles[chosen[0]]
    best_cols = list(families[chosen[0]])

    remaining = [f for f in present if f not in chosen]
    improved = True
    while improved and remaining:
        improved = False
        trial_scores = {}
        for fam in remaining:
            cols = sorted(set(best_cols) | set(families[fam]))
            r = evaluate_subset("cumulative_" + "+".join(chosen + [fam]), cols,
                                pooled, boundaries, metadata, config, n_boot)
            rows.append(r)
            trial_scores[(fam, tuple(cols))] = (_score(r), r)
        # pick the best-improving addition, if any
        best_add = max(trial_scores.items(), key=lambda kv: kv[1][0])
        (fam, cols), (score, _row) = best_add
        if score > best_score:
            best_score = score
            best_cols = list(cols)
            chosen.append(fam)
            remaining = [f for f in remaining if f != fam]
            improved = True
    return rows


# ---------------------------------------------------------------------------
# Part D — shape-space as a replacement candidate
# ---------------------------------------------------------------------------

def study_shape_space(pooled_default, boundaries_default, families_default,
                      metadata, config, n_boot):
    """Add shape_space and shape_space+per_keypoint_speed rows IF the
    shape_space extraction exists (run: compare.py --extract --feature-mode
    shape_space first). Evaluated as REPLACEMENTS, not additions."""
    ss_dir = _features_dir("shape_space")
    if not os.path.exists(os.path.join(ss_dir, "index.json")):
        print("[skip] shape_space rows — results/features/shape_space/ not found. "
              "Run: python compare.py --extract --feature-mode shape_space")
        return []

    pooled_ss, boundaries_ss, names_ss, stems_ss = load_project_features("shape_space")
    rows = []
    ss_cols = list(range(pooled_ss.shape[1]))
    rows.append(evaluate_subset("shape_space", ss_cols, pooled_ss, boundaries_ss,
                                metadata, config, n_boot))

    # shape_space + per_keypoint_speed: align per stem (same frames per video).
    speed_cols = families_default.get("per_keypoint_speed", [])
    common = [s for s in boundaries_ss if s in boundaries_default]
    if speed_cols and common:
        blocks_ss, blocks_speed, new_bounds, cur = [], [], {}, 0
        aligned = True
        for stem in common:
            a_ss, b_ss = boundaries_ss[stem]
            a_d, b_d = boundaries_default[stem]
            if (b_ss - a_ss) != (b_d - a_d):
                aligned = False
                break
            blocks_ss.append(pooled_ss[a_ss:b_ss])
            blocks_speed.append(pooled_default[a_d:b_d][:, speed_cols])
            new_bounds[stem] = (cur, cur + (b_ss - a_ss))
            cur += (b_ss - a_ss)
        if aligned and blocks_ss:
            combined = np.hstack([np.vstack(blocks_ss), np.vstack(blocks_speed)])
            cols = list(range(combined.shape[1]))
            rows.append(evaluate_subset("shape_space+per_keypoint_speed", cols,
                                        combined, new_bounds, metadata, config, n_boot))
        else:
            print("[skip] shape_space+per_keypoint_speed — per-video frame counts "
                  "differ between default and shape_space extractions.")
    return rows


# ---------------------------------------------------------------------------
# Part C4 — grooming/freezing separability (best-effort, conditional)
# ---------------------------------------------------------------------------

def label_separation_note() -> str:
    """Report whether human clip labels exist to check state separation.
    The full clip->frame remap is done in the findings doc's workflow; here
    we only detect availability so the report can state it honestly."""
    ann = os.path.join(_vc.get_results_dir(), "annotations", "annotations.csv")
    if os.path.exists(ann):
        return f"annotations found ({ann}) — see FEATURE_ABLATION_FINDINGS.md for the separation check."
    return "no results/annotations/annotations.csv — grooming/freezing separation check skipped."


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _project_name() -> str:
    try:
        import project_manager as _pm
        return _pm.get_active_project(ROOT).name
    except Exception:
        return "active_project"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", choices=["all", "baseline", "leave_one_out", "cumulative", "shape_space"],
                        default="all", help="Which study to run (default: all)")
    parser.add_argument("--umap-dims", type=int, default=10)
    parser.add_argument("--min-cluster-size", type=int, default=50)
    parser.add_argument("--min-samples", type=int, default=None)
    parser.add_argument("--fit-sample", type=int, default=200_000,
                        help="Max frames to fit UMAP/HDBSCAN on (rest assigned); matches production caps")
    parser.add_argument("--n-boot", type=int, default=200,
                        help="Bootstrap resamples for the repeatability R confidence interval")
    args = parser.parse_args()

    config = AblationConfig(umap_dims=args.umap_dims, min_cluster_size=args.min_cluster_size,
                            min_samples=args.min_samples, fit_sample=args.fit_sample)

    project = _project_name()
    print(f"=== Feature ablation study — project: {project} ===")
    print(label_separation_note())

    pooled, boundaries, feature_names, stems = load_project_features("default")
    families = classify_families(feature_names)
    metadata = load_metadata(stems)
    print(f"Loaded {pooled.shape[0]:,} frames × {pooled.shape[1]} features from {len(boundaries)} videos.")
    print("Feature families: " + ", ".join(f"{f}({len(idx)})" for f, idx in families.items()))

    rows: list[dict] = []
    if args.study in ("all", "baseline"):
        rows += study_baseline(families, pooled, boundaries, metadata, config, args.n_boot)
    if args.study in ("all", "leave_one_out"):
        rows += study_leave_one_out(families, pooled, boundaries, metadata, config, args.n_boot)
    if args.study in ("all", "cumulative"):
        rows += study_cumulative(families, pooled, boundaries, metadata, config, args.n_boot)
    if args.study in ("all", "shape_space"):
        rows += study_shape_space(pooled, boundaries, families, metadata, config, args.n_boot)

    # De-duplicate identical subset names, keeping the last.
    df = pd.DataFrame(rows).drop_duplicates(subset="subset", keep="last")

    out_dir = os.path.join(_vc.get_results_dir(), "ablation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"feature_ablation_{project}.csv")
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        existing = existing[~existing["subset"].isin(df["subset"])]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(out_path, index=False)

    print(f"\n=== Ablation table (results/ablation/feature_ablation_{project}.csv) ===")
    print(df.to_string(index=False))
    print("\nNo winner declared — review the table and see "
          "docs/FEATURE_ABLATION_FINDINGS.md to record the per-project recommendation.")


if __name__ == "__main__":
    main()
