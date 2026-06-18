#!/usr/bin/env python3
"""
feature_reduction_test.py — Test whether locomotion emerges as a clean cluster
when using minimal feature subsets.

Usage
-----
    python feature_reduction_test.py

Samples 50k frames from results/features/, tests 5 feature subsets,
and reports clustering quality per subset.
"""

import json
import os
import sys
import time

import numpy as np
import pandas as pd

RESULTS_DIR = os.path.join("results", "feature_reduction")
FEATURES_DIR = os.path.join("results", "features")
N_SAMPLE = 50_000
HDBSCAN_MIN_CLUSTER = 500
HDBSCAN_MIN_SAMPLES = 5
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Feature name constants — loaded from index.json when available, with
# hardcoded K=8 fallbacks for backward compatibility.
# ---------------------------------------------------------------------------
SPEED_KP = [f"speed_kp{i}" for i in range(8)]
DIST_PAIRS = [f"dist_pair{i}" for i in range(28)]
UNIVERSAL_SCALAR_FEATS = [
    "centroid_speed", "body_orientation", "elongation", "angular_velocity",
    "movement_entropy",
]
SEMANTIC_SCALAR_FEATS = ["rearing_score", "head_angle"]
SCALAR_FEATS = UNIVERSAL_SCALAR_FEATS + SEMANTIC_SCALAR_FEATS
# Temporal window features (indices after scalars in the flat array)
TEMPORAL_FEATS = [
    "speed_mean_window", "speed_std_window", "speed_max_window", "speed_p90_window",
    "dist_mean_window", "dist_std_window", "angle_mean_window", "angle_std_window",
    "entropy_mean_window", "accel_mean_window",
]

ALL_NAMES_51 = SPEED_KP + DIST_PAIRS + SCALAR_FEATS + TEMPORAL_FEATS
# Wavelet names (40 = 8 kp × 5 freqs) — appended if use_wavelets=True
WAVELET_NAMES = [f"wavelet_kp{k}_f{f}" for k in range(8) for f in range(5)]
ALL_NAMES_91 = ALL_NAMES_51 + WAVELET_NAMES


def _load_feature_names_from_index():
    """Load authoritative feature names from index.json if available."""
    idx_path = os.path.join(FEATURES_DIR, "index.json")
    if not os.path.exists(idx_path):
        return None
    try:
        with open(idx_path) as f:
            meta = json.load(f).get("_meta", {})
        names = meta.get("feature_names", [])
        return names if names else None
    except Exception:
        return None


def _name_to_idx(names_in_subset, all_names):
    """Return column indices for the requested feature names."""
    name_to_i = {n: i for i, n in enumerate(all_names)}
    missing = [n for n in names_in_subset if n not in name_to_i]
    if missing:
        print(f"  [WARN] Feature names not found (skipped): {missing}")
    return [name_to_i[n] for n in names_in_subset if n in name_to_i]


def load_sample(n=N_SAMPLE):
    """Load up to n frames sampled uniformly from all feature files."""
    idx_path = os.path.join(FEATURES_DIR, "index.json")
    if not os.path.exists(idx_path):
        sys.exit("[ERROR] results/features/index.json not found. Run: python compare.py --extract")
    with open(idx_path) as f:
        index = json.load(f)
    if not index:
        sys.exit("[ERROR] Feature index is empty.")

    print(f"Loading up to {n:,} frames from {len(index)} videos...")
    all_feat = []
    for stem, info in index.items():
        fp = info.get("features_path", "").replace("\\", "/")
        if not os.path.exists(fp):
            continue
        arr = np.load(fp)
        all_feat.append(arr)

    pooled = np.vstack(all_feat).astype(np.float32)
    print(f"  Total frames available: {pooled.shape[0]:,}  features: {pooled.shape[1]}")

    n_total = pooled.shape[0]
    if n_total > n:
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(n_total, n, replace=False)
        pooled = pooled[np.sort(idx)]
    print(f"  Sampled: {pooled.shape[0]:,} frames")
    return pooled


def build_subsets(n_feats):
    """Return list of (name, indices) pairs for each feature subset."""
    index_names = _load_feature_names_from_index()
    if index_names and len(index_names) == n_feats:
        all_names = index_names
    else:
        all_names = ALL_NAMES_91 if n_feats >= 91 else ALL_NAMES_51

    subset_1_names = ["centroid_speed", "elongation", "angular_velocity", "rearing_score"]
    subset_2_names = subset_1_names + [
        "speed_kp2", "speed_kp3", "speed_kp6", "body_orientation",
    ]
    subset_3_names = subset_2_names + SPEED_KP + TEMPORAL_FEATS
    subset_4_names = ALL_NAMES_51
    subset_5_names = ALL_NAMES_91

    return [
        ("subset_1 (4 core)",      subset_1_names),
        ("subset_2 (8 kinematics)", subset_2_names),
        ("subset_3 (~20 features)", subset_3_names),
        ("subset_4 (51 no-wavelet)", subset_4_names),
        ("subset_5 (91 full)",      subset_5_names),
    ]


def run_subset(name, col_indices, data, out_dir):
    """Standardize, UMAP-reduce, HDBSCAN-cluster, score, and save plot."""
    from sklearn.preprocessing import StandardScaler
    import umap as umap_lib
    import hdbscan as hdbscan_lib
    from sklearn.metrics import silhouette_score

    subset_data = data[:, col_indices].astype(np.float64)
    n_feat = subset_data.shape[1]

    # Standardize
    scaler = StandardScaler()
    scaled = scaler.fit_transform(subset_data)

    # UMAP: 2D for vis, 3D for clustering
    t0 = time.time()
    reducer_2d = umap_lib.UMAP(n_components=2, n_neighbors=30, min_dist=0.0,
                                random_state=RANDOM_SEED, low_memory=True, verbose=False)
    emb_2d = reducer_2d.fit_transform(scaled)

    reducer_3d = umap_lib.UMAP(n_components=3, n_neighbors=30, min_dist=0.0,
                                random_state=RANDOM_SEED, low_memory=True, verbose=False)
    emb_3d = reducer_3d.fit_transform(scaled)
    umap_time = time.time() - t0

    # HDBSCAN on 3D embedding
    clusterer = hdbscan_lib.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_method="eom",
    )
    clusterer.fit(emb_3d)
    labels = clusterer.labels_

    n_clusters = int(len(np.unique(labels[labels >= 0])))
    noise_frac = float((labels == -1).mean())

    if n_clusters == 0:
        return {"name": name, "n_features": n_feat, "n_clusters": 0,
                "dominant_frac": float("nan"), "noise_frac": noise_frac,
                "silhouette": float("nan"), "loco_speed": float("nan"), "loco_frac": float("nan")}

    # Dominant state fraction
    valid = labels >= 0
    if valid.sum() > 0:
        counts = np.bincount(labels[valid])
        dominant_frac = float(counts.max() / valid.sum())
    else:
        dominant_frac = float("nan")

    # Silhouette on valid frames (subsample for speed)
    sil = float("nan")
    if n_clusters >= 2 and valid.sum() > 0:
        try:
            n_sil = min(10_000, int(valid.sum()))
            rng = np.random.default_rng(RANDOM_SEED)
            sil_idx = rng.choice(np.where(valid)[0], n_sil, replace=False)
            sil = float(silhouette_score(emb_3d[sil_idx], labels[sil_idx], sample_size=None))
        except Exception:
            sil = float("nan")

    # Locomotion cluster: highest mean centroid_speed
    speed_col_local = None
    for i_global, col_i in enumerate(col_indices):
        if col_i == (ALL_NAMES_91.index("centroid_speed") if len(col_indices) <= len(ALL_NAMES_91)
                     else ALL_NAMES_51.index("centroid_speed")):
            speed_col_local = i_global
            break
    # Simpler: look up centroid_speed index in ALL_NAMES_51 (index 36)
    # The col_indices map into the original data columns, so we need original idx
    # centroid_speed is always at index 36 in the 51-feat array
    CENTROID_SPEED_IDX = 36
    loco_speed = float("nan")
    loco_frac = float("nan")
    if CENTROID_SPEED_IDX in col_indices and n_clusters > 0:
        speed_col_in_subset = col_indices.index(CENTROID_SPEED_IDX)
        cluster_speeds = []
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                cluster_speeds.append((k, float(subset_data[mask, speed_col_in_subset].mean())))
        if cluster_speeds:
            loco_k, loco_speed = max(cluster_speeds, key=lambda x: x[1])
            loco_frac = float((labels == loco_k).sum() / len(labels))

    # Save UMAP scatter plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "")
        fig, ax = plt.subplots(figsize=(8, 6))
        unique_labels = sorted(set(labels))
        cmap = plt.cm.tab20(np.linspace(0, 1, max(20, len(unique_labels))))
        for i, k in enumerate(unique_labels):
            mask = labels == k
            color = "#cccccc" if k == -1 else cmap[i % 20]
            alpha = 0.2 if k == -1 else 0.6
            size = 1 if k == -1 else 2
            ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                       c=[color], s=size, alpha=alpha, linewidths=0,
                       label=f"State {k}" if k >= 0 else "Noise")
        ax.set_title(f"{name}\n{n_feat} features, {n_clusters} clusters, sil={sil:.2f}", fontsize=10)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.axis("off")
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{safe_name}_umap.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"    Saved: {out_path}")
    except Exception as e:
        print(f"    [WARN] Could not save plot: {e}")

    print(f"    UMAP fit time: {umap_time:.1f}s | clusters={n_clusters} | "
          f"dom={dominant_frac*100:.1f}% | noise={noise_frac*100:.1f}% | sil={sil:.3f}")

    return {
        "name": name, "n_features": n_feat, "n_clusters": n_clusters,
        "dominant_frac": dominant_frac, "noise_frac": noise_frac,
        "silhouette": sil, "loco_speed": loco_speed, "loco_frac": loco_frac,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    pooled = load_sample(N_SAMPLE)
    n_feats = pooled.shape[1]
    print(f"\nFeature vector size: {n_feats} (91 = with wavelets, 51 = without)")

    subsets = build_subsets(n_feats)
    results = []
    t_start = time.time()

    for i, (name, feat_names) in enumerate(subsets):
        col_indices = _name_to_idx(feat_names, ALL_NAMES_91 if n_feats >= 91 else ALL_NAMES_51)
        if not col_indices:
            print(f"\n[SKIP] {name}: no valid feature indices")
            continue
        # Restrict to features that exist in this data
        col_indices = [c for c in col_indices if c < n_feats]
        if not col_indices:
            print(f"\n[SKIP] {name}: features exceed data width ({n_feats})")
            continue

        print(f"\n[{i+1}/{len(subsets)}] {name} ({len(col_indices)} features)...")
        try:
            res = run_subset(name, col_indices, pooled, RESULTS_DIR)
            results.append(res)
        except Exception as e:
            print(f"  [ERROR] {e}")
            results.append({"name": name, "n_features": len(col_indices), "n_clusters": 0,
                            "dominant_frac": float("nan"), "noise_frac": float("nan"),
                            "silhouette": float("nan"), "loco_speed": float("nan"),
                            "loco_frac": float("nan")})

        elapsed = time.time() - t_start
        if elapsed > 270:
            print(f"\n[WARN] Elapsed {elapsed:.0f}s — stopping early to stay under 5 min limit.")
            break

    # ---- Comparison table ----
    print(f"\n{'='*75}")
    print(f"{'Subset':<30} | {'N feat':>6} | {'Clusters':>8} | {'Dom%':>5} | {'Noise%':>6} | {'Sil':>6}")
    print(f"{'-'*75}")
    for r in results:
        dom  = f"{r['dominant_frac']*100:.1f}" if not np.isnan(r['dominant_frac'] or float('nan')) else "-"
        noi  = f"{r['noise_frac']*100:.1f}" if not np.isnan(r['noise_frac'] or float('nan')) else "-"
        sil  = f"{r['silhouette']:.3f}" if not np.isnan(r['silhouette'] or float('nan')) else "-"
        print(f"{r['name']:<30} | {r['n_features']:>6} | {r['n_clusters']:>8} | {dom:>5} | {noi:>6} | {sil:>6}")

    # ---- Recommendation ----
    valid = [r for r in results if r["n_clusters"] >= 8 and not np.isnan(r.get("silhouette") or float("nan"))]
    if valid:
        best = max(valid, key=lambda r: (r["silhouette"], -r["noise_frac"]))
        print(f"\nRecommended subset: {best['name']}")
        print(f"  n_clusters={best['n_clusters']}, sil={best['silhouette']:.3f}, noise={best['noise_frac']*100:.1f}%")
    else:
        print("\nNo subset met the n_clusters >= 8 criterion with valid silhouette.")

    # ---- Locomotion-like cluster ----
    print(f"\nLocomotion-like cluster (highest mean centroid_speed):")
    for r in results:
        if not np.isnan(r.get("loco_speed") or float("nan")):
            print(f"  {r['name']:<30} speed={r['loco_speed']:.4f}, frac={r['loco_frac']*100:.1f}%")
        else:
            print(f"  {r['name']:<30} centroid_speed not in subset")

    # Save results CSV
    df = pd.DataFrame(results)
    out_csv = os.path.join(RESULTS_DIR, "subset_comparison.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved: {out_csv}")
    print(f"UMAP plots saved: {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
