"""
compare.py — Cross-video behavioral analysis for VIEB
======================================================
Fits a single shared clusterer across all 222 videos so behavioral states
are directly comparable, then joins with metadata.csv to compare groups.

Usage
-----
Step 1 – extract features from every video (run once, ~minutes):
    python compare.py --extract

Step 2 – fit shared model and label all videos:
    python compare.py --cluster
    python compare.py --cluster --n-clusters 6   # force a fixed number of states

Step 3 – generate comparison plots and summary table:
    python compare.py --report
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd


# Step 1: Feature extraction

def cmd_extract(fps: float = 30.0):
    from ml import PoseFeatureExtractor
    from main import load_pose, _find_dlc_csv

    videos = sorted(glob.glob("raw_videos/*.mp4"))
    if not videos:
        sys.exit("No .mp4 files found in raw_videos/")

    os.makedirs("results/features", exist_ok=True)

    index_path = "results/features/index.json"
    index = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)

    extractor = PoseFeatureExtractor(fps=fps)
    new_count = 0
    skip_count = 0

    print(f"Extracting features from {len(videos)} videos...")
    for video_path in videos:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join("results", "features", f"{stem}_features.npy")

        if os.path.exists(out_path):
            skip_count += 1
            continue

        csv_path = _find_dlc_csv(video_path)
        if csv_path is None:
            print(f"  SKIP (no DLC CSV): {stem}")
            continue

        print(f"  {stem}")
        pose, conf, _ = load_pose(csv_path)
        features_dict = extractor.extract_features(pose, confidence=conf)
        features_flat = extractor._flatten_features(features_dict)

        np.save(out_path, features_flat.astype(np.float32))
        index[stem] = {
            "video_path": video_path,
            "csv_path": csv_path,
            "n_frames": int(pose.shape[0]),
            "n_features": int(features_flat.shape[1]),
            "features_path": out_path,
        }
        new_count += 1

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone. Extracted {new_count} new, skipped {skip_count} already done.")
    print(f"Total in index: {len(index)} videos")
    print(f"Feature files saved to results/features/")


# ---------------------------------------------------------------------------
# Step 2: Shared clustering
# ---------------------------------------------------------------------------

def cmd_cluster(fps: float = 30.0, n_clusters: int = None):
    import joblib
    from ml import BehaviorPreprocessor, BehaviorClusterer

    index_path = "results/features/index.json"
    if not os.path.exists(index_path):
        sys.exit("No index found. Run --extract first.")
    with open(index_path) as f:
        index = json.load(f)
    if not index:
        sys.exit("Index is empty. Run --extract first.")

    os.makedirs("results/shared", exist_ok=True)

    # Load all feature matrices
    stems = sorted(index.keys())
    print(f"Loading features from {len(stems)} videos...")
    all_features = []
    boundaries = {}
    cursor = 0
    for stem in stems:
        feat = np.load(index[stem]["features_path"])
        boundaries[stem] = (cursor, cursor + len(feat))
        cursor += len(feat)
        all_features.append(feat)

    pooled = np.vstack(all_features).astype(np.float64)
    print(f"Pooled matrix: {pooled.shape[0]:,} frames × {pooled.shape[1]} features")

    # Fit shared preprocessor
    print("\nFitting shared preprocessor (standardize + PCA)...")
    preprocessor = BehaviorPreprocessor(use_pca=True, pca_variance=0.95)
    pooled_norm = preprocessor.fit_transform(pooled)
    preprocessor.save("results/shared/preprocessor.pkl")
    print(f"  Reduced to {pooled_norm.shape[1]} PCA components")

    # Fit shared clusterer
    print("\nFitting shared clusterer on pooled data...")
    clusterer = BehaviorClusterer(
        method="kmeans",
        n_clusters=n_clusters,
        auto_tune=(n_clusters is None),
        max_clusters=12,
    )
    clusterer.fit(pooled_norm)
    n_found = int(len(np.unique(clusterer.labels_[clusterer.labels_ >= 0])))
    print(f"  Behavioral states discovered: {n_found}")

    # Persist the underlying sklearn model (BehaviorClusterer has no save method)
    joblib.dump(clusterer.model, "results/shared/clusterer.pkl")
    cluster_info = {
        "n_clusters": n_found,
        "cluster_centers": clusterer.cluster_centers_.tolist()
        if clusterer.cluster_centers_ is not None else None,
    }
    with open("results/shared/cluster_info.json", "w") as f:
        json.dump(cluster_info, f, indent=2)

    # Per-video labels using the shared model
    print(f"\nLabeling all {len(stems)} videos with shared cluster assignments...")
    for stem in stems:
        start, end = boundaries[stem]
        video_norm = pooled_norm[start:end]
        labels = clusterer.predict(video_norm)
        np.save(f"results/shared/{stem}_labels.npy", labels.astype(np.int32))

    # Print global state distribution
    all_labels = np.concatenate([
        np.load(f"results/shared/{s}_labels.npy") for s in stems
    ])
    print(f"\nGlobal state distribution ({all_labels.shape[0]:,} frames total):")
    for k in range(n_found):
        pct = (all_labels == k).mean() * 100
        n_frames = (all_labels == k).sum()
        print(f"  State {k}: {pct:5.1f}%  ({n_frames:,} frames)")

    print(f"\nShared models → results/shared/")
    print(f"Per-video labels → results/shared/<stem>_labels.npy")


# ---------------------------------------------------------------------------
# Step 3: Comparison report
# ---------------------------------------------------------------------------

def cmd_report(fps: float = 30.0):
    import matplotlib.pyplot as plt
    from scipy import stats

    for path in ["results/features/index.json", "results/shared/cluster_info.json"]:
        if not os.path.exists(path):
            sys.exit(f"Missing {path}. Run --extract and --cluster first.")

    with open("results/features/index.json") as f:
        index = json.load(f)
    with open("results/shared/cluster_info.json") as f:
        cluster_info = json.load(f)
    n_clusters = cluster_info["n_clusters"]
    state_cols = [f"state_{k}_frac" for k in range(n_clusters)]

    # Build per-video summary
    rows = []
    for stem in sorted(index.keys()):
        labels_path = f"results/shared/{stem}_labels.npy"
        if not os.path.exists(labels_path):
            continue
        labels = np.load(labels_path)
        row = {"stem": stem}
        for k in range(n_clusters):
            row[f"state_{k}_frac"] = float((labels == k).mean())
        rows.append(row)

    df_states = pd.DataFrame(rows)

    if not os.path.exists("metadata.csv"):
        sys.exit("metadata.csv not found.")
    meta = pd.read_csv("metadata.csv")
    meta["stem"] = meta["filename"].str.replace(r"\.mp4$", "", regex=True)

    df = df_states.merge(meta, on="stem", how="left")

    os.makedirs("results/comparison", exist_ok=True)
    df.to_csv("results/comparison/summary_table.csv", index=False)
    print(f"Summary table saved: results/comparison/summary_table.csv  ({len(df)} videos)")

    # ---- Plots ----
    def boxplot_by_group(group_col, save_path, group_label):
        valid = df[group_col].dropna()
        groups = sorted(valid.unique())
        if len(groups) < 2:
            print(f"  SKIP {save_path}: only {len(groups)} group(s) in '{group_col}'")
            return

        fig, axes = plt.subplots(1, n_clusters, figsize=(3 * n_clusters, 5), sharey=False)
        if n_clusters == 1:
            axes = [axes]

        for ax, col in zip(axes, state_cols):
            data = [df[df[group_col] == g][col].dropna().values for g in groups]
            bp = ax.boxplot(data, labels=[str(g) for g in groups], patch_artist=True)
            colors = plt.cm.tab10(np.linspace(0, 0.5, len(groups)))
            for patch, color in zip(bp["boxes"], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

            # Mann-Whitney U between first two groups if exactly 2
            if len(groups) == 2 and len(data[0]) > 0 and len(data[1]) > 0:
                _, p = stats.mannwhitneyu(data[0], data[1], alternative="two-sided")
                stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                y_max = max(np.max(d) if len(d) else 0 for d in data)
                ax.annotate(
                    stars,
                    xy=(1.5, y_max * 1.05),
                    ha="center", fontsize=10,
                )

            ax.set_title(f"State {col.split('_')[1]}")
            ax.set_ylabel("Fraction of session")
            ax.set_xlabel(group_label)

        plt.suptitle(f"Behavioral State Occupancy by {group_label}", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    # Fear comparison (only if column is filled in)
    if df["fear"].notna().any():
        boxplot_by_group("fear", "results/comparison/state_by_fear.png", "Fear Condition")
    else:
        print("  SKIP state_by_fear.png: 'fear' column in metadata.csv is empty (fill it in)")

    if "day" in df.columns:
        boxplot_by_group("day", "results/comparison/state_by_day.png", "Day")

    if "context" in df.columns:
        boxplot_by_group("context", "results/comparison/state_by_context.png", "Context")

    if "experiment" in df.columns:
        boxplot_by_group("experiment", "results/comparison/state_by_experiment.png", "Experiment (CFC vs CFD)")

    # Statistical summary to terminal
    print(f"\n--- Group means (state fractions) ---")
    for group_col in ["fear", "day", "context", "experiment"]:
        if group_col not in df.columns:
            continue
        if df[group_col].notna().sum() == 0:
            continue
        print(f"\nBy {group_col}:")
        group_means = df.groupby(group_col)[state_cols].mean().round(3)
        print(group_means.to_string())

    print(f"\nResults in results/comparison/")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cross-video behavioral analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--extract", action="store_true",
                        help="Extract and save pose features from all videos")
    parser.add_argument("--cluster", action="store_true",
                        help="Fit shared clusterer across all videos")
    parser.add_argument("--report", action="store_true",
                        help="Generate comparison plots using metadata.csv")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--n-clusters", type=int, default=None,
                        help="Fix number of behavioral states (default: auto-detect)")
    args = parser.parse_args()

    if not any([args.extract, args.cluster, args.report]):
        parser.print_help()
        sys.exit(1)

    if args.extract:
        cmd_extract(fps=args.fps)
    if args.cluster:
        cmd_cluster(fps=args.fps, n_clusters=args.n_clusters)
    if args.report:
        cmd_report(fps=args.fps)


if __name__ == "__main__":
    main()
