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
# Step 4: State windows — per-video frame ranges for manual inspection
# ---------------------------------------------------------------------------

def cmd_windows(fps: float = 30.0):
    """Export CSVs listing start/end frames for every state bout in every video."""
    for path in ["results/features/index.json", "results/shared/cluster_info.json"]:
        if not os.path.exists(path):
            sys.exit(f"Missing {path}. Run --extract and --cluster first.")

    with open("results/features/index.json") as f:
        index = json.load(f)
    with open("results/shared/cluster_info.json") as f:
        cluster_info = json.load(f)
    n_clusters = cluster_info["n_clusters"]

    os.makedirs("results/inspection", exist_ok=True)

    all_rows = []
    for stem in sorted(index.keys()):
        labels_path = f"results/shared/{stem}_labels.npy"
        if not os.path.exists(labels_path):
            continue
        labels = np.load(labels_path)

        # Run-length encode: find consecutive runs of the same state
        change_points = np.where(np.diff(labels) != 0)[0] + 1
        starts = np.concatenate([[0], change_points])
        ends = np.concatenate([change_points, [len(labels)]])
        values = labels[starts]

        rows = []
        for s, e, v in zip(starts, ends, values):
            dur_sec = (e - s) / fps
            rows.append({
                "stem": stem,
                "state": int(v),
                "start_frame": int(s),
                "end_frame": int(e - 1),
                "start_sec": round(s / fps, 2),
                "end_sec": round((e - 1) / fps, 2),
                "duration_sec": round(dur_sec, 2),
            })

        # Save per-video file
        df_vid = pd.DataFrame(rows)
        df_vid.to_csv(f"results/inspection/{stem}_windows.csv", index=False)
        all_rows.extend(rows)

    # Save combined file
    df_all = pd.DataFrame(all_rows)
    df_all.to_csv("results/inspection/all_windows.csv", index=False)

    print(f"State windows exported:")
    print(f"  Per-video: results/inspection/<stem>_windows.csv")
    print(f"  Combined:  results/inspection/all_windows.csv  ({len(df_all):,} bouts total)")
    print(f"\nTo find frames to inspect in a specific video:")
    print(f"  Open results/inspection/<stem>_windows.csv")
    print(f"  Find rows where state=<N>, scrub video to start_frame")

    # Print a sample: longest bout per state (good examples to start with)
    print(f"\nLongest single bout per state (good frames to check first):")
    for k in range(n_clusters):
        sub = df_all[df_all["state"] == k]
        if sub.empty:
            continue
        best = sub.loc[sub["duration_sec"].idxmax()]
        print(f"  State {k}: {best['stem']}")
        print(f"    frames {int(best['start_frame'])}–{int(best['end_frame'])}  "
              f"({best['duration_sec']:.1f}s at {best['start_sec']:.1f}s into video)")


# ---------------------------------------------------------------------------
# Step 5: Per-animal ethogram grid
# ---------------------------------------------------------------------------

def cmd_ethograms(fps: float = 30.0):
    """Plot a color-coded state timeline for every session of every animal."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap

    for path in ["results/features/index.json", "results/shared/cluster_info.json",
                 "metadata.csv"]:
        if not os.path.exists(path):
            sys.exit(f"Missing {path}. Run --extract, --cluster, and ensure metadata.csv exists.")

    with open("results/features/index.json") as f:
        index = json.load(f)
    with open("results/shared/cluster_info.json") as f:
        cluster_info = json.load(f)
    n_clusters = cluster_info["n_clusters"]

    meta = pd.read_csv("metadata.csv")
    meta["stem"] = meta["filename"].str.replace(r"\.mp4$", "", regex=True)

    # State color palette (consistent across all plots)
    STATE_COLORS = plt.cm.tab10(np.linspace(0, 0.9, n_clusters))
    cmap = ListedColormap(STATE_COLORS)

    os.makedirs("results/comparison", exist_ok=True)

    animals = sorted(meta["animal_id"].dropna().unique())
    print(f"Generating ethograms for {len(animals)} animals...")

    for animal_id in animals:
        animal_sessions = meta[meta["animal_id"] == animal_id].sort_values(
            ["experiment", "day", "context"]
        )

        # Filter to sessions that have labels
        valid = []
        for _, row in animal_sessions.iterrows():
            stem = row["stem"]
            if os.path.exists(f"results/shared/{stem}_labels.npy"):
                valid.append(row)

        if not valid:
            continue

        n_sessions = len(valid)
        max_frames = max(
            len(np.load(f"results/shared/{row['stem']}_labels.npy"))
            for row in valid
        )
        max_sec = max_frames / fps

        fig, ax = plt.subplots(figsize=(14, max(3, n_sessions * 0.5 + 1)))

        for i, row in enumerate(valid):
            stem = row["stem"]
            labels = np.load(f"results/shared/{stem}_labels.npy")
            n_frames = len(labels)
            time_axis = np.arange(n_frames) / fps

            # Draw colored segments using imshow row
            ax.imshow(
                labels[np.newaxis, :],
                aspect="auto",
                extent=[0, n_frames / fps, i - 0.4, i + 0.4],
                cmap=cmap,
                vmin=0,
                vmax=n_clusters - 1,
                interpolation="nearest",
            )

            # Y-axis label: experiment + day + context
            exp = row.get("experiment", "")
            day = row.get("day", "")
            ctx = row.get("context", "")
            ax.text(
                -0.5, i, f"{exp} D{day} {ctx}",
                va="center", ha="right", fontsize=7,
            )

        ax.set_xlim(0, max_sec)
        ax.set_ylim(-0.6, n_sessions - 0.4)
        ax.set_yticks([])
        ax.set_xlabel("Time (seconds)")
        ax.set_title(f"Animal {animal_id} — Behavioral State Timeline")

        # Legend
        patches = [
            mpatches.Patch(color=STATE_COLORS[k], label=f"State {k}")
            for k in range(n_clusters)
        ]
        ax.legend(handles=patches, loc="upper right", fontsize=8, ncol=n_clusters)

        plt.tight_layout()
        save_path = f"results/comparison/ethogram_{animal_id}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {save_path}")

    print(f"\nEthograms saved to results/comparison/ethogram_<animal_id>.png")


# ---------------------------------------------------------------------------
# Step 6: Pose profiles — mean skeleton per state
# ---------------------------------------------------------------------------

def cmd_pose_profiles(fps: float = 30.0, max_frames_per_state: int = 2000):
    """Plot the mean body posture (skeleton) for each behavioral state."""
    import matplotlib.pyplot as plt
    from main import load_pose

    for path in ["results/features/index.json", "results/shared/cluster_info.json"]:
        if not os.path.exists(path):
            sys.exit(f"Missing {path}. Run --extract and --cluster first.")

    with open("results/features/index.json") as f:
        index = json.load(f)
    with open("results/shared/cluster_info.json") as f:
        cluster_info = json.load(f)
    n_clusters = cluster_info["n_clusters"]

    # Keypoint names (must match config.yaml order)
    KEYPOINTS = ["left_ear", "right_ear", "nose", "center",
                 "left_hip", "right_hip", "tail_base", "tail_tip"]
    # Skeleton connections as (i, j) index pairs
    SKELETON = [
        (2, 0),  # nose → left_ear
        (2, 1),  # nose → right_ear
        (0, 1),  # left_ear → right_ear
        (2, 3),  # nose → center
        (3, 4),  # center → left_hip
        (3, 5),  # center → right_hip
        (4, 6),  # left_hip → tail_base
        (5, 6),  # right_hip → tail_base
        (6, 7),  # tail_base → tail_tip
    ]

    # Accumulate normalized pose frames per state
    # pose_accum[k] = list of (K, 2) arrays
    pose_accum = {k: [] for k in range(n_clusters)}
    frames_collected = {k: 0 for k in range(n_clusters)}

    stems = sorted(index.keys())
    print(f"Loading pose data from {len(stems)} videos to build pose profiles...")

    for stem in stems:
        # Stop collecting a state once we have enough frames
        if all(frames_collected[k] >= max_frames_per_state for k in range(n_clusters)):
            break

        labels_path = f"results/shared/{stem}_labels.npy"
        if not os.path.exists(labels_path):
            continue

        labels = np.load(labels_path)
        csv_path = index[stem].get("csv_path")
        if not csv_path or not os.path.exists(csv_path):
            continue

        pose, conf, bodyparts = load_pose(csv_path)

        # Normalize each frame to body-centered frame
        # 1. Translate: center on centroid of all keypoints
        # 2. Rotate: align nose(2) → tail_base(6) axis to point upward
        for k in range(n_clusters):
            if frames_collected[k] >= max_frames_per_state:
                continue

            frame_indices = np.where(labels == k)[0]
            if len(frame_indices) == 0:
                continue

            # Sample evenly across the video
            n_sample = min(len(frame_indices), max_frames_per_state - frames_collected[k])
            sampled = frame_indices[np.round(
                np.linspace(0, len(frame_indices) - 1, n_sample)
            ).astype(int)]

            for t in sampled:
                pts = pose[t].copy()  # (K, 2)

                # Skip frames with NaN (low confidence tracking)
                if np.any(np.isnan(pts)):
                    continue

                # Translate: center on centroid
                centroid = pts.mean(axis=0)
                pts -= centroid

                # Rotate: align nose-to-tailbase axis to point "up" (0, -1)
                nose = pts[2]
                tail = pts[6]
                axis = tail - nose
                axis_len = np.linalg.norm(axis)
                if axis_len < 1e-6:
                    continue
                axis /= axis_len

                # Target direction: nose should be at top (0, -1 in image coords)
                target = np.array([0.0, -1.0])
                angle = np.arctan2(axis[1], axis[0]) - np.arctan2(target[1], target[0])
                cos_a, sin_a = np.cos(-angle), np.sin(-angle)
                R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
                pts = pts @ R.T

                # Scale: normalize by nose-to-tailbase distance
                scale = np.linalg.norm(pts[2] - pts[6])
                if scale < 1e-6:
                    continue
                pts /= scale

                pose_accum[k].append(pts)
                frames_collected[k] += 1

    print(f"Frames collected per state: { {k: len(v) for k, v in pose_accum.items()} }")

    # Plot
    fig, axes = plt.subplots(1, n_clusters, figsize=(4 * n_clusters, 5))
    if n_clusters == 1:
        axes = [axes]

    STATE_COLORS = plt.cm.tab10(np.linspace(0, 0.9, n_clusters))

    for k, ax in enumerate(axes):
        frames = pose_accum[k]
        if not frames:
            ax.set_title(f"State {k}\n(no data)")
            ax.axis("off")
            continue

        pts_stack = np.stack(frames)          # (N, K, 2)
        mean_pose = pts_stack.mean(axis=0)    # (K, 2)
        std_pose = pts_stack.std(axis=0)      # (K, 2)

        color = STATE_COLORS[k]

        # Draw skeleton edges
        for (i, j) in SKELETON:
            ax.plot(
                [mean_pose[i, 0], mean_pose[j, 0]],
                [mean_pose[i, 1], mean_pose[j, 1]],
                color=color, linewidth=2.5, zorder=1,
            )

        # Draw keypoint uncertainty ellipses (std as error bars)
        for ki in range(len(KEYPOINTS)):
            ax.errorbar(
                mean_pose[ki, 0], mean_pose[ki, 1],
                xerr=std_pose[ki, 0], yerr=std_pose[ki, 1],
                fmt="o", color=color, markersize=6,
                elinewidth=1, ecolor="gray", alpha=0.7, zorder=2,
            )
            ax.annotate(
                KEYPOINTS[ki].replace("_", "\n"),
                xy=(mean_pose[ki, 0], mean_pose[ki, 1]),
                fontsize=5, ha="center", va="bottom",
                xytext=(0, 6), textcoords="offset points",
                color="dimgray",
            )

        n_frames = len(frames)
        global_frac = n_frames / sum(len(v) for v in pose_accum.values()) * 100
        ax.set_title(f"State {k}\n({n_frames:,} frames sampled)")
        ax.set_aspect("equal")
        ax.invert_yaxis()   # image convention: y increases downward
        ax.set_xlabel("← right     left →")
        ax.set_ylabel("← tail     nose →")
        ax.axhline(0, color="lightgray", linewidth=0.5, zorder=0)
        ax.axvline(0, color="lightgray", linewidth=0.5, zorder=0)

    plt.suptitle(
        "Mean Body Posture per Behavioral State\n"
        "(body-centered, normalized to nose–tail length)",
        fontsize=11,
    )
    plt.tight_layout()
    save_path = "results/comparison/pose_profiles.png"
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nPose profiles saved: {save_path}")


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
    parser.add_argument("--windows", action="store_true",
                        help="Export per-video state bout frame ranges for manual inspection")
    parser.add_argument("--ethograms", action="store_true",
                        help="Plot per-animal session-by-session ethogram timelines")
    parser.add_argument("--pose-profiles", action="store_true",
                        help="Plot mean skeleton posture for each behavioral state")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--n-clusters", type=int, default=None,
                        help="Fix number of behavioral states (default: auto-detect)")
    args = parser.parse_args()

    if not any([args.extract, args.cluster, args.report,
                args.windows, args.ethograms, args.pose_profiles]):
        parser.print_help()
        sys.exit(1)

    if args.extract:
        cmd_extract(fps=args.fps)
    if args.cluster:
        cmd_cluster(fps=args.fps, n_clusters=args.n_clusters)
    if args.report:
        cmd_report(fps=args.fps)
    if args.windows:
        cmd_windows(fps=args.fps)
    if args.ethograms:
        cmd_ethograms(fps=args.fps)
    if args.pose_profiles:
        cmd_pose_profiles(fps=args.fps)


if __name__ == "__main__":
    main()
