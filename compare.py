"""
compare.py — Cross-video behavioral analysis for VIEB
======================================================
Fits a single shared clusterer across all 222 videos so behavioral states
are directly comparable, then joins with metadata.csv to compare groups.

Usage
-----
Step 1:  python compare.py --extract              [--no-wavelets]
Step 2:  python compare.py --cluster              [--min-cluster-size N]
Step 3:  python compare.py --report
         python compare.py --summarize
"""

import argparse
import glob
import io
import json
import os
import sys

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# Step 1: Feature extraction

def cmd_extract(fps: float = 30.0, use_wavelets: bool = True):
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

    extractor = PoseFeatureExtractor(fps=fps, use_wavelets=use_wavelets)
    new_count = 0
    skip_count = 0

    if not use_wavelets:
        print("(wavelets disabled)")
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
# HMM smoother (pure numpy — no extra dependencies)
# ---------------------------------------------------------------------------

def _fit_hmm(labels: np.ndarray, n_states: int) -> dict:
    """
    Estimate HMM parameters from a label sequence.

    Expects labels to contain only valid state indices (0..n_states-1).
    Noise frames (-1) must be filtered out before calling this function.

    Fits:
      - prior: initial state distribution
      - A:     transition matrix (n_states × n_states)
      - B:     emission matrix (soft identity — allows Viterbi to correct
               isolated wrong-state frames)
    """
    # Prior: fraction of time in each state
    prior = np.bincount(labels, minlength=n_states).astype(float)
    prior /= prior.sum()
    prior = np.maximum(prior, 1e-10)

    # Transition matrix — skip pairs that cross a noise boundary.
    # Since caller passes pre-filtered valid labels (with noise removed but
    # video boundaries still present), we just count all consecutive pairs.
    a_labels = labels[:-1].astype(int)
    b_labels = labels[1:].astype(int)
    flat = a_labels * n_states + b_labels
    A = np.bincount(flat, minlength=n_states * n_states).reshape(n_states, n_states).astype(float)
    row_sums = A.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    A /= row_sums

    # Emission: soft identity with small noise floor so Viterbi can correct
    # isolated wrong-state frames.
    eps = 0.05
    B = np.full((n_states, n_states), eps / (n_states - 1))
    np.fill_diagonal(B, 1.0 - eps)

    return {"prior": prior, "A": A, "B": B, "n_states": n_states}


def _smooth_with_noise(labels: np.ndarray, hmm_params: dict) -> np.ndarray:
    """
    Run HMM Viterbi on each contiguous non-noise segment; preserve -1 labels.

    Splits the sequence at noise (-1) boundaries, decodes each segment
    independently, then stitches back.
    """
    smoothed = labels.copy()
    T = len(labels)
    t = 0
    while t < T:
        if labels[t] < 0:
            t += 1
            continue
        seg_start = t
        while t < T and labels[t] >= 0:
            t += 1
        seg = labels[seg_start:t]
        if len(seg) > 1:
            smoothed[seg_start:t] = _hmm_viterbi(seg, **hmm_params)
    return smoothed


def _hmm_viterbi(obs: np.ndarray, prior, A, B, n_states: int) -> np.ndarray:
    """
    Viterbi decoding: find the most likely state sequence given observations.

    Works in log-space to avoid underflow on long sequences.
    """
    T = len(obs)
    log_A = np.log(np.maximum(A, 1e-300))
    log_B = np.log(np.maximum(B, 1e-300))
    log_prior = np.log(np.maximum(prior, 1e-300))

    # delta[t, s] = log-prob of best path ending in state s at time t
    delta = np.full((T, n_states), -np.inf)
    psi   = np.zeros((T, n_states), dtype=np.int32)

    delta[0] = log_prior + log_B[:, obs[0]]

    for t in range(1, T):
        trans = delta[t - 1, :, None] + log_A          # (n_states, n_states)
        psi[t]   = np.argmax(trans, axis=0)
        delta[t] = np.max(trans, axis=0) + log_B[:, obs[t]]

    # Backtrack
    path = np.empty(T, dtype=np.int32)
    path[-1] = np.argmax(delta[-1])
    for t in range(T - 2, -1, -1):
        path[t] = psi[t + 1, path[t + 1]]

    return path


# ---------------------------------------------------------------------------
# Step 2: Shared clustering (UMAP + HDBSCAN)
# ---------------------------------------------------------------------------

def cmd_cluster(fps: float = 30.0, n_clusters: int = None, min_cluster_size: int = 50):
    import joblib
    import umap as umap_lib
    import hdbscan as hdbscan_lib
    from ml import BehaviorPreprocessor

    index_path = "results/features/index.json"
    if not os.path.exists(index_path):
        sys.exit("No index found. Run --extract first.")
    with open(index_path) as f:
        index = json.load(f)
    if not index:
        sys.exit("Index is empty. Run --extract first.")

    os.makedirs("results/shared", exist_ok=True)

    # ---- Load all feature matrices ----
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

    # ---- Standardize (no PCA — UMAP handles reduction) ----
    print("\nFitting shared standardizer...")
    preprocessor = BehaviorPreprocessor(use_pca=False)
    pooled_scaled = preprocessor.fit_transform(pooled)
    preprocessor.save("results/shared/preprocessor.pkl")
    print(f"  Standardized to {pooled_scaled.shape[1]} features")

    # ---- UMAP reduction ----
    print("\nFitting UMAP (n_components=10, n_neighbors=30)...")
    n_total = pooled_scaled.shape[0]
    n_sample = min(200_000, n_total)
    if n_total > n_sample:
        rng = np.random.default_rng(42)
        sample_idx = np.sort(rng.choice(n_total, n_sample, replace=False))
        fit_data = pooled_scaled[sample_idx]
        print(f"  Fitting on {n_sample:,}-frame sample, then transforming all {n_total:,}...")
    else:
        fit_data = pooled_scaled
        print(f"  Fitting on all {n_total:,} frames...")

    reducer = umap_lib.UMAP(
        n_components=10,
        n_neighbors=30,
        min_dist=0.0,
        random_state=42,
        low_memory=True,
        verbose=True,
    )
    reducer.fit(fit_data)
    pooled_umap = reducer.transform(pooled_scaled)
    joblib.dump(reducer, "results/shared/umap_reducer.pkl")
    print(f"  UMAP embedding: {pooled_umap.shape}")

    # ---- HDBSCAN clustering ----
    print(f"\nFitting HDBSCAN (min_cluster_size={min_cluster_size})...")
    clusterer_model = hdbscan_lib.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=10,
        cluster_selection_method="eom",
    )
    clusterer_model.fit(pooled_umap)
    all_raw_labels = clusterer_model.labels_.astype(np.int32)  # -1 = noise
    n_found = int(len(np.unique(all_raw_labels[all_raw_labels >= 0])))
    n_noise = int((all_raw_labels == -1).sum())
    print(f"  Behavioral states discovered: {n_found}")
    print(f"  Noise frames: {n_noise:,} ({100 * n_noise / len(all_raw_labels):.1f}%)")

    if n_found == 0:
        sys.exit("HDBSCAN found no clusters. Try a smaller --min-cluster-size.")

    # Cluster centers in standardized feature space (for characterize.py compatibility)
    cluster_centers = []
    for k in range(n_found):
        mask = all_raw_labels == k
        if mask.any():
            cluster_centers.append(pooled_scaled[mask].mean(axis=0).tolist())
        else:
            cluster_centers.append([0.0] * pooled_scaled.shape[1])

    joblib.dump(clusterer_model, "results/shared/clusterer.pkl")
    cluster_info = {
        "n_clusters": n_found,
        "cluster_centers": cluster_centers,
        "method": "umap+hdbscan",
        "min_cluster_size": min_cluster_size,
    }
    with open("results/shared/cluster_info.json", "w") as f:
        json.dump(cluster_info, f, indent=2)

    # ---- Per-video labels (slice from pooled HDBSCAN result) ----
    print(f"\nSlicing per-video labels ({len(stems)} videos)...")
    raw_labels_all = []
    for stem in stems:
        start, end = boundaries[stem]
        raw_labels_all.append(all_raw_labels[start:end])

    # ---- HMM smoothing on non-noise segments ----
    print("\nFitting HMM smoother on non-noise labels...")
    all_raw_concat = np.concatenate(raw_labels_all)
    valid_labels = all_raw_concat[all_raw_concat >= 0]

    if len(valid_labels) > 0 and n_found > 1:
        hmm_params = _fit_hmm(valid_labels, n_found)
        smoothed_labels_all = [_smooth_with_noise(lbl, hmm_params) for lbl in raw_labels_all]
    else:
        print("  Skipping HMM (no valid labels or single cluster)")
        smoothed_labels_all = raw_labels_all

    # ---- Save smoothed labels ----
    for stem, smoothed in zip(stems, smoothed_labels_all):
        np.save(f"results/shared/{stem}_labels.npy", smoothed.astype(np.int32))

    all_labels = np.concatenate(smoothed_labels_all)
    n_valid_total = int((all_labels >= 0).sum())
    print(f"\nGlobal state distribution ({n_valid_total:,} valid frames, "
          f"{(all_labels == -1).sum():,} noise):")
    for k in range(n_found):
        pct = float((all_labels == k).sum()) / len(all_labels) * 100
        n_frames = int((all_labels == k).sum())
        print(f"  State {k}: {pct:5.1f}%  ({n_frames:,} frames)")

    print(f"\nShared models → results/shared/")
    print(f"Per-video labels → results/shared/<stem>_labels.npy")


# ---------------------------------------------------------------------------
# Transition matrix helpers
# ---------------------------------------------------------------------------

def _compute_transition_matrix(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Row-normalised transition probability matrix, ignoring noise (-1) frames.

    Returns
    -------
    T : np.ndarray, shape (n_clusters, n_clusters)
        T[i, j] = P(next state is j | current state is i)
    """
    counts = np.zeros((n_clusters, n_clusters), dtype=np.float64)
    a = labels[:-1]
    b = labels[1:]
    valid = (a >= 0) & (b >= 0)
    for ai, bi in zip(a[valid], b[valid]):
        counts[ai, bi] += 1
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return counts / row_sums


def _plot_transition_heatmaps(group_matrices: dict, n_clusters: int, save_path: str):
    """
    Side-by-side heatmaps of mean transition matrices per group (context).
    """
    import matplotlib.pyplot as plt

    groups = sorted(group_matrices.keys())
    n_groups = len(groups)
    if n_groups == 0:
        return

    fig, axes = plt.subplots(1, n_groups, figsize=(4 * n_groups, 3.5))
    if n_groups == 1:
        axes = [axes]

    vmax = max(m.max() for m in group_matrices.values() if m is not None)

    for ax, grp in zip(axes, groups):
        mat = group_matrices[grp]
        im = ax.imshow(mat, vmin=0, vmax=vmax, cmap="Blues", aspect="auto")
        ax.set_title(f"Context {grp}")
        ax.set_xlabel("To state")
        ax.set_ylabel("From state")
        ax.set_xticks(range(n_clusters))
        ax.set_yticks(range(n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if mat[i, j] > vmax * 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Mean State Transition Probabilities by Context", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Step 3: Comparison report
# ---------------------------------------------------------------------------

def _plot_animal_trajectories(df, state_cols, n_clusters):
    """
    Line plot: each animal's state occupancy across days.
    One subplot per behavioral state; one line per animal.
    Reveals which animals show consistent fear-related state changes vs. which don't.
    """
    import matplotlib.pyplot as plt

    animals = sorted(df["animal_id"].dropna().unique())
    days = sorted(df["day"].dropna().unique())

    if len(animals) < 2 or len(days) < 2:
        print("  SKIP animal_trajectories.png: need ≥2 animals and ≥2 days")
        return

    fig, axes = plt.subplots(1, n_clusters, figsize=(3 * n_clusters, 5), sharey=False)
    if n_clusters == 1:
        axes = [axes]

    colors = plt.cm.tab20(np.linspace(0, 1, len(animals)))

    for ax, col in zip(axes, state_cols):
        for animal, color in zip(animals, colors):
            animal_df = df[df["animal_id"] == animal].copy()
            animal_df = animal_df.dropna(subset=["day", col])
            if len(animal_df) < 2:
                continue
            day_mean = animal_df.groupby("day")[col].mean()
            ax.plot(day_mean.index, day_mean.values, marker="o", color=color,
                    linewidth=1.5, markersize=4, label=str(animal), alpha=0.8)

        ax.set_title(f"State {col.split('_')[1]}")
        ax.set_xlabel("Day")
        ax.set_ylabel("Fraction of session")
        ax.grid(True, alpha=0.3)

    # Single legend outside the last axis
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Animal", bbox_to_anchor=(1.01, 0.5),
                   loc="center left", fontsize=8)

    plt.suptitle("Per-Animal Behavioral State Trajectory Across Days", fontsize=12)
    plt.tight_layout()
    save_path = "results/comparison/animal_trajectories.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


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

    # Build per-video summary + transition matrices
    rows = []
    trans_rows = []  # flattened transition probabilities per video
    for stem in sorted(index.keys()):
        labels_path = f"results/shared/{stem}_labels.npy"
        if not os.path.exists(labels_path):
            continue
        labels = np.load(labels_path)
        row = {"stem": stem}
        for k in range(n_clusters):
            row[f"state_{k}_frac"] = float((labels == k).mean())
        rows.append(row)

        # Transition matrix
        tmat = _compute_transition_matrix(labels, n_clusters)
        trans_row = {"stem": stem}
        for i in range(n_clusters):
            for j in range(n_clusters):
                trans_row[f"trans_{i}_{j}"] = float(tmat[i, j])
        trans_rows.append(trans_row)

    df_states = pd.DataFrame(rows)

    if not os.path.exists("metadata.csv"):
        sys.exit("metadata.csv not found.")
    meta = pd.read_csv("metadata.csv")
    meta["stem"] = meta["filename"].str.replace(r"\.mp4$", "", regex=True)

    df = df_states.merge(meta, on="stem", how="left")

    os.makedirs("results/comparison", exist_ok=True)
    df.to_csv("results/comparison/summary_table.csv", index=False)
    print(f"Summary table saved: results/comparison/summary_table.csv  ({len(df)} videos)")

    # ---- Transition matrix outputs ----
    df_trans = pd.DataFrame(trans_rows).merge(
        meta[["stem", "context", "day", "animal_id", "experiment"]].drop_duplicates("stem"),
        on="stem", how="left"
    )
    trans_cols = [c for c in df_trans.columns if c.startswith("trans_")]
    # Join full metadata for transition_table.csv
    df_trans_full = df_states.merge(
        pd.DataFrame(trans_rows), on="stem", how="left"
    ).merge(meta, on="stem", how="left")
    df_trans_full.to_csv("results/comparison/transition_table.csv", index=False)
    print(f"Transition table saved: results/comparison/transition_table.csv")

    # Heatmap per context
    if "context" in df_trans.columns and df_trans["context"].notna().any():
        group_matrices = {}
        for ctx, grp in df_trans.groupby("context"):
            mats = []
            for _, row in grp.iterrows():
                mat = np.array([[row[f"trans_{i}_{j}"] for j in range(n_clusters)]
                                for i in range(n_clusters)])
                mats.append(mat)
            group_matrices[ctx] = np.stack(mats).mean(axis=0)
        _plot_transition_heatmaps(
            group_matrices, n_clusters,
            "results/comparison/transition_by_context.png"
        )

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

    if "animal_id" in df.columns:
        boxplot_by_group("animal_id", "results/comparison/state_by_animal.png", "Animal ID")

    # Per-animal trajectory across days
    if "animal_id" in df.columns and "day" in df.columns:
        _plot_animal_trajectories(df, state_cols, n_clusters)

    # Statistical summary to terminal
    print(f"\n--- Group means (state fractions) ---")
    for group_col in ["fear", "day", "context", "experiment", "animal_id"]:
        if group_col not in df.columns:
            continue
        if df[group_col].notna().sum() == 0:
            continue
        print(f"\nBy {group_col}:")
        group_means = df.groupby(group_col)[state_cols].mean().round(3)
        print(group_means.to_string())

    print(f"\nResults in results/comparison/")




# ---------------------------------------------------------------------------
# Step 4: Per-animal scalar summary
# ---------------------------------------------------------------------------

def _identify_freeze_state(n_clusters: int) -> int:
    """
    Return the cluster ID with the lowest mean speed across all keypoints.

    Uses speed features (first 8 columns = per-keypoint speed) from the
    saved feature files, avoiding a full reload of all data.
    """
    index_path = "results/features/index.json"
    if not os.path.exists(index_path):
        sys.exit("No feature index found. Run --extract first.")
    with open(index_path) as f:
        index = json.load(f)

    speed_sum = np.zeros(n_clusters, dtype=np.float64)
    speed_cnt = np.zeros(n_clusters, dtype=np.float64)

    for stem, info in index.items():
        labels_path = f"results/shared/{stem}_labels.npy"
        if not os.path.exists(labels_path) or not os.path.exists(info["features_path"]):
            continue
        labels = np.load(labels_path)
        # First 8 features are per-keypoint speeds
        feats = np.load(info["features_path"])[:, :8].astype(np.float64)
        if feats.shape[0] != len(labels):
            continue
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                speed_sum[k] += feats[mask].mean(axis=1).sum()
                speed_cnt[k] += mask.sum()

    mean_speed = np.where(speed_cnt > 0, speed_sum / speed_cnt, np.inf)
    freeze_state = int(np.argmin(mean_speed))
    print(f"  Mean speed per cluster: "
          + ", ".join(f"S{k}={mean_speed[k]:.2f}" for k in range(n_clusters)))
    print(f"  → Freeze state identified: State {freeze_state}")
    return freeze_state


def cmd_summarize():
    """
    Per-animal scalar summary:
      - AUC of freeze-state occupancy across days (trapezoidal rule)
      - Mean discrimination ratio: (freeze_A - freeze_B) / (freeze_A + freeze_B)

    The freeze state is identified automatically as the cluster with the
    lowest mean keypoint speed.

    Output: results/comparison/animal_scalars.csv
    """
    summary_path = "results/comparison/summary_table.csv"
    if not os.path.exists(summary_path):
        sys.exit("summary_table.csv not found. Run --report first.")

    for path in ["results/shared/cluster_info.json"]:
        if not os.path.exists(path):
            sys.exit(f"Missing {path}. Run --cluster first.")

    with open("results/shared/cluster_info.json") as f:
        cluster_info = json.load(f)
    n_clusters = cluster_info["n_clusters"]

    df = pd.read_csv(summary_path)
    required = {"animal_id", "day", "context"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"summary_table.csv is missing columns: {missing}")

    print("Identifying freeze state (lowest mean speed)...")
    freeze_state = _identify_freeze_state(n_clusters)
    freeze_col = f"state_{freeze_state}_frac"

    if freeze_col not in df.columns:
        sys.exit(f"Column '{freeze_col}' not found in summary_table.csv.")

    rows = []
    for animal_id, group in df.groupby("animal_id"):
        if pd.isna(animal_id):
            continue

        # AUC of freeze-state occupancy across days
        group_sorted = group.dropna(subset=["day", freeze_col]).sort_values("day")
        if len(group_sorted) >= 2:
            auc = float(np.trapz(group_sorted[freeze_col].values,
                                 group_sorted["day"].values))
        elif len(group_sorted) == 1:
            auc = float(group_sorted[freeze_col].iloc[0])
        else:
            auc = float("nan")

        # Discrimination ratio per day: (freeze_A - freeze_B) / (freeze_A + freeze_B)
        disc_ratios = []
        for day, day_group in group.groupby("day"):
            ctx_means = (
                day_group.dropna(subset=["context", freeze_col])
                         .groupby("context")[freeze_col].mean()
            )
            if "A" in ctx_means.index and "B" in ctx_means.index:
                fa, fb = ctx_means["A"], ctx_means["B"]
                denom = fa + fb
                if denom > 0:
                    disc_ratios.append((fa - fb) / denom)

        mean_disc = float(np.mean(disc_ratios)) if disc_ratios else float("nan")

        rows.append({
            "animal_id": animal_id,
            "freeze_state": freeze_state,
            "freeze_auc": round(auc, 4),
            "mean_discrimination_ratio": round(mean_disc, 4) if not np.isnan(mean_disc) else float("nan"),
            "n_sessions": len(group),
            "n_days": int(group["day"].nunique()),
        })

    if not rows:
        sys.exit("No animals found with valid data in summary_table.csv.")

    os.makedirs("results/comparison", exist_ok=True)
    out = pd.DataFrame(rows).sort_values("animal_id")
    out.to_csv("results/comparison/animal_scalars.csv", index=False)
    print(f"\nSaved: results/comparison/animal_scalars.csv  ({len(out)} animals)")
    print(f"\n{out.to_string(index=False)}")


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
                        help="Fit shared UMAP+HDBSCAN clusterer across all videos")
    parser.add_argument("--report", action="store_true",
                        help="Generate comparison plots using metadata.csv")
    parser.add_argument("--summarize", action="store_true",
                        help="Per-animal AUC + discrimination ratio (requires --report output)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--n-clusters", type=int, default=None,
                        help="(ignored for HDBSCAN — kept for CLI compatibility)")
    parser.add_argument("--min-cluster-size", type=int, default=50,
                        help="HDBSCAN min_cluster_size (default: 50)")
    parser.add_argument("--no-wavelets", action="store_true",
                        help="Skip Morlet wavelet features during --extract (faster)")
    args = parser.parse_args()

    if not any([args.extract, args.cluster, args.report, args.summarize]):
        parser.print_help()
        sys.exit(1)

    if args.extract:
        cmd_extract(fps=args.fps, use_wavelets=not args.no_wavelets)
    if args.cluster:
        cmd_cluster(fps=args.fps, min_cluster_size=args.min_cluster_size)
    if args.report:
        cmd_report(fps=args.fps)
    if args.summarize:
        cmd_summarize()


if __name__ == "__main__":
    main()
