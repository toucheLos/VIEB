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
import platform
import sys

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# GPU detection and hardware banner
# ---------------------------------------------------------------------------

def _detect_gpu() -> bool:
    """Return True if cuML (RAPIDS) is importable and CUDA is available."""
    if platform.system() == "Windows":
        return False  # cuML has no Windows wheels; requires WSL2
    try:
        import cuml  # noqa: F401
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()  # raises if no CUDA device
        return True
    except Exception:
        return False


def _get_gpu_name() -> str | None:
    """Return GPU name string via nvidia-smi, or None if unavailable."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        name = result.stdout.strip().splitlines()[0].strip()
        if name:
            return name
    except Exception:
        pass
    return None


def _print_hardware_banner():
    """Print a summary of available hardware at startup."""
    import multiprocessing
    cpu = platform.processor() or platform.machine()
    n_cores = multiprocessing.cpu_count()
    gpu_name = _get_gpu_name()
    on_windows = platform.system() == "Windows"
    gpu_accel = _detect_gpu()

    print("=" * 60)
    print("VIEB  —  Hardware")
    print(f"  CPU : {cpu} ({n_cores} logical cores)")
    if gpu_name:
        print(f"  GPU : {gpu_name}  [CUDA available]")
        if on_windows:
            print("        GPU acceleration (cuML) requires WSL2 on Windows")
        elif gpu_accel:
            print("        cuML available — GPU acceleration ready for --cluster")
        else:
            print("        cuML not installed — install via pip for GPU acceleration")
    else:
        print("  GPU : none detected")
    print("=" * 60)
    print()



# Step 1: Feature extraction

def cmd_extract(fps: float = 30.0, use_wavelets: bool = True):
    from ml import PoseFeatureExtractor
    from pose_io import load_pose, _find_dlc_csv

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

def _run_validation_report(
    stems, train_stems, test_stems, boundaries,
    smoothed_labels_all, probs_all, n_found, min_cluster_size,
):
    """Compute and print train/test state distribution comparison."""
    stem_to_idx = {s: i for i, s in enumerate(stems)}
    train_set = set(train_stems)
    test_set  = set(test_stems)

    def _per_video_fracs(stem_set):
        fracs = []
        for s in stem_set:
            i = stem_to_idx[s]
            lbl = smoothed_labels_all[i]
            row = []
            for k in range(n_found):
                row.append(float((lbl == k).sum()) / max(1, len(lbl)))
            fracs.append(row)
        return np.array(fracs) if fracs else np.zeros((0, n_found))

    train_fracs = _per_video_fracs(train_set)
    test_fracs  = _per_video_fracs(test_set)

    train_mean = train_fracs.mean(axis=0) if len(train_fracs) else np.zeros(n_found)
    test_mean  = test_fracs.mean(axis=0)  if len(test_fracs)  else np.zeros(n_found)
    deltas = np.abs(train_mean - test_mean)
    mean_delta = float(deltas.mean())
    generalization = round(1.0 - mean_delta, 4)

    if generalization >= 0.9:
        quality = "excellent"
    elif generalization >= 0.8:
        quality = "good"
    else:
        quality = "poor"

    print(f"\n=== Clustering Validation (Train/Test Split) ===")
    print(f"Train videos: {len(train_stems)}  Test videos: {len(test_stems)}")
    n_train_fr = sum(boundaries[s][1] - boundaries[s][0] for s in train_stems)
    n_test_fr  = sum(boundaries[s][1] - boundaries[s][0] for s in test_stems)
    print(f"Train frames: {n_train_fr:,}  Test frames: {n_test_fr:,}")
    print(f"\nState distribution comparison:")
    print(f"{'State':>6} | {'Train%':>7} | {'Test%':>6} | {'Delta':>6}")
    print("-" * 36)
    per_state_delta = {}
    for k in range(n_found):
        tr = train_mean[k] * 100
        te = test_mean[k] * 100
        d  = deltas[k] * 100
        per_state_delta[str(k)] = round(float(deltas[k]), 6)
        print(f"  {k:>4} | {tr:>6.1f}% | {te:>5.1f}% | {d:>5.1f}%")
    print(f"\nMean delta: {mean_delta * 100:.1f}%")
    print(f"Generalization score: {generalization:.3f} ({quality})")

    if generalization < 0.8:
        print(f"\nWARNING: clustering may not generalize well.")
        print(f"Try increasing --min-cluster-size.")

    report = {
        "generalization_score": generalization,
        "train_stems": sorted(train_stems),
        "test_stems":  sorted(test_stems),
        "per_state_delta": per_state_delta,
        "mean_delta": round(mean_delta, 6),
    }
    os.makedirs("results/shared", exist_ok=True)
    with open("results/shared/validation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nValidation report saved: results/shared/validation_report.json")


def cmd_cluster(fps: float = 30.0, n_clusters: int = None, min_cluster_size: int = 50, min_samples: int = None, umap_dims: int = 10, validate: bool = False):
    import joblib
    from ml import BehaviorPreprocessor

    if _detect_gpu():
        use_gpu = True
    else:
        use_gpu = False
        if platform.system() == "Windows":
            print("[GPU] Running on CPU (cuML requires WSL2 on Windows).")
        else:
            print("[GPU] Running on CPU (cuML not available).")

    if use_gpu:
        from cuml.manifold import UMAP as UMAPClass
        from cuml.cluster import HDBSCAN as HDBSCANClass
        print("[GPU] Using cuML UMAP + HDBSCAN")
    else:
        import umap as umap_lib
        import hdbscan as hdbscan_lib
        UMAPClass = umap_lib.UMAP
        HDBSCANClass = hdbscan_lib.HDBSCAN

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

    # ---- Train/test split (video-level, not frame-level) ----
    if validate:
        rng_split = np.random.default_rng(42)
        shuffled = rng_split.permutation(len(stems)).tolist()
        n_train = int(len(stems) * 0.8)
        train_stems = sorted([stems[i] for i in shuffled[:n_train]])
        test_stems  = sorted([stems[i] for i in shuffled[n_train:]])
        print(f"\n=== Clustering Validation (Train/Test Split) ===")
        print(f"Train videos: {len(train_stems)}  Test videos: {len(test_stems)}")
        print(f"Test set stems: {test_stems}")
    else:
        train_stems = stems
        test_stems  = []

    print(f"Loading features from {len(stems)} videos...")
    all_features = []
    boundaries = {}
    cursor = 0
    for stem in stems:
        feat = np.load(index[stem]["features_path"].replace("\\", "/"))
        boundaries[stem] = (cursor, cursor + len(feat))
        cursor += len(feat)
        all_features.append(feat)

    pooled = np.vstack(all_features).astype(np.float64)
    print(f"Pooled matrix: {pooled.shape[0]:,} frames × {pooled.shape[1]} features")

    # ---- Standardize (no PCA — UMAP handles reduction) ----
    print("\nFitting shared standardizer...")
    preprocessor = BehaviorPreprocessor(use_pca=False)

    if validate:
        # Fit only on train frames
        train_indices = np.concatenate([
            np.arange(boundaries[s][0], boundaries[s][1]) for s in train_stems
        ])
        train_frames = pooled[train_indices]
        preprocessor.fit(train_frames)
        pooled_scaled = preprocessor.transform(pooled)
    else:
        pooled_scaled = preprocessor.fit_transform(pooled)

    preprocessor.save("results/shared/preprocessor.pkl")
    print(f"  Standardized to {pooled_scaled.shape[1]} features")

    # ---- UMAP reduction ----
    print(f"\nFitting UMAP (n_components={umap_dims}, n_neighbors=30)...")
    umap_save_path = "results/shared/umap_reducer.pkl"
    if os.path.exists(umap_save_path):
        try:
            _saved = joblib.load(umap_save_path)
            if getattr(_saved, 'n_components', None) != umap_dims:
                print(f"  [info] Saved UMAP reducer has different n_components; refitting with {umap_dims}.")
        except Exception:
            pass

    if validate:
        # Fit UMAP on train frames only
        train_scaled = pooled_scaled[train_indices]
        n_train_frames = len(train_scaled)
        n_sample = min(200_000, n_train_frames)
        if n_train_frames > n_sample:
            rng = np.random.default_rng(42)
            sample_idx = np.sort(rng.choice(n_train_frames, n_sample, replace=False))
            fit_data = train_scaled[sample_idx]
            print(f"  Fitting UMAP on {n_sample:,}-frame train sample...")
        else:
            fit_data = train_scaled
            print(f"  Fitting UMAP on {n_train_frames:,} train frames...")
        umap_kwargs = dict(n_components=umap_dims, n_neighbors=30, min_dist=0.0, random_state=42)
        if not use_gpu:
            umap_kwargs.update(low_memory=True, verbose=False)
        reducer = UMAPClass(**umap_kwargs)
        reducer.fit(fit_data)
        # Transform all frames (train + test) through the fitted UMAP
        pooled_umap = reducer.transform(pooled_scaled)
        train_n_frames = int(sum(boundaries[s][1] - boundaries[s][0] for s in train_stems))
        test_n_frames  = int(sum(boundaries[s][1] - boundaries[s][0] for s in test_stems))
        print(f"  Train frames: {train_n_frames:,}  Test frames: {test_n_frames:,}")
    else:
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
        umap_kwargs = dict(n_components=umap_dims, n_neighbors=30, min_dist=0.0, random_state=42)
        if not use_gpu:
            umap_kwargs.update(low_memory=True, verbose=True)
        reducer = UMAPClass(**umap_kwargs)
        reducer.fit(fit_data)
        pooled_umap = reducer.transform(pooled_scaled)

    if hasattr(pooled_umap, "to_numpy"):
        pooled_umap = pooled_umap.to_numpy()
    elif hasattr(pooled_umap, "get"):
        pooled_umap = pooled_umap.get()
    pooled_umap = np.asarray(pooled_umap, dtype=np.float32)
    joblib.dump(reducer, "results/shared/umap_reducer.pkl")
    print(f"  UMAP embedding: {pooled_umap.shape}")

    # ---- HDBSCAN clustering ----
    effective_min_samples = min_samples if min_samples is not None else min_cluster_size
    print(f"\nFitting HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={effective_min_samples})...")

    if validate:
        # Fit HDBSCAN on train frames only
        train_umap = pooled_umap[train_indices]
        clusterer_model = HDBSCANClass(
            min_cluster_size=min_cluster_size,
            min_samples=effective_min_samples,
            cluster_selection_method="eom",
        )
        clusterer_model.fit(train_umap)
        train_raw_labels = clusterer_model.labels_
        if hasattr(train_raw_labels, "to_numpy"):
            train_raw_labels = train_raw_labels.to_numpy()
        elif hasattr(train_raw_labels, "get"):
            train_raw_labels = train_raw_labels.get()
        train_raw_labels = np.asarray(train_raw_labels, dtype=np.int32)

        # Assign test frames using approximate_predict (CPU) or transform (GPU)
        test_indices = np.concatenate([
            np.arange(boundaries[s][0], boundaries[s][1]) for s in test_stems
        ]) if test_stems else np.array([], dtype=np.int64)

        if len(test_indices) > 0:
            test_umap = pooled_umap[test_indices]
            if use_gpu:
                try:
                    test_result = clusterer_model.transform(test_umap)
                    if hasattr(test_result, "to_numpy"):
                        test_result = test_result.to_numpy()
                    elif hasattr(test_result, "get"):
                        test_result = test_result.get()
                    test_raw_labels = np.asarray(test_result[:, 0] if test_result.ndim > 1 else test_result, dtype=np.int32)
                    test_probs = np.ones(len(test_raw_labels), dtype=np.float32)
                    test_probs[test_raw_labels < 0] = 0.0
                except Exception:
                    test_raw_labels = np.full(len(test_indices), -1, dtype=np.int32)
                    test_probs = np.zeros(len(test_indices), dtype=np.float32)
            else:
                try:
                    from hdbscan import approximate_predict
                    test_raw_labels, test_probs = approximate_predict(clusterer_model, test_umap)
                    test_raw_labels = np.asarray(test_raw_labels, dtype=np.int32)
                    test_probs = np.asarray(test_probs, dtype=np.float32)
                except Exception as e:
                    print(f"  [WARN] approximate_predict failed: {e}. Using noise labels for test.")
                    test_raw_labels = np.full(len(test_indices), -1, dtype=np.int32)
                    test_probs = np.zeros(len(test_indices), dtype=np.float32)
        else:
            test_raw_labels = np.array([], dtype=np.int32)
            test_probs = np.array([], dtype=np.float32)

        # Build pooled labels array combining train and test
        all_raw_labels = np.full(len(pooled_umap), -1, dtype=np.int32)
        all_raw_labels[train_indices] = train_raw_labels
        if len(test_indices) > 0:
            all_raw_labels[test_indices] = test_raw_labels

        # Train probabilities from clusterer
        train_probs_raw = getattr(clusterer_model, "probabilities_", None)
        if train_probs_raw is not None:
            if hasattr(train_probs_raw, "to_numpy"):
                train_probs_raw = train_probs_raw.to_numpy()
            elif hasattr(train_probs_raw, "get"):
                train_probs_raw = train_probs_raw.get()
            train_probs_raw = np.asarray(train_probs_raw, dtype=np.float32)
        else:
            train_probs_raw = np.where(train_raw_labels >= 0, 1.0, 0.0).astype(np.float32)

        all_probs = np.zeros(len(pooled_umap), dtype=np.float32)
        all_probs[train_indices] = train_probs_raw
        if len(test_indices) > 0:
            all_probs[test_indices] = test_probs

    else:
        clusterer_model = HDBSCANClass(
            min_cluster_size=min_cluster_size,
            min_samples=effective_min_samples,
            cluster_selection_method="eom",
        )
        clusterer_model.fit(pooled_umap)
        raw_labels = clusterer_model.labels_
        if hasattr(raw_labels, "to_numpy"):
            raw_labels = raw_labels.to_numpy()
        elif hasattr(raw_labels, "get"):
            raw_labels = raw_labels.get()
        all_raw_labels = np.asarray(raw_labels, dtype=np.int32)

        raw_probs = getattr(clusterer_model, "probabilities_", None)
        if raw_probs is not None:
            if hasattr(raw_probs, "to_numpy"):
                raw_probs = raw_probs.to_numpy()
            elif hasattr(raw_probs, "get"):
                raw_probs = raw_probs.get()
            all_probs = np.asarray(raw_probs, dtype=np.float32)
        else:
            all_probs = np.where(all_raw_labels >= 0, 1.0, 0.0).astype(np.float32)

    n_found = int(len(np.unique(all_raw_labels[all_raw_labels >= 0])))
    n_noise = int((all_raw_labels == -1).sum())
    print(f"  Behavioral states discovered: {n_found}")
    print(f"  Noise frames: {n_noise:,} ({100 * n_noise / len(all_raw_labels):.1f}%)")

    # ---- Confidence stats ----
    non_noise_probs = all_probs[all_raw_labels >= 0]
    if len(non_noise_probs) > 0:
        mean_conf = float(non_noise_probs.mean())
        low_conf_frac = float((non_noise_probs < 0.5).sum() / len(non_noise_probs))
    else:
        mean_conf = 0.0
        low_conf_frac = 0.0
    print(f"  Mean cluster confidence: {mean_conf:.3f}")
    print(f"  Low confidence frames (<0.5): {100 * low_conf_frac:.1f}%")

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
        "mean_confidence": round(mean_conf, 4),
        "low_confidence_frac": round(low_conf_frac, 4),
    }
    with open("results/shared/cluster_info.json", "w") as f:
        json.dump(cluster_info, f, indent=2)

    # ---- Per-video labels (slice from pooled HDBSCAN result) ----
    print(f"\nSlicing per-video labels ({len(stems)} videos)...")
    raw_labels_all = []
    raw_probs_all = []
    for stem in stems:
        start, end = boundaries[stem]
        raw_labels_all.append(all_raw_labels[start:end])
        raw_probs_all.append(all_probs[start:end])

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

    # ---- Save smoothed labels and probabilities ----
    for stem, smoothed, probs in zip(stems, smoothed_labels_all, raw_probs_all):
        np.save(f"results/shared/{stem}_labels.npy", smoothed.astype(np.int32))
        np.save(f"results/shared/{stem}_probs.npy", probs.astype(np.float32))

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
    print(f"Per-video probabilities → results/shared/<stem>_probs.npy")

    # ---- Validation report ----
    if validate:
        _run_validation_report(
            stems, train_stems, test_stems, boundaries,
            smoothed_labels_all, raw_probs_all, n_found,
            min_cluster_size,
        )


# ---------------------------------------------------------------------------
# Step 2.5: Collapse similar states (post-clustering merge)
# ---------------------------------------------------------------------------

def cmd_collapse(threshold: float = 0.5):
    """
    Merge behavioral states whose centroids have cosine similarity > threshold.

    Operates on the existing results/shared/ outputs without re-running UMAP or
    HDBSCAN. Updates cluster_info.json and remaps all _labels.npy files in-place.
    """
    from collections import defaultdict

    cluster_info_path = "results/shared/cluster_info.json"
    if not os.path.exists(cluster_info_path):
        sys.exit("No cluster_info.json found. Run --cluster first.")
    index_path = "results/features/index.json"
    if not os.path.exists(index_path):
        sys.exit("No feature index found. Run --extract first.")

    with open(cluster_info_path) as f:
        cluster_info = json.load(f)
    with open(index_path) as f:
        index = json.load(f)

    n_clusters = cluster_info["n_clusters"]
    centers = np.array(cluster_info["cluster_centers"], dtype=np.float64)  # (K, D)

    # Pairwise cosine similarity between centroids
    norms = np.linalg.norm(centers, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    centers_normed = centers / norms
    sim = centers_normed @ centers_normed.T  # (K, K)

    # Collect pairs above threshold (upper triangle, skip diagonal)
    merge_edges = [
        (i, j)
        for i in range(n_clusters)
        for j in range(i + 1, n_clusters)
        if sim[i, j] > threshold
    ]
    print(f"Cosine similarity threshold: {threshold}")
    print(f"Pairs above threshold: {len(merge_edges)}")

    # Union-find to build connected components
    parent = list(range(n_clusters))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in merge_edges:
        pi, pj = _find(i), _find(j)
        if pi != pj:
            parent[pi] = pj

    groups = defaultdict(list)
    for k in range(n_clusters):
        groups[_find(k)].append(k)

    sorted_groups = sorted(groups.values(), key=lambda g: min(g))
    n_new = len(sorted_groups)

    old_to_new = {}
    for new_id, group in enumerate(sorted_groups):
        for old_id in group:
            old_to_new[old_id] = new_id

    print(f"\nCollapsing {n_clusters} → {n_new} states")
    for new_id, group in enumerate(sorted_groups):
        if len(group) > 1:
            print(f"  New state {new_id}: merged from original states {sorted(group)}")

    if n_new == n_clusters:
        print("No merges at this threshold. Try a higher --collapse-threshold.")
        return

    # Remap label files and count frames per old cluster for weighted center averaging
    stems = sorted(index.keys())
    frame_counts = np.zeros(n_clusters, dtype=np.int64)

    print(f"\nRemapping {len(stems)} label files...")
    for stem in stems:
        labels_path = f"results/shared/{stem}_labels.npy"
        if not os.path.exists(labels_path):
            continue
        labels = np.load(labels_path)
        for k in range(n_clusters):
            frame_counts[k] += int((labels == k).sum())
        new_labels = labels.copy()
        for old_id, new_id in old_to_new.items():
            new_labels[labels == old_id] = new_id  # uses original `labels` as mask
        np.save(labels_path, new_labels.astype(np.int32))

    # New cluster centers: weighted mean of merged old centers by frame count
    new_centers = []
    for group in sorted_groups:
        total = sum(int(frame_counts[k]) for k in group)
        if total == 0:
            new_centers.append(centers[group[0]].tolist())
        else:
            weighted = sum(frame_counts[k] * centers[k] for k in group)
            new_centers.append((weighted / total).tolist())

    cluster_info["n_clusters"] = n_new
    cluster_info["cluster_centers"] = new_centers
    cluster_info["collapse_threshold"] = threshold
    cluster_info["collapse_map"] = {str(k): v for k, v in old_to_new.items()}

    with open(cluster_info_path, "w") as f:
        json.dump(cluster_info, f, indent=2)

    print(f"\nUpdated results/shared/cluster_info.json  ({n_clusters} → {n_new} states)")
    print("All _labels.npy files remapped in-place.")
    print("Run --report (and --summarize / characterize.py) to rebuild downstream outputs.")


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


def cmd_report(fps: float = 30.0, min_confidence: float = 0.0):
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

    if min_confidence > 0.0:
        print(f"Applying min-confidence filter: {min_confidence} "
              f"(frames with prob < {min_confidence} excluded from state fractions)")

    # Build per-video summary + transition matrices
    rows = []
    trans_rows = []  # flattened transition probabilities per video
    for stem in sorted(index.keys()):
        labels_path = f"results/shared/{stem}_labels.npy"
        if not os.path.exists(labels_path):
            continue
        labels = np.load(labels_path)

        if min_confidence > 0.0:
            probs_path = f"results/shared/{stem}_probs.npy"
            if os.path.exists(probs_path):
                probs = np.load(probs_path)
                valid = (labels >= 0) & (probs >= min_confidence)
            else:
                valid = labels >= 0
            denom = int(valid.sum())
            row = {"stem": stem}
            for k in range(n_clusters):
                row[f"state_{k}_frac"] = float((labels[valid] == k).sum() / denom) if denom > 0 else 0.0
        else:
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
# Step 4: Per-animal scalar summary (delegates to quantify.py)
# ---------------------------------------------------------------------------

def cmd_summarize():
    """
    Deprecated thin wrapper — delegates to quantify.py build_master_table().

    Kept for CLI backwards compatibility. Prefer:
        python quantify.py --build
    """
    print("NOTE: --summarize now delegates to quantify.py build_master_table().")
    print("      For the full master table, run: python quantify.py --build")
    print()
    try:
        from quantify import build_master_table
        build_master_table()
    except ImportError:
        sys.exit("[ERROR] quantify.py not found in project directory.")


def cmd_quantify(cohort: str | None = None, min_confidence: float = 0.0):
    """Build master_table.csv via quantify.py."""
    try:
        from quantify import build_master_table, compute_contrast_vector
        build_master_table(cohort_path=cohort, min_confidence=min_confidence)
    except ImportError:
        sys.exit("[ERROR] quantify.py not found in project directory.")

    print("\nComputing behavioral contrast vectors...")
    try:
        contrast_df = compute_contrast_vector(
            summary_csv="results/comparison/summary_table.csv",
            output_dir="results/quantification",
            cohort_csv=cohort,
        )

        master_path = "results/quantification/master_table.csv"
        if os.path.exists(master_path) and "animal_id" in contrast_df.columns:
            master = pd.read_csv(master_path)
            master["animal_id"] = master["animal_id"].astype(str)
            contrast_df["animal_id"] = contrast_df["animal_id"].astype(str)
            master = master.merge(
                contrast_df[["animal_id", "contrast_magnitude",
                             "dominant_fear_state", "dominant_safety_state"]],
                on="animal_id", how="left",
            )
            master.to_csv(master_path, index=False)
            print("contrast_magnitude added to master_table.csv")
    except Exception as e:
        print(f"[WARN] Contrast vector computation failed: {e}")

    print("\nComputing state learning rates...")
    try:
        from quantify import compute_state_learning_rates
        lr_df = compute_state_learning_rates(
            "results/comparison/summary_table.csv",
            output_dir="results/quantification",
            cohort_csv=cohort,
        )
        master_path = "results/quantification/master_table.csv"
        if os.path.exists(master_path) and not lr_df.empty:
            master = pd.read_csv(master_path)
            master["animal_id"] = master["animal_id"].astype(str)
            lr_reset = lr_df.reset_index()
            lr_reset["animal_id"] = lr_reset["animal_id"].astype(str)
            keep_cols = ["animal_id", "fear_learning_rate", "fear_learning_r2"]
            keep_cols = [c for c in keep_cols if c in lr_reset.columns]
            master = master.merge(lr_reset[keep_cols], on="animal_id", how="left")
            master.to_csv(master_path, index=False)
            print("fear_learning_rate added to master_table.csv")
    except Exception as e:
        print(f"[WARN] Learning rate computation failed: {e}")


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
                        help="[deprecated] Use --quantify instead")
    parser.add_argument("--quantify", action="store_true",
                        help="Build master_table.csv with all per-animal scalars")
    parser.add_argument("--collapse", action="store_true",
                        help="Merge similar states by centroid cosine similarity (run after --cluster)")
    parser.add_argument("--collapse-threshold", type=float, default=0.5,
                        help="Cosine similarity threshold for --collapse (default: 0.5)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--n-clusters", type=int, default=None,
                        help="(ignored for HDBSCAN — kept for CLI compatibility)")
    parser.add_argument("--min-cluster-size", type=int, default=50,
                        help="HDBSCAN min_cluster_size (default: 50)")
    parser.add_argument("--hdbscan-min-samples", type=int, default=None,
                        help="HDBSCAN min_samples. Defaults to min_cluster_size if not set.")
    parser.add_argument("--umap-dims", type=int, default=10,
                        help="UMAP n_components (default: 10). Try 3 for better HDBSCAN performance.")
    parser.add_argument("--no-wavelets", action="store_true",
                        help="Skip Morlet wavelet features during --extract (faster)")
    parser.add_argument("--validate", action="store_true",
                        help="With --cluster: run 80/20 train/test split validation (seed=42)")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="With --report/--quantify: exclude frames with prob < threshold")
    parser.add_argument("--cohort", metavar="FILE", default=None,
                        help="Cohort CSV/Excel for --quantify (auto-detected if omitted)")
    args = parser.parse_args()

    if not any([args.extract, args.cluster, args.collapse, args.report, args.summarize, args.quantify]):
        parser.print_help()
        sys.exit(1)

    _print_hardware_banner()

    if args.extract:
        cmd_extract(fps=args.fps, use_wavelets=not args.no_wavelets)
    if args.cluster:
        cmd_cluster(fps=args.fps, min_cluster_size=args.min_cluster_size,
                    min_samples=args.hdbscan_min_samples, umap_dims=args.umap_dims,
                    validate=args.validate)
    if args.collapse:
        cmd_collapse(threshold=args.collapse_threshold)
    if args.report:
        cmd_report(fps=args.fps, min_confidence=args.min_confidence)
    if args.summarize:
        cmd_summarize()
    if args.quantify:
        cmd_quantify(cohort=args.cohort, min_confidence=args.min_confidence)


if __name__ == "__main__":
    main()
