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

import vieb_config as _vc
def _raw(): return _vc.get_raw_videos_dir()
def _res(): return _vc.get_results_dir()
def _meta(): return _vc.get_metadata_path()

ROOT = os.path.dirname(os.path.abspath(__file__))


def _fix_features_path(path: str) -> str:
    """Normalize stored feature paths across machines.

    Older index files may contain absolute Windows paths from another machine.
    Prefer remapping any embedded `results/features/...` suffix onto this
    machine's project ROOT, then fall back to the current results/features
    directory using the filename.
    """
    if not path:
        return path
    if os.path.exists(path):
        return path

    normalized = path.replace("\\", os.sep)
    if os.path.exists(normalized):
        return normalized

    marker = f"results{os.sep}features{os.sep}"
    lower_normalized = normalized.lower()
    marker_idx = lower_normalized.rfind(marker)
    if marker_idx >= 0:
        root_relative = normalized[marker_idx:]
        root_path = os.path.join(ROOT, root_relative)
        if os.path.exists(root_path):
            return root_path

    return os.path.join(_res(), "features", os.path.basename(normalized))

# ---------------------------------------------------------------------------
# GPU detection and hardware banner
# ---------------------------------------------------------------------------

def _detect_gpu() -> bool:
    """Return True if cuML (RAPIDS) is importable and CUDA is actually usable."""
    if platform.system() == "Windows":
        return False  # cuML has no Windows wheels; requires WSL2
    try:
        import cuml  # noqa: F401
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        # Trigger elementwise kernel compilation — same path cuML UMAP.fit() uses.
        # cp.array() alone doesn't compile kernels; cp.copyto / cp.full do.
        import numpy as _np
        _a = cp.zeros(4, dtype=_np.float32)
        cp.copyto(_a, cp.full(4, 1.0, dtype=_np.float32))
        return True
    except Exception:
        return False


def _cuml_importable() -> bool:
    """Return True if cuML can be imported (regardless of driver compatibility)."""
    try:
        import cuml  # noqa: F401
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
        elif _cuml_importable():
            print("        cuML installed but GPU unusable (driver/CUDA version mismatch) — using CPU")
        else:
            print("        cuML not installed — install via pip for GPU acceleration")
    else:
        print("  GPU : none detected")
    print("=" * 60)
    print()



# ---------------------------------------------------------------------------
# Kinematic label generation — derives descriptive state names from data
# ---------------------------------------------------------------------------

# Feature indices for key scalar kinematic features (8 keypoints → first 36
# are per-keypoint speeds + pairwise distances).
_KIN_FEATURES = {
    36: "velocity",
    38: "elongation",
    39: "angular velocity",
    40: "movement variability",
    41: "rearing",
}


def _generate_kinematic_labels(
    cluster_centers: list[list[float]],
    bout_stats: dict[int, dict],
    n_keypoints: int = 8,
) -> dict[int, str]:
    """Derive descriptive heuristic labels from cluster center kinematics.

    For each state, finds the 2 most distinctive features (by z-score relative
    to the population of states) and produces labels like "high velocity, low
    elongation" or "short bouts, high rearing".
    """
    centers = np.array(cluster_centers, dtype=np.float64)
    n_states, n_feat = centers.shape

    # Adjust feature indices based on actual keypoint count
    n_speeds = n_keypoints
    n_dists = n_keypoints * (n_keypoints - 1) // 2
    base = n_speeds + n_dists
    kin_features = {
        base + 0: "velocity",
        base + 2: "elongation",
        base + 3: "angular velocity",
        base + 4: "movement variability",
        base + 5: "rearing",
    }

    labels: dict[int, str] = {}
    for k in range(n_states):
        descriptors: list[tuple[float, float, str]] = []

        for feat_idx, name in kin_features.items():
            if feat_idx >= n_feat:
                continue
            values = centers[:, feat_idx]
            std = values.std()
            if std < 1e-8:
                continue
            z = (centers[k, feat_idx] - values.mean()) / std
            descriptors.append((abs(z), z, name))

        # Bout duration
        dur = bout_stats.get(k, {}).get("mean_dur")
        if dur is not None:
            all_durs = [v["mean_dur"] for v in bout_stats.values()
                        if v.get("mean_dur") is not None]
            if len(all_durs) > 1:
                dur_std = np.std(all_durs)
                if dur_std > 1e-8:
                    z = (dur - np.mean(all_durs)) / dur_std
                    descriptors.append((abs(z), z, "bout duration"))

        descriptors.sort(key=lambda x: x[0], reverse=True)

        parts: list[str] = []
        for _, z, name in descriptors[:2]:
            if name == "bout duration":
                if z > 0.5:
                    parts.append("long bouts")
                elif z < -0.5:
                    parts.append("short bouts")
            else:
                if z > 0.5:
                    parts.append(f"high {name}")
                elif z < -0.5:
                    parts.append(f"low {name}")

        labels[k] = ", ".join(parts) if parts else f"State {k}"

    return labels


def _extract_kinematic_values(
    center: list[float],
    n_keypoints: int = 8,
) -> dict[str, float]:
    """Extract key kinematic feature values from a single cluster center."""
    arr = np.array(center, dtype=np.float64)
    n_speeds = n_keypoints
    n_dists = n_keypoints * (n_keypoints - 1) // 2
    base = n_speeds + n_dists

    mapping = {
        "mean_centroid_speed": base + 0,
        "mean_elongation": base + 2,
        "mean_angular_vel": base + 3,
        "mean_movement_entropy": base + 4,
        "mean_rearing_score": base + 5,
        "mean_head_angle": base + 6,
    }
    result: dict[str, float] = {}
    for col_name, idx in mapping.items():
        if idx < len(arr):
            result[col_name] = float(arr[idx])
        else:
            result[col_name] = float("nan")
    return result


# Step 1: Feature extraction

def _load_extractor_config():
    """Load keypoint_roles, object_keypoints (from config.json) and
    bodypart_names (from DLC config.yaml, if present)."""
    keypoint_roles = {}
    object_keypoints = []
    try:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        with open(cfg_path, encoding="utf-8") as _f:
            _cfg_data = json.load(_f)
            keypoint_roles = _cfg_data.get("keypoint_roles", {})
            object_keypoints = _cfg_data.get("object_keypoints", [])
    except Exception:
        pass

    bodypart_names = None
    try:
        dlc_cfg_path = _vc.get_dlc_config_path()
        if dlc_cfg_path and os.path.exists(dlc_cfg_path):
            import yaml as _yaml
            with open(dlc_cfg_path, encoding="utf-8") as _f:
                _dlc_cfg = _yaml.safe_load(_f)
            bodypart_names = _dlc_cfg.get("bodyparts") or None
    except Exception:
        pass

    return keypoint_roles, object_keypoints, bodypart_names


def _cmd_extract_h5(fps: float = 30.0, use_wavelets: bool = True):
    """Feature extraction from a single shared H5 pose file (video-less mode).

    For standard multi-key H5 files, iterates metadata.csv rows and resolves
    each row to a key inside the H5 file. For concatenated-table H5 files,
    iterates the unique session/source values directly from the H5.
    """
    from ml import PoseFeatureExtractor
    from pose_io import inspect_h5, load_pose_h5
    from h5_manifest import detect_concatenated_table, load_manifest, resolve_h5_key

    h5_path = _vc.get_h5_path()
    if not h5_path or not os.path.exists(h5_path):
        sys.exit(f"H5 pose source configured but file not found: {h5_path!r}")

    if not os.path.exists(_meta()):
        sys.exit(f"H5 mode requires a metadata CSV. Not found: {_meta()}")

    meta = pd.read_csv(_meta(), dtype=str).fillna("")
    meta = _vc.normalize_metadata_columns(meta)

    h5_info = inspect_h5(h5_path)
    h5_keys = h5_info["keys"]
    if not h5_keys:
        sys.exit(f"No keys found in H5 file: {h5_path}")

    configured_source_col = _vc.get_h5_source_col() or None
    session_source_col = configured_source_col or "source_file"
    concatenated_key = detect_concatenated_table(
        h5_info,
        h5_source_col=session_source_col,
    )
    manifest_value_col = session_source_col if concatenated_key is not None else "h5_key"
    manifest = load_manifest(_vc.get_h5_manifest_path(), value_col=manifest_value_col)

    os.makedirs(os.path.join(_res(), "features"), exist_ok=True)

    index_path = os.path.join(_res(), "features", "index.json")
    index = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)

    keypoint_roles, object_keypoints, bodypart_names = _load_extractor_config()

    extractor = None
    new_count = 0
    skip_count = 0

    if not use_wavelets:
        print("(wavelets disabled)")
    if concatenated_key is not None:
        with pd.HDFStore(h5_path, mode="r") as store:
            raw_key = (
                concatenated_key if concatenated_key.startswith("/")
                else f"/{concatenated_key}"
            )
            df = store[raw_key]
        if session_source_col not in df.columns:
            sys.exit(
                f"Concatenated H5 key '{concatenated_key}' does not contain "
                f"column {session_source_col!r}"
            )
        source_values = [
            str(v)
            for v in df[session_source_col].dropna().astype(str).unique().tolist()
            if str(v)
        ]
        print(
            f"Extracting features from {len(source_values)} H5 sessions "
            f"(concatenated mode: {h5_path}, key={concatenated_key})..."
        )
        from pose_io import _pose_from_dlc_df, _pose_from_flat_df

        for source_value in source_values:
            stem = os.path.splitext(os.path.basename(source_value))[0]
            out_path = os.path.join(_res(), "features", f"{stem}_features.npy")

            if os.path.exists(out_path):
                skip_count += 1
                continue

            session_df = df[df[session_source_col] == source_value]
            if session_df.empty:
                print(
                    f"  SKIP (no rows found for {session_source_col}={source_value!r})"
                )
                continue

            print(f"  {stem}  ({session_source_col}={source_value!r})")
            if isinstance(session_df.columns, pd.MultiIndex) and session_df.columns.nlevels >= 3:
                pose, conf, h5_bodyparts = _pose_from_dlc_df(session_df)
            else:
                pose, conf, h5_bodyparts = _pose_from_flat_df(session_df)

            if extractor is None:
                extractor = PoseFeatureExtractor(
                    fps=fps,
                    use_wavelets=use_wavelets,
                    keypoint_roles=keypoint_roles,
                    bodypart_names=bodypart_names or h5_bodyparts,
                    object_keypoints=object_keypoints,
                )

            features_dict = extractor.extract_features(pose, confidence=conf)
            features_flat = extractor._flatten_features(features_dict)

            np.save(out_path, features_flat.astype(np.float32))
            index[stem] = {
                "video_path": None,
                "csv_path": None,
                "h5_path": h5_path,
                "h5_key": source_value,
                "n_frames": int(pose.shape[0]),
                "n_keypoints": int(pose.shape[1]),
                "n_features": int(features_flat.shape[1]),
                "features_path": out_path,
            }
            new_count += 1
    else:
        print(f"Extracting features from {len(meta)} metadata rows (H5 mode: {h5_path})...")
        for i, (_, row) in enumerate(meta.iterrows()):
            row_dict = row.to_dict()
            filename = row_dict.get("filename", "")
            stem = (
                os.path.splitext(filename)[0]
                if filename else row_dict.get("animal_id", f"row{i}")
            )
            out_path = os.path.join(_res(), "features", f"{stem}_features.npy")

            if os.path.exists(out_path):
                skip_count += 1
                continue

            try:
                h5_key, strategy = resolve_h5_key(row_dict, h5_keys, manifest, i)
            except ValueError as e:
                print(f"  SKIP ({e})")
                continue

            print(f"  {stem}  (h5_key={h5_key!r}, match={strategy})")
            pose, conf, h5_bodyparts = load_pose_h5(
                h5_path,
                key=h5_key,
                source_col=configured_source_col,
            )

            if extractor is None:
                extractor = PoseFeatureExtractor(
                    fps=fps,
                    use_wavelets=use_wavelets,
                    keypoint_roles=keypoint_roles,
                    bodypart_names=bodypart_names or h5_bodyparts,
                    object_keypoints=object_keypoints,
                )

            features_dict = extractor.extract_features(pose, confidence=conf)
            features_flat = extractor._flatten_features(features_dict)

            np.save(out_path, features_flat.astype(np.float32))
            index[stem] = {
                "video_path": None,
                "csv_path": None,
                "h5_path": h5_path,
                "h5_key": h5_key,
                "n_frames": int(pose.shape[0]),
                "n_keypoints": int(pose.shape[1]),
                "n_features": int(features_flat.shape[1]),
                "features_path": out_path,
            }
            new_count += 1

    first_entry = next((v for k, v in index.items() if k != '_meta'), {})
    if extractor is not None:
        feat_meta = extractor.get_feature_meta(
            int(first_entry.get("n_keypoints", 8))
        )
    else:
        feat_meta = {
            "n_keypoints": int(first_entry.get("n_keypoints", 8)),
            "n_features": int(first_entry.get("n_features", 91)),
            "use_wavelets": use_wavelets,
            "feature_names": [],
            "semantic_features": [],
        }
    index["_meta"] = {
        "n_keypoints": feat_meta["n_keypoints"],
        "n_features": feat_meta["n_features"],
        "use_wavelets": feat_meta["use_wavelets"],
        "feature_names": feat_meta["feature_names"],
        "semantic_features": feat_meta["semantic_features"],
        "vieb_version": "1.0",
        "pose_source": "h5",
    }

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone. Extracted {new_count} new, skipped {skip_count} already done.")
    print(f"Total in index: {len(index)} videos")
    print(f"Feature files saved to results/features/")


def _check_metadata_before_extract():
    """Warn (non-fatal) if metadata.csv is missing required animal_id/context
    values. Feature extraction can still proceed without metadata, but
    downstream comparison/quantification will fail or be meaningless."""
    from metadata_generator import validate_metadata_csv

    meta_path = _meta()
    if not os.path.exists(meta_path):
        print(f"[warn] metadata.csv not found at {meta_path} — "
              f"downstream comparison steps will need it.")
        return

    report = validate_metadata_csv(meta_path)
    if not report["valid"]:
        print(f"[warn] metadata.csv has incomplete rows ({meta_path}):")
        for msg in report["messages"]:
            print(f"  - {msg}")
        print("  Fill in these rows before running 'compare.py --report' or '--quantify'.")


def cmd_extract(fps: float = 30.0, use_wavelets: bool = True):
    from ml import PoseFeatureExtractor
    from pose_io import load_pose, _find_dlc_csv

    _check_metadata_before_extract()

    pose_source = _vc.get_pose_source()

    if pose_source == "h5":
        return _cmd_extract_h5(fps=fps, use_wavelets=use_wavelets)

    videos = sorted(glob.glob(os.path.join(_raw(), "*.mp4")))
    if not videos:
        sys.exit("No .mp4 files found in raw_videos/")

    os.makedirs(os.path.join(_res(), "features"), exist_ok=True)

    index_path = os.path.join(_res(), "features", "index.json")
    index = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)

    keypoint_roles, object_keypoints, bodypart_names = _load_extractor_config()

    extractor = PoseFeatureExtractor(
        fps=fps,
        use_wavelets=use_wavelets,
        keypoint_roles=keypoint_roles,
        bodypart_names=bodypart_names,
        object_keypoints=object_keypoints,
    )
    new_count = 0
    skip_count = 0

    if not use_wavelets:
        print("(wavelets disabled)")
    print(f"Extracting features from {len(videos)} videos...")
    for video_path in videos:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        out_path = os.path.join(_res(), "features", f"{stem}_features.npy")

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
            "n_keypoints": int(pose.shape[1]),
            "n_features": int(features_flat.shape[1]),
            "features_path": out_path,
        }
        new_count += 1

    first_entry = next((v for k, v in index.items() if k != '_meta'), {})
    feat_meta = extractor.get_feature_meta(
        int(first_entry.get("n_keypoints", 8))
    )
    index["_meta"] = {
        "n_keypoints": feat_meta["n_keypoints"],
        "n_features": feat_meta["n_features"],
        "use_wavelets": feat_meta["use_wavelets"],
        "feature_names": feat_meta["feature_names"],
        "semantic_features": feat_meta["semantic_features"],
        "vieb_version": "1.0",
    }

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone. Extracted {new_count} new, skipped {skip_count} already done.")
    print(f"Total in index: {len(index)} videos")
    print(f"Feature files saved to results/features/")


def _load_pose_for_index_entry(stem: str, entry: dict):
    """Load (pose, conf, bodyparts) for an index.json entry, using its
    recorded pose source (CSV/DLC or shared H5)."""
    from pose_io import load_pose, load_pose_h5, _find_dlc_csv

    if entry.get("h5_path"):
        return load_pose_h5(
            entry["h5_path"],
            key=entry.get("h5_key"),
            source_col=_vc.get_h5_source_col() or None,
        )

    pose_path = entry.get("csv_path")
    if not pose_path or not os.path.exists(pose_path):
        video_path = entry.get("video_path")
        pose_path = _find_dlc_csv(video_path) if video_path else None
    if not pose_path or not os.path.exists(pose_path):
        return None
    return load_pose(pose_path)


def cmd_fix_features(fps: float = 30.0):
    """Re-extract feature files for videos whose feature dimension does not
    match the current config.json extraction settings (use_wavelets).

    Only the mismatched videos are re-extracted — videos that already match
    the target dimension are left untouched. Updates index["_meta"]
    afterwards so it reflects the now-consistent settings.
    """
    from collections import Counter
    from ml import PoseFeatureExtractor

    index_path = os.path.join(_res(), "features", "index.json")
    if not os.path.exists(index_path):
        sys.exit("No index found. Run --extract first.")
    with open(index_path) as f:
        index = json.load(f)

    stems = sorted(k for k in index.keys() if k != "_meta")
    if not stems:
        sys.exit("Index is empty. Run --extract first.")

    counts = Counter(index[s].get("n_features") for s in stems)
    print(f"Feature dimensions across {len(stems)} videos: {dict(counts)}")
    if len(counts) <= 1:
        print("All videos already have a consistent feature dimension. Nothing to fix.")
        return

    use_wavelets = _vc.get_use_wavelets()
    keypoint_roles, object_keypoints, bodypart_names = _load_extractor_config()
    print(f"Re-extracting mismatched videos using config.json settings "
          f"(use_wavelets={use_wavelets})...")

    extractor = PoseFeatureExtractor(
        fps=fps,
        use_wavelets=use_wavelets,
        keypoint_roles=keypoint_roles,
        bodypart_names=bodypart_names,
        object_keypoints=object_keypoints,
    )

    # ---- Determine the target feature dimension by extracting one video ----
    target_n_features = None
    target_n_keypoints = None
    for stem in stems:
        result = _load_pose_for_index_entry(stem, index[stem])
        if result is None:
            continue
        pose, conf, _ = result
        features_dict = extractor.extract_features(pose, confidence=conf)
        features_flat = extractor._flatten_features(features_dict)
        target_n_features = int(features_flat.shape[1])
        target_n_keypoints = int(pose.shape[1])
        if index[stem].get("n_features") != target_n_features:
            out_path = index[stem]["features_path"]
            np.save(out_path, features_flat.astype(np.float32))
            index[stem]["n_features"] = target_n_features
            index[stem]["n_keypoints"] = target_n_keypoints
            index[stem]["n_frames"] = int(pose.shape[0])
            print(f"  Re-extracted {stem}: {target_n_features} features")
        break

    if target_n_features is None:
        sys.exit("Could not load pose data for any video; cannot determine target feature dimension.")

    print(f"Target dimension: {target_n_features} features ({target_n_keypoints} keypoints)")

    # ---- Re-extract every remaining video whose dimension doesn't match ----
    mismatched = [s for s in stems if index[s].get("n_features") != target_n_features]
    print(f"Re-extracting {len(mismatched)} mismatched video(s)...")

    fixed_count = 0
    skip_count = 0
    for stem in mismatched:
        result = _load_pose_for_index_entry(stem, index[stem])
        if result is None:
            print(f"  SKIP {stem}: pose data not found, cannot re-extract")
            skip_count += 1
            continue
        pose, conf, _ = result
        features_dict = extractor.extract_features(pose, confidence=conf)
        features_flat = extractor._flatten_features(features_dict)
        if features_flat.shape[1] != target_n_features:
            print(f"  WARNING {stem}: re-extraction produced {features_flat.shape[1]} features "
                  f"(expected {target_n_features}); leaving unchanged")
            skip_count += 1
            continue

        out_path = index[stem]["features_path"]
        np.save(out_path, features_flat.astype(np.float32))
        index[stem]["n_features"] = int(features_flat.shape[1])
        index[stem]["n_keypoints"] = int(pose.shape[1])
        index[stem]["n_frames"] = int(pose.shape[0])
        fixed_count += 1
        print(f"  Re-extracted {stem}: {features_flat.shape[1]} features")

    # ---- Update index metadata to reflect the now-consistent settings ----
    old_meta = index.get("_meta", {})
    new_meta = {
        "n_keypoints": target_n_keypoints,
        "n_features": target_n_features,
        "use_wavelets": use_wavelets,
        "vieb_version": old_meta.get("vieb_version", "1.0"),
    }
    if "pose_source" in old_meta:
        new_meta["pose_source"] = old_meta["pose_source"]
    index["_meta"] = new_meta

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    counts_after = Counter(index[s].get("n_features") for s in stems)
    print(f"\nDone. Re-extracted {fixed_count}, skipped {skip_count}.")
    print(f"Feature dimensions after fix: {dict(counts_after)}")
    if len(counts_after) > 1:
        print("WARNING: dimensions are still inconsistent. Some videos may be "
              "missing pose data — re-run after restoring it, or re-extract "
              "from scratch with --extract.")


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
    os.makedirs(os.path.join(_res(), "shared"), exist_ok=True)
    with open(os.path.join(_res(), "shared", "validation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nValidation report saved: results/shared/validation_report.json")


# ---------------------------------------------------------------------------
# Run versioning helpers
# ---------------------------------------------------------------------------

def _next_run_n(runs_dir: str) -> int:
    """Return the next integer N for naming a run directory."""
    max_n = 0
    if os.path.isdir(runs_dir):
        for name in os.listdir(runs_dir):
            if name.startswith("run_") and os.path.isdir(os.path.join(runs_dir, name)):
                try:
                    n = int(name.split("_")[1])
                    max_n = max(max_n, n)
                except (IndexError, ValueError):
                    pass
    return max_n + 1


def _auto_save_previous_run() -> str | None:
    """Copy results/shared/ to results/runs/{run_id}/ before a new run overwrites it."""
    import shutil
    from datetime import datetime as _dt

    shared_dir = os.path.join(_res(), "shared")
    cluster_info_path = os.path.join(shared_dir, "cluster_info.json")
    if not os.path.exists(cluster_info_path):
        return None

    with open(cluster_info_path) as f:
        ci = json.load(f)

    manifest_path = os.path.join(shared_dir, "run_manifest.json")
    existing_manifest: dict = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            existing_manifest = json.load(f)

    prev_mcs = existing_manifest.get("min_cluster_size", ci.get("min_cluster_size", 50))
    prev_umap = existing_manifest.get("umap_dims", ci.get("umap_dims", 10))
    prev_hdbscan_sample = existing_manifest.get(
        "hdbscan_sample",
        ci.get("hdbscan_sample", 0),
    )

    runs_dir = os.path.join(_res(), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    n = _next_run_n(runs_dir)
    now = _dt.now()
    run_id = f"run_{n:03d}_{now.strftime('%Y%m%d_%H%M')}_mcs{prev_mcs}_umap{prev_umap}"
    run_dir = os.path.join(runs_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    for fname in os.listdir(shared_dir):
        src = os.path.join(shared_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(run_dir, fname))

    # Compute noise_frac from existing label files
    noise_frac = 0.0
    try:
        total_frames = 0
        noise_frames = 0
        for fname in os.listdir(shared_dir):
            if fname.endswith("_labels.npy"):
                lbl = np.load(os.path.join(shared_dir, fname))
                total_frames += len(lbl)
                noise_frames += int((lbl == -1).sum())
        if total_frames > 0:
            noise_frac = float(noise_frames / total_frames)
    except Exception:
        pass

    manifest = {
        "run_id": run_id,
        "date": existing_manifest.get("date", now.strftime("%Y-%m-%d %H:%M")),
        "min_cluster_size": int(prev_mcs),
        "umap_dims": int(prev_umap),
        "hdbscan_min_samples": int(existing_manifest.get("hdbscan_min_samples", 0)),
        "hdbscan_sample": int(prev_hdbscan_sample),
        "n_clusters": int(ci.get("n_clusters", 0)),
        "mean_confidence": float(ci.get("mean_confidence", 0.0)),
        "low_confidence_frac": float(ci.get("low_confidence_frac", 0.0)),
        "noise_frac": round(noise_frac, 4),
        "saved": False,
    }
    with open(os.path.join(run_dir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Auto-saved previous run to results/runs/{run_id}/")
    return run_id


def _write_current_run_manifest(
    min_cluster_size: int,
    umap_dims: int,
    effective_min_samples: int,
    hdbscan_sample: int,
    n_found: int,
    mean_conf: float,
    low_conf_frac: float,
    noise_frac: float,
) -> str:
    """Write results/shared/run_manifest.json and update config.json."""
    from datetime import datetime as _dt

    runs_dir = os.path.join(_res(), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    n = _next_run_n(runs_dir)
    now = _dt.now()
    run_id = f"run_{n:03d}_{now.strftime('%Y%m%d_%H%M')}_mcs{min_cluster_size}_umap{umap_dims}"

    manifest = {
        "run_id": run_id,
        "date": now.strftime("%Y-%m-%d %H:%M"),
        "min_cluster_size": min_cluster_size,
        "umap_dims": umap_dims,
        "hdbscan_min_samples": effective_min_samples,
        "hdbscan_sample": hdbscan_sample,
        "n_clusters": n_found,
        "mean_confidence": round(mean_conf, 4),
        "low_confidence_frac": round(low_conf_frac, 4),
        "noise_frac": round(noise_frac, 4),
        "saved": False,
    }
    with open(os.path.join(_res(), "shared", "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    cfg_data: dict = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg_data = json.load(f)
        except Exception:
            pass
    cfg_data["current_run_saved"] = False
    cfg_data["current_run_id"] = run_id
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2)

    return run_id


def _extract_hdbscan_outputs(clusterer_model) -> tuple[np.ndarray, np.ndarray]:
    """Normalize HDBSCAN label/probability outputs across CPU/GPU backends."""
    labels = clusterer_model.labels_
    if hasattr(labels, "to_numpy"):
        labels = labels.to_numpy()
    elif hasattr(labels, "get"):
        labels = labels.get()
    labels = np.asarray(labels, dtype=np.int32)

    probs = getattr(clusterer_model, "probabilities_", None)
    if probs is not None:
        if hasattr(probs, "to_numpy"):
            probs = probs.to_numpy()
        elif hasattr(probs, "get"):
            probs = probs.get()
        probs = np.asarray(probs, dtype=np.float32)
    else:
        probs = np.where(labels >= 0, 1.0, 0.0).astype(np.float32)

    return labels, probs


def _batched_approximate_predict(clusterer_model, points, batch_size=25_000):
    """Batch approximate_predict to avoid OOM on multi-million-frame datasets.

    HDBSCAN's prediction path allocates temporary nearest-neighbor heaps whose
    size scales with both batch size and `2 * min_samples`. Conservative,
    adaptive batching keeps assignment feasible after sampled HDBSCAN fitting.
    """
    from hdbscan import approximate_predict

    if len(points) <= batch_size:
        return approximate_predict(clusterer_model, points)
    all_labels = np.empty(len(points), dtype=np.int32)
    all_probs = np.empty(len(points), dtype=np.float32)
    start = 0
    current_batch_size = int(max(1, batch_size))
    while start < len(points):
        end = min(start + current_batch_size, len(points))
        try:
            batch_labels, batch_probs = approximate_predict(
                clusterer_model,
                points[start:end],
            )
        except MemoryError:
            if current_batch_size <= 1_000:
                raise
            current_batch_size = max(1_000, current_batch_size // 2)
            print(
                f"  [hdbscan] approximate_predict OOM at batch size "
                f"{end - start:,}; retrying with {current_batch_size:,}..."
            )
            continue
        except Exception as e:
            if (
                "Unable to allocate" in str(e)
                or e.__class__.__name__ == "_ArrayMemoryError"
            ) and current_batch_size > 1_000:
                current_batch_size = max(1_000, current_batch_size // 2)
                print(
                    f"  [hdbscan] approximate_predict memory pressure at batch size "
                    f"{end - start:,}; retrying with {current_batch_size:,}..."
                )
                continue
            raise

        all_labels[start:end] = np.asarray(batch_labels, dtype=np.int32)
        all_probs[start:end] = np.asarray(batch_probs, dtype=np.float32)
        start = end
    return all_labels, all_probs


def _fit_cpu_hdbscan_with_assignment(
    HDBSCANClass,
    pooled_umap: np.ndarray,
    fit_indices: np.ndarray,
    predict_indices: np.ndarray,
    min_cluster_size: int,
    effective_min_samples: int,
) -> tuple[object, np.ndarray, np.ndarray]:
    """
    Fit CPU HDBSCAN on `fit_indices`, then assign any remaining frames.

    UMAP already uses a fitting sample for memory reasons. HDBSCAN needs its
    own sampling control because clustering the full embedding can still blow
    up on multi-million-frame datasets. On CPU, `approximate_predict()` lets
    us fit on a manageable subset while reconstructing labels for every frame.
    """
    clusterer_model = HDBSCANClass(
        min_cluster_size=min_cluster_size,
        min_samples=effective_min_samples,
        cluster_selection_method="eom",
        prediction_data=True,
    )
    clusterer_model.fit(pooled_umap[fit_indices])
    fit_labels, fit_probs = _extract_hdbscan_outputs(clusterer_model)

    all_raw_labels = np.full(len(pooled_umap), -1, dtype=np.int32)
    all_probs = np.zeros(len(pooled_umap), dtype=np.float32)
    all_raw_labels[fit_indices] = fit_labels
    all_probs[fit_indices] = fit_probs

    if len(predict_indices) > 0:
        pred_labels, pred_probs = _batched_approximate_predict(
            clusterer_model,
            pooled_umap[predict_indices],
        )
        all_raw_labels[predict_indices] = np.asarray(pred_labels, dtype=np.int32)
        all_probs[predict_indices] = np.asarray(pred_probs, dtype=np.float32)

    return clusterer_model, all_raw_labels, all_probs


def _mark_run_saved() -> None:
    """Set saved=true in results/shared/run_manifest.json and config.json."""
    manifest_path = os.path.join(_res(), "shared", "run_manifest.json")
    if not os.path.exists(manifest_path):
        print("[warn] No run_manifest.json found — nothing to mark as saved.")
        return
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["saved"] = True
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    cfg_data: dict = {}
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg_data = json.load(f)
        except Exception:
            pass
    cfg_data["current_run_saved"] = True
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg_data, f, indent=2)
    print("Run saved.")


def cmd_cluster(
    fps: float = 30.0,
    n_clusters: int = None,
    min_cluster_size: int = 50,
    min_samples: int = None,
    umap_dims: int = 10,
    validate: bool = False,
    hdbscan_sample: int = 300000,
):
    import joblib
    from ml import BehaviorPreprocessor

    hdbscan_sample = max(1, int(hdbscan_sample))

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

    index_path = os.path.join(_res(), "features", "index.json")
    if not os.path.exists(index_path):
        sys.exit("No index found. Run --extract first.")
    with open(index_path) as f:
        index = json.load(f)
    if not index:
        sys.exit("Index is empty. Run --extract first.")

    # ---- Validate feature dimension consistency ----
    meta_info = index.get("_meta")
    if meta_info:
        expected_n_features = meta_info.get("n_features")
        expected_n_keypoints = meta_info.get("n_keypoints")
    else:
        first_entry = next((v for k, v in index.items() if k != '_meta'), {})
        expected_n_features = first_entry.get("n_features")
        expected_n_keypoints = first_entry.get("n_keypoints")

    mismatches = []
    for stem, entry in index.items():
        if stem == '_meta':
            continue
        nf = entry.get("n_features")
        if nf is not None and expected_n_features is not None and nf != expected_n_features:
            mismatches.append((stem, nf, entry.get("n_keypoints", "?")))

    if mismatches:
        exp_nk = expected_n_keypoints if expected_n_keypoints is not None else "?"
        print(f"[ERROR] Feature dimension mismatch across videos:")
        print(f"  Expected: {expected_n_features} features ({exp_nk} keypoints)")
        for stem, nf, nk in mismatches:
            print(f"  Mismatch: {stem} → {nf} features ({nk} keypoints)")
        print("Re-run compare.py --extract to regenerate features with consistent settings.")
        sys.exit(1)
    else:
        nk_str = f"{expected_n_keypoints} keypoints" if expected_n_keypoints is not None else "unknown keypoints"
        print(f"[OK] Feature dimensions consistent: {expected_n_features} features, {nk_str}")

    os.makedirs(os.path.join(_res(), "shared"), exist_ok=True)

    # ---- Run versioning: auto-save previous run if it exists and wasn't saved ----
    _prev_cluster_info = os.path.join(_res(), "shared", "cluster_info.json")
    if os.path.exists(_prev_cluster_info):
        _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        _cfg_data: dict = {}
        if os.path.exists(_cfg_path):
            try:
                with open(_cfg_path, encoding="utf-8") as _f:
                    _cfg_data = json.load(_f)
            except Exception:
                pass
        if not _cfg_data.get("current_run_saved", False):
            _auto_save_previous_run()

    # ---- Load all feature matrices ----
    stems = sorted(k for k in index.keys() if k != '_meta')

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
        feat_path = _fix_features_path(index[stem].get("features_path", ""))
        if not os.path.exists(feat_path):
            print(f"  SKIP {stem}: feature file not found at {feat_path}")
            continue
        feat = np.load(feat_path)
        boundaries[stem] = (cursor, cursor + len(feat))
        cursor += len(feat)
        all_features.append(feat)

    if not all_features:
        sys.exit("No feature files could be loaded. Check results/features/index.json.")

    stems = [s for s in stems if s in boundaries]
    train_stems = [s for s in train_stems if s in boundaries]
    test_stems = [s for s in test_stems if s in boundaries]

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

    preprocessor.save(os.path.join(_res(), "shared", "preprocessor.pkl"))
    print(f"  Standardized to {pooled_scaled.shape[1]} features")

    # ---- UMAP reduction ----
    print(f"\nFitting UMAP (n_components={umap_dims}, n_neighbors=30)...")
    umap_save_path = os.path.join(_res(), "shared", "umap_reducer.pkl")
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
        try:
            reducer.fit(fit_data)
            pooled_umap = reducer.transform(pooled_scaled)
        except Exception as _gpu_err:
            if use_gpu:
                print(f"  GPU UMAP failed ({_gpu_err}); falling back to CPU…")
                import umap as _umap_lib, hdbscan as _hdbscan_lib
                use_gpu = False
                UMAPClass = _umap_lib.UMAP
                HDBSCANClass = _hdbscan_lib.HDBSCAN
                umap_kwargs.update(low_memory=True, verbose=False)
                reducer = UMAPClass(**umap_kwargs)
                reducer.fit(fit_data)
                pooled_umap = reducer.transform(pooled_scaled)
            else:
                raise
        # Transform all frames (train + test) through the fitted UMAP
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
        try:
            reducer.fit(fit_data)
            pooled_umap = reducer.transform(pooled_scaled)
        except Exception as _gpu_err:
            if use_gpu:
                print(f"  GPU UMAP failed ({_gpu_err}); falling back to CPU…")
                import umap as _umap_lib, hdbscan as _hdbscan_lib
                use_gpu = False
                UMAPClass = _umap_lib.UMAP
                HDBSCANClass = _hdbscan_lib.HDBSCAN
                umap_kwargs.update(low_memory=True, verbose=True)
                reducer = UMAPClass(**umap_kwargs)
                reducer.fit(fit_data)
                pooled_umap = reducer.transform(pooled_scaled)
            else:
                raise

    if hasattr(pooled_umap, "to_numpy"):
        pooled_umap = pooled_umap.to_numpy()
    elif hasattr(pooled_umap, "get"):
        pooled_umap = pooled_umap.get()
    pooled_umap = np.asarray(pooled_umap, dtype=np.float32)
    joblib.dump(reducer, os.path.join(_res(), "shared", "umap_reducer.pkl"))
    print(f"  UMAP embedding: {pooled_umap.shape}")

    # ---- HDBSCAN clustering ----
    # UMAP and HDBSCAN need separate sampling controls: UMAP can fit on a
    # subset and transform everything cheaply, but HDBSCAN still builds its
    # clustering structure on the embedding it is fit on.
    effective_min_samples = min_samples if min_samples is not None else min_cluster_size
    if use_gpu and effective_min_samples > 1023:
        print(f"  [info] cuML HDBSCAN requires min_samples <= 1023; clamping {effective_min_samples} -> 1023.")
        effective_min_samples = 1023
    print(
        f"\nFitting HDBSCAN (min_cluster_size={min_cluster_size}, "
        f"min_samples={effective_min_samples}, hdbscan_sample={hdbscan_sample})..."
    )

    if validate:
        # Validation still fits only on the train partition; if needed, we
        # subsample within that train partition before HDBSCAN fitting.
        train_fit_indices = np.asarray(train_indices, dtype=np.int64)
        test_indices = np.concatenate([
            np.arange(boundaries[s][0], boundaries[s][1]) for s in test_stems
        ]) if test_stems else np.array([], dtype=np.int64)
        if len(train_fit_indices) > hdbscan_sample:
            rng = np.random.default_rng(42)
            sampled_pos = np.sort(
                rng.choice(len(train_fit_indices), hdbscan_sample, replace=False)
            )
            fit_indices = train_fit_indices[sampled_pos]
            print(
                f"  Fitting HDBSCAN on {len(fit_indices):,}-frame train sample, "
                f"then assigning remaining train/test frames..."
            )
        else:
            fit_indices = train_fit_indices
            print(f"  Fitting HDBSCAN on all {len(fit_indices):,} train frames...")
        actual_hdbscan_sample = int(len(fit_indices))

        predict_mask = np.ones(len(pooled_umap), dtype=bool)
        predict_mask[fit_indices] = False
        predict_indices = np.flatnonzero(predict_mask)

        if use_gpu and len(train_fit_indices) > hdbscan_sample:
            # cuML HDBSCAN has no approximate_predict equivalent, so when we
            # subsample the fit set we fall back to CPU HDBSCAN for full-frame
            # assignment. This preserves downstream outputs and avoids OOM.
            print("  cuML HDBSCAN sampling requires CPU fallback for full-frame assignment...")
            import hdbscan as _hdbscan_lib
            use_gpu = False
            HDBSCANClass = _hdbscan_lib.HDBSCAN

        if use_gpu:
            clusterer_model = HDBSCANClass(
                min_cluster_size=min_cluster_size,
                min_samples=effective_min_samples,
                cluster_selection_method="eom",
            )
            try:
                clusterer_model.fit(pooled_umap[fit_indices])
            except Exception as _gpu_err:
                print(f"  GPU HDBSCAN failed ({_gpu_err}); falling back to CPU…")
                import hdbscan as _hdbscan_lib
                use_gpu = False
                HDBSCANClass = _hdbscan_lib.HDBSCAN
                clusterer_model, all_raw_labels, all_probs = _fit_cpu_hdbscan_with_assignment(
                    HDBSCANClass,
                    pooled_umap,
                    fit_indices,
                    predict_indices,
                    min_cluster_size,
                    effective_min_samples,
                )
            else:
                fit_labels, fit_probs = _extract_hdbscan_outputs(clusterer_model)
                all_raw_labels = np.full(len(pooled_umap), -1, dtype=np.int32)
                all_probs = np.zeros(len(pooled_umap), dtype=np.float32)
                all_raw_labels[fit_indices] = fit_labels
                all_probs[fit_indices] = fit_probs
                if len(test_indices) > 0:
                    try:
                        test_result = clusterer_model.transform(pooled_umap[test_indices])
                        if hasattr(test_result, "to_numpy"):
                            test_result = test_result.to_numpy()
                        elif hasattr(test_result, "get"):
                            test_result = test_result.get()
                        test_raw_labels = np.asarray(
                            test_result[:, 0] if test_result.ndim > 1 else test_result,
                            dtype=np.int32,
                        )
                        test_probs = np.ones(len(test_raw_labels), dtype=np.float32)
                        test_probs[test_raw_labels < 0] = 0.0
                    except Exception:
                        test_raw_labels = np.full(len(test_indices), -1, dtype=np.int32)
                        test_probs = np.zeros(len(test_indices), dtype=np.float32)
                    all_raw_labels[test_indices] = test_raw_labels
                    all_probs[test_indices] = test_probs
        else:
            clusterer_model, all_raw_labels, all_probs = _fit_cpu_hdbscan_with_assignment(
                HDBSCANClass,
                pooled_umap,
                fit_indices,
                predict_indices,
                min_cluster_size,
                effective_min_samples,
            )

    else:
        all_indices = np.arange(len(pooled_umap), dtype=np.int64)
        if len(all_indices) > hdbscan_sample:
            rng = np.random.default_rng(42)
            sampled_pos = np.sort(
                rng.choice(len(all_indices), hdbscan_sample, replace=False)
            )
            fit_indices = all_indices[sampled_pos]
            print(
                f"  Fitting HDBSCAN on {len(fit_indices):,}-frame embedding sample, "
                f"then assigning remaining frames..."
            )
        else:
            fit_indices = all_indices
            print(f"  Fitting HDBSCAN on all {len(fit_indices):,} embedded frames...")
        actual_hdbscan_sample = int(len(fit_indices))

        predict_mask = np.ones(len(pooled_umap), dtype=bool)
        predict_mask[fit_indices] = False
        predict_indices = np.flatnonzero(predict_mask)

        if use_gpu and len(all_indices) > hdbscan_sample:
            # cuML can fit the sample, but it cannot assign every withheld
            # frame without a CPU-style approximate_predict path.
            print("  cuML HDBSCAN sampling requires CPU fallback for full-frame assignment...")
            import hdbscan as _hdbscan_lib
            use_gpu = False
            HDBSCANClass = _hdbscan_lib.HDBSCAN

        if use_gpu:
            clusterer_model = HDBSCANClass(
                min_cluster_size=min_cluster_size,
                min_samples=effective_min_samples,
                cluster_selection_method="eom",
            )
            try:
                clusterer_model.fit(pooled_umap[fit_indices])
            except Exception as _gpu_err:
                print(f"  GPU HDBSCAN failed ({_gpu_err}); falling back to CPU…")
                import hdbscan as _hdbscan_lib
                use_gpu = False
                HDBSCANClass = _hdbscan_lib.HDBSCAN
                clusterer_model, all_raw_labels, all_probs = _fit_cpu_hdbscan_with_assignment(
                    HDBSCANClass,
                    pooled_umap,
                    fit_indices,
                    predict_indices,
                    min_cluster_size,
                    effective_min_samples,
                )
            else:
                all_raw_labels, all_probs = _extract_hdbscan_outputs(clusterer_model)
        else:
            clusterer_model, all_raw_labels, all_probs = _fit_cpu_hdbscan_with_assignment(
                HDBSCANClass,
                pooled_umap,
                fit_indices,
                predict_indices,
                min_cluster_size,
                effective_min_samples,
            )

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

    joblib.dump(clusterer_model, os.path.join(_res(), "shared", "clusterer.pkl"))
    cluster_info = {
        "n_clusters": n_found,
        "cluster_centers": cluster_centers,
        "method": "umap+hdbscan",
        "min_cluster_size": min_cluster_size,
        "hdbscan_sample": actual_hdbscan_sample,
        "mean_confidence": round(mean_conf, 4),
        "low_confidence_frac": round(low_conf_frac, 4),
    }
    with open(os.path.join(_res(), "shared", "cluster_info.json"), "w") as f:
        json.dump(cluster_info, f, indent=2)

    meta_info = dict(index.get("_meta", {}))
    meta_info["hdbscan_sample"] = actual_hdbscan_sample
    index["_meta"] = meta_info
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

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
        np.save(os.path.join(_res(), "shared", f"{stem}_labels.npy"), smoothed.astype(np.int32))
        np.save(os.path.join(_res(), "shared", f"{stem}_probs.npy"), probs.astype(np.float32))

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

    # ---- Write run manifest for this run ----
    _noise_frac = float(n_noise / len(all_raw_labels)) if len(all_raw_labels) > 0 else 0.0
    _current_run_id = _write_current_run_manifest(
        min_cluster_size=min_cluster_size,
        umap_dims=umap_dims,
        effective_min_samples=effective_min_samples,
        hdbscan_sample=actual_hdbscan_sample,
        n_found=n_found,
        mean_conf=mean_conf,
        low_conf_frac=low_conf_frac,
        noise_frac=_noise_frac,
    )
    print(f"Run manifest → results/shared/run_manifest.json  (run_id: {_current_run_id})")

    # ---- Validation report ----
    if validate:
        _run_validation_report(
            stems, train_stems, test_stems, boundaries,
            smoothed_labels_all, raw_probs_all, n_found,
            min_cluster_size,
        )

# ---------------------------------------------------------------------------
# Step 2b: Apply the existing shared cluster model to new videos only
# ---------------------------------------------------------------------------

def cmd_apply_existing(fps: float = 30.0):
    """
    Apply the existing saved preprocessor/UMAP/HDBSCAN models to videos that
    have features extracted but no labels yet. Does not refit anything, so
    cluster IDs for previously processed videos are unchanged.
    """
    import joblib

    shared_dir = os.path.join(_res(), "shared")
    for fname in ("preprocessor.pkl", "umap_reducer.pkl", "clusterer.pkl", "cluster_info.json"):
        if not os.path.exists(os.path.join(shared_dir, fname)):
            sys.exit(f"Missing {fname} in results/shared/. Run 'compare.py --cluster' (full fit) first.")

    index_path = os.path.join(_res(), "features", "index.json")
    if not os.path.exists(index_path):
        sys.exit("No index found. Run --extract first.")
    with open(index_path) as f:
        index = json.load(f)

    stems = sorted(k for k in index.keys() if k != "_meta")
    new_stems = [s for s in stems if not os.path.exists(os.path.join(shared_dir, f"{s}_labels.npy"))]

    if not new_stems:
        print("No new videos to cluster — every video in the feature index already has labels.")
        return

    print(f"Applying existing cluster model to {len(new_stems)} new video(s):")
    for s in new_stems:
        print(f"  {s}")

    preprocessor = joblib.load(os.path.join(shared_dir, "preprocessor.pkl"))
    reducer = joblib.load(os.path.join(shared_dir, "umap_reducer.pkl"))
    clusterer_model = joblib.load(os.path.join(shared_dir, "clusterer.pkl"))
    with open(os.path.join(shared_dir, "cluster_info.json")) as f:
        cluster_info = json.load(f)
    n_found = int(cluster_info.get("n_clusters", 0))

    boundaries = {}
    cursor = 0
    feats = []
    for stem in new_stems:
        feat_path = _fix_features_path(index[stem].get("features_path", ""))
        if not os.path.exists(feat_path):
            print(f"  SKIP {stem}: feature file not found at {feat_path}")
            continue
        feat = np.load(feat_path).astype(np.float64)
        boundaries[stem] = (cursor, cursor + len(feat))
        cursor += len(feat)
        feats.append(feat)

    new_stems = [s for s in new_stems if s in boundaries]
    if not new_stems:
        sys.exit("No feature files could be loaded for the new videos.")

    pooled = np.vstack(feats)
    pooled_scaled = preprocessor.transform(pooled)
    pooled_umap = reducer.transform(pooled_scaled)
    if hasattr(pooled_umap, "to_numpy"):
        pooled_umap = pooled_umap.to_numpy()
    elif hasattr(pooled_umap, "get"):
        pooled_umap = pooled_umap.get()
    pooled_umap = np.asarray(pooled_umap, dtype=np.float32)

    try:
        raw_labels, raw_probs = _batched_approximate_predict(clusterer_model, pooled_umap)
        raw_labels = np.asarray(raw_labels, dtype=np.int32)
        raw_probs = np.asarray(raw_probs, dtype=np.float32)
    except Exception as e:
        print(f"  [WARN] approximate_predict failed: {e}. Marking all new frames as noise.")
        raw_labels = np.full(len(pooled_umap), -1, dtype=np.int32)
        raw_probs = np.zeros(len(pooled_umap), dtype=np.float32)

    raw_labels_all = []
    raw_probs_all = []
    for stem in new_stems:
        start, end = boundaries[stem]
        raw_labels_all.append(raw_labels[start:end])
        raw_probs_all.append(raw_probs[start:end])

    print("\nSmoothing labels with HMM...")
    all_raw_concat = np.concatenate(raw_labels_all)
    valid_labels = all_raw_concat[all_raw_concat >= 0]
    if len(valid_labels) > 0 and n_found > 1:
        hmm_params = _fit_hmm(valid_labels, n_found)
        smoothed_labels_all = [_smooth_with_noise(lbl, hmm_params) for lbl in raw_labels_all]
    else:
        smoothed_labels_all = raw_labels_all

    for stem, smoothed, probs in zip(new_stems, smoothed_labels_all, raw_probs_all):
        np.save(os.path.join(shared_dir, f"{stem}_labels.npy"), smoothed.astype(np.int32))
        np.save(os.path.join(shared_dir, f"{stem}_probs.npy"), probs.astype(np.float32))
        n_noise = int((smoothed < 0).sum())
        print(f"  {stem}: {len(smoothed):,} frames, {n_noise:,} noise ({100 * n_noise / len(smoothed):.1f}%)")

    print(f"\nDone. Applied existing cluster model to {len(new_stems)} video(s).")
    print("Per-video labels → results/shared/<stem>_labels.npy")
    print("Existing models and cluster_info.json were not modified.")


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

    cluster_info_path = os.path.join(_res(), "shared", "cluster_info.json")
    if not os.path.exists(cluster_info_path):
        sys.exit("No cluster_info.json found. Run --cluster first.")
    index_path = os.path.join(_res(), "features", "index.json")
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
        labels_path = os.path.join(_res(), "shared", f"{stem}_labels.npy")
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

    # Update run_manifest.json so the Cluster Runs view reflects the post-collapse count
    manifest_path = os.path.join(_res(), "shared", "run_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)
        manifest["n_clusters"] = n_new
        manifest["collapse_threshold"] = threshold
        manifest["saved"] = False
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    # Mark the run as unsaved: the saved snapshot (if any) no longer matches shared/
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg_data = json.load(f)
            cfg_data["current_run_saved"] = False
            with open(cfg_path, "w", encoding="utf-8") as f:
                json.dump(cfg_data, f, indent=2)
        except Exception:
            pass

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
    save_path = os.path.join(_res(), "comparison", "animal_trajectories.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def cmd_report(fps: float = 30.0, min_confidence: float = 0.0):
    import matplotlib.pyplot as plt
    from scipy import stats

    for path in [os.path.join(_res(), "features", "index.json"),
                  os.path.join(_res(), "shared", "cluster_info.json")]:
        if not os.path.exists(path):
            sys.exit(f"Missing {path}. Run --extract and --cluster first.")

    with open(os.path.join(_res(), "features", "index.json")) as f:
        index = json.load(f)
    with open(os.path.join(_res(), "shared", "cluster_info.json")) as f:
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
        labels_path = os.path.join(_res(), "shared", f"{stem}_labels.npy")
        if not os.path.exists(labels_path):
            continue
        labels = np.load(labels_path)

        if min_confidence > 0.0:
            probs_path = os.path.join(_res(), "shared", f"{stem}_probs.npy")
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

    if not os.path.exists(_meta()):
        sys.exit("metadata.csv not found.")
    meta = pd.read_csv(_meta())
    meta = _vc.normalize_metadata_columns(meta)
    meta["stem"] = meta["filename"].str.replace(r"\.mp4$", "", regex=True)

    df = df_states.merge(meta, on="stem", how="left")

    os.makedirs(os.path.join(_res(), "comparison"), exist_ok=True)
    df.to_csv(os.path.join(_res(), "comparison", "summary_table.csv"), index=False)
    print(f"Summary table saved: results/comparison/summary_table.csv  ({len(df)} videos)")

    # ---- Characterization: bouts.csv + state_summary.csv ----
    char_dir = os.path.join(_res(), "characterization")
    os.makedirs(char_dir, exist_ok=True)
    all_bouts = []
    meta_cols = [c for c in ["context", "animal_id", "day", "experiment"] if c in df.columns]
    for stem, row_df in df.groupby("stem"):
        labels_path = os.path.join(_res(), "shared", f"{stem}_labels.npy")
        if not os.path.exists(labels_path):
            continue
        lbl = np.load(labels_path)
        meta_r = {c: row_df[c].iloc[0] for c in meta_cols}
        i = 0
        while i < len(lbl):
            if lbl[i] < 0:
                i += 1
                continue
            s = int(lbl[i])
            j = i
            while j < len(lbl) and lbl[j] == s:
                j += 1
            all_bouts.append({
                "stem": stem, "state": s,
                "start_frame": i, "end_frame": j - 1,
                "start_sec": i / fps, "end_sec": (j - 1) / fps,
                "duration_sec": (j - i) / fps,
                **meta_r,
            })
            i = j
    bouts_df = pd.DataFrame(all_bouts) if all_bouts else pd.DataFrame(
        columns=["stem", "state", "start_frame", "end_frame",
                 "start_sec", "end_sec", "duration_sec"] + meta_cols
    )
    bouts_df.to_csv(os.path.join(char_dir, "bouts.csv"), index=False)
    print(f"Bouts saved: results/characterization/bouts.csv  ({len(bouts_df)} bouts)")

    # Per-state context fractions from summary_table
    ctx_fracs: dict[int, dict[str, float]] = {}
    if "context" in df.columns:
        for ctx, grp in df.groupby("context"):
            for k in range(n_clusters):
                col = f"state_{k}_frac"
                if col in grp.columns:
                    ctx_fracs.setdefault(k, {})[str(ctx)] = float(grp[col].mean())

    # Compute bout stats and kinematic labels from cluster centers
    n_kp = index.get("_meta", {}).get("n_keypoints", 8)
    centers = cluster_info.get("cluster_centers", [])
    bout_stats: dict[int, dict] = {}
    ss_rows = []
    for k in range(n_clusters):
        grp_b = bouts_df[bouts_df["state"] == k] if not bouts_df.empty else pd.DataFrame()
        mean_dur = float(grp_b["duration_sec"].mean()) if not grp_b.empty else float("nan")
        bout_stats[k] = {"mean_dur": mean_dur if not np.isnan(mean_dur) else None}

    kin_labels = _generate_kinematic_labels(centers, bout_stats, n_keypoints=n_kp)

    for k in range(n_clusters):
        grp_b = bouts_df[bouts_df["state"] == k] if not bouts_df.empty else pd.DataFrame()
        row_s = {
            "state": k,
            "heuristic_label": kin_labels.get(k, f"State {k}"),
            "mean_bout_dur_sec": float(grp_b["duration_sec"].mean()) if not grp_b.empty else float("nan"),
            "median_bout_dur_sec": float(grp_b["duration_sec"].median()) if not grp_b.empty else float("nan"),
            "n_bouts": len(grp_b),
        }
        if k < len(centers):
            row_s.update(_extract_kinematic_values(centers[k], n_keypoints=n_kp))
        row_s.update({f"context_{ctx}_frac": v for ctx, v in ctx_fracs.get(k, {}).items()})
        ss_rows.append(row_s)

    ss_path = os.path.join(char_dir, "state_summary.csv")
    pd.DataFrame(ss_rows).to_csv(ss_path, index=False)
    print(f"State summary saved: results/characterization/state_summary.csv  ({n_clusters} states)")

    # ---- Transition matrix outputs ----
    _trans_meta_cols = ["stem"] + [
        c for c in ["context", "day", "animal_id", "experiment"] if c in meta.columns
    ]
    df_trans = pd.DataFrame(trans_rows).merge(
        meta[_trans_meta_cols].drop_duplicates("stem"),
        on="stem", how="left"
    )
    trans_cols = [c for c in df_trans.columns if c.startswith("trans_")]
    # Join full metadata for transition_table.csv
    df_trans_full = df_states.merge(
        pd.DataFrame(trans_rows), on="stem", how="left"
    ).merge(meta, on="stem", how="left")
    df_trans_full.to_csv(os.path.join(_res(), "comparison", "transition_table.csv"), index=False)
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
            os.path.join(_res(), "comparison", "transition_by_context.png")
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
        boxplot_by_group("fear", os.path.join(_res(), "comparison", "state_by_fear.png"), "Fear Condition")
    else:
        print("  SKIP state_by_fear.png: 'fear' column in metadata.csv is empty (fill it in)")

    if "day" in df.columns:
        boxplot_by_group("day", os.path.join(_res(), "comparison", "state_by_day.png"), "Day")

    if "context" in df.columns:
        boxplot_by_group("context", os.path.join(_res(), "comparison", "state_by_context.png"), "Context")

    if "experiment" in df.columns:
        boxplot_by_group("experiment", os.path.join(_res(), "comparison", "state_by_experiment.png"), "Experiment (CFC vs CFD)")

    if "animal_id" in df.columns:
        boxplot_by_group("animal_id", os.path.join(_res(), "comparison", "state_by_animal.png"), "Animal ID")

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


def cmd_event_align(min_confidence: float = 0.0):
    """Peri-event behavioral state alignment for discrete experiments."""
    column_map = _vc.get_column_map()

    from event_alignment import load_events, compute_peri_event_profiles, compute_event_contrast

    events_df = load_events(_meta(), column_map)
    if events_df is None:
        print("No event column configured. "
              "Set 'event' in column mapping to use event alignment.")
        return

    print(f"Event column: '{column_map.get('event', '')}'")
    print(f"Event labels: {sorted(events_df['event_label'].unique())}")
    label_counts = events_df.groupby("event_label").size()
    for lbl, n in label_counts.items():
        print(f"  {lbl}: {n} session(s)")

    index_path = os.path.join(_res(), "features", "index.json")
    if not os.path.exists(index_path):
        sys.exit("No feature index found. Run --extract first.")
    with open(index_path) as f:
        index = json.load(f)

    print("\nComputing peri-event state profiles...")
    profiles = compute_peri_event_profiles(
        index, events_df, min_confidence=min_confidence
    )

    if not profiles:
        print("No profiles computed. Check that _labels.npy files exist for event sessions.")
        return

    print("\nComputing event contrast vectors...")
    contrast = compute_event_contrast(profiles)

    print("\n=== Peri-Event Summary ===")
    print(f"{'Event Label':20}  {'Dominant State':14}")
    print("-" * 38)
    for label, fracs in profiles.items():
        dom = int(np.argmax(fracs))
        print(f"  {label:18}  State {dom}")

    if contrast:
        print("\n=== Event Contrast ===")
        print(f"{'Pair':32}  {'Magnitude':10}  {'Dom A':6}  {'Dom B':6}")
        print("-" * 62)
        for key, info in contrast.items():
            print(f"  {key:30}  {info['contrast_magnitude']:.4f}    "
                  f"  {info['dominant_state_A']}       {info['dominant_state_B']}")


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
            summary_csv=os.path.join(_res(), "comparison", "summary_table.csv"),
            output_dir=os.path.join(_res(), "quantification"),
            cohort_csv=cohort,
        )

        master_path = os.path.join(_res(), "quantification", "master_table.csv")
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
            os.path.join(_res(), "comparison", "summary_table.csv"),
            output_dir=os.path.join(_res(), "quantification"),
            cohort_csv=cohort,
        )
        master_path = os.path.join(_res(), "quantification", "master_table.csv")
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
# Cluster diagnostic (delegates to diagnose_clusters.py)
# ---------------------------------------------------------------------------

def cmd_diagnose(
    mcs_list: list | None = None,
    umap_dims: int = 10,
    min_samples: int | None = None,
    umap_sweep: bool = False,
    hdbscan_jobs: int = 1,
):
    """Run the MCS sweep from diagnose_clusters.py and print a recommendation."""
    try:
        from diagnose_clusters import cmd_sweep, cmd_umap_sweep, _DEFAULT_MCS
    except ImportError:
        sys.exit("[ERROR] diagnose_clusters.py not found in project directory.")

    sweep_mcs = mcs_list if mcs_list else _DEFAULT_MCS
    rec = cmd_sweep(
        sweep_mcs,
        umap_dims=umap_dims,
        min_samples=min_samples,
        hdbscan_jobs=hdbscan_jobs,
    )
    if umap_sweep:
        rec_mcs = rec["mcs"] if rec else None
        cmd_umap_sweep(rec_mcs, umap_dims=umap_dims, min_samples=min_samples,
                       hdbscan_jobs=hdbscan_jobs)


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
    parser.add_argument("--fix-features", action="store_true",
                        help="Re-extract only videos whose feature dimension doesn't match "
                             "the current config.json settings (resolves --extract/--no-wavelets "
                             "mismatches without a full re-extraction)")
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
    parser.add_argument("--diagnose", action="store_true",
                        help="Sweep min_cluster_size values and recommend best setting (runs diagnose_clusters.py)")
    parser.add_argument("--diagnose-mcs", type=str, default=None,
                        metavar="LIST",
                        help="Comma-separated min_cluster_size values for --diagnose "
                             "(default: 50,100,200,300,500,750,1000,1500,2000,3000)")
    parser.add_argument("--umap-sweep", action="store_true",
                        help="With --diagnose: also sweep UMAP n_neighbors on a 50k-frame sample")
    parser.add_argument("--hdbscan-jobs", type=int, default=1,
                        help="Parallel jobs for HDBSCAN core-distance (--diagnose only, default: 1)")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--n-clusters", type=int, default=None,
                        help="(ignored for HDBSCAN — kept for CLI compatibility)")
    parser.add_argument("--min-cluster-size", type=int, default=50,
                        help="HDBSCAN min_cluster_size (default: 50)")
    parser.add_argument("--hdbscan-min-samples", type=int, default=None,
                        help="HDBSCAN min_samples. Defaults to min_cluster_size if not set.")
    parser.add_argument("--hdbscan-sample", type=int, default=300000,
                        help="Max frames used to fit HDBSCAN before assigning remaining frames "
                             "(default: 300000)")
    parser.add_argument("--umap-dims", type=int, default=10,
                        help="UMAP n_components (default: 10). Try 3 for better HDBSCAN performance.")
    parser.add_argument("--no-wavelets", action="store_true",
                        help="Skip Morlet wavelet features during --extract (faster)")
    parser.add_argument("--validate", action="store_true",
                        help="With --cluster: run 80/20 train/test split validation (seed=42)")
    parser.add_argument("--apply-existing", action="store_true",
                        help="With --cluster: apply the existing saved model to new videos only "
                             "(fast, does not refit preprocessor/UMAP/HDBSCAN)")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        help="With --report/--quantify: exclude frames with prob < threshold")
    parser.add_argument("--cohort", metavar="FILE", default=None,
                        help="Cohort CSV/Excel for --quantify (auto-detected if omitted)")
    parser.add_argument("--save-run", action="store_true",
                        help="Mark the cluster run as saved after --cluster completes")
    parser.add_argument("--event-align", action="store_true",
                        help="Peri-event behavioral state analysis (discrete experiments only)")
    args = parser.parse_args()

    if not any([args.extract, args.fix_features, args.cluster, args.collapse, args.diagnose,
                args.report, args.summarize, args.quantify, args.save_run,
                args.event_align]):
        parser.print_help()
        sys.exit(1)

    _print_hardware_banner()

    if args.extract:
        cmd_extract(fps=args.fps, use_wavelets=not args.no_wavelets)
    if args.fix_features:
        cmd_fix_features(fps=args.fps)
    if args.diagnose:
        mcs_list = None
        if args.diagnose_mcs:
            try:
                mcs_list = [int(x.strip()) for x in args.diagnose_mcs.split(",") if x.strip()]
            except ValueError:
                sys.exit("--diagnose-mcs must be a comma-separated list of integers, e.g. 100,200,500")
        cmd_diagnose(
            mcs_list=mcs_list,
            umap_dims=args.umap_dims,
            min_samples=args.hdbscan_min_samples,
            umap_sweep=args.umap_sweep,
            hdbscan_jobs=args.hdbscan_jobs,
        )
    if args.cluster:
        if args.apply_existing:
            cmd_apply_existing(fps=args.fps)
        else:
            cmd_cluster(fps=args.fps, min_cluster_size=args.min_cluster_size,
                        min_samples=args.hdbscan_min_samples, umap_dims=args.umap_dims,
                        validate=args.validate, hdbscan_sample=args.hdbscan_sample)
        if args.save_run:
            _mark_run_saved()
    if args.save_run and not args.cluster:
        _mark_run_saved()
    if args.collapse:
        cmd_collapse(threshold=args.collapse_threshold)
    if args.report:
        cmd_report(fps=args.fps, min_confidence=args.min_confidence)
    if args.summarize:
        cmd_summarize()
    if args.quantify:
        cmd_quantify(cohort=args.cohort, min_confidence=args.min_confidence)
    if args.event_align:
        cmd_event_align(min_confidence=args.min_confidence)


if __name__ == "__main__":
    main()
