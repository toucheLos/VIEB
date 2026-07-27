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
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import vieb_config as _vc
def _raw(): return _vc.get_raw_videos_dir()
def _res(): return _vc.get_results_dir()
def _meta(): return _vc.get_metadata_path()

ROOT = os.path.dirname(os.path.abspath(__file__))


def _project_config_path() -> str:
    """Return the active project's config.json path (never the repo-root one)."""
    return _vc.get_project_config_path()


def _print_project_path_diagnostics(repo_root: str | None = None, app_config_path: str | None = None, *, repair: bool = False):
    import project_manager as _pm

    root = repo_root or ROOT
    app_path = app_config_path or os.path.join(root, "app_config.json")
    project = _pm.get_active_project(root, app_path)
    project_path = Path(project)
    paths = _pm.resolve_project_paths(root, app_path)
    print(f"Active project: {project}")
    print(f"Metadata path: {paths['metadata'].path} (origin: {paths['metadata'].origin})")
    print(f"Results dir: {paths['results'].path} (origin: {paths['results'].origin})")
    print(f"Raw videos dir: {paths['raw_videos'].path} (origin: {paths['raw_videos'].origin})")
    print(f"Config path: {paths['config'].path}")
    for key in ("metadata", "results", "raw_videos"):
        info = paths[key]
        if not info.valid:
            raise _pm.ProjectSelectionError(
                f"Refusing project path for {key}: {info.message} Complete Stage 0: Onboarding before running the pipeline."
            )
        if not info.path.exists():
            print(f"[WARN] Resolved {key} path does not exist:")
            print(f"  {info.path}")
            candidate = _pm.detect_doubled_project_segment(info.path, project_path, root)
            if candidate is not None:
                print(f"[WARN]   This looks like a doubled path from a pre-refactor config.json.")
                print(f"[WARN]   A working path was found: {candidate}")
                if repair:
                    _pm.repair_project_config_path(project_path, key, candidate)
                    print(f"[FIX]    config.json updated: paths.{key} -> {candidate}")
                else:
                    print(f"[FIX]    Re-run with --repair-paths to update config.json automatically,")
                    print(f"         or edit config.json's paths.{key} to the path above.")
    return paths


def _slug(text: object) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip().lower()).strip("_")
    return slug or "group"


def _state_group_filename(group_col: str) -> str:
    compat = {"animal_id": "state_by_animal.png"}
    return compat.get(group_col, f"state_by_{_slug(group_col)}.png")


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
        cfg_path = _project_config_path()
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


_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")


def _resolve_h5_video_path(stem, candidates, video_path_map, raw_videos_dir):
    """Resolve a video path for an H5-extracted session.

    Tries each raw (non-normalized) candidate string against video_path_map
    (keyed by h5_manifest._normalize()), then falls back to
    raw_videos_dir/<stem><ext> for each known video extension. Returns None
    if nothing resolves — that is an expected, valid outcome, not an error.
    """
    from h5_manifest import _normalize

    for cand in candidates:
        if not cand:
            continue
        norm = _normalize(cand)
        if norm in video_path_map:
            return video_path_map[norm]
    if raw_videos_dir:
        for ext in _VIDEO_EXTS:
            candidate_path = os.path.join(raw_videos_dir, f"{stem}{ext}")
            if os.path.exists(candidate_path):
                return candidate_path
    return None


def _cmd_extract_h5(fps: float = 30.0, use_wavelets: bool = True):
    """Feature extraction from a single shared H5 pose file (video-less mode).

    For standard multi-key H5 files, iterates metadata.csv rows and resolves
    each row to a key inside the H5 file. For concatenated-table H5 files,
    iterates the unique session/source values directly from the H5.
    """
    from ml import PoseFeatureExtractor
    from pose_io import inspect_h5, load_pose_h5
    from h5_manifest import detect_concatenated_table, load_manifest, load_video_paths, resolve_h5_key

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
    video_path_map = load_video_paths(_vc.get_h5_manifest_path(), value_col=manifest_value_col)
    try:
        raw_videos_dir = _vc.get_raw_videos_dir()
    except Exception:
        raw_videos_dir = ""

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
            video_path = _resolve_h5_video_path(
                stem, [stem, source_value], video_path_map, raw_videos_dir
            )
            index[stem] = {
                "video_path": video_path,
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
            video_path = _resolve_h5_video_path(
                stem, [stem, h5_key, row_dict.get("animal_id", "")], video_path_map, raw_videos_dir
            )
            index[stem] = {
                "video_path": video_path,
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

    # Print feature availability report
    report = extractor.get_feature_availability_report()
    if report.get("groups"):
        resolved = [g for g, info in report["groups"].items() if info["resolved"]]
        skipped_groups = [g for g, info in report["groups"].items() if not info["resolved"]]
        if resolved:
            print(f"Keypoint groups resolved: {', '.join(resolved)}")
        if skipped_groups:
            print(f"Keypoint groups missing: {', '.join(skipped_groups)}")
        if report.get("skipped_features"):
            for feat, reason in report["skipped_features"].items():
                print(f"  Skipping {feat}: {reason}")

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
    # Start from a copy so no existing field is silently dropped.
    new_meta = dict(old_meta)
    new_meta["n_keypoints"] = target_n_keypoints
    new_meta["n_features"] = target_n_features
    new_meta["use_wavelets"] = use_wavelets
    new_meta.setdefault("vieb_version", "1.0")
    # Rebuild feature_names and semantic_features from the now-configured extractor.
    _fm = extractor.get_feature_meta(n_keypoints=target_n_keypoints)
    new_meta["feature_names"] = _fm.get("feature_names", [])
    new_meta["semantic_features"] = _fm.get("semantic_features", [])
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


def cmd_backfill_video_paths():
    """Backfill video_path for existing index.json entries created before H5
    extraction started populating it (see _cmd_extract_h5()).

    Only touches entries with h5_path set and video_path missing/None.
    Every other field on every entry (including entries this doesn't touch)
    is preserved exactly as loaded — mirrors cmd_fix_features()'s discipline
    of never dropping unrelated index.json fields.
    """
    from h5_manifest import load_video_paths

    index_path = os.path.join(_res(), "features", "index.json")
    if not os.path.exists(index_path):
        sys.exit("No index found. Run --extract first.")
    with open(index_path) as f:
        index = json.load(f)

    # Extraction may have used either "h5_key" or the configured H5 source
    # column (e.g. "source_file") as the manifest's value_col, depending on
    # whether the H5 was a concatenated table — an existing index.json gives
    # no cheap way to tell which was used for a given entry, so try both
    # candidate key spaces and merge (first match wins).
    manifest_path = _vc.get_h5_manifest_path()
    video_path_map = load_video_paths(manifest_path, value_col="h5_key")
    source_col = _vc.get_h5_source_col() or None
    if source_col:
        for k, v in load_video_paths(manifest_path, value_col=source_col).items():
            video_path_map.setdefault(k, v)

    try:
        raw_videos_dir = _vc.get_raw_videos_dir()
    except Exception:
        raw_videos_dir = ""

    backfilled = 0
    already_had = 0
    skipped_no_h5 = 0
    still_missing = 0

    for stem, entry in index.items():
        if stem == "_meta" or not isinstance(entry, dict):
            continue
        if not entry.get("h5_path"):
            skipped_no_h5 += 1
            continue
        if entry.get("video_path"):
            already_had += 1
            continue
        candidates = [stem, entry.get("h5_key", "")]
        video_path = _resolve_h5_video_path(stem, candidates, video_path_map, raw_videos_dir)
        if video_path:
            entry["video_path"] = video_path
            backfilled += 1
            print(f"  {stem}: video_path -> {video_path}")
        else:
            still_missing += 1

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    print(f"\nDone. Backfilled {backfilled} entr{'y' if backfilled == 1 else 'ies'}.")
    print(f"  Already had video_path : {already_had}")
    print(f"  Skipped (no h5_path)   : {skipped_no_h5}")
    print(f"  Still unresolved       : {still_missing}"
          + ("" if not still_missing else
             " (no manifest video-path column match and no matching file "
             f"under raw_videos_dir with extensions {_VIDEO_EXTS})"))


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
    from cluster_run_manager import ClusterRunManager
    mgr = ClusterRunManager(os.path.dirname(runs_dir))
    return mgr._next_run_n()


def _auto_save_previous_run() -> str | None:
    """Copy results/shared/ to results/runs/{run_id}/ before a new run overwrites it."""
    from cluster_run_manager import ClusterRunManager

    shared_dir = os.path.join(_res(), "shared")
    cluster_info_path = os.path.join(shared_dir, "cluster_info.json")
    if not os.path.exists(cluster_info_path):
        return None

    manifest_path = os.path.join(shared_dir, "run_manifest.json")
    if not os.path.exists(manifest_path):
        return None

    with open(manifest_path) as f:
        existing_manifest = json.load(f)

    run_id = existing_manifest.get("run_id", "")
    if not run_id:
        return None

    cfg_path = _project_config_path()
    mgr = ClusterRunManager(_res(), config_path=cfg_path)
    mgr.save_run(run_id)
    print(f"Auto-saved previous run to results/runs/{run_id}/")
    return run_id


def _migrate_index_meta(meta: dict) -> dict:
    """Normalize old _meta formats to the current schema in-memory.

    - ``feature_count`` (old key) → ``n_features``
    - ``use_wavelets`` absent → left as absent (not defaulted to True/False)

    Does NOT write to disk; callers persist if needed.
    """
    if not isinstance(meta, dict):
        return {}
    out = dict(meta)
    if "n_features" not in out and "feature_count" in out:
        out["n_features"] = out["feature_count"]
    return out


def _write_current_run_manifest(
    min_cluster_size: int,
    umap_dims: int,
    effective_min_samples: int,
    hdbscan_sample: int,
    n_found: int,
    mean_conf: float,
    low_conf_frac: float,
    noise_frac: float,
    *,
    min_samples_requested: int = 0,
    runtime_seconds: float = 0.0,
    assignment_method: str = "",
) -> str:
    """Write results/shared/run_manifest.json and update config.json."""
    from cluster_run_manager import ClusterRunConfig, ClusterRunManager

    cfg_path = _project_config_path()
    mgr = ClusterRunManager(_res(), config_path=cfg_path)
    run_cfg = ClusterRunConfig(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples_requested,
        umap_dims=umap_dims,
        hdbscan_sample=hdbscan_sample,
    )
    run_id = mgr.create_run_id(run_cfg)

    from datetime import datetime as _dt
    now = _dt.now()

    manifest = {
        "run_id": run_id,
        "status": "completed",
        "date": now.strftime("%Y-%m-%d %H:%M"),
        "started_at": "",
        "finished_at": now.isoformat(),
        "runtime_seconds": round(runtime_seconds, 1),
        "min_cluster_size": min_cluster_size,
        "min_samples_requested": min_samples_requested,
        "min_samples_resolved": effective_min_samples,
        "umap_dims": umap_dims,
        "hdbscan_sample": hdbscan_sample,
        "hdbscan_min_samples": effective_min_samples,
        "n_clusters": n_found,
        "mean_confidence": round(mean_conf, 4),
        "low_confidence_frac": round(low_conf_frac, 4),
        "noise_frac": round(noise_frac, 4),
        "assignment_method": assignment_method,
        "saved": False,
    }

    # Attach feature metadata so the manifest is self-contained.
    _index_path = os.path.join(_res(), "features", "index.json")
    if os.path.exists(_index_path):
        try:
            with open(_index_path) as _f:
                _idx = json.load(_f)
            _feat_meta = _migrate_index_meta(_idx.get("_meta", {}))
            _nf = _feat_meta.get("n_features")
            if _nf is not None:
                manifest["n_features"] = int(_nf)
            _uw = _feat_meta.get("use_wavelets")
            manifest["use_wavelets"] = bool(_uw) if _uw is not None else None
        except Exception:
            pass

    with open(os.path.join(_res(), "shared", "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

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


def _write_overview_summary(
    summary: pd.DataFrame,
    cluster_info: dict,
    feature_index: dict,
    *,
    active_run_id: str | None = None,
) -> None:
    """Write the small Overview-only summary used by GUI startup."""
    shared_dir = os.path.join(_res(), "shared")
    os.makedirs(shared_dir, exist_ok=True)

    n_states = int(cluster_info.get("n_clusters", 0) or 0)
    state_cols = [
        f"state_{i}_frac"
        for i in range(n_states)
        if f"state_{i}_frac" in summary.columns
    ]
    state_means = {
        str(int(col.split("_")[1])): float(summary[col].mean())
        for col in state_cols
    }
    noise_fraction = None
    if state_cols:
        noise_fraction = max(0.0, 1.0 - float(summary[state_cols].sum(axis=1).mean()))

    total_frames = 0
    if isinstance(feature_index, dict):
        for key, value in feature_index.items():
            if key == "_meta" or not isinstance(value, dict):
                continue
            total_frames += int(value.get("n_frames", 0) or 0)

    run_manifest = {}
    manifest_path = os.path.join(shared_dir, "run_manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, encoding="utf-8") as f:
                run_manifest = json.load(f)
        except Exception:
            run_manifest = {}

    from datetime import datetime as _dt
    now = _dt.now().isoformat()
    overview = {
        "schema_version": 1,
        "generated_at": now,
        "last_run_time": run_manifest.get("finished_at") or run_manifest.get("date") or now,
        "active_run_id": active_run_id or run_manifest.get("run_id") or "",
        "total_videos": int(len(summary)),
        "total_frames": int(total_frames),
        "n_states": n_states,
        "noise_fraction": noise_fraction,
        "state_means": state_means,
        "markers": {
            "features": os.path.exists(os.path.join(_res(), "features", "index.json")),
            "clusters": os.path.exists(os.path.join(_res(), "shared", "cluster_info.json")),
            "report": True,
            "summary": True,
            "motifs": (
                os.path.exists(os.path.join(_res(), "comparison", "motifs.csv"))
                or os.path.exists(os.path.join(_res(), "motifs", "motif_summary.csv"))
            ),
        },
    }
    with open(os.path.join(shared_dir, "overview_summary.json"), "w", encoding="utf-8") as f:
        json.dump(overview, f, indent=2)
    print("Overview summary saved: results/shared/overview_summary.json")


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


def _assign_by_nearest_centroid(
    fit_points: np.ndarray,
    fit_labels: np.ndarray,
    predict_points: np.ndarray,
    noise_distance_factor: float = 3.0,
    batch_size: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign labels to non-sampled frames by nearest cluster centroid.

    Used for GPU HDBSCAN where approximate_predict is unavailable.
    Computes the centroid of each cluster in UMAP space, then assigns
    each predict_point to its nearest centroid.
    Points farther than noise_distance_factor * (median within-cluster radius)
    are labelled -1 (noise) with probability 0.

    Batches the distance computation to avoid OOM on large predict sets.
    """
    unique_labels = np.unique(fit_labels[fit_labels >= 0])
    if len(unique_labels) == 0:
        return (
            np.full(len(predict_points), -1, dtype=np.int32),
            np.zeros(len(predict_points), dtype=np.float32),
        )

    centroids = np.stack([
        fit_points[fit_labels == lbl].mean(axis=0)
        for lbl in unique_labels
    ])  # (n_clusters, n_dims)

    # Noise threshold: noise_distance_factor × median within-cluster radius
    noise_threshold: float | None = None
    if noise_distance_factor > 0:
        intra_dists = []
        for i, lbl in enumerate(unique_labels):
            pts = fit_points[fit_labels == lbl]
            if len(pts) > 1:
                intra_dists.append(float(np.median(np.linalg.norm(pts - centroids[i], axis=1))))
        if intra_dists:
            noise_threshold = noise_distance_factor * float(np.median(intra_dists))

    n_predict = len(predict_points)
    pred_labels = np.full(n_predict, -1, dtype=np.int32)
    pred_probs  = np.zeros(n_predict, dtype=np.float32)

    for start in range(0, n_predict, batch_size):
        end = min(start + batch_size, n_predict)
        batch = predict_points[start:end]
        diffs = batch[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=-1)  # (batch, n_clusters)
        nearest_idx  = np.argmin(dists, axis=1)
        nearest_dist = dists[np.arange(len(batch)), nearest_idx]
        batch_labels = unique_labels[nearest_idx].astype(np.int32)
        batch_probs  = (1.0 / (1.0 + nearest_dist)).astype(np.float32)
        if noise_threshold is not None:
            noise_mask = nearest_dist > noise_threshold
            batch_labels[noise_mask] = -1
            batch_probs[noise_mask]  = 0.0
        pred_labels[start:end] = batch_labels
        pred_probs[start:end]  = batch_probs

    return pred_labels, pred_probs


def _mark_run_saved() -> None:
    """Set saved=true in results/shared/run_manifest.json, save to runs/, and update config.json."""
    from cluster_run_manager import ClusterRunManager

    manifest_path = os.path.join(_res(), "shared", "run_manifest.json")
    if not os.path.exists(manifest_path):
        print("[warn] No run_manifest.json found — nothing to mark as saved.")
        return
    with open(manifest_path) as f:
        manifest = json.load(f)
    manifest["saved"] = True
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    run_id = manifest.get("run_id", "")
    if run_id:
        cfg_path = _project_config_path()
        mgr = ClusterRunManager(_res(), config_path=cfg_path)
        mgr.save_run(run_id)

    cfg_path = _project_config_path()
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
    import time as _time
    import joblib
    from ml import BehaviorPreprocessor

    _t0 = _time.perf_counter()
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
    meta_info = _migrate_index_meta(index.get("_meta") or {}) or None
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
        _cfg_path = _project_config_path()
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
    if min_samples is not None:
        effective_min_samples = min_samples
    else:
        effective_min_samples = max(10, min(100, min_cluster_size // 10))
    if use_gpu and effective_min_samples > 1023:
        print(f"  [info] cuML HDBSCAN requires min_samples <= 1023; clamping {effective_min_samples} -> 1023.")
        effective_min_samples = 1023
    print(
        f"\nFitting HDBSCAN (min_cluster_size={min_cluster_size}, "
        f"min_samples={effective_min_samples}, hdbscan_sample={hdbscan_sample})..."
    )

    n_embedded = len(pooled_umap)

    if validate:
        train_fit_indices = np.asarray(train_indices, dtype=np.int64)
        test_indices = np.concatenate([
            np.arange(boundaries[s][0], boundaries[s][1]) for s in test_stems
        ]) if test_stems else np.array([], dtype=np.int64)
        base_indices = train_fit_indices
    else:
        base_indices = np.arange(n_embedded, dtype=np.int64)

    # ---- Sampling (GPU and CPU both respect hdbscan_sample) ----
    n_base = len(base_indices)
    if n_base > hdbscan_sample:
        rng = np.random.default_rng(42)
        sampled_pos = np.sort(rng.choice(n_base, hdbscan_sample, replace=False))
        fit_indices = base_indices[sampled_pos]
        _partition = "train " if validate else ""
        print(
            f"  Fitting HDBSCAN on {len(fit_indices):,} sampled {_partition}frames "
            f"out of {n_embedded:,} total embedded "
            f"({n_base:,} {_partition}frames); "
            f"full-frame HDBSCAN disabled."
        )
        print(
            f"  Assigning remaining {n_embedded - len(fit_indices):,} frames "
            f"using {'approximate_predict' if not use_gpu else 'nearest-centroid'} assignment."
        )
    else:
        fit_indices = base_indices
        print(
            f"  Fitting HDBSCAN on all {len(fit_indices):,} "
            f"{'train ' if validate else ''}embedded frames "
            f"(total embedded: {n_embedded:,})."
        )

    # Safety guard: never silently fit on more frames than the sample limit.
    if len(fit_indices) > hdbscan_sample:
        raise RuntimeError(
            f"HDBSCAN safety guard: about to fit on {len(fit_indices):,} frames "
            f"but hdbscan_sample={hdbscan_sample:,}. "
            "This would likely OOM. This is a code bug — please report it."
        )

    actual_hdbscan_sample = int(len(fit_indices))

    predict_mask = np.ones(n_embedded, dtype=bool)
    predict_mask[fit_indices] = False
    predict_indices = np.flatnonzero(predict_mask)

    # ---- GPU HDBSCAN ----
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
            all_raw_labels = np.full(n_embedded, -1, dtype=np.int32)
            all_probs = np.zeros(n_embedded, dtype=np.float32)
            all_raw_labels[fit_indices] = fit_labels
            all_probs[fit_indices] = fit_probs
            if len(predict_indices) > 0:
                print(f"  Assigning {len(predict_indices):,} non-sampled frames via nearest-centroid…")
                pred_labels, pred_probs = _assign_by_nearest_centroid(
                    pooled_umap[fit_indices],
                    fit_labels,
                    pooled_umap[predict_indices],
                )
                all_raw_labels[predict_indices] = pred_labels
                all_probs[predict_indices] = pred_probs

    # ---- CPU HDBSCAN ----
    if not use_gpu:
        clusterer_model, all_raw_labels, all_probs = _fit_cpu_hdbscan_with_assignment(
            HDBSCANClass,
            pooled_umap,
            fit_indices,
            predict_indices,
            min_cluster_size,
            effective_min_samples,
        )

    if len(predict_indices) == 0:
        _assignment_method = "direct"
    elif use_gpu:
        _assignment_method = "nearest_centroid"
    else:
        _assignment_method = "approximate_predict"
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
    _runtime = _time.perf_counter() - _t0
    _noise_frac = float(n_noise / len(all_raw_labels)) if len(all_raw_labels) > 0 else 0.0
    _min_samples_requested = min_samples if min_samples is not None else 0
    _current_run_id = _write_current_run_manifest(
        min_cluster_size=min_cluster_size,
        umap_dims=umap_dims,
        effective_min_samples=effective_min_samples,
        hdbscan_sample=actual_hdbscan_sample,
        n_found=n_found,
        mean_conf=mean_conf,
        low_conf_frac=low_conf_frac,
        noise_frac=_noise_frac,
        min_samples_requested=_min_samples_requested,
        runtime_seconds=_runtime,
        assignment_method=_assignment_method,
    )
    print(f"Run manifest → results/shared/run_manifest.json  (run_id: {_current_run_id})")

    # ---- Clustering diagnostics ----
    try:
        _generate_diagnostics(
            all_labels=np.concatenate(smoothed_labels_all),
            all_probs=np.concatenate(raw_probs_all),
            pooled_umap=pooled_umap,
        )
        print("Diagnostics → results/diagnostics/")
    except Exception as _diag_err:
        print(f"[warn] Diagnostics generation failed: {_diag_err}")

    # ---- Auto-save run to results/runs/ ----
    try:
        from cluster_run_manager import ClusterRunManager
        _cfg_path = _project_config_path()
        _mgr = ClusterRunManager(_res(), config_path=_cfg_path)
        _mgr.save_run(_current_run_id)
        print(f"Run auto-saved → results/runs/{_current_run_id}/")
    except Exception as _save_err:
        print(f"[warn] Could not auto-save run: {_save_err}")

    # ---- Validation report ----
    if validate:
        _run_validation_report(
            stems, train_stems, test_stems, boundaries,
            smoothed_labels_all, raw_probs_all, n_found,
            min_cluster_size,
        )


# ---------------------------------------------------------------------------
# Clustering diagnostics
# ---------------------------------------------------------------------------

def _compute_state_bouts(labels_1d: np.ndarray) -> dict[int, list[int]]:
    """Return {state_id: [bout_len_frames, ...]} for one video. Skips noise (-1)."""
    if len(labels_1d) == 0:
        return {}
    changes = np.where(np.diff(labels_1d) != 0)[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(labels_1d)]])
    bouts: dict[int, list[int]] = {}
    for s, e in zip(starts, ends):
        state = int(labels_1d[s])
        if state >= 0:
            bouts.setdefault(state, []).append(int(e - s))
    return bouts


def _gini(values: list[float]) -> float:
    """Gini coefficient of a list of non-negative values. 0 = equal, 1 = maximal inequality."""
    if not values or sum(values) == 0:
        return 0.0
    arr = sorted(values)
    n = len(arr)
    cumsum = 0.0
    for i, v in enumerate(arr):
        cumsum += (2 * (i + 1) - n - 1) * v
    return cumsum / (n * sum(arr))


def _generate_diagnostics(
    all_labels: np.ndarray | None = None,
    all_probs: np.ndarray | None = None,
    pooled_umap: np.ndarray | None = None,
) -> dict:
    """Generate clustering quality diagnostics to results/diagnostics/.

    When called from cmd_cluster(), receives live arrays.
    When called standalone (--diagnostics), loads from disk.
    """
    diag_dir = os.path.join(_res(), "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    shared_dir = os.path.join(_res(), "shared")
    ci_path = os.path.join(shared_dir, "cluster_info.json")
    if not os.path.exists(ci_path):
        print("[diagnostics] No cluster_info.json found. Run --cluster first.")
        return {}

    with open(ci_path) as f:
        ci = json.load(f)
    n_clusters = int(ci.get("n_clusters", 0))
    if n_clusters == 0:
        return {}

    # ---- FPS for converting frames → seconds ----
    try:
        import vieb_config as _vc_diag
        _cfg_diag = _vc_diag._load_config()
        fps = float(_cfg_diag.get("fps", 30))
    except Exception:
        fps = 30.0
    fps = max(1.0, fps)
    min_short_frames = max(1, int(fps * 0.5))  # bouts shorter than 0.5 s are "short"

    # ---- Load per-video label files; compute bout metrics in the same pass ----
    all_bout_lists: dict[int, list[int]] = {}  # state → list of bout lengths (frames)
    video_presence: dict[int, int] = {}
    label_arrays_disk: list[np.ndarray] = []
    prob_arrays_disk: list[np.ndarray] = []

    for fname in sorted(os.listdir(shared_dir)):
        if not fname.endswith("_labels.npy"):
            continue
        lbl = np.load(os.path.join(shared_dir, fname))
        # video presence
        present = set(int(s) for s in np.unique(lbl) if s >= 0)
        for s in present:
            video_presence[s] = video_presence.get(s, 0) + 1
        # bout metrics (per-video, no cross-video artifacts)
        for state, lengths in _compute_state_bouts(lbl).items():
            all_bout_lists.setdefault(state, []).extend(lengths)
        # for disk-load path
        if all_labels is None:
            label_arrays_disk.append(lbl)
            stem = fname.replace("_labels.npy", "")
            prob_path = os.path.join(shared_dir, f"{stem}_probs.npy")
            if os.path.exists(prob_path):
                prob_arrays_disk.append(np.load(prob_path))
            else:
                prob_arrays_disk.append(np.where(lbl >= 0, 1.0, 0.0).astype(np.float32))

    if all_labels is None:
        if not label_arrays_disk:
            print("[diagnostics] No label files found.")
            return {}
        all_labels = np.concatenate(label_arrays_disk)
        all_probs = np.concatenate(prob_arrays_disk)

    if all_probs is None:
        all_probs = np.where(all_labels >= 0, 1.0, 0.0).astype(np.float32)

    total_frames = len(all_labels)
    noise_count = int((all_labels == -1).sum())
    noise_frac = noise_count / max(1, total_frames)

    # ---- State occupancy ----
    occ_rows = []
    for k in range(-1, n_clusters):
        mask = all_labels == k
        count = int(mask.sum())
        if count == 0 and k >= 0:
            continue
        frac = count / max(1, total_frames)
        occ_rows.append({
            "state": k,
            "frame_count": count,
            "fraction": round(frac, 6),
            "n_videos_present": video_presence.get(k, 0) if k >= 0 else 0,
        })

    occ_df = pd.DataFrame(occ_rows)
    occ_df.to_csv(os.path.join(diag_dir, "state_occupancy.csv"), index=False)

    valid_mask = all_labels >= 0
    state_fracs = []
    for k in range(n_clusters):
        state_fracs.append(float((all_labels == k).sum()) / max(1, total_frames))
    largest_frac = max(state_fracs) if state_fracs else 0.0
    smallest_frac = min(f for f in state_fracs if f > 0) if any(f > 0 for f in state_fracs) else 0.0
    dominant_state_id = int(np.argmax(state_fracs)) if state_fracs else 0

    non_noise_probs = all_probs[valid_mask]
    mean_conf = float(non_noise_probs.mean()) if len(non_noise_probs) > 0 else 0.0
    low_conf_frac = float((non_noise_probs < 0.5).sum() / max(1, len(non_noise_probs)))

    # ---- Bout metrics ----
    # Global short-bout fraction: frames in short bouts / total non-noise frames
    total_short_frames = 0
    total_non_noise_frames = int(valid_mask.sum())
    bout_metrics: dict[str, dict] = {}
    dur_rows = []
    for k in range(n_clusters):
        lengths = all_bout_lists.get(k, [])
        if not lengths:
            continue
        arr = np.array(lengths, dtype=np.float64)
        short_count = int((arr < min_short_frames).sum())
        total_short_frames += int((arr[arr < min_short_frames]).sum())
        bm = {
            "n_bouts": len(lengths),
            "mean_dur_frames": round(float(arr.mean()), 2),
            "mean_dur_s": round(float(arr.mean()) / fps, 3),
            "median_dur_s": round(float(np.median(arr)) / fps, 3),
            "std_dur_s": round(float(arr.std()) / fps, 3),
            "short_bout_frac": round(short_count / max(1, len(lengths)), 4),
        }
        bout_metrics[str(k)] = bm
        dur_rows.append({
            "state": k,
            "n_bouts": bm["n_bouts"],
            "mean_dur_frames": bm["mean_dur_frames"],
            "mean_dur_s": bm["mean_dur_s"],
            "median_dur_s": bm["median_dur_s"],
            "std_dur_s": bm["std_dur_s"],
            "short_bout_frac": bm["short_bout_frac"],
        })

    global_short_frac = total_short_frames / max(1, total_non_noise_frames)

    if dur_rows:
        pd.DataFrame(dur_rows).to_csv(
            os.path.join(diag_dir, "state_duration_summary.csv"), index=False
        )

    # ---- Cluster balance metrics ----
    pos_fracs = [f for f in state_fracs if f > 0]
    if len(pos_fracs) > 1:
        import math
        ent = -sum(f * math.log(f) for f in pos_fracs if f > 0)
        max_ent = math.log(len(pos_fracs))
        state_entropy = round(ent / max_ent, 4) if max_ent > 0 else 1.0
    else:
        state_entropy = 0.0
    imbalance_score = round(_gini(pos_fracs), 4)

    # ---- Load index metadata for feature info ----
    index_path = os.path.join(_res(), "features", "index.json")
    n_features = 0
    use_wavelets = None  # None = unknown; never infer True/False when key is absent
    semantic_features: list[str] = []
    keypoint_groups: dict = {}
    if os.path.exists(index_path):
        with open(index_path) as f:
            idx = json.load(f)
        meta = _migrate_index_meta(idx.get("_meta", {}))
        n_features = int(meta.get("n_features", 0))
        _uw = meta.get("use_wavelets")
        use_wavelets = bool(_uw) if _uw is not None else None
        semantic_features = meta.get("semantic_features", [])
        keypoint_groups = meta.get("keypoint_groups", {})

    # ---- Load run manifest for params ----
    manifest_path = os.path.join(shared_dir, "run_manifest.json")
    manifest: dict = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    umap_dims = int(manifest.get("umap_dims", ci.get("umap_dims", 10)))
    min_cluster_size = int(manifest.get("min_cluster_size", ci.get("min_cluster_size", 50)))
    hdbscan_sample = int(manifest.get("hdbscan_sample", ci.get("hdbscan_sample", 0)))
    hdbscan_min_samples = int(manifest.get("hdbscan_min_samples", 0))

    # ---- Warnings ----
    warnings: list[dict] = []
    if n_clusters <= 3:
        warnings.append({
            "level": "warning",
            "message": f"Only {n_clusters} states found. Try lowering --min-cluster-size.",
            "action": f"python compare.py --cluster --min-cluster-size {max(10, min_cluster_size // 2)}",
        })
    if noise_frac == 0 and largest_frac > 0.85:
        warnings.append({
            "level": "warning",
            "message": "0% noise and one dominant state (>85%) may indicate overly smooth clustering.",
            "action": "Try increasing UMAP dims or lowering min_samples.",
        })
    if largest_frac > 0.90:
        warnings.append({
            "level": "error",
            "message": f"Largest state occupies {largest_frac*100:.1f}% of frames.",
            "action": f"python compare.py --cluster --min-cluster-size {max(10, min_cluster_size // 2)}",
        })
    if noise_frac > 0.50:
        warnings.append({
            "level": "error",
            "message": f"Over 50% of frames are noise ({noise_frac*100:.1f}%). HDBSCAN may be too aggressive.",
            "action": f"python compare.py --cluster --min-cluster-size {max(10, min_cluster_size // 2)}",
        })
    if n_clusters > 15:
        warnings.append({
            "level": "warning",
            "message": f"Many states discovered ({n_clusters}). Consider raising --min-cluster-size to merge similar states.",
            "action": f"python compare.py --cluster --min-cluster-size {min_cluster_size * 2}",
        })
    if umap_dims < 5:
        warnings.append({
            "level": "warning",
            "message": f"UMAP dimension is very low ({umap_dims}). May compress behavioral structure. Try --umap-dims 10.",
            "action": "python compare.py --cluster --umap-dims 10",
        })
    if global_short_frac > 0.35:
        warnings.append({
            "level": "warning",
            "message": f"More than 35% of frames are in very short bouts (<0.5 s). States may be over-split.",
            "action": f"python compare.py --cluster --min-cluster-size {min_cluster_size * 2}",
        })
    if hdbscan_sample > 0 and hdbscan_sample < total_frames // 20:
        warnings.append({
            "level": "info",
            "message": (
                f"HDBSCAN fit on {hdbscan_sample:,} of {total_frames:,} frames. "
                "Generalization may be limited."
            ),
            "action": "python compare.py --cluster --hdbscan-sample 0  (to use all frames)",
        })
    if n_features > 0 and n_features < 20:
        warnings.append({
            "level": "warning",
            "message": f"Low feature count ({n_features}). Consider enabling wavelets.",
            "action": "python compare.py --extract  (without --no-wavelets)",
        })
    if not semantic_features:
        skipped_groups = [g for g, info in keypoint_groups.items()
                         if isinstance(info, dict) and not info.get("resolved", True)]
        if skipped_groups:
            warnings.append({
                "level": "info",
                "message": f"Semantic features skipped (missing keypoint groups: {', '.join(skipped_groups)}).",
                "action": "Map keypoint groups in Settings > Column Mapping.",
            })

    # ---- Health verdict ----
    levels = {w["level"] for w in warnings}
    if "error" in levels:
        health_status = "failed"
    elif "warning" in levels:
        health_status = "suspicious"
    else:
        health_status = "good"

    diagnostics = {
        "n_states": n_clusters,
        "n_frames": total_frames,
        "noise_frac": round(noise_frac, 4),
        "largest_state_frac": round(largest_frac, 4),
        "smallest_state_frac": round(smallest_frac, 4),
        "dominant_state_id": dominant_state_id,
        "mean_confidence": round(mean_conf, 4),
        "low_confidence_frac": round(low_conf_frac, 4),
        "state_entropy": state_entropy,
        "imbalance_score": imbalance_score,
        "short_bout_frac": round(global_short_frac, 4),
        "bout_metrics": bout_metrics,
        "n_features": n_features,
        "use_wavelets": use_wavelets,
        "umap_dims": umap_dims,
        "min_cluster_size": min_cluster_size,
        "hdbscan_sample": hdbscan_sample,
        "hdbscan_min_samples": hdbscan_min_samples,
        "health_status": health_status,
        "warnings": warnings,
    }

    with open(os.path.join(diag_dir, "cluster_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)

    # ---- UMAP 2D sample for scatter visualization ----
    _save_umap_sample(all_labels, all_probs, pooled_umap, diag_dir, n_clusters)
    _save_umap_embedding_plot(diag_dir)

    # ---- Overview plot ----
    _save_diagnostics_plot(diagnostics, occ_df, diag_dir)

    return diagnostics


def _save_umap_sample(
    all_labels: np.ndarray,
    all_probs: np.ndarray,
    pooled_umap: np.ndarray | None,
    diag_dir: str,
    n_clusters: int,
    max_points: int = 50_000,
) -> None:
    """Save a 2D UMAP scatter sample to umap_sample.csv."""
    if pooled_umap is None:
        return

    n_total = len(all_labels)
    if n_total == 0:
        return

    n_sample = min(max_points, n_total)
    rng = np.random.default_rng(42)
    idx = np.sort(rng.choice(n_total, n_sample, replace=False))

    sampled_umap = pooled_umap[idx]
    sampled_labels = all_labels[idx]
    sampled_probs = all_probs[idx]

    if sampled_umap.shape[1] == 2:
        umap_2d = sampled_umap
    else:
        try:
            import umap as umap_lib
            reducer_2d = umap_lib.UMAP(
                n_components=2, n_neighbors=30, min_dist=0.1,
                random_state=42, low_memory=True, verbose=False,
            )
            umap_2d = reducer_2d.fit_transform(sampled_umap)
        except Exception:
            return

    sample_df = pd.DataFrame({
        "umap_1": umap_2d[:, 0],
        "umap_2": umap_2d[:, 1],
        "label": sampled_labels,
        "prob": sampled_probs,
    })
    sample_df.to_csv(os.path.join(diag_dir, "umap_sample.csv"), index=False)


def _save_umap_embedding_plot(diag_dir: str) -> None:
    """Save a standalone UMAP embedding visualization colored by state."""
    umap_path = os.path.join(diag_dir, "umap_sample.csv")
    if not os.path.exists(umap_path):
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
    except ImportError:
        return

    umap_df = pd.read_csv(umap_path)
    if umap_df.empty:
        return

    valid = umap_df[umap_df["label"] >= 0]
    noise = umap_df[umap_df["label"] < 0]
    fig, ax = _plt.subplots(figsize=(7, 6))
    if not noise.empty:
        ax.scatter(
            noise["umap_1"], noise["umap_2"],
            c="#CCCCCC", s=1, alpha=0.25, rasterized=True, label="Noise",
        )
    if not valid.empty:
        scatter = ax.scatter(
            valid["umap_1"], valid["umap_2"],
            c=valid["label"], cmap="tab20", s=1.2, alpha=0.55,
            rasterized=True,
        )
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="State")
    ax.set_title("UMAP Embedding by State", fontsize=11, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(diag_dir, "umap_embedding_by_state.png"), dpi=150)
    _plt.close(fig)


def _save_diagnostics_plot(diagnostics: dict, occ_df: pd.DataFrame, diag_dir: str) -> None:
    """Save a 2x2 diagnostic overview plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
    except ImportError:
        return

    n_clusters = diagnostics["n_states"]
    fig, axes = _plt.subplots(2, 2, figsize=(12, 10))

    # ---- [0,0] State occupancy bars ----
    ax = axes[0, 0]
    state_rows = occ_df[occ_df["state"] >= 0].sort_values("state")
    if not state_rows.empty:
        states = state_rows["state"].values
        fracs = state_rows["fraction"].values * 100
        colors = _plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 1)))
        bar_colors = [colors[int(s) % len(colors)] for s in states]
        ax.barh(range(len(states)), fracs, color=bar_colors, alpha=0.85)
        ax.set_yticks(range(len(states)))
        ax.set_yticklabels([f"S{s}" for s in states], fontsize=8)
        ax.set_xlabel("Occupancy (%)")
        ax.invert_yaxis()
        noise_row = occ_df[occ_df["state"] == -1]
        if not noise_row.empty:
            noise_pct = float(noise_row["fraction"].values[0]) * 100
            ax.set_title(f"State Occupancy (noise: {noise_pct:.1f}%)", fontsize=10, fontweight="bold")
        else:
            ax.set_title("State Occupancy", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- [0,1] UMAP scatter ----
    ax2 = axes[0, 1]
    umap_path = os.path.join(diag_dir, "umap_sample.csv")
    if os.path.exists(umap_path):
        umap_df = pd.read_csv(umap_path)
        valid = umap_df[umap_df["label"] >= 0]
        noise = umap_df[umap_df["label"] < 0]
        if not noise.empty:
            ax2.scatter(noise["umap_1"], noise["umap_2"], c="#CCCCCC", s=1, alpha=0.3, rasterized=True)
        if not valid.empty:
            scatter = ax2.scatter(
                valid["umap_1"], valid["umap_2"],
                c=valid["label"], cmap="tab20", s=1, alpha=0.5, rasterized=True,
            )
        ax2.set_title("UMAP Embedding (sampled)", fontsize=10, fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "No UMAP sample available", ha="center", va="center",
                 transform=ax2.transAxes, color="#999")
    ax2.set_xticks([])
    ax2.set_yticks([])

    # ---- [1,0] Confidence histogram ----
    ax3 = axes[1, 0]
    if os.path.exists(umap_path):
        umap_df = pd.read_csv(umap_path)
        probs = umap_df["prob"].values
        ax3.hist(probs, bins=50, color="#4E79A7", alpha=0.8, edgecolor="white", linewidth=0.3)
        ax3.axvline(0.5, color="#E63946", linewidth=1.5, linestyle="--", label="0.5 threshold")
        ax3.set_xlabel("HDBSCAN Probability")
        ax3.set_ylabel("Count")
        ax3.legend(fontsize=8)
    ax3.set_title("Confidence Distribution", fontsize=10, fontweight="bold")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    # ---- [1,1] Parameter summary + warnings ----
    ax4 = axes[1, 1]
    ax4.axis("off")
    lines = [
        f"States: {diagnostics['n_states']}",
        f"Frames: {diagnostics['n_frames']:,}",
        f"Noise: {diagnostics['noise_frac']*100:.1f}%",
        f"Largest state: {diagnostics['largest_state_frac']*100:.1f}%",
        f"Mean confidence: {diagnostics['mean_confidence']:.3f}",
        f"Low confidence (<0.5): {diagnostics['low_confidence_frac']*100:.1f}%",
        "",
        f"UMAP dims: {diagnostics['umap_dims']}",
        f"min_cluster_size: {diagnostics['min_cluster_size']}",
        f"Features: {diagnostics['n_features']}",
        f"Wavelets: {'yes' if diagnostics['use_wavelets'] else 'no'}",
    ]
    if diagnostics["warnings"]:
        lines.append("")
        lines.append("WARNINGS:")
        for w in diagnostics["warnings"]:
            icon = "!" if w["level"] == "error" else "*"
            lines.append(f"  {icon} {w['message']}")

    ax4.text(0.05, 0.95, "\n".join(lines), transform=ax4.transAxes,
             fontsize=9, verticalalignment="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#F5F5F5", alpha=0.8))
    ax4.set_title("Run Parameters", fontsize=10, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(diag_dir, "cluster_overview.png"), dpi=150)
    _plt.close(fig)


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
    cfg_path = _project_config_path()
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


def _plot_transition_heatmaps(
    group_matrices: dict,
    n_clusters: int,
    save_path: str,
    group_label: str = "Context",
):
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
        ax.set_title(f"{group_label} {grp}")
        ax.set_xlabel("To state")
        ax.set_ylabel("From state")
        ax.set_xticks(range(n_clusters))
        ax.set_yticks(range(n_clusters))
        for i in range(n_clusters):
            for j in range(n_clusters):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        fontsize=7, color="white" if mat[i, j] > vmax * 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle(f"Mean State Transition Probabilities by {group_label}", fontsize=11)
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


def _save_state_occupancy_plot(df: pd.DataFrame, state_cols: list[str], save_path: str) -> None:
    """Persist a compact global state occupancy figure for export/artifacts."""
    if df.empty or not state_cols:
        return
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    means = df[state_cols].mean().fillna(0.0)
    state_ids = [int(c.split("_")[1]) for c in state_cols]
    fig_h = max(3.5, 0.24 * len(state_ids) + 1.4)
    fig, ax = plt.subplots(figsize=(7.0, fig_h))
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(state_ids))))
    ax.barh(range(len(state_ids)), means.values * 100, color=colors, alpha=0.85)
    ax.set_yticks(range(len(state_ids)))
    ax.set_yticklabels([f"S{s}" for s in state_ids], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean session occupancy (%)")
    ax.set_title("State Occupancy", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, color="#EEEEEE", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def _write_configured_correlations(df: pd.DataFrame, state_cols: list[str]) -> None:
    schema = _vc.get_metadata_schema()
    rows = []
    for corr in schema.get("correlations", []):
        if not isinstance(corr, dict) or not corr.get("enabled", False):
            continue
        for col in corr.get("columns", []):
            if col not in df.columns:
                print(f"[info] Correlation skipped for {col}: column not found.")
                continue
            x = pd.to_numeric(df[col], errors="coerce")
            if x.notna().sum() < 3:
                print(f"[info] Correlation skipped for {col}: need at least 3 numeric values.")
                continue
            for target in state_cols:
                y = pd.to_numeric(df[target], errors="coerce")
                valid = x.notna() & y.notna()
                if valid.sum() < 3:
                    continue
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = float(np.corrcoef(x[valid], y[valid])[0, 1])
                rows.append({
                    "analysis": corr.get("name", "Configured correlations"),
                    "column": col,
                    "target_type": "state_fraction",
                    "target": target,
                    "n": int(valid.sum()),
                    "pearson_r": r,
                })
    if not rows:
        return
    out = pd.DataFrame(rows)
    out_path = os.path.join(_res(), "comparison", "correlations.csv")
    out.to_csv(out_path, index=False)
    print(f"Correlations saved: results/comparison/correlations.csv ({len(out)} rows)")


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
    if "stem" not in meta.columns:
        sys.exit("[ERROR] metadata.csv needs a session identifier column. "
                 "Map one to 'session_id' or provide filename/source_file.")

    df = df_states.merge(meta, on="stem", how="left")

    os.makedirs(os.path.join(_res(), "comparison"), exist_ok=True)
    try:
        from analysis_design import write_analysis_design
        analysis_design = write_analysis_design(df, _res(), _vc._load_config())
    except Exception as e:
        print(f"[warn] Could not write analysis design: {e}")
        analysis_design = {
            "subject_col": "animal_id" if "animal_id" in df.columns else None,
            "time_col": "day" if "day" in df.columns else None,
            "time_order": sorted(df["day"].dropna().unique().tolist()) if "day" in df.columns else None,
            "condition_cols": ["context"] if "context" in df.columns else [],
            "group_cols": [],
            "continuous_cols": [],
            "detected_mode": "time_and_condition" if {"day", "context"}.issubset(df.columns) else "minimal",
        }
    try:
        import metadata_schema as _ms
        schema_report = _ms.metadata_schema_report(meta, _vc._load_config())
        schema_report["summary_rows"] = int(len(df))
        schema_report["unmatched_stems"] = sorted(
            set(df_states["stem"].astype(str)) - set(meta["stem"].astype(str))
        )
        os.makedirs(_res(), exist_ok=True)
        with open(os.path.join(_res(), "metadata_schema_report.json"), "w", encoding="utf-8") as f:
            json.dump(schema_report, f, indent=2, default=str)
        print("Metadata schema report saved: results/metadata_schema_report.json")
    except Exception as e:
        print(f"[warn] Could not write metadata schema report: {e}")
    df.to_csv(os.path.join(_res(), "comparison", "summary_table.csv"), index=False)
    print(f"Summary table saved: results/comparison/summary_table.csv  ({len(df)} videos)")
    _write_overview_summary(
        df,
        cluster_info,
        index,
        active_run_id=cluster_info.get("run_id") or cluster_info.get("active_run_id"),
    )

    # ---- Characterization: bouts.csv + state_summary.csv ----
    char_dir = os.path.join(_res(), "characterization")
    os.makedirs(char_dir, exist_ok=True)
    all_bouts = []
    design_meta_cols = [
        analysis_design.get("subject_col"),
        analysis_design.get("time_col"),
        *(analysis_design.get("condition_cols") or []),
        *(analysis_design.get("group_cols") or []),
    ]
    meta_cols = []
    for c in ["context", "animal_id", "day", "experiment", *design_meta_cols]:
        if c and c in df.columns and c not in meta_cols:
            meta_cols.append(c)
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

    try:
        from sequence_artifacts import build_sequence_artifacts
        build_sequence_artifacts(
            df,
            analysis_design,
            _res(),
            fps=fps,
            n_clusters=n_clusters,
        )
    except Exception as e:
        print(f"[warn] Sequence artifact generation failed: {e}")

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
    _save_state_occupancy_plot(
        df,
        state_cols,
        os.path.join(char_dir, "state_occupancy.png"),
    )

    # ---- Transition matrix outputs ----
    _trans_meta_cols = ["stem"]
    for c in ["context", "day", "animal_id", "experiment", *design_meta_cols]:
        if c and c in meta.columns and c not in _trans_meta_cols:
            _trans_meta_cols.append(c)
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

    try:
        from report_plots import generate_mode_driven_plots
        generate_mode_driven_plots(
            df,
            df_trans_full,
            bouts_df,
            analysis_design,
            _res(),
        )
    except Exception as e:
        print(f"[warn] Mode-driven plot generation failed: {e}")

    analysis_groups = _vc.get_enabled_analysis_groups(df)

    # Heatmaps for configured grouping variables.
    transition_groups = [
        g for g in analysis_groups
        if g.get("available", True) and "transition_matrix" in g.get("plots", [])
    ]
    for group in transition_groups:
        group_col = group["column"]
        if group_col not in df_trans.columns or not df_trans[group_col].notna().any():
            print(f"[info] Transition report skipped for {group_col}: column missing or empty.")
            continue
        group_matrices = {}
        for value, grp in df_trans.groupby(group_col):
            mats = []
            for _, row in grp.iterrows():
                mat = np.array([[row[f"trans_{i}_{j}"] for j in range(n_clusters)]
                                for i in range(n_clusters)])
                mats.append(mat)
            group_matrices[value] = np.stack(mats).mean(axis=0)
        filename = (
            "transition_by_context.png"
            if group_col == "context"
            else f"transition_by_{_slug(group_col)}.png"
        )
        _plot_transition_heatmaps(
            group_matrices, n_clusters,
            os.path.join(_res(), "comparison", filename),
            group_label=group.get("name", group_col),
        )

    # ---- Plots ----
    def boxplot_by_group(group_col, save_path, group_label):
        if group_col not in df.columns:
            print(f"  SKIP {os.path.basename(save_path)}: '{group_col}' column not found")
            return
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
            bp = ax.boxplot(data, tick_labels=[str(g) for g in groups], patch_artist=True)
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

    # State-fraction comparisons for configured grouping variables.
    for group in analysis_groups:
        group_col = group["column"]
        if "state_fraction" not in group.get("plots", []):
            continue
        if not group.get("available", True):
            if group_col == "fear":
                print("[info] No fear column found; skipping fear-specific report.")
            else:
                print(f"[info] State-fraction report skipped for {group_col}: {group.get('skip_reason')}")
            continue
        filename = _state_group_filename(group_col)
        label = group.get("name", group_col.replace("_", " ").title())
        boxplot_by_group(
            group_col,
            os.path.join(_res(), "comparison", filename),
            label,
        )

    # Per-animal trajectory across days
    if "animal_id" in df.columns and "day" in df.columns:
        _plot_animal_trajectories(df, state_cols, n_clusters)

    # Statistical summary to terminal
    print(f"\n--- Group means (state fractions) ---")
    for group in analysis_groups:
        group_col = group["column"]
        if group_col not in df.columns:
            continue
        if df[group_col].notna().sum() == 0:
            continue
        print(f"\nBy {group.get('name', group_col)}:")
        group_means = df.groupby(group_col)[state_cols].mean().round(3)
        print(group_means.to_string())

    _write_configured_correlations(df, state_cols)

    motif_groups = [
        g for g in analysis_groups
        if g.get("available", True) and "motif_enrichment" in g.get("plots", [])
    ]
    if motif_groups:
        for group in motif_groups:
            group_col = group["column"]
            values = df[group_col].dropna().astype(str).unique().tolist()
            if len(values) < 2:
                print(f"[info] Motif report skipped for {group_col}: need at least two values.")
                continue
            try:
                print(f"\n--- Motifs by {group.get('name', group_col)} ---")
                prefix = "motifs" if group_col == "context" else f"motifs_by_{_slug(group_col)}"
                cmd_motifs(min_confidence=min_confidence, group_col=group_col, output_prefix=prefix)
            except SystemExit as e:
                print(f"[info] Motif report skipped: {e}")
    else:
        print("[info] Motif report skipped: no configured motif enrichment groups are available.")

    print(f"\nResults in results/comparison/")




# ---------------------------------------------------------------------------
# Step 4: Per-animal scalar summary (delegates to quantify.py)
# ---------------------------------------------------------------------------

def _motif_counts(labels: np.ndarray, n: int) -> dict[tuple[int, ...], int]:
    """Count valid overlapping state motifs of length n."""
    counts: dict[tuple[int, ...], int] = {}
    if len(labels) < n:
        return counts
    for i in range(len(labels) - n + 1):
        window = labels[i:i + n]
        if np.all(window >= 0):
            motif = tuple(int(v) for v in window)
            counts[motif] = counts.get(motif, 0) + 1
    return counts


def _format_motif(motif: tuple[int, ...]) -> str:
    """Format as Python tuple string for ast.literal_eval compatibility."""
    return str(motif)


def _is_degenerate_motif(motif: tuple[int, ...]) -> bool:
    """True if every state in the tuple is identical, e.g. (48, 48), (3, 3, 3).

    Such motifs encode bout duration (one state persisting), not sequence
    structure, so they are excluded from motif ranking. Mixed tuples like
    (12, 47) or (3, 3, 7) are not degenerate and are kept.
    """
    return len(set(motif)) <= 1


def _pick_motif_contexts(context_values: list[str]) -> tuple[str, str]:
    available = sorted({str(v) for v in context_values if str(v) and str(v) != "nan"})
    if len(available) < 2:
        sys.exit("Motif discovery needs at least two context values in metadata.csv.")

    label_a = str(_vc.get_condition_a_label())
    label_b = str(_vc.get_condition_b_label())
    if label_a in available and label_b in available and label_a != label_b:
        return label_a, label_b
    if "A" in available and "B" in available:
        return "A", "B"
    return available[0], available[1]


def _pick_motif_groups(values: list[str], group_col: str) -> tuple[str, str]:
    if group_col == "context":
        return _pick_motif_contexts(values)
    available = sorted({str(v) for v in values if str(v) and str(v) != "nan"})
    if len(available) < 2:
        sys.exit(f"Motif discovery needs at least two values in '{group_col}'.")
    return available[0], available[1]


def _plot_motif_heatmap(df: pd.DataFrame, save_path: str, limit: int = 20) -> None:
    import matplotlib.pyplot as plt

    if df.empty:
        return

    plot_df = df.copy()
    if "abs_log2_enrichment" in plot_df.columns:
        plot_df = plot_df.sort_values("abs_log2_enrichment", ascending=False)
    else:
        plot_df = plot_df.sort_values("enrichment_ratio", ascending=False)
    plot_df = plot_df.head(limit)

    mat = plot_df[["context_A_freq", "context_B_freq"]].to_numpy(dtype=float)
    labels = plot_df["motif"].astype(str).tolist()
    group_a = str(plot_df["context_A"].iloc[0]) if "context_A" in plot_df.columns and not plot_df.empty else "Group A"
    group_b = str(plot_df["context_B"].iloc[0]) if "context_B" in plot_df.columns and not plot_df.empty else "Group B"

    fig_h = max(4.0, 0.28 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(6.5, fig_h))
    im = ax.imshow(mat, cmap="magma", aspect="auto")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([group_a, group_b])
    ax.set_title("Top Group-Enriched Motifs")
    ax.set_xlabel("Frequency within group")
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            ax.text(c, r, f"{mat[r, c]:.3f}", ha="center", va="center",
                    color="white" if mat[r, c] > mat.max() * 0.55 else "black",
                    fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


def cmd_motifs(
    min_confidence: float = 0.0,
    group_col: str = "context",
    output_prefix: str = "motifs",
):
    """
    Discover bigram/trigram state motifs enriched between two metadata groups.

    Outputs:
      - results/comparison/motifs.csv
      - results/comparison/motif_heatmap.png
    """
    index_path = os.path.join(_res(), "features", "index.json")
    cluster_info_path = os.path.join(_res(), "shared", "cluster_info.json")
    if not os.path.exists(index_path):
        sys.exit("No feature index found. Run --extract first.")
    if not os.path.exists(cluster_info_path):
        sys.exit("No cluster_info.json found. Run --cluster first.")
    if not os.path.exists(_meta()):
        sys.exit("metadata.csv not found.")

    with open(index_path) as f:
        index = json.load(f)

    meta = pd.read_csv(_meta())
    meta = _vc.normalize_metadata_columns(meta)
    if "stem" not in meta.columns or group_col not in meta.columns:
        sys.exit(f"metadata.csv must include a session identifier and '{group_col}' for motif discovery.")
    meta_by_stem = meta.drop_duplicates("stem").set_index("stem")

    context_a, context_b = _pick_motif_groups(meta[group_col].dropna().astype(str).tolist(), group_col)
    print(f"Extracting motifs by {group_col}: {context_a} vs {context_b}")
    if min_confidence > 0.0:
        print(f"Applying min-confidence filter: {min_confidence}")

    counts = {
        context_a: {"bigram": {}, "trigram": {}},
        context_b: {"bigram": {}, "trigram": {}},
    }
    totals = {
        context_a: {"bigram": 0, "trigram": 0},
        context_b: {"bigram": 0, "trigram": 0},
    }
    videos_used = {context_a: 0, context_b: 0}

    for stem in sorted(k for k in index.keys() if k != "_meta"):
        if stem not in meta_by_stem.index:
            continue
        ctx = str(meta_by_stem.loc[stem, group_col])
        if ctx not in (context_a, context_b):
            continue
        labels_path = os.path.join(_res(), "shared", f"{stem}_labels.npy")
        if not os.path.exists(labels_path):
            continue
        labels = np.load(labels_path).astype(np.int32)
        if min_confidence > 0.0:
            probs_path = os.path.join(_res(), "shared", f"{stem}_probs.npy")
            if os.path.exists(probs_path):
                probs = np.load(probs_path)
                labels = labels.copy()
                labels[probs < min_confidence] = -1

        videos_used[ctx] += 1
        for n, typ in ((2, "bigram"), (3, "trigram")):
            c = _motif_counts(labels, n)
            totals[ctx][typ] += sum(c.values())
            dest = counts[ctx][typ]
            for motif, value in c.items():
                dest[motif] = dest.get(motif, 0) + value

    rows = []
    for typ in ("bigram", "trigram"):
        motifs = sorted(set(counts[context_a][typ]) | set(counts[context_b][typ]))
        total_a = totals[context_a][typ]
        total_b = totals[context_b][typ]
        for motif in motifs:
            # Skip degenerate motifs (a single state repeating); these encode
            # bout duration, reported separately in bout_duration_by_context.csv.
            if _is_degenerate_motif(motif):
                continue
            count_a = int(counts[context_a][typ].get(motif, 0))
            count_b = int(counts[context_b][typ].get(motif, 0))
            freq_a = count_a / total_a if total_a else 0.0
            freq_b = count_b / total_b if total_b else 0.0
            eps_a = 0.5 / total_a if total_a else 0.5
            eps_b = 0.5 / total_b if total_b else 0.5
            ratio = (freq_a + eps_a) / (freq_b + eps_b)
            se = np.sqrt(1.0 / (count_a + 0.5) + 1.0 / (count_b + 0.5))
            ci_low = float(np.exp(np.log(ratio) - 1.96 * se))
            ci_high = float(np.exp(np.log(ratio) + 1.96 * se))
            rows.append({
                "motif": _format_motif(motif),
                "type": typ,
                "context_A": context_a,
                "context_B": context_b,
                "context_A_count": count_a,
                "context_B_count": count_b,
                "context_A_freq": freq_a,
                "context_B_freq": freq_b,
                "enrichment_ratio": ratio,
                "log2_enrichment": float(np.log2(ratio)),
                "abs_log2_enrichment": float(abs(np.log2(ratio))),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "flagged": bool(ratio >= 2.0 or ratio <= 0.5),
            })

    out_dir = os.path.join(_res(), "comparison")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"{output_prefix}.csv")
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(
            ["abs_log2_enrichment", "type", "motif"],
            ascending=[False, True, True],
        )
    df.to_csv(out_csv, index=False)
    print(f"Motifs saved: results/comparison/{output_prefix}.csv  ({len(df)} motifs)")
    print(f"Videos used: {context_a}={videos_used[context_a]}, {context_b}={videos_used[context_b]}")
    print(f"Totals: bigrams {totals[context_a]['bigram']:,}/{totals[context_b]['bigram']:,}, "
          f"trigrams {totals[context_a]['trigram']:,}/{totals[context_b]['trigram']:,}")

    if not df.empty:
        heatmap_name = "motif_heatmap.png" if output_prefix == "motifs" else f"{output_prefix}_heatmap.png"
        _plot_motif_heatmap(df, os.path.join(out_dir, heatmap_name))

    # ---- Supplementary bout-based motif outputs → results/motifs/ ----
    if group_col == "context":
        _write_bout_motifs(meta_by_stem, index, context_a, context_b, min_confidence)


def _write_bout_motifs(
    meta_by_stem: pd.DataFrame,
    index: dict,
    context_a: str,
    context_b: str,
    min_confidence: float,
):
    """Produce bout-sequence n-gram tables in results/motifs/."""
    bouts_path = os.path.join(_res(), "characterization", "bouts.csv")
    if not os.path.exists(bouts_path):
        print("[info] Bout-based motif tables skipped: run --report first to generate bouts.csv")
        return

    bouts = pd.read_csv(bouts_path)
    if bouts.empty:
        return

    motifs_dir = os.path.join(_res(), "motifs")
    os.makedirs(motifs_dir, exist_ok=True)

    # Enrich bouts with prev_state / next_state
    enriched_rows = []
    for stem, grp in bouts.groupby("stem"):
        grp = grp.sort_values("start_frame").reset_index(drop=True)
        states = grp["state"].tolist()
        for i, row in grp.iterrows():
            r = row.to_dict()
            idx = grp.index.get_loc(i)
            r["prev_state"] = int(states[idx - 1]) if idx > 0 else -1
            r["next_state"] = int(states[idx + 1]) if idx < len(states) - 1 else -1
            enriched_rows.append(r)

    enriched_df = pd.DataFrame(enriched_rows)
    enriched_df.to_csv(os.path.join(motifs_dir, "bouts.csv"), index=False)
    print(f"  Bout sequences: results/motifs/bouts.csv ({len(enriched_df)} bouts)")

    # Extract n-grams from bout sequences (not frame labels)
    seq_rows = []
    global_counts: dict[str, dict[tuple, int]] = {"bigram": {}, "trigram": {}}
    ctx_counts: dict[str, dict[str, dict[tuple, int]]] = {}
    ctx_totals: dict[str, dict[str, int]] = {}

    for stem, grp in bouts.groupby("stem"):
        grp = grp.sort_values("start_frame")
        states = grp["state"].tolist()
        ctx = str(meta_by_stem.loc[stem, "context"]) if stem in meta_by_stem.index else ""

        for n, typ in ((2, "bigram"), (3, "trigram")):
            for i in range(len(states) - n + 1):
                motif = tuple(int(s) for s in states[i:i + n])
                global_counts[typ][motif] = global_counts[typ].get(motif, 0) + 1

                if ctx:
                    ctx_counts.setdefault(ctx, {}).setdefault(typ, {})
                    ctx_counts[ctx][typ][motif] = ctx_counts[ctx][typ].get(motif, 0) + 1
                    ctx_totals.setdefault(ctx, {}).setdefault(typ, 0)
                    ctx_totals[ctx][typ] += 1

                meta_cols = {}
                if stem in meta_by_stem.index:
                    for c in ["context", "animal_id", "day", "experiment"]:
                        if c in meta_by_stem.columns:
                            meta_cols[c] = meta_by_stem.loc[stem, c]

                seq_rows.append({
                    "stem": stem,
                    "type": typ,
                    "motif": str(motif),
                    "position": i,
                    **meta_cols,
                })

    seq_df = pd.DataFrame(seq_rows)
    seq_df.to_csv(os.path.join(motifs_dir, "motif_sequences.csv"), index=False)
    print(f"  Motif sequences: results/motifs/motif_sequences.csv ({len(seq_df)} occurrences)")

    # Global frequency summary
    summary_rows = []
    for typ in ("bigram", "trigram"):
        total = sum(global_counts[typ].values())
        for motif, count in sorted(global_counts[typ].items(), key=lambda x: -x[1]):
            summary_rows.append({
                "type": typ,
                "motif": str(motif),
                "count": count,
                "frequency": count / total if total else 0.0,
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(motifs_dir, "motif_summary.csv"), index=False)
    print(f"  Motif summary: results/motifs/motif_summary.csv ({len(summary_df)} motifs)")

    # Context enrichment (bout-based)
    all_contexts = sorted(ctx_counts.keys())
    enrichment_rows = []
    for typ in ("bigram", "trigram"):
        all_motifs = sorted(global_counts[typ].keys())
        global_total = sum(global_counts[typ].values())
        for motif in all_motifs:
            global_freq = global_counts[typ][motif] / global_total if global_total else 0.0
            for ctx in all_contexts:
                ctx_total = ctx_totals.get(ctx, {}).get(typ, 0)
                ctx_count = ctx_counts.get(ctx, {}).get(typ, {}).get(motif, 0)
                ctx_freq = ctx_count / ctx_total if ctx_total else 0.0
                ratio = ctx_freq / global_freq if global_freq > 0 else 0.0
                enrichment_rows.append({
                    "type": typ,
                    "motif": str(motif),
                    "context": ctx,
                    "count": ctx_count,
                    "frequency": ctx_freq,
                    "global_frequency": global_freq,
                    "enrichment_ratio": ratio,
                })
    enrichment_df = pd.DataFrame(enrichment_rows)
    enrichment_df.to_csv(os.path.join(motifs_dir, "motif_context_enrichment.csv"), index=False)
    print(f"  Context enrichment: results/motifs/motif_context_enrichment.csv ({len(enrichment_df)} rows)")

    # ---- Bout Duration by Context (separate from sequence motifs) ----
    # Repeated-state "motifs" really measure how long a single state persists.
    # That is a legitimate but distinct analysis, reported here rather than as a
    # sequence motif.
    _write_bout_duration_by_context(bouts, meta_by_stem)

    print(f"Motifs → results/motifs/")


def _write_bout_duration_by_context(bouts: pd.DataFrame, meta_by_stem: pd.DataFrame):
    """Per-(state, context) bout-duration summary → comparison/bout_duration_by_context.csv."""
    if bouts is None or bouts.empty or "state" not in bouts.columns:
        return

    df = bouts.copy()
    # Ensure a context column (bouts.csv has it when context is in metadata;
    # otherwise derive from metadata by stem).
    if "context" not in df.columns or df["context"].isna().all():
        df["context"] = df["stem"].map(
            lambda s: str(meta_by_stem.loc[s, "context"])
            if s in meta_by_stem.index and "context" in meta_by_stem.columns else ""
        )
    df["context"] = df["context"].fillna("").astype(str)
    df = df[df["context"].str.strip() != ""]
    if df.empty:
        return

    # Duration: prefer existing column, else derive from frames (1 frame == 1 unit
    # if start/end frames are present; left in frames is fine for relative compare,
    # but we keep duration_sec semantics when available).
    if "duration_sec" not in df.columns:
        df = df.assign(duration_sec=df.get("end_frame", 0) - df.get("start_frame", 0) + 1)
    df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce")
    df = df.dropna(subset=["duration_sec"])
    if df.empty:
        return

    global_mean = df.groupby("state")["duration_sec"].mean()
    rows = []
    for (state, ctx), grp in df.groupby(["state", "context"]):
        g_mean = float(global_mean.get(state, float("nan")))
        ctx_mean = float(grp["duration_sec"].mean())
        rows.append({
            "state_id": int(state),
            "context": ctx,
            "bout_count": int(len(grp)),
            "mean_bout_dur_sec": round(ctx_mean, 4),
            "median_bout_dur_sec": round(float(grp["duration_sec"].median()), 4),
            "global_mean_dur_sec": round(g_mean, 4) if g_mean == g_mean else float("nan"),
            "duration_enrichment": (
                round(ctx_mean / g_mean, 4) if g_mean and g_mean == g_mean else float("nan")
            ),
        })

    out_df = pd.DataFrame(rows).sort_values(["state_id", "context"])
    out_dir = os.path.join(_res(), "comparison")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "bout_duration_by_context.csv")
    out_df.to_csv(out_path, index=False)
    print(f"  Bout duration by context: results/comparison/bout_duration_by_context.csv ({len(out_df)} rows)")


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


def _save_contrast_vector_comparison_plot(contrast_df: pd.DataFrame, save_path: str) -> None:
    """Persist a comparison figure for per-cohort or per-animal contrast vectors."""
    if contrast_df is None or contrast_df.empty or "contrast_magnitude" not in contrast_df.columns:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plot_df = contrast_df.copy()
    plot_df = plot_df[plot_df["contrast_magnitude"].notna()]
    if plot_df.empty:
        return
    label_col = "cohort_label" if "cohort_label" in plot_df.columns else "animal_id"
    if label_col not in plot_df.columns:
        return

    plot_df[label_col] = plot_df[label_col].astype(str)
    plot_df = plot_df.sort_values("contrast_magnitude", ascending=True)
    fig_h = max(3.5, 0.28 * len(plot_df) + 1.5)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(plot_df))))
    ax.barh(plot_df[label_col].values, plot_df["contrast_magnitude"].astype(float).values, color=colors, alpha=0.85)
    ax.set_xlabel("Contrast magnitude")
    ax.set_ylabel("Cohort" if label_col == "cohort_label" else "Animal")
    ax.set_title("Behavioral Contrast Vector Comparison", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, color="#EEEEEE", zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def cmd_quantify(cohort: str | None = None, min_confidence: float = 0.0):
    """Build master_table.csv via quantify.py."""
    try:
        from quantify import build_master_table, compute_contrast_vector
        build_master_table(cohort_path=cohort, min_confidence=min_confidence)
    except ImportError:
        sys.exit("[ERROR] quantify.py not found in project directory.")

    from quantify import coerce_id_column

    print("\nComputing behavioral contrast vectors...")
    try:
        contrast_df = compute_contrast_vector(
            summary_csv=os.path.join(_res(), "comparison", "summary_table.csv"),
            output_dir=os.path.join(_res(), "quantification"),
            cohort_csv=cohort,
        )
        _save_contrast_vector_comparison_plot(
            contrast_df,
            os.path.join(_res(), "comparison", "contrast_vector_comparison.png"),
        )

        master_path = os.path.join(_res(), "quantification", "master_table.csv")
        if os.path.exists(master_path) and "animal_id" in contrast_df.columns:
            master = pd.read_csv(master_path)
            master = coerce_id_column(master)
            contrast_df = coerce_id_column(contrast_df)
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
            master = coerce_id_column(master)
            lr_reset = coerce_id_column(lr_df.reset_index())
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
    parser.add_argument("--backfill-video-paths", action="store_true",
                        help="Re-resolve missing video_path fields in results/features/index.json "
                             "for H5-extracted entries, using the H5 manifest's video-path column "
                             "or a raw_videos_dir extension match (does not re-extract features)")
    parser.add_argument("--cluster", action="store_true",
                        help="Fit shared UMAP+HDBSCAN clusterer across all videos")
    parser.add_argument("--report", action="store_true",
                        help="Generate comparison plots using metadata.csv")
    parser.add_argument("--summarize", action="store_true",
                        help="[deprecated] Use --quantify instead")
    parser.add_argument("--quantify", action="store_true",
                        help="Build master_table.csv with all per-animal scalars")
    parser.add_argument("--motifs", "--motif", action="store_true", dest="motifs",
                        help="Discover context-enriched bigram/trigram state motifs")
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
                        help="With --report/--motifs/--quantify: exclude frames with prob < threshold")
    parser.add_argument("--cohort", metavar="FILE", default=None,
                        help="Cohort CSV/Excel for --quantify (auto-detected if omitted)")
    parser.add_argument("--save-run", action="store_true",
                        help="Mark the cluster run as saved after --cluster completes")
    parser.add_argument("--set-active", metavar="RUN_ID", default=None,
                        help="Set a saved run as the active clustering run")
    parser.add_argument("--list-runs", action="store_true",
                        help="List all saved clustering runs")
    parser.add_argument("--event-align", action="store_true",
                        help="Peri-event behavioral state analysis (discrete experiments only)")
    parser.add_argument("--diagnostics", action="store_true",
                        help="Generate clustering quality diagnostics to results/diagnostics/")
    parser.add_argument("--motif-clips", action="store_true",
                        help="Generate exemplar video clips for top motifs")
    parser.add_argument("--top-motifs", type=int, default=10,
                        help="Number of top motifs to generate clips for (default: 10)")
    parser.add_argument("--clips-per-motif", type=int, default=5,
                        help="Max clips per motif (default: 5)")
    parser.add_argument("--clip-padding-sec", type=float, default=1.0,
                        help="Padding in seconds around motif clip boundaries (default: 1.0)")
    parser.add_argument("--repair-paths", action="store_true",
                        help="Automatically rewrite config.json when a resolved metadata/results/"
                             "raw_videos path looks like a doubled pre-refactor path. Without this "
                             "flag, only a warning + suggested fix is printed.")
    args = parser.parse_args()

    if not any([args.extract, args.fix_features, args.backfill_video_paths, args.cluster,
                args.collapse, args.diagnose,
                args.report, args.summarize, args.quantify, args.motifs, args.save_run,
                args.event_align, args.diagnostics, args.motif_clips,
                args.set_active, args.list_runs, args.repair_paths]):
        parser.print_help()
        sys.exit(1)

    _print_hardware_banner()
    try:
        _print_project_path_diagnostics(repair=args.repair_paths)
    except Exception as exc:
        sys.exit(str(exc))

    if args.list_runs:
        from cluster_run_manager import ClusterRunManager
        _cfg_path = _project_config_path()
        _mgr = ClusterRunManager(_res(), config_path=_cfg_path)
        _runs = _mgr.list_runs()
        _active = _mgr.get_active_run()
        if not _runs:
            print("No saved clustering runs.")
        else:
            print(f"{'Run ID':<55} {'States':>6} {'Noise':>7} {'MCS':>6} {'MS':>4} {'UMAP':>5} {'Status':<10} {'Active'}")
            print("-" * 105)
            for _m in _runs:
                _act = " *" if _m.run_id == _active else ""
                print(
                    f"{_m.run_id:<55} {_m.n_clusters:>6} "
                    f"{_m.noise_frac * 100:>6.1f}% {_m.min_cluster_size:>6} "
                    f"{_m.min_samples_resolved:>4} {_m.umap_dims:>5} "
                    f"{_m.status:<10}{_act}"
                )

    if args.set_active:
        from cluster_run_manager import ClusterRunManager
        _cfg_path = _project_config_path()
        _mgr = ClusterRunManager(_res(), config_path=_cfg_path)
        try:
            _mgr.set_active_run(args.set_active)
            print(f"Active run set to: {args.set_active}")
        except FileNotFoundError:
            sys.exit(f"Run not found: {args.set_active}")

    if args.extract:
        cmd_extract(fps=args.fps, use_wavelets=not args.no_wavelets)
    if args.fix_features:
        cmd_fix_features(fps=args.fps)
    if args.backfill_video_paths:
        cmd_backfill_video_paths()
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
    if args.motifs:
        cmd_motifs(min_confidence=args.min_confidence)
    if args.quantify:
        cmd_quantify(cohort=args.cohort, min_confidence=args.min_confidence)
    if args.event_align:
        cmd_event_align(min_confidence=args.min_confidence)
    if args.diagnostics:
        _generate_diagnostics()
    if args.motif_clips:
        from generate_clips import cmd_motif_clips
        cmd_motif_clips(
            fps=args.fps,
            top_motifs=args.top_motifs,
            clips_per_motif=args.clips_per_motif,
            clip_padding_sec=args.clip_padding_sec,
        )


if __name__ == "__main__":
    main()
