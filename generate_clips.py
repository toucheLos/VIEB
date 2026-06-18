"""
generate_clips.py — Export exemplar video clips for each behavioral state
=========================================================================

Generates video clips from the clustering results. For each state 0..N-1:
  - Longest bouts of consecutive frames labeled as that state
  - Bouts closest to the cluster centroid (most typical)
  - Bouts from the most context-enriched context

Output directory: clips/state_{id}/
  longest_01.mp4, longest_02.mp4, ...
  typical_01.mp4, typical_02.mp4, ...
  context_{X}_01.mp4, ...

Usage:
    python generate_clips.py
    python generate_clips.py --n-clips 15
    python generate_clips.py --clip-purity 0.95
    python generate_clips.py --output clips/
    python generate_clips.py --fps 30 --max-clip-frames 300
"""

import argparse
import io
import json
import os
import sys
import warnings

# Ensure UTF-8 output on Windows so → and other unicode chars don't crash
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

import platform

import vieb_config as _vc
def _res(): return _vc.get_results_dir()
def _meta(): return _vc.get_metadata_path()


# ---------------------------------------------------------------------------
# Feature index lookup — resolved dynamically from index.json metadata
# ---------------------------------------------------------------------------

def _load_feature_index() -> dict:
    """Return {feature_name: column_index} from index.json metadata."""
    from ml.feature_extraction import resolve_feature_indices
    idx_path = os.path.join(_res(), "features", "index.json")
    if os.path.exists(idx_path):
        try:
            with open(idx_path) as f:
                meta = json.load(f).get("_meta", {})
            names = meta.get("feature_names", [])
            if names:
                return resolve_feature_indices(names)
        except Exception:
            pass
    return {}

SMOOTH_FRAMES = 15   # 0.5 s at 30 fps
MIN_BOUT_FRAMES = 6  # 0.2 s


# ---------------------------------------------------------------------------
# Shared helpers — verbatim copies from characterize.py
# ---------------------------------------------------------------------------

def _resolve_video_path(path: str) -> str:
    """Resolve a video path that may be relative (from index.json built on another OS)."""
    if os.path.exists(path):
        return path
    proj_root = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(proj_root, path)
    if os.path.exists(abs_path):
        return abs_path
    raw_dir = _vc.get_raw_videos_dir()
    if raw_dir:
        candidate = os.path.join(raw_dir, os.path.basename(path))
        if os.path.exists(candidate):
            return candidate
    return path


def _load_prereqs():
    for path in [os.path.join(_res(), "features", "index.json"),
                 os.path.join(_res(), "shared", "cluster_info.json"),
                 os.path.join(_res(), "comparison", "summary_table.csv")]:
        if not os.path.exists(path):
            sys.exit(f"Missing {path}. Run compare.py --extract / --cluster / --report first.")

    with open(os.path.join(_res(), "features", "index.json")) as f:
        index = {k: v for k, v in json.load(f).items() if "features_path" in v}
    with open(os.path.join(_res(), "shared", "cluster_info.json")) as f:
        cluster_info = json.load(f)
    df_summary = pd.read_csv(os.path.join(_res(), "comparison", "summary_table.csv"))

    meta = pd.DataFrame()
    if os.path.exists(_meta()):
        meta = pd.read_csv(_meta())
        meta = _vc.normalize_metadata_columns(meta)
        meta["stem"] = meta["filename"].str.replace(r"\.mp4$", "", regex=True)

    return index, cluster_info, df_summary, meta


def _smooth_labels(labels):
    return median_filter(labels.astype(float), size=SMOOTH_FRAMES).round().astype(np.int32)


def _rle_bouts(labels):
    """Run-length encode, return (state, start, end_inclusive) for runs >= MIN_BOUT_FRAMES."""
    if len(labels) == 0:
        return []
    changes = np.where(np.diff(labels) != 0)[0] + 1
    starts = np.concatenate([[0], changes])
    ends = np.concatenate([changes, [len(labels)]])
    return [
        (int(labels[s]), int(s), int(e - 1))
        for s, e in zip(starts, ends)
        if e - s >= MIN_BOUT_FRAMES
    ]


def _build_bouts_df(index, fps, meta):
    """Build smoothed bouts DataFrame across all videos."""
    ctx_map = {}
    animal_map = {}
    day_map = {}
    exp_map = {}
    if not meta.empty:
        for _, row in meta.iterrows():
            s = row["stem"]
            ctx_map[s]    = str(row.get("context", ""))
            animal_map[s] = str(row.get("animal_id", ""))
            day_map[s]    = str(row.get("day", ""))
            exp_map[s]    = str(row.get("experiment", ""))

    rows = []
    for stem in sorted(index.keys()):
        lp = os.path.join(_res(), "shared", f"{stem}_labels.npy")
        if not os.path.exists(lp):
            continue
        labels = _smooth_labels(np.load(lp))
        for state, s, e in _rle_bouts(labels):
            rows.append({
                "stem": stem,
                "state": state,
                "start_frame": s,
                "end_frame": e,
                "start_sec": round(s / fps, 2),
                "end_sec": round(e / fps, 2),
                "duration_sec": round((e - s + 1) / fps, 2),
                "context": ctx_map.get(stem, ""),
                "animal_id": animal_map.get(stem, ""),
                "day": day_map.get(stem, ""),
                "experiment": exp_map.get(stem, ""),
                "video_path": _resolve_video_path(index[stem]["video_path"]),
                "features_path": index[stem]["features_path"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B: Clip extraction — verbatim copies from characterize.py
# ---------------------------------------------------------------------------

def _export_clip(video_path, start_frame, end_frame, out_path,
                 fps=30.0, pad_to_secs=5.0, max_secs=10.0):
    import cv2

    video_path = str(video_path).replace("\\", "/")
    if not os.path.exists(video_path):
        print(f"    WARN: video not found: {video_path}")
        return False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bout_len = end_frame - start_frame + 1
    target   = max(bout_len, int(pad_to_secs * orig_fps))
    target   = min(target, int(max_secs * orig_fps))
    center   = (start_frame + end_frame) // 2
    cs       = max(0, center - target // 2)
    ce       = min(total - 1, cs + target - 1)
    cs       = max(0, ce - target + 1)
    n_out    = ce - cs + 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, cs)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), orig_fps, (w, h))

    for _ in range(n_out):
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(frame)

    cap.release()
    writer.release()
    return True


def _expand_clip(labels, anchor, target_state, clip_purity, max_frames):
    left = anchor
    right = anchor + 1
    while (right - left) < max_frames:
        # Try expanding left
        new_left = left - 1
        if new_left >= 0:
            trial = labels[new_left:right]
            purity = (trial == target_state).mean()
            if purity >= clip_purity:
                left = new_left
                continue
        # Try expanding right
        new_right = right + 1
        if new_right <= len(labels):
            trial = labels[left:new_right]
            purity = (trial == target_state).mean()
            if purity >= clip_purity:
                right = new_right
                continue
        break  # neither direction meets purity
    return left, right


def cmd_clips(fps=30.0, n_clips=None, clip_purity=0.95, max_clip_frames=300,
              output_dir=None):
    import cv2  # fail fast if not installed

    index, cluster_info, df_summary, meta = _load_prereqs()
    n_clusters = cluster_info["n_clusters"]
    centers    = np.array(cluster_info["cluster_centers"])
    clips_written = 0
    clips_attempted = 0

    base_clips_dir = output_dir if output_dir else _vc.get_clips_dir()

    # Load or build bouts
    bouts_csv = os.path.join(_res(), "characterization", "bouts.csv")
    if os.path.exists(bouts_csv):
        bouts_df = pd.read_csv(bouts_csv)
        vp_map = {s: _resolve_video_path(info["video_path"]) for s, info in index.items() if "video_path" in info}
        fp_map = {s: info["features_path"] for s, info in index.items() if "features_path" in info}
        bouts_df["video_path"]    = bouts_df["stem"].map(vp_map)
        bouts_df["features_path"] = bouts_df["stem"].map(fp_map)
    else:
        bouts_df = _build_bouts_df(index, fps, meta)

    # Load preprocessor for "typical" ranking
    preprocessor = None
    pp_path = os.path.join(_res(), "shared", "preprocessor.pkl")
    if os.path.exists(pp_path):
        from ml import BehaviorPreprocessor
        preprocessor = BehaviorPreprocessor.load(pp_path)

    # Which context is most enriched per state?
    ctx_report_path = os.path.join(_res(), "characterization", "context_report.csv")
    state_best_ctx = {}
    if os.path.exists(ctx_report_path):
        cr = pd.read_csv(ctx_report_path)
        enrich_cols = [c for c in cr.columns if c.endswith("_enrichment")]
        for _, r in cr.iterrows():
            k = int(r["state"])
            best, best_val = None, -np.inf
            for col in enrich_cols:
                ctx = col.replace("_enrichment", "")
                val = r[col]
                try:
                    val = float(val)
                except (TypeError, ValueError):
                    continue
                if not np.isnan(val) and val > best_val:
                    best, best_val = ctx, val
            if best:
                state_best_ctx[k] = best

    # Cache smoothed labels per stem for purity-based clip expansion
    labels_cache = {}
    for stem in index.keys():
        lp = os.path.join(_res(), "shared", f"{stem}_labels.npy")
        if os.path.exists(lp):
            labels_cache[stem] = _smooth_labels(np.load(lp))

    skipped_states = []
    for k in range(n_clusters):
        kb = bouts_df[bouts_df["state"] == k].copy()
        if kb.empty:
            print(f"\nState {k}: SKIPPED — no bouts found in bouts.csv.")
            skipped_states.append(k)
            continue

        out_dir = os.path.join(base_clips_dir, f"state_{k}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nState {k}: {len(kb)} bouts → {out_dir}")

        # ── Longest bouts ──────────────────────────────────────────────────
        n_ok = 0
        for i, (_, b) in enumerate(kb.nlargest(n_clips or len(kb), "duration_sec").iterrows()):
            stem = b["stem"]
            anchor = (int(b["start_frame"]) + int(b["end_frame"])) // 2
            if stem in labels_cache:
                left, right = _expand_clip(labels_cache[stem], anchor, k, clip_purity, max_clip_frames)
            else:
                left, right = int(b["start_frame"]), int(b["end_frame"]) + 1
            clips_attempted += 1
            ok = _export_clip(b["video_path"], left, right - 1,
                              os.path.join(out_dir, f"longest_{i+1:02d}.mp4"), fps=fps,
                              pad_to_secs=5.0, max_secs=max_clip_frames / fps)
            clips_written += int(ok)
            n_ok += int(ok)
        print(f"  longest: {n_ok}/{len(kb)} clips written")

        # ── Typical bouts (nearest to cluster centroid in PCA space) ───────
        if preprocessor is not None:
            ck = centers[k]
            # Probe feature dimension using the first available bout
            _feat_dim = None
            for _, _b in kb.iterrows():
                _fp = _b.get("features_path", "")
                if _fp and os.path.exists(str(_fp)):
                    _feat_dim = preprocessor.transform(
                        np.load(str(_fp))[:1].astype(np.float64)
                    ).shape[1]
                    break

            if _feat_dim is not None and _feat_dim != len(ck):
                print(
                    f"  typical: skipped — feature dim ({_feat_dim}D) does not match "
                    f"cluster center dim ({len(ck)}D). "
                    "Re-run compare.py --cluster to fix."
                )
            elif _feat_dim is not None:
                dists = []
                for _, b in kb.iterrows():
                    fp = b.get("features_path", "")
                    if not fp or not os.path.exists(str(fp)):
                        dists.append(np.inf)
                        continue
                    feats = np.load(fp)[int(b["start_frame"]):int(b["end_frame"]) + 1].astype(np.float64)
                    if len(feats) == 0:
                        dists.append(np.inf)
                        continue
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        pca = preprocessor.transform(feats)
                    dists.append(float(np.linalg.norm(pca.mean(axis=0) - ck)))
                kb["_dist"] = dists

                n_ok = 0
                for i, (_, b) in enumerate(kb.nsmallest(n_clips or len(kb), "_dist").iterrows()):
                    stem = b["stem"]
                    anchor = (int(b["start_frame"]) + int(b["end_frame"])) // 2
                    if stem in labels_cache:
                        left, right = _expand_clip(labels_cache[stem], anchor, k, clip_purity, max_clip_frames)
                    else:
                        left, right = int(b["start_frame"]), int(b["end_frame"]) + 1
                    clips_attempted += 1
                    ok = _export_clip(b["video_path"], left, right - 1,
                                      os.path.join(out_dir, f"typical_{i+1:02d}.mp4"), fps=fps,
                                      pad_to_secs=5.0, max_secs=max_clip_frames / fps)
                    clips_written += int(ok)
                    n_ok += int(ok)
                print(f"  typical: {n_ok}/{len(kb)} clips written")

        # ── Context-specific bouts ─────────────────────────────────────────
        best_ctx = state_best_ctx.get(k)
        if best_ctx:
            ctx_bouts = kb[kb["context"] == best_ctx]
            n_ok = 0
            for i, (_, b) in enumerate(ctx_bouts.nlargest(n_clips or len(ctx_bouts), "duration_sec").iterrows()):
                stem = b["stem"]
                anchor = (int(b["start_frame"]) + int(b["end_frame"])) // 2
                if stem in labels_cache:
                    left, right = _expand_clip(labels_cache[stem], anchor, k, clip_purity, max_clip_frames)
                else:
                    left, right = int(b["start_frame"]), int(b["end_frame"]) + 1
                clips_attempted += 1
                ok = _export_clip(b["video_path"], left, right - 1,
                                  os.path.join(out_dir, f"context_{best_ctx}_{i+1:02d}.mp4"), fps=fps,
                                  pad_to_secs=5.0, max_secs=max_clip_frames / fps)
                clips_written += int(ok)
                n_ok += int(ok)
            print(f"  context-{best_ctx}: {n_ok}/{len(ctx_bouts)} clips written")

    if skipped_states:
        print(
            f"\nWARNING: States {skipped_states} had no bouts in bouts.csv and were skipped.\n"
            "  bouts.csv may be stale — re-run:  python characterize.py  (without --clips)\n"
            "  then re-run:                       python generate_clips.py"
        )

    failed = clips_attempted - clips_written
    if clips_attempted == 0:
        raise RuntimeError(
            "No clips were attempted — bouts.csv may be empty or no states were found."
        )
    if clips_written == 0:
        raise RuntimeError(
            f"All {clips_attempted} clip exports failed.\n"
            "Most likely cause: video files could not be opened.\n"
            f"  raw_videos dir : {_vc.get_raw_videos_dir()}\n"
            "Check that the directory is mounted and readable, then re-run."
        )
    if failed:
        print(f"\nWARNING: {failed}/{clips_attempted} clips failed to export.")
    print(f"\nDone: {clips_written}/{clips_attempted} clips saved under {base_clips_dir}/state_<id>/")


def main():
    parser = argparse.ArgumentParser(
        description="Export exemplar video clips for each behavioral state",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n-clips", type=int, default=15,
        help="Max clips per category per state (default: 15)",
    )
    parser.add_argument(
        "--clip-purity", type=float, default=0.95,
        help="Minimum fraction of frames in a clip that must belong to the "
             "target state (default: 0.95). Range: 0.0 to 1.0.",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory for clips (default: clips/ from vieb_config)",
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument(
        "--max-clip-frames", type=int, default=300,
        help="Hard cap on clip length in frames (default: 300 = 10 s at 30 fps).",
    )
    args = parser.parse_args()
    cmd_clips(
        fps=args.fps,
        n_clips=args.n_clips,
        clip_purity=args.clip_purity,
        max_clip_frames=args.max_clip_frames,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
