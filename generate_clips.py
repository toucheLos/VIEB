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
DEFAULT_STATE_EXEMPLARS = 3
MIN_EXEMPLAR_DURATION_SEC = 1.0
BOUNDARY_MARGIN_SEC = 0.25


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


def _clean_value(value, default=""):
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass
    return value


def _state_exemplar_columns() -> list[str]:
    return [
        "state_id", "rank", "clip_path", "stem", "session_id",
        "source_video", "subject_id", "animal_id", "context", "condition",
        "day", "timepoint", "start_frame", "end_frame", "duration_sec",
        "exemplar_score", "selection_reason", "mean_confidence",
        "typicality_distance", "skipped_reason",
    ]


def _state_exemplar_row(
    state_id, rank="", clip_path="", stem="", source_video="", start_frame="",
    end_frame="", duration_sec="", exemplar_score="", selection_reason="",
    mean_confidence="", typicality_distance="", skipped_reason="", **meta,
) -> dict:
    row = {
        "state_id": state_id,
        "rank": rank,
        "clip_path": clip_path,
        "stem": stem,
        "session_id": meta.get("session_id", ""),
        "source_video": source_video,
        "subject_id": meta.get("subject_id", ""),
        "animal_id": meta.get("animal_id", ""),
        "context": meta.get("context", ""),
        "condition": meta.get("condition", ""),
        "day": meta.get("day", ""),
        "timepoint": meta.get("timepoint", ""),
        "start_frame": start_frame,
        "end_frame": end_frame,
        "duration_sec": duration_sec,
        "exemplar_score": exemplar_score,
        "selection_reason": selection_reason,
        "mean_confidence": mean_confidence,
        "typicality_distance": typicality_distance,
        "skipped_reason": skipped_reason,
    }
    return {col: row.get(col, "") for col in _state_exemplar_columns()}


def _candidate_meta(row) -> dict:
    return {
        "session_id": str(_clean_value(row.get("session_id", row.get("stem", "")), "")),
        "subject_id": str(_clean_value(row.get("subject_id", ""), "")),
        "animal_id": str(_clean_value(row.get("animal_id", ""), "")),
        "context": str(_clean_value(row.get("context", ""), "")),
        "condition": str(_clean_value(row.get("condition", ""), "")),
        "day": str(_clean_value(row.get("day", ""), "")),
        "timepoint": str(_clean_value(row.get("timepoint", ""), "")),
    }


def _safe_float(value, default=np.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_prob_slice_mean(stem: str, start: int, end: int) -> tuple[float, str]:
    prob_path = os.path.join(_res(), "shared", f"{stem}_probs.npy")
    if not os.path.exists(prob_path):
        return np.nan, "confidence unavailable"
    try:
        probs = np.load(prob_path)
        if len(probs) == 0:
            return np.nan, "confidence unavailable"
        s = max(0, min(start, len(probs) - 1))
        e = max(s, min(end, len(probs) - 1))
        return float(np.nanmean(probs[s:e + 1])), "confidence scored"
    except Exception:
        return np.nan, "confidence unavailable"


def _bout_typicality_distance(row, center, preprocessor=None) -> float:
    fp = row.get("features_path", "")
    if not fp or not os.path.exists(str(fp)):
        return np.nan
    try:
        start = int(row.get("start_frame", 0))
        end = int(row.get("end_frame", start))
        feats = np.load(str(fp))[start:end + 1].astype(np.float64)
        if len(feats) == 0:
            return np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if preprocessor is not None:
                feats = preprocessor.transform(feats)
        if len(center) != feats.shape[1]:
            return np.nan
        return float(np.linalg.norm(feats.mean(axis=0) - np.asarray(center, dtype=np.float64)))
    except Exception:
        return np.nan


def _video_frame_count(video_path: str, index_info: dict | None = None) -> int | None:
    if index_info:
        for key in ("n_frames", "frames", "frame_count"):
            val = index_info.get(key)
            try:
                if val is not None and int(val) > 0:
                    return int(val)
            except (TypeError, ValueError):
                pass
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total if total > 0 else None
    except Exception:
        return None


def select_state_exemplars(
    bouts_df: pd.DataFrame,
    cluster_info: dict,
    index: dict,
    fps: float = 30.0,
    exemplars_per_state: int = DEFAULT_STATE_EXEMPLARS,
    min_duration_sec: float = MIN_EXEMPLAR_DURATION_SEC,
    boundary_margin_sec: float = BOUNDARY_MARGIN_SEC,
    preprocessor=None,
) -> tuple[list[dict], list[dict]]:
    """Return selected exemplar candidate rows and skipped diagnostic rows.

    Selection is deterministic and conservative: filter unusable bouts first,
    rank clear/typical/high-confidence bouts, then greedily diversify across
    sessions, subjects, contexts, and days.
    """
    if bouts_df is None or bouts_df.empty:
        return [], []

    n_clusters = int(cluster_info.get("n_clusters", 0) or 0)
    centers = cluster_info.get("cluster_centers", [])
    margin_frames = max(0, int(round(boundary_margin_sec * fps)))
    selected: list[dict] = []
    skipped: list[dict] = []

    state_col = next((c for c in ("state", "state_id", "cluster_id") if c in bouts_df.columns), None)
    if state_col is None:
        return [], [
            _state_exemplar_row("", skipped_reason="bouts table has no state column")
        ]

    for state_id in range(n_clusters):
        state_bouts = bouts_df[bouts_df[state_col] == state_id].copy()
        if state_bouts.empty:
            skipped.append(_state_exemplar_row(state_id, skipped_reason="no bouts for state"))
            continue

        candidates: list[dict] = []
        for _, bout in state_bouts.iterrows():
            stem = str(_clean_value(bout.get("stem", ""), ""))
            meta = _candidate_meta(bout)
            video_path = str(_clean_value(bout.get("video_path", ""), ""))
            if not video_path and stem in index:
                video_path = _resolve_video_path(index[stem].get("video_path", ""))
            start = int(_safe_float(bout.get("start_frame", 0), 0))
            end = int(_safe_float(bout.get("end_frame", start), start))
            duration = _safe_float(bout.get("duration_sec", (end - start + 1) / fps))

            base_row = _state_exemplar_row(
                state_id, stem=stem, source_video=video_path,
                start_frame=start, end_frame=end,
                duration_sec=round(duration, 4), **meta,
            )
            if not stem:
                skipped.append({**base_row, "skipped_reason": "missing stem"})
                continue
            if duration < min_duration_sec:
                skipped.append({**base_row, "skipped_reason": "short bout"})
                continue
            if not video_path or not os.path.exists(video_path):
                skipped.append({**base_row, "skipped_reason": "source video missing"})
                continue

            total_frames = _video_frame_count(video_path, index.get(stem, {}))
            if total_frames is not None and (
                start < margin_frames or end > total_frames - margin_frames - 1
            ):
                skipped.append({**base_row, "skipped_reason": "too close to video boundary"})
                continue

            mean_conf, conf_reason = _load_prob_slice_mean(stem, start, end)
            center = centers[state_id] if state_id < len(centers) else []
            distance = _bout_typicality_distance(bout, center, preprocessor)
            if np.isfinite(distance):
                typicality_score = 1.0 / (1.0 + distance)
                typical_reason = "near state centroid"
            else:
                typicality_score = 0.0
                typical_reason = "typicality unavailable"
            conf_score = float(mean_conf) if np.isfinite(mean_conf) else 0.5
            duration_score = min(1.0, duration / max(min_duration_sec, 1.0) / 5.0)
            score = (0.55 * typicality_score) + (0.30 * conf_score) + (0.15 * duration_score)
            reason = f"{typical_reason}; {conf_reason}; duration {duration:.2f}s"
            candidates.append({
                **base_row,
                "duration_sec": round(duration, 4),
                "exemplar_score": round(float(score), 6),
                "selection_reason": reason,
                "mean_confidence": round(float(mean_conf), 6) if np.isfinite(mean_conf) else "",
                "typicality_distance": round(float(distance), 6) if np.isfinite(distance) else "",
                "_sort_score": float(score),
            })

        candidates.sort(key=lambda c: (-c["_sort_score"], -_safe_float(c["duration_sec"], 0), c["stem"]))
        for rank, cand in enumerate(_select_diverse_candidates(candidates, exemplars_per_state), start=1):
            cand = {k: v for k, v in cand.items() if not k.startswith("_")}
            cand["rank"] = rank
            selected.append(cand)

    return selected, skipped


def _write_state_exemplar_manifest(rows: list[dict], skipped_rows: list[dict]) -> str:
    out_path = os.path.join(_res(), "characterization", "state_exemplars.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    all_rows = rows + skipped_rows
    df = pd.DataFrame(all_rows, columns=_state_exemplar_columns())
    df.to_csv(out_path, index=False)
    return out_path


def _project_relative_clip_path(clip_path: str) -> str:
    """Store clip paths relative to the active project's data root when possible."""
    project_data_root = os.path.abspath(os.path.dirname(_res()))
    try:
        return os.path.relpath(clip_path, project_data_root).replace("\\", "/")
    except ValueError:
        return os.path.abspath(clip_path)


def cmd_clips(fps=30.0, n_clips=None, clip_purity=0.95, max_clip_frames=300,
              output_dir=None, exemplars_per_state=DEFAULT_STATE_EXEMPLARS,
              min_exemplar_duration_sec=MIN_EXEMPLAR_DURATION_SEC):
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

    exemplar_rows, exemplar_skips = select_state_exemplars(
        bouts_df,
        cluster_info,
        index,
        fps=fps,
        exemplars_per_state=exemplars_per_state,
        min_duration_sec=min_exemplar_duration_sec,
        preprocessor=preprocessor,
    )
    exemplar_by_state: dict[int, list[dict]] = {}
    for row in exemplar_rows:
        try:
            exemplar_by_state.setdefault(int(row["state_id"]), []).append(row)
        except (TypeError, ValueError):
            continue
    written_exemplar_rows: list[dict] = []

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

        # ── Curated exemplars ───────────────────────────────────────────────
        n_ok = 0
        selected_rows = exemplar_by_state.get(k, [])
        for cand in selected_rows:
            rank = int(cand.get("rank", n_ok + 1) or n_ok + 1)
            out_path = os.path.join(out_dir, f"clip_{rank:03d}.mp4")
            clips_attempted += 1
            ok = _export_clip(
                cand["source_video"],
                int(cand["start_frame"]),
                int(cand["end_frame"]),
                out_path,
                fps=fps,
                pad_to_secs=5.0,
                max_secs=max_clip_frames / fps,
            )
            row_out = dict(cand)
            if ok:
                row_out["clip_path"] = _project_relative_clip_path(out_path)
                row_out["skipped_reason"] = ""
                clips_written += 1
                n_ok += 1
            else:
                row_out["clip_path"] = ""
                row_out["skipped_reason"] = "clip export failed"
            written_exemplar_rows.append(row_out)
        if selected_rows:
            print(f"  exemplars: {n_ok}/{len(selected_rows)} clips written")
        else:
            print("  exemplars: no valid representative bouts")

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

    manifest_path = _write_state_exemplar_manifest(
        written_exemplar_rows,
        exemplar_skips,
    )
    print(f"\nState exemplar manifest: {manifest_path}")

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


# ---------------------------------------------------------------------------
# C: Motif exemplar clip generation
# ---------------------------------------------------------------------------

def _safe_path_part(value: str, default: str = "motif") -> str:
    """Return a filesystem-safe path component with no traversal."""
    import re
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_-")
    return safe or default


def _parse_motif_items(motif_str: str) -> list[str]:
    """Parse a motif tuple/list without assuming labels have project meaning."""
    import ast
    try:
        value = ast.literal_eval(str(motif_str))
    except (ValueError, SyntaxError):
        return []
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return [str(value)]


def _motif_dir_name(motif_str: str, motif_type: str) -> str:
    """Convert '(55, 66)' + 'bigram' -> 'bigram_55_66' safely."""
    items = _parse_motif_items(motif_str)
    if items:
        motif_part = "_".join(_safe_path_part(item, "state") for item in items)
    else:
        motif_part = _safe_path_part(motif_str)
    return _safe_path_part(f"{motif_type}_{motif_part}", "motif")


def _empty_index_row(motif, motif_type, rank, selection_reason, skipped_reason=""):
    return {
        "motif": motif,
        "type": motif_type,
        "clip_path": "",
        "stem": "",
        "subject_id": "",
        "animal_id": "",
        "context": "",
        "group": "",
        "start_frame": "",
        "end_frame": "",
        "duration_sec": "",
        "source_video": "",
        "rank": rank,
        "selection_reason": selection_reason,
        "skipped_reason": skipped_reason,
    }


def _first_present(row, names, default=""):
    for name in names:
        if name in row:
            value = row.get(name)
            if pd.notna(value) and str(value) != "nan":
                return value
    return default


def _select_diverse_candidates(candidates, limit):
    """Greedily spread selections across videos, subjects, and groups."""
    selected = []
    remaining = list(candidates)
    seen_stems = set()
    seen_subjects = set()
    seen_groups = set()

    while remaining and len(selected) < limit:
        best_idx = 0
        best_score = None
        for i, cand in enumerate(remaining):
            score = (
                int(cand.get("stem", "") in seen_stems),
                int(cand.get("subject_id", "") in seen_subjects or cand.get("animal_id", "") in seen_subjects),
                int(cand.get("group", "") in seen_groups),
                -float(cand.get("duration_sec", 0) or 0),
                str(cand.get("stem", "")),
            )
            if best_score is None or score < best_score:
                best_score = score
                best_idx = i
        cand = remaining.pop(best_idx)
        selected.append(cand)
        if cand.get("stem"):
            seen_stems.add(cand["stem"])
        subject = cand.get("subject_id") or cand.get("animal_id")
        if subject:
            seen_subjects.add(subject)
        if cand.get("group"):
            seen_groups.add(cand["group"])
    return selected


def _build_motif_sequences_from_bouts(bouts_df, meta_by_stem=None):
    """Derive bout-level bigram/trigram occurrences from a bouts table.

    Mirrors the schema of compare.py's motif_sequences.csv so cmd_motif_clips can
    run without a prior `compare.py --motifs`. `position` is the bout index within
    each stem's bouts sorted by start_frame, matching how cmd_motif_clips resolves
    frame ranges. Bout sequences are inherently non-degenerate (consecutive bouts
    always differ in state), so no degenerate filtering is needed here.
    """
    cols = ["stem", "type", "motif", "position", "context",
            "animal_id", "day", "experiment"]
    if bouts_df is None or bouts_df.empty or "state" not in bouts_df.columns:
        return pd.DataFrame(columns=cols)

    rows = []
    for stem, grp in bouts_df.groupby("stem"):
        grp = grp.sort_values("start_frame")
        states = grp["state"].tolist()
        first = grp.iloc[0]

        def _meta_val(col):
            if col in grp.columns:
                v = first.get(col, "")
                return "" if pd.isna(v) else str(v)
            if meta_by_stem is not None and stem in meta_by_stem.index \
                    and col in meta_by_stem.columns:
                v = meta_by_stem.loc[stem, col]
                return "" if pd.isna(v) else str(v)
            return ""

        context = _meta_val("context")
        animal_id = _meta_val("animal_id")
        day = _meta_val("day")
        experiment = _meta_val("experiment")

        for n, typ in ((2, "bigram"), (3, "trigram")):
            for i in range(len(states) - n + 1):
                motif = tuple(int(s) for s in states[i:i + n])
                rows.append({
                    "stem": str(stem), "type": typ, "motif": str(motif),
                    "position": i, "context": context, "animal_id": animal_id,
                    "day": day, "experiment": experiment,
                })
    return pd.DataFrame(rows, columns=cols)


def cmd_motif_clips(
    fps=None, top_motifs=10, clips_per_motif=5,
    clip_padding_sec=1.0, output_dir=None,
    motif_source=None,
):
    """Generate exemplar video clips for top motif occurrences.

    Reads motif occurrence rows and bout boundaries, then exports short clips
    spanning the full behavior sequence represented by the motif.
    """
    import cv2  # noqa: F401 - fail fast if not installed

    cfg = {}
    try:
        cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception:
        pass
    if fps is None:
        fps = float(cfg.get("fps", 30))

    seqs_csv = motif_source or os.path.join(_res(), "motifs", "motif_sequences.csv")
    bouts_csv = os.path.join(_res(), "motifs", "bouts.csv")
    if not os.path.exists(bouts_csv):
        bouts_csv = os.path.join(_res(), "characterization", "bouts.csv")
    index_path = os.path.join(_res(), "features", "index.json")

    # index.json is the one hard requirement — it maps stems to videos/features.
    if not os.path.exists(index_path):
        sys.exit(
            f"Missing results/features/index.json: {index_path}. "
            "Run compare.py --extract / --cluster first, then re-run with --motif-clips."
        )
    with open(index_path) as f:
        index = {k: v for k, v in json.load(f).items() if isinstance(v, dict) and "features_path" in v}

    # An explicitly requested motif source must exist; the default may be rebuilt.
    if motif_source and not os.path.exists(motif_source):
        sys.exit(f"Missing motif occurrence source: {motif_source}.")

    # Load metadata once (used for bout context and the fallback sequence build).
    meta = pd.DataFrame()
    if os.path.exists(_meta()):
        try:
            meta = _vc.normalize_metadata_columns(pd.read_csv(_meta()))
            if "stem" not in meta.columns and "filename" in meta.columns:
                meta["stem"] = meta["filename"].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
        except Exception:
            meta = pd.DataFrame()

    # Bouts: read an existing table, otherwise build from labels + index.
    if os.path.exists(bouts_csv):
        bouts_df = pd.read_csv(bouts_csv)
    else:
        bouts_df = _build_bouts_df(index, fps, meta)
        if bouts_df is None or bouts_df.empty:
            sys.exit(
                "No bouts available to build motif clips. "
                "Run compare.py --cluster (and optionally --report) first."
            )

    # Motif occurrences: read an existing table, otherwise derive them from the
    # bout sequences so a single click works right after clustering.
    if os.path.exists(seqs_csv):
        seqs_df = pd.read_csv(seqs_csv)
    else:
        meta_by_stem_fb = (
            meta.drop_duplicates("stem").set_index("stem")
            if "stem" in meta.columns else None
        )
        seqs_df = _build_motif_sequences_from_bouts(bouts_df, meta_by_stem_fb)
        if seqs_df.empty:
            sys.exit(
                "No motif sequences could be derived from bouts. "
                "Ensure clustering produced sessions with multiple bouts."
            )
        print(f"[info] {seqs_csv} not found — derived motif sequences from bouts on the fly.")

    required = {"stem", "motif", "position"}
    missing_cols = required - set(seqs_df.columns)
    if missing_cols:
        sys.exit(f"Missing required columns in {seqs_csv}: {sorted(missing_cols)}")
    if "type" not in seqs_df.columns:
        seqs_df["type"] = "motif"

    video_map = {}
    for stem, info in index.items():
        vp = info.get("video_path")
        if vp:
            video_map[stem] = _resolve_video_path(vp)

    meta_by_stem = {}
    if os.path.exists(_meta()):
        try:
            meta = _vc.normalize_metadata_columns(pd.read_csv(_meta()))
            if "stem" not in meta.columns and "filename" in meta.columns:
                meta["stem"] = meta["filename"].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
            for _, row in meta.iterrows():
                meta_by_stem[str(row.get("stem", ""))] = row
        except Exception:
            meta_by_stem = {}

    motif_scores = (
        seqs_df.groupby(["motif", "type"], dropna=False)
        .size()
        .reset_index(name="count")
    )
    motif_scores["selection_reason"] = "common motif occurrence count"
    for summary_path in [
        os.path.join(_res(), "motifs", "motif_context_enrichment.csv"),
        os.path.join(_res(), "comparison", "motifs.csv"),
        os.path.join(_res(), "motifs", "motif_summary.csv"),
    ]:
        if not os.path.exists(summary_path):
            continue
        try:
            summary = pd.read_csv(summary_path)
        except Exception:
            continue
        if not {"motif", "type"}.issubset(summary.columns):
            continue
        keep = ["motif", "type"]
        for col in ["abs_log2_enrichment", "enrichment_ratio", "count", "frequency"]:
            if col in summary.columns and col not in keep:
                keep.append(col)
        motif_scores = motif_scores.merge(
            summary[keep], on=["motif", "type"], how="left", suffixes=("", "_summary")
        )
        if "abs_log2_enrichment" in motif_scores.columns:
            motif_scores["selection_reason"] = "enriched motif ranked by abs_log2_enrichment"
        elif "enrichment_ratio" in motif_scores.columns:
            motif_scores["selection_reason"] = "enriched motif ranked by enrichment_ratio"
        break

    sort_cols = []
    ascending = []
    for col in ["abs_log2_enrichment", "enrichment_ratio", "count_summary", "count"]:
        if col in motif_scores.columns:
            sort_cols.append(col)
            ascending.append(False)
    sort_cols.extend(["motif", "type"])
    ascending.extend([True, True])
    top = motif_scores.sort_values(sort_cols, ascending=ascending).head(top_motifs)
    if top.empty:
        print(f"No motifs found in {seqs_csv}.")
        return

    base_dir = output_dir or os.path.join(_res(), "motifs", "clips")
    pad_frames = int(clip_padding_sec * fps)
    index_rows = []
    total_written = 0
    total_attempted = 0

    # Build per-stem bout lists (sorted by start_frame)
    stem_bouts: dict[str, list[dict]] = {}
    for _, row in bouts_df.iterrows():
        stem = str(row.get("stem", ""))
        if stem not in stem_bouts:
            stem_bouts[stem] = []
        stem_bouts[stem].append(row.to_dict())
    for stem in stem_bouts:
        stem_bouts[stem].sort(key=lambda b: b.get("start_frame", 0))

    print(f"Generating motif clips for top {len(top)} motifs...")
    for rank, (_, mrow) in enumerate(top.iterrows()):
        motif_str = str(mrow["motif"])
        motif_type = str(mrow.get("type", "bigram"))
        dir_name = _motif_dir_name(motif_str, motif_type)
        out_dir = os.path.join(base_dir, dir_name)
        os.makedirs(out_dir, exist_ok=True)

        motif_items = _parse_motif_items(motif_str)
        if not motif_items:
            index_rows.append(_empty_index_row(
                motif_str, motif_type, rank + 1,
                str(mrow.get("selection_reason", "selected motif")),
                "could not parse motif sequence",
            ))
            print(f"  SKIP: cannot parse motif {motif_str!r}")
            continue

        motif_len = len(motif_items)

        # Find occurrences from motif_sequences.csv
        matches = seqs_df[
            (seqs_df["motif"].astype(str) == motif_str) &
            (seqs_df["type"].astype(str) == motif_type)
        ].copy()

        if matches.empty:
            print(f"  {dir_name}: no occurrences in motif_sequences.csv")
            index_rows.append(_empty_index_row(
                motif_str, motif_type, rank + 1,
                str(mrow.get("selection_reason", "selected motif")),
                "no occurrences in motif source",
            ))
            continue

        # Resolve frame ranges from bout positions
        clip_candidates = []
        for _, occ in matches.iterrows():
            stem = str(occ.get("stem", ""))
            try:
                pos = int(occ.get("position", -1))
            except (TypeError, ValueError):
                pos = -1
            meta_row = meta_by_stem.get(stem, {})
            subject_id = str(_first_present(occ, ["subject_id"], _first_present(meta_row, ["subject_id"], "")))
            animal_id = str(_first_present(occ, ["animal_id"], _first_present(meta_row, ["animal_id"], "")))
            group = str(_first_present(occ, ["group"], _first_present(meta_row, ["group", "cohort"], "")))
            context = str(_first_present(occ, ["context"], _first_present(meta_row, ["context"], "")))
            video_path = video_map.get(stem, "")
            skipped_reason = ""
            if stem not in stem_bouts:
                skipped_reason = "stem not found in bouts.csv"
            elif not video_path:
                skipped_reason = "source video not listed in feature index"
            elif not os.path.exists(str(video_path)):
                skipped_reason = "source video missing"
            bouts = stem_bouts.get(stem, [])
            if not skipped_reason and pos < 0:
                skipped_reason = "invalid motif bout position"
            elif not skipped_reason and pos + motif_len - 1 >= len(bouts):
                skipped_reason = "motif bout position outside available bouts"

            if skipped_reason:
                index_rows.append({
                    "motif": motif_str,
                    "type": motif_type,
                    "clip_path": "",
                    "stem": stem,
                    "subject_id": subject_id,
                    "animal_id": animal_id,
                    "context": context,
                    "group": group,
                    "start_frame": "",
                    "end_frame": "",
                    "duration_sec": "",
                    "source_video": video_path,
                    "rank": rank + 1,
                    "selection_reason": str(mrow.get("selection_reason", "selected motif")),
                    "skipped_reason": skipped_reason,
                })
                continue

            first_bout = bouts[pos]
            last_bout = bouts[pos + motif_len - 1]
            start_frame = max(0, int(first_bout["start_frame"]) - pad_frames)
            end_frame = int(last_bout["end_frame"]) + pad_frames
            duration = (end_frame - start_frame + 1) / fps

            clip_candidates.append({
                "stem": stem,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "duration_sec": round(duration, 2),
                "subject_id": subject_id or str(first_bout.get("subject_id", "")),
                "animal_id": animal_id or str(first_bout.get("animal_id", "")),
                "context": context or str(first_bout.get("context", "")),
                "group": group or str(first_bout.get("group", "")),
                "video_path": video_path,
                "selection_reason": str(mrow.get("selection_reason", "selected motif")),
            })

        # Rank by duration descending, then greedily diversify across videos/subjects/groups.
        clip_candidates.sort(key=lambda c: (-c["duration_sec"], c["stem"]))
        selected = _select_diverse_candidates(clip_candidates, clips_per_motif)

        n_ok = 0
        for ci, cand in enumerate(selected):
            clip_path = os.path.join(out_dir, f"clip_{ci+1:03d}.mp4")
            total_attempted += 1
            ok = _export_clip(
                cand["video_path"], cand["start_frame"], cand["end_frame"],
                clip_path, fps=fps,
                pad_to_secs=cand["duration_sec"], max_secs=cand["duration_sec"] + 2,
            )
            if ok:
                total_written += 1
                n_ok += 1
            index_rows.append({
                "motif": motif_str,
                "type": motif_type,
                "clip_path": os.path.relpath(clip_path, _res()) if ok else "",
                "stem": cand["stem"],
                "subject_id": cand["subject_id"],
                "animal_id": cand["animal_id"],
                "context": cand["context"],
                "group": cand["group"],
                "start_frame": cand["start_frame"],
                "end_frame": cand["end_frame"],
                "duration_sec": cand["duration_sec"],
                "source_video": cand["video_path"],
                "rank": ci + 1,
                "selection_reason": cand["selection_reason"],
                "skipped_reason": "" if ok else "clip export failed",
            })

        print(f"  {dir_name}: {n_ok}/{len(selected)} clips written")

    # Write index CSV
    idx_path = os.path.join(_res(), "motifs", "motif_exemplars.csv")
    if index_rows:
        os.makedirs(os.path.dirname(idx_path), exist_ok=True)
        pd.DataFrame(index_rows).to_csv(idx_path, index=False)
        print(f"\nMotif exemplar index: {idx_path}")
    else:
        os.makedirs(os.path.dirname(idx_path), exist_ok=True)
        pd.DataFrame(columns=list(_empty_index_row("", "", "", "").keys())).to_csv(idx_path, index=False)
        print(f"\nMotif exemplar index: {idx_path} (empty)")

    failed = total_attempted - total_written
    if total_attempted == 0:
        print("\nNo motif clips attempted. Check motif_sequences.csv and video availability.")
    elif total_written == 0:
        print(f"\nAll {total_attempted} clips failed. Check video paths.")
    else:
        if failed:
            print(f"\nWARNING: {failed}/{total_attempted} clips failed to export.")
        print(f"\nDone: {total_written}/{total_attempted} motif clips saved under {base_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Export exemplar video clips for each behavioral state",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--motif-clips", action="store_true",
        help="Generate exemplar clips for top motifs instead of per-state clips",
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
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument(
        "--max-clip-frames", type=int, default=300,
        help="Hard cap on clip length in frames (default: 300 = 10 s at 30 fps).",
    )
    parser.add_argument(
        "--state-exemplars", type=int, default=DEFAULT_STATE_EXEMPLARS,
        help="Curated exemplar clips per state (default: 3).",
    )
    parser.add_argument(
        "--min-exemplar-duration-sec", type=float, default=MIN_EXEMPLAR_DURATION_SEC,
        help="Minimum bout duration for curated state exemplars (default: 1.0).",
    )
    parser.add_argument(
        "--top-motifs", type=int, default=10,
        help="Number of top motifs to generate clips for (default: 10)",
    )
    parser.add_argument(
        "--clips-per-motif", type=int, default=5,
        help="Max clips per motif (default: 5)",
    )
    parser.add_argument(
        "--clip-padding-sec", type=float, default=1.0,
        help="Padding in seconds around motif clip boundaries (default: 1.0)",
    )
    parser.add_argument(
        "--motif-source", type=str, default=None,
        help="Motif occurrence CSV (default: results/motifs/motif_sequences.csv)",
    )
    args = parser.parse_args()

    resolved_fps = args.fps
    if resolved_fps is None:
        try:
            cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
            with open(cfg_path, encoding="utf-8") as f:
                resolved_fps = float(json.load(f).get("fps", 30))
        except Exception:
            resolved_fps = 30.0

    if args.motif_clips:
        cmd_motif_clips(
            fps=resolved_fps,
            top_motifs=args.top_motifs,
            clips_per_motif=args.clips_per_motif,
            clip_padding_sec=args.clip_padding_sec,
            output_dir=args.output,
            motif_source=args.motif_source,
        )
    else:
        cmd_clips(
            fps=resolved_fps,
            n_clips=args.n_clips,
            clip_purity=args.clip_purity,
            max_clip_frames=args.max_clip_frames,
            output_dir=args.output,
            exemplars_per_state=args.state_exemplars,
            min_exemplar_duration_sec=args.min_exemplar_duration_sec,
        )


if __name__ == "__main__":
    main()
