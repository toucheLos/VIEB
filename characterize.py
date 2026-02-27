"""
characterize.py — Behavioral state interpretation for VIEB
===========================================================
Turns unlabeled cluster IDs into interpretable behaviors and surfaces
context-specific patterns.

Outputs (results/characterization/):
  state_summary.csv       — kinematic profiles + heuristic labels per state
  context_report.csv      — A/B/C enrichment, effect sizes, bootstrap CIs
  hidden_behaviors.csv    — rare states enriched in a context + anomaly bouts
  bouts.csv               — all bouts with metadata (smoothed labels, 0.5s window)
  labels_per_frame.csv    — per-frame state + context for every video
  context_fractions.png   — bar plot of state occupancy by context

  clips/state_<id>/       — exemplar clips (--clips flag):
    longest_NN.mp4        — longest bouts
    typical_NN.mp4        — bouts closest to cluster centroid
    context_<X>_NN.mp4   — bouts from the most-enriched context

Usage:
    python characterize.py              # all outputs except clips (~1 min)
    python characterize.py --clips      # also export video clips (~slow)
    python characterize.py --n-clips 10 # change clips per category (default 15)
"""

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import median_filter

# ---------------------------------------------------------------------------
# Feature index constants (from ml/feature_extraction.py _flatten_features)
# [0:8]  speeds  [8:36]  pairwise dists  [36] centroid_spd
# [37] orientation  [38] elongation  [39] angular_vel  [40] entropy
# [41:49] temporal window features
# ---------------------------------------------------------------------------
IDX_SPEEDS       = slice(0, 8)
IDX_DISTS        = slice(8, 36)
IDX_CENTROID_SPD = 36
IDX_ELONGATION   = 38
IDX_ANGULAR_VEL  = 39
IDX_ENTROPY      = 40
# nose(2)→tail_base(6): pair rank = 7+6+3 = 16, so feature col = 8+16 = 24
IDX_BODY_LENGTH  = 24

SMOOTH_FRAMES = 15   # 0.5 s at 30 fps
MIN_BOUT_FRAMES = 6  # 0.2 s


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_prereqs():
    for path in ["results/features/index.json",
                 "results/shared/cluster_info.json",
                 "results/comparison/summary_table.csv"]:
        if not os.path.exists(path):
            sys.exit(f"Missing {path}. Run compare.py --extract / --cluster / --report first.")

    with open("results/features/index.json") as f:
        index = json.load(f)
    with open("results/shared/cluster_info.json") as f:
        cluster_info = json.load(f)
    df_summary = pd.read_csv("results/comparison/summary_table.csv")

    meta = pd.DataFrame()
    if os.path.exists("metadata.csv"):
        meta = pd.read_csv("metadata.csv")
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
        lp = f"results/shared/{stem}_labels.npy"
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
                "video_path": index[stem]["video_path"],
                "features_path": index[stem]["features_path"],
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# A: State kinematic summary + heuristic labels
# ---------------------------------------------------------------------------

def _heuristic_label(row, thr):
    spd = row["mean_centroid_speed"]
    ang = row["mean_angular_vel"]
    ent = row["mean_entropy"]
    dur = row["mean_bout_dur_sec"]

    if spd < thr["spd_lo"] and ang < thr["ang_lo"] and dur > thr["dur_hi"]:
        return "freezing-like"
    if spd > thr["spd_hi"]:
        return "locomotion"
    if thr["spd_lo"] <= spd <= thr["spd_hi"] and ang > thr["ang_hi"] and ent > thr["ent_hi"]:
        return "exploration-like"
    if spd < thr["spd_hi"] and ent > thr["ent_hi"]:
        return "grooming/comfort-like"
    return "mixed"


def _build_state_summary(index, cluster_info, df_summary, meta, bouts_df, fps):
    n_clusters = cluster_info["n_clusters"]
    rows = []

    for k in range(n_clusters):
        spd_vals, ang_vals, ent_vals, blen_vals, elong_vals = [], [], [], [], []

        for stem in sorted(index.keys()):
            lp = f"results/shared/{stem}_labels.npy"
            fp = index[stem]["features_path"]
            if not os.path.exists(lp) or not os.path.exists(fp):
                continue
            labels = _smooth_labels(np.load(lp))
            feats  = np.load(fp)
            mask = labels == k
            if mask.sum() == 0:
                continue
            f = feats[mask]
            spd_vals.append(f[:, IDX_CENTROID_SPD])
            ang_vals.append(np.abs(f[:, IDX_ANGULAR_VEL]))
            ent_vals.append(f[:, IDX_ENTROPY])
            blen_vals.append(f[:, IDX_BODY_LENGTH])
            elong_vals.append(f[:, IDX_ELONGATION])

        if not spd_vals:
            continue

        spd   = np.concatenate(spd_vals)
        ang   = np.concatenate(ang_vals)
        ent   = np.concatenate(ent_vals)
        blen  = np.concatenate(blen_vals)
        elong = np.concatenate(elong_vals)

        col   = f"state_{k}_frac"
        ctx_fracs = {}
        for ctx in ["A", "B", "C"]:
            sub = df_summary[df_summary["context"] == ctx][col] if "context" in df_summary else pd.Series(dtype=float)
            ctx_fracs[f"context_{ctx}_frac"] = float(sub.mean()) if len(sub) else float("nan")

        kb = bouts_df[bouts_df["state"] == k]
        rows.append({
            "state": k,
            "heuristic_label": "",
            **ctx_fracs,
            "mean_centroid_speed":    float(spd.mean()),
            "median_centroid_speed":  float(np.median(spd)),
            "mean_angular_vel":       float(ang.mean()),
            "median_angular_vel":     float(np.median(ang)),
            "mean_body_length_px":    float(blen.mean()),
            "median_body_length_px":  float(np.median(blen)),
            "mean_elongation":        float(elong.mean()),
            "median_elongation":      float(np.median(elong)),
            "mean_entropy":           float(ent.mean()),
            "median_entropy":         float(np.median(ent)),
            "mean_bout_dur_sec":      float(kb["duration_sec"].mean()) if len(kb) else 0.0,
            "median_bout_dur_sec":    float(kb["duration_sec"].median()) if len(kb) else 0.0,
            "n_bouts":                int(len(kb)),
        })

    df = pd.DataFrame(rows)

    # Data-driven thresholds
    thr = {
        "spd_lo":  float(np.percentile(df["mean_centroid_speed"], 30)),
        "spd_hi":  float(np.percentile(df["mean_centroid_speed"], 70)),
        "ang_lo":  float(np.percentile(df["mean_angular_vel"],    30)),
        "ang_hi":  float(np.percentile(df["mean_angular_vel"],    60)),
        "ent_hi":  float(np.percentile(df["mean_entropy"],        60)),
        "dur_hi":  float(np.percentile(df["mean_bout_dur_sec"],   60)),
    }

    df["heuristic_label"] = df.apply(
        lambda r: f"state_{r['state']}: {_heuristic_label(r, thr)} (candidate)", axis=1
    )
    return df


# ---------------------------------------------------------------------------
# C: Context contrast report
# ---------------------------------------------------------------------------

def _bootstrap_diff(a, b, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    diffs = [rng.choice(a, len(a), replace=True).mean() -
             rng.choice(b, len(b), replace=True).mean()
             for _ in range(n)]
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _cohen_d(a, b):
    pool = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return float((a.mean() - b.mean()) / pool) if pool > 1e-10 else 0.0


def _build_context_report(cluster_info, df_summary):
    n_clusters = cluster_info["n_clusters"]
    ctx_data = {c: df_summary[df_summary["context"] == c]
                for c in ["A", "B", "C"]}

    rows = []
    for k in range(n_clusters):
        col  = f"state_{k}_frac"
        row  = {"state": k}
        vals = {}

        for ctx, sub in ctx_data.items():
            if len(sub) > 0:
                row[f"{ctx}_frac"] = float(sub[col].mean())
                vals[ctx] = sub[col].values

        for c1, c2 in [("A", "B"), ("A", "C"), ("B", "C")]:
            if c1 in vals and c2 in vals:
                diff = float(vals[c1].mean() - vals[c2].mean())
                ci_lo, ci_hi = _bootstrap_diff(vals[c1], vals[c2])
                row[f"{c1}_minus_{c2}"]        = round(diff, 4)
                row[f"{c1}_minus_{c2}_ci_lo"]  = round(ci_lo, 4)
                row[f"{c1}_minus_{c2}_ci_hi"]  = round(ci_hi, 4)
                row[f"cohen_d_{c1}_{c2}"]      = round(_cohen_d(vals[c1], vals[c2]), 3)

        # Enrichment scores (each context vs. mean of the other two)
        for ctx in ["A", "B", "C"]:
            others = [vals[c].mean() for c in ["A", "B", "C"] if c != ctx and c in vals]
            if ctx in vals and others:
                row[f"{ctx}_enrichment"] = round(vals[ctx].mean() - np.mean(others), 4)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Add per-state enrichment rankings
    for ctx in ["A", "B", "C"]:
        col = f"{ctx}_enrichment"
        if col in df:
            df[f"{ctx}_rank"] = df[col].rank(ascending=False).astype(int)

    return df


def _plot_context_fractions(df_ctx, n_clusters, out_dir):
    import matplotlib.pyplot as plt

    present = [c for c in ["A", "B", "C"] if f"{c}_frac" in df_ctx.columns]
    if not present:
        return

    colors = {"A": "#E74C3C", "B": "#3498DB", "C": "#2ECC71"}
    x = np.arange(n_clusters)
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(8, 2 * n_clusters), 5))
    for i, ctx in enumerate(present):
        ax.bar(x + i * w, df_ctx[f"{ctx}_frac"], w,
               label=f"Context {ctx}  (shock={'yes' if ctx=='B' else 'no'})",
               color=colors.get(ctx, f"C{i}"), alpha=0.85)

    ax.set_xticks(x + w * (len(present) - 1) / 2)
    ax.set_xticklabels([
        f"State {int(r['state'])}\n{r.get('heuristic_label', '').split(': ')[-1].replace(' (candidate)', '')}"
        if "heuristic_label" in df_ctx.columns else f"State {int(r['state'])}"
        for _, r in df_ctx.iterrows()
    ], fontsize=8)
    ax.set_ylabel("Mean fraction of session")
    ax.set_title("Behavioral State Occupancy by Context")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "context_fractions.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# D: Hidden behavior finder
# ---------------------------------------------------------------------------

def _find_hidden_behaviors(index, cluster_info, df_summary, bouts_df, out_dir):
    n_clusters = cluster_info["n_clusters"]
    centers = np.array(cluster_info["cluster_centers"])
    rows = []

    # --- Definition 1: enriched states ---
    for k in range(n_clusters):
        col = f"state_{k}_frac"
        overall = float(df_summary[col].mean())
        for ctx in ["A", "B", "C"]:
            sub = df_summary[df_summary["context"] == ctx][col]
            if len(sub) == 0:
                continue
            ctx_frac = float(sub.mean())
            ratio = ctx_frac / overall if overall > 0.001 else 0.0
            if overall < 0.15 and ratio > 2.0:
                n_bouts = int(len(bouts_df[(bouts_df["state"] == k) & (bouts_df["context"] == ctx)]))
                rows.append({
                    "type": "context_enriched_state",
                    "state": k,
                    "context": ctx,
                    "overall_frac": round(overall, 4),
                    "context_frac": round(ctx_frac, 4),
                    "enrichment_ratio": round(ratio, 2),
                    "n_exemplar_bouts": n_bouts,
                    "note": f"state_{k} is {ratio:.1f}x more common in context {ctx}",
                })

    # --- Definition 2: anomaly bouts (top 1% by distance to cluster center) ---
    preprocessor_path = "results/shared/preprocessor.pkl"
    if os.path.exists(preprocessor_path):
        from ml import BehaviorPreprocessor
        preprocessor = BehaviorPreprocessor.load(preprocessor_path)

        all_dist_by_stem = {}
        for stem in sorted(index.keys()):
            lp = f"results/shared/{stem}_labels.npy"
            fp = index[stem]["features_path"]
            if not os.path.exists(lp) or not os.path.exists(fp):
                continue
            labels = np.load(lp)
            feats  = np.load(fp).astype(np.float64)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pca = preprocessor.transform(feats)
            if pca.shape[0] != len(labels):
                continue
            assigned = centers[labels]  # (T, n_pca)
            dist = np.linalg.norm(pca - assigned, axis=1)
            all_dist_by_stem[stem] = dist

        if all_dist_by_stem:
            all_dists = np.concatenate(list(all_dist_by_stem.values()))
            threshold = float(np.percentile(all_dists, 99))

            anomaly_rows = []
            for stem, dist in all_dist_by_stem.items():
                is_anom = dist > threshold
                changes = np.where(np.diff(is_anom.astype(int)) != 0)[0] + 1
                starts = np.concatenate([[0], changes])
                ends   = np.concatenate([changes, [len(is_anom)]])
                ctx = bouts_df[bouts_df["stem"] == stem]["context"].values
                ctx_val = ctx[0] if len(ctx) > 0 else ""
                for s, e in zip(starts, ends):
                    if is_anom[s] and (e - s) >= MIN_BOUT_FRAMES:
                        anomaly_rows.append({
                            "stem": stem,
                            "start_frame": int(s),
                            "end_frame": int(e - 1),
                            "duration_sec": round((e - s) / 30.0, 2),
                            "context": ctx_val,
                            "mean_distance": round(float(dist[s:e].mean()), 3),
                            "video_path": index[stem]["video_path"],
                        })

            if anomaly_rows:
                df_anom = pd.DataFrame(anomaly_rows)
                anom_path = os.path.join(out_dir, "anomaly_bouts.csv")
                df_anom.drop(columns=["video_path"], errors="ignore").to_csv(anom_path, index=False)
                print(f"  Anomaly bouts (top 1%): {len(df_anom)} bouts → {anom_path}")
                ctx_counts = df_anom["context"].value_counts().to_dict()
                rows.append({
                    "type": "high_reconstruction_error",
                    "state": -1,
                    "context": str(ctx_counts),
                    "overall_frac": round(0.01, 4),
                    "context_frac": float("nan"),
                    "enrichment_ratio": float("nan"),
                    "n_exemplar_bouts": len(df_anom),
                    "note": f"Top 1% frames by distance to cluster center. "
                            f"Context breakdown: {ctx_counts}",
                })

    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "hidden_behaviors.csv")
    df.to_csv(path, index=False)
    print(f"Hidden behaviors → {path}  ({len(df)} candidates)")
    for _, r in df.iterrows():
        if r["type"] == "context_enriched_state":
            print(f"  {r['note']}")


# ---------------------------------------------------------------------------
# B: Clip extraction
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


def cmd_clips(fps=30.0, n_clips=15):
    import cv2  # fail fast if not installed

    index, cluster_info, df_summary, meta = _load_prereqs()
    n_clusters = cluster_info["n_clusters"]
    centers    = np.array(cluster_info["cluster_centers"])

    # Load or build bouts
    bouts_csv = "results/characterization/bouts.csv"
    if os.path.exists(bouts_csv):
        bouts_df = pd.read_csv(bouts_csv)
        vp_map = {s: info["video_path"] for s, info in index.items()}
        fp_map = {s: info["features_path"] for s, info in index.items()}
        bouts_df["video_path"]    = bouts_df["stem"].map(vp_map)
        bouts_df["features_path"] = bouts_df["stem"].map(fp_map)
    else:
        bouts_df = _build_bouts_df(index, fps, meta)

    # Load preprocessor for "typical" ranking
    preprocessor = None
    pp_path = "results/shared/preprocessor.pkl"
    if os.path.exists(pp_path):
        from ml import BehaviorPreprocessor
        preprocessor = BehaviorPreprocessor.load(pp_path)

    # Which context is most enriched per state?
    ctx_report_path = "results/characterization/context_report.csv"
    state_best_ctx = {}
    if os.path.exists(ctx_report_path):
        cr = pd.read_csv(ctx_report_path)
        for _, r in cr.iterrows():
            k = int(r["state"])
            best, best_val = None, -np.inf
            for ctx in ["A", "B", "C"]:
                col = f"{ctx}_enrichment"
                if col in r and not np.isnan(r[col]) and r[col] > best_val:
                    best, best_val = ctx, r[col]
            if best:
                state_best_ctx[k] = best

    for k in range(n_clusters):
        kb = bouts_df[bouts_df["state"] == k].copy()
        if kb.empty:
            continue

        out_dir = os.path.join("clips", f"state_{k}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"\nState {k}: {len(kb)} bouts → {out_dir}")

        # ── Longest bouts ──────────────────────────────────────────────────
        for i, (_, b) in enumerate(kb.nlargest(n_clips, "duration_sec").iterrows()):
            _export_clip(b["video_path"], int(b["start_frame"]), int(b["end_frame"]),
                         os.path.join(out_dir, f"longest_{i+1:02d}.mp4"), fps=fps)
        print(f"  longest: {min(n_clips, len(kb))} clips")

        # ── Typical bouts (nearest to cluster centroid in PCA space) ───────
        if preprocessor is not None:
            ck = centers[k]
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

            for i, (_, b) in enumerate(kb.nsmallest(n_clips, "_dist").iterrows()):
                _export_clip(b["video_path"], int(b["start_frame"]), int(b["end_frame"]),
                             os.path.join(out_dir, f"typical_{i+1:02d}.mp4"), fps=fps)
            print(f"  typical: {min(n_clips, len(kb))} clips")

        # ── Context-specific bouts ─────────────────────────────────────────
        best_ctx = state_best_ctx.get(k)
        if best_ctx:
            ctx_bouts = kb[kb["context"] == best_ctx]
            for i, (_, b) in enumerate(ctx_bouts.nlargest(n_clips, "duration_sec").iterrows()):
                _export_clip(b["video_path"], int(b["start_frame"]), int(b["end_frame"]),
                             os.path.join(out_dir, f"context_{best_ctx}_{i+1:02d}.mp4"), fps=fps)
            print(f"  context-{best_ctx}: {min(n_clips, len(ctx_bouts))} clips")

    print(f"\nClips saved under clips/state_<id>/")


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def cmd_summarize(fps=30.0):
    index, cluster_info, df_summary, meta = _load_prereqs()
    n_clusters = cluster_info["n_clusters"]
    out_dir = "results/characterization"
    os.makedirs(out_dir, exist_ok=True)

    print("Building bouts (smoothed labels, 0.5 s window)...")
    bouts_df = _build_bouts_df(index, fps, meta)

    # ── A: State summary ──────────────────────────────────────────────────
    print(f"Computing kinematic profiles for {n_clusters} states...")
    df_states = _build_state_summary(index, cluster_info, df_summary, meta, bouts_df, fps)
    df_states.to_csv(os.path.join(out_dir, "state_summary.csv"), index=False)
    print(f"\nState summary → {out_dir}/state_summary.csv")
    for _, r in df_states.iterrows():
        ctx_line = "  ".join(
            f"{c}={r.get(f'context_{c}_frac', float('nan')):.1%}"
            for c in ["A", "B", "C"]
        )
        print(f"  {r['heuristic_label']}")
        print(f"    speed={r['mean_centroid_speed']:.2f}  "
              f"ang_vel={r['mean_angular_vel']:.3f}  "
              f"bout={r['mean_bout_dur_sec']:.2f}s  [{ctx_line}]")

    # ── C: Context contrast ───────────────────────────────────────────────
    print("\nComputing context enrichment + bootstrap CIs...")
    df_ctx = _build_context_report(cluster_info, df_summary)
    # Merge heuristic labels for x-tick labels in plot
    if "heuristic_label" in df_states.columns:
        df_ctx = df_ctx.merge(df_states[["state", "heuristic_label"]], on="state", how="left")
    df_ctx.to_csv(os.path.join(out_dir, "context_report.csv"), index=False)
    print(f"Context report → {out_dir}/context_report.csv")

    print("\n  Context enrichment summary:")
    for _, r in df_ctx.iterrows():
        enrichments = {c: r.get(f"{c}_enrichment", float("nan")) for c in ["A", "B", "C"]}
        best = max(enrichments, key=lambda c: enrichments[c] if not np.isnan(enrichments[c]) else -np.inf)
        print(f"  State {int(r['state'])}: most enriched in Context {best} "
              f"(Δ={enrichments[best]:+.3f})")

    _plot_context_fractions(df_ctx, n_clusters, out_dir)
    print(f"Context plot → {out_dir}/context_fractions.png")

    # ── D: Hidden behaviors ───────────────────────────────────────────────
    print("\nSearching for hidden behaviors...")
    _find_hidden_behaviors(index, cluster_info, df_summary, bouts_df, out_dir)

    # ── E: bouts.csv ─────────────────────────────────────────────────────
    bouts_out = bouts_df.drop(columns=["video_path", "features_path"], errors="ignore")
    bouts_out.to_csv(os.path.join(out_dir, "bouts.csv"), index=False)
    print(f"\nBouts → {out_dir}/bouts.csv  ({len(bouts_df):,} bouts)")

    # ── E: labels_per_frame.csv ──────────────────────────────────────────
    print("Writing labels_per_frame.csv (large file)...")
    ctx_map = {}
    if not meta.empty:
        ctx_map = meta.set_index("stem")["context"].to_dict()

    parts = []
    for stem in sorted(index.keys()):
        lp = f"results/shared/{stem}_labels.npy"
        if not os.path.exists(lp):
            continue
        labels = _smooth_labels(np.load(lp))
        n = len(labels)
        parts.append(pd.DataFrame({
            "stem": stem,
            "frame": np.arange(n),
            "state": labels,
            "context": ctx_map.get(stem, ""),
        }))
    pd.concat(parts, ignore_index=True).to_csv(
        os.path.join(out_dir, "labels_per_frame.csv"), index=False
    )
    print(f"Labels per frame → {out_dir}/labels_per_frame.csv")
    print(f"\nAll outputs in {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Behavioral state characterization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--clips",    action="store_true",
                        help="Export exemplar video clips per state (slow)")
    parser.add_argument("--n-clips",  type=int, default=15,
                        help="Clips per category per state (default 15)")
    parser.add_argument("--fps",      type=float, default=30.0)
    args = parser.parse_args()

    cmd_summarize(fps=args.fps)
    if args.clips:
        print("\n--- Exporting clips ---")
        cmd_clips(fps=args.fps, n_clips=args.n_clips)


if __name__ == "__main__":
    main()
