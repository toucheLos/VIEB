#!/usr/bin/env python3
"""
quantify.py — Per-animal scalar quantification and Jess correlation for VIEB.

Produces results/quantification/master_table.csv — one row per animal,
all behavioral scalars needed for the paper.

Usage
-----
    python quantify.py --build          # build master_table.csv
    python quantify.py --jess FILE      # correlate with Jess protein data
    python quantify.py --build --jess FILE  # both

Columns in master_table.csv
----------------------------
  Identity   : animal_id, cohort_label, sex, age_group, genotype, treatment
  Fear       : fear_index, context_discrimination
  Learning   : discrimination_ratio_mean, discrimination_ratio_peak_day, fear_auc
  Diversity  : behavioral_diversity (Shannon entropy)
  Transitions: transition_entropy_A, transition_entropy_B
  Occupancy  : s{k}_frac_A, s{k}_frac_B, s{k}_delta  (per non-dominant state)
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

_OUT    = Path("results/quantification")
_COMP   = Path("results/comparison")
_CHAR   = Path("results/characterization")
_SHARED = Path("results/shared")


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def _require(path, hint=""):
    p = Path(path)
    if not p.exists():
        msg = f"[ERROR] Required file not found: {p}"
        if hint:
            msg += f"\nHint: {hint}"
        sys.exit(msg)
    return pd.read_csv(p)


def _load_cluster_info():
    import json
    p = _SHARED / "cluster_info.json"
    if not p.exists():
        sys.exit("[ERROR] results/shared/cluster_info.json not found. Run compare.py --cluster first.")
    with open(p) as f:
        return json.load(f)


def _load_cohort(cohort_path=None):
    if cohort_path and os.path.exists(cohort_path):
        ext = os.path.splitext(cohort_path)[1].lower()
        if ext in (".xlsx", ".xls", ".xlsm"):
            from cohort_loader import load_cohort_excel
            return load_cohort_excel(cohort_path)
        return pd.read_csv(cohort_path)
    for c in ("cohort_normalized.csv",):
        if os.path.exists(c):
            return pd.read_csv(c)
    return None


def _state_cols(df):
    return sorted(
        [c for c in df.columns if c.startswith("state_") and c.endswith("_frac")],
        key=lambda c: int(c.split("_")[1]),
    )


def _dominant_state_id(summary, cluster_info):
    if "dominant_state" in cluster_info:
        return int(cluster_info["dominant_state"])
    cols = _state_cols(summary)
    if not cols:
        return None
    return int(summary[cols].mean().idxmax().split("_")[1])


def _identify_fear_states(context_report_path, n_top=3):
    if not Path(context_report_path).exists():
        return []
    cr = pd.read_csv(context_report_path)
    if "A_minus_B" not in cr.columns or "state" not in cr.columns:
        return []
    return [int(s) for s in cr.nlargest(n_top, "A_minus_B")["state"].tolist()]


def _ctx_col(summary):
    return next((c for c in ("context", "Context", "ctx") if c in summary.columns), None)


def _ctx_groups(summary):
    col = _ctx_col(summary)
    if col is None:
        return [], []
    vals = summary[col].dropna().astype(str).unique()
    return [v for v in vals if v.upper().startswith("A")], [v for v in vals if v.upper().startswith("B")]


# ---------------------------------------------------------------------------
# Scalar computation functions
# ---------------------------------------------------------------------------

def compute_discrimination_ratio(summary, fear_states, dominant_id):
    """Per-animal mean and peak-day discrimination ratio using fear-enriched states."""
    col = _ctx_col(summary)
    if col is None:
        return pd.DataFrame()
    ctx_A, ctx_B = _ctx_groups(summary)
    sc = _state_cols(summary)
    if dominant_id is not None:
        sc = [c for c in sc if int(c.split("_")[1]) != dominant_id]
    if not fear_states and sc:
        a_mean = summary[summary[col].isin(ctx_A)][sc].mean()
        b_mean = summary[summary[col].isin(ctx_B)][sc].mean()
        use_cols = [(a_mean - b_mean).idxmax()]
    else:
        use_cols = [f"state_{k}_frac" for k in fear_states if f"state_{k}_frac" in summary.columns]
        if not use_cols:
            use_cols = sc[:1] if sc else []
    if not use_cols:
        return pd.DataFrame()
    rows = []
    for animal_id, grp in summary.groupby("animal_id"):
        disc_by_day = []
        for day, day_grp in grp.groupby("day"):
            a = day_grp[day_grp[col].isin(ctx_A)]
            b = day_grp[day_grp[col].isin(ctx_B)]
            if a.empty or b.empty:
                continue
            fa = float(a[use_cols].mean().mean())
            fb = float(b[use_cols].mean().mean())
            denom = fa + fb
            if denom > 1e-9:
                disc_by_day.append((int(day), (fa - fb) / denom))
        if not disc_by_day:
            rows.append({"animal_id": animal_id,
                         "discrimination_ratio_mean": float("nan"),
                         "discrimination_ratio_peak_day": float("nan")})
        else:
            rows.append({
                "animal_id": animal_id,
                "discrimination_ratio_mean": round(float(np.mean([d for _, d in disc_by_day])), 4),
                "discrimination_ratio_peak_day": float(max(disc_by_day, key=lambda x: x[1])[0]),
            })
    return pd.DataFrame(rows)


def compute_fear_auc(summary, fear_states, dominant_id):
    """AUC of fear-state occupancy in context A across days (trapezoidal)."""
    col = _ctx_col(summary)
    if col is None:
        return pd.DataFrame()
    ctx_A, _ = _ctx_groups(summary)
    sc = _state_cols(summary)
    if dominant_id is not None:
        sc = [c for c in sc if int(c.split("_")[1]) != dominant_id]
    if not fear_states and sc:
        a_mean = summary[summary[col].isin(ctx_A)][sc].mean()
        use_cols = [a_mean.idxmax()]
    else:
        use_cols = [f"state_{k}_frac" for k in fear_states if f"state_{k}_frac" in summary.columns]
        if not use_cols:
            use_cols = sc[:1] if sc else []
    rows = []
    for animal_id, grp in summary.groupby("animal_id"):
        a_rows = grp[grp[col].isin(ctx_A)].copy().dropna(subset=["day"]).sort_values("day")
        if use_cols:
            a_rows["_fear"] = a_rows[use_cols].mean(axis=1)
        else:
            rows.append({"animal_id": animal_id, "fear_auc": float("nan")}); continue
        if len(a_rows) >= 2:
            auc = float(np.trapz(a_rows["_fear"].values, a_rows["day"].values))
        elif len(a_rows) == 1:
            auc = float(a_rows["_fear"].iloc[0])
        else:
            auc = float("nan")
        rows.append({"animal_id": animal_id, "fear_auc": round(auc, 4)})
    return pd.DataFrame(rows)


def compute_behavioral_diversity(summary, dominant_id):
    """Shannon entropy of per-animal mean state occupancy (excluding dominant state)."""
    sc = _state_cols(summary)
    if dominant_id is not None:
        sc = [c for c in sc if int(c.split("_")[1]) != dominant_id]
    if not sc:
        return pd.DataFrame()
    rows = []
    for animal_id, grp in summary.groupby("animal_id"):
        p = grp[sc].mean().values
        p = np.maximum(p, 0)
        total = p.sum()
        if total > 1e-9:
            p /= total; p = p[p > 0]
            entropy = float(-np.sum(p * np.log(p)))
        else:
            entropy = float("nan")
        rows.append({"animal_id": animal_id, "behavioral_diversity": round(entropy, 4)})
    return pd.DataFrame(rows)


def compute_transition_entropy(summary, dominant_id):
    """Shannon entropy of per-animal transition matrix per context."""
    col = _ctx_col(summary)
    if col is None:
        return pd.DataFrame()
    trans_path = _COMP / "transition_table.csv"
    if not trans_path.exists():
        return pd.DataFrame()
    trans = pd.read_csv(trans_path)
    trans_cols = [c for c in trans.columns if c.startswith("trans_")]
    if not trans_cols or "animal_id" not in trans.columns:
        return pd.DataFrame()
    ctx_A, ctx_B = _ctx_groups(summary)
    rows = []
    for animal_id, grp in trans.groupby("animal_id"):
        def _ent(sub):
            if sub.empty: return float("nan")
            p = sub[trans_cols].mean().values
            p = np.maximum(p, 0); total = p.sum()
            if total < 1e-9: return float("nan")
            p /= total; p = p[p > 0]
            return float(-np.sum(p * np.log(p)))
        rows.append({
            "animal_id": animal_id,
            "transition_entropy_A": round(_ent(grp[grp[col].isin(ctx_A)]), 4),
            "transition_entropy_B": round(_ent(grp[grp[col].isin(ctx_B)]), 4),
        })
    return pd.DataFrame(rows)


def build_occupancy_scalars(summary, dominant_id):
    """Per-animal state occupancy in context A, context B, and delta (A-B)."""
    col = _ctx_col(summary)
    if col is None:
        return pd.DataFrame()
    ctx_A, ctx_B = _ctx_groups(summary)
    sc = _state_cols(summary)
    if dominant_id is not None:
        sc = [c for c in sc if int(c.split("_")[1]) != dominant_id]
    rows = []
    for animal_id, grp in summary.groupby("animal_id"):
        a = grp[grp[col].isin(ctx_A)]
        b = grp[grp[col].isin(ctx_B)]
        entry = {"animal_id": animal_id}
        for c in sc:
            sid = c.split("_")[1]
            fa = float(a[c].mean()) if not a.empty else float("nan")
            fb = float(b[c].mean()) if not b.empty else float("nan")
            entry[f"s{sid}_frac_A"] = round(fa, 4)
            entry[f"s{sid}_frac_B"] = round(fb, 4)
            entry[f"s{sid}_delta"] = round(fa - fb, 4) if not (np.isnan(fa) or np.isnan(fb)) else float("nan")
        rows.append(entry)
    return pd.DataFrame(rows)


def build_master_table(cohort_path=None, out_dir=_OUT):
    """Assemble all per-animal scalars into master_table.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pipeline outputs...")
    summary = _require(_COMP / "summary_table.csv",
                       "Run: python compare.py --extract --cluster --report")
    cluster_info = _load_cluster_info()
    cohort = _load_cohort(cohort_path)

    dominant_id = _dominant_state_id(summary, cluster_info)
    print(f"  Dominant state (excluded): {dominant_id}")

    fear_states = _identify_fear_states(_CHAR / "context_report.csv")
    print(f"  Fear-enriched states: {fear_states or 'none — auto-detecting'}")

    animal_ids = sorted(summary["animal_id"].dropna().unique().tolist())
    master = pd.DataFrame({"animal_id": animal_ids})

    if cohort is not None:
        id_cols = ["animal_id"] + [c for c in
                   ("cohort_label", "sex", "age_group", "genotype", "treatment")
                   if c in cohort.columns]
        master = master.merge(cohort[id_cols].drop_duplicates("animal_id"),
                              on="animal_id", how="left")
        print(f"  Cohort: {len(cohort)} animals")
    else:
        print("  No cohort file — identity columns will be missing")

    # Fear index
    try:
        from fear_index import cohort_normalize, compute_fear_index
        if cohort is not None:
            normed = cohort_normalize(summary, cohort)
            fi_df = compute_fear_index(normed)
            master = master.merge(
                fi_df[["fear_index", "context_discrimination"]].reset_index(),
                on="animal_id", how="left")
            normed.reset_index().to_csv(out_dir / "cohort_normalized_profiles.csv", index=False)
            fi_df.reset_index().to_csv(out_dir / "fear_index.csv", index=False)
            print("  fear_index computed")
        else:
            master["fear_index"] = float("nan")
            master["context_discrimination"] = float("nan")
    except Exception as e:
        warnings.warn(f"fear_index failed: {e}")
        master["fear_index"] = float("nan")
        master["context_discrimination"] = float("nan")

    steps = [
        ("discrimination ratio",
         lambda: compute_discrimination_ratio(summary, fear_states, dominant_id),
         ["discrimination_ratio_mean", "discrimination_ratio_peak_day"]),
        ("fear AUC",
         lambda: compute_fear_auc(summary, fear_states, dominant_id),
         ["fear_auc"]),
        ("behavioral diversity",
         lambda: compute_behavioral_diversity(summary, dominant_id),
         ["behavioral_diversity"]),
        ("transition entropy",
         lambda: compute_transition_entropy(summary, dominant_id),
         ["transition_entropy_A", "transition_entropy_B"]),
    ]
    for name, fn, cols in steps:
        print(f"  Computing {name}...")
        df = fn()
        if not df.empty:
            master = master.merge(df, on="animal_id", how="left")
        else:
            for c in cols:
                master[c] = float("nan")

    print("  Computing per-state occupancy scalars...")
    occ_df = build_occupancy_scalars(summary, dominant_id)
    if not occ_df.empty:
        master = master.merge(occ_df, on="animal_id", how="left")

    out_path = out_dir / "master_table.csv"
    master.to_csv(out_path, index=False)
    n_animals = len(master)
    n_cols = len(master.columns)
    n_nan = master.drop(columns=["animal_id"], errors="ignore").isna().sum().sum()
    print(f"\nMaster table: {n_animals} animals x {n_cols} columns")
    print(f"  Missing values: {n_nan} ({100*n_nan/max(1,n_animals*(n_cols-1)):.1f}%)")
    print(f"  Saved to {out_path}")
    for col in ("fear_index", "discrimination_ratio_mean", "fear_auc"):
        if col in master.columns and master[col].isna().mean() > 0.5:
            print(f"  WARNING: {col} is NaN for {master[col].isna().mean():.0%} of animals")
    return master


# ---------------------------------------------------------------------------
# Behavioral contrast vector
# ---------------------------------------------------------------------------

def compute_contrast_vector(
    summary_csv: str,
    output_dir: str = "results/quantification",
    cohort_csv: str = None,
    per_condition: bool = True,
) -> "pd.DataFrame":
    """
    Compute behavioral contrast vector and scalar.

    per_condition=True (default):
        One vector per cohort group, pooling ALL sessions from every animal
        in that cohort.  p_A / p_B are cohort-level means (sessions first,
        then average per state).  Saves condition_contrast.csv.
        Requires cohort_csv; falls back to per-animal with a warning if absent.

    per_condition=False:
        One vector per animal (original behaviour).
        Saves contrast_vectors.csv.
    """
    import json as _json

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(summary_csv)
    summary["animal_id"] = summary["animal_id"].astype(str)

    # Shared setup: dominant state, non-dominant cols, context detection
    all_state_cols = _state_cols(summary)
    if not all_state_cols:
        sys.exit("[ERROR] No state_N_frac columns found in summary_table.csv")
    dominant_col = summary[all_state_cols].mean().idxmax()
    dominant_id  = int(dominant_col.split("_")[1])
    sc        = [c for c in all_state_cols if int(c.split("_")[1]) != dominant_id]
    state_ids = [int(c.split("_")[1]) for c in sc]

    col = _ctx_col(summary)
    if col is None:
        sys.exit("[ERROR] No context column found in summary_table.csv")
    ctx_vals   = summary[col].dropna().astype(str)
    ctx_A_vals = [v for v in ctx_vals.unique() if v.upper().startswith("A")]
    ctx_B_vals = [v for v in ctx_vals.unique() if v.upper().startswith("B")]

    cohort_df = _load_cohort(cohort_csv)

    # ---- Per-condition mode ----------------------------------------
    if per_condition:
        if cohort_df is None or "cohort_label" not in cohort_df.columns:
            print("[WARN] per_condition=True requires a cohort file with cohort_label "
                  "— falling back to per-animal mode")
            return compute_contrast_vector(
                summary_csv, output_dir, cohort_csv, per_condition=False
            )

        cohort_df = cohort_df.copy()
        cohort_df["animal_id"] = cohort_df["animal_id"].astype(str)
        summary_c = summary.merge(
            cohort_df[["animal_id", "cohort_label"]].drop_duplicates("animal_id"),
            on="animal_id", how="left",
        )
        summary_c["cohort_label"] = summary_c["cohort_label"].fillna("Unknown")

        rows = []
        for label in sorted(summary_c["cohort_label"].unique()):
            grp    = summary_c[summary_c["cohort_label"] == label]
            a_sess = grp[grp[col].isin(ctx_A_vals)]
            b_sess = grp[grp[col].isin(ctx_B_vals)]
            n_animals = grp["animal_id"].nunique()

            p_A = a_sess[sc].mean().values if not a_sess.empty else np.full(len(sc), np.nan)
            p_B = b_sess[sc].mean().values if not b_sess.empty else np.full(len(sc), np.nan)

            if np.all(np.isnan(p_A)) or np.all(np.isnan(p_B)):
                print(f"  [WARN] Cohort {label}: missing Context A or B — NaN")
                rows.append({
                    "cohort_label": label,
                    "n_animals": n_animals,
                    "n_A_sessions": len(a_sess),
                    "n_B_sessions": len(b_sess),
                    "contrast_magnitude": float("nan"),
                    "dominant_fear_state": float("nan"),
                    "dominant_safety_state": float("nan"),
                    "contrast_vector_json": _json.dumps([float("nan")] * len(sc)),
                    "p_A_json": _json.dumps(p_A.tolist()),
                    "p_B_json": _json.dumps(p_B.tolist()),
                })
            else:
                contrast = p_A - p_B
                magnitude = float(np.linalg.norm(contrast) / np.sqrt(2))
                dom_fear_idx = int(np.argmax(contrast))
                dom_safe_idx = int(np.argmin(contrast))
                rows.append({
                    "cohort_label": label,
                    "n_animals": n_animals,
                    "n_A_sessions": len(a_sess),
                    "n_B_sessions": len(b_sess),
                    "contrast_magnitude": round(magnitude, 4),
                    "dominant_fear_state": state_ids[dom_fear_idx],
                    "dominant_safety_state": state_ids[dom_safe_idx],
                    "contrast_vector_json": _json.dumps([round(float(x), 6) for x in contrast]),
                    "p_A_json": _json.dumps([round(float(x), 6) for x in p_A]),
                    "p_B_json": _json.dumps([round(float(x), 6) for x in p_B]),
                })

        result   = pd.DataFrame(rows)
        out_path = out_dir / "condition_contrast.csv"
        result.to_csv(out_path, index=False, encoding="utf-8")

        print("\n=== Per-Condition Behavioral Contrast Vectors ===")
        print(f"Cohorts computed: {len(result)}")
        valid = result[result["contrast_magnitude"].notna()]
        print(f"Cohorts with valid contrast: {len(valid)}")
        for _, r in valid.iterrows():
            print(f"  {r['cohort_label']}: magnitude={r['contrast_magnitude']:.3f}"
                  f"  (n_animals={int(r['n_animals'])},"
                  f" A_sessions={int(r['n_A_sessions'])},"
                  f" B_sessions={int(r['n_B_sessions'])})")
        print(f"Saved: {out_path}")
        return result

    # ---- Per-animal mode -------------------------------------------
    rows = []
    animals     = sorted(summary["animal_id"].dropna().unique().tolist())
    valid_count = 0

    for animal_id in animals:
        grp    = summary[summary["animal_id"] == animal_id]
        a_rows = grp[grp[col].isin(ctx_A_vals)]
        b_rows = grp[grp[col].isin(ctx_B_vals)]

        p_A = a_rows[sc].mean().values if not a_rows.empty else np.full(len(sc), np.nan)
        p_B = b_rows[sc].mean().values if not b_rows.empty else np.full(len(sc), np.nan)

        if np.all(np.isnan(p_A)) or np.all(np.isnan(p_B)):
            print(f"  [WARN] Animal {animal_id}: missing Context A or B — NaN")
            rows.append({
                "animal_id": animal_id,
                "contrast_magnitude": float("nan"),
                "dominant_fear_state": float("nan"),
                "dominant_safety_state": float("nan"),
                "n_A_sessions": len(a_rows),
                "n_B_sessions": len(b_rows),
                "contrast_vector_json": _json.dumps([float("nan")] * len(sc)),
                "p_A_json": _json.dumps(p_A.tolist()),
                "p_B_json": _json.dumps(p_B.tolist()),
            })
        else:
            contrast = p_A - p_B
            magnitude = float(np.linalg.norm(contrast) / np.sqrt(2))
            dom_fear_idx = int(np.argmax(contrast))
            dom_safe_idx = int(np.argmin(contrast))
            rows.append({
                "animal_id": animal_id,
                "contrast_magnitude": round(magnitude, 4),
                "dominant_fear_state": state_ids[dom_fear_idx],
                "dominant_safety_state": state_ids[dom_safe_idx],
                "n_A_sessions": len(a_rows),
                "n_B_sessions": len(b_rows),
                "contrast_vector_json": _json.dumps([round(float(x), 6) for x in contrast]),
                "p_A_json": _json.dumps([round(float(x), 6) for x in p_A]),
                "p_B_json": _json.dumps([round(float(x), 6) for x in p_B]),
            })
            valid_count += 1

    result = pd.DataFrame(rows)

    valid = result["contrast_magnitude"].dropna()
    if len(valid) > 0 and (valid == 0.0).all():
        print("[DIAG] All contrast_magnitude == 0.0 — checking first 3 animals:")
        for _, r in result.head(3).iterrows():
            print(f"  {r['animal_id']}: p_A={r['p_A_json'][:80]}  p_B={r['p_B_json'][:80]}")

    if cohort_df is not None:
        cohort_df["animal_id"] = cohort_df["animal_id"].astype(str)
        cohort_cols = [c for c in cohort_df.columns if c != "animal_id"]
        result = result.merge(
            cohort_df[["animal_id"] + cohort_cols].drop_duplicates("animal_id"),
            on="animal_id", how="left",
        )

    out_path = out_dir / "contrast_vectors.csv"
    result.to_csv(out_path, index=False, encoding="utf-8")

    print("\n=== Behavioral Contrast Vectors ===")
    print(f"Animals computed: {len(result)}/{len(animals)}")
    print(f"Animals with valid contrast: {valid_count}")
    if valid_count > 0:
        mags = result["contrast_magnitude"].dropna()
        print(f"Mean contrast magnitude: {mags.mean():.3f} ± {mags.std():.3f} (std)")

    if "cohort_label" in result.columns:
        print("By cohort:")
        for label, grp in result.groupby("cohort_label"):
            mags = grp["contrast_magnitude"].dropna()
            if len(mags) > 0:
                print(f"  {label}: mean={mags.mean():.3f} ± {mags.std():.3f} (n={len(mags)})")

    fear_states = result["dominant_fear_state"].dropna()
    if len(fear_states) > 0:
        from collections import Counter
        counts = Counter(int(s) for s in fear_states)
        print("Dominant fear states across all animals:")
        for sid, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            print(f"  State {sid}: {cnt} animals")

    print(f"Saved: {out_path}")
    return result


def compute_cohort_contrast(
    contrast_csv: str,
    cohort_csv: str,
    groupby: str = "cohort_label",
    n_bootstrap: int = 1000,
    seed: int = 42,
    output_dir: str = "results/quantification",
) -> tuple:
    """
    Compute mean contrast vector and statistics per cohort.
    Returns (cohort_df, stats_df).
    """
    import json as _json
    from scipy.stats import mannwhitneyu

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(contrast_csv)
    df["animal_id"] = df["animal_id"].astype(str)

    # Ensure groupby column present
    if groupby not in df.columns:
        cohort_df = _load_cohort(cohort_csv)
        if cohort_df is not None:
            cohort_df["animal_id"] = cohort_df["animal_id"].astype(str)
            if groupby in cohort_df.columns:
                df = df.merge(cohort_df[["animal_id", groupby]].drop_duplicates("animal_id"),
                              on="animal_id", how="left")
        if groupby not in df.columns:
            print(f"[WARN] Column '{groupby}' not found — treating all animals as one cohort 'All'")
            df[groupby] = "All"

    df[groupby] = df[groupby].fillna("Unknown")

    rng = np.random.default_rng(seed)
    cohort_rows = []
    groups = sorted(df[groupby].unique())

    for label in groups:
        sub = df[df[groupby] == label].copy()
        valid = sub[sub["contrast_magnitude"].notna()]
        if len(valid) == 0:
            cohort_rows.append({
                groupby: label, "n_animals": len(sub),
                "mean_magnitude": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                "mean_contrast_vector_json": "[]",
            })
            continue

        # Parse contrast vectors
        vectors = []
        for cv_str in valid["contrast_vector_json"]:
            try:
                v = np.array(_json.loads(cv_str), dtype=float)
                if not np.any(np.isnan(v)):
                    vectors.append(v)
            except Exception:
                pass

        magnitudes = valid["contrast_magnitude"].values

        # Bootstrap CI on mean magnitude
        boot_means = np.array([
            rng.choice(magnitudes, size=len(magnitudes), replace=True).mean()
            for _ in range(n_bootstrap)
        ])
        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))

        mean_contrast = np.mean(vectors, axis=0).tolist() if vectors else []

        cohort_rows.append({
            groupby: label,
            "n_animals": len(valid),
            "mean_magnitude": round(float(magnitudes.mean()), 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
            "mean_contrast_vector_json": _json.dumps([round(float(x), 6) for x in mean_contrast]),
        })

    cohort_result = pd.DataFrame(cohort_rows)
    # Rename groupby column to cohort_label for consistent downstream use
    if groupby != "cohort_label":
        cohort_result = cohort_result.rename(columns={groupby: "cohort_label"})
        df = df.rename(columns={groupby: "cohort_label"})
        groups_renamed = [str(g) for g in groups]
    else:
        groups_renamed = list(groups)

    # Pairwise Mann-Whitney U
    stat_rows = []
    gl = list(groups_renamed)
    for i in range(len(gl)):
        for j in range(i + 1, len(gl)):
            a_lab, b_lab = gl[i], gl[j]
            col_label = "cohort_label" if groupby != "cohort_label" else groupby
            a_mags = df[(df[col_label] == groups[i])]["contrast_magnitude"].dropna().values
            b_mags = df[(df[col_label] == groups[j])]["contrast_magnitude"].dropna().values
            if len(a_mags) < 2 or len(b_mags) < 2:
                continue
            try:
                u, p = mannwhitneyu(a_mags, b_mags, alternative="two-sided")
            except Exception:
                continue
            stat_rows.append({
                "cohort_A": a_lab, "cohort_B": b_lab,
                "n_A": len(a_mags), "n_B": len(b_mags),
                "mean_fi_A": round(float(a_mags.mean()), 4),
                "mean_fi_B": round(float(b_mags.mean()), 4),
                "U_stat": round(float(u), 2),
                "p_uncorrected": round(float(p), 4),
            })

    stats_df = pd.DataFrame(stat_rows)
    if not stats_df.empty:
        # BH FDR correction
        n_tests = len(stats_df)
        sorted_idx = stats_df["p_uncorrected"].argsort().values
        ranks = np.arange(1, n_tests + 1)
        p_sorted = stats_df["p_uncorrected"].values[sorted_idx]
        bh = p_sorted * n_tests / ranks
        # Enforce monotonicity from right
        for k in range(n_tests - 2, -1, -1):
            bh[k] = min(bh[k], bh[k + 1])
        bh = np.minimum(bh, 1.0)
        p_fdr = np.empty(n_tests)
        p_fdr[sorted_idx] = bh
        stats_df["p_fdr"] = p_fdr.round(4)
        stats_df["significant"] = stats_df["p_fdr"] < 0.05

        print("\nPairwise cohort comparisons (contrast magnitude):")
        print(stats_df[["cohort_A", "cohort_B", "n_A", "n_B",
                         "mean_fi_A", "mean_fi_B", "U_stat",
                         "p_uncorrected", "p_fdr", "significant"]].to_string(index=False))

    cohort_out = out_dir / "cohort_contrast_vectors.csv"
    stats_out  = out_dir / "cohort_contrast_stats.csv"
    cohort_result.to_csv(cohort_out, index=False, encoding="utf-8")
    stats_df.to_csv(stats_out, index=False, encoding="utf-8")
    print(f"Saved: {cohort_out}")
    print(f"Saved: {stats_out}")
    return cohort_result, stats_df


# ---------------------------------------------------------------------------
# Cohort distance matrix (from per-condition contrast vectors)
# ---------------------------------------------------------------------------

def compute_cohort_distances(
    condition_contrast_csv: str,
    output_dir: str = "results/quantification",
) -> "pd.DataFrame":
    """
    Pairwise L2 distances between cohort contrast vectors from condition_contrast.csv.
    d(A,B) = ||contrast_A - contrast_B||_2  (symmetric by construction).
    Saves cohort_distance_matrix.csv.
    """
    import json as _json

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p = Path(condition_contrast_csv)
    if not p.exists():
        sys.exit(f"[ERROR] {p} not found. Run: python quantify.py --contrast first.")

    df = pd.read_csv(p)
    if "cohort_label" not in df.columns:
        print("[WARN] No cohort_label column in condition_contrast.csv — skipping distance matrix")
        return pd.DataFrame()

    labels  = df["cohort_label"].tolist()
    vectors = []
    for cv_str in df["contrast_vector_json"]:
        try:
            v = np.array(_json.loads(cv_str), dtype=float)
        except Exception:
            v = np.full(1, np.nan)
        vectors.append(v)

    n = len(labels)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                dist_mat[i, j] = 0.0
            else:
                diff = vectors[i] - vectors[j]
                dist_mat[i, j] = (
                    float(np.linalg.norm(diff)) if not np.any(np.isnan(diff)) else np.nan
                )

    dist_df  = pd.DataFrame(dist_mat, index=labels, columns=labels)
    out_path = out_dir / "cohort_distance_matrix.csv"
    dist_df.to_csv(out_path, encoding="utf-8")

    print("\n=== Cohort Behavioral Distance Matrix ===")
    print(dist_df.round(4).to_string())
    print(f"Saved: {out_path}")
    return dist_df


# ---------------------------------------------------------------------------
# Jess correlation
# ---------------------------------------------------------------------------

def run_jess_correlation(master_table_csv, jess_csv, output_dir):
    """Correlate every numeric behavioral column against every Jess protein."""
    from scipy.stats import pearsonr, spearmanr

    master = pd.read_csv(master_table_csv)
    jess = pd.read_csv(jess_csv)
    jess["animal_id"] = jess["animal_id"].astype(str)
    master["animal_id"] = master["animal_id"].astype(str)

    merged = master.merge(jess, on="animal_id", how="inner")
    print(f"{len(merged)}/{len(master)} animals matched")

    jess_proteins = [c for c in jess.columns if c != "animal_id"]
    id_cols = {"animal_id", "cohort_label", "sex", "genotype", "treatment",
               "age_group", "age_weeks"}
    behav_cols = [c for c in master.columns
                  if c not in id_cols and pd.api.types.is_numeric_dtype(master[c])]

    rows = []
    for bv in behav_cols:
        for jp in jess_proteins:
            sub = merged[[bv, jp]].dropna()
            n = len(sub)
            if n < 3:
                continue
            if n < 10:
                warnings.warn(f"Only {n} pairs for {bv} vs {jp}")
            try:
                pr, pp = pearsonr(sub[bv].values, sub[jp].values)
                sr, sp = spearmanr(sub[bv].values, sub[jp].values)
            except Exception:
                continue
            rows.append({"behavioral_var": bv, "jess_protein": jp,
                         "pearson_r": round(pr, 4), "pearson_p": round(pp, 4),
                         "spearman_rho": round(sr, 4), "spearman_p": round(sp, 4),
                         "n_pairs": n})

    if not rows:
        print("No correlations computed.")
        return pd.DataFrame()

    result = pd.DataFrame(rows)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "jess_correlations.csv", index=False)

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pivot = result.pivot(index="behavioral_var", columns="jess_protein", values="pearson_r")
        pivot_p = result.pivot(index="behavioral_var", columns="jess_protein", values="pearson_p")
        fig, ax = plt.subplots(figsize=(max(6, len(jess_proteins)*0.8),
                                        max(6, len(behav_cols)*0.4)))
        im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=7)
        fig.colorbar(im, ax=ax, label="Pearson r")
        for i, bv in enumerate(pivot.index):
            for j, jp in enumerate(pivot.columns):
                try:
                    p = pivot_p.loc[bv, jp]
                except Exception:
                    p = 1.0
                marker = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                if marker:
                    ax.text(j, i, marker, ha="center", va="center", fontsize=6,
                            color="black" if abs(pivot.loc[bv, jp]) < 0.5 else "white")
        ax.set_title("Jess Protein x Behavioral Variable Correlations")
        fig.tight_layout()
        fig.savefig(out_dir / "correlation_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Heatmap saved to {out_dir / 'correlation_heatmap.png'}")
    except Exception as e:
        print(f"Could not generate heatmap: {e}")

    print(f"\nTop 20 by Pearson r:")
    print(result.nlargest(20, "pearson_r")[
        ["behavioral_var", "jess_protein", "pearson_r", "pearson_p"]
    ].to_string(index=False))
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build master quantification table and run Jess correlations")
    parser.add_argument("--build", action="store_true",
                        help="Build results/quantification/master_table.csv")
    parser.add_argument("--jess", metavar="FILE",
                        help="Jess protein CSV/Excel file")
    parser.add_argument("--contrast", action="store_true",
                        help="Compute behavioral contrast vectors (per-condition + per-animal)")
    parser.add_argument("--per-animal", action="store_true",
                        help="With --contrast: run per-animal mode only (skip per-condition)")
    parser.add_argument("--cohort", metavar="FILE", default=None,
                        help="Cohort CSV/Excel (auto-detected if omitted)")
    parser.add_argument("--out", metavar="DIR", default=str(_OUT),
                        help="Output directory (default: results/quantification)")
    args = parser.parse_args()

    if not args.build and not args.jess and not args.contrast:
        parser.print_help(); sys.exit(1)

    if args.build:
        build_master_table(cohort_path=args.cohort, out_dir=args.out)

    if args.contrast:
        summary_csv = str(_COMP / "summary_table.csv")
        if not Path(summary_csv).exists():
            sys.exit("[ERROR] results/comparison/summary_table.csv not found. "
                     "Run: python compare.py --extract --cluster --report first.")
        cohort_path = args.cohort
        if cohort_path is None:
            for c in ("cohort_normalized.csv",):
                if os.path.exists(c):
                    cohort_path = c
                    break

        per_animal_only = getattr(args, "per_animal", False)

        if not per_animal_only:
            # Per-condition (default) — one vector per cohort group
            compute_contrast_vector(
                summary_csv=summary_csv,
                output_dir=args.out,
                cohort_csv=cohort_path,
                per_condition=True,
            )
            condition_csv = str(Path(args.out) / "condition_contrast.csv")
            if Path(condition_csv).exists():
                compute_cohort_distances(condition_csv, output_dir=args.out)

        # Per-animal always runs (both modes) unless --per-animal forces solo
        compute_contrast_vector(
            summary_csv=summary_csv,
            output_dir=args.out,
            cohort_csv=cohort_path,
            per_condition=False,
        )

        # Cohort-level bootstrap + Mann-Whitney on per-animal magnitudes
        if not per_animal_only:
            animal_csv = str(Path(args.out) / "contrast_vectors.csv")
            if Path(animal_csv).exists():
                tmp = pd.read_csv(animal_csv)
                if "cohort_label" in tmp.columns:
                    compute_cohort_contrast(
                        contrast_csv=animal_csv,
                        cohort_csv=cohort_path,
                        output_dir=args.out,
                    )

    if args.jess:
        master_path = Path(args.out) / "master_table.csv"
        if not master_path.exists():
            sys.exit(f"[ERROR] {master_path} not found. Run --build first.")
        ext = os.path.splitext(args.jess)[1].lower()
        if ext in (".xlsx", ".xls", ".xlsm"):
            jess_df = pd.read_excel(args.jess)
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tf:
                jess_df.to_csv(tf, index=False)
                jess_path = tf.name
        else:
            jess_path = args.jess
        run_jess_correlation(str(master_path), jess_path, args.out)


if __name__ == "__main__":
    main()
