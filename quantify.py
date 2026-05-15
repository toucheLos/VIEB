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
    parser.add_argument("--cohort", metavar="FILE", default=None,
                        help="Cohort CSV/Excel (auto-detected if omitted)")
    parser.add_argument("--out", metavar="DIR", default=str(_OUT),
                        help="Output directory (default: results/quantification)")
    args = parser.parse_args()

    if not args.build and not args.jess:
        parser.print_help(); sys.exit(1)

    if args.build:
        build_master_table(cohort_path=args.cohort, out_dir=args.out)

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
