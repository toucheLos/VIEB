"""
fear_index.py — Cohort-normalized fear quantification for VIEB.

Three computations that serve the paper's core claim:

  1. cohort_normalize()        — remove baseline differences between cohorts
  2. compute_fear_index()      — scalar measure of fear-specific behavior
  3. cohort_fear_profiles()    — compare fear signatures across cohorts (TMZ vs Vehicle)

All functions work on summary_table.csv (one row per session per animal) and
cohort_normalized.csv (one row per animal with genotype/age/sex/treatment).

Usage
-----
    from fear_index import cohort_normalize, compute_fear_index, cohort_fear_profiles
    import pandas as pd

    summary  = pd.read_csv("results/comparison/summary_table.csv")
    cohort   = pd.read_csv("cohort_normalized.csv")

    normed   = cohort_normalize(summary, cohort)
    fi       = compute_fear_index(normed)
    profiles = cohort_fear_profiles(normed, cohort)

Or from the CLI:
    python fear_index.py [--cohort cohort_normalized.csv] [--out results/quantification/]
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure UTF-8 output on Windows terminals that default to cp1252
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _state_cols(df: pd.DataFrame) -> list[str]:
    return sorted(
        [c for c in df.columns if c.startswith("state_") and c.endswith("_frac")],
        key=lambda c: int(c.split("_")[1]),
    )


def _dominant_state(df: pd.DataFrame) -> int | None:
    """Return the state ID with the highest mean occupancy (the noise/dominant state)."""
    cols = _state_cols(df)
    if not cols:
        return None
    means = df[cols].mean()
    return int(means.idxmax().split("_")[1])


def _non_dominant_cols(df: pd.DataFrame, dominant: int | None) -> list[str]:
    cols = _state_cols(df)
    if dominant is None:
        return cols
    return [c for c in cols if int(c.split("_")[1]) != dominant]


# ---------------------------------------------------------------------------
# Step 1 — Cohort normalization
# ---------------------------------------------------------------------------

def cohort_normalize(
    summary: pd.DataFrame,
    cohort: pd.DataFrame,
    context_col: str = "context",
    groupby: str = "cohort_label",
) -> pd.DataFrame:
    """
    Remove baseline behavioral differences between cohorts.

    For each cohort (Age × Treatment × Sex group defined by cohort_label):
      1. Compute the cohort mean state occupancy across ALL sessions and contexts.
      2. Subtract that mean from each animal's per-context occupancy vector.

    The result is an animal × state matrix of *deviation from cohort baseline*
    separately for context A and context B.

    Parameters
    ----------
    summary  : DataFrame with columns stem, animal_id, context, day, state_*_frac, ...
    cohort   : DataFrame with columns animal_id, cohort_label (+ sex, age_group, etc.)
    context_col : column name in summary that encodes the behavioral context ("A", "B", "C")
    groupby  : column in cohort that defines the normalization group

    Returns
    -------
    DataFrame indexed by animal_id with columns:
        {state_k}_A_raw, {state_k}_B_raw,
        {state_k}_A_norm, {state_k}_B_norm
    for each non-dominant state k.
    """
    required = {"animal_id", context_col}
    missing = required - set(summary.columns)
    if missing:
        raise ValueError(f"summary_table is missing columns: {missing}")
    if "animal_id" not in cohort.columns or groupby not in cohort.columns:
        raise ValueError(f"cohort must have 'animal_id' and '{groupby}' columns")

    dom = _dominant_state(summary)
    sc = _non_dominant_cols(summary, dom)
    if not sc:
        raise ValueError("No state_*_frac columns found in summary_table")

    ctx_vals = summary[context_col].dropna().astype(str).unique()
    ctx_A = [v for v in ctx_vals if v.upper().startswith("A")]
    ctx_B = [v for v in ctx_vals if v.upper().startswith("B")]

    # Per-animal, per-context mean occupancy
    rows = []
    for animal_id, grp in summary.groupby("animal_id"):
        a_rows = grp[grp[context_col].isin(ctx_A)]
        b_rows = grp[grp[context_col].isin(ctx_B)]
        entry = {"animal_id": animal_id}
        for col in sc:
            entry[f"{col}_A_raw"] = float(a_rows[col].mean()) if len(a_rows) else float("nan")
            entry[f"{col}_B_raw"] = float(b_rows[col].mean()) if len(b_rows) else float("nan")
        rows.append(entry)

    raw_df = pd.DataFrame(rows).set_index("animal_id")

    # Join cohort labels
    cmap = cohort.set_index("animal_id")[groupby].to_dict()
    raw_df["_cohort"] = pd.Series({k: cmap.get(k) for k in raw_df.index})

    # Subtract cohort mean per context column
    norm_df = raw_df.copy()
    raw_cols_A = [f"{col}_A_raw" for col in sc]
    raw_cols_B = [f"{col}_B_raw" for col in sc]

    for grp_label, grp_idx in raw_df.groupby("_cohort").groups.items():
        cohort_mean_A = raw_df.loc[grp_idx, raw_cols_A].mean()
        cohort_mean_B = raw_df.loc[grp_idx, raw_cols_B].mean()
        for col, raw_col in zip(sc, raw_cols_A):
            norm_df.loc[grp_idx, f"{col}_A_norm"] = (
                raw_df.loc[grp_idx, raw_col] - cohort_mean_A[raw_col]
            )
        for col, raw_col in zip(sc, raw_cols_B):
            norm_df.loc[grp_idx, f"{col}_B_norm"] = (
                raw_df.loc[grp_idx, raw_col] - cohort_mean_B[raw_col]
            )

    norm_df = norm_df.drop(columns=["_cohort"])
    return norm_df


# ---------------------------------------------------------------------------
# Step 2 — Fear index
# ---------------------------------------------------------------------------

def compute_fear_index(
    normalized: pd.DataFrame,
    state_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute per-animal fear index from cohort-normalized occupancy profiles.

    fear_index = dot(fear_profile, fear_profile - safe_profile)

    where:
      fear_profile = normalized A-context occupancy vector  (deviations from cohort mean)
      safe_profile = normalized B-context occupancy vector

    Interpretation:
      > 0  : animal shows more fear-specific behavior relative to its cohort
      < 0  : animal shows less fear-specific behavior than its cohort average
      ≈ 0  : no context discrimination at the cohort-normalized level

    The formula is the dot product of the fear profile with the A−B contrast,
    which rewards animals whose fear profile aligns with the fear direction
    AND has a large magnitude.

    Parameters
    ----------
    normalized  : output of cohort_normalize()
    state_cols  : which non-dominant state columns to use (auto-detected if None)

    Returns
    -------
    DataFrame with columns: animal_id, fear_index, fear_profile_norm, safe_profile_norm,
    context_discrimination (scalar: mean A_norm − mean B_norm)
    """
    if state_cols is None:
        # Detect from column names: state_*_A_norm
        state_cols = sorted(
            {c.replace("_A_norm", "").replace("_A_raw", "")
             for c in normalized.columns
             if "_A_norm" in c},
            key=lambda c: int(c.split("_")[1]) if c.split("_")[1].isdigit() else 0,
        )
    if not state_cols:
        raise ValueError("No normalized state columns found; run cohort_normalize() first")

    a_cols = [f"{s}_A_norm" for s in state_cols]
    b_cols = [f"{s}_B_norm" for s in state_cols]

    missing = [c for c in a_cols + b_cols if c not in normalized.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    rows = []
    for animal_id in normalized.index:
        fear_vec = normalized.loc[animal_id, a_cols].values.astype(float)
        safe_vec = normalized.loc[animal_id, b_cols].values.astype(float)

        contrast = fear_vec - safe_vec
        fi = float(np.dot(fear_vec, contrast))

        # Context discrimination: mean A_norm − mean B_norm (simple scalar)
        disc = float(np.nanmean(fear_vec) - np.nanmean(safe_vec))

        rows.append({
            "animal_id": animal_id,
            "fear_index": round(fi, 6),
            "context_discrimination": round(disc, 6),
            "fear_profile_l2": round(float(np.linalg.norm(fear_vec)), 6),
            "safe_profile_l2": round(float(np.linalg.norm(safe_vec)), 6),
        })

    return pd.DataFrame(rows).set_index("animal_id")


# ---------------------------------------------------------------------------
# Step 3 — Cohort fear profiles
# ---------------------------------------------------------------------------

def cohort_fear_profiles(
    normalized: pd.DataFrame,
    cohort: pd.DataFrame,
    groupby: str = "cohort_label",
    state_cols: list[str] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compare cohort-mean fear profiles to ask whether TMZ-treated animals show
    a different behavioral signature of fear than vehicle controls.

    For each cohort defined by `groupby`:
      - Compute mean normalized A-context occupancy vector (fear profile)
      - Compute mean normalized B-context occupancy vector (safe profile)
      - Compute the cohort's mean fear_index
      - Bootstrap 95% CI on the fear_index

    Returns a DataFrame with one row per cohort and per-state fear profile means,
    plus fear_index mean ± CI.

    This table is what goes into the paper's cohort comparison figure.
    """
    if state_cols is None:
        state_cols = sorted(
            {c.replace("_A_norm", "")
             for c in normalized.columns if "_A_norm" in c},
            key=lambda c: int(c.split("_")[1]) if c.split("_")[1].isdigit() else 0,
        )

    a_cols = [f"{s}_A_norm" for s in state_cols]
    b_cols = [f"{s}_B_norm" for s in state_cols]

    # Join cohort labels
    cmap = cohort.set_index("animal_id")[groupby].to_dict()
    norm = normalized.copy()
    norm["_cohort"] = pd.Series({k: cmap.get(k) for k in norm.index})

    fi_df = compute_fear_index(normalized, state_cols=state_cols)
    norm = norm.join(fi_df["fear_index"], how="left")

    rng = np.random.default_rng(seed)
    rows = []
    for label, grp in norm.groupby("_cohort"):
        n = len(grp)
        a_means = grp[a_cols].mean().values
        b_means = grp[b_cols].mean().values

        fear_indices = grp["fear_index"].dropna().values
        if len(fear_indices) >= 2:
            bs = [rng.choice(fear_indices, len(fear_indices), replace=True).mean()
                  for _ in range(n_bootstrap)]
            ci_lo, ci_hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
        else:
            ci_lo = ci_hi = float(fear_indices.mean()) if len(fear_indices) else float("nan")

        row = {
            groupby: label,
            "n_animals": n,
            "mean_fear_index": round(float(np.nanmean(fear_indices)), 6),
            "fear_index_ci_lo": round(ci_lo, 6),
            "fear_index_ci_hi": round(ci_hi, 6),
        }
        for col, val in zip(a_cols, a_means):
            row[f"{col}_mean"] = round(float(val), 6)
        for col, val in zip(b_cols, b_means):
            row[f"{col}_mean"] = round(float(val), 6)

        rows.append(row)

    return pd.DataFrame(rows).sort_values("mean_fear_index", ascending=False)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_data(cohort_path: str | None):
    summary_path = "results/comparison/summary_table.csv"
    if not os.path.exists(summary_path):
        sys.exit("[ERROR] results/comparison/summary_table.csv not found. "
                 "Run: python compare.py --extract --cluster --report")

    summary = pd.read_csv(summary_path)

    if cohort_path is None:
        for candidate in ("cohort_normalized.csv",):
            if os.path.exists(candidate):
                cohort_path = candidate
                break

    if cohort_path is None or not os.path.exists(cohort_path):
        sys.exit(
            "[ERROR] Cohort file not found.\n"
            "Pass --cohort <file.csv/.xlsx> or run:\n"
            "  python prepare_cohort.py --input <cohort.xlsx> --output cohort_normalized.csv"
        )

    ext = os.path.splitext(cohort_path)[1].lower()
    if ext in (".xlsx", ".xls", ".xlsm"):
        from cohort_loader import load_cohort_excel
        cohort = load_cohort_excel(cohort_path)
    else:
        cohort = pd.read_csv(cohort_path)

    return summary, cohort


def main():
    parser = argparse.ArgumentParser(
        description="Compute cohort-normalized fear index",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cohort", metavar="FILE", default=None,
                        help="Cohort CSV/Excel (auto-detected if omitted)")
    parser.add_argument("--out", metavar="DIR", default="results/quantification",
                        help="Output directory (default: results/quantification)")
    parser.add_argument("--groupby", default="cohort_label",
                        help="Column to define cohorts (default: cohort_label)")
    args = parser.parse_args()

    summary, cohort = _load_data(args.cohort)
    os.makedirs(args.out, exist_ok=True)

    print("Step 1: cohort normalization...")
    normed = cohort_normalize(summary, cohort, groupby=args.groupby)
    normed_path = os.path.join(args.out, "cohort_normalized_profiles.csv")
    normed.reset_index().to_csv(normed_path, index=False)
    print(f"  → {normed_path}  ({len(normed)} animals)")

    print("\nStep 2: per-animal fear index...")
    fi = compute_fear_index(normed)
    fi_path = os.path.join(args.out, "fear_index.csv")
    fi.reset_index().to_csv(fi_path, index=False)
    print(f"  → {fi_path}")
    print(f"\n  fear_index distribution:")
    print(fi[["fear_index", "context_discrimination"]].describe().round(4).to_string())

    print("\nStep 3: cohort fear profiles...")
    profiles = cohort_fear_profiles(normed, cohort, groupby=args.groupby)
    profiles_path = os.path.join(args.out, "cohort_fear_profiles.csv")
    profiles.to_csv(profiles_path, index=False)
    print(f"  → {profiles_path}")
    print(f"\n  Cohort fear index comparison:")
    print(profiles[[args.groupby, "n_animals", "mean_fear_index",
                    "fear_index_ci_lo", "fear_index_ci_hi"]].to_string(index=False))

    print(f"\nAll outputs in {args.out}/")


if __name__ == "__main__":
    main()
