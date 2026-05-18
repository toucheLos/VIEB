"""
cohort_analysis.py — Cohort-level behavioral state analysis for VIEB.

Usage:
    python cohort_analysis.py --cohort cohort_normalized.csv --output results/cohort/
    python cohort_analysis.py --cohort cohort.xlsx --groupby genotype_treatment --dry-run
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import warnings
from itertools import combinations
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# FDR correction (Benjamini-Hochberg)
# ---------------------------------------------------------------------------

def _bh_correct(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction; falls back to manual if statsmodels absent."""
    try:
        from statsmodels.stats.multitest import multipletests
        _, p_corrected, _, _ = multipletests(pvalues, method="fdr_bh")
        return p_corrected
    except ImportError:
        pass
    n = len(pvalues)
    if n == 0:
        return np.array([])
    order = np.argsort(pvalues)
    adjusted = np.minimum(1.0, pvalues[order] * n / (np.arange(1, n + 1)))
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])
    result = np.empty(n)
    result[order] = adjusted
    return result


# ---------------------------------------------------------------------------
# Cohort label builder
# ---------------------------------------------------------------------------

GROUPBY_OPTIONS = {"age", "treatment", "sex", "genotype",
                   "age_treatment", "genotype_treatment", "age_sex", "full"}


def _clean(s: str) -> str:
    return str(s).strip().replace(" ", "_").replace("-", "_")


def _safe_fname(s: str) -> str:
    return re.sub(r"[^\w\-]", "_", s)[:60]


def _build_cohort_group(df_cohort: pd.DataFrame, groupby: str) -> pd.Series:
    """Return a Series with one cohort group label per row."""
    c = df_cohort
    if groupby == "age":
        return c["age_group"].apply(_clean)
    if groupby == "treatment":
        return c["treatment"].apply(_clean)
    if groupby == "sex":
        return c["sex"].apply(_clean)
    if groupby == "genotype":
        return c["genotype"].apply(_clean)
    if groupby == "age_treatment":
        return c["age_group"].apply(_clean) + "_" + c["treatment"].apply(_clean)
    if groupby == "genotype_treatment":
        return c["genotype"].apply(_clean) + "_" + c["treatment"].apply(_clean)
    if groupby == "age_sex":
        return c["age_group"].apply(_clean) + "_" + c["sex"].apply(_clean)
    if groupby == "full":
        return c["cohort_label"]
    raise ValueError(f"Unknown --groupby: {groupby!r}")


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

def _state_colors(n: int) -> np.ndarray:
    if n <= 10:
        return plt.cm.tab10(np.linspace(0, 1, max(n, 1)))
    return plt.cm.tab20(np.linspace(0, 1, max(n, 1)))


def _cohort_colors(n: int) -> np.ndarray:
    if n <= 9:
        return plt.cm.Set1(np.linspace(0, 0.9, max(n, 1)))
    return plt.cm.tab20(np.linspace(0, 1, max(n, 1)))


# ---------------------------------------------------------------------------
# Scalar metric resolution
# ---------------------------------------------------------------------------

SCALAR_METRICS = [
    ("fear_AUC",               ["fear_AUC", "fear_AUC_normalized", "freeze_auc", "freeze_AUC"]),
    ("disc_ratio",             ["disc_ratio", "mean_discrimination_ratio", "discrimination_ratio"]),
    ("behavioral_diversity_A", ["behavioral_diversity_A"]),
    ("behavioral_diversity_B", ["behavioral_diversity_B"]),
    ("n_states_used_A",        ["n_states_used_A"]),
    ("n_sessions",             ["n_sessions"]),
    ("n_days",                 ["n_days"]),
]


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _bar_means_ses(ax, x, means, sems, colors, bar_width=0.7, **kwargs):
    sems_safe = [s if (s is not None and not np.isnan(s)) else 0.0 for s in sems]
    ax.bar(x, means, width=bar_width, color=colors, yerr=sems_safe,
           capsize=3, edgecolor="black", linewidth=0.5, alpha=0.85, **kwargs)


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run(args):
    out_dir = args.output
    dry = args.dry_run

    if not dry:
        os.makedirs(out_dir, exist_ok=True)

    # ===== Task 1: Load and merge ===========================================
    print("Loading cohort file...")
    from cohort_loader import load_cohort_excel
    df_cohort = load_cohort_excel(args.cohort)
    df_cohort["cohort_group"] = _build_cohort_group(df_cohort, args.groupby)
    animal_to_group = df_cohort.set_index("animal_id")["cohort_group"].to_dict()
    cohort_ids = set(df_cohort["animal_id"].unique())

    print(f"Loading summary table: {args.summary}")
    if not os.path.exists(args.summary):
        sys.exit(f"[ERROR] summary_table.csv not found: {args.summary}")
    df_summary = pd.read_csv(args.summary)

    if "animal_id" not in df_summary.columns:
        sys.exit("[ERROR] summary_table.csv has no 'animal_id' column.")
    df_summary["animal_id"] = pd.to_numeric(df_summary["animal_id"], errors="coerce")
    df_summary = df_summary.dropna(subset=["animal_id"]).copy()
    df_summary["animal_id"] = df_summary["animal_id"].astype(int)

    # Warn about unmatched animal_ids and exclude them
    summary_ids = set(df_summary["animal_id"].unique())
    unmatched = sorted(summary_ids - cohort_ids)
    if unmatched:
        print(f"\n[WARNING] {len(unmatched)} animal_id(s) in summary_table have no cohort "
              f"match and will be excluded: {unmatched}")

    df_summary = df_summary[df_summary["animal_id"].isin(cohort_ids)].copy()
    df_summary["cohort_group"] = df_summary["animal_id"].map(animal_to_group)
    df_summary = df_summary.dropna(subset=["cohort_group"])

    state_cols = sorted(
        [c for c in df_summary.columns if c.startswith("state_") and c.endswith("_frac")],
        key=lambda c: int(c.split("_")[1]),
    )
    if not state_cols:
        sys.exit("[ERROR] No state_*_frac columns found in summary_table.csv.")
    state_ids = [int(c.split("_")[1]) for c in state_cols]

    # Print cohort summary
    print("\n" + "-" * 58)
    print(f"{'Cohort':<34} {'N animals':>10}  {'N sessions':>10}")
    print("-" * 58)
    for group in sorted(df_summary["cohort_group"].unique()):
        grp = df_summary[df_summary["cohort_group"] == group]
        print(f"{group:<34} {grp['animal_id'].nunique():>10}  {len(grp):>10}")
    print("-" * 58)

    # ===== Task 2: Identify dominant state =================================
    agg_dict = {col: "mean" for col in state_cols}
    agg_dict["cohort_group"] = "first"
    per_animal = df_summary.groupby("animal_id").agg(agg_dict).reset_index()

    global_mean = per_animal[state_cols].mean()
    dominant_idx = int(global_mean.values.argmax())
    dominant_state = state_ids[dominant_idx]
    print(f"\nDominant state: {dominant_state} (excluded from analysis)")

    non_dominant_ids = [s for s in state_ids if s != dominant_state]
    non_dominant_cols = [f"state_{s}_frac" for s in non_dominant_ids]
    cohort_groups = sorted(per_animal["cohort_group"].unique())
    n_cohorts = len(cohort_groups)

    # Load heuristic labels if available
    state_labels: dict[int, str] = {}
    if os.path.exists(args.state_summary):
        try:
            df_ss = pd.read_csv(args.state_summary)
            if {"state", "heuristic_label"}.issubset(df_ss.columns):
                for _, row in df_ss.iterrows():
                    state_labels[int(row["state"])] = str(row["heuristic_label"])
        except Exception as e:
            print(f"  [WARNING] Could not read state_summary.csv: {e}")
    for sid in state_ids:
        state_labels.setdefault(sid, f"state_{sid}")

    # ===== Task 3: cohort_state_profiles.csv ================================
    print("\nComputing cohort state profiles...")
    profile_rows = []
    for group in cohort_groups:
        grp_df = per_animal[per_animal["cohort_group"] == group]
        for sid in non_dominant_ids:
            col = f"state_{sid}_frac"
            values = grp_df[col].dropna().values
            n = len(values)
            profile_rows.append({
                "cohort_group":   group,
                "state_id":       sid,
                "heuristic_label": state_labels[sid],
                "mean_fraction":  float(np.mean(values)) if n > 0 else np.nan,
                "se_fraction":    float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
                "n_animals":      n,
            })
    df_profiles = pd.DataFrame(profile_rows)

    _save(df_profiles, out_dir, "cohort_state_profiles.csv", dry)

    # ===== Task 4: cohort_behavioral_metrics.csv ===========================
    print("\nLoading animal scalars...")
    available_metrics: list[tuple[str, str]] = []
    df_metrics_out: Optional[pd.DataFrame] = None

    if not os.path.exists(args.scalars):
        print(f"  [WARNING] animal_scalars.csv not found: {args.scalars} — skipping Task 4.")
    else:
        df_scalars = pd.read_csv(args.scalars)
        df_scalars["animal_id"] = pd.to_numeric(df_scalars["animal_id"], errors="coerce")
        df_scalars = df_scalars.dropna(subset=["animal_id"]).copy()
        df_scalars["animal_id"] = df_scalars["animal_id"].astype(int)
        df_scalars["cohort_group"] = df_scalars["animal_id"].map(animal_to_group)
        df_scalars = df_scalars.dropna(subset=["cohort_group"])

        for canonical, candidates in SCALAR_METRICS:
            col = _find_col(df_scalars, candidates)
            if col is None:
                print(f"  [WARNING] Metric '{canonical}' not found in animal_scalars.csv — skipping.")
            else:
                available_metrics.append((canonical, col))

        metric_rows = []
        for group in cohort_groups:
            grp_df = df_scalars[df_scalars["cohort_group"] == group]
            row: dict = {"cohort_group": group, "n_animals": len(grp_df)}
            for canonical, col in available_metrics:
                vals = pd.to_numeric(grp_df[col], errors="coerce").dropna().values
                n = len(vals)
                row[f"{canonical}_mean"] = float(np.mean(vals)) if n > 0 else np.nan
                row[f"{canonical}_se"]   = float(np.std(vals, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
                row[f"{canonical}_n"]    = n
            metric_rows.append(row)

        df_metrics_out = pd.DataFrame(metric_rows)
        _save(df_metrics_out, out_dir, "cohort_behavioral_metrics.csv", dry)

    # ===== Task 5: cohort_statistics.csv ===================================
    print("\nRunning pairwise Mann-Whitney U tests + BH FDR correction...")
    stat_rows = []
    for sid in non_dominant_ids:
        col = f"state_{sid}_frac"
        for grp_a, grp_b in combinations(cohort_groups, 2):
            vals_a = per_animal[per_animal["cohort_group"] == grp_a][col].dropna().values
            vals_b = per_animal[per_animal["cohort_group"] == grp_b][col].dropna().values
            if len(vals_a) < 2 or len(vals_b) < 2:
                continue
            n_a, n_b = len(vals_a), len(vals_b)
            mean_a, mean_b = float(np.mean(vals_a)), float(np.mean(vals_b))
            fold_change = mean_a / (mean_b + 1e-6)
            u_stat, p_uncorr = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
            z_abs = abs(stats.norm.ppf(max(float(p_uncorr), 1e-300) / 2))
            effect_r = z_abs / np.sqrt(n_a + n_b) * np.sign(mean_a - mean_b)
            stat_rows.append({
                "state_id":       sid,
                "heuristic_label": state_labels[sid],
                "cohort_A":       grp_a,
                "cohort_B":       grp_b,
                "n_A":            n_a,
                "n_B":            n_b,
                "mean_A":         mean_a,
                "mean_B":         mean_b,
                "fold_change":    float(fold_change),
                "effect_size_r":  float(effect_r),
                "U_stat":         float(u_stat),
                "p_uncorrected":  float(p_uncorr),
                "p_fdr":          np.nan,
                "significant_fdr": False,
            })

    df_stats = pd.DataFrame(stat_rows)
    n_sig = 0
    if not df_stats.empty:
        p_fdr = _bh_correct(df_stats["p_uncorrected"].values)
        df_stats["p_fdr"] = p_fdr
        df_stats["significant_fdr"] = p_fdr < 0.05
        n_sig = int(df_stats["significant_fdr"].sum())

        n_tested = len(df_stats)
        print(f"  {n_tested} state/cohort pairs tested")
        print(f"  {n_sig} significant after FDR correction (p < 0.05)")

        top5 = df_stats[df_stats["significant_fdr"]].nlargest(5, "fold_change")
        if not top5.empty:
            print("  Top 5 significant states:")
            hdr = f"  {'state_id':<10} {'label':<35} {'comparison':<35} {'fold_change':>12} {'p_fdr':>10}"
            print(hdr)
            for _, row in top5.iterrows():
                comp = f"{row['cohort_A']} vs {row['cohort_B']}"
                print(f"  {int(row['state_id']):<10} {row['heuristic_label'][:34]:<35} "
                      f"{comp[:34]:<35} {row['fold_change']:>12.3f} {row['p_fdr']:>10.4f}")

        _save(df_stats, out_dir, "cohort_statistics.csv", dry)
        df_sig = df_stats[df_stats["significant_fdr"]].sort_values("fold_change", ascending=False)
        _save(df_sig, out_dir, "cohort_significant_states.csv", dry)
    else:
        print("  [WARNING] No pairwise comparisons possible (need ≥2 animals per group in ≥2 groups).")

    # ===== Task 6: cohort_state_profiles.png ================================
    if not dry:
        print("\nGenerating cohort_state_profiles.png...")
        colors_by_state = _state_colors(len(non_dominant_ids))
        state_color_map = {sid: colors_by_state[i] for i, sid in enumerate(non_dominant_ids)}

        max_frac = df_profiles["mean_fraction"].max()
        y_max = (max_frac if not np.isnan(max_frac) else 0.1) * 1.1

        fig, axes = plt.subplots(1, n_cohorts, figsize=(5 * n_cohorts, 5), sharey=True)
        if n_cohorts == 1:
            axes = [axes]

        x = np.arange(len(non_dominant_ids))
        for ax, group in zip(axes, cohort_groups):
            grp_prof = df_profiles[df_profiles["cohort_group"] == group].set_index("state_id")
            n_animals_grp = per_animal[per_animal["cohort_group"] == group]["animal_id"].nunique()
            bar_colors = [state_color_map[sid] for sid in non_dominant_ids]
            means = [grp_prof.loc[sid, "mean_fraction"] if sid in grp_prof.index else 0.0
                     for sid in non_dominant_ids]
            sems  = [grp_prof.loc[sid, "se_fraction"]   if sid in grp_prof.index else 0.0
                     for sid in non_dominant_ids]
            _bar_means_ses(ax, x, means, sems, bar_colors)
            ax.set_xticks(x)
            ax.set_xticklabels([str(s) for s in non_dominant_ids], fontsize=8)
            ax.set_xlabel("State ID")
            ax.set_title(f"{group}\n(N={n_animals_grp})", fontsize=9)
            ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)

        axes[0].set_ylim(0, y_max)
        axes[0].set_ylabel("Mean fraction")
        plt.suptitle("Cohort State Profiles (non-dominant states)", fontsize=11)
        plt.tight_layout()
        _savefig(fig, out_dir, "cohort_state_profiles.png")

    # ===== Task 7: cohort_comparison.png ====================================
    if not dry and n_cohorts >= 2:
        print("\nGenerating cohort_comparison.png...")

        # Per-state between-cohort spread (max mean – min mean)
        cohort_state_means: dict[tuple, float] = {}
        for group in cohort_groups:
            gp = df_profiles[df_profiles["cohort_group"] == group].set_index("state_id")
            for sid in non_dominant_ids:
                cohort_state_means[(group, sid)] = (
                    float(gp.loc[sid, "mean_fraction"]) if sid in gp.index else 0.0
                )

        state_spread = {
            sid: max(cohort_state_means[(g, sid)] for g in cohort_groups) -
                 min(cohort_state_means[(g, sid)] for g in cohort_groups)
            for sid in non_dominant_ids
        }
        top20 = sorted(non_dominant_ids, key=lambda s: -state_spread[s])[:20]
        c_colors = _cohort_colors(n_cohorts)

        if n_cohorts <= 4:
            fig, ax = plt.subplots(figsize=(16, 6))
            _draw_grouped_bars(ax, top20, cohort_groups, df_profiles, cohort_state_means,
                               c_colors, "Top 20 States by Between-Cohort Difference")
            _savefig(fig, out_dir, "cohort_comparison.png")
        else:
            pairs = list(combinations(cohort_groups, 2))
            print(f"  {n_cohorts} cohorts > 4 — generating {len(pairs)} pairwise charts.")
            for i, (grp_a, grp_b) in enumerate(pairs):
                fig, ax = plt.subplots(figsize=(16, 6))
                pair_groups = [grp_a, grp_b]
                pair_colors = np.array([c_colors[cohort_groups.index(grp_a)],
                                        c_colors[cohort_groups.index(grp_b)]])
                _draw_grouped_bars(ax, top20, pair_groups, df_profiles, cohort_state_means,
                                   pair_colors, f"Cohort Comparison: {grp_a} vs {grp_b}")
                fname = f"cohort_comparison_{i+1:02d}_{_safe_fname(grp_a)}_vs_{_safe_fname(grp_b)}.png"
                _savefig(fig, out_dir, fname)

    # ===== Task 8: cohort_metrics.png =======================================
    if not dry and df_metrics_out is not None and available_metrics:
        print("\nGenerating cohort_metrics.png...")
        n_metrics = len(available_metrics)
        n_cols = max(1, (n_metrics + 1) // 2)
        n_rows = 2
        c_colors_m = _cohort_colors(n_cohorts)
        x = np.arange(n_cohorts)
        bar_width = max(0.3, 0.6 / max(n_cohorts / 3, 1))

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_metrics, 8), squeeze=False)
        axes_flat = axes.flatten()

        for ax_idx, (canonical, _col) in enumerate(available_metrics):
            ax = axes_flat[ax_idx]
            means = [df_metrics_out.loc[df_metrics_out["cohort_group"] == g,
                                        f"{canonical}_mean"].values[0]
                     if (df_metrics_out["cohort_group"] == g).any() else np.nan
                     for g in cohort_groups]
            sems  = [df_metrics_out.loc[df_metrics_out["cohort_group"] == g,
                                        f"{canonical}_se"].values[0]
                     if (df_metrics_out["cohort_group"] == g).any() else np.nan
                     for g in cohort_groups]
            _bar_means_ses(ax, x, means, sems, c_colors_m, bar_width=bar_width)
            if canonical == "disc_ratio":
                ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xticks(x)
            ax.set_xticklabels(cohort_groups, rotation=45, ha="right", fontsize=7)
            ax.set_title(canonical, fontsize=9)
            ax.set_ylabel("Mean ± SE")
            ax.grid(True, axis="y", alpha=0.3)

        for ax_idx in range(len(available_metrics), len(axes_flat)):
            axes_flat[ax_idx].set_visible(False)

        plt.suptitle("Behavioral Metrics by Cohort", fontsize=11)
        plt.tight_layout()
        _savefig(fig, out_dir, "cohort_metrics.png")

    # ===== Task 9: Summary print ============================================
    total_animals_analyzed = per_animal["animal_id"].nunique()
    total_cohort_animals   = len(df_cohort)
    total_sessions         = len(df_summary)

    # Metric means for summary lines
    fear_auc_line   = "(not available)"
    disc_ratio_line = "(not available)"
    if df_metrics_out is not None:
        if f"fear_AUC_mean" in df_metrics_out.columns:
            parts = [f"{row['cohort_group']}={row['fear_AUC_mean']:.4f}"
                     for _, row in df_metrics_out.iterrows()
                     if pd.notna(row.get("fear_AUC_mean"))]
            if parts:
                fear_auc_line = ", ".join(parts)
        if "disc_ratio_mean" in df_metrics_out.columns:
            parts = [f"{row['cohort_group']}={row['disc_ratio_mean']:.4f}"
                     for _, row in df_metrics_out.iterrows()
                     if pd.notna(row.get("disc_ratio_mean"))]
            if parts:
                disc_ratio_line = ", ".join(parts)

    top5_lines = []
    if not df_stats.empty and n_sig > 0:
        for _, row in df_stats[df_stats["significant_fdr"]].nlargest(5, "fold_change").iterrows():
            top5_lines.append(
                f"  State {int(row['state_id'])}: {row['heuristic_label'][:30]} | "
                f"{row['cohort_A']} vs {row['cohort_B']} | "
                f"FC={row['fold_change']:.3f}, p_fdr={row['p_fdr']:.4f}"
            )

    output_files = [
        "cohort_state_profiles.csv",
        "cohort_behavioral_metrics.csv",
        "cohort_statistics.csv",
        "cohort_significant_states.csv",
        "cohort_state_profiles.png",
        "cohort_comparison.png",
        "cohort_metrics.png",
    ]

    print("\n" + "=" * 60)
    print("VIEB — Cohort Analysis Complete")
    print("=" * 60)
    print(f"Cohorts analyzed: {n_cohorts}")
    print(f"Animals included: {total_animals_analyzed}/{total_cohort_animals}")
    print(f"Sessions included: {total_sessions}")
    print(f"States analyzed: {len(non_dominant_ids)} (dominant state {dominant_state} excluded)")
    if dry:
        print("\n[dry-run] No files written.")
    else:
        print(f"\nOutputs saved to {out_dir}:")
        for f in output_files:
            exists = os.path.exists(os.path.join(out_dir, f))
            print(f"  {'  ' if exists else '  [SKIPPED] '}{f}")
    n_tested = len(df_stats)
    print(f"\nSignificant state differences (FDR p < 0.05):")
    print(f"  {n_sig} state/cohort pairs (of {n_tested} tested)")
    for line in top5_lines:
        print(line)
    print(f"\nBehavioral metric differences:")
    print(f"  fear_AUC:   {fear_auc_line}")
    print(f"  disc_ratio: {disc_ratio_line}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _save(df: pd.DataFrame, out_dir: str, fname: str, dry: bool) -> None:
    path = os.path.join(out_dir, fname)
    if dry:
        print(f"[dry-run] Would save: {path}")
    else:
        df.to_csv(path, index=False, encoding="utf-8")
        print(f"  Saved: {path}")


def _savefig(fig: plt.Figure, out_dir: str, fname: str) -> None:
    path = os.path.join(out_dir, fname)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _draw_grouped_bars(
    ax,
    state_ids: list[int],
    cohort_groups: list[str],
    df_profiles: pd.DataFrame,
    cohort_state_means: dict,
    colors: np.ndarray,
    title: str,
) -> None:
    n_states = len(state_ids)
    n_cohorts = len(cohort_groups)
    bar_width = 0.8 / n_cohorts
    x = np.arange(n_states)

    for i, group in enumerate(cohort_groups):
        gp = df_profiles[df_profiles["cohort_group"] == group].set_index("state_id")
        means = [cohort_state_means.get((group, s), 0.0) for s in state_ids]
        sems  = []
        for s in state_ids:
            if s in gp.index:
                v = gp.loc[s, "se_fraction"]
                sems.append(float(v) if not np.isnan(v) else 0.0)
            else:
                sems.append(0.0)
        offset = (i - n_cohorts / 2 + 0.5) * bar_width
        ax.bar(x + offset, means, width=bar_width, label=group,
               color=colors[i], alpha=0.85, yerr=sems,
               capsize=3, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in state_ids], rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("State ID")
    ax.set_ylabel("Mean fraction")
    ax.set_title(title)
    ax.legend(title="Cohort", fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="VIEB — Cohort-level behavioral state analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cohort", required=True,
                        help="Path to cohort_normalized.csv or .xlsx")
    parser.add_argument("--summary", default="results/comparison/summary_table.csv",
                        help="Path to summary_table.csv (default: results/comparison/summary_table.csv)")
    parser.add_argument("--scalars", default="results/comparison/animal_scalars.csv",
                        help="Path to animal_scalars.csv (default: results/comparison/animal_scalars.csv)")
    parser.add_argument("--state-summary", dest="state_summary",
                        default="results/characterization/state_summary.csv",
                        help="Path to state_summary.csv")
    parser.add_argument("--output", default="results/cohort/",
                        help="Output directory (default: results/cohort/)")
    parser.add_argument("--groupby", default="age_treatment",
                        choices=sorted(GROUPBY_OPTIONS),
                        help="How to define cohort groups (default: age_treatment)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be computed without writing any files")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
