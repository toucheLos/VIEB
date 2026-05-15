"""
plot_cohort.py — Paper-ready cohort analysis figures for VIEB
=============================================================

Generates 5 figures from behavioral_fingerprints.csv + cohort metadata.

Usage
-----
python plot_cohort.py --all [--cohort cohort_normalized.csv]
python plot_cohort.py --heatmap
python plot_cohort.py --umap
python plot_cohort.py --profiles
python plot_cohort.py --deviation
python plot_cohort.py --importance
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

_ROOT   = os.path.dirname(os.path.abspath(__file__))
_COMP   = os.path.join(_ROOT, "results", "comparison")
_CHAR   = os.path.join(_ROOT, "results", "characterization")
_SHARED = os.path.join(_ROOT, "results", "shared")
_DPI    = 150
_FIG_FMTS = (".png",)

# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_fingerprints() -> pd.DataFrame | None:
    p = os.path.join(_COMP, "behavioral_fingerprints.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, index_col="animal_id")
    df.index = df.index.astype(int)
    return df


def _load_deviation_scores() -> pd.DataFrame | None:
    p = os.path.join(_COMP, "deviation_scores.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def _load_reverse_results() -> dict | None:
    p = os.path.join(_COMP, "reverse_model_results.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _load_summary() -> pd.DataFrame | None:
    p = os.path.join(_COMP, "summary_table.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def _load_cluster_info() -> dict:
    p = os.path.join(_SHARED, "cluster_info.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_cohort(cohort_path: str | None) -> pd.DataFrame | None:
    if not cohort_path:
        norm = os.path.join(_ROOT, "cohort_normalized.csv")
        if os.path.exists(norm):
            cohort_path = norm
        else:
            cfg_p = os.path.join(_ROOT, "config.json")
            if os.path.exists(cfg_p):
                try:
                    cfg = json.load(open(cfg_p, encoding="utf-8"))
                    cohort_path = cfg.get("cohort_csv_path") or cfg.get("cohort_xlsx_path")
                except Exception:
                    pass
    if not cohort_path or not os.path.exists(cohort_path):
        return None
    ext = os.path.splitext(cohort_path)[1].lower()
    if ext in (".xlsx", ".xls"):
        from cohort_loader import load_cohort_excel
        return load_cohort_excel(cohort_path)
    return pd.read_csv(cohort_path)


from behavioral_fingerprint import _group_features, _sorted_feature_order


# ---------------------------------------------------------------------------
# Figure 1 — Behavioral fingerprint heatmap
# ---------------------------------------------------------------------------

def plot_fingerprint_heatmap(
    fp: pd.DataFrame,
    cohort_df: pd.DataFrame | None,
    dev_scores: pd.DataFrame | None,
    save_path: str,
    color_by: str = "cohort_label",
) -> None:
    """
    Rows: animals sorted by cohort_label then composite_z.
    Columns: behavioral features grouped by category.
    """
    feat_cols = _sorted_feature_order(list(fp.columns))
    feat_cols = [c for c in feat_cols if c in fp.columns]

    # Z-score columns for display
    fp_z = fp[feat_cols].copy()
    fp_z = (fp_z - fp_z.mean()) / fp_z.std().clip(lower=1e-10)
    fp_z = fp_z.clip(-3, 3)

    # Sort rows
    sort_df = pd.DataFrame({"animal_id": fp.index})
    if cohort_df is not None and "cohort_label" in cohort_df.columns:
        cmap = cohort_df.set_index("animal_id")["cohort_label"].to_dict()
        sort_df["cohort_label"] = sort_df["animal_id"].map(cmap).fillna("Unknown")
    else:
        sort_df["cohort_label"] = "Unknown"

    if dev_scores is not None and "composite_z" in dev_scores.columns:
        dev_map = dev_scores.set_index("animal_id")["composite_z"].to_dict()
        sort_df["composite_z"] = sort_df["animal_id"].map(dev_map).fillna(0.0)
    else:
        sort_df["composite_z"] = 0.0

    sort_df = sort_df.sort_values(["cohort_label", "composite_z"], ascending=[True, False])
    row_order = list(sort_df["animal_id"])
    fp_sorted = fp_z.loc[row_order]

    n_animals  = len(row_order)
    n_features = len(feat_cols)
    fig_w = max(14, n_features * 0.12)
    fig_h = max(6,  n_animals  * 0.35)

    fig = plt.figure(figsize=(fig_w + 2, fig_h))
    # Main heatmap axes + annotation axes
    ax_main  = fig.add_axes([0.12, 0.15, 0.72, 0.75])
    ax_color = fig.add_axes([0.02, 0.15, 0.04, 0.75])  # row annotations

    im = ax_main.imshow(
        fp_sorted.values, aspect="auto", cmap="RdBu_r",
        vmin=-3, vmax=3, interpolation="nearest",
    )
    ax_main.set_xticks([])
    ax_main.set_yticks(range(n_animals))
    ax_main.set_yticklabels([str(a) for a in row_order], fontsize=7)
    ax_main.set_xlabel("Behavioral features (grouped: state fracs A/B/Δ | bout dur | scalars | motifs)",
                        fontsize=8)

    # Group separators on x-axis
    grps  = _group_features(feat_cols)
    order = ["frac_A", "frac_B", "delta", "bout_dur", "scalar", "motif"]
    x     = 0
    for grp in order:
        n = len(grps[grp])
        if n > 0:
            ax_main.axvline(x - 0.5, color="#555", lw=0.6, alpha=0.5)
            ax_main.text(x + n / 2 - 0.5, -1.5, grp.replace("_", " "),
                         ha="center", va="top", fontsize=6, color="#444")
            x += n

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_main, fraction=0.015, pad=0.01)
    cbar.set_label("Z-score (clipped ±3)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    # Row annotation: cohort/sex/genotype color strip
    if cohort_df is not None and color_by in cohort_df.columns:
        ann_col = cohort_df.set_index("animal_id")[color_by].to_dict()
        vals    = [str(ann_col.get(a, "?")) for a in row_order]
        unique  = sorted(set(vals))
        palette = plt.cm.tab20(np.linspace(0, 1, max(1, len(unique))))
        val_to_color = {v: palette[i] for i, v in enumerate(unique)}
        ann_mat = np.array([[list(val_to_color[v])] for v in vals])
        ax_color.imshow(ann_mat, aspect="auto", interpolation="nearest")
        ax_color.set_xticks([])
        ax_color.set_yticks([])
        ax_color.set_title(color_by.replace("_", " "), fontsize=7, pad=2)
        # Legend
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, fc=val_to_color[v], label=v[:20])
            for v in unique
        ]
        ax_color.legend(handles=legend_elements, loc="upper left",
                        bbox_to_anchor=(1.0, 1.0), fontsize=6, framealpha=0.8)
    else:
        ax_color.axis("off")

    plt.suptitle("Behavioral Fingerprint Heatmap", fontsize=12, y=0.98)
    plt.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 2 — UMAP of animals in behavioral space
# ---------------------------------------------------------------------------

def plot_animal_umap(
    fp: pd.DataFrame,
    cohort_df: pd.DataFrame | None,
    dev_scores: pd.DataFrame | None,
    save_path: str,
) -> None:
    """Each point = one animal; color by cohort group; size ∝ composite_z."""
    try:
        import umap as umap_lib
    except ImportError:
        print("  [SKIP] umap-learn not installed — skipping animal_umap.png")
        return

    from sklearn.decomposition import PCA

    n_animals, n_feats = fp.shape
    if n_animals < 4:
        print(f"  [SKIP] animal_umap.png: only {n_animals} animals (need ≥ 4)")
        return

    X = fp.values.astype(np.float64)

    # PCA pre-reduction to avoid curse of dimensionality
    n_pca = min(n_animals - 1, 20, n_feats)
    if n_feats > n_pca:
        X = PCA(n_components=n_pca, random_state=42).fit_transform(X)

    # UMAP on (n_animals, n_pca)
    n_neighbors = max(2, min(n_animals - 1, 10))
    reducer = umap_lib.UMAP(
        n_components=2, n_neighbors=n_neighbors,
        min_dist=0.2, random_state=42,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb = reducer.fit_transform(X)   # (n_animals, 2)

    # Compose color dimensions
    color_dims = {}
    if cohort_df is not None:
        for col in ("genotype", "treatment", "sex", "age_group"):
            if col in cohort_df.columns:
                cmap = cohort_df.set_index("animal_id")[col].to_dict()
                color_dims[col] = [str(cmap.get(int(a), "?")) for a in fp.index]

    dev_map: dict[int, float] = {}
    if dev_scores is not None and "composite_z" in dev_scores.columns:
        dev_map = dict(zip(dev_scores["animal_id"].astype(int),
                           dev_scores["composite_z"].astype(float)))
    sizes = np.array([max(30, dev_map.get(int(a), 0) * 80 + 30) for a in fp.index])
    outlier_ids = [int(a) for a in fp.index if dev_map.get(int(a), 0) > 2.0]

    n_dims = len(color_dims) if color_dims else 1
    fig, axes = plt.subplots(1, n_dims, figsize=(5 * n_dims, 5))
    if n_dims == 1:
        axes = [axes]

    color_keys = list(color_dims.keys()) if color_dims else ["all"]
    for ax, key in zip(axes, color_keys):
        if key == "all":
            vals   = ["all"] * n_animals
            unique = ["all"]
        else:
            vals   = color_dims[key]
            unique = sorted(set(vals))

        palette = plt.cm.tab10(np.linspace(0, 1, max(1, len(unique))))
        val_to_c = {v: palette[i] for i, v in enumerate(unique)}
        colors   = [val_to_c[v] for v in vals]

        sc = ax.scatter(emb[:, 0], emb[:, 1],
                        c=colors, s=sizes, alpha=0.85, edgecolors="#333", linewidths=0.4)
        ax.set_title(f"Color: {key}", fontsize=10)
        ax.set_xlabel("UMAP 1", fontsize=9)
        ax.set_ylabel("UMAP 2", fontsize=9)
        ax.tick_params(labelsize=8)

        # Legend
        for v, c in val_to_c.items():
            ax.scatter([], [], c=[c], label=v[:20], s=50, alpha=0.85, edgecolors="#333")
        ax.legend(fontsize=7, framealpha=0.8, loc="best")

        # Annotate outliers
        for aid in outlier_ids:
            idx = list(fp.index).index(aid)
            ax.annotate(str(aid), xy=(emb[idx, 0], emb[idx, 1]),
                        fontsize=7, color="red",
                        xytext=(4, 4), textcoords="offset points")

    plt.suptitle("Animal Behavioral Space (UMAP)\n"
                 "Point size ∝ composite deviation score  ·  red labels = outliers (z>2)",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 3 — Cohort state profiles
# ---------------------------------------------------------------------------

def plot_cohort_state_profiles(
    summary: pd.DataFrame,
    cohort_df: pd.DataFrame,
    cluster_info: dict,
    save_path: str,
) -> None:
    """
    Grid: rows=genotype, cols=treatment.
    Bar chart of mean state occupancy ± SE per panel.
    """
    n_clusters = int(cluster_info.get("n_clusters", 0))
    if n_clusters == 0:
        print("  [SKIP] cohort_state_profiles.png: n_clusters = 0")
        return

    dom = cluster_info.get("dominant_state", None)
    state_cols = [f"state_{k}_frac" for k in range(n_clusters)
                  if k != dom and f"state_{k}_frac" in summary.columns]
    state_ids  = [int(c.split("_")[1]) for c in state_cols]
    if not state_cols:
        return

    cdf = cohort_df.copy()
    cdf["animal_id"] = cdf["animal_id"].astype(str)
    summary = summary.copy()
    summary["animal_id"] = summary["animal_id"].astype(str)
    merged = summary.merge(cdf[["animal_id", "genotype", "treatment"]], on="animal_id", how="left")
    merged[["genotype", "treatment"]] = merged[["genotype", "treatment"]].fillna("Unknown")

    genotypes  = sorted(merged["genotype"].unique())
    treatments = sorted(merged["treatment"].unique())
    n_g = len(genotypes)
    n_t = len(treatments)

    # Per-animal mean (collapse across sessions first)
    per_animal = merged.groupby(["animal_id", "genotype", "treatment"])[state_cols].mean().reset_index()

    colors = plt.cm.tab20(np.linspace(0, 1, max(1, len(state_ids))))

    # Determine shared y-axis max
    y_max = float(per_animal[state_cols].max().max()) * 1.15

    fig, axes = plt.subplots(
        n_g, n_t,
        figsize=(max(4, 2.5 * n_t), max(4, 3 * n_g)),
        sharey=True, sharex=True,
        squeeze=False,
    )

    for gi, geno in enumerate(genotypes):
        for ti, treat in enumerate(treatments):
            ax = axes[gi][ti]
            sub = per_animal[(per_animal["genotype"] == geno) & (per_animal["treatment"] == treat)]
            if len(sub) == 0:
                ax.text(0.5, 0.5, "n=0", ha="center", va="center", transform=ax.transAxes,
                        fontsize=8, color="#aaa")
                ax.axis("off")
                continue
            means = sub[state_cols].mean().values
            sems  = sub[state_cols].sem().values
            ax.bar(state_ids, means, color=colors[:len(state_ids)],
                   yerr=sems, capsize=3, error_kw={"elinewidth": 1, "ecolor": "#555"})
            ax.set_ylim(0, y_max)
            ax.tick_params(labelsize=7)
            n_g_group = len(sub)
            title = f"{geno}\n{treat} (N={n_g_group})"
            ax.set_title(title, fontsize=8)
            if gi == n_g - 1:
                ax.set_xlabel("State ID", fontsize=7)
            if ti == 0:
                ax.set_ylabel("Fraction", fontsize=7)

    plt.suptitle("Cohort State Profiles: Mean Occupancy ± SE by Genotype × Treatment",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 4 — Deviation score distribution
# ---------------------------------------------------------------------------

def plot_deviation_distributions(
    dev_scores: pd.DataFrame,
    save_path: str,
) -> None:
    """Boxplot per cohort_label + per-point overlay; red line at z=2."""
    if "composite_z" not in dev_scores.columns:
        print("  [SKIP] deviation_distributions.png: no composite_z column")
        return

    cohorts = sorted(dev_scores["cohort_label"].dropna().unique())
    if len(cohorts) == 0:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(cohorts) * 1.2), 5))
    data  = [dev_scores[dev_scores["cohort_label"] == c]["composite_z"].dropna().values
             for c in cohorts]

    bp = ax.boxplot(data, labels=[c[:25] for c in cohorts], patch_artist=True,
                    widths=0.5, showfliers=False)
    palette = plt.cm.tab20(np.linspace(0, 1, max(1, len(cohorts))))
    for patch, color in zip(bp["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    rng = np.random.default_rng(42)
    for xi, (d, c) in enumerate(zip(data, cohorts), start=1):
        jitter = rng.uniform(-0.15, 0.15, size=len(d))
        ax.scatter(np.full(len(d), xi) + jitter, d,
                   s=40, zorder=3, alpha=0.8, edgecolors="#333", linewidths=0.5,
                   color=palette[xi - 1])
        # Annotate outliers with animal_id
        for aid, z in zip(dev_scores[dev_scores["cohort_label"] == c]["animal_id"],
                          dev_scores[dev_scores["cohort_label"] == c]["composite_z"]):
            if z > 2.0:
                ax.annotate(str(int(aid)), xy=(xi, z), fontsize=7, color="red",
                            xytext=(3, 2), textcoords="offset points")

    ax.axhline(2.0, color="red", linewidth=1.2, linestyle="--", label="Outlier threshold (z=2)")
    ax.set_xlabel("Cohort", fontsize=10)
    ax.set_ylabel("Composite deviation score (z)", fontsize=10)
    ax.set_title("Behavioral Deviation from Cohort Norm", fontsize=12)
    ax.legend(fontsize=9)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# Figure 5 — Feature importance from reverse model
# ---------------------------------------------------------------------------

def plot_feature_importance(
    rev_results: dict,
    save_path: str,
) -> None:
    """Horizontal bar charts: top-10 features per target variable."""
    if not rev_results:
        print("  [SKIP] feature_importance.png: no reverse model results")
        return

    targets = list(rev_results.keys())
    n_t = len(targets)
    fig, axes = plt.subplots(1, n_t, figsize=(6 * n_t, 5))
    if n_t == 1:
        axes = [axes]

    for ax, tgt in zip(axes, targets):
        res  = rev_results[tgt]
        acc  = res.get("loo_accuracy", float("nan"))
        top10 = res.get("top10_features", [])
        if not top10:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        feats  = [d["feature"]    for d in top10]
        imps   = [d["importance"] for d in top10]
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(feats)))
        y_pos  = range(len(feats))

        ax.barh(list(y_pos), imps, color=colors[::-1])
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels([f[:35] for f in feats[::-1]], fontsize=8)
        ax.set_xlabel("Feature importance", fontsize=9)
        ax.set_title(f"-> {tgt}\nLOO accuracy: {acc:.2f}", fontsize=10)
        ax.invert_yaxis()

    plt.suptitle("Top-10 Behavioral Features for Cohort Prediction", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cohort analysis figures for VIEB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--all",       action="store_true", help="Generate all 5 figures")
    parser.add_argument("--heatmap",   action="store_true", help="Figure 1: fingerprint heatmap")
    parser.add_argument("--umap",      action="store_true", help="Figure 2: animal UMAP")
    parser.add_argument("--profiles",  action="store_true", help="Figure 3: cohort state profiles")
    parser.add_argument("--deviation", action="store_true", help="Figure 4: deviation distributions")
    parser.add_argument("--importance",action="store_true", help="Figure 5: feature importance")
    parser.add_argument("--cohort",    type=str, default=None, metavar="FILE")
    parser.add_argument("--color-by",  type=str, default="cohort_label",
                        help="Row annotation column for heatmap (default: cohort_label)")
    args = parser.parse_args()

    do_hm  = args.heatmap   or args.all
    do_um  = args.umap      or args.all
    do_pr  = args.profiles  or args.all
    do_dv  = args.deviation or args.all
    do_im  = args.importance or args.all

    if not any([do_hm, do_um, do_pr, do_dv, do_im]):
        parser.print_help()
        sys.exit(1)

    os.makedirs(_COMP, exist_ok=True)

    fp       = _load_fingerprints()
    cohort   = _load_cohort(args.cohort)
    dev      = _load_deviation_scores()
    rev      = _load_reverse_results()
    summary  = _load_summary()
    ci       = _load_cluster_info()

    if fp is None:
        sys.exit("[ERROR] behavioral_fingerprints.csv not found. "
                 "Run: python behavioral_fingerprint.py --fingerprints")

    if do_hm:
        plot_fingerprint_heatmap(
            fp, cohort, dev,
            os.path.join(_COMP, "fingerprint_heatmap.png"),
            color_by=args.color_by,
        )

    if do_um:
        plot_animal_umap(
            fp, cohort, dev,
            os.path.join(_COMP, "animal_umap.png"),
        )

    if do_pr:
        if cohort is None or summary is None:
            print("  [SKIP] cohort_state_profiles.png: cohort file or summary_table missing.")
        else:
            plot_cohort_state_profiles(
                summary, cohort, ci,
                os.path.join(_COMP, "cohort_state_profiles.png"),
            )

    if do_dv:
        if dev is None:
            print("  [SKIP] deviation_distributions.png: deviation_scores.csv missing. "
                  "Run --deviation first.")
        else:
            plot_deviation_distributions(dev, os.path.join(_COMP, "deviation_distributions.png"))

    if do_im:
        if rev is None:
            print("  [SKIP] feature_importance.png: reverse_model_results.json missing. "
                  "Run --reverse first.")
        else:
            plot_feature_importance(rev, os.path.join(_COMP, "feature_importance.png"))

    print("\nDone — figures saved to results/comparison/")


if __name__ == "__main__":
    main()
