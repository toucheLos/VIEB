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
# Contrast vector helpers
# ---------------------------------------------------------------------------

_QUANT = os.path.join(_ROOT, "results", "quantification")
_COHORT_COLORS = ["#4E79A7", "#E07B39", "#59A14F", "#B07AA1"]


def _load_contrast(contrast_csv: str) -> pd.DataFrame:
    import json as _json
    p = contrast_csv if os.path.isabs(contrast_csv) else os.path.join(_ROOT, contrast_csv)
    if not os.path.exists(p):
        sys.exit(f"[ERROR] {p} not found.\nRun: python quantify.py --contrast first")
    df = pd.read_csv(p)
    df["animal_id"] = df["animal_id"].astype(str)
    return df


def _parse_json_col(series: "pd.Series") -> np.ndarray:
    import json as _json
    rows = []
    for val in series:
        try:
            v = np.array(_json.loads(val), dtype=float)
        except Exception:
            v = np.array([])
        rows.append(v)
    # Pad to uniform length
    max_len = max((len(r) for r in rows), default=0)
    out = np.full((len(rows), max_len), np.nan)
    for i, r in enumerate(rows):
        out[i, :len(r)] = r
    return out


def _cohort_color_map(labels):
    unique = sorted(set(str(l) for l in labels))
    return {l: _COHORT_COLORS[i % len(_COHORT_COLORS)] for i, l in enumerate(unique)}


def _ax_clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, color="#e0e0e0", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Contrast Figure 1 — Diverging bar chart per cohort
# ---------------------------------------------------------------------------

def _session_bootstrap_cis(
    summary: "pd.DataFrame",
    cohort_label: str,
    cohort_label_col: str,
    sc: list,
    ctx_A_vals: list,
    ctx_B_vals: list,
    ctx_col: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> tuple:
    """Bootstrap 95% CI for a per-condition contrast vector by resampling sessions."""
    rng = np.random.default_rng(seed)
    grp    = summary[summary[cohort_label_col] == cohort_label]
    a_mat  = grp[grp[ctx_col].isin(ctx_A_vals)][sc].values.astype(float)
    b_mat  = grp[grp[ctx_col].isin(ctx_B_vals)][sc].values.astype(float)
    if len(a_mat) == 0 or len(b_mat) == 0:
        return np.full(len(sc), np.nan), np.full(len(sc), np.nan)
    boot = []
    for _ in range(n_bootstrap):
        ai = rng.integers(0, len(a_mat), len(a_mat))
        bi = rng.integers(0, len(b_mat), len(b_mat))
        boot.append(a_mat[ai].mean(axis=0) - b_mat[bi].mean(axis=0))
    boot_mat = np.array(boot)
    return np.percentile(boot_mat, 2.5, axis=0), np.percentile(boot_mat, 97.5, axis=0)


def plot_contrast_bars(
    condition_contrast_csv: str,
    summary_csv: str,
    cohort_csv: str = None,
    state_summary_csv: str = None,
    output_path: str = "results/quantification/contrast_bars.png",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> None:
    """
    Diverging bar chart of per-condition contrast vector per cohort.
    Primary input is condition_contrast.csv (pooled-session vectors).
    Error bars = bootstrap 95% CI computed by resampling sessions.
    """
    import json as _json

    p = condition_contrast_csv if os.path.isabs(condition_contrast_csv) \
        else os.path.join(_ROOT, condition_contrast_csv)
    if not os.path.exists(p):
        sys.exit(f"[ERROR] {p} not found.\nRun: python quantify.py --contrast first")
    cond_df = pd.read_csv(p)

    summary = pd.read_csv(
        summary_csv if os.path.isabs(summary_csv) else os.path.join(_ROOT, summary_csv)
    )
    summary["animal_id"] = summary["animal_id"].astype(str)

    # Attach cohort labels to summary for session bootstrap
    cohort_label_col = "cohort_label"
    if cohort_csv:
        cohort_raw = pd.read_csv(
            cohort_csv if os.path.isabs(cohort_csv) else os.path.join(_ROOT, cohort_csv)
        )
        cohort_raw["animal_id"] = cohort_raw["animal_id"].astype(str)
        if cohort_label_col in cohort_raw.columns:
            summary = summary.merge(
                cohort_raw[["animal_id", cohort_label_col]].drop_duplicates("animal_id"),
                on="animal_id", how="left",
            )
            summary[cohort_label_col] = summary[cohort_label_col].fillna("Unknown")

    ctx_col = next((c for c in ("context", "Context", "ctx") if c in summary.columns), None)
    if ctx_col:
        ctx_vals   = summary[ctx_col].dropna().astype(str)
        ctx_A_vals = [v for v in ctx_vals.unique() if v.upper().startswith("A")]
        ctx_B_vals = [v for v in ctx_vals.unique() if v.upper().startswith("B")]
    else:
        ctx_A_vals = ctx_B_vals = []

    groups = (list(cond_df["cohort_label"]) if "cohort_label" in cond_df.columns
              else ["All"])
    n_cohorts = len(groups)
    color_map = _cohort_color_map(groups)

    # State labels
    state_labels = {}
    if state_summary_csv and os.path.exists(state_summary_csv):
        ss = pd.read_csv(state_summary_csv)
        if "state" in ss.columns and "heuristic_label" in ss.columns:
            state_labels = dict(zip(ss["state"].astype(int), ss["heuristic_label"].astype(str)))

    # Collect all sc columns from cond_df json to know n_states
    sample_vec = _parse_json_col(cond_df["contrast_vector_json"])
    n_states = sample_vec.shape[1]
    state_ids = list(range(n_states))

    fig, axes = plt.subplots(1, n_cohorts, figsize=(5 * n_cohorts, 5), sharey=True)
    if n_cohorts == 1:
        axes = [axes]

    for ax, label in zip(axes, groups):
        row = cond_df[cond_df["cohort_label"] == label] if "cohort_label" in cond_df.columns \
              else cond_df.iloc[:1]
        if row.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        contrast_vec = np.array(_json.loads(row.iloc[0]["contrast_vector_json"]), dtype=float)
        xs = np.arange(len(contrast_vec))

        # Bootstrap CI via session resampling
        if ctx_col and cohort_label_col in summary.columns:
            sc = [f"state_{i}_frac" for i in state_ids
                  if f"state_{i}_frac" in summary.columns]
            if sc:
                ci_lo, ci_hi = _session_bootstrap_cis(
                    summary, label, cohort_label_col, sc,
                    ctx_A_vals, ctx_B_vals, ctx_col,
                    n_bootstrap=n_bootstrap, seed=seed,
                )
                err_lo = np.abs(contrast_vec - ci_lo)
                err_hi = np.abs(ci_hi - contrast_vec)
                yerr   = [err_lo, err_hi]
            else:
                yerr = None
        else:
            yerr = None

        bar_colors = [("#C0392B" if v >= 0 else "#2980B9") for v in contrast_vec]
        ax.bar(xs, contrast_vec, color=bar_colors,
               yerr=yerr, capsize=3,
               error_kw={"elinewidth": 1, "ecolor": "#555"}, zorder=3)
        ax.axhline(0, color="#888", linewidth=0.8, zorder=2)

        if state_labels and len(contrast_vec) > 0:
            top_pos_idx = int(np.nanargmax(contrast_vec))
            top_neg_idx = int(np.nanargmin(contrast_vec))
            for idx in {top_pos_idx, top_neg_idx}:
                sid = state_ids[idx] if idx < len(state_ids) else idx
                lbl = state_labels.get(sid, "")
                if lbl:
                    ax.text(xs[idx], contrast_vec[idx], lbl[:12],
                            ha="center",
                            va="bottom" if contrast_vec[idx] >= 0 else "top",
                            fontsize=7, color="#666")

        n_animals = int(row.iloc[0].get("n_animals", "?"))
        ax.set_title(f"{label}\n(n={n_animals})", fontsize=9)
        ax.set_xlabel("State ID", fontsize=8)
        _ax_clean(ax)

    axes[0].set_ylabel("p_A − p_B", fontsize=9)
    fig.suptitle("Behavioral Context Contrast by Cohort", fontsize=12, y=1.02)
    fig.text(0.5, -0.03,
             "Positive bars = states more active during fear (Context A).\n"
             "Negative bars = states more active during safety (Context B).\n"
             "Error bars = 95% bootstrap CI (session resampling).",
             ha="center", fontsize=8, color="#555", style="italic")

    out_full = output_path if os.path.isabs(output_path) else os.path.join(_ROOT, output_path)
    os.makedirs(os.path.dirname(out_full), exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_full, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_full}")


# ---------------------------------------------------------------------------
# Contrast Figure 1b — Overlay of all cohort contrast vectors (paper figure)
# ---------------------------------------------------------------------------

def plot_contrast_overlay(
    condition_contrast_csv: str,
    summary_csv: str,
    cohort_csv: str = None,
    state_summary_csv: str = None,
    output_path: str = "results/quantification/contrast_overlay.png",
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> None:
    """
    Overlay all cohort contrast vectors on one axes.
    One colored line per cohort with shaded 95% CI band (session bootstrap).
    Primary paper figure for the contrast vector analysis.
    """
    import json as _json

    p = condition_contrast_csv if os.path.isabs(condition_contrast_csv) \
        else os.path.join(_ROOT, condition_contrast_csv)
    if not os.path.exists(p):
        sys.exit(f"[ERROR] {p} not found.\nRun: python quantify.py --contrast first")
    cond_df = pd.read_csv(p)

    summary = pd.read_csv(
        summary_csv if os.path.isabs(summary_csv) else os.path.join(_ROOT, summary_csv)
    )
    summary["animal_id"] = summary["animal_id"].astype(str)

    cohort_label_col = "cohort_label"
    if cohort_csv:
        cohort_raw = pd.read_csv(
            cohort_csv if os.path.isabs(cohort_csv) else os.path.join(_ROOT, cohort_csv)
        )
        cohort_raw["animal_id"] = cohort_raw["animal_id"].astype(str)
        if cohort_label_col in cohort_raw.columns:
            summary = summary.merge(
                cohort_raw[["animal_id", cohort_label_col]].drop_duplicates("animal_id"),
                on="animal_id", how="left",
            )
            summary[cohort_label_col] = summary[cohort_label_col].fillna("Unknown")

    ctx_col = next((c for c in ("context", "Context", "ctx") if c in summary.columns), None)
    if ctx_col:
        ctx_vals   = summary[ctx_col].dropna().astype(str)
        ctx_A_vals = [v for v in ctx_vals.unique() if v.upper().startswith("A")]
        ctx_B_vals = [v for v in ctx_vals.unique() if v.upper().startswith("B")]
    else:
        ctx_A_vals = ctx_B_vals = []

    groups    = (list(cond_df["cohort_label"]) if "cohort_label" in cond_df.columns
                 else ["All"])
    color_map = _cohort_color_map(groups)

    state_labels = {}
    if state_summary_csv and os.path.exists(state_summary_csv):
        ss = pd.read_csv(state_summary_csv)
        if "state" in ss.columns and "heuristic_label" in ss.columns:
            state_labels = dict(zip(ss["state"].astype(int), ss["heuristic_label"].astype(str)))

    sample_mat = _parse_json_col(cond_df["contrast_vector_json"])
    n_states   = sample_mat.shape[1]
    state_ids  = list(range(n_states))
    xs         = np.arange(n_states)

    fig, ax = plt.subplots(figsize=(max(10, n_states * 0.35), 5))

    for label in groups:
        row = cond_df[cond_df["cohort_label"] == label] if "cohort_label" in cond_df.columns \
              else cond_df.iloc[:1]
        if row.empty:
            continue
        color        = color_map.get(str(label), "#888")
        contrast_vec = np.array(_json.loads(row.iloc[0]["contrast_vector_json"]), dtype=float)

        ci_lo = ci_hi = None
        if ctx_col and cohort_label_col in summary.columns:
            sc = [f"state_{i}_frac" for i in state_ids
                  if f"state_{i}_frac" in summary.columns]
            if sc:
                ci_lo, ci_hi = _session_bootstrap_cis(
                    summary, label, cohort_label_col, sc,
                    ctx_A_vals, ctx_B_vals, ctx_col,
                    n_bootstrap=n_bootstrap, seed=seed,
                )

        n_animals = int(row.iloc[0].get("n_animals", "?"))
        ax.plot(xs, contrast_vec, color=color, linewidth=2,
                label=f"{label} (n={n_animals})", zorder=4)
        if ci_lo is not None and ci_hi is not None:
            ax.fill_between(xs, ci_lo, ci_hi, color=color, alpha=0.15, zorder=3)

    ax.axhline(0, color="#888", linewidth=0.8, linestyle="--", zorder=2)

    # X-axis state labels if available
    if state_labels:
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [state_labels.get(i, str(i)) for i in state_ids],
            rotation=45, ha="right", fontsize=7,
        )
    else:
        ax.set_xticks(xs)
        ax.set_xticklabels([str(i) for i in state_ids], fontsize=8)

    ax.set_xlabel("State ID", fontsize=9)
    ax.set_ylabel("p_A − p_B", fontsize=9)
    ax.legend(fontsize=8, framealpha=0.9)
    _ax_clean(ax)

    ax.set_title("Fear vs Safety Context Contrast by Cohort", fontsize=12)
    fig.text(0.5, -0.03,
             "Lines = per-condition contrast vector (pooled sessions). "
             "Shaded area = 95% bootstrap CI (session resampling).",
             ha="center", fontsize=8, color="#555", style="italic")

    out_full = output_path if os.path.isabs(output_path) else os.path.join(_ROOT, output_path)
    os.makedirs(os.path.dirname(out_full), exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_full, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_full}")


# ---------------------------------------------------------------------------
# Contrast Figure 2 — Heatmap of individual animal contrast vectors
# ---------------------------------------------------------------------------

def plot_contrast_heatmap(
    contrast_csv: str,
    output_path: str = "results/quantification/contrast_heatmap.png",
) -> None:
    df = _load_contrast(contrast_csv)

    cohort_col = "cohort_label" if "cohort_label" in df.columns else None
    if cohort_col is None:
        df["cohort_label"] = "All"
        cohort_col = "cohort_label"

    df = df.sort_values(
        [cohort_col, "contrast_magnitude"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)

    mat = _parse_json_col(df["contrast_vector_json"])  # (n_animals, n_states)
    n_animals, n_states = mat.shape

    cohort_labels = df[cohort_col].tolist()
    color_map = _cohort_color_map(cohort_labels)
    magnitudes = df["contrast_magnitude"].fillna(0).values

    vmax = float(np.nanmax(np.abs(mat))) if np.any(~np.isnan(mat)) else 1.0
    vmax = max(vmax, 1e-6)

    fig_w = max(10, n_states * 0.3)
    fig_h = max(8, n_animals * 0.4)

    fig = plt.figure(figsize=(fig_w + 2.5, fig_h))
    # axes layout: [cohort strip] [magnitude strip] [main heatmap] [colorbar]
    left_strip_w = 0.015
    mag_strip_w  = 0.04
    gap          = 0.005
    main_left    = 0.08 + left_strip_w + mag_strip_w + 2 * gap
    main_width   = 0.72
    bottom       = 0.12
    height       = 0.78

    ax_strip  = fig.add_axes([0.04, bottom, left_strip_w, height])
    ax_mag    = fig.add_axes([0.04 + left_strip_w + gap, bottom, mag_strip_w, height])
    ax_main   = fig.add_axes([main_left, bottom, main_width, height])
    ax_cbar   = fig.add_axes([main_left + main_width + 0.01, bottom, 0.015, height])

    # Cohort color strip
    strip_colors = np.array([[mcolors.to_rgb(color_map[str(l)])] for l in cohort_labels])
    ax_strip.imshow(strip_colors, aspect="auto", interpolation="nearest")
    ax_strip.set_xticks([]); ax_strip.set_yticks([])

    # Magnitude strip (horizontal bars as image column)
    mag_img = magnitudes[:, np.newaxis]
    unique_cohorts = list(dict.fromkeys(str(l) for l in cohort_labels))
    mag_colors = np.zeros((n_animals, 1, 4))
    for i, l in enumerate(cohort_labels):
        r, g, b = mcolors.to_rgb(color_map[str(l)])
        mag_colors[i, 0] = (r, g, b, 0.6)
    # Draw magnitude as a simple bar by scaling x
    ax_mag.imshow(mag_img, aspect="auto", interpolation="nearest",
                  cmap="Greys", vmin=0, vmax=1)
    ax_mag.set_xticks([]); ax_mag.set_yticks([])

    # Main heatmap
    im = ax_main.imshow(mat, aspect="auto", cmap="RdBu_r",
                        vmin=-vmax, vmax=vmax, interpolation="nearest")

    # White lines between cohort groups
    prev = cohort_labels[0]
    for i, l in enumerate(cohort_labels):
        if l != prev:
            ax_main.axhline(i - 0.5, color="white", linewidth=1.5)
        prev = l

    ax_main.set_yticks(range(n_animals))
    ax_main.set_yticklabels(df["animal_id"].tolist(), fontsize=6)
    x_ticks = range(n_states)
    ax_main.set_xticks(list(x_ticks))
    x_labels = [str(i) for i in range(n_states)]
    rotation = 45 if n_states > 20 else 0
    ax_main.set_xticklabels(x_labels, fontsize=7, rotation=rotation)
    ax_main.set_xlabel("State ID", fontsize=8)

    fig.colorbar(im, cax=ax_cbar, label="p_A − p_B")
    ax_cbar.tick_params(labelsize=7)

    fig.suptitle("Per-Animal Behavioral Contrast Vectors", fontsize=12, y=0.99)
    fig.text(0.5, 0.01,
             "Rows sorted by cohort then by contrast magnitude. "
             "Red = more active in fear context.",
             ha="center", fontsize=8, color="#555")

    out_full = output_path if os.path.isabs(output_path) else os.path.join(_ROOT, output_path)
    os.makedirs(os.path.dirname(out_full), exist_ok=True)
    fig.savefig(out_full, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_full}")


# ---------------------------------------------------------------------------
# Contrast Figure 3 — Contrast magnitude per cohort (bar + dots)
# ---------------------------------------------------------------------------

def plot_contrast_magnitude(
    contrast_csv: str,
    cohort_contrast_csv: str,
    stats_csv: str = None,
    output_path: str = "results/quantification/contrast_magnitude.png",
) -> None:
    import json as _json

    df = _load_contrast(contrast_csv)

    p = cohort_contrast_csv if os.path.isabs(cohort_contrast_csv) else os.path.join(_ROOT, cohort_contrast_csv)
    if not os.path.exists(p):
        sys.exit(f"[ERROR] {p} not found.")
    cdf = pd.read_csv(p)

    cohort_col = "cohort_label" if "cohort_label" in df.columns else None
    if cohort_col is None:
        df["cohort_label"] = "All"
        cohort_col = "cohort_label"

    groups = list(cdf["cohort_label"]) if "cohort_label" in cdf.columns else ["All"]
    n_cohorts = len(groups)
    color_map = _cohort_color_map(groups)

    fig, ax = plt.subplots(figsize=(max(6, n_cohorts * 2), 5))
    rng = np.random.default_rng(42)

    bar_xs = np.arange(n_cohorts)
    for xi, label in enumerate(groups):
        row = cdf[cdf["cohort_label"] == label].iloc[0] if not cdf[cdf["cohort_label"] == label].empty else None
        color = color_map.get(str(label), "#888888")

        if row is not None:
            mean_mag = float(row["mean_magnitude"])
            ci_lo    = float(row["ci_lo"])
            ci_hi    = float(row["ci_hi"])
            ax.bar(xi, mean_mag, color=color, alpha=0.8, zorder=3)
            ax.errorbar(xi, mean_mag,
                        yerr=[[mean_mag - ci_lo], [ci_hi - mean_mag]],
                        fmt="none", color="#333", capsize=5, linewidth=1.5, zorder=4)

        # Individual dots
        sub = df[df[cohort_col] == label]["contrast_magnitude"].dropna().values
        jitter = rng.uniform(-0.1, 0.1, size=len(sub))
        ax.scatter(xi + jitter, sub, color=color, s=36, zorder=5,
                   edgecolors="#333", linewidths=0.4)

    ax.axhline(0, color="#aaa", linestyle="--", linewidth=0.5, label="no discrimination")
    ax.set_xticks(bar_xs)
    labels_text = [str(g) for g in groups]
    rotation = 30 if max(len(g) for g in labels_text) > 10 else 0
    ax.set_xticklabels(labels_text, rotation=rotation, ha="right" if rotation else "center", fontsize=9)
    ax.set_ylabel("Contrast Magnitude (||p_A−p_B||₂ / √2)", fontsize=9)
    ax.set_xlabel("Cohort", fontsize=9)
    ax.set_title("Behavioral Context Discrimination by Cohort", fontsize=11)
    _ax_clean(ax)

    # Significance brackets
    if stats_csv and os.path.exists(stats_csv):
        sdf = pd.read_csv(stats_csv)
        sig = sdf[sdf["significant"] == True] if "significant" in sdf.columns else pd.DataFrame()
        if not sig.empty:
            y_top = ax.get_ylim()[1]
            bracket_step = (y_top - ax.get_ylim()[0]) * 0.06
            for _, srow in sig.iterrows():
                try:
                    xi1 = groups.index(srow["cohort_A"])
                    xi2 = groups.index(srow["cohort_B"])
                except ValueError:
                    continue
                p_fdr = float(srow["p_fdr"])
                stars = "**" if p_fdr < 0.01 else "*"
                bh = y_top + bracket_step
                ax.plot([xi1, xi1, xi2, xi2], [bh - bracket_step * 0.3, bh, bh, bh - bracket_step * 0.3],
                        color="#333", linewidth=1.0)
                ax.text((xi1 + xi2) / 2, bh + bracket_step * 0.1, stars,
                        ha="center", va="bottom", fontsize=10)
                y_top = bh + bracket_step

    fig.text(0.5, -0.04,
             "Higher = greater behavioral change between fear and safety contexts. "
             "Points = individual animals.",
             ha="center", fontsize=8, color="#555", style="italic")

    out_full = output_path if os.path.isabs(output_path) else os.path.join(_ROOT, output_path)
    os.makedirs(os.path.dirname(out_full), exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_full, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_full}")


# ---------------------------------------------------------------------------
# Contrast Figure 4 — Contrast magnitude × Jess protein scatter
# ---------------------------------------------------------------------------

def plot_contrast_scatter(
    contrast_csv: str,
    jess_csv: str,
    output_path: str = "results/quantification/contrast_scatter.png",
) -> None:
    if not os.path.exists(jess_csv):
        print("Jess data not yet available — skipping contrast scatter plot.")
        return

    from scipy.stats import linregress

    df = _load_contrast(contrast_csv)
    jess = pd.read_csv(jess_csv)
    jess["animal_id"] = jess["animal_id"].astype(str)
    merged = df.merge(jess, on="animal_id", how="inner")

    proteins = [c for c in jess.columns if c != "animal_id"]
    proteins = [p for p in proteins
                if merged[["contrast_magnitude", p]].dropna().shape[0] >= 5]
    if not proteins:
        print("[WARN] Fewer than 5 animals with both contrast_magnitude and protein data — skipping scatter.")
        return

    cohort_col = "cohort_label" if "cohort_label" in merged.columns else None
    if cohort_col:
        color_map = _cohort_color_map(merged[cohort_col].fillna("Unknown"))
    else:
        color_map = {"all": "#4E79A7"}

    n_proteins = len(proteins)
    fig, axes = plt.subplots(1, n_proteins, figsize=(5 * n_proteins, 5), squeeze=False)
    axes = axes[0]

    for ax, prot in zip(axes, proteins):
        sub = merged[["animal_id", "contrast_magnitude", prot] +
                     ([cohort_col] if cohort_col else [])].dropna()
        if len(sub) < 5:
            print(f"  [WARN] Only {len(sub)} animals for {prot} — skipping")
            continue

        x = sub["contrast_magnitude"].values
        y = sub[prot].values

        # Scatter
        if cohort_col:
            for _, row in sub.iterrows():
                c = color_map.get(str(row[cohort_col]), "#888")
                ax.scatter(row["contrast_magnitude"], row[prot], color=c, s=48,
                           edgecolors="#333", linewidths=0.4, zorder=4)
        else:
            ax.scatter(x, y, color="#4E79A7", s=48, edgecolors="#333", linewidths=0.4, zorder=4)

        # Regression line + CI
        slope, intercept, r, p_val, _ = linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="#888", linewidth=1.5, zorder=3)

        # 95% CI via bootstrap
        rng = np.random.default_rng(42)
        boot_lines = []
        for _ in range(500):
            idx = rng.integers(0, len(x), len(x))
            s2, i2, *_ = linregress(x[idx], y[idx])
            boot_lines.append(s2 * x_line + i2)
        ci_lo = np.percentile(boot_lines, 2.5, axis=0)
        ci_hi = np.percentile(boot_lines, 97.5, axis=0)
        ax.fill_between(x_line, ci_lo, ci_hi, alpha=0.15, color="#888")

        # Annotation
        p_str = f"{p_val:.3f}" if p_val >= 0.001 else f"{p_val:.2e}"
        xpos = 0.97 if r >= 0 else 0.03
        ha   = "right" if r >= 0 else "left"
        ax.text(xpos, 0.97, f"r = {r:.2f}, p = {p_str}",
                transform=ax.transAxes, ha=ha, va="top", fontsize=9, color="#555")

        ax.set_xlabel("Contrast Magnitude", fontsize=9)
        ax.set_ylabel(prot, fontsize=9)
        ax.set_title(prot, fontsize=10)
        _ax_clean(ax)

    fig.suptitle("Contrast Magnitude × Protein Expression", fontsize=12)
    fig.text(0.5, -0.03,
             "Each point = one animal. Line = linear regression. Shaded area = 95% CI.",
             ha="center", fontsize=8, color="#555", style="italic")

    out_full = output_path if os.path.isabs(output_path) else os.path.join(_ROOT, output_path)
    os.makedirs(os.path.dirname(out_full), exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_full, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_full}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cohort analysis figures for VIEB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--all",       action="store_true", help="Generate all figures")
    parser.add_argument("--heatmap",   action="store_true", help="Figure 1: fingerprint heatmap")
    parser.add_argument("--umap",      action="store_true", help="Figure 2: animal UMAP")
    parser.add_argument("--profiles",  action="store_true", help="Figure 3: cohort state profiles")
    parser.add_argument("--deviation", action="store_true", help="Figure 4: deviation distributions")
    parser.add_argument("--importance",action="store_true", help="Figure 5: feature importance")
    parser.add_argument("--contrast",  action="store_true", help="Contrast vector figures (bars, heatmap, magnitude, scatter)")
    parser.add_argument("--cohort",    type=str, default=None, metavar="FILE")
    parser.add_argument("--color-by",  type=str, default="cohort_label",
                        help="Row annotation column for heatmap (default: cohort_label)")
    parser.add_argument("--jess",      type=str, default=None, metavar="FILE",
                        help="Jess protein CSV for contrast scatter (optional)")
    args = parser.parse_args()

    do_hm  = args.heatmap   or args.all
    do_um  = args.umap      or args.all
    do_pr  = args.profiles  or args.all
    do_dv  = args.deviation or args.all
    do_im  = args.importance or args.all
    do_ct  = args.contrast  or args.all

    if not any([do_hm, do_um, do_pr, do_dv, do_im, do_ct]):
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

    if do_ct:
        condition_ct_csv  = os.path.join(_QUANT, "condition_contrast.csv")
        contrast_csv      = os.path.join(_QUANT, "contrast_vectors.csv")
        cohort_ct_csv     = os.path.join(_QUANT, "cohort_contrast_vectors.csv")
        stats_ct_csv      = os.path.join(_QUANT, "cohort_contrast_stats.csv")
        state_summary_csv = os.path.join(_CHAR,  "state_summary.csv")
        summary_csv       = os.path.join(_COMP,  "summary_table.csv")
        cohort_arg        = args.cohort

        if not os.path.exists(condition_ct_csv):
            print(f"[ERROR] {condition_ct_csv} not found.\nRun: python quantify.py --contrast first")
        else:
            # Primary bar chart — per-condition vectors, session-level bootstrap CI
            plot_contrast_bars(
                condition_contrast_csv=condition_ct_csv,
                summary_csv=summary_csv,
                cohort_csv=cohort_arg,
                state_summary_csv=state_summary_csv if os.path.exists(state_summary_csv) else None,
                output_path=os.path.join(_QUANT, "contrast_bars.png"),
            )
            # Overlay — all cohorts on one axes (paper figure)
            plot_contrast_overlay(
                condition_contrast_csv=condition_ct_csv,
                summary_csv=summary_csv,
                cohort_csv=cohort_arg,
                state_summary_csv=state_summary_csv if os.path.exists(state_summary_csv) else None,
                output_path=os.path.join(_QUANT, "contrast_overlay.png"),
            )
            if os.path.exists(contrast_csv):
                plot_contrast_heatmap(
                    contrast_csv=contrast_csv,
                    output_path=os.path.join(_QUANT, "contrast_heatmap.png"),
                )
                plot_contrast_magnitude(
                    contrast_csv=contrast_csv,
                    cohort_contrast_csv=cohort_ct_csv,
                    stats_csv=stats_ct_csv if os.path.exists(stats_ct_csv) else None,
                    output_path=os.path.join(_QUANT, "contrast_magnitude.png"),
                )
                jess_path = args.jess or os.path.join(_QUANT, "jess_correlations.csv")
                plot_contrast_scatter(
                    contrast_csv=contrast_csv,
                    jess_csv=jess_path,
                    output_path=os.path.join(_QUANT, "contrast_scatter.png"),
                )

    print("\nDone — figures saved to results/comparison/ and results/quantification/")


if __name__ == "__main__":
    main()
