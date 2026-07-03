"""Mode-driven report plots for VIEB."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def safe_get_state_columns(df: pd.DataFrame | None) -> list[str]:
    if df is None:
        return []
    cols = [c for c in df.columns if str(c).startswith("state_") and str(c).endswith("_frac")]

    def _key(col: str) -> int:
        try:
            return int(str(col).split("_")[1])
        except Exception:
            return 10**9

    return sorted(cols, key=_key)


def _skip(name: str, reason: str) -> None:
    print(f"  SKIP {name}: {reason}")


def _mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        return None


def _ensure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _state_ids(state_cols: list[str]) -> list[int]:
    return [int(c.split("_")[1]) for c in state_cols]


def _ordered_values(df: pd.DataFrame, col: str, order: list[Any] | None = None) -> list[Any]:
    values = [v for v in df[col].dropna().unique().tolist() if str(v).strip() != ""]
    if order:
        order_s = [str(v) for v in order]
        values_s = {str(v): v for v in values}
        ordered = [values_s[str(v)] for v in order if str(v) in values_s]
        rest = [v for v in values if str(v) not in set(order_s)]
        return ordered + sorted(rest, key=lambda v: str(v))
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=lambda v: str(v))


def _transition_cols(df: pd.DataFrame | None) -> list[str]:
    if df is None:
        return []
    return [c for c in df.columns if str(c).startswith("trans_")]


def _transition_matrix_from_rows(df: pd.DataFrame, n_states: int) -> np.ndarray | None:
    if df.empty:
        return None
    cols = _transition_cols(df)
    if not cols:
        return None
    mat = np.zeros((n_states, n_states), dtype=float)
    found = False
    for i in range(n_states):
        for j in range(n_states):
            col = f"trans_{i}_{j}"
            if col in df.columns:
                mat[i, j] = pd.to_numeric(df[col], errors="coerce").mean()
                found = True
    return mat if found else None


def state_summary_plot(summary: pd.DataFrame, results_dir: Path) -> None:
    name = "state_summary_plot.png"
    plt = _mpl()
    state_cols = safe_get_state_columns(summary)
    if plt is None:
        return _skip(name, "matplotlib is not installed")
    if summary is None or summary.empty or not state_cols:
        return _skip(name, "summary table has no state fraction columns")
    means = summary[state_cols].apply(pd.to_numeric, errors="coerce").mean().fillna(0)
    ids = _state_ids(state_cols)
    fig, ax = plt.subplots(figsize=(7, max(3.5, 0.24 * len(ids) + 1.4)))
    ax.barh(range(len(ids)), means.values * 100, color=plt.cm.tab20(np.linspace(0, 1, max(1, len(ids)))))
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels([f"S{s}" for s in ids], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean session occupancy (%)")
    ax.set_title("State Summary")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out = results_dir / "comparison" / name
    _ensure(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def umap_by_state(results_dir: Path) -> None:
    name = "umap_by_state.png"
    plt = _mpl()
    path = results_dir / "diagnostics" / "umap_sample.csv"
    if plt is None:
        return _skip(name, "matplotlib is not installed")
    if not path.exists():
        return _skip(name, "diagnostics/umap_sample.csv not found")
    df = pd.read_csv(path)
    if df.empty or not {"umap_1", "umap_2", "label"}.issubset(df.columns):
        return _skip(name, "UMAP sample is empty or missing columns")
    fig, ax = plt.subplots(figsize=(7, 6))
    valid = df[df["label"] >= 0]
    noise = df[df["label"] < 0]
    if not noise.empty:
        ax.scatter(noise["umap_1"], noise["umap_2"], c="#CCCCCC", s=1, alpha=0.25, rasterized=True)
    if not valid.empty:
        sc = ax.scatter(valid["umap_1"], valid["umap_2"], c=valid["label"], cmap="tab20", s=1, alpha=0.55, rasterized=True)
        fig.colorbar(sc, ax=ax, label="State")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("UMAP by State")
    fig.tight_layout()
    out = results_dir / "comparison" / name
    _ensure(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def umap_by_time_or_condition(summary: pd.DataFrame, design: dict, results_dir: Path) -> None:
    name = "umap_by_time_or_condition.png"
    # Current umap_sample.csv is frame-level and has no stem/session key, so it
    # cannot be joined to metadata without adding a new sampling schema.
    return _skip(name, "UMAP sample does not include session metadata for time/condition coloring")


def state_transition_matrix(transition: pd.DataFrame | None, state_cols: list[str], results_dir: Path) -> None:
    name = "state_transition_matrix.png"
    plt = _mpl()
    if plt is None:
        return _skip(name, "matplotlib is not installed")
    n_states = len(state_cols)
    if transition is None or transition.empty or n_states == 0:
        return _skip(name, "transition table or state columns are missing")
    mat = _transition_matrix_from_rows(transition, n_states)
    if mat is None:
        return _skip(name, "transition columns are missing")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(1e-9, float(np.nanmax(mat))))
    ax.set_title("Mean State Transition Matrix")
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_xticks(range(n_states))
    ax.set_yticks(range(n_states))
    fig.colorbar(im, ax=ax, label="Probability")
    fig.tight_layout()
    out = results_dir / "comparison" / name
    _ensure(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def state_duration_distribution(bouts: pd.DataFrame | None, results_dir: Path) -> None:
    name = "state_duration_distribution.png"
    plt = _mpl()
    if plt is None:
        return _skip(name, "matplotlib is not installed")
    if bouts is None or bouts.empty or not {"state", "duration_sec"}.issubset(bouts.columns):
        return _skip(name, "bouts table is missing state/duration_sec")
    states = sorted(bouts["state"].dropna().unique().tolist())
    groups = [pd.to_numeric(bouts[bouts["state"] == s]["duration_sec"], errors="coerce").dropna().values for s in states]
    if not states or not any(len(g) for g in groups):
        return _skip(name, "no valid bout durations")
    fig, ax = plt.subplots(figsize=(max(7, 0.35 * len(states)), 5))
    ax.boxplot(groups, labels=[f"S{int(s)}" for s in states], showfliers=False)
    ax.set_xlabel("State")
    ax.set_ylabel("Bout duration (s)")
    ax.set_title("State Duration Distribution")
    ax.set_yscale("log")
    fig.tight_layout()
    out = results_dir / "comparison" / name
    _ensure(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def state_occupancy_over_time(summary: pd.DataFrame, design: dict, results_dir: Path) -> None:
    name = "state_occupancy_over_time.png"
    plt = _mpl()
    time_col = design.get("time_col")
    state_cols = safe_get_state_columns(summary)
    if plt is None:
        return _skip(name, "matplotlib is not installed")
    if not time_col or time_col not in summary.columns:
        return _skip(name, "time column is missing")
    if not state_cols:
        return _skip(name, "state columns are missing")
    order = _ordered_values(summary, time_col, design.get("time_order"))
    grouped = summary.groupby(time_col)[state_cols].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(9, 5))
    ids = _state_ids(state_cols)
    for col, sid in zip(state_cols, ids):
        ax.plot(range(len(grouped)), grouped[col], marker="o", linewidth=1.4, label=f"S{sid}")
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels([str(v) for v in grouped.index], rotation=30, ha="right")
    ax.set_xlabel(str(time_col).replace("_", " ").title())
    ax.set_ylabel("Mean fraction of session")
    ax.set_title("State Occupancy Over Time")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    out = results_dir / "comparison" / name
    _ensure(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def state_duration_over_time(bouts: pd.DataFrame | None, design: dict, results_dir: Path) -> None:
    name = "state_duration_over_time.png"
    plt = _mpl()
    time_col = design.get("time_col")
    if plt is None:
        return _skip(name, "matplotlib is not installed")
    if bouts is None or bouts.empty or not time_col or time_col not in bouts.columns:
        return _skip(name, "bouts table lacks the time column")
    if not {"state", "duration_sec"}.issubset(bouts.columns):
        return _skip(name, "bouts table is missing state/duration_sec")
    grouped = bouts.groupby([time_col, "state"])["duration_sec"].mean().reset_index()
    order = _ordered_values(bouts, time_col, design.get("time_order"))
    fig, ax = plt.subplots(figsize=(9, 5))
    for state, grp in grouped.groupby("state"):
        vals = grp.set_index(time_col).reindex(order)["duration_sec"]
        ax.plot(range(len(order)), vals, marker="o", linewidth=1.2, label=f"S{int(state)}")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([str(v) for v in order], rotation=30, ha="right")
    ax.set_xlabel(str(time_col).replace("_", " ").title())
    ax.set_ylabel("Mean bout duration (s)")
    ax.set_title("State Duration Over Time")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    out = results_dir / "comparison" / name
    _ensure(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def transition_entropy_over_time(transition: pd.DataFrame | None, design: dict, results_dir: Path) -> None:
    name = "transition_entropy_over_time.png"
    plt = _mpl()
    time_col = design.get("time_col")
    cols = _transition_cols(transition)
    if plt is None:
        return _skip(name, "matplotlib is not installed")
    if transition is None or transition.empty or not time_col or time_col not in transition.columns:
        return _skip(name, "transition table lacks the time column")
    if not cols:
        return _skip(name, "transition columns are missing")
    df = transition.copy()
    probs = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy()
    probs = np.maximum(probs, 0)
    row_sums = probs.sum(axis=1, keepdims=True)
    valid = row_sums[:, 0] > 0
    entropy = np.full(len(df), np.nan)
    p = probs[valid] / row_sums[valid]
    terms = np.zeros_like(p)
    positive = p > 0
    terms[positive] = p[positive] * np.log(p[positive])
    entropy[valid] = -np.sum(terms, axis=1)
    df["_transition_entropy"] = entropy
    order = _ordered_values(df, time_col, design.get("time_order"))
    grouped = df.groupby(time_col)["_transition_entropy"].mean().reindex(order)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(len(grouped)), grouped.values, marker="o", linewidth=2)
    ax.set_xticks(range(len(grouped)))
    ax.set_xticklabels([str(v) for v in grouped.index], rotation=30, ha="right")
    ax.set_xlabel(str(time_col).replace("_", " ").title())
    ax.set_ylabel("Mean transition entropy")
    ax.set_title("Transition Entropy Over Time")
    fig.tight_layout()
    out = results_dir / "comparison" / name
    _ensure(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def per_subject_state_trajectories(summary: pd.DataFrame, design: dict, results_dir: Path) -> None:
    name = "per_subject_state_trajectories.png"
    plt = _mpl()
    subject_col = design.get("subject_col")
    time_col = design.get("time_col")
    state_cols = safe_get_state_columns(summary)
    if plt is None:
        return _skip(name, "matplotlib is not installed")
    if not subject_col or subject_col not in summary.columns or not time_col or time_col not in summary.columns:
        return _skip(name, "subject or time column is missing")
    if not state_cols:
        return _skip(name, "state columns are missing")
    order = _ordered_values(summary, time_col, design.get("time_order"))
    dominant = summary[state_cols].mean().idxmax()
    fig, ax = plt.subplots(figsize=(8, 5))
    for subject, grp in summary.groupby(subject_col):
        vals = grp.groupby(time_col)[dominant].mean().reindex(order)
        ax.plot(range(len(order)), vals.values, marker="o", alpha=0.45, linewidth=1.0)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([str(v) for v in order], rotation=30, ha="right")
    ax.set_xlabel(str(time_col).replace("_", " ").title())
    ax.set_ylabel(str(dominant).replace("_frac", " fraction"))
    ax.set_title("Per-Subject State Trajectories")
    fig.tight_layout()
    out = results_dir / "comparison" / name
    _ensure(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def change_from_baseline(summary: pd.DataFrame, design: dict, results_dir: Path) -> None:
    name = "change_from_baseline.png"
    plt = _mpl()
    time_col = design.get("time_col")
    state_cols = safe_get_state_columns(summary)
    order = design.get("time_order") or []
    if plt is None:
        return _skip(name, "matplotlib is not installed")
    if not time_col or time_col not in summary.columns or not order:
        return _skip(name, "time column or time_order is missing")
    if not state_cols:
        return _skip(name, "state columns are missing")
    grouped = summary.groupby(time_col)[state_cols].mean()
    baseline = order[0]
    key = next((idx for idx in grouped.index if str(idx) == str(baseline)), None)
    if key is None:
        return _skip(name, "baseline timepoint is not present")
    delta = grouped.subtract(grouped.loc[key], axis=1)
    ordered = [idx for val in order for idx in grouped.index if str(idx) == str(val)]
    delta = delta.reindex(ordered)
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(delta[state_cols].T.values, cmap="RdBu_r", aspect="auto")
    ax.set_yticks(range(len(state_cols)))
    ax.set_yticklabels([f"S{s}" for s in _state_ids(state_cols)], fontsize=8)
    ax.set_xticks(range(len(delta.index)))
    ax.set_xticklabels([str(v) for v in delta.index], rotation=30, ha="right")
    ax.set_title("Change From Baseline")
    fig.colorbar(im, ax=ax, label="Delta occupancy")
    fig.tight_layout()
    out = results_dir / "comparison" / name
    _ensure(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def state_occupancy_by_time_and_condition(summary: pd.DataFrame, design: dict, results_dir: Path) -> None:
    name = "state_occupancy_by_time_and_condition.png"
    plt = _mpl()
    time_col = design.get("time_col")
    condition_col = (design.get("condition_cols") or [None])[0]
    state_cols = safe_get_state_columns(summary)
    if plt is None:
        return _skip(name, "matplotlib is not installed")
    if not time_col or time_col not in summary.columns or not condition_col or condition_col not in summary.columns:
        return _skip(name, "time or condition column is missing")
    if not state_cols:
        return _skip(name, "state columns are missing")
    dominant = summary[state_cols].mean().idxmax()
    order = _ordered_values(summary, time_col, design.get("time_order"))
    fig, ax = plt.subplots(figsize=(8, 5))
    for cond, grp in summary.groupby(condition_col):
        vals = grp.groupby(time_col)[dominant].mean().reindex(order)
        ax.plot(range(len(order)), vals.values, marker="o", linewidth=2, label=str(cond))
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([str(v) for v in order], rotation=30, ha="right")
    ax.set_xlabel(str(time_col).replace("_", " ").title())
    ax.set_ylabel(str(dominant).replace("_frac", " fraction"))
    ax.set_title("State Occupancy by Time and Condition")
    ax.legend(title=str(condition_col))
    fig.tight_layout()
    out = results_dir / "comparison" / name
    _ensure(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def condition_contrast_over_time(summary: pd.DataFrame, design: dict, results_dir: Path) -> None:
    name = "condition_contrast_over_time.png"
    plt = _mpl()
    time_col = design.get("time_col")
    condition_col = (design.get("condition_cols") or [None])[0]
    state_cols = safe_get_state_columns(summary)
    if plt is None:
        return _skip(name, "matplotlib is not installed")
    if not time_col or time_col not in summary.columns or not condition_col or condition_col not in summary.columns:
        return _skip(name, "time or condition column is missing")
    conditions = sorted(summary[condition_col].dropna().astype(str).unique().tolist())
    if len(conditions) < 2:
        return _skip(name, "need at least two condition values")
    if not state_cols:
        return _skip(name, "state columns are missing")
    a, b = conditions[:2]
    order = _ordered_values(summary, time_col, design.get("time_order"))
    rows = []
    for t in order:
        sub = summary[summary[time_col].astype(str) == str(t)]
        av = sub[sub[condition_col].astype(str) == a][state_cols].mean()
        bv = sub[sub[condition_col].astype(str) == b][state_cols].mean()
        if av.notna().any() and bv.notna().any():
            rows.append((t, float(np.linalg.norm((av - bv).fillna(0).values))))
    if not rows:
        return _skip(name, "no paired condition/time rows")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(len(rows)), [v for _, v in rows], marker="o", linewidth=2)
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([str(t) for t, _ in rows], rotation=30, ha="right")
    ax.set_xlabel(str(time_col).replace("_", " ").title())
    ax.set_ylabel("State-vector distance")
    ax.set_title(f"Condition Contrast Over Time: {b} vs {a}")
    fig.tight_layout()
    out = results_dir / "comparison" / name
    _ensure(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def context_enriched_states(summary: pd.DataFrame, design: dict, results_dir: Path) -> None:
    name = "context_enriched_states.png"
    plt = _mpl()
    condition_col = (design.get("condition_cols") or [None])[0]
    state_cols = safe_get_state_columns(summary)
    if plt is None:
        return _skip(name, "matplotlib is not installed")
    if not condition_col or condition_col not in summary.columns:
        return _skip(name, "condition column is missing")
    if not state_cols:
        return _skip(name, "state columns are missing")
    grouped = summary.groupby(condition_col)[state_cols].mean()
    if len(grouped) < 2:
        return _skip(name, "need at least two condition values")
    fig, ax = plt.subplots(figsize=(max(7, 0.35 * len(state_cols)), max(3.5, 0.5 * len(grouped))))
    im = ax.imshow(grouped.values, cmap="YlOrRd", aspect="auto")
    ax.set_yticks(range(len(grouped.index)))
    ax.set_yticklabels([str(v) for v in grouped.index])
    ax.set_xticks(range(len(state_cols)))
    ax.set_xticklabels([f"S{s}" for s in _state_ids(state_cols)], rotation=45, ha="right", fontsize=8)
    ax.set_title("Condition-Enriched States")
    fig.colorbar(im, ax=ax, label="Mean occupancy")
    fig.tight_layout()
    out = results_dir / "comparison" / name
    _ensure(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def transition_by_condition(transition: pd.DataFrame | None, design: dict, state_cols: list[str], results_dir: Path) -> None:
    name = "transition_by_condition.png"
    plt = _mpl()
    condition_col = (design.get("condition_cols") or [None])[0]
    if plt is None:
        return _skip(name, "matplotlib is not installed")
    if transition is None or transition.empty or not condition_col or condition_col not in transition.columns:
        return _skip(name, "transition table lacks condition column")
    n_states = len(state_cols)
    if n_states == 0:
        return _skip(name, "state columns are missing")
    groups = []
    for cond, grp in transition.groupby(condition_col):
        mat = _transition_matrix_from_rows(grp, n_states)
        if mat is not None:
            groups.append((cond, mat))
    if not groups:
        return _skip(name, "no transition matrices by condition")
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 3.8), squeeze=False)
    vmax = max(float(np.nanmax(mat)) for _, mat in groups)
    for ax, (cond, mat) in zip(axes[0], groups):
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(vmax, 1e-9))
        ax.set_title(str(cond))
        ax.set_xlabel("To")
        ax.set_ylabel("From")
    fig.suptitle("Transition by Condition")
    fig.tight_layout()
    out = results_dir / "comparison" / name
    _ensure(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def condition_state_trajectories(summary: pd.DataFrame, design: dict, results_dir: Path) -> None:
    name = "condition_state_trajectories.png"
    # Same data contract as state_occupancy_by_time_and_condition, but all
    # states are shown as small multiples to preserve per-state trajectories.
    plt = _mpl()
    time_col = design.get("time_col")
    condition_col = (design.get("condition_cols") or [None])[0]
    state_cols = safe_get_state_columns(summary)
    if plt is None:
        return _skip(name, "matplotlib is not installed")
    if not time_col or time_col not in summary.columns or not condition_col or condition_col not in summary.columns:
        return _skip(name, "time or condition column is missing")
    if not state_cols:
        return _skip(name, "state columns are missing")
    order = _ordered_values(summary, time_col, design.get("time_order"))
    n = len(state_cols)
    ncols = min(4, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.5 * nrows), squeeze=False)
    for ax, col in zip(axes.ravel(), state_cols):
        for cond, grp in summary.groupby(condition_col):
            vals = grp.groupby(time_col)[col].mean().reindex(order)
            ax.plot(range(len(order)), vals.values, marker="o", linewidth=1.2, label=str(cond))
        ax.set_title(col.replace("_frac", "").replace("_", " ").title(), fontsize=9)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([str(v) for v in order], rotation=30, ha="right", fontsize=7)
    for ax in axes.ravel()[len(state_cols):]:
        ax.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.suptitle("Condition State Trajectories")
    fig.tight_layout()
    out = results_dir / "comparison" / name
    _ensure(out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def generate_mode_driven_plots(
    summary: pd.DataFrame,
    transition: pd.DataFrame | None,
    bouts: pd.DataFrame | None,
    design: dict,
    results_dir: str | os.PathLike,
) -> None:
    results = Path(results_dir)
    state_cols = safe_get_state_columns(summary)
    mode = design.get("detected_mode") or "minimal"

    print(f"\n--- Mode-driven plots ({mode}) ---")
    state_summary_plot(summary, results)
    umap_by_state(results)
    umap_by_time_or_condition(summary, design, results)
    state_transition_matrix(transition, state_cols, results)
    state_duration_distribution(bouts, results)

    if mode == "time_only":
        state_occupancy_over_time(summary, design, results)
        state_duration_over_time(bouts, design, results)
        transition_entropy_over_time(transition, design, results)
        per_subject_state_trajectories(summary, design, results)
        change_from_baseline(summary, design, results)
    elif mode == "time_and_condition":
        state_occupancy_by_time_and_condition(summary, design, results)
        condition_contrast_over_time(summary, design, results)
        context_enriched_states(summary, design, results)
        transition_by_condition(transition, design, state_cols, results)
        condition_state_trajectories(summary, design, results)
