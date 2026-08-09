"""Figures for the cross-method comparison.

Six figures, each answering one question that a table answers badly:

1. `effect_vs_power`   — why "35 of 37 states significant" is not a win
2. `retrieval_timecourse` — does the top state tell the fear-learning story?
3. `state_count_sweep` — is the Koopman state count an output or a parameter?
4. `partition_geometry` — what shape is each partition?
5. `implied_timescales` — the §3 gate, and what its failure looks like
6. `ranking`           — the composite, decomposed into its axes

Palette is the dataviz reference instance, slots 1-5 in fixed order, validated
on the adjacent pairlist in both modes (worst adjacent CVD ΔE 9.1 light / 8.4
dark; normal-vision 19.6 / 19.3). Three light-mode slots sit below 3:1 against
the surface, so the **relief rule** applies throughout: every series is directly
labeled and every figure has a table twin in the report. Figure 1 is a scatter,
an all-pairs form the 5-slot set cannot clear, so it drops categorical color
entirely and identifies its five points by direct label plus marker shape.

Usage:
    python -m results_analysis.plots --report ~/vieb2-results/_report
"""

from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# dataviz reference palette, light mode, slots 1-5 in fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8880"
GRID = "#e4e3de"
SURFACE = "#fcfcfb"
# Status colors are reserved and never reused as a series (dataviz rule).
CRITICAL = "#c9252d"

ARM_ORDER = ["MoSeq (reference)", "pca-HDBSCAN", "pca-Koopman",
             "diffusion-HDBSCAN", "diffusion-Koopman"]
COLOR = dict(zip(ARM_ORDER, SERIES))

# The design, in experimental order. Keys are `arm_profile`'s.
# Single-line tick labels. The phase names live in the caption instead: at
# 45 degrees, two-line ticks overprint their neighbours.
PHASES = [
    ("CFC d0 A", "d0 A"), ("CFC d1 A/NS", "d1 A"), ("CFC d2 C", "d2 C"),
    ("CFD d3 A", "d3 A"), ("CFD d4 A", "d4 A"), ("CFD d5 A", "d5 A"),
    ("CFD d6 A", "d6 A"), ("CFD d7 A", "d7 A"),
]
PHASE_CAPTION = ("d0 conditioning (shock)   ·   d1 retrieval test (no shock)"
                 "   ·   d2 novel context   ·   d3–d7 discrimination")
PHASES_B = ["CFD d3 B", "CFD d4 B", "CFD d5 B", "CFD d6 B", "CFD d7 B"]


def _style(ax, title=None, xlabel=None, ylabel=None):
    """Recessive grid and axes; ink for text, never a series color."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=1.0)
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=11.5, loc="left",
                     fontweight="600", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK_2, fontsize=9.5)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_2, fontsize=9.5)
    return ax


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=170, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"[plots] {path}")
    return path


def fig_effect_vs_power(rows, out_dir):
    """The headline dissociation: significance count vs effect size.

    Single-hue marks with direct labels rather than five categorical colors --
    a scatter is an all-pairs form, where the 5-slot set fails the normal-vision
    floor (ΔE 12.9 against a 15.0 gate). Marker shape separates the two
    algorithm families, so nothing here is carried by color alone.
    """
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    _style(ax, "Effect size does not follow significance count",
           "significant states / tested states",
           "largest paired occupancy shift at retrieval")

    # pca-HDBSCAN's largest median difference is exactly 0.0, which a log axis
    # cannot place. Pinning it to the floor and saying so is honest; plotting
    # it at 1e-4 unmarked would read as "small effect" rather than "none".
    floor = 3e-4
    for r in rows:
        if r.get("effect") is None or r.get("coverage") is None:
            continue
        ref = r["reference"]
        zero = r["effect"] <= 0
        marker = "D" if ref else ("o" if r["algorithm"] == "koopman" else "s")
        y = floor if zero else r["effect"]
        ax.scatter(r["coverage"], y, s=150 if ref else 110, marker=marker,
                   color=SERIES[0] if ref else INK_2, edgecolor=SURFACE,
                   linewidth=2, zorder=3)
        note = ("  median diff exactly 0" if zero else "")
        # Labels sit left of the two right-most points so none runs off-axis.
        left = r["coverage"] > 0.9
        ax.annotate(f"{r['arm']}\n{r['n_significant']}/{r['n_tested']} sig{note}",
                    (r["coverage"], y), textcoords="offset points",
                    xytext=(-13 if left else 13, 0), fontsize=8.5, color=INK,
                    va="center", ha="right" if left else "left")

    ax.set_yscale("log")
    ax.set_xlim(0.40, 1.10)
    ax.set_ylim(floor * 0.6, 2.0)
    ax.axhline(0.05, color=MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
               zorder=1)
    ax.annotate("5 percentage points — a shift a human scorer could see",
                (0.415, 0.058), fontsize=8, color=MUTED)
    ax.legend(handles=[
        Line2D([], [], marker="D", linestyle="none", color=SERIES[0],
               markersize=9, label="reference (MoSeq)"),
        Line2D([], [], marker="o", linestyle="none", color=INK_2,
               markersize=8, label="Koopman basins"),
        Line2D([], [], marker="s", linestyle="none", color=INK_2,
               markersize=8, label="HDBSCAN density"),
    ], frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    return _save(fig, out_dir, "1_effect_vs_power")


def fig_retrieval_timecourse(comparison, discrimination, rows, out_dir):
    """Top state's occupancy across the whole design, per method.

    The contrasts say whether there is a difference; this says whether the
    difference is the *shape* fear conditioning should produce -- low while
    naive, high on re-exposure, intermediate in a novel context, and A-vs-B
    separated across the discrimination days.
    """
    panels = []
    ms = comparison.get("moseq") or {}
    if ms.get("arm_profile"):
        panels.append(("MoSeq (reference)", ms["arm_profile"],
                       ms["top"]["syllable"]))
    for name in ARM_ORDER[1:]:
        d = (discrimination.get("arms") or {}).get(name, {})
        if d.get("arm_profile") and d.get("score", {}).get("top_state") is not None:
            panels.append((name, d["arm_profile"], d["score"]["top_state"]))

    fig, axes = plt.subplots(1, len(panels), figsize=(3.05 * len(panels), 4.2),
                             sharex=True, sharey=True)
    axes = [axes] if len(panels) == 1 else list(axes)
    labels = [lab for _, lab in PHASES]
    x = range(len(PHASES))

    for ax, (name, profile, state) in zip(axes, panels):
        key = f"s{int(state)}"
        ya = [profile.get(k, {}).get(key) for k, _ in PHASES]
        yb = [None, None, None] + [profile.get(k, {}).get(key)
                                   for k in PHASES_B]
        color = COLOR[name]
        ax.plot(x, ya, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.6, zorder=3)
        xb = [i for i, v in zip(x, yb) if v is not None]
        vb = [v for v in yb if v is not None]
        ax.plot(xb, vb, color=color, linewidth=2, linestyle=(0, (4, 3)),
                marker="o", markersize=5, markerfacecolor=SURFACE,
                markeredgecolor=color, markeredgewidth=1.6, zorder=3)
        _style(ax, f"{name}\nstate {state}", None,
               "occupancy" if ax is axes[0] else None)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, fontsize=7.5, color=INK_2, rotation=45,
                           ha="right")
        ax.set_ylim(0, 1.12)
        # Only the two contrast points are labeled, and they are pushed apart
        # vertically -- at 0.95/0.97 the two labels would otherwise overprint.
        pts = [(i, ya[i]) for i in (0, 1) if ya[i] is not None]
        close = len(pts) == 2 and abs(pts[0][1] - pts[1][1]) < 0.10
        if close and min(v for _, v in pts) < 0.12:
            # Both near the floor: pushing one down would land it in the tick
            # labels, so separate them horizontally instead.
            offsets = [(-6, 11), (6, 11)]
        elif close:
            offsets = [(0, -16), (0, 11)] if pts[0][1] <= pts[1][1] \
                else [(0, 11), (0, -16)]
        else:
            offsets = [(0, 11), (0, 11)]
        for (i, v), off in zip(pts, offsets):
            ax.annotate(f"{v:.2f}", (i, v), textcoords="offset points",
                        xytext=off, ha="center", fontsize=8.5, color=INK,
                        fontweight="600")

    # One figure-level legend: repeating it in five panels spent space and
    # collided with the data in the flattest one.
    fig.legend(handles=[
        Line2D([], [], color=INK_2, linewidth=2, marker="o", markersize=6,
               markeredgecolor=SURFACE, label="Context A (conditioned)"),
        Line2D([], [], color=INK_2, linewidth=2, linestyle=(0, (4, 3)),
               marker="o", markersize=5, markerfacecolor=SURFACE,
               markeredgecolor=INK_2, label="Context B (safe)"),
    ], frameon=False, fontsize=9, labelcolor=INK_2, ncol=2,
        loc="upper right", bbox_to_anchor=(0.998, 1.10))
    fig.suptitle("Does the top state tell the fear-learning story?",
                 color=INK, fontsize=13, fontweight="600", x=0.005, ha="left",
                 y=1.10)
    fig.text(0.005, -0.10, PHASE_CAPTION, fontsize=8.5, color=MUTED,
             ha="left")
    fig.subplots_adjust(wspace=0.12)
    return _save(fig, out_dir, "2_retrieval_timecourse")


def fig_state_count_sweep(comparison, out_dir):
    """n_attractors against n_regions -- the parsimony claim, tested."""
    sweep = comparison.get("n_regions_sweep") or []
    by = {}
    for r in sweep:
        by.setdefault(r["latent"], []).append(r)

    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    _style(ax, "The Koopman state count tracks its resolution parameter",
           "--n-regions (the parameter)", "attractors found (the 'output')")

    for i, (latent, pts) in enumerate(sorted(by.items())):
        pts = sorted(pts, key=lambda r: r["n_regions"])
        xs = [p["n_regions"] for p in pts]
        ys = [p["n_attractors"] for p in pts]
        color = SERIES[1] if latent == "pca" else SERIES[2]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=3)
        slope = ((math.log(ys[-1]) - math.log(ys[0])) /
                 (math.log(xs[-1]) - math.log(xs[0])))
        # Labels sit below-right of the last point, clear of the title band
        # and of the reference diagonal that runs above the pca series.
        ax.annotate(f"{latent}   n ∝ r^{slope:.2f}", (xs[-1], ys[-1]),
                    textcoords="offset points", xytext=(-8, -18),
                    fontsize=9.5, color=INK, ha="right", fontweight="600")

    lim = [10, 230]
    ax.plot(lim, lim, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)),
            zorder=1)
    ax.annotate("one attractor per region —\nthe state count IS the parameter",
                (12.5, 95), fontsize=8.5, color=MUTED, ha="left", va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([12, 24, 48, 96, 192])
    ax.set_xticklabels(["12", "24", "48", "96", "192"])
    ax.set_yticks([5, 10, 25, 50, 100, 200])
    ax.set_yticklabels(["5", "10", "25", "50", "100", "200"])
    return _save(fig, out_dir, "3_state_count_sweep")


def fig_partition_geometry(comparison, out_dir):
    """What shape each partition is, in the one convention that compares."""
    arms = comparison.get("arms") or {}
    names = [n for n in ARM_ORDER[1:] if n in arms]
    metrics = [
        ("largest_state_frac_clean", "largest state\n(of clustered frames)"),
        ("noise_frac", "unlabeled frames"),
        ("state_entropy_clean", "state entropy\n(normalized)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3))
    for ax, (key, title) in zip(axes, metrics):
        vals = [arms[n].get(key) or 0.0 for n in names]
        # 2px surface gap between adjacent bars: width 0.62 on unit spacing.
        bars = ax.bar(range(len(names)), vals, width=0.62,
                      color=[COLOR[n] for n in names],
                      edgecolor=SURFACE, linewidth=2, zorder=3)
        _style(ax, title)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace("-", "\n") for n in names], fontsize=8.5,
                           color=INK_2)
        ax.set_ylim(0, 1.05)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:.3f}", (b.get_x() + b.get_width() / 2, v),
                        textcoords="offset points", xytext=(0, 5), ha="center",
                        fontsize=9, color=INK, fontweight="600")
    fig.suptitle("Partition shape — the two HDBSCAN arms are one state plus a "
                 "tail", color=INK, fontsize=13, fontweight="600", x=0.005,
                 ha="left", y=1.13)
    return _save(fig, out_dir, "4_partition_geometry")


def fig_implied_timescales(comparison, out_dir):
    """The §3 gate: t2 against tau, and the local exponent that is its shape."""
    ts = (comparison.get("timescales") or {}).get("arms") or {}
    if not ts:
        return None

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8))
    _style(ax, "Implied timescale t₂ never flattens",
           "lag τ (s)", "implied timescale t₂ (s)")
    _style(ax2, "…and the growth is scale-free, not an artifact",
           "lag τ (s)", "d log t₂ / d log τ  (over 4× lag windows)")

    # The two arms lie almost on top of each other -- that near-coincidence is
    # the finding (the restored channels change nothing), and it is exactly
    # what makes endpoint direct labels overprint. A legend carries identity
    # here instead; two series is the case the skill allows it for.
    handles = []
    for i, (key, label) in enumerate((("channels", "11D  pose PCs + locomotor"),
                                      ("pose_only", "9D  pose PCs only (control)"))):
        arm = ts.get(key)
        if not arm:
            continue
        color = SERIES[i + 3]
        taus = arm["taus_s"]
        t2 = arm["t2_s"]
        xs = [t for t, y in zip(taus, t2) if y]
        ys = [y for y in t2 if y]
        ax.plot(xs, ys, color=color, linewidth=2.4, zorder=3 - i * 0.5,
                alpha=1.0 if i == 0 else 0.95)
        ex = arm.get("local_exponents") or []
        ax2.plot([e["tau_lo"] for e in ex], [e["exponent"] for e in ex],
                 color=color, linewidth=2.4, marker="o", markersize=5,
                 markeredgecolor=SURFACE, markeredgewidth=1.4,
                 zorder=3 - i * 0.5)
        handles.append(Line2D([], [], color=color, linewidth=2.4, label=label))

    lo = list(ts.get("channels", {}).get("taus_s", [1]))
    ax.plot(lo, lo, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax.annotate("t₂ = τ  (resolution floor)", (lo[-1], lo[-1]),
                textcoords="offset points", xytext=(-4, -16), ha="right",
                fontsize=8.5, color=MUTED)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK_2,
              loc="upper left")

    ax2.axhline(0, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    ax2.annotate("0 = a plateau (what the gate needed)", (0.037, 0.035),
                 fontsize=8.5, color=MUTED)
    ax2.axhline(1, color=CRITICAL, linewidth=1.2, linestyle=(0, (2, 3)),
                zorder=1)
    ax2.annotate("1 = the trivial large-τ artifact", (0.037, 1.04),
                 fontsize=8.5, color=CRITICAL)
    ax2.set_xscale("log")
    ax2.set_ylim(-0.18, 1.24)
    ax2.legend(handles=handles, frameon=False, fontsize=9, labelcolor=INK_2,
               loc="lower right")
    fig.subplots_adjust(wspace=0.24)
    return _save(fig, out_dir, "5_implied_timescales")


def fig_ranking(ranking, out_dir):
    """The composite, decomposed. A single bar would hide the disagreement."""
    rows = ranking["rows"]
    axes_order = ["effect", "specificity", "coverage", "resolution",
                  "parsimony", "cleanliness"]
    fig, ax = plt.subplots(figsize=(10.4, 0.85 * len(rows) + 2.4))
    _style(ax, "Composite score, decomposed into its axes",
           "weighted contribution to the composite")

    weights = ranking["weights"]
    ys = range(len(rows))
    palette6 = SERIES + ["#4a3aa7"]
    for y, r in zip(ys, rows):
        left = 0.0
        for j, axis in enumerate(axes_order):
            v = r["axes_scaled"].get(axis)
            if v is None:
                continue
            w = weights[axis] * v
            ax.barh(y, w, left=left, height=0.6, color=palette6[j],
                    edgecolor=SURFACE, linewidth=2, zorder=3)
            left += w
        ax.annotate(f"{r['composite']:.3f}", (left, y),
                    textcoords="offset points", xytext=(8, 0), va="center",
                    fontsize=10, color=INK, fontweight="600")
        if r["axes_missing"]:
            ax.annotate(f"(no {', '.join(r['axes_missing'])})", (left, y),
                        textcoords="offset points", xytext=(52, 0),
                        va="center", fontsize=8, color=MUTED)

    ax.set_yticks(list(ys))
    ax.set_yticklabels([f"{r['rank']}. {r['arm']}" for r in rows], fontsize=10,
                       color=INK)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.legend(handles=[Line2D([], [], marker="s", linestyle="none",
                              color=palette6[j], markersize=10,
                              label=f"{a}  (w={weights[a]:.3f})")
                       for j, a in enumerate(axes_order)],
              frameon=False, fontsize=9, labelcolor=INK_2, ncol=3,
              loc="lower right", bbox_to_anchor=(1.0, -0.20))
    return _save(fig, out_dir, "6_ranking")


def run(report_dir, out_dir=None):
    out_dir = out_dir or os.path.join(report_dir, "figures")
    with open(os.path.join(report_dir, "model_comparison.json")) as fh:
        comparison = json.load(fh)
    with open(os.path.join(report_dir, "discrimination.json")) as fh:
        discrimination = json.load(fh)
    with open(os.path.join(report_dir, "ranking.json")) as fh:
        ranking = json.load(fh)

    made = [
        fig_effect_vs_power(ranking["rows"], out_dir),
        fig_retrieval_timecourse(comparison, discrimination, ranking["rows"],
                                 out_dir),
        fig_state_count_sweep(comparison, out_dir),
        fig_partition_geometry(comparison, out_dir),
        fig_implied_timescales(comparison, out_dir),
        fig_ranking(ranking, out_dir),
    ]
    return [m for m in made if m]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report",
                    default=os.path.expanduser("~/vieb2-results/_report"))
    ap.add_argument("--out", default=None, help="default <report>/figures")
    args = ap.parse_args(argv)
    run(args.report, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
