"""Generate five summary plots from VIEB pipeline outputs.

Saves all figures to results/plots/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

OUT_DIR = "results/plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load data ────────────────────────────────────────────────────────────────
state_summary = pd.read_csv("results/characterization/state_summary.csv")
context_report = pd.read_csv("results/characterization/context_report.csv")
summary_table  = pd.read_csv("results/comparison/summary_table.csv")
bouts          = pd.read_csv("results/characterization/bouts.csv")

STATES = list(range(9))   # 0-8; exclude 9
STATE_COLS = [f"state_{s}_frac" for s in STATES]

ss = state_summary[state_summary["state"].isin(STATES)].set_index("state").loc[STATES]
cr = context_report[context_report["state"].isin(STATES)].set_index("state").loc[STATES]

# ── Colour palette: one colour per state ─────────────────────────────────────
cmap9 = matplotlib.colormaps.get_cmap("tab10").resampled(9)
state_colors = {s: cmap9(i) for i, s in enumerate(STATES)}

# ─────────────────────────────────────────────────────────────────────────────
# 1. Kinematic profile heatmap
# ─────────────────────────────────────────────────────────────────────────────
def minmax(col):
    mn, mx = col.min(), col.max()
    if mx == mn:
        return col * 0.0
    return (col - mn) / (mx - mn)

# rearing_score is not present; use elongation as the closest proxy
# Units: centroid_speed px/s, angular_vel rad/s, bout_dur s, elongation dimensionless ratio
kin_cols = {
    "speed\n(px/s)":       "mean_centroid_speed",
    "ang_vel\n(rad/s)":    "mean_angular_vel",
    "bout_dur\n(s)":       "mean_bout_dur_sec",
    "elongation\n(ratio)": "mean_elongation",
}
available = {k: v for k, v in kin_cols.items() if v in ss.columns}

raw  = pd.DataFrame({k: ss[v] for k, v in available.items()}, index=STATES)
heat = pd.DataFrame({k: minmax(ss[v]) for k, v in available.items()}, index=STATES)

# Format strings per column: show raw value so reader can see actual magnitude
fmt = {
    "speed\n(px/s)":       lambda v: f"{v:.1f}",
    "ang_vel\n(rad/s)":    lambda v: f"{v:.2f}",
    "bout_dur\n(s)":       lambda v: f"{v:.1f}",
    "elongation\n(ratio)": lambda v: f"{v:.2f}",
}

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(heat.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
ax.set_xticks(range(len(heat.columns)))
ax.set_xticklabels(heat.columns, rotation=0, ha="center", fontsize=9)
ax.set_yticks(range(len(STATES)))
ax.set_yticklabels([f"State {s}" for s in STATES], fontsize=9)
ax.set_title("Kinematic profile (states 0–8)\nColor = min-max normalized; cell text = raw value", fontsize=11)
for i in range(len(STATES)):
    for j, col in enumerate(heat.columns):
        raw_val = raw.iloc[i, j]
        cell_txt = fmt[col](raw_val)
        text_color = "white" if heat.iloc[i, j] > 0.65 else "black"
        ax.text(j, i, cell_txt, ha="center", va="center", fontsize=7.5, color=text_color)
plt.colorbar(im, ax=ax, label="Normalized intensity (0 = min, 1 = max per column)")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/1_kinematic_heatmap.png", dpi=150)
plt.close(fig)
print("Saved: 1_kinematic_heatmap.png")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Context enrichment bar chart
# ─────────────────────────────────────────────────────────────────────────────
x = np.arange(len(STATES))
w = 0.28
ctx_labels = ["A", "B", "C"]
ctx_fracs  = [cr["A_frac"], cr["B_frac"], cr["C_frac"]]
ctx_colors = ["#4878CF", "#D65F5F", "#6ACC65"]

fig, ax = plt.subplots(figsize=(10, 4))
for i, (label, frac, color) in enumerate(zip(ctx_labels, ctx_fracs, ctx_colors)):
    ax.bar(x + (i - 1) * w, frac.values, w, label=f"Context {label}", color=color, alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f"State {s}" for s in STATES], rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Mean fraction of session", fontsize=10)
ax.set_title("Context enrichment per state (states 0–8)", fontsize=11)
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/2_context_enrichment.png", dpi=150)
plt.close(fig)
print("Saved: 2_context_enrichment.png")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-animal state occupancy heatmap
# ─────────────────────────────────────────────────────────────────────────────
# One row per (animal_id). Average state fracs across videos for that animal.
animal_fracs = (
    summary_table[["animal_id"] + STATE_COLS]
    .groupby("animal_id")[STATE_COLS]
    .mean()
)
animal_fracs.columns = STATES

# Sort by total non-state-9 occupancy (already excluded from STATE_COLS) – desc
animal_fracs["_total"] = animal_fracs[STATES].sum(axis=1)
animal_fracs = animal_fracs.sort_values("_total", ascending=False).drop(columns="_total")

fig, ax = plt.subplots(figsize=(10, max(4, len(animal_fracs) * 0.28)))
im = ax.imshow(animal_fracs.values, aspect="auto", cmap="Blues")
ax.set_xticks(range(len(STATES)))
ax.set_xticklabels([f"S{s}" for s in STATES], fontsize=9)
ax.set_yticks(range(len(animal_fracs)))
ax.set_yticklabels(animal_fracs.index.astype(str), fontsize=7)
ax.set_xlabel("State", fontsize=10)
ax.set_ylabel("Animal ID", fontsize=10)
ax.set_title("Per-animal state occupancy (states 0–8; sorted by total non-noise)", fontsize=11)
plt.colorbar(im, ax=ax, label="Mean fraction")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/3_animal_occupancy.png", dpi=150)
plt.close(fig)
print("Saved: 3_animal_occupancy.png")

# ─────────────────────────────────────────────────────────────────────────────
# 4. State occupancy by day line plot
# ─────────────────────────────────────────────────────────────────────────────
day_fracs = (
    summary_table[["day"] + STATE_COLS]
    .groupby("day")[STATE_COLS]
    .mean()
)
day_fracs.columns = STATES
days = sorted(day_fracs.index)

fig, ax = plt.subplots(figsize=(9, 5))
for s in STATES:
    vals = [day_fracs.loc[d, s] if d in day_fracs.index else np.nan for d in days]
    ax.plot(days, vals, marker="o", color=state_colors[s], label=f"State {s}", linewidth=1.8)
ax.set_xlabel("Day", fontsize=10)
ax.set_ylabel("Mean fraction of session", fontsize=10)
ax.set_title("State occupancy across days (states 0–8)", fontsize=11)
ax.set_xticks(days)
ax.legend(ncol=3, fontsize=8, loc="upper right")
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/4_state_by_day.png", dpi=150)
plt.close(fig)
print("Saved: 4_state_by_day.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Bout duration distribution boxplot
# ─────────────────────────────────────────────────────────────────────────────
bouts_filt = bouts[bouts["state"].isin(STATES)]
groups = [bouts_filt.loc[bouts_filt["state"] == s, "duration_sec"].values for s in STATES]

fig, ax = plt.subplots(figsize=(10, 5))
bp = ax.boxplot(
    groups,
    labels=[f"State {s}" for s in STATES],
    patch_artist=True,
    showfliers=True,
    flierprops=dict(marker="o", markersize=2, alpha=0.3),
    medianprops=dict(color="black", linewidth=2),
)
for patch, s in zip(bp["boxes"], STATES):
    patch.set_facecolor(state_colors[s])
    patch.set_alpha(0.75)
ax.set_xlabel("State", fontsize=10)
ax.set_ylabel("Bout duration (s, log scale)", fontsize=10)
ax.set_title("Bout duration distribution per state (states 0–8)", fontsize=11)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"{y:g} s"))
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/5_bout_duration.png", dpi=150)
plt.close(fig)
print("Saved: 5_bout_duration.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. State 9 fraction by day × context (fear conditioning curve)
# ─────────────────────────────────────────────────────────────────────────────
ctx_colors_map = {"A": "#4878CF", "B": "#D65F5F", "C": "#6ACC65"}

day_ctx = (
    summary_table[["day", "context", "state_9_frac"]]
    .groupby(["day", "context"])["state_9_frac"]
    .agg(["mean", "sem"])
    .reset_index()
)

fig, ax = plt.subplots(figsize=(9, 5))
for ctx, grp in day_ctx.groupby("context"):
    grp = grp.sort_values("day")
    color = ctx_colors_map.get(ctx, "gray")
    ax.plot(grp["day"], grp["mean"], marker="o", color=color,
            label=f"Context {ctx}", linewidth=2)
    ax.fill_between(grp["day"],
                    grp["mean"] - grp["sem"],
                    grp["mean"] + grp["sem"],
                    color=color, alpha=0.15)
ax.set_xlabel("Day", fontsize=10)
ax.set_ylabel("Mean fraction of session (state 9)", fontsize=10)
ax.set_title("State 9 occupancy by day and context\n(shading = ±1 SEM across animals)", fontsize=11)
ax.set_xticks(sorted(day_ctx["day"].unique()))
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/6_state9_by_day_context.png", dpi=150)
plt.close(fig)
print("Saved: 6_state9_by_day_context.png")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Per-animal state 9 trajectory across days
# ─────────────────────────────────────────────────────────────────────────────
# One line per animal; color = dominant context for that animal
animal_day = (
    summary_table[["animal_id", "day", "context", "state_9_frac"]]
    .groupby(["animal_id", "day"])
    .agg(state_9_frac=("state_9_frac", "mean"),
         context=("context", lambda x: x.mode()[0]))
    .reset_index()
)

# Assign each animal a single representative context (most frequent)
animal_ctx = (
    summary_table.groupby("animal_id")["context"]
    .agg(lambda x: x.mode()[0])
    .to_dict()
)

animals = sorted(animal_day["animal_id"].unique())
fig, ax = plt.subplots(figsize=(10, 5))
for animal in animals:
    sub = animal_day[animal_day["animal_id"] == animal].sort_values("day")
    ctx = animal_ctx.get(animal, "A")
    color = ctx_colors_map.get(ctx, "gray")
    ax.plot(sub["day"], sub["state_9_frac"], marker="o", color=color,
            alpha=0.5, linewidth=1.2, markersize=4)

# Legend proxy
for ctx, color in ctx_colors_map.items():
    ax.plot([], [], color=color, linewidth=2, label=f"Context {ctx} (dominant)")
ax.set_xlabel("Day", fontsize=10)
ax.set_ylabel("Mean fraction of session (state 9)", fontsize=10)
ax.set_title("Per-animal state 9 trajectory across days\n(color = animal's most frequent context)", fontsize=11)
ax.set_xticks(sorted(animal_day["day"].unique()))
ax.legend(fontsize=9)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/7_animal_state9_trajectories.png", dpi=150)
plt.close(fig)
print("Saved: 7_animal_state9_trajectories.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. State × context heatmap (all 10 states, mean fraction)
# ─────────────────────────────────────────────────────────────────────────────
ALL_STATES = list(range(10))
ALL_STATE_COLS = [f"state_{s}_frac" for s in ALL_STATES]

ctx_state = (
    summary_table[["context"] + ALL_STATE_COLS]
    .groupby("context")[ALL_STATE_COLS]
    .mean()
)
ctx_state.columns = ALL_STATES

fig, axes = plt.subplots(1, 2, figsize=(13, 3.5),
                         gridspec_kw={"width_ratios": [9, 1], "wspace": 0.05})

# Left panel: states 0–8 (fine-grained scale)
im0 = axes[0].imshow(ctx_state[list(range(9))].values,
                     aspect="auto", cmap="YlOrRd")
axes[0].set_xticks(range(9))
axes[0].set_xticklabels([f"State {s}" for s in range(9)], fontsize=9)
axes[0].set_yticks(range(3))
axes[0].set_yticklabels(ctx_state.index, fontsize=10)
axes[0].set_title("States 0–8 (rare behaviors)", fontsize=10)
for i, ctx in enumerate(ctx_state.index):
    for j in range(9):
        v = ctx_state.loc[ctx, j]
        axes[0].text(j, i, f"{v:.3f}", ha="center", va="center",
                     fontsize=7.5,
                     color="white" if ctx_state.loc[ctx, j] > ctx_state[list(range(9))].values.max() * 0.65 else "black")
plt.colorbar(im0, ax=axes[0], label="Mean fraction")

# Right panel: state 9 (separate scale)
im1 = axes[1].imshow(ctx_state[[9]].values,
                     aspect="auto", cmap="Blues")
axes[1].set_xticks([0])
axes[1].set_xticklabels(["State 9"], fontsize=9)
axes[1].set_yticks(range(3))
axes[1].set_yticklabels([], fontsize=10)
axes[1].set_title("State 9", fontsize=10)
for i, ctx in enumerate(ctx_state.index):
    v = ctx_state.loc[ctx, 9]
    axes[1].text(0, i, f"{v:.2f}", ha="center", va="center",
                 fontsize=8, color="white" if v > 0.9 else "black")
plt.colorbar(im1, ax=axes[1], label="Mean fraction")

fig.suptitle("Mean state occupancy by context (all 10 states)", fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/8_state_context_heatmap_full.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: 8_state_context_heatmap_full.png")

print("\nAll plots written to", OUT_DIR)
