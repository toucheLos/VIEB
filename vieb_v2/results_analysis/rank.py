"""Rank the arms, and say what the ranking is a ranking of.

A single "best model" number over these arms would be dishonest -- the arms
disagree about what a state even is, and the metrics disagree about which arm
wins. So the ranking is built from named axes, each arm scores on each, and the
composite is a stated weighting rather than an implied one.

The one judgement embedded here, and the reason the ranking comes out as it
does:

**`n_significant` is a power metric, not an effect metric.** With 298 animals
paired within-subject, a two-percentage-point shift in a state that occupies 96%
of every session clears q<1e-30. `diffusion-HDBSCAN` scores 35/37 significant on
exactly that basis: its top state moves 0.951 -> 0.971. Counting significant
states therefore ranks it first and ranks the arm with an 0.55 -> 0.39 shift
fourth. `max_abs_median_diff` is what separates them, so effect size is weighted
above hit rate and the two are never summed into one column.

The axes:

| axis | what it measures | why it is not the others |
|---|---|---|
| `effect` | largest paired occupancy shift at retrieval | behavior, not power |
| `specificity` | retrieval effect / novel-context effect | is the shift about *this* context, or about any context change? |
| `coverage` | significant / tested states | does the whole state set carry signal, or one state? |
| `resolution` | 1 - largest_state_frac (clustered) | a 96%-one-state partition is not a decomposition |
| `parsimony` | is the state count an output or a parameter? | #55/#57's own criterion, applied |
| `cleanliness` | 1 - noise_frac | frames the method refuses to label |

`specificity` is the axis on which a VIEB arm beats MoSeq, and it would be
invisible in any single-contrast summary.
"""

from __future__ import annotations

import json
import math
import os

# Weights. Effect dominates because it is the only axis that is about behavior
# rather than about the shape of the partition; parsimony and cleanliness are
# hygiene, not evidence, so they are worth a fifth of it each.
WEIGHTS = {
    "effect": 0.35,
    "specificity": 0.20,
    "coverage": 0.15,
    "resolution": 0.15,
    "parsimony": 0.075,
    "cleanliness": 0.075,
}

RETRIEVAL = "retrieval_day1A_vs_day0A"
NOVEL = "novel_context_day2C_vs_day0A"


def _safe_ratio(a, b, cap=50.0):
    if not b or not math.isfinite(b) or b <= 0:
        return float("nan")
    return min(a / b, cap)


def build_rows(comparison, discrimination):
    """One row per arm, MoSeq included as the reference."""
    rows = []

    ms = comparison.get("moseq") or {}
    if ms:
        retr = ms["contrasts"][RETRIEVAL]
        novel = ms["contrasts"].get(NOVEL, {})
        n_tested = ms.get("n_syllables_tested") or retr.get("n_syllables")
        rows.append({
            "arm": "MoSeq (reference)",
            "latent": "keypoint-MoSeq (own)",
            "algorithm": "AR-HMM syllables",
            "reference": True,
            "n_states": 48,
            "n_tested": n_tested,
            "n_significant": retr["n_significant"],
            "effect": retr["max_abs_median_diff"],
            "novel_effect": novel.get("max_abs_median_diff"),
            "specificity": _safe_ratio(retr["max_abs_median_diff"],
                                       novel.get("max_abs_median_diff")),
            "coverage": retr["n_significant"] / n_tested if n_tested else 0.0,
            # MoSeq's own report carries no partition geometry, and inventing
            # one from its syllable table would compare a fitted quantity
            # against a measured one. Left absent, and the composite is
            # renormalized over the axes an arm actually has.
            "largest_state_frac": None,
            "noise_frac": None,
            "resolution": None,
            "cleanliness": None,
            "parsimony": 0.0,   # kappa sets the syllable count
            "parsimony_note": "syllable count set by the stickiness prior",
            "null_frac_at_or_above": (ms.get("shuffle_null", {})
                                      .get("frac_null_at_or_above_observed")),
        })

    sweep = comparison.get("n_regions_sweep") or []
    by_latent = {}
    for r in sweep:
        by_latent.setdefault(r["latent"], []).append(r)

    for name, arm in (comparison.get("arms") or {}).items():
        d = (discrimination.get("arms") or {}).get(name, {})
        s = d.get("score") or {}
        cons = d.get("contrasts") or {}
        retr = cons.get(RETRIEVAL, {})
        novel = cons.get(NOVEL, {})
        effect = s.get("max_abs_median_diff")
        novel_effect = novel.get("max_abs_median_diff")

        # Parsimony: an arm whose state count tracks its own resolution
        # parameter scores 0. Measured as 1 - d log(states) / d log(param)
        # across the sweep, clipped to [0, 1]. HDBSCAN was never swept, so it
        # is scored `None` -- untested, not passed.
        parsimony, note = None, "min_cluster_size never swept"
        if arm.get("algorithm") == "koopman":
            pts = sorted(by_latent.get(arm["latent"], []),
                         key=lambda r: r["n_regions"])
            pts = [p for p in pts if p.get("n_attractors")]
            if len(pts) >= 2:
                slope = ((math.log(pts[-1]["n_attractors"]) -
                          math.log(pts[0]["n_attractors"])) /
                         (math.log(pts[-1]["n_regions"]) -
                          math.log(pts[0]["n_regions"])))
                parsimony = max(0.0, min(1.0, 1.0 - slope))
                note = (f"n_attractors ~ n_regions^{slope:.2f} over "
                        f"{pts[0]['n_regions']}-{pts[-1]['n_regions']}")

        largest = arm.get("largest_state_frac_clean")
        rows.append({
            "arm": name,
            "latent": arm.get("latent"),
            "algorithm": arm.get("algorithm"),
            "reference": False,
            "n_states": arm.get("n_states"),
            "n_tested": s.get("n_states_tested"),
            "n_significant": s.get("n_significant"),
            "effect": effect,
            "novel_effect": novel_effect,
            "specificity": _safe_ratio(effect, novel_effect),
            "coverage": s.get("hit_rate"),
            "largest_state_frac": largest,
            "noise_frac": arm.get("noise_frac"),
            "resolution": (1.0 - largest) if largest is not None else None,
            "cleanliness": (1.0 - arm["noise_frac"])
                           if arm.get("noise_frac") is not None else None,
            "parsimony": parsimony,
            "parsimony_note": note,
            "noise_speed_ratio": arm.get("noise_speed_ratio"),
            "state_entropy_clean": arm.get("state_entropy_clean"),
            "null_frac_at_or_above": s.get("null_frac_at_or_above"),
            "testable_frac": s.get("testable_frac"),
            "top_state": s.get("top_state"),
            "top_state_mean_a": s.get("top_state_mean_a"),
            "top_state_mean_b": s.get("top_state_mean_b"),
            "n_retrieval_significant": retr.get("n_significant"),
        })
    return rows


def normalize(rows, axes=tuple(WEIGHTS)):
    """Min-max each axis over the arms that have it, then weight.

    Arms missing an axis are renormalized over the axes they do have rather
    than scored 0 on it, so "not measured" never reads as "measured badly".
    """
    scaled = {}
    for axis in axes:
        vals = [r[axis] for r in rows
                if r.get(axis) is not None and math.isfinite(r[axis])]
        lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
        span = hi - lo
        # Every arm tied on an axis means every arm is equally best on it, so
        # they all score 1. The naive `span or 1.0` guard gives them all 0
        # instead, which silently deletes the axis from the composite.
        scaled[axis] = {id(r): (1.0 if span == 0 else (r[axis] - lo) / span)
                        for r in rows
                        if r.get(axis) is not None and math.isfinite(r[axis])}

    for r in rows:
        present = {a: scaled[a][id(r)] for a in axes if id(r) in scaled[a]}
        wsum = sum(WEIGHTS[a] for a in present) or 1.0
        r["axes_scaled"] = present
        r["axes_missing"] = [a for a in axes if a not in present]
        r["composite"] = sum(WEIGHTS[a] * v for a, v in present.items()) / wsum
    rows.sort(key=lambda r: -r["composite"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def run(report_dir):
    with open(os.path.join(report_dir, "model_comparison.json")) as fh:
        comparison = json.load(fh)
    with open(os.path.join(report_dir, "discrimination.json")) as fh:
        discrimination = json.load(fh)
    rows = normalize(build_rows(comparison, discrimination))
    out = {"weights": WEIGHTS, "rows": rows}
    dest = os.path.join(report_dir, "ranking.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[rank] wrote {dest}")
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report",
                    default=os.path.expanduser("~/vieb2-results/_report"))
    args = ap.parse_args(argv)
    res = run(args.report)

    def fmt(v, spec="6.3f"):
        return "     -" if v is None or (isinstance(v, float) and
                                         not math.isfinite(v)) \
            else format(v, spec)

    print(f"\n{'#':>2s} {'arm':22s} {'comp':>6s} {'effect':>7s} {'spec':>6s} "
          f"{'cover':>6s} {'resol':>6s} {'parsi':>6s} {'clean':>6s}")
    for r in res["rows"]:
        print(f"{r['rank']:2d} {r['arm']:22s} {r['composite']:6.3f} "
              f"{fmt(r['effect'], '7.4f')} {fmt(r['specificity'])} "
              f"{fmt(r['coverage'])} {fmt(r['resolution'])} "
              f"{fmt(r['parsimony'])} {fmt(r['cleanliness'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
