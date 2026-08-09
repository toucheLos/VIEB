"""Harvest every run's JSON into one normalized record per arm.

The runs write four incompatible report shapes -- `latent_comparison.json`,
`hdbscan_report.json`, `koopman_report_r<N>.json`, `timescales_*.json` -- plus
MoSeq's control, which is not a VIEB report at all. Reading them together by
hand is where the arithmetic errors come from, so it happens once, here.

Two normalizations are load-bearing:

**Entropy convention.** `state_entropy` is reported in two conventions in every
report (`v1_convention` counts noise frames in the denominator,
`clustered_only` does not) and they are not interchangeable
(`hpc/README.md:134`). Both are carried through under explicit names; nothing
here picks one silently.

**Frame denominator.** The `koopman_*` family was built before the h5/csv dedup
(#59) and sees 28,626,107 frames against the transfer-operator family's
22,355,989. `n_frames` is recorded per arm and `deduped` flags which family an
arm belongs to, so a reader cannot accidentally compare a rate from one against
a count from the other.

Usage:
    python -m results_analysis.collect --results-root ~/vieb2-results \
        --out ~/vieb2-results/_report
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re

# Every arm that produced per-frame states, as
# (name, latent, algorithm, directory, report filename).
ARM_REPORTS = (
    ("pca-HDBSCAN", "pca", "hdbscan", "koopman_pca", "hdbscan_report.json"),
    ("pca-Koopman", "pca", "koopman", "koopman_pca", "koopman_report_r48.json"),
    ("diffusion-HDBSCAN", "diffusion", "hdbscan", "koopman_diffusion",
     "hdbscan_report.json"),
    ("diffusion-Koopman", "diffusion", "koopman", "koopman_diffusion",
     "koopman_report_r48.json"),
)

_SWEEP_RE = re.compile(r"koopman_(?P<latent>pca|diffusion)_r(?P<n>\d+)$")


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def _arm_record(name, latent, algorithm, report):
    """Flatten one report into the shared schema."""
    m = report.get("metrics", {})
    s = report.get("speed", {})
    v1 = m.get("v1_convention", {})
    clean = m.get("clustered_only", {})
    rec = {
        "arm": name,
        "latent": latent,
        "algorithm": algorithm,
        "n_states": m.get("n_states"),
        "n_frames": m.get("n_frames"),
        "noise_frac": m.get("noise_frac"),
        # Both conventions, named. See module docstring.
        "largest_state_frac_v1": v1.get("largest_state_frac"),
        "largest_state_frac_clean": clean.get("largest_state_frac"),
        "state_entropy_v1": v1.get("state_entropy"),
        "state_entropy_clean": clean.get("state_entropy"),
        "state_fracs_clean": clean.get("state_fracs"),
        "noise_speed_ratio": s.get("noise_speed_ratio"),
        "size_speed_rank_corr": s.get("size_speed_rank_corr"),
        "clustered_speed": s.get("clustered_speed"),
        "noise_speed": s.get("noise_speed"),
        # The koopman_* family predates the dedup; see #59.
        "deduped": False,
    }
    inner = report.get("report", {})
    if algorithm == "koopman":
        rec["params"] = {k: inner.get(k) for k in
                         ("n_regions", "knn", "knn_sample", "knn_subsampled",
                          "v_threshold", "coherence_tol", "min_attractor_frac",
                          "global_rank", "global_residual", "min_edge_frac",
                          "min_edge_count")}
        rec["topology"] = {
            "n_attractors": inner.get("n_attractors"),
            "n_fixed_points": inner.get("n_fixed_points"),
            "n_limit_cycles": inner.get("n_limit_cycles"),
            "transition_fraction": inner.get("transition_fraction"),
        }
        # Koopman's -1 is "near a separatrix", not "unclustered". Recorded so a
        # reader does not read the two families' noise columns as one quantity.
        rec["noise_means"] = "near a separatrix (a transition), not unclustered"
    else:
        rec["params"] = {k: inner.get(k) for k in
                         ("min_cluster_size", "min_samples", "hdbscan_sample",
                          "hdbscan_backend")}
        rec["noise_means"] = "unclustered by HDBSCAN"
    return rec


def collect_arms(root):
    arms = {}
    for name, latent, algorithm, sub, fname in ARM_REPORTS:
        report = _load(os.path.join(root, sub, fname))
        if report is None:
            continue
        arms[name] = _arm_record(name, latent, algorithm, report)
    return arms


def collect_latents(root, run_dir=None):
    """PCA vs diffusion: cost, dimensionality, and spectrum."""
    if run_dir is None:
        hits = sorted(glob.glob(os.path.join(root, "run_*",
                                             "latent_comparison.json")))
        if not hits:
            return {}
        run_dir = os.path.dirname(hits[-1])
    data = _load(os.path.join(run_dir, "latent_comparison.json"))
    if data is None:
        return {}
    out = {"source": run_dir}
    for arm in ("pca", "diffusion"):
        if arm not in data:
            continue
        lat = data[arm].get("latent", {})
        out[arm] = {
            "n_components": lat.get("n_components"),
            "explained_variance": lat.get("explained_variance"),
            "explained_variance_ratio": lat.get("explained_variance_ratio"),
            "eigenvalues": lat.get("eigenvalues"),
            "spectral_gap": lat.get("spectral_gap"),
            "epsilon": lat.get("epsilon"),
            "n_landmarks": lat.get("n_landmarks"),
            "n_landmarks_pruned": lat.get("n_landmarks_pruned"),
            "seconds": data[arm].get("seconds"),
            "backend": data[arm].get("backend"),
        }
    return out


def collect_sweep(root):
    """The --n-regions sweep: does the state count track the parameter?

    #57 made the case that a state count is only an output if the parameter
    that could fake it has been varied. This is the answer to that, and it is
    the reason the sweep is collected rather than only r=48.
    """
    rows = []
    for path in sorted(glob.glob(os.path.join(root, "koopman_*_r*"))):
        m = _SWEEP_RE.search(os.path.basename(path))
        if not m:
            continue
        n = int(m.group("n"))
        rep = _load(os.path.join(path, f"koopman_report_r{n}.json"))
        if rep is None:
            continue
        inner, metrics = rep.get("report", {}), rep.get("metrics", {})
        rows.append({
            "latent": m.group("latent"),
            "n_regions": n,
            "n_attractors": inner.get("n_attractors"),
            "n_limit_cycles": inner.get("n_limit_cycles"),
            "transition_fraction": inner.get("transition_fraction"),
            "state_entropy_v1": metrics.get("v1_convention", {}).get("state_entropy"),
            "largest_state_frac_v1": metrics.get("v1_convention", {}).get("largest_state_frac"),
        })
    # r=48 lives in the base directories, not in a koopman_*_r48 dir.
    for latent, sub in (("pca", "koopman_pca"), ("diffusion", "koopman_diffusion")):
        rep = _load(os.path.join(root, sub, "koopman_report_r48.json"))
        if rep is None:
            continue
        inner, metrics = rep.get("report", {}), rep.get("metrics", {})
        rows.append({
            "latent": latent,
            "n_regions": inner.get("n_regions", 48),
            "n_attractors": inner.get("n_attractors"),
            "n_limit_cycles": inner.get("n_limit_cycles"),
            "transition_fraction": inner.get("transition_fraction"),
            "state_entropy_v1": metrics.get("v1_convention", {}).get("state_entropy"),
            "largest_state_frac_v1": metrics.get("v1_convention", {}).get("largest_state_frac"),
        })
    rows.sort(key=lambda r: (r["latent"], r["n_regions"]))
    return rows


def _local_exponent(taus, t2):
    """d log t2 / d log tau over sliding ~4x lag windows.

    A plateau is exponent 0; the trivial large-tau artifact is 1. Reporting the
    exponent rather than the raw curve is what separates "no plateau" from
    "no plateau *and* scale-free", which is the stronger claim.
    """
    out = []
    for i, tau in enumerate(taus):
        j = next((k for k in range(i + 1, len(taus)) if taus[k] >= 4 * tau),
                 None)
        if j is None or not t2[i] or not t2[j]:
            continue
        out.append({
            "tau_lo": taus[i], "tau_hi": taus[j],
            "exponent": (math.log(t2[j]) - math.log(t2[i])) /
                        (math.log(taus[j]) - math.log(taus[i])),
        })
    return out


def collect_timescales(root, run_dir=None):
    """The transfer-operator arm: implied timescales and the §3 gate."""
    if run_dir is None:
        hits = sorted(glob.glob(os.path.join(root, "to_align_*")))
        if not hits:
            return {}
        run_dir = hits[-1]
    out = {"source": run_dir, "arms": {}}
    for key, fname in (("channels", "timescales_channels.json"),
                       ("pose_only", "timescales_pose_only.json")):
        data = _load(os.path.join(run_dir, fname))
        if data is None:
            continue
        taus = data["taus_s"]
        # its_s[i][0] is the stationary process (infinite timescale, stored as
        # null); t2 is index 1.
        t2 = [row[1] if len(row) > 1 else None for row in data["its_s"]]
        t3 = [row[2] if len(row) > 2 else None for row in data["its_s"]]
        diag = data.get("diagnostics", [])
        out["arms"][key] = {
            "n_states": data.get("n_states"),
            "pose_only": data.get("pose_only"),
            "horizon_s": data.get("horizon_s"),
            "median_recording_s": data.get("median_recording_s"),
            "n_recordings": data.get("n_recordings"),
            "taus_s": taus,
            "t2_s": t2,
            "t3_s": t3,
            "t2_over_tau": [(a / b) if a else None for a, b in zip(t2, taus)],
            "counts_symmetrized_t2": [d.get("counts_symmetrized_t2")
                                      for d in diag],
            "local_exponents": _local_exponent(taus, t2),
            "gate": data.get("gate", {}),
            # The estimator's own health. The value of "no plateau" depends
            # entirely on these being clean.
            "health": {
                "min_states_kept": min((d.get("n_states_kept", 0) for d in diag),
                                       default=None),
                "max_dropped_frame_frac": max((d.get("dropped_frame_frac", 0)
                                               for d in diag), default=None),
                "max_leak_frac": max((d.get("leak_frac", 0) for d in diag),
                                     default=None),
                "max_components": max((d.get("n_components", 0) for d in diag),
                                      default=None),
                "any_near_reducible": any(d.get("near_reducible")
                                          for d in diag),
                "min_pairs": min((d.get("n_pairs", 0) for d in diag),
                                 default=None),
            },
            "deduped": True,
        }
    degen = _load(os.path.join(run_dir, "degeneracy.json"))
    if degen:
        out["degeneracy"] = {
            "verdict": degen.get("verdict"),
            "verdict_model": degen.get("verdict_model"),
            "nonlinear_gain": degen.get("nonlinear_gain"),
            "models": {k: {kk: vv for kk, vv in v.items()
                           if not isinstance(vv, (list, dict))}
                       for k, v in degen.get("models", {}).items()},
        }
    rep = _load(os.path.join(run_dir, "recordings_report.json"))
    if rep:
        out["recordings"] = rep
    return out


def collect_moseq(root):
    """The reference arm. Not a VIEB run -- the standard the others clear."""
    for sub in ("transfer_operator/moseq_control",
                "transfer_operator/moseq_control_trunc"):
        data = _load(os.path.join(root, sub, "moseq_control.json"))
        if data is None:
            continue
        # `run()` splits the per-syllable rows out of the contrast dicts into a
        # top-level `rows` map, so the contrast block carries only the summary.
        rows = (data.get("rows", {}).get("retrieval_day1A_vs_day0A")
                or data["contrasts"]["retrieval_day1A_vs_day0A"].get("rows", []))
        best = max(rows, key=lambda r: abs(r["median_diff"])) if rows else {}
        return {
            "source": os.path.join(root, sub),
            "max_frames": data.get("max_frames"),
            "n_recordings": data.get("n_recordings"),
            "n_animals": data.get("n_animals"),
            "n_syllables_tested": data.get("n_syllables_tested"),
            "contrasts": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                          for k, v in data["contrasts"].items()},
            "shuffle_null": data.get("shuffle_null"),
            "arm_profile": data.get("arm_profile"),
            "top": {"syllable": best.get("syllable"),
                    "median_diff": best.get("median_diff"),
                    "rank_biserial": best.get("rank_biserial"),
                    "mean_a": best.get("mean_a"), "mean_b": best.get("mean_b"),
                    "q": best.get("q")},
            "deduped": True,
        }
    return {}


def run(results_root, out_dir):
    out = {
        "results_root": results_root,
        "latents": collect_latents(results_root),
        "arms": collect_arms(results_root),
        "koopman_comparison": _load(os.path.join(results_root,
                                                 "koopman_comparison.json")),
        "n_regions_sweep": collect_sweep(results_root),
        "timescales": collect_timescales(results_root),
        "moseq": collect_moseq(results_root),
    }
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, "model_comparison.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[collect] wrote {dest}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-root",
                    default=os.path.expanduser("~/vieb2-results"))
    ap.add_argument("--out", default=None, help="default <results-root>/_report")
    args = ap.parse_args(argv)
    out_dir = args.out or os.path.join(args.results_root, "_report")
    res = run(args.results_root, out_dir)

    print(f"\n{'arm':22s} {'states':>7s} {'noise':>7s} {'largest':>8s} "
          f"{'H_v1':>7s} {'H_clean':>8s} {'nsr':>7s}")
    for name, a in res["arms"].items():
        print(f"{name:22s} {a['n_states']:7d} {a['noise_frac']:7.3f} "
              f"{a['largest_state_frac_v1']:8.3f} {a['state_entropy_v1']:7.3f} "
              f"{a['state_entropy_clean']:8.3f} {a['noise_speed_ratio']:7.2f}")
    print("\nn_regions sweep (state count vs the parameter that could fake it):")
    for r in res["n_regions_sweep"]:
        print(f"  {r['latent']:9s} r={r['n_regions']:4d} -> "
              f"{r['n_attractors']:4d} attractors, "
              f"{r['n_limit_cycles']} limit cycles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
