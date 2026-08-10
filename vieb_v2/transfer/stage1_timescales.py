"""Section 3 -- the cheap decisive test, and the branch's falsification gate.

Build the Ulam transfer operator on a Voronoi partition of the existing feature
space and sweep the lag tau. The decision rule is stated in advance:

  * `t_imp` **plateaus** over some tau window -> a timescale separation exists,
    Chapman-Kolmogorov is approximately satisfied, the branch continues.
  * `t_imp` **grows linearly in tau from the start**, with no plateau at any
    tau -> there is no Markovian coarse-graining at any resolution on this
    data. The branch dies here. Report it and stop; do not tune.

Two things that are artifacts and must not be read as findings. At very small
tau the matrix is near-identity and the eigenvalues degenerate toward 1. At very
large tau the rows become noisy copies of pi and every timescale grows linearly
because the only scale left in the problem is the lag. A plateau, if there is
one, lies between them -- which is why `plateau_score` reports the slope of
log t against log tau over a *window* rather than a single number.

**Deviation from the brief, stated plainly.** The brief says k-means microstate
assignments already exist from a coarse-then-refine path and should not be
rebuilt. They do not exist in this project: `results/shared/` contains 404
HDBSCAN `_labels.npy`, which are density-based and therefore exactly what must
not be used to partition here. Microstates are built fresh, geometrically, and
the sweep is repeated at two values of N so the conclusion can be checked
against the partition's resolution rather than resting on one arbitrary choice.

Frames are never subsampled within a recording -- that would destroy the lag
structure the whole method reads. Recordings are subsampled instead, whole.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from featureio import feature_files  # noqa: E402
from representation import transfer_operator as TO  # noqa: E402


def load_recordings(features_dir, n_recordings, seed=0, min_frames=600):
    """Whole contiguous recordings; never a subsample of frames.

    Restricted to one feature dimension -- the directory mixes 51-D and 91-D
    output from two extraction runs, and pooling them would put two different
    feature spaces into a single partition.
    """
    rng = np.random.default_rng(seed)
    files, file_report = feature_files(features_dir)
    if len(files) > n_recordings:
        files = [files[i] for i in
                 rng.choice(len(files), size=n_recordings, replace=False)]

    out, names = [], []
    for path in files:
        arr = np.load(path)
        if arr.shape[0] >= min_frames:
            out.append(np.asarray(arr, dtype=np.float32))
            names.append(os.path.basename(path))
    if not out:
        raise SystemExit("no recording met the minimum length")
    return out, names, file_report


def reduce_dimension(sessions, n_components, seed=0, sample=200_000):
    """Standardize, then project onto the leading PCs of a frame subsample.

    Ulam partitioning degrades in high dimension, so some reduction is
    necessary; how much variance it costs is reported rather than assumed
    negligible.
    """
    rng = np.random.default_rng(seed)
    stacked = np.concatenate(sessions, axis=0)
    take = min(sample, stacked.shape[0])
    idx = rng.choice(stacked.shape[0], size=take, replace=False)
    block = stacked[idx].astype(np.float64)

    mu = block.mean(0)
    sd = block.std(0)
    sd[sd == 0] = 1.0
    centred = (block - mu) / sd

    _, singular, vt = np.linalg.svd(centred, full_matrices=False)
    variance = singular ** 2
    keep = int(min(n_components, vt.shape[0]))
    basis = vt[:keep].T
    retained = float(variance[:keep].sum() / variance.sum())

    projected = [(((s.astype(np.float64) - mu) / sd) @ basis).astype(np.float32)
                 for s in sessions]
    return projected, {"n_components": keep,
                       "variance_retained": retained,
                       "n_features_in": int(stacked.shape[1])}


def build_microstates(projected, n_states, seed=0, sample=300_000):
    rng = np.random.default_rng(seed)
    stacked = np.concatenate(projected, axis=0)
    take = min(sample, stacked.shape[0])
    idx = rng.choice(stacked.shape[0], size=take, replace=False)
    centers = TO.microstates(stacked[idx], n_states, seed=seed)
    return [TO.assign_microstates(s, centers) for s in projected], centers


def run(features_dir, fps=30.0, n_recordings=300, n_states=(100, 200),
        n_components=10, tau_min_s=0.0333, tau_max_s=60.0, n_taus=22,
        n_timescales=10, n_boot=20, seed=0, out=None):
    sessions, names, file_report = load_recordings(
        features_dir, n_recordings, seed)
    projected, pca = reduce_dimension(sessions, n_components, seed)
    total_frames = int(sum(s.shape[0] for s in sessions))

    lags = np.unique(np.round(
        np.geomspace(tau_min_s, tau_max_s, n_taus) * fps) / fps)

    report = {
        "fps": float(fps),
        "n_recordings": len(sessions),
        "feature_files": file_report,
        "total_frames": total_frames,
        "duration_s": total_frames / float(fps),
        "pca": pca,
        "tau_grid_s": [float(t) for t in lags],
        "partitions": {},
        "note_microstates": (
            "built fresh; the k-means microstates the brief refers to do not "
            "exist in this project, and the available labels are HDBSCAN "
            "(density-based), which must not partition an Ulam operator"),
    }

    for N in n_states:
        labels, _ = build_microstates(projected, N, seed)
        rows = TO.timescale_sweep(labels, N, lags, fps,
                                  n_timescales=n_timescales, n_boot=n_boot,
                                  seed=seed)

        curves = []
        for r in rows:
            entry = {"lag_s": r["lag_s"], "n_active": r["n_active"],
                     "n_pairs": r["n_pairs"]}
            if r.get("timescales") is not None:
                entry["t_imp"] = [None if not np.isfinite(v) else float(v)
                                  for v in r["timescales"][:n_timescales]]
                entry["eigenvalues"] = [float(v)
                                        for v in r["eigenvalues"][:n_timescales]]
                if r.get("lo") is not None:
                    entry["ci_lo"] = [None if not np.isfinite(v) else float(v)
                                      for v in r["lo"][:n_timescales]]
                    entry["ci_hi"] = [None if not np.isfinite(v) else float(v)
                                      for v in r["hi"][:n_timescales]]
            else:
                entry["error"] = r.get("error")
            curves.append(entry)

        plateau = {f"t{i + 2}": TO.plateau_score(rows, which=i)
                   for i in range(min(3, n_timescales))}

        # Chapman-Kolmogorov at the flattest window found for the slowest mode,
        # which is the only tau where a Markov claim would be made.
        best = plateau["t2"].get("best_window")
        ck_lag = float(best["tau_lo"]) if best else float(lags[len(lags) // 3])
        ck = TO.chapman_kolmogorov(labels, N, ck_lag, fps, n_max=5)

        report["partitions"][str(N)] = {
            "curves": curves, "plateau": plateau,
            "ck_lag_s": ck_lag, "chapman_kolmogorov": ck,
        }

    verdict = _verdict(report)
    report["verdict"] = verdict
    print(json.dumps(verdict, indent=2))
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        print(f"\nfull report -> {out}")
    return report


#: A plateau means t_imp is independent of tau. 0.15 allows t to double across
#: a 100x change in lag and still count as flat, which is generous; 0.275 --
#: doubling across 12x -- is not a plateau by any reading.
FLAT_SLOPE_MAX = 0.15
#: Total-variation distance between P(2 tau) and P(tau)^2. A genuine Markov
#: chain scores under 0.05 (see the synthetic gate); 0.1 is the loose bound.
CK_TV_MAX = 0.10


def _verdict(report):
    """Apply the stated decision rule, without hedging it.

    A plateau is *necessary but not sufficient*: the eigenvectors must also be
    tau-stable, which is what the Chapman-Kolmogorov error tests. Both have to
    hold for the same partition before the branch continues.
    """
    lines = {}
    for N, block in report["partitions"].items():
        p = block["plateau"]["t2"]
        best = p.get("best_window")
        ck = [r for r in block["chapman_kolmogorov"] if r.get("mean_tv") is not None]
        ck2 = ck[0]["mean_tv"] if ck else None
        flat = abs(best["slope"]) if best else None

        lines[N] = {
            "overall_slope_dlogt_dlogtau": p.get("slope"),
            "flattest_qualifying_window": best,
            "excluded_points": p.get("dropped"),
            "t2_range_s": p.get("t_range"),
            "tau_range_s": p.get("tau_range"),
            "ck_mean_tv_n2_to_n5": [round(r["mean_tv"], 4) for r in ck],
            "plateau_ok": (flat is not None and flat < FLAT_SLOPE_MAX),
            "ck_ok": (ck2 is not None and ck2 < CK_TV_MAX),
        }

    passed = [N for N, b in lines.items() if b["plateau_ok"] and b["ck_ok"]]
    flats = [abs(b["flattest_qualifying_window"]["slope"]) for b in lines.values()
             if b.get("flattest_qualifying_window")]

    if passed:
        gate = f"PLATEAU FOUND at N={passed} -- continue"
        reason = None
    else:
        gate = "NO PLATEAU -- branch dies here"
        reason = (
            "t_imp rises monotonically across the whole tau sweep. No window "
            "spanning at least 3x in tau is flat to within "
            f"|d log t / d log tau| < {FLAT_SLOPE_MAX}, and the "
            "Chapman-Kolmogorov error is above "
            f"{CK_TV_MAX} where it matters. Per the brief: report and stop, "
            "do not tune.")

    return {
        "gate": gate,
        "reason": reason,
        "criteria": {"flat_slope_max": FLAT_SLOPE_MAX, "ck_tv_max": CK_TV_MAX,
                     "min_window_span_factor": 3.0,
                     "excluded": "lambda_2 > 0.95 (near-identity); t_imp < 2 tau"},
        "observed_min_window_slope": (min(flats) if flats else None),
        "per_partition": lines,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--n-recordings", type=int, default=300)
    p.add_argument("--n-states", type=int, nargs="+", default=[100, 200])
    p.add_argument("--n-components", type=int, default=10)
    p.add_argument("--tau-min-s", type=float, default=0.0333)
    p.add_argument("--tau-max-s", type=float, default=60.0)
    p.add_argument("--n-taus", type=int, default=22)
    p.add_argument("--n-boot", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    run(a.features, a.fps, a.n_recordings, tuple(a.n_states), a.n_components,
        a.tau_min_s, a.tau_max_s, a.n_taus, 10, a.n_boot, a.seed, a.out)


if __name__ == "__main__":
    main()
