#!/usr/bin/env python
"""Sweep the sticky HMM's kappa, and ask whether the dominant syllable is a duration artifact.

This is the cheapest possible improvement to the current best method, and it tests
this branch's hypothesis from the *other* side. keypoint-MoSeq's dwell times are
geometric, ``P(d=n) = p^(n-1)(1-p)``, so raising kappa raises the mean duration and
**never moves the mode off 1 frame**. If the 43% dominant syllable is really a
duration artifact — a single state absorbing everything short — then kappa should
move ``largest_state_frac`` substantially. If it does not, the dominance is about
the representation, not the duration model, and the HSMM will not fix it either.

Either answer is worth having before spending a GPU-week on the HSMM.

    python scripts/kappa_sweep.py --baseline          # measure the existing fit
    python scripts/kappa_sweep.py --kappa 1e5,1e6,1e7,1e8

Refits the AR-HMM at each kappa from the *same* PCA, latent_dim, K and seed as the
baseline, so kappa is the only thing that varies.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

DEFAULT_PROJECT = "~/moseq/luna_demo"
DEFAULT_RESULTS = "~/moseq/luna_demo/2026_07_26-19_54_24/results"
DEFAULT_KAPPAS = (1e5, 1e6, 1e7, 1e8)


def bout_stats(labels: np.ndarray) -> dict:
    """Run-length statistics for one recording's label sequence."""
    if labels.size == 0:
        return {"durations": np.zeros(0, dtype=np.int64)}
    cuts = np.flatnonzero(np.diff(labels)) + 1
    bounds = np.concatenate([[0], cuts, [labels.size]])
    return {"states": labels[bounds[:-1]], "durations": np.diff(bounds)}


def summarize(states: np.ndarray, durations: np.ndarray, fps: float = 30.0) -> dict:
    """The three numbers this sweep is about: occupancy, mode, spread."""
    total = int(durations.sum())
    occ: dict[int, int] = {}
    for s, d in zip(states.tolist(), durations.tolist()):
        occ[s] = occ.get(s, 0) + d
    top = max(occ.items(), key=lambda kv: kv[1]) if occ else (-1, 0)
    counts = np.bincount(durations) if durations.size else np.zeros(1, dtype=np.int64)
    mode = int(np.argmax(counts))
    return {
        "n_states_used": len(occ),
        "largest_state_id": int(top[0]),
        "largest_state_frac": top[1] / total if total else float("nan"),
        "n_bouts": int(durations.size),
        "bout_mode_frames": mode,
        "bout_mode_s": mode / fps,
        "mode_is_one_frame": mode == 1,
        "bout_median_frames": float(np.median(durations)) if durations.size else float("nan"),
        "bout_median_s": float(np.median(durations)) / fps if durations.size else float("nan"),
        "bout_mean_frames": float(durations.mean()) if durations.size else float("nan"),
        "bout_cv": float(durations.std() / durations.mean()) if durations.size else float("nan"),
    }


def measure_baseline(results_dir: str, fps: float = 30.0, limit: int | None = None) -> dict:
    """Read the syllables kpms already wrote. No refit — this is the reference row."""
    import pandas as pd

    files = sorted(glob.glob(os.path.expanduser(results_dir) + "/*.csv"))
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"no result csvs under {results_dir}")
    all_s, all_d = [], []
    for f in files:
        s = pd.read_csv(f, usecols=["syllable"])["syllable"].to_numpy()
        b = bout_stats(s)
        all_s.append(b["states"])
        all_d.append(b["durations"])
    out = summarize(np.concatenate(all_s), np.concatenate(all_d), fps)
    out.update({"kappa": "baseline (as fitted)", "n_recordings": len(files),
                "source": os.path.expanduser(results_dir)})
    return out


def refit_at_kappa(kappa: float, project_dir: str, *, num_iters: int, seed: int,
                   fit_sample_size: int, fps: float = 30.0) -> dict:
    """Refit the AR-HMM at one kappa, everything else held at the baseline."""
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.random as jr
    import keypoint_moseq as kpms
    from jax_moseq.models import arhmm
    from jax_moseq.utils.debugging import convert_data_precision

    project_dir = os.path.expanduser(project_dir)
    config = kpms.load_config(project_dir)
    coordinates, confidences, _ = kpms.load_keypoints(
        config["video_dir"], format="deeplabcut", extension="h5", recursive=True
    )

    keys = sorted(coordinates)
    if fit_sample_size and len(keys) > fit_sample_size:
        import random

        random.Random(seed).shuffle(keys)
        keys = keys[:fit_sample_size]
    coordinates = {k: coordinates[k] for k in keys}
    confidences = {k: confidences[k] for k in keys}

    data, _ = kpms.format_data(coordinates, confidences, **config)
    data = convert_data_precision(data)
    pca = kpms.load_pca(project_dir)
    x = jax.numpy.asarray(kpms.pca.project(pca, data["Y"], **config)[..., : config["latent_dim"]]) \
        if hasattr(kpms, "pca") else None

    model = kpms.init_model(data, pca=pca, **config)
    model = kpms.update_hypparams(model, kappa=float(kappa))

    ar_data = {"x": model["states"]["x"], "mask": data["mask"]}
    m = {"seed": jr.PRNGKey(seed), "states": {"z": model["states"]["z"]},
         "params": {k: model["params"][k] for k in ("betas", "pi", "Ab", "Q")},
         "hypparams": model["hypparams"]}
    for _ in range(num_iters):
        m = arhmm.resample_model(data=ar_data, **m)

    z = np.asarray(m["states"]["z"])
    mask = np.asarray(ar_data["mask"])[:, -z.shape[1]:].astype(bool)
    all_s, all_d = [], []
    for i in range(z.shape[0]):
        row = z[i][mask[i]]
        if row.size == 0:
            continue
        b = bout_stats(row)
        all_s.append(b["states"])
        all_d.append(b["durations"])
    out = summarize(np.concatenate(all_s), np.concatenate(all_d), fps)
    out.update({"kappa": float(kappa), "n_recordings": int(z.shape[0]),
                "num_iters": num_iters, "seed": seed})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", default=DEFAULT_PROJECT)
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--kappa", default=",".join(str(k) for k in DEFAULT_KAPPAS))
    ap.add_argument("--baseline", action="store_true",
                    help="only measure the existing fit, do not refit")
    ap.add_argument("--num-iters", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit-sample-size", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None,
                    help="baseline: read only the first N result csvs")
    ap.add_argument("--out", default="~/vieb-runs/kappa_sweep.json")
    args = ap.parse_args(argv)

    rows = [measure_baseline(args.results, limit=args.limit)]
    print(json.dumps(rows[0], indent=2))

    if not args.baseline:
        for kappa in [float(k) for k in args.kappa.split(",")]:
            print(f"\n--- refitting at kappa = {kappa:g} ---", flush=True)
            try:
                row = refit_at_kappa(
                    kappa, args.project, num_iters=args.num_iters, seed=args.seed,
                    fit_sample_size=args.fit_sample_size,
                )
            except Exception as exc:
                row = {"kappa": float(kappa), "error": f"{type(exc).__name__}: {exc}"}
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)

    out = Path(os.path.expanduser(args.out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))

    print(f"\n{'kappa':>22}  {'largest':>8} {'n_used':>7} {'mode_f':>7} {'median_f':>9} {'cv':>6}")
    for r in rows:
        if "error" in r:
            print(f"{str(r['kappa']):>22}  ERROR {r['error'][:60]}")
            continue
        k = r["kappa"] if isinstance(r["kappa"], str) else f"{r['kappa']:g}"
        print(f"{k:>22}  {r['largest_state_frac']:8.4f} {r['n_states_used']:7d} "
              f"{r['bout_mode_frames']:7d} {r['bout_median_frames']:9.1f} {r['bout_cv']:6.2f}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
