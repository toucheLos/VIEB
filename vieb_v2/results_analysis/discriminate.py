"""Score each state-discovery arm on the MoSeq control's own axis.

`scripts/moseq_control.py` asked whether any Keypoint-MoSeq syllable shifts in
Context A after conditioning. It does: 33 of 35, null 0/100. That result is the
standard, and it was never applied to VIEB's own states -- the four arms in
`koopman_comparison.json` are compared to each other on state count, entropy and
noise fraction, which are properties of a partition and say nothing about
whether the partition tracks behavior.

This module runs the identical contrast on `labels.npz` / `koopman_labels.npz`,
reusing `moseq_control`'s statistics verbatim so the numbers are commensurable:
per-animal means, paired Wilcoxon, BH-FDR, and the same within-animal sign-flip
null.

## The index problem, and why it is solvable here

`labels.npz` carries an `index` of `(recording_idx, frame_idx)` and nothing
else -- no path, no recording id. `recordings.py:1` says the map back to a file
"is not merely absent, it is not reconstructible after the fact", because
`load_sessions` silently drops unreadable files into a `skipped` list that
shifts every later index and is never persisted.

That is true in general and false for this particular run, verifiably:
`find_pose_files` returns 4,925 paths, `aligned.npz["lengths"]` has 4,925
entries, and `frame_count(paths[i]) == lengths[i]` for every i. A skipped file
would break that correspondence -- lengths would be short by one and every
subsequent entry misaligned -- so an exact match over all 4,925 is a positive
check that the skip list was empty, not an assumption that it was.

`verify_index` performs that check and **raises** rather than warning. A silent
off-by-one here would attribute every recording's behavior to a neighbouring
animal and still produce plausible-looking p-values.

## Deduplication

The `koopman_*` runs predate the h5/csv dedup (#59), so 1,079 recordings appear
twice. Occupancy is per-recording and then averaged per animal, so a duplicate
does not double-count frames the way it double-counted the stationary measure --
but it does weight those 1,079 sessions twice inside their animal's mean. They
are collapsed here, h5 preferred, matching `recordings.DEFAULT_PREFERENCE`.

Usage:
    python -m results_analysis.discriminate --results-root ~/vieb2-results \
        --out ~/vieb2-results/_report
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from representation import recordings as R  # noqa: E402
from representation.pose_loader import find_pose_files  # noqa: E402
from scripts.moseq_control import (  # noqa: E402
    arm_profile,
    paired_contrast,
    shuffle_null,
)

# Mirrors moseq_control's floor so the two are testing comparably-supported
# states. A state occupying <0.1% of a session in fewer than 50 recordings
# cannot carry a within-animal paired test and only spends FDR budget.
MIN_STATE_FRAC = 0.001
MIN_RECORDINGS_PRESENT = 50

# The arms to score, as (name, directory, labels filename). Koopman's `-1`
# means "near a separatrix" rather than "unclustered", so the two families'
# noise fractions are not the same quantity -- both are reported, neither is
# folded into occupancy.
DEFAULT_ARMS = (
    ("pca-HDBSCAN", "koopman_pca", "labels.npz"),
    ("pca-Koopman", "koopman_pca", "koopman_labels.npz"),
    ("diffusion-HDBSCAN", "koopman_diffusion", "labels.npz"),
    ("diffusion-Koopman", "koopman_diffusion", "koopman_labels.npz"),
)


def verify_index(pose_dir, lengths):
    """Rebuild recording_idx -> path, proving the correspondence as it goes.

    Returns the path list, ordered so `paths[i]` is the recording `index[:,0]`
    calls `i`. Raises if the reconstruction cannot be proven exact.
    """
    paths = find_pose_files(pose_dir)
    if len(paths) != len(lengths):
        raise ValueError(
            f"{len(paths)} pose files under {pose_dir} against "
            f"{len(lengths)} recordings in the checkpoint. The pose directory "
            "has changed since the run; the index cannot be reconstructed.")

    counts = np.array([R.frame_count(p) for p in paths])
    bad = np.flatnonzero(counts != np.asarray(lengths))
    if bad.size:
        raise ValueError(
            f"{bad.size} of {len(paths)} frame counts disagree with the "
            f"checkpoint (first at index {int(bad[0])}: file has "
            f"{int(counts[bad[0]])} frames, checkpoint says "
            f"{int(lengths[bad[0]])}). A file was skipped at run time or has "
            "changed on disk, so every later index is shifted.")
    return paths


def _design(paths):
    """Parsed experimental design per recording, plus the dedup grouping."""
    fields, rids = [], []
    for p in paths:
        rid = R.normalize_id(p)
        parsed = R.parse_id(rid)
        if parsed is None:
            raise ValueError(f"unparseable recording id: {rid}")
        rids.append(rid)
        fields.append(parsed)
    return rids, fields


def state_occupancy(labels_path, n_recordings, max_frames=None):
    """Per-recording occupancy over the non-negative state labels.

    Rows sum to 1 over *assigned* frames. Noise is excluded from the
    denominator rather than made a state: HDBSCAN's `-1` and Koopman's `-1`
    mean different things (unclustered vs near-separatrix), so a shared
    denominator that included them would not compare like with like. The noise
    fraction is returned separately, per recording, and tested on its own.

    `max_frames` truncates every recording to its first N frames, using the
    checkpoint's own `index[:,1]`. Session length is confounded with arm --
    Context A runs ~6,302 frames against ~5,392 for B and C, because the shock
    protocol needs the extra time -- so a state whose rate drifts within a
    session separates the arms with no behavioral difference at all.
    `moseq_control` removes that explanation by truncating; without the same
    treatment the VIEB arms are not being scored on MoSeq's terms.
    """
    z = np.load(labels_path, allow_pickle=True)
    labels = z["labels"].astype(np.int64, copy=False)
    index = z["index"]
    rec = index[:, 0].astype(np.int64, copy=False)

    n_states = int(labels.max()) + 1
    if n_states < 1:
        raise ValueError(f"{labels_path} has no non-noise labels")

    if max_frames is not None:
        keep = index[:, 1] < max_frames
        labels, rec = labels[keep], rec[keep]

    valid = labels >= 0
    flat = rec[valid] * n_states + labels[valid]
    counts = np.bincount(flat, minlength=n_recordings * n_states)
    counts = counts.reshape(n_recordings, n_states).astype(np.float64)

    total = np.bincount(rec, minlength=n_recordings).astype(np.float64)
    assigned = counts.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        occ = np.where(assigned[:, None] > 0, counts / assigned[:, None], 0.0)
        noise = np.where(total > 0, 1.0 - assigned / total, np.nan)

    meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
    return occ, noise, meta


def dedupe_rows(occ, noise, rids, fields, paths,
                prefer=R.DEFAULT_PREFERENCE):
    """Collapse duplicate recordings, keeping the preferred format's row."""
    rank = {ext: i for i, ext in enumerate(prefer)}
    best = {}
    for i, rid in enumerate(rids):
        ext = os.path.splitext(paths[i])[1].lower()
        key = rank.get(ext, len(prefer))
        if rid not in best or key < best[rid][0]:
            best[rid] = (key, i)
    keep = sorted(idx for _, idx in best.values())
    dropped = len(rids) - len(keep)
    return (occ[keep], noise[keep], [rids[i] for i in keep],
            [fields[i] for i in keep], dropped)


def filter_states(occ):
    """Drop states too sparse to carry a paired test. Returns (occ, ids)."""
    present = (occ > MIN_STATE_FRAC).sum(axis=0)
    keep = np.flatnonzero(present >= MIN_RECORDINGS_PRESENT)
    return occ[:, keep], keep


def contrasts(occ, fields, state_ids, seed=0, n_repeats=100):
    """The three MoSeq contrasts, run on a VIEB state set."""
    exp = np.array([f["experiment"] for f in fields])
    day = np.array([f["day"] for f in fields])
    ctx = np.array([f["context"] for f in fields])
    ns = np.array([f["no_shock"] for f in fields])

    day0_A = (exp == "CFC") & (day == 0) & (ctx == "A")
    day1_A = (exp == "CFC") & (day == 1) & (ctx == "A") & ns
    day2_C = (exp == "CFC") & (day == 2) & (ctx == "C")
    cfd_A = (exp == "CFD") & (ctx == "A")
    cfd_B = (exp == "CFD") & (ctx == "B")

    out = {}
    out["retrieval_day1A_vs_day0A"] = paired_contrast(
        occ, fields, day1_A, day0_A, state_ids,
        label="day 1 Context A (retrieval, no shock) vs day 0 Context A")
    out["novel_context_day2C_vs_day0A"] = paired_contrast(
        occ, fields, day2_C, day0_A, state_ids,
        label="day 2 Context C (novel) vs day 0 Context A")
    out["discrimination_cfdA_vs_cfdB"] = paired_contrast(
        occ, fields, cfd_A, cfd_B, state_ids,
        label="CFD Context A vs Context B, pooled over days 3-7")

    null = shuffle_null(occ, fields, day1_A, day0_A, state_ids, seed=seed,
                        n_repeats=n_repeats,
                        observed=out["retrieval_day1A_vs_day0A"]
                        .get("n_significant"))
    profile = arm_profile(occ, fields, state_ids)
    return out, null, profile


def score(contrast, null, n_states_total, n_tested):
    """Collapse one arm's retrieval contrast into comparable scalars.

    `hit_rate` is over *tested* states, `yield_rate` over every state the
    method emitted. They diverge exactly when a method produces many states too
    sparse to test, which is the failure mode a raw significant-count hides:
    43 states of which 6 are testable is not better than 6 of which 6 are.
    """
    rows = contrast.get("rows", [])
    if not rows:
        return {"error": contrast.get("error", "no rows")}
    n_sig = int(contrast["n_significant"])
    best = max(rows, key=lambda r: abs(r["median_diff"]))
    return {
        "n_states_total": int(n_states_total),
        "n_states_tested": int(n_tested),
        "testable_frac": float(n_tested / n_states_total) if n_states_total else 0.0,
        "n_significant": n_sig,
        "hit_rate": float(n_sig / n_tested) if n_tested else 0.0,
        "yield_rate": float(n_sig / n_states_total) if n_states_total else 0.0,
        "max_abs_median_diff": float(abs(best["median_diff"])),
        "max_abs_rank_biserial": float(max(abs(r["rank_biserial"]) for r in rows)),
        "top_state": int(best["syllable"]),
        "top_state_mean_a": float(best["mean_a"]),
        "top_state_mean_b": float(best["mean_b"]),
        "top_state_q": float(best["q"]),
        "null_frac_at_or_above": float(null.get("frac_null_at_or_above_observed",
                                                float("nan"))),
        "null_median_significant": float(null.get("median_significant",
                                                  float("nan"))),
    }


def run_arm(name, run_dir, labels_file, pose_dir, paths, rids, fields,
            seed=0, n_repeats=100, max_frames=None):
    labels_path = os.path.join(run_dir, labels_file)
    if not os.path.exists(labels_path):
        return {"arm": name, "error": f"missing {labels_path}"}

    occ, noise, meta = state_occupancy(labels_path, len(paths),
                                       max_frames=max_frames)
    n_states_total = occ.shape[1]
    occ, noise, rid_k, fields_k, n_dropped = dedupe_rows(
        occ, noise, rids, fields, paths)
    occ_t, state_ids = filter_states(occ)
    if state_ids.size < 2:
        return {"arm": name, "n_states_total": int(n_states_total),
                "n_states_tested": int(state_ids.size),
                "error": "fewer than 2 testable states"}

    cons, null, profile = contrasts(occ_t, fields_k, state_ids, seed=seed,
                                    n_repeats=n_repeats)
    return {
        "arm": name,
        "run_dir": run_dir,
        "labels_file": labels_file,
        "max_frames": max_frames,
        "n_recordings": len(rid_k),
        "n_duplicates_dropped": int(n_dropped),
        "n_animals": len({f["animal"] for f in fields_k}),
        "mean_noise_frac": float(np.nanmean(noise)),
        "score": score(cons["retrieval_day1A_vs_day0A"], null,
                       n_states_total, state_ids.size),
        "contrasts": cons,
        "shuffle_null": null,
        "arm_profile": profile,
        "state_ids_tested": [int(s) for s in state_ids],
        "checkpoint_meta": {k: meta.get(k) for k in
                            ("latent_method", "n_lags", "lag_stride",
                             "min_cluster_size", "hdbscan_backend",
                             "n_regions", "knn_subsampled")
                            if k in meta},
    }


def run(results_root, out_dir, pose_dir, aligned=None, arms=DEFAULT_ARMS,
        seed=0, n_repeats=100, max_frames=None, name="discrimination"):
    aligned = aligned or os.path.join(results_root, "run_20260804_160351",
                                      "aligned.npz")
    lengths = np.load(aligned, allow_pickle=True)["lengths"]
    paths = verify_index(pose_dir, lengths)
    rids, fields = _design(paths)

    out = {"aligned": aligned, "pose_dir": pose_dir,
           "n_recordings_raw": len(paths), "max_frames": max_frames,
           "index_verified": True, "arms": {}}
    # `arm` deliberately, not `name`: the loop variable shadowed the output
    # basename and every run wrote itself to <last arm>.json.
    for arm, sub, labels_file in arms:
        run_dir = os.path.join(results_root, sub)
        print(f"[discriminate] {arm} ...", flush=True)
        out["arms"][arm] = run_arm(arm, run_dir, labels_file, pose_dir,
                                   paths, rids, fields, seed=seed,
                                   n_repeats=n_repeats, max_frames=max_frames)

    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, f"{name}.json")
    with open(dest, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"[discriminate] wrote {dest}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results-root",
                    default=os.path.expanduser("~/vieb2-results"))
    ap.add_argument("--out", default=None,
                    help="default <results-root>/_report")
    ap.add_argument("--pose-dir",
                    default=os.path.expanduser("~/dlc-training/raw_videos"))
    ap.add_argument("--aligned", default=None,
                    help="aligned.npz whose `lengths` define the index order")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-repeats", type=int, default=100,
                    help="shuffle-null repeats (0 skips the null)")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="truncate every recording to its first N frames, so "
                         "session length cannot separate the arms (MoSeq's "
                         "control used 5381)")
    ap.add_argument("--name", default="discrimination",
                    help="output basename under --out")
    args = ap.parse_args(argv)

    out_dir = args.out or os.path.join(args.results_root, "_report")
    res = run(args.results_root, out_dir, args.pose_dir, aligned=args.aligned,
              seed=args.seed, n_repeats=args.n_repeats,
              max_frames=args.max_frames, name=args.name)

    print(f"\n{'arm':22s} {'states':>7s} {'tested':>7s} {'sig':>5s} "
          f"{'hit':>6s} {'yield':>6s} {'max|d|':>7s} {'null>=obs':>9s}")
    for name, a in res["arms"].items():
        s = a.get("score", {})
        if "error" in a or "error" in s:
            print(f"{name:22s} {a.get('error') or s.get('error')}")
            continue
        print(f"{name:22s} {s['n_states_total']:7d} {s['n_states_tested']:7d} "
              f"{s['n_significant']:5d} {s['hit_rate']:6.3f} "
              f"{s['yield_rate']:6.3f} {s['max_abs_median_diff']:7.4f} "
              f"{s['null_frac_at_or_above']:9.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
