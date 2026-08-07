"""Positive control: does Keypoint-MoSeq already find a context-discriminating state?

This runs against output that already exists, so it costs nothing and it bounds
how broken the v2 representation actually is. The argument for running it first:
if a method operating on the *same* pose data already separates the post-shock
condition, then the claim that egocentric alignment destroys the behaviorally
relevant signal is weaker than it looks, and the ordering of everything
downstream should be revisited.

It is also the standard every later method has to clear. A metastable
decomposition that is elegant and does not separate Context A after conditioning
is a negative result.

The design, read off the filenames (298 animals, every animal in every phase):

    CFC  day 0   Context A              conditioning, shock delivered
    CFC  day 1   Context A, No Shock    retrieval test   <-- the contrast
    CFC  day 2   Context C              novel-context generalization
    CFD  day 3-7 Context A and B        discrimination, both contexts per day

Three contrasts, all paired within animal because every animal appears in every
phase, so between-animal variance cancels:

    retrieval      day 1 A/NS  vs  day 0 A     the post-shock effect
    novel_context  day 2 C     vs  day 0 A     does any context change do this?
    discrimination CFD A       vs  CFD B       day-matched, pooled over days

plus a within-animal label-shuffle null that must come back empty.

One honest limit: MoSeq gives per-frame syllables but the shock timestamps are
not in this data, so "post-shock" is necessarily session-level -- day 1 versus
day 0 -- not pre- versus post-shock within the conditioning session. Day 0 mixes
naive and conditioned frames, which makes the primary contrast *conservative*:
a real effect is diluted by the naive portion of the day 0 baseline.

Usage:
    python -m scripts.moseq_control --moseq-dir DIR --out DIR [--json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from representation.recordings import normalize_id, parse_id  # noqa: E402

# Fraction of a session below which a syllable is not worth testing. MoSeq's own
# reporting drops the tail this way; testing a syllable that appears in three
# recordings burns FDR budget on a row nobody can interpret.
MIN_SYLLABLE_FRAC = 0.001
MIN_RECORDINGS_PRESENT = 50


def load_occupancy(moseq_dir, limit=None, max_frames=None):
    """Per-recording syllable occupancy fractions.

    Returns (recording_ids, fields, occupancy, syllables) where `occupancy` is
    (R, S) float64 rows summing to 1 and `fields` is the parsed design per
    recording. Joining is by normalized recording id: MoSeq keeps the full
    `...DLC_Resnet50_...csv` name, which is exactly the mismatch that makes a
    naive cross-method join return nothing.

    `max_frames` truncates every recording to its first N frames. Session
    length is confounded with arm here -- Context A sessions run ~6,302 frames
    against ~5,392 for Context B/C, because the shock protocol needs the extra
    time -- so a syllable whose rate drifts within a session would separate the
    arms with no behavioral difference at all. Truncating to a common length
    removes that explanation.
    """
    import pandas as pd

    paths = sorted(glob.glob(os.path.join(moseq_dir, "*.csv")))
    if limit:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"no MoSeq result csv files under {moseq_dir}")

    ids, fields, counters = [], [], []
    for path in paths:
        rid = normalize_id(path)
        parsed = parse_id(rid)
        if parsed is None:
            continue
        syl = pd.read_csv(path, usecols=["syllable"])["syllable"].to_numpy()
        if max_frames is not None:
            syl = syl[:max_frames]
        syl = syl[syl >= 0]
        if syl.size == 0:
            continue
        ids.append(rid)
        fields.append(parsed)
        counters.append(np.bincount(syl))

    if not ids:
        raise ValueError(f"no parseable MoSeq recordings under {moseq_dir}")

    width = max(c.size for c in counters)
    counts = np.zeros((len(ids), width), dtype=np.float64)
    for i, c in enumerate(counters):
        counts[i, :c.size] = c
    occupancy = counts / counts.sum(axis=1, keepdims=True)

    # Keep only syllables that are actually used, so FDR is not spent on
    # all-zero columns from the sparse tail of the id space.
    present = (occupancy > MIN_SYLLABLE_FRAC).sum(axis=0)
    keep = np.flatnonzero(present >= MIN_RECORDINGS_PRESENT)
    return ids, fields, occupancy[:, keep], keep


def _per_animal(occupancy, fields, mask):
    """Mean occupancy per animal over the recordings selected by `mask`.

    Per-animal rather than per-session, matching the project convention: an
    animal contributing five sessions to one arm and one to the other would
    otherwise weight the arms unequally.
    """
    by_animal = {}
    for i in np.flatnonzero(mask):
        by_animal.setdefault(fields[i]["animal"], []).append(occupancy[i])
    return {a: np.mean(v, axis=0) for a, v in by_animal.items()}


def _benjamini_hochberg(pvals):
    """BH step-up FDR. Hand-rolled to avoid a statsmodels dependency."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n, dtype=float)
    out[order] = np.clip(ranked, 0, 1)
    return out


def paired_contrast(occupancy, fields, mask_a, mask_b, syllables,
                    label="", rng=None):
    """Paired per-animal comparison of syllable occupancy between two arms.

    `mask_b` is the baseline: reported differences are arm A minus arm B, so a
    positive `median_diff` means the syllable is used *more* in arm A.

    Uses Wilcoxon signed-rank rather than a t-test because occupancy fractions
    are bounded, zero-inflated and visibly non-normal. Effect size is reported
    as the matched-pairs rank-biserial correlation, which is the effect size
    Wilcoxon actually implies.
    """
    from scipy import stats

    a_means = _per_animal(occupancy, fields, mask_a)
    b_means = _per_animal(occupancy, fields, mask_b)
    animals = sorted(set(a_means) & set(b_means))
    if len(animals) < 5:
        return {"label": label, "n_animals": len(animals),
                "error": "too few paired animals"}

    A = np.array([a_means[x] for x in animals])
    B = np.array([b_means[x] for x in animals])
    diff = A - B

    rows = []
    for j in range(diff.shape[1]):
        d = diff[:, j]
        if np.allclose(d, 0):
            rows.append({"syllable": int(syllables[j]), "p": 1.0,
                         "median_diff": 0.0, "rank_biserial": 0.0,
                         "mean_a": float(A[:, j].mean()),
                         "mean_b": float(B[:, j].mean())})
            continue
        try:
            _, p = stats.wilcoxon(d, zero_method="wilcox",
                                  alternative="two-sided")
        except ValueError:
            p = 1.0
        # Signed rank-biserial, computed from W+ and W- directly. scipy's
        # two-sided `statistic` is min(W+, W-), so deriving the effect size from
        # it discards the direction and reports every syllable as positive.
        nz = d[d != 0]
        if nz.size:
            ranks = stats.rankdata(np.abs(nz))
            w_pos = float(ranks[nz > 0].sum())
            w_neg = float(ranks[nz < 0].sum())
            rbc = (w_pos - w_neg) / (w_pos + w_neg)
        else:
            rbc = 0.0
        rows.append({"syllable": int(syllables[j]), "p": float(p),
                     "median_diff": float(np.median(d)),
                     "rank_biserial": rbc,
                     "mean_a": float(A[:, j].mean()),
                     "mean_b": float(B[:, j].mean())})

    q = _benjamini_hochberg([r["p"] for r in rows])
    for r, qq in zip(rows, q):
        r["q"] = float(qq)
    rows.sort(key=lambda r: (r["q"], -abs(r["median_diff"])))

    return {"label": label, "n_animals": len(animals),
            "n_syllables": len(rows),
            "n_significant": int(sum(r["q"] < 0.05 for r in rows)),
            "max_abs_median_diff": float(max(abs(r["median_diff"]) for r in rows)),
            "rows": rows}


def shuffle_null(occupancy, fields, mask_a, mask_b, syllables, seed=0,
                 n_repeats=100, observed=None):
    """Within-animal arm swap -- the exact sign-flip null for a paired test.

    Swapping an animal's two arms negates its difference vector, so this is the
    randomization null that Wilcoxon's signed-rank statistic assumes, generated
    empirically rather than trusted. It checks that the pairing, the FDR and the
    per-animal averaging are wired correctly: a contrast that cannot beat its
    own null is not a finding.

    The whole distribution is reported, not the mean. Occupancy fractions are
    compositional -- they sum to 1 across syllables -- so the tests are strongly
    dependent and the rejection count is overdispersed: usually exactly zero,
    occasionally a burst. A mean alone hides both halves of that.
    """
    rng = np.random.default_rng(seed)
    animals = {f["animal"] for f in fields}
    counts = []
    for _ in range(n_repeats):
        a, b = mask_a.copy(), mask_b.copy()
        flip = {x for x in animals if rng.random() < 0.5}
        for i, f in enumerate(fields):
            if f["animal"] in flip and (mask_a[i] or mask_b[i]):
                a[i], b[i] = mask_b[i], mask_a[i]
        res = paired_contrast(occupancy, fields, a, b, syllables, label="null")
        counts.append(res.get("n_significant", 0))

    counts = np.asarray(counts)
    out = {"n_repeats": n_repeats,
           "mean_significant": float(counts.mean()),
           "median_significant": float(np.median(counts)),
           "p90_significant": float(np.percentile(counts, 90)),
           "max_significant": int(counts.max()),
           "frac_repeats_with_none": float((counts == 0).mean())}
    if observed is not None:
        out["observed_significant"] = int(observed)
        out["frac_null_at_or_above_observed"] = float((counts >= observed).mean())
    return out


def arm_profile(occupancy, fields, syllables):
    """Mean occupancy per syllable in every cell of the design.

    The contrasts answer "is there a difference"; this answers "what is the
    time course", which is what makes a difference interpretable. A state that
    is low during conditioning, high at retrieval, only mildly raised in a novel
    context, and increasingly A-vs-B separated across the discrimination days is
    telling a story that a p-value cannot.
    """
    arms = {}
    for i, f in enumerate(fields):
        key = (f"{f['experiment']} d{f['day']} {f['context']}"
               f"{'/NS' if f['no_shock'] else ''}")
        arms.setdefault(key, []).append(occupancy[i])
    return {
        key: {"n": len(rows),
              **{f"s{int(s)}": round(float(np.mean(rows, axis=0)[j]), 5)
                 for j, s in enumerate(syllables)}}
        for key, rows in sorted(arms.items())
    }


def run(moseq_dir, out_dir, limit=None, seed=0, max_frames=None):
    ids, fields, occ, syllables = load_occupancy(moseq_dir, limit=limit,
                                                 max_frames=max_frames)
    exp = np.array([f["experiment"] for f in fields])
    day = np.array([f["day"] for f in fields])
    ctx = np.array([f["context"] for f in fields])
    nos = np.array([f["no_shock"] for f in fields])

    m_day0_A = (exp == "CFC") & (day == 0) & (ctx == "A")
    m_day1_A = (exp == "CFC") & (day == 1) & (ctx == "A") & nos
    m_day2_C = (exp == "CFC") & (day == 2) & (ctx == "C")
    m_cfd_A = (exp == "CFD") & (ctx == "A")
    m_cfd_B = (exp == "CFD") & (ctx == "B")

    contrasts = {
        "retrieval_day1A_vs_day0A": paired_contrast(
            occ, fields, m_day1_A, m_day0_A, syllables,
            "day 1 Context A (retrieval, no shock) vs day 0 Context A"),
        "novel_context_day2C_vs_day0A": paired_contrast(
            occ, fields, m_day2_C, m_day0_A, syllables,
            "day 2 Context C (novel) vs day 0 Context A"),
        "discrimination_cfdA_vs_cfdB": paired_contrast(
            occ, fields, m_cfd_A, m_cfd_B, syllables,
            "CFD Context A vs Context B, pooled over days 3-7"),
    }
    null = shuffle_null(
        occ, fields, m_day1_A, m_day0_A, syllables, seed=seed,
        observed=contrasts["retrieval_day1A_vs_day0A"].get("n_significant"))

    summary = {
        "moseq_dir": moseq_dir,
        "max_frames": max_frames,
        "n_recordings": len(ids),
        "n_syllables_tested": int(len(syllables)),
        "syllables_tested": [int(s) for s in syllables],
        "n_animals": len({f["animal"] for f in fields}),
        "arm_sizes": {"day0_A": int(m_day0_A.sum()),
                      "day1_A_no_shock": int(m_day1_A.sum()),
                      "day2_C": int(m_day2_C.sum()),
                      "cfd_A": int(m_cfd_A.sum()),
                      "cfd_B": int(m_cfd_B.sum())},
        "contrasts": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                      for k, v in contrasts.items()},
        "shuffle_null": null,
        "arm_profile": arm_profile(occ, fields, syllables),
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "moseq_control.json"), "w",
                  encoding="utf-8") as handle:
            json.dump({**summary, "rows": {k: v.get("rows", [])
                                           for k, v in contrasts.items()}},
                      handle, indent=2)
        import csv
        with open(os.path.join(out_dir, "moseq_syllable_contrasts.csv"), "w",
                  newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["contrast", "syllable", "mean_a", "mean_b",
                             "median_diff", "rank_biserial", "p", "q"])
            for name, res in contrasts.items():
                for r in res.get("rows", []):
                    writer.writerow([name, r["syllable"], r["mean_a"],
                                     r["mean_b"], r["median_diff"],
                                     r["rank_biserial"], r["p"], r["q"]])
    return summary, contrasts, null


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--moseq-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-frames", type=int, default=None,
                    help="truncate every recording to its first N frames, "
                         "removing the session-length confound between arms")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    summary, contrasts, null = run(args.moseq_dir, args.out,
                                   limit=args.limit, seed=args.seed,
                                   max_frames=args.max_frames)
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"MoSeq control: {summary['n_recordings']} recordings, "
          f"{summary['n_animals']} animals, "
          f"{summary['n_syllables_tested']} syllables tested")
    print(f"  arms: {summary['arm_sizes']}")
    for name, res in contrasts.items():
        if "error" in res:
            print(f"\n{name}: {res['error']}")
            continue
        print(f"\n{name}  (n={res['n_animals']} animals)")
        print(f"  {res['n_significant']}/{res['n_syllables']} syllables at FDR q<0.05")
        for r in res["rows"][:6]:
            flag = "*" if r["q"] < 0.05 else " "
            print(f"   {flag} syllable {r['syllable']:>3}  "
                  f"{r['mean_b']:.4f} -> {r['mean_a']:.4f}  "
                  f"d={r['median_diff']:+.4f}  rbc={r['rank_biserial']:+.3f}  "
                  f"q={r['q']:.3g}")
    print(f"\nsign-flip null over {null['n_repeats']} repeats: "
          f"median {null['median_significant']:.0f}, "
          f"mean {null['mean_significant']:.2f}, "
          f"p90 {null['p90_significant']:.0f}, "
          f"max {null['max_significant']} significant; "
          f"{null['frac_repeats_with_none']:.0%} of repeats found none")
    if "frac_null_at_or_above_observed" in null:
        print(f"  null reached the observed {null['observed_significant']} "
              f"in {null['frac_null_at_or_above_observed']:.0%} of repeats")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
