"""Section 8 -- positive control against MoSeq output that already exists.

Does any Keypoint-MoSeq syllable shift in Context A after shock? The run is
already on disk, so this costs nothing to ask, and it bounds how broken the
representation actually is:

  * a syllable *does* discriminate -> the representation carries the signal,
    section 2's degeneracy claim is weaker than stated, and the ordering of the
    branch should be revisited.
  * nothing discriminates -> no decomposition built on this representation can
    separate Context A post-shock either, and an elegant one that fails to is a
    negative result, not a partial success.

**Not run on the dev machine.** `results.h5` needs h5py, which is absent there,
cannot be installed under PEP-668, and is in none of the local virtualenvs. It
is meant for the cluster, where the environment is real. Run it there:

    module load python/3.11.4
    source "/path/to/venv/bin/activate"
    which python3                       # verify before trusting anything
    python3 stage8_moseq_control.py \\
        --results "/path/to/2026_07_16-12_47_43/results.h5" \\
        --metadata "/path/to/metadata.csv" \\
        --out "/path/to/moseq_control.json"

Install h5py on the login node; compute nodes have no outbound internet.

Design notes. Usage is a per-recording fraction of frames, so recordings of
different length are comparable. The unit of analysis is the *animal*, not the
recording: several sessions from one mouse are not independent samples, and
pooling them would inflate every p-value's confidence. Recording ids are
normalized the same way the comparator expects, since VIEB and MoSeq key
recordings differently and a silent low-overlap join is the failure mode that
looks like a result.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re

import numpy as np

#: MoSeq and VIEB key recordings differently. Strip the DLC suffix and the
#: extension so the two agree; asserted rather than assumed downstream.
_DLC_SUFFIX = re.compile(r"DLC_.*$|DLC[a-zA-Z]*_resnet.*$", re.IGNORECASE)


def normalize_recording_id(name):
    """Strip directory, extension and any trailing DLC_* marker."""
    stem = os.path.basename(str(name))
    for ext in (".mp4", ".avi", ".h5", ".csv", ".npy"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
    stem = _DLC_SUFFIX.sub("", stem)
    return stem.strip("_. ")


def load_syllables(results_path):
    """{recording_id: syllable array} from a keypoint-MoSeq results.h5.

    The layout has varied between keypoint-MoSeq versions, so the syllable
    dataset is located by name rather than by a fixed path.
    """
    import h5py

    out = {}
    with h5py.File(results_path, "r") as fh:
        for key in fh.keys():
            group = fh[key]
            if not hasattr(group, "keys"):
                continue
            for field in ("syllable", "syllables", "z", "states"):
                if field in group:
                    out[normalize_recording_id(key)] = np.asarray(group[field]).ravel()
                    break
    if not out:
        raise SystemExit(
            f"no syllable dataset found in {results_path!r}; inspect its "
            f"structure and extend load_syllables")
    return out


def load_metadata(metadata_path):
    with open(metadata_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["_id"] = normalize_recording_id(row.get("filename", ""))
    return rows


def usage_matrix(syllables, rows, n_syllables=None):
    """Per-recording syllable usage fractions, aligned to metadata rows."""
    ids = set(syllables)
    matched = [r for r in rows if r["_id"] in ids]
    overlap = len(matched) / max(1, len(rows))
    if not matched:
        raise SystemExit(
            "zero overlap between MoSeq recording ids and metadata; "
            "normalization is wrong -- do not report a comparison")

    if n_syllables is None:
        n_syllables = int(max(int(v.max()) for v in syllables.values() if v.size)) + 1

    usage = np.zeros((len(matched), n_syllables))
    for i, row in enumerate(matched):
        seq = syllables[row["_id"]]
        seq = seq[seq >= 0]
        if seq.size:
            usage[i] = np.bincount(seq, minlength=n_syllables)[:n_syllables] / seq.size
    return usage, matched, n_syllables, overlap


def _fdr(pvalues):
    """Benjamini-Hochberg, without statsmodels."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * p.size / np.arange(1, p.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(ranked)
    out[order] = np.clip(ranked, 0, 1)
    return out


def run(results_path, metadata_path, shock_day=0, context="A", out=None,
        min_overlap=0.5):
    from scipy.stats import mannwhitneyu

    syllables = load_syllables(results_path)
    rows = load_metadata(metadata_path)
    usage, matched, n_syll, overlap = usage_matrix(syllables, rows)

    # Assert overlap before claiming a comparison ran, rather than after.
    if overlap < min_overlap:
        raise SystemExit(
            f"only {overlap:.1%} of metadata rows matched a MoSeq recording; "
            f"below the {min_overlap:.0%} floor. Fix id normalization first -- "
            f"a low-overlap join produces a confident-looking wrong answer.")

    in_context = np.array([r.get("context") == context for r in matched])
    day = np.array([float(r["day"]) if str(r.get("day", "")).strip() else np.nan
                    for r in matched])
    animal = np.array([r.get("animal_id", "") for r in matched])

    pre = in_context & (day <= shock_day)
    post = in_context & (day > shock_day)
    if pre.sum() < 3 or post.sum() < 3:
        raise SystemExit(
            f"not enough Context {context} sessions either side of day "
            f"{shock_day} (pre={int(pre.sum())}, post={int(post.sum())})")

    # Collapse to one value per animal per side: sessions from one mouse are
    # not independent.
    def per_animal(mask):
        table = {}
        for i in np.flatnonzero(mask):
            table.setdefault(animal[i], []).append(usage[i])
        return np.array([np.mean(v, axis=0) for v in table.values()])

    a_pre, a_post = per_animal(pre), per_animal(post)

    results = []
    for s in range(n_syll):
        x, y = a_pre[:, s], a_post[:, s]
        if x.size < 3 or y.size < 3 or (np.all(x == x[0]) and np.all(y == y[0])):
            continue
        stat, p = mannwhitneyu(x, y, alternative="two-sided")
        results.append({"syllable": s, "pre_mean": float(x.mean()),
                        "post_mean": float(y.mean()),
                        "delta": float(y.mean() - x.mean()),
                        "u": float(stat), "p": float(p)})

    if results:
        q = _fdr([r["p"] for r in results])
        for r, value in zip(results, q):
            r["q_bh"] = float(value)
        results.sort(key=lambda r: r["p"])

    significant = [r for r in results if r.get("q_bh", 1.0) < 0.05]
    report = {
        "control": f"any MoSeq syllable shifting in Context {context} post-shock",
        "n_syllables": int(n_syll),
        "n_animals_pre": int(a_pre.shape[0]),
        "n_animals_post": int(a_post.shape[0]),
        "id_overlap": float(overlap),
        "n_significant_fdr05": len(significant),
        "significant": significant[:20],
        "top_by_p": results[:10],
        "verdict": ("SIGNAL PRESENT -- the representation carries it; revisit "
                    "the ordering of section 2"
                    if significant else
                    "NO SIGNAL -- no syllable discriminates Context "
                    f"{context} post-shock at FDR 0.05"),
    }
    print(json.dumps(report, indent=2))
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", required=True, help="keypoint-MoSeq results.h5")
    p.add_argument("--metadata", required=True)
    p.add_argument("--shock-day", type=float, default=0)
    p.add_argument("--context", default="A")
    p.add_argument("--min-overlap", type=float, default=0.5)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    run(a.results, a.metadata, a.shock_day, a.context, a.out, a.min_overlap)


if __name__ == "__main__":
    main()
