"""Section 2a -- how much locomotor information survives egocentric alignment.

`representation/align.py` subtracts the per-frame centroid and applies a
per-frame rotation, so the v2 aligned space is purely postural: translation and
heading are gone by construction. Delay embedding recovers derivatives of what
was *measured*; it cannot recover what was subtracted before measurement. If
freezing and steady locomotion are degenerate in that input, nothing downstream
can separate them and the rest of the branch is built on sand.

The test: can a classifier told *only about posture* tell a fast frame from a
slow one?

  AUC ~ 0.5  alignment destroyed the freeze/locomote distinction.
  AUC >~ 0.8 locomotion leaves a postural signature, and restoring the explicit
             channels is an improvement rather than a rescue.

**This is a proxy and must be reported as one.** The stated measurement wants
raw pre-alignment centroid speed as the target and the v2 aligned pose as the
predictor. No per-recording raw pose exists in this project -- only training
artifacts -- so both sides are taken from v1's engineered features instead:

  target     `centroid_speed`, computed by v1 from raw pose before any
             alignment, so it is the right quantity
  predictors the translation- and heading-invariant subset only -- 28 pairwise
             keypoint distances plus elongation, rearing_score and head_angle

`body_orientation` is deliberately excluded despite being available: it is a
heading, and heading is exactly what v2's alignment removes. Including it would
flatter the result. Every speed-derived channel is excluded for the obvious
reason.

What this cannot tell you: the answer is about v1's feature space, not v2's
aligned space. They are different bases over related information, so read the
number as an estimate of the effect's size, not as the measurement itself.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featureio import feature_files  # noqa: E402

# Column positions in v1's 51-feature vector; authoritative list lives in the
# features directory's index.json under _meta.feature_names.
POSTURAL = list(range(8, 36)) + [38, 41, 42]     # dist_pair0-27, elongation,
#                                                  rearing_score, head_angle
CENTROID_SPEED = 36
EXCLUDED = {"speed_kp0-7": list(range(0, 8)),
            "centroid_speed": [36],
            "body_orientation (heading, removed by alignment)": [37],
            "angular_velocity": [39],
            "movement_entropy (speed-derived)": [40],
            "all *_window": list(range(43, 51))}


def load_sample(features_dir, n_recordings, n_frames, seed=0):
    """Random frames from random recordings. Order does not matter here.

    Restricted to files matching the dimension index.json declares: the
    directory mixes 51-D and 91-D output, and these column positions are only
    valid for the 51-D layout.
    """
    rng = np.random.default_rng(seed)
    files, file_report = feature_files(features_dir)
    if len(files) > n_recordings:
        files = [files[i] for i in
                 rng.choice(len(files), size=n_recordings, replace=False)]

    X, y, group = [], [], []
    for g, path in enumerate(files):
        arr = np.load(path, mmap_mode="r")
        if arr.shape[0] < 10:
            continue
        take = min(n_frames, arr.shape[0])
        idx = np.sort(rng.choice(arr.shape[0], size=take, replace=False))
        block = np.asarray(arr[idx], dtype=np.float64)
        X.append(block[:, POSTURAL])
        y.append(block[:, CENTROID_SPEED])
        group.append(np.full(take, g))

    return (np.concatenate(X), np.concatenate(y), np.concatenate(group),
            len(files), file_report)


def fit_logistic(X, y, l2=1.0, max_iter=400):
    """L2-regularized logistic regression via L-BFGS.

    Hand-rolled because sklearn is not importable in every environment this has
    to run in; scipy is.
    """
    from scipy.optimize import minimize

    signed = np.where(y > 0, 1.0, -1.0)
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    n = Xb.shape[0]

    def objective(w):
        margin = signed * (Xb @ w)
        # log(1 + exp(-m)) evaluated stably for large |m|.
        loss = np.logaddexp(0.0, -margin).mean() + 0.5 * l2 * (w[:-1] ** 2).sum() / n
        prob = -signed / (1.0 + np.exp(margin))
        grad = Xb.T @ prob / n
        grad[:-1] += l2 * w[:-1] / n
        return loss, grad

    start = np.zeros(Xb.shape[1])
    result = minimize(objective, start, jac=True, method="L-BFGS-B",
                      options={"maxiter": max_iter})
    return result.x


def auc_score(y, score):
    """Rank-based AUC (Mann-Whitney U), ties averaged."""
    y = np.asarray(y) > 0
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(score.size, dtype=np.float64)
    ranks[order] = np.arange(1, score.size + 1)
    # Average ranks within ties so a constant predictor scores exactly 0.5.
    sorted_scores = score[order]
    start = 0
    for stop in range(1, score.size + 1):
        if stop == score.size or sorted_scores[stop] != sorted_scores[start]:
            if stop - start > 1:
                ranks[order[start:stop]] = ranks[order[start:stop]].mean()
            start = stop
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def evaluate(X, label, group, train_groups, test_groups, l2=1.0):
    train = np.isin(group, train_groups)
    test = np.isin(group, test_groups)
    if train.sum() < 50 or test.sum() < 50:
        return np.nan
    mu, sd = X[train].mean(0), X[train].std(0)
    sd[sd == 0] = 1.0
    w = fit_logistic((X[train] - mu) / sd, label[train], l2)
    scores = np.hstack([(X[test] - mu) / sd, np.ones((test.sum(), 1))]) @ w
    return auc_score(label[test], scores)


def run(features_dir, n_recordings=400, n_frames=1500, n_boot=200, seed=0,
        l2=1.0, out=None):
    rng = np.random.default_rng(seed)
    X, speed, group, n_files, file_report = load_sample(
        features_dir, n_recordings, n_frames, seed)

    lo, hi = np.percentile(speed, [100 / 3.0, 200 / 3.0])
    keep = (speed <= lo) | (speed >= hi)
    X, label, group = X[keep], (speed[keep] >= hi).astype(int), group[keep]

    groups = np.unique(group)
    rng.shuffle(groups)
    cut = int(0.8 * groups.size)
    point = evaluate(X, label, group, groups[:cut], groups[cut:], l2)

    # Bootstrap over *recordings*: frames within a recording are strongly
    # dependent, so a frame-level bootstrap would understate the interval.
    draws = []
    for _ in range(n_boot):
        shuffled = rng.permutation(groups)
        cut = int(0.8 * shuffled.size)
        train = rng.choice(shuffled[:cut], size=cut, replace=True)
        value = evaluate(X, label, group, train, shuffled[cut:], l2)
        if np.isfinite(value):
            draws.append(value)

    report = {
        "auc": None if not np.isfinite(point) else float(point),
        "ci95": ([float(np.percentile(draws, 2.5)),
                  float(np.percentile(draws, 97.5))] if draws else None),
        "n_boot": len(draws),
        "n_recordings": int(n_files),
        "feature_files": file_report,
        "n_frames_scored": int(X.shape[0]),
        "n_predictors": len(POSTURAL),
        "tercile_edges": [float(lo), float(hi)],
        "target": "centroid_speed (v1, computed pre-alignment)",
        "predictors": "translation/heading-invariant v1 features only",
        "excluded": {k: v for k, v in EXCLUDED.items()},
        "is_proxy": True,
        "proxy_caveat": (
            "v1 feature space, not the v2 aligned space; no per-recording raw "
            "pose exists in this project. Read as an estimate of effect size, "
            "not as the stated measurement."),
    }

    print(json.dumps(report, indent=2))
    if out:
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--features", required=True, help="directory of *_features.npy")
    p.add_argument("--n-recordings", type=int, default=400)
    p.add_argument("--n-frames", type=int, default=1500)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    run(a.features, a.n_recordings, a.n_frames, a.n_boot, a.seed, a.l2, a.out)


if __name__ == "__main__":
    main()
