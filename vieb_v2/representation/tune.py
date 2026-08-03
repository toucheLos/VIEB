"""Choose the delay-embedding window from the data instead of by guess.

Two standard estimators for the lag spacing tau:

  * the first 1/e crossing of the PC-score autocorrelation, and
  * the first minimum of time-delayed mutual information (Fraser & Swinney),
    which unlike autocorrelation responds to nonlinear dependence.

Neither is authoritative. They are reported side by side with the behavioural
timescales they imply in seconds, because the right window is ultimately a
statement about which behaviors should count as one event: too short and a
stride is split, too long and the window straddles a behavior boundary.

One caution carries over from the maths: avoid a tau commensurate with a
periodic behavior's period, where p_{t-tau} is approximately p_t and the extra
lag block adds nothing.
"""

from __future__ import annotations

import numpy as np


def autocorrelation(scores, max_lag=60):
    """Mean autocorrelation across PC dimensions, lags 0..max_lag."""
    scores = np.asarray(scores, dtype=np.float64)
    x = scores - scores.mean(axis=0)
    var = (x ** 2).mean(axis=0)
    keep = var > 1e-12
    if not keep.any():
        return np.zeros(max_lag + 1)
    x, var = x[:, keep], var[keep]

    n = x.shape[0]
    out = np.ones(max_lag + 1)
    for lag in range(1, min(max_lag, n - 1) + 1):
        out[lag] = float(((x[lag:] * x[:-lag]).mean(axis=0) / var).mean())
    return out


def first_crossing(acf, threshold=1.0 / np.e):
    """First lag where the autocorrelation drops below `threshold`."""
    below = np.flatnonzero(acf < threshold)
    return int(below[0]) if below.size else None


def delayed_mutual_information(scores, max_lag=60, bins=16, component=0):
    """Time-delayed mutual information of one PC, in nats, for lags 0..max_lag.

    Uses a fixed 2-D histogram, which is crude but adequate for locating the
    first minimum -- which is all it is used for.
    """
    x = np.asarray(scores, dtype=np.float64)[:, component]
    n = x.size
    out = np.zeros(min(max_lag, n - 2) + 1)
    edges = np.histogram_bin_edges(x, bins=bins)

    for lag in range(out.size):
        a, b = (x, x) if lag == 0 else (x[lag:], x[:-lag])
        joint, _, _ = np.histogram2d(a, b, bins=[edges, edges])
        joint = joint / joint.sum()
        px = joint.sum(axis=1, keepdims=True)
        py = joint.sum(axis=0, keepdims=True)
        nz = joint > 0
        out[lag] = float((joint[nz] * np.log(joint[nz] / (px @ py)[nz])).sum())
    return out


def first_minimum(values):
    """First local minimum of a sequence."""
    for i in range(1, len(values) - 1):
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
            return int(i)
    return None


def _combine(tau_acf, tau_mi, ratio_tol=3.0):
    """Reconcile the two estimators. -> (tau, agreement)

    Averaging is only defensible when they broadly agree. When they disagree by
    more than `ratio_tol` the mean is a number neither estimator supports, so
    the autocorrelation value is preferred -- it is the more conservative and
    more standard choice for a lag spacing -- and the disagreement is surfaced
    for the caller to act on rather than buried.
    """
    available = [t for t in (tau_acf, tau_mi) if t]
    if not available:
        return 1, "no estimate; defaulted to 1"
    if len(available) == 1:
        return max(1, available[0]), "only one estimator returned a value"

    lo, hi = min(available), max(available)
    if lo > 0 and hi / lo > ratio_tol:
        return max(1, tau_acf or lo), (
            f"estimators disagree {hi / lo:.0f}x "
            f"(acf={tau_acf}, mi={tau_mi}); using the autocorrelation value"
        )
    return max(1, int(round(float(np.mean(available))))), "estimators agree"


def suggest(scores_list, fps=30.0, max_lag=60, n_lags=4):
    """Suggest tau (and the resulting window) from PC scores.

    scores_list : list of per-recording (T, q) score arrays
    Returns a dict of estimates -- suggestions to weigh, not a decision.
    """
    pooled = np.concatenate([np.asarray(s) for s in scores_list], axis=0)

    acf = autocorrelation(pooled, max_lag)
    tau_acf = first_crossing(acf)
    mi = delayed_mutual_information(pooled, max_lag)
    tau_mi = first_minimum(mi)

    tau, agreement = _combine(tau_acf, tau_mi)

    return {
        "fps": fps,
        "tau_autocorrelation": tau_acf,
        "tau_mutual_information": tau_mi,
        "tau_suggested": tau,
        "estimator_agreement": agreement,
        "n_lags": n_lags,
        "window_frames": n_lags * tau + 1,
        "window_seconds": (n_lags * tau + 1) / fps,
        "autocorrelation": [float(v) for v in acf],
        "mutual_information": [float(v) for v in mi],
        "reference_timescales_s": {
            "stride_cycle": "0.1-0.2",
            "rear": "~0.5",
            "freeze_bout": "seconds",
        },
    }
