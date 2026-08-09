"""Observation channels: postural PCs plus the two the alignment removed.

`align_session` subtracts the per-frame centroid and applies a per-frame
rotation, so the aligned space is purely postural by construction. Translation
and heading are not attenuated in it -- they are gone. Delay embedding recovers
derivatives of what was measured; it cannot recover what was subtracted before
measurement. Freezing and steady locomotion are therefore candidates to be
degenerate in the pose-only arm, which is what `degeneracy()` measures and
what restoring these channels is meant to fix.

Two decisions here are load-bearing:

* Every temporal parameter is in **seconds** and converted through fps at the
  boundary. Luna is 30 fps and Spence is 250 fps; a hardcoded frame count is an
  8x different real-world window between the two rigs.
* Derivatives are taken **per recording**, never across a boundary. `np.gradient`
  is length-preserving and one-sided at the edges, so a recording contributes no
  pair that spans its own start or end, and no recording ever sees another's
  frames.

Smoothing before differentiation is not cosmetic. Measured against MoSeq's
independently estimated centroid on this data, raw per-frame speed agrees at
r=0.61; the same speed smoothed over 5 frames agrees at 0.76 and over 15 frames
at 0.82, while the *positions* agree to a 0.6 px mean offset throughout. The
disagreement is high-frequency keypoint jitter differentiated into the speed
estimate, not a wiring error, so the default window is set well above zero.
"""

from __future__ import annotations

import numpy as np

# Seconds, not frames -- see the module docstring.
DEFAULT_SMOOTH_S = 0.25

CHANNEL_NAMES = ("centroid_speed", "angular_velocity")


def _odd_window(smooth_s, fps):
    """Smoothing window in frames, odd so the filter is exactly centred.

    An even-width moving average has no integer centre and shifts the signal by
    half a frame. That half-frame lag is the same class of error as the
    centred-convolution bug found in ExBias, and it is silent.
    """
    w = int(round(float(smooth_s) * float(fps)))
    if w < 2:
        return 1
    return w + 1 if w % 2 == 0 else w


def smooth(x, window):
    """Centred moving average along axis 0, length-preserving, edge-replicating.

    Hand-rolled rather than pulled from scipy.ndimage so the centring is visible
    in the source and covered by a test that asserts zero phase shift.
    """
    window = int(window)
    if window <= 1 or x.shape[0] <= 1:
        return np.asarray(x, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    half = window // 2
    pad = [(half, half)] + [(0, 0)] * (x.ndim - 1)
    padded = np.pad(x, pad, mode="edge")
    kernel = np.ones(window) / window
    if x.ndim == 1:
        return np.convolve(padded, kernel, mode="valid")
    out = np.empty_like(x)
    for j in range(x.shape[1]):
        out[:, j] = np.convolve(padded[:, j], kernel, mode="valid")
    return out


def speed_and_turn(centroid, theta, fps, smooth_s=DEFAULT_SMOOTH_S):
    """Centroid speed (px/s) and angular velocity (rad/s) for one recording.

    `theta` is the rotation the alignment applied, so heading is -theta; it is
    unwrapped before differentiating, otherwise every crossing of the branch cut
    becomes a spurious 2*pi/dt spike.
    """
    centroid = np.asarray(centroid, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    if centroid.ndim != 2 or centroid.shape[1] != 2:
        raise ValueError(f"centroid must be (T, 2), got {centroid.shape}")
    if theta.shape[0] != centroid.shape[0]:
        raise ValueError(
            f"theta has {theta.shape[0]} frames, centroid has "
            f"{centroid.shape[0]} -- they must describe the same recording")

    fps = float(fps)
    t = centroid.shape[0]
    if t < 2:
        return (np.zeros(t, dtype=np.float64), np.zeros(t, dtype=np.float64))

    w = _odd_window(smooth_s, fps)
    xy = smooth(centroid, w)
    heading = smooth(np.unwrap(-theta), w)

    # gradient is centred in the interior and one-sided at the two ends, so it
    # is length-preserving and never reaches outside this recording.
    vx = np.gradient(xy[:, 0]) * fps
    vy = np.gradient(xy[:, 1]) * fps
    return np.hypot(vx, vy), np.gradient(heading) * fps


def channels_all(frame, fps, smooth_s=DEFAULT_SMOOTH_S):
    """`speed_and_turn` over a list of per-recording (T, 3) frame arrays."""
    out = []
    for i, f in enumerate(frame):
        f = np.asarray(f)
        if f.ndim != 2 or f.shape[1] != 3:
            raise ValueError(
                f"recording {i}: expected (T, 3) [centroid_x, centroid_y, "
                f"theta], got {f.shape}")
        speed, turn = speed_and_turn(f[:, :2], f[:, 2], fps,
                                     smooth_s=smooth_s)
        out.append(np.column_stack([speed, turn]))
    return out


def build(scores, frame, fps, smooth_s=DEFAULT_SMOOTH_S, include=True):
    """Standardise pose PCs and the two restored channels, then concatenate.

    Each channel is scaled to unit variance *before* concatenation. Raw
    concatenation would let whichever block carries the larger numeric scale set
    the geometry: centroid speed is in px/s and runs to hundreds, while a pose PC
    score is order one. The pre-standardisation scales are returned rather than
    only logged, so the ratio that would have applied is auditable after the
    fact.

    Returns (observations, report). `include=False` standardises and returns the
    pose block alone, which is the control arm for the degeneracy test -- the
    two arms then differ only in the presence of the channels, not in scaling.
    """
    if isinstance(scores, np.ndarray):
        raise TypeError(
            "scores must be a list of per-recording arrays, not one "
            "concatenated array -- derivatives must not cross a boundary")
    lengths = [len(s) for s in scores]

    pose = np.concatenate([np.asarray(s, dtype=np.float64) for s in scores])
    pose_scale = pose.std(axis=0)
    pose_scale = np.where(pose_scale > 0, pose_scale, 1.0)
    blocks = [(pose - pose.mean(axis=0)) / pose_scale]
    names = [f"pc{i + 1}" for i in range(pose.shape[1])]
    report = {
        "n_pose_components": int(pose.shape[1]),
        "pose_scale": [float(v) for v in pose_scale],
        "fps": float(fps),
        "smooth_s": float(smooth_s),
        "smooth_frames": int(_odd_window(smooth_s, fps)),
        "channels_included": bool(include),
    }

    if include:
        if frame is None:
            raise ValueError("include=True needs the pose_frame arrays")
        if len(frame) != len(scores):
            raise ValueError(
                f"{len(frame)} frame array(s) but {len(scores)} score array(s)")
        for i, (f, n) in enumerate(zip(frame, lengths)):
            if len(f) != n:
                raise ValueError(
                    f"recording {i}: frame has {len(f)} rows, scores have {n}")
        chan = np.concatenate(channels_all(frame, fps, smooth_s=smooth_s))
        chan_scale = chan.std(axis=0)
        chan_scale = np.where(chan_scale > 0, chan_scale, 1.0)
        blocks.append((chan - chan.mean(axis=0)) / chan_scale)
        names += list(CHANNEL_NAMES)
        report["channel_scale"] = {
            k: float(v) for k, v in zip(CHANNEL_NAMES, chan_scale)}
        report["channel_median"] = {
            k: float(v) for k, v in zip(CHANNEL_NAMES, np.median(chan, axis=0))}
        # The number the standardisation exists to neutralise.
        report["scale_ratio_before_standardising"] = float(
            np.max(chan_scale) / np.max(pose_scale))

    obs = np.concatenate(blocks, axis=1)
    report["names"] = names
    report["n_observations"] = int(obs.shape[1])
    return _split(obs, lengths), report


def _split(flat, lengths):
    """Back into per-recording arrays; the boundary is the whole point."""
    return list(np.split(flat, np.cumsum(lengths)[:-1]))


# ------------------------------------------------------------- degeneracy (2a)

def _fit_predict(model, x_tr, y_tr, x_te, seed):
    """One fold. Standardisation is fitted on train only, never on all rows."""
    if model == "logistic":
        from sklearn.linear_model import LogisticRegression
        mu, sd = x_tr.mean(axis=0), x_tr.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit((x_tr - mu) / sd, y_tr)
        return clf.predict_proba((x_te - mu) / sd)[:, 1]
    if model == "boosted":
        # Trees are scale-invariant, so no standardisation, and none is
        # needed for the comparison to be fair.
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
        clf.fit(x_tr, y_tr)
        return clf.predict_proba(x_te)[:, 1]
    raise ValueError(f"unknown model {model!r}")


def degeneracy(scores, frame, fps, smooth_s=DEFAULT_SMOOTH_S, n_folds=5,
               seed=0, n_boot=200, subsample=400_000, model="logistic"):
    """Can aligned pose alone tell fast frames from slow ones?

    Splits frames into terciles of raw centroid speed and fits a logistic
    regression separating the top tercile from the bottom using *aligned pose
    only*. The middle tercile is discarded: it is the ambiguous band, and
    including it would measure the regression's calibration rather than whether
    the distinction survives alignment at all.

    The held-out split is **by recording**, not by frame. Neighbouring frames of
    one recording are nearly identical, so a random frame split puts near-copies
    of each training frame in the test set and reports an AUC that mostly
    measures autocorrelation. Grouping by recording is what makes the number
    mean "generalises to an unseen animal-session".

    AUC ~ 0.5  -- alignment destroyed the freeze/locomote distinction.
    AUC >~ 0.8 -- locomotion leaves a postural signature, and restoring the
                  channels is an improvement rather than a rescue.

    `model` is worth varying before reading either verdict. A logistic
    regression measures the *linearly decodable* signature, so on its own it
    cannot distinguish "the information is absent" from "the information is
    present but curved". On this data the two differ by about 0.1 AUC, which
    straddles the 0.8 line, so the linear number alone would have been read as
    a stronger degeneracy claim than the data supports.

    The confidence interval is a bootstrap over recordings of the out-of-fold
    predictions, so it reflects between-recording variability rather than the
    frame count, which is arbitrary.
    """
    from sklearn.metrics import roc_auc_score

    if isinstance(scores, np.ndarray):
        raise TypeError("scores must be a list of per-recording arrays")
    if len(frame) != len(scores):
        raise ValueError(
            f"{len(frame)} frame array(s) but {len(scores)} score array(s)")

    rng = np.random.default_rng(seed)
    chan = channels_all(frame, fps, smooth_s=smooth_s)
    speed = np.concatenate([c[:, 0] for c in chan])
    lengths = [len(s) for s in scores]
    rec = np.repeat(np.arange(len(scores)), lengths)
    pose = np.concatenate([np.asarray(s, dtype=np.float64) for s in scores])

    lo, hi = np.percentile(speed, [100 / 3, 200 / 3])
    keep = (speed <= lo) | (speed >= hi)
    x, y, g = pose[keep], (speed[keep] >= hi).astype(int), rec[keep]

    if subsample and x.shape[0] > subsample:
        idx = rng.choice(x.shape[0], size=int(subsample), replace=False)
        idx.sort()
        x, y, g = x[idx], y[idx], g[idx]

    uniq = np.unique(g)
    if uniq.size < 2:
        raise ValueError("need at least 2 recordings to hold one out")
    n_folds = int(min(n_folds, uniq.size))
    fold_of = {r: i % n_folds for i, r in enumerate(rng.permutation(uniq))}
    folds = np.array([fold_of[r] for r in g])

    oof = np.full(y.shape[0], np.nan)
    for f in range(n_folds):
        tr, te = folds != f, folds == f
        if y[tr].min() == y[tr].max() or not te.any():
            continue
        oof[te] = _fit_predict(model, x[tr], y[tr], x[te], seed)

    scored = np.isfinite(oof)
    auc = float(roc_auc_score(y[scored], oof[scored]))

    # Bootstrap over recordings, not frames: the unit that varies is the
    # session, and resampling frames would just re-measure the frame count.
    boot, present = [], np.unique(g[scored])
    for _ in range(int(n_boot)):
        pick = rng.choice(present, size=present.size, replace=True)
        sel = np.concatenate([np.flatnonzero(scored & (g == r)) for r in pick])
        if np.unique(y[sel]).size < 2:
            continue
        boot.append(roc_auc_score(y[sel], oof[sel]))
    ci = ([float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
          if boot else [float("nan"), float("nan")])

    return {
        "auc": auc,
        "model": model,
        "ci95": ci,
        "n_boot_ok": len(boot),
        "n_frames_scored": int(scored.sum()),
        "n_recordings": int(present.size),
        "n_folds": n_folds,
        "speed_tercile_edges": [float(lo), float(hi)],
        "speed_units": "px/s",
        "median_speed_slow": float(np.median(speed[speed <= lo])),
        "median_speed_fast": float(np.median(speed[speed >= hi])),
        "smooth_s": float(smooth_s),
        "grouping": "held out by recording",
    }
