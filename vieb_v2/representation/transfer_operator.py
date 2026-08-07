"""Transfer (Perron-Frobenius) operator on a Ulam partition.

Every decomposition tried on this data so far has clustered *frames*, and every
one of them was defeated by the same fact: frames arrive at uniform time
intervals, so sampling density along a trajectory goes as 1/|v|. A slow
behavior is dense for a reason that has nothing to do with it being a distinct
behavior. The measured symptoms are `noise_speed_ratio` 9.67 (PCA) / 18.96
(diffusion) and `size_speed_rank_corr` -0.508.

The operator estimated here is a different object from the diffusion map that
failed with `spectral_gap` = 0.0034. That one was built from a Gaussian kernel
on *spatial distances*: proximity defined the operator, so a dense region was a
strongly-connected region by construction. This one is built by counting
*observed transitions* at lag tau: the animal's trajectory defines the operator.
Two different operators over the same points; the first one's flat spectrum
implies nothing about the second's.

What that buys is the separation the previous attempts could not make.
Occupancy lives in `pi` (the stationary distribution) and state identity lives
in the eigenvectors of `P`. A state that is rare and fast has small `pi` but can
still carry its own eigenvector. Density-based clustering has no such split --
there, measure *is* identity, which is why a 20:1 dwell ratio swallowed the
short state whole.

Two honest limits, stated here because they belong in the writeup:

  * Reversibilizing gives an **upper bound** on relaxation times when the
    dynamics is irreversible, and behavior is irreversible. `t_imp` from
    `P_r` is a bound, not an exact timescale.
  * `chi` is `pi`-weighted, so a set carrying little stationary mass
    contributes little to global coherence. This removes the *systematic* bias
    toward slow states; it does not solve sampling of rare fast ones.

Temporal conventions. Every public parameter is in **seconds** and converted to
frames through `fps` at the boundary. Luna is 30fps and Spence is 250fps, so a
hardcoded frame count is an 8x different real-world window between labs. There
is no public entry point here that accepts a frame count.
"""

from __future__ import annotations

import numpy as np

# Below this many microstates a dense symmetric eigensolve is exact, faster and
# far better conditioned than ARPACK, which needs k < N-1 and struggles when k
# approaches N. ARPACK is used above it, as specified.
_DENSE_EIGEN_MAX = 800


# ---------------------------------------------------------------------------
# Seconds -> frames, the only place the conversion happens
# ---------------------------------------------------------------------------

def to_frames(seconds, fps):
    """Convert a duration in seconds to a whole number of frames (>= 1)."""
    if seconds is None:
        raise ValueError("duration must be given in seconds, not frames")
    frames = int(round(float(seconds) * float(fps)))
    return max(1, frames)


def valid_pair_mask(length, lag_frames):
    """Which frames in a recording of `length` can start a lag-tau pair.

    Explicit and returned rather than implied by a slice, so the boundary
    guarantee is assertable after the fact. A pair straddling two recordings is
    not a slow transition, it is a splice between two animals, and at long tau
    it is indistinguishable from the metastable transitions this whole method
    exists to measure.
    """
    mask = np.zeros(int(length), dtype=bool)
    if length > lag_frames:
        mask[: int(length) - int(lag_frames)] = True
    return mask


# ---------------------------------------------------------------------------
# Ulam estimator
# ---------------------------------------------------------------------------

def ulam_counts(labels, n_states, lag_s, fps):
    """C_ij(tau) = # observed transitions i -> j at lag tau.

    labels : list of (T_r,) integer microstate sequences, one per recording.
             A list, never a concatenated array -- see `valid_pair_mask`.

    Counts are accumulated per recording and summed. Negative labels are
    treated as unobserved and drop the pairs that touch them.
    """
    if isinstance(labels, np.ndarray):
        raise TypeError(
            "labels must be a list of per-recording sequences, not one "
            "concatenated array; concatenating would count transitions across "
            "recording boundaries, which at long tau are indistinguishable "
            "from the slow transitions being measured"
        )

    lag = to_frames(lag_s, fps)
    counts = np.zeros((n_states, n_states), dtype=np.float64)
    n_pairs = 0

    for seq in labels:
        seq = np.asarray(seq)
        mask = valid_pair_mask(seq.size, lag)
        if not mask.any():
            continue
        src = seq[:-lag][mask[:seq.size - lag]]
        dst = seq[lag:][mask[:seq.size - lag]]
        good = (src >= 0) & (dst >= 0)
        # The boundary guarantee, asserted rather than trusted.
        assert src.size == max(0, seq.size - lag), "pair mask crossed a boundary"
        if good.any():
            # bincount on the flattened pair index, not np.add.at: the latter
            # is an unbuffered scatter and costs minutes rather than seconds
            # once a real project's millions of pairs are involved.
            flat = src[good].astype(np.intp) * n_states + dst[good]
            counts += np.bincount(
                flat, minlength=n_states * n_states
            ).reshape(n_states, n_states)
            n_pairs += int(good.sum())

    return counts, n_pairs


def row_normalize(counts):
    """P_ij = C_ij / sum_j C_ij, keeping only states seen as a source.

    Returns (P, active) where `active` indexes the retained states. Empty rows
    cannot be normalized and carry no dynamical information; dropping them is
    reported rather than papered over with a uniform row, which would invent
    transitions that were never observed.
    """
    totals = counts.sum(axis=1)
    active = np.flatnonzero(totals > 0)
    if active.size == 0:
        raise ValueError("no state was ever observed as a transition source")

    sub = counts[np.ix_(active, active)]
    totals = sub.sum(axis=1)
    keep = totals > 0
    if not keep.all():
        active = active[keep]
        sub = counts[np.ix_(active, active)]
        totals = sub.sum(axis=1)

    return sub / totals[:, None], active


def stationary_distribution(P, tol=1e-12, max_iter=10_000):
    """pi P = pi, the left eigenvector at eigenvalue 1.

    Solved by power iteration on P^T from the uniform vector. For a count-based
    estimator this converges quickly and, unlike a general eigensolver, cannot
    return a complex or negative-valued vector that would make the
    reversibilization below undefined.
    """
    n = P.shape[0]
    pi = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        nxt = pi @ P
        total = nxt.sum()
        if total <= 0:
            raise ValueError("transition matrix has no stationary distribution")
        nxt = nxt / total
        if np.abs(nxt - pi).sum() < tol:
            pi = nxt
            break
        pi = nxt

    if not np.all(pi > 0):
        floor = pi[pi > 0].min() * 1e-6 if np.any(pi > 0) else 1e-12
        pi = np.maximum(pi, floor)
        pi = pi / pi.sum()
    return pi


def reversibilize(P, pi):
    """P_r = 0.5 * (P + P~), with P~_ij = pi_j P_ji / pi_i.

    P_r satisfies detailed balance with respect to pi, which makes its spectrum
    real and its eigenvectors orthogonal in the pi-weighted inner product. That
    is what licenses the timescale reading; it also means the timescales are an
    upper bound for the true irreversible dynamics.
    """
    tilde = (pi[None, :] * P.T) / pi[:, None]
    return 0.5 * (P + tilde)


# ---------------------------------------------------------------------------
# Spectrum
# ---------------------------------------------------------------------------

def leading_eigen(P_r, pi, k):
    """Top-k eigenpairs of a pi-reversible P_r, largest algebraic first.

    Solved on the symmetrized S = D^{1/2} P_r D^{-1/2}, which is similar to P_r
    and genuinely symmetric, so the eigenvalues come back real instead of with
    numerical imaginary parts that would have to be discarded by hand.
    """
    n = P_r.shape[0]
    k = int(min(k, n))
    root = np.sqrt(pi)
    S = (root[:, None] * P_r) / root[None, :]
    S = 0.5 * (S + S.T)  # kill asymmetry left by finite precision

    if n <= _DENSE_EIGEN_MAX or k >= n - 1:
        from scipy.linalg import eigh

        vals, vecs = eigh(S)
        order = np.argsort(vals)[::-1][:k]
        vals, vecs = vals[order], vecs[:, order]
    else:
        from scipy.sparse.linalg import eigsh

        vals, vecs = eigsh(S, k=k, which="LA")
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]

    return vals, vecs / root[:, None]


def implied_timescales(eigenvalues, lag_s):
    """t_i = -tau / log(lambda_i), in seconds.

    The stationary eigenvalue at 1 has infinite timescale and is returned as
    NaN rather than inf, so it plots and averages without special-casing.
    """
    vals = np.asarray(eigenvalues, dtype=np.float64)
    out = np.full(vals.shape, np.nan)
    usable = (vals > 0) & (vals < 1.0 - 1e-12)
    out[usable] = -float(lag_s) / np.log(vals[usable])
    return out


# ---------------------------------------------------------------------------
# The falsification gate (section 3)
# ---------------------------------------------------------------------------

def timescale_sweep(labels, n_states, lag_seconds, fps, n_timescales=15,
                    n_boot=0, seed=0):
    """Implied timescales across a sweep of tau, with bootstrap CIs.

    Bootstrap resamples whole *recordings*, not frames. Frames within a
    recording are strongly dependent, so a frame-level bootstrap would report
    confidence intervals narrower than the data supports by roughly the square
    root of the correlation time.

    Read the result against the two artifacts that are not findings: at very
    small tau the matrix is near-identity and every eigenvalue degenerates
    toward 1; at very large tau the rows become noisy copies of pi and all
    timescales grow linearly. A plateau, if there is one, lies between them.
    """
    rng = np.random.default_rng(seed)
    n_rec = len(labels)
    rows = []

    for lag_s in lag_seconds:
        record = {"lag_s": float(lag_s), "timescales": None, "eigenvalues": None,
                  "n_active": 0, "n_pairs": 0, "lo": None, "hi": None}
        try:
            counts, n_pairs = ulam_counts(labels, n_states, lag_s, fps)
            P, active = row_normalize(counts)
            pi = stationary_distribution(P)
            vals, _ = leading_eigen(reversibilize(P, pi), pi, n_timescales + 1)
            record["timescales"] = implied_timescales(vals[1:], lag_s)
            record["eigenvalues"] = vals[1:]
            record["n_active"] = int(active.size)
            record["n_pairs"] = int(n_pairs)
        except Exception as exc:                      # noqa: BLE001
            record["error"] = str(exc)
            rows.append(record)
            continue

        if n_boot:
            draws = []
            for _ in range(n_boot):
                pick = rng.integers(0, n_rec, size=n_rec)
                try:
                    c, _ = ulam_counts([labels[i] for i in pick], n_states,
                                       lag_s, fps)
                    Pb, _ = row_normalize(c)
                    pib = stationary_distribution(Pb)
                    vb, _ = leading_eigen(reversibilize(Pb, pib), pib,
                                          n_timescales + 1)
                    draws.append(implied_timescales(vb[1:], lag_s))
                except Exception:                     # noqa: BLE001
                    continue
            if draws:
                width = max(len(d) for d in draws)
                padded = np.full((len(draws), width), np.nan)
                for i, d in enumerate(draws):
                    padded[i, :len(d)] = d
                record["lo"] = np.nanpercentile(padded, 2.5, axis=0)
                record["hi"] = np.nanpercentile(padded, 97.5, axis=0)

        rows.append(record)
    return rows


def plateau_score(rows, which=0, near_identity_max=0.95, min_resolvable=2.0,
                  min_span_factor=3.0):
    """How flat `t_imp` is over tau, and whether it just tracks tau.

    The decision rule this branch lives or dies by: a plateau means a Markovian
    coarse-graining exists at some resolution; `t_imp` growing in tau with no
    plateau anywhere means there is none at any resolution.

    `slope` is d log t / d log tau. A pure "no coarse-graining" result trends
    toward 1, because the only timescale the operator can see is the lag
    itself. A genuine plateau gives slope ~ 0 over a *wide* window.

    Two regions are excluded before any window is fitted, because the brief is
    explicit that neither is a finding:

      * `lambda_2 > near_identity_max` -- at very small tau the matrix is near
        identity and every eigenvalue degenerates toward 1. Flatness here is an
        artifact of the operator barely having moved, and an earlier version of
        this function reported exactly that as the branch's plateau.
      * `t_imp < min_resolvable * tau` -- a timescale shorter than a couple of
        lags is not resolved by that lag.

    Windows must also span at least `min_span_factor` in tau. Without that the
    search returns whichever three adjacent points happen to be flattest, which
    on a monotonically rising curve is guaranteed to find something.
    """
    taus, ts, dropped = [], [], {"near_identity": 0, "unresolvable": 0}
    for r in rows:
        if r.get("timescales") is None or which >= len(r["timescales"]):
            continue
        value = r["timescales"][which]
        if not (np.isfinite(value) and value > 0):
            continue
        eig = r.get("eigenvalues")
        if eig is not None and which < len(eig) and eig[which] > near_identity_max:
            dropped["near_identity"] += 1
            continue
        if value < min_resolvable * r["lag_s"]:
            dropped["unresolvable"] += 1
            continue
        taus.append(r["lag_s"])
        ts.append(value)

    if len(taus) < 4:
        return {"slope": None, "best_window": None, "n_points": len(taus),
                "dropped": dropped}

    taus, ts = np.asarray(taus), np.asarray(ts)
    log_tau, log_t = np.log(taus), np.log(ts)
    overall = float(np.polyfit(log_tau, log_t, 1)[0])
    min_span = float(np.log(min_span_factor))

    best = None
    for start in range(len(taus)):
        for stop in range(start + 3, len(taus) + 1):
            span = float(np.log(taus[stop - 1] / taus[start]))
            if span < min_span:
                continue
            window_slope = float(np.polyfit(log_tau[start:stop],
                                            log_t[start:stop], 1)[0])
            if best is None or abs(window_slope) < abs(best["slope"]):
                best = {"slope": window_slope, "span_factor": float(np.exp(span)),
                        "tau_lo": float(taus[start]), "tau_hi": float(taus[stop - 1]),
                        "t_mean": float(np.mean(ts[start:stop]))}

    return {"slope": overall, "best_window": best, "n_points": len(taus),
            "dropped": dropped, "t_range": [float(ts.min()), float(ts.max())],
            "tau_range": [float(taus.min()), float(taus.max())]}


def chapman_kolmogorov(labels, n_states, lag_s, fps, n_max=5):
    """Compare P(n tau) estimated directly against P(tau)^n.

    A timescale plateau is necessary but not sufficient for Markovianity: the
    eigenvectors have to be tau-stable too. This is the test that catches a
    partition whose timescales look flat while its transition structure is
    still changing with tau.
    """
    counts, _ = ulam_counts(labels, n_states, lag_s, fps)
    P1, active = row_normalize(counts)
    out = []
    for n in range(2, int(n_max) + 1):
        try:
            counts_n, _ = ulam_counts(labels, n_states, lag_s * n, fps)
        except Exception as exc:                      # noqa: BLE001
            out.append({"n": n, "error": str(exc)})
            continue

        sub = np.array([[counts_n[a, b] for b in active] for a in active],
                       dtype=np.float64)
        totals = sub.sum(axis=1)
        keep = np.flatnonzero(totals > 0)
        if keep.size < 2:
            out.append({"n": n, "error": "too few active states at this lag"})
            continue

        direct = sub[np.ix_(keep, keep)]
        direct = direct / direct.sum(axis=1)[:, None]
        predicted = np.linalg.matrix_power(P1, n)[np.ix_(keep, keep)]
        predicted = predicted / np.maximum(predicted.sum(axis=1)[:, None], 1e-300)

        # Total-variation distance per row is the interpretable error here: it
        # is a probability, bounded in [0, 1], unlike a Frobenius norm whose
        # scale depends on how many states there are.
        tv = 0.5 * np.abs(direct - predicted).sum(axis=1)
        out.append({"n": n, "lag_s": float(lag_s * n),
                    "mean_tv": float(tv.mean()), "max_tv": float(tv.max()),
                    "median_tv": float(np.median(tv)), "n_states": int(keep.size)})
    return out


# ---------------------------------------------------------------------------
# Metastable decomposition (section 5d/5e)
# ---------------------------------------------------------------------------

def coherence(P_r, pi, members):
    """chi(S) = sum_{i,j in S} pi_i P_ij / sum_{i in S} pi_i.

    The probability of still being in S one lag later, given stationarity. 1
    means perfectly metastable; for a memoryless chain it collapses to pi(S),
    which is what makes it a usable stopping rule rather than a free parameter.
    """
    members = np.asarray(members)
    if members.size == 0:
        return 0.0
    mass = pi[members].sum()
    if mass <= 0:
        return 0.0
    return float((pi[members, None] * P_r[np.ix_(members, members)]).sum() / mass)


def split_by_phi2(P_r, pi, phi2=None):
    """Split along phi_2 at the threshold maximizing min(chi(S+), chi(S-)).

    phi_2 takes only N distinct values, so every candidate threshold is scanned
    and the global maximum taken. It is a cheap matrix product and there is no
    reason to optimize it into something that can find a local maximum instead.
    """
    n = P_r.shape[0]
    if n < 2:
        return None
    if phi2 is None:
        _, vecs = leading_eigen(P_r, pi, 2)
        if vecs.shape[1] < 2:
            return None
        phi2 = vecs[:, 1]

    order = np.argsort(phi2)
    best = None
    for cut in range(1, n):
        minus, plus = order[:cut], order[cut:]
        score = min(coherence(P_r, pi, minus), coherence(P_r, pi, plus))
        if best is None or score > best["chi"]:
            best = {"chi": score, "threshold": float(phi2[order[cut - 1]]),
                    "minus": minus.copy(), "plus": plus.copy(),
                    "chi_minus": coherence(P_r, pi, minus),
                    "chi_plus": coherence(P_r, pi, plus),
                    "pi_minus": float(pi[minus].sum()),
                    "pi_plus": float(pi[plus].sum())}
    return best


def escape_rate(P_r, pi, members, lag_s):
    """Rate of leaving S per second, from the stationary flux out of it."""
    members = np.asarray(members)
    mass = pi[members].sum()
    if mass <= 0 or members.size == 0:
        return None
    stay = (pi[members, None] * P_r[np.ix_(members, members)]).sum() / mass
    return float((1.0 - stay) / float(lag_s))


def recursive_split(P_r, pi, min_chi=0.8, min_mass=0.01, max_depth=4):
    """Subdivide the largest-measure macrostate, recursively.

    A dominant set is not a failure in this framework -- it is level one of the
    hierarchy, and the previous pipelines' 96-99% largest state was a report of
    depth-1 structure being read as the whole answer.

    Stopping rule, fixed in advance and recorded on every node: stop when the
    best available split has chi below `min_chi`, when a child would carry less
    than `min_mass` of the stationary measure, or at `max_depth`.
    """
    root = {"members": np.arange(P_r.shape[0]), "depth": 0,
            "pi": float(pi.sum()), "chi": coherence(P_r, pi, np.arange(P_r.shape[0])),
            "children": [], "stop_reason": None}
    leaves = [root]

    for _ in range(1024):
        candidates = [n for n in leaves
                      if n["stop_reason"] is None and n["depth"] < max_depth
                      and n["members"].size >= 2]
        if not candidates:
            break
        node = max(candidates, key=lambda n: n["pi"])

        members = node["members"]
        sub_pi = pi[members] / pi[members].sum()
        sub_P = P_r[np.ix_(members, members)]
        sub_P = sub_P / np.maximum(sub_P.sum(axis=1)[:, None], 1e-300)

        best = split_by_phi2(sub_P, sub_pi)
        if best is None:
            node["stop_reason"] = "no split available"
            continue
        if best["chi"] < min_chi:
            node["stop_reason"] = f"chi {best['chi']:.3f} < {min_chi}"
            continue

        child_mass = [best["pi_minus"] * node["pi"], best["pi_plus"] * node["pi"]]
        if min(child_mass) < min_mass:
            node["stop_reason"] = f"child mass {min(child_mass):.4f} < {min_mass}"
            continue

        node["children"] = []
        for side, mass in zip(("minus", "plus"), child_mass):
            child = {"members": members[best[side]], "depth": node["depth"] + 1,
                     "pi": float(mass), "chi": float(best[f"chi_{side}"]),
                     "children": [], "stop_reason": None}
            node["children"].append(child)
        node["split_chi"] = float(best["chi"])
        leaves.remove(node)
        leaves.extend(node["children"])

    return root, leaves


# ---------------------------------------------------------------------------
# Partition (section 5b) and entropy rate (section 5a)
# ---------------------------------------------------------------------------

def microstates(points, n_states, seed=0, n_iter=60, batch_size=4096):
    """k-means++ seeded MiniBatchKMeans into Voronoi microstates.

    Hand-rolled because sklearn is not importable in every environment this has
    to run in, and because the "existing MiniBatchKMeans coarse path" this was
    meant to reuse does not exist in the repository. Density-based clustering is
    deliberately not offered here: the whole point of the Ulam construction is
    that the partition should be geometric and the *operator* should decide
    what is a state.
    """
    points = np.asarray(points, dtype=np.float64)
    rng = np.random.default_rng(seed)
    n = points.shape[0]
    n_states = int(min(n_states, n))

    centers = np.empty((n_states, points.shape[1]))
    centers[0] = points[rng.integers(n)]
    closest = ((points - centers[0]) ** 2).sum(axis=1)
    for i in range(1, n_states):
        total = closest.sum()
        pick = (rng.choice(n, p=closest / total) if total > 0 else rng.integers(n))
        centers[i] = points[pick]
        closest = np.minimum(closest, ((points - centers[i]) ** 2).sum(axis=1))

    counts = np.zeros(n_states)
    for _ in range(n_iter):
        batch = points[rng.choice(n, size=min(batch_size, n), replace=False)]
        assign = assign_microstates(batch, centers)
        for j in np.unique(assign):
            mask = assign == j
            counts[j] += mask.sum()
            centers[j] += (mask.sum() / counts[j]) * (batch[mask].mean(axis=0)
                                                      - centers[j])
    return centers


def assign_microstates(points, centers, chunk=8192):
    out = np.empty(points.shape[0], dtype=int)
    for start in range(0, points.shape[0], chunk):
        block = points[start:start + chunk]
        d = ((block[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        out[start:start + chunk] = d.argmin(axis=1)
    return out


def delay_embed(sessions, k_lags, lag_stride_s, fps):
    """Stack K time-shifted copies per recording, boundary-masked.

    Returns (X_K, index) with index[i] = (recording, frame). Lags never cross a
    boundary: a K-frame stack straddling two recordings is silently garbage and
    is the single easiest way to fabricate a slow timescale.
    """
    if isinstance(sessions, np.ndarray):
        raise TypeError("sessions must be a list of per-recording arrays")

    stride = to_frames(lag_stride_s, fps)
    span = int(k_lags) * stride
    blocks, index = [], []
    for r, scores in enumerate(sessions):
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape[0] <= span:
            continue
        stacked = np.concatenate(
            [scores[span - ell * stride: scores.shape[0] - ell * stride]
             for ell in range(int(k_lags) + 1)], axis=1)
        blocks.append(stacked)
        frames = np.arange(span, span + stacked.shape[0])
        index.append(np.stack([np.full(frames.size, r), frames], axis=1))

    if not blocks:
        raise ValueError(
            f"every recording is shorter than the {span + 1}-frame delay window")
    return np.concatenate(blocks, 0), np.concatenate(index, 0).astype(int)


def entropy_rate(labels, n_states, dt_s, fps):
    """h = -(1/dt) sum_ij pi_i P_ij(dt) log P_ij(dt), in nats per second.

    Used to choose K*: sweep K and take the smallest where h stops rising.
    Beware the finite-size trap -- h is *underestimated* at large N from
    undersampling, and that looks exactly like a plateau, so the h(N) curves
    must be shown per K rather than only the chosen value.
    """
    counts, n_pairs = ulam_counts(labels, n_states, dt_s, fps)
    P, active = row_normalize(counts)
    pi = stationary_distribution(P)
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(P > 0, np.log(P), 0.0)
    return {"h": float(-(pi[:, None] * P * logs).sum() / float(dt_s)),
            "n_active": int(active.size), "n_pairs": int(n_pairs)}


__all__ = [
    "to_frames", "valid_pair_mask", "ulam_counts", "row_normalize",
    "stationary_distribution", "reversibilize", "leading_eigen",
    "implied_timescales", "timescale_sweep", "plateau_score",
    "chapman_kolmogorov", "coherence", "split_by_phi2", "escape_rate",
    "recursive_split", "microstates", "assign_microstates", "delay_embed",
    "entropy_rate",
]
