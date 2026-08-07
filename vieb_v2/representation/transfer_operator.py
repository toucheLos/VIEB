"""Perron-Frobenius (transfer) operator on a discretized state space.

Every previous partitioning attempt on this data failed the same way: the
largest state took 42%, then 99.19%, then 96.40% across engineered, PCA and
diffusion-map features with the clusterer held fixed. The measured reason is
that density in frame space is confounded with duration. Frames arrive at
uniform time intervals, so sampling density along a trajectory goes as 1/|v|,
and a clusterer that finds modes of a density is therefore finding *slow*
behavior, not *distinct* behavior. `noise_speed_ratio` = 9.67 and
`size_speed_rank_corr` = -0.508 are two views of that single fact.

The transfer operator dissolves the confound rather than relabelling it. It is
estimated by counting *observed transitions at lag tau*, so occupancy and
identity come out as separate objects: the stationary distribution `pi` says how
much time is spent where, and the eigenvectors say what the slow processes are.
A rare, fast state is still a state; it simply carries little measure.

This is a different operator from the diffusion map's, on the same points. That
one was built from a Gaussian kernel on spatial distances, so proximity defined
it and its spectrum came back flat (spectral_gap 0.0034, all eigenvalues in
[0.935, 0.988]). This one is defined by the animal's own trajectory. The first
result says nothing about the second.

What this module does NOT claim:

- Reversibilizing gives an *upper bound* on relaxation times when the dynamics
  are irreversible, and behavior is irreversible. `implied_timescales` on `P_r`
  is a bound, not an equality. `spectrum_nonreversible` exists so the rotational
  part can be inspected directly rather than being quietly symmetrized away.
- Ulam partitioning degrades in high dimension. If the partition needs heavy
  dimensional reduction to work, that is a real cost, not a detail.
- A plateau in the implied timescales is necessary for a Markovian
  coarse-graining but not sufficient -- the eigenvectors must be tau-stable too,
  which is what `ck_test` is for. And a plateau alone does not imply two
  metastable states: a one-dimensional Ornstein-Uhlenbeck relaxation plateaus
  beautifully and is unimodal. `plateau_gate` therefore reports the spectral gap
  and the eigenvector sign structure beside the plateau, and the caller must
  require all three.

Dense numpy throughout, deliberately. At N <= ~1000 microstates every matrix is
8 MB and `eigh` takes ~0.15 s, exactly convergent. ARPACK would add convergence
failures on the nearly-degenerate spectra that a near-decomposable pooled chain
produces, and carries a trap: for a symmetric operator with eigenvalues in
[-1, 1], `which="LM"` collects eigenvalues near *-1* -- period-2 alternating
modes -- alongside those near +1, silently reporting them as slow processes.
Sorting a dense `eigh` by algebraic value cannot make that mistake.
"""

from __future__ import annotations

import numpy as np

from . import koopman

# Below this, a state's row of the count matrix is too thin for its transition
# probabilities to mean anything, and it drags the connected set with it.
DEFAULT_MIN_COUNT = 1

# A process decaying faster than the lag cannot be resolved by that lag: this is
# the standard "y = tau" line on an implied-timescale plot.
#
# This was 5x on the first pass, on the reasoning that a timescale two lags long
# is fitted from a handful of eigenvalue digits. The Ornstein-Uhlenbeck system in
# the synthetic gate measured that reasoning and found it wrong: OU's timescale
# is flat to within 5% of its analytic value across a thirtyfold lag range, and a
# 5x margin rejected all but the two shortest lags of it -- excluding precisely
# the regime where the estimate is most accurate. 1.0 is the defensible line.
DEFAULT_LAG_MARGIN = 1.0

# Ratio between consecutive implied timescales required to call a process
# spectrally separated. Note what the gate measured: an Ornstein-Uhlenbeck
# process, which has no metastable structure whatsoever, has eigenvalues
# exp(-n*theta*t) and therefore consecutive timescale ratios of exactly
# n+1 : n -- t2/t3 = 2.0. So a gap threshold at or below 2 cannot distinguish
# one-dimensional relaxation from metastability, at any tuning. The eigenvector
# sign structure is the criterion that can, which is why `plateau_gate` requires
# it separately and refuses to call a verdict passed on the gap alone.
DEFAULT_MIN_SPECTRAL_GAP = 2.0


def pair_index(state_ids, tau):
    """Per-recording (s_t, s_{t+tau}) pairs, flattened, with provenance.

    Returns (rows, cols, rec) as int arrays of equal length, where `rec` is the
    index of the recording each pair came from -- the hook the block bootstrap
    resamples on.

    `state_ids` must be a *list* of per-recording arrays, never a concatenated
    block. Passing one array raises, mirroring `koopman.snapshot_pairs` and
    `embed.embed_all`: a pair straddling two recordings is silently garbage, and
    a type error at the boundary is the only enforcement that cannot be
    forgotten. Recordings shorter than the lag contribute nothing rather than
    wrapping around.
    """
    if isinstance(state_ids, np.ndarray):
        raise TypeError(
            "state_ids must be a list of per-recording arrays, not one "
            "concatenated array -- pairs must never cross a recording boundary")
    tau = int(tau)
    if tau < 1:
        raise ValueError(f"tau must be >= 1 frame, got {tau}")

    rows, cols, recs, skipped = [], [], [], 0
    for r, ids in enumerate(state_ids):
        ids = np.asarray(ids)
        if ids.ndim != 1:
            raise ValueError(f"recording {r} must be 1-D, got {ids.shape}")
        if ids.size <= tau:
            skipped += 1
            continue
        rows.append(ids[:-tau])
        cols.append(ids[tau:])
        recs.append(np.full(ids.size - tau, r, dtype=np.int32))

    if not rows:
        return (np.empty(0, np.int64), np.empty(0, np.int64),
                np.empty(0, np.int32))
    return (np.concatenate(rows).astype(np.int64),
            np.concatenate(cols).astype(np.int64),
            np.concatenate(recs))


def count_matrix(rows, cols, n_states, rec=None, rec_weights=None):
    """Dense (n, n) transition counts.

    Sliding windows, not strided: consecutive pairs overlap. That is the correct
    maximum-likelihood estimator for the transition matrix. It does inflate the
    *apparent* sample size, which would matter for likelihood-based error bars --
    but the error bars here come from the recording-block bootstrap, which is
    immune to it. Do not "fix" this by striding; that throws away most of the
    data for no gain.

    `rec_weights` multiplies each recording's contribution, which is how a
    bootstrap replicate is formed without recounting anything.
    """
    n = int(n_states)
    if rows.size == 0:
        return np.zeros((n, n), dtype=np.float64)
    if rows.min() < 0 or cols.min() < 0:
        raise ValueError("negative state ids -- mask noise frames out before "
                         "building the operator, do not pass -1 through")
    if rows.max() >= n or cols.max() >= n:
        raise ValueError(f"state id >= n_states={n}")

    key = rows * n + cols
    if rec_weights is None:
        counts = np.bincount(key, minlength=n * n).astype(np.float64)
    else:
        if rec is None:
            raise ValueError("rec is required when rec_weights is given")
        counts = np.bincount(key, weights=np.asarray(rec_weights, float)[rec],
                             minlength=n * n)
    return counts.reshape(n, n)


def counts_at_lag(state_ids, n_states, tau):
    """`pair_index` then `count_matrix`, for callers that need neither part."""
    rows, cols, rec = pair_index(state_ids, tau)
    return count_matrix(rows, cols, n_states, rec)


def connected_set(counts, min_count=DEFAULT_MIN_COUNT):
    """Largest communicating class of the count matrix, by count *mass*.

    Returns (keep, report). Taking the largest strongly-connected component by
    node count would be a mistake: a 400-node component of states each visited
    twice is not the chain we want to describe. Mass is the right ranking.

    A state with no outgoing counts at this lag is censoring, not absorption --
    it was only ever seen in the final tau frames of the recordings it appears
    in. It falls out of the SCC naturally as a non-terminal singleton, which is
    the correct treatment. Adding a self-loop instead would manufacture an
    eigenvalue at exactly 1, a second ergodic class, and a spurious infinite
    implied timescale.
    """
    counts = np.asarray(counts, dtype=np.float64)
    n = counts.shape[0]
    adjacency = (counts >= float(min_count)).astype(np.int8)
    components, _ = koopman.sccs(adjacency)

    total = counts.sum()
    if not components or total <= 0:
        return np.empty(0, dtype=int), {
            "n_components": 0, "mass_share_of_largest": 0.0,
            "n_states_kept": 0, "n_states_dropped": n,
            "dropped_frame_frac": 1.0, "leak_frac": 1.0}

    masses = [counts[np.ix_(c, c)].sum() for c in components]
    best = int(np.argmax(masses))
    keep = np.array(components[best], dtype=int)

    kept_out = counts[keep, :].sum()
    inside = counts[np.ix_(keep, keep)].sum()
    report = {
        "n_components": len(components),
        "component_sizes": sorted((len(c) for c in components), reverse=True)[:10],
        "component_mass_shares": sorted(
            (float(m / total) for m in masses), reverse=True)[:10],
        "mass_share_of_largest": float(masses[best] / total),
        "n_states_kept": int(keep.size),
        "n_states_dropped": int(n - keep.size),
        "dropped_frame_frac": float(1.0 - kept_out / total),
        "leak_frac": float(1.0 - inside / kept_out) if kept_out > 0 else 1.0,
    }
    return keep, report


def restrict(counts, keep):
    """Submatrix on `keep`, plus the outgoing mass discarded by doing so.

    Slicing and renormalizing conditions on transitions that stay in the set,
    which is the standard MSM choice but is an implicit reflecting boundary.
    `leak_frac` is how much that assumption is being asked to carry.
    """
    counts = np.asarray(counts, dtype=np.float64)
    sub = counts[np.ix_(keep, keep)]
    out = counts[keep, :].sum()
    leak = float(1.0 - sub.sum() / out) if out > 0 else 1.0
    return sub, leak


def row_normalize(counts):
    """Counts -> row-stochastic P. Raises on an empty row.

    Callers must have run `connected_set` first. A zero row would leave P
    substochastic, at which point every eigenvalue interpretation is void; this
    refuses rather than papering over it.
    """
    counts = np.asarray(counts, dtype=np.float64)
    totals = counts.sum(axis=1)
    empty = np.flatnonzero(totals <= 0)
    if empty.size:
        raise ValueError(
            f"{empty.size} state(s) have no outgoing counts (first: "
            f"{empty[:5].tolist()}) -- run connected_set before row_normalize")
    return counts / totals[:, None]


def stationary(P, tol=1e-13, max_iter=200_000):
    """Stationary distribution, by power iteration, cross-checked densely.

    Returns (pi, report). The cross-check is not ceremony: a near-decomposable
    chain -- which is exactly what pooling heterogeneous recordings produces --
    has a second eigenvalue indistinguishable from 1, and power iteration will
    happily converge to an arbitrary mixture of the two ergodic classes without
    complaint. `subdominant_nonreversible` is the number that exposes it.
    """
    P = np.asarray(P, dtype=np.float64)
    n = P.shape[0]
    pi = np.full(n, 1.0 / n)
    iters = max_iter
    for i in range(max_iter):
        nxt = pi @ P
        s = nxt.sum()
        if s <= 0:
            raise ValueError("power iteration collapsed -- P is not stochastic")
        nxt /= s
        if np.abs(nxt - pi).sum() < tol:
            pi = nxt
            iters = i + 1
            break
        pi = nxt

    evals = np.linalg.eigvals(P)
    order = np.argsort(-np.abs(evals))
    subdominant = float(np.abs(evals[order[1]])) if n > 1 else 0.0

    w, v = np.linalg.eig(P.T)
    idx = int(np.argmin(np.abs(w - 1.0)))
    dense = np.real(v[:, idx])
    if dense.sum() < 0:
        dense = -dense
    dense = np.clip(dense, 0, None)
    dense = dense / dense.sum() if dense.sum() > 0 else pi

    report = {
        "iterations": iters,
        "converged": iters < max_iter,
        "l1_vs_dense_eig": float(np.abs(pi - dense).sum()),
        "dominant_eigenvalue_gap": float(abs(1.0 - np.abs(evals[order[0]]))),
        "subdominant_nonreversible": subdominant,
        "near_reducible": bool(subdominant > 1 - 1e-6),
        "pi_min": float(pi.min()),
        "pi_max": float(pi.max()),
    }
    return pi, report


def reversible(P, pi):
    """Symmetric conjugate of the reversibilized operator.

    The reversibilization is P_r = (P + P~)/2 with P~_ij = pi_j P_ji / pi_i, so
    the flux pi_i P_r,ij is the symmetrized flux. Neither P~ nor P_r is ever
    formed: both are ill-conditioned when pi spans orders of magnitude, and what
    the spectrum actually needs is

        S = diag(sqrt(pi)) P_r diag(1/sqrt(pi))

    which is symmetric, so `eigh` gives real eigenvalues and orthogonal vectors
    with no chance of picking up a spurious imaginary part.

    S @ sqrt(pi) == sqrt(pi) exactly, since P_r is stochastic. The caller-facing
    check on that identity costs microseconds and catches a wrong pi, which is
    otherwise a very quiet failure.
    """
    P = np.asarray(P, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    if np.any(pi <= 0):
        raise ValueError("pi has non-positive entries -- restrict to the "
                         "connected set before reversibilizing")
    flux = pi[:, None] * P
    sq = np.sqrt(pi)
    S = (flux + flux.T) / (2.0 * np.outer(sq, sq))
    S = 0.5 * (S + S.T)                     # kill the last ulp of asymmetry

    residual = float(np.abs(S @ sq - sq).max())
    if residual > 1e-8:
        raise ValueError(
            f"reversibilization self-check failed (|S@sqrt(pi) - sqrt(pi)| = "
            f"{residual:.2e}) -- pi is not the stationary distribution of P")
    return S


def spectrum(S, pi, k=10):
    """Leading eigenpairs of the symmetric conjugate, largest algebraic first.

    Sorting by algebraic value rather than magnitude is the whole reason for
    symmetrizing. An eigenvalue near -1 is a period-2 alternating mode: real,
    interesting, and emphatically not a slow process. Magnitude-sorting would
    file it as one.

    Returns eigenvalues (k,), `phi` right eigenvectors of P_r (pi-normalized,
    deterministically signed) and `psi` left eigenvectors. Index 0 is the
    stationary pair, so the first non-trivial process is index 1.
    """
    S = np.asarray(S, dtype=np.float64)
    pi = np.asarray(pi, dtype=np.float64)
    k = int(min(k, S.shape[0]))

    w, u = np.linalg.eigh(S)
    order = np.argsort(w)[::-1][:k]
    w, u = w[order], u[:, order]

    sq = np.sqrt(pi)
    phi = u / sq[:, None]
    psi = u * sq[:, None]

    norms = np.sqrt((pi[:, None] * phi ** 2).sum(axis=0))
    phi = phi / np.where(norms > 0, norms, 1.0)
    # Deterministic sign: the largest-magnitude entry is made positive, so
    # bootstrap replicates and reruns are comparable without post-hoc flipping.
    for j in range(phi.shape[1]):
        lead = int(np.argmax(np.abs(phi[:, j])))
        if phi[lead, j] < 0:
            phi[:, j] *= -1
            psi[:, j] *= -1
    return {"eigenvalues": w, "phi": phi, "psi": psi}


def spectrum_nonreversible(P, k=10):
    """Eigenvalues of P itself, magnitude-sorted, complex retained.

    Reversibilization destroys rotation: a limit cycle's complex pair is
    symmetrized into decay. This exists so that loss is visible rather than
    assumed, and so `koopman.classify_modes` can be pointed at the result.
    """
    evals = np.linalg.eigvals(np.asarray(P, dtype=np.float64))
    return evals[np.argsort(-np.abs(evals))][:int(min(k, evals.size))]


def implied_timescales(eigenvalues, tau, fps, lag_margin=DEFAULT_LAG_MARGIN,
                       horizon_s=None):
    """t_i = -tau*dt / log(lambda_i), with the unresolvable cases marked.

    Returns (t_seconds, flags). t[0] is inf by construction (lambda_1 = 1) and
    must never be plotted as a timescale.

    An eigenvalue <= 0 gives NaN and is *counted*, not clipped: a negative
    eigenvalue means a period-2 alternating mode, which is a real signal about
    the lag being commensurate with an oscillation. Clipping it to a small
    positive number would report a fast process where there is a rhythmic one.
    """
    lam = np.asarray(eigenvalues, dtype=np.float64)
    dt = 1.0 / float(fps)
    lag_s = float(tau) * dt

    t = np.full(lam.shape, np.nan)
    flags = {"n_nonpositive": 0, "n_degenerate": 0, "n_unresolved": 0,
             "n_beyond_horizon": 0}

    for i, l in enumerate(lam):
        if l >= 1.0 - 1e-12:
            t[i] = np.inf
            if i > 0:
                flags["n_degenerate"] += 1
            continue
        if l <= 0.0:
            flags["n_nonpositive"] += 1
            continue
        t[i] = -lag_s / np.log(l)

    finite = np.isfinite(t)
    unresolved = finite & (t < lag_margin * lag_s)
    flags["n_unresolved"] = int(unresolved.sum())
    flags["unresolved"] = unresolved
    if horizon_s is not None:
        beyond = finite & (t > horizon_s)
        flags["n_beyond_horizon"] = int(beyond.sum())
        flags["beyond_horizon"] = beyond
    else:
        flags["beyond_horizon"] = np.zeros_like(finite)
    return t, flags


def operator_at_lag(state_ids, n_states, tau, min_count=DEFAULT_MIN_COUNT,
                    k=10, rows=None, cols=None, rec=None, rec_weights=None):
    """Full pipeline at one lag: counts -> connected set -> P -> pi -> S -> spectrum.

    `rows`/`cols`/`rec` may be supplied to reuse a cached pair index, which is
    what makes the bootstrap affordable -- the pairs do not change between
    replicates, only their weights.
    """
    if rows is None:
        rows, cols, rec = pair_index(state_ids, tau)
    counts = count_matrix(rows, cols, n_states, rec, rec_weights)

    keep, conn = connected_set(counts, min_count=min_count)
    if keep.size < 2:
        return {"tau": int(tau), "ok": False, "reason": "connected set < 2 states",
                "connected": conn}

    sub, leak = restrict(counts, keep)
    P = row_normalize(sub)
    pi, pi_report = stationary(P)
    S = reversible(P, pi)
    spec = spectrum(S, pi, k=k)

    return {"tau": int(tau), "ok": True, "keep": keep, "counts": counts,
            "P": P, "pi": pi, "S": S, "leak_frac": leak,
            "eigenvalues": spec["eigenvalues"], "phi": spec["phi"],
            "psi": spec["psi"], "connected": conn, "stationary": pi_report,
            "n_pairs": int(rows.size)}


def counts_symmetrized_timescale(counts, fps, tau, k=3):
    """Second timescale from the *count*-symmetrized estimator, for comparison.

    C_sym = (C + C^T)/2 with pi proportional to its row sums is the other
    standard reversible estimator. On near-stationary data the two agree to
    several digits. Divergence is a direct, quantitative alarm that the pooled
    data is not one stationary process -- which is the single most likely way
    for this whole analysis to produce a beautiful and meaningless plateau.
    """
    counts = np.asarray(counts, dtype=np.float64)
    sym = 0.5 * (counts + counts.T)
    totals = sym.sum(axis=1)
    keep = np.flatnonzero(totals > 0)
    if keep.size < 2:
        return None
    sym = sym[np.ix_(keep, keep)]
    P = sym / sym.sum(axis=1)[:, None]
    pi = sym.sum(axis=1) / sym.sum()
    S = reversible(P, pi)
    w = spectrum(S, pi, k=k)["eigenvalues"]
    t, _ = implied_timescales(w, tau, fps)
    return float(t[1]) if t.size > 1 else None


def make_taus(fps, tau_max_s, n_tau=28, tau_min_frames=1):
    """Log-spaced lags in frames, deduplicated after rounding.

    Seconds in, frames out, at the boundary and nowhere else. A hardcoded frame
    count would mean an 8x different real-world window between a 30 fps and a
    250 fps rig.
    """
    lo = max(1, int(tau_min_frames)) / float(fps)
    taus_s = np.geomspace(lo, float(tau_max_s), int(n_tau))
    frames = np.unique(np.maximum(1, np.rint(taus_s * float(fps))).astype(int))
    return frames


def its_sweep(state_ids, n_states, taus, fps, k=10, min_count=DEFAULT_MIN_COUNT,
              lag_margin=DEFAULT_LAG_MARGIN, horizon_s=None):
    """Implied timescales across a lag sweep.

    Returns a dict with `its` of shape (n_tau, k) in seconds, plus the
    per-lag connected-set and stationary diagnostics. Two regions of the curve
    are artifacts and must not be read as findings: at very small tau the matrix
    is near-identity and every eigenvalue degenerates toward 1, and at very
    large tau the rows become noisy copies of pi and all timescales grow
    linearly. A plateau, if there is one, lives between them.
    """
    taus = np.asarray(taus, dtype=int)
    its = np.full((taus.size, k), np.nan)
    diagnostics, ok = [], np.zeros(taus.size, dtype=bool)

    for i, tau in enumerate(taus):
        res = operator_at_lag(state_ids, n_states, int(tau),
                              min_count=min_count, k=k)
        if not res["ok"]:
            diagnostics.append({"tau": int(tau), "ok": False,
                                "reason": res.get("reason")})
            continue
        t, flags = implied_timescales(res["eigenvalues"], tau, fps,
                                      lag_margin=lag_margin,
                                      horizon_s=horizon_s)
        its[i, :t.size] = t
        ok[i] = True
        diagnostics.append({
            "tau": int(tau), "tau_s": float(tau) / fps, "ok": True,
            "n_pairs": res["n_pairs"], "leak_frac": res["leak_frac"],
            "n_states_kept": res["connected"]["n_states_kept"],
            "n_states_dropped": res["connected"]["n_states_dropped"],
            "dropped_frame_frac": res["connected"]["dropped_frame_frac"],
            "mass_share_of_largest": res["connected"]["mass_share_of_largest"],
            "n_components": res["connected"]["n_components"],
            "subdominant_nonreversible":
                res["stationary"]["subdominant_nonreversible"],
            "near_reducible": res["stationary"]["near_reducible"],
            "n_nonpositive_eigenvalues": flags["n_nonpositive"],
            "n_unresolved": flags["n_unresolved"],
            "n_beyond_horizon": flags["n_beyond_horizon"],
            "counts_symmetrized_t2": counts_symmetrized_timescale(
                res["counts"], fps, tau),
        })

    return {"taus": taus, "taus_s": taus.astype(float) / fps, "its": its,
            "ok": ok, "diagnostics": diagnostics, "fps": float(fps),
            "lag_margin": lag_margin, "horizon_s": horizon_s}


def _plateau_for_series(taus_s, t, window_ratio, tol, lag_margin, horizon_s):
    """Widest lag window over which one timescale is flat and resolved."""
    n = taus_s.size
    best = None
    for i in range(n):
        if not np.isfinite(t[i]):
            continue
        for j in range(n - 1, i, -1):
            if not np.isfinite(t[j]):
                continue
            ratio = taus_s[j] / taus_s[i]
            if ratio < window_ratio:
                break
            seg_t, seg_tau = t[i:j + 1], taus_s[i:j + 1]
            if not np.all(np.isfinite(seg_t)) or np.any(seg_t <= 0):
                continue
            if np.any(seg_t < lag_margin * seg_tau):
                continue
            if horizon_s is not None and np.any(seg_t > horizon_s):
                continue
            if seg_t.max() / seg_t.min() > 1.0 + tol:
                continue
            cand = {"tau_lo_s": float(seg_tau[0]), "tau_hi_s": float(seg_tau[-1]),
                    "tau_ratio": float(ratio),
                    "plateau_s": float(np.median(seg_t)),
                    "flatness": float(seg_t.max() / seg_t.min()),
                    "n_lags": int(j - i + 1)}
            if best is None or cand["tau_ratio"] > best["tau_ratio"]:
                best = cand
            break
    return best


def plateau_gate(sweep, processes=(1, 2, 3), window_ratio=4.0, tol=0.20,
                 min_spectral_gap=DEFAULT_MIN_SPECTRAL_GAP,
                 phi_sign_structure=None):
    """The falsification criterion, evaluated.

    A plateau alone is not enough and this refuses to pretend otherwise. An
    Ornstein-Uhlenbeck process has one genuine relaxation timescale, plateaus
    cleanly, and has no metastable structure whatsoever -- its second
    eigenvector is monotone in the coordinate, not sign-split. So the gate
    reports three things and the verdict requires all of them:

      plateau        t_i flat over a lag window spanning at least `window_ratio`
      spectral gap   lambda_i / lambda_{i+1} separating the process from the bulk
      sign structure phi_i genuinely splitting the state space, not ramping

    `phi_sign_structure` is supplied by the caller (it needs the partition's
    geometry, which this module does not have). Absent it, the gate reports
    `None` for that condition and `passed` stays conditional.
    """
    taus_s, its = sweep["taus_s"], sweep["its"]
    lag_margin = sweep.get("lag_margin", DEFAULT_LAG_MARGIN)
    horizon_s = sweep.get("horizon_s")

    out = {"processes": {}, "window_ratio": window_ratio, "tol": tol}
    any_plateau = False
    for p in processes:
        if p >= its.shape[1]:
            continue
        plateau = _plateau_for_series(taus_s, its[:, p], window_ratio, tol,
                                      lag_margin, horizon_s)
        entry = {"plateau": plateau, "has_plateau": plateau is not None}
        if plateau is not None:
            any_plateau = True
            idx = int(np.argmin(np.abs(taus_s - plateau["tau_lo_s"])))
            if p + 1 < its.shape[1] and np.isfinite(its[idx, p + 1]) and \
                    its[idx, p + 1] > 0:
                entry["timescale_ratio_to_next"] = float(
                    its[idx, p] / its[idx, p + 1])
                entry["has_spectral_gap"] = bool(
                    entry["timescale_ratio_to_next"] >= min_spectral_gap)
            else:
                entry["timescale_ratio_to_next"] = None
                entry["has_spectral_gap"] = None
        entry["has_sign_structure"] = (
            None if phi_sign_structure is None else bool(phi_sign_structure.get(p)))
        entry["passed"] = bool(
            entry["has_plateau"]
            and entry.get("has_spectral_gap") is not False
            and entry.get("has_sign_structure") is not False)
        out["processes"][p] = entry

    out["any_plateau"] = any_plateau
    out["passed"] = any(e["passed"] for e in out["processes"].values())
    if not any_plateau:
        out["verdict"] = ("no plateau at any lag: there is no Markovian "
                          "coarse-graining at this resolution on this data")
    elif out["passed"]:
        out["verdict"] = "plateau with spectral separation"
    else:
        out["verdict"] = ("plateau present but without spectral separation or "
                          "sign structure -- consistent with a single "
                          "relaxation, not with metastability")
    return out


def ck_test(state_ids, n_states, tau, fps, n_max=5, min_count=DEFAULT_MIN_COUNT,
            min_length=None, max_lag_frac=0.28):
    """Chapman-Kolmogorov: P(n*tau) estimated directly vs P(tau)^n.

    A plateau in the implied timescales is necessary for Markovianity but not
    sufficient -- the eigenvectors have to be tau-stable too, and this is what
    tests that.

    The feasibility guard is not optional. With ~3-minute recordings, n=5 at
    tau=60 s needs a 9,000-frame lag against a 6,321-frame maximum: the count
    matrix comes back empty and surfaces as a confusing zero-row error rather
    than as "you asked for something the data cannot answer". Refuse instead,
    and say which number was the binding one.

    Error is the pi-weighted mean total variation between the two matrices:
    bounded in [0, 1] so a threshold means the same thing at any N, weighted so
    a microstate the animal never occupies cannot dominate.
    """
    base = operator_at_lag(state_ids, n_states, tau, min_count=min_count)
    if not base["ok"]:
        return {"ok": False, "reason": base.get("reason")}

    keep, P, pi = base["keep"], base["P"], base["pi"]
    if min_length is None:
        min_length = min(int(np.asarray(s).size) for s in state_ids)
    budget = int(max_lag_frac * min_length)

    rows = []
    for n in range(2, int(n_max) + 1):
        lag = int(tau) * n
        if lag > budget:
            rows.append({"n": n, "lag": lag, "ok": False,
                         "reason": (f"n*tau={lag} exceeds {max_lag_frac:g} x "
                                    f"min recording length {min_length} "
                                    f"({budget} frames)")})
            continue
        counts_n = counts_at_lag(state_ids, n_states, lag)
        sub = counts_n[np.ix_(keep, keep)]
        totals = sub.sum(axis=1)
        live = totals > 0
        if live.sum() < 2:
            rows.append({"n": n, "lag": lag, "ok": False,
                         "reason": "no counts on the base connected set"})
            continue
        direct = np.zeros_like(sub)
        direct[live] = sub[live] / totals[live, None]
        predicted = np.linalg.matrix_power(P, n)

        row_tv = 0.5 * np.abs(direct - predicted).sum(axis=1)
        w = pi[live] / pi[live].sum()
        rows.append({
            "n": n, "lag": lag, "ok": True,
            "lag_s": lag / float(fps),
            "err": float((w * row_tv[live]).sum()),
            "err_max_row": float(row_tv[live].max()),
            "n_pairs": int(sub.sum()),
            "n_rows_scored": int(live.sum()),
        })

    return {"ok": True, "tau": int(tau), "tau_s": tau / float(fps),
            "min_length": int(min_length), "budget_frames": budget,
            "n_states_kept": int(keep.size), "rows": rows}


def holdout_ck(state_ids, n_states, tau, fps, n_max=5, seed=0,
               min_count=DEFAULT_MIN_COUNT, max_lag_frac=0.28):
    """CK with P(tau) fitted on half the recordings and tested on the other half.

    The sharpest single discriminator available against the failure mode this
    analysis is most exposed to. Pooling 3,846 recordings that span contexts,
    days, shocked and unshocked animals assumes one stationary process generated
    all of them. When it did not, the pooled chain is near-decomposable --
    sub-populations occupy different regions and the transitions between them
    are never observed -- and that manufactures a slow apparent process with a
    convincing plateau. A pooling artifact does not transfer across a split of
    the recordings; a real process does.
    """
    rng = np.random.default_rng(seed)
    n_rec = len(state_ids)
    if n_rec < 4:
        return {"ok": False, "reason": "need at least 4 recordings to split"}
    perm = rng.permutation(n_rec)
    half = n_rec // 2
    a = [state_ids[i] for i in perm[:half]]
    b = [state_ids[i] for i in perm[half:]]

    fit = operator_at_lag(a, n_states, tau, min_count=min_count)
    test = operator_at_lag(b, n_states, tau, min_count=min_count)
    if not fit["ok"] or not test["ok"]:
        return {"ok": False, "reason": "one half has no usable connected set"}

    shared = np.intersect1d(fit["keep"], test["keep"])
    if shared.size < 2:
        return {"ok": False, "reason": "halves share fewer than 2 states"}

    pos = {s: i for i, s in enumerate(fit["keep"])}
    idx = np.array([pos[s] for s in shared])
    P_fit = fit["P"][np.ix_(idx, idx)]
    P_fit = P_fit / P_fit.sum(axis=1, keepdims=True)
    pi_fit = fit["pi"][idx] / fit["pi"][idx].sum()

    min_length = min(int(np.asarray(s).size) for s in b)
    budget = int(max_lag_frac * min_length)

    rows = []
    for n in range(2, int(n_max) + 1):
        lag = int(tau) * n
        if lag > budget:
            rows.append({"n": n, "lag": lag, "ok": False,
                         "reason": f"n*tau={lag} exceeds budget {budget}"})
            continue
        counts_n = counts_at_lag(b, n_states, lag)
        sub = counts_n[np.ix_(shared, shared)]
        totals = sub.sum(axis=1)
        live = totals > 0
        if live.sum() < 2:
            rows.append({"n": n, "lag": lag, "ok": False,
                         "reason": "no held-out counts"})
            continue
        direct = np.zeros_like(sub)
        direct[live] = sub[live] / totals[live, None]
        predicted = np.linalg.matrix_power(P_fit, n)
        row_tv = 0.5 * np.abs(direct - predicted).sum(axis=1)
        w = pi_fit[live] / pi_fit[live].sum()
        rows.append({"n": n, "lag": lag, "ok": True,
                     "err_holdout": float((w * row_tv[live]).sum()),
                     "err_max_row": float(row_tv[live].max()),
                     "n_pairs": int(sub.sum())})

    return {"ok": True, "tau": int(tau), "n_shared_states": int(shared.size),
            "n_recordings_fit": len(a), "n_recordings_test": len(b),
            "rows": rows}


def _match_modes(phi_ref, pi_ref, phi_boot, k):
    """Match bootstrap eigenvectors to the point estimate by pi-weighted overlap.

    Percentile intervals on "the 2nd largest timescale" are intervals on a
    *rank*, which is biased upward whenever two processes are close -- precisely
    when the interval matters. Matching the eigenvectors instead tracks the same
    physical process across replicates, and the overlap itself reports whether
    the process is identified at all.
    """
    m = min(k, phi_ref.shape[1], phi_boot.shape[1])
    overlap = np.abs(np.einsum("i,ij,ik->jk", pi_ref, phi_ref[:, :m],
                               phi_boot[:, :m]))
    matched = np.full(m, -1, dtype=int)
    quality = np.zeros(m)
    taken = set()
    for j in np.argsort(-overlap.max(axis=1)):
        order = np.argsort(-overlap[j])
        for cand in order:
            if int(cand) not in taken:
                matched[j] = int(cand)
                quality[j] = overlap[j, cand]
                taken.add(int(cand))
                break
    return matched, quality


def bootstrap_its(state_ids, n_states, taus, fps, n_boot=200, seed=0, k=6,
                  min_count=DEFAULT_MIN_COUNT):
    """Block bootstrap over recordings.

    Recordings are the natural block: transitions never cross them, so the count
    matrix is exactly a sum of per-recording matrices and a replicate is a
    *reweighting* rather than a recount. That makes each replicate one weighted
    bincount plus one eigh.
    """
    rng = np.random.default_rng(seed)
    taus = np.asarray(taus, dtype=int)
    n_rec = len(state_ids)

    out = {"taus": taus, "taus_s": taus.astype(float) / fps, "n_boot": n_boot,
           "lo": np.full((taus.size, k), np.nan),
           "hi": np.full((taus.size, k), np.nan),
           "point": np.full((taus.size, k), np.nan),
           "mode_stability": np.full((taus.size, k), np.nan),
           "same_support_frac": np.full(taus.size, np.nan)}

    for i, tau in enumerate(taus):
        rows, cols, rec = pair_index(state_ids, int(tau))
        if rows.size == 0:
            continue
        base = operator_at_lag(state_ids, n_states, int(tau),
                              min_count=min_count, k=k,
                              rows=rows, cols=cols, rec=rec)
        if not base["ok"]:
            continue
        t0, _ = implied_timescales(base["eigenvalues"], tau, fps)
        out["point"][i, :t0.size] = t0

        samples = np.full((n_boot, k), np.nan)
        quality = np.full((n_boot, k), np.nan)
        same_support = 0
        for b in range(n_boot):
            w = rng.multinomial(n_rec, np.full(n_rec, 1.0 / n_rec)).astype(float)
            res = operator_at_lag(state_ids, n_states, int(tau),
                                  min_count=min_count, k=k, rows=rows,
                                  cols=cols, rec=rec, rec_weights=w)
            if not res["ok"]:
                continue
            if res["keep"].size == base["keep"].size and \
                    np.array_equal(res["keep"], base["keep"]):
                same_support += 1
                matched, q = _match_modes(base["phi"], base["pi"], res["phi"], k)
            else:
                shared = np.intersect1d(base["keep"], res["keep"])
                if shared.size < 2:
                    continue
                bi = np.searchsorted(base["keep"], shared)
                ri = np.searchsorted(res["keep"], shared)
                pi_s = base["pi"][bi] / base["pi"][bi].sum()
                matched, q = _match_modes(base["phi"][bi], pi_s,
                                          res["phi"][ri], k)
            t, _ = implied_timescales(res["eigenvalues"], tau, fps)
            for j in range(min(k, matched.size)):
                if matched[j] >= 0 and matched[j] < t.size:
                    samples[b, j] = t[matched[j]]
                    quality[b, j] = q[j]

        with np.errstate(invalid="ignore"):
            out["lo"][i] = np.nanpercentile(samples, 2.5, axis=0)
            out["hi"][i] = np.nanpercentile(samples, 97.5, axis=0)
            out["mode_stability"][i] = np.nanmean(quality, axis=0)
        out["same_support_frac"][i] = same_support / max(1, n_boot)

    return out


__all__ = [
    "DEFAULT_MIN_COUNT", "DEFAULT_LAG_MARGIN", "DEFAULT_MIN_SPECTRAL_GAP",
    "pair_index", "count_matrix", "counts_at_lag", "connected_set", "restrict",
    "row_normalize", "stationary", "reversible", "spectrum",
    "spectrum_nonreversible", "implied_timescales", "operator_at_lag",
    "counts_symmetrized_timescale", "make_taus", "its_sweep", "plateau_gate",
    "ck_test", "holdout_ck", "bootstrap_its",
]
