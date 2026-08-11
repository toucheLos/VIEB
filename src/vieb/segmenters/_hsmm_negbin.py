"""The negative-binomial duration model, as an HMM embedding.

Deliberately free of any ``vieb`` import: this is the mathematics, and the §4
synthetic gate tests it directly without the harness, the loader or the registry
in the way. ``hsmm.py`` is the thin adapter that gives it the ``Segmenter``
contract.

Why this model
--------------
A sticky HDP-HMM carries persistence in a self-transition probability ``p``,
which forces geometric dwell times ``P(d=n) = p^(n-1)(1-p)``. That is
monotonically decreasing, so the **mode is 1 frame at every value of kappa** —
raising kappa moves the mean and never the mode. Measured on the Luna baseline
at kappa=1e6: mode 1 frame, median 14, mean 35.4.

Here the transition matrix has a **zero diagonal** and persistence lives entirely
in an explicit duration::

    z_s ~ Cat(Pi_{z_{s-1}})           Pi_kk = 0
    d_s ~ NegBin(r_{z_s}, p_{z_s})    frames
    x_t ~ N(A_{z_s} x_{t-1} + b, Q)   reused from jax_moseq, untouched

The embedding
-------------
Naive HSMM message passing is ``O(T K^2 D_max)``, hopeless at T = 22.4M. But a
``NegBin(r, p)`` duration is *exactly* a chain of ``r`` sub-states each with
geometric holding ``p`` (Johnson & Willsky), so the HSMM is an HMM on ``K * r``
states and the existing forward-backward runs unchanged. No ``D_max``, no
truncation: the duration distribution has unbounded support.

Sub-state ``(k, i)`` lives at flat index ``k * r_max + i``:

* ``i < r_k - 1``  -- stay ``p_k``, advance to ``(k, i+1)`` with ``1 - p_k``
* ``i = r_k - 1``  -- stay ``p_k``, leave to ``(j, 0)`` with ``(1-p_k) Pi_kj``
* ``i >= r_k``     -- padding, so a ragged ``r_k`` fits a rectangular array.
  Self-loop row (a valid stochastic row), zero initial mass, and nothing
  transitions into it, so it is unreachable rather than merely unlikely.

Total time in the chain is a sum of ``r`` geometrics on ``{1, 2, ...}``, i.e.
support ``d >= r`` with ``P(d=n) = C(n-1, r-1) (1-p)^r p^(n-r)``. At ``r = 1``
this is exactly the sticky HMM's geometric dwell time, which is what makes the
``r = 1`` likelihood-ratio test of §5 a nested-model test rather than an
analogy.

Cost. Emission likelihoods are computed for ``K`` states and tiled, so the AR
term stays ``O(T K)``; only message passing grows, to ``O(T (K r_max)^2)`` for
this dense embedding. That is the price of reusing dynamax's tested
forward-backward verbatim instead of writing a structured scan; see
``docs/DECISIONS.md``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from dynamax.hidden_markov_model.inference import hmm_filter, hmm_posterior_sample
from jax_moseq.utils.autoregression import ar_log_likelihood, get_nlags
from jax_moseq.utils.transitions import count_transitions, sample_betas, sample_pi

na = jnp.newaxis

#: Largest ``r`` the grid admits. The embedded chain is ``K * R_MAX`` states, so
#: this is the one number that sets the cost of every sweep.
R_MAX = 10


# --------------------------------------------------------------------------
# the duration distribution
# --------------------------------------------------------------------------


def negbin_logpmf(d, r, p):
    """``log P(d)`` for a sum of ``r`` geometric holds with stay probability ``p``.

    Support is ``d >= r``; below it the density is zero, which is *not* a numerical
    guard but the model: a chain of ``r`` sub-states cannot be traversed in fewer
    than ``r`` frames. It is also what stops ``r`` from running away — ``r`` above a
    state's shortest uncensored bout is rejected outright.
    """
    d = jnp.asarray(d, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.float64)
    p = jnp.clip(jnp.asarray(p, dtype=jnp.float64), 1e-12, 1 - 1e-12)
    from jax.scipy.special import gammaln

    log_binom = gammaln(d) - gammaln(r) - gammaln(d - r + 1.0)
    out = log_binom + r * jnp.log1p(-p) + (d - r) * jnp.log(p)
    return jnp.where(d >= r, out, -jnp.inf)


def negbin_mode(r: int, p: float, d_max: int = 100_000) -> int:
    """Argmax of the pmf, found by evaluation rather than by a closed form.

    The whole claim of this branch is *where the mode is*, so it is measured the
    same way the empirical mode is measured, not asserted from an identity that
    could be off by one.
    """
    r = int(r)
    hi = max(r + 1, min(d_max, int(np.ceil(4 * r / max(1e-9, 1 - p))) + 10))
    d = np.arange(r, hi + 1)
    lp = np.asarray(negbin_logpmf(jnp.asarray(d), r, p))
    return int(d[int(np.argmax(lp))])


def negbin_mean(r: int, p: float) -> float:
    """``E[d] = r / (1 - p)``."""
    return float(r) / max(1e-12, 1.0 - float(p))


# --------------------------------------------------------------------------
# the embedding
# --------------------------------------------------------------------------


def embed_transition_matrix(pi, p, r, r_max: int = R_MAX):
    """``(K, K)`` super-state matrix -> ``(K r_max, K r_max)`` embedded matrix.

    ``pi`` must have a zero diagonal; a self-transition here would be persistence
    entering through the back door, on top of the duration that is supposed to be
    carrying all of it.
    """
    pi = jnp.asarray(pi, dtype=jnp.float64)
    p = jnp.asarray(p, dtype=jnp.float64)
    r = jnp.asarray(r, dtype=jnp.int32)
    K = pi.shape[0]

    idx = jnp.arange(r_max)[na, :]                 # (1, r_max)
    active = idx < r[:, na]                        # (K, r_max)
    last = idx == (r[:, na] - 1)

    kk = jnp.broadcast_to(jnp.arange(K)[:, na], (K, r_max))
    ii = jnp.broadcast_to(idx, (K, r_max))

    T = jnp.zeros((K, r_max, K, r_max), dtype=jnp.float64)

    # Stay. Padding sub-states get a self-loop so their row is stochastic; they
    # are never entered, so the value is bookkeeping, not behaviour.
    T = T.at[kk, ii, kk, ii].set(jnp.where(active, p[:, na], 1.0))

    # Advance within the chain. At i = r_max-1 the clamped target is the stay
    # cell, but the value there is 0, so the add is a no-op.
    advance = jnp.where(active & ~last, 1.0 - p[:, na], 0.0)
    T = T.at[kk, ii, kk, jnp.minimum(ii + 1, r_max - 1)].add(advance)

    # Leave, from the last sub-state only, into sub-state 0 of another state.
    exit_mass = jnp.where(last, 1.0 - p[:, na], 0.0)          # (K, r_max)
    T = T.at[:, :, :, 0].add(exit_mass[:, :, na] * pi[:, na, :])

    return T.reshape(K * r_max, K * r_max)


def embed_initial_distribution(K: int, r_max: int = R_MAX, weights=None):
    """All initial mass on sub-state 0 of each state.

    ``sample_hmm_stateseq`` hardcodes a uniform initial distribution over every
    state, which here would start chains inside padding sub-states and mid-chain.
    That is why this module calls ``hmm_posterior_sample`` directly: same dynamax
    routine, just not through that one wrapper.
    """
    w = jnp.ones(K) / K if weights is None else jnp.asarray(weights, dtype=jnp.float64)
    pi0 = jnp.zeros((K, r_max), dtype=jnp.float64).at[:, 0].set(w)
    return pi0.reshape(K * r_max)


def embed_log_likelihoods(lls, r_max: int = R_MAX):
    """``(T, K)`` -> ``(T, K r_max)``: every sub-state of ``k`` emits as ``k``.

    ``jnp.repeat`` (not ``tile``) so the layout matches ``k * r_max + i``.
    """
    return jnp.repeat(jnp.asarray(lls), r_max, axis=-1)


def collapse(z_emb, r_max: int = R_MAX):
    """Embedded sub-state ids -> super-state ids."""
    return jnp.asarray(z_emb) // r_max


# --------------------------------------------------------------------------
# Gibbs steps
# --------------------------------------------------------------------------


def _ar_lls(x, Ab, Q):
    """Per-state AR log-likelihoods, ``(N, T-nlags, K)``. Reused verbatim."""
    from functools import partial

    lls = jax.lax.map(partial(ar_log_likelihood, x), (Ab, Q))
    return jnp.moveaxis(lls, 0, -1)


def resample_stateseqs(seed, x, mask, Ab, Q, pi, p, r, r_max: int = R_MAX):
    """Resample the sub-state sequence, then collapse it.

    Returns ``(z, z_emb)``. ``z`` feeds ``resample_ar_params`` unchanged — the AR
    model never learns that the embedding exists.
    """
    nlags = get_nlags(Ab)
    K = pi.shape[0]
    lls = embed_log_likelihoods(_ar_lls(x, Ab, Q), r_max)
    pi_emb = embed_transition_matrix(pi, p, r, r_max)
    pi0 = embed_initial_distribution(K, r_max)
    m = mask.astype(jnp.float64)[:, nlags:]

    def one(key, ll, mk):
        return hmm_posterior_sample(key, pi0, pi_emb, ll * mk[:, na])[1]

    z_emb = jax.vmap(one)(jr.split(seed, mask.shape[0]), lls, m)
    return collapse(z_emb, r_max), z_emb


def resample_stay_probs(seed, z_emb, mask, K, r_max=R_MAX, a0=1.0, b0=1.0, nlags=3):
    """Conjugate Beta update for ``p_k`` from the sub-chain's own transitions.

    Every within-state step is one Bernoulli(``p_k``) trial for "stay", whether it
    advances a sub-state or leaves for another state — which is exactly why the
    embedding makes the duration model conjugate.
    """
    z_emb = jnp.asarray(z_emb)
    m = mask.astype(jnp.float64)[:, nlags:][:, 1:]
    src, dst = z_emb[:, :-1], z_emb[:, 1:]
    k_src = src // r_max
    stayed = (src == dst).astype(jnp.float64)

    n_stay = jnp.zeros(K).at[k_src].add(stayed * m)
    n_move = jnp.zeros(K).at[k_src].add((1.0 - stayed) * m)
    return jr.beta(seed, a0 + n_stay, b0 + n_move), n_stay, n_move


def resample_transitions(seed, z, mask, betas, alpha, gamma, nlags=3):
    """Segment-level transitions, with the diagonal removed.

    Counts are over *segments*, not frames: a frame-level count would be
    overwhelmingly diagonal and would re-learn the persistence the duration model
    is meant to own. ``sample_betas``/``sample_pi`` are reused at ``kappa=0`` so
    the prior stays comparable to the baseline's ``Dir(alpha * beta)``; the
    diagonal is then zeroed and the rows renormalized, since ``pi`` is sampled
    from a prior that does not know the diagonal is structurally absent.
    """
    K = len(betas)
    counts = segment_transition_counts(np.asarray(z), np.asarray(mask), K, nlags=nlags)
    counts = jnp.asarray(counts, dtype=jnp.float64)
    seeds = jr.split(seed)
    betas = sample_betas(seeds[0], counts, betas, alpha, 0.0, gamma)
    pi = sample_pi(seeds[1], counts, betas, alpha, 0.0)
    pi = pi * (1.0 - jnp.eye(K))
    pi = pi / jnp.clip(pi.sum(axis=1, keepdims=True), 1e-12)
    return betas, pi


def _negbin_logpmf_np(d, r, p):
    """numpy twin of ``negbin_logpmf``. The MH step runs per state on short arrays,
    where jax dispatch costs more than the arithmetic."""
    from scipy.special import gammaln

    d = np.asarray(d, dtype=np.float64)
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    r = float(r)
    out = (gammaln(d) - gammaln(r) - gammaln(d - r + 1.0)
           + r * np.log1p(-p) + (d - r) * np.log(p))
    return np.where(d >= r, out, -np.inf)


def resample_r(seed, durations, states, p, r, K, r_max=R_MAX, stay_prior=(1.0, 1.0)):
    """Metropolis-Hastings on the integer grid ``1..r_max``, one step per state.

    Returns ``(r, p)``: the two move **together**, and that is the whole design of
    this step.

    Proposing ``r`` at fixed ``p`` does not work, and fails in a way that looks like
    the model being right. The chain's mean duration is ``r / (1-p)``, so a move
    ``r -> r+1`` at fixed ``p`` also doubles the mean; the observed durations then
    have a far lower likelihood and the move is rejected essentially always. The
    sampler stays pinned at its ``r = 1`` initialization and reports geometric
    durations for data that is not geometric.

    So the proposal moves along the **constant-mean manifold**: propose ``r'``, then
    take ``p'`` with ``r'/(1-p') = r/(1-p)``. The two hypotheses being compared then
    have the same mean duration and differ only in *shape*, which is the comparison
    that is actually of interest. The map ``p -> p'`` is linear with Jacobian
    ``r'/r``, included below; the +/-1 proposal is asymmetric at the grid edges, so
    the Hastings ratio is included too rather than assumed away.

    ``durations``/``states`` must already exclude recording-edge segments: those are
    censored, and scoring a truncated bout as if it were complete biases ``r`` down.
    """
    rng = np.random.default_rng(int(jr.randint(seed, (), 0, 2**30)))
    r = np.asarray(r, dtype=np.int32).copy()
    p = np.asarray(p, dtype=np.float64).copy()
    durations = np.asarray(durations)
    states = np.asarray(states)
    a0, b0 = stay_prior

    def log_beta_prior(q):
        return (a0 - 1.0) * np.log(max(q, 1e-300)) + (b0 - 1.0) * np.log(max(1 - q, 1e-300))

    for k in range(K):
        d = durations[states == k]
        if d.size == 0:
            continue
        cur, p_cur = int(r[k]), float(p[k])
        moves = [m for m in (cur - 1, cur + 1) if 1 <= m <= r_max]
        if not moves:
            continue
        prop = int(rng.choice(moves))

        # constant-mean reparameterization
        p_prop = 1.0 - (prop / cur) * (1.0 - p_cur)
        if not (0.0 < p_prop < 1.0):
            continue

        back = [m for m in (prop - 1, prop + 1) if 1 <= m <= r_max]
        log_hastings = np.log(len(moves)) - np.log(len(back))
        log_jacobian = np.log(prop / cur)

        ll_prop = float(np.sum(_negbin_logpmf_np(d, prop, p_prop))) + log_beta_prior(p_prop)
        if not np.isfinite(ll_prop):
            continue
        ll_cur = float(np.sum(_negbin_logpmf_np(d, cur, p_cur))) + log_beta_prior(p_cur)

        if np.log(rng.random() + 1e-300) < (ll_prop - ll_cur + log_hastings + log_jacobian):
            r[k], p[k] = prop, p_prop
    return r, p


# --------------------------------------------------------------------------
# segments
# --------------------------------------------------------------------------


def segment_durations(z, mask, nlags: int = 3, drop_censored: bool = True):
    """Run-length encode each row separately. Returns ``(states, durations)``.

    Rows are recordings, so a segment cannot span a recording boundary by
    construction — the boundary is not a place the encoder can run through, it is
    the end of the array it is running through.

    ``drop_censored`` removes each row's first and last segment: both are cut off
    by the recording rather than by the animal, and their lengths are lower bounds.
    """
    z = np.asarray(z)
    m = np.asarray(mask)[:, nlags:].astype(bool) if mask is not None else None
    states, durs = [], []
    for i in range(z.shape[0]):
        row = z[i]
        if m is not None:
            row = row[m[i][: row.shape[0]]]
        if row.size == 0:
            continue
        cuts = np.flatnonzero(np.diff(row)) + 1
        bounds = np.concatenate([[0], cuts, [row.size]])
        s = row[bounds[:-1]]
        d = np.diff(bounds)
        if drop_censored and s.size > 2:
            s, d = s[1:-1], d[1:-1]
        elif drop_censored:
            continue
        states.append(s)
        durs.append(d)
    if not states:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64)
    return np.concatenate(states).astype(np.int32), np.concatenate(durs).astype(np.int64)


def segment_transition_counts(z, mask, K, nlags: int = 3):
    """``(K, K)`` counts of consecutive *distinct* states, per row.

    Uncensored/censored is irrelevant here — a transition observed at a row edge is
    still a transition. Only within-row pairs are counted.
    """
    z = np.asarray(z)
    m = np.asarray(mask)[:, nlags:].astype(bool) if mask is not None else None
    counts = np.zeros((K, K), dtype=np.int64)
    for i in range(z.shape[0]):
        row = z[i]
        if m is not None:
            row = row[m[i][: row.shape[0]]]
        if row.size < 2:
            continue
        seq = row[np.concatenate([[0], np.flatnonzero(np.diff(row)) + 1])]
        if seq.size < 2:
            continue
        np.add.at(counts, (seq[:-1], seq[1:]), 1)
    return counts


def duration_summary(states, durations, K: int, fps: float) -> list[dict]:
    """Per-state mode / median / CV, in frames and seconds.

    The mode is the number this branch exists to move, so it is reported first and
    reported empirically.
    """
    out = []
    for k in range(K):
        d = np.asarray(durations)[np.asarray(states) == k]
        if d.size == 0:
            out.append({"state": k, "n_bouts": 0})
            continue
        counts = np.bincount(d)
        mode = int(np.argmax(counts))
        mean = float(d.mean())
        out.append({
            "state": k,
            "n_bouts": int(d.size),
            "mode_frames": mode,
            "mode_s": mode / fps,
            "median_frames": float(np.median(d)),
            "median_s": float(np.median(d)) / fps,
            "mean_frames": mean,
            "cv": float(d.std() / mean) if mean > 0 else float("nan"),
            "mode_is_one_frame": mode == 1,
        })
    return out


# --------------------------------------------------------------------------
# likelihood, for the model comparisons of §4.2, §4.3 and §5
# --------------------------------------------------------------------------


def marginal_log_likelihood(x, mask, Ab, Q, pi, p=None, r=None, r_max=R_MAX):
    """Marginal log-likelihood, summed over recordings.

    With ``p``/``r`` this scores the HSMM; without them it scores a plain HMM using
    ``pi`` as given — so the sticky-HMM and HSMM numbers in §4.2 come from the same
    function, over the same emission model, and differ only in the transition
    structure. That is the only way the comparison means anything.
    """
    nlags = get_nlags(Ab)
    K = pi.shape[0]
    lls = _ar_lls(x, Ab, Q)
    m = mask.astype(jnp.float64)[:, nlags:]

    if p is None or r is None:
        pi_use, pi0, lls_use = jnp.asarray(pi), jnp.ones(K) / K, lls
    else:
        pi_use = embed_transition_matrix(pi, p, r, r_max)
        pi0 = embed_initial_distribution(K, r_max)
        lls_use = embed_log_likelihoods(lls, r_max)

    def one(ll, mk):
        return hmm_filter(pi0, pi_use, ll * mk[:, na]).marginal_loglik

    return float(jnp.sum(jax.vmap(one)(lls_use, m)))
