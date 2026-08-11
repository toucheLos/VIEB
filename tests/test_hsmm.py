"""§4 — the synthetic gate for the ``hsmm`` arm. Nothing touches Luna until these pass.

These are not contract tests. They ask whether the duration model *works*: whether
it recovers durations it was shown, whether it beats the sticky HMM on data with
non-geometric durations, whether it declines to invent structure on data without
any, and whether it respects recording boundaries.

Run them in the jax environment, which is not the repo venv::

    module load python/3.11.4
    JAX_PLATFORMS=cpu PYTHONPATH=src ~/moseq/venv-moseq/bin/python -m pytest tests/test_hsmm.py -v

The repo venv is numpy 2.x and has no jax, so the main suite skips this file.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax", reason="hsmm needs the jax_moseq environment")
pytest.importorskip("jax_moseq", reason="hsmm needs the jax_moseq environment")
jax.config.update("jax_enable_x64", True)

from sklearn.metrics import adjusted_rand_score  # noqa: E402

from vieb.data.dataset import PoseDataset  # noqa: E402
from vieb.segmenters._hsmm_negbin import (  # noqa: E402
    embed_transition_matrix,
    negbin_logpmf,
    negbin_mode,
    segment_durations,
)
from vieb.segmenters.hsmm import (  # noqa: E402
    HSMMSegmenter,
    batch_by_recording,
    fit_hsmm,
    fit_sticky_hmm,
)

#: (r, p) chosen so the duration modes land on 8/15/30/60/120 frames, verified by
#: ``negbin_mode``. Deliberately non-geometric: every one has its mode away from 1,
#: which is the structure a sticky HMM cannot represent at any kappa.
TRUE_DURATIONS = [(3, 0.71531), (4, 0.78714), (5, 0.86274), (6, 0.91566), (8, 0.94212)]
TARGET_MODES = [8, 15, 30, 60, 121]


# --------------------------------------------------------------------------
# synthetic generators
# --------------------------------------------------------------------------


def make_ar_params(K: int, d: int = 2, sigma: float = 0.12, seed: int = 0):
    """``K`` clearly distinct AR(1) dynamics: different rotation, different centre.

    Distinct on purpose — the gate is about the *duration* model, so the emission
    model must not be the thing that fails.
    """
    rng = np.random.default_rng(seed)
    Ab = np.zeros((K, d, d + 1))
    Q = np.zeros((K, d, d))
    for k in range(K):
        theta = 2 * np.pi * k / K
        rho = 0.55 + 0.08 * k
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        A = rho * R
        centre = 6.0 * np.array([np.cos(theta), np.sin(theta)])
        Ab[k, :, :d] = A
        Ab[k, :, d] = (np.eye(d) - A) @ centre        # fixed point at `centre`
        Q[k] = (sigma ** 2) * np.eye(d)
    return Ab, Q, rng


def sample_state_sequence(T, K, durations, pi, rng, geometric=False):
    """State sequence with explicit durations and a zero-diagonal transition matrix."""
    z, t, k = [], 0, int(rng.integers(K))
    while t < T:
        r, p = durations[k]
        d = int(rng.geometric(1 - p, size=r).sum()) if not geometric else int(rng.geometric(1 - p))
        d = max(1, d)
        z.extend([k] * d)
        t += d
        k = int(rng.choice(K, p=pi[k]))
    return np.asarray(z[:T], dtype=np.int32)


def generate(n_rec=6, T=1500, K=5, durations=None, seed=0, geometric=False):
    """Return ``(x, mask, z_true, Ab, Q)`` shaped like the fitter expects."""
    durations = durations or TRUE_DURATIONS
    Ab, Q, rng = make_ar_params(K, seed=seed)
    pi = rng.random((K, K)) + 0.5
    np.fill_diagonal(pi, 0.0)
    pi /= pi.sum(1, keepdims=True)

    d = Ab.shape[1]
    x = np.zeros((n_rec, T, d))
    z_true = np.zeros((n_rec, T), dtype=np.int32)
    for i in range(n_rec):
        z = sample_state_sequence(T, K, durations, pi, rng, geometric=geometric)
        z_true[i] = z
        state = rng.normal(size=d)
        for t in range(T):
            A, b = Ab[z[t], :, :d], Ab[z[t], :, d]
            state = A @ state + b + rng.normal(scale=np.sqrt(Q[z[t], 0, 0]), size=d)
            x[i, t] = state
    return x, np.ones((n_rec, T)), z_true, Ab, Q


FIT = dict(num_states=8, latent_dim=2, nlags=1, alpha=5.7, gamma=1000.0,
           S_0_scale=0.01, K_0_scale=10.0)


def occupied_states(z, floor=0.02):
    """State ids holding more than ``floor`` of all frames."""
    used, counts = np.unique(np.asarray(z), return_counts=True)
    return used[counts > floor * np.asarray(z).size]


def fitted_modes(params, states):
    """The model's own claim about each state's duration mode.

    Not ``bincount(durations).argmax()``. That estimator is far too noisy to gate
    on: measured against *ground truth* at this scale it returns [8, 17, 20, 43,
    135] for true [8, 15, 30, 60, 121], so a correct fit fails it. See
    ``docs/DECISIONS.md`` #74.
    """
    r, p = np.asarray(params["r"]), np.asarray(params["p"])
    return sorted(negbin_mode(int(r[k]), float(p[k])) for k in states)


# --------------------------------------------------------------------------
# §4.1 recovery
# --------------------------------------------------------------------------


class TestRecovery:
    """Generate from a known HSMM; assert the durations and the states come back."""

    @pytest.fixture(scope="class")
    @classmethod
    def fitted(cls):
        x, mask, z_true, _, _ = generate(n_rec=10, T=3500, seed=1)
        params, z, _ = fit_hsmm(x, mask, seed=0, num_iters=150, **FIT)
        return x, mask, z_true, params, np.asarray(z)

    def test_duration_modes_within_20_percent(self, fitted):
        _, _, _, params, z = fitted
        modes = fitted_modes(params, occupied_states(z))
        unmatched = []
        for target in TARGET_MODES:
            if not modes:
                unmatched.append((target, None))
                continue
            best = min(modes, key=lambda m: abs(m - target))
            if abs(best - target) / target > 0.20:
                unmatched.append((target, best))
        assert not unmatched, (
            f"duration modes not recovered within 20%: {unmatched}; fitted {modes}, "
            f"true {TARGET_MODES}"
        )

    def test_state_ari_above_0_8(self, fitted):
        _, _, z_true, _, z = fitted
        a = z_true[:, 1:].ravel()           # nlags=1 offset
        assert adjusted_rand_score(a, z.ravel()) > 0.8

    def test_finds_the_true_number_of_states(self, fitted):
        """K=8 is a weak limit, not a target: 5 states generated it."""
        _, _, _, _, z = fitted
        assert len(occupied_states(z)) == 5

    def test_recovered_r_is_above_one(self, fitted):
        """The data has no geometric state, so no occupied state should sit at r=1."""
        _, _, _, params, z = fitted
        r = np.asarray(params["r"])[occupied_states(z)]
        assert (r > 1).all(), f"r for occupied states: {r}"


# --------------------------------------------------------------------------
# §4.2 the one that matters
# --------------------------------------------------------------------------


class TestBeatsStickyHMM:
    """On data generated *with* non-geometric durations, the HSMM must win.

    A tie means the duration model is inert and the implementation is wrong. That
    is the whole point of this test, so it is written to fail loudly rather than
    to pass on a technicality.
    """

    def test_held_out_log_likelihood(self):
        from vieb.segmenters._hsmm_negbin import marginal_log_likelihood

        x, mask, _, _, _ = generate(n_rec=8, T=1500, seed=2)
        train, train_m = x[:6], mask[:6]
        test, test_m = x[6:], mask[6:]

        params, _, _ = fit_hsmm(train, train_m, seed=0, num_iters=120, **FIT)
        hsmm_ll = marginal_log_likelihood(
            test, test_m, params["Ab"], params["Q"], params["pi"],
            params["p"], params["r"],
        )

        sticky = fit_sticky_hmm(train, train_m, seed=0, num_iters=120, **FIT)
        sp = sticky["params"]
        hmm_ll = marginal_log_likelihood(
            test, test_m, sp["Ab"], sp["Q"], sp["pi"],
        )

        assert hsmm_ll > hmm_ll, (
            f"HSMM held-out ll {hsmm_ll:.1f} did not beat sticky HMM {hmm_ll:.1f} on "
            f"data with non-geometric durations — the duration model is inert"
        )

    def test_sticky_hmm_cannot_move_its_mode(self):
        """Why the comparison is fair: a geometric dwell has mode 1 at every kappa."""
        for kappa in (1e4, 1e6, 1e8):
            p = kappa / (kappa + 10.0)
            d = np.arange(1, 5000)
            pmf = (p ** (d - 1)) * (1 - p)
            assert int(d[np.argmax(pmf)]) == 1


# --------------------------------------------------------------------------
# §4.3 null
# --------------------------------------------------------------------------


class TestNull:
    """Genuinely geometric durations: the HSMM must not invent a mode."""

    @pytest.fixture(scope="class")
    @classmethod
    def fitted(cls):
        geo = [(1, p) for _, p in TRUE_DURATIONS]
        x, mask, _, _, _ = generate(n_rec=6, T=1500, seed=3,
                                    durations=geo, geometric=True)
        params, z, _ = fit_hsmm(x, mask, seed=0, num_iters=120, **FIT)
        return x, mask, params, np.asarray(z)

    def test_does_not_invent_a_mode_away_from_one(self, fitted):
        _, mask, params, z = fitted
        used, counts = np.unique(z, return_counts=True)
        major = used[counts > 0.02 * z.size]
        modes = [negbin_mode(int(params["r"][k]), float(params["p"][k])) for k in major]
        assert np.mean([m == 1 for m in modes]) >= 0.5, (
            f"on geometric data the fitted duration modes are {modes}; a mode away "
            f"from 1 here is a false positive"
        )

    def test_likelihood_ratio_against_nested_r1_does_not_reject(self, fitted):
        from scipy.stats import chi2

        x, mask, params, z = fitted
        states, durs = segment_durations(z, mask, nlags=FIT["nlags"])
        r, p = np.asarray(params["r"]), np.asarray(params["p"])

        used = np.unique(states)
        ll_full = ll_null = 0.0
        for k in used:
            d = durs[states == k]
            if d.size == 0:
                continue
            ll_full += float(np.sum(np.asarray(negbin_logpmf(d, int(r[k]), float(p[k])))))
            ll_null += float(np.sum(np.asarray(negbin_logpmf(d, 1, float(p[k])))))

        stat = 2 * (ll_full - ll_null)
        df = max(1, len(used))
        assert chi2.sf(max(stat, 0.0), df) > 0.05, (
            f"LR test rejected r=1 on geometric data (stat {stat:.2f}, df {df}) — "
            f"a false positive for the duration model"
        )


# --------------------------------------------------------------------------
# §4.4 boundaries
# --------------------------------------------------------------------------


class TestBoundaries:
    """No segment spans a recording boundary; no AR term crosses one."""

    def _dataset(self, lengths):
        n = sum(lengths)
        return PoseDataset(
            keypoints=np.zeros((n, 3, 2)),
            recording_index=np.repeat(np.arange(len(lengths)), lengths).astype(np.int64),
            recording_ids=[f"rec_{i}" for i in range(len(lengths))],
            keypoint_names=["nose", "center", "tail_base"],
            fps=30.0,
        )

    def test_batching_puts_each_recording_in_its_own_row(self):
        data = self._dataset([40, 25])
        X = np.arange(65 * 2, dtype=float).reshape(65, 2)
        x, mask, slices = batch_by_recording(X, data)
        assert x.shape == (2, 40, 2)
        assert mask[0].sum() == 40 and mask[1].sum() == 25
        assert np.array_equal(x[1, :25], X[40:])
        # padding is masked, so no AR lag reads a neighbouring recording
        assert mask[1, 25:].sum() == 0

    def test_no_segment_spans_a_boundary(self):
        # One recording all state 0, the next all state 0 as well: a naive encoder
        # run over the concatenation would report ONE bout of 30, not two of 15.
        z = np.zeros((2, 15), dtype=np.int32)
        mask = np.ones((2, 15 + 1))
        states, durs = segment_durations(z, mask, nlags=1, drop_censored=False)
        assert states.tolist() == [0, 0]
        assert durs.tolist() == [15, 15]

    def test_predict_leaves_the_ar_lead_in_unassigned(self):
        data = self._dataset([60, 45])
        x, mask, _, _, _ = generate(n_rec=2, T=60, K=5, seed=5)
        X = np.concatenate([x[0], x[1][:45]], axis=0)
        seg = HSMMSegmenter(num_iters=8, predict_iters=2, fit_sample_size=2, **FIT)
        seg.fit(X, data, seed=0)
        out = seg.predict(X, data)
        # exactly nlags frames per recording, at each recording's head
        assert out.frame_labels[0] == -1 and out.frame_labels[60] == -1
        assert int((out.frame_labels < 0).sum()) == FIT["nlags"] * 2

    def test_transition_counts_ignore_cross_row_pairs(self):
        from vieb.segmenters._hsmm_negbin import segment_transition_counts

        # row 0 ends in state 2, row 1 starts in state 3. That pair is NOT a
        # transition — it is two different animals.
        z = np.array([[0, 0, 2, 2], [3, 3, 1, 1]], dtype=np.int32)
        counts = segment_transition_counts(z, np.ones((2, 5)), 4, nlags=1)
        assert counts[2, 3] == 0
        assert counts[0, 2] == 1 and counts[3, 1] == 1


# --------------------------------------------------------------------------
# the embedding itself
# --------------------------------------------------------------------------


class TestEmbedding:
    def test_rows_are_stochastic_and_padding_is_unreachable(self):
        import jax.numpy as jnp

        K, r_max = 4, 10
        rng = np.random.default_rng(0)
        pi = rng.random((K, K))
        np.fill_diagonal(pi, 0)
        pi /= pi.sum(1, keepdims=True)
        p = np.array([0.5, 0.8, 0.9, 0.95])
        r = np.array([1, 2, 5, 10])

        T = np.asarray(embed_transition_matrix(jnp.asarray(pi), jnp.asarray(p),
                                               jnp.asarray(r), r_max))
        assert np.allclose(T.sum(1), 1.0)
        assert (T >= 0).all()

        Tn = T.reshape(K, r_max, K, r_max)
        for k in range(K):
            for i in range(int(r[k]), r_max):
                inbound = Tn[:, :, k, i].sum() - Tn[k, i, k, i]
                assert inbound == 0.0, f"padding sub-state ({k},{i}) is reachable"

    def test_zero_diagonal_means_no_self_transition_between_segments(self):
        import jax.numpy as jnp

        K, r_max = 3, 4
        pi = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]])
        p = np.full(K, 0.8)
        r = np.full(K, 2)
        T = np.asarray(embed_transition_matrix(jnp.asarray(pi), jnp.asarray(p),
                                               jnp.asarray(r), r_max)).reshape(
            K, r_max, K, r_max)
        # from the last sub-state of k, all exiting mass goes to other states
        for k in range(K):
            assert T[k, 1, k, 0] == 0.0

    @pytest.mark.parametrize("r,p", TRUE_DURATIONS)
    def test_pmf_normalizes(self, r, p):
        import jax.numpy as jnp

        d = jnp.arange(r, 60000)
        assert abs(float(jnp.exp(negbin_logpmf(d, r, p)).sum()) - 1.0) < 1e-6

    def test_r_one_is_exactly_geometric(self):
        """The nested model of §5's LR test is the sticky HMM's dwell time itself."""
        import jax.numpy as jnp

        d = np.arange(1, 400)
        p = 0.9
        assert np.allclose(
            np.asarray(jnp.exp(negbin_logpmf(jnp.asarray(d), 1, p))),
            (p ** (d - 1)) * (1 - p),
        )

    @pytest.mark.parametrize("r,p,expected", list(zip(*zip(*TRUE_DURATIONS), TARGET_MODES)))
    def test_mode_moves_off_one_for_r_above_one(self, r, p, expected):
        assert negbin_mode(r, p) == expected
        assert negbin_mode(1, p) == 1
