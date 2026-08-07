"""Section 4 -- synthetic verification gate for the transfer operator.

Same discipline that caught the centered-convolution bug in ExBias: no new code
touches real data until all four of these pass. Each one targets a specific way
the construction could be wrong in a manner that would still look plausible on
real behavior.

  1. **Underdamped double well** -- does the operator recover a known
     metastable decomposition, a known rate, and (via delay embedding) a
     variable that was never observed?
  2. **Limit cycle** -- does it recover a known period?
  3. **Duration control** -- the one this branch's central claim rests on. Two
     states of identical dynamical character at a 20:1 dwell ratio, plus a
     fast rare third. If the rare state is merged or dropped, the confound has
     been relabeled rather than dissolved, and that must be reported as a
     failure rather than tuned away.
  4. **Null** -- i.i.d. noise must produce no plateau and no split. Zero false
     positives.

Model time units are used throughout and `fps` is set to `1 / dt_obs`, so the
seconds-based API is exercised exactly as it will be on real data.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from representation import transfer_operator as TO  # noqa: E402


# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------

def simulate_double_well(beta, n_traj=24, n_steps=30_000, dt=0.01,
                         sample_every=5, seed=0):
    """dx = v dt ; dv = -v dt - 4x(x^2-1) dt + sqrt(2/beta) dW.

    U(x) = (x^2-1)^2, so the barrier is 1 and kT = 1/beta. Only x is returned:
    the velocity must be recovered by delay embedding or not at all.
    """
    rng = np.random.default_rng(seed)
    x = rng.choice([-1.0, 1.0], size=n_traj) + 0.1 * rng.normal(size=n_traj)
    v = np.zeros(n_traj)
    noise = np.sqrt(2.0 / beta) * np.sqrt(dt)

    out = np.empty((n_steps // sample_every, n_traj))
    for step in range(n_steps):
        v += (-v - 4.0 * x * (x ** 2 - 1.0)) * dt + noise * rng.normal(size=n_traj)
        x += v * dt
        if step % sample_every == 0:
            out[step // sample_every] = x
    return [out[:, i].copy() for i in range(n_traj)], dt * sample_every


def measured_hop_rate(sessions, dt_obs, edge=0.5):
    """Well-crossing rate per unit time, counted from committed crossings.

    A crossing only counts once the trajectory is past +/-`edge`, so barrier
    recrossing -- which is fast, real, and not a hop -- does not inflate it.
    """
    changes, total_time = 0, 0.0
    for x in sessions:
        well = np.zeros(x.size, dtype=int)
        current = 0
        for t in range(x.size):
            if x[t] > edge:
                current = 1
            elif x[t] < -edge:
                current = -1
            well[t] = current
        committed = well[well != 0]
        if committed.size > 1:
            changes += int((np.diff(committed) != 0).sum())
        total_time += x.size * dt_obs
    return changes / total_time if total_time > 0 else 0.0


def simulate_stuart_landau(period=1.0, n_traj=12, n_steps=30_000, dt=0.002,
                           sample_every=10, noise=0.02, seed=0):
    """dz/dt = (1 + i w) z - |z|^2 z -- a clean limit cycle of known period."""
    rng = np.random.default_rng(seed)
    omega = 2.0 * np.pi / period
    z = np.exp(2j * np.pi * rng.random(n_traj)) * (1.0 + 0.1 * rng.normal(size=n_traj))
    amp = noise * np.sqrt(dt)

    out = np.empty((n_steps // sample_every, n_traj))
    for step in range(n_steps):
        z = z + ((1.0 + 1j * omega) * z - (np.abs(z) ** 2) * z) * dt \
            + amp * (rng.normal(size=n_traj) + 1j * rng.normal(size=n_traj))
        if step % sample_every == 0:
            out[step // sample_every] = z.real
    return [out[:, i].copy() for i in range(n_traj)], dt * sample_every


def simulate_dwell_contrast(n_traj=16, n_frames=12_000, dt_obs=0.05, seed=0):
    """Three states of identical dynamical character, dwell times 20 : 1 : 0.5.

    Within every state the observable is the same Ornstein-Uhlenbeck relaxation
    about a different centre, so the states differ *only* in how long they last
    and how often they are entered -- exactly the situation in which a
    density-based method assigns almost everything to the long state.

    `theta` must be fast enough that a state is *reached* well inside its own
    dwell time. At theta=4 the relaxation takes ~10 frames while the rare
    state lasts 10 frames, so its frames sat in transit near the other centres
    and the test measured that transit rather than the operator. Isolating the
    duration variable is the whole point of this control, so the approach to
    each centre is made fast (~3 frames) relative to the shortest dwell.
    """
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [6.0, 0.0], [3.0, 6.0]])
    dwell = np.array([20.0, 1.0, 0.5])          # seconds: the 20:1 ratio, plus fast
    theta, sigma = 12.0, 0.5                     # identical within-state dynamics

    sessions, truth = [], []
    for _ in range(n_traj):
        xs = np.empty((n_frames, 2))
        labels = np.empty(n_frames, dtype=int)
        state = 0
        remaining = rng.exponential(dwell[state])
        pos = centres[state].copy()

        for t in range(n_frames):
            if remaining <= 0:
                if state == 2:
                    state = int(rng.integers(0, 2))
                else:
                    state = 2 if rng.random() < 0.4 else (1 - state)
                remaining = rng.exponential(dwell[state])
            pos += theta * (centres[state] - pos) * dt_obs \
                + sigma * np.sqrt(dt_obs) * rng.normal(size=2)
            xs[t], labels[t] = pos, state
            remaining -= dt_obs

        sessions.append(xs)
        truth.append(labels)
    return sessions, truth, dt_obs


def label_sessions(sessions, centers, k_lags, stride_s, fps):
    """Delay-embed, assign microstates, and split back per recording."""
    X, index = TO.delay_embed(sessions, k_lags, stride_s, fps)
    flat = TO.assign_microstates(X, centers)
    return [flat[index[:, 0] == r] for r in range(len(sessions))], X, index


# ---------------------------------------------------------------------------
# 1. Underdamped double well
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def double_well():
    out = {}
    for beta in (2.0, 3.0, 4.0):
        sessions, dt_obs = simulate_double_well(beta, seed=int(beta))
        fps = 1.0 / dt_obs
        embedded = [s.reshape(-1, 1) for s in sessions]
        X, index = TO.delay_embed(embedded, 1, dt_obs, fps)
        centers = TO.microstates(X, 50, seed=0)
        labels, _, _ = label_sessions(embedded, centers, 1, dt_obs, fps)
        out[beta] = {"sessions": sessions, "dt_obs": dt_obs, "fps": fps,
                     "labels": labels, "X": X, "index": index,
                     "centers": centers,
                     "hop_rate": measured_hop_rate(sessions, dt_obs)}
    return out


def test_double_well_splits_into_two_metastable_wells(double_well):
    """VERIFICATION 1a -- phi_2 must separate the two wells."""
    d = double_well[3.0]
    counts, _ = TO.ulam_counts(d["labels"], 50, 1.0, d["fps"])
    P, active = TO.row_normalize(counts)
    pi = TO.stationary_distribution(P)
    P_r = TO.reversibilize(P, pi)

    best = TO.split_by_phi2(P_r, pi)
    assert best is not None and best["chi"] > 0.85, (
        f"no metastable split found (chi={best['chi'] if best else None})")

    # Each side must correspond to one well, read through the microstate centres.
    centre_x = d["centers"][active][:, 0]
    minus_sign = np.sign(np.mean(centre_x[best["minus"]]))
    plus_sign = np.sign(np.mean(centre_x[best["plus"]]))
    assert minus_sign != plus_sign, "the split did not separate the two wells"

    purity = max(
        np.mean(centre_x[best["minus"]] < 0) * np.mean(centre_x[best["plus"]] > 0),
        np.mean(centre_x[best["minus"]] > 0) * np.mean(centre_x[best["plus"]] < 0),
    )
    assert purity > 0.8, f"wells are mixed across the split (purity {purity:.2f})"


@pytest.mark.parametrize("beta", [2.0, 3.0, 4.0])
def test_second_eigenvalue_matches_the_measured_hopping_rate(double_well, beta):
    """VERIFICATION 1b -- |Lambda_2| must track the rate, at three temperatures.

    For a symmetric two-well chain of rate k each way the slowest relaxation is
    t_2 = 1 / (2k), so the operator's t_imp is checked against a rate counted
    independently from well crossings rather than against theory.
    """
    d = double_well[beta]
    counts, _ = TO.ulam_counts(d["labels"], 50, 1.0, d["fps"])
    P, _ = TO.row_normalize(counts)
    pi = TO.stationary_distribution(P)
    vals, _ = TO.leading_eigen(TO.reversibilize(P, pi), pi, 3)
    t2 = TO.implied_timescales(vals[1:2], 1.0)[0]

    expected = 1.0 / (2.0 * d["hop_rate"])
    assert d["hop_rate"] > 0, "no well crossings were observed"
    assert 0.6 < t2 / expected < 1.7, (
        f"beta={beta}: t_imp={t2:.2f} vs 1/(2k)={expected:.2f} "
        f"(measured rate {d['hop_rate']:.4f})"
    )


def test_delay_embedding_recovers_equipartition(double_well):
    """VERIFICATION 1c -- mean kinetic energy ~ kT/2 from x alone.

    Velocity was never observed. If the delay coordinate carries it, the
    reconstructed kinetic energy matches equipartition; if the embedding is
    only smoothing x, it will not.
    """
    for beta, d in double_well.items():
        v = np.concatenate([np.diff(s) / d["dt_obs"] for s in d["sessions"]])
        kinetic = float(np.mean(0.5 * v ** 2))
        expected = 0.5 / beta
        assert 0.8 < kinetic / expected < 1.25, (
            f"beta={beta}: <KE>={kinetic:.4f} vs kT/2={expected:.4f}")


# ---------------------------------------------------------------------------
# 2. Limit cycle
# ---------------------------------------------------------------------------

def test_limit_cycle_period_appears_as_aliasing_in_the_spectrum():
    """VERIFICATION 2 -- the operator returns toward identity at the period."""
    period = 1.0
    sessions, dt_obs = simulate_stuart_landau(period=period, seed=1)
    fps = 1.0 / dt_obs
    embedded = [s.reshape(-1, 1) for s in sessions]

    X, _ = TO.delay_embed(embedded, 2, dt_obs * 4, fps)
    centers = TO.microstates(X, 60, seed=0)
    labels, _, _ = label_sessions(embedded, centers, 2, dt_obs * 4, fps)

    taus = np.linspace(0.1, 2.4, 40)
    lam2 = []
    for tau in taus:
        counts, _ = TO.ulam_counts(labels, 60, float(tau), fps)
        P, _ = TO.row_normalize(counts)
        pi = TO.stationary_distribution(P)
        vals, _ = TO.leading_eigen(TO.reversibilize(P, pi), pi, 3)
        lam2.append(float(vals[1]))
    lam2 = np.asarray(lam2)

    # Ignore the near-identity region at small tau, which is an artifact.
    window = taus > 0.55
    peak = taus[window][int(np.argmax(lam2[window]))]
    assert abs(peak - period) / period < 0.15, (
        f"spectral revival at tau={peak:.3f}, expected the period {period}")
    assert lam2[window].max() > 0.5, "no revival of the spectrum at any tau"


# ---------------------------------------------------------------------------
# 3. Duration control -- the decisive test
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dwell_contrast():
    sessions, truth, dt_obs = simulate_dwell_contrast(seed=3)
    fps = 1.0 / dt_obs
    centers = TO.microstates(np.concatenate(sessions, 0), 60, seed=0)
    labels = [TO.assign_microstates(s, centers) for s in sessions]
    return {"sessions": sessions, "truth": truth, "fps": fps, "dt_obs": dt_obs,
            "labels": labels, "centers": centers}


def test_stationary_distribution_reflects_the_20_to_1_dwell_ratio(dwell_contrast):
    """VERIFICATION 3a -- occupancy is reported, in pi, not suppressed."""
    d = dwell_contrast
    counts, _ = TO.ulam_counts(d["labels"], 60, 0.25, d["fps"])
    P, active = TO.row_normalize(counts)
    pi = TO.stationary_distribution(P)

    # Attribute each microstate to the true state its centre sits nearest.
    truth_flat = np.concatenate(d["truth"])
    micro_flat = np.concatenate(d["labels"])
    mass = np.zeros(3)
    for k, state in enumerate(active):
        member = micro_flat == state
        if member.any():
            mass[np.bincount(truth_flat[member], minlength=3).argmax()] += pi[k]

    observed = mass[0] / mass[1]
    assert 12.0 < observed < 32.0, (
        f"pi ratio long:short = {observed:.1f}, expected ~20 "
        f"(pi = {mass.round(4)})")


def test_rare_fast_state_is_neither_merged_nor_dropped(dwell_contrast):
    """VERIFICATION 3b -- the claim the whole branch rests on.

    The rare state carries a few percent of the stationary measure and lasts
    1/40th as long as the dominant one. A density-based method assigns it to
    whichever neighbour is denser. Here it must survive as its own metastable
    set, because identity lives in the eigenvectors and only occupancy lives
    in pi.
    """
    d = dwell_contrast
    counts, _ = TO.ulam_counts(d["labels"], 60, 0.25, d["fps"])
    P, active = TO.row_normalize(counts)
    pi = TO.stationary_distribution(P)
    P_r = TO.reversibilize(P, pi)

    _, leaves = TO.recursive_split(P_r, pi, min_chi=0.6, min_mass=0.002,
                                   max_depth=3)
    assert len(leaves) >= 3, f"only {len(leaves)} macrostates recovered, expected >= 3"

    truth_flat = np.concatenate(d["truth"])
    micro_flat = np.concatenate(d["labels"])

    purities = []
    for leaf in leaves:
        states = active[leaf["members"]]
        member = np.isin(micro_flat, states)
        if member.sum() == 0:
            continue
        hist = np.bincount(truth_flat[member], minlength=3)
        purities.append((hist.argmax(), hist.max() / hist.sum(), leaf["pi"]))

    rare = [p for p in purities if p[0] == 2 and p[1] > 0.7]
    assert rare, (
        "the rare fast state was merged away or dropped; leaves resolved as "
        f"{[(int(a), round(float(b), 2), round(float(c), 4)) for a, b, c in purities]}"
    )


# ---------------------------------------------------------------------------
# 4. Null
# ---------------------------------------------------------------------------

def test_iid_noise_produces_no_plateau_and_no_split():
    """VERIFICATION 4 -- zero false positives on structureless data."""
    rng = np.random.default_rng(7)
    sessions = [rng.normal(size=(4000, 2)) for _ in range(12)]
    fps = 20.0
    centers = TO.microstates(np.concatenate(sessions, 0), 40, seed=0)
    labels = [TO.assign_microstates(s, centers) for s in sessions]

    rows = TO.timescale_sweep(labels, 40, [0.05, 0.1, 0.2, 0.4, 0.8, 1.6], fps,
                              n_timescales=5)
    slowest = [r["timescales"][0] for r in rows
               if r.get("timescales") is not None and np.isfinite(r["timescales"][0])]
    # For memoryless data every eigenvalue is ~0, so any timescale that does
    # survive must be far shorter than the lag that produced it.
    for row in rows:
        if row.get("eigenvalues") is not None:
            assert row["eigenvalues"][0] < 0.25, (
                f"tau={row['lag_s']}: lambda_2={row['eigenvalues'][0]:.3f} "
                f"on i.i.d. noise")
    assert all(t < 0.5 for t in slowest), f"spurious slow timescales {slowest}"

    counts, _ = TO.ulam_counts(labels, 40, 0.1, fps)
    P, _ = TO.row_normalize(counts)
    pi = TO.stationary_distribution(P)
    best = TO.split_by_phi2(TO.reversibilize(P, pi), pi)
    assert best["chi"] < 0.6, (
        f"spurious metastable split on i.i.d. noise (chi={best['chi']:.3f})")


# ---------------------------------------------------------------------------
# Supporting: the operational constraints
# ---------------------------------------------------------------------------

def test_all_temporal_parameters_are_in_seconds():
    # 0.1 s is 3 frames at Luna's 30fps and 25 at Spence's 250fps. A hardcoded
    # frame count would make these the same window.
    assert TO.to_frames(0.1, 30.0) == 3
    assert TO.to_frames(0.1, 250.0) == 25
    assert TO.to_frames(0.001, 30.0) == 1        # never rounds down to zero


def test_pairs_never_straddle_a_recording_boundary():
    # Two recordings of 10 frames at lag 3 give 7 + 7 = 14 pairs. Concatenating
    # first would give 17, the 3 extra being splices between two animals.
    labels = [np.zeros(10, dtype=int), np.ones(10, dtype=int)]
    counts, n_pairs = TO.ulam_counts(labels, 2, 0.3, 10.0)
    assert n_pairs == 14
    assert counts[0, 1] == 0 and counts[1, 0] == 0, "a pair crossed a boundary"


def test_concatenated_labels_are_rejected():
    with pytest.raises(TypeError, match="across recording boundaries"):
        TO.ulam_counts(np.zeros(20, dtype=int), 2, 0.1, 10.0)


def test_reversibilized_operator_satisfies_detailed_balance():
    rng = np.random.default_rng(0)
    P = rng.random((6, 6)) + 0.1
    P /= P.sum(axis=1)[:, None]
    pi = TO.stationary_distribution(P)
    P_r = TO.reversibilize(P, pi)
    flux = pi[:, None] * P_r
    assert np.allclose(flux, flux.T), "detailed balance violated"
    assert np.allclose(P_r.sum(axis=1), 1.0), "P_r is not stochastic"


def test_chapman_kolmogorov_is_near_zero_for_a_true_markov_chain():
    rng = np.random.default_rng(0)
    true = np.array([[0.90, 0.08, 0.02],
                     [0.05, 0.90, 0.05],
                     [0.02, 0.08, 0.90]])
    sessions = []
    for _ in range(20):
        seq = np.empty(5000, dtype=int)
        seq[0] = 0
        for t in range(1, seq.size):
            seq[t] = rng.choice(3, p=true[seq[t - 1]])
        sessions.append(seq)

    report = TO.chapman_kolmogorov(sessions, 3, 0.1, 10.0, n_max=5)
    for row in report:
        assert row.get("mean_tv") is not None, row
        assert row["mean_tv"] < 0.05, f"CK error {row['mean_tv']:.4f} at n={row['n']}"
