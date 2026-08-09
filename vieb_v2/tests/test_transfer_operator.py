<<<<<<< HEAD
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
=======
"""Synthetic verification of the transfer operator. This is a gate, not a suite.

No real data goes near this code until every system here passes. The discipline
is the one that caught the centered-convolution bug in ExBias: each system has an
answer known in closed form, and the test asserts that answer rather than
asserting that the code ran.

Five systems, and the reason each earns its place:

1. Underdamped double well, observing position only. The observable is *not*
   Markov -- velocity is hidden -- so this is the honest version of the problem
   behavior poses. Checks the metastable split, the hopping rate against Kramers
   across three temperatures, and that the delay embedding really does recover
   the hidden velocity, via equipartition.

2. Limit cycle. Reversibilization symmetrizes rotation into decay, so this
   pins down both what survives (period-aliasing in the timescales, the complex
   pair in the non-reversible spectrum) and what does not. A method whose
   failure modes are undocumented is worse than one with fewer of them.

3. Duration control. The whole claim of this branch. Two states with identical
   dynamical character and an 18:1 dwell ratio, plus a third fast rare state.
   `pi` must report the occupancy ratio while the rare state survives the
   connected set. If this fails, the branch's central claim is false and the
   correct response is to say so, not to tune tau until it passes.

4. i.i.d. noise. Zero false positives, and specifically `plateau_gate` must
   return False -- a gate that cannot fail is not a gate.

5. Ornstein-Uhlenbeck. A genuine relaxation timescale with a clean plateau and
   *no* metastability. Without it the gate reduces to "did the timescales
   plateau", and a plateau is exactly what boring one-dimensional relaxation
   gives you. This is what forces the verdict to require plateau AND spectral
   gap AND sign structure.
>>>>>>> 1deb112fe3f70a3b9c20d11ea35f7ec43986b068
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

<<<<<<< HEAD
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
=======
from representation import transfer_operator as top  # noqa: E402
from representation.koopman import classify_modes  # noqa: E402


# --------------------------------------------------------------------------
# 1. Underdamped double well, observing x only
# --------------------------------------------------------------------------
# U(x) = (x^2 - 1)^2, minima at +-1, barrier height 1 at x = 0.
# dx = v dt ; dv = (-U'(x) - gamma v) dt + sqrt(2 gamma kT) dW
DW_GAMMA = 1.0
DW_DT = 0.005
DW_STRIDE = 20                      # -> dt_obs = 0.1, fps = 10
DW_FPS = 1.0 / (DW_DT * DW_STRIDE)
DW_FRAMES = 5000
DW_TRAJ = 40
DW_KT = (0.25, 0.325, 0.4)
DW_BARRIER = 1.0
DW_BINS = 50
DW_EDGES = np.linspace(-2.0, 2.0, DW_BINS + 1)
# The lag at which t2 has converged. Measured: t2 falls 326 -> 171 -> 119 -> 98
# -> 90 -> 87 s over tau = 2, 5, 10, 20, 40, 80 at kT = 0.25. The short-lag
# inflation is the near-identity artifact the module docstring names -- with
# velocity hidden, a particle that has barely moved looks like it stayed -- so
# reading t2 at tau = 5 measures the artifact, not the barrier.
DW_TAU = 40


def _double_well(seed=0):
    """One integration, vectorized over (temperature, trajectory)."""
    rng = np.random.default_rng(seed)
    kT = np.repeat(np.array(DW_KT), DW_TRAJ)[:, None].ravel()
    n = kT.size
    x = rng.choice([-1.0, 1.0], size=n)
    v = rng.normal(0, np.sqrt(kT))
    noise = np.sqrt(2.0 * DW_GAMMA * kT * DW_DT)

    out = np.empty((DW_FRAMES, n))
    for step in range(DW_FRAMES * DW_STRIDE):
        force = -4.0 * x * (x * x - 1.0)
        v += (force - DW_GAMMA * v) * DW_DT + noise * rng.standard_normal(n)
        x += v * DW_DT
        if step % DW_STRIDE == DW_STRIDE - 1:
            out[step // DW_STRIDE] = x
    return out, kT


def _bin_states(x, edges):
    return np.clip(np.digitize(x, edges[1:-1]), 0, edges.size - 2)


def _hop_time(x, fps, lo=-0.8, hi=0.8):
    """Mean time between well-to-well transitions, with hysteresis.

    Counting bare sign changes would count every recrossing of the barrier top
    as a hop and overestimate the rate by an order of magnitude.
    """
    side = np.zeros(x.size, dtype=np.int8)
    side[x > hi] = 1
    side[x < lo] = -1
    committed = side[side != 0]
    if committed.size < 2:
        return np.nan
    n_hops = int((np.diff(committed) != 0).sum())
    return (x.size / fps) / n_hops if n_hops else np.nan


@pytest.fixture(scope="module")
def double_well():
    traj, kT = _double_well(seed=0)
    per_temp = {}
    for i, temp in enumerate(DW_KT):
        cols = np.flatnonzero(kT == temp)
        xs = [traj[:, c] for c in cols]
        states = [_bin_states(x, DW_EDGES) for x in xs]
        res = top.operator_at_lag(states, DW_BINS, tau=DW_TAU, k=6)
        t, _ = top.implied_timescales(res["eigenvalues"], DW_TAU, DW_FPS)
        hops = np.array([_hop_time(x, DW_FPS) for x in xs])
        per_temp[temp] = {"x": xs, "states": states, "res": res, "its": t,
                          "hop_time": float(np.nanmean(hops))}
    return per_temp


def test_the_double_well_splits_into_two_metastable_sets_on_the_second_eigenvector(
        double_well):
    """phi_2's sign must separate the wells, and change sign at the barrier."""
    data = double_well[DW_KT[1]]
    res, phi = data["res"], data["res"]["phi"]
    centers = 0.5 * (DW_EDGES[:-1] + DW_EDGES[1:])[res["keep"]]

    left = phi[centers < -0.5, 1]
    right = phi[centers > 0.5, 1]
    assert np.mean(left > 0) > 0.95 or np.mean(left < 0) > 0.95
    assert np.sign(np.median(left)) == -np.sign(np.median(right))

    # The sign change must happen at the barrier, not off in one well.
    order = np.argsort(centers)
    c, p = centers[order], phi[order, 1]
    crossings = np.flatnonzero(np.diff(np.sign(p)) != 0)
    assert crossings.size >= 1
    split = c[crossings[0]]
    assert abs(split) < 0.2, f"phi_2 changes sign at x={split:.3f}, not the barrier"


def test_the_second_timescale_matches_the_measured_hopping_rate(double_well):
    """For a symmetric two-state chain the relaxation time is half the mean
    time between hops. Asserted at every temperature, not just a lucky one."""
    for temp in DW_KT:
        data = double_well[temp]
        t2_msm = data["its"][1]
        t2_true = data["hop_time"] / 2.0
        assert np.isfinite(t2_msm) and t2_msm > 0
        ratio = t2_msm / t2_true
        assert abs(np.log(ratio)) < np.log(1.5), (
            f"kT={temp}: MSM t2={t2_msm:.2f}s vs empirical {t2_true:.2f}s")


def test_the_hopping_slows_monotonically_as_temperature_falls(double_well):
    t2 = [double_well[temp]["its"][1] for temp in DW_KT]
    assert t2[0] > t2[1] > t2[2], f"t2 not monotone in kT: {t2}"


def test_the_hopping_rate_is_arrhenius_in_temperature(double_well):
    """log(1/t2) vs 1/kT must have slope -barrier. 30% is the honest tolerance:
    Kramers prefactors are temperature-dependent and add real curvature."""
    inv_kt = np.array([1.0 / temp for temp in DW_KT])
    log_rate = np.array([np.log(1.0 / double_well[temp]["its"][1])
                         for temp in DW_KT])
    slope = np.polyfit(inv_kt, log_rate, 1)[0]
    assert abs(slope + DW_BARRIER) < 0.3 * DW_BARRIER, (
        f"Arrhenius slope {slope:.3f}, expected {-DW_BARRIER}")


def test_delay_embedding_recovers_the_hidden_velocity_by_equipartition(
        double_well):
    """The observable is position only. If the delay embedding is doing what it
    is supposed to, the reconstructed velocity carries mean kinetic energy
    kT/2 -- an exact analytic answer, not a fitted one."""
    ratios = []
    for temp in DW_KT:
        x = np.concatenate([xi for xi in double_well[temp]["x"]])
        v_est = np.diff(x) * DW_FPS
        var = float(np.var(v_est))
        assert abs(var - temp) < 0.25 * temp, (
            f"kT={temp}: Var(v_est)={var:.4f}, expected ~{temp}")
        ratios.append(var / temp)
    assert np.ptp(ratios) < 0.10, f"Var(v)/kT not constant: {ratios}"


# --------------------------------------------------------------------------
# 2. Limit cycle
# --------------------------------------------------------------------------
LC_B, LC_K = 2.0, 1.0
LC_PERIOD_S = 1.0
LC_FPS = 30.0
LC_FRAMES = 900
LC_TRAJ = 30
LC_BINS = 24


@pytest.fixture(scope="module")
def limit_cycle():
    """r' = -k r (r - b), theta' = omega. Stable cycle at r = b, period exact."""
    rng = np.random.default_rng(1)
    omega = 2.0 * np.pi / LC_PERIOD_S
    dt = 1.0 / LC_FPS
    states = []
    for _ in range(LC_TRAJ):
        r = LC_B + rng.normal(0, 0.05)
        th = rng.uniform(0, 2 * np.pi)
        angles = np.empty(LC_FRAMES)
        for i in range(LC_FRAMES):
            r += -LC_K * r * (r - LC_B) * dt + rng.normal(0, 0.005)
            th += omega * dt
            angles[i] = th % (2 * np.pi)
        states.append(np.minimum((angles / (2 * np.pi) * LC_BINS).astype(int),
                                 LC_BINS - 1))
    return states


def _t2_at(states, tau, n_states, fps):
    res = top.operator_at_lag(states, n_states, tau, k=4)
    if not res["ok"]:
        return np.nan
    return top.implied_timescales(res["eigenvalues"], tau, fps)[0][1]


def _eigs_at(states, tau, n_states, k=8):
    return top.operator_at_lag(states, n_states, tau, k=k)["eigenvalues"]


def test_the_orbit_period_shows_up_as_aliasing_in_the_spectrum(limit_cycle):
    """Aliasing lives in the eigenvalues, not in the timescales.

    For a rotation discretized into B angular bins, P(tau) is a shift by k bins
    and its reversibilization has eigenvalues cos(2*pi*k*m/B). At a full period
    k = B and every eigenvalue returns to +1 -- the operator is the identity. At
    a half period k = B/2 and the eigenvalues are (-1)^m, so half of them sit at
    -1: period-2 alternating modes.

    Those negative eigenvalues make t_imp NaN *by design* -- the module refuses
    to clip them into small positive timescales, precisely because doing so
    would report a fast process where there is a rhythmic one. So a test phrased
    on t_imp measures nothing here; the eigenvalues are where the period is
    legible.
    """
    # One full orbit: the trajectory has returned, so P is literally the
    # identity. Every state is its own communicating class and there is no
    # operator to speak of -- the strongest aliasing signature there is, and
    # `operator_at_lag` is right to refuse rather than report timescales.
    counts_full = top.counts_at_lag(limit_cycle, LC_BINS, 30)
    assert np.trace(counts_full) / counts_full.sum() > 0.95
    full = top.operator_at_lag(limit_cycle, LC_BINS, 30)
    assert full["ok"] is False
    assert full["connected"]["n_components"] == LC_BINS
    assert full["connected"]["mass_share_of_largest"] < 0.1

    # Half an orbit: P is a shift by B/2, so states pair into 2-cycles and the
    # spectrum is exactly +-1 -- a period-2 alternating mode.
    half = top.operator_at_lag(limit_cycle, LC_BINS, 15, k=4)
    assert half["ok"]
    assert half["connected"]["n_components"] == LC_BINS // 2
    assert half["eigenvalues"].min() < -0.9, (
        f"no period-2 mode at half an orbit: {half['eigenvalues'].min():.3f}")
    t_half, flags_half = top.implied_timescales(half["eigenvalues"], 15, LC_FPS)
    assert flags_half["n_nonpositive"] >= 1
    assert np.isnan(t_half[-1])

    # A quarter orbit: one well-mixed component, no inversion.
    quarter = top.operator_at_lag(limit_cycle, LC_BINS, 7, k=4)
    assert quarter["connected"]["n_components"] == 1
    assert quarter["connected"]["mass_share_of_largest"] > 0.95
    assert quarter["eigenvalues"].min() > 0.0


def test_the_nonreversible_spectrum_reports_the_rotation_directly(limit_cycle):
    """Cross-checked through koopman.classify_modes, which also proves the two
    modules share one convention for turning an eigenvalue into a frequency."""
    res = top.operator_at_lag(limit_cycle, LC_BINS, tau=1, k=6)
    evals = top.spectrum_nonreversible(res["P"], k=6)
    modes = classify_modes(evals, LC_FPS)
    periods = [m["period_frames"] for m in modes
               if m["oscillatory"] and m["period_frames"]]
    assert any(27 <= p <= 33 for p in periods), (
        f"no mode near the 30-frame orbit period: {periods}")


def test_reversibilization_destroys_the_rotation(limit_cycle):
    """The documented cost. P_r's spectrum is real by construction, and none of
    its timescales corresponds to the orbit period -- rotation has been
    symmetrized into decay. Stated here rather than discovered later."""
    res = top.operator_at_lag(limit_cycle, LC_BINS, tau=1, k=8)
    assert np.all(np.isreal(res["eigenvalues"]))
    t, _ = top.implied_timescales(res["eigenvalues"], 1, LC_FPS)
    finite = t[np.isfinite(t) & (t > 0)]
    assert not np.any(np.abs(finite - LC_PERIOD_S) < 0.2 * LC_PERIOD_S), (
        f"a reversible timescale matched the period: {finite}")


# --------------------------------------------------------------------------
# 3. Duration control -- the claim this branch rests on
# --------------------------------------------------------------------------
DC_DWELL = {0: 600, 1: 30, 2: 6}        # frames; A slow, B fast, C very fast
DC_MEANS = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
DC_SIGMA = 0.6
DC_RECORDINGS = 60
DC_FRAMES = 6000
DC_FPS = 30.0
# 120, not 60. Measured: state C is 313 frames out of 360,000, and k-means at
# k = 60 allocates it no centre at all -- its 313 points are absorbed into a
# neighbouring cluster and the state vanishes before the operator ever sees it.
# k = 120 resolves it, k = 200 gives it two centres. That is a discretization
# limit on rare states, entirely separate from the operator, and it is the thing
# to watch on real data where no ground truth says whether a state was lost.
DC_MICRO = 120
# t2 of the exact discrete generator is 31.8 frames, so a lag above ~30 frames
# cannot resolve it. Measured convergence: t2 = 0.53, 0.63, 0.74, 0.83, 0.89 s
# at tau = 2, 3, 5, 8, 12 against a true 1.06 s -- the usual short-lag
# underestimate, approached from below.
DC_TAU = 8


def _duration_chain(seed=0):
    """A -> B always; B -> A w.p. 0.9, B -> C w.p. 0.1; C -> B always.

    Visit frequencies are nA : nB : nC = 0.9 : 1 : 0.1, so A is entered
    slightly *less* often than B, while occupying 18x more time. That is a
    sharper statement of the claim than equal visit rates would be.
    """
    rng = np.random.default_rng(seed)
    labels, obs = [], []
    for _ in range(DC_RECORDINGS):
        lab = np.empty(DC_FRAMES, dtype=int)
        state, i = 0, 0
        while i < DC_FRAMES:
            dwell = max(1, rng.geometric(1.0 / DC_DWELL[state]))
            end = min(DC_FRAMES, i + dwell)
            lab[i:end] = state
            i = end
            if state == 0:
                state = 1
            elif state == 1:
                state = 2 if rng.random() < 0.1 else 0
            else:
                state = 1
        labels.append(lab)
        obs.append(DC_MEANS[lab] + rng.normal(0, DC_SIGMA, (DC_FRAMES, 3)))
    return labels, obs


def _measure_truth(labels):
    """Occupancy and entry counts read off the true state sequence.

    Measured rather than derived, so the assertions are against what this
    realization actually contains -- finite recordings truncated mid-dwell do
    not reproduce the asymptotic chain exactly, and an assertion against theory
    would be testing the theory.
    """
    pooled = np.concatenate(labels)
    occupancy = np.array([(pooled == s).mean() for s in range(3)])
    entries = np.zeros(3, dtype=int)
    for lab in labels:
        changes = np.flatnonzero(np.diff(lab) != 0) + 1
        for s in range(3):
            entries[s] += int((lab[changes] == s).sum()) + int(lab[0] == s)
    return occupancy, entries


@pytest.fixture(scope="module")
def duration_control():
    from sklearn.cluster import MiniBatchKMeans
    labels, obs = _duration_chain(seed=0)
    pooled = np.concatenate(obs)
    km = MiniBatchKMeans(n_clusters=DC_MICRO, random_state=0, n_init=10,
                         batch_size=2048).fit(pooled)
    micro = [km.predict(o) for o in obs]
    res = top.operator_at_lag(micro, DC_MICRO, tau=DC_TAU, k=6)
    true_of_micro = np.full(DC_MICRO, -1)
    all_lab, all_mic = np.concatenate(labels), np.concatenate(micro)
    for m in range(DC_MICRO):
        sel = all_lab[all_mic == m]
        if sel.size:
            true_of_micro[m] = np.bincount(sel, minlength=3).argmax()
    occupancy, entries = _measure_truth(labels)
    return {"labels": labels, "micro": micro, "res": res,
            "true_of_micro": true_of_micro,
            "occupancy": occupancy, "entries": entries}


def _pi_by_true_state(data):
    mapping = data["true_of_micro"][data["res"]["keep"]]
    pi = data["res"]["pi"]
    return np.array([pi[mapping == s].sum() for s in range(3)]), mapping


def test_the_stationary_distribution_encodes_dwell_time_not_visit_frequency(
        duration_control):
    """The named test, and the branch's central claim.

    A and B are entered essentially equally often -- measured 607 against 605,
    a 0.3% difference -- while A occupies 19x the time. A method that finds
    modes of a sampling density cannot separate those two facts, because the
    density is the occupancy. The transfer operator reports the occupancy in pi
    and leaves the identity of the states to the eigenvectors, so the confound
    is dissolved rather than relabelled.
    """
    occupancy, entries = (duration_control["occupancy"],
                          duration_control["entries"])
    pi_true, _ = _pi_by_true_state(duration_control)

    # The premise: near-equal entry rates, wildly unequal occupancy.
    assert 0.85 < entries[0] / entries[1] < 1.15, (
        f"A and B are not entered comparably often: {entries}")
    assert occupancy[0] / occupancy[1] > 15

    # The claim: pi recovers the occupancy, not the entry rate.
    assert pi_true[0] / pi_true[1] > 15, (
        f"pi(A)/pi(B) = {pi_true[0] / pi_true[1]:.2f}, occupancy ratio "
        f"{occupancy[0] / occupancy[1]:.2f}")
    assert np.allclose(pi_true[:2], occupancy[:2], atol=0.01), (
        f"pi = {pi_true} vs measured occupancy {occupancy}")


def test_the_rare_fast_state_survives_the_connected_set(duration_control):
    """C carries under 0.1% of the measure across 313 frames in 360,000. It must
    be neither pruned away nor inflated -- low measure is not invisibility."""
    pi_true, mapping = _pi_by_true_state(duration_control)
    occupancy = duration_control["occupancy"]

    assert (mapping == 2).sum() >= 1, "every microstate for C was pruned"
    assert pi_true[2] > 0
    assert 0.4 * occupancy[2] < pi_true[2] < 2.5 * occupancy[2], (
        f"pi(C) = {pi_true[2]:.5f}, measured occupancy {occupancy[2]:.5f}")


def test_a_coarse_partition_loses_the_rare_state_before_the_operator_sees_it():
    """The discretization limit, made explicit.

    At k = 60 microstates, k-means gives state C no centre -- its 313 points are
    absorbed into a neighbour and it is gone before any operator is built. This
    is not something the transfer operator can repair, and on real data nothing
    announces it. Recorded here so the k = 120 choice above reads as a measured
    requirement rather than a preference.
    """
    from sklearn.cluster import MiniBatchKMeans
    labels, obs = _duration_chain(seed=0)
    all_lab, pooled = np.concatenate(labels), np.concatenate(obs)

    found = {}
    for k in (60, 120):
        km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=10,
                             batch_size=2048).fit(pooled)
        assign = km.predict(pooled)
        majority = np.array([
            np.bincount(all_lab[assign == j], minlength=3).argmax()
            if (assign == j).any() else -1 for j in range(k)])
        found[k] = int((majority == 2).sum())

    assert found[60] == 0, "expected k=60 to lose the rare state"
    assert found[120] >= 1, "expected k=120 to resolve the rare state"


def test_the_slowest_timescale_approaches_the_analytic_generator(
        duration_control):
    """Against the exact discrete-time transition matrix, not a fit.

    The relaxation is fast (31.8 frames) even though A's dwell is 600, because
    equilibration rate is k_AB + k_BA and B empties quickly. Occupancy is skewed;
    equilibration is not slow. Those are different statements and the operator
    keeps them apart.
    """
    Q = np.zeros((3, 3))
    Q[0, 1] = 1.0 / DC_DWELL[0]
    Q[1, 0] = 0.9 / DC_DWELL[1]
    Q[1, 2] = 0.1 / DC_DWELL[1]
    Q[2, 1] = 1.0 / DC_DWELL[2]
    np.fill_diagonal(Q, -Q.sum(axis=1))
    T = np.eye(3) + Q                       # geometric dwells -> discrete time
    lam = np.sort(np.real(np.linalg.eigvals(T)))[::-1]
    t2_true = -1.0 / np.log(lam[1]) / DC_FPS

    t, _ = top.implied_timescales(duration_control["res"]["eigenvalues"],
                                  DC_TAU, DC_FPS)
    assert abs(np.log(t[1] / t2_true)) < np.log(1.45), (
        f"MSM t2={t[1]:.3f}s vs generator {t2_true:.3f}s")

    # Approached from below, as the short-lag discretization bias predicts.
    shorter, _ = top.implied_timescales(
        top.operator_at_lag(duration_control["micro"], DC_MICRO, tau=3,
                            k=4)["eigenvalues"], 3, DC_FPS)
    assert shorter[1] < t[1] < t2_true


# --------------------------------------------------------------------------
# 4. Null -- i.i.d. Gaussian
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def iid_noise():
    from sklearn.cluster import MiniBatchKMeans
    rng = np.random.default_rng(0)
    obs = [rng.standard_normal((2000, 5)) for _ in range(100)]
    km = MiniBatchKMeans(n_clusters=100, random_state=0, n_init=10,
                         batch_size=2048).fit(np.concatenate(obs))
    return [km.predict(o) for o in obs]


def test_iid_noise_has_no_slow_process(iid_noise):
    res = top.operator_at_lag(iid_noise, 100, tau=1, k=6)
    assert res["ok"]
    assert res["eigenvalues"][1] < 0.10, (
        f"lambda_2 = {res['eigenvalues'][1]:.4f} on i.i.d. noise")


def test_timescales_grow_linearly_with_lag_on_noise(iid_noise):
    """Linear growth in tau *is* the null signature, and the brief names it as
    the branch's falsification condition.

    With no real slow process, lambda_2 sits at the sampling floor and does not
    move with the lag, so t = -tau*dt/log(lambda_2) is proportional to tau by
    construction. Asserting that timescales stay *flat* on noise would be
    asserting the opposite of the truth. What must hold is that t/tau is
    constant -- no plateau anywhere -- and that every point is below the
    resolution line, which is what stops the gate from reading it as a process.
    """
    taus = np.array([1, 2, 4, 8, 16])
    t = np.array([_t2_at(iid_noise, int(x), 100, 30.0) for x in taus])
    per_lag = t / (taus / 30.0)

    assert np.all(np.diff(t) > 0), f"t2 not growing with lag on noise: {t}"
    assert per_lag.max() / per_lag.min() < 2.0, (
        f"growth is not linear in tau: t/tau = {np.round(per_lag, 3)}")
    assert np.all(t < taus / 30.0), (
        "noise timescales should all sit below the resolution line")


def test_the_plateau_gate_rejects_noise(iid_noise):
    """The gate itself, not just the eigenvalues. A gate that cannot fail is
    not a gate."""
    sweep = top.its_sweep(iid_noise, 100, [1, 2, 4, 8, 16, 32], 30.0, k=5)
    gate = top.plateau_gate(sweep)
    assert gate["passed"] is False
    assert "no plateau" in gate["verdict"] or not gate["any_plateau"]


# --------------------------------------------------------------------------
# 5. Ornstein-Uhlenbeck -- a plateau without metastability
# --------------------------------------------------------------------------
OU_THETA, OU_SIGMA, OU_FPS = 2.0, 1.0, 30.0
OU_BINS, OU_FRAMES, OU_RECORDINGS = 50, 3000, 60


@pytest.fixture(scope="module")
def ou_process():
    """Exact discrete sampling -- no integration error to argue about."""
    rng = np.random.default_rng(2)
    dt = 1.0 / OU_FPS
    a = np.exp(-OU_THETA * dt)
    sd_step = OU_SIGMA * np.sqrt((1 - a ** 2) / (2 * OU_THETA))
    sd_stat = OU_SIGMA / np.sqrt(2 * OU_THETA)
    edges = np.linspace(-3 * sd_stat, 3 * sd_stat, OU_BINS + 1)

    states = []
    for _ in range(OU_RECORDINGS):
        x = np.empty(OU_FRAMES)
        x[0] = rng.normal(0, sd_stat)
        noise = rng.normal(0, sd_step, OU_FRAMES)
        for i in range(1, OU_FRAMES):
            x[i] = a * x[i - 1] + noise[i]
        states.append(_bin_states(x, edges))
    return states, edges


def test_the_ou_process_plateaus_at_its_known_relaxation_time(ou_process):
    """A textbook plateau: measured flat to within 5% of 1/theta from tau = 1
    to tau = 30 frames, a thirtyfold range."""
    states, _ = ou_process
    taus = top.make_taus(OU_FPS, tau_max_s=1.0, n_tau=16)
    sweep = top.its_sweep(states, OU_BINS, taus, OU_FPS, k=5)
    gate = top.plateau_gate(sweep, processes=(1,), window_ratio=4.0, tol=0.20)

    entry = gate["processes"][1]
    assert entry["has_plateau"], "no plateau on a process that has one exactly"
    plateau = entry["plateau"]["plateau_s"]
    assert abs(plateau - 1.0 / OU_THETA) < 0.15 * (1.0 / OU_THETA), (
        f"plateau at {plateau:.3f}s, expected {1.0 / OU_THETA:.3f}s")
    assert entry["plateau"]["tau_ratio"] >= 4.0


def test_a_plateau_alone_does_not_imply_two_metastable_states(ou_process):
    """The reason this system is in the gate.

    OU has one genuine relaxation timescale, plateaus beautifully, and has no
    metastable structure at all -- its density is unimodal. Two things follow,
    and both are load-bearing for how the verdict is phrased:

    1. The spectral-gap criterion cannot exclude it *at any threshold at or
       below 2*. OU's eigenvalues are exp(-n*theta*t), so its consecutive
       timescale ratios are exactly (n+1):n and t2/t3 = 2.00 -- indistinguishable
       from a real gap by ratio alone.
    2. phi_2 ramps monotonically across the coordinate rather than splitting it.
       That is the criterion that separates relaxation from metastability.

    So the gate must require sign structure, and must not pass on plateau plus
    gap. The final assertion checks exactly that.
    """
    from scipy import stats

    states, edges = ou_process
    res = top.operator_at_lag(states, OU_BINS, tau=8, k=5)
    centers = 0.5 * (edges[:-1] + edges[1:])[res["keep"]]

    rho = stats.spearmanr(centers, res["phi"][:, 1]).statistic
    assert abs(rho) > 0.95, f"phi_2 is not monotone in x (rho={rho:.3f})"
    interior = np.sign(res["phi"][np.argsort(centers), 1])
    assert int((np.diff(interior) != 0).sum()) == 1, (
        "phi_2 should cross zero once, as a ramp does, not partition the space")

    t, _ = top.implied_timescales(res["eigenvalues"], 8, OU_FPS)
    ratio = t[1] / t[2]
    assert 1.8 < ratio < 2.3, (
        f"expected OU's analytic 2:1 timescale ratio, got {ratio:.2f}")

    # Plateau and gap both present; only the sign structure rejects it.
    taus = top.make_taus(OU_FPS, tau_max_s=1.0, n_tau=16)
    sweep = top.its_sweep(states, OU_BINS, taus, OU_FPS, k=5)
    assert top.plateau_gate(sweep, processes=(1,),
                            phi_sign_structure={1: False})["passed"] is False


# --------------------------------------------------------------------------
# Machinery -- boundary rules and numerical self-checks
# --------------------------------------------------------------------------
def test_pair_index_refuses_a_concatenated_array():
    """The boundary rule is enforced by the type system, not by discipline: a
    pair straddling two recordings is silently garbage."""
    with pytest.raises(TypeError, match="never cross a recording boundary"):
        top.pair_index(np.zeros(100, dtype=int), tau=1)


def test_pairs_never_straddle_a_recording_boundary():
    a = np.zeros(10, dtype=int)
    b = np.ones(10, dtype=int)
    rows, cols, rec = top.pair_index([a, b], tau=1)
    assert rows.size == 18                      # 9 + 9, not 19
    assert not np.any(rows != cols)             # no 0 -> 1 pair exists
    assert set(np.unique(rec)) == {0, 1}


def test_recordings_shorter_than_the_lag_contribute_nothing():
    rows, _, _ = top.pair_index([np.zeros(5, int), np.zeros(50, int)], tau=10)
    assert rows.size == 40


def test_row_normalize_refuses_an_empty_row():
    counts = np.array([[3.0, 1.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="no outgoing counts"):
        top.row_normalize(counts)


def test_connected_set_drops_a_censored_state_without_a_self_loop():
    """A state seen only at the end of every recording has no outgoing counts.
    Adding a self-loop would manufacture a second eigenvalue at exactly 1 and a
    spurious infinite timescale; dropping it is correct."""
    counts = np.array([[10.0, 5.0, 0.0],
                       [6.0, 9.0, 1.0],
                       [0.0, 0.0, 0.0]])
    keep, report = top.connected_set(counts)
    assert set(keep.tolist()) == {0, 1}
    assert report["n_states_dropped"] == 1
    P = top.row_normalize(top.restrict(counts, keep)[0])
    pi, pi_report = top.stationary(P)
    assert pi_report["near_reducible"] is False
    assert np.isfinite(top.implied_timescales(
        top.spectrum(top.reversible(P, pi), pi, k=2)["eigenvalues"],
        1, 30.0)[0][1])


def test_connected_set_ranks_components_by_mass_not_by_node_count():
    """A large component of barely-visited states is not the chain we want."""
    n = 12
    counts = np.zeros((n, n))
    for i in range(2):                              # tiny, heavy component
        for j in range(2):
            counts[i, j] = 5000.0
    for i in range(2, n):                           # large, feather-light one
        counts[i, 2 + (i - 1) % (n - 2)] = 1.0
        counts[i, 2 + (i + 1) % (n - 2)] = 1.0
    keep, report = top.connected_set(counts)
    assert set(keep.tolist()) == {0, 1}
    assert report["mass_share_of_largest"] > 0.9


def test_reversibilization_self_check_catches_a_wrong_stationary_vector():
    P = np.array([[0.9, 0.1], [0.2, 0.8]])
    pi, _ = top.stationary(P)
    top.reversible(P, pi)                            # correct pi: fine
    with pytest.raises(ValueError, match="self-check failed"):
        top.reversible(P, np.array([0.5, 0.5]))


def test_the_symmetric_conjugate_reproduces_a_known_two_state_timescale():
    """A two-state chain has t2 = -tau*dt / log(1 - p - q), exactly."""
    p, q, fps, tau = 0.1, 0.2, 30.0, 1
    P = np.array([[1 - p, p], [q, 1 - q]])
    pi, _ = top.stationary(P)
    assert pi[0] == pytest.approx(q / (p + q))
    spec = top.spectrum(top.reversible(P, pi), pi, k=2)
    assert spec["eigenvalues"][1] == pytest.approx(1 - p - q, abs=1e-12)
    t, _ = top.implied_timescales(spec["eigenvalues"], tau, fps)
    assert t[1] == pytest.approx(-(tau / fps) / np.log(1 - p - q))
    assert np.isinf(t[0])


def test_implied_timescales_flag_rather_than_clip_a_negative_eigenvalue():
    """A negative eigenvalue is a period-2 alternating mode -- a real signal
    about the lag, not a small positive number."""
    t, flags = top.implied_timescales(np.array([1.0, -0.5]), 1, 30.0)
    assert np.isnan(t[1])
    assert flags["n_nonpositive"] == 1


def test_counts_are_reweighted_not_recounted_by_the_bootstrap_hook():
    a, b = np.array([0, 1, 0, 1, 0]), np.array([1, 1, 1, 1, 1])
    rows, cols, rec = top.pair_index([a, b], tau=1)
    plain = top.count_matrix(rows, cols, 2, rec)
    doubled = top.count_matrix(rows, cols, 2, rec, rec_weights=[2.0, 2.0])
    dropped = top.count_matrix(rows, cols, 2, rec, rec_weights=[1.0, 0.0])
    assert np.allclose(doubled, 2 * plain)
    assert dropped[1, 1] == 0                       # recording b removed entirely
    assert dropped.sum() == 4


def test_ck_test_refuses_a_lag_the_recordings_cannot_support():
    """n=5 at a 60s lag needs 9,000 frames against a 6,321-frame maximum. That
    must surface as a stated refusal, not as an empty count matrix."""
    states = [np.random.default_rng(i).integers(0, 4, 300) for i in range(20)]
    res = top.ck_test(states, 4, tau=60, fps=30.0, n_max=5)
    assert res["ok"]
    refused = [r for r in res["rows"] if not r["ok"]]
    assert refused, "no row was refused despite the lag exceeding the budget"
    assert "min recording length" in refused[-1]["reason"]


def test_ck_error_is_near_zero_for_a_genuinely_markov_chain():
    """The positive control for the CK test itself."""
    rng = np.random.default_rng(0)
    P = np.array([[0.90, 0.08, 0.02],
                  [0.05, 0.90, 0.05],
                  [0.02, 0.08, 0.90]])
    states = []
    for r in range(40):
        s = np.empty(4000, dtype=int)
        s[0] = rng.integers(0, 3)
        for i in range(1, 4000):
            s[i] = rng.choice(3, p=P[s[i - 1]])
        states.append(s)
    res = top.ck_test(states, 3, tau=5, fps=30.0, n_max=4)
    errs = [r["err"] for r in res["rows"] if r["ok"]]
    assert len(errs) == 3
    assert max(errs) < 0.02, f"CK error on an exactly Markov chain: {errs}"


def _markov_chain(P, n_rec, n_frames, seed=0):
    rng = np.random.default_rng(seed)
    n = P.shape[0]
    out = []
    for _ in range(n_rec):
        s = np.empty(n_frames, dtype=int)
        s[0] = rng.integers(0, n)
        u = rng.random(n_frames)
        cdf = np.cumsum(P, axis=1)
        for i in range(1, n_frames):
            s[i] = int(np.searchsorted(cdf[s[i - 1]], u[i]))
        out.append(s)
    return out


def test_bootstrap_intervals_bracket_the_point_estimate():
    P = np.array([[0.90, 0.08, 0.02],
                  [0.05, 0.90, 0.05],
                  [0.02, 0.08, 0.90]])
    states = _markov_chain(P, n_rec=30, n_frames=2000, seed=0)
    boot = top.bootstrap_its(states, 3, [5, 10], 30.0, n_boot=40, seed=0, k=3)

    for i in range(2):
        point, lo, hi = boot["point"][i, 1], boot["lo"][i, 1], boot["hi"][i, 1]
        assert np.isfinite(point) and lo <= point <= hi
        assert hi / lo < 2.0, "interval implausibly wide on 60k well-mixed frames"
    assert np.all(boot["same_support_frac"] == 1.0)
    assert np.all(boot["mode_stability"][:, 1] > 0.9), (
        "a well-separated process should be identified in every replicate")


def test_the_bootstrap_reweights_recordings_as_whole_blocks():
    """Transitions never cross a recording, so the count matrix is exactly a sum
    of per-recording matrices -- which is what makes a replicate a reweighting
    rather than a recount, and what makes the interval honest about the fact
    that recordings, not frames, are the independent units."""
    states = _markov_chain(np.array([[0.8, 0.2], [0.3, 0.7]]), 4, 500, seed=1)
    rows, cols, rec = top.pair_index(states, 1)
    total = top.count_matrix(rows, cols, 2, rec)
    parts = [top.count_matrix(rows, cols, 2, rec,
                              rec_weights=np.eye(4)[r]) for r in range(4)]
    assert np.allclose(sum(parts), total)


def test_holdout_ck_agrees_with_pooled_ck_on_a_genuinely_markov_chain():
    """The positive control for the holdout: when the process really is one
    stationary chain, fitting on half the recordings and testing on the other
    half must give the same answer as fitting and testing on all of them."""
    P = np.array([[0.90, 0.08, 0.02],
                  [0.05, 0.90, 0.05],
                  [0.02, 0.08, 0.90]])
    states = _markov_chain(P, n_rec=40, n_frames=3000, seed=2)
    pooled = top.ck_test(states, 3, tau=5, fps=30.0, n_max=4)
    held = top.holdout_ck(states, 3, tau=5, fps=30.0, n_max=4, seed=0)

    assert held["ok"] and held["n_shared_states"] == 3
    for a, b in zip([r for r in pooled["rows"] if r["ok"]],
                    [r for r in held["rows"] if r["ok"]]):
        assert b["err_holdout"] < 0.05
        assert abs(b["err_holdout"] - a["err"]) < 0.05


def test_stratifying_reveals_a_slow_process_that_pooling_manufactured():
    """The failure mode this analysis is most exposed to, and its mitigation.

    Two sub-populations that occupy different regions and are never observed
    moving between them do not become one process by being concatenated. The
    pooled chain is near-decomposable, and near-decomposability *is*
    metastability mathematically -- the slow eigenvalue is real, the plateau
    will look convincing, and nothing about the pooled numbers says the states
    are animals rather than behaviors.

    Nothing in the operator can detect this, and it is important to say so. What
    detects it is refitting within each group: here the pooled chain reports a
    slow process two orders of magnitude longer than anything present in either
    group alone. That is why `--stratify` is required rather than advisory.
    """
    fast = np.array([[0.5, 0.5], [0.5, 0.5]])
    eps = 2e-4
    P = np.zeros((4, 4))
    P[:2, :2] = fast * (1 - eps)
    P[2:, 2:] = fast * (1 - eps)
    P[0, 2] = P[1, 3] = eps
    P[2, 0] = P[3, 1] = eps
    P = P / P.sum(axis=1, keepdims=True)

    group_a = _markov_chain(np.array([[0.5, 0.5], [0.5, 0.5]]), 20, 4000, seed=3)
    group_b = [s + 2 for s in
               _markov_chain(np.array([[0.5, 0.5], [0.5, 0.5]]), 20, 4000, seed=4)]
    # A handful of cross-group frames, so the pooled chain is connected.
    bridged = [s.copy() for s in group_a + group_b]
    bridged[0][::2000] = 2
    bridged[-1][::2000] = 1

    pooled = top.operator_at_lag(bridged, 4, tau=1, k=4)
    assert pooled["ok"]
    assert pooled["connected"]["n_components"] == 1, "expected one class"
    t_pooled, _ = top.implied_timescales(pooled["eigenvalues"], 1, 30.0)

    within = top.operator_at_lag(group_a, 4, tau=1, k=4)
    t_within, _ = top.implied_timescales(within["eigenvalues"], 1, 30.0)

    assert t_pooled[1] > 50 * t_within[1], (
        f"pooled t2={t_pooled[1]:.4f}s vs within-group {t_within[1]:.4f}s -- "
        "the pooling artifact should dominate")


def test_make_taus_converts_seconds_at_the_boundary_and_deduplicates():
    """Seconds in, frames out. A hardcoded frame count would mean an 8x
    different real-world window between a 30 fps and a 250 fps rig."""
    at30 = top.make_taus(30.0, tau_max_s=20.0, n_tau=28)
    at250 = top.make_taus(250.0, tau_max_s=20.0, n_tau=28)
    assert at30.min() == 1 and at30.max() == 600
    assert at250.max() == 5000
    assert np.all(np.diff(at30) > 0)                # strictly increasing
    assert at30.size == np.unique(at30).size
>>>>>>> 1deb112fe3f70a3b9c20d11ea35f7ec43986b068
