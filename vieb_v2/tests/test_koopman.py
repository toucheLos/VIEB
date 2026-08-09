"""Known-answer verification of the topology extractor.

This method extracts topology. If the extraction is wrong on a system whose
answer is known analytically, nothing it says about real data is
interpretable -- so these tests are a gate, not a regression net.

**The test system, and why it is not the one originally suggested.** The brief
proposed a damped oscillator plus a Van der Pol oscillator in separate regions
of the plane. That system verifies two of the four required properties but
cannot verify the other two:

  * *No separatrix exists between them.* Two dynamically disjoint regions share
    no boundary at which a small perturbation flips the outcome, so there is
    nothing for the separatrix check to find. It would pass vacuously.
  * *Van der Pol's period has no closed form* except in the mu -> 0 limit. A
    test asserting a hard-coded period would be asserting a number obtained the
    same way the code obtains it -- it would pass for the wrong reason.

Used here instead is a single planar system in polar form:

    r' = -k r (r - a)(r - b),    theta' = omega        with 0 < a < b

whose entire topology is known in closed form:

  * a stable **fixed point** at r = 0            (r' < 0 for 0 < r < a)
  * an unstable limit cycle at r = a -- **the separatrix** (r' changes sign)
  * a stable **limit cycle** at r = b, of period exactly 2*pi/omega, because
    theta' is constant so the circulation is uniform
  * **basins** known per frame: r < a falls to the fixed point, r > a to the
    cycle

That is all four properties in one connected system, each with an analytic
answer rather than a numerically-estimated one.

Parameters are chosen so the two attractors are genuinely distinguishable:
`k` sets both the contraction onto the origin (rate k*a*b) and the transverse
contraction onto the cycle (rate k*b*(b-a)), and it is tuned so the origin's
|lambda| sits below 1 - unit_tol while the cycle's stays within unit_tol of 1.
The cycle's modulus is sqrt of the transverse multiplier -- the along-flow
direction is neutral -- which is why an attracting cycle still reports
|lambda| ~ 1.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from representation.koopman import (  # noqa: E402
    NOISE_LABEL, classify_modes, dmd, extract_topology, snapshot_pairs,
)

FPS = 30.0
DT = 1.0 / FPS
A = 1.0                      # unstable cycle -- the separatrix
B = 2.0                      # stable limit cycle
K = 1.0                      # radial rate
PERIOD_S = 1.0               # of the stable cycle, exactly
OMEGA = 2.0 * np.pi / PERIOD_S
NOISE = 0.005


def _velocity(p):
    """The vector field, in Cartesian coordinates."""
    x, y = p[..., 0], p[..., 1]
    r = np.hypot(x, y)
    radial = -K * r * (r - A) * (r - B)
    safe = np.where(r > 1e-12, r, 1.0)
    return np.stack([radial * (x / safe) - OMEGA * y,
                     radial * (y / safe) + OMEGA * x], axis=-1)


def _integrate(p0, n_frames, rng):
    """Fixed-step RK4 with a small stochastic kick each frame."""
    out = np.empty((n_frames, 2))
    p = np.asarray(p0, dtype=np.float64)
    for t in range(n_frames):
        out[t] = p
        k1 = _velocity(p)
        k2 = _velocity(p + 0.5 * DT * k1)
        k3 = _velocity(p + 0.5 * DT * k2)
        k4 = _velocity(p + DT * k3)
        p = p + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        p = p + rng.normal(scale=NOISE, size=2)
    return out


def _make_sessions(seed=0):
    """Recordings seeded across both basins and along the separatrix."""
    rng = np.random.default_rng(seed)
    sessions = []
    for radius, count, n_frames in [((0.15, 0.80), 8, 400),    # inside
                                    ((0.88, 1.12), 6, 600),    # on the boundary
                                    ((1.25, 3.00), 12, 600)]:  # outside
        for _ in range(count):
            r0 = rng.uniform(*radius)
            angle = rng.uniform(0, 2 * np.pi)
            sessions.append(_integrate(
                [r0 * np.cos(angle), r0 * np.sin(angle)], n_frames, rng))
    return sessions


@pytest.fixture(scope="module")
def topology():
    """One extraction, shared by the four verification checks."""
    return extract_topology(_make_sessions(), fps=FPS, n_regions=60, seed=0,
                            knn=12, plausible_hz=(0.2, 12.0))


@pytest.fixture(scope="module")
def radii(topology):
    """Radius of every frame, in the same row order as `labels`."""
    return np.hypot(*np.concatenate(_make_sessions(), axis=0).T)


# ---------------------------------------------------------------------------
# The four required checks
# ---------------------------------------------------------------------------

def test_fixed_point_is_found(topology):
    """VERIFICATION 1 -- found, with |lambda| < 1 and ||v|| ~ 0."""
    fixed = topology["fixed_points"]
    assert len(fixed) >= 1, "the stable fixed point at r=0 was not found"

    point = max(fixed, key=lambda a: a["basin_frames"])
    assert point["max_magnitude"] < 1.0, (
        f"fixed point is not contracting: max |lambda| = {point['max_magnitude']}"
    )

    cycle_speed = max(a["v_norm"] for a in topology["attractors"])
    assert point["v_norm"] < 0.1 * cycle_speed, (
        f"||v|| at the fixed point ({point['v_norm']:.4f}) is not ~0 relative "
        f"to the flow elsewhere ({cycle_speed:.4f})"
    )
    assert point["basin_frames"] > 0


def test_limit_cycle_is_found_with_the_correct_period(topology):
    """VERIFICATION 2 -- found, with the analytically known period."""
    cycles = topology["limit_cycles"]
    assert len(cycles) >= 1, "the stable limit cycle at r=b was not found"

    cycle = max(cycles, key=lambda a: a["basin_frames"])

    # 3b's own criterion: |lambda| near 1 with a nonzero imaginary part.
    assert 0.9 < cycle["max_magnitude"] <= 1.05
    assert cycle["eigen_period_s"] is not None

    # theta' is constant, so the true period is exactly 2*pi/omega.
    assert cycle["eigen_period_s"] == pytest.approx(PERIOD_S, rel=0.15), (
        f"eigenvalue period {cycle['eigen_period_s']:.3f}s != {PERIOD_S}s"
    )
    assert cycle["empirical_period_s"] == pytest.approx(PERIOD_S, rel=0.15), (
        f"return-time period {cycle['empirical_period_s']:.3f}s != {PERIOD_S}s"
    )
    assert cycle["plausible"] is True


def test_basin_assignment_matches_the_analytic_basins(topology, radii):
    """VERIFICATION 3 -- labels agree with the closed-form basins.

    Frames within 0.2 of the separatrix are excluded: that band is what
    VERIFICATION 4 is about, and the analytic basin there is not what the data
    shows once noise can carry a trajectory across r = a.
    """
    labels = topology["labels"]
    fixed_ids = {a["attractor"] for a in topology["fixed_points"]}
    cycle_ids = {a["attractor"] for a in topology["limit_cycles"]}

    truth_is_fixed = radii < A
    scored = (labels >= 0) & (np.abs(radii - A) > 0.2)
    assert scored.sum() > 0.5 * labels.size, "too few frames left to score"

    predicted_fixed = np.isin(labels, list(fixed_ids))
    predicted_cycle = np.isin(labels, list(cycle_ids))
    assert (predicted_fixed | predicted_cycle)[scored].all(), (
        "some scored frames fell into an attractor that is neither the fixed "
        "point nor the limit cycle"
    )

    correct = np.where(truth_is_fixed, predicted_fixed, predicted_cycle)
    accuracy = float(correct[scored].mean())
    assert accuracy > 0.9, f"basin accuracy {accuracy:.3f} against known basins"


def test_separatrix_is_detected(topology, radii):
    """VERIFICATION 4 -- flagged frames sit on the known circle r = a."""
    labels = topology["labels"]
    flagged = labels == NOISE_LABEL

    assert flagged.any(), "no separatrix frames were detected at all"
    assert flagged.mean() < 0.5, (
        f"{flagged.mean():.1%} of frames flagged -- the basins were not resolved"
    )

    distance = np.abs(radii - A)
    assert distance[flagged].mean() < distance[~flagged].mean(), (
        "flagged frames are no closer to the separatrix than unflagged ones"
    )
    near_band = float((distance[flagged] < 0.4).mean())
    assert near_band > 0.5, (
        f"only {near_band:.1%} of flagged frames lie near r={A}; the detector "
        f"is flagging something other than the separatrix"
    )


# ---------------------------------------------------------------------------
# Supporting checks
# ---------------------------------------------------------------------------

def test_snapshot_pairs_never_straddle_a_recording_boundary():
    # Three recordings of 10, 7 and 5 frames give (10-1)+(7-1)+(5-1) = 19
    # pairs. Concatenating first would give 22-1 = 21 -- the two extra are the
    # fabricated cross-animal transitions.
    sessions = [np.arange(n * 2, dtype=float).reshape(n, 2)
                for n in (10, 7, 5)]
    X, Xp = snapshot_pairs(sessions)

    assert X.shape[1] == 19
    assert Xp.shape[1] == 19
    # Every pair must be one frame apart; a straddling pair would jump.
    assert np.allclose(Xp - X, 2.0)


def test_a_concatenated_array_is_rejected():
    stacked = np.arange(40, dtype=float).reshape(20, 2)
    with pytest.raises(TypeError, match="fabricated"):
        snapshot_pairs(stacked)


def test_region_pairs_are_selected_by_origin_not_by_both_endpoints():
    sessions = [np.arange(20, dtype=float).reshape(10, 2)]
    ids = [np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])]
    X, _ = snapshot_pairs(sessions, ids, region=0)
    # Frames 0,1,2,6,7 originate in region 0, so all five of their forward
    # steps count -- including 2->3 and 7->8, which leave the region. A local
    # operator is the flow map *on* a region; the image need not be the region.
    assert X.shape[1] == 5


def test_dmd_recovers_a_known_linear_operator():
    rng = np.random.default_rng(0)
    angle = 0.3
    true = np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]]) * 0.95

    trajectory = np.empty((300, 2))
    trajectory[0] = [1.0, 0.5]
    for t in range(1, 300):
        trajectory[t] = true @ trajectory[t - 1] + rng.normal(scale=1e-6, size=2)

    X, Xp = snapshot_pairs([trajectory])
    fit = dmd(X, Xp)

    recovered = np.sort_complex(fit["eigenvalues"])
    expected = np.sort_complex(np.linalg.eigvals(true))
    assert np.allclose(recovered, expected, atol=1e-3)
    assert fit["residual"] < 1e-3


def test_growing_modes_are_flagged():
    modes = classify_modes(np.array([1.4 + 0.0j, 0.5 + 0.0j, 0.99 + 0.05j]), FPS)
    assert [m["band"] for m in modes] == ["growing", "decaying", "oscillatory"]
    assert [m["flagged"] for m in modes] == [True, False, False]


def test_frequency_is_reported_in_hz_from_the_eigenvalue_angle():
    # A mode advancing pi/15 rad per frame at 30 fps completes one cycle every
    # 30 frames -- 1 Hz.
    lam = np.exp(1j * (2 * np.pi / 30))
    mode = classify_modes(np.array([lam]), FPS)[0]
    assert mode["frequency_hz"] == pytest.approx(1.0)
    assert mode["period_frames"] == pytest.approx(30.0)


def test_topology_reports_that_no_clustering_was_run(topology):
    assert topology["report"]["clustering"] == "none -- states are basins of attraction"
    assert topology["report"]["n_fixed_points"] >= 1
    assert topology["report"]["n_limit_cycles"] >= 1
    assert 0.0 < topology["report"]["transition_fraction"] < 0.5


def test_labels_carry_the_noise_label_convention(topology):
    """-1 must mean the same thing it means in metrics, so the existing
    confound diagnostics score basin labels without modification."""
    from representation.metrics import cluster_metrics, speed_diagnostics

    labels = topology["labels"]
    stacked = np.concatenate(_make_sessions(), axis=0)

    metrics = cluster_metrics(labels)
    assert metrics["n_states"] == topology["report"]["n_attractors"]
    assert metrics["noise_frac"] == pytest.approx(
        topology["report"]["transition_fraction"])

    speed = speed_diagnostics(labels, stacked)
    assert speed["noise_speed_ratio"] is not None
    assert speed["size_speed_rank_corr"] is not None


def test_save_topology_matches_the_labels_npz_shape(topology, tmp_path):
    from representation import checkpoints, koopman

    path = koopman.save_topology(str(tmp_path / "koopman_labels.npz"), topology)
    data, meta = checkpoints.load(path)

    assert set(data.files) >= {"labels", "probabilities", "index", "meta"}
    assert data["labels"].shape == topology["labels"].shape
    assert data["index"].shape[1] == 2
    assert "separatrix" in meta["noise_label_means"]


# ---------------------------------------------------------------------------
# Scale: the subsampled separatrix index
# ---------------------------------------------------------------------------

def test_subsampled_knn_index_leaves_the_topology_standing():
    """VERIFICATION 4b -- the approximation must not change the answer.

    Above `knn_sample` frames the separatrix neighbour index is fitted on a
    subsample rather than on all N. `_make_sessions` is 14000 frames, so the
    threshold is forced far below that to exercise the path the real run takes
    -- at project scale (28.6M frames) it is always the subsampled one.
    """
    sessions = _make_sessions()
    n_frames = sum(len(s) for s in sessions)

    exact = extract_topology(sessions, fps=FPS, n_regions=60, seed=0, knn=12,
                             plausible_hz=(0.2, 12.0), knn_sample=0)
    approx = extract_topology(sessions, fps=FPS, n_regions=60, seed=0, knn=12,
                              plausible_hz=(0.2, 12.0), knn_sample=2000)

    assert exact["report"]["knn_subsampled"] is False
    assert approx["report"]["knn_subsampled"] is True
    assert approx["report"]["knn_sample"] == 2000

    # The topology is a property of the flow, not of the neighbour index --
    # subsampling only decides which frames sit near a separatrix.
    for key in ("n_attractors", "n_fixed_points", "n_limit_cycles"):
        assert approx["report"][key] == exact["report"][key], key

    assert approx["labels"].shape == (n_frames,)
    assert approx["probabilities"].shape == (n_frames,)
    assert 0.0 < approx["report"]["transition_fraction"] < 0.5

    # Every frame is still labelled -- only the neighbour pool was sampled.
    assert (approx["labels"] >= NOISE_LABEL).all()
    agree = float((approx["labels"] == exact["labels"]).mean())
    assert agree > 0.9, f"subsampling moved {1 - agree:.1%} of frames"


def test_subsampled_index_is_deterministic_under_seed():
    sessions = _make_sessions()
    kw = dict(fps=FPS, n_regions=60, seed=0, knn=12, plausible_hz=(0.2, 12.0),
              knn_sample=2000)
    first = extract_topology(sessions, **kw)
    second = extract_topology(sessions, **kw)
    assert np.array_equal(first["labels"], second["labels"])
    assert np.array_equal(first["probabilities"], second["probabilities"])
