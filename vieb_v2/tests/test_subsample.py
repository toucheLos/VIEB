import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from representation.subsample import (  # noqa: E402
    arclength_indices, per_bout_indices, subsample_sessions,
)


def test_per_bout_equalises_long_and_short_bouts():
    # The duration-to-mass advantage: a 300-frame bout and a 15-frame bout
    # should contribute the same number of points afterwards.
    labels = np.r_[np.zeros(300, dtype=int), np.ones(15, dtype=int)]
    idx = per_bout_indices(labels, n_per_bout=10)
    assert (idx < 300).sum() == 10
    assert (idx >= 300).sum() == 10


def test_per_bout_keeps_short_bouts_whole():
    labels = np.r_[np.zeros(3, dtype=int), np.ones(50, dtype=int)]
    idx = per_bout_indices(labels, n_per_bout=10)
    assert (idx < 3).sum() == 3          # cannot take more than exist
    assert (idx >= 3).sum() == 10


def test_per_bout_treats_repeated_visits_as_separate_bouts():
    labels = np.array([0] * 20 + [1] * 20 + [0] * 20)
    idx = per_bout_indices(labels, n_per_bout=5)
    assert idx.size == 15                # three bouts, five each


def test_per_bout_samples_are_spread_not_clumped():
    labels = np.zeros(100, dtype=int)
    idx = per_bout_indices(labels, n_per_bout=5)
    assert idx.min() == 0 and idx.max() == 99


def test_arclength_cancels_the_speed_term():
    # Linear sampling density is 1/speed; sampling uniformly in arclength
    # should give both behaviors the same density per unit path length.
    # delta must exceed both step sizes for this to be achievable -- see
    # test_arclength_cannot_upsample_a_fast_behavior.
    slow = np.cumsum(np.full((1000, 1), 0.01), axis=0)   # path length ~10
    fast = np.cumsum(np.full((1000, 1), 1.00), axis=0)   # path length ~1000

    n_slow = arclength_indices(slow, delta=2.0).size
    n_fast = arclength_indices(fast, delta=2.0).size

    # Both are sampled every 2.0 of arclength, so counts track path length
    # (100x) rather than duration (equal), which is the whole point.
    assert 80 < n_fast / n_slow < 125


def test_arclength_cannot_upsample_a_fast_behavior():
    # A real limitation of the mitigation, not a bug: resampling can only
    # discard frames, never invent them. When delta is finer than the
    # frame-to-frame step, every frame is already kept and the density stays
    # capped at the frame rate. So arclength resampling can bring a slow
    # behavior DOWN to a fast one's density but can never raise the fast one
    # up -- the fast behavior is under-sampled at acquisition time.
    fast = np.cumsum(np.full((1000, 1), 1.0), axis=0)    # step 1.0 per frame
    kept = arclength_indices(fast, delta=0.1)            # 10x finer than step
    assert kept.size == 1000                              # capped, not 10000


def test_arclength_on_a_stationary_trajectory_keeps_almost_nothing():
    still = np.zeros((500, 2))
    assert arclength_indices(still, delta=0.1).size <= 1


def test_arclength_rejects_bad_delta():
    with pytest.raises(ValueError):
        arclength_indices(np.zeros((10, 2)), delta=0.0)


def test_subsample_none_is_the_identity():
    sessions = [np.zeros((10, 2)), np.zeros((5, 2))]
    out, idx = subsample_sessions(sessions, "none")
    assert [len(o) for o in out] == [10, 5]
    assert np.array_equal(idx[0], np.arange(10))


def test_subsample_indices_map_back_to_original_frames():
    # Cluster sizes stop encoding occupancy once subsampled, so the index has
    # to be usable to recover durations afterwards.
    rng = np.random.default_rng(0)
    sessions = [rng.normal(size=(200, 2)).cumsum(axis=0)]
    out, idx = subsample_sessions(sessions, "arclength", delta=1.0)
    assert np.allclose(out[0], sessions[0][idx[0]])


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        subsample_sessions([np.zeros((5, 2))], "magic")
