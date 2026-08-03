import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from representation.metrics import (  # noqa: E402
    cluster_metrics, speed_diagnostics,
)


def test_v1_convention_divides_by_total_frames_including_noise():
    # v1 (compare.py:2110-2114) divides state counts by total_frames, which
    # includes noise, so the fractions deliberately do NOT sum to 1.
    labels = np.array([0] * 5 + [1] * 3 + [-1] * 2)
    m = cluster_metrics(labels)

    assert m["n_states"] == 2
    assert m["noise_frac"] == 0.2
    assert np.isclose(m["v1_convention"]["largest_state_frac"], 0.5)   # 5/10
    assert np.isclose(sum(m["v1_convention"]["state_fracs"]), 0.8)     # 1 - noise
    # The clean convention normalises over clustered frames only.
    assert np.isclose(m["clustered_only"]["largest_state_frac"], 5 / 8)
    assert np.isclose(sum(m["clustered_only"]["state_fracs"]), 1.0)


def test_entropy_matches_v1_normalised_formula():
    labels = np.array([0] * 6 + [1] * 3 + [-1] * 1)
    m = cluster_metrics(labels)

    fracs = [0.6, 0.3]  # over total_frames = 10, as v1 computes them
    expected = (-sum(f * math.log(f) for f in fracs)) / math.log(2)
    assert np.isclose(m["v1_convention"]["state_entropy"], expected)


def test_entropy_is_zero_for_a_single_state():
    m = cluster_metrics(np.array([0] * 10))
    assert m["v1_convention"]["state_entropy"] == 0.0
    assert m["n_states"] == 1


def test_balanced_states_give_maximal_normalised_entropy():
    labels = np.repeat([0, 1, 2, 3], 25)
    m = cluster_metrics(labels)
    assert np.isclose(m["clustered_only"]["state_entropy"], 1.0)


def test_all_noise_is_handled():
    m = cluster_metrics(np.full(10, -1))
    assert m["n_states"] == 0
    assert m["noise_frac"] == 1.0
    assert m["v1_convention"]["largest_state_frac"] == 0.0


def test_noise_speed_ratio_detects_a_speed_biased_noise_label():
    # The confound's actual prediction: unclustered frames are the fast ones.
    rng = np.random.default_rng(0)
    slow = rng.normal(size=(200, 2)) * 0.01
    fast = np.cumsum(rng.normal(size=(200, 2)), axis=0)
    scores = np.vstack([slow, fast])
    labels = np.r_[np.zeros(200, dtype=int), np.full(200, -1)]

    d = speed_diagnostics(labels, scores)
    assert d["noise_speed_ratio"] > 2.0


def test_noise_speed_ratio_near_one_when_no_bias():
    rng = np.random.default_rng(1)
    scores = rng.normal(size=(400, 2))
    labels = np.r_[np.zeros(200, dtype=int), np.full(200, -1)]
    d = speed_diagnostics(labels, scores)
    assert 0.7 < d["noise_speed_ratio"] < 1.4


def test_per_cluster_reports_size_speed_and_bout_length():
    labels = np.array([0] * 4 + [1] * 6)
    scores = np.arange(10, dtype=float).reshape(10, 1)
    d = speed_diagnostics(labels, scores)

    assert d["per_cluster"][0]["size"] == 4
    assert d["per_cluster"][1]["size"] == 6
    assert d["per_cluster"][0]["mean_bout_frames"] == 4.0


def test_mean_bout_length_averages_separate_runs():
    labels = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0])
    scores = np.zeros((10, 1))
    d = speed_diagnostics(labels, scores)
    assert d["per_cluster"][0]["mean_bout_frames"] == 3.0  # runs of 2 and 4
