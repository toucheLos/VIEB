"""Tests for ml/validation_stats.py (statistics-methods branch): the
Nakagawa & Schielzeth repeatability statistic and the transition-graph
modularity / bridge-state check."""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# compute_repeatability_R
# ---------------------------------------------------------------------------

def test_repeatability_high_when_animal_dominates_variance():
    from ml.validation_stats import compute_repeatability_R

    animals = np.repeat(["a1", "a2", "a3", "a4", "a5"], 4)
    days = np.tile([1, 2, 3, 4], 5)
    animal_effect = {"a1": 0.1, "a2": 0.3, "a3": 0.5, "a4": 0.7, "a5": 0.9}
    rng = np.random.default_rng(0)
    state0 = np.array([animal_effect[a] for a in animals]) + rng.normal(0, 0.02, 20)

    df = pd.DataFrame({"animal_id": animals, "day": days, "state_0_frac": state0})
    result = compute_repeatability_R(df, ["state_0_frac"])

    assert not result["skipped"]
    assert result["mean_R"] > 0.9
    assert result["n_states_scored"] == 1


def test_repeatability_low_when_variance_is_within_animal_noise():
    """With a larger sample (10 animals x 8 sessions), a null (no true
    between-animal effect) case's R estimate should land closer to 0 than
    the high-repeatability case's R (>0.9) does — a small positive R from
    sampling noise alone is expected and not a failure."""
    from ml.validation_stats import compute_repeatability_R

    n_animals, n_sessions = 10, 8
    animals = np.repeat([f"a{i}" for i in range(n_animals)], n_sessions)
    days = np.tile(range(n_sessions), n_animals)
    rng = np.random.default_rng(0)
    state0 = rng.normal(0.5, 0.2, n_animals * n_sessions)  # pure noise, independent of animal

    df = pd.DataFrame({"animal_id": animals, "day": days, "state_0_frac": state0})
    result = compute_repeatability_R(df, ["state_0_frac"])

    assert not result["skipped"]
    assert result["mean_R"] < 0.5


def test_repeatability_skips_gracefully_with_insufficient_sessions():
    from ml.validation_stats import compute_repeatability_R

    df = pd.DataFrame({"animal_id": ["a1", "a2"], "day": [1, 1], "state_0_frac": [0.1, 0.2]})
    result = compute_repeatability_R(df, ["state_0_frac"])

    assert result["skipped"]
    assert result["mean_R"] is None
    assert "day" in result["reason"]


def test_repeatability_skips_gracefully_without_animal_column():
    from ml.validation_stats import compute_repeatability_R

    df = pd.DataFrame({"state_0_frac": [0.1, 0.2, 0.3]})
    result = compute_repeatability_R(df, ["state_0_frac"], animal_col="animal_id")

    assert result["skipped"]
    assert "animal_id" in result["reason"]


def test_repeatability_missing_state_column_reported_per_state():
    from ml.validation_stats import compute_repeatability_R

    df = pd.DataFrame({
        "animal_id": ["a1", "a1", "a2", "a2"],
        "day": [1, 2, 1, 2],
        "state_0_frac": [0.1, 0.15, 0.5, 0.55],
    })
    result = compute_repeatability_R(df, ["state_0_frac", "state_1_frac"])
    assert result["per_state"]["state_1_frac"]["skipped"]
    assert not result["per_state"]["state_0_frac"]["skipped"]


# ---------------------------------------------------------------------------
# compute_transition_modularity
# ---------------------------------------------------------------------------

def test_modularity_flags_bridge_state_between_two_blocks():
    from ml.validation_stats import compute_transition_modularity

    n = 6
    counts = np.zeros((n, n))
    block1, block2 = [0, 1, 2], [3, 4, 5]
    for i in block1:
        for j in block1:
            if i != j:
                counts[i, j] = 20
    for i in block2:
        for j in block2:
            if i != j:
                counts[i, j] = 20
    # State 2 bridges roughly evenly into block2
    counts[2, 3] = counts[3, 2] = 15
    counts[2, 4] = counts[4, 2] = 15

    result = compute_transition_modularity(counts)

    assert not result["skipped"]
    assert result["modularity_Q"] > 0
    assert 2 in result["possible_split_states"]
    assert isinstance(result["possible_split_states"], list)
    assert all(isinstance(x, int) for x in result["possible_split_states"])


def test_modularity_no_bridge_states_in_cleanly_separated_graph():
    from ml.validation_stats import compute_transition_modularity

    n = 6
    counts = np.zeros((n, n))
    for i in [0, 1, 2]:
        for j in [0, 1, 2]:
            if i != j:
                counts[i, j] = 20
    for i in [3, 4, 5]:
        for j in [3, 4, 5]:
            if i != j:
                counts[i, j] = 20

    result = compute_transition_modularity(counts)
    assert not result["skipped"]
    assert result["possible_split_states"] == []


def test_modularity_skips_gracefully_with_too_few_states():
    from ml.validation_stats import compute_transition_modularity

    counts = np.array([[0.0, 5.0], [5.0, 0.0]])
    result = compute_transition_modularity(counts)
    assert result["skipped"]
    assert result["possible_split_states"] == []


def test_modularity_skips_gracefully_with_no_transitions():
    from ml.validation_stats import compute_transition_modularity

    counts = np.zeros((4, 4))
    result = compute_transition_modularity(counts)
    assert result["skipped"]


def test_modularity_output_shape_matches_video_stories_hook_expectation():
    """views/video_stories.py's load_possible_split_states() reads
    cluster_info.json["possible_split_states"] and accepts either a plain
    list of ints, or a dict of {state_id: truthy}. We emit a plain list."""
    from ml.validation_stats import compute_transition_modularity

    n = 4
    counts = np.full((n, n), 5.0)
    np.fill_diagonal(counts, 0.0)
    result = compute_transition_modularity(counts)

    possible_split = result["possible_split_states"]
    assert isinstance(possible_split, list)
    # Simulate exactly what load_possible_split_states() does with our output.
    reconstructed = {int(x) for x in possible_split}
    assert reconstructed == set(possible_split)
