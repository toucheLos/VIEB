"""The statistics behind the MoSeq positive control.

These guard two things that were wrong on the first pass and would not have been
visible in the output. The rank-biserial effect size was derived from scipy's
two-sided `statistic`, which is `min(W+, W-)` -- so every syllable reported a
positive effect regardless of direction, including ones whose occupancy had
halved. And the sign-flip null needs to actually be a null: if it cannot come
back empty on exchangeable data, a contrast beating it means nothing.

Synthetic data throughout, with the answer known in advance.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.moseq_control import (  # noqa: E402
    _benjamini_hochberg, arm_profile, paired_contrast, shuffle_null,
)


def _design(n_animals=40, effect=0.0, seed=0):
    """Two arms per animal. `effect` shifts syllable 0 in arm A only."""
    rng = np.random.default_rng(seed)
    fields, rows = [], []
    for a in range(n_animals):
        base = rng.uniform(0.2, 0.4, size=3)
        for day, bump in ((0, 0.0), (1, effect)):
            occ = base + rng.normal(0, 0.01, size=3)
            occ[0] += bump
            occ = np.clip(occ, 1e-6, None)
            rows.append(occ / occ.sum())
            fields.append({"animal": f"m{a}", "day": day, "context": "A",
                           "experiment": "CFC", "no_shock": day == 1})
    return np.array(rows), fields


def _masks(fields):
    day = np.array([f["day"] for f in fields])
    return day == 1, day == 0


def test_benjamini_hochberg_is_monotone_and_bounded():
    p = np.array([0.001, 0.02, 0.03, 0.5, 0.9])
    q = _benjamini_hochberg(p)
    assert np.all(q >= p - 1e-12)
    assert np.all(np.diff(q[np.argsort(p)]) >= -1e-12)
    assert q.max() <= 1.0


def test_benjamini_hochberg_leaves_a_single_test_alone():
    assert _benjamini_hochberg([0.03])[0] == pytest.approx(0.03)


def test_rank_biserial_carries_the_direction_of_the_effect():
    """The bug this test exists for: scipy's two-sided statistic is
    min(W+, W-), so an effect size derived from it is positive for a decrease
    as well as an increase."""
    occ, fields = _design(effect=+0.20, seed=1)
    a, b = _masks(fields)
    up = paired_contrast(occ, fields, a, b, np.arange(3))
    down = paired_contrast(occ, fields, b, a, np.arange(3))

    s0_up = next(r for r in up["rows"] if r["syllable"] == 0)
    s0_down = next(r for r in down["rows"] if r["syllable"] == 0)

    assert s0_up["median_diff"] > 0 and s0_up["rank_biserial"] > 0.5
    assert s0_down["median_diff"] < 0 and s0_down["rank_biserial"] < -0.5
    assert s0_up["rank_biserial"] == pytest.approx(-s0_down["rank_biserial"])


def test_a_real_effect_is_detected_and_localized_to_the_right_syllable():
    occ, fields = _design(effect=+0.20, seed=2)
    a, b = _masks(fields)
    res = paired_contrast(occ, fields, a, b, np.arange(3))
    by_id = {r["syllable"]: r for r in res["rows"]}
    assert by_id[0]["q"] < 0.05
    assert by_id[0]["median_diff"] > 0.05
    assert res["n_animals"] == 40


def test_no_effect_gives_no_detections():
    occ, fields = _design(effect=0.0, seed=3)
    a, b = _masks(fields)
    assert paired_contrast(occ, fields, a, b, np.arange(3))["n_significant"] == 0


def test_paired_contrast_reports_arm_a_minus_arm_b():
    """Sign convention: `mask_b` is the baseline, so a positive median_diff
    means more of that syllable in arm A. Getting this backwards would invert
    every biological conclusion while looking entirely reasonable."""
    occ, fields = _design(effect=+0.20, seed=4)
    a, b = _masks(fields)
    res = paired_contrast(occ, fields, a, b, np.arange(3))
    s0 = next(r for r in res["rows"] if r["syllable"] == 0)
    assert s0["mean_a"] > s0["mean_b"]
    assert s0["median_diff"] == pytest.approx(s0["mean_a"] - s0["mean_b"], abs=0.03)


def test_paired_contrast_refuses_too_few_pairs():
    occ, fields = _design(n_animals=3, effect=0.2, seed=5)
    a, b = _masks(fields)
    assert "error" in paired_contrast(occ, fields, a, b, np.arange(3))


def test_paired_contrast_uses_animals_not_sessions_as_the_unit():
    """One animal contributing many sessions must not outvote the others."""
    occ, fields = _design(n_animals=20, effect=0.0, seed=6)
    extra_occ = np.tile(np.array([[0.9, 0.05, 0.05]]), (30, 1))
    extra_fields = [{"animal": "m0", "day": 1, "context": "A",
                     "experiment": "CFC", "no_shock": True} for _ in range(30)]
    occ2 = np.vstack([occ, extra_occ])
    fields2 = fields + extra_fields
    a, b = _masks(fields2)
    res = paired_contrast(occ2, fields2, a, b, np.arange(3))
    assert res["n_animals"] == 20
    assert res["n_significant"] == 0


def test_the_sign_flip_null_comes_back_empty_on_exchangeable_data():
    """If the null cannot find zero, beating it proves nothing."""
    occ, fields = _design(effect=0.0, seed=7)
    a, b = _masks(fields)
    null = shuffle_null(occ, fields, a, b, np.arange(3), seed=0, n_repeats=40)
    assert null["median_significant"] == 0
    assert null["frac_repeats_with_none"] > 0.8


def test_the_sign_flip_null_rarely_reaches_a_real_effect():
    """Rarely, not never. Occupancy fractions are compositional -- they sum to
    1 -- so with only three syllables the tests are almost perfectly dependent
    and the null's rejection count is overdispersed: usually zero, occasionally
    a burst that reaches a modest observed count. Asserting an exact zero here
    would be asserting away that dependence. (On the real 35-syllable data the
    observed count is 33 and the null reaches it in 0 of 100 repeats.)"""
    occ, fields = _design(effect=+0.20, seed=8)
    a, b = _masks(fields)
    observed = paired_contrast(occ, fields, a, b, np.arange(3))["n_significant"]
    null = shuffle_null(occ, fields, a, b, np.arange(3), seed=0, n_repeats=40,
                        observed=observed)
    assert observed >= 1
    assert null["median_significant"] < observed
    assert null["frac_null_at_or_above_observed"] <= 0.05


def test_arm_profile_groups_every_cell_of_the_design():
    occ, fields = _design(effect=0.1, seed=9)
    profile = arm_profile(occ, fields, np.arange(3))
    assert set(profile) == {"CFC d0 A", "CFC d1 A/NS"}
    assert profile["CFC d0 A"]["n"] == 40
    assert profile["CFC d1 A/NS"]["s0"] > profile["CFC d0 A"]["s0"]
