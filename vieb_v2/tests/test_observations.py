"""Tests for the restored observation channels.

The centring test is not ceremony. A half-frame shift from an even-width moving
average is exactly the class of error that cost hours in ExBias, it is invisible
in any summary statistic, and it biases every derivative built on top of it.
"""

from __future__ import annotations

import numpy as np
import pytest

from representation import observations as obs


# ------------------------------------------------------------------ smoothing

def test_smooth_does_not_shift_a_peak():
    """Zero phase shift, asserted as symmetry about the true peak.

    Not via argmax: smoothing a delta with a boxcar produces a flat plateau of
    `window` tied values, and argmax returns the first of them, which looks like
    a leftward shift of half a window and is not one. Symmetry about index 50 is
    the property that actually distinguishes a centred filter from a lagged one.
    """
    x = np.zeros(101)
    x[50] = 1.0
    for window in (3, 5, 11, 21):
        out = obs.smooth(x, window)
        assert np.allclose(out[:50][::-1], out[51:]), f"asymmetric at w={window}"
        assert out[50] == pytest.approx(1.0 / window)
        assert out.sum() == pytest.approx(1.0)


def test_smooth_averages_the_window_centred_on_each_sample():
    """The defining identity, checked against an explicit slice."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=60)
    out = obs.smooth(x, 5)
    for k in (10, 25, 49):
        assert out[k] == pytest.approx(x[k - 2:k + 3].mean())


def test_smooth_preserves_length_and_is_edge_safe():
    x = np.arange(20, dtype=float)
    out = obs.smooth(x, 5)
    assert out.shape == x.shape
    # Edge replication, not zero padding: a ramp stays monotone at the ends.
    assert np.all(np.diff(out) > 0)


def test_smooth_window_is_forced_odd():
    """An even window has no integer centre, so it would shift by half a frame."""
    assert obs._odd_window(0.2, 30) % 2 == 1   # 6 frames -> 7
    assert obs._odd_window(0.1, 30) % 2 == 1   # 3 frames -> 3
    assert obs._odd_window(0.0, 30) == 1       # disabled


def test_smooth_is_identity_for_trivial_window():
    x = np.random.default_rng(0).normal(size=30)
    assert np.allclose(obs.smooth(x, 1), x)


# ---------------------------------------------------------- speed and turning

def test_speed_recovers_a_known_constant_velocity():
    fps, t = 30.0, 300
    # 4 px/frame east = 120 px/s.
    xy = np.column_stack([4.0 * np.arange(t), np.zeros(t)])
    speed, turn = obs.speed_and_turn(xy, np.zeros(t), fps, smooth_s=0.0)
    assert np.allclose(speed, 120.0)
    assert np.allclose(turn, 0.0)


def test_angular_velocity_recovers_a_known_turn_rate():
    fps, t = 30.0, 600
    # heading = -theta, so a clockwise theta ramp is a counter-clockwise heading.
    theta = -0.02 * np.arange(t)
    _, turn = obs.speed_and_turn(np.zeros((t, 2)), theta, fps, smooth_s=0.0)
    assert np.allclose(turn, 0.02 * fps)


def test_angular_velocity_has_no_branch_cut_spikes():
    """Unwrapping first: a steady turn through +/-pi must stay steady."""
    fps, t = 30.0, 900
    theta = np.angle(np.exp(1j * 0.05 * np.arange(t)))   # wrapped into (-pi, pi]
    _, turn = obs.speed_and_turn(np.zeros((t, 2)), theta, fps, smooth_s=0.0)
    expected = -0.05 * fps
    assert np.allclose(turn, expected, atol=1e-6), (
        f"max deviation {np.abs(turn - expected).max():.3f} rad/s -- a 2*pi/dt "
        f"spike would be ~{2 * np.pi * fps:.0f}")


def test_frozen_and_locomoting_are_separated_by_speed():
    fps = 30.0
    frozen = np.zeros((300, 2))
    walking = np.column_stack([3.0 * np.arange(300), np.zeros(300)])
    s_frozen, _ = obs.speed_and_turn(frozen, np.zeros(300), fps)
    s_walk, _ = obs.speed_and_turn(walking, np.zeros(300), fps)
    assert s_frozen.max() < 1e-9
    # Interior only -- see the edge-attenuation test below.
    assert s_walk[10:-10].min() > 80.0


def test_smoothing_attenuates_derivatives_at_a_recording_edge():
    """A known, bounded cost of edge replication, asserted so it stays known.

    Replicating the endpoint flattens a ramp within half a window of each end,
    so speed is under-estimated there. At the default 0.25 s / 30 fps that is 4
    frames at each end of a ~5,400-frame recording -- under 0.2% of frames, and
    biased toward zero rather than toward a spurious spike.
    """
    fps, t = 30.0, 300
    ramp = np.column_stack([3.0 * np.arange(t), np.zeros(t)])
    speed, _ = obs.speed_and_turn(ramp, np.zeros(t), fps, smooth_s=0.25)
    # half frames of edge-replicated smoothing, plus one more because the
    # centred gradient at index k reads k-1 and k+1.
    edge = obs._odd_window(0.25, fps) // 2 + 1
    assert np.allclose(speed[edge:-edge], 90.0)
    assert speed[0] < 90.0                      # attenuated, not amplified
    assert speed[0] > 0.0


def test_short_recording_does_not_crash():
    for t in (0, 1):
        speed, turn = obs.speed_and_turn(np.zeros((t, 2)), np.zeros(t), 30.0)
        assert speed.shape == (t,) and turn.shape == (t,)


def test_theta_length_mismatch_raises():
    with pytest.raises(ValueError, match="same recording"):
        obs.speed_and_turn(np.zeros((10, 2)), np.zeros(9), 30.0)


# ----------------------------------------------------- the boundary invariant

def test_derivatives_never_cross_a_recording_boundary():
    """Two recordings far apart in arena space must produce no junction spike."""
    fps = 30.0
    a = np.zeros((200, 2))                      # frozen at the origin
    b = np.full((200, 2), 5000.0)               # frozen 5000 px away
    frame = [np.column_stack([r, np.zeros(len(r))]) for r in (a, b)]
    chan = obs.channels_all(frame, fps, smooth_s=0.0)

    assert len(chan) == 2
    for c in chan:
        assert c[:, 0].max() < 1e-9, "a frozen recording must have zero speed"

    # The spike a concatenated derivative would produce, for scale.
    naive = np.abs(np.gradient(np.concatenate([a, b])[:, 0])) * fps
    assert naive.max() > 1e4


def test_channels_all_rejects_wrong_shape():
    with pytest.raises(ValueError, match=r"\(T, 3\)"):
        obs.channels_all([np.zeros((10, 2))], 30.0)


# ----------------------------------------------------------------- build()

def _toy(seed=0, n_rec=4, t=500, q=6):
    rng = np.random.default_rng(seed)
    scores = [rng.normal(size=(t, q)) for _ in range(n_rec)]
    frame = []
    for _ in range(n_rec):
        xy = np.cumsum(rng.normal(scale=2.0, size=(t, 2)), axis=0)
        frame.append(np.column_stack([xy, rng.normal(scale=0.1, size=t)]))
    return scores, frame


def test_build_standardises_every_channel_to_unit_variance():
    scores, frame = _toy()
    out, report = obs.build(scores, frame, fps=30.0)
    flat = np.concatenate(out)
    assert flat.shape[1] == 6 + 2
    assert np.allclose(flat.std(axis=0), 1.0, atol=1e-8), (
        "an unstandardised block would let px/s set the geometry")
    assert report["names"][-2:] == list(obs.CHANNEL_NAMES)


def test_build_reports_the_scale_ratio_it_neutralises():
    scores, frame = _toy()
    _, report = obs.build(scores, frame, fps=30.0)
    assert report["scale_ratio_before_standardising"] > 1.0
    assert set(report["channel_scale"]) == set(obs.CHANNEL_NAMES)
    assert len(report["pose_scale"]) == 6


def test_build_preserves_the_per_recording_split():
    scores, frame = _toy(n_rec=5, t=137)
    out, _ = obs.build(scores, frame, fps=30.0)
    assert [len(o) for o in out] == [137] * 5


def test_build_without_channels_is_the_pose_only_control():
    scores, frame = _toy()
    out, report = obs.build(scores, frame, fps=30.0, include=False)
    assert np.concatenate(out).shape[1] == 6
    assert report["channels_included"] is False
    # Both arms standardise, so they differ only in the extra channels.
    assert np.allclose(np.concatenate(out).std(axis=0), 1.0, atol=1e-8)


def test_build_rejects_a_concatenated_array():
    scores, frame = _toy()
    with pytest.raises(TypeError, match="must not cross a boundary"):
        obs.build(np.concatenate(scores), frame, fps=30.0)


def test_build_rejects_mismatched_lengths():
    scores, frame = _toy()
    frame[1] = frame[1][:-5]
    with pytest.raises(ValueError, match="frame has"):
        obs.build(scores, frame, fps=30.0)


def test_smoothing_window_is_seconds_not_frames():
    """The same seconds must mean the same real duration at either rig's fps."""
    assert obs._odd_window(0.25, 30) == 9      # 7.5 -> 8 -> 9 frames, 0.30 s
    assert obs._odd_window(0.25, 250) == 63    # 62.5 -> 62 -> 63 frames, 0.25 s
    # A hardcoded 9 frames would be 0.30 s on Luna and 0.036 s on Spence -- the
    # 8x discrepancy the seconds-at-the-boundary rule exists to prevent.


# -------------------------------------------------------------- degeneracy

def test_degeneracy_detects_a_postural_signature_when_one_exists():
    """Pose that encodes speed by construction must be separable."""
    rng = np.random.default_rng(1)
    scores, frame = [], []
    for _ in range(12):
        # Bouts, not per-frame coin flips. Speed is smoothed over ~9 frames, so
        # a state that flips every frame is erased by the smoothing before the
        # regression ever sees it -- which would test the smoother, not the
        # representation. Real freezing and locomotion come in bouts anyway.
        bouts = rng.random(20) < 0.5
        fast = np.repeat(bouts, 20)
        t = fast.size
        step = np.where(fast, 6.0, 0.01)[:, None]
        xy = np.cumsum(step * np.ones((t, 2)), axis=0)
        s = rng.normal(size=(t, 5))
        s[:, 0] += 4.0 * fast            # first PC carries the state
        scores.append(s)
        frame.append(np.column_stack([xy, rng.normal(scale=0.05, size=t)]))

    res = obs.degeneracy(scores, frame, fps=30.0, n_boot=40, seed=0)
    assert res["auc"] > 0.9, res
    assert res["ci95"][0] > 0.5
    assert res["n_recordings"] == 12


def test_degeneracy_returns_chance_when_pose_is_independent_of_speed():
    """The null the whole test exists to be able to report."""
    rng = np.random.default_rng(2)
    scores, frame = [], []
    for _ in range(12):
        t = 400
        xy = np.cumsum(rng.normal(scale=3.0, size=(t, 2)), axis=0)
        scores.append(rng.normal(size=(t, 5)))     # independent of xy
        frame.append(np.column_stack([xy, rng.normal(scale=0.05, size=t)]))

    res = obs.degeneracy(scores, frame, fps=30.0, n_boot=40, seed=0)
    assert 0.40 < res["auc"] < 0.60, res
    # Both bounds near chance. Not "the CI covers 0.5 exactly": with thousands
    # of frames the interval is a few thousandths wide, so it can sit just off
    # 0.5 while the estimate is chance for every practical purpose.
    assert 0.45 < res["ci95"][0] <= res["ci95"][1] < 0.55, res
    assert res["ci95"][1] - res["ci95"][0] < 0.10


def test_degeneracy_holds_out_by_recording():
    scores, frame = _toy(n_rec=6, t=300)
    res = obs.degeneracy(scores, frame, fps=30.0, n_boot=10, seed=0)
    assert res["grouping"] == "held out by recording"
    assert res["n_folds"] == 5
    assert res["median_speed_fast"] > res["median_speed_slow"]


def test_degeneracy_boosted_beats_logistic_on_a_curved_signature():
    """The case the linear number cannot distinguish from absence.

    Speed depends on the *radius* in PC space, which no linear boundary can
    separate. Logistic regression should sit near chance while the boosted model
    recovers it -- so a low logistic AUC alone never licenses "the information
    is gone".
    """
    rng = np.random.default_rng(7)
    scores, frame = [], []
    for _ in range(14):
        # State first, in bouts; geometry conditioned on it. Two concentric
        # rings share a centre, so no halfspace separates them, while an
        # axis-aligned tree can carve out the inner one.
        fast = np.repeat(rng.random(20) < 0.5, 20)
        t = fast.size
        radius = np.where(fast, 2.5, 0.6) + rng.normal(scale=0.1, size=t)
        angle = rng.uniform(0, 2 * np.pi, size=t)
        s = np.column_stack([radius * np.cos(angle), radius * np.sin(angle),
                             rng.normal(size=t), rng.normal(size=t)])
        xy = np.cumsum(np.where(fast, 6.0, 0.01)[:, None] * np.ones((t, 2)),
                       axis=0)
        scores.append(s)
        frame.append(np.column_stack([xy, rng.normal(scale=0.05, size=t)]))

    lin = obs.degeneracy(scores, frame, fps=30.0, n_boot=0, seed=0,
                         model="logistic")
    nl = obs.degeneracy(scores, frame, fps=30.0, n_boot=0, seed=0,
                        model="boosted")
    assert lin["auc"] < 0.65, lin
    assert nl["auc"] > lin["auc"] + 0.15, (lin, nl)
    assert nl["model"] == "boosted"


def test_degeneracy_rejects_an_unknown_model():
    scores, frame = _toy()
    with pytest.raises(ValueError, match="unknown model"):
        obs.degeneracy(scores, frame, fps=30.0, n_boot=0, model="svm")


def test_degeneracy_needs_two_recordings():
    scores, frame = _toy(n_rec=1)
    with pytest.raises(ValueError, match="at least 2 recordings"):
        obs.degeneracy(scores, frame, fps=30.0)
