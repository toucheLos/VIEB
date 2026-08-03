import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from representation.align import (  # noqa: E402
    align_all, align_session, null_leakage, solve_rotation,
)


def _rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def _shape(K=7, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(K, 2))
    return x - x.mean(axis=0)


def test_solve_rotation_recovers_applied_angle():
    # Rotating the reference by +phi must be undone by -phi.
    ref = _shape()
    rng = np.random.default_rng(1)
    for phi in rng.uniform(-np.pi, np.pi, size=50):
        rotated = (ref @ _rot(phi).T)[None]
        theta = solve_rotation(rotated, ref, np.ones((1, ref.shape[0])))
        err = abs((theta[0] - (-phi) + np.pi) % (2 * np.pi) - np.pi)
        assert err < 1e-9


def test_solve_rotation_matches_kabsch():
    ref = _shape(seed=2)
    rng = np.random.default_rng(3)
    pose = ref[None] + 0.2 * rng.normal(size=(20, *ref.shape))
    pose = pose - pose.mean(axis=1, keepdims=True)
    theta = solve_rotation(pose, ref, np.ones(pose.shape[:2]))
    for t in range(pose.shape[0]):
        m = pose[t].T @ ref
        u, _, vt = np.linalg.svd(m)
        v = vt.T
        r = v @ np.diag([1, np.sign(np.linalg.det(v @ u.T))]) @ u.T
        err = abs((theta[t] - np.arctan2(r[1, 0], r[0, 0]) + np.pi)
                  % (2 * np.pi) - np.pi)
        assert err < 1e-9


def test_alignment_is_invariant_to_translation_and_rotation():
    # The whole point: same posture, different place and heading, same output.
    rng = np.random.default_rng(4)
    base = _shape(seed=5)
    pose = base[None] + 0.15 * rng.normal(size=(40, *base.shape))

    moved = np.stack([
        pose[t] @ _rot(rng.uniform(-np.pi, np.pi)).T + rng.normal(size=2) * 25
        for t in range(pose.shape[0])
    ])

    ref = _shape(seed=5)
    a1, _, _ = align_session(pose, reference=ref)
    a2, _, _ = align_session(moved, reference=ref)
    assert np.allclose(a1, a2, atol=1e-9)


def test_weighted_reduces_to_unweighted():
    rng = np.random.default_rng(6)
    pose = rng.normal(size=(30, 7, 2)) * 3
    ref = _shape(seed=7)
    a_none, _, _ = align_session(pose, conf=None, reference=ref)
    a_ones, _, _ = align_session(pose, conf=np.ones((30, 7)), reference=ref)
    assert np.allclose(a_none, a_ones, atol=1e-12)


def test_low_confidence_keypoint_is_downweighted():
    # A single wildly-misplaced keypoint should not swing the alignment when
    # its likelihood says it is untrustworthy.
    ref = _shape(seed=8)
    pose = np.repeat(ref[None], 5, axis=0)
    corrupted = pose.copy()
    corrupted[:, 0] += np.array([50.0, -40.0])

    conf = np.ones((5, ref.shape[0]))
    conf[:, 0] = 1e-4  # DLC says this point is garbage

    good, _, _ = align_session(pose, reference=ref)
    weighted, _, _ = align_session(corrupted, conf=conf, reference=ref)
    unweighted, _, _ = align_session(corrupted, reference=ref)

    # Compare only the trustworthy keypoints.
    err_w = np.abs(weighted[:, 1:] - good[:, 1:]).max()
    err_u = np.abs(unweighted[:, 1:] - good[:, 1:]).max()
    assert err_w < err_u


def test_uniform_weights_give_exactly_three_null_directions():
    # Rank 2K-3: two from centering, one from the alignment stationarity
    # condition being a fixed linear functional of the aligned pose.
    rng = np.random.default_rng(9)
    K = 7
    base = _shape(K=K, seed=10)
    pose = np.stack([
        (base + 0.3 * rng.normal(size=base.shape)) @ _rot(
            rng.uniform(-np.pi, np.pi)).T + rng.normal(size=2) * 8
        for _ in range(4000)
    ])
    aligned, _, _ = align_session(pose)
    flat = aligned.reshape(aligned.shape[0], -1)
    ev = np.sort(np.linalg.eigvalsh(np.cov(flat.T)))[::-1]
    n_zero = int((np.abs(ev) <= ev[0] * 1e-10).sum())
    assert n_zero == 3, ev
    assert flat.shape[1] - n_zero == 2 * K - 3


def test_unweighted_recentering_restores_translation_nulls():
    # Per-frame weights break the exact rank structure; the post-hoc unweighted
    # re-centering in align_session must restore at least the two translation
    # nulls, so PCA cannot spend components on confidence fluctuation.
    rng = np.random.default_rng(11)
    K = 7
    base = _shape(K=K, seed=12)
    poses, confs = [], []
    for _ in range(4000):
        poses.append((base + 0.3 * rng.normal(size=base.shape))
                     @ _rot(rng.uniform(-np.pi, np.pi)).T
                     + rng.normal(size=2) * 8)
        confs.append(rng.uniform(0.5, 1.0, size=K))
    aligned, _, _ = align_session(np.stack(poses), np.stack(confs))
    flat = aligned.reshape(aligned.shape[0], -1)
    ev = np.sort(np.linalg.eigvalsh(np.cov(flat.T)))[::-1]
    assert int((np.abs(ev) <= ev[0] * 1e-10).sum()) >= 2, ev


def test_null_leakage_grows_with_confidence_variability():
    # Steadier likelihoods should leak less variance into the null directions.
    K = 7
    base = _shape(K=K, seed=13)

    def leak(lo):
        rng = np.random.default_rng(14)
        poses, confs = [], []
        for _ in range(3000):
            poses.append((base + 0.3 * rng.normal(size=base.shape))
                         @ _rot(rng.uniform(-np.pi, np.pi)).T)
            confs.append(rng.uniform(lo, 1.0, size=K))
        aligned, _, _ = align_session(np.stack(poses), np.stack(confs))
        return null_leakage(aligned.reshape(aligned.shape[0], -1))

    assert leak(0.98) < leak(0.5)


def test_align_all_shares_one_reference():
    rng = np.random.default_rng(15)
    sessions = [(rng.normal(size=(50, 7, 2)) * 2, None) for _ in range(3)]
    aligned, reference = align_all(sessions)
    assert len(aligned) == 3
    assert reference.shape == (7, 2)
    # Re-aligning any session to that reference must reproduce it exactly.
    again, _, _ = align_session(sessions[1][0], None, reference=reference)
    assert np.allclose(again, aligned[1], atol=1e-12)
