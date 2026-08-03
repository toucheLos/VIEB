import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from representation.embed import (  # noqa: E402
    difference_matrix, embed_all, embed_session, valid_frames,
)


def test_embedding_stacks_the_right_frames():
    scores = np.arange(10, dtype=float).reshape(10, 1)
    emb = embed_session(scores, n_lags=2, lag_stride=1)
    assert emb.shape == (8, 3)
    # Row 0 is frame 2 with its two predecessors: [2, 1, 0].
    assert np.allclose(emb[0], [2.0, 1.0, 0.0])
    assert np.allclose(emb[-1], [9.0, 8.0, 7.0])


def test_lag_stride_widens_the_window():
    scores = np.arange(20, dtype=float).reshape(20, 1)
    emb = embed_session(scores, n_lags=2, lag_stride=3)
    assert emb.shape == (14, 3)
    assert np.allclose(emb[0], [6.0, 3.0, 0.0])


def test_lags_never_cross_a_recording_boundary():
    # The failure this guards against: embedding a concatenated array splices
    # the end of one animal's session onto the start of another's.
    sessions = [
        np.zeros((10, 1)),
        np.ones((10, 1)),
        np.full((10, 1), 2.0),
    ]
    emb, index = embed_all(sessions, n_lags=3, lag_stride=1)

    # Every row must be internally constant -- any chimeric row would mix
    # values from two different recordings.
    assert np.allclose(emb.min(axis=1), emb.max(axis=1))

    # And every row's frame index must be >= the window span.
    assert (index[:, 1] >= 3).all()
    for r, session in enumerate(sessions):
        assert (index[index[:, 0] == r, 1] < len(session)).all()


def test_frame_count_matches_the_valid_index_set():
    sessions = [np.zeros((n, 2)) for n in (30, 45, 12)]
    n_lags, stride = 4, 2
    emb, index = embed_all(sessions, n_lags, stride)
    expected = sum(max(0, n - n_lags * stride) for n in (30, 45, 12))
    assert emb.shape[0] == expected == index.shape[0]


def test_recordings_shorter_than_the_window_are_dropped():
    sessions = [np.zeros((3, 2)), np.zeros((40, 2))]
    emb, index = embed_all(sessions, n_lags=5, lag_stride=2)
    # Only the second recording is long enough for an 11-frame window.
    assert set(index[:, 0]) == {1}
    assert emb.shape[0] == 40 - 10


def test_concatenated_array_is_rejected():
    # Passing one array instead of a list is precisely the boundary-crossing
    # mistake, so it must fail loudly rather than silently embed across it.
    with pytest.raises(TypeError):
        embed_all(np.zeros((100, 2)), n_lags=3)


def test_zero_lags_is_the_identity():
    scores = np.random.default_rng(0).normal(size=(20, 3))
    assert np.allclose(embed_session(scores, n_lags=0), scores)


def test_geometric_weights_downweight_older_lags():
    scores = np.ones((10, 1))
    emb = embed_session(scores, n_lags=2, lag_stride=1, weights=0.5)
    assert np.allclose(emb[0], [1.0, 0.5, 0.25])


def test_valid_frames_matches_embedding_rows():
    assert np.array_equal(valid_frames(10, 3, 1), np.arange(3, 10))
    assert valid_frames(3, 5, 1).size == 0


def test_delay_stack_and_derivative_stack_are_linearly_equivalent():
    # B is invertible, so delay embedding spans the same space as the
    # (position, velocity, acceleration, ...) stack -- a superset of v1's
    # hand-built kinematic features.
    for n_lags in range(1, 9):
        b = difference_matrix(n_lags)
        assert abs(abs(np.linalg.det(b)) - 1.0) < 1e-9
        # ...but not an isometry, so the two give different clusterings.
        assert not np.allclose(b @ b.T, np.eye(n_lags + 1))


def test_difference_matrix_computes_actual_differences():
    scores = np.array([[1.0], [3.0], [7.0]])          # frames 0,1,2
    emb = embed_session(scores, n_lags=2, lag_stride=1)  # [p2, p1, p0]
    derivs = difference_matrix(2) @ emb[0]
    assert np.isclose(derivs[0], 7.0)                  # p_t
    assert np.isclose(derivs[1], 7.0 - 3.0)            # first difference
    assert np.isclose(derivs[2], 7.0 - 2 * 3.0 + 1.0)  # second difference
