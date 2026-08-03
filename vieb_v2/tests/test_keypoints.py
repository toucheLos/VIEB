import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from representation import keypoints  # noqa: E402


def test_tail_tip_is_dropped_and_nose_is_kept():
    idx = keypoints.select_indices(keypoints.DEFAULT_BODYPARTS)
    kept = [keypoints.DEFAULT_BODYPARTS[i] for i in idx]
    assert "nose" in kept
    assert "tail_tip" not in kept
    assert "tail_base" in kept   # only the far end of the tail goes
    assert len(kept) == 7


def test_dropping_reduces_pose_and_conf_together():
    pose = np.arange(8 * 2 * 5, dtype=float).reshape(5, 8, 2)
    conf = np.ones((5, 8))
    pose2, conf2, kept = keypoints.select(pose, conf, keypoints.DEFAULT_BODYPARTS)

    assert pose2.shape == (5, 7, 2)
    assert conf2.shape == (5, 7)
    assert len(kept) == 7
    # Rank of the aligned covariance is 2K-3, so dropping one point takes it
    # from 13 to 11.
    assert 2 * pose2.shape[1] - 3 == 11


def test_conf_may_be_absent():
    pose = np.zeros((4, 8, 2))
    pose2, conf2, _ = keypoints.select(pose, None, keypoints.DEFAULT_BODYPARTS)
    assert pose2.shape == (4, 7, 2)
    assert conf2 is None


def test_name_matching_is_case_and_whitespace_insensitive():
    names = ["Nose", " TAIL_TIP ", "center"]
    kept = [names[i] for i in keypoints.select_indices(names)]
    assert kept == ["Nose", "center"]


def test_ordering_is_preserved():
    names = ["center", "tail_tip", "nose", "left_ear"]
    kept = [names[i] for i in keypoints.select_indices(names)]
    assert kept == ["center", "nose", "left_ear"]


def test_dropping_everything_is_an_error():
    with pytest.raises(ValueError):
        keypoints.select_indices(["tail_tip"])
