import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from representation import keypoints  # noqa: E402
from representation.pipeline import run  # noqa: E402


def _rot(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s], [s, c]])


def _synthetic_sessions(n_sessions=3, n_frames=400, seed=0, with_conf=True):
    """Recordings with two behaviors, placed and oriented at random."""
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(8, 2))
    sessions = []
    for _ in range(n_sessions):
        frames = []
        for t in range(n_frames):
            # Alternate between two postures in 50-frame bouts.
            posture = base if (t // 50) % 2 == 0 else base * np.array([1.0, 0.5])
            pose = posture + 0.05 * rng.normal(size=(8, 2))
            pose = pose @ _rot(rng.uniform(-np.pi, np.pi)).T
            pose = pose + rng.normal(size=2) * 20      # random arena position
            frames.append(pose)
        pose = np.stack(frames)
        conf = rng.uniform(0.9, 1.0, size=(n_frames, 8)) if with_conf else None
        sessions.append((pose, conf))
    return sessions


def test_pipeline_runs_end_to_end():
    result = run(_synthetic_sessions(), n_lags=3, lag_stride=2,
                 min_cluster_size=30)
    assert result["labels"].shape[0] == result["embedded"].shape[0]
    assert result["metrics"]["n_states"] >= 1


def test_tail_tip_is_dropped_and_reported():
    report = run(_synthetic_sessions(), n_lags=2, min_cluster_size=30)["report"]
    assert report["n_keypoints"] == 7
    assert report["dropped_keypoints"] == ["tail_tip"]
    assert report["pose_dim"] == 14
    assert report["expected_rank"] == 11


def test_component_count_is_logged():
    # Required by the brief: the number of PCs selected must be recorded.
    report = run(_synthetic_sessions(), n_lags=2, min_cluster_size=30)["report"]
    pca = report["pca"]
    assert 1 <= pca["n_components"] <= 11        # cannot exceed rank 2K-3
    assert len(pca["eigenvalues"]) == 14
    assert 0.0 < pca["explained_variance"] <= 1.0


def test_null_leakage_is_reported():
    report = run(_synthetic_sessions(), n_lags=2, min_cluster_size=30)["report"]
    # Confidence weighting refills the null directions; with likelihoods in
    # [0.9, 1] the leak should be small but is reported either way.
    assert 0.0 <= report["null_direction_leakage"] < 0.05


def test_embedding_dimension_matches_lags_and_components():
    result = run(_synthetic_sessions(), n_lags=3, lag_stride=2,
                 min_cluster_size=30)
    q = result["report"]["pca"]["n_components"]
    assert result["report"]["embedding_dim"] == q * 4
    assert result["report"]["window_frames"] == 7


def test_index_maps_rows_back_to_recording_and_frame():
    result = run(_synthetic_sessions(n_sessions=3), n_lags=3, lag_stride=2,
                 min_cluster_size=30)
    index = result["index"]
    assert set(index[:, 0]) == {0, 1, 2}
    # No row may come from a frame inside the window at a recording's start.
    assert (index[:, 1] >= 6).all()


def test_noise_label_is_preserved_not_filled_in():
    # Force noise by demanding implausibly large clusters.
    result = run(_synthetic_sessions(), n_lags=2, min_cluster_size=400)
    labels = result["labels"]
    assert (labels == -1).any()
    assert result["metrics"]["noise_frac"] > 0


def test_pipeline_reports_no_umap():
    # v2 clusters the embedding directly; v1 reduces with UMAP first. The
    # asymmetry is recorded rather than hidden.
    assert run(_synthetic_sessions(), n_lags=2,
               min_cluster_size=30)["report"]["umap"] is False


def test_works_without_confidences():
    result = run(_synthetic_sessions(with_conf=False), n_lags=2,
                 min_cluster_size=30)
    assert result["labels"].size > 0


def test_recordings_too_short_for_the_window_are_an_error():
    short = [(np.zeros((4, 8, 2)), None)]
    with pytest.raises(ValueError, match="delay embedding"):
        run(short, n_lags=10, lag_stride=2)


def test_seed_stability_is_optional_and_reported():
    result = run(_synthetic_sessions(), n_lags=2, min_cluster_size=30,
                 stability_repeats=3)
    assert result["seed_stability"]["n_repeats"] == 3
    assert len(result["seed_stability"]["n_states"]) == 3


def test_custom_bodypart_order_is_respected():
    names = ["tail_tip", "nose", "center", "left_ear", "right_ear",
             "left_hip", "right_hip", "tail_base"]
    report = run(_synthetic_sessions(), bodyparts=names, n_lags=2,
                 min_cluster_size=30)["report"]
    assert report["n_keypoints"] == 7
    assert report["dropped_keypoints"] == ["tail_tip"]
    assert keypoints.DEFAULT_BODYPARTS != names   # order genuinely differed
