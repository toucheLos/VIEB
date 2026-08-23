"""Tests for HDBSCAN subsampling in compare.py --cluster."""

from __future__ import annotations

import json
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class FakeUMAP:
    """Small stand-in that behaves like an identity projection."""

    def __init__(self, n_components=2, **kwargs):
        self.n_components = n_components
        self.fit_rows = 0

    def fit(self, X):
        self.fit_rows = len(X)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        return X[:, :self.n_components]


class FakeHDBSCAN:
    """CPU HDBSCAN stub with deterministic labels and probabilities."""

    fit_sizes: list[int] = []

    def __init__(
        self,
        min_cluster_size,
        min_samples,
        cluster_selection_method,
        prediction_data=False,
    ):
        self.prediction_data = prediction_data
        self.labels_ = np.array([], dtype=np.int32)
        self.probabilities_ = np.array([], dtype=np.float32)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        FakeHDBSCAN.fit_sizes.append(len(X))
        self.labels_ = (X[:, 0] >= 0).astype(np.int32)
        self.probabilities_ = np.full(len(X), 0.95, dtype=np.float32)
        return self


def fake_approximate_predict(model, X):
    X = np.asarray(X, dtype=np.float32)
    labels = (X[:, 0] >= 0).astype(np.int32)
    probs = np.full(len(X), 0.8, dtype=np.float32)
    return labels, probs


def test_cmd_cluster_hdbscan_sampling_reconstructs_full_outputs(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    results_dir = project_dir / "results"
    features_dir = results_dir / "features"
    shared_dir = results_dir / "shared"
    features_dir.mkdir(parents=True)
    project_dir.mkdir(exist_ok=True)

    feature_data = {
        "vidA": np.array([[-3.0, 0.1, 1.0], [-2.0, 0.2, 1.0], [2.0, 0.3, 1.0], [3.0, 0.4, 1.0]], dtype=np.float32),
        "vidB": np.array([[-1.0, 0.5, 1.0], [1.0, 0.6, 1.0], [2.0, 0.7, 1.0]], dtype=np.float32),
        "vidC": np.array([[-4.0, 0.8, 1.0], [-3.0, 0.9, 1.0], [3.0, 1.0, 1.0], [4.0, 1.1, 1.0], [5.0, 1.2, 1.0]], dtype=np.float32),
    }

    index = {"_meta": {"n_features": 3, "n_keypoints": 2, "use_wavelets": False}}
    for stem, feats in feature_data.items():
        feat_path = features_dir / f"{stem}_features.npy"
        np.save(feat_path, feats)
        index[stem] = {
            "video_path": None,
            "csv_path": None,
            "n_frames": int(len(feats)),
            "n_keypoints": 2,
            "n_features": int(feats.shape[1]),
            "features_path": str(feat_path),
        }

    with open(features_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    config_path = project_dir / "config.json"
    config_path.write_text(json.dumps({"results_dir": str(results_dir)}))
    app_config_path = tmp_path / "app_config.json"
    app_config_path.write_text(json.dumps({"active_project": str(project_dir)}))

    import vieb_config as vc
    monkeypatch.setattr(vc, "_APP_CONFIG_PATH", str(app_config_path))
    monkeypatch.setattr(vc, "_CONFIG_PATH", str(config_path))

    import compare

    monkeypatch.setattr(compare, "_detect_gpu", lambda: False)
    monkeypatch.setitem(sys.modules, "umap", types.SimpleNamespace(UMAP=FakeUMAP))
    monkeypatch.setitem(
        sys.modules,
        "hdbscan",
        types.SimpleNamespace(
            HDBSCAN=FakeHDBSCAN,
            approximate_predict=fake_approximate_predict,
        ),
    )

    FakeHDBSCAN.fit_sizes.clear()

    compare.cmd_cluster(
        fps=30.0,
        min_cluster_size=2,
        umap_dims=2,
        validate=False,
        hdbscan_sample=4,
    )

    assert FakeHDBSCAN.fit_sizes == [4]

    with open(features_dir / "index.json") as f:
        index_after = json.load(f)
    assert index_after["_meta"]["hdbscan_sample"] == 4

    with open(shared_dir / "cluster_info.json") as f:
        cluster_info = json.load(f)
    assert cluster_info["hdbscan_sample"] == 4
    assert cluster_info["n_clusters"] == 2

    with open(shared_dir / "run_manifest.json") as f:
        run_manifest = json.load(f)
    assert run_manifest["hdbscan_sample"] == 4

    for stem, feats in feature_data.items():
        labels = np.load(shared_dir / f"{stem}_labels.npy")
        probs = np.load(shared_dir / f"{stem}_probs.npy")
        assert len(labels) == len(feats)
        assert len(probs) == len(feats)


# ---------------------------------------------------------------------------
# Unit tests: _assign_by_nearest_centroid
# ---------------------------------------------------------------------------

def test_assign_by_nearest_centroid_correct_cluster():
    """Each predict point is assigned to the nearest cluster centroid."""
    from compare import _assign_by_nearest_centroid

    rng = np.random.default_rng(0)
    # Two clearly separated clusters in 2D
    fit_pts = np.vstack([
        rng.normal([-5, 0], 0.1, (50, 2)),   # cluster 0
        rng.normal([ 5, 0], 0.1, (50, 2)),   # cluster 1
    ])
    fit_labels = np.array([0] * 50 + [1] * 50, dtype=np.int32)

    # Predict points clearly near cluster 0 or cluster 1
    predict_pts = np.array([[-5.1, 0.0], [5.1, 0.0], [-4.9, 0.0]])
    labels, probs = _assign_by_nearest_centroid(fit_pts, fit_labels, predict_pts,
                                                noise_distance_factor=0.0)
    assert labels[0] == 0
    assert labels[1] == 1
    assert labels[2] == 0
    assert all(p > 0 for p in probs)


def test_assign_by_nearest_centroid_noise_threshold():
    """Points far from all centroids are marked as noise (-1)."""
    from compare import _assign_by_nearest_centroid

    rng = np.random.default_rng(1)
    fit_pts = rng.normal([0, 0], 0.05, (100, 2))   # tight cluster 0
    fit_labels = np.zeros(100, dtype=np.int32)

    # Outlier 100 units away
    predict_pts = np.array([[100.0, 100.0]])
    labels, probs = _assign_by_nearest_centroid(fit_pts, fit_labels, predict_pts,
                                                noise_distance_factor=3.0)
    assert labels[0] == -1
    assert probs[0] == 0.0


def test_assign_by_nearest_centroid_all_noise_fit():
    """When all fit labels are -1, predict points are all assigned noise."""
    from compare import _assign_by_nearest_centroid

    fit_pts = np.zeros((10, 3), dtype=np.float32)
    fit_labels = np.full(10, -1, dtype=np.int32)
    predict_pts = np.ones((5, 3), dtype=np.float32)
    labels, probs = _assign_by_nearest_centroid(fit_pts, fit_labels, predict_pts)
    assert all(l == -1 for l in labels)
    assert all(p == 0.0 for p in probs)


def test_assign_by_nearest_centroid_batching_matches_unbatched():
    """Results must be identical regardless of batch size."""
    from compare import _assign_by_nearest_centroid

    rng = np.random.default_rng(2)
    fit_pts = np.vstack([rng.normal([-3, 0], 0.5, (60, 2)),
                         rng.normal([ 3, 0], 0.5, (60, 2))])
    fit_labels = np.array([0]*60 + [1]*60, dtype=np.int32)
    predict_pts = rng.uniform(-4, 4, (200, 2))

    labels_large, probs_large = _assign_by_nearest_centroid(
        fit_pts, fit_labels, predict_pts, noise_distance_factor=0.0, batch_size=1_000_000
    )
    labels_small, probs_small = _assign_by_nearest_centroid(
        fit_pts, fit_labels, predict_pts, noise_distance_factor=0.0, batch_size=17
    )
    np.testing.assert_array_equal(labels_large, labels_small)
    np.testing.assert_allclose(probs_large, probs_small, rtol=1e-5)


def test_hdbscan_safety_guard_triggers(tmp_path, monkeypatch):
    """If fit_indices would exceed hdbscan_sample, cmd_cluster raises RuntimeError."""
    import compare

    # Patch _detect_gpu so the CPU path is taken with hdbscan_sample=2 but 12 total frames
    project_dir = tmp_path / "project"
    results_dir = project_dir / "results"
    features_dir = results_dir / "features"
    features_dir.mkdir(parents=True)

    feats = np.zeros((12, 3), dtype=np.float32)
    feat_path = features_dir / "vid_features.npy"
    np.save(feat_path, feats)
    index = {
        "_meta": {"n_features": 3, "n_keypoints": 2, "use_wavelets": False},
        "vid": {
            "video_path": None, "csv_path": None,
            "n_frames": 12, "n_keypoints": 2,
            "n_features": 3, "features_path": str(feat_path),
        },
    }
    (features_dir / "index.json").write_text(json.dumps(index))
    config_path = project_dir / "config.json"
    config_path.write_text(json.dumps({"results_dir": str(results_dir)}))
    app_config_path = tmp_path / "app_config.json"
    app_config_path.write_text(json.dumps({"active_project": str(project_dir)}))

    import vieb_config as vc
    monkeypatch.setattr(vc, "_APP_CONFIG_PATH", str(app_config_path))
    monkeypatch.setattr(vc, "_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(compare, "_detect_gpu", lambda: False)

    # Patch sampling logic to skip the rng.choice so fit_indices = all 12 frames,
    # but set hdbscan_sample=2 so the guard trips.
    original_arange = np.arange

    def patched_arange(*args, **kwargs):
        arr = original_arange(*args, **kwargs)
        return arr

    import types as _types
    hdbscan_mod = _types.SimpleNamespace(
        HDBSCAN=FakeHDBSCAN,
        approximate_predict=fake_approximate_predict,
    )
    monkeypatch.setitem(sys.modules, "hdbscan", hdbscan_mod)
    monkeypatch.setitem(sys.modules, "umap", _types.SimpleNamespace(UMAP=FakeUMAP))

    # Monkeypatch numpy.random.default_rng inside compare so choice returns wrong size
    class _RNG:
        def choice(self, n, size, replace):
            # Return ALL indices instead of a sample — simulates the bug
            return np.arange(n)
        def shuffle(self, x): np.random.shuffle(x)

    monkeypatch.setattr(compare.np.random, "default_rng", lambda seed=None: _RNG())

    import pytest
    with pytest.raises(RuntimeError, match="safety guard"):
        compare.cmd_cluster(fps=30.0, min_cluster_size=2, umap_dims=2,
                            validate=False, hdbscan_sample=2)

