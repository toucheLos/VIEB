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

