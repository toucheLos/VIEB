"""Tests for feature metadata schema migration, display logic, and run manifest enrichment."""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# _migrate_index_meta
# ---------------------------------------------------------------------------


def test_migrate_feature_count_to_n_features():
    from compare import _migrate_index_meta

    old = {"feature_count": 91, "use_wavelets": True, "vieb_version": "1.0"}
    result = _migrate_index_meta(old)
    assert result["n_features"] == 91
    assert result["use_wavelets"] is True
    assert result["vieb_version"] == "1.0"


def test_migrate_preserves_n_features_when_present():
    from compare import _migrate_index_meta

    meta = {"n_features": 51, "feature_count": 99}
    result = _migrate_index_meta(meta)
    # n_features already present, must not be overwritten by feature_count
    assert result["n_features"] == 51


def test_migrate_leaves_use_wavelets_absent():
    """Missing use_wavelets must NOT be defaulted to True or False."""
    from compare import _migrate_index_meta

    meta = {"n_features": 51}
    result = _migrate_index_meta(meta)
    assert "use_wavelets" not in result


def test_migrate_empty_dict():
    from compare import _migrate_index_meta

    assert _migrate_index_meta({}) == {}


def test_migrate_non_dict_returns_empty():
    from compare import _migrate_index_meta

    assert _migrate_index_meta(None) == {}
    assert _migrate_index_meta("bad") == {}


# ---------------------------------------------------------------------------
# _generate_diagnostics — use_wavelets must not default to True
# ---------------------------------------------------------------------------


def _setup_diag(tmp_path, monkeypatch, meta: dict):
    results = tmp_path / "results"
    shared = results / "shared"
    shared.mkdir(parents=True)
    (results / "features").mkdir()
    (results / "diagnostics").mkdir()

    labels = np.random.randint(0, 5, size=5000).astype(np.int32)
    probs = np.ones(5000, dtype=np.float32)
    np.save(shared / "vid1_labels.npy", labels)
    np.save(shared / "vid1_probs.npy", probs)

    ci = {"n_clusters": 5, "cluster_centers": [[0.0] * 10] * 5,
          "method": "umap+hdbscan", "min_cluster_size": 50}
    (shared / "cluster_info.json").write_text(json.dumps(ci))
    (shared / "run_manifest.json").write_text(json.dumps(
        {"run_id": "r", "umap_dims": 10, "min_cluster_size": 50, "hdbscan_min_samples": 1}
    ))

    (results / "features" / "index.json").write_text(json.dumps({"_meta": meta}))

    import vieb_config
    monkeypatch.setattr(vieb_config, "get_results_dir", lambda: str(results))


def test_diagnostics_use_wavelets_unknown_when_absent(tmp_path, monkeypatch):
    _setup_diag(tmp_path, monkeypatch, meta={"n_features": 51})
    from compare import _generate_diagnostics

    diag = _generate_diagnostics()
    # Must be None/null — not defaulted to True
    assert diag["use_wavelets"] is None


def test_diagnostics_use_wavelets_true_when_set(tmp_path, monkeypatch):
    _setup_diag(tmp_path, monkeypatch, meta={"n_features": 91, "use_wavelets": True})
    from compare import _generate_diagnostics

    diag = _generate_diagnostics()
    assert diag["use_wavelets"] is True


def test_diagnostics_use_wavelets_false_when_set(tmp_path, monkeypatch):
    _setup_diag(tmp_path, monkeypatch, meta={"n_features": 51, "use_wavelets": False})
    from compare import _generate_diagnostics

    diag = _generate_diagnostics()
    assert diag["use_wavelets"] is False


def test_diagnostics_migrates_feature_count(tmp_path, monkeypatch):
    """Old index.json using feature_count (not n_features) must still produce correct n_features."""
    _setup_diag(tmp_path, monkeypatch, meta={"feature_count": 91, "use_wavelets": True})
    from compare import _generate_diagnostics

    diag = _generate_diagnostics()
    assert diag["n_features"] == 91


# ---------------------------------------------------------------------------
# _write_current_run_manifest — must embed n_features and use_wavelets
# ---------------------------------------------------------------------------


def _setup_manifest(tmp_path, monkeypatch, meta: dict):
    results = tmp_path / "results"
    shared = results / "shared"
    shared.mkdir(parents=True)
    (results / "features").mkdir()

    (results / "features" / "index.json").write_text(json.dumps({"_meta": meta}))

    import vieb_config
    monkeypatch.setattr(vieb_config, "get_results_dir", lambda: str(results))

    config = {"use_wavelets": True}
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps(config))

    import compare
    import cluster_run_manager as crm

    class _FakeMgr:
        def __init__(self, *a, **kw): pass
        def create_run_id(self, cfg): return "run_test"

    monkeypatch.setattr(compare, "_res", lambda: str(results))
    monkeypatch.setattr(crm, "ClusterRunManager", _FakeMgr)

    return compare, shared


def test_run_manifest_includes_n_features(tmp_path, monkeypatch):
    compare, shared = _setup_manifest(
        tmp_path, monkeypatch, meta={"n_features": 91, "use_wavelets": True}
    )
    compare._write_current_run_manifest(
        min_cluster_size=50, umap_dims=10, effective_min_samples=5,
        hdbscan_sample=100000, n_found=6, mean_conf=0.8, low_conf_frac=0.05,
        noise_frac=0.1,
    )
    manifest = json.loads((shared / "run_manifest.json").read_text())
    assert manifest["n_features"] == 91
    assert manifest["use_wavelets"] is True


def test_run_manifest_use_wavelets_null_when_absent(tmp_path, monkeypatch):
    compare, shared = _setup_manifest(
        tmp_path, monkeypatch, meta={"n_features": 51}
    )
    compare._write_current_run_manifest(
        min_cluster_size=50, umap_dims=10, effective_min_samples=5,
        hdbscan_sample=100000, n_found=4, mean_conf=0.75, low_conf_frac=0.1,
        noise_frac=0.05,
    )
    manifest = json.loads((shared / "run_manifest.json").read_text())
    assert manifest["n_features"] == 51
    # use_wavelets must be None (null in JSON) — not True/False
    assert manifest["use_wavelets"] is None


def test_run_manifest_no_index_does_not_crash(tmp_path, monkeypatch):
    """_write_current_run_manifest must succeed even without index.json."""
    results = tmp_path / "results"
    shared = results / "shared"
    shared.mkdir(parents=True)

    import vieb_config
    monkeypatch.setattr(vieb_config, "get_results_dir", lambda: str(results))

    import compare
    import cluster_run_manager as crm

    class _FakeMgr:
        def __init__(self, *a, **kw): pass
        def create_run_id(self, cfg): return "run_no_idx"

    monkeypatch.setattr(compare, "_res", lambda: str(results))
    monkeypatch.setattr(crm, "ClusterRunManager", _FakeMgr)

    compare._write_current_run_manifest(
        min_cluster_size=50, umap_dims=10, effective_min_samples=5,
        hdbscan_sample=100000, n_found=4, mean_conf=0.75, low_conf_frac=0.1,
        noise_frac=0.05,
    )
    manifest = json.loads((shared / "run_manifest.json").read_text())
    assert manifest["n_clusters"] == 4
    # n_features and use_wavelets must be absent (not default values)
    assert "n_features" not in manifest
