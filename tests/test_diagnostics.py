import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _setup_diagnostics(tmp_path, monkeypatch, n_clusters=5, dominant_frac=None):
    """Create minimal results/shared/ with label files and cluster_info."""
    results = tmp_path / "results"
    shared = results / "shared"
    shared.mkdir(parents=True)
    (results / "features").mkdir()
    (results / "diagnostics").mkdir()

    n_frames = 10000
    if dominant_frac is not None:
        labels = np.zeros(n_frames, dtype=np.int32)
        dom_count = int(n_frames * dominant_frac)
        labels[dom_count:] = np.random.randint(1, n_clusters, size=n_frames - dom_count)
    else:
        labels = np.random.randint(0, n_clusters, size=n_frames).astype(np.int32)
    probs = np.random.uniform(0.3, 1.0, size=n_frames).astype(np.float32)

    np.save(shared / "vid1_labels.npy", labels)
    np.save(shared / "vid1_probs.npy", probs)

    ci = {
        "n_clusters": n_clusters,
        "cluster_centers": [[0.0] * 10] * n_clusters,
        "method": "umap+hdbscan",
        "min_cluster_size": 100,
        "hdbscan_sample": 5000,
        "mean_confidence": 0.8,
        "low_confidence_frac": 0.05,
    }
    with open(shared / "cluster_info.json", "w") as f:
        json.dump(ci, f)

    manifest = {
        "run_id": "test_run",
        "umap_dims": 10,
        "min_cluster_size": 100,
        "hdbscan_min_samples": 1,
        "hdbscan_sample": 5000,
    }
    with open(shared / "run_manifest.json", "w") as f:
        json.dump(manifest, f)

    index = {"_meta": {"n_features": 51, "use_wavelets": True, "semantic_features": ["rearing_score"]}}
    with open(results / "features" / "index.json", "w") as f:
        json.dump(index, f)

    import vieb_config
    monkeypatch.setattr(vieb_config, "get_results_dir", lambda: str(results))

    return results


def test_generate_diagnostics_creates_files(tmp_path, monkeypatch):
    results = _setup_diagnostics(tmp_path, monkeypatch)
    from compare import _generate_diagnostics
    diag = _generate_diagnostics()

    assert (results / "diagnostics" / "cluster_diagnostics.json").exists()
    assert (results / "diagnostics" / "state_occupancy.csv").exists()
    assert diag["n_states"] == 5
    assert diag["n_frames"] == 10000
    assert 0 <= diag["noise_frac"] <= 1


def test_warnings_few_states(tmp_path, monkeypatch):
    results = _setup_diagnostics(tmp_path, monkeypatch, n_clusters=2)
    from compare import _generate_diagnostics
    diag = _generate_diagnostics()

    warnings = diag.get("warnings", [])
    msgs = [w["message"] for w in warnings]
    assert any("Only 2 states" in m for m in msgs)


def test_warnings_dominant_state(tmp_path, monkeypatch):
    results = _setup_diagnostics(tmp_path, monkeypatch, n_clusters=5, dominant_frac=0.92)
    from compare import _generate_diagnostics
    diag = _generate_diagnostics()

    warnings = diag.get("warnings", [])
    msgs = [w["message"] for w in warnings]
    assert any("90" in m or "Largest state" in m for m in msgs)


def test_diagnostics_with_passed_arrays(tmp_path, monkeypatch):
    results = _setup_diagnostics(tmp_path, monkeypatch)
    from compare import _generate_diagnostics

    labels = np.random.randint(0, 5, size=5000).astype(np.int32)
    probs = np.random.uniform(0.5, 1.0, size=5000).astype(np.float32)

    diag = _generate_diagnostics(all_labels=labels, all_probs=probs)
    assert diag["n_frames"] == 5000
    assert diag["n_states"] == 5


def test_diagnostics_handles_no_umap(tmp_path, monkeypatch):
    results = _setup_diagnostics(tmp_path, monkeypatch)
    from compare import _generate_diagnostics
    diag = _generate_diagnostics(pooled_umap=None)

    assert (results / "diagnostics" / "cluster_diagnostics.json").exists()
    assert not (results / "diagnostics" / "umap_sample.csv").exists()
