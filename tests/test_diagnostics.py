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
    pooled_umap = np.random.normal(size=(5000, 2)).astype(np.float32)

    diag = _generate_diagnostics(all_labels=labels, all_probs=probs, pooled_umap=pooled_umap)
    assert diag["n_frames"] == 5000
    assert diag["n_states"] == 5
    assert (results / "diagnostics" / "umap_sample.csv").exists()
    assert (results / "diagnostics" / "umap_embedding_by_state.png").exists()


def test_diagnostics_handles_no_umap(tmp_path, monkeypatch):
    results = _setup_diagnostics(tmp_path, monkeypatch)
    from compare import _generate_diagnostics
    diag = _generate_diagnostics(pooled_umap=None)

    assert (results / "diagnostics" / "cluster_diagnostics.json").exists()
    assert not (results / "diagnostics" / "umap_sample.csv").exists()


# ── New tests ──────────────────────────────────────────────────────────────────


def _make_bout_labels(n_clusters=4, fps=30):
    """Build labels with known bout structure: alternating 30-frame and 5-frame bouts."""
    rng = np.random.default_rng(0)
    segments = []
    for _ in range(200):
        state = int(rng.integers(0, n_clusters))
        length = 30 if len(segments) % 2 == 0 else 5
        segments.extend([state] * length)
    return np.array(segments, dtype=np.int32)


def test_health_status_good(tmp_path, monkeypatch):
    results = _setup_diagnostics(tmp_path, monkeypatch, n_clusters=5)
    from compare import _generate_diagnostics
    diag = _generate_diagnostics()
    assert diag["health_status"] in ("good", "suspicious", "failed")  # field exists
    # Balanced 5-state clustering should not be "failed"
    assert diag["health_status"] != "failed" or diag.get("noise_frac", 0) > 0.5


def test_health_status_suspicious(tmp_path, monkeypatch):
    # 2 states triggers "Only N states" warning → suspicious
    results = _setup_diagnostics(tmp_path, monkeypatch, n_clusters=2)
    from compare import _generate_diagnostics
    diag = _generate_diagnostics()
    assert diag["health_status"] in ("suspicious", "failed")


def test_health_status_failed(tmp_path, monkeypatch):
    # dominant_frac=0.95 triggers largest-state error
    results = _setup_diagnostics(tmp_path, monkeypatch, n_clusters=5, dominant_frac=0.95)
    from compare import _generate_diagnostics
    diag = _generate_diagnostics()
    assert diag["health_status"] == "failed"


def test_zero_noise_collapsed_warning(tmp_path, monkeypatch):
    results = _setup_diagnostics(tmp_path, monkeypatch, n_clusters=5, dominant_frac=0.92)
    # Force noise_frac = 0: overwrite labels file with no -1
    shared = results / "shared"
    n = 10000
    labels = np.zeros(n, dtype=np.int32)
    labels[int(n * 0.92):] = np.random.randint(1, 5, size=n - int(n * 0.92))
    np.save(shared / "vid1_labels.npy", labels)

    from compare import _generate_diagnostics
    diag = _generate_diagnostics()
    msgs = [w["message"] for w in diag.get("warnings", [])]
    assert any("dominant" in m.lower() or "noise" in m.lower() or "Largest" in m for m in msgs)


def test_high_noise_warning(tmp_path, monkeypatch):
    results = _setup_diagnostics(tmp_path, monkeypatch, n_clusters=5)
    shared = results / "shared"
    # 60% noise frames
    n = 10000
    labels = np.full(n, -1, dtype=np.int32)
    labels[int(n * 0.6):] = np.random.randint(0, 5, size=n - int(n * 0.6))
    np.save(shared / "vid1_labels.npy", labels)

    from compare import _generate_diagnostics
    diag = _generate_diagnostics()
    msgs = [w["message"] for w in diag.get("warnings", [])]
    assert any("50%" in m or "noise" in m.lower() for m in msgs)
    assert diag["health_status"] == "failed"


def test_state_duration_summary_written(tmp_path, monkeypatch):
    results = _setup_diagnostics(tmp_path, monkeypatch, n_clusters=4)
    shared = results / "shared"
    labels = _make_bout_labels(n_clusters=4)
    np.save(shared / "vid1_labels.npy", labels)
    probs = np.ones(len(labels), dtype=np.float32)
    np.save(shared / "vid1_probs.npy", probs)

    from compare import _generate_diagnostics
    _generate_diagnostics()

    csv_path = results / "diagnostics" / "state_duration_summary.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert set(["state", "n_bouts", "mean_dur_s", "median_dur_s", "short_bout_frac"]).issubset(df.columns)
    assert len(df) > 0
    assert df["n_bouts"].sum() > 0


def test_bout_metrics_in_json(tmp_path, monkeypatch):
    results = _setup_diagnostics(tmp_path, monkeypatch, n_clusters=4)
    shared = results / "shared"
    labels = _make_bout_labels(n_clusters=4)
    np.save(shared / "vid1_labels.npy", labels)
    np.save(shared / "vid1_probs.npy", np.ones(len(labels), dtype=np.float32))

    from compare import _generate_diagnostics
    diag = _generate_diagnostics()

    assert "bout_metrics" in diag
    bm = diag["bout_metrics"]
    assert len(bm) > 0
    first = next(iter(bm.values()))
    assert "n_bouts" in first
    assert "mean_dur_s" in first
    assert "short_bout_frac" in first


def test_short_bout_frac_in_json(tmp_path, monkeypatch):
    results = _setup_diagnostics(tmp_path, monkeypatch, n_clusters=4)
    shared = results / "shared"
    labels = _make_bout_labels(n_clusters=4)
    np.save(shared / "vid1_labels.npy", labels)
    np.save(shared / "vid1_probs.npy", np.ones(len(labels), dtype=np.float32))

    from compare import _generate_diagnostics
    diag = _generate_diagnostics()

    assert "short_bout_frac" in diag
    assert 0.0 <= diag["short_bout_frac"] <= 1.0


def test_imbalance_score_balanced(tmp_path, monkeypatch):
    results = _setup_diagnostics(tmp_path, monkeypatch, n_clusters=5)
    shared = results / "shared"
    # Perfectly balanced labels
    n = 10000
    labels = np.tile(np.arange(5, dtype=np.int32), n // 5)
    np.save(shared / "vid1_labels.npy", labels)
    np.save(shared / "vid1_probs.npy", np.ones(n, dtype=np.float32))

    from compare import _generate_diagnostics
    diag = _generate_diagnostics()

    assert diag["imbalance_score"] < 0.15  # near-zero for balanced


def test_imbalance_score_collapsed(tmp_path, monkeypatch):
    results = _setup_diagnostics(tmp_path, monkeypatch, n_clusters=5, dominant_frac=0.95)
    from compare import _generate_diagnostics
    diag = _generate_diagnostics()
    assert diag["imbalance_score"] > 0.5


def test_graceful_missing_label_files(tmp_path, monkeypatch):
    results = _setup_diagnostics(tmp_path, monkeypatch, n_clusters=5)
    # Remove the label file
    (results / "shared" / "vid1_labels.npy").unlink()

    from compare import _generate_diagnostics
    diag = _generate_diagnostics()
    assert diag == {}  # returns empty dict gracefully
