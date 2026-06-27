import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cluster_run_manager import (
    ClusterRunConfig,
    ClusterRunManifest,
    ClusterRunManager,
    ClusterRunQueue,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_shared(tmp_path, n_clusters=5):
    """Create a minimal results/shared/ with label files, cluster_info, and manifest."""
    results = tmp_path / "results"
    shared = results / "shared"
    shared.mkdir(parents=True)
    (results / "diagnostics").mkdir()

    labels = np.random.randint(0, n_clusters, size=1000).astype(np.int32)
    probs = np.random.uniform(0.3, 1.0, size=1000).astype(np.float32)
    np.save(shared / "vid1_labels.npy", labels)
    np.save(shared / "vid1_probs.npy", probs)

    ci = {
        "n_clusters": n_clusters,
        "cluster_centers": [[0.0] * 10] * n_clusters,
        "method": "umap+hdbscan",
        "min_cluster_size": 2000,
        "hdbscan_sample": 5000,
        "mean_confidence": 0.85,
        "low_confidence_frac": 0.05,
    }
    (shared / "cluster_info.json").write_text(json.dumps(ci), encoding="utf-8")
    (shared / "preprocessor.pkl").write_bytes(b"fake-pkl")
    (shared / "umap_reducer.pkl").write_bytes(b"fake-pkl")
    (shared / "clusterer.pkl").write_bytes(b"fake-pkl")

    manifest = {
        "run_id": "run_001_20260626_1400_mcs2000_ms100_umap10",
        "status": "completed",
        "date": "2026-06-26 14:00",
        "min_cluster_size": 2000,
        "min_samples_requested": 0,
        "min_samples_resolved": 100,
        "umap_dims": 10,
        "hdbscan_sample": 5000,
        "n_clusters": n_clusters,
        "mean_confidence": 0.85,
        "low_confidence_frac": 0.05,
        "noise_frac": 0.12,
        "saved": False,
    }
    (shared / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    diag = {
        "n_states": n_clusters,
        "n_frames": 1000,
        "noise_frac": 0.12,
        "largest_state_frac": 0.35,
        "health_status": "good",
        "warnings": [{"level": "info", "message": "test warning"}],
    }
    (results / "diagnostics" / "cluster_diagnostics.json").write_text(
        json.dumps(diag), encoding="utf-8"
    )

    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"current_run_id": "", "current_run_saved": False}), encoding="utf-8")

    return results, cfg_path


# ---------------------------------------------------------------------------
# ClusterRunConfig tests
# ---------------------------------------------------------------------------

def test_min_samples_auto_resolution():
    cfg = ClusterRunConfig(min_cluster_size=2000, min_samples=0)
    resolved = cfg.resolve_min_samples()
    assert resolved == 100  # max(10, min(100, 2000 // 10)) = 100


def test_min_samples_zero_resolves_nonzero():
    for mcs in [50, 100, 500, 1000, 2000, 5000]:
        cfg = ClusterRunConfig(min_cluster_size=mcs, min_samples=0)
        assert cfg.resolve_min_samples() > 0


def test_min_samples_explicit_passthrough():
    cfg = ClusterRunConfig(min_cluster_size=2000, min_samples=50)
    assert cfg.resolve_min_samples() == 50


def test_min_samples_auto_small_mcs():
    cfg = ClusterRunConfig(min_cluster_size=30, min_samples=0)
    assert cfg.resolve_min_samples() == 10  # max(10, min(100, 3)) = 10


def test_min_samples_auto_large_mcs():
    cfg = ClusterRunConfig(min_cluster_size=5000, min_samples=0)
    assert cfg.resolve_min_samples() == 100  # max(10, min(100, 500)) = 100


def test_config_to_dict():
    cfg = ClusterRunConfig(min_cluster_size=2000, min_samples=0, umap_dims=7)
    d = cfg.to_dict()
    assert d["min_cluster_size"] == 2000
    assert d["min_samples"] == 0
    assert d["resolved_min_samples"] == 100
    assert d["umap_dims"] == 7


# ---------------------------------------------------------------------------
# ClusterRunManifest tests
# ---------------------------------------------------------------------------

def test_manifest_round_trip():
    m = ClusterRunManifest(
        run_id="run_001",
        status="completed",
        n_clusters=5,
        noise_frac=0.12,
        health_status="good",
        warnings_count=1,
        warnings=["test warning"],
    )
    d = m.to_dict()
    m2 = ClusterRunManifest.from_dict(d)
    assert m2.run_id == "run_001"
    assert m2.status == "completed"
    assert m2.n_clusters == 5
    assert m2.warnings == ["test warning"]


def test_legacy_manifest_compat():
    old_manifest = {
        "run_id": "run_005_20260622_2320_mcs2_umap2",
        "date": "2026-06-22 23:20",
        "min_cluster_size": 2,
        "umap_dims": 2,
        "hdbscan_min_samples": 0,
        "hdbscan_sample": 5000,
        "n_clusters": 3,
        "mean_confidence": 0.7,
        "low_confidence_frac": 0.1,
        "noise_frac": 0.2,
        "saved": True,
    }
    m = ClusterRunManifest.from_legacy(old_manifest)
    assert m.run_id == "run_005_20260622_2320_mcs2_umap2"
    assert m.status == "completed"
    assert m.min_cluster_size == 2
    assert m.min_samples_requested == 0
    assert m.min_samples_resolved == 0
    assert m.saved is True


def test_failed_run_preserves_error():
    m = ClusterRunManifest(
        run_id="run_010",
        status="failed",
        error_message="CUDA out of memory",
        n_clusters=0,
    )
    d = m.to_dict()
    m2 = ClusterRunManifest.from_dict(d)
    assert m2.status == "failed"
    assert m2.error_message == "CUDA out of memory"


# ---------------------------------------------------------------------------
# ClusterRunManager tests
# ---------------------------------------------------------------------------

def test_create_run_id_format(tmp_path):
    results, cfg_path = _setup_shared(tmp_path)
    mgr = ClusterRunManager(results, config_path=cfg_path)
    cfg = ClusterRunConfig(min_cluster_size=2100, min_samples=0, umap_dims=7)
    run_id = mgr.create_run_id(cfg)
    assert run_id.startswith("run_")
    assert "_mcs2100_" in run_id
    assert "_ms100_" in run_id
    assert "_umap7" in run_id
    assert re.match(r"run_\d{3}_\d{8}_\d{4}_mcs\d+_ms\d+_umap\d+", run_id)


def test_save_and_list_runs(tmp_path):
    results, cfg_path = _setup_shared(tmp_path)
    mgr = ClusterRunManager(results, config_path=cfg_path)

    run_id = "run_001_20260626_1400_mcs2000_ms100_umap10"
    run_dir = mgr.save_run(run_id)

    assert run_dir.is_dir()
    assert (run_dir / "cluster_info.json").exists()
    assert (run_dir / "vid1_labels.npy").exists()
    assert (run_dir / "run_manifest.json").exists()
    assert (run_dir / "cluster_diagnostics.json").exists()

    runs = mgr.list_runs()
    assert len(runs) == 1
    assert runs[0].run_id == run_id


def test_set_active_run_syncs_to_shared(tmp_path):
    results, cfg_path = _setup_shared(tmp_path)
    mgr = ClusterRunManager(results, config_path=cfg_path)

    run_id = "run_001_20260626_1400_mcs2000_ms100_umap10"
    mgr.save_run(run_id)

    # Clear shared and verify it's empty of cluster files
    mgr._clear_cluster_outputs_from_shared()
    assert not (results / "shared" / "cluster_info.json").exists()

    # Activate the run
    mgr.set_active_run(run_id)

    assert (results / "shared" / "cluster_info.json").exists()
    assert (results / "shared" / "vid1_labels.npy").exists()
    assert (results / "shared" / "run_manifest.json").exists()

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["active_cluster_run"] == run_id
    assert cfg["current_run_id"] == run_id
    assert cfg["current_run_saved"] is True


def test_get_active_run(tmp_path):
    results, cfg_path = _setup_shared(tmp_path)
    mgr = ClusterRunManager(results, config_path=cfg_path)

    assert mgr.get_active_run() == ""

    run_id = "run_001_20260626_1400_mcs2000_ms100_umap10"
    mgr.save_run(run_id)
    mgr.set_active_run(run_id)

    assert mgr.get_active_run() == run_id


def test_delete_run(tmp_path):
    results, cfg_path = _setup_shared(tmp_path)
    mgr = ClusterRunManager(results, config_path=cfg_path)

    run_id = "run_001_20260626_1400_mcs2000_ms100_umap10"
    mgr.save_run(run_id)
    assert len(mgr.list_runs()) == 1

    mgr.delete_run(run_id)
    assert len(mgr.list_runs()) == 0
    assert not (results / "runs" / run_id).exists()


def test_active_run_path_resolution(tmp_path):
    results, cfg_path = _setup_shared(tmp_path)
    mgr = ClusterRunManager(results, config_path=cfg_path)
    run_id = "run_042_20260627_0900_mcs1500_ms100_umap7"
    assert mgr.get_run_dir(run_id) == results / "runs" / run_id


def test_completed_run_has_diagnostics(tmp_path):
    results, cfg_path = _setup_shared(tmp_path)
    mgr = ClusterRunManager(results, config_path=cfg_path)

    run_id = "run_001_20260626_1400_mcs2000_ms100_umap10"
    mgr.save_run(run_id)

    runs = mgr.list_runs()
    assert len(runs) == 1
    m = runs[0]
    assert m.largest_state_occupancy == 0.35
    assert m.health_status == "good"
    assert m.warnings_count == 1


def test_hdbscan_sample_stored(tmp_path):
    results, cfg_path = _setup_shared(tmp_path)
    mgr = ClusterRunManager(results, config_path=cfg_path)

    run_id = "run_001_20260626_1400_mcs2000_ms100_umap10"
    mgr.save_run(run_id)

    m = mgr.load_run_manifest(run_id)
    assert m is not None
    assert m.hdbscan_sample == 5000


def test_comparison_table(tmp_path):
    results, cfg_path = _setup_shared(tmp_path)
    mgr = ClusterRunManager(results, config_path=cfg_path)

    run_id = "run_001_20260626_1400_mcs2000_ms100_umap10"
    mgr.save_run(run_id)

    table = mgr.comparison_table()
    assert len(table) == 1
    row = table[0]
    assert "run_id" in row
    assert "n_clusters" in row
    assert "noise_frac" in row
    assert "health_status" in row
    assert row["run_id"] == run_id


def test_set_active_nonexistent_raises(tmp_path):
    results, cfg_path = _setup_shared(tmp_path)
    mgr = ClusterRunManager(results, config_path=cfg_path)
    try:
        mgr.set_active_run("nonexistent_run")
        assert False, "Expected FileNotFoundError"
    except FileNotFoundError:
        pass


def test_multiple_runs_listed_in_order(tmp_path):
    results, cfg_path = _setup_shared(tmp_path)
    mgr = ClusterRunManager(results, config_path=cfg_path)

    for i, mcs in enumerate([1000, 2000, 3000], start=1):
        run_id = f"run_{i:03d}_20260626_1400_mcs{mcs}_ms100_umap10"
        run_dir = results / "runs" / run_id
        run_dir.mkdir(parents=True)
        manifest = {
            "run_id": run_id,
            "status": "completed",
            "date": "2026-06-26 14:00",
            "min_cluster_size": mcs,
            "min_samples_requested": 0,
            "min_samples_resolved": 100,
            "umap_dims": 10,
            "hdbscan_sample": 5000,
            "n_clusters": 5,
            "noise_frac": 0.1,
        }
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    runs = mgr.list_runs()
    assert len(runs) == 3
    assert runs[0].min_cluster_size == 1000
    assert runs[1].min_cluster_size == 2000
    assert runs[2].min_cluster_size == 3000


# ---------------------------------------------------------------------------
# ClusterRunQueue tests
# ---------------------------------------------------------------------------

def test_queue_multiple_configs():
    configs = [
        ClusterRunConfig(min_cluster_size=1000, min_samples=0, umap_dims=7),
        ClusterRunConfig(min_cluster_size=1500, min_samples=0, umap_dims=7),
        ClusterRunConfig(min_cluster_size=2100, min_samples=0, umap_dims=7),
    ]
    q = ClusterRunQueue(configs)
    assert not q.is_done
    assert q.current_index == 0

    c1 = q.next_config()
    assert c1 is not None
    assert c1.min_cluster_size == 1000

    c2 = q.next_config()
    assert c2 is not None
    assert c2.min_cluster_size == 1500

    c3 = q.next_config()
    assert c3 is not None
    assert c3.min_cluster_size == 2100

    assert q.next_config() is None
    assert q.is_done


def test_queue_stop_after_current():
    configs = [
        ClusterRunConfig(min_cluster_size=1000),
        ClusterRunConfig(min_cluster_size=2000),
        ClusterRunConfig(min_cluster_size=3000),
    ]
    q = ClusterRunQueue(configs)

    c1 = q.next_config()
    assert c1 is not None

    q.stop_after_current()
    assert q.next_config() is None
    assert q.is_done


def test_queue_add_remove_duplicate():
    q = ClusterRunQueue()
    q.add(ClusterRunConfig(min_cluster_size=1000))
    q.add(ClusterRunConfig(min_cluster_size=2000))
    assert len(q.configs) == 2

    q.duplicate(0)
    assert len(q.configs) == 3
    assert q.configs[0].min_cluster_size == 1000
    assert q.configs[1].min_cluster_size == 1000

    q.remove(1)
    assert len(q.configs) == 2
    assert q.configs[1].min_cluster_size == 2000


def test_queue_cancel():
    q = ClusterRunQueue([ClusterRunConfig()])
    q.cancel()
    assert q.is_done
    assert q.next_config() is None
