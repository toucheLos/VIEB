import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from representation import run_registry  # noqa: E402


def test_missing_registry_reads_as_empty(tmp_path):
    assert run_registry.load(str(tmp_path / "nothing")) == []


def test_record_round_trips(tmp_path):
    out = str(tmp_path)
    run_registry.record(out, "diffusion", {"min_cluster_size": 50},
                        metrics={"n_states": 7})
    runs = run_registry.load(out)
    assert len(runs) == 1
    assert runs[0]["latent_method"] == "diffusion"
    assert runs[0]["params"]["min_cluster_size"] == 50
    assert runs[0]["metrics"]["n_states"] == 7
    assert runs[0]["run_id"] and runs[0]["timestamp"]


def test_records_accumulate_and_keep_order(tmp_path):
    out = str(tmp_path)
    for method in ("pca", "diffusion", "pca"):
        run_registry.record(out, method, {})
    assert [r["latent_method"] for r in run_registry.load(out)] \
        == ["pca", "diffusion", "pca"]


def test_run_ids_are_unique(tmp_path):
    out = str(tmp_path)
    for _ in range(20):
        run_registry.record(out, "pca", {})
    ids = [r["run_id"] for r in run_registry.load(out)]
    assert len(set(ids)) == 20


def test_cli_and_gui_runs_share_one_registry(tmp_path):
    # The Cluster Runs page must show runs regardless of where they started.
    out = str(tmp_path)
    run_registry.record(out, "pca", {}, source="cli")
    run_registry.record(out, "diffusion", {}, source="gui")
    assert {r["source"] for r in run_registry.load(out)} == {"cli", "gui"}


def test_corrupt_registry_does_not_crash(tmp_path):
    out = str(tmp_path)
    with open(run_registry.registry_path(out), "w") as fh:
        fh.write("{ not json")
    assert run_registry.load(out) == []


def test_numpy_values_survive_serialisation(tmp_path):
    import numpy as np

    out = str(tmp_path)
    run_registry.record(out, "pca",
                        {"threshold": np.float64(0.95),
                         "components": np.int64(7),
                         "eigenvalues": np.array([1.0, 0.5])})
    with open(run_registry.registry_path(out)) as fh:
        json.load(fh)      # would raise if numpy types leaked through


def test_summarise_flattens_the_table_rows(tmp_path):
    out = str(tmp_path)
    run_registry.record(
        out, "diffusion", {},
        metrics={"n_states": 8, "noise_frac": 0.12,
                 "clustered_only": {"largest_state_frac": 0.31,
                                    "state_entropy": 0.87}})
    row = run_registry.summarise(run_registry.load(out))[0]
    assert row["latent_method"] == "diffusion"
    assert row["n_states"] == 8
    assert row["noise_frac"] == 0.12
    assert row["largest_state_frac"] == 0.31
    assert row["state_entropy"] == 0.87


def test_summarise_tolerates_missing_metrics(tmp_path):
    out = str(tmp_path)
    run_registry.record(out, "pca", {})
    row = run_registry.summarise(run_registry.load(out))[0]
    assert row["n_states"] is None
    assert row["latent_method"] == "pca"


def test_write_is_atomic(tmp_path):
    # A crash mid-write must not truncate the file and lose every earlier run.
    out = str(tmp_path)
    run_registry.record(out, "pca", {})
    run_registry.record(out, "diffusion", {})
    assert not os.path.exists(run_registry.registry_path(out) + ".tmp")
    assert len(run_registry.load(out)) == 2
