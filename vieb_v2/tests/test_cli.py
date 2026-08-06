import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cli  # noqa: E402
from representation import gpu  # noqa: E402


def _write_pose(directory, n_videos=3, n_frames=300, seed=0):
    """Write synthetic DLC-format CSVs, matching the real bodypart set."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    bps = ["left_ear", "right_ear", "nose", "center",
           "left_hip", "right_hip", "tail_base", "tail_tip"]
    base = rng.normal(size=(8, 2)) * 10
    os.makedirs(directory, exist_ok=True)

    for v in range(n_videos):
        frames = []
        for t in range(n_frames):
            posture = base if (t // 40) % 2 == 0 else base * [1.0, 0.55]
            p = posture + rng.normal(size=(8, 2)) * 0.4
            th = rng.uniform(-np.pi, np.pi)
            r = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            frames.append(p @ r.T + rng.normal(size=2) * 40)
        pose = np.stack(frames)

        data = np.zeros((n_frames, len(bps) * 3))
        for k in range(len(bps)):
            data[:, k * 3] = pose[:, k, 0]
            data[:, k * 3 + 1] = pose[:, k, 1]
            data[:, k * 3 + 2] = rng.uniform(0.9, 1.0, n_frames)
        cols = pd.MultiIndex.from_product([["S"], bps, ["x", "y", "likelihood"]])
        pd.DataFrame(data, columns=cols).to_csv(
            os.path.join(directory, f"vid{v}DLC_test.csv"))
    return directory


@pytest.fixture
def project(tmp_path):
    return (str(_write_pose(str(tmp_path / "pose"))), str(tmp_path / "out"))


def test_missing_pose_exits_needs_attention_and_names_dlc(capsys, tmp_path):
    # The single most likely failure today, so it must be actionable rather
    # than a traceback.
    code = cli.main(["run", "--pose", str(tmp_path / "nope"),
                     "--out", str(tmp_path / "out")])
    assert code == cli.NEEDS_ATTENTION
    out = capsys.readouterr().out
    assert "setup_dlc_training.py --analyze" in out


def test_missing_checkpoint_exits_needs_attention(tmp_path):
    assert cli.main(["pca", "--out", str(tmp_path / "empty")]) \
        == cli.NEEDS_ATTENTION


def test_full_run_produces_all_checkpoints(project):
    pose, out = project
    assert cli.main(["run", "--pose", pose, "--out", out,
                     "--min-cluster-size", "30", "--gpu", "off"]) == cli.OK
    for name in ("aligned.npz", "scores.npz", "embedded.npz", "labels.npz"):
        assert os.path.exists(os.path.join(out, name)), name


def test_checkpoints_preserve_recording_boundaries(project):
    # A flattened blob would let the embed stage cross recordings, so the
    # per-recording lengths must survive the round trip.
    pose, out = project
    cli.main(["align", "--pose", pose, "--out", out, "--gpu", "off"])
    data = np.load(os.path.join(out, "aligned.npz"))
    assert list(data["lengths"]) == [300, 300, 300]
    assert data["stacked"].shape[0] == 900


def test_embedding_never_crosses_a_boundary_via_cli(project):
    pose, out = project
    cli.main(["align", "--pose", pose, "--out", out, "--gpu", "off"])
    cli.main(["pca", "--out", out])
    cli.main(["embed", "--out", out, "--n-lags", "3", "--lag-stride", "2"])

    index = np.load(os.path.join(out, "embedded.npz"))["index"]
    assert set(index[:, 0]) == {0, 1, 2}
    assert (index[:, 1] >= 6).all()          # window span is 6 frames
    assert index.shape[0] == 3 * (300 - 6)


def test_tail_tip_dropped_through_the_cli(project, capsys):
    pose, out = project
    cli.main(["align", "--pose", pose, "--out", out, "--gpu", "off"])
    out_text = capsys.readouterr().out
    assert "tail_tip" in out_text
    assert "K=7" in out_text and "rank ceiling 11" in out_text


def test_staged_and_single_run_agree(project, tmp_path):
    pose, out = project
    staged = str(tmp_path / "staged")
    cli.main(["run", "--pose", pose, "--out", out,
              "--min-cluster-size", "30", "--gpu", "off"])
    for argv in (["align", "--pose", pose, "--out", staged, "--gpu", "off"],
                 ["pca", "--out", staged],
                 ["embed", "--out", staged],
                 ["cluster", "--out", staged, "--min-cluster-size", "30",
                  "--gpu", "off"]):
        assert cli.main(argv) == cli.OK

    a = np.load(os.path.join(out, "labels.npz"))["labels"]
    b = np.load(os.path.join(staged, "labels.npz"))["labels"]
    assert np.array_equal(a, b)


def test_gpu_off_forces_cpu_backend(project):
    pose, out = project
    cli.main(["run", "--pose", pose, "--out", out,
              "--min-cluster-size", "30", "--gpu", "off"])
    import json
    meta = json.loads(str(np.load(os.path.join(out, "labels.npz"))["meta"]))
    assert meta["hdbscan_backend"] == "cpu"


def test_gpu_on_fails_loudly_when_unavailable(monkeypatch, project):
    # Silently falling back would burn a GPU allocation on CPU work.
    pose, out = project
    monkeypatch.setenv("VIEB_FORCE_CPU", "1")
    gpu.reset_cache()
    try:
        assert cli.main(["run", "--pose", pose, "--out", out,
                         "--gpu", "on"]) == cli.FAILED
    finally:
        gpu.reset_cache()


def test_gpu_auto_falls_back_quietly(monkeypatch, project):
    pose, out = project
    monkeypatch.setenv("VIEB_FORCE_CPU", "1")
    gpu.reset_cache()
    try:
        assert cli.main(["run", "--pose", pose, "--out", out,
                         "--min-cluster-size", "30", "--gpu", "auto"]) == cli.OK
    finally:
        gpu.reset_cache()


def test_doctor_runs_without_a_gpu(monkeypatch, capsys):
    monkeypatch.setenv("VIEB_FORCE_CPU", "1")
    gpu.reset_cache()
    try:
        assert cli.main(["doctor"]) == cli.OK
        assert "HDBSCAN on GPU   False" in capsys.readouterr().out
    finally:
        gpu.reset_cache()


def test_doctor_reports_the_recommended_stack(monkeypatch, capsys):
    monkeypatch.setattr(gpu, "detect_nvidia_driver", lambda: {
        "ok": True, "driver": "575.57.08", "driver_tuple": (575, 57, 8),
        "cuda": "12.9", "gpu_name": "NVIDIA A100-SXM4-80GB", "error": None})
    assert cli.main(["doctor"]) == cli.OK
    out = capsys.readouterr().out
    assert "575.57.08" in out
    assert "RAPIDS 26.04" in out


def test_print_packages_emits_only_pip_arguments(monkeypatch, capsys):
    # install_gpu.sh pipes this straight into pip, so a stray log line on
    # stdout would become a bogus package name.
    monkeypatch.setattr(gpu, "detect_nvidia_driver", lambda: {
        "ok": True, "driver": "575.57.08", "driver_tuple": (575, 57, 8),
        "cuda": "12.9", "gpu_name": None, "error": None})
    assert cli.main(["doctor", "--print-packages"]) == cli.OK

    captured = capsys.readouterr()
    lines = captured.out.strip().splitlines()
    stack = gpu.select_gpu_stack((575, 57, 8))
    assert lines == stack["packages"]
    assert not any(line.startswith("[vieb2]") for line in lines)


def test_print_packages_says_nothing_on_stdout_without_a_driver(monkeypatch,
                                                                capsys):
    # The login-node case: exit non-zero and keep stdout empty so the caller's
    # package array comes back empty rather than holding an error string.
    monkeypatch.setattr(gpu, "detect_nvidia_driver", lambda: {
        "ok": False, "driver": None, "driver_tuple": None, "cuda": None,
        "gpu_name": None, "error": "nvidia-smi not found"})
    assert cli.main(["doctor", "--print-packages"]) == cli.NEEDS_ATTENTION

    captured = capsys.readouterr()
    assert captured.out.strip() == ""
    assert "gpu partition" in captured.err


def test_print_packages_rejects_a_driver_too_old_for_any_stack(monkeypatch,
                                                               capsys):
    monkeypatch.setattr(gpu, "detect_nvidia_driver", lambda: {
        "ok": True, "driver": "470.1", "driver_tuple": (470, 1),
        "cuda": "11.4", "gpu_name": None, "error": None})
    assert cli.main(["doctor", "--print-packages"]) == cli.NEEDS_ATTENTION
    assert capsys.readouterr().out.strip() == ""


def test_degenerate_extension_warning_is_silent_when_zero(capsys):
    cli._report_degenerate_extensions({"n_degenerate_extensions": 0,
                                       "degenerate_extension_frac": 0.0})
    assert capsys.readouterr().out == ""


def test_degenerate_extension_warning_names_the_count_and_fraction(capsys):
    cli._report_degenerate_extensions({"n_degenerate_extensions": 3,
                                       "degenerate_extension_frac": 0.5})
    out = capsys.readouterr().out
    assert "3" in out
    assert "50.0000%" in out
    assert "WARNING" in out


def test_sample_then_predict_labels_every_frame(project):
    # A full project is ~2.3M frames, so HDBSCAN is fitted on a subsample and
    # the rest labelled by approximate_predict. Every frame must still get one.
    pose, out = project
    cli.main(["align", "--pose", pose, "--out", out, "--gpu", "off"])
    cli.main(["pca", "--out", out])
    cli.main(["embed", "--out", out])
    assert cli.main(["cluster", "--out", out, "--min-cluster-size", "30",
                     "--hdbscan-sample", "200", "--gpu", "off"]) == cli.OK

    data = np.load(os.path.join(out, "labels.npz"))
    assert data["labels"].shape[0] == data["index"].shape[0]
    assert np.isfinite(data["probabilities"]).all()


def test_json_output_is_written(project, tmp_path):
    pose, out = project
    metrics = str(tmp_path / "metrics.json")
    cli.main(["run", "--pose", pose, "--out", out, "--min-cluster-size", "30",
              "--gpu", "off", "--json", metrics])
    import json
    with open(metrics) as fh:
        payload = json.load(fh)
    assert "metrics" in payload and "backend" in payload


def test_limit_restricts_recordings(project, capsys):
    pose, out = project
    cli.main(["align", "--pose", pose, "--out", out, "--limit", "2",
              "--gpu", "off"])
    assert list(np.load(os.path.join(out, "aligned.npz"))["lengths"]) \
        == [300, 300]


def test_cli_can_resume_from_a_gui_written_run(project, tmp_path):
    """A GUI run and a CLI run must be indistinguishable on disk.

    The GUI runs the pipeline in memory; if it did not also write the stage
    checkpoints, the Pipeline page would report 0/4 after a successful run and
    the CLI could not re-cluster without redoing alignment.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    QApplication.instance() or QApplication([])

    from app.worker import PipelineWorker
    from representation import checkpoints

    pose, out = project
    worker = PipelineWorker(pose, out, {"latent_method": "pca",
                                        "min_cluster_size": 30})
    finished = []
    worker.finished_ok.connect(finished.append)
    worker.failed.connect(lambda msg: finished.append(AssertionError(msg)))
    worker.run()                      # synchronous; no thread needed
    assert finished and not isinstance(finished[0], Exception), finished

    # All four stage checkpoints present, exactly as `cli run` leaves them.
    assert sorted(checkpoints.completed_stages(out)) == \
        sorted(checkpoints.STAGE_FILES)

    # ...and the CLI picks up where the GUI left off.
    assert cli.main(["cluster", "--out", out, "--min-cluster-size", "40",
                     "--gpu", "off"]) in (cli.OK, cli.NEEDS_ATTENTION)


def test_every_subcommand_has_help():
    parser = cli.build_parser()
    for name in ("doctor", "align", "pca", "embed", "cluster", "run",
                 "tune", "sweep", "benchmark"):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([name, "--help"])
        assert exc.value.code == 0


# ---------------------------------------------------------------------------
# Koopman basins beside HDBSCAN states
# ---------------------------------------------------------------------------

def _two_arms(n_recordings=3, window=8, seed=0):
    """Koopman labels every frame; HDBSCAN labels all but the delay window."""
    rng = np.random.default_rng(seed)
    lengths = [200, 150, 250][:n_recordings]

    k_index = np.concatenate([
        np.stack([np.full(n, r), np.arange(n)], axis=1)
        for r, n in enumerate(lengths)])
    h_index = np.concatenate([
        np.stack([np.full(n - window, r), np.arange(window, n)], axis=1)
        for r, n in enumerate(lengths)])

    k_labels = rng.integers(0, 5, len(k_index)).astype(np.int32)
    lookup = {rf: k_labels[i] for i, rf in enumerate(map(tuple, k_index))}
    h_labels = np.array([lookup[rf] for rf in map(tuple, h_index)],
                        dtype=np.int32)
    return ({"labels": h_labels, "index": h_index, "n_frames": h_labels.size},
            {"labels": k_labels, "index": k_index, "n_frames": k_labels.size})


def test_koopman_and_hdbscan_are_joined_on_index_not_position():
    """The two label arrays differ in length by one delay window per recording,
    so a positional comparison silently offsets every recording after the
    first. Same labels in, ARI 1.0 out -- only if the join is on (rec, frame).
    """
    hdb, koop = _two_arms()
    joined = cli._join_on_index(hdb, koop)

    assert joined["n_joined"] == hdb["n_frames"]
    assert joined["n_dropped_by_delay_window"] == 3 * 8
    assert joined["adjusted_rand"] == pytest.approx(1.0)

    # The trap this guards against: identical partitions, compared row by row.
    positional = cli._adjusted_rand(hdb["labels"],
                                    koop["labels"][:hdb["n_frames"]])
    assert positional < 0.5, (
        "a positional compare agreed here, so this test would not catch the "
        "offset it exists to catch")


def test_join_survives_recordings_absent_from_one_arm():
    """A recording dropped entirely by one arm must not shift the others."""
    hdb, koop = _two_arms()
    keep = hdb["index"][:, 0] != 1
    hdb = {"labels": hdb["labels"][keep], "index": hdb["index"][keep],
           "n_frames": int(keep.sum())}

    joined = cli._join_on_index(hdb, koop)
    assert joined["n_joined"] == hdb["n_frames"]
    assert joined["adjusted_rand"] == pytest.approx(1.0)


def test_koopman_subcommand_writes_labels_beside_hdbscan(project, capsys):
    """End to end: both label sets survive, and both score with one metric set.

    Koopman's -1 is metrics.NOISE_LABEL, so nothing here is special-cased for
    the method that produced the labels.
    """
    from representation import checkpoints

    pose, out = project
    assert cli.main(["run", "--pose", pose, "--out", out,
                     "--min-cluster-size", "30", "--gpu", "off"]) == cli.OK
    cli.main(["koopman", "--out", out, "--n-regions", "8"])

    # Both must exist: they are built from different frame sets, so one
    # overwriting the other would make the comparison impossible.
    assert os.path.exists(os.path.join(out, cli.LABELS))
    assert os.path.exists(os.path.join(out, checkpoints.KOOPMAN_LABELS))

    koop = cli._score_labels(out, checkpoints.KOOPMAN_LABELS, cli.SCORES, None)
    hdb = cli._score_labels(out, cli.LABELS, cli.EMBEDDED, "embedded")
    assert koop["metrics"]["n_states"] == koop["report"]["n_attractors"]

    # Koopman keeps the delay window HDBSCAN drops, so it is strictly longer.
    assert koop["n_frames"] > hdb["n_frames"]

    joined = cli._join_on_index(hdb, koop)
    assert joined["n_joined"] == hdb["n_frames"]
    assert joined["n_dropped_by_delay_window"] > 0


def test_compare_koopman_declares_no_winner(project, capsys):
    pose, out = project
    assert cli.main(["run", "--pose", pose, "--out", out,
                     "--min-cluster-size", "30", "--gpu", "off"]) == cli.OK
    cli.main(["koopman", "--out", out, "--n-regions", "8"])
    capsys.readouterr()

    assert cli.main(["compare-koopman", "--pca-out", out]) == cli.OK
    printed = capsys.readouterr().out
    assert "No winner is declared." in printed
    assert "joined on (recording, frame), not position" in printed
    # The two -1s share a value, not a meaning -- the table must say so.
    assert "share a value, not a meaning" in printed
