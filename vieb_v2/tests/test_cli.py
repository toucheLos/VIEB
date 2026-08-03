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


def test_every_subcommand_has_help():
    parser = cli.build_parser()
    for name in ("doctor", "align", "pca", "embed", "cluster", "run",
                 "tune", "sweep", "benchmark"):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([name, "--help"])
        assert exc.value.code == 0
