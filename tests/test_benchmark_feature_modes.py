"""Integration test for benchmark_feature_modes.py (Part E): confirms the
comparison pipeline runs end-to-end across all feature modes on tiny
synthetic data without crashing, and produces the expected comparison
table shape. Does NOT assert any representation is "better" — per the
task, this only confirms the comparison pipeline itself works; picking a
winner is left to the researcher.

Runs entirely in-process (monkeypatches benchmark_feature_modes._run_stage
to call compare.py's cmd_extract/cmd_cluster/cmd_report functions directly,
with UMAP/HDBSCAN mocked via the sys.modules injection pattern from
tests/test_cluster_sampling.py) rather than spawning real subprocesses —
real subprocesses would each re-read this repo's actual config.json/
app_config.json (the live desktop app's project pointer), which a test
must never touch.
"""

from __future__ import annotations

import json
import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

BODYPARTS = ["nose", "left_ear", "right_ear", "tail_base", "center",
             "left_hip", "right_hip", "tail_tip"]
SCORER = "synthetic"


def _make_pose_df(n_frames: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = pd.MultiIndex.from_product(
        [[SCORER], BODYPARTS, ["x", "y", "likelihood"]],
        names=["scorer", "bodyparts", "coords"],
    )
    t = np.linspace(0, 10, n_frames)
    data = rng.random((n_frames, len(BODYPARTS) * 3)) * 0.1
    for i in range(len(BODYPARTS)):
        data[:, i * 3] += np.sin(t + i) * 5 + 20
        data[:, i * 3 + 1] += np.cos(t + i) * 5 + 20
    return pd.DataFrame(data, columns=cols)


class FakeUMAP:
    def __init__(self, n_components=2, **kwargs):
        self.n_components = n_components

    def fit(self, X):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] >= self.n_components:
            return X[:, :self.n_components]
        pad = np.zeros((X.shape[0], self.n_components - X.shape[1]), dtype=np.float32)
        return np.concatenate([X, pad], axis=1)


class FakeHDBSCAN:
    def __init__(self, min_cluster_size, min_samples, cluster_selection_method, prediction_data=False):
        self.labels_ = np.array([], dtype=np.int32)
        self.probabilities_ = np.array([], dtype=np.float32)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.labels_ = (X[:, 0] >= np.median(X[:, 0])).astype(np.int32)
        self.probabilities_ = np.full(len(X), 0.9, dtype=np.float32)
        return self


def _fake_approximate_predict(model, X):
    X = np.asarray(X, dtype=np.float32)
    labels = (X[:, 0] >= 0).astype(np.int32)
    probs = np.full(len(X), 0.8, dtype=np.float32)
    return labels, probs


@pytest.fixture
def synthetic_project(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    results_dir = project_dir / "results"
    raw_dir = project_dir / "raw_videos"
    raw_dir.mkdir(parents=True)
    results_dir.mkdir(parents=True)

    meta_path = project_dir / "metadata.csv"
    pd.DataFrame({
        "filename": ["vidA.mp4", "vidB.mp4", "vidC.mp4", "vidD.mp4"],
        "date": ["20250101"] * 4,
        "box": [1, 1, 1, 1],
        "experiment": ["CFC"] * 4,
        "day": [0, 1, 0, 1],
        "context": ["A", "A", "B", "B"],
        "no_shock": ["no"] * 4,
        "animal_id": ["1", "1", "2", "2"],
        "fear": ["", "", "", ""],
    }).to_csv(meta_path, index=False)

    config = {
        "pose_source": "csv",
        "results_dir": str(results_dir),
        "raw_videos_dir": str(raw_dir),
        "metadata_csv_path": str(meta_path),
        "use_wavelets": True,
    }
    config_path = project_dir / "config.json"
    config_path.write_text(json.dumps(config))
    app_config_path = tmp_path / "app_config.json"
    app_config_path.write_text(json.dumps({"active_project": str(project_dir)}))

    import vieb_config as vc
    monkeypatch.setattr(vc, "_APP_CONFIG_PATH", str(app_config_path))
    monkeypatch.setattr(vc, "_CONFIG_PATH", str(config_path))

    import compare
    monkeypatch.setattr(compare, "_load_extractor_config", lambda: ({}, [], BODYPARTS))
    monkeypatch.setattr(compare, "_detect_gpu", lambda: False)
    monkeypatch.setitem(sys.modules, "umap", types.SimpleNamespace(UMAP=FakeUMAP))
    monkeypatch.setitem(
        sys.modules, "hdbscan",
        types.SimpleNamespace(HDBSCAN=FakeHDBSCAN, approximate_predict=_fake_approximate_predict),
    )

    for stem, seed in (("vidA", 1), ("vidB", 2), ("vidC", 3), ("vidD", 4)):
        (raw_dir / f"{stem}.mp4").touch()
        _make_pose_df(n_frames=80, seed=seed).to_csv(raw_dir / f"{stem}DLC_resnet50.csv")

    return project_dir, results_dir


def test_benchmark_runs_all_modes_end_to_end(synthetic_project, monkeypatch):
    import benchmark_feature_modes as bfm
    import compare

    def fake_run_stage(stage_flag, mode, extra_args):
        if stage_flag == "--extract":
            compare.cmd_extract(fps=30.0, use_wavelets=True, feature_mode=mode)
        elif stage_flag == "--cluster":
            compare.cmd_cluster(fps=30.0, min_cluster_size=2, umap_dims=2,
                                 validate=False, hdbscan_sample=500, feature_mode=mode)
        elif stage_flag == "--report":
            compare.cmd_report(fps=30.0, min_confidence=0.0, feature_mode=mode)
        else:
            raise ValueError(stage_flag)
        return 0.01

    monkeypatch.setattr(bfm, "_run_stage", fake_run_stage)
    monkeypatch.setattr(bfm, "_peak_rss_mb", lambda: 123.0)

    modes = "default,shape_space,delay_embedding,topological"
    try:
        import ripser  # noqa: F401
    except ImportError:
        modes = "default,shape_space,delay_embedding"

    old_argv = sys.argv
    sys.argv = ["benchmark_feature_modes.py", "--modes", modes]
    try:
        bfm.main()
    finally:
        sys.argv = old_argv

    _, results_dir = synthetic_project
    out_path = results_dir / "benchmark" / "feature_mode_comparison.csv"
    assert out_path.exists()

    df = pd.read_csv(out_path)
    expected_modes = set(modes.split(","))
    assert set(df["feature_mode"]) == expected_modes
    assert len(df) == len(expected_modes)

    expected_cols = {
        "feature_mode", "project_name", "n_states", "noise_frac", "mean_confidence",
        "repeatability_mean_R", "modularity_Q", "n_possible_split_states",
        "extract_runtime_sec", "cluster_runtime_sec", "report_runtime_sec",
        "total_runtime_sec", "peak_rss_mb",
    }
    assert expected_cols.issubset(set(df.columns))

    # Every mode should have produced a real cluster count — the test only
    # confirms the pipeline runs, not that any mode is "better".
    assert (df["n_states"] > 0).all()
