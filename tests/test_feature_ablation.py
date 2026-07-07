"""Tests for feature_ablation.py (feature ablation & dimensionality study)
and the new metrics in ml/validation_stats.py (DBCV, ARI-stability, R
bootstrap CI)."""

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


def _real_feature_names(use_wavelets=True):
    from ml.feature_extraction import PoseFeatureExtractor
    ext = PoseFeatureExtractor(fps=30.0, use_wavelets=use_wavelets,
                               bodypart_names=BODYPARTS, keypoint_roles={})
    return ext.get_feature_names(8)


# ---------------------------------------------------------------------------
# Family classifier
# ---------------------------------------------------------------------------

def test_classify_families_partitions_all_columns_exactly_once():
    import feature_ablation as fa
    names = _real_feature_names(use_wavelets=True)
    fams = fa.classify_families(names)

    all_idx = [i for cols in fams.values() for i in cols]
    assert len(all_idx) == len(names), "every column must be classified"
    assert len(all_idx) == len(set(all_idx)), "no column may be in two families"

    assert len(fams["per_keypoint_speed"]) == 8
    assert len(fams["pairwise_distances"]) == 28
    assert len(fams["wavelets"]) == 40
    assert len(fams["temporal_window_stats"]) == 8
    assert len(fams["semantic"]) == 2
    for scalar in ("centroid_speed", "body_orientation", "elongation",
                   "angular_velocity", "movement_entropy"):
        assert len(fams[scalar]) == 1


def test_classify_families_handles_naming_collisions():
    """speed_kp vs speed_*_window, dist_pair vs dist_*_window,
    angular_velocity vs angular_vel_*_window must not cross-contaminate."""
    import feature_ablation as fa
    names = _real_feature_names(use_wavelets=True)
    fams = fa.classify_families(names)

    # The temporal-window columns must NOT land in speed/dist/angular families.
    for col in fams["per_keypoint_speed"]:
        assert names[col].startswith("speed_kp")
        assert not names[col].endswith("_window")
    for col in fams["pairwise_distances"]:
        assert names[col].startswith("dist_pair")
    for col in fams["temporal_window_stats"]:
        assert names[col].endswith("_window")
    # The scalar angular_velocity is exactly that name, not a window stat.
    assert names[fams["angular_velocity"][0]] == "angular_velocity"


def test_classify_families_no_wavelets_51():
    import feature_ablation as fa
    names = _real_feature_names(use_wavelets=False)
    fams = fa.classify_families(names)
    assert "wavelets" not in fams  # empty families are dropped
    assert sum(len(c) for c in fams.values()) == len(names) == 51


def test_classify_families_rejects_unknown_name():
    import feature_ablation as fa
    with pytest.raises(ValueError, match="Unclassifiable"):
        fa.classify_families(["speed_kp0", "mystery_feature"])


# ---------------------------------------------------------------------------
# New metrics in ml/validation_stats.py
# ---------------------------------------------------------------------------

def test_dbcv_high_for_separated_blobs_skips_for_one_cluster():
    from ml.validation_stats import compute_dbcv
    rng = np.random.default_rng(0)
    emb = np.vstack([rng.normal([-6, 0], 0.3, (80, 2)),
                     rng.normal([6, 0], 0.3, (80, 2))])
    labels = np.array([0] * 80 + [1] * 80)
    res = compute_dbcv(emb, labels)
    assert not res["skipped"]
    assert res["dbcv"] > 0.5

    one = compute_dbcv(emb, np.zeros(160, dtype=int))
    assert one["skipped"]
    assert one["dbcv"] is None


def test_ari_stability_perfect_for_deterministic_partition():
    from ml.validation_stats import compute_ari_stability
    rng = np.random.default_rng(0)
    pts = rng.normal(0, 1, (200, 2))

    def recluster(idx):
        return (pts[idx, 0] >= 0).astype(int)  # deterministic, resample-invariant

    res = compute_ari_stability(recluster, n_samples=200, n_runs=5)
    assert not res["skipped"]
    assert res["ari_stability"] == pytest.approx(1.0)
    assert res["n_pairs"] == 10


def test_ari_stability_skips_with_too_few_samples():
    from ml.validation_stats import compute_ari_stability
    res = compute_ari_stability(lambda idx: np.zeros(len(idx)), n_samples=2, n_runs=5)
    assert res["skipped"]


def test_repeatability_bootstrap_ci_brackets_point_estimate():
    from ml.validation_stats import compute_repeatability_R
    rng = np.random.default_rng(0)
    animals = np.repeat([f"a{i}" for i in range(6)], 4)
    days = np.tile([1, 2, 3, 4], 6)
    eff = {f"a{i}": 0.1 + 0.15 * i for i in range(6)}
    s0 = np.array([eff[a] for a in animals]) + rng.normal(0, 0.02, 24)
    df = pd.DataFrame({"animal_id": animals, "day": days, "state_0_frac": s0})

    res = compute_repeatability_R(df, ["state_0_frac"], n_boot=200)
    assert "R_ci_low" in res and "R_ci_high" in res
    assert res["R_ci_low"] <= res["mean_R"] <= res["R_ci_high"]
    assert res["n_boot"] == 200


def test_repeatability_default_return_shape_unchanged_without_nboot():
    """cmd_report calls this without n_boot — the return shape must be
    exactly the original (no CI keys) so that path is untouched."""
    from ml.validation_stats import compute_repeatability_R
    df = pd.DataFrame({
        "animal_id": ["a1", "a1", "a2", "a2"],
        "day": [1, 2, 1, 2],
        "state_0_frac": [0.1, 0.15, 0.5, 0.55],
    })
    res = compute_repeatability_R(df, ["state_0_frac"])
    assert set(res.keys()) == {"skipped", "reason", "per_state", "mean_R", "n_states_scored"}


# ---------------------------------------------------------------------------
# Harness internals: subset selection, occupancy, transitions
# ---------------------------------------------------------------------------

def test_leave_zero_families_out_reproduces_full_column_set():
    """The union of all family indices must equal the full column range —
    i.e. removing zero families leaves the baseline exactly."""
    import feature_ablation as fa
    names = _real_feature_names(use_wavelets=True)
    fams = fa.classify_families(names)
    all_cols = fa._all_indices(fams)
    assert all_cols == list(range(len(names)))


def test_standardization_audit_passes_for_normal_data_and_flags_bypass():
    import feature_ablation as fa
    rng = np.random.default_rng(0)
    # Well-behaved matrix — audit must pass.
    good = rng.normal(5.0, 3.0, (500, 6))
    scaled = fa._standardize(good, audit=True)
    assert scaled.shape == good.shape

    # A monkeypatched preprocessor that returns UN-standardized data must trip
    # the audit assertion (a feature bypassing standardization).
    class _NoOpPre:
        def __init__(self, **k): pass
        def fit_transform(self, x): return np.asarray(x, dtype=np.float64)
    import ml
    orig = ml.BehaviorPreprocessor
    ml.BehaviorPreprocessor = _NoOpPre
    try:
        with pytest.raises(AssertionError, match="Standardization audit"):
            fa._standardize(rng.normal(100.0, 50.0, (200, 4)), audit=True)
    finally:
        ml.BehaviorPreprocessor = orig


def test_occupancy_and_transition_helpers():
    import feature_ablation as fa
    # 2 videos, 2 animals; labels chosen so occupancy differs by animal.
    labels = np.array([0, 0, 0, 1] + [1, 1, 1, 0])
    boundaries = {"vidA": (0, 4), "vidB": (4, 8)}
    meta = pd.DataFrame({"stem": ["vidA", "vidB"], "animal_id": ["1", "2"], "day": [1, 1]})
    occ, state_cols = fa._occupancy_df(labels, boundaries, meta, n_states=2)
    assert state_cols == ["state_0_frac", "state_1_frac"]
    assert occ.loc[occ.stem == "vidA", "state_0_frac"].iloc[0] == pytest.approx(0.75)
    assert "animal_id" in occ.columns

    counts = fa._transition_counts(labels, boundaries, n_states=2)
    # vidA: 0->0,0->0,0->1 ; vidB: 1->1,1->1,1->0
    assert counts[0, 0] == 2
    assert counts[0, 1] == 1
    assert counts[1, 1] == 2
    assert counts[1, 0] == 1


# ---------------------------------------------------------------------------
# End-to-end harness on synthetic data with mocked UMAP/HDBSCAN
# ---------------------------------------------------------------------------

class _FakeUMAP:
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


class _FakeHDBSCAN:
    def __init__(self, min_cluster_size, min_samples, cluster_selection_method, prediction_data=False):
        self.labels_ = np.array([], dtype=np.int32)

    def fit(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.labels_ = (X[:, 0] >= np.median(X[:, 0])).astype(np.int32)
        return self


def _fake_approximate_predict(model, X):
    X = np.asarray(X, dtype=np.float32)
    return (X[:, 0] >= 0).astype(np.int32), np.full(len(X), 0.9, dtype=np.float32)


def _make_project(tmp_path, name, seed, n_videos=4):
    project_dir = tmp_path / name
    results_dir = project_dir / "results"
    features_dir = results_dir / "features"
    features_dir.mkdir(parents=True)

    rng = np.random.default_rng(seed)
    feature_names = _real_feature_names(use_wavelets=True)
    n_feats = len(feature_names)

    index = {"_meta": {"n_keypoints": 8, "n_features": n_feats,
                       "use_wavelets": True, "feature_names": feature_names}}
    for v in range(n_videos):
        stem = f"vid{v}"
        arr = rng.normal(0, 1, (60, n_feats)).astype(np.float32)
        # give first column a clean bimodal split so fake HDBSCAN finds 2 states
        arr[:30, 0] -= 5
        arr[30:, 0] += 5
        fp = features_dir / f"{stem}_features.npy"
        np.save(fp, arr)
        index[stem] = {"n_frames": 60, "n_keypoints": 8, "n_features": n_feats,
                       "features_path": str(fp)}
    (features_dir / "index.json").write_text(json.dumps(index))

    meta_path = project_dir / "metadata.csv"
    pd.DataFrame({
        "filename": [f"vid{v}.mp4" for v in range(n_videos)],
        "animal_id": [str(1 + (v % 2)) for v in range(n_videos)],
        "day": [1 + (v // 2) for v in range(n_videos)],
        "context": ["A" if v % 2 == 0 else "B" for v in range(n_videos)],
    }).to_csv(meta_path, index=False)

    config = {"pose_source": "csv", "results_dir": str(results_dir),
              "metadata_csv_path": str(meta_path)}
    (project_dir / "config.json").write_text(json.dumps(config))
    return project_dir, results_dir


@pytest.fixture
def mocked_cluster(monkeypatch):
    monkeypatch.setitem(sys.modules, "umap", types.SimpleNamespace(UMAP=_FakeUMAP))
    monkeypatch.setitem(sys.modules, "hdbscan", types.SimpleNamespace(
        HDBSCAN=_FakeHDBSCAN, approximate_predict=_fake_approximate_predict))


def _point_config(tmp_path, project_dir, results_dir, monkeypatch):
    app_config = tmp_path / f"app_{project_dir.name}.json"
    app_config.write_text(json.dumps({"active_project": str(project_dir)}))
    import vieb_config as vc
    monkeypatch.setattr(vc, "_APP_CONFIG_PATH", str(app_config))
    monkeypatch.setattr(vc, "_CONFIG_PATH", str(project_dir / "config.json"))


def test_harness_runs_end_to_end_without_reextraction(tmp_path, monkeypatch, mocked_cluster):
    project_dir, results_dir = _make_project(tmp_path, "proj_luna", seed=1)
    _point_config(tmp_path, project_dir, results_dir, monkeypatch)

    import feature_ablation as fa
    pooled, boundaries, names, stems = fa.load_project_features("default")
    families = fa.classify_families(names)
    metadata = fa.load_metadata(stems)
    config = fa.AblationConfig(umap_dims=2, min_cluster_size=2, fit_sample=10_000)

    rows = fa.study_leave_one_out(families, pooled, boundaries, metadata, config, n_boot=20)
    df = pd.DataFrame(rows)

    expected_cols = {"subset", "n_features", "dbcv", "repeatability_R", "R_ci_low",
                     "R_ci_high", "ari_stability", "modularity_Q", "noise_frac", "n_states"}
    assert expected_cols.issubset(df.columns)
    # baseline + one row per non-empty family
    assert (df["subset"] == "all_features").sum() == 1
    assert (df["n_states"] >= 1).all()
    # per_keypoint_speed removal must reduce feature count below baseline
    base_n = df.loc[df.subset == "all_features", "n_features"].iloc[0]
    minus_speed = df.loc[df.subset == "minus_per_keypoint_speed", "n_features"].iloc[0]
    assert minus_speed == base_n - len(families["per_keypoint_speed"])


def test_two_projects_stay_isolated(tmp_path, monkeypatch, mocked_cluster):
    """Running two projects must never pool their frames — each loads only
    its own features."""
    import feature_ablation as fa

    luna_dir, luna_res = _make_project(tmp_path, "luna", seed=1, n_videos=4)
    spence_dir, spence_res = _make_project(tmp_path, "spence", seed=2, n_videos=6)

    _point_config(tmp_path, luna_dir, luna_res, monkeypatch)
    pooled_l, bounds_l, _, _ = fa.load_project_features("default")

    _point_config(tmp_path, spence_dir, spence_res, monkeypatch)
    pooled_s, bounds_s, _, _ = fa.load_project_features("default")

    assert len(bounds_l) == 4
    assert len(bounds_s) == 6
    assert pooled_l.shape[0] == 4 * 60
    assert pooled_s.shape[0] == 6 * 60
    # different frame totals confirm no accidental pooling
    assert pooled_l.shape[0] != pooled_s.shape[0]
