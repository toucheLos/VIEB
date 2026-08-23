import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless: avoid Qt backend crash in cmd_motifs heatmap

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_motif_dir_name():
    from generate_clips import _motif_dir_name
    assert _motif_dir_name("(5, 10)", "bigram") == "bigram_5_10"
    assert _motif_dir_name("(1, 2, 3)", "trigram") == "trigram_1_2_3"
    assert _motif_dir_name("(0, 0)", "bigram") == "bigram_0_0"


def _setup_motif_project(tmp_path, monkeypatch):
    """Create minimal results/ with motifs data for testing."""
    results = tmp_path / "results"
    (results / "comparison").mkdir(parents=True)
    (results / "motifs").mkdir()
    (results / "features").mkdir()
    (results / "shared").mkdir()
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "vid1.mp4").touch()
    (tmp_path / "raw" / "vid2.mp4").touch()

    # motifs.csv - top motifs
    motifs = pd.DataFrame({
        "motif": ["(0, 1)", "(1, 2)"],
        "type": ["bigram", "bigram"],
        "context_A_freq": [0.1, 0.05],
        "context_B_freq": [0.02, 0.01],
        "enrichment_ratio": [5.0, 5.0],
        "log2_enrichment": [2.32, 2.32],
        "abs_log2_enrichment": [2.32, 2.32],
        "flagged": [True, True],
    })
    motifs.to_csv(results / "comparison" / "motifs.csv", index=False)

    # motif_sequences.csv
    seqs = pd.DataFrame({
        "stem": ["vid1", "vid1", "vid2"],
        "type": ["bigram", "bigram", "bigram"],
        "motif": ["(0, 1)", "(0, 1)", "(1, 2)"],
        "position": [0, 2, 0],
        "context": ["A", "A", "B"],
        "animal_id": ["m1", "m1", "m2"],
        "day": ["1", "1", "2"],
        "experiment": ["exp1", "exp1", "exp1"],
    })
    seqs.to_csv(results / "motifs" / "motif_sequences.csv", index=False)

    # bouts.csv
    bouts = pd.DataFrame({
        "stem": ["vid1"] * 4 + ["vid2"] * 3,
        "state": [0, 1, 2, 0, 1, 2, 1],
        "start_frame": [0, 100, 200, 300, 0, 100, 200],
        "end_frame": [99, 199, 299, 399, 99, 199, 299],
        "start_sec": [0, 3.33, 6.67, 10, 0, 3.33, 6.67],
        "end_sec": [3.3, 6.63, 9.97, 13.3, 3.3, 6.63, 9.97],
        "duration_sec": [3.3, 3.3, 3.3, 3.3, 3.3, 3.3, 3.3],
        "context": ["A"] * 4 + ["B"] * 3,
        "animal_id": ["m1"] * 4 + ["m2"] * 3,
    })
    bouts.to_csv(results / "motifs" / "bouts.csv", index=False)

    # index.json
    index = {
        "vid1": {
            "video_path": str(tmp_path / "raw" / "vid1.mp4"),
            "features_path": str(results / "features" / "vid1_features.npy"),
            "n_frames": 400,
        },
        "vid2": {
            "video_path": str(tmp_path / "raw" / "vid2.mp4"),
            "features_path": str(results / "features" / "vid2_features.npy"),
            "n_frames": 300,
        },
        "_meta": {"n_features": 51},
    }
    with open(results / "features" / "index.json", "w") as f:
        json.dump(index, f)

    # Config
    cfg = {
        "fps": 30,
        "results_dir": str(results),
        "paths": {
            "raw_videos": str(tmp_path / "raw"),
            "pose_files": "",
            "pose_h5": None,
            "metadata": str(tmp_path / "metadata.csv"),
            "results": str(results),
        },
    }
    with open(tmp_path / "config.json", "w") as f:
        json.dump(cfg, f)
    with open(tmp_path / "app_config.json", "w") as f:
        json.dump({"active_project": str(tmp_path)}, f)

    import vieb_config
    monkeypatch.setattr(vieb_config, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(vieb_config, "_APP_CONFIG_PATH", str(tmp_path / "app_config.json"))
    monkeypatch.setattr(vieb_config, "_CONFIG_PATH", str(tmp_path / "config.json"))
    monkeypatch.setattr(vieb_config, "get_results_dir", lambda: str(results))

    # Monkeypatch _export_clip to always return True (no real video)
    import generate_clips
    monkeypatch.setattr(generate_clips, "_export_clip", lambda *a, **kw: True)

    return results


def test_cmd_motif_clips_writes_index(tmp_path, monkeypatch):
    results = _setup_motif_project(tmp_path, monkeypatch)
    from generate_clips import cmd_motif_clips
    cmd_motif_clips(fps=30, top_motifs=2, clips_per_motif=2)

    index_path = results / "motifs" / "motif_exemplars.csv"
    assert index_path.exists()
    df = pd.read_csv(index_path)
    required_cols = {"motif", "type", "clip_path", "stem", "animal_id",
                     "context", "start_frame", "end_frame", "duration_sec",
                     "source_video", "rank", "selection_reason", "skipped_reason"}
    assert required_cols.issubset(set(df.columns))
    assert len(df) > 0


def test_motif_clips_creates_dirs(tmp_path, monkeypatch):
    results = _setup_motif_project(tmp_path, monkeypatch)
    from generate_clips import cmd_motif_clips
    cmd_motif_clips(fps=30, top_motifs=1, clips_per_motif=1)

    clips_dir = results / "motifs" / "clips"
    assert clips_dir.exists()
    subdirs = list(clips_dir.iterdir())
    assert len(subdirs) >= 1
    assert subdirs[0].name.startswith("bigram_")


def test_motif_clips_skips_missing_video(tmp_path, monkeypatch):
    results = _setup_motif_project(tmp_path, monkeypatch)
    (tmp_path / "raw" / "vid1.mp4").unlink()
    from generate_clips import cmd_motif_clips
    cmd_motif_clips(fps=30, top_motifs=1, clips_per_motif=1)

    index_path = results / "motifs" / "motif_exemplars.csv"
    assert index_path.exists()
    df = pd.read_csv(index_path)
    skipped = df[df["skipped_reason"].fillna("") == "source video missing"]
    assert len(skipped) > 0


def test_motif_clips_explicit_missing_source_gives_clear_error(tmp_path, monkeypatch):
    results = _setup_motif_project(tmp_path, monkeypatch)
    from generate_clips import cmd_motif_clips
    try:
        cmd_motif_clips(fps=30, motif_source=str(results / "motifs" / "does_not_exist.csv"))
    except SystemExit as exc:
        assert "Missing motif occurrence source" in str(exc)
    else:
        raise AssertionError("expected SystemExit for an explicit missing motif source")


def test_motif_clips_fallback_builds_from_bouts(tmp_path, monkeypatch):
    """When motif_sequences.csv is absent, clips are derived from bouts on the fly."""
    results = _setup_motif_project(tmp_path, monkeypatch)
    (results / "motifs" / "motif_sequences.csv").unlink()
    from generate_clips import cmd_motif_clips
    cmd_motif_clips(fps=30, top_motifs=3, clips_per_motif=2)

    index_path = results / "motifs" / "motif_exemplars.csv"
    assert index_path.exists()
    df = pd.read_csv(index_path)
    # The bout sequences in the fixture yield multi-state motifs like (0,1),(1,2).
    written = df[df["clip_path"].fillna("") != ""]
    assert len(written) > 0


def test_motif_clips_missing_index_gives_clear_error(tmp_path, monkeypatch):
    results = _setup_motif_project(tmp_path, monkeypatch)
    (results / "features" / "index.json").unlink()
    from generate_clips import cmd_motif_clips
    try:
        cmd_motif_clips(fps=30)
    except SystemExit as exc:
        assert "index.json" in str(exc)
    else:
        raise AssertionError("expected SystemExit when index.json is missing")


def test_motif_dir_name_sanitizes_paths():
    from generate_clips import _motif_dir_name
    name = _motif_dir_name("../bad motif/(1, 2)", "tri/../gram")
    assert "/" not in name
    assert "\\" not in name
    assert ".." not in name
    assert name


# ── Degenerate motif filtering ────────────────────────────────────────────

def test_is_degenerate_motif():
    from compare import _is_degenerate_motif
    # All-identical → degenerate (bout duration, not sequence structure)
    assert _is_degenerate_motif((48, 48))
    assert _is_degenerate_motif((46, 46, 46))
    assert _is_degenerate_motif((3,))
    # Mixed → kept, including partially-repeated tuples
    assert not _is_degenerate_motif((12, 47))
    assert not _is_degenerate_motif((12, 47, 8))
    assert not _is_degenerate_motif((3, 3, 7))


def test_cmd_motifs_excludes_degenerate_motifs(tmp_path, monkeypatch):
    """cmd_motifs must drop (n, n)/(n, n, n) but keep genuine multi-state motifs."""
    import compare

    results = tmp_path / "results"
    (results / "comparison").mkdir(parents=True)
    (results / "features").mkdir()
    (results / "shared").mkdir()

    # Two sessions in two contexts. Labels chosen to produce both degenerate
    # (0,0)/(1,1) and mixed (0,1)/(1,0) motifs.
    np.save(results / "shared" / "a_labels.npy",
            np.array([0, 0, 0, 1, 1, 1], dtype=np.int32))
    np.save(results / "shared" / "b_labels.npy",
            np.array([0, 1, 0, 1, 0, 1], dtype=np.int32))

    index = {
        "a": {"features_path": str(results / "features" / "a.npy")},
        "b": {"features_path": str(results / "features" / "b.npy")},
        "_meta": {"n_features": 51},
    }
    with open(results / "features" / "index.json", "w") as f:
        json.dump(index, f)

    with open(results / "shared" / "cluster_info.json", "w") as f:
        json.dump({"n_clusters": 2, "cluster_centers": [[0.0], [1.0]]}, f)

    meta = pd.DataFrame({
        "stem": ["a", "b"],
        "filename": ["a.mp4", "b.mp4"],
        "context": ["A", "B"],
        "animal_id": ["m1", "m2"],
    })
    meta.to_csv(tmp_path / "metadata.csv", index=False)

    import vieb_config
    monkeypatch.setattr(vieb_config, "get_results_dir", lambda: str(results))
    monkeypatch.setattr(vieb_config, "get_metadata_path", lambda: str(tmp_path / "metadata.csv"))
    monkeypatch.setattr(vieb_config, "get_condition_a_label", lambda: "A")
    monkeypatch.setattr(vieb_config, "get_condition_b_label", lambda: "B")
    monkeypatch.setattr(vieb_config, "normalize_metadata_columns", lambda df: df)

    compare.cmd_motifs(group_col="context")

    df = pd.read_csv(results / "comparison" / "motifs.csv")
    motifs = set(df["motif"].astype(str))
    # No degenerate motifs survive
    for deg in ("(0, 0)", "(1, 1)", "(0, 0, 0)", "(1, 1, 1)"):
        assert deg not in motifs, f"degenerate motif {deg} should be excluded"
    # At least one genuine multi-state motif remains
    assert any(s in motifs for s in ("(0, 1)", "(1, 0)"))


# ── Bout duration by context ──────────────────────────────────────────────

def test_bout_duration_by_context_written(tmp_path, monkeypatch):
    import compare

    results = tmp_path / "results"
    (results / "comparison").mkdir(parents=True)

    import vieb_config
    monkeypatch.setattr(vieb_config, "get_results_dir", lambda: str(results))

    bouts = pd.DataFrame({
        "stem": ["v1", "v1", "v2", "v2"],
        "state": [0, 0, 0, 1],
        "duration_sec": [2.0, 4.0, 1.0, 3.0],
        "context": ["A", "A", "B", "B"],
    })
    compare._write_bout_duration_by_context(bouts, pd.DataFrame())

    out = results / "comparison" / "bout_duration_by_context.csv"
    assert out.exists()
    df = pd.read_csv(out)
    cols = {"state_id", "context", "bout_count", "mean_bout_dur_sec",
            "median_bout_dur_sec", "duration_enrichment"}
    assert cols.issubset(set(df.columns))
    # State 0 appears in both contexts (repeated-state duration is reported here,
    # not as a sequence motif).
    s0 = df[df["state_id"] == 0]
    assert set(s0["context"]) == {"A", "B"}
    a_mean = float(s0[s0["context"] == "A"]["mean_bout_dur_sec"].iloc[0])
    assert abs(a_mean - 3.0) < 1e-6  # mean of [2.0, 4.0]
