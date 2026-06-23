import json
import os
import sys

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
    cfg = {"fps": 30, "results_dir": str(results)}
    with open(tmp_path / "config.json", "w") as f:
        json.dump(cfg, f)

    import vieb_config
    monkeypatch.setattr(vieb_config, "get_results_dir", lambda: str(results))

    # Monkeypatch _export_clip to always return True (no real video)
    import generate_clips
    monkeypatch.setattr(generate_clips, "_export_clip", lambda *a, **kw: True)

    return results


def test_cmd_motif_clips_writes_index(tmp_path, monkeypatch):
    results = _setup_motif_project(tmp_path, monkeypatch)
    from generate_clips import cmd_motif_clips
    cmd_motif_clips(fps=30, top_motifs=2, clips_per_motif=2)

    index_path = results / "motifs" / "motif_clip_index.csv"
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

    index_path = results / "motifs" / "motif_clip_index.csv"
    assert index_path.exists()
    df = pd.read_csv(index_path)
    skipped = df[df["skipped_reason"].fillna("") == "source video missing"]
    assert len(skipped) > 0


def test_motif_clips_missing_source_gives_clear_error(tmp_path, monkeypatch):
    results = _setup_motif_project(tmp_path, monkeypatch)
    (results / "motifs" / "motif_sequences.csv").unlink()
    from generate_clips import cmd_motif_clips
    try:
        cmd_motif_clips(fps=30)
    except SystemExit as exc:
        assert "Missing motif occurrence source" in str(exc)
    else:
        raise AssertionError("expected SystemExit for missing motif_sequences.csv")


def test_motif_dir_name_sanitizes_paths():
    from generate_clips import _motif_dir_name
    name = _motif_dir_name("../bad motif/(1, 2)", "tri/../gram")
    assert "/" not in name
    assert "\\" not in name
    assert ".." not in name
    assert name
