"""Tests for generate_clips.py tolerating missing/unresolvable video_path
entries (H5-extracted sessions with video_path=None) instead of crashing."""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_resolve_video_path_none_and_empty_return_none():
    from generate_clips import _resolve_video_path
    assert _resolve_video_path(None) is None
    assert _resolve_video_path("") is None


def test_resolve_video_path_unfindable_returns_none(tmp_path, monkeypatch):
    from generate_clips import _resolve_video_path
    import vieb_config as vc
    monkeypatch.setattr(vc, "get_raw_videos_dir", lambda: str(tmp_path / "raw"))
    assert _resolve_video_path(str(tmp_path / "does_not_exist.mp4")) is None


def _setup_project(tmp_path, monkeypatch):
    """Build a minimal results/ tree with a mix of valid/None/unresolvable
    video_path entries, following test_motif_clips.py's established pattern."""
    results = tmp_path / "results"
    (results / "comparison").mkdir(parents=True)
    (results / "characterization").mkdir()
    (results / "features").mkdir()
    (results / "shared").mkdir()
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "vid_ok.mp4").touch()

    n_frames = 300
    for stem in ("vid_ok", "vid_none", "vid_unresolvable"):
        np.save(results / "features" / f"{stem}_features.npy",
                np.random.rand(n_frames, 5).astype(np.float32))
        labels = np.zeros(n_frames, dtype=np.int32)
        labels[100:200] = 1
        np.save(results / "shared" / f"{stem}_labels.npy", labels)

    index = {
        "vid_ok": {
            "video_path": str(tmp_path / "raw" / "vid_ok.mp4"),
            "features_path": str(results / "features" / "vid_ok_features.npy"),
            "n_frames": n_frames,
        },
        "vid_none": {
            "video_path": None,
            "features_path": str(results / "features" / "vid_none_features.npy"),
            "n_frames": n_frames,
        },
        "vid_unresolvable": {
            "video_path": str(tmp_path / "raw" / "does_not_exist.mp4"),
            "features_path": str(results / "features" / "vid_unresolvable_features.npy"),
            "n_frames": n_frames,
        },
        "_meta": {"n_features": 5},
    }
    with open(results / "features" / "index.json", "w") as f:
        json.dump(index, f)

    cluster_info = {"n_clusters": 2, "cluster_centers": [[0, 0, 0, 0, 0], [1, 1, 1, 1, 1]]}
    with open(results / "shared" / "cluster_info.json", "w") as f:
        json.dump(cluster_info, f)

    summary = pd.DataFrame({
        "stem": ["vid_ok", "vid_none", "vid_unresolvable"],
        "animal_id": ["a1", "a2", "a3"],
        "state_0_frac": [0.6, 0.6, 0.6],
        "state_1_frac": [0.4, 0.4, 0.4],
    })
    summary.to_csv(results / "comparison" / "summary_table.csv", index=False)

    (tmp_path / "metadata.csv").write_text(
        "filename,animal_id,context,day,experiment\n"
        "vid_ok.mp4,a1,A,1,exp\nvid_none.mp4,a2,A,1,exp\nvid_unresolvable.mp4,a3,A,1,exp\n"
    )

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
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    (tmp_path / "app_config.json").write_text(json.dumps({"active_project": str(tmp_path)}))

    import vieb_config
    monkeypatch.setattr(vieb_config, "PROJECT_ROOT", str(tmp_path))
    monkeypatch.setattr(vieb_config, "_APP_CONFIG_PATH", str(tmp_path / "app_config.json"))
    monkeypatch.setattr(vieb_config, "_CONFIG_PATH", str(tmp_path / "config.json"))
    monkeypatch.setattr(vieb_config, "get_results_dir", lambda: str(results))
    monkeypatch.setattr(vieb_config, "get_raw_videos_dir", lambda: str(tmp_path / "raw"))
    monkeypatch.setattr(vieb_config, "get_clips_dir", lambda: str(tmp_path / "clips"))

    import generate_clips
    monkeypatch.setattr(generate_clips, "_export_clip", lambda *a, **kw: True)

    return results


def test_build_bouts_df_tolerates_none_video_path(tmp_path, monkeypatch):
    results = _setup_project(tmp_path, monkeypatch)
    from generate_clips import _build_bouts_df, _load_prereqs

    index, cluster_info, df_summary, meta = _load_prereqs()
    bouts_df = _build_bouts_df(index, fps=30.0, meta=meta)

    assert not bouts_df.empty
    none_rows = bouts_df[bouts_df["stem"] == "vid_none"]
    assert not none_rows.empty
    assert none_rows["video_path"].isna().all()
    ok_rows = bouts_df[bouts_df["stem"] == "vid_ok"]
    assert (ok_rows["video_path"] == str(tmp_path / "raw" / "vid_ok.mp4")).all()


def test_cmd_clips_mixed_valid_none_unresolvable_no_crash_and_summary(tmp_path, monkeypatch, capsys):
    results = _setup_project(tmp_path, monkeypatch)
    from generate_clips import cmd_clips

    cmd_clips(fps=30.0, n_clips=5, output_dir=str(tmp_path / "clips"))

    captured = capsys.readouterr()
    assert "sessions have a usable local video" in captured.out
    assert "1 missing video_path" in captured.out
    assert "1 unresolvable locally" in captured.out
    # Clips were attempted/written only for the resolvable session (_export_clip
    # is monkeypatched to always succeed, so "Done: N/N" confirms nothing failed).
    assert "Done:" in captured.out
    assert "0/0" not in captured.out


def test_cmd_clips_from_existing_bouts_csv_tolerates_none_video_path(tmp_path, monkeypatch):
    """Exercises the vp_map rewrite path (bouts.csv already exists) rather
    than _build_bouts_df."""
    results = _setup_project(tmp_path, monkeypatch)
    from generate_clips import _build_bouts_df, _load_prereqs, cmd_clips

    index, cluster_info, df_summary, meta = _load_prereqs()
    bouts_df = _build_bouts_df(index, fps=30.0, meta=meta)
    bouts_df.to_csv(results / "characterization" / "bouts.csv", index=False)

    # Should not raise despite the None/unresolvable video_path entries.
    cmd_clips(fps=30.0, n_clips=5, output_dir=str(tmp_path / "clips"))


def test_cmd_clips_all_videos_missing_raises(tmp_path, monkeypatch):
    results = _setup_project(tmp_path, monkeypatch)
    from generate_clips import cmd_clips

    # Break the one resolvable video too, so nothing is usable.
    (tmp_path / "raw" / "vid_ok.mp4").unlink()

    try:
        cmd_clips(fps=30.0, n_clips=5, output_dir=str(tmp_path / "clips"))
    except RuntimeError as exc:
        assert "No usable local video files" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when no video is usable")
