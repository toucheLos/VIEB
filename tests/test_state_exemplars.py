import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _touch_video(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-real-video")
    return path


def _base_project(tmp_path, monkeypatch):
    results = tmp_path / "results"
    shared = results / "shared"
    features = results / "features"
    char = results / "characterization"
    for p in (shared, features, char):
        p.mkdir(parents=True, exist_ok=True)
    raw = tmp_path / "raw"
    raw.mkdir()
    clips = tmp_path / "clips"
    clips.mkdir()

    import generate_clips
    monkeypatch.setattr(generate_clips, "_res", lambda: str(results))
    monkeypatch.setattr(generate_clips._vc, "get_results_dir", lambda: str(results))
    monkeypatch.setattr(generate_clips._vc, "get_clips_dir", lambda: str(clips))
    monkeypatch.setattr(generate_clips._vc, "get_raw_videos_dir", lambda: str(raw))
    monkeypatch.setattr(generate_clips, "_meta", lambda: str(tmp_path / "metadata.csv"))
    monkeypatch.setattr(generate_clips, "_video_frame_count", lambda video_path, index_info=None: 1000)
    return results, raw, clips


def _cluster_info(n_clusters=2):
    return {"n_clusters": n_clusters, "cluster_centers": [[0.0, 0.0], [10.0, 10.0]][:n_clusters]}


def test_select_state_exemplars_chooses_each_state(tmp_path, monkeypatch):
    results, raw, _clips = _base_project(tmp_path, monkeypatch)
    v1 = _touch_video(raw / "s1.mp4")
    v2 = _touch_video(raw / "s2.mp4")
    np.save(results / "shared" / "s1_probs.npy", np.full(300, 0.95, dtype=np.float32))
    np.save(results / "shared" / "s2_probs.npy", np.full(300, 0.90, dtype=np.float32))
    np.save(results / "features" / "s1_features.npy", np.zeros((300, 2), dtype=np.float32))
    np.save(results / "features" / "s2_features.npy", np.full((300, 2), 10, dtype=np.float32))
    bouts = pd.DataFrame([
        {"stem": "s1", "state": 0, "start_frame": 30, "end_frame": 120, "duration_sec": 3.0, "video_path": str(v1), "features_path": str(results / "features" / "s1_features.npy"), "animal_id": "a1", "context": "A"},
        {"stem": "s2", "state": 1, "start_frame": 40, "end_frame": 150, "duration_sec": 3.7, "video_path": str(v2), "features_path": str(results / "features" / "s2_features.npy"), "animal_id": "a2", "context": "B"},
    ])
    from generate_clips import select_state_exemplars
    selected, skipped = select_state_exemplars(bouts, _cluster_info(), {}, fps=30, exemplars_per_state=1)
    assert {r["state_id"] for r in selected} == {0, 1}
    assert all(r["clip_path"] == "" for r in selected)
    assert not [r for r in skipped if r.get("skipped_reason")]


def test_select_state_exemplars_filters_short_bouts(tmp_path, monkeypatch):
    _results, raw, _clips = _base_project(tmp_path, monkeypatch)
    v1 = _touch_video(raw / "s1.mp4")
    bouts = pd.DataFrame([
        {"stem": "s1", "state": 0, "start_frame": 30, "end_frame": 40, "duration_sec": 0.2, "video_path": str(v1)},
    ])
    from generate_clips import select_state_exemplars
    selected, skipped = select_state_exemplars(bouts, {"n_clusters": 1, "cluster_centers": [[]]}, {}, fps=30)
    assert selected == []
    assert skipped[0]["skipped_reason"] == "short bout"


def test_select_state_exemplars_skips_missing_video(tmp_path, monkeypatch):
    _results, raw, _clips = _base_project(tmp_path, monkeypatch)
    missing = raw / "missing.mp4"
    bouts = pd.DataFrame([
        {"stem": "s1", "state": 0, "start_frame": 30, "end_frame": 120, "duration_sec": 3.0, "video_path": str(missing)},
    ])
    from generate_clips import select_state_exemplars
    selected, skipped = select_state_exemplars(bouts, {"n_clusters": 1, "cluster_centers": [[]]}, {}, fps=30)
    assert selected == []
    assert skipped[0]["skipped_reason"] == "source video missing"


def test_select_state_exemplars_handles_missing_metadata_and_confidence(tmp_path, monkeypatch):
    _results, raw, _clips = _base_project(tmp_path, monkeypatch)
    v1 = _touch_video(raw / "s1.mp4")
    bouts = pd.DataFrame([
        {"stem": "s1", "state": 0, "start_frame": 30, "end_frame": 120, "duration_sec": 3.0, "video_path": str(v1)},
    ])
    from generate_clips import select_state_exemplars
    selected, _skipped = select_state_exemplars(bouts, {"n_clusters": 1, "cluster_centers": [[]]}, {}, fps=30)
    assert len(selected) == 1
    assert selected[0]["animal_id"] == ""
    assert selected[0]["mean_confidence"] == ""
    assert "confidence unavailable" in selected[0]["selection_reason"]


def test_select_state_exemplars_diversifies_sessions(tmp_path, monkeypatch):
    results, raw, _clips = _base_project(tmp_path, monkeypatch)
    rows = []
    for i, animal in enumerate(["a1", "a1", "a2"], start=1):
        stem = f"s{i}"
        video = _touch_video(raw / f"{stem}.mp4")
        np.save(results / "features" / f"{stem}_features.npy", np.zeros((300, 2), dtype=np.float32))
        rows.append({
            "stem": stem, "state": 0, "start_frame": 30, "end_frame": 150,
            "duration_sec": 4.0 - (i * 0.1), "video_path": str(video),
            "features_path": str(results / "features" / f"{stem}_features.npy"),
            "animal_id": animal,
        })
    from generate_clips import select_state_exemplars
    selected, _skipped = select_state_exemplars(pd.DataFrame(rows), {"n_clusters": 1, "cluster_centers": [[0, 0]]}, {}, fps=30, exemplars_per_state=2)
    assert {r["animal_id"] for r in selected} == {"a1", "a2"}


def test_cmd_clips_writes_state_exemplar_manifest_with_relative_paths(tmp_path, monkeypatch):
    results, raw, clips = _base_project(tmp_path, monkeypatch)
    v1 = _touch_video(raw / "s1.mp4")
    (results / "comparison").mkdir(exist_ok=True)
    pd.DataFrame({"stem": ["s1"]}).to_csv(results / "comparison" / "summary_table.csv", index=False)
    pd.DataFrame([
        {"stem": "s1", "state": 0, "start_frame": 30, "end_frame": 120, "duration_sec": 3.0, "video_path": str(v1), "features_path": str(results / "features" / "s1_features.npy"), "animal_id": "a1"},
    ]).to_csv(results / "characterization" / "bouts.csv", index=False)
    np.save(results / "shared" / "s1_labels.npy", np.zeros(300, dtype=np.int32))
    np.save(results / "features" / "s1_features.npy", np.zeros((300, 2), dtype=np.float32))
    (results / "features" / "index.json").write_text(json.dumps({
        "s1": {"video_path": str(v1), "features_path": str(results / "features" / "s1_features.npy"), "n_frames": 300},
        "_meta": {"feature_names": ["f0", "f1"]},
    }))
    (results / "shared" / "cluster_info.json").write_text(json.dumps({"n_clusters": 1, "cluster_centers": [[0, 0]]}))
    import generate_clips
    monkeypatch.setattr(generate_clips, "_export_clip", lambda *args, **kwargs: True)
    generate_clips.cmd_clips(fps=30, n_clips=1, exemplars_per_state=1)
    out = results / "characterization" / "state_exemplars.csv"
    assert out.exists()
    df = pd.read_csv(out)
    ok = df[df["skipped_reason"].fillna("") == ""]
    assert len(ok) == 1
    assert str(ok.iloc[0]["clip_path"]).startswith("clips/state_0/clip_001.mp4")
    assert (clips / "state_0").exists()


def test_state_characterization_does_not_load_clip_until_selected(tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    results, _raw, clips = _base_project(tmp_path, monkeypatch)
    (clips / "state_0").mkdir(parents=True)
    (clips / "state_0" / "clip_001.mp4").write_bytes(b"video")
    (results / "characterization").mkdir(exist_ok=True)
    pd.DataFrame([
        {"state_id": 0, "rank": 1, "clip_path": "clips/state_0/clip_001.mp4", "animal_id": "a1", "context": "A", "duration_sec": 3.0, "selection_reason": "near state centroid", "skipped_reason": ""},
    ]).to_csv(results / "characterization" / "state_exemplars.csv", index=False)
    monkeypatch.setattr("views.state_characterization.RESULTS", results)
    import vieb_config
    monkeypatch.setattr(vieb_config, "get_results_dir", lambda: str(results))
    from views.state_characterization import StateCharacterizationView
    view = StateCharacterizationView({})
    calls = []
    class Player:
        def load(self, path):
            calls.append(path)
        def play(self):
            calls.append("play")
    view._player = Player()
    view.update_data({
        "state_summary": pd.DataFrame({"state": [0], "n_bouts": [1], "mean_bout_dur_sec": [3.0]}),
        "cluster_info": {"n_clusters": 1, "cluster_centers": [[]]},
    })
    view._state_list.setCurrentRow(0)
    app.processEvents()
    assert calls == []
    view._load_clip()
    assert calls and calls[0].endswith("clip_001.mp4")
