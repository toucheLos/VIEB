"""Tests for the Video Stories panel (Analysis > Video Stories).

Follows this repo's convention (see tests/test_gui_artifact_scroll.py,
tests/test_motif_clip_playback.py): offscreen QApplication for anything that
needs real Qt widgets, and otherwise prefer exercising the module-level pure
functions directly since they don't need Qt at all.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import pytest

pytest.importorskip("PyQt5")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import views.video_stories as vs  # noqa: E402


_QAPP = None


def _app():
    # Keep a persistent reference — an unreferenced QApplication return value
    # gets garbage collected, leaving QApplication.instance() None again and
    # crashing the next QWidget construction ("Must construct a QApplication
    # before a QWidget").
    global _QAPP
    if _QAPP is None:
        from PyQt5.QtWidgets import QApplication
        _QAPP = QApplication.instance() or QApplication([])
    return _QAPP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_native_sequences(results_dir, spence=True):
    seq_dir = results_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)
    if spence:
        stories = pd.DataFrame({
            "video_id": ["r12_baseline", "r12_week2", "r12_week3"],
            "subject_id": ["rat12", "rat12", "rat12"],
            "timepoint": ["baseline", "week2", "week3"],
            "condition": ["", "", ""],
            "duration_sec": [10.0, 10.0, 10.0],
            "dominant_state": [0, 1, 1],
            "state_entropy": [0.4, 0.5, 0.5],
            "n_bouts": [2, 2, 2],
            "n_transitions": [1, 1, 1],
            "transition_rate": [0.1, 0.1, 0.1],
            "mean_bout_duration": [5.0, 5.0, 5.0],
            "short_bout_fraction": [0.0, 0.0, 0.0],
            "state_sequence_rle": ["0:150|1:150"] * 3,
            "top_motifs": ["(0, 1):3", "", ""],
        })
        bouts = pd.DataFrame({
            "video_id": ["r12_baseline", "r12_baseline", "r12_week2", "r12_week2"],
            "subject_id": ["rat12"] * 4,
            "timepoint": ["baseline", "baseline", "week2", "week2"],
            "condition": [""] * 4,
            "state": [0, 1, 1, 0],
            "start_frame": [0, 150, 0, 120],
            "end_frame": [149, 299, 119, 239],
            "start_sec": [0.0, 5.0, 0.0, 4.0],
            "end_sec": [4.97, 9.97, 3.97, 7.97],
            "duration_sec": [5.0, 5.0, 4.0, 4.0],
            "confidence_mean": [0.9, 0.8, 0.85, 0.75],
        })
    else:
        stories = pd.DataFrame({
            "video_id": ["m3_dayA_context1"],
            "subject_id": ["mouse3"],
            "timepoint": ["day1"],
            "condition": ["contextA"],
            "duration_sec": [8.0],
            "dominant_state": [2],
            "state_entropy": [0.3],
            "n_bouts": [1],
            "n_transitions": [0],
            "transition_rate": [0.0],
            "mean_bout_duration": [8.0],
            "short_bout_fraction": [0.0],
            "state_sequence_rle": ["2:240"],
            "top_motifs": [""],
        })
        bouts = pd.DataFrame({
            "video_id": ["m3_dayA_context1"],
            "subject_id": ["mouse3"],
            "timepoint": ["day1"],
            "condition": ["contextA"],
            "state": [2],
            "start_frame": [0],
            "end_frame": [239],
            "start_sec": [0.0],
            "end_sec": [7.97],
            "duration_sec": [8.0],
            "confidence_mean": [0.95],
        })
    stories.to_csv(seq_dir / "video_stories.csv", index=False)
    bouts.to_csv(seq_dir / "video_story_bouts.csv", index=False)
    return stories, bouts


def _write_analysis_design(results_dir, time_order):
    import json
    (results_dir).mkdir(parents=True, exist_ok=True)
    (results_dir / "analysis_design.json").write_text(json.dumps({"time_order": time_order}))


# ---------------------------------------------------------------------------
# Pure-logic tests
# ---------------------------------------------------------------------------

def test_load_story_data_reads_native_csvs(tmp_path):
    _write_native_sequences(tmp_path)
    stories, bouts, source = vs.load_story_data(tmp_path)
    assert source == "native"
    assert len(stories) == 3
    assert len(bouts) == 4
    assert set(stories["video_id"]) == {"r12_baseline", "r12_week2", "r12_week3"}


def test_load_story_data_returns_none_when_nothing_present(tmp_path):
    stories, bouts, source = vs.load_story_data(tmp_path)
    assert stories is None and bouts is None and source is None


def test_build_fallback_stories_from_legacy_outputs():
    legacy_bouts = pd.DataFrame({
        "stem": ["v2"],
        "state": [3],
        "start_frame": [0],
        "end_frame": [89],
        "start_sec": [0.0],
        "end_sec": [2.9],
        "duration_sec": [2.9],
        "context": ["B"],
        "animal_id": [9],
        "day": ["week3"],
        "experiment": ["e"],
        "no_shock": [True],
    })
    summary = pd.DataFrame({"stem": ["v2"], "state_0_frac": [0.1], "state_3_frac": [0.9]})
    stories, bouts = vs.build_fallback_stories(legacy_bouts, summary)
    row = stories.iloc[0]
    assert row["video_id"] == "v2"
    assert row["subject_id"] == 9
    assert row["timepoint"] == "week3"
    assert row["condition"] == "B"
    # dominant_state must resolve to the real state id (3), not the column position (1).
    assert row["dominant_state"] == 3
    assert "video_id" in bouts.columns


def test_load_story_data_falls_back_to_legacy_when_native_missing(tmp_path):
    (tmp_path / "characterization").mkdir()
    (tmp_path / "comparison").mkdir()
    pd.DataFrame({
        "stem": ["v2"], "state": [3], "start_frame": [0], "end_frame": [89],
        "start_sec": [0.0], "end_sec": [2.9], "duration_sec": [2.9],
        "context": ["B"], "animal_id": [9], "day": ["week3"],
        "experiment": ["e"], "no_shock": [True],
    }).to_csv(tmp_path / "characterization" / "bouts.csv", index=False)
    pd.DataFrame({"stem": ["v2"], "state_0_frac": [0.1], "state_3_frac": [0.9]}).to_csv(
        tmp_path / "comparison" / "summary_table.csv", index=False
    )
    stories, bouts, source = vs.load_story_data(tmp_path)
    assert source == "fallback"
    assert stories.iloc[0]["video_id"] == "v2"


def test_find_bout_at_time_orders_and_finds_short_bouts():
    bouts = pd.DataFrame({
        "start_sec": [3.0, 0.0, 1.0, 1.05],
        "end_sec": [5.0, 0.9, 1.04, 2.9],
        "state": [0, 0, 1, 2],
    })
    # A sub-second bout (1.00-1.04s) must still be found precisely.
    hit = vs.find_bout_at_time(bouts, 1.02)
    assert hit["state"] == 1
    # A point outside any bout returns None.
    assert vs.find_bout_at_time(bouts, 0.95) is None
    # A point in the widest/last-listed bout still resolves correctly regardless of input order.
    hit2 = vs.find_bout_at_time(bouts, 4.0)
    assert hit2["state"] == 0
    assert hit2["start_sec"] == 3.0


def test_compute_clip_window_hits_target_for_short_bout():
    start, end = vs.compute_clip_window(10.0, 10.3)
    assert end - start == pytest.approx(5.0)
    assert (start + end) / 2 == pytest.approx(10.15, abs=0.05)


def test_compute_clip_window_expands_to_minimum():
    start, end = vs.compute_clip_window(
        10.0, 10.05, target_clip_sec=1.0, min_clip_sec=3.0, pad_before_sec=0.1, pad_after_sec=0.1,
    )
    assert end - start == pytest.approx(3.0)


def test_compute_clip_window_clamps_at_zero_by_shifting_right():
    start, end = vs.compute_clip_window(0.1, 0.15)
    assert start == 0.0
    assert end - start == pytest.approx(5.0)


def test_compute_clip_window_clamps_at_video_duration():
    start, end = vs.compute_clip_window(10.0, 10.3, video_duration_sec=11.0)
    assert end == 11.0
    assert start >= 0.0


def test_order_timepoints_sorts_spence_workflow(tmp_path):
    _write_analysis_design(
        tmp_path,
        ["baseline", "week2", "week3", "week4", "week6", "week7", "week8", "week9"],
    )
    values = ["week9", "baseline", "week3", "week2", "week6", "week7", "week8", "week4"]
    assert vs.order_timepoints(values, tmp_path) == [
        "baseline", "week2", "week3", "week4", "week6", "week7", "week8", "week9",
    ]


def test_order_timepoints_falls_back_to_natural_sort_without_design(tmp_path):
    values = ["b", "a", "c"]
    assert vs.order_timepoints(values, tmp_path) == ["a", "b", "c"]


def test_load_state_labels_reads_existing_file_no_second_store(tmp_path):
    validation_dir = tmp_path / "validation"
    validation_dir.mkdir()
    pd.DataFrame({
        "state_id": [3, 7],
        "label": ["asymmetric gait", ""],
        "category": ["", ""],
    }).to_csv(validation_dir / "state_labels.csv", index=False)
    labels = vs.load_state_labels(tmp_path)
    assert labels[3]["label"] == "asymmetric gait"
    assert vs.state_label_text(3, labels) == "State 3 — tentative label: asymmetric gait"
    assert vs.state_label_text(7, labels) == "State 7 — unlabeled"
    assert vs.state_label_text(99, labels) == "State 99 — unlabeled"


def test_load_state_labels_missing_file_returns_empty(tmp_path):
    assert vs.load_state_labels(tmp_path) == {}


def test_parse_and_membership():
    motifs = vs.parse_top_motifs("(5, 3):26;(3, 5):24")
    assert motifs == [((5, 3), 26), ((3, 5), 24)]
    hits = vs.bout_motif_membership([5, 3, 5, 3, 2], 1, motifs)
    assert (5, 3) in hits
    assert (3, 5) in hits


def test_resolve_source_video_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(vs._vc, "get_raw_videos_dir", lambda: str(tmp_path / "nope"))
    assert vs.resolve_source_video("does_not_exist", {}) is None


# ---------------------------------------------------------------------------
# Qt widget tests
# ---------------------------------------------------------------------------

def test_selector_cascade_updates_selected_story(tmp_path, monkeypatch):
    _app()
    _write_native_sequences(tmp_path, spence=True)
    monkeypatch.setattr(vs, "RESULTS", tmp_path)

    view = vs.VideoStoriesView(cfg={"fps": 30.0})
    assert view._timepoint_active is True
    assert view._current_story["video_id"] == "r12_baseline"

    idx = view._timepoint_combo.findText("week2")
    assert idx >= 0
    view._timepoint_combo.setCurrentIndex(idx)
    assert view._current_story["video_id"] == "r12_week2"
    assert list(view._current_bouts["state"]) == [1, 0]  # sorted by start_frame


def test_luna_workflow_metadata_renders_in_summary(tmp_path, monkeypatch):
    _app()
    _write_native_sequences(tmp_path, spence=False)
    monkeypatch.setattr(vs, "RESULTS", tmp_path)

    view = vs.VideoStoriesView(cfg={"fps": 30.0})
    story = view._current_story
    assert story["subject_id"] == "mouse3"
    assert story["timepoint"] == "day1"
    assert story["condition"] == "contextA"
    # Single subject/timepoint/condition -> selector levels should collapse.
    assert view._subject_active is False
    assert view._timepoint_active is False
    assert view._condition_active is False


def test_timeline_click_finds_correct_bout(tmp_path, monkeypatch):
    _app()
    _write_native_sequences(tmp_path, spence=True)
    monkeypatch.setattr(vs, "RESULTS", tmp_path)
    view = vs.VideoStoriesView(cfg={"fps": 30.0})

    class _FakeEvent:
        inaxes = view._timeline_canvas.ax
        xdata = 1.0

    opened = []
    view._open_segment_dialog = lambda bout: opened.append(bout)
    view._on_timeline_click(_FakeEvent())
    assert len(opened) == 1
    assert opened[0]["state"] == 0


def test_missing_source_video_shows_inline_message_no_crash(tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QDialog
    _app()
    _write_native_sequences(tmp_path, spence=True)
    monkeypatch.setattr(vs, "RESULTS", tmp_path)
    monkeypatch.setattr(vs._vc, "get_raw_videos_dir", lambda: str(tmp_path / "no_raw_videos"))
    monkeypatch.setattr(QDialog, "exec_", lambda self: None)

    view = vs.VideoStoriesView(cfg={"fps": 30.0})
    bout = view._current_bouts.iloc[0].to_dict()
    view._open_segment_dialog(bout)  # must not raise


# ---------------------------------------------------------------------------
# Part B: Journey / comparison layer
# ---------------------------------------------------------------------------

def _write_journey_dataset(results_dir, subject, timepoints, condition=""):
    """timepoints: list of (timepoint, [(state, duration_sec), ...]) for one
    subject, one video per timepoint. Rows are written in a shuffled order
    to prove chronological rendering comes from sorting, not file order."""
    import random

    seq_dir = results_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    story_rows, bout_rows, journey_rows = [], [], []
    for i, (tp, states) in enumerate(timepoints):
        video_id = f"{subject}_{tp}"
        t = 0.0
        for state, dur in states:
            bout_rows.append({
                "video_id": video_id, "subject_id": subject, "timepoint": tp, "condition": condition,
                "state": state, "start_frame": int(t * 30), "end_frame": int((t + dur) * 30) - 1,
                "start_sec": t, "end_sec": t + dur, "duration_sec": dur, "confidence_mean": 0.9,
            })
            t += dur
        story_rows.append({
            "video_id": video_id, "subject_id": subject, "timepoint": tp, "condition": condition,
            "duration_sec": t, "dominant_state": states[0][0], "state_entropy": 0.5 + i * 0.1,
            "n_bouts": len(states), "n_transitions": len(states) - 1, "transition_rate": 0.1 + i * 0.05,
            "mean_bout_duration": t / len(states), "short_bout_fraction": 0.0,
            "state_sequence_rle": "", "top_motifs": "",
        })
        journey_rows.append({
            "subject_id": subject, "timepoint": tp, "distance_from_baseline": round(i * 0.2, 3),
            "dominant_state": states[0][0], "state_entropy": round(0.5 + i * 0.1, 3),
            "transition_rate": round(0.1 + i * 0.05, 3),
            "state_occupancy_vector": "[1,0]", "story_similarity_to_baseline": round(1.0 - i * 0.1, 3),
        })

    shuffled_stories = story_rows[:]
    random.Random(0).shuffle(shuffled_stories)
    shuffled_bouts = bout_rows[:]
    random.Random(1).shuffle(shuffled_bouts)

    pd.DataFrame(shuffled_stories).to_csv(seq_dir / "video_stories.csv", index=False)
    pd.DataFrame(shuffled_bouts).to_csv(seq_dir / "video_story_bouts.csv", index=False)
    pd.DataFrame(journey_rows).to_csv(seq_dir / "subject_journeys.csv", index=False)


def test_select_representative_video_prefers_current_and_falls_back():
    stories = pd.DataFrame({
        "video_id": ["b", "a"], "subject_id": ["rat1", "rat1"],
        "timepoint": ["week2", "week2"], "condition": ["", ""],
    })
    assert vs.select_representative_video(stories, "rat1", "week2") == "a"
    assert vs.select_representative_video(stories, "rat1", "week2", prefer_video_id="b") == "b"
    assert vs.select_representative_video(stories, "rat1", "week2", prefer_video_id="zzz") == "a"
    assert vs.select_representative_video(stories, "rat1", "week9") is None


def test_comparison_strips_render_in_order_spence(tmp_path, monkeypatch):
    _app()
    _write_journey_dataset(tmp_path, "rat519", [
        ("week9", [(0, 2.0)]),
        ("baseline", [(1, 2.0)]),
        ("week2", [(0, 2.0)]),
    ])
    _write_analysis_design(
        tmp_path, ["baseline", "week2", "week3", "week4", "week6", "week7", "week8", "week9"]
    )
    monkeypatch.setattr(vs, "RESULTS", tmp_path)

    view = vs.VideoStoriesView(cfg={"fps": 30.0})
    view._compare_toggle.setChecked(True)
    labels = [t.get_text() for t in view._compare_canvas.ax.get_yticklabels()]
    assert labels == ["baseline", "week2", "week9"]


def test_comparison_strips_render_in_order_luna_day_based(tmp_path, monkeypatch):
    _app()
    _write_journey_dataset(tmp_path, "mouse3", [
        ("day10", [(2, 2.0)]),
        ("day2", [(1, 2.0)]),
        ("day1", [(0, 2.0)]),
    ])
    _write_analysis_design(tmp_path, ["day1", "day2", "day10"])
    monkeypatch.setattr(vs, "RESULTS", tmp_path)

    view = vs.VideoStoriesView(cfg={"fps": 30.0})
    labels = [t.get_text() for t in view._compare_canvas.ax.get_yticklabels()]
    assert labels == ["day1", "day2", "day10"]


def test_compare_shows_placeholder_when_insufficient_timepoints(tmp_path, monkeypatch):
    _app()
    _write_native_sequences(tmp_path, spence=False)  # single timepoint fixture
    monkeypatch.setattr(vs, "RESULTS", tmp_path)
    view = vs.VideoStoriesView(cfg={"fps": 30.0})
    assert len(view._compare_canvas.ax.collections) == 0


def test_derived_metrics_read_journeys_columns_without_recompute(tmp_path, monkeypatch):
    _app()
    _write_journey_dataset(tmp_path, "rat519", [
        ("baseline", [(0, 2.0)]),
        ("week2", [(1, 2.0)]),
    ])
    _write_analysis_design(tmp_path, ["baseline", "week2"])
    monkeypatch.setattr(vs, "RESULTS", tmp_path)

    view = vs.VideoStoriesView(cfg={"fps": 30.0})
    journeys = pd.read_csv(tmp_path / "sequences" / "subject_journeys.csv").set_index("timepoint")

    ax1 = view._metrics_canvas1.ax
    trans_rate_line, entropy_line = ax1.lines[0], ax1.lines[1]
    assert list(trans_rate_line.get_ydata()) == pytest.approx(
        [journeys.loc["baseline", "transition_rate"], journeys.loc["week2", "transition_rate"]]
    )
    assert list(entropy_line.get_ydata()) == pytest.approx(
        [journeys.loc["baseline", "state_entropy"], journeys.loc["week2", "state_entropy"]]
    )

    dist_line = view._metrics_canvas2.ax.lines[0]
    assert list(dist_line.get_ydata()) == pytest.approx(
        [journeys.loc["baseline", "distance_from_baseline"], journeys.loc["week2", "distance_from_baseline"]]
    )


def test_derived_metrics_placeholder_when_no_journey_rows(tmp_path, monkeypatch):
    _app()
    _write_native_sequences(tmp_path, spence=True)  # no subject_journeys.csv written
    monkeypatch.setattr(vs, "RESULTS", tmp_path)
    view = vs.VideoStoriesView(cfg={"fps": 30.0})
    assert len(view._metrics_canvas1.ax.lines) == 0
    assert len(view._metrics_canvas2.ax.lines) == 0


def test_possible_split_states_absent_returns_empty_set(tmp_path):
    assert vs.load_possible_split_states(tmp_path) == set()


def test_possible_split_states_malformed_json_returns_empty_set(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "cluster_info.json").write_text("{not valid json")
    assert vs.load_possible_split_states(tmp_path) == set()


def test_possible_split_states_present_renders_hatch_without_crashing(tmp_path, monkeypatch):
    import json
    _app()
    _write_native_sequences(tmp_path, spence=True)
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "cluster_info.json").write_text(json.dumps({"possible_split_states": [1]}))
    monkeypatch.setattr(vs, "RESULTS", tmp_path)

    assert vs.load_possible_split_states(tmp_path) == {1}
    view = vs.VideoStoriesView(cfg={"fps": 30.0})  # first story's bouts contain states 0 and 1
    # one collection for normal (state 0) bouts, one for the flagged (state 1) hatch overlay
    assert len(view._timeline_canvas.ax.collections) == 2


def test_artifact_links_emit_correct_categories(tmp_path, monkeypatch):
    from PyQt5.QtWidgets import QPushButton
    _app()
    _write_native_sequences(tmp_path, spence=True)
    monkeypatch.setattr(vs, "RESULTS", tmp_path)
    view = vs.VideoStoriesView(cfg={"fps": 30.0})

    emitted = []
    view.navigate_artifacts_category.connect(lambda c: emitted.append(c))
    buttons = {b.text(): b for b in view._compare_content.findChildren(QPushButton)}
    buttons["Open Video Stories in Artifacts"].click()
    buttons["Open Story Clips in Artifacts"].click()
    buttons["Open Motif Clips in Artifacts"].click()
    buttons["Open State Clips in Artifacts"].click()
    assert emitted == ["Video Stories", "Video Stories", "Motifs", "Clips"]
