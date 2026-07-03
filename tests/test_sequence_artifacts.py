from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sequence_artifacts import build_sequence_artifacts  # noqa: E402


def test_build_sequence_artifacts_stories_and_journeys(tmp_path):
    results = tmp_path / "results"
    shared = results / "shared"
    shared.mkdir(parents=True)

    labels_a = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int32)
    labels_b = np.array([1, 1, 1, 0], dtype=np.int32)
    np.save(shared / "vidA_labels.npy", labels_a)
    np.save(shared / "vidB_labels.npy", labels_b)
    np.save(shared / "vidA_probs.npy", np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 1.0, 0.9]))
    np.save(shared / "vidB_probs.npy", np.ones(len(labels_b)))

    summary = pd.DataFrame({
        "stem": ["vidA", "vidB"],
        "animal_id": ["s1", "s1"],
        "day": [0, 1],
        "context": ["A", "A"],
        "state_0_frac": [0.5, 0.25],
        "state_1_frac": [0.5, 0.75],
    })
    design = {
        "subject_col": "animal_id",
        "time_col": "day",
        "time_order": [0, 1],
        "condition_cols": ["context"],
        "detected_mode": "time_and_condition",
    }

    outputs = build_sequence_artifacts(summary, design, results, fps=2.0, n_clusters=2)

    bouts = outputs["video_story_bouts"]
    stories = outputs["video_stories"]
    journeys = outputs["subject_journeys"]

    assert (results / "sequences" / "video_story_bouts.csv").exists()
    assert (results / "sequences" / "video_stories.csv").exists()
    assert (results / "sequences" / "subject_journeys.csv").exists()

    first = bouts.iloc[0]
    assert first["video_id"] == "vidA"
    assert first["subject_id"] == "s1"
    assert first["timepoint"] == 0
    assert first["condition"] == "A"
    assert first["state"] == 0
    assert first["start_frame"] == 0
    assert first["end_frame"] == 1
    assert first["duration_sec"] == 1.0
    assert round(float(first["confidence_mean"]), 3) == 0.85

    story_a = stories.set_index("video_id").loc["vidA"]
    assert story_a["dominant_state"] == 0
    assert story_a["n_bouts"] == 4
    assert story_a["n_transitions"] == 3
    assert story_a["state_sequence_rle"] == "0:2|1:2|0:2|1:2"
    assert story_a["top_motifs"] == "(0, 1):2"

    journeys = journeys.sort_values("timepoint")
    assert journeys["distance_from_baseline"].tolist() == [0.0, 0.25]
    assert journeys.iloc[0]["story_similarity_to_baseline"] == 1.0
    assert json.loads(journeys.iloc[1]["state_occupancy_vector"]) == [0.25, 0.75]


def test_build_sequence_artifacts_time_only_axes(tmp_path):
    results = tmp_path / "results"
    shared = results / "shared"
    shared.mkdir(parents=True)
    np.save(shared / "rat1_baseline_labels.npy", np.array([0, 0, 0, 1], dtype=np.int32))
    np.save(shared / "rat1_week2_labels.npy", np.array([1, 1, 1, 0], dtype=np.int32))

    summary = pd.DataFrame({
        "stem": ["rat1_baseline", "rat1_week2"],
        "animal_id": ["rat1", "rat1"],
        "day": ["baseline", "week2"],
        "state_0_frac": [0.75, 0.25],
        "state_1_frac": [0.25, 0.75],
    })
    design = {
        "subject_col": "animal_id",
        "time_col": "day",
        "time_order": ["baseline", "week2"],
        "condition_cols": [],
        "detected_mode": "time_only",
    }

    outputs = build_sequence_artifacts(summary, design, results, fps=1.0, n_clusters=2)

    stories = outputs["video_stories"].sort_values("timepoint")
    assert stories["condition"].fillna("").tolist() == ["", ""]
    journeys = outputs["subject_journeys"].sort_values("timepoint")
    assert journeys["subject_id"].tolist() == ["rat1", "rat1"]
    assert journeys["distance_from_baseline"].tolist() == [0.0, 0.5]
