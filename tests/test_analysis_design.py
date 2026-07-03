from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis_design import detect_analysis_design  # noqa: E402


def test_luna_metadata_detects_time_and_condition():
    meta = pd.read_csv(os.path.join(os.path.dirname(__file__), "..", "metadata.csv"))

    design = detect_analysis_design(meta, {})

    assert design["subject_col"] == "animal_id"
    assert design["time_col"] == "day"
    assert "context" in design["condition_cols"]
    assert design["detected_mode"] == "time_and_condition"


def test_spence_style_metadata_detects_time_only_with_mapping():
    meta = pd.DataFrame({
        "source_file": [
            "rat1_baseline.csv",
            "rat1_week2.csv",
            "rat2_baseline.csv",
            "rat2_week2.csv",
        ],
        "rat": ["r1", "r1", "r2", "r2"],
        "timepoint": ["baseline", "week2", "baseline", "week2"],
    })
    cfg = {
        "metadata_schema": {
            "id_column": "source_file",
            "column_map": {
                "session_id": "source_file",
                "animal_id": "rat",
                "day": "timepoint",
            },
            "analysis_groups": [],
        }
    }

    design = detect_analysis_design(meta, cfg)

    assert design["subject_col"] == "animal_id"
    assert design["time_col"] == "day"
    assert design["time_order"] == ["baseline", "week2"]
    assert design["condition_cols"] == []
    assert design["detected_mode"] == "time_only"
