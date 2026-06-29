import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_state_occupancy_plot_uses_project_path(tmp_path):
    from compare import _save_state_occupancy_plot

    project_results = tmp_path / "project_a" / "results"
    repo_results = tmp_path / "results"
    df = pd.DataFrame({
        "state_0_frac": [0.2, 0.4],
        "state_1_frac": [0.8, 0.6],
    })

    out = project_results / "characterization" / "state_occupancy.png"
    _save_state_occupancy_plot(df, ["state_0_frac", "state_1_frac"], str(out))

    assert out.exists()
    assert not (repo_results / "characterization" / "state_occupancy.png").exists()


def test_contrast_vector_plot_uses_project_path(tmp_path):
    from compare import _save_contrast_vector_comparison_plot

    project_results = tmp_path / "project_b" / "results"
    repo_results = tmp_path / "results"
    df = pd.DataFrame({
        "cohort_label": ["A", "B"],
        "contrast_magnitude": [0.15, 0.42],
    })

    out = project_results / "comparison" / "contrast_vector_comparison.png"
    _save_contrast_vector_comparison_plot(df, str(out))

    assert out.exists()
    assert not (repo_results / "comparison" / "contrast_vector_comparison.png").exists()
