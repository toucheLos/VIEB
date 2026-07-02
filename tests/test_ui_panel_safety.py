from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import user_interface as ui  # noqa: E402
from views import dlc_setup  # noqa: E402


def test_safe_get_state_columns_sorted():
    df = pd.DataFrame({
        "state_10_frac": [0.1],
        "state_2_frac": [0.2],
        "other": [1],
        "state_1_frac": [0.3],
    })

    assert ui.safe_get_state_columns(df) == ["state_1_frac", "state_2_frac", "state_10_frac"]


def test_safe_infer_target_state_with_configured_groups():
    df = pd.DataFrame({
        "context": ["baseline", "drug", "baseline", "drug"],
        "state_0_frac": [0.4, 0.2, 0.5, 0.3],
        "state_1_frac": [0.1, 0.7, 0.2, 0.8],
    })

    target, reason = ui.safe_infer_target_state(df, "context", "baseline", "drug")

    assert reason == ""
    assert target == "state_1_frac"


def test_safe_infer_target_state_all_na_skips():
    df = pd.DataFrame({
        "timepoint": ["pre", "post"],
        "state_0_frac": [np.nan, np.nan],
    })

    target, reason = ui.safe_infer_target_state(df, "timepoint", "pre", "post")

    assert target is None
    assert "no valid state data" in reason


def test_safe_infer_target_state_no_state_columns_skips():
    df = pd.DataFrame({"context": ["baseline", "drug"]})

    target, reason = ui.safe_infer_target_state(df, "context", "baseline", "drug")

    assert target is None
    assert "no state fraction columns" in reason


def test_panel_available_disabled_learning_curve():
    ok, reason = ui.panel_available(
        pd.DataFrame({"context": ["baseline"]}),
        {"ui_panels": {"learning_curves": {"enabled": False}}},
        "learning_curves",
    )

    assert not ok
    assert "disabled" in reason


def test_panel_available_missing_optional_column():
    ok, reason = ui.panel_available(
        pd.DataFrame({"state_0_frac": [0.5]}),
        {},
        "transition_by_context",
    )

    assert not ok
    assert "context" in reason


def test_running_status_text_idle_running_without_command():
    assert ui._running_status_text("") == "running"
    assert ui._running_status_text(None) == "running"


def test_running_status_text_includes_command():
    assert ui._running_status_text("python compare.py --report") == "running: python compare.py --report"


def test_running_status_text_elides_long_command():
    text = ui._running_status_text("python " + "x" * 100, max_chars=24)

    assert len(text) == 24
    assert text.startswith("running: python")
    assert text.endswith("\u2026")


def test_dlc_gpu_state_from_log():
    assert dlc_setup._gpu_state_from_log("GPU detected: NVIDIA RTX") == "active"
    assert dlc_setup._gpu_state_from_log("Using CUDA device 0 for pose estimation.") == "active"
    assert dlc_setup._gpu_state_from_log("No GPU detected — using CPU") == "inactive"
    assert dlc_setup._gpu_state_from_log("unrelated line") is None


def test_dlc_log_autoscroll_decision():
    assert dlc_setup._should_stick_to_bottom(100, 100)
    assert dlc_setup._should_stick_to_bottom(94, 100)
    assert not dlc_setup._should_stick_to_bottom(50, 100)


def test_dlc_setup_running_status_wiring():
    dlc_src = open(os.path.join(os.path.dirname(__file__), "..", "views", "dlc_setup.py"), encoding="utf-8").read()
    main_src = open(os.path.join(os.path.dirname(__file__), "..", "user_interface.py"), encoding="utf-8").read()

    assert "worker_running = pyqtSignal(bool)" in dlc_src
    assert "worker_command = pyqtSignal(str)" in dlc_src
    assert "def stop_worker(self)" in dlc_src
    assert "self._dlc.worker_running.connect" in main_src
    assert "self._dlc.worker_command.connect" in main_src
    assert "self._dlc.stop_worker()" in main_src
