from __future__ import annotations

import inspect
import os
import sys

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

pytest.importorskip("PyQt5")

import user_interface as ui  # noqa: E402
from views.analysis import AnalysisView  # noqa: E402
from views.cluster_runs import ClusterRunsView  # noqa: E402


class _StatusBar:
    def __init__(self):
        self.messages = []

    def showMessage(self, text, timeout=0):
        self.messages.append((text, timeout))


class _Button:
    def __init__(self):
        self.enabled = True

    def setEnabled(self, enabled):
        self.enabled = enabled


class _Signal:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)


class _Label:
    def __init__(self):
        self.text = ""
        self.visible = None

    def setText(self, text):
        self.text = text

    def setVisible(self, visible):
        self.visible = visible


def test_cluster_runs_appears_in_workspace_navigation_order():
    assert ui._NAV_VIEWS == [
        "Overview",
        "Pipeline",
        "Cluster Runs",
        "Analysis",
        "Artifacts",
        "Settings",
        "Help",
    ]


def test_cluster_runs_is_not_remapped_to_pipeline_on_startup():
    source = inspect.getsource(ui.MainWindow._build)
    assert '"Cluster Runs": "Pipeline"' not in source


def test_reload_data_button_runs_report_regen(monkeypatch):
    """Reload Data merges with Regenerate Report — clicking it calls _run_report_regen."""
    win = ui.MainWindow.__new__(ui.MainWindow)
    status = _StatusBar()
    called = {"regen": 0}

    monkeypatch.setattr(win, "statusBar", lambda: status)
    monkeypatch.setattr(win, "_run_report_regen", lambda: called.__setitem__("regen", called["regen"] + 1))

    ui.MainWindow._on_reload_clicked(win)

    assert called["regen"] == 1


def test_report_regeneration_launches_compare_report(monkeypatch):
    created = {}

    class Worker:
        def __init__(self, args):
            created["args"] = args
            self.log = _Signal()
            self.done = _Signal()
            self.started = False

        def start(self):
            self.started = True
            created["started"] = True

    win = ui.MainWindow.__new__(ui.MainWindow)
    win._reload_btn = _Button()
    status = _StatusBar()
    monkeypatch.setattr(win, "statusBar", lambda: status)
    monkeypatch.setattr(ui, "SubprocessWorker", Worker)

    ui.MainWindow._run_report_regen(win)

    assert created["args"] == ["compare.py", "--report"]
    assert created["started"] is True
    assert status.messages[-1][0] == "Running compare.py --report…"
    assert win._reload_btn.enabled is False


def test_analysis_active_run_banner_uses_run_manifest():
    view = AnalysisView.__new__(AnalysisView)
    view._data = {
        "run_manifest": {
            "run_id": "run_004",
            "n_clusters": 7,
            "noise_frac": 0.125,
            "min_samples_requested": 0,
            "min_samples_resolved": 42,
        }
    }
    view._active_run_banner = _Label()
    view._active_run_bar = _Label()

    AnalysisView._update_active_run_banner(view)

    assert "Active cluster: run_004" in view._active_run_banner.text
    assert "7 states" in view._active_run_banner.text
    assert "noise: 12.5%" in view._active_run_banner.text
    assert "min_samples=Auto" in view._active_run_banner.text
    assert view._active_run_bar.visible is True


def test_cluster_runs_view_has_refresh_method():
    """ClusterRunsView must expose a refresh() method for external callers."""
    assert callable(getattr(ClusterRunsView, "refresh", None))
