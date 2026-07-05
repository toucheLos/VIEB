"""Tests for the embedded motif clip player added to the Transitions & Motifs
tab's Section D (mirrors the clip-nav pattern already used by
views/state_characterization.py's per-state exemplar player).

Follows this repo's convention (see tests/test_workspace_cluster_runs.py) of
bypassing AnalysisView.__init__ via __new__ and using lightweight duck-typed
stand-ins instead of real Qt widgets, since constructing real QTableWidget
instances is not exercised elsewhere in this suite and is best avoided.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt  # noqa: E402

import views.analysis as analysis_mod  # noqa: E402
from views.analysis import AnalysisView  # noqa: E402


class _Signal:
    def __init__(self):
        self._slots = []

    def connect(self, slot):
        self._slots.append(slot)

    def emit(self):
        for slot in list(self._slots):
            slot()


class _Item:
    def __init__(self, text="", user_role_data=None):
        self._text = text
        self._data = {}
        if user_role_data is not None:
            self._data[Qt.UserRole] = user_role_data

    def text(self):
        return self._text

    def data(self, role):
        return self._data.get(role)


class _FakeTable:
    """Duck-typed stand-in for the parts of QTableWidget this feature uses."""

    def __init__(self, rows):
        # rows: list of 6-tuples of _Item
        self._rows = rows
        self._current_row = -1
        self.itemSelectionChanged = _Signal()

    def rowCount(self):
        return len(self._rows)

    def currentRow(self):
        return self._current_row

    def item(self, row, col):
        if 0 <= row < len(self._rows):
            return self._rows[row][col]
        return None

    def selectRow(self, row):
        self._current_row = row
        self.itemSelectionChanged.emit()


class _CheckBox:
    def __init__(self, checked=False):
        self._checked = checked

    def isChecked(self):
        return self._checked

    def setChecked(self, value):
        self._checked = value


class _Button:
    def __init__(self):
        self.enabled = True

    def setEnabled(self, value):
        self.enabled = value


class _Label:
    def __init__(self):
        self._text = ""

    def setText(self, text):
        self._text = text

    def text(self):
        return self._text


class _Player:
    def __init__(self):
        self.loaded = []
        self.play_count = 0

    def load(self, path):
        self.loaded.append(path)

    def play(self):
        self.play_count += 1


def _make_rows(entries):
    """entries: list of (motif, type, clip_name, animal, context, duration, full_path)."""
    rows = []
    for motif, mtype, clip_name, animal, context, duration, full_path in entries:
        rows.append((
            _Item(motif), _Item(mtype), _Item(clip_name, user_role_data=full_path),
            _Item(animal), _Item(context), _Item(duration),
        ))
    return rows


def _view(entries=()):
    view = AnalysisView.__new__(AnalysisView)
    view._t2_current_clip_path = None
    view._t2_clip_status = _Label()
    view._t2_export_clip_btn = _Button()
    view._t2_prev_clip_btn = _Button()
    view._t2_next_clip_btn = _Button()
    view._t2_autoplay_cb = _CheckBox()
    view._t2_player = _Player()
    view._t2_clips_table = _FakeTable(_make_rows(entries))
    # Production wires this in _build_tab2(); reconnect it here so
    # selectRow()-driven navigation (prev/next/autoplay-advance) loads clips
    # the same way it does in the real UI.
    view._t2_clips_table.itemSelectionChanged.connect(view._on_motif_clip_selected)
    return view


def _sample_entries(tmp_path):
    clip_a = tmp_path / "clip_a.mp4"
    clip_b = tmp_path / "clip_b.mp4"
    clip_a.write_text("fake", encoding="utf-8")
    clip_b.write_text("fake", encoding="utf-8")
    return [
        ("(1, 2)", "bigram", "clip_a.mp4", "rat1", "A", "1.2s", str(clip_a)),
        ("(3, 4)", "trigram", "clip_b.mp4", "rat2", "B", "2.4s", str(clip_b)),
    ]


def test_selected_clip_path_reads_user_role(tmp_path):
    entries = _sample_entries(tmp_path)
    view = _view(entries)

    view._t2_clips_table.selectRow(0)
    assert view._t2_selected_clip_path() == Path(entries[0][6])

    view._t2_clips_table.selectRow(1)
    assert view._t2_selected_clip_path() == Path(entries[1][6])


def test_prev_next_clip_move_current_row_and_load(tmp_path):
    entries = _sample_entries(tmp_path)
    view = _view(entries)
    view._t2_clips_table.selectRow(0)

    view._t2_next_clip()
    assert view._t2_clips_table.currentRow() == 1
    assert view._t2_player.loaded[-1] == entries[1][6]

    # Already at the last row: next is a no-op.
    view._t2_next_clip()
    assert view._t2_clips_table.currentRow() == 1

    view._t2_prev_clip()
    assert view._t2_clips_table.currentRow() == 0
    assert view._t2_player.loaded[-1] == entries[0][6]

    # Already at the first row: prev is a no-op.
    view._t2_prev_clip()
    assert view._t2_clips_table.currentRow() == 0


def test_update_clip_nav_buttons_clamped_at_ends(tmp_path):
    entries = _sample_entries(tmp_path)
    view = _view(entries)

    view._t2_clips_table.selectRow(0)
    view._t2_update_clip_nav_buttons()
    assert view._t2_prev_clip_btn.enabled is False
    assert view._t2_next_clip_btn.enabled is True

    view._t2_clips_table.selectRow(1)
    view._t2_update_clip_nav_buttons()
    assert view._t2_prev_clip_btn.enabled is True
    assert view._t2_next_clip_btn.enabled is False


def test_video_finished_wraps_around_when_autoplay_checked(tmp_path):
    entries = _sample_entries(tmp_path)
    view = _view(entries)
    view._t2_autoplay_cb.setChecked(True)
    view._t2_clips_table.selectRow(1)  # last row

    view._on_t2_video_finished()

    assert view._t2_clips_table.currentRow() == 0
    assert view._t2_player.loaded[-1] == entries[0][6]


def test_video_finished_does_nothing_when_autoplay_unchecked(tmp_path):
    entries = _sample_entries(tmp_path)
    view = _view(entries)
    view._t2_autoplay_cb.setChecked(False)
    view._t2_clips_table.selectRow(1)

    view._on_t2_video_finished()

    assert view._t2_clips_table.currentRow() == 1  # unchanged


def test_video_finished_replays_single_clip(tmp_path):
    entries = _sample_entries(tmp_path)[:1]
    view = _view(entries)
    view._t2_autoplay_cb.setChecked(True)
    view._t2_clips_table.selectRow(0)
    view._t2_player.loaded.clear()

    view._on_t2_video_finished()

    assert view._t2_clips_table.currentRow() == 0
    assert view._t2_player.loaded[-1] == entries[0][6]


def test_export_clip_builds_prefixed_destination_and_dedupes(tmp_path, monkeypatch):
    entries = _sample_entries(tmp_path)
    view = _view(entries)
    view._t2_clips_table.selectRow(0)

    export_dir = tmp_path / "results" / "exports"
    monkeypatch.setattr(analysis_mod, "RESULTS", tmp_path / "results")

    copied = []
    monkeypatch.setattr(analysis_mod.shutil, "copy2", lambda src, dst: copied.append((src, dst)))
    monkeypatch.setattr(
        analysis_mod.QMessageBox, "information",
        staticmethod(lambda *a, **k: None),
    )

    view._export_t2_clip()

    assert len(copied) == 1
    dest = copied[0][1]
    assert dest.parent == export_dir
    assert dest.name == "motif_1_2_bigram_clip_a.mp4"

    # A second export of the same clip should de-dupe with a numeric suffix.
    dest.write_text("existing", encoding="utf-8")
    view._export_t2_clip()
    assert copied[1][1].name == "motif_1_2_bigram_clip_a_1.mp4"


def test_export_clip_noop_when_nothing_loaded(monkeypatch):
    view = _view()
    info_calls = []
    monkeypatch.setattr(
        analysis_mod.QMessageBox, "information",
        staticmethod(lambda *a, **k: info_calls.append((a, k))),
    )
    copy_calls = []
    monkeypatch.setattr(analysis_mod.shutil, "copy2", lambda *a, **k: copy_calls.append(a))

    view._export_t2_clip()

    assert not copy_calls
    assert len(info_calls) == 1


def test_load_clip_handles_missing_file(tmp_path):
    missing = tmp_path / "gone.mp4"
    entries = [("(1, 2)", "bigram", "gone.mp4", "rat1", "A", "1.2s", str(missing))]
    view = _view(entries)

    view._t2_clips_table.selectRow(0)

    assert view._t2_current_clip_path is None
    assert "not found" in view._t2_clip_status.text()
    assert view._t2_export_clip_btn.enabled is False
