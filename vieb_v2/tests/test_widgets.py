import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_QAPP = None


def _app():
    global _QAPP
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    if _QAPP is None:
        _QAPP = QApplication.instance() or QApplication([])
    return _QAPP


# ------------------------------------------------------------------- theme

def test_theme_has_no_qt_dependency():
    # theme must be importable without a display, so tests and tooling can
    # read tokens without spinning up Qt.
    import subprocess
    result = subprocess.run(
        [sys.executable, "-c",
         "import sys; sys.path.insert(0, %r); "
         "from app import theme; print(theme.ACCENT); "
         "assert 'PyQt5' not in sys.modules" %
         os.path.join(os.path.dirname(__file__), "..")],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_status_palette_covers_every_stage_state():
    from app import theme
    from app.stage_row import STATES
    for state in STATES:
        assert state in theme.STATUS
        assert state in theme.STATUS_ICONS
        assert len(theme.STATUS[state]) == 3


def test_style_builders_reference_tokens_not_literals():
    # The point of the port: changing a token changes the rendered style.
    from app import theme
    assert theme.ACCENT in theme.primary_button_style()
    assert theme.TEXT in theme.heading_style()
    assert theme.CONTENT_BG in theme.card_style()


# -------------------------------------------------------------------- Card

def test_card_shows_and_updates_its_value():
    _app()
    from app.widgets import Card
    card = Card("States", "-")
    assert card.value() == "-"
    card.set(8)
    assert card.value() == "8"


# ---------------------------------------------------------------- Terminal

def test_terminal_appends_and_clears():
    _app()
    from app.widgets import Terminal
    term = Terminal()
    term.append_line("hello")
    term.append_line("world")
    assert "hello" in term.toPlainText() and "world" in term.toPlainText()
    term._clear()
    assert term.toPlainText() == ""


def test_terminal_clear_callback_fires():
    _app()
    from app.widgets import Terminal
    called = []
    term = Terminal(on_clear=lambda: called.append(1))
    term.append_line("x")
    term._clear()
    assert called == [1]


def test_terminal_overlay_repositions_on_resize():
    # The overlay is moved manually rather than laid out, so a resize must
    # keep it inside the widget or the buttons drift off-screen.
    _app()
    from app.widgets import Terminal
    term = Terminal()
    term.resize(600, 300)
    assert term._overlay.x() + term._overlay.width() <= 600
    assert term._overlay.y() + term._overlay.height() <= 300


# --------------------------------------------------------------- StageRow

def test_stage_row_cycles_all_states():
    _app()
    from app.stage_row import STATES, StageRow
    row = StageRow("align", 1, "Align", "desc")
    for state in STATES:
        row.set_state(state)
        assert row.state == state


def test_stage_row_rejects_unknown_state():
    _app()
    import pytest
    from app.stage_row import StageRow
    row = StageRow("align", 1, "Align", "desc")
    with pytest.raises(ValueError):
        row.set_state("finished")


def test_running_stage_cannot_be_started_again():
    _app()
    from app.stage_row import StageRow
    row = StageRow("align", 1, "Align", "desc")
    row.set_state("running")
    assert not row._run_button.isEnabled()
    row.set_state("done")
    assert row._run_button.isEnabled()


def test_stage_row_emits_run_request_with_its_key():
    _app()
    from app.stage_row import StageRow
    row = StageRow("cluster", 4, "Cluster", "desc")
    seen = []
    row.run_requested.connect(seen.append)
    row._run_button.click()
    assert seen == ["cluster"]


def test_stage_row_expands_and_collapses():
    _app()
    from app.stage_row import StageRow
    row = StageRow("align", 1, "Align", "desc")
    assert not row.is_expanded
    row._toggle()
    assert row.is_expanded
    row._toggle()
    assert not row.is_expanded


def test_stage_row_carries_no_backend_coupling():
    # v1's StageRow emitted run_diagnose / run_subcluster /
    # navigate_cluster_runs -- pipeline specifics baked into a reusable widget.
    # This one must expose only generic signals.
    from app.stage_row import StageRow
    signals = {n for n in dir(StageRow) if not n.startswith("_")}
    assert "run_requested" in signals and "toggled" in signals
    for leaked in ("run_diagnose", "run_subcluster", "navigate_cluster_runs",
                   "mark_completed", "run_from_here"):
        assert leaked not in signals
