"""Collapsible pipeline stage row.

Ports v1's StageRow (user_interface.py:2050): a card whose header shows a
status glyph, the stage name, a timing hint and an expand arrow, with the
description and controls revealed underneath.

Improvement over v1: this widget knows nothing about the pipeline. v1's version
emits `run_diagnose`, `run_subcluster` and `navigate_cluster_runs` -- specific
pipeline actions baked into a reusable widget, which is why it cannot be reused
anywhere else. This one emits only `run_requested` and `toggled`; the page that
owns the rows decides what a stage means.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app import theme

STATES = ("pending", "running", "done", "error")


class _ClickableHeader(QFrame):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class StageRow(QFrame):
    """One stage of a pipeline, expandable to show detail and a Run button."""

    run_requested = pyqtSignal(str)      # stage key
    toggled = pyqtSignal(str, bool)      # stage key, expanded

    def __init__(self, key, index, name, description, parent=None):
        super().__init__(parent)
        self.key = key
        self.index = index
        self.name = name
        self._state = "pending"
        self._expanded = False

        self.setObjectName("stageCard")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_header())
        outer.addWidget(self._build_body(description))

        self.set_state("pending")
        self.set_expanded(False)

    # ------------------------------------------------------------- building

    def _build_header(self):
        header = _ClickableHeader()
        header.setCursor(Qt.PointingHandCursor)
        header.setStyleSheet("background:transparent;border:none;")
        header.clicked.connect(self._toggle)

        row = QHBoxLayout(header)
        row.setContentsMargins(14, 10, 14, 10)

        self._icon = QLabel(theme.STATUS_ICONS["pending"])
        self._icon.setFixedWidth(20)
        row.addWidget(self._icon)

        title = QLabel(f"{self.index}. {self.name}")
        title.setStyleSheet(
            f"font-weight:bold;color:{theme.TEXT};"
            f"background:transparent;border:none;")
        row.addWidget(title, stretch=1)

        self._detail = QLabel("")
        self._detail.setStyleSheet(
            theme.label_style(theme.TEXT_SUBTLE, theme.FONT_SIZE_BODY, mono=True))
        row.addWidget(self._detail)

        self._arrow = QToolButton()
        self._arrow.setArrowType(Qt.RightArrow)
        self._arrow.setCursor(Qt.PointingHandCursor)
        self._arrow.setFixedSize(18, 18)
        self._arrow.setStyleSheet(
            f"QToolButton{{color:{theme.TEXT_MUTED};background:transparent;"
            f"border:none;padding:0;margin:0;}}"
            f"QToolButton:hover{{background:{theme.LINK_WASH};border-radius:3px;}}")
        self._arrow.clicked.connect(self._toggle)
        row.addWidget(self._arrow)
        return header

    def _build_body(self, description):
        self._body = QWidget()
        layout = QVBoxLayout(self._body)
        layout.setContentsMargins(40, 0, 14, 10)
        layout.setSpacing(8)

        desc = QLabel(description)
        desc.setWordWrap(True)
        desc.setStyleSheet(
            theme.label_style(theme.TEXT_MUTED, theme.FONT_SIZE_BODY))
        layout.addWidget(desc)

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        self._run_button = QPushButton("Run stage")
        self._run_button.setCursor(Qt.PointingHandCursor)
        self._run_button.setStyleSheet(theme.quiet_button_style())
        self._run_button.clicked.connect(
            lambda: self.run_requested.emit(self.key))
        controls.addWidget(self._run_button)
        controls.addStretch()
        layout.addLayout(controls)
        return self._body

    # ------------------------------------------------------------ behaviour

    def _toggle(self):
        self.set_expanded(not self._expanded)

    def set_expanded(self, expanded):
        self._expanded = bool(expanded)
        self._body.setVisible(self._expanded)
        self._arrow.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.toggled.emit(self.key, self._expanded)

    @property
    def is_expanded(self):
        return self._expanded

    @property
    def state(self):
        return self._state

    def set_state(self, state, detail=""):
        """Set status, restyling the card. Unknown states fall back to pending."""
        if state not in STATES:
            raise ValueError(f"state must be one of {STATES}; got {state!r}")
        self._state = state
        self.setStyleSheet(theme.status_style(state))
        self._icon.setText(theme.STATUS_ICONS[state])
        self._icon.setStyleSheet(theme.status_icon_style(state))
        self._detail.setText(detail)
        # A running stage should not be startable again from the same row.
        self._run_button.setEnabled(state != "running")

    def set_detail(self, text):
        self._detail.setText(str(text))
