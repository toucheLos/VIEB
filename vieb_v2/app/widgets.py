"""Shared widgets ported from v1's GUI.

  Card      v1's _Card (user_interface.py:1487) -- a stat tile
  Terminal  v1's _TerminalWidget (:1976) -- dark log with overlaid Copy/Clear
  SectionTitle / scroll_content
            v1's _section_title / _scroll_content_widget (views/analysis.py:134)

All styling comes from `theme`, so these are the only place the visual idiom is
defined rather than being restated at each call site as v1 does.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from app import theme


class Card(QFrame):
    """A titled metric tile: small uppercase caption over a large mono value."""

    def __init__(self, title, value="-", parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(theme.card_style())
        self.setFixedHeight(theme.CARD_HEIGHT)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)

        caption = QLabel(title)
        caption.setStyleSheet(theme.caption_style())
        layout.addWidget(caption)

        self._value = QLabel(str(value))
        self._value.setFont(QFont("Consolas", theme.FONT_SIZE_METRIC, QFont.Bold))
        self._value.setStyleSheet(
            f"color:{theme.TEXT};border:none;background:transparent;")
        layout.addWidget(self._value)

    def set(self, value):
        self._value.setText(str(value))

    def value(self):
        return self._value.text()


class SectionTitle(QLabel):
    """Bold heading used to separate blocks within a page."""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFont(QFont("Arial", 11, QFont.Bold))
        self.setStyleSheet(
            f"color:{theme.TEXT};padding-top:8px;padding-bottom:2px;"
            f"background:transparent;border:none;")


class Terminal(QTextEdit):
    """Read-only dark log with Copy/Clear buttons overlaid bottom-right.

    The overlay is repositioned in resizeEvent rather than laid out, which is
    how v1 does it -- a layout would reserve space and shrink the text area.
    """

    def __init__(self, on_clear=None, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self._on_clear = on_clear
        self.setStyleSheet(
            f"QTextEdit{{background:{theme.TERMINAL_BG};color:{theme.TERMINAL_FG};"
            f"font-family:{theme.MONO_FAMILY};font-size:{theme.FONT_SIZE_BODY}px;"
            f"border:1px solid {theme.BORDER};border-radius:{theme.RADIUS_SMALL}px;}}")

        self._overlay = QWidget(self)
        row = QHBoxLayout(self._overlay)
        row.setContentsMargins(4, 4, 4, 4)
        row.setSpacing(4)
        row.addWidget(self._overlay_button("Copy", self._copy))
        row.addWidget(self._overlay_button("Clear", self._clear))
        self._overlay.adjustSize()
        self._overlay.raise_()

    def _overlay_button(self, text, slot):
        button = QPushButton(text)
        button.setFixedHeight(20)
        button.setCursor(Qt.ArrowCursor)
        button.setStyleSheet(
            "QPushButton{background:rgba(45,45,45,210);color:#aaa;"
            "border:1px solid #555;border-radius:3px;font-size:10px;"
            "padding:1px 7px;}"
            "QPushButton:hover{background:rgba(80,80,80,230);color:#eee;}")
        button.clicked.connect(slot)
        return button

    def _copy(self):
        QApplication.clipboard().setText(self.toPlainText())

    def _clear(self):
        self.clear()
        if self._on_clear:
            self._on_clear()

    def append_line(self, text):
        self.append(str(text))
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._overlay.adjustSize()
        self._overlay.move(
            max(0, self.width() - self._overlay.width() - 2),
            max(0, self.height() - self._overlay.height() - 2),
        )


def scroll_content():
    """Return (scroll_area, layout) for a vertically scrolling page body.

    Pages with variable-height content need this; without it a long page
    silently clips instead of scrolling.
    """
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setStyleSheet("background:transparent;")

    content = QWidget()
    content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
    layout = QVBoxLayout(content)
    layout.setSpacing(16)
    area.setWidget(content)
    return area, layout


def card_row(*cards):
    """Lay cards out horizontally with even spacing."""
    holder = QWidget()
    row = QHBoxLayout(holder)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(12)
    for card in cards:
        row.addWidget(card)
    return holder


def separator():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet(
        f"color:{theme.BORDER};background:{theme.BORDER};border:none;"
        f"max-height:1px;")
    return line
