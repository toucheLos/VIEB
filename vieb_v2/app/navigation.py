"""Sidebar navigation for the VIEB v2 shell."""

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app import theme

# Single source of truth for nav order and labels. The pages registered in
# MainWindow are keyed off these exact strings.
NAV_ITEMS = [
    "Overview",
    "States & Motifs",
    "Journeys",
    "Analysis",
    "Artifacts",
]

_NAV_BTN_QSS = f"""
    QPushButton {{
        text-align: left;
        padding: 0 18px;
        border: none;
        border-left: 3px solid transparent;
        background: transparent;
        font-size: 13px;
        color: {theme.TEXT_MUTED};
        font-family: {theme.FONT_FAMILY};
    }}
    QPushButton:hover {{
        background: rgba(0, 0, 0, 0.03);
        color: {theme.TEXT};
    }}
    QPushButton:checked {{
        border-left-color: {theme.ACCENT};
        background: rgba(78, 121, 167, 0.08);
        color: {theme.TEXT};
        font-weight: 600;
    }}
"""

_SIDEBAR_QSS = f"""
    QWidget#sidebar {{
        background: {theme.SIDEBAR_BG};
        border-right: 1px solid {theme.BORDER};
    }}
"""


class Navigation(QWidget):
    """Fixed-width sidebar listing the top-level pages.

    Emits `page_selected` with the nav item's label when the user clicks one.
    Selection state lives in an exclusive QButtonGroup, so highlighting the
    active item is handled by Qt rather than by hand.
    """

    page_selected = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(theme.SIDEBAR_WIDTH)
        self.setStyleSheet(_SIDEBAR_QSS)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 18, 0, 10)
        layout.setSpacing(0)

        layout.addLayout(self._build_brand_row())

        self.buttons = {}
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)

        for name in NAV_ITEMS:
            button = QPushButton(name)
            button.setCheckable(True)
            button.setFixedHeight(theme.NAV_ITEM_HEIGHT)
            button.setCursor(Qt.PointingHandCursor)
            button.setStyleSheet(_NAV_BTN_QSS)
            button.clicked.connect(lambda _checked, n=name: self.page_selected.emit(n))
            self._group.addButton(button)
            layout.addWidget(button)
            self.buttons[name] = button

        layout.addStretch()

    def _build_brand_row(self):
        row = QHBoxLayout()
        row.setContentsMargins(18, 0, 18, 20)

        logo = QLabel("VIEB")
        logo.setStyleSheet(
            f"font-family:{theme.MONO_FAMILY};font-size:16px;font-weight:600;"
            f"letter-spacing:2px;color:{theme.TEXT};"
            "background:transparent;border:none;"
        )
        version = QLabel("v2")
        version.setStyleSheet(
            f"font-family:{theme.MONO_FAMILY};font-size:10px;color:{theme.TEXT_FAINT};"
            "background:transparent;border:none;"
        )

        row.addWidget(logo)
        row.addWidget(version)
        row.addStretch()
        return row

    def set_active(self, name):
        """Highlight `name` without emitting `page_selected`."""
        button = self.buttons.get(name)
        if button is not None and not button.isChecked():
            button.setChecked(True)
