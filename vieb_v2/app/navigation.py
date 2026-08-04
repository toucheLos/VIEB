"""Sidebar navigation.

Ports v1's sidebar structure (user_interface.py:5230): brand row with a
collapse toggle, an output-directory switcher, a WORKSPACE section label, the
nav buttons, a separator and a footer status line.

NAV_ITEMS is the union of v1's information architecture and v2's own pages --
v1's Overview/Pipeline/Cluster Runs/Analysis/Artifacts/Settings/Help, plus
States & Motifs (v2's rename of v1's State Characterization, and the intended
home for the video player) and Journeys.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app import theme
from app.nav_button import NavBtn

# Single source of truth for nav order and labels. MainWindow keys its pages
# off these exact strings.
NAV_ITEMS = [
    "Overview",
    "Pipeline",
    "States & Motifs",
    "Journeys",
    "Cluster Runs",
    "Analysis",
    "Artifacts",
    "Settings",
    "Help",
]

_SIDEBAR_QSS = f"""
    QWidget#sidebar {{
        background: {theme.SIDEBAR_BG};
        border-right: 1px solid {theme.BORDER};
    }}
"""


class Navigation(QWidget):
    """Fixed-width sidebar listing the top-level pages.

    Emits `page_selected` with the nav item's label. Selection lives in the
    buttons' checked state managed here rather than in a QButtonGroup, because
    collapsing restyles every button and the group added nothing beyond
    exclusivity we already enforce in `set_active`.
    """

    page_selected = pyqtSignal(str)
    output_clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("sidebar")
        self.setFixedWidth(theme.SIDEBAR_WIDTH)
        self.setStyleSheet(_SIDEBAR_QSS)

        self._collapsed = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 18, 0, 10)
        layout.setSpacing(0)

        layout.addLayout(self._build_brand_row())
        layout.addWidget(self._build_output_row())
        layout.addWidget(self._build_section_label())

        self.buttons = {}
        for name in NAV_ITEMS:
            button = NavBtn(name)
            button.clicked.connect(
                lambda _checked, n=name: self._on_clicked(n))
            layout.addWidget(button)
            self.buttons[name] = button

        layout.addStretch()
        layout.addWidget(self._build_separator())
        layout.addWidget(self._build_footer())

    # ------------------------------------------------------------- building

    def _build_brand_row(self):
        row = QHBoxLayout()
        row.setContentsMargins(18, 0, 12, 20)

        self._logo = QLabel("VIEB")
        self._logo.setStyleSheet(
            f"font-family:{theme.MONO_FAMILY};font-size:16px;font-weight:600;"
            f"letter-spacing:2px;color:{theme.TEXT};"
            f"background:transparent;border:none;")
        self._version = QLabel("v2")
        self._version.setStyleSheet(
            f"font-family:{theme.MONO_FAMILY};font-size:10px;"
            f"color:{theme.TEXT_FAINT};background:transparent;border:none;")

        row.addWidget(self._logo)
        row.addWidget(self._version)
        row.addStretch()

        self._collapse_button = QPushButton("«")
        self._collapse_button.setFixedSize(22, 22)
        self._collapse_button.setCursor(Qt.PointingHandCursor)
        self._collapse_button.setToolTip("Collapse sidebar")
        self._collapse_button.setStyleSheet(
            f"QPushButton{{background:transparent;color:{theme.TEXT_FAINT};"
            f"border:none;font-size:13px;}}"
            f"QPushButton:hover{{color:{theme.TEXT};background:{theme.HOVER_BG};"
            f"border-radius:4px;}}")
        self._collapse_button.clicked.connect(self.toggle_collapsed)
        row.addWidget(self._collapse_button)
        return row

    def _build_output_row(self):
        self._output_holder = QWidget()
        row = QHBoxLayout(self._output_holder)
        row.setContentsMargins(18, 0, 18, 12)

        self._output_button = QPushButton("—  ▼")
        self._output_button.setCursor(Qt.PointingHandCursor)
        self._output_button.setToolTip("Output directory for this session")
        self._output_button.setStyleSheet(
            f"QPushButton{{background:{theme.ACCENT};color:#FFFFFF;"
            f"font-weight:bold;font-size:11px;border:none;"
            f"border-radius:{theme.RADIUS_SMALL}px;padding:6px 12px;"
            f"text-align:left;}}"
            f"QPushButton:hover{{background:{theme.ACCENT_HOVER};}}")
        self._output_button.clicked.connect(self.output_clicked.emit)
        row.addWidget(self._output_button, stretch=1)
        return self._output_holder

    def _build_section_label(self):
        self._section_label = QLabel("WORKSPACE")
        self._section_label.setStyleSheet(
            f"font-size:10px;font-weight:600;letter-spacing:2px;"
            f"color:{theme.TEXT_FAINT};padding:6px 18px;"
            f"background:transparent;border:none;")
        return self._section_label

    def _build_separator(self):
        self._separator = QFrame()
        self._separator.setFrameShape(QFrame.HLine)
        self._separator.setStyleSheet(
            f"color:{theme.BORDER};background:{theme.BORDER};border:none;"
            f"max-height:1px;")
        return self._separator

    def _build_footer(self):
        self._footer = QLabel("No run yet")
        self._footer.setWordWrap(True)
        self._footer.setStyleSheet(
            f"font-family:{theme.MONO_FAMILY};font-size:10px;"
            f"color:{theme.TEXT_FAINT};padding:10px 18px;"
            f"background:transparent;border:none;")
        return self._footer

    # ------------------------------------------------------------ behaviour

    def _on_clicked(self, name):
        self.set_active(name)
        self.page_selected.emit(name)

    def set_active(self, name):
        """Highlight `name` exclusively, without emitting page_selected."""
        if name not in self.buttons:
            return
        for label, button in self.buttons.items():
            button.setChecked(label == name)

    def active(self):
        for label, button in self.buttons.items():
            if button.isChecked():
                return label
        return None

    @property
    def is_collapsed(self):
        return self._collapsed

    def toggle_collapsed(self):
        self.set_collapsed(not self._collapsed)

    def set_collapsed(self, collapsed):
        """Collapse to icon-only. The active page is unaffected."""
        self._collapsed = bool(collapsed)
        self.setFixedWidth(theme.SIDEBAR_COLLAPSED_WIDTH if collapsed
                           else theme.SIDEBAR_WIDTH)

        for button in self.buttons.values():
            button.set_collapsed(collapsed)

        for widget in (self._logo, self._version, self._output_holder,
                       self._section_label, self._separator, self._footer):
            widget.setVisible(not collapsed)

        self._collapse_button.setText("»" if collapsed else "«")
        self._collapse_button.setToolTip(
            "Expand sidebar" if collapsed else "Collapse sidebar")

    def set_output_label(self, text):
        import os

        self._output_button.setText(f"{os.path.basename(text) or text}  ▼")
        self._output_button.setToolTip(text)

    def set_footer(self, text):
        self._footer.setText(str(text))
