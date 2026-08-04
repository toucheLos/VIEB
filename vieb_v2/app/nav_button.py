"""Sidebar navigation button.

Ports v1's NavBtn (user_interface.py:4903): left-aligned label with a leading
glyph, and a 3px accent bar down the left edge when active. The collapsed mode
drops to icon-only at 56px with the label moved into a tooltip.

Improvement over v1: the two style strings are built from `theme` tokens rather
than duplicated hex, and the collapsed state keeps the label reachable by
keyboard (accessibleName) instead of only as a hover tooltip.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QPushButton

from app import theme

# Glyphs carried over from v1's _NAV_ICONS (user_interface.py:628), extended
# for the pages v2 adds.
NAV_ICONS = {
    "Overview": "⊞",
    "Pipeline": "▶",
    "States & Motifs": "▣",
    "Journeys": "⤳",
    "Cluster Runs": "⊙",
    "Analysis": "◈",
    "Artifacts": "◪",
    "Settings": "≡",
    "Help": "?",
}

_BASE = """
    QPushButton {{
        text-align: {align};
        padding: {padding};
        border: none;
        border-left: 3px solid transparent;
        background: transparent;
        font-size: {size}px;
        color: {muted};
        font-family: {family};
    }}
    QPushButton:hover {{
        background: {hover};
        color: {text};
    }}
    QPushButton:checked {{
        border-left-color: {accent};
        background: {wash};
        color: {text};
        font-weight: 600;
    }}
    QPushButton:focus {{
        outline: none;
        border-left-color: {accent};
    }}
"""


def _style(collapsed):
    return _BASE.format(
        align="center" if collapsed else "left",
        padding="0" if collapsed else "0 18px",
        size=15 if collapsed else theme.FONT_SIZE_NAV,
        muted=theme.TEXT_MUTED,
        family=theme.FONT_FAMILY,
        hover=theme.HOVER_BG,
        text=theme.TEXT,
        accent=theme.ACCENT,
        wash=theme.ACCENT_WASH,
    )


class NavBtn(QPushButton):
    """A checkable sidebar entry that can collapse to icon-only."""

    def __init__(self, label, parent=None):
        self._label = label
        self._icon = NAV_ICONS.get(label, "·")
        super().__init__(f"  {self._icon}   {label}", parent)

        self.setCheckable(True)
        self.setFixedHeight(theme.NAV_ITEM_HEIGHT)
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(_style(collapsed=False))
        # Keeps the destination announced to assistive tech even once the
        # visible text is reduced to a glyph.
        self.setAccessibleName(label)
        self._collapsed = False

    @property
    def label(self):
        return self._label

    @property
    def is_collapsed(self):
        return self._collapsed

    def set_collapsed(self, collapsed):
        self._collapsed = bool(collapsed)
        if collapsed:
            self.setText(self._icon)
            self.setToolTip(self._label)
        else:
            self.setText(f"  {self._icon}   {self._label}")
            self.setToolTip("")
        self.setStyleSheet(_style(collapsed))
