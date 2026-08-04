"""Design tokens for VIEB v2.

The palette, type and metrics come from v1's GUI so v2 reads as the same
application. The difference is that v1 repeats these values as hex literals in
thousands of places across ~25k lines, which makes a palette change a
find-and-replace across the codebase. Here they are defined once and every
widget composes from them.

Nothing in this module imports Qt, so it can be read by tests and tooling
without a display.
"""

from __future__ import annotations

# ── Surfaces ────────────────────────────────────────────────────────────────
SIDEBAR_BG = "#F4F4F4"
CONTENT_BG = "#FFFFFF"
SUBTLE_BG = "#FAFAFA"
BORDER = "#E5E5E5"
BORDER_STRONG = "#DADCE0"
HOVER_BG = "rgba(0, 0, 0, 0.03)"

# ── Text ────────────────────────────────────────────────────────────────────
TEXT = "#1A1A1A"
TEXT_MUTED = "#6B6B6B"
TEXT_FAINT = "#9B9B9B"
TEXT_SUBTLE = "#888888"

# ── Accent / interaction ────────────────────────────────────────────────────
ACCENT = "#4E79A7"
ACCENT_HOVER = "#3D6291"
ACCENT_WASH = "rgba(78, 121, 167, 0.08)"
LINK = "#1a73e8"
LINK_WASH = "#E8F0FE"
LINK_TEXT = "#0B57D0"

# ── Status palette ──────────────────────────────────────────────────────────
# (background, border, foreground) per stage state -- v1's StageRow._COLORS.
STATUS = {
    "done": ("#e8f5e9", "#a5d6a7", "#2e7d32"),
    "running": ("#e3f2fd", "#90caf9", "#1565c0"),
    "pending": ("#fafafa", "#e0e0e0", "#999999"),
    "error": ("#ffebee", "#ef9a9a", "#c62828"),
}
STATUS_ICONS = {"done": "✓", "running": "▶", "pending": "○", "error": "✕"}

# ── Terminal ────────────────────────────────────────────────────────────────
TERMINAL_BG = "#1E1E1E"
TERMINAL_FG = "#D4D4D4"

# ── Type ────────────────────────────────────────────────────────────────────
FONT_FAMILY = "'Segoe UI', Arial, sans-serif"
MONO_FAMILY = "'Consolas', 'IBM Plex Mono', monospace"

FONT_SIZE_SMALL = 10
FONT_SIZE_BODY = 12
FONT_SIZE_NAV = 13
FONT_SIZE_TITLE = 18
FONT_SIZE_METRIC = 22

# ── Metrics ─────────────────────────────────────────────────────────────────
SIDEBAR_WIDTH = 220
SIDEBAR_COLLAPSED_WIDTH = 56
NAV_ITEM_HEIGHT = 38
CARD_HEIGHT = 90
RADIUS = 6
RADIUS_SMALL = 4
PAGE_MARGIN = 28


# ── QSS builders ────────────────────────────────────────────────────────────
# Pages compose these instead of pasting style strings, so a token change
# propagates rather than needing every call site edited.

def page_background():
    return f"background:{CONTENT_BG};"


def heading_style(size=FONT_SIZE_TITLE):
    return (f"font-family:{FONT_FAMILY};font-size:{size}px;font-weight:600;"
            f"color:{TEXT};background:transparent;")


def label_style(color=TEXT_MUTED, size=FONT_SIZE_BODY, mono=False):
    family = MONO_FAMILY if mono else FONT_FAMILY
    return (f"font-family:{family};font-size:{size}px;color:{color};"
            f"background:transparent;border:none;")


def caption_style():
    """Small uppercase label -- v1's idiom for card titles and section eyebrows."""
    return (f"color:{TEXT_FAINT};font-size:{FONT_SIZE_SMALL}px;font-weight:600;"
            f"letter-spacing:1px;text-transform:uppercase;"
            f"background:transparent;border:none;")


def card_style():
    return (f"QFrame{{background:{CONTENT_BG};border:1px solid {BORDER};"
            f"border-radius:{RADIUS}px;}}")


def primary_button_style():
    return (f"QPushButton{{background:{ACCENT};color:#FFFFFF;font-weight:600;"
            f"border:none;border-radius:{RADIUS_SMALL}px;padding:8px 16px;}}"
            f"QPushButton:hover{{background:{ACCENT_HOVER};}}"
            f"QPushButton:disabled{{background:{TEXT_FAINT};}}")


def quiet_button_style():
    return (f"QPushButton{{background:transparent;border:1px solid {BORDER};"
            f"border-radius:{RADIUS_SMALL}px;color:{TEXT_MUTED};"
            f"padding:6px 12px;}}"
            f"QPushButton:hover{{color:{LINK};border-color:{LINK};"
            f"background:{LINK_WASH};}}")


def status_style(state):
    """Border and background for a stage row in the given state."""
    bg, border, _fg = STATUS.get(state, STATUS["pending"])
    return (f"QFrame#stageCard{{background:{bg};border:1px solid {border};"
            f"border-radius:{RADIUS}px;}}")


def status_icon_style(state):
    _bg, _border, fg = STATUS.get(state, STATUS["pending"])
    return (f"color:{fg};font-size:14px;font-weight:bold;"
            f"background:transparent;border:none;")


LIGHT_MENU_QSS = (
    f"QMenu#lightPopupMenu{{background:{CONTENT_BG};color:{TEXT};"
    f"border:1px solid {BORDER_STRONG};border-radius:{RADIUS_SMALL}px;"
    f"padding:4px;}}"
    f"QMenu#lightPopupMenu::item{{background:transparent;color:{TEXT};"
    f"padding:6px 28px 6px 12px;border-radius:3px;}}"
    f"QMenu#lightPopupMenu::item:selected{{background:{LINK_WASH};"
    f"color:{LINK_TEXT};}}"
    f"QMenu#lightPopupMenu::separator{{height:1px;background:{BORDER};"
    f"margin:4px 6px;}}"
)


def style_light_menu(menu):
    """Apply the light popup styling to a QMenu (v1's _style_light_menu)."""
    menu.setObjectName("lightPopupMenu")
    menu.setStyleSheet(LIGHT_MENU_QSS)
    return menu
