"""Top-level pages for the VIEB v2 shell.

`PlaceholderPage` holds the shared body for pages whose content is not built
yet. It is styled like a real page -- heading, summary of what it will hold,
and a stated reason it is empty -- rather than a bare "coming soon" label, so
an unfinished page still looks intentional beside the finished ones.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from app import theme


class PlaceholderPage(QWidget):
    """A page whose content is not implemented yet."""

    TITLE = ""
    SUMMARY = ""
    BLOCKED_ON = ""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(theme.page_background())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.PAGE_MARGIN, 24,
                                  theme.PAGE_MARGIN, 24)
        layout.setSpacing(10)

        heading = QLabel(self.TITLE)
        heading.setStyleSheet(theme.heading_style())
        layout.addWidget(heading)

        if self.SUMMARY:
            summary = QLabel(self.SUMMARY)
            summary.setWordWrap(True)
            summary.setStyleSheet(
                theme.label_style(theme.TEXT_MUTED, theme.FONT_SIZE_BODY))
            layout.addWidget(summary)

        layout.addStretch()

        pending = QLabel(self.BLOCKED_ON or "Not implemented yet.")
        pending.setAlignment(Qt.AlignCenter)
        pending.setWordWrap(True)
        pending.setStyleSheet(
            f"font-family:{theme.FONT_FAMILY};font-size:13px;"
            f"color:{theme.TEXT_FAINT};background:{theme.SUBTLE_BG};"
            f"border:1px dashed {theme.BORDER};"
            f"border-radius:{theme.RADIUS}px;padding:28px;")
        layout.addWidget(pending)
        layout.addStretch()
