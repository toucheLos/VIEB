"""Top-level pages for the VIEB v2 shell.

Every page is currently a placeholder. `PlaceholderPage` holds the shared body
so each page module only has to declare its title.
"""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget

from app import theme


class PlaceholderPage(QWidget):
    """An empty page showing its own name, centered."""

    TITLE = ""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{theme.CONTENT_BG};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        label = QLabel(f"{self.TITLE} — coming soon")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet(
            f"font-family:{theme.FONT_FAMILY};font-size:15px;"
            f"color:{theme.TEXT_FAINT};background:transparent;border:none;"
        )
        layout.addWidget(label)
