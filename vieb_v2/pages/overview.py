"""Overview page -- the most recent run at a glance.

Reads the shared run registry, so a run launched from the CLI or a batch job
shows up here without the GUI having observed it.
"""

from __future__ import annotations

from PyQt5.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app import theme
from app.widgets import Card, SectionTitle, card_row


class OverviewPage(QWidget):
    TITLE = "Overview"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(theme.page_background())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.PAGE_MARGIN, 24,
                                  theme.PAGE_MARGIN, 24)
        layout.setSpacing(14)

        heading = QLabel("Overview")
        heading.setStyleSheet(theme.heading_style())
        layout.addWidget(heading)

        layout.addLayout(self._build_source_row())

        layout.addWidget(SectionTitle("Latest run"))
        self.cards = {
            "latent": Card("Latent space", "-"),
            "states": Card("States", "-"),
            "noise": Card("Noise", "-"),
            "entropy": Card("Entropy", "-"),
        }
        layout.addWidget(card_row(*self.cards.values()))

        self.detail_label = QLabel("")
        self.detail_label.setWordWrap(True)
        self.detail_label.setStyleSheet(
            theme.label_style(theme.TEXT_MUTED, mono=True))
        layout.addWidget(self.detail_label)

        layout.addStretch()
        self.refresh()

    def _build_source_row(self):
        row = QHBoxLayout()
        label = QLabel("Output directory")
        label.setStyleSheet(theme.label_style())
        self.out_edit = QLineEdit("results/v2")
        refresh = QPushButton("Refresh")
        refresh.setStyleSheet(theme.quiet_button_style())
        refresh.clicked.connect(self.refresh)
        row.addWidget(label)
        row.addWidget(self.out_edit, stretch=1)
        row.addWidget(refresh)
        return row

    def refresh(self):
        from representation import run_registry

        runs = run_registry.load(self.out_edit.text().strip())
        if not runs:
            for card in self.cards.values():
                card.set("-")
            self.detail_label.setText(
                "No runs recorded yet. Configure one on the Analysis page, or "
                "run:  python -m vieb_v2.cli run --pose <dir>")
            return 0

        rows = run_registry.summarise(runs)
        latest = rows[-1]
        self.cards["latent"].set(latest["latent_method"] or "-")
        self.cards["states"].set(_fmt(latest["n_states"]))
        self.cards["noise"].set(_pct(latest["noise_frac"]))
        self.cards["entropy"].set(_num(latest["state_entropy"]))

        self.detail_label.setText(
            f"run {latest['run_id']} · {latest['timestamp'].replace('T', ' ')} "
            f"· started from {latest['source']} · {len(runs)} run(s) recorded")
        return len(runs)


def _fmt(value):
    return "-" if value is None else str(value)


def _pct(value):
    return "-" if value is None else f"{value * 100:.1f}%"


def _num(value):
    return "-" if value is None else f"{value:.3f}"
