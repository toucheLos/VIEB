"""Cluster Runs page -- history of pipeline runs.

Shows which latent space produced each run, so PCA and diffusion results are
distinguishable at a glance rather than by remembering what was configured.
Reads the shared registry, so runs launched from the CLI appear here too.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app import theme

COLUMNS = ["Run", "When", "Latent", "Source", "States", "Noise",
           "Largest state", "Entropy"]


class ClusterRunsPage(QWidget):
    TITLE = "Cluster Runs"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{theme.CONTENT_BG};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(14)

        heading = QLabel("Cluster runs")
        heading.setStyleSheet(
            f"font-family:{theme.FONT_FAMILY};font-size:18px;font-weight:600;"
            f"color:{theme.TEXT};background:transparent;")
        layout.addWidget(heading)

        row = QHBoxLayout()
        self.out_edit = QLineEdit("results/v2")
        refresh = QPushButton("Refresh")
        refresh.setCursor(Qt.PointingHandCursor)
        refresh.clicked.connect(self.refresh)
        row.addWidget(QLabel("Output directory"))
        row.addWidget(self.out_edit, stretch=1)
        row.addWidget(refresh)
        layout.addLayout(row)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        layout.addWidget(self.table, stretch=1)

        self.empty_label = QLabel(
            "No runs yet. Configure one on the Analysis page, or run\n"
            "    python -m vieb_v2.cli run --pose <dir> --latent-method pca")
        self.empty_label.setStyleSheet(
            f"color:{theme.TEXT_FAINT};background:transparent;"
            f"font-family:{theme.MONO_FAMILY};font-size:12px;")
        layout.addWidget(self.empty_label)

        self.refresh()

    def refresh(self):
        from representation import run_registry

        runs = run_registry.load(self.out_edit.text().strip())
        rows = run_registry.summarise(runs)
        rows.reverse()                       # newest first

        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, value in enumerate([
                row["run_id"],
                row["timestamp"].replace("T", " "),
                row["latent_method"],
                row["source"],
                _fmt(row["n_states"]),
                _pct(row["noise_frac"]),
                _pct(row["largest_state_frac"]),
                _fmt(row["state_entropy"], 3),
            ]):
                self.table.setItem(r, c, QTableWidgetItem(str(value)))

        self.table.setVisible(bool(rows))
        self.empty_label.setVisible(not rows)
        return len(rows)


def _fmt(value, places=None):
    if value is None:
        return "-"
    if places is not None and isinstance(value, float):
        return f"{value:.{places}f}"
    return value


def _pct(value):
    return "-" if value is None else f"{value * 100:.1f}%"
