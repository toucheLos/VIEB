"""Artifacts page -- files the pipeline has written.

Lists what is actually on disk in the output directory rather than what the
pipeline believes it produced, so a missing or truncated checkpoint is visible.
"""

from __future__ import annotations

import os
from datetime import datetime

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

COLUMNS = ["File", "Kind", "Size", "Modified"]

# What each artifact is, so the table explains itself.
KINDS = {
    "aligned.npz": "Aligned pose",
    "scores.npz": "Latent coordinates",
    "embedded.npz": "Delay embedding",
    "labels.npz": "Cluster labels",
    "runs.json": "Run registry",
    "latent_comparison.json": "PCA vs diffusion",
}


class ArtifactsPage(QWidget):
    TITLE = "Artifacts"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(theme.page_background())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.PAGE_MARGIN, 24,
                                  theme.PAGE_MARGIN, 24)
        layout.setSpacing(14)

        heading = QLabel("Artifacts")
        heading.setStyleSheet(theme.heading_style())
        layout.addWidget(heading)

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
        layout.addLayout(row)

        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table, stretch=1)

        self.empty_label = QLabel("")
        self.empty_label.setWordWrap(True)
        self.empty_label.setStyleSheet(
            theme.label_style(theme.TEXT_FAINT, mono=True))
        layout.addWidget(self.empty_label)

        self.refresh()

    def refresh(self):
        out_dir = self.out_edit.text().strip()
        entries = []
        if os.path.isdir(out_dir):
            for name in sorted(os.listdir(out_dir)):
                path = os.path.join(out_dir, name)
                if not os.path.isfile(path):
                    continue
                stat = os.stat(path)
                entries.append((
                    name,
                    KINDS.get(name, _kind_from_suffix(name)),
                    _size(stat.st_size),
                    datetime.fromtimestamp(stat.st_mtime).strftime(
                        "%Y-%m-%d %H:%M"),
                ))

        self.table.setRowCount(len(entries))
        for r, entry in enumerate(entries):
            for c, value in enumerate(entry):
                self.table.setItem(r, c, QTableWidgetItem(str(value)))

        self.table.setVisible(bool(entries))
        self.empty_label.setVisible(not entries)
        if not entries:
            self.empty_label.setText(
                f"Nothing in {out_dir!r} yet. Artifacts appear as each stage "
                f"completes -- see the Pipeline page.")
        return len(entries)


def _kind_from_suffix(name):
    return {".npz": "Checkpoint", ".json": "Report",
            ".csv": "Table", ".png": "Figure"}.get(
        os.path.splitext(name)[1], "File")


def _size(num_bytes):
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024:
            return f"{num_bytes:.0f} {unit}" if unit == "B" else \
                f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TB"
