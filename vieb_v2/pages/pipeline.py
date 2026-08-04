"""Pipeline page -- v2's four stages, with live output.

Uses v1's StageRow idiom, but for the stages v2 actually has:
align -> latent -> delay embed -> cluster. v1's nine stages (feature
extraction, UMAP, collapse, report, quantification, motifs, clips...) describe
a different pipeline and are not translated here.

Stage status is read from the checkpoints the CLI already writes into the
output directory, so a run started from the terminal shows up here too.
"""

from __future__ import annotations

import os

from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from app import theme
from app.stage_row import StageRow
from app.widgets import SectionTitle, Terminal

# key -> (display name, checkpoint file, description)
STAGES = [
    ("align", "Align",
     "aligned.npz",
     "Load DLC pose, drop noisy keypoints (tail_tip), and remove translation "
     "and rotation by weighted Procrustes so the same posture maps to the "
     "same coordinates wherever it happened in the arena."),
    ("latent", "Latent space",
     "scores.npz",
     "Reduce aligned pose with PCA (linear, exact distances) or diffusion "
     "maps (nonlinear, distance is random-walk connectivity). Fitted once "
     "across every recording, never per recording."),
    ("embed", "Delay embedding",
     "embedded.npz",
     "Stack each frame with its preceding lags so a clustered point is a "
     "short trajectory rather than an instant. Lags never cross a recording "
     "boundary."),
    ("cluster", "Cluster",
     "labels.npz",
     "HDBSCAN over the delay-embedded coordinates. The -1 noise label is "
     "kept, never force-assigned."),
]


class PipelinePage(QWidget):
    TITLE = "Pipeline"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(theme.page_background())
        self.out_dir = "results/v2"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.PAGE_MARGIN, 24,
                                  theme.PAGE_MARGIN, 24)
        layout.setSpacing(12)

        heading = QLabel("Pipeline")
        heading.setStyleSheet(theme.heading_style())
        layout.addWidget(heading)

        subtitle = QLabel(
            "Each stage writes a checkpoint, so a later stage can be re-run "
            "without repeating the earlier ones.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(theme.label_style())
        layout.addWidget(subtitle)

        self.rows = {}
        for index, (key, name, _file, description) in enumerate(STAGES, start=1):
            row = StageRow(key, index, name, description)
            row.run_requested.connect(self._on_run_requested)
            layout.addWidget(row)
            self.rows[key] = row

        layout.addWidget(SectionTitle("Output"))
        self.terminal = Terminal()
        self.terminal.setMinimumHeight(160)
        layout.addWidget(self.terminal, stretch=1)

        layout.addLayout(self._build_controls())
        self.refresh()

    def _build_controls(self):
        row = QHBoxLayout()
        refresh = QPushButton("Refresh status")
        refresh.setStyleSheet(theme.quiet_button_style())
        refresh.clicked.connect(self.refresh)
        row.addWidget(refresh)
        row.addStretch()

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(
            theme.label_style(theme.TEXT_FAINT, mono=True))
        row.addWidget(self.status_label)
        return row

    # ------------------------------------------------------------ behaviour

    def set_out_dir(self, path):
        self.out_dir = path
        self.refresh()

    def refresh(self):
        """Mark stages done when their checkpoint exists on disk.

        Deliberately filesystem-driven rather than tracking in-process state:
        the CLI writes the same checkpoints, so a run done in a terminal or a
        batch job is reflected here without the GUI having observed it.
        """
        done = 0
        for key, _name, filename, _desc in STAGES:
            path = os.path.join(self.out_dir, filename)
            if os.path.exists(path):
                size = os.path.getsize(path) / 1e6
                self.rows[key].set_state("done", f"{size:.1f} MB")
                done += 1
            else:
                self.rows[key].set_state("pending", "")
        self.status_label.setText(f"{done}/{len(STAGES)} stages complete")
        return done

    def mark_running(self, key, detail=""):
        if key in self.rows:
            self.rows[key].set_state("running", detail)

    def mark_error(self, key, detail=""):
        if key in self.rows:
            self.rows[key].set_state("error", detail)

    def log(self, text):
        self.terminal.append_line(text)

    def _on_run_requested(self, key):
        # Wiring individual stages to the worker is the next pass; for now the
        # page reports what it would run rather than pretending to run it.
        name = next(n for k, n, _f, _d in STAGES if k == key)
        self.log(f"$ python -m vieb_v2.cli {key} --out {self.out_dir}")
        self.log(f"  ({name} is not yet runnable from this page -- "
                 f"use the Analysis page or the CLI)")
