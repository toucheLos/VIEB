from __future__ import annotations
import json
from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox, QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QSpinBox,
    QVBoxLayout, QWidget,
)

from _utils import ROOT, RESULTS, _DEFAULT_CFG, _save_cfg, _load_cfg


class SettingsView(QWidget):
    settings_changed = pyqtSignal(dict)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        t = QLabel("Settings")
        t.setFont(QFont("Arial", 18, QFont.Bold))
        lay.addWidget(t)
        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        r = 0

        def _help_btn(title, body):
            """Small (?) button that opens an info dialog."""
            b = QPushButton("?")
            b.setFixedSize(20, 20)
            b.setFlat(True)
            b.setStyleSheet(
                "QPushButton{border:1px solid #aaa;border-radius:10px;color:#555;"
                "background:#f5f5f5;font-size:10px;font-weight:bold;}"
                "QPushButton:hover{background:#e8f0fe;color:#1a73e8;border-color:#1a73e8;}"
            )
            b.setToolTip(body)
            b.clicked.connect(lambda: QMessageBox.information(None, title, body))
            return b

        def row(label_text, widget, tooltip=""):
            nonlocal r
            lbl = QLabel(label_text)
            if tooltip:
                lbl.setToolTip(tooltip)
                widget.setToolTip(tooltip)
            form.addWidget(lbl, r, 0)
            if tooltip:
                hw = QHBoxLayout()
                hw.setContentsMargins(0, 0, 0, 0)
                hw.setSpacing(4)
                hw.addWidget(widget)
                hw.addWidget(_help_btn(label_text, tooltip))
                form.addLayout(hw, r, 1)
            else:
                form.addWidget(widget, r, 1)
            r += 1

        def dir_row(label_text, key, tooltip=""):
            nonlocal r
            le = QLineEdit(self.cfg.get(key, ""))
            if tooltip:
                le.setToolTip(tooltip)
            browse = QPushButton("Browse...")
            browse.clicked.connect(lambda: self._browse(le))
            h = QHBoxLayout()
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            h.addWidget(le)
            h.addWidget(browse)
            if tooltip:
                h.addWidget(_help_btn(label_text, tooltip))
            lbl = QLabel(label_text)
            if tooltip:
                lbl.setToolTip(tooltip)
            form.addWidget(lbl, r, 0)
            form.addLayout(h, r, 1)
            r += 1
            return le

        ab = cfg.get("arena_bounds", _DEFAULT_CFG["arena_bounds"])
        self._xmin = QSpinBox(); self._xmin.setRange(0, 9999); self._xmin.setValue(ab["x_min"])
        self._ymin = QSpinBox(); self._ymin.setRange(0, 9999); self._ymin.setValue(ab["y_min"])
        self._xmax = QSpinBox(); self._xmax.setRange(0, 9999); self._xmax.setValue(ab["x_max"])
        self._ymax = QSpinBox(); self._ymax.setRange(0, 9999); self._ymax.setValue(ab["y_max"])

        _arena_tip = (
            "Pixel coordinates of the arena boundary in the raw video frame.\n"
            "Used to compute distance-to-wall features in feature extraction.\n"
            "Set to the full frame size (e.g. 0–1280, 0–960) if unsure."
        )
        row("Arena x_min", self._xmin, _arena_tip)
        row("Arena y_min", self._ymin, _arena_tip)
        row("Arena x_max", self._xmax, _arena_tip)
        row("Arena y_max", self._ymax, _arena_tip)

        self._results = dir_row(
            "Results directory", "results_dir",
            "Where all pipeline output files are saved: feature arrays, cluster models,\n"
            "comparison plots, and characterization CSVs.\n"
            "Default: results/ inside the VIEB project folder."
        )
        self._raw = dir_row(
            "Raw videos directory", "raw_videos_dir",
            "Folder containing your .mp4 video files and their DLC pose CSV files.\n"
            "DLC CSVs must be in the same folder as the corresponding .mp4.\n"
            "Default: raw_videos/ inside the VIEB project folder."
        )

        self._ctx_groups = QLineEdit(str(self.cfg.get("context_groups", "A,B,C")))
        row(
            "Context groups (comma-separated)", self._ctx_groups,
            "Labels for the experimental contexts in your metadata.csv 'context' column.\n"
            "Example: 'A,B,C' for three contexts (A=conditioned, B=test, C=novel).\n"
            "Must exactly match the values in the context column of metadata.csv."
        )

        self._fps = QSpinBox()
        self._fps.setRange(1, 240)
        self._fps.setValue(int(self.cfg.get("fps", 30)))
        row(
            "FPS", self._fps,
            "Frame rate of your videos in frames per second.\n"
            "Used to convert frame counts to seconds in all reports and bout durations.\n"
            "Typical values: 25 (PAL), 30 (NTSC), 60 (high-speed)."
        )

        lay.addLayout(form)

        save = QPushButton("Save Settings")
        save.clicked.connect(self._save)
        lay.addWidget(save)
        lay.addStretch()

    def _browse(self, le):
        d = QFileDialog.getExistingDirectory(self, "Select Directory", le.text())
        if d:
            le.setText(d)

    def _save(self):
        self.cfg["arena_bounds"] = {
            "x_min": self._xmin.value(),
            "y_min": self._ymin.value(),
            "x_max": self._xmax.value(),
            "y_max": self._ymax.value(),
        }
        self.cfg["results_dir"] = self._results.text()
        self.cfg["raw_videos_dir"] = self._raw.text()
        self.cfg["context_groups"] = self._ctx_groups.text().strip() or "A,B,C"
        self.cfg["fps"] = self._fps.value()
        _save_cfg(self.cfg)
        self.settings_changed.emit(self.cfg)
        QMessageBox.information(self, "Settings", "Saved.")
