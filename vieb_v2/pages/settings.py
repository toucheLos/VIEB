"""Settings page -- defaults the other pages start from.

Persisted to JSON next to the results so a project's configuration travels with
it. The Analysis page reads these as its initial values, which means the
pose/output directories and parameters only have to be typed once.
"""

from __future__ import annotations

import json
import os

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app import theme

CONFIG_NAME = "vieb2_settings.json"

DEFAULTS = {
    "pose_dir": "results/pose",
    "out_dir": "results/v2",
    "latent_method": "pca",
    "var_threshold": 0.95,
    "n_components": 8,
    "alpha": 1.0,
    "n_lags": 4,
    "lag_stride": 2,
    "min_cluster_size": 50,
}


def config_path():
    return os.path.join(os.path.expanduser("~"), ".config", CONFIG_NAME)


def load_settings():
    """Read saved settings, falling back to defaults for anything missing."""
    path = config_path()
    values = dict(DEFAULTS)
    if os.path.exists(path):
        try:
            with open(path) as fh:
                stored = json.load(fh)
            if isinstance(stored, dict):
                # Merge rather than replace, so a config written by an older
                # version doesn't drop keys added since.
                values.update({k: v for k, v in stored.items() if k in DEFAULTS})
        except (json.JSONDecodeError, OSError):
            pass
    return values


def save_settings(values):
    path = config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(values, fh, indent=2)
    os.replace(tmp, path)
    return path


class SettingsPage(QWidget):
    TITLE = "Settings"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(theme.page_background())

        layout = QVBoxLayout(self)
        layout.setContentsMargins(theme.PAGE_MARGIN, 24,
                                  theme.PAGE_MARGIN, 24)
        layout.setSpacing(14)

        heading = QLabel("Settings")
        heading.setStyleSheet(theme.heading_style())
        layout.addWidget(heading)

        subtitle = QLabel("Defaults used when a run is configured on the "
                          "Analysis page.")
        subtitle.setStyleSheet(theme.label_style())
        layout.addWidget(subtitle)

        layout.addWidget(self._paths_box())
        layout.addWidget(self._latent_box())
        layout.addWidget(self._cluster_box())
        layout.addLayout(self._buttons())
        layout.addStretch()

        self.apply_values(load_settings())

    # --------------------------------------------------------------- groups

    def _paths_box(self):
        box = QGroupBox("Paths")
        form = QFormLayout(box)
        self.pose_edit = QLineEdit()
        self.out_edit = QLineEdit()

        browse = QPushButton("Browse...")
        browse.setCursor(Qt.PointingHandCursor)
        browse.setStyleSheet(theme.quiet_button_style())
        browse.clicked.connect(self._browse_pose)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(self.pose_edit, stretch=1)
        row.addWidget(browse)
        holder = QWidget()
        holder.setLayout(row)

        form.addRow("Pose directory", holder)
        form.addRow("Output directory", self.out_edit)
        return box

    def _latent_box(self):
        box = QGroupBox("Latent space")
        form = QFormLayout(box)
        self.var_threshold = QDoubleSpinBox()
        self.var_threshold.setRange(0.5, 1.0)
        self.var_threshold.setSingleStep(0.01)
        self.n_components = QSpinBox()
        self.n_components.setRange(2, 64)
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.0, 1.0)
        self.alpha.setSingleStep(0.1)
        self.alpha.setToolTip(
            "Diffusion maps only. 1.0 (Laplace-Beltrami) removes most of the "
            "sampling-density effect; below 1, densely sampled - slow - "
            "regions get compressed in the embedding.")
        form.addRow("PCA variance retained", self.var_threshold)
        form.addRow("Diffusion components", self.n_components)
        form.addRow("Diffusion alpha", self.alpha)
        return box

    def _cluster_box(self):
        box = QGroupBox("Embedding and clustering")
        form = QFormLayout(box)
        self.n_lags = QSpinBox()
        self.n_lags.setRange(0, 32)
        self.lag_stride = QSpinBox()
        self.lag_stride.setRange(1, 32)
        self.min_cluster_size = QSpinBox()
        self.min_cluster_size.setRange(2, 100_000)
        form.addRow("Lags", self.n_lags)
        form.addRow("Lag stride", self.lag_stride)
        form.addRow("Min cluster size", self.min_cluster_size)
        return box

    def _buttons(self):
        row = QHBoxLayout()
        save = QPushButton("Save")
        save.setFixedWidth(120)
        save.setCursor(Qt.PointingHandCursor)
        save.setStyleSheet(theme.primary_button_style())
        save.clicked.connect(self.save)

        reset = QPushButton("Reset to defaults")
        reset.setCursor(Qt.PointingHandCursor)
        reset.setStyleSheet(theme.quiet_button_style())
        reset.clicked.connect(lambda: self.apply_values(DEFAULTS))

        self.status_label = QLabel("")
        self.status_label.setStyleSheet(
            theme.label_style(theme.TEXT_FAINT, mono=True))

        row.addWidget(save)
        row.addWidget(reset)
        row.addWidget(self.status_label, stretch=1)
        return row

    # ------------------------------------------------------------ behaviour

    def _browse_pose(self):
        chosen = QFileDialog.getExistingDirectory(self, "Select pose directory")
        if chosen:
            self.pose_edit.setText(chosen)

    def values(self):
        return {
            "pose_dir": self.pose_edit.text().strip(),
            "out_dir": self.out_edit.text().strip(),
            "latent_method": DEFAULTS["latent_method"],
            "var_threshold": self.var_threshold.value(),
            "n_components": self.n_components.value(),
            "alpha": self.alpha.value(),
            "n_lags": self.n_lags.value(),
            "lag_stride": self.lag_stride.value(),
            "min_cluster_size": self.min_cluster_size.value(),
        }

    def apply_values(self, values):
        self.pose_edit.setText(values["pose_dir"])
        self.out_edit.setText(values["out_dir"])
        self.var_threshold.setValue(values["var_threshold"])
        self.n_components.setValue(values["n_components"])
        self.alpha.setValue(values["alpha"])
        self.n_lags.setValue(values["n_lags"])
        self.lag_stride.setValue(values["lag_stride"])
        self.min_cluster_size.setValue(values["min_cluster_size"])

    def save(self):
        path = save_settings(self.values())
        self.status_label.setText(f"saved to {path}")
        return path
