"""Analysis page -- configure and launch a clustering run.

The latent-space choice lives here because it is the decision this page exists
to support: PCA is exact but linear, diffusion maps are nonlinear with a defined
distance. Runs execute on a worker thread so the window stays responsive, and
every run is written to the registry the Cluster Runs page reads.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app import theme


class AnalysisPage(QWidget):
    TITLE = "Analysis"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{theme.CONTENT_BG};")
        self._worker = None
        self.run_recorded = None      # MainWindow sets this to refresh history

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(14)

        layout.addWidget(self._heading())
        layout.addWidget(self._paths_box())
        layout.addWidget(self._latent_box())
        layout.addWidget(self._pipeline_box())
        layout.addLayout(self._run_row())
        layout.addWidget(self._results_label())
        layout.addStretch()

        self._on_method_changed()

    # ------------------------------------------------------------- widgets

    def _heading(self):
        label = QLabel("Clustering run")
        label.setStyleSheet(
            f"font-family:{theme.FONT_FAMILY};font-size:18px;font-weight:600;"
            f"color:{theme.TEXT};background:transparent;")
        return label

    def _paths_box(self):
        box = QGroupBox("Data")
        form = QFormLayout(box)
        self.pose_edit = QLineEdit("results/pose")
        self.out_edit = QLineEdit("results/v2")

        browse = QPushButton("Browse...")
        browse.setCursor(Qt.PointingHandCursor)
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
        outer = QVBoxLayout(box)

        self.pca_radio = QRadioButton("PCA -- linear, exact distances")
        self.diffusion_radio = QRadioButton(
            "Diffusion maps -- nonlinear; distance is random-walk connectivity")
        self.pca_radio.setChecked(True)

        self._method_group = QButtonGroup(self)
        self._method_group.addButton(self.pca_radio)
        self._method_group.addButton(self.diffusion_radio)
        self._method_group.buttonToggled.connect(self._on_method_changed)

        outer.addWidget(self.pca_radio)
        outer.addWidget(self.diffusion_radio)

        self.pca_params = QWidget()
        pca_form = QFormLayout(self.pca_params)
        pca_form.setContentsMargins(24, 4, 0, 0)
        self.var_threshold = QDoubleSpinBox()
        self.var_threshold.setRange(0.5, 1.0)
        self.var_threshold.setSingleStep(0.01)
        self.var_threshold.setValue(0.95)
        pca_form.addRow("Variance retained", self.var_threshold)
        outer.addWidget(self.pca_params)

        self.diffusion_params = QWidget()
        dm_form = QFormLayout(self.diffusion_params)
        dm_form.setContentsMargins(24, 4, 0, 0)
        self.n_components = QSpinBox()
        self.n_components.setRange(2, 64)
        self.n_components.setValue(8)
        self.alpha = QDoubleSpinBox()
        self.alpha.setRange(0.0, 1.0)
        self.alpha.setSingleStep(0.1)
        self.alpha.setValue(1.0)
        self.alpha.setToolTip(
            "1.0 (Laplace-Beltrami) removes most of the sampling-density "
            "effect. Below 1, densely sampled - slow - regions get compressed "
            "in the embedding.")
        self.diffusion_time = QSpinBox()
        self.diffusion_time.setRange(1, 32)
        self.diffusion_time.setValue(1)
        self.n_landmarks = QSpinBox()
        self.n_landmarks.setRange(100, 50_000)
        self.n_landmarks.setSingleStep(500)
        self.n_landmarks.setValue(3000)
        dm_form.addRow("Components", self.n_components)
        dm_form.addRow("Alpha", self.alpha)
        dm_form.addRow("Diffusion time", self.diffusion_time)
        dm_form.addRow("Landmarks", self.n_landmarks)
        outer.addWidget(self.diffusion_params)
        return box

    def _pipeline_box(self):
        box = QGroupBox("Embedding and clustering")
        form = QFormLayout(box)
        self.n_lags = QSpinBox()
        self.n_lags.setRange(0, 32)
        self.n_lags.setValue(4)
        self.lag_stride = QSpinBox()
        self.lag_stride.setRange(1, 32)
        self.lag_stride.setValue(2)
        self.min_cluster_size = QSpinBox()
        self.min_cluster_size.setRange(2, 100_000)
        self.min_cluster_size.setValue(50)
        form.addRow("Lags", self.n_lags)
        form.addRow("Lag stride", self.lag_stride)
        form.addRow("Min cluster size", self.min_cluster_size)
        return box

    def _run_row(self):
        row = QHBoxLayout()
        self.run_button = QPushButton("Run")
        self.run_button.setCursor(Qt.PointingHandCursor)
        self.run_button.setFixedWidth(120)
        self.run_button.setStyleSheet(
            f"QPushButton{{background:{theme.ACCENT};color:#FFFFFF;"
            f"font-weight:600;border:none;border-radius:4px;padding:8px 16px;}}"
            f"QPushButton:disabled{{background:{theme.TEXT_FAINT};}}")
        self.run_button.clicked.connect(self.start_run)

        self.status_label = QLabel("idle")
        self.status_label.setStyleSheet(
            f"color:{theme.TEXT_MUTED};background:transparent;"
            f"font-family:{theme.MONO_FAMILY};font-size:12px;")

        row.addWidget(self.run_button)
        row.addWidget(self.status_label, stretch=1)
        return row

    def _results_label(self):
        self.results_label = QLabel("")
        self.results_label.setWordWrap(True)
        self.results_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.results_label.setStyleSheet(
            f"color:{theme.TEXT};background:transparent;"
            f"font-family:{theme.MONO_FAMILY};font-size:12px;"
            f"border-top:1px solid {theme.BORDER};padding-top:12px;")
        return self.results_label

    # ------------------------------------------------------------ behaviour

    def _on_method_changed(self, *_):
        diffusion = self.diffusion_radio.isChecked()
        self.pca_params.setVisible(not diffusion)
        self.diffusion_params.setVisible(diffusion)

    def _browse_pose(self):
        chosen = QFileDialog.getExistingDirectory(self, "Select pose directory")
        if chosen:
            self.pose_edit.setText(chosen)

    def latent_method(self):
        return "diffusion" if self.diffusion_radio.isChecked() else "pca"

    def options(self):
        """Exactly the keyword arguments `pipeline.run` accepts."""
        return {
            "latent_method": self.latent_method(),
            "var_threshold": self.var_threshold.value(),
            "n_components": self.n_components.value(),
            "alpha": self.alpha.value(),
            "diffusion_time": self.diffusion_time.value(),
            "n_landmarks": self.n_landmarks.value(),
            "n_lags": self.n_lags.value(),
            "lag_stride": self.lag_stride.value(),
            "min_cluster_size": self.min_cluster_size.value(),
        }

    def start_run(self):
        if self._worker is not None and self._worker.isRunning():
            return

        from app.worker import PipelineWorker

        self.run_button.setEnabled(False)
        self.results_label.setText("")
        self.status_label.setText("starting...")

        self._worker = PipelineWorker(
            self.pose_edit.text().strip(), self.out_edit.text().strip(),
            self.options(), parent=self)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished_ok.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.start()

    def _on_progress(self, stage):
        self.status_label.setText(stage)

    def _on_finished(self, result):
        self.run_button.setEnabled(True)
        self.status_label.setText("done")

        metrics = result.get("metrics") or {}
        clean = metrics.get("clustered_only", {})
        report = result.get("report", {})
        lines = [
            f"latent          {report.get('latent_method')}",
            f"components      {report.get('latent', {}).get('n_components')}",
            f"n_states        {metrics.get('n_states')}",
            f"noise_frac      {metrics.get('noise_frac', 0):.4f}",
            f"largest_state   {clean.get('largest_state_frac', 0):.4f}",
            f"state_entropy   {clean.get('state_entropy', 0):.4f}",
        ]
        ratio = (result.get("speed_diagnostics") or {}).get("noise_speed_ratio")
        if ratio:
            lines.append(f"noise/clustered speed  {ratio:.2f}")
        self.results_label.setText("\n".join(lines))

        if callable(self.run_recorded):
            self.run_recorded()

    def _on_failed(self, message):
        self.run_button.setEnabled(True)
        self.status_label.setText("failed")
        # A worker-thread exception would otherwise vanish and leave the Run
        # button dead; surface it where it can be read and copied.
        self.results_label.setText(message)
