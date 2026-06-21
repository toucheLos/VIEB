from __future__ import annotations
import glob
import json
import os
import sys
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QRadioButton, QScrollArea, QTextEdit, QVBoxLayout, QWidget,
)

from _utils import ROOT, RESULTS, _has_pose_csvs, _probe_wsl_cuml
from _workers import SubprocessWorker
from views.dlc_setup import _StepCard, _PRIMARY_BTN_STYLE, _ClickableLabel, _translate_log


class AddVideosView(QWidget):
    """Guided flow for adding new videos to a project that already has a
    trained DLC model and a fitted shared cluster model.

    Walks the user through exactly four steps, each scoped to *new* videos
    only: run pose estimation, extract features, cluster (apply existing
    model or refit), then proceed to the rest of the pipeline.
    """

    navigate_dlc = pyqtSignal()
    navigate_pipeline = pyqtSignal()
    worker_running = pyqtSignal(bool)
    pipeline_done = pyqtSignal()

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self._worker = None
        self._action_buttons: list[QPushButton] = []
        self._active_step_num: int | None = None
        self._step_results: dict[int, bool] = {}
        self._steps: dict[int, _StepCard] = {}
        self._build()
        self.refresh()

    # ── Layout ────────────────────────────────────────────────────────────

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        scroll.setWidget(content)
        root.addWidget(scroll)

        outer = QVBoxLayout(content)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(12)

        title = QLabel("Add Videos")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        outer.addWidget(title)

        subtitle = QLabel(
            "Run the analysis pipeline on videos you've added since your last run. "
            "Each step below only processes the new videos — existing results "
            "are left alone."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color:#555;")
        outer.addWidget(subtitle)

        self._gpu_badge = QLabel("⏳ Checking GPU…")
        self._gpu_badge.setStyleSheet(
            "background:#f5f5f5;border:1px solid #ddd;border-radius:4px;"
            "padding:4px 10px;color:#555;font-size:12px;"
        )
        outer.addWidget(self._gpu_badge)
        QTimer.singleShot(800, self._probe_gpu_async)

        self._banner_frame = QFrame()
        self._banner_frame.setObjectName("statusBanner")
        banner_lay = QHBoxLayout(self._banner_frame)
        banner_lay.setContentsMargins(14, 10, 14, 10)
        self._banner_label = QLabel("")
        self._banner_label.setWordWrap(True)
        self._banner_label.setTextFormat(Qt.RichText)
        self._banner_label.setStyleSheet("background:transparent;border:none;")
        banner_lay.addWidget(self._banner_label, stretch=1)
        outer.addWidget(self._banner_frame)

        self._steps_container = QWidget()
        self._steps_lay = QVBoxLayout(self._steps_container)
        self._steps_lay.setContentsMargins(0, 0, 0, 0)
        self._steps_lay.setSpacing(8)
        outer.addWidget(self._steps_container)

        # Log (collapsed by default)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(180)
        self._log.setStyleSheet(
            "background:#151515;color:#cfd8dc;font-family:Consolas;font-size:11px;"
        )
        self._log.hide()

        log_hdr_row = QHBoxLayout()
        self._log_header = _ClickableLabel("Show log  ▾")
        self._log_header.setStyleSheet("color:#555;font-size:11px;padding:4px 0;")
        self._log_header.setCursor(Qt.PointingHandCursor)
        self._log_header.clicked.connect(self._toggle_log)
        log_hdr_row.addWidget(self._log_header)
        log_hdr_row.addStretch()
        copy_btn = QPushButton("Copy")
        copy_btn.setFlat(True)
        copy_btn.clicked.connect(
            lambda: QApplication.clipboard().setText(self._log.toPlainText())
        )
        log_hdr_row.addWidget(copy_btn)
        clear_log_btn = QPushButton("Clear")
        clear_log_btn.setFlat(True)
        clear_log_btn.clicked.connect(self._log.clear)
        log_hdr_row.addWidget(clear_log_btn)
        outer.addLayout(log_hdr_row)
        outer.addWidget(self._log)

        outer.addStretch()

    def _toggle_log(self):
        visible = not self._log.isVisible()
        self._log.setVisible(visible)
        self._log_header.setText("Hide log  ▴" if visible else "Show log  ▾")

    # ── GPU detection ────────────────────────────────────────────────────

    def _probe_gpu_async(self):
        class _ProbeThread(QThread):
            result = pyqtSignal(bool)
            def run(self):
                self.result.emit(_probe_wsl_cuml())

        self._gpu_thread = _ProbeThread(self)
        self._gpu_thread.result.connect(self._on_gpu_probe)
        self._gpu_thread.start()

    def _on_gpu_probe(self, ok: bool):
        if ok:
            self._gpu_badge.setText("✓ GPU (WSL2 + cuML) — pose estimation and clustering use your GPU")
            self._gpu_badge.setStyleSheet(
                "background:#e8f5e9;border:1px solid #a5d6a7;border-radius:4px;"
                "padding:4px 10px;color:#1b5e20;font-size:12px;"
            )
        else:
            self._gpu_badge.setText("CPU mode — pose estimation and clustering run on CPU")
            self._gpu_badge.setStyleSheet(
                "background:#fff8e1;border:1px solid #ffe082;border-radius:4px;"
                "padding:4px 10px;color:#795548;font-size:12px;"
            )

    # ── Status helpers ───────────────────────────────────────────────────

    def _raw_dir(self) -> Path:
        return Path(self.cfg.get("raw_videos_dir", str(ROOT / "raw_videos")))

    def _count_total_videos(self) -> int:
        raw_dir = self._raw_dir()
        return len(list(raw_dir.glob("*.mp4"))) if raw_dir.exists() else 0

    def _count_pose_csvs(self) -> int:
        raw_dir = self._raw_dir()
        return len(list(raw_dir.glob("*DLC*.csv"))) if raw_dir.exists() else 0

    def _feature_index(self) -> dict:
        index_path = RESULTS / "features" / "index.json"
        if not index_path.exists():
            return {}
        try:
            with open(index_path) as f:
                return json.load(f)
        except Exception:
            return {}

    def _count_extracted(self) -> int:
        index = self._feature_index()
        return len([k for k in index if k != "_meta"])

    def _stems_without_labels(self) -> list[str]:
        index = self._feature_index()
        shared_dir = RESULTS / "shared"
        return [
            stem for stem in index if stem != "_meta"
            and not (shared_dir / f"{stem}_labels.npy").exists()
        ]

    def _has_shared_model(self) -> bool:
        shared_dir = RESULTS / "shared"
        return all(
            (shared_dir / f).exists()
            for f in ("preprocessor.pkl", "umap_reducer.pkl", "clusterer.pkl", "cluster_info.json")
        )

    def _has_trained_dlc(self) -> bool:
        try:
            import vieb_config
            project_path = vieb_config.get_dlc_project_path()
        except Exception:
            project_path = None
        if not project_path:
            return False
        return bool(list(Path(project_path).glob("dlc-models/**/train/snapshot-*.index")))

    # ── Step rebuilding ──────────────────────────────────────────────────

    def refresh(self):
        while self._steps_lay.count():
            item = self._steps_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        self._action_buttons = []
        self._steps = {}

        total_videos = self._count_total_videos()
        csv_count = self._count_pose_csvs()
        extracted = self._count_extracted()
        pending_cluster = self._stems_without_labels()

        new_for_pose = max(0, total_videos - csv_count)
        new_for_features = max(0, csv_count - extracted)
        new_for_cluster = len(pending_cluster)

        if new_for_pose == 0 and new_for_features == 0 and new_for_cluster == 0:
            self._banner_frame.setStyleSheet(
                "QFrame#statusBanner{background:#e8f5e9;border:1px solid #a5d6a7;border-radius:6px;}"
            )
            self._banner_label.setText(
                "✓  Everything is up to date — every video in raw_videos/ has been "
                "tracked, had features extracted, and been clustered.<br>"
                "Add new .mp4 files to raw_videos/ and revisit this page when you're ready to process them."
            )
        else:
            self._banner_frame.setStyleSheet(
                "QFrame#statusBanner{background:#e3f2fd;border:1px solid #90caf9;border-radius:6px;}"
            )
            self._banner_label.setText(
                f"{total_videos} video(s) in raw_videos/. "
                f"{new_for_pose} need pose estimation, "
                f"{new_for_features} need feature extraction, "
                f"{new_for_cluster} need clustering.<br>"
                "Work through the steps below in order."
            )

        if not self._has_trained_dlc() and not _has_pose_csvs(self._raw_dir()):
            note = _StepCard(
                1, "Set up DeepLabCut first",
                "Add Videos assumes you already have a trained DLC model (or existing pose "
                "CSV/H5 files). Go to DLC Setup to finish that one-time setup."
            )
            dlc_btn = QPushButton("Open DLC Setup")
            dlc_btn.setMinimumHeight(34)
            dlc_btn.clicked.connect(self.navigate_dlc.emit)
            note.body_layout().addWidget(dlc_btn, alignment=Qt.AlignLeft)
            note.set_status("current")
            self._steps_lay.addWidget(note)
            return

        # Step 1: Run pose estimation on new videos
        step1 = self._add_step(
            1, "Run Pose Estimation",
            "Run the trained DLC model on videos that don't have pose CSVs yet."
        )
        info1 = QLabel(f"{csv_count}/{max(total_videos, csv_count)} video(s) have pose data.")
        info1.setStyleSheet("color:#666;font-size:11px;")
        step1.body_layout().addWidget(info1)
        pose_btn = QPushButton("Run Pose Estimation")
        pose_btn.setMinimumHeight(34)
        pose_btn.setStyleSheet(_PRIMARY_BTN_STYLE)
        pose_btn.setToolTip(
            "Runs setup_dlc_training.py --analyze. DeepLabCut automatically skips "
            "videos that already have output."
        )
        pose_btn.clicked.connect(self._run_pose_estimation)
        step1.body_layout().addWidget(pose_btn, alignment=Qt.AlignLeft)
        self._action_buttons.append(pose_btn)
        if self._step_results.get(1) is False:
            step1.set_status("error")
        elif new_for_pose == 0:
            step1.set_status("done", expanded=False)
        else:
            step1.set_status("current")

        # Step 2: Feature extraction
        step2 = self._add_step(
            2, "Extract Features",
            "Extract behavioral features for the videos that now have pose data. "
            "Videos that were already extracted are skipped."
        )
        info2 = QLabel(f"{extracted}/{max(csv_count, extracted)} video(s) have features extracted.")
        info2.setStyleSheet("color:#666;font-size:11px;")
        step2.body_layout().addWidget(info2)
        extract_btn = QPushButton("Extract Features")
        extract_btn.setMinimumHeight(34)
        extract_btn.setStyleSheet(_PRIMARY_BTN_STYLE)
        extract_btn.setToolTip("Runs compare.py --extract.")
        extract_btn.clicked.connect(self._run_extract)
        step2.body_layout().addWidget(extract_btn, alignment=Qt.AlignLeft)
        self._action_buttons.append(extract_btn)
        if self._step_results.get(2) is False:
            step2.set_status("error")
        elif new_for_pose > 0:
            step2.set_status("pending", expanded=False)
        elif new_for_features == 0:
            step2.set_status("done", expanded=False)
        else:
            step2.set_status("current")

        # Step 3: Cluster
        step3 = self._add_step(
            3, "Cluster New Videos",
            "Assign behavioral states to the new videos."
        )
        if new_for_cluster:
            info3 = QLabel(
                f"{new_for_cluster} video(s) need state labels: " + ", ".join(pending_cluster)
            )
        else:
            info3 = QLabel("All extracted videos already have state labels.")
        info3.setWordWrap(True)
        info3.setStyleSheet("color:#666;font-size:11px;")
        step3.body_layout().addWidget(info3)

        has_model = self._has_shared_model()
        if has_model:
            self._apply_radio = QRadioButton(
                "Apply existing cluster model (fast — keeps existing state labels unchanged)"
            )
            self._refit_radio = QRadioButton(
                "Refit clustering on everything (slower — may discover new states, "
                "renumbers all videos)"
            )
            self._apply_radio.setChecked(True)
            step3.body_layout().addWidget(self._apply_radio)
            step3.body_layout().addWidget(self._refit_radio)
        else:
            self._apply_radio = None
            self._refit_radio = None
            note3 = QLabel(
                "No existing shared cluster model found yet — this will run a full fit."
            )
            note3.setStyleSheet("color:#888;font-size:11px;")
            step3.body_layout().addWidget(note3)

        cluster_btn = QPushButton("Run Clustering")
        cluster_btn.setMinimumHeight(34)
        cluster_btn.setStyleSheet(_PRIMARY_BTN_STYLE)
        cluster_btn.clicked.connect(self._run_cluster)
        step3.body_layout().addWidget(cluster_btn, alignment=Qt.AlignLeft)
        self._action_buttons.append(cluster_btn)
        if self._step_results.get(3) is False:
            step3.set_status("error")
        elif new_for_pose > 0 or new_for_features > 0:
            step3.set_status("pending", expanded=False)
        elif new_for_cluster == 0:
            step3.set_status("done", expanded=False)
        else:
            step3.set_status("current")

        # Step 4: Re-run reports
        step4 = self._add_step(
            4, "Update Reports & Quantification",
            "Once new videos are clustered, re-run the reporting and quantification "
            "stages so they include the new data."
        )
        pipeline_btn = QPushButton("Go to Pipeline →")
        pipeline_btn.setMinimumHeight(34)
        pipeline_btn.clicked.connect(self.navigate_pipeline.emit)
        step4.body_layout().addWidget(pipeline_btn, alignment=Qt.AlignLeft)
        if new_for_pose > 0 or new_for_features > 0 or new_for_cluster > 0:
            step4.set_status("pending", expanded=False)
        else:
            step4.set_status("current")

    def _add_step(self, number: int, title: str, description: str) -> _StepCard:
        card = _StepCard(number, title, description)
        self._steps_lay.addWidget(card)
        self._steps[number] = card
        return card

    # ── Actions ──────────────────────────────────────────────────────────

    def _run_pose_estimation(self):
        self._active_step_num = 1
        self._run_subprocess(["setup_dlc_training.py", "--analyze"], use_dlc_python=True)

    def _run_extract(self):
        self._active_step_num = 2
        fps = self.cfg.get("fps", 30)
        args = ["compare.py", "--extract", "--fps", str(fps)]
        if not self.cfg.get("use_wavelets", True):
            args.append("--no-wavelets")
        self._run_subprocess(args)

    def _run_cluster(self):
        self._active_step_num = 3
        fps = self.cfg.get("fps", 30)
        if self._apply_radio is not None and self._apply_radio.isChecked():
            args = ["compare.py", "--cluster", "--apply-existing", "--fps", str(fps)]
        else:
            args = [
                "compare.py", "--cluster", "--fps", str(fps),
                "--min-cluster-size", str(self.cfg.get("min_cluster_size", 50)),
                "--umap-dims", str(self.cfg.get("umap_dims", 10)),
            ]
            hdbscan_min_samples = self.cfg.get("hdbscan_min_samples", 0)
            if hdbscan_min_samples:
                args += ["--hdbscan-min-samples", str(hdbscan_min_samples)]
        self._run_subprocess(args)

    def _run_subprocess(self, args: list[str], use_dlc_python: bool = False):
        if self._worker and self._worker.isRunning():
            self._log_human("⚠ A task is already running. Wait for it to finish.")
            return
        if self._active_step_num is not None:
            self._step_results.pop(self._active_step_num, None)
        self._set_buttons_enabled(False)
        self.worker_running.emit(True)
        python_exe = (self.cfg.get("dlc_python") or sys.executable) if use_dlc_python else sys.executable
        self._worker = SubprocessWorker(args, python_exe=python_exe)
        self._worker.log.connect(self._on_raw_log)
        self._worker.done.connect(self._on_worker_done)
        self._worker.start()

    def _on_worker_done(self, ok: bool):
        self._set_buttons_enabled(True)
        self.worker_running.emit(False)
        if ok:
            self._log_human("✓ Task completed successfully.")
        else:
            self._log_human("✕ Task failed — check the log above for details.")
        if self._active_step_num is not None:
            self._step_results[self._active_step_num] = ok
        self._active_step_num = None
        self.refresh()
        self.pipeline_done.emit()

    def stop_worker(self):
        if self._worker and self._worker.isRunning():
            self._worker.stop()

    def _set_buttons_enabled(self, enabled: bool):
        for b in self._action_buttons:
            b.setEnabled(enabled)

    # ── Logging ──────────────────────────────────────────────────────────

    def _on_raw_log(self, text: str):
        human = _translate_log(text)
        if human is not None:
            self._log.insertPlainText(human + "\n")
        else:
            self._log.insertPlainText(text)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _log_human(self, msg: str):
        self._log.insertPlainText(msg + "\n")
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())
