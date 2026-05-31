from __future__ import annotations
import os
import sys
import subprocess
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QFrame, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QScrollArea, QTextEdit, QVBoxLayout, QWidget,
)

from _utils import (
    ROOT, RESULTS, STAGES, _has_pose_csvs, _open_folder, _save_cfg,
    wsl_cuml_available, wsl_cuml_reset_cache, _probe_wsl_cuml, _state_key, _MPL,
)
from _workers import PipelineRunner, SubprocessWorker
from _widgets import StageRow
from _dialogs import WslSetupDialog, DiagnoseDialog

if _MPL:
    from _utils import Figure, FigureCanvas

    def _state_key(stage_id):
        return str(stage_id)

    class WslSetupDialog:
        def __init__(self, parent=None):
            pass
        def exec_(self):
            pass

    class DiagnoseDialog:
        def __init__(self, parent=None):
            pass
        def show(self):
            pass
        def start(self):
            pass

from datetime import datetime


class RunPipelineView(QWidget):
    pipeline_done = pyqtSignal()
    worker_running = pyqtSignal(bool)
    navigate_dlc = pyqtSignal()

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._rows: dict[int, StageRow] = {}
        self._worker = None
        self._active_stages = set()
        self._build()
        # Probe WSL2 GPU in the background so the badge updates without blocking
        self._wsl_thread = None
        QTimer.singleShot(800, self._probe_gpu_async)

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(12)

        top = QHBoxLayout()
        t = QLabel("Pipeline")
        t.setFont(QFont("Arial", 18, QFont.Bold))
        top.addWidget(t)
        top.addStretch()
        self._run_full = QPushButton("Run Full Pipeline (Stages 2–11)")
        self._run_full.setToolTip(
            "Run all behavioral analysis stages in order.\n"
            "Prerequisite: complete DLC Setup (pose estimation) first."
        )
        self._run_full.clicked.connect(self.run_full_pipeline)
        top.addWidget(self._run_full)
        lay.addLayout(top)

        # GPU status badge (updated after WSL probe completes)
        gpu_row = QHBoxLayout()
        self._gpu_badge = QLabel("⏳ Checking GPU…")
        self._gpu_badge.setStyleSheet(
            "background:#f5f5f5;border:1px solid #ddd;border-radius:4px;"
            "padding:4px 10px;color:#555;font-size:12px;"
        )
        gpu_row.addWidget(self._gpu_badge)
        gpu_row.addStretch()
        self._gpu_setup_btn = QPushButton("Set up GPU acceleration")
        self._gpu_setup_btn.setFixedHeight(26)
        self._gpu_setup_btn.clicked.connect(self._open_wsl_setup)
        gpu_row.addWidget(self._gpu_setup_btn)
        lay.addLayout(gpu_row)

        self._status = QLabel("")
        self._status.setStyleSheet("color:#666;")
        lay.addWidget(self._status)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        v = QVBoxLayout(holder)
        for stage in STAGES:
            row = StageRow(stage, self.cfg)
            if stage["id"] == 1:
                row.run_stage.connect(lambda _: self.navigate_dlc.emit())
                row._run_btn.setText("Open DLC Setup ▶")
                row._from_btn.hide()
            else:
                row.run_stage.connect(self._run_stage)
                row.run_from_here.connect(self._run_from_here)
            row.mark_completed.connect(self._mark_completed)
            row.changed.connect(self._param_changed)
            if stage["id"] == 5:
                row.run_diagnose.connect(self._run_diagnose)
                row.run_subcluster.connect(self._run_subcluster)
            self._rows[stage["id"]] = row
            v.addWidget(row)
        v.addStretch()
        scroll.setWidget(holder)
        lay.addWidget(scroll)

        log_header = QHBoxLayout()
        log_header.addStretch()
        self._copy_log_btn = QPushButton("Copy Output")
        self._copy_log_btn.setFixedHeight(24)
        self._copy_log_btn.setToolTip("Copy all pipeline output to clipboard")
        self._copy_log_btn.clicked.connect(
            lambda: QApplication.clipboard().setText(self._global_log.toPlainText())
        )
        log_header.addWidget(self._copy_log_btn)
        lay.addLayout(log_header)

        self._global_log = QTextEdit()
        self._global_log.setReadOnly(True)
        self._global_log.setMaximumHeight(180)
        self._global_log.setStyleSheet("background:#151515;color:#cfd8dc;font-family:Consolas;")
        lay.addWidget(self._global_log)

    def update_stage0_status(self, dlc_path: str | None):
        """Update the Stage 1 (DLC) row status based on the current DLC project."""
        row = self._rows.get(1)
        if row is None:
            return
        if dlc_path:
            row.set_status("done")
        else:
            row.set_status("pending")

    def _probe_gpu_async(self):
        """Run WSL2 probe in a background QThread; update badge when done."""
        class _ProbeThread(QThread):
            result = pyqtSignal(bool)
            def run(self):
                self.result.emit(_probe_wsl_cuml())

        if self._wsl_thread and self._wsl_thread.isRunning():
            return
        self._wsl_thread = _ProbeThread(self)
        self._wsl_thread.result.connect(self._on_gpu_probe)
        self._wsl_thread.start()

    def _on_gpu_probe(self, ok: bool):
        global _WSL_CUML
        _WSL_CUML = ok
        self.refresh_gpu_badge()

    def refresh_gpu_badge(self):
        if _WSL_CUML is True:
            self._gpu_badge.setText("✓ GPU (WSL2 + cuML) — clustering uses your RTX GPU")
            self._gpu_badge.setStyleSheet(
                "background:#e8f5e9;border:1px solid #a5d6a7;border-radius:4px;"
                "padding:4px 10px;color:#1b5e20;font-size:12px;"
            )
            self._gpu_setup_btn.setText("GPU Setup")
        elif _WSL_CUML is False:
            self._gpu_badge.setText("CPU mode — clustering runs on CPU (WSL2 + cuML not found)")
            self._gpu_badge.setStyleSheet(
                "background:#fff8e1;border:1px solid #ffe082;border-radius:4px;"
                "padding:4px 10px;color:#795548;font-size:12px;"
            )
            self._gpu_setup_btn.setText("Set up GPU acceleration")
        else:
            self._gpu_badge.setText("⏳ Checking GPU…")

    def _open_wsl_setup(self):
        if sys.platform != "win32":
            return
        dlg = WslSetupDialog(self)
        dlg.exec_()
        # Re-probe after dialog closes in case user just finished setup
        wsl_cuml_reset_cache()
        self._probe_gpu_async()

    def _param_changed(self, key, value):
        self.cfg[key] = value

    def _mark_completed(self, sid, completed):
        key = _state_key(sid)
        self.cfg.setdefault("stage_status", {})[key] = "done" if completed else "pending"
        if completed:
            self.cfg.setdefault("stage_last_run", {})[key] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cfg["last_completed_stage"] = STAGES[sid - 1]["name"]
            self._rows[sid].set_status("done")
        else:
            self._rows[sid].set_status("pending")
        self._rows[sid].set_last_run(self.cfg.get("stage_last_run", {}).get(key))
        _save_cfg(self.cfg)

    def estimate_times(self, data):
        idx = data.get("feature_index") or {}
        ci = data.get("cluster_info") or {}
        n_videos = len(idx) if isinstance(idx, dict) else 0
        n_frames = sum(int(v.get("n_frames", 0)) for v in idx.values()) if isinstance(idx, dict) else 0
        for sid, row in self._rows.items():
            if sid == 1:
                mins = max(5, int(n_videos * 0.4))
            elif sid == 2:
                mins = max(2, int(n_frames / 120000))
            elif sid in (3, 4, 5, 6):
                mins = max(5, int(n_frames / 90000))
            elif sid == 7:
                mins = 2
            elif sid == 8:
                mins = 2
            elif sid == 9:
                mins = 1
            elif sid == 10:
                mins = 1
            else:
                mins = 3 + (5 if self.cfg.get("export_clips") else 0)
            row.set_eta(f"~{mins} min")

    def update_from_cfg(self, statuses: dict | None = None):
        ss = statuses if statuses is not None else self.cfg.get("stage_status", {})
        ts = self.cfg.get("stage_last_run", {})
        for sid, row in self._rows.items():
            row.set_status(ss.get(_state_key(sid), "pending"))
            row.set_last_run(ts.get(_state_key(sid)))

    def _append_log(self, line):
        self._global_log.insertPlainText(line)
        sb = self._global_log.verticalScrollBar()
        sb.setValue(sb.maximum())
        for sid in self._active_stages:
            self._rows[sid].append_log(line)

    def _set_buttons(self, enabled):
        self._run_full.setEnabled(enabled)
        for row in self._rows.values():
            row.set_enabled(enabled)

    def _start_worker(self, stage_ids):
        if self._worker and self._worker.isRunning():
            return
        self._worker = PipelineRunner(stage_ids, self.cfg)
        self._active_stages = set(stage_ids)
        self._worker.log.connect(self._append_log)
        self._worker.stage_started.connect(self._on_stage_started)
        self._worker.stage_done.connect(self._on_stage_done)
        self._worker.all_done.connect(self._on_all_done)
        self._set_buttons(False)
        self.worker_running.emit(True)
        self._worker.start()

    def _on_stage_started(self, sid):
        self._rows[sid].set_status("running")
        self._status.setText(f"Running stage {sid}: {STAGES[sid - 1]['name']}")

    def _on_stage_done(self, sid, ok):
        key = _state_key(sid)
        self.cfg.setdefault("stage_status", {})[key] = "done" if ok else "error"
        if ok:
            self.cfg.setdefault("stage_last_run", {})[key] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cfg["last_completed_stage"] = STAGES[sid - 1]["name"]
        self._rows[sid].set_status("done" if ok else "error")
        self._rows[sid].set_last_run(self.cfg["stage_last_run"].get(key))
        _save_cfg(self.cfg)

    def _on_all_done(self, ok):
        self._set_buttons(True)
        self.worker_running.emit(False)
        self._status.setText("Pipeline completed." if ok else "Pipeline failed.")
        if ok:
            self.pipeline_done.emit()

    def update_cluster_quality(self, data: dict):
        """Update the Stage 5 quality badge from loaded cluster/summary data."""
        stage5_row = self._rows.get(5)
        if stage5_row is None:
            return
        ci = data.get("cluster_info")
        summary = data.get("summary")
        if ci is None or summary is None:
            return
        n_clusters = int(ci.get("n_clusters", 0))
        if n_clusters == 0:
            return
        # Compute dominant state: state with highest mean fraction
        state_cols = [c for c in summary.columns if c.startswith("state_") and c.endswith("_frac")]
        if not state_cols:
            return
        means = summary[state_cols].mean()
        dom_col = means.idxmax()
        dom_state_id = int(dom_col.split("_")[1])
        dom_frac = float(means[dom_col])
        stage5_row.set_cluster_quality(dom_frac, dom_state_id)

    def _run_diagnose(self):
        """Launch diagnose_clusters.py in a background thread; show output in a dialog."""
        dlg = DiagnoseDialog(self)
        dlg.show()
        dlg.start()

    def _run_subcluster(self, dom_state_id: int):
        if dom_state_id < 0:
            return
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "Pipeline busy",
                                    "Wait for the current pipeline step to finish.")
            return
        reply = QMessageBox.question(
            self, "Fix dominant state",
            f"Split state {dom_state_id} into sub-states using\n"
            f"  compare.py --subcluster --state {dom_state_id}\n\n"
            f"This rewrites all label files and re-runs --report.\n"
            f"Continue?",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self._start_worker_cmd(
            [sys.executable, "compare.py", "--subcluster", "--state", str(dom_state_id)],
            label=f"Splitting state {dom_state_id}…",
        )

    def _start_worker_cmd(self, cmd: list, label: str):
        """Run an arbitrary subprocess command using a lightweight SubprocessWorker."""
        import subprocess

        class _CmdThread(QThread):
            log = pyqtSignal(str)
            done = pyqtSignal(bool)

            def __init__(self, cmd):
                super().__init__()
                self._cmd = cmd

            def run(self):
                try:
                    proc = subprocess.Popen(
                        self._cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace",
                        cwd=str(ROOT),
                    )
                    for line in proc.stdout:
                        self.log.emit(line)
                    proc.wait()
                    self.done.emit(proc.returncode == 0)
                except Exception as exc:
                    self.log.emit(f"[error] {exc}\n")
                    self.done.emit(False)

        self._status.setText(label)
        self._set_buttons(False)
        self.worker_running.emit(True)
        t = _CmdThread(cmd)
        t.log.connect(self._append_log)
        t.done.connect(lambda ok: (
            self._set_buttons(True),
            self.worker_running.emit(False),
            self._status.setText("Done." if ok else "Failed."),
            self.pipeline_done.emit() if ok else None,
        ))
        t.setParent(self)
        t.start()

    def _build_sequence(self, start_sid=1, from_here=False):
        all_ids = [s["id"] for s in STAGES if s["id"] >= start_sid] if from_here else [start_sid]
        if not from_here and start_sid in (4, 5, 6):
            all_ids = list(range(3, start_sid + 1))
        if not from_here and start_sid == 3:
            all_ids = [3]
        if from_here and start_sid in (4, 5, 6):
            all_ids = [3] + [s for s in range(7, 12)]

        if not self.cfg.get("enable_state_collapse", False):
            all_ids = [i for i in all_ids if i != 7]
        if not self.cfg.get("export_clips", False) and from_here:
            pass
        if _has_pose_csvs(Path(self.cfg.get("raw_videos_dir", str(ROOT / "raw_videos")))):
            if 1 in all_ids:
                all_ids.remove(1)
                self._rows[1].set_status("done")
        if from_here:
            done_ids = {
                int(k)
                for k, v in self.cfg.get("stage_status", {}).items()
                if str(v) == "done" and str(k).isdigit()
            }
            all_ids = [sid for sid in all_ids if sid not in done_ids]
        return all_ids

    def run_full_pipeline(self):
        self._start_worker(self._build_sequence(2, from_here=True))

    def _run_stage(self, sid):
        self._start_worker(self._build_sequence(sid, from_here=False))

    def _run_from_here(self, sid):
        self._start_worker(self._build_sequence(sid, from_here=True))
