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
    ROOT, RESULTS, STAGES, APP_CONFIG_PATH, _has_pose_csvs, _open_folder, _save_cfg,
    wsl_cuml_available, wsl_cuml_reset_cache, _probe_wsl_cuml, _state_key, _MPL,
)
from _workers import PipelineRunner, SubprocessWorker
from _widgets import StageRow

_STAGE_BY_ID = {s["id"]: s for s in STAGES}
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
    navigate_help = pyqtSignal(str)

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
        self._run_full = QPushButton("Run Full Pipeline")
        self._run_full.setToolTip(
            "Run all behavioral analysis stages in order."
        )
        self._run_full.clicked.connect(self.run_full_pipeline)
        top.addWidget(self._run_full)
        lay.addLayout(top)

        # GPU badge — created here, inserted into the scroll after Stage 0's row
        self._gpu_badge = QLabel("⏳ Checking GPU…")
        self._gpu_badge.setStyleSheet(
            "background:#f5f5f5;border:1px solid #ddd;border-radius:4px;"
            "padding:4px 10px;color:#555;font-size:12px;"
        )
        self._gpu_setup_btn = QPushButton("Set up GPU acceleration")
        self._gpu_setup_btn.setFixedHeight(26)
        self._gpu_setup_btn.clicked.connect(self._open_env_setup)

        self._status = QLabel("")
        self._status.setStyleSheet("color:#666;")
        lay.addWidget(self._status)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        v = QVBoxLayout(holder)
        for stage in STAGES:
            if stage["id"] == 1:
                # Stage 1 (DLC pose estimation) is accessible via the
                # sidebar's DLC Setup view; don't render it here too.
                continue
            row = StageRow(stage, self.cfg)
            if stage["id"] == 0:
                row.run_stage.connect(lambda _: self._check_stage0_readiness())
                row._run_btn.setText("Check Project Readiness")
                row._from_btn.hide()
                if self._stage0_complete():
                    row.set_status("done")
            else:
                row.run_stage.connect(self._run_stage)
                row.run_from_here.connect(self._run_from_here)
            row.mark_completed.connect(self._mark_completed)
            row.changed.connect(self._param_changed)
            row.navigate_help.connect(self.navigate_help.emit)
            if stage["id"] == 5:
                row.run_diagnose.connect(self._run_diagnose)
                row.run_subcluster.connect(self._run_subcluster)
            self._rows[stage["id"]] = row
            v.addWidget(row)
            # GPU status badge lives under Stage 0, not in the top-level header
            if stage["id"] == 0:
                gpu_widget = QWidget()
                gpu_layout = QHBoxLayout(gpu_widget)
                gpu_layout.setContentsMargins(28, 0, 4, 4)
                gpu_layout.addWidget(self._gpu_badge)
                gpu_layout.addStretch()
                gpu_layout.addWidget(self._gpu_setup_btn)
                v.addWidget(gpu_widget)
        # ── Clustering Diagnostics panel ──
        self._diag_frame = QFrame()
        self._diag_frame.setFrameShape(QFrame.StyledPanel)
        self._diag_frame.setStyleSheet(
            "QFrame { background: #FAFAFA; border: 1px solid #E0E0E0; border-radius: 6px; }"
        )
        df_lay = QVBoxLayout(self._diag_frame)
        df_lay.setContentsMargins(16, 12, 16, 12)
        df_lay.setSpacing(8)

        diag_hdr = QHBoxLayout()
        diag_title = QLabel("Clustering Diagnostics")
        diag_title.setFont(QFont("Arial", 12, QFont.Bold))
        diag_hdr.addWidget(diag_title)
        diag_hdr.addStretch()
        self._diag_regen_btn = QPushButton("Regenerate")
        self._diag_regen_btn.setFixedHeight(26)
        self._diag_regen_btn.clicked.connect(self._regen_diagnostics)
        diag_hdr.addWidget(self._diag_regen_btn)
        df_lay.addLayout(diag_hdr)

        self._diag_params = QLabel("")
        self._diag_params.setWordWrap(True)
        self._diag_params.setStyleSheet("font-size: 11px; color: #444; font-family: monospace;")
        df_lay.addWidget(self._diag_params)

        self._diag_warnings_lay = QVBoxLayout()
        self._diag_warnings_lay.setSpacing(4)
        df_lay.addLayout(self._diag_warnings_lay)

        if _MPL:
            from _widgets import MplCanvas
            self._diag_occ_canvas = MplCanvas(figsize=(8, 2.5))
            self._diag_occ_canvas.setMinimumHeight(180)
            df_lay.addWidget(self._diag_occ_canvas)

            self._diag_umap_canvas = MplCanvas(figsize=(5, 4))
            self._diag_umap_canvas.setMinimumHeight(280)
            df_lay.addWidget(self._diag_umap_canvas)

            self._diag_conf_canvas = MplCanvas(figsize=(5, 2.5))
            self._diag_conf_canvas.setMinimumHeight(180)
            df_lay.addWidget(self._diag_conf_canvas)
        else:
            self._diag_occ_canvas = None
            self._diag_umap_canvas = None
            self._diag_conf_canvas = None

        self._diag_frame.hide()
        v.addWidget(self._diag_frame)

        v.addStretch()
        scroll.setWidget(holder)
        lay.addWidget(scroll)

        self._global_log = QTextEdit()
        self._global_log.setReadOnly(True)
        self._global_log.setFixedHeight(180)
        self._global_log.setStyleSheet(
            "background:#151515;color:#cfd8dc;font-family:Consolas;font-size:11px;"
        )
        self._global_log.hide()

        log_hdr = QHBoxLayout()
        self._log_toggle = QPushButton("Show log  ▾")
        self._log_toggle.setFlat(True)
        self._log_toggle.setStyleSheet("color:#555;font-size:11px;")
        self._log_toggle.setCursor(Qt.PointingHandCursor)
        self._log_toggle.clicked.connect(self._toggle_log)
        log_hdr.addWidget(self._log_toggle)
        log_hdr.addStretch()
        copy_btn = QPushButton("Copy")
        copy_btn.setFlat(True)
        copy_btn.clicked.connect(
            lambda: QApplication.clipboard().setText(self._global_log.toPlainText())
        )
        log_hdr.addWidget(copy_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.setFlat(True)
        clear_btn.clicked.connect(self._global_log.clear)
        log_hdr.addWidget(clear_btn)
        lay.addLayout(log_hdr)
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

    @staticmethod
    def _venv_exists() -> bool:
        venv = ROOT / "venv"
        if sys.platform == "win32":
            return (venv / "Scripts" / "python.exe").exists()
        return (venv / "bin" / "python").exists()

    def _stage0_complete(self) -> bool:
        try:
            import project_manager as _pm
            project = _pm.get_active_project(ROOT, APP_CONFIG_PATH)
            return _pm.onboarding_complete(project)
        except Exception:
            return False

    def _check_stage0_readiness(self):
        """Stage 0 action: lightweight project readiness check."""
        try:
            import project_manager as _pm
            _pm.get_active_project(ROOT, APP_CONFIG_PATH)
        except Exception:
            QMessageBox.information(
                self, "Stage 0: Onboarding",
                "No valid project selected. Complete Stage 0: Onboarding before running the pipeline.\n\n"
                "Use the Pipeline view to create or open a project.",
            )
            return
        row = self._rows.get(0)
        if row and self._stage0_complete():
            row.set_status("done")

    def _open_env_setup(self):
        """GPU/environment setup — separate from Stage 0 project readiness."""
        if sys.platform == "win32":
            dlg = WslSetupDialog(self)
            dlg.exec_()
            wsl_cuml_reset_cache()
            self._probe_gpu_async()
        else:
            QMessageBox.information(
                self, "Environment Setup",
                "Run the following command in a terminal to set up the environment:\n\n"
                "    python setup.py\n\n"
                "The script will create the venv and optionally install GPU extras.\n"
                "Once complete, restart the application.",
            )

    def _param_changed(self, key, value):
        self.cfg[key] = value

    def _mark_completed(self, sid, completed):
        key = _state_key(sid)
        self.cfg.setdefault("stage_status", {})[key] = "done" if completed else "pending"
        if completed:
            self.cfg.setdefault("stage_last_run", {})[key] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cfg["last_completed_stage"] = _STAGE_BY_ID[sid]["name"]
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
            elif sid == 3:
                mins = max(5, int(n_frames / 90000))
            elif sid == 4:
                mins = 2
            elif sid == 5:
                mins = 2
            elif sid == 6:
                mins = 1
            elif sid == 7:
                mins = 1
            else:
                mins = 8 if sid == 8 else 3
            row.set_eta(f"~{mins} min")

    def update_from_cfg(self, statuses: dict | None = None):
        ss = statuses if statuses is not None else self.cfg.get("stage_status", {})
        ts = self.cfg.get("stage_last_run", {})
        for sid, row in self._rows.items():
            row.set_status(ss.get(_state_key(sid), "pending"))
            row.set_last_run(ts.get(_state_key(sid)))

    def _toggle_log(self):
        visible = not self._global_log.isVisible()
        self._global_log.setVisible(visible)
        self._log_toggle.setText("Hide log  ▴" if visible else "Show log  ▾")

    def _append_log(self, line):
        if not self._global_log.isVisible():
            self._global_log.show()
            self._log_toggle.setText("Hide log  ▴")
        self._global_log.insertPlainText(line)
        sb = self._global_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _set_buttons(self, enabled):
        self._run_full.setEnabled(enabled)
        for sid, row in self._rows.items():
            if sid != 0:  # Stage 0 is not a pipeline step; keep its button usable
                row.set_enabled(enabled)

    def _check_metadata(self, stage_ids) -> bool:
        """If feature extraction (stage 2) is about to run, warn the user about
        incomplete metadata.csv rows. Returns False if the user wants to abort."""
        if 2 not in stage_ids:
            return True
        try:
            import vieb_config as _vc
            from metadata_generator import validate_metadata_csv
            report = validate_metadata_csv(_vc.get_metadata_path())
        except Exception:
            return True
        if report["valid"]:
            return True
        details = "\n".join(f"- {m}" for m in report["messages"])
        reply = QMessageBox.warning(
            self, "Incomplete metadata.csv",
            f"metadata.csv has rows missing 'animal_id' or 'context':\n\n{details}\n\n"
            "Feature extraction can still run, but comparison/quantification\n"
            "will fail or be meaningless until these are filled in.\n\n"
            "Continue anyway?",
            QMessageBox.Yes | QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _start_worker(self, stage_ids):
        if self._worker and self._worker.isRunning():
            return
        if not self._check_metadata(stage_ids):
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
        self._status.setText(f"Running stage {sid}: {_STAGE_BY_ID[sid]['name']}")

    def _on_stage_done(self, sid, ok):
        key = _state_key(sid)
        self.cfg.setdefault("stage_status", {})[key] = "done" if ok else "error"
        if ok:
            self.cfg.setdefault("stage_last_run", {})[key] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.cfg["last_completed_stage"] = _STAGE_BY_ID[sid]["name"]
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
        """Update the Stage 3 quality badge from loaded cluster/summary data."""
        stage3_row = self._rows.get(3)
        if stage3_row is None:
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
        stage3_row.set_cluster_quality(dom_frac, dom_state_id)

    def update_diagnostics(self, data: dict):
        """Populate the diagnostics panel from loaded data."""
        diag = data.get("diagnostics")
        occ = data.get("state_occupancy")
        if not diag:
            self._diag_frame.hide()
            return

        self._diag_frame.show()

        # Resolved min_samples info from run manifest
        rm = data.get("run_manifest") or {}
        ms_req = rm.get("min_samples_requested", None)
        ms_res = rm.get("min_samples_resolved", rm.get("hdbscan_min_samples", ""))
        if ms_req == 0 and ms_res:
            ms_text = f"min_samples: Auto (resolved to {ms_res})"
        elif ms_res:
            ms_text = f"min_samples: {ms_res}"
        else:
            ms_text = ""

        # Parameters summary
        lines = [
            f"States: {diag.get('n_states', '?')}   "
            f"Frames: {diag.get('n_frames', 0):,}   "
            f"Noise: {diag.get('noise_frac', 0)*100:.1f}%",
            f"Largest state: {diag.get('largest_state_frac', 0)*100:.1f}%   "
            f"Mean confidence: {diag.get('mean_confidence', 0):.3f}   "
            f"Low conf (<0.5): {diag.get('low_confidence_frac', 0)*100:.1f}%",
            f"UMAP dims: {diag.get('umap_dims', '?')}   "
            f"min_cluster_size: {diag.get('min_cluster_size', '?')}   "
            f"Features: {diag.get('n_features', '?')}   "
            f"Wavelets: {'yes' if diag.get('use_wavelets') else 'no'}",
        ]
        if ms_text:
            lines.append(ms_text)
        self._diag_params.setText("\n".join(lines))

        # Clear old warnings
        while self._diag_warnings_lay.count():
            item = self._diag_warnings_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        warnings = diag.get("warnings", [])
        if not warnings:
            ok_lbl = QLabel("No warnings.")
            ok_lbl.setStyleSheet("color: #2e7d32; font-size: 11px; padding: 2px 0;")
            self._diag_warnings_lay.addWidget(ok_lbl)
        for w in warnings:
            level = w.get("level", "info")
            if level == "error":
                color, icon = "#c62828", "!"
            elif level == "warning":
                color, icon = "#e65100", "*"
            else:
                color, icon = "#1565c0", "i"
            lbl = QLabel(f"  {icon}  {w.get('message', '')}")
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color: {color}; font-size: 11px; padding: 2px 0;")
            if w.get("action"):
                lbl.setToolTip(w["action"])
            self._diag_warnings_lay.addWidget(lbl)

        if not _MPL or not occ is not None:
            return

        import numpy as np

        # State occupancy bar chart
        if self._diag_occ_canvas and occ is not None and not occ.empty:
            canvas = self._diag_occ_canvas
            canvas.fig.clf()
            ax = canvas.fig.add_subplot(111)
            state_rows = occ[occ["state"] >= 0].sort_values("state")
            if not state_rows.empty:
                states = state_rows["state"].values
                fracs = state_rows["fraction"].values * 100
                from _utils import _state_colors
                colors = _state_colors(max(len(states), 1))
                bar_colors = [colors[int(s) % len(colors)] for s in states]
                ax.barh(range(len(states)), fracs, color=bar_colors, alpha=0.85)
                ax.set_yticks(range(len(states)))
                ax.set_yticklabels([f"S{s}" for s in states], fontsize=7)
                ax.set_xlabel("Occupancy (%)", fontsize=9)
                ax.invert_yaxis()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_title("State Occupancy", fontsize=10, fontweight="bold", loc="left")
            canvas.fig.tight_layout()
            canvas.draw()

        # UMAP scatter
        umap_path = RESULTS / "diagnostics" / "umap_sample.csv"
        if self._diag_umap_canvas and umap_path.exists():
            import pandas as pd
            canvas = self._diag_umap_canvas
            canvas.fig.clf()
            ax = canvas.fig.add_subplot(111)
            try:
                udf = pd.read_csv(umap_path)
                valid = udf[udf["label"] >= 0]
                noise = udf[udf["label"] < 0]
                if not noise.empty:
                    ax.scatter(noise["umap_1"], noise["umap_2"],
                               c="#CCCCCC", s=1, alpha=0.3, rasterized=True)
                if not valid.empty:
                    ax.scatter(valid["umap_1"], valid["umap_2"],
                               c=valid["label"], cmap="tab20", s=1, alpha=0.5, rasterized=True)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title("UMAP Embedding (sampled)", fontsize=10, fontweight="bold", loc="left")
            except Exception:
                ax.text(0.5, 0.5, "Error loading UMAP sample", ha="center", va="center",
                        transform=ax.transAxes, color="#999")
            canvas.fig.tight_layout()
            canvas.draw()

        # Confidence histogram
        if self._diag_conf_canvas and umap_path.exists():
            import pandas as pd
            canvas = self._diag_conf_canvas
            canvas.fig.clf()
            ax = canvas.fig.add_subplot(111)
            try:
                udf = pd.read_csv(umap_path)
                ax.hist(udf["prob"].values, bins=50, color="#4E79A7", alpha=0.8,
                        edgecolor="white", linewidth=0.3)
                ax.axvline(0.5, color="#E63946", linewidth=1.5, linestyle="--", label="0.5")
                ax.set_xlabel("HDBSCAN Probability", fontsize=9)
                ax.set_ylabel("Count", fontsize=9)
                ax.legend(fontsize=8)
                ax.set_title("Confidence Distribution", fontsize=10, fontweight="bold", loc="left")
            except Exception:
                pass
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            canvas.fig.tight_layout()
            canvas.draw()

    def _regen_diagnostics(self):
        if self._worker and self._worker.isRunning():
            QMessageBox.information(self, "Pipeline busy",
                                    "Wait for the current pipeline step to finish.")
            return
        self._start_worker_cmd(
            [sys.executable, "compare.py", "--diagnostics"],
            label="Regenerating diagnostics…",
        )

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
        # Get all stage IDs in order
        all_stage_ids = [s["id"] for s in STAGES]

        if from_here:
            # Include start_sid and everything after
            ids = [sid for sid in all_stage_ids if sid >= start_sid]
        else:
            # Run only this specific stage
            ids = [start_sid]

        # Filter out collapse stage if not enabled
        if not self.cfg.get("enable_state_collapse", False):
            ids = [i for i in ids if i != 4]

        # Filter out DLC stage if pose CSVs already exist
        try:
            import vieb_config as _vc
            raw_dir = Path(_vc.get_raw_videos_dir())
        except Exception:
            raw_dir = Path("__missing_project__")
        if _has_pose_csvs(raw_dir) and 1 in ids:
            ids.remove(1)

        # Never run stage 1 from pipeline runner (it opens DLC Setup)
        if 1 in ids and from_here:
            ids.remove(1)

        return [i for i in ids if i not in (0, 9)]

    def run_full_pipeline(self):
        self._start_worker(self._build_sequence(1, from_here=True))

    def _run_stage(self, sid):
        self._start_worker(self._build_sequence(sid, from_here=False))

    def _run_from_here(self, sid):
        self._start_worker(self._build_sequence(sid, from_here=True))
