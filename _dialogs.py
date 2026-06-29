from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QPushButton, QProgressBar,
    QScrollArea, QSpinBox, QStackedWidget, QTabWidget, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from _utils import (
    ROOT, RESULTS, CLIPS, CONFIG_PATH, _load_cfg, _save_cfg,
    _load_projects, _save_projects, _register_project,
    _open_folder, _MPL, _CV2,
    _wsl_check_installed, _wsl_check_distro, _wsl_check_venv,
    _wsl_elevate_install, wsl_cuml_reset_cache,
    _wsl_path, _wsl_python, _probe_wsl_cuml,
    detect_nvidia_driver, select_gpu_stack, gpu_stack_message,
)
from _workers import ExportWorker, SubprocessWorker

if _MPL:
    from _utils import plt, PdfPages, Figure, mpl_cm, mpimg

# ---------------------------------------------------------------------------
# Module-level WSL2 GPU state (mirrors gui.py _WSL_CUML global)
# ---------------------------------------------------------------------------
_WSL_CUML: bool | None = None


class _CreateProjectDialog(QDialog):
    """Dialog to create a new DLC project via deeplabcut.create_new_project()."""

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.result_path: str | None = None
        self.setWindowTitle("Create New DLC Project")
        self.resize(480, 280)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        form = QGridLayout()
        r = 0

        def field(label, placeholder, tip=""):
            nonlocal r
            lbl = QLabel(label)
            le = QLineEdit()
            le.setPlaceholderText(placeholder)
            if tip:
                le.setToolTip(tip)
                lbl.setToolTip(tip)
            form.addWidget(lbl, r, 0)
            form.addWidget(le, r, 1)
            r += 1
            return le

        self._proj_name = field(
            "Project name", "e.g. VIEB",
            "Short name for this DLC project (no spaces recommended).",
        )
        self._experimenter = field(
            "Experimenter name", "e.g. Carlos",
            "Your name — used to name the project directory (VIEB-<name>-<date>).",
        )

        # Video directory row (with Browse button)
        lbl_vd = QLabel("Videos directory")
        self._videos_dir = QLineEdit(self.cfg.get("raw_videos_dir", str(ROOT / "raw_videos")))
        self._videos_dir.setToolTip("Directory containing your .mp4 videos.")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_videos)
        form.addWidget(lbl_vd, r, 0)
        vrow = QHBoxLayout()
        vrow.addWidget(self._videos_dir)
        vrow.addWidget(browse)
        form.addLayout(vrow, r, 1)
        r += 1

        lay.addLayout(form)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        lay.addWidget(self._status)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._create)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    def _browse_videos(self):
        d = QFileDialog.getExistingDirectory(self, "Select Videos Directory", str(ROOT))
        if d:
            self._videos_dir.setText(d)

    def _create(self):
        name = self._proj_name.text().strip()
        experimenter = self._experimenter.text().strip()
        videos_dir = self._videos_dir.text().strip()

        if not name or not experimenter:
            self._status.setText("⚠ Project name and experimenter name are required.")
            self._status.setStyleSheet("color:#c62828;")
            return
        if not os.path.isdir(videos_dir):
            self._status.setText(f"⚠ Videos directory not found: {videos_dir}")
            self._status.setStyleSheet("color:#c62828;")
            return

        try:
            import deeplabcut
        except ImportError:
            self._status.setText(
                "⚠ DeepLabCut is not installed.\n"
                "Run: pip install 'deeplabcut[tf]'  or  pip install -e '.[tracking]'"
            )
            self._status.setStyleSheet("color:#c62828;")
            return

        self._status.setText("Creating project…")
        self._status.setStyleSheet("color:#333;")
        try:
            video_files = [
                str(p) for p in Path(videos_dir).glob("*.mp4")
            ]
            project_path = deeplabcut.create_new_project(
                name, experimenter, video_files,
                working_directory=str(ROOT),
                copy_videos=False,
            )
            import vieb_config
            vieb_config.set_dlc_project_path(project_path)
            _register_project(project_path)
            self.result_path = project_path
            self._status.setText(f"✓ Created: {project_path}")
            self._status.setStyleSheet("color:#2e7d32;")
            self.accept()
        except Exception as exc:
            self._status.setText(f"✕ Failed: {exc}")
            self._status.setStyleSheet("color:#c62828;")


class WslSetupDialog(QDialog):
    """
    Automated GPU setup wizard.

    Walks through 4 prerequisites in order, automating each step where possible:
      1. WSL2 installed
      2. Linux distro registered
      3. venv_wsl Python environment created
      4. cuML (RAPIDS) importable

    Each step is attempted automatically; if it fails the user sees a one-click
    fix button with live log output.
    """

    _checks_done = pyqtSignal(list)   # list of 4 bools

    _STEPS = [
        ("WSL2 installed",          "Windows Subsystem for Linux 2 kernel"),
        ("Linux distro registered", "Ubuntu (or another Linux distribution)"),
        ("Python environment",      "venv_wsl with pipeline dependencies"),
        ("cuML GPU library",        "RAPIDS cuML + CUDA device accessible"),
    ]

    # Bash script that creates venv_wsl and installs everything
    _SETUP_SCRIPT = """\
set -e
cd {wsl_root}
echo "--- Installing system packages ---"
sudo apt-get update -q
sudo apt-get install -y -q python3 python3-venv python3-pip
echo "--- Creating venv_wsl ---"
python3 -m venv venv_wsl
echo "--- Upgrading pip ---"
venv_wsl/bin/pip install --upgrade pip -q
echo "--- Installing pipeline dependencies ---"
venv_wsl/bin/pip install numpy pandas scikit-learn umap-learn hdbscan joblib -q
echo "--- Installing cuML (RAPIDS) --- this may take several minutes ---"
venv_wsl/bin/pip install --extra-index-url https://pypi.nvidia.com cuml-cu12==24.12.0 cudf-cu12==24.12.0 cupy-cuda12x==12.2.0 cuda-python==12.2.1 "cuda-toolkit[cublas,cufft,curand,cusolver,cusparse]==12.2.2" nvidia-cuda-runtime-cu12==12.2.140 nvidia-cuda-nvrtc-cu12==12.2.140 nvidia-nvjitlink-cu12==12.2.140 nvidia-cublas-cu12==12.2.5.6 nvidia-cufft-cu12==11.0.8.103 nvidia-curand-cu12==10.3.3.141 nvidia-cusolver-cu12==11.5.2.141 nvidia-cusparse-cu12==12.1.2.141
echo "--- Verifying GPU ---"
venv_wsl/bin/python -c "import cuml; import cupy; cupy.cuda.runtime.getDeviceCount(); print('cuml_ok')"
echo "=== Setup complete ==="
"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GPU Acceleration Setup")
        self.resize(660, 600)
        self.setModal(True)
        self._active_thread = None
        self._checks_done.connect(self._on_checks_done)
        self._build()
        QTimer.singleShot(300, self._run_checks)

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(10)

        title = QLabel("GPU Acceleration Setup")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        lay.addWidget(title)

        intro = QLabel(
            "UMAP and HDBSCAN (the slowest steps) can use your RTX GPU via WSL2. "
            "This wizard checks and installs everything automatically."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color:#444;")
        lay.addWidget(intro)

        # ── Checklist ────────────────────────────────────────────────────────
        check_frame = QFrame()
        check_frame.setStyleSheet(
            "QFrame{background:#f8f8f8;border:1px solid #ddd;border-radius:6px;}"
        )
        cf_lay = QVBoxLayout(check_frame)
        cf_lay.setContentsMargins(14, 10, 14, 10)
        cf_lay.setSpacing(6)
        self._check_icons = []
        self._check_labels = []
        for i, (name, detail) in enumerate(self._STEPS):
            row = QHBoxLayout()
            icon = QLabel("⏳")
            icon.setFixedWidth(22)
            icon.setStyleSheet("font-size:14px;")
            lbl = QLabel(f"<b>{name}</b>  <span style='color:#777;font-size:11px;'>— {detail}</span>")
            lbl.setTextFormat(Qt.RichText)
            row.addWidget(icon)
            row.addWidget(lbl, stretch=1)
            cf_lay.addLayout(row)
            self._check_icons.append(icon)
            self._check_labels.append(lbl)
        lay.addWidget(check_frame)

        # ── Current action description ────────────────────────────────────────
        self._action_lbl = QLabel("Checking your system…")
        self._action_lbl.setStyleSheet("color:#1a73e8;font-weight:bold;")
        lay.addWidget(self._action_lbl)

        # ── Log output ────────────────────────────────────────────────────────
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(200)
        self._log.setStyleSheet(
            "background:#111;color:#d4d4d4;font-family:Consolas;font-size:11px;"
            "border:1px solid #333;border-radius:4px;"
        )
        lay.addWidget(self._log)

        # ── BIOS note ────────────────────────────────────────────────────────
        bios_note = QLabel(
            "ℹ  If WSL2 won't start: enable SVM (AMD) or VT-x (Intel) in your BIOS settings."
        )
        bios_note.setWordWrap(True)
        bios_note.setStyleSheet("color:#777;font-size:11px;")
        lay.addWidget(bios_note)

        # ── Buttons ──────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        self._action_btn = QPushButton("Checking…")
        self._action_btn.setEnabled(False)
        self._action_btn.setFixedHeight(32)
        self._action_btn.clicked.connect(self._on_action)
        btn_row.addWidget(self._action_btn, stretch=1)
        self._recheck_btn = QPushButton("Re-check")
        self._recheck_btn.setFixedHeight(32)
        self._recheck_btn.setEnabled(False)
        self._recheck_btn.clicked.connect(self._run_checks)
        btn_row.addWidget(self._recheck_btn)
        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(32)
        close_btn.clicked.connect(self.accept)
        btn_row.addWidget(close_btn)
        lay.addLayout(btn_row)

        self._pending_action = None   # callable, set by _on_checks_done

    # ── Check phase ──────────────────────────────────────────────────────────

    def _run_checks(self):
        self._recheck_btn.setEnabled(False)
        self._action_btn.setEnabled(False)
        self._action_btn.setText("Checking…")
        self._action_lbl.setText("Checking your system…")
        self._action_lbl.setStyleSheet("color:#1a73e8;font-weight:bold;")
        for icon in self._check_icons:
            icon.setText("⏳")

        class _CheckThread(QThread):
            done = pyqtSignal(list)
            def run(self):
                results = [
                    _wsl_check_installed(),
                    _wsl_check_distro(),
                    _wsl_check_venv(),
                    _probe_wsl_cuml(),
                ]
                self.done.emit(results)

        t = _CheckThread(self)
        t.done.connect(self._checks_done)
        self._active_thread = t
        t.start()

    def _set_icon(self, idx: int, ok: bool | None):
        if ok is True:
            self._check_icons[idx].setText("✅")
        elif ok is False:
            self._check_icons[idx].setText("❌")
        else:
            self._check_icons[idx].setText("⏳")

    def _on_checks_done(self, results: list):
        global _WSL_CUML
        wsl_ok, distro_ok, venv_ok, cuml_ok = results
        _WSL_CUML = cuml_ok

        for i, ok in enumerate(results):
            self._set_icon(i, ok)

        self._recheck_btn.setEnabled(True)

        if cuml_ok:
            self._action_lbl.setText("✅  GPU ready — clustering will use your RTX GPU automatically.")
            self._action_lbl.setStyleSheet("color:#1b5e20;font-weight:bold;")
            self._action_btn.setVisible(False)
            return

        if not wsl_ok:
            self._action_lbl.setText("Step 1: Install WSL2 (requires a system restart).")
            self._action_lbl.setStyleSheet("color:#e65100;font-weight:bold;")
            self._action_btn.setText("Install WSL2 (Admin)")
            self._action_btn.setEnabled(True)
            self._pending_action = self._do_install_wsl2
            return

        if not distro_ok:
            self._action_lbl.setText("Step 2: Install Ubuntu Linux inside WSL2.")
            self._action_lbl.setStyleSheet("color:#e65100;font-weight:bold;")
            self._action_btn.setText("Install Ubuntu")
            self._action_btn.setEnabled(True)
            self._pending_action = self._do_install_distro
            return

        if not venv_ok or not cuml_ok:
            self._action_lbl.setText(
                "Step 3/4: Create Python environment and install GPU libraries "
                "(downloads ~2 GB — takes 5-15 min)."
            )
            self._action_lbl.setStyleSheet("color:#e65100;font-weight:bold;")
            self._action_btn.setText("Set up GPU environment")
            self._action_btn.setEnabled(True)
            self._pending_action = self._do_setup_env

    def _on_action(self):
        if self._pending_action:
            self._pending_action()

    # ── Actions ──────────────────────────────────────────────────────────────

    def _do_install_wsl2(self):
        self._log_line("Opening elevated prompt to run: wsl --install\n")
        self._log_line("After installation finishes, restart your PC, then click Re-check.\n")
        try:
            _wsl_elevate_install("--install")
        except Exception as e:
            self._log_line(f"[error] {e}\n")
        self._action_btn.setEnabled(False)

    def _do_install_distro(self):
        self._log_line("Installing Ubuntu (wsl --install -d Ubuntu)…\n")
        self._log_line(
            "A new terminal will open. Set a username and password when prompted,\n"
            "then close it and click Re-check here.\n"
        )
        try:
            _wsl_elevate_install("--install -d Ubuntu")
        except Exception as e:
            self._log_line(f"[error] {e}\n")
        self._action_btn.setEnabled(False)

    def _do_setup_env(self):
        self._action_btn.setEnabled(False)
        self._recheck_btn.setEnabled(False)
        self._action_lbl.setText("Installing… (this takes several minutes — do not close the window)")
        self._action_lbl.setStyleSheet("color:#1a73e8;font-weight:bold;")

        import shlex
        wsl_root = _wsl_path(str(ROOT))
        script = self._SETUP_SCRIPT.format(wsl_root=shlex.quote(wsl_root))

        class _SetupThread(QThread):
            log = pyqtSignal(str)
            done = pyqtSignal(bool)

            def __init__(self, script):
                super().__init__()
                self._script = script

            def run(self):
                try:
                    proc = subprocess.Popen(
                        ["wsl", "bash", "-lc", self._script],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                        text=True, encoding="utf-8", errors="replace",
                    )
                    for line in proc.stdout:
                        self.log.emit(line)
                    proc.wait()
                    self.done.emit(proc.returncode == 0)
                except Exception as exc:
                    self.log.emit(f"[error] {exc}\n")
                    self.done.emit(False)

        t = _SetupThread(script)
        t.log.connect(self._log_line)
        t.done.connect(self._on_setup_done)
        self._active_thread = t
        t.start()

    def _on_setup_done(self, ok: bool):
        self._recheck_btn.setEnabled(True)
        if ok:
            self._action_lbl.setText("Installation complete — click Re-check to verify.")
            self._action_lbl.setStyleSheet("color:#1b5e20;font-weight:bold;")
            wsl_cuml_reset_cache()
            QTimer.singleShot(500, self._run_checks)
        else:
            self._action_lbl.setText("Installation failed — see log above for details.")
            self._action_lbl.setStyleSheet("color:#b71c1c;font-weight:bold;")
            self._action_btn.setText("Retry")
            self._action_btn.setEnabled(True)

    def _log_line(self, line: str):
        self._log.insertPlainText(line)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())


class DiagnoseDialog(QDialog):
    """Runs diagnose_clusters.py in a background thread; streams output to a log panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cluster Diagnostic")
        self.resize(760, 520)
        self.setModal(False)
        self._thread = None
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        hdr = QLabel("HDBSCAN min_cluster_size sweep")
        hdr.setFont(QFont("Arial", 12, QFont.Bold))
        lay.addWidget(hdr)

        sub = QLabel(
            "Tests multiple min_cluster_size values on the existing UMAP embedding.\n"
            "When complete, the recommended setting is shown at the bottom."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet("color:#555;")
        lay.addWidget(sub)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet("background:#111;color:#d4d4d4;font-family:Consolas;font-size:12px;")
        lay.addWidget(self._log, stretch=1)

        btn_row = QHBoxLayout()
        self._status_lbl = QLabel("Waiting to start…")
        self._status_lbl.setStyleSheet("color:#777;")
        btn_row.addWidget(self._status_lbl, stretch=1)
        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.close)
        btn_row.addWidget(self._close_btn)
        lay.addLayout(btn_row)

    def start(self):
        import subprocess

        class _DiagThread(QThread):
            log = pyqtSignal(str)
            done = pyqtSignal(bool)

            def run(self):
                try:
                    proc = subprocess.Popen(
                        [sys.executable, "diagnose_clusters.py"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
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

        self._thread = _DiagThread(self)
        self._thread.log.connect(self._on_log)
        self._thread.done.connect(self._on_done)
        self._status_lbl.setText("Running…")
        self._thread.start()

    def _on_log(self, line: str):
        self._log.insertPlainText(line)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def _on_done(self, ok: bool):
        self._status_lbl.setText("Done." if ok else "Failed — see output above.")
        self._status_lbl.setStyleSheet(f"color:{'#1b5e20' if ok else '#b71c1c'};font-weight:bold;")


class LinuxGpuSetupDialog(QDialog):
    """Guide the user through installing cuML (RAPIDS) on Linux for GPU-accelerated clustering."""

    def __init__(self, gpu_name: str | None = None, parent=None):
        super().__init__(parent)
        self._driver_info = detect_nvidia_driver()
        self._stack = select_gpu_stack(self._driver_info.get("driver_tuple"))
        self._gpu_name = gpu_name or self._driver_info.get("gpu_name") or "NVIDIA GPU"
        self._worker = None
        self.setWindowTitle("GPU Acceleration Setup")
        self.resize(620, 480)
        self.setModal(True)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(10)

        hdr = QLabel(f"GPU detected: {self._gpu_name}")
        hdr.setFont(QFont("Arial", 13, QFont.Bold))
        lay.addWidget(hdr)

        info = QLabel(
            "cuML (RAPIDS) enables much faster UMAP + HDBSCAN clustering on your GPU.\n\n"
            + gpu_stack_message(self._driver_info)
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        if self._stack:
            install_cmd = (
                "pip install --extra-index-url https://pypi.nvidia.com "
                + " ".join(self._stack["packages"])
            )
        else:
            install_cmd = "Upgrade the NVIDIA driver, then re-open this setup dialog."

        cmd_box = QTextEdit(install_cmd)
        cmd_box.setReadOnly(True)
        cmd_box.setMaximumHeight(64)
        cmd_box.setStyleSheet(
            "background:#1e1e1e;color:#cfd8dc;font-family:Consolas,monospace;"
            "font-size:12px;border-radius:4px;padding:4px;"
        )
        lay.addWidget(cmd_box)

        btn_row = QHBoxLayout()
        self._install_btn = QPushButton("Run Install")
        self._install_btn.setToolTip(
            "Install the pinned RAPIDS/cuML packages compatible with this driver.\n"
            "This may take several minutes on the first run."
        )
        self._install_btn.setEnabled(self._stack is not None)
        self._install_btn.clicked.connect(self._run_install)
        btn_row.addWidget(self._install_btn)

        self._verify_btn = QPushButton("Verify")
        self._verify_btn.clicked.connect(self._run_verify)
        btn_row.addWidget(self._verify_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet(
            "background:#151515;color:#cfd8dc;font-family:Consolas,monospace;font-size:11px;"
        )
        lay.addWidget(self._log)

        self._status = QLabel("")
        self._status.setStyleSheet("font-weight:bold;")
        lay.addWidget(self._status)

        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        lay.addWidget(close, alignment=Qt.AlignRight)

    def _append(self, line: str):
        self._log.insertPlainText(line)
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())

    def _run_install(self):
        if self._worker and self._worker.isRunning():
            return
        if self._stack is None:
            self._status.setText("Upgrade the NVIDIA driver before installing GPU acceleration.")
            self._status.setStyleSheet("color:#b71c1c;font-weight:bold;")
            return
        self._install_btn.setEnabled(False)
        self._verify_btn.setEnabled(False)
        self._status.setText("")
        self._log.clear()
        self._append("Running install…\n")
        args = [
            "-m", "pip", "install",
            "--extra-index-url", "https://pypi.nvidia.com",
        ] + list(self._stack["packages"])
        self._append("Installing " + self._stack["label"] + "\n")
        self._worker = SubprocessWorker(args)
        self._worker.log.connect(self._append)
        self._worker.done.connect(self._on_install_done)
        self._worker.start()

    def _on_install_done(self, ok: bool):
        self._install_btn.setEnabled(True)
        self._verify_btn.setEnabled(True)
        if ok:
            self._status.setText("Install succeeded — click Verify to confirm GPU access.")
            self._status.setStyleSheet("color:#1b5e20;font-weight:bold;")
            self._run_verify()
        else:
            self._status.setText("Install failed — see output above.")
            self._status.setStyleSheet("color:#b71c1c;font-weight:bold;")

    def _run_verify(self):
        if self._worker and self._worker.isRunning():
            return
        self._verify_btn.setEnabled(False)
        self._log.clear()
        self._append("Verifying cuML + GPU…\n")
        verify_script = (
            "import cuml; import cupy as cp; "
            "a = cp.array([1.0, 2.0, 3.0]); assert float(a.sum()) == 6.0; "
            "print('cuML GPU verified OK')"
        )
        self._worker = SubprocessWorker(["-c", verify_script])
        self._worker.log.connect(self._append)
        self._worker.done.connect(self._on_verify_done)
        self._worker.start()

    def _on_verify_done(self, ok: bool):
        self._verify_btn.setEnabled(True)
        if ok:
            self._status.setText("GPU acceleration is ready — close this dialog and re-run the pipeline.")
            self._status.setStyleSheet("color:#1b5e20;font-weight:bold;")
        else:
            self._status.setText("Verification failed — check the output above.")
            self._status.setStyleSheet("color:#b71c1c;font-weight:bold;")


class ExportResultsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Export Results")
        self.resize(600, 520)
        self._worker = None
        lay = QVBoxLayout(self)
        self._checks = {}
        options = [
            ("summary", "Summary table (summary_table.csv)"),
            ("animal", "Animal scalars (animal_scalars.csv)"),
            ("states", "State profiles (state_summary.csv)"),
            ("transitions", "Transition matrices (transition_table.csv)"),
            ("motifs", "Motif enrichment (motifs.csv)"),
            ("plots", "All comparison plots (PNG files)"),
            ("clips", "Video clips (copies clips/ directory)"),
            ("pdf", "Full report (single PDF summary)"),
        ]
        for key, text in options:
            cb = QCheckBox(text)
            self._checks[key] = cb
            lay.addWidget(cb)
        self._run = QPushButton("Choose Destination and Export")
        self._run.clicked.connect(self._start)
        lay.addWidget(self._run)
        self._prog = QProgressBar()
        self._prog.setRange(0, 1)
        lay.addWidget(self._prog)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setStyleSheet("background:#111;color:#ddd;")
        lay.addWidget(self._log)
        b = QDialogButtonBox(QDialogButtonBox.Close)
        b.rejected.connect(self.reject)
        lay.addWidget(b)

    def _start(self):
        opts = {k: v.isChecked() for k, v in self._checks.items()}
        if not any(opts.values()):
            QMessageBox.information(self, "Export", "Select at least one output.")
            return
        d = QFileDialog.getExistingDirectory(self, "Select destination", str(ROOT))
        if not d:
            return
        self._run.setEnabled(False)
        self._prog.setRange(0, 0)
        self._worker = ExportWorker(opts, Path(d))
        self._worker.log.connect(lambda t: self._log.insertPlainText(t))
        self._worker.done.connect(self._done)
        self._worker.start()

    def _done(self, ok):
        self._prog.setRange(0, 1)
        self._prog.setValue(1 if ok else 0)
        self._run.setEnabled(True)
        QMessageBox.information(self, "Export", "Export complete." if ok else "Export failed.")


def export_pdf_report():
    if not _MPL:
        raise RuntimeError("matplotlib is required for PDF export.")

    from datetime import datetime

    summary = RESULTS / "comparison" / "summary_table.csv"
    state_summary = RESULTS / "characterization" / "state_summary.csv"
    transition = RESULTS / "comparison" / "transition_table.csv"
    animal_scalars = RESULTS / "comparison" / "animal_scalars.csv"
    motifs = RESULTS / "comparison" / "motifs.csv"
    cluster_info = RESULTS / "shared" / "cluster_info.json"
    fi_path = RESULTS / "features" / "index.json"

    df = pd.read_csv(summary) if summary.exists() else pd.DataFrame()
    ss = pd.read_csv(state_summary) if state_summary.exists() else pd.DataFrame()
    tt = pd.read_csv(transition) if transition.exists() else pd.DataFrame()
    asc = pd.read_csv(animal_scalars) if animal_scalars.exists() else pd.DataFrame()
    mot = pd.read_csv(motifs) if motifs.exists() else pd.DataFrame()
    ci = json.loads(cluster_info.read_text(encoding="utf-8")) if cluster_info.exists() else {"n_clusters": 0}
    fi = json.loads(fi_path.read_text(encoding="utf-8")) if fi_path.exists() else {}

    out = RESULTS / f"VIEB_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    with PdfPages(out) as pdf:
        fig, ax = plt.subplots(figsize=(8.3, 11.7))
        n_videos = len(df) if not df.empty else 0
        n_frames = sum(int(v.get("n_frames", 0)) for v in fi.values()) if isinstance(fi, dict) else 0
        n_states = int(ci.get("n_clusters", 0))
        noise = "-"
        if not df.empty:
            sc = [c for c in df.columns if c.startswith("state_") and c.endswith("_frac")]
            if sc:
                noise = f"{(1 - df[sc].sum(axis=1).mean()) * 100:.1f}%"
        ax.axis("off")
        ax.text(0.1, 0.9, "VIEB Dataset Overview", fontsize=20, weight="bold")
        ax.text(0.1, 0.78, f"Videos: {n_videos}\nFrames: {n_frames:,}\nStates: {n_states}\nNoise: {noise}", fontsize=14)
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8))
        if not df.empty and "context" in df.columns:
            sc = [c for c in df.columns if c.startswith("state_") and c.endswith("_frac")]
            by_ctx = df.groupby("context")[sc].mean()
            by_ctx.plot(kind="bar", ax=ax)
            ax.set_title("State occupancy by context")
        else:
            ax.text(0.5, 0.5, "No summary_table.csv available", ha="center", va="center")
        pdf.savefig(fig); plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        if not tt.empty and "context" in tt.columns:
            n = int(ci.get("n_clusters", 0))
            for idx, ctx in enumerate(["A", "B"]):
                sub = tt[tt["context"] == ctx]
                cols = [f"trans_{i}_{j}" for i in range(n) for j in range(n) if f"trans_{i}_{j}" in sub.columns]
                if cols and not sub.empty:
                    mat = sub[cols].mean().values.reshape(n, n)
                    im = axes[idx].imshow(mat, cmap="Blues", aspect="auto")
                    axes[idx].set_title(f"Context {ctx}")
                    fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8))
        if not ss.empty:
            mcols = [c for c in ["mean_centroid_speed", "mean_angular_vel", "mean_body_length_px", "mean_elongation", "mean_entropy"] if c in ss.columns]
            mat = ss[mcols].values
            im = ax.imshow(mat, cmap="viridis", aspect="auto")
            ax.set_yticks(range(len(ss)))
            ax.set_yticklabels(ss["state"] if "state" in ss.columns else range(len(ss)))
            ax.set_xticks(range(len(mcols)))
            ax.set_xticklabels(mcols, rotation=45, ha="right")
            ax.set_title("Kinematic profile heatmap")
            fig.colorbar(im, ax=ax)
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8))
        if not asc.empty and {"freeze_auc", "mean_discrimination_ratio"}.issubset(asc.columns):
            ax.scatter(asc["freeze_auc"], asc["mean_discrimination_ratio"])
            for _, r in asc.iterrows():
                ax.annotate(str(r.get("animal_id", "")), (r["freeze_auc"], r["mean_discrimination_ratio"]))
            ax.set_xlabel("AUC")
            ax.set_ylabel("Discrimination ratio")
            ax.set_title("Animal scalars")
        pdf.savefig(fig); plt.close(fig)

        fig, ax = plt.subplots(figsize=(11, 8))
        ax.axis("off")
        top = mot.sort_values("enrichment_ratio", ascending=False).head(10) if not mot.empty else pd.DataFrame()
        if not top.empty:
            table_data = top[["motif", "type", "enrichment_ratio"]].astype(str).values.tolist()
            tbl = ax.table(cellText=table_data, colLabels=["Motif", "Type", "Enrichment"], loc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(10)
            tbl.scale(1, 1.5)
            ax.set_title("Top 10 enriched motifs")
        else:
            ax.text(0.5, 0.5, "No motif data available", ha="center", va="center")
        pdf.savefig(fig); plt.close(fig)
    return out
