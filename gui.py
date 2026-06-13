#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""VIEB GUI - Video Interpreter for Experimental Behavior."""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import sys
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from PyQt5.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal
    from PyQt5.QtGui import QFont, QIcon, QImage, QKeySequence, QPixmap
    from PyQt5.QtWidgets import (
        QApplication,
        QAbstractItemView,
        QCheckBox,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QFileDialog,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPushButton,
        QProgressBar,
        QScrollArea,
        QSlider,
        QSpinBox,
        QStackedWidget,
        QStatusBar,
        QTabWidget,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError:
    print("PyQt5 not found. Install with: pip install PyQt5")
    sys.exit(1)

_MPL = False
try:
    import matplotlib

    matplotlib.use("Qt5Agg")
    import matplotlib.cm as mpl_cm
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    _MPL = True
except Exception:
    pass

_CV2 = False
try:
    import cv2

    _CV2 = True
except Exception:
    pass

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
CLIPS = ROOT / "clips"
CONFIG_PATH = ROOT / "config.json"
VALIDATION_DIR = RESULTS / "validation"

STAGES = [
    {
        "id": 1,
        "name": "Cohort Preparation",
        "desc": "Normalize and validate your cohort Excel/CSV file against metadata.csv.",
        "cmd": "python prepare_cohort.py --input {cohort_input_file} --output cohort_normalized.csv --metadata metadata.csv",
    },
    {
        "id": 2,
        "name": "Pose Estimation (DLC)",
        "desc": "Run DeepLabCut analysis to generate pose CSV files for all videos.",
        "cmd": "python setup_dlc_training.py --analyze",
    },
    {
        "id": 3,
        "name": "Feature Extraction",
        "desc": "Extract ~51 kinematic features per frame from tracked keypoints.",
        "cmd": "python compare.py --extract [--no-wavelets]",
    },
    {
        "id": 4,
        "name": "Shared Clustering",
        "desc": "Pool all frames, standardize, run UMAP → HDBSCAN → HMM smoothing.",
        "cmd": "python compare.py --cluster --min-cluster-size {N}",
    },
    {
        "id": 5,
        "name": "Comparison Report",
        "desc": "Build summary tables, transition matrices, and group comparison plots.",
        "cmd": "python compare.py --report --cohort cohort_normalized.csv",
    },
    {
        "id": 6,
        "name": "State Characterization",
        "desc": "Generate kinematic profiles, context enrichment, bout statistics, and optional video clips.",
        "cmd": "python characterize.py [--clips]",
    },
    {
        "id": 7,
        "name": "Quantification",
        "desc": "Compute per-animal fear index, discrimination ratio, and behavioral diversity.",
        "cmd": "python quantify.py --build --cohort cohort_normalized.csv",
    },
    {
        "id": 8,
        "name": "Fear Index",
        "desc": "Cohort-normalized fear quantification and cohort-level comparison profiles.",
        "cmd": "python fear_index.py --cohort cohort_normalized.csv",
    },
    {
        "id": 9,
        "name": "Behavioral Fingerprint",
        "desc": "Per-animal fingerprints, deviation scores, and forward/reverse cohort models.",
        "cmd": "python behavioral_fingerprint.py --all --cohort cohort_normalized.csv",
    },
    {
        "id": 10,
        "name": "Jess Protein Correlation (optional)",
        "desc": "Correlate behavioral scalars against Jess western blot protein data.",
        "cmd": "python quantify.py --jess {jess_file}",
        "optional": True,
    },
]

_DEFAULT_CFG = {
    "arena_bounds": {"x_min": 0, "y_min": 0, "x_max": 1280, "y_max": 960},
    "results_dir": str(RESULTS),
    "raw_videos_dir": str(ROOT / "raw_videos"),
    "fps": 30,
    "window_size": [1280, 800],
    "last_view": "Overview",
    "min_cluster_size": 200,
    "use_wavelets": True,
    "export_clips": False,
    "cohort_input_file": "",
    "cohort_file": "cohort_normalized.csv",
    "jess_file": "",
    "onboarding_complete": False,
    "project_name": "VIEB Project",
    "last_completed_stage": "",
    "stage_status": {},
    "stage_last_run": {},
    "context_groups": "A,B,C",
}

_SPINNER = ["|", "/", "-", "\\"]
_NAV_VIEWS = [
    "Overview",
    "Run Pipeline",
    "Browse States",
    "Validation",
    "Transition Matrix",
    "Motifs",
    "Animal Explorer",
    "Settings",
]


def _load_cfg():
    cfg = json.loads(json.dumps(_DEFAULT_CFG))
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    if "arena_bounds" not in cfg:
        cfg["arena_bounds"] = dict(_DEFAULT_CFG["arena_bounds"])
    for k, v in _DEFAULT_CFG.items():
        if k not in cfg:
            cfg[k] = v
    return cfg


def _save_cfg(cfg):
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def _fmt_ts(ts):
    if not ts:
        return "-"
    if isinstance(ts, str):
        return ts
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def _state_key(stage_id):
    return str(stage_id)


def _results_exist():
    return RESULTS.exists() and any(RESULTS.iterdir())


def _find_dlc_project():
    for p in ROOT.glob("VIEB-*/config.yaml"):
        return p.parent
    return None


def _has_pose_csvs(raw_videos_dir: Path):
    if not raw_videos_dir.exists():
        return False
    return any(raw_videos_dir.glob("*DLC*.csv")) or any(raw_videos_dir.glob("*.csv"))


def _open_folder(path: Path):
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception:
        pass


class _Capture(QObject):
    text = pyqtSignal(str)
    encoding = "utf-8"
    errors = "replace"

    def write(self, s):
        if s:
            self.text.emit(s)

    def flush(self):
        pass

    def isatty(self):
        return False


if _MPL:
    class MplCanvas(FigureCanvas):
        def __init__(self, parent=None, figsize=(6, 4)):
            self.fig = Figure(figsize=figsize, tight_layout=True)
            super().__init__(self.fig)
            self.setParent(parent)
            self.ax = self.fig.add_subplot(111)

else:
    class MplCanvas(QWidget):
        def __init__(self, parent=None, figsize=(6, 4)):
            super().__init__(parent)
            self.fig = None
            self.ax = None

        def draw(self):
            pass


class DataLoader(QThread):
    loaded = pyqtSignal(dict)
    error = pyqtSignal(str)

    def run(self):
        data = {}
        try:
            def _csv(rel):
                p = RESULTS / rel
                return pd.read_csv(p) if p.exists() else None

            def _json(rel):
                p = RESULTS / rel
                return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

            data["summary"] = _csv("comparison/summary_table.csv")
            data["state_summary"] = _csv("characterization/state_summary.csv")
            data["context_report"] = _csv("characterization/context_report.csv")
            data["transition_table"] = _csv("comparison/transition_table.csv")
            data["bouts"] = _csv("characterization/bouts.csv")
            data["motifs"] = _csv("comparison/motifs.csv")
            data["cluster_info"] = _json("shared/cluster_info.json")
            data["feature_index"] = _json("features/index.json")
            data["animal_scalars"] = _csv("comparison/animal_scalars.csv")
            data["labels_per_frame"] = _csv("characterization/labels_per_frame.csv")
            data["validation_labels"] = _csv("validation/frame_labels.csv")
            data["validation_sample"] = _csv("validation/current_sample.csv")
            data["master_table"] = _csv("quantification/master_table.csv")
            data["fear_index"] = _csv("quantification/fear_index.csv")
            data["cohort_fear_profiles"] = _csv("quantification/cohort_fear_profiles.csv")
            data["behavioral_fingerprints"] = _csv("comparison/behavioral_fingerprints.csv")
            data["deviation_scores"] = _csv("comparison/deviation_scores.csv")
        except Exception as e:
            self.error.emit(str(e))
            return
        self.loaded.emit(data)


class PipelineRunner(QThread):
    log = pyqtSignal(str)
    stage_started = pyqtSignal(int)
    stage_done = pyqtSignal(int, bool)
    all_done = pyqtSignal(bool)

    def __init__(self, stage_ids: list[int], cfg: dict):
        super().__init__()
        self.stage_ids = stage_ids
        self.cfg = cfg

    def _run_subprocess(self, args):
        p = subprocess.Popen(
            [sys.executable, *args],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert p.stdout is not None
        for line in p.stdout:
            self.log.emit(line)
        rc = p.wait()
        return rc == 0

    def run(self):
        ok_all = True
        mcs = int(self.cfg.get("min_cluster_size", 200))
        use_wavelets = bool(self.cfg.get("use_wavelets", True))
        export_clips = bool(self.cfg.get("export_clips", False))
        cohort_file = self.cfg.get("cohort_file", "cohort_normalized.csv")
        cohort_input = self.cfg.get("cohort_input_file", "")
        jess_file = self.cfg.get("jess_file", "")

        for sid in self.stage_ids:
            self.stage_started.emit(sid)
            try:
                if sid == 1:
                    if not cohort_input:
                        self.log.emit("Stage 1 skipped: no cohort input file configured.\n")
                        self.stage_done.emit(sid, True)
                        continue
                    ok = self._run_subprocess([
                        "prepare_cohort.py",
                        "--input", cohort_input,
                        "--output", cohort_file,
                        "--metadata", "metadata.csv",
                    ])
                elif sid == 2:
                    ok = self._run_subprocess(["setup_dlc_training.py", "--analyze"])
                elif sid == 3:
                    args = ["compare.py", "--extract"]
                    if not use_wavelets:
                        args.append("--no-wavelets")
                    ok = self._run_subprocess(args)
                elif sid == 4:
                    ok = self._run_subprocess(["compare.py", "--cluster", "--min-cluster-size", str(mcs)])
                elif sid == 5:
                    ok = self._run_subprocess(["compare.py", "--report", "--cohort", cohort_file])
                elif sid == 6:
                    args = ["characterize.py"]
                    if export_clips:
                        args.append("--clips")
                    ok = self._run_subprocess(args)
                elif sid == 7:
                    ok = self._run_subprocess(["quantify.py", "--build", "--cohort", cohort_file])
                elif sid == 8:
                    ok = self._run_subprocess(["fear_index.py", "--cohort", cohort_file])
                elif sid == 9:
                    ok = self._run_subprocess(["behavioral_fingerprint.py", "--all", "--cohort", cohort_file])
                elif sid == 10:
                    if not jess_file:
                        self.log.emit("Stage 10 skipped: no Jess protein file configured.\n")
                        self.stage_done.emit(sid, True)
                        continue
                    ok = self._run_subprocess(["quantify.py", "--jess", jess_file])
                else:
                    ok = True
                if not ok:
                    raise RuntimeError(f"Stage {sid} subprocess returned non-zero exit code.")
                self.stage_done.emit(sid, True)
            except Exception:
                self.log.emit(traceback.format_exc())
                self.stage_done.emit(sid, False)
                ok_all = False
                break
        self.all_done.emit(ok_all)


class SubprocessWorker(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(bool)

    def __init__(self, args: list[str]):
        super().__init__()
        self.args = args

    def run(self):
        try:
            p = subprocess.Popen(
                [sys.executable, *self.args],
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            assert p.stdout is not None
            for line in p.stdout:
                self.log.emit(line)
            self.done.emit(p.wait() == 0)
        except Exception:
            self.log.emit(traceback.format_exc())
            self.done.emit(False)


class ClipGenerationWorker(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(bool)

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg

    def run(self):
        cap = _Capture()
        cap.text.connect(self.log)
        old_out, old_err = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = cap
        ok = False
        try:
            sys.path.insert(0, str(ROOT))
            from characterize import cmd_clips

            cmd_clips(fps=float(self.cfg.get("fps", 30)))
            ok = True
        except Exception:
            print(traceback.format_exc())
        finally:
            sys.stdout, sys.stderr = old_out, old_err
        self.done.emit(ok)


class ExportWorker(QThread):
    log = pyqtSignal(str)
    done = pyqtSignal(bool)

    def __init__(self, options: dict, destination: Path):
        super().__init__()
        self.options = options
        self.destination = destination

    def _copy_file(self, rel):
        src = RESULTS / rel
        if src.exists():
            dst = self.destination / src.name
            shutil.copy2(src, dst)
            self.log.emit(f"Copied: {src} -> {dst}\n")

    def _copy_plot_pngs(self):
        comp = RESULTS / "comparison"
        if not comp.exists():
            return
        for p in comp.glob("*.png"):
            shutil.copy2(p, self.destination / p.name)
            self.log.emit(f"Copied plot: {p.name}\n")

    def _copy_clips(self):
        if CLIPS.exists():
            dst = self.destination / "clips"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(CLIPS, dst)
            self.log.emit("Copied clips directory.\n")

    def run(self):
        try:
            self.destination.mkdir(parents=True, exist_ok=True)
            if self.options.get("summary"):
                self._copy_file(Path("comparison") / "summary_table.csv")
            if self.options.get("animal"):
                self._copy_file(Path("comparison") / "animal_scalars.csv")
            if self.options.get("states"):
                self._copy_file(Path("characterization") / "state_summary.csv")
            if self.options.get("transitions"):
                self._copy_file(Path("comparison") / "transition_table.csv")
            if self.options.get("motifs"):
                self._copy_file(Path("comparison") / "motifs.csv")
            if self.options.get("plots"):
                self._copy_plot_pngs()
            if self.options.get("clips"):
                self._copy_clips()
            if self.options.get("pdf"):
                pdf = export_pdf_report()
                shutil.copy2(pdf, self.destination / pdf.name)
                self.log.emit(f"Copied report: {pdf.name}\n")
            self.done.emit(True)
        except Exception:
            self.log.emit(traceback.format_exc())
            self.done.emit(False)


class VideoPlayer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._cap = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._next_frame)
        self._fps = 30.0
        self._speed = 1.0
        self._total = 0
        self._cur = 0
        self._loop = False
        self._playing = False
        self._frame_buf = None
        self.setFocusPolicy(Qt.StrongFocus)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._display = QLabel("No video loaded", alignment=Qt.AlignCenter)
        self._display.setMinimumSize(320, 220)
        self._display.setStyleSheet("background:#111;color:#999;")
        lay.addWidget(self._display)

        ctrl = QHBoxLayout()
        self._btn_play = QPushButton("Play")
        self._btn_play.clicked.connect(self.toggle_play)
        ctrl.addWidget(self._btn_play)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.sliderMoved.connect(self.seek)
        ctrl.addWidget(self._slider)

        self._loop_btn = QCheckBox("Loop")
        self._loop_btn.toggled.connect(lambda v: setattr(self, "_loop", v))
        ctrl.addWidget(self._loop_btn)

        ctrl.addWidget(QLabel("Speed"))
        self._speed_combo = QComboBox()
        self._speed_combo.addItems(["0.25x", "0.5x", "1x"])
        self._speed_combo.setCurrentText("1x")
        self._speed_combo.currentTextChanged.connect(self._set_speed)
        ctrl.addWidget(self._speed_combo)
        lay.addLayout(ctrl)

    def _set_speed(self, text):
        val = text.replace("x", "")
        self._speed = float(val)
        if self._playing:
            self.play()

    def load(self, path: str):
        if not _CV2:
            self._display.setText("OpenCV unavailable")
            return
        self.pause()
        if self._cap:
            self._cap.release()
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            self._display.setText("Cannot open clip")
            return
        self._fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._slider.setMaximum(max(0, self._total - 1))
        self._show(0)

    def _show(self, idx):
        if not self._cap:
            return
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self._cap.read()
        if not ret:
            return
        h, w = frame.shape[:2]
        max_w, max_h = self._display.width(), self._display.height()
        scale = min(max_w / w, max_h / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        frame = cv2.resize(frame, (nw, nh))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._frame_buf = frame.copy()
        img = QImage(self._frame_buf.data, nw, nh, 3 * nw, QImage.Format_RGB888)
        self._display.setPixmap(QPixmap.fromImage(img))
        self._cur = idx
        self._slider.blockSignals(True)
        self._slider.setValue(idx)
        self._slider.blockSignals(False)

    def _next_frame(self):
        nxt = self._cur + 1
        if nxt >= self._total:
            if self._loop:
                nxt = 0
            else:
                self.pause()
                return
        self._show(nxt)

    def play(self):
        if not self._cap:
            return
        self._playing = True
        self._btn_play.setText("Pause")
        interval = max(1, int(1000 / max(0.01, self._fps * self._speed)))
        self._timer.start(interval)

    def pause(self):
        self._playing = False
        self._btn_play.setText("Play")
        self._timer.stop()

    def toggle_play(self):
        self.pause() if self._playing else self.play()

    def seek(self, idx):
        self.pause()
        self._show(idx)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Space:
            self.toggle_play()
            e.accept()
            return
        super().keyPressEvent(e)

    def closeEvent(self, e):
        self.pause()
        if self._cap:
            self._cap.release()
        super().closeEvent(e)


class _Card(QFrame):
    def __init__(self, title, value="-"):
        super().__init__()
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("QFrame{background:#f5f5f5;border:none;border-radius:8px;}")
        self.setFixedHeight(95)
        lay = QVBoxLayout(self)
        t = QLabel(title)
        t.setStyleSheet("color:#666;")
        lay.addWidget(t)
        self._value = QLabel(value)
        self._value.setFont(QFont("Arial", 22, QFont.Bold))
        lay.addWidget(self._value)

    def set(self, value):
        self._value.setText(str(value))


class OverviewView(QWidget):
    export_requested = pyqtSignal()

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(16)

        top = QHBoxLayout()
        title = QLabel("Overview")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        top.addWidget(title)
        top.addStretch()
        self._export_btn = QPushButton("Export Results")
        self._export_btn.clicked.connect(self.export_requested.emit)
        top.addWidget(self._export_btn)
        lay.addLayout(top)

        row = QHBoxLayout()
        self._c_videos = _Card("Total Videos")
        self._c_frames = _Card("Total Frames")
        self._c_states = _Card("States")
        self._c_noise = _Card("Noise %")
        for c in (self._c_videos, self._c_frames, self._c_states, self._c_noise):
            row.addWidget(c)
        lay.addLayout(row)

        box = QGroupBox("Mean State Occupancy")
        bl = QVBoxLayout(box)
        if _MPL:
            self._canvas = MplCanvas(figsize=(8, 4))
            bl.addWidget(self._canvas)
        else:
            self._canvas = None
            bl.addWidget(QLabel("Install matplotlib to view charts."))
        lay.addWidget(box)

        self._run_lbl = QLabel("Last run: -")
        self._run_lbl.setStyleSheet("color:#777;")
        lay.addWidget(self._run_lbl)
        lay.addStretch()

    def update_data(self, data):
        summary = data.get("summary")
        ci = data.get("cluster_info")
        fi = data.get("feature_index")
        if summary is None:
            self._c_videos.set("-")
            return
        self._c_videos.set(len(summary))
        self._c_states.set(ci.get("n_clusters", 0) if ci else "-")

        total = 0
        if isinstance(fi, dict):
            for v in fi.values():
                if isinstance(v, dict):
                    total += int(v.get("n_frames", 0))
        self._c_frames.set(f"{total:,}" if total else "-")

        state_cols = [c for c in summary.columns if c.startswith("state_") and c.endswith("_frac")]
        if state_cols:
            mean_sum = summary[state_cols].sum(axis=1).mean()
            noise = (1 - float(mean_sum)) * 100
            self._c_noise.set(f"{noise:.1f}%")
            if self._canvas and ci:
                n = int(ci.get("n_clusters", 0))
                vals = [summary.get(f"state_{i}_frac", pd.Series([0])).mean() for i in range(n)]
                self._canvas.ax.clear()
                self._canvas.ax.bar(range(n), vals, color=mpl_cm.tab20(np.linspace(0, 1, max(1, n))))
                self._canvas.ax.set_xlabel("State ID")
                self._canvas.ax.set_ylabel("Fraction")
                self._canvas.fig.tight_layout()
                self._canvas.draw()
        p = RESULTS / "comparison" / "summary_table.csv"
        if p.exists():
            self._run_lbl.setText(f"Last run: {_fmt_ts(p.stat().st_mtime)}")


class StageRow(QFrame):
    run_stage = pyqtSignal(int)
    run_from_here = pyqtSignal(int)
    mark_completed = pyqtSignal(int, bool)
    changed = pyqtSignal(str, object)

    def __init__(self, stage: dict, cfg: dict):
        super().__init__()
        self.stage = stage
        self.cfg = cfg
        self.logs = deque(maxlen=20)
        self._build()

    def _build(self):
        self.setStyleSheet("QFrame{border:none;background:#fff;}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)

        top = QHBoxLayout()
        self._icon = QLabel("○")
        self._icon.setFixedWidth(18)
        top.addWidget(self._icon)
        name = QLabel(f"Stage {self.stage['id']}: {self.stage['name']}")
        name.setFont(QFont("Arial", 11, QFont.Bold))
        top.addWidget(name)
        top.addStretch()
        self._ts = QLabel("Last run: -")
        self._eta = QLabel("ETA: -")
        top.addWidget(self._ts)
        top.addWidget(QLabel("  "))
        top.addWidget(self._eta)
        self._expand = QToolButton()
        self._expand.setText("▾")
        self._expand.setCheckable(True)
        self._expand.clicked.connect(self._toggle)
        top.addWidget(self._expand)
        self._done_cb = QCheckBox()
        self._done_cb.setToolTip("Mark stage as completed")
        self._done_cb.toggled.connect(lambda v: self.mark_completed.emit(self.stage["id"], v))
        top.addWidget(self._done_cb)
        lay.addLayout(top)

        desc_text = self.stage["desc"]
        if self.stage.get("optional"):
            desc_text += "  (optional)"
        d = QLabel(desc_text)
        d.setStyleSheet("color:#555;")
        lay.addWidget(d)

        self._details = QWidget()
        dl = QVBoxLayout(self._details)
        cmd = QLabel(f"CLI: {self.stage['cmd']}")
        cmd.setStyleSheet("font-family:Consolas;color:#222;")
        dl.addWidget(cmd)

        params = QHBoxLayout()
        sid = self.stage["id"]
        if sid == 1:
            self._cohort_input = QLineEdit(self.cfg.get("cohort_input_file", ""))
            self._cohort_input.setPlaceholderText("Path to cohort Excel/CSV…")
            self._cohort_input.textChanged.connect(lambda v: self.changed.emit("cohort_input_file", v))
            browse = QPushButton("Browse…")
            browse.clicked.connect(self._browse_cohort_input)
            params.addWidget(QLabel("Cohort input:"))
            params.addWidget(self._cohort_input)
            params.addWidget(browse)
        if sid == 4:
            self._mcs = QSpinBox()
            self._mcs.setRange(50, 10000)
            self._mcs.setValue(int(self.cfg.get("min_cluster_size", 200)))
            self._mcs.setSingleStep(50)
            self._mcs.valueChanged.connect(lambda v: self.changed.emit("min_cluster_size", v))
            self._wave = QCheckBox("Use Morlet wavelets")
            self._wave.setChecked(bool(self.cfg.get("use_wavelets", True)))
            self._wave.toggled.connect(lambda v: self.changed.emit("use_wavelets", v))
            params.addWidget(QLabel("min_cluster_size:"))
            params.addWidget(self._mcs)
            params.addWidget(self._wave)
        if sid == 6:
            self._clips = QCheckBox("Export video clips")
            self._clips.setChecked(bool(self.cfg.get("export_clips", False)))
            self._clips.toggled.connect(lambda v: self.changed.emit("export_clips", v))
            params.addWidget(self._clips)
        if sid == 10:
            self._jess_input = QLineEdit(self.cfg.get("jess_file", ""))
            self._jess_input.setPlaceholderText("Path to Jess protein Excel/CSV… (optional)")
            self._jess_input.textChanged.connect(lambda v: self.changed.emit("jess_file", v))
            browse_j = QPushButton("Browse…")
            browse_j.clicked.connect(self._browse_jess)
            params.addWidget(QLabel("Jess file:"))
            params.addWidget(self._jess_input)
            params.addWidget(browse_j)
        params.addStretch()
        dl.addLayout(params)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(120)
        self._log.setStyleSheet("background:#181818;color:#d4d4d4;font-family:Consolas;")
        dl.addWidget(self._log)

        acts = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.clicked.connect(lambda: self.run_stage.emit(self.stage["id"]))
        self._from_btn = QPushButton("Run from here")
        self._from_btn.clicked.connect(lambda: self.run_from_here.emit(self.stage["id"]))
        acts.addWidget(self._run_btn)
        acts.addWidget(self._from_btn)
        acts.addStretch()
        dl.addLayout(acts)

        self._details.hide()
        lay.addWidget(self._details)

    def _browse_cohort_input(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select cohort file", "", "Spreadsheets (*.xlsx *.xls *.csv)")
        if f:
            self._cohort_input.setText(f)

    def _browse_jess(self):
        f, _ = QFileDialog.getOpenFileName(self, "Select Jess protein file", "", "Spreadsheets (*.xlsx *.xls *.csv)")
        if f:
            self._jess_input.setText(f)

    def _toggle(self):
        self._details.setVisible(self._expand.isChecked())
        self._expand.setText("▴" if self._expand.isChecked() else "▾")

    def set_eta(self, text):
        self._eta.setText(f"ETA: {text}")

    def set_status(self, status):
        icon_map = {
            "pending": ("○", "#888"),
            "running": ("◔", "#e0a400"),
            "done": ("✓", "green"),
            "error": ("✕", "red"),
        }
        icon, color = icon_map.get(status, ("○", "#888"))
        self._icon.setText(icon)
        self._icon.setStyleSheet(f"color:{color};font-weight:bold;")
        self._done_cb.blockSignals(True)
        self._done_cb.setChecked(status == "done")
        self._done_cb.blockSignals(False)

    def set_last_run(self, ts):
        self._ts.setText(f"Last run: {_fmt_ts(ts)}")

    def append_log(self, line):
        self.logs.append(line.rstrip("\n"))
        self._log.setPlainText("\n".join(self.logs))
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def set_enabled(self, enabled):
        self._run_btn.setEnabled(enabled)
        self._from_btn.setEnabled(enabled)


class RunPipelineView(QWidget):
    pipeline_done = pyqtSignal()
    worker_running = pyqtSignal(bool)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._rows: dict[int, StageRow] = {}
        self._worker = None
        self._active_stages = set()
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(12)

        top = QHBoxLayout()
        t = QLabel("Run Pipeline")
        t.setFont(QFont("Arial", 18, QFont.Bold))
        top.addWidget(t)
        top.addStretch()
        self._run_full = QPushButton("Run Full Pipeline")
        self._run_full.clicked.connect(self.run_full_pipeline)
        top.addWidget(self._run_full)
        lay.addLayout(top)

        diag_bar = QHBoxLayout()
        diag_lbl = QLabel("Diagnostics:")
        diag_lbl.setStyleSheet("color:#555;font-size:12px;")
        diag_bar.addWidget(diag_lbl)
        self._diag_cluster_btn = QPushButton("Diagnose Clusters")
        self._diag_cluster_btn.setToolTip("Sweep HDBSCAN min_cluster_size values and report cluster quality metrics")
        self._diag_cluster_btn.clicked.connect(self._run_diagnose_clusters)
        diag_bar.addWidget(self._diag_cluster_btn)
        self._diag_plots_btn = QPushButton("Generate Summary Plots")
        self._diag_plots_btn.setToolTip("Generate paper-ready summary plots from existing results")
        self._diag_plots_btn.clicked.connect(self._run_make_plots)
        diag_bar.addWidget(self._diag_plots_btn)
        diag_bar.addStretch()
        lay.addLayout(diag_bar)

        self._status = QLabel("")
        self._status.setStyleSheet("color:#666;")
        lay.addWidget(self._status)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        holder = QWidget()
        v = QVBoxLayout(holder)
        for stage in STAGES:
            row = StageRow(stage, self.cfg)
            row.run_stage.connect(self._run_stage)
            row.run_from_here.connect(self._run_from_here)
            row.mark_completed.connect(self._mark_completed)
            row.changed.connect(self._param_changed)
            self._rows[stage["id"]] = row
            v.addWidget(row)
        v.addStretch()
        scroll.setWidget(holder)
        lay.addWidget(scroll)

        self._global_log = QTextEdit()
        self._global_log.setReadOnly(True)
        self._global_log.setMaximumHeight(180)
        self._global_log.setStyleSheet("background:#151515;color:#cfd8dc;font-family:Consolas;")
        lay.addWidget(self._global_log)

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
        n_videos = len(idx) if isinstance(idx, dict) else 0
        n_frames = sum(int(v.get("n_frames", 0)) for v in idx.values()) if isinstance(idx, dict) else 0
        eta = {
            1: 1,
            2: max(5, int(n_videos * 0.4)),
            3: max(2, int(n_frames / 120000)),
            4: max(5, int(n_frames / 90000)),
            5: 2,
            6: 3 + (5 if self.cfg.get("export_clips") else 0),
            7: 2,
            8: 1,
            9: 2,
            10: 1,
        }
        for sid, row in self._rows.items():
            row.set_eta(f"~{eta.get(sid, 2)} min")

    def update_from_cfg(self):
        ss = self.cfg.get("stage_status", {})
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
        self._diag_cluster_btn.setEnabled(enabled)
        self._diag_plots_btn.setEnabled(enabled)
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

    def _run_diagnostic(self, script_args, btn_label):
        if self._worker and self._worker.isRunning():
            return
        diag_worker = SubprocessWorker(script_args)
        diag_worker.log.connect(self._append_log)
        self._diag_cluster_btn.setEnabled(False)
        self._diag_plots_btn.setEnabled(False)
        self._status.setText(f"Running {btn_label}…")

        def _done(ok):
            self._diag_cluster_btn.setEnabled(True)
            self._diag_plots_btn.setEnabled(True)
            self._status.setText(f"{btn_label} {'completed.' if ok else 'failed.'}")

        diag_worker.done.connect(_done)
        diag_worker.start()
        self._diag_worker = diag_worker

    def _run_diagnose_clusters(self):
        mcs = self.cfg.get("min_cluster_size", 200)
        self._run_diagnostic(["diagnose_clusters.py", "--min-cluster-sizes", str(mcs)], "Diagnose Clusters")

    def _run_make_plots(self):
        self._run_diagnostic(["make_plots.py"], "Generate Summary Plots")

    def _build_sequence(self, start_sid=1, from_here=False):
        all_ids = [s["id"] for s in STAGES if s["id"] >= start_sid] if from_here else [start_sid]
        # Skip Jess correlation unless a file is configured
        if not self.cfg.get("jess_file", ""):
            all_ids = [i for i in all_ids if i != 10]
        # Skip cohort prep if normalized file already exists and no input file given
        if 1 in all_ids and not self.cfg.get("cohort_input_file", "") and (ROOT / "cohort_normalized.csv").exists():
            all_ids.remove(1)
            self._rows[1].set_status("done")
        # Skip DLC if pose CSVs already exist
        if 2 in all_ids and _has_pose_csvs(Path(self.cfg.get("raw_videos_dir", str(ROOT / "raw_videos")))):
            all_ids.remove(2)
            self._rows[2].set_status("done")
        if from_here:
            done_ids = {
                int(k)
                for k, v in self.cfg.get("stage_status", {}).items()
                if str(v) == "done" and str(k).isdigit()
            }
            all_ids = [sid for sid in all_ids if sid not in done_ids]
        return all_ids

    def run_full_pipeline(self):
        self._start_worker(self._build_sequence(1, from_here=True))

    def _run_stage(self, sid):
        self._start_worker(self._build_sequence(sid, from_here=False))

    def _run_from_here(self, sid):
        self._start_worker(self._build_sequence(sid, from_here=True))


def _thumb_from_video(path: Path | None, size=(180, 110)):
    if not _CV2:
        return None
    if path is None:
        return None
    cap = cv2.VideoCapture(str(path))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    frame = cv2.resize(frame, size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    buf = frame.copy()
    img = QImage(buf.data, size[0], size[1], 3 * size[0], QImage.Format_RGB888)
    return QPixmap.fromImage(img)


class StateDetailDialog(QDialog):
    clips_generated = pyqtSignal()
    request_clip_generation = pyqtSignal(int)

    def __init__(self, sid, s_row=None, c_row=None, bouts_df=None, cfg=None, parent=None):
        super().__init__(parent)
        self.sid = sid
        self.s_row = s_row
        self.c_row = c_row
        self.bouts_df = bouts_df
        self.cfg = cfg or {}
        self._player = None
        self._clip_sections = []
        self.setWindowTitle(f"State {sid} Details")
        self.resize(840, 580)
        self.setMinimumSize(520, 360)
        self.setSizeGripEnabled(True)
        self.setModal(False)
        self._build()

    def _clip_groups(self):
        sd = CLIPS / f"state_{self.sid}"
        if not sd.exists():
            return {}, False
        return {
            "Longest Bouts": sorted(sd.glob("longest_*.mp4"))[:5],
            "Most Typical": sorted(sd.glob("typical_*.mp4"))[:5],
            "Context-Enriched": sorted(sd.glob("context_*.mp4"))[:5],
        }, True

    def _meta_for(self, clip: Path):
        if self.bouts_df is None or self.bouts_df.empty:
            return "duration: -, bout: -, context: -, animal: -, day: -"
        sub = self.bouts_df[self.bouts_df["state"] == self.sid]
        if sub.empty:
            return "duration: -, bout: -, context: -, animal: -, day: -"
        if clip.name.startswith("context_"):
            c = clip.name.split("_")[1]
            sub = sub[sub["context"].astype(str) == c]
        sub = sub.sort_values("duration_sec", ascending=False)
        row = sub.iloc[0]
        return (
            f"duration: {float(row.get('duration_sec', 0)):.2f}s, "
            f"bout: {int(row.get('end_frame', 0) - row.get('start_frame', 0) + 1)} fr, "
            f"context: {row.get('context', '-')}, animal: {row.get('animal_id', '-')}, day: {row.get('day', '-')}"
        )

    def _build_kinematics(self, tabs):
        w = QWidget()
        lay = QVBoxLayout(w)
        if _MPL and self.s_row is not None:
            c = MplCanvas(figsize=(6, 3))
            metrics = {
                "Speed": self.s_row.get("mean_centroid_speed", 0),
                "AngVel": self.s_row.get("mean_angular_vel", 0),
                "BodyLen": self.s_row.get("mean_body_length_px", 0),
                "Elongation": self.s_row.get("mean_elongation", 0),
                "Entropy": self.s_row.get("mean_entropy", 0),
                "BoutSec": self.s_row.get("mean_bout_dur_sec", 0),
            }
            c.ax.bar(list(metrics.keys()), [float(v or 0) for v in metrics.values()], color="#4a90d9")
            c.ax.set_title("Kinematic Profile")
            c.fig.tight_layout()
            lay.addWidget(c)
        else:
            lay.addWidget(QLabel("Run Characterization + Clip Export to generate state kinematics."))
        tabs.addTab(w, "Kinematics")

    def _start_generate(self):
        self.request_clip_generation.emit(self.sid)
        QMessageBox.information(
            self,
            "Background Job Started",
            "Clip generation is running in the background. You can keep navigating the app.",
        )
        self.close()

    def _build_clips(self, tabs):
        w = QWidget()
        lay = QVBoxLayout(w)
        groups, has_any = self._clip_groups()
        self._clip_sections = []
        if has_any and _CV2:
            self._player = VideoPlayer()
            lay.addWidget(self._player)
            for sec, clips in groups.items():
                g = QGroupBox(sec)
                gl = QVBoxLayout(g)
                row_host = QWidget()
                row = QHBoxLayout(row_host)
                row.setContentsMargins(8, 4, 8, 4)
                row.setSpacing(10)
                section_cards = []
                if not clips:
                    row.addWidget(QLabel("No clips in this section."))
                for clip in clips:
                    card = QWidget()
                    cl = QVBoxLayout(card)
                    cl.setContentsMargins(4, 4, 4, 4)
                    cl.setSpacing(4)
                    b = QPushButton("")
                    pm = _thumb_from_video(clip, size=(360, 220))
                    if pm is not None:
                        b.setIcon(QIcon(pm))
                        b.setIconSize(pm.size())
                    b.setMinimumSize(120, 80)
                    b.clicked.connect(lambda _, p=clip: self._player.load(str(p)))
                    name_lbl = QLabel(clip.name)
                    name_lbl.setAlignment(Qt.AlignCenter)
                    name_lbl.setWordWrap(True)
                    meta_lbl = QLabel(self._meta_for(clip))
                    meta_lbl.setWordWrap(True)
                    cl.addWidget(b)
                    cl.addWidget(name_lbl)
                    cl.addWidget(meta_lbl)
                    row.addWidget(card)
                    section_cards.append(
                        {
                            "card": card,
                            "button": b,
                            "pixmap": pm,
                        }
                    )
                row.addStretch()
                sc = QScrollArea()
                sc.setWidgetResizable(True)
                sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
                sc.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                sc.setWidget(row_host)
                gl.addWidget(sc)
                lay.addWidget(g)
                self._clip_sections.append({"scroll": sc, "cards": section_cards})
            first = next((c for clips in groups.values() for c in clips), None)
            if first:
                self._player.load(str(first))
            QTimer.singleShot(0, self._update_clip_card_sizes)
        else:
            lay.addWidget(QLabel("Clips are not available for this state yet."))
            self._gen_btn = QPushButton("Generate clips for this state")
            self._gen_btn.clicked.connect(self._start_generate)
            lay.addWidget(self._gen_btn)
            lay.addWidget(QLabel("Generation runs in background so you can continue browsing."))
        tabs.addTab(w, "Video Clips")

    def _update_clip_card_sizes(self):
        if not self._clip_sections:
            return
        for sec in self._clip_sections:
            cards = sec["cards"]
            if not cards:
                continue
            viewport_w = sec["scroll"].viewport().width()
            n_cards = len(cards)
            avail = max(220, viewport_w - (n_cards + 1) * 10)
            card_w = max(150, min(300, int(avail / max(1, n_cards))))
            thumb_w = int(card_w * 0.92)
            thumb_h = max(88, int(thumb_w * 0.62))
            for item in cards:
                card = item["card"]
                btn = item["button"]
                pm = item["pixmap"]
                card.setFixedWidth(card_w)
                btn.setFixedSize(thumb_w, thumb_h)
                if pm is not None:
                    scaled = pm.scaled(
                        max(40, thumb_w - 8),
                        max(30, thumb_h - 8),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                    btn.setIcon(QIcon(scaled))
                    btn.setIconSize(scaled.size())

    def _build(self):
        lay = QVBoxLayout(self)
        hdr = QLabel(f"State {self.sid}")
        hdr.setFont(QFont("Arial", 14, QFont.Bold))
        lay.addWidget(hdr)
        tabs = QTabWidget()
        self._build_kinematics(tabs)
        self._build_clips(tabs)
        lay.addWidget(tabs)
        btn = QDialogButtonBox(QDialogButtonBox.Close)
        btn.rejected.connect(self.reject)
        lay.addWidget(btn)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update_clip_card_sizes()


class StateCard(QFrame):
    clicked = pyqtSignal(int)

    def __init__(self, sid, s_row=None, c_row=None, strength=None):
        super().__init__()
        self.sid = sid
        self.setFrameShape(QFrame.StyledPanel)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(200, 235)
        self.setStyleSheet(
            "QFrame{background:#fff;border:none;border-radius:8px;}"
            "QFrame:hover{background:#f5f9ff;}"
        )
        lay = QVBoxLayout(self)
        th = QLabel(alignment=Qt.AlignCenter)
        th.setFixedSize(170, 100)
        th.setStyleSheet("background:#eee;border-radius:4px;")
        pm = _thumb_from_video(next(iter((CLIPS / f"state_{sid}").glob("*.mp4")), None)) if (CLIPS / f"state_{sid}").exists() else None
        if pm:
            th.setPixmap(pm)
        else:
            th.setText(f"State {sid}")
        lay.addWidget(th, alignment=Qt.AlignCenter)
        lab = QLabel(f"State {sid}")
        lab.setFont(QFont("Arial", 10, QFont.Bold))
        lay.addWidget(lab)
        if s_row is not None:
            hl = str(s_row.get("heuristic_label", ""))
            lay.addWidget(QLabel((hl[:32] + "...") if len(hl) > 32 else hl))
        if c_row is not None:
            scores = {
                "A": float(c_row.get("A_enrichment", 0) or 0),
                "B": float(c_row.get("B_enrichment", 0) or 0),
                "C": float(c_row.get("C_enrichment", 0) or 0),
            }
            best = max(scores, key=scores.get)
            lay.addWidget(QLabel(f"Context {best} enriched"))
        if strength is not None and not np.isnan(strength):
            lay.addWidget(QLabel(f"Strength: {100 * float(strength):.1f}%"))
        lay.addStretch()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.sid)


class BrowseStatesView(QWidget):
    navigate_to_pipeline = pyqtSignal()
    request_clip_generation = pyqtSignal(int)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._data = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        t = QLabel("Browse States")
        t.setFont(QFont("Arial", 18, QFont.Bold))
        lay.addWidget(t)
        flt = QHBoxLayout()
        flt.addWidget(QLabel("Context"))
        self._context = QComboBox()
        self._context.currentTextChanged.connect(self._rerender)
        flt.addWidget(self._context)
        flt.addWidget(QLabel("Day"))
        self._day = QComboBox()
        self._day.currentTextChanged.connect(self._rerender)
        flt.addWidget(self._day)
        flt.addStretch()
        lay.addLayout(flt)
        self._status = QLabel("No data loaded.")
        lay.addWidget(self._status)
        self._placeholder_btn = QPushButton("Run Characterization + Clip Export")
        self._placeholder_btn.clicked.connect(self.navigate_to_pipeline.emit)
        self._placeholder_btn.hide()
        lay.addWidget(self._placeholder_btn)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        self._gw = QWidget()
        self._grid = QGridLayout(self._gw)
        self._grid.setSpacing(12)
        scroll.setWidget(self._gw)
        lay.addWidget(scroll)
        self._detail_dialogs = []

    def _configure_filters(self, summary):
        self._context.blockSignals(True)
        self._day.blockSignals(True)
        self._context.clear()
        self._day.clear()
        self._context.addItem("All")
        self._day.addItem("All")
        groups = [s.strip() for s in str(self.cfg.get("context_groups", "A,B,C")).split(",") if s.strip()]
        data_ctx = []
        if summary is not None and "context" in summary.columns:
            data_ctx = sorted(summary["context"].dropna().astype(str).unique().tolist())
        existing = set()
        for i in range(self._context.count()):
            existing.add(self._context.itemText(i))
        for g in groups:
            if g not in existing:
                self._context.addItem(g)
                existing.add(g)
        for c in data_ctx:
            if c not in existing:
                self._context.addItem(c)
        if summary is not None and "day" in summary.columns:
            for d in sorted(summary["day"].dropna().astype(str).unique()):
                self._day.addItem(str(d))
        self._context.blockSignals(False)
        self._day.blockSignals(False)

    def _filtered_summary(self):
        summary = self._data.get("summary")
        if summary is None:
            return None
        df = summary.copy()
        ctx = self._context.currentText()
        day = self._day.currentText()
        if ctx and ctx != "All" and "context" in df.columns:
            df = df[df["context"].astype(str) == ctx]
        if day and day != "All" and "day" in df.columns:
            df = df[df["day"].astype(str) == day]
        return df

    def _rerender(self):
        self._render_cards()

    def update_data(self, data):
        self._data = data
        self._configure_filters(data.get("summary"))
        self._render_cards()

    def _render_cards(self):
        while self._grid.count():
            i = self._grid.takeAt(0)
            if i.widget():
                i.widget().deleteLater()
        ci = self._data.get("cluster_info")
        ss = self._data.get("state_summary")
        cr = self._data.get("context_report")
        if not ci:
            self._status.setText("Run HDBSCAN Clustering to generate this data.")
            self._placeholder_btn.show()
            return
        self._placeholder_btn.hide()
        n = int(ci.get("n_clusters", 0))
        if n <= 0:
            self._status.setText("No states discovered.")
            return
        rest = -1
        summary = self._filtered_summary()
        strength_by_state = {}
        if summary is not None and not summary.empty:
            fracs = [(i, float(summary.get(f"state_{i}_frac", pd.Series([0])).mean())) for i in range(n)]
            rest = max(fracs, key=lambda x: x[1])[0]
            strength_by_state = {i: v for i, v in fracs}
        ctx = self._context.currentText() or "All"
        day = self._day.currentText() or "All"
        self._status.setText(f"{n} states discovered. Context: {ctx}, Day: {day}. Dominant state {rest} excluded.")
        row = col = 0
        for sid in range(n):
            if sid == rest:
                continue
            s_row = None
            c_row = None
            if ss is not None and "state" in ss.columns:
                r = ss[ss["state"] == sid]
                if not r.empty:
                    s_row = r.iloc[0]
            if cr is not None and "state" in cr.columns:
                r = cr[cr["state"] == sid]
                if not r.empty:
                    c_row = r.iloc[0]
            c = StateCard(sid, s_row, c_row, strength=strength_by_state.get(sid))
            c.clicked.connect(self._open_detail)
            self._grid.addWidget(c, row, col)
            col += 1
            if col >= 5:
                col = 0
                row += 1

    def _open_detail(self, sid):
        ss = self._data.get("state_summary")
        cr = self._data.get("context_report")
        bouts = self._data.get("bouts")
        s_row = c_row = None
        if ss is not None and "state" in ss.columns:
            r = ss[ss["state"] == sid]
            if not r.empty:
                s_row = r.iloc[0]
        if cr is not None and "state" in cr.columns:
            r = cr[cr["state"] == sid]
            if not r.empty:
                c_row = r.iloc[0]
        dlg = StateDetailDialog(sid, s_row, c_row, bouts_df=bouts, cfg=self.cfg, parent=self)
        dlg.request_clip_generation.connect(self.request_clip_generation.emit)
        dlg.clips_generated.connect(lambda: None)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        self._detail_dialogs.append(dlg)
        self._detail_dialogs = [d for d in self._detail_dialogs if d.isVisible()]


class ValidationView(QWidget):
    navigate_to_pipeline = pyqtSignal()

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._data = {}
        self._sample = None
        self._cursor = 0
        self._pose_cache = {}
        self._cap_cache = {}
        self._feature_cache = {}
        self._label_map = {
            Qt.Key_F: "freeze",
            Qt.Key_W: "walk",
            Qt.Key_G: "groom",
            Qt.Key_R: "rear",
            Qt.Key_O: "other",
            Qt.Key_S: "skip",
        }
        self.setFocusPolicy(Qt.StrongFocus)
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        title = QLabel("Validation")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        outer.addWidget(title)

        split = QHBoxLayout()
        outer.addLayout(split, stretch=1)

        left = QGroupBox("Frame Sampler")
        ll = QVBoxLayout(left)
        self._video_combo = QComboBox()
        self._state_combo = QComboBox()
        self._n_slider = QSlider(Qt.Horizontal)
        self._n_slider.setRange(10, 200)
        self._n_slider.setValue(50)
        self._n_lbl = QLabel("Frames: 50")
        self._n_slider.valueChanged.connect(lambda v: self._n_lbl.setText(f"Frames: {v}"))
        self._sample_btn = QPushButton("Sample Frames From Video")
        self._sample_btn.clicked.connect(self._sample_frames)
        ll.addWidget(QLabel("Video to Validate"))
        ll.addWidget(self._video_combo)
        ll.addWidget(QLabel("State"))
        ll.addWidget(self._state_combo)
        ll.addWidget(self._n_lbl)
        ll.addWidget(self._n_slider)
        ll.addWidget(self._sample_btn)
        self._progress_lbl = QLabel("0 of 0 frames labeled")
        ll.addWidget(self._progress_lbl)
        split.addWidget(left, stretch=1)

        center = QGroupBox("Frame Display")
        cl = QVBoxLayout(center)
        self._frame = QLabel("Select a video and sample frames to begin", alignment=Qt.AlignCenter)
        self._frame.setMinimumSize(540, 360)
        self._frame.setStyleSheet("background:#111;color:#999;")
        cl.addWidget(self._frame)
        self._frame_info = QLabel("State: - | kinematics: -")
        cl.addWidget(self._frame_info)
        self._shortcuts = QLabel("Shortcuts: F=freeze, W=walk, G=groom, R=rear, O=other, S=skip")
        self._shortcuts.setStyleSheet("color:#666;")
        cl.addWidget(self._shortcuts)
        split.addWidget(center, stretch=2)

        right = QGroupBox("Label Assignment")
        rl = QVBoxLayout(right)
        for name in ("Freeze", "Walk", "Groom", "Rear", "Other"):
            b = QPushButton(name)
            b.setMinimumHeight(44)
            b.clicked.connect(lambda _, n=name.lower(): self._assign(n))
            rl.addWidget(b)
        skip = QPushButton("Skip")
        skip.setMinimumHeight(44)
        skip.clicked.connect(lambda: self._assign("skip"))
        rl.addWidget(skip)
        rl.addStretch()
        split.addWidget(right, stretch=1)

        bottom = QGroupBox("Results")
        bl = QVBoxLayout(bottom)
        if _MPL:
            self._cm_canvas = MplCanvas(figsize=(5, 3))
            bl.addWidget(self._cm_canvas)
        else:
            self._cm_canvas = None
            bl.addWidget(QLabel("Install matplotlib for confusion matrix heatmap."))
        self._agree = QLabel("Agreement per state: -")
        bl.addWidget(self._agree)
        self._export_btn = QPushButton("Export labels CSV")
        self._export_btn.clicked.connect(self._export_validation)
        bl.addWidget(self._export_btn)
        outer.addWidget(bottom)

    def update_data(self, data):
        self._data = data
        ci = data.get("cluster_info")
        lpf = data.get("labels_per_frame")
        if ci is None or lpf is None or lpf.empty:
            self._sample_btn.setEnabled(False)
            self._progress_lbl.setText("Run Characterization + Clip Export to generate this data.")
            return
        self._sample_btn.setEnabled(True)
        n = int(ci.get("n_clusters", 0))
        summary = data.get("summary")
        dominant = -1
        if summary is not None:
            dominant = max(
                [(i, float(summary.get(f"state_{i}_frac", pd.Series([0])).mean())) for i in range(n)],
                key=lambda x: x[1],
            )[0]
        self._state_combo.clear()
        for sid in range(n):
            if sid != dominant:
                self._state_combo.addItem(f"State {sid}", sid)
        self._video_combo.clear()
        if "stem" in lpf.columns:
            stems = sorted(lpf["stem"].dropna().astype(str).unique().tolist())
            for s in stems:
                self._video_combo.addItem(s, s)
        sample = data.get("validation_sample")
        if sample is not None and not sample.empty:
            # Resume previous work only when same selected video/state exists in sample
            current_video = self._video_combo.currentData() if self._video_combo.count() else None
            current_state = self._state_combo.currentData() if self._state_combo.count() else None
            resumed = sample
            if current_video is not None and "stem" in sample.columns:
                resumed = resumed[resumed["stem"].astype(str) == str(current_video)]
            if current_state is not None and "cluster_label" in resumed.columns:
                resumed = resumed[resumed["cluster_label"] == int(current_state)]
            self._sample = resumed.reset_index(drop=True) if not resumed.empty else None
            if self._sample is not None:
                self._cursor = int((self._sample["manual_label"].fillna("") != "").sum())
                self._show_current()
                self._refresh_results()

    def _sample_frames(self):
        lpf = self._data.get("labels_per_frame")
        fi = self._data.get("feature_index") or {}
        if lpf is None or lpf.empty:
            return
        stem = self._video_combo.currentData()
        if not stem:
            QMessageBox.information(self, "Validation", "Select a video to validate.")
            return
        sid = int(self._state_combo.currentData())
        n = int(self._n_slider.value())
        sub = lpf[(lpf["state"] == sid) & (lpf["stem"].astype(str) == str(stem))]
        if sub.empty:
            QMessageBox.information(self, "Validation", "No frames available for this state in selected video.")
            return
        if len(sub) > n:
            sub = sub.sample(n=n, random_state=42)
        sub = sub.copy()
        sub.rename(columns={"frame": "frame_idx"}, inplace=True)
        sub["cluster_label"] = sub["state"]
        sub["manual_label"] = ""
        sub["timestamp"] = ""
        for idx, row in sub.iterrows():
            stem = row["stem"]
            info = fi.get(stem, {}) if isinstance(fi, dict) else {}
            sub.at[idx, "video_path"] = info.get("video_path", str(ROOT / "raw_videos" / f"{stem}.mp4"))
            sub.at[idx, "csv_path"] = info.get("csv_path", "")
        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        sub.to_csv(VALIDATION_DIR / "current_sample.csv", index=False)
        self._sample = sub.reset_index(drop=True)
        self._cursor = 0
        self._show_current()
        self._refresh_results()

    def _load_pose(self, csv_path):
        if csv_path in self._pose_cache:
            return self._pose_cache[csv_path]
        try:
            from main import load_pose

            pose, conf, _ = load_pose(csv_path)
            self._pose_cache[csv_path] = (pose, conf)
            return pose, conf
        except Exception:
            return None, None

    def _draw_frame(self, row):
        if not _CV2:
            return
        video = row.get("video_path", "")
        frame_idx = int(row.get("frame_idx", 0))
        csv_path = row.get("csv_path", "")
        cap = self._cap_cache.get(video)
        if cap is None:
            cap = cv2.VideoCapture(str(video))
            self._cap_cache[video] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            self._frame.setText("Could not load frame.")
            return
        pose, _ = self._load_pose(csv_path)
        if pose is not None and frame_idx < len(pose):
            pts = pose[frame_idx]
            colors = [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
                (255, 0, 255),
                (0, 255, 255),
                (255, 128, 0),
                (128, 0, 255),
            ]
            for i, pt in enumerate(pts):
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(frame, (x, y), 4, colors[i], -1)
            edges = [(2, 3), (3, 6), (0, 1)]
            for a, b in edges:
                pa = tuple(np.int32(pts[a]))
                pb = tuple(np.int32(pts[b]))
                cv2.line(frame, pa, pb, (255, 255, 255), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        mxw, mxh = self._frame.width(), self._frame.height()
        sc = min(mxw / w, mxh / h)
        nw, nh = int(w * sc), int(h * sc)
        frame = cv2.resize(frame, (nw, nh))
        buf = frame.copy()
        img = QImage(buf.data, nw, nh, 3 * nw, QImage.Format_RGB888)
        self._frame.setPixmap(QPixmap.fromImage(img))

    def _kinematic_text(self, row):
        fi = self._data.get("feature_index") or {}
        stem = row.get("stem", "")
        frame_idx = int(row.get("frame_idx", 0))
        info = fi.get(stem, {}) if isinstance(fi, dict) else {}
        fp = info.get("features_path", "")
        if not fp:
            return "-"
        arr = self._feature_cache.get(fp)
        if arr is None and Path(fp).exists():
            arr = np.load(fp)
            self._feature_cache[fp] = arr
        if arr is None or frame_idx >= len(arr):
            return "-"
        feat = arr[frame_idx]
        return f"speed={feat[36]:.3f}, ang_vel={feat[39]:.3f}, entropy={feat[40]:.3f}"

    def _show_current(self):
        if self._sample is None or self._sample.empty:
            self._frame.setText("No sample loaded.")
            return
        unl = self._sample["manual_label"].fillna("") == ""
        if unl.sum() == 0:
            self._frame.setText("All frames labeled.")
            return
        idxs = self._sample.index[unl]
        self._cursor = int(idxs[0])
        row = self._sample.loc[self._cursor]
        self._draw_frame(row)
        self._frame_info.setText(
            f"State {int(row.get('cluster_label', -1))} | "
            f"frame {int(row.get('frame_idx', 0))} | {self._kinematic_text(row)}"
        )
        done = int((self._sample["manual_label"].fillna("") != "").sum())
        self._progress_lbl.setText(f"{done} of {len(self._sample)} frames labeled")

    def _assign(self, manual_label):
        if self._sample is None or self._sample.empty:
            return
        self._sample.at[self._cursor, "manual_label"] = manual_label
        self._sample.at[self._cursor, "timestamp"] = datetime.now().isoformat(timespec="seconds")
        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        self._sample.to_csv(VALIDATION_DIR / "current_sample.csv", index=False)
        self._sample.to_csv(VALIDATION_DIR / "frame_labels.csv", index=False)
        self._refresh_results()
        self._show_current()

    def _refresh_results(self):
        if self._sample is None or self._sample.empty:
            return
        done = self._sample[self._sample["manual_label"].fillna("") != ""]
        if done.empty:
            self._agree.setText("Agreement per state: -")
            if self._cm_canvas:
                self._cm_canvas.ax.clear()
                self._cm_canvas.ax.text(0.5, 0.5, "No labels yet", ha="center", va="center")
                self._cm_canvas.draw()
            return
        cros = pd.crosstab(done["cluster_label"], done["manual_label"])
        if self._cm_canvas:
            self._cm_canvas.ax.clear()
            self._cm_canvas.ax.imshow(cros.values, aspect="auto", cmap="Blues")
            self._cm_canvas.ax.set_xticks(range(len(cros.columns)))
            self._cm_canvas.ax.set_xticklabels(cros.columns, rotation=45, ha="right")
            self._cm_canvas.ax.set_yticks(range(len(cros.index)))
            self._cm_canvas.ax.set_yticklabels(cros.index)
            self._cm_canvas.ax.set_xlabel("Manual Label")
            self._cm_canvas.ax.set_ylabel("Cluster")
            self._cm_canvas.fig.tight_layout()
            self._cm_canvas.draw()
        agreements = []
        for sid, grp in done.groupby("cluster_label"):
            top = grp["manual_label"].value_counts().max()
            agreements.append(f"S{sid}: {100 * top / len(grp):.1f}%")
        self._agree.setText("Agreement per state: " + ", ".join(agreements))

    def _export_validation(self):
        p = VALIDATION_DIR / "frame_labels.csv"
        if not p.exists():
            QMessageBox.information(self, "Validation", "No labels to export yet.")
            return
        d = QFileDialog.getExistingDirectory(self, "Select Destination", str(ROOT))
        if not d:
            return
        dst = Path(d) / "frame_labels.csv"
        shutil.copy2(p, dst)
        QMessageBox.information(self, "Validation", f"Exported to {dst}")

    def keyPressEvent(self, e):
        if e.key() in self._label_map:
            self._assign(self._label_map[e.key()])
            e.accept()
            return
        super().keyPressEvent(e)


class TransitionMatrixView(QWidget):
    def __init__(self):
        super().__init__()
        self._data = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        t = QLabel("Transition Matrix")
        t.setFont(QFont("Arial", 18, QFont.Bold))
        lay.addWidget(t)
        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Filter by"))
        self._f_combo = QComboBox()
        self._f_combo.addItems(["All", "Day", "Experiment"])
        self._f_combo.currentTextChanged.connect(self._update_vals)
        ctrl.addWidget(self._f_combo)
        self._v_combo = QComboBox()
        self._v_combo.currentTextChanged.connect(self._plot)
        ctrl.addWidget(self._v_combo)
        ctrl.addStretch()
        lay.addLayout(ctrl)
        if _MPL:
            self._canvas = MplCanvas(figsize=(12, 5))
            lay.addWidget(self._canvas)
        else:
            self._canvas = None
            lay.addWidget(QLabel("Install matplotlib."))

    def update_data(self, data):
        self._data = data
        self._update_vals(self._f_combo.currentText())
        self._plot()

    def _update_vals(self, f):
        self._v_combo.clear()
        summary = self._data.get("summary")
        if summary is None:
            self._v_combo.addItem("All")
            return
        if f == "Day" and "day" in summary.columns:
            vals = ["All"] + [str(v) for v in sorted(summary["day"].dropna().unique())]
        elif f == "Experiment" and "experiment" in summary.columns:
            vals = ["All"] + sorted(summary["experiment"].dropna().astype(str).unique())
        else:
            vals = ["All"]
        self._v_combo.addItems(vals)

    def _plot(self):
        if not self._canvas:
            return
        png = RESULTS / "comparison" / "transition_by_context.png"
        summary = self._data.get("summary")
        ci = self._data.get("cluster_info")
        ftype = self._f_combo.currentText()
        val = self._v_combo.currentText()
        if png.exists() and (ftype == "All" or val == "All"):
            img = mpimg.imread(str(png))
            self._canvas.ax.clear()
            self._canvas.ax.imshow(img)
            self._canvas.ax.axis("off")
            self._canvas.draw()
            return
        if summary is None or ci is None:
            self._canvas.ax.clear()
            self._canvas.ax.text(0.5, 0.5, "Run Report Generation to generate this data.", ha="center", va="center")
            self._canvas.draw()
            return
        n = int(ci.get("n_clusters", 0))
        df = summary.copy()
        if ftype == "Day" and val != "All" and "day" in df.columns:
            df = df[df["day"].astype(str) == val]
        if ftype == "Experiment" and val != "All" and "experiment" in df.columns:
            df = df[df["experiment"].astype(str) == val]
        if "context" not in df.columns:
            return
        state_cols = [f"state_{i}_frac" for i in range(n) if f"state_{i}_frac" in df.columns]
        contexts = sorted(df["context"].dropna().unique())[:2]
        self._canvas.fig.clear()
        for i, ctx in enumerate(contexts):
            vals = df[df["context"] == ctx][state_cols].mean().values
            mat = np.outer(vals, vals)
            if mat.max() > 0:
                mat = mat / mat.max()
            ax = self._canvas.fig.add_subplot(1, len(contexts), i + 1)
            im = ax.imshow(mat, cmap="Blues", aspect="auto")
            ax.set_title(f"Context {ctx}")
            ax.set_xlabel("To")
            ax.set_ylabel("From")
            self._canvas.fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        self._canvas.fig.tight_layout()
        self._canvas.draw()


class MotifsView(QWidget):
    navigate_to_pipeline = pyqtSignal()

    def __init__(self):
        super().__init__()
        self._data = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        t = QLabel("Motifs")
        t.setFont(QFont("Arial", 18, QFont.Bold))
        lay.addWidget(t)
        top = QHBoxLayout()
        top.addWidget(QLabel("Type"))
        self._type = QComboBox()
        self._type.addItems(["All", "bigram", "trigram"])
        self._type.currentTextChanged.connect(self._refresh)
        top.addWidget(self._type)
        top.addStretch()
        lay.addLayout(top)
        self._summary = QLabel("Motif summary: -")
        lay.addWidget(self._summary)
        if _MPL:
            self._canvas = MplCanvas(figsize=(9, 3))
            lay.addWidget(self._canvas)
        else:
            self._canvas = None
        self._table = QTableWidget()
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(
            ["Motif", "Type", "Context A Freq", "Context B Freq", "Enrichment", "CI"]
        )
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setSortingEnabled(True)
        lay.addWidget(self._table)
        self._empty = QLabel("Run Motif Discovery to generate this data.")
        self._empty.setAlignment(Qt.AlignCenter)
        self._empty.setStyleSheet("color:#777;")
        lay.addWidget(self._empty)
        self._btn = QPushButton("Go to Run Pipeline")
        self._btn.clicked.connect(self.navigate_to_pipeline.emit)
        lay.addWidget(self._btn)
        self._heat_lbl = QLabel()
        self._heat_lbl.setAlignment(Qt.AlignCenter)
        self._heat_lbl.hide()
        lay.addWidget(self._heat_lbl)

    def update_data(self, data):
        self._data = data
        self._refresh()

    def _refresh(self):
        motifs = self._data.get("motifs") if isinstance(self._data, dict) else None
        if motifs is None or motifs.empty:
            self._table.hide()
            self._empty.show()
            self._btn.show()
            self._summary.setText("Motif summary: no data")
            if self._canvas:
                self._canvas.ax.clear()
                self._canvas.ax.text(0.5, 0.5, "No motif data", ha="center", va="center")
                self._canvas.draw()
            return
        m = motifs.copy()
        t = self._type.currentText()
        if t != "All" and "type" in m.columns:
            m = m[m["type"].astype(str) == t]
        self._table.show()
        self._empty.hide()
        self._btn.hide()
        self._table.setRowCount(len(m))
        cols = [
            ("motif", 0),
            ("type", 1),
            ("context_A_freq", 2),
            ("context_B_freq", 3),
            ("enrichment_ratio", 4),
            ("ci_low", 5),
        ]
        for r, (_, row) in enumerate(m.iterrows()):
            for c, idx in cols:
                if c == "ci_low":
                    txt = f"[{row.get('ci_low', '')}, {row.get('ci_high', '')}]"
                else:
                    txt = str(row.get(c, ""))
                it = QTableWidgetItem(txt)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self._table.setItem(r, idx, it)
        if "flagged" in m.columns:
            n_flag = int(m["flagged"].fillna(False).sum())
        else:
            n_flag = int((m.get("enrichment_ratio", pd.Series(dtype=float)) >= 2.0).sum())
        self._summary.setText(f"Motif summary: {len(m)} motifs, {n_flag} enriched candidates")
        if self._canvas:
            self._canvas.ax.clear()
            top = m.sort_values("enrichment_ratio", ascending=False).head(10)
            labels = [str(v) for v in top.get("motif", pd.Series(dtype=str)).tolist()]
            vals = [float(v) for v in top.get("enrichment_ratio", pd.Series(dtype=float)).fillna(0).tolist()]
            if labels:
                self._canvas.ax.barh(range(len(labels)), vals, color="#4a90d9")
                self._canvas.ax.set_yticks(range(len(labels)))
                self._canvas.ax.set_yticklabels([l[:24] + ("..." if len(l) > 24 else "") for l in labels], fontsize=8)
                self._canvas.ax.invert_yaxis()
                self._canvas.ax.set_xlabel("Enrichment Ratio")
                self._canvas.ax.set_title("Top Enriched Motifs")
            else:
                self._canvas.ax.text(0.5, 0.5, "No motifs for selected type", ha="center", va="center")
            self._canvas.fig.tight_layout()
            self._canvas.draw()
        heat = RESULTS / "comparison" / "motif_heatmap.png"
        if heat.exists():
            pm = QPixmap(str(heat))
            if not pm.isNull():
                self._heat_lbl.setPixmap(pm.scaledToWidth(680, Qt.SmoothTransformation))
                self._heat_lbl.show()
            else:
                self._heat_lbl.hide()
        else:
            self._heat_lbl.hide()


class AnimalExplorerView(QWidget):
    def __init__(self):
        super().__init__()
        self._data = {}
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        t = QLabel("Animal Explorer")
        t.setFont(QFont("Arial", 18, QFont.Bold))
        lay.addWidget(t)
        top = QHBoxLayout()
        top.addWidget(QLabel("Animal ID"))
        self._combo = QComboBox()
        self._combo.currentTextChanged.connect(self._update)
        top.addWidget(self._combo)
        self._all_animals = QCheckBox("Plot all animals")
        self._all_animals.toggled.connect(lambda _: self._update(self._combo.currentText()))
        top.addWidget(self._all_animals)
        top.addWidget(QLabel("Focus State"))
        self._focus_state = QComboBox()
        self._focus_state.currentTextChanged.connect(lambda _: self._update(self._combo.currentText()))
        top.addWidget(self._focus_state)
        top.addStretch()
        lay.addLayout(top)

        if _MPL:
            self._line = MplCanvas(figsize=(10, 3))
            lay.addWidget(self._line)
        else:
            self._line = None
            lay.addWidget(QLabel("Install matplotlib to view charts."))

        panels = QTabWidget()
        lay.addWidget(panels, stretch=1)

        self._session_table = QTableWidget()
        self._session_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._session_table.cellDoubleClicked.connect(self._open_row_folder)
        panels.addTab(self._session_table, "Session Table")

        self._disc_w = QWidget()
        dl = QVBoxLayout(self._disc_w)
        if _MPL:
            self._disc_canvas = MplCanvas(figsize=(10, 3))
            dl.addWidget(self._disc_canvas)
        else:
            self._disc_canvas = None
            dl.addWidget(QLabel("Install matplotlib to view chart."))
        panels.addTab(self._disc_w, "Discrimination Trajectory")

        self._heat_w = QWidget()
        hl = QVBoxLayout(self._heat_w)
        if _MPL:
            self._heat_canvas = MplCanvas(figsize=(5, 4))
            hl.addWidget(self._heat_canvas)
        else:
            self._heat_canvas = None
            hl.addWidget(QLabel("Install matplotlib to view chart."))
        panels.addTab(self._heat_w, "Transition Heatmap")

    def update_data(self, data):
        self._data = data
        summary = data.get("summary")
        if summary is None or "animal_id" not in summary.columns:
            return
        self._combo.blockSignals(True)
        self._combo.clear()
        animals = sorted(summary["animal_id"].dropna().astype(str).unique())
        self._combo.addItems(animals)
        self._combo.blockSignals(False)
        ci = data.get("cluster_info") or {}
        n = int(ci.get("n_clusters", 0))
        self._focus_state.blockSignals(True)
        self._focus_state.clear()
        for sid in range(n):
            self._focus_state.addItem(f"State {sid}", sid)
        self._focus_state.blockSignals(False)
        if animals:
            self._update(animals[0])

    def _open_row_folder(self, row, _col):
        stem_item = self._session_table.item(row, 0)
        if not stem_item:
            return
        stem = stem_item.text()
        path = CLIPS / stem
        if not path.exists():
            path = CLIPS
        _open_folder(path)

    def _update(self, animal_id):
        summary = self._data.get("summary")
        ci = self._data.get("cluster_info")
        if summary is None or not animal_id:
            return
        sub = summary[summary["animal_id"].astype(str) == str(animal_id)].copy()
        if sub.empty:
            return
        n = int(ci.get("n_clusters", 0)) if ci else 0
        state_cols = [f"state_{i}_frac" for i in range(n) if f"state_{i}_frac" in sub.columns]
        noise = 1 - sub[state_cols].sum(axis=1) if state_cols else pd.Series([0] * len(sub))

        headers = ["stem", "date", "day", "context", "experiment"] + state_cols + ["noise_frac"]
        self._session_table.setColumnCount(len(headers))
        self._session_table.setHorizontalHeaderLabels(headers)
        self._session_table.setRowCount(len(sub))
        for r, (_, row) in enumerate(sub.iterrows()):
            for c, h in enumerate(headers):
                if h == "noise_frac":
                    txt = f"{float(noise.iloc[r]):.3f}"
                else:
                    txt = str(row.get(h, ""))
                it = QTableWidgetItem(txt)
                it.setFlags(it.flags() & ~Qt.ItemIsEditable)
                self._session_table.setItem(r, c, it)
        self._session_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        if self._line:
            self._line.ax.clear()
            if state_cols and "day" in summary.columns:
                if self._all_animals.isChecked():
                    sid = int(self._focus_state.currentData() or 0)
                    col = f"state_{sid}_frac"
                    if col in summary.columns:
                        colors = mpl_cm.tab20(np.linspace(0, 1, max(1, summary["animal_id"].nunique())))
                        for i, (aid, grp) in enumerate(summary.groupby("animal_id")):
                            if grp.empty:
                                continue
                            by_day = grp.groupby("day")[col].mean()
                            self._line.ax.plot(
                                by_day.index,
                                by_day.values,
                                label=str(aid),
                                color=colors[i % len(colors)],
                                linewidth=1.4,
                                alpha=0.85,
                            )
                        self._line.ax.set_title(f"State {sid} Occupancy - All Animals")
                        self._line.ax.legend(fontsize=7, ncol=2)
                else:
                    by_day = sub.groupby("day")[state_cols].mean()
                    colors = mpl_cm.tab20(np.linspace(0, 1, max(1, len(state_cols))))
                    for i, col in enumerate(state_cols[:20]):
                        sid = int(col.split("_")[1])
                        self._line.ax.plot(days := by_day.index, by_day[col], label=f"S{sid}", color=colors[i], linewidth=1.4)
                    self._line.ax.set_title(f"State Occupancy - Animal {animal_id}")
                self._line.ax.set_xlabel("Day")
                self._line.ax.set_ylabel("Fraction")
                self._line.fig.tight_layout()
            self._line.draw()

        if self._disc_canvas:
            self._disc_canvas.ax.clear()
            ratios = []
            days = sorted(sub["day"].dropna().unique()) if "day" in sub.columns else []
            for day in days:
                g = sub[sub["day"] == day]
                if "context" in g.columns:
                    cols = [c for c in state_cols if c in g.columns]
                    if not cols:
                        continue
                    freeze_col = cols[0]
                    a = g[g["context"] == "A"][freeze_col].mean()
                    b = g[g["context"] == "B"][freeze_col].mean()
                    if pd.notna(a) and pd.notna(b) and (a + b) > 0:
                        ratios.append(((a - b) / (a + b), day))
            if ratios:
                xs = [d for _, d in ratios]
                ys = [v for v, _ in ratios]
                self._disc_canvas.ax.plot(xs, ys, marker="o")
                self._disc_canvas.ax.axhline(0, color="gray", linestyle="--")
            scalars = self._data.get("animal_scalars")
            auc = np.nan
            if scalars is not None and "animal_id" in scalars.columns:
                r = scalars[scalars["animal_id"].astype(str) == str(animal_id)]
                if not r.empty:
                    auc = float(r.iloc[0].get("freeze_auc", np.nan))
            self._disc_canvas.ax.set_title(f"Discrimination Trajectory (AUC={auc:.3f})")
            self._disc_canvas.ax.set_xlabel("Day")
            self._disc_canvas.ax.set_ylabel("Discrimination Ratio")
            self._disc_canvas.fig.tight_layout()
            self._disc_canvas.draw()

        if self._heat_canvas:
            self._heat_canvas.ax.clear()
            tt = self._data.get("transition_table")
            if tt is not None and "animal_id" in tt.columns:
                sub_t = tt[tt["animal_id"].astype(str) == str(animal_id)]
                # Derive actual state count from column names rather than n_clusters,
                # which can differ from the transition table when HDBSCAN produces noise labels.
                present = set()
                for col in sub_t.columns:
                    if col.startswith("trans_"):
                        parts = col.split("_")
                        if len(parts) == 3:
                            try:
                                present.add(int(parts[1]))
                                present.add(int(parts[2]))
                            except ValueError:
                                pass
                n_actual = max(present) + 1 if present else 0
                if n_actual > 0:
                    mat = np.zeros((n_actual, n_actual))
                    trans_cols = [f"trans_{i}_{j}" for i in range(n_actual) for j in range(n_actual)
                                  if f"trans_{i}_{j}" in sub_t.columns]
                    means = sub_t[trans_cols].mean()
                    for col in trans_cols:
                        parts = col.split("_")
                        mat[int(parts[1]), int(parts[2])] = means[col]
                    im = self._heat_canvas.ax.imshow(mat, cmap="Blues", aspect="auto")
                    self._heat_canvas.ax.set_title("Mean Transition Matrix")
                    self._heat_canvas.ax.set_xlabel("To")
                    self._heat_canvas.ax.set_ylabel("From")
                    self._heat_canvas.fig.colorbar(im, ax=self._heat_canvas.ax, fraction=0.046, pad=0.04)
            self._heat_canvas.fig.tight_layout()
            self._heat_canvas.draw()


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
        r = 0

        def row(label, widget):
            nonlocal r
            form.addWidget(QLabel(label), r, 0)
            form.addWidget(widget, r, 1)
            r += 1

        ab = cfg.get("arena_bounds", _DEFAULT_CFG["arena_bounds"])
        self._xmin = QSpinBox(); self._xmin.setRange(0, 9999); self._xmin.setValue(ab["x_min"])
        self._ymin = QSpinBox(); self._ymin.setRange(0, 9999); self._ymin.setValue(ab["y_min"])
        self._xmax = QSpinBox(); self._xmax.setRange(0, 9999); self._xmax.setValue(ab["x_max"])
        self._ymax = QSpinBox(); self._ymax.setRange(0, 9999); self._ymax.setValue(ab["y_max"])
        row("Arena x_min", self._xmin)
        row("Arena y_min", self._ymin)
        row("Arena x_max", self._xmax)
        row("Arena y_max", self._ymax)

        def dir_row(label, key):
            nonlocal r
            le = QLineEdit(self.cfg.get(key, ""))
            b = QPushButton("Browse...")
            b.clicked.connect(lambda: self._browse(le))
            h = QHBoxLayout()
            h.addWidget(le)
            h.addWidget(b)
            form.addWidget(QLabel(label), r, 0)
            form.addLayout(h, r, 1)
            r += 1
            return le

        self._results = dir_row("Results directory", "results_dir")
        self._raw = dir_row("Raw videos directory", "raw_videos_dir")

        def file_row(label, key, filt="All files (*.*)"):
            nonlocal r
            le = QLineEdit(self.cfg.get(key, ""))
            b = QPushButton("Browse…")
            b.clicked.connect(lambda: self._browse_file(le, filt))
            h = QHBoxLayout()
            h.addWidget(le)
            h.addWidget(b)
            form.addWidget(QLabel(label), r, 0)
            form.addLayout(h, r, 1)
            r += 1
            return le

        self._cohort_input = file_row("Cohort input file (Excel/CSV)", "cohort_input_file", "Spreadsheets (*.xlsx *.xls *.csv)")
        self._cohort_file = file_row("Cohort normalized file (output)", "cohort_file", "CSV files (*.csv)")
        self._jess_file = file_row("Jess protein file (optional)", "jess_file", "Spreadsheets (*.xlsx *.xls *.csv)")

        self._ctx_groups = QLineEdit(str(self.cfg.get("context_groups", "A,B,C")))
        row("Context groups (comma-separated)", self._ctx_groups)
        self._fps = QSpinBox()
        self._fps.setRange(1, 256)
        self._fps.setValue(int(self.cfg.get("fps", 30)))
        row("FPS", self._fps)
        lay.addLayout(form)

        save = QPushButton("Save Settings")
        save.clicked.connect(self._save)
        lay.addWidget(save)
        lay.addStretch()

    def _browse(self, le):
        d = QFileDialog.getExistingDirectory(self, "Select Directory", le.text())
        if d:
            le.setText(d)

    def _browse_file(self, le, filt):
        f, _ = QFileDialog.getOpenFileName(self, "Select file", le.text(), filt)
        if f:
            le.setText(f)

    def _save(self):
        self.cfg["arena_bounds"] = {
            "x_min": self._xmin.value(),
            "y_min": self._ymin.value(),
            "x_max": self._xmax.value(),
            "y_max": self._ymax.value(),
        }
        self.cfg["results_dir"] = self._results.text()
        self.cfg["raw_videos_dir"] = self._raw.text()
        self.cfg["cohort_input_file"] = self._cohort_input.text().strip()
        self.cfg["cohort_file"] = self._cohort_file.text().strip() or "cohort_normalized.csv"
        self.cfg["jess_file"] = self._jess_file.text().strip()
        self.cfg["context_groups"] = self._ctx_groups.text().strip() or "A,B,C"
        self.cfg["fps"] = self._fps.value()
        _save_cfg(self.cfg)
        self.settings_changed.emit(self.cfg)
        QMessageBox.information(self, "Settings", "Saved.")


class NavBtn(QPushButton):
    def __init__(self, text):
        super().__init__(text)
        self.setCheckable(True)
        self.setFixedHeight(42)
        self.setStyleSheet(
            """
            QPushButton{text-align:left;padding-left:20px;border:none;background:transparent;font-size:13px;}
            QPushButton:hover{background:#e8f0fe;}
            QPushButton:checked{background:#d2e3fc;color:#1a73e8;font-weight:bold;border-left:3px solid #1a73e8;}
            """
        )


class OnboardingDialog(QDialog):
    completed = pyqtSignal()

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._worker = None
        self._pipeline = None
        self.setWindowTitle("Welcome to VIEB")
        self.setModal(True)
        self.setWindowFlags(self.windowFlags() | Qt.WindowMaximizeButtonHint)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(30, 30, 30, 30)
        self._steps = QLabel("Step 1 - Project Setup   >   Step 2 - Pose Estimation   >   Step 3 - Pipeline Configuration")
        self._steps.setFont(QFont("Arial", 12, QFont.Bold))
        lay.addWidget(self._steps)
        self._stack = QStackedWidget()
        lay.addWidget(self._stack, stretch=1)
        self._build_step1()
        self._build_step2()
        self._build_step3()
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(150)
        self._log.setStyleSheet("background:#111;color:#ddd;")
        lay.addWidget(self._log)

    def _build_step1(self):
        w = QWidget()
        l = QVBoxLayout(w)
        l.addWidget(QLabel("Raw videos directory"))
        h = QHBoxLayout()
        self._raw_dir = QLineEdit(self.cfg.get("raw_videos_dir", str(ROOT / "raw_videos")))
        b = QPushButton("Browse...")
        b.clicked.connect(self._browse_raw)
        h.addWidget(self._raw_dir)
        h.addWidget(b)
        l.addLayout(h)
        l.addWidget(QLabel("Project name"))
        self._proj = QLineEdit(self.cfg.get("project_name", "VIEB Project"))
        l.addWidget(self._proj)
        self._have_dlc = QCheckBox("I have already run DeepLabCut pose estimation")
        l.addWidget(self._have_dlc)
        nxt = QPushButton("Continue")
        nxt.clicked.connect(self._step1_next)
        l.addWidget(nxt)
        l.addStretch()
        self._stack.addWidget(w)

    def _build_step2(self):
        w = QWidget()
        l = QVBoxLayout(w)
        card = QLabel(
            "VIEB uses DeepLabCut to track 8 body keypoints per frame.\n"
            "This requires labeling ~20 frames per video to train the model.\n"
            "This step takes several hours but only needs to be done once."
        )
        card.setStyleSheet("background:#eef4ff;padding:12px;border:1px solid #c8dafc;border-radius:8px;")
        l.addWidget(card)
        self._btn_label = QPushButton("Start Labeling Queue")
        self._btn_train = QPushButton("Train Model")
        self._btn_analyze = QPushButton("Run Pose Estimation on All Videos")
        self._btn_label.clicked.connect(lambda: self._run_setup(["setup_dlc_training.py", "--label"]))
        self._btn_train.clicked.connect(lambda: self._run_setup(["setup_dlc_training.py", "--train"]))
        self._btn_analyze.clicked.connect(lambda: self._run_setup(["setup_dlc_training.py", "--analyze"]))
        l.addWidget(self._btn_label)
        l.addWidget(self._btn_train)
        l.addWidget(self._btn_analyze)
        self._csv_prog = QProgressBar()
        l.addWidget(self._csv_prog)
        nxt = QPushButton("Continue to Pipeline Configuration")
        nxt.clicked.connect(lambda: self._stack.setCurrentIndex(2))
        l.addWidget(nxt)
        l.addStretch()
        self._stack.addWidget(w)
        self._csv_timer = QTimer(self)
        self._csv_timer.timeout.connect(self._update_csv_progress)
        self._csv_timer.start(2000)

    def _build_step3(self):
        w = QWidget()
        l = QVBoxLayout(w)
        self._mcs = QSlider(Qt.Horizontal)
        self._mcs.setRange(500, 5000)
        self._mcs.setValue(int(self.cfg.get("min_cluster_size", 2000)))
        self._mcs_lbl = QLabel(f"min_cluster_size: {self._mcs.value()}")
        self._mcs.valueChanged.connect(lambda v: self._mcs_lbl.setText(f"min_cluster_size: {v}"))
        l.addWidget(self._mcs_lbl)
        l.addWidget(self._mcs)
        self._wave = QCheckBox("Use Morlet wavelets")
        self._wave.setChecked(bool(self.cfg.get("use_wavelets", True)))
        l.addWidget(self._wave)
        g = QGridLayout()
        self._xmin = QSpinBox(); self._xmin.setRange(0, 9999); self._xmin.setValue(int(self.cfg.get("arena_bounds", {}).get("x_min", 0)))
        self._ymin = QSpinBox(); self._ymin.setRange(0, 9999); self._ymin.setValue(int(self.cfg.get("arena_bounds", {}).get("y_min", 0)))
        self._xmax = QSpinBox(); self._xmax.setRange(0, 9999); self._xmax.setValue(int(self.cfg.get("arena_bounds", {}).get("x_max", 1280)))
        self._ymax = QSpinBox(); self._ymax.setRange(0, 9999); self._ymax.setValue(int(self.cfg.get("arena_bounds", {}).get("y_max", 960)))
        g.addWidget(QLabel("x_min"), 0, 0); g.addWidget(self._xmin, 0, 1)
        g.addWidget(QLabel("y_min"), 0, 2); g.addWidget(self._ymin, 0, 3)
        g.addWidget(QLabel("x_max"), 1, 0); g.addWidget(self._xmax, 1, 1)
        g.addWidget(QLabel("y_max"), 1, 2); g.addWidget(self._ymax, 1, 3)
        l.addLayout(g)
        self._run = QPushButton("Run Full Pipeline")
        self._run.clicked.connect(self._run_full)
        l.addWidget(self._run)
        self._run_prog = QProgressBar()
        self._run_prog.setRange(0, 1)
        l.addWidget(self._run_prog)
        l.addStretch()
        self._stack.addWidget(w)

    def _browse_raw(self):
        d = QFileDialog.getExistingDirectory(self, "Select Raw Videos", self._raw_dir.text())
        if d:
            self._raw_dir.setText(d)

    def _step1_next(self):
        self.cfg["raw_videos_dir"] = self._raw_dir.text()
        self.cfg["project_name"] = self._proj.text().strip() or "VIEB Project"
        _save_cfg(self.cfg)
        self._stack.setCurrentIndex(2 if self._have_dlc.isChecked() else 1)

    def _run_setup(self, args):
        if self._worker and self._worker.isRunning():
            return
        self._worker = SubprocessWorker(args)
        self._worker.log.connect(lambda t: self._log.insertPlainText(t))
        self._worker.done.connect(lambda ok: self._log.insertPlainText(f"\n{'OK' if ok else 'FAIL'}: {' '.join(args)}\n"))
        self._worker.start()

    def _update_csv_progress(self):
        raw = Path(self._raw_dir.text())
        vids = list(raw.glob("*.mp4")) if raw.exists() else []
        csvs = list(raw.glob("*DLC*.csv")) if raw.exists() else []
        total = max(1, len(vids))
        self._csv_prog.setRange(0, total)
        self._csv_prog.setValue(min(total, len(csvs)))
        self._csv_prog.setFormat(f"{len(csvs)} / {len(vids)} videos with CSV")

    def _on_stage_started(self, sid):
        self._log.insertPlainText(f"\nStage {sid} started...\n")

    def _on_stage_done(self, sid, ok):
        self._log.insertPlainText(f"Stage {sid} {'done' if ok else 'failed'}.\n")
        self._run_prog.setValue(min(1, self._run_prog.value() + 1))

    def _run_full(self):
        if self._pipeline and self._pipeline.isRunning():
            return
        self.cfg["min_cluster_size"] = int(self._mcs.value())
        self.cfg["use_wavelets"] = bool(self._wave.isChecked())
        self.cfg["arena_bounds"] = {
            "x_min": self._xmin.value(),
            "y_min": self._ymin.value(),
            "x_max": self._xmax.value(),
            "y_max": self._ymax.value(),
        }
        _save_cfg(self.cfg)
        stages = [2, 3, 8, 9, 10, 11] if self._have_dlc.isChecked() else [1, 2, 3, 8, 9, 10, 11]
        self._run_prog.setRange(0, len(stages))
        self._run_prog.setValue(0)
        self._pipeline = PipelineRunner(stages, self.cfg)
        self._pipeline.log.connect(lambda t: self._log.insertPlainText(t))
        self._pipeline.stage_started.connect(self._on_stage_started)
        self._pipeline.stage_done.connect(self._on_stage_done)
        self._pipeline.all_done.connect(self._pipeline_done)
        self._pipeline.start()

    def _pipeline_done(self, ok):
        if ok:
            self.cfg["onboarding_complete"] = True
            _save_cfg(self.cfg)
            self.completed.emit()
            self.accept()
        else:
            QMessageBox.warning(self, "Onboarding", "Pipeline failed. Review logs and retry.")


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


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.cfg = _load_cfg()
        self.setWindowTitle("VIEB - Video Interpreter for Experimental Behavior")
        self.setMinimumSize(1024, 768)
        w, h = self.cfg.get("window_size", [1280, 800])
        self.resize(w, h)
        self._pulse_idx = 0
        self._running = False
        self._pipeline_running = False
        self._clip_running = False
        self._clip_worker = None
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse)
        self._build()
        self._load_data()
        QTimer.singleShot(200, self._maybe_onboarding)

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        ml = QHBoxLayout(central)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(0)

        sidebar = QWidget()
        sidebar.setFixedWidth(255)
        sidebar.setStyleSheet("background:#f8f9fa;border-right:1px solid #e0e0e0;")
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(0, 0, 0, 0)
        logo = QLabel("  VIEB")
        logo.setFont(QFont("Arial", 16, QFont.Bold))
        logo.setStyleSheet("padding:20px 20px 12px 20px;color:#1a73e8;border-bottom:1px solid #e0e0e0;")
        sl.addWidget(logo)

        self._nav = {}
        for name in _NAV_VIEWS:
            b = NavBtn(name)
            b.clicked.connect(lambda _, n=name: self._switch(n))
            sl.addWidget(b)
            self._nav[name] = b
        sl.addStretch()
        ml.addWidget(sidebar)

        self._stack = QStackedWidget()
        ml.addWidget(self._stack)
        self._views = {}

        def add(name, widget):
            self._views[name] = widget
            self._stack.addWidget(widget)

        self._ov = OverviewView()
        self._ov.export_requested.connect(self._show_export_dialog)
        add("Overview", self._ov)

        self._pv = RunPipelineView(self.cfg)
        self._pv.pipeline_done.connect(self._load_data)
        self._pv.worker_running.connect(self._set_running)
        add("Run Pipeline", self._pv)

        self._sv = BrowseStatesView(self.cfg)
        self._sv.navigate_to_pipeline.connect(lambda: self._switch("Run Pipeline"))
        self._sv.request_clip_generation.connect(self._start_background_clip_generation)
        add("Browse States", self._sv)

        self._vv = ValidationView(self.cfg)
        self._vv.navigate_to_pipeline.connect(lambda: self._switch("Run Pipeline"))
        add("Validation", self._vv)

        self._tv = TransitionMatrixView()
        add("Transition Matrix", self._tv)

        self._mv = MotifsView()
        self._mv.navigate_to_pipeline.connect(lambda: self._switch("Run Pipeline"))
        add("Motifs", self._mv)

        self._av = AnimalExplorerView()
        add("Animal Explorer", self._av)

        self._setv = SettingsView(self.cfg)
        self._setv.settings_changed.connect(self._settings_changed)
        add("Settings", self._setv)

        self._build_status_bar()
        self._switch(self.cfg.get("last_view", "Overview"))

    def _build_status_bar(self):
        sb: QStatusBar = self.statusBar()
        sb.setFixedHeight(30)
        self._sb_run = QLabel("Last run: -")
        self._sb_vid = QLabel("Videos: -")
        self._sb_states = QLabel("States: -")
        self._sb_stage = QLabel("Last: -")
        self._sb_noise = QLabel("Noise: -")
        self._sb_noise.setToolTip("X% of frames excluded as noise (transitions, tracking errors)")
        self._sb_dot = QLabel("●")
        self._sb_dot.setStyleSheet("color:#999;")
        for w in [self._sb_run, QLabel(" | "), self._sb_vid, QLabel(" | "), self._sb_states, QLabel(" | "), self._sb_stage, QLabel(" | "), self._sb_noise, QLabel(" | "), self._sb_dot]:
            sb.addWidget(w)

    def _settings_changed(self, cfg):
        self.cfg = cfg
        _save_cfg(self.cfg)
        self._load_data()

    def _set_running(self, running: bool):
        self._pipeline_running = running
        self._sync_running()

    def _sync_running(self):
        self._running = self._pipeline_running or self._clip_running
        if self._running:
            self._pulse_timer.start(300)
        else:
            self._pulse_timer.stop()
            self._sb_dot.setStyleSheet("color:#999;")

    def _start_background_clip_generation(self, _sid: int):
        if self._clip_worker and self._clip_worker.isRunning():
            self.statusBar().showMessage("Clip generation already running in background.", 5000)
            return
        self._clip_worker = ClipGenerationWorker(self.cfg)
        self._clip_worker.log.connect(self._on_clip_log)
        self._clip_worker.done.connect(self._on_clip_done)
        self._clip_running = True
        self._sync_running()
        self.statusBar().showMessage("Background clip generation started.", 5000)
        self._clip_worker.start()

    def _on_clip_log(self, text):
        if text.strip():
            self.statusBar().showMessage("Generating clips in background...", 2000)

    def _on_clip_done(self, ok):
        self._clip_running = False
        self._sync_running()
        self.statusBar().showMessage("Clip generation complete." if ok else "Clip generation failed.", 7000)
        if ok:
            self._load_data()

    def _pulse(self):
        if not self._running:
            self._sb_dot.setStyleSheet("color:#999;")
            return
        self._pulse_idx = (self._pulse_idx + 1) % 2
        color = "#2e7d32" if self._pulse_idx else "#7bd87f"
        self._sb_dot.setStyleSheet(f"color:{color};")

    def _switch(self, name):
        if name not in self._views:
            return
        for n, b in self._nav.items():
            b.setChecked(n == name)
        self._stack.setCurrentWidget(self._views[name])
        self.cfg["last_view"] = name

    def _load_data(self):
        self._loader = DataLoader()
        self._loader.loaded.connect(self._on_loaded)
        self._loader.error.connect(lambda e: self.statusBar().showMessage(f"Load error: {e}", 6000))
        self._loader.start()

    def _on_loaded(self, data):
        summary = data.get("summary")
        ci = data.get("cluster_info")
        if summary is not None:
            self._sb_vid.setText(f"Videos: {len(summary)}")
        if ci:
            self._sb_states.setText(f"States: {ci.get('n_clusters', '-')}")
        p = RESULTS / "comparison" / "summary_table.csv"
        if p.exists():
            self._sb_run.setText(f"Last run: {_fmt_ts(p.stat().st_mtime)}")
        self._sb_stage.setText(f"Last: {self.cfg.get('last_completed_stage', '-') or '-'}")
        if summary is not None:
            state_cols = [c for c in summary.columns if c.startswith("state_") and c.endswith("_frac")]
            if state_cols:
                noise = (1 - summary[state_cols].sum(axis=1).mean()) * 100
                self._sb_noise.setText(f"Noise: {noise:.1f}%")

        self._pv.estimate_times(data)
        self._pv.update_from_cfg()
        self._ov.update_data(data)
        self._sv.update_data(data)
        self._vv.update_data(data)
        self._tv.update_data(data)
        self._mv.update_data(data)
        self._av.update_data(data)

    def _show_export_dialog(self):
        dlg = ExportResultsDialog(self)
        dlg.exec_()

    def _maybe_onboarding(self):
        if self.cfg.get("onboarding_complete"):
            return
        raw_dir = Path(self.cfg.get("raw_videos_dir", str(ROOT / "raw_videos")))
        no_results = not _results_exist()
        no_dlc_project = _find_dlc_project() is None
        if no_results and no_dlc_project:
            dlg = OnboardingDialog(self.cfg, self)
            dlg.completed.connect(lambda: self._load_data())
            dlg.showMaximized()
            dlg.exec_()

    def closeEvent(self, e):
        self.cfg["window_size"] = [self.width(), self.height()]
        _save_cfg(self.cfg)
        super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("VIEB")
    app.setStyle("Fusion")
    app.setStyleSheet("QLabel{border:none;} QGroupBox{border:none;} QGroupBox::title{subcontrol-origin: margin;}")
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
