from __future__ import annotations
import math
import os
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QSlider, QToolButton,
    QVBoxLayout, QWidget,
)

from _utils import _fmt_ts, _CV2, _MPL, _state_colors

if _CV2:
    import cv2

if _MPL:
    from _utils import FigureCanvas, Figure


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


class VideoPlayer(QWidget):
    video_finished = pyqtSignal()

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
                self.video_finished.emit()
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


class KinematicsPanel(QWidget):
    """
    Three time-series panel shown below a video player.

    Shows centroid_speed (blue), angular_velocity (orange), rearing_score (green)
    normalised 0-1 over the clip's frame range. A vertical red cursor updates
    at 30fps as the video plays (driven by an external QTimer call to set_frame).

    If features are unavailable the panel hides itself silently.
    """

    _FEAT_NAMES = ("centroid_speed", "angular_velocity", "rearing_score")

    def __init__(self, parent=None):
        super().__init__(parent)
        self._features = None      # (T_clip, 3) normalised
        self._n_frames = 0
        self._cursor_frame = 0
        self._cursor_line = None
        self.setMaximumHeight(130)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)
        lay.setSpacing(0)
        if _MPL:
            self._canvas = FigureCanvas(Figure(figsize=(8, 1.4), tight_layout=True))
            self._canvas.figure.patch.set_facecolor("#1a1a1a")
            self._ax = self._canvas.figure.add_subplot(111)
            self._ax.set_facecolor("#1a1a1a")
            lay.addWidget(self._canvas)
        else:
            self._canvas = None
            self._ax = None
            lay.addWidget(QLabel("Install matplotlib for kinematic overlay."))

    @staticmethod
    def _resolve_feat_indices() -> dict:
        """Load feature_names from index.json and return {name: index} mapping."""
        import json
        from ml.feature_extraction import resolve_feature_indices
        idx_path = os.path.join("results", "features", "index.json")
        if os.path.exists(idx_path):
            try:
                with open(idx_path) as f:
                    meta = json.load(f).get("_meta", {})
                names = meta.get("feature_names", [])
                if names:
                    return resolve_feature_indices(names)
            except Exception:
                pass
        return {}

    def load_clip(self, features_path: str, start_frame: int, end_frame: int):
        """Load feature slice for a clip and draw the time series."""
        self._features = None
        self._n_frames = 0
        self._cursor_frame = 0
        if not _MPL or self._ax is None:
            return
        if not features_path or not os.path.exists(features_path):
            self._ax.clear()
            self._canvas.draw()
            return
        try:
            arr = np.load(features_path)
            n_total = arr.shape[0]
            s = max(0, min(start_frame, n_total - 1))
            e = max(s + 1, min(end_frame, n_total))
            feat_slice = arr[s:e]

            feat_indices = self._resolve_feat_indices()
            cols = []
            for name in self._FEAT_NAMES:
                idx = feat_indices.get(name)
                if idx is not None and idx < arr.shape[1]:
                    raw = feat_slice[:, idx].astype(float)
                    lo, hi = raw.min(), raw.max()
                    normed = (raw - lo) / (hi - lo + 1e-9)
                    cols.append(normed)
                else:
                    cols.append(np.zeros(len(feat_slice)))

            self._features = np.column_stack(cols)
            self._n_frames = len(feat_slice)
        except Exception:
            self._features = None
            self._n_frames = 0

        self._redraw()

    def _redraw(self):
        if not _MPL or self._ax is None:
            return
        ax = self._ax
        ax.clear()
        ax.set_facecolor("#1a1a1a")

        if self._features is not None and self._n_frames > 0:
            x = np.arange(self._n_frames)
            colors = ["#4a90d9", "#e67e22", "#2ecc71"]
            labels = ["speed", "ang_vel", "rearing"]
            for i, (color, label) in enumerate(zip(colors, labels)):
                ax.plot(x, self._features[:, i], color=color, linewidth=0.8,
                        label=label, alpha=0.9)
            # Cursor
            self._cursor_line = ax.axvline(
                x=self._cursor_frame, color="#e74c3c", linewidth=1.2, alpha=0.85)
            ax.legend(loc="upper right", fontsize=6, framealpha=0.3,
                      labelcolor="white", facecolor="#1a1a1a")
            ax.set_xlim(0, max(1, self._n_frames - 1))
            ax.set_ylim(-0.05, 1.05)
        else:
            ax.text(0.5, 0.5, "Features not available", ha="center", va="center",
                    color="#666", fontsize=8, transform=ax.transAxes)

        for spine in ax.spines.values():
            spine.set_color("#333")
        ax.tick_params(colors="#888", labelsize=6)
        ax.set_xlabel("Frame", color="#888", fontsize=6)

        self._canvas.figure.tight_layout(pad=0.2)
        self._canvas.draw()

    def set_frame(self, frame_idx: int):
        """Update cursor position — called by QTimer at ~30fps."""
        if self._features is None or self._n_frames == 0:
            return
        frame_idx = max(0, min(frame_idx, self._n_frames - 1))
        if frame_idx == self._cursor_frame:
            return
        self._cursor_frame = frame_idx
        if self._cursor_line is not None:
            self._cursor_line.set_xdata([frame_idx, frame_idx])
            self._canvas.draw_idle()

    def clear(self):
        self._features = None
        self._n_frames = 0
        self._cursor_frame = 0
        self._cursor_line = None
        if _MPL and self._ax:
            self._ax.clear()
            self._canvas.draw()


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


_STAGE_HELP: dict[int, str] = {
    0:  "stage_0_onboarding",
    1:  "stage_1_dlc",
    2:  "stage_2_features",
    3:  "stage_3_clustering",
    4:  "stage_4_collapse",
    5:  "stage_5_comparison",
    6:  "stage_6_quantification",
    7:  "stage_7_motifs",
    8:  "stage_8_clips",
    9:  "stage_9_add_videos",
}


class _ClickableHeader(QFrame):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class StageRow(QFrame):
    run_stage = pyqtSignal(int)
    run_from_here = pyqtSignal(int)
    mark_completed = pyqtSignal(int, bool)
    changed = pyqtSignal(str, object)
    run_diagnose = pyqtSignal()
    run_subcluster = pyqtSignal(int)
    navigate_help = pyqtSignal(str)

    _COLORS = {
        "done":    ("#e8f5e9", "#a5d6a7", "#2e7d32"),
        "running": ("#e3f2fd", "#90caf9", "#1565c0"),
        "pending": ("#fafafa", "#e0e0e0", "#999999"),
        "error":   ("#ffebee", "#ef9a9a", "#c62828"),
    }
    _ICONS = {"done": "✓", "running": "▶", "pending": "○", "error": "✕"}
    _ARROW_STYLE = (
        "QToolButton{color:#5f6368;background:transparent;border:none;"
        "padding:0;margin:0;}"
        "QToolButton:hover{background:#edf2f7;border-radius:3px;}"
    )

    def __init__(self, stage: dict, cfg: dict):
        super().__init__()
        self.stage = stage
        self.cfg = cfg
        self._dom_state_id = -1
        self._build()

    def _build(self):
        self.setObjectName("stageCard")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = _ClickableHeader()
        header.setCursor(Qt.PointingHandCursor)
        header.setStyleSheet("background:transparent;border:none;")
        header.clicked.connect(self._toggle)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(14, 10, 14, 10)

        self._icon = QLabel("○")
        self._icon.setFixedWidth(20)
        hl.addWidget(self._icon)

        title = QLabel(f"Stage {self.stage['id']}: {self.stage['name']}")
        title.setStyleSheet(
            "font-weight:bold;color:#333;background:transparent;border:none;"
        )
        hl.addWidget(title, stretch=1)

        _help_anchor = _STAGE_HELP.get(self.stage["id"])
        if _help_anchor:
            _hb = QToolButton()
            _hb.setText("?")
            _hb.setFixedSize(20, 20)
            _hb.setToolTip("Open Help for this stage")
            _hb.setCursor(Qt.PointingHandCursor)
            _hb.setStyleSheet(
                "QToolButton{border:1px solid #aaa;border-radius:10px;color:#555;"
                "background:#f5f5f5;font-size:10px;font-weight:bold;}"
                "QToolButton:hover{background:#e8f0fe;color:#1a73e8;border-color:#1a73e8;}"
            )
            _hb.clicked.connect(lambda _, a=_help_anchor: self.navigate_help.emit(a))
            hl.addWidget(_hb)

        self._ts = QLabel("")
        self._ts.setStyleSheet(
            "color:#888;font-size:11px;background:transparent;border:none;"
        )
        hl.addWidget(self._ts)

        self._eta = QLabel("")
        self._eta.setStyleSheet(
            "color:#888;font-size:11px;background:transparent;border:none;"
        )
        hl.addWidget(self._eta)

        self._arrow = QToolButton()
        self._arrow.setArrowType(Qt.RightArrow)
        self._arrow.setCursor(Qt.PointingHandCursor)
        self._arrow.setFixedSize(18, 18)
        self._arrow.setStyleSheet(self._ARROW_STYLE)
        self._arrow.clicked.connect(self._toggle)
        hl.addWidget(self._arrow)

        outer.addWidget(header)

        self._desc = QLabel(self.stage["desc"])
        self._desc.setWordWrap(True)
        self._desc.setStyleSheet(
            "color:#666;font-size:11px;padding:0 14px 8px 40px;"
            "background:transparent;border:none;"
        )
        outer.addWidget(self._desc)

        self._body = QWidget()
        self._body.setStyleSheet("background:transparent;")
        bl = QVBoxLayout(self._body)
        bl.setContentsMargins(40, 0, 14, 14)
        bl.setSpacing(8)

        if self.stage["id"] in (3, 4, 5, 6):
            note = QLabel("Runs stages 3–6 together.")
            note.setStyleSheet("color:#0b57d0;font-size:11px;")
            bl.addWidget(note)

        if self.stage["id"] == 5:
            self._quality_badge = QLabel("Cluster Quality: —")
            self._quality_badge.setStyleSheet(
                "background:#f5f5f5;border:1px solid #ddd;border-radius:4px;"
                "padding:3px 8px;color:#555;font-size:12px;"
            )
            bl.addWidget(self._quality_badge)

        has_params = False
        params = QHBoxLayout()
        if self.stage["id"] in (2, 3):
            self._mcs = QSlider(Qt.Horizontal)
            self._mcs.setRange(500, 5000)
            self._mcs.setValue(int(self.cfg.get("min_cluster_size", 2000)))
            self._mcs.valueChanged.connect(lambda v: self.changed.emit("min_cluster_size", v))
            self._wave = QCheckBox("Use Morlet wavelets")
            self._wave.setChecked(bool(self.cfg.get("use_wavelets", True)))
            self._wave.toggled.connect(lambda v: self.changed.emit("use_wavelets", v))
            params.addWidget(QLabel("min_cluster_size"))
            params.addWidget(self._mcs)
            params.addWidget(self._wave)
            has_params = True
        if self.stage["id"] == 7:
            self._collapse = QCheckBox("Enable state collapsing")
            self._collapse.setChecked(bool(self.cfg.get("enable_state_collapse", False)))
            self._collapse.toggled.connect(lambda v: self.changed.emit("enable_state_collapse", v))
            params.addWidget(self._collapse)
            has_params = True
        if self.stage["id"] == 11:
            self._clips = QCheckBox("Export video clips")
            self._clips.setChecked(bool(self.cfg.get("export_clips", False)))
            self._clips.toggled.connect(lambda v: self.changed.emit("export_clips", v))
            params.addWidget(self._clips)
            has_params = True
        if self.stage["id"] == 5:
            self._diagnose_btn = QPushButton("Diagnose")
            self._diagnose_btn.setFixedHeight(26)
            self._diagnose_btn.setToolTip(
                "Run diagnose_clusters.py to sweep min_cluster_size values\n"
                "and find the best setting for your data."
            )
            self._diagnose_btn.clicked.connect(self.run_diagnose.emit)
            self._fix_btn = QPushButton("Fix dominant state")
            self._fix_btn.setFixedHeight(26)
            self._fix_btn.setToolTip(
                "Re-cluster the dominant state into sub-states using a\n"
                "second UMAP pass (compare.py --subcluster --state N)."
            )
            self._fix_btn.clicked.connect(
                lambda: self.run_subcluster.emit(self._dom_state_id)
            )
            self._fix_btn.hide()
            _diag_help = QToolButton()
            _diag_help.setText("?")
            _diag_help.setFixedSize(20, 20)
            _diag_help.setToolTip("Open Help: Diagnose Clustering Parameters")
            _diag_help.setCursor(Qt.PointingHandCursor)
            _diag_help.setStyleSheet(
                "QToolButton{border:1px solid #aaa;border-radius:10px;color:#555;"
                "background:#f5f5f5;font-size:10px;font-weight:bold;}"
                "QToolButton:hover{background:#e8f0fe;color:#1a73e8;border-color:#1a73e8;}"
            )
            _diag_help.clicked.connect(lambda: self.navigate_help.emit("diagnose"))
            _fix_help = QToolButton()
            _fix_help.setText("?")
            _fix_help.setFixedSize(20, 20)
            _fix_help.setToolTip("Open Help: Split Dominant State")
            _fix_help.setCursor(Qt.PointingHandCursor)
            _fix_help.setStyleSheet(
                "QToolButton{border:1px solid #aaa;border-radius:10px;color:#555;"
                "background:#f5f5f5;font-size:10px;font-weight:bold;}"
                "QToolButton:hover{background:#e8f0fe;color:#1a73e8;border-color:#1a73e8;}"
            )
            _fix_help.clicked.connect(lambda: self.navigate_help.emit("split_dominant"))
            params.addWidget(self._diagnose_btn)
            params.addWidget(_diag_help)
            params.addWidget(self._fix_btn)
            params.addWidget(_fix_help)
            has_params = True
        params.addStretch()
        if has_params:
            bl.addLayout(params)

        acts = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.setMinimumHeight(34)
        self._run_btn.clicked.connect(lambda: self.run_stage.emit(self.stage["id"]))
        self._from_btn = QPushButton("Run from here")
        self._from_btn.setMinimumHeight(34)
        self._from_btn.clicked.connect(lambda: self.run_from_here.emit(self.stage["id"]))
        self._done_cb = QCheckBox("Mark done")
        self._done_cb.setToolTip("Mark stage as completed")
        self._done_cb.toggled.connect(lambda v: self.mark_completed.emit(self.stage["id"], v))
        acts.addWidget(self._run_btn)
        acts.addWidget(self._from_btn)
        acts.addStretch()
        acts.addWidget(self._done_cb)
        bl.addLayout(acts)

        self._body.hide()
        self._desc.hide()
        outer.addWidget(self._body)

        self.set_status("pending")

    def _set_expanded(self, expanded: bool):
        self._body.setVisible(expanded)
        self._desc.setVisible(expanded)
        self._arrow.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)

    def _toggle(self):
        self._set_expanded(not self._body.isVisible())

    def set_eta(self, text):
        self._eta.setText(f"ETA: {text}")

    def set_status(self, status):
        self._icon.setText(self._ICONS.get(status, "○"))
        bg, border, icon_color = self._COLORS.get(status, self._COLORS["pending"])
        self.setStyleSheet(
            f"QFrame#stageCard{{background:{bg};border:1px solid {border};"
            f"border-radius:6px;}}"
        )
        self._icon.setStyleSheet(
            f"background:transparent;border:none;font-size:13px;"
            f"font-weight:bold;color:{icon_color};"
        )
        self._set_expanded(status == "running")
        self._done_cb.blockSignals(True)
        self._done_cb.setChecked(status == "done")
        self._done_cb.blockSignals(False)

    def set_last_run(self, ts):
        self._ts.setText(f"Last run: {_fmt_ts(ts)}" if ts else "")

    def set_cluster_quality(self, dominant_frac: float, dom_state_id: int):
        if self.stage["id"] != 5:
            return
        self._dom_state_id = dom_state_id
        pct = dominant_frac * 100
        if dominant_frac < 0.40:
            color_bg, color_border, color_text = "#e8f5e9", "#a5d6a7", "#1b5e20"
            label = f"Cluster Quality: ✓ Good — dominant state {dom_state_id} = {pct:.1f}%"
        elif dominant_frac < 0.60:
            color_bg, color_border, color_text = "#fff8e1", "#ffe082", "#795548"
            label = f"Cluster Quality: ⚠ Moderate — dominant state {dom_state_id} = {pct:.1f}%"
        else:
            color_bg, color_border, color_text = "#ffebee", "#ef9a9a", "#b71c1c"
            label = f"Cluster Quality: ✕ Poor — dominant state {dom_state_id} = {pct:.1f}% (consider fixing)"
        self._quality_badge.setText(label)
        self._quality_badge.setStyleSheet(
            f"background:{color_bg};border:1px solid {color_border};"
            f"border-radius:4px;padding:3px 8px;color:{color_text};font-size:12px;"
        )
        self._fix_btn.setVisible(dominant_frac > 0.50)

    def set_enabled(self, enabled):
        self._run_btn.setEnabled(enabled)
        self._from_btn.setEnabled(enabled)


_NAVBTN_CSS = (
    "QPushButton{text-align:left;padding-left:20px;border:none;"
    "background:#F0F0F0;color:#333333;font-size:13px;}"
    "QPushButton:hover{background:#e8f0fe;color:#1a73e8;}"
    "QPushButton:checked{background:#d2e3fc;color:#1a73e8;font-weight:bold;"
    "border-left:3px solid #1a73e8;padding-left:17px;}"
)

_NAV_ICONS = {
    "Overview":       "⊞",
    "Pipeline":       "▶",
    "Browse States":  "▣",
    "Validation":     "✓",
    "Quantification": "∑",
    "Advanced":       "⚙",
    "Settings":       "≡",
}


class NavBtn(QPushButton):
    def __init__(self, text):
        icon = _NAV_ICONS.get(text, "")
        display = f"{icon}  {text}" if icon else text
        super().__init__(display)
        self.setCheckable(True)
        self.setFixedHeight(42)
        self.setStyleSheet(_NAVBTN_CSS)
