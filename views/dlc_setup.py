from __future__ import annotations
import os
import sys
from pathlib import Path

import pandas as pd

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFrame, QGridLayout,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QScrollArea, QTextEdit, QVBoxLayout, QWidget, QDialogButtonBox,
)

# Canonical role names shown in the keypoint-mapping dropdowns
_ROLE_OPTIONS = [
    "— unassigned —",
    "nose",
    "left_ear",
    "right_ear",
    "center/centroid",
    "left_hip",
    "right_hip",
    "tail_base",
    "tail_tip",
]

from _utils import ROOT, _save_cfg, _register_project, _find_dlc_project
from _workers import SubprocessWorker
from _dialogs import _CreateProjectDialog
from dlc_project_utils import has_trained_dlc_model, normalize_dlc_project_path

try:
    from vieb_config import get_dlc_project_path
except ImportError:
    get_dlc_project_path = lambda: None

try:
    from pretrained_manager import list_available_pretrained
except ImportError:
    list_available_pretrained = lambda: []

_PRIMARY_BTN_STYLE = (
    "QPushButton{background-color:#4E79A7;color:white;border-radius:6px;"
    "font-weight:bold;font-size:12pt;padding:8px;}"
    "QPushButton:hover{background-color:#3d6291;}"
    "QPushButton:disabled{background-color:#b0bec5;}"
)

_LABELING_GUIDE = """
<h3>Labeling Guide — DeepLabCut + Napari</h3>

<b>What you are doing:</b><br>
Clicking on 8 body keypoints in each extracted frame so DeepLabCut can
learn what the mouse looks like from above.

<b>Keypoints (label in this order):</b><br>
<ol>
  <li>left_ear</li><li>right_ear</li><li>nose</li><li>center (mid-body)</li>
  <li>left_hip</li><li>right_hip</li><li>tail_base</li><li>tail_tip</li>
</ol>

<b>Controls in Napari:</b><br>
<ul>
  <li><b>Click</b> on the image to place a point for the active keypoint.</li>
  <li>Use the <b>Points layer selector</b> on the left to switch between keypoints.</li>
  <li><b>Left / Right arrow</b> — move between frames.</li>
  <li><b>Ctrl+Z</b> — undo last point.</li>
  <li><b>Ctrl+S</b> — save your work (do this before closing!).</li>
</ul>

<b>Tips:</b><br>
<ul>
  <li>Label at least 5 frames per video before training.</li>
  <li>It is OK to skip a keypoint if it is fully hidden (leave the layer empty for that frame).</li>
  <li>Save with Ctrl+S before closing Napari — unsaved work is lost.</li>
</ul>
"""

_LOG_PATTERNS = [
    # (substring_to_match,  human_message)
    ("GPU detected",          "✓ GPU found — training and inference will be faster."),
    ("No GPU detected",       "ℹ No GPU found — running on CPU. Training will be slow."),
    ("Analyzing",             "⏳ Running pose estimation on videos…"),
    ("Analysis complete",     "✓ Pose estimation complete. All videos processed."),
    ("Extracting features",   "⏳ Extracting behavioral features from pose CSVs…"),
    ("Done. Extracted",       "✓ Feature extraction complete. Ready to cluster."),
    ("Fitting UMAP",          "⏳ Fitting UMAP dimensionality reduction (may take several minutes)…"),
    ("Fitting HDBSCAN",       "⏳ Fitting HDBSCAN clustering…"),
    ("Behavioral states discovered", None),  # pass-through (already readable)
    ("HMM smoother",          "⏳ Smoothing state assignments with HMM…"),
    ("Per-video labels",      "⏳ Saving per-video state labels…"),
    ("Summary table saved",   "✓ Report generated. Comparison plots written to results/comparison/"),
    ("Extracting motifs",     "⏳ Computing bigram/trigram enrichment between contexts…"),
    ("Motifs →",              "✓ Motif discovery complete."),
    ("Collapse mapping",      "✓ States collapsed. Labels rewritten."),
    ("Training complete",     "✓ Model training complete. Run Evaluate to check accuracy."),
    ("Evaluation complete",   "✓ Evaluation complete. Check evaluation-results/ folder."),
    ("[VIEB] Error:",         None),  # pass-through errors verbatim
]


def _translate_log(raw: str) -> str | None:
    """Return a human-readable message for raw CLI output, or None to use raw."""
    stripped = raw.strip()
    if not stripped:
        return None
    for pattern, msg in _LOG_PATTERNS:
        if pattern in stripped:
            return msg  # None means pass raw through
    return None  # default: pass raw through


def _gpu_state_from_log(raw: str) -> str | None:
    """Return GPU badge state implied by a DLC log line."""
    text = raw.lower()
    if "no gpu detected" in text or "using cpu" in text or "torch not installed" in text:
        return "inactive"
    if "gpu detected" in text or "using cuda device" in text:
        return "active"
    return None


def _should_stick_to_bottom(value: int, maximum: int, tolerance: int = 8) -> bool:
    """True when new log output should keep the terminal pinned to the bottom."""
    return maximum - value <= tolerance


_DLC_NOT_INSTALLED_MSG = (
    "⚠️  DeepLabCut is not installed in this environment.\n"
    "\n"
    "To install it, run this command in your terminal:\n"
    "    pip install -e \".[deeplabcut]\"\n"
    "\n"
    "Then restart VIEB.\n"
    "\n"
    "Note: If you already have DLC pose CSVs or an H5 file, you do not need\n"
    "DeepLabCut. Go to Settings → Pose Data Source to configure your existing\n"
    "pose files and skip this step entirely."
)


def _map_color(val: float) -> str:
    if val >= 95:
        return "#2e7d32"
    if val >= 80:
        return "#e65100"
    return "#c62828"


def _rmse_color(val: float) -> str:
    if val <= 5:
        return "#2e7d32"
    if val <= 10:
        return "#e65100"
    return "#c62828"


class _ClickableLabel(QLabel):
    """A QLabel that emits `clicked` on left mouse-button press — used for
    lightweight hyperlink-style headers."""

    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class _ClickableFrame(QFrame):
    """A QFrame that emits `clicked` on left mouse-button press."""

    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class _ModeCard(_ClickableFrame):
    """A clickable card representing one 'what's your situation?' option."""

    def __init__(self, icon: str, title: str, description: str):
        super().__init__()
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(100)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)

        top = QLabel(f"{icon}  {title}")
        top.setWordWrap(True)
        top.setStyleSheet("font-weight:bold;color:#333;background:transparent;border:none;")
        lay.addWidget(top)

        desc = QLabel(description)
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#666;font-size:11px;background:transparent;border:none;")
        lay.addWidget(desc)
        lay.addStretch()

        self.set_selected(False)

    def set_selected(self, selected: bool):
        if selected:
            self.setStyleSheet(
                "QFrame{background:#e3f2fd;border:2px solid #1565c0;border-radius:6px;}"
            )
        else:
            self.setStyleSheet(
                "QFrame{background:#fff;border:1px solid #e0e0e0;border-radius:6px;}"
                "QFrame:hover{background:#f5f9ff;}"
            )


class _StepCard(QFrame):
    """A single collapsible step in the guided setup wizard.

    Status is one of 'done' (✓, collapsed by default), 'current'
    (▶, expanded, highlighted), or 'pending' (○, collapsed, greyed).
    Clicking the header toggles the expanded state regardless of status.
    """

    _COLORS = {
        "done":    ("#e8f5e9", "#a5d6a7", "#2e7d32"),
        "current": ("#e3f2fd", "#90caf9", "#1565c0"),
        "pending": ("#fafafa", "#e0e0e0", "#999999"),
        "error":   ("#ffebee", "#ef9a9a", "#c62828"),
    }
    _ICONS = {"done": "✓", "current": "▶", "pending": "○", "error": "✕"}

    def __init__(self, number: int, title: str, description: str):
        super().__init__()
        self.setObjectName("stepCard")
        sp = self.sizePolicy()
        sp.setHeightForWidth(True)
        self.setSizePolicy(sp)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._header = header = _ClickableFrame()
        header.setCursor(Qt.PointingHandCursor)
        header.setStyleSheet("background:transparent;border:none;")
        header.clicked.connect(self.toggle)
        hl = QHBoxLayout(header)
        hl.setContentsMargins(14, 10, 14, 10)
        self._icon_lbl = QLabel("○")
        self._icon_lbl.setFixedWidth(20)
        hl.addWidget(self._icon_lbl)
        self._title_lbl = QLabel(f"Step {number}: {title}")
        self._title_lbl.setStyleSheet("font-weight:bold;color:#333;background:transparent;border:none;")
        hl.addWidget(self._title_lbl, stretch=1)
        self._arrow_lbl = QLabel("▾")
        self._arrow_lbl.setStyleSheet("color:#999;background:transparent;border:none;")
        hl.addWidget(self._arrow_lbl)
        outer.addWidget(header)

        self._desc = QLabel(description)
        self._desc.setWordWrap(True)
        self._desc.setStyleSheet("color:#666;font-size:11px;padding:0 14px 8px 40px;background:transparent;border:none;")
        outer.addWidget(self._desc)

        self._body = QWidget()
        self._body.setStyleSheet("background:transparent;")
        self._body_lay = QVBoxLayout(self._body)
        self._body_lay.setContentsMargins(40, 0, 14, 14)
        self._body_lay.setSpacing(8)
        outer.addWidget(self._body)

        self.set_status("pending")

    def body_layout(self) -> QVBoxLayout:
        return self._body_lay

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.updateGeometry()

    def toggle(self):
        expanded = not self._body.isVisible()
        self._body.setVisible(expanded)
        self._desc.setVisible(expanded)
        self._arrow_lbl.setText("▾" if expanded else "▸")

    def set_status(self, status: str, expanded: bool | None = None):
        if expanded is None:
            expanded = status == "current"
        self._icon_lbl.setText(self._ICONS.get(status, "○"))
        self._body.setVisible(expanded)
        self._desc.setVisible(expanded)
        self._arrow_lbl.setText("▾" if expanded else "▸")
        bg, border, icon_color = self._COLORS.get(status, self._COLORS["pending"])
        self.setStyleSheet(
            f"QFrame#stepCard{{background:{bg};border:1px solid {border};border-radius:6px;}}"
        )
        self._icon_lbl.setStyleSheet(
            f"background:transparent;border:none;font-size:13px;font-weight:bold;color:{icon_color};"
        )


class DLCSetupView(QWidget):
    """Guided, situation-aware DeepLabCut setup page.

    Lets the user pick "what's your situation" (starting from scratch,
    already have a trained model, use a pretrained model, or already have
    pose CSV/H5 files) and walks them through only the steps that apply.
    """

    navigate_pipeline = pyqtSignal()
    navigate_settings = pyqtSignal()
    worker_running = pyqtSignal(bool)
    worker_command = pyqtSignal(str)

    _MODES = [
        ("scratch", "🆕", "Starting from scratch",
         "I have raw videos and need to label frames and train a tracking model."),
        ("existing", "📂", "I already have a trained DLC model",
         "Link my existing DeepLabCut project and run it on my videos."),
        ("pretrained", "⚡", "Use a ready-made model",
         "Use a pretrained model included with VIEB — no labeling or training needed."),
        ("have_pose", "📄", "I already have pose data (CSV/H5)",
         "Skip DeepLabCut entirely — point VIEB to existing pose files."),
    ]

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self._worker: SubprocessWorker | None = None
        self._dlc_error_shown = False
        self._action_buttons: list[QPushButton] = []
        self._mode_cards: dict[str, _ModeCard] = {}
        self._keypoint_combos: dict[str, QComboBox] = {}
        self._keypoint_object_checks: dict[str, QCheckBox] = {}
        self._pretrained_selection = ""
        self._running_command = ""
        self._base_running_command = ""
        self._gpu_state = "unknown"

        try:
            self._project_path = get_dlc_project_path() or ""
        except Exception:
            self._project_path = ""
        if not self._project_path:
            dlc_dir = _find_dlc_project()
            self._project_path = str(dlc_dir) if dlc_dir else ""

        self._build()
        self._detect_and_show_status()
        self._select_mode(self._detect_mode())

    # ── Top-level layout ─────────────────────────────────────────────────────

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

        title = QLabel("DLC Setup")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        outer.addWidget(title)

        subtitle = QLabel(
            "Get pose-tracking data for your videos. Pick the option below that matches "
            "your situation, then follow the steps in order."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color:#555;")
        outer.addWidget(subtitle)

        # ── GPU status badge ─────────────────────────────────────────────────
        self._gpu_badge = QFrame()
        self._gpu_badge.setObjectName("gpuBadge")
        gpu_lay = QHBoxLayout(self._gpu_badge)
        gpu_lay.setContentsMargins(10, 6, 10, 6)
        gpu_lay.setSpacing(8)
        self._gpu_icon = QLabel("●")
        self._gpu_icon.setStyleSheet("background:transparent;border:none;font-size:13px;")
        self._gpu_label = QLabel("GPU Unknown")
        self._gpu_label.setStyleSheet(
            "background:transparent;border:none;font-weight:bold;font-size:11px;"
        )
        self._gpu_detail = QLabel("Run a DLC command to detect the active hardware.")
        self._gpu_detail.setStyleSheet("background:transparent;border:none;color:#666;font-size:11px;")
        self._gpu_detail.setWordWrap(True)
        gpu_lay.addWidget(self._gpu_icon)
        gpu_lay.addWidget(self._gpu_label)
        gpu_lay.addWidget(self._gpu_detail, stretch=1)
        outer.addWidget(self._gpu_badge)
        self._set_gpu_state("unknown")

        # ── Status banner ─────────────────────────────────────────────────────
        self._banner_frame = QFrame()
        self._banner_frame.setObjectName("statusBanner")
        banner_lay = QHBoxLayout(self._banner_frame)
        banner_lay.setContentsMargins(14, 10, 14, 10)
        self._banner_label = QLabel("")
        self._banner_label.setWordWrap(True)
        self._banner_label.setTextFormat(Qt.RichText)
        self._banner_label.setStyleSheet("background:transparent;border:none;")
        banner_lay.addWidget(self._banner_label, stretch=1)
        self._banner_btn = QPushButton("Proceed to Pipeline →")
        self._banner_btn.setToolTip(
            "Open the Run Pipeline tab to run feature extraction, clustering, and analysis"
        )
        self._banner_btn.setStyleSheet(
            "QPushButton{background-color:#43a047;color:white;border-radius:4px;"
            "padding:6px 14px;font-weight:bold;}"
            "QPushButton:hover{background-color:#388e3c;}"
        )
        self._banner_btn.clicked.connect(self.navigate_pipeline.emit)
        self._banner_btn.hide()
        banner_lay.addWidget(self._banner_btn)
        outer.addWidget(self._banner_frame)

        # ── Mode selector ────────────────────────────────────────────────────
        mode_title = QLabel("What's your situation?")
        mode_title.setStyleSheet("font-weight:bold;color:#333;")
        outer.addWidget(mode_title)

        mode_box = QWidget()
        mode_grid = QGridLayout(mode_box)
        mode_grid.setContentsMargins(0, 0, 0, 0)
        mode_grid.setSpacing(10)
        for i, (mode_id, icon, mtitle, mdesc) in enumerate(self._MODES):
            card = _ModeCard(icon, mtitle, mdesc)
            card.clicked.connect(lambda m=mode_id: self._select_mode(m))
            mode_grid.addWidget(card, i // 2, i % 2)
            self._mode_cards[mode_id] = card
        outer.addWidget(mode_box)

        # ── Steps container (rebuilt whenever the mode changes) ─────────────
        self._steps_container = QWidget()
        self._steps_lay = QVBoxLayout(self._steps_container)
        self._steps_lay.setContentsMargins(0, 0, 0, 0)
        self._steps_lay.setSpacing(8)
        outer.addWidget(self._steps_container)

        # ── Log (collapsed by default) ───────────────────────────────────────
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
        copy_log_btn = QPushButton("Copy")
        copy_log_btn.setFlat(True)
        copy_log_btn.clicked.connect(self._copy_log)
        log_hdr_row.addWidget(copy_log_btn)
        clear_log_btn = QPushButton("Clear")
        clear_log_btn.setFlat(True)
        clear_log_btn.clicked.connect(self._log.clear)
        log_hdr_row.addWidget(clear_log_btn)
        outer.addLayout(log_hdr_row)
        outer.addWidget(self._log)

        # ── Bottom bar ────────────────────────────────────────────────────────
        bottom_row = QHBoxLayout()

        guide_btn = QPushButton("Labeling Guide")
        guide_btn.setFlat(True)
        guide_btn.setToolTip("Show step-by-step instructions for labeling frames in Napari")
        guide_btn.clicked.connect(self._show_labeling_guide)
        bottom_row.addWidget(guide_btn)

        bottom_row.addStretch()

        self._bottom_proceed_btn = QPushButton("Proceed to Pipeline →")
        self._bottom_proceed_btn.setToolTip(
            "Open the Run Pipeline tab to run feature extraction, clustering, and analysis"
        )
        self._bottom_proceed_btn.clicked.connect(self.navigate_pipeline.emit)
        bottom_row.addWidget(self._bottom_proceed_btn)

        outer.addLayout(bottom_row)
        outer.addStretch()

    # ── Log collapsing ───────────────────────────────────────────────────────

    def _toggle_log(self):
        visible = not self._log.isVisible()
        self._log.setVisible(visible)
        self._log_header.setText("Hide log  ▴" if visible else "Show log  ▾")

    def _copy_log(self):
        from PyQt5.QtWidgets import QApplication
        QApplication.clipboard().setText(self._log.toPlainText())

    def _set_gpu_state(self, state: str, detail: str | None = None):
        self._gpu_state = state
        styles = {
            "active": {
                "text": "GPU Active",
                "detail": "DeepLabCut is using CUDA for this command.",
                "bg": "#e8f5e9",
                "border": "#a5d6a7",
                "color": "#2e7d32",
            },
            "inactive": {
                "text": "GPU Inactive",
                "detail": "DeepLabCut is running on CPU for this command.",
                "bg": "#fff8e1",
                "border": "#ffe082",
                "color": "#e65100",
            },
            "unknown": {
                "text": "GPU Unknown",
                "detail": "Run a DLC command to detect the active hardware.",
                "bg": "#f5f5f5",
                "border": "#d9d9d9",
                "color": "#777",
            },
        }
        spec = styles.get(state, styles["unknown"])
        self._gpu_badge.setStyleSheet(
            f"QFrame#gpuBadge{{background:{spec['bg']};border:1px solid {spec['border']};"
            "border-radius:6px;}}"
        )
        self._gpu_icon.setStyleSheet(
            f"background:transparent;border:none;font-size:13px;color:{spec['color']};"
        )
        self._gpu_label.setText(spec["text"])
        self._gpu_label.setStyleSheet(
            f"background:transparent;border:none;font-weight:bold;font-size:11px;color:{spec['color']};"
        )
        self._gpu_detail.setText(detail or spec["detail"])

    # ── Status banner / overall detection ────────────────────────────────────

    def _count_pose_csvs(self) -> int:
        raw_dir = Path(self.cfg.get("raw_videos_dir", str(ROOT / "raw_videos")))
        return len(list(raw_dir.glob("*DLC*.csv"))) if raw_dir.exists() else 0

    def _count_total_videos(self) -> int:
        raw_dir = Path(self.cfg.get("raw_videos_dir", str(ROOT / "raw_videos")))
        return len(list(raw_dir.glob("*.mp4"))) if raw_dir.exists() else 0

    def _count_extracted_videos(self) -> int:
        if not self._project_path:
            return 0
        labeled_dir = Path(self._project_path) / "labeled-data"
        if not labeled_dir.exists():
            return 0
        return sum(
            1 for d in labeled_dir.iterdir()
            if d.is_dir() and any(f.suffix == ".png" for f in d.iterdir())
        )

    def _count_labeled_videos(self) -> int:
        if not self._project_path:
            return 0
        labeled_dir = Path(self._project_path) / "labeled-data"
        if not labeled_dir.exists():
            return 0
        return sum(
            1 for d in labeled_dir.iterdir()
            if d.is_dir() and any(
                f.name.startswith("CollectedData") and f.suffix == ".h5" for f in d.iterdir()
            )
        )

    def _detect_and_show_status(self):
        """Refresh the top status banner based on current pose-CSV count."""
        self._dlc_csv_count = self._count_pose_csvs()
        self._update_banner()

    def _update_banner(self):
        csv_count = getattr(self, "_dlc_csv_count", 0)
        total = self._count_total_videos()

        if csv_count > 0:
            self._banner_frame.setStyleSheet(
                "QFrame#statusBanner{background:#e8f5e9;border:1px solid #a5d6a7;"
                "border-radius:6px;}"
            )
            self._banner_label.setText(
                f"✓  Pose estimation complete — {csv_count}/{max(total, csv_count)} video(s) "
                "have CSV files.<br>You do not need to redo this step unless adding new videos."
            )
            self._banner_btn.show()
        elif self._project_path:
            self._banner_frame.setStyleSheet(
                "QFrame#statusBanner{background:#fff8e1;border:1px solid #ffe082;"
                "border-radius:6px;}"
            )
            self._banner_label.setText(
                "⚠  DLC project linked but no pose CSVs found yet.<br>"
                "Follow the steps below to generate them."
            )
            self._banner_btn.hide()
        else:
            self._banner_frame.setStyleSheet(
                "QFrame#statusBanner{background:#e3f2fd;border:1px solid #90caf9;"
                "border-radius:6px;}"
            )
            self._banner_label.setText(
                "ℹ  Pick the option below that matches your situation to get started.<br>"
                "If you already have CSV or H5 pose files, choose "
                "\"I already have pose data\"."
            )
            self._banner_btn.hide()

        self._bottom_proceed_btn.setVisible(csv_count == 0)

    # ── Mode selection / step rebuilding ─────────────────────────────────────

    def _detect_mode(self) -> str:
        csv_count = self._count_pose_csvs()
        pose_source = "csv"
        try:
            import vieb_config
            pose_source = vieb_config.get_pose_source()
        except Exception:
            pass

        if not self._project_path and pose_source == "h5" and self.cfg.get("h5_path"):
            return "have_pose"
        if not self._project_path and csv_count > 0:
            return "have_pose"
        if self._project_path and self._validate_model(warn=False):
            return "existing"
        if not self._project_path:
            try:
                if list_available_pretrained():
                    return "pretrained"
            except Exception:
                pass
        return "scratch"

    def _select_mode(self, mode_id: str):
        self._mode = mode_id
        for m, card in self._mode_cards.items():
            card.set_selected(m == mode_id)
        self._rebuild_steps()

    def _add_step(self, number: int, title: str, description: str) -> _StepCard:
        card = _StepCard(number, title, description)
        self._steps_lay.addWidget(card)
        return card

    def _rebuild_steps(self):
        while self._steps_lay.count():
            item = self._steps_lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

        self._action_buttons = []
        self._keypoint_combos = {}
        self._keypoint_object_checks = {}

        if self._mode == "scratch":
            self._build_steps_scratch()
        elif self._mode == "existing":
            self._build_steps_existing()
        elif self._mode == "pretrained":
            self._build_steps_pretrained()
        else:
            self._build_steps_have_pose()

    # ── Shared building blocks ────────────────────────────────────────────────

    def _project_status_text(self) -> tuple[str, str]:
        path = self._project_path
        if not path:
            return "", "#666"
        if not os.path.isdir(path):
            return "⚠ Directory not found.", "#c62828"
        if not os.path.exists(os.path.join(path, "config.yaml")):
            return "⚠ config.yaml not found in this directory.", "#c62828"
        return f"✓ Valid DLC project — {os.path.join(path, 'config.yaml')}", "#2e7d32"

    def _build_project_section(self, layout: QVBoxLayout):
        """Project path field + Browse + status + 'create new project' link."""
        path_row = QHBoxLayout()
        path_le = QLineEdit(self._project_path)
        path_le.setReadOnly(True)
        path_le.setPlaceholderText("No DLC project linked yet…")
        path_row.addWidget(path_le, stretch=1)

        browse_btn = QPushButton("Browse…")
        browse_btn.setToolTip("Select an existing DLC config.yaml to load that project")
        browse_btn.clicked.connect(self._browse_project)
        path_row.addWidget(browse_btn)
        layout.addLayout(path_row)

        status_text, status_color = self._project_status_text()
        if status_text:
            status_lbl = QLabel(status_text)
            status_lbl.setWordWrap(True)
            status_lbl.setStyleSheet(f"color:{status_color};")
            layout.addWidget(status_lbl)

        create_link = _ClickableLabel("＋ Create a new DLC project from scratch")
        create_link.setStyleSheet("color:#1a73e8;text-decoration:underline;font-size:11px;")
        create_link.setCursor(Qt.PointingHandCursor)
        create_link.setToolTip("Create a brand-new DLC project directory")
        create_link.clicked.connect(self._create_project)
        layout.addWidget(create_link)

    def _load_bodyparts(self) -> list:
        if not self._project_path:
            return []
        config_yaml = os.path.join(self._project_path, "config.yaml")
        if not os.path.exists(config_yaml):
            return []
        try:
            import yaml as _yaml
            with open(config_yaml, encoding="utf-8") as fh:
                dlc_cfg = _yaml.safe_load(fh)
            return dlc_cfg.get("bodyparts", []) if isinstance(dlc_cfg, dict) else []
        except Exception:
            return []

    @staticmethod
    def _match_role(name: str) -> str:
        """Case-insensitive heuristic: map a DLC keypoint name to a canonical role."""
        n = name.lower().replace(" ", "_").replace("-", "_").replace("/", "_")
        if n == "nose":
            return "nose"
        if n in ("left_ear", "leftear", "l_ear", "lear"):
            return "left_ear"
        if n in ("right_ear", "rightear", "r_ear", "rear"):
            return "right_ear"
        if n in ("center", "centroid", "centre", "mid", "midbody", "mid_body",
                 "center_centroid", "center_centroid"):
            return "center/centroid"
        if n in ("left_hip", "lefthip", "l_hip", "lhip"):
            return "left_hip"
        if n in ("right_hip", "righthip", "r_hip", "rhip"):
            return "right_hip"
        if n in ("tail_base", "tailbase", "tail_root", "tailroot"):
            return "tail_base"
        if n in ("tail_tip", "tailtip", "tail_end", "tailend", "tail"):
            return "tail_tip"
        return "— unassigned —"

    def _make_object_toggle_handler(self, combo: QComboBox):
        """Closure that disables/replaces the role combo when 'Object' is toggled."""
        state = {"prev": "— unassigned —"}

        def handler(checked: bool):
            if checked:
                state["prev"] = combo.currentText() if combo.isEnabled() else "— unassigned —"
                combo.clear()
                combo.addItem("— object point —")
                combo.setEnabled(False)
            else:
                combo.clear()
                for role in _ROLE_OPTIONS:
                    combo.addItem(role)
                prev = state["prev"]
                idx = _ROLE_OPTIONS.index(prev) if prev in _ROLE_OPTIONS else 0
                combo.setCurrentIndex(idx)
                combo.setEnabled(True)

        return handler

    def _build_keypoint_section(self, layout: QVBoxLayout, bodyparts: list):
        """Keypoint → role / object mapping grid + Save button."""
        saved_roles: dict = self.cfg.get("keypoint_roles", {})
        saved_objects: list = self.cfg.get("object_keypoints", [])

        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(5)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 0)

        hdr_kp = QLabel("Keypoint Name")
        hdr_kp.setStyleSheet("font-weight:bold;color:#333;")
        hdr_role = QLabel("Role")
        hdr_role.setStyleSheet("font-weight:bold;color:#333;")
        hdr_obj = QLabel("Object")
        hdr_obj.setStyleSheet("font-weight:bold;color:#333;")
        hdr_obj.setToolTip(
            "Check this if the keypoint tracks an object (treat, pellet, target) rather than\n"
            "a body part. Object keypoints contribute to distance features but not to posture\n"
            "calculations."
        )
        grid.addWidget(hdr_kp, 0, 0)
        grid.addWidget(hdr_role, 0, 1)
        grid.addWidget(hdr_obj, 0, 2)

        for row_idx, name in enumerate(bodyparts, start=1):
            lbl = QLabel(name)
            combo = QComboBox()
            for role in _ROLE_OPTIONS:
                combo.addItem(role)
            matched = saved_roles.get(name) or self._match_role(name)
            if matched in _ROLE_OPTIONS:
                combo.setCurrentIndex(_ROLE_OPTIONS.index(matched))

            chk = QCheckBox()
            chk.setToolTip(
                "Check if this keypoint tracks an object, not a body part.\n"
                "Object keypoints contribute to distance features but not posture calculations."
            )
            chk.toggled.connect(self._make_object_toggle_handler(combo))
            if name in saved_objects:
                chk.setChecked(True)

            grid.addWidget(lbl, row_idx, 0)
            grid.addWidget(combo, row_idx, 1)
            grid.addWidget(chk, row_idx, 2)
            self._keypoint_combos[name] = combo
            self._keypoint_object_checks[name] = chk

        grid.setRowStretch(len(bodyparts) + 1, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(220)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(container)
        layout.addWidget(scroll)

        save_btn = QPushButton("Save Keypoint Mapping")
        save_btn.setToolTip(
            "Write the mapping to config.json under 'keypoint_roles' "
            "(used by feature_extraction.py for correct postural scalars)"
        )
        save_btn.clicked.connect(self._save_keypoint_mapping)
        layout.addWidget(save_btn, alignment=Qt.AlignRight)

    def _save_keypoint_mapping(self):
        if not self._keypoint_combos:
            QMessageBox.information(
                self, "Nothing to Save",
                "Import a DLC project first to generate the keypoint mapping."
            )
            return
        mapping = {
            name: combo.currentText()
            for name, combo in self._keypoint_combos.items()
            if combo.currentText() not in ("— unassigned —", "— object point —")
        }
        object_kps = [
            name for name, chk in self._keypoint_object_checks.items()
            if chk.isChecked()
        ]
        self.cfg["keypoint_roles"] = mapping
        self.cfg["object_keypoints"] = object_kps
        _save_cfg(self.cfg)
        parts = [f"{len(mapping)} keypoint role(s)"]
        if object_kps:
            parts.append(f"{len(object_kps)} object keypoint(s): {', '.join(object_kps)}")
        QMessageBox.information(
            self, "Keypoint Mapping Saved",
            f"Saved {' and '.join(parts)} to config.json."
        )

    @staticmethod
    def _make_stat_card(value_text: str, label_text: str) -> tuple[QLabel, QFrame]:
        """Build a single stat card (white box with a big value and a small label)."""
        card = QFrame()
        card.setStyleSheet(
            "QFrame{background:white;border:1px solid #e8e8e8;border-radius:4px;}"
        )
        card.setMinimumWidth(100)
        v = QVBoxLayout(card)
        v.setContentsMargins(8, 8, 16, 8)
        v.setAlignment(Qt.AlignCenter)
        value_lbl = QLabel(value_text)
        value_lbl.setAlignment(Qt.AlignCenter)
        value_lbl.setStyleSheet(
            "font-size:16pt;font-weight:bold;color:#333;background:transparent;border:none;"
        )
        label_lbl = QLabel(label_text)
        label_lbl.setAlignment(Qt.AlignCenter)
        label_lbl.setStyleSheet("color:#666;font-size:11px;background:transparent;border:none;")
        v.addWidget(value_lbl)
        v.addWidget(label_lbl)
        return value_lbl, card

    @staticmethod
    def _load_eval_results(dlc_project_path: str) -> dict | None:
        """Return the best-snapshot row from a DLC CombinedEvaluation-results.csv, or None."""
        if not dlc_project_path:
            return None
        patterns = [
            "evaluation-results-pytorch/iteration-*/CombinedEvaluation-results.csv",
            "evaluation-results/iteration-*/CombinedEvaluation-results.csv",
        ]
        for pattern in patterns:
            for csv_path in sorted(Path(dlc_project_path).glob(pattern)):
                try:
                    df = pd.read_csv(csv_path)
                except Exception:
                    continue
                if df.empty:
                    continue
                df.columns = [str(c).strip() for c in df.columns]
                if "test mAP" not in df.columns:
                    continue
                best_idx = df["test mAP"].idxmax()
                row = df.loc[best_idx]
                wanted = (
                    "train mAP", "train mAR", "test mAP", "test mAR",
                    "train rmse", "train rmse_pcutoff", "test rmse", "test rmse_pcutoff",
                    "Training epochs", "Shuffle number",
                )
                result = {}
                for col in wanted:
                    if col in df.columns:
                        val = row[col]
                        result[col] = val.item() if hasattr(val, "item") else val
                return result
        return None

    def _build_eval_section(self, layout: QVBoxLayout, results: dict | None, project_path: str):
        """Stat cards + thumbnails for evaluation results, or a hint if none exist."""
        if not results:
            note = QLabel("Run \"Evaluate Model\" below to see accuracy metrics here.")
            note.setWordWrap(True)
            note.setStyleSheet("color:#888;font-size:11px;")
            layout.addWidget(note)
            return

        cards_row = QHBoxLayout()
        test_map_lbl, card1 = self._make_stat_card("--", "Test mAP")
        test_rmse_lbl, card2 = self._make_stat_card("--", "Test RMSE")
        train_map_lbl, card3 = self._make_stat_card("--", "Train mAP")
        epochs_lbl, card4 = self._make_stat_card("--", "Epochs")
        for card in (card1, card2, card3, card4):
            cards_row.addWidget(card)
        cards_row.addStretch()
        layout.addLayout(cards_row)

        def _set_card(label: QLabel, text: str, color: str | None = None):
            label.setText(text)
            base = "font-size:16pt;font-weight:bold;background:transparent;border:none;"
            label.setStyleSheet(base + f"color:{color};" if color else base + "color:#333;")

        test_map = results.get("test mAP")
        test_rmse = results.get("test rmse_pcutoff")
        train_map = results.get("train mAP")
        epochs = results.get("Training epochs")

        if test_map is not None:
            _set_card(test_map_lbl, f"{float(test_map):.1f}%", _map_color(float(test_map)))
        if test_rmse is not None:
            _set_card(test_rmse_lbl, f"{float(test_rmse):.2f}px", _rmse_color(float(test_rmse)))
        if train_map is not None:
            _set_card(train_map_lbl, f"{float(train_map):.1f}%", _map_color(float(train_map)))
        if epochs is not None:
            _set_card(epochs_lbl, f"{int(epochs)}")

        note = QLabel(
            "Evaluated on held-out test frames. RMSE = mean keypoint position error in pixels."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#888;font-size:11px;")
        layout.addWidget(note)

        thumb_row = QHBoxLayout()
        pattern = "evaluation-results-pytorch/iteration-*/*/LabeledImages_*/*.png"
        pngs = sorted(Path(project_path).glob(pattern))[:3] if project_path else []
        if pngs:
            thumb_lbl = QLabel("Predicted keypoints on test frames")
            thumb_lbl.setStyleSheet("color:#888;font-size:11px;")
            layout.addWidget(thumb_lbl)
            for png in pngs:
                thumb = _ClickableLabel()
                pix = QPixmap(str(png))
                if not pix.isNull():
                    thumb.setPixmap(pix.scaled(120, 80, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                thumb.setFixedSize(120, 80)
                thumb.setCursor(Qt.PointingHandCursor)
                thumb.setStyleSheet("border:1px solid #ccc;background:white;")
                thumb.setToolTip("Click to view full size")
                thumb.clicked.connect(lambda p=png: self._show_full_image(p))
                thumb_row.addWidget(thumb)
            thumb_row.addStretch()
            layout.addLayout(thumb_row)

    def _show_full_image(self, png_path: Path) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(png_path.name)
        lay = QVBoxLayout(dlg)
        lbl = QLabel()
        pix = QPixmap(str(png_path))
        if not pix.isNull():
            if pix.width() > 1000 or pix.height() > 800:
                pix = pix.scaled(1000, 800, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            lbl.setPixmap(pix)
        lay.addWidget(lbl)
        btns = QDialogButtonBox(QDialogButtonBox.Ok)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        dlg.exec_()

    # ── Step flows per mode ────────────────────────────────────────────────────

    def _build_steps_scratch(self):
        project_ok = bool(self._project_path) and os.path.exists(
            os.path.join(self._project_path, "config.yaml")
        )
        extracted = self._count_extracted_videos()
        labeled = self._count_labeled_videos()
        total_videos = self._count_total_videos()
        model_trained = self._validate_model(warn=False)
        eval_results = self._load_eval_results(self._project_path)
        csv_count = self._count_pose_csvs()

        # Step 1: Connect project
        step1 = self._add_step(
            1, "Connect your DLC project",
            "Link an existing DeepLabCut project, or create a brand-new one to get started."
        )
        self._build_project_section(step1.body_layout())
        step1.set_status("done" if project_ok else "current")

        # Step 2: Extract frames
        step2 = self._add_step(
            2, "Prepare frames for labeling",
            "Pull a sample of frames from your videos so you can mark body parts on them."
        )
        extract_btn = QPushButton("Extract Frames")
        extract_btn.setMinimumHeight(34)
        extract_btn.setToolTip(
            "Register your videos and extract a representative sample of frames (kmeans sampling)."
        )
        extract_btn.clicked.connect(self._extract_frames)
        step2.body_layout().addWidget(extract_btn, alignment=Qt.AlignLeft)
        self._action_buttons.append(extract_btn)
        if extracted > 0:
            info = QLabel(f"✓ Frames extracted for {extracted} video(s).")
            info.setStyleSheet("color:#2e7d32;font-size:11px;")
            step2.body_layout().addWidget(info)
        if not project_ok:
            step2.set_status("pending", expanded=False)
        elif extracted > 0:
            step2.set_status("done", expanded=False)
        else:
            step2.set_status("current")

        # Step 3: Label keypoints
        step3 = self._add_step(
            3, "Label keypoints",
            "Open the labeling tool and click on each of the 8 body parts in every frame."
        )
        row = QHBoxLayout()
        label_btn = QPushButton("Continue Labeling")
        label_btn.setMinimumHeight(34)
        label_btn.setToolTip("Launch the Napari labeling interface for the next unlabeled video.")
        label_btn.clicked.connect(self._open_labeling)
        self._action_buttons.append(label_btn)
        guide_btn = QPushButton("Labeling Guide")
        guide_btn.setFlat(True)
        guide_btn.clicked.connect(self._show_labeling_guide)
        row.addWidget(label_btn)
        row.addWidget(guide_btn)
        row.addStretch()
        step3.body_layout().addLayout(row)
        progress = QLabel(f"Progress: {labeled}/{max(extracted, labeled)} video(s) labeled.")
        progress.setStyleSheet("color:#666;font-size:11px;")
        step3.body_layout().addWidget(progress)
        if extracted == 0:
            step3.set_status("pending", expanded=False)
        elif labeled >= extracted:
            step3.set_status("done", expanded=False)
        else:
            step3.set_status("current")

        # Step 4: Train
        step4 = self._add_step(
            4, "Train the model",
            "Teach DeepLabCut to find these body parts automatically. Training can take "
            "30 minutes to 2 hours depending on your hardware."
        )
        train_btn = QPushButton("Train Model")
        train_btn.setMinimumHeight(34)
        train_btn.setToolTip(
            "Train the ResNet50 DLC model on your labeled frames.\n"
            "This can take 30 min – 2 hrs with a GPU."
        )
        train_btn.clicked.connect(lambda: self._run_dlc_subprocess(["setup_dlc_training.py", "--train"]))
        step4.body_layout().addWidget(train_btn, alignment=Qt.AlignLeft)
        self._action_buttons.append(train_btn)
        if labeled == 0:
            step4.set_status("pending", expanded=False)
        elif model_trained:
            step4.set_status("done", expanded=False)
        else:
            step4.set_status("current")

        # Step 5: Evaluate
        step5 = self._add_step(
            5, "Check accuracy",
            "See how well the trained model performs on test frames it has not seen before."
        )
        self._build_eval_section(step5.body_layout(), eval_results, self._project_path)
        eval_btn = QPushButton("Evaluate Model")
        eval_btn.setMinimumHeight(34)
        eval_btn.setToolTip("Evaluate the trained model and produce accuracy metrics (mAP).")
        eval_btn.clicked.connect(lambda: self._run_dlc_subprocess(["setup_dlc_training.py", "--evaluate"]))
        step5.body_layout().addWidget(eval_btn, alignment=Qt.AlignLeft)
        self._action_buttons.append(eval_btn)
        if not model_trained:
            step5.set_status("pending", expanded=False)
        elif eval_results:
            step5.set_status("done", expanded=False)
        else:
            step5.set_status("current")

        # Step 6: Run pose estimation
        step6 = self._add_step(
            6, "Run on all your videos",
            "Use the trained model to generate pose-tracking data (CSV files) for every video."
        )
        info = QLabel(f"{csv_count}/{max(total_videos, csv_count)} video(s) have pose data.")
        info.setStyleSheet("color:#666;font-size:11px;")
        step6.body_layout().addWidget(info)
        analyze_btn = QPushButton("🎯  Run Pose Estimation")
        analyze_btn.setMinimumHeight(40)
        analyze_btn.setStyleSheet(_PRIMARY_BTN_STYLE)
        analyze_btn.setToolTip("Run the trained DLC model on all videos to generate pose CSV files.")
        analyze_btn.clicked.connect(self._run_pose_estimation)
        step6.body_layout().addWidget(analyze_btn)
        self._action_buttons.append(analyze_btn)
        if not model_trained:
            step6.set_status("pending", expanded=False)
        elif csv_count >= total_videos and total_videos > 0:
            step6.set_status("done", expanded=False)
        else:
            step6.set_status("current")

    def _build_steps_existing(self):
        project_ok = bool(self._project_path) and os.path.exists(
            os.path.join(self._project_path, "config.yaml")
        )
        model_trained = self._validate_model(warn=False)
        eval_results = self._load_eval_results(self._project_path)
        csv_count = self._count_pose_csvs()
        total_videos = self._count_total_videos()

        step1 = self._add_step(
            1, "Import your DLC project",
            "Point VIEB to your existing config.yaml. VIEB only needs config.yaml and "
            "the dlc-models/ folder in the same directory."
        )
        self._build_project_section(step1.body_layout())
        step1.set_status("done" if (project_ok and model_trained) else "current")

        next_num = 2
        bodyparts = self._load_bodyparts() if project_ok else []
        if bodyparts:
            step_kp = self._add_step(
                next_num, "Check keypoint roles (optional)",
                "Map each tracked body part to its role so VIEB computes posture features correctly."
            )
            self._build_keypoint_section(step_kp.body_layout(), bodyparts)
            step_kp.set_status("done" if self.cfg.get("keypoint_roles") else "pending", expanded=False)
            next_num += 1

        step_eval = self._add_step(
            next_num, "Check accuracy (optional)",
            "See how well this model performs on test frames, if evaluation results are available."
        )
        self._build_eval_section(step_eval.body_layout(), eval_results, self._project_path)
        eval_btn = QPushButton("Evaluate Model")
        eval_btn.setToolTip("Evaluate the model and produce accuracy metrics (mAP).")
        eval_btn.clicked.connect(lambda: self._run_dlc_subprocess(["setup_dlc_training.py", "--evaluate"]))
        step_eval.body_layout().addWidget(eval_btn, alignment=Qt.AlignLeft)
        self._action_buttons.append(eval_btn)
        step_eval.set_status("done" if eval_results else "pending", expanded=False)
        next_num += 1

        step_run = self._add_step(
            next_num, "Run on all your videos",
            "Use this model to generate pose-tracking data (CSV files) for every video."
        )
        info = QLabel(f"{csv_count}/{max(total_videos, csv_count)} video(s) have pose data.")
        info.setStyleSheet("color:#666;font-size:11px;")
        step_run.body_layout().addWidget(info)
        analyze_btn = QPushButton("🎯  Run Pose Estimation")
        analyze_btn.setMinimumHeight(40)
        analyze_btn.setStyleSheet(_PRIMARY_BTN_STYLE)
        analyze_btn.setToolTip("Run the trained DLC model on all videos to generate pose CSV files.")
        analyze_btn.clicked.connect(self._run_pose_estimation)
        step_run.body_layout().addWidget(analyze_btn)
        self._action_buttons.append(analyze_btn)
        if not (project_ok and model_trained):
            step_run.set_status("pending", expanded=False)
        elif csv_count >= total_videos and total_videos > 0:
            step_run.set_status("done", expanded=False)
        else:
            step_run.set_status("current")

    def _build_steps_pretrained(self):
        try:
            models = list_available_pretrained()
        except Exception:
            models = []

        step1 = self._add_step(
            1, "Choose a ready-made model",
            "Use a model that's already trained on mouse keypoints — no labeling or training needed."
        )
        if not models:
            note = QLabel(
                "No pretrained models found in pretrained/. Download a model package from "
                "GitHub Releases and unzip it into the pretrained/ folder, then reopen this page."
            )
            note.setWordWrap(True)
            note.setStyleSheet("color:#888;font-size:11px;")
            step1.body_layout().addWidget(note)
            step1.set_status("current")
        else:
            combo = QComboBox()
            for m in models:
                combo.addItem(m.get("model_name", "?"), m)
            if self._pretrained_selection:
                idx = combo.findText(self._pretrained_selection)
                if idx >= 0:
                    combo.setCurrentIndex(idx)
            self._pretrained_selection = combo.currentText()

            desc_lbl = QLabel("")
            desc_lbl.setWordWrap(True)
            desc_lbl.setStyleSheet("color:#666;font-size:11px;")

            def _update_desc(_text, combo=combo, desc_lbl=desc_lbl):
                self._pretrained_selection = combo.currentText()
                info = combo.currentData() or {}
                parts = []
                if info.get("description"):
                    parts.append(str(info["description"]))
                if info.get("keypoints"):
                    parts.append("Keypoints: " + ", ".join(info["keypoints"]))
                desc_lbl.setText(" — ".join(parts))

            combo.currentTextChanged.connect(_update_desc)
            _update_desc(combo.currentText())

            step1.body_layout().addWidget(combo)
            step1.body_layout().addWidget(desc_lbl)
            step1.set_status("done", expanded=False)

        step2 = self._add_step(
            2, "Run on all your videos",
            "Apply this model to every video to generate pose-tracking data (CSV files)."
        )
        csv_count = self._count_pose_csvs()
        total_videos = self._count_total_videos()
        info = QLabel(f"{csv_count}/{max(total_videos, csv_count)} video(s) have pose data.")
        info.setStyleSheet("color:#666;font-size:11px;")
        step2.body_layout().addWidget(info)
        analyze_btn = QPushButton("🎯  Run Pose Estimation")
        analyze_btn.setMinimumHeight(40)
        analyze_btn.setStyleSheet(_PRIMARY_BTN_STYLE)
        analyze_btn.setEnabled(bool(models))
        analyze_btn.clicked.connect(self._use_pretrained)
        step2.body_layout().addWidget(analyze_btn)
        self._action_buttons.append(analyze_btn)
        if not models:
            step2.set_status("pending", expanded=False)
        elif csv_count >= total_videos and total_videos > 0:
            step2.set_status("done", expanded=False)
        else:
            step2.set_status("current")

    def _build_steps_have_pose(self):
        csv_count = self._count_pose_csvs()
        total_videos = self._count_total_videos()
        pose_source = "csv"
        try:
            import vieb_config
            pose_source = vieb_config.get_pose_source()
        except Exception:
            pass

        step = self._add_step(
            1, "Use your existing pose data",
            "VIEB can use pose-tracking files you already have — DeepLabCut is not required."
        )
        if pose_source == "h5":
            h5_path = self.cfg.get("h5_path", "")
            info = QLabel(f"Pose source is set to H5: {h5_path or '(not set yet)'}")
        else:
            info = QLabel(
                f"{csv_count}/{max(total_videos, csv_count)} video(s) already have a "
                "DLC pose CSV in raw_videos/."
            )
        info.setWordWrap(True)
        info.setStyleSheet("color:#666;font-size:11px;")
        step.body_layout().addWidget(info)

        settings_btn = QPushButton("Open Settings → Pose Data Source")
        settings_btn.setToolTip("Configure where VIEB should read pose data from (CSV or H5).")
        settings_btn.clicked.connect(self.navigate_settings.emit)
        step.body_layout().addWidget(settings_btn, alignment=Qt.AlignLeft)

        note = QLabel(
            "Use this if you ran DeepLabCut (or another pose tracker) outside of VIEB "
            "and already have CSV or H5 pose files for your videos."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#888;font-size:11px;")
        step.body_layout().addWidget(note)

        has_pose = csv_count > 0 or (pose_source == "h5" and self.cfg.get("h5_path"))
        step.set_status("done" if has_pose else "current")

    # ── Project management ────────────────────────────────────────────────────

    def _set_project_path(self, path: str):
        project_path = normalize_dlc_project_path(path)
        path = str(project_path) if project_path else path
        self._project_path = path
        try:
            import vieb_config
            vieb_config.set_dlc_project_path(path)
        except Exception:
            pass
        self.cfg["dlc_project_path"] = path
        _save_cfg(self.cfg)
        _register_project(path)
        self._detect_and_show_status()

    def _browse_project(self):
        config_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select DLC config.yaml",
            str(ROOT),
            "DLC Config (config.yaml);;All files (*)",
        )
        if not config_file:
            return
        import yaml as _yaml
        try:
            with open(config_file, encoding="utf-8") as f:
                dlc_cfg = _yaml.safe_load(f)
            if not isinstance(dlc_cfg, dict) or "bodyparts" not in dlc_cfg:
                QMessageBox.warning(
                    self,
                    "Invalid DLC Config",
                    f"The file does not appear to be a valid DLC config.yaml\n({config_file})",
                )
                return
        except Exception as exc:
            QMessageBox.warning(self, "Parse Error", f"Could not read config.yaml:\n{exc}")
            return
        self._set_project_path(os.path.dirname(config_file))
        self._rebuild_steps()

    def _create_project(self):
        dlg = _CreateProjectDialog(self.cfg, self)
        if dlg.exec_() == QDialog.Accepted and dlg.result_path:
            self._set_project_path(dlg.result_path)
            self._rebuild_steps()

    # ── Pretrained models ─────────────────────────────────────────────────────

    def _use_pretrained(self):
        name = self._pretrained_selection
        if not name:
            QMessageBox.information(
                self,
                "No Model",
                "No pretrained models found in pretrained/\n"
                "Download a model from GitHub Releases and unzip it into pretrained/",
            )
            return
        raw_dir = self.cfg.get("raw_videos_dir", str(ROOT / "raw_videos")).replace("\\", "\\\\")
        code = (
            "from pretrained_manager import load_pretrained_model, analyze_with_pretrained; "
            f"load_pretrained_model({name!r}, {raw_dir!r}); "
            f"analyze_with_pretrained({name!r}, {raw_dir!r})"
        )
        self._log_human(f"⏳ Loading pretrained model '{name}' and running pose estimation…")
        self._run_dlc_subprocess(["-c", code])

    # ── DLC subprocess helpers ────────────────────────────────────────────────

    def _run_dlc_subprocess(self, args: list[str]):
        if self._worker and self._worker.isRunning():
            self._log_human("⚠ A task is already running. Wait for it to finish.")
            return
        self._dlc_error_shown = False
        self._set_gpu_state("unknown")
        self._set_buttons_enabled(False)
        dlc_python = self.cfg.get("dlc_python") or str(ROOT / "venv-dlc" / "bin" / "python")
        if not os.path.exists(dlc_python):
            dlc_python = sys.executable
        display_python = Path(dlc_python).name or "python"
        self._base_running_command = " ".join([display_python, *args])
        self._running_command = self._base_running_command
        self._worker = SubprocessWorker(args, python_exe=dlc_python)
        self._worker.log.connect(self._on_raw_log)
        self._worker.done.connect(self._on_worker_done)
        self.worker_command.emit(self._running_command)
        self.worker_running.emit(True)
        self._worker.start()

    def stop_worker(self):
        if self._worker and self._worker.isRunning():
            self._log_human("Stopping DLC command…")
            self._worker.stop()

    def _extract_frames(self):
        if not self._validate_project():
            return
        self._log_human("⏳ Registering videos and extracting sample frames…")
        self._run_dlc_subprocess(["-c", "import setup_dlc_training as s; s.add_videos_to_config(); s.extract_frames()"])

    def _open_labeling(self):
        if not self._validate_project():
            return
        self._show_labeling_guide(before_launch=True)
        self._run_dlc_subprocess(["setup_dlc_training.py", "--label"])

    def _run_pose_estimation(self):
        if not self._validate_project():
            return
        if not self._validate_model(warn=True):
            return
        self._log_human("⏳ Running pose estimation on all videos…")
        self._run_dlc_subprocess(["setup_dlc_training.py", "--analyze"])

    def _validate_project(self) -> bool:
        if not self._project_path or not os.path.exists(os.path.join(self._project_path, "config.yaml")):
            QMessageBox.warning(
                self,
                "No DLC Project",
                "Please connect a DLC project first (Step 1).",
            )
            return False
        return True

    def _validate_model(self, warn: bool = True) -> bool:
        """Return True if a trained DLC model snapshot exists."""
        if not self._project_path:
            return False
        if not has_trained_dlc_model(self._project_path):
            if warn:
                QMessageBox.information(
                    self,
                    "No Trained Model Found",
                    "No trained model snapshot found under dlc-models/ or "
                    "dlc-models-pytorch/. If you imported an existing model, "
                    "make sure its config.yaml is in the same project folder as "
                    "the trained snapshot files.",
                )
            return False
        return True

    def _on_worker_done(self, ok: bool):
        self._set_buttons_enabled(True)
        self._detect_and_show_status()
        if ok:
            self._log_human("✓ Task completed successfully.")
        else:
            self._check_dlc_error("")
            self._log_human("✕ Task failed — check the log above for details.")
        self.worker_running.emit(False)
        self._running_command = ""
        self._base_running_command = ""
        self._rebuild_steps()

    def _check_dlc_error(self, text: str) -> bool:
        """Detect a missing-DeepLabCut/torch import error and surface a helpful message."""
        if "ModuleNotFoundError: No module named 'deeplabcut'" in text or \
                "ModuleNotFoundError: No module named deeplabcut" in text or \
                "ModuleNotFoundError: No module named 'torch'" in text or \
                "ModuleNotFoundError: No module named torch" in text:
            if not self._dlc_error_shown:
                self._dlc_error_shown = True
                sb = self._log.verticalScrollBar()
                stick = _should_stick_to_bottom(sb.value(), sb.maximum())
                self._log.append(
                    f"<pre style='color:#ffb300;'>{_DLC_NOT_INSTALLED_MSG}</pre>"
                )
                if stick:
                    sb.setValue(sb.maximum())
                QMessageBox.warning(self, "DeepLabCut Not Installed", _DLC_NOT_INSTALLED_MSG)
            return True
        return False

    def _set_buttons_enabled(self, enabled: bool):
        for b in self._action_buttons:
            b.setEnabled(enabled)

    # ── Logging ───────────────────────────────────────────────────────────────

    def _append_log_text(self, text: str):
        sb = self._log.verticalScrollBar()
        stick = _should_stick_to_bottom(sb.value(), sb.maximum())
        self._log.insertPlainText(text)
        if stick:
            sb.setValue(sb.maximum())

    def _on_raw_log(self, text: str):
        gpu_state = _gpu_state_from_log(text)
        if gpu_state:
            self._set_gpu_state(gpu_state, text.strip())

        human = _translate_log(text)
        if human is not None:
            self._append_log_text(human + "\n")
            if gpu_state and text.strip() and text.strip() != human:
                self._append_log_text(text)
        else:
            self._append_log_text(text)
        self._check_dlc_error(text)

    def _log_human(self, msg: str):
        self._append_log_text(msg + "\n")

    # ── Labeling guide ───────────────────────────────────────────────────────

    def _show_labeling_guide(self, before_launch: bool = False):
        dlg = QDialog(self)
        dlg.setWindowTitle("Labeling Guide — DeepLabCut + Napari")
        dlg.resize(540, 480)
        lay = QVBoxLayout(dlg)
        if before_launch:
            note = QLabel(
                "<b>Please read this before Napari opens.</b> "
                "Save your work with Ctrl+S before closing."
            )
            note.setStyleSheet(
                "background:#fff3cd;border:1px solid #ffc107;border-radius:4px;padding:8px;"
            )
            note.setWordWrap(True)
            lay.addWidget(note)
        lbl = QLabel(_LABELING_GUIDE)
        lbl.setWordWrap(True)
        lbl.setTextFormat(Qt.RichText)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(lbl)
        lay.addWidget(scroll)
        btns = QDialogButtonBox(QDialogButtonBox.Ok)
        btns.accepted.connect(dlg.accept)
        lay.addWidget(btns)
        dlg.exec_()
