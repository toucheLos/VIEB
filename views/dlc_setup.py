from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFrame, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QScrollArea, QSpinBox, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
    QDialogButtonBox,
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

from _utils import ROOT, RESULTS, _load_cfg, _save_cfg, _register_project, _load_projects
from _workers import SubprocessWorker
from _dialogs import _CreateProjectDialog

try:
    from vieb_config import get_dlc_project_path, set_dlc_project_path, get_dlc_config_path
except ImportError:
    get_dlc_project_path = lambda: None
    set_dlc_project_path = lambda p: None
    get_dlc_config_path = lambda: None

try:
    from pretrained_manager import list_available_pretrained, load_pretrained_model, analyze_with_pretrained
except ImportError:
    list_available_pretrained = lambda: []
    load_pretrained_model = lambda *a, **kw: None
    analyze_with_pretrained = lambda *a, **kw: []

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


def _find_dlc_project():
    for p in ROOT.glob("VIEB-*/config.yaml"):
        return p.parent
    return None


class _ClickableLabel(QLabel):
    """A QLabel that emits `clicked` on left mouse-button press — used for
    lightweight hyperlink-style and collapsible-section headers."""

    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class DLCSetupView(QWidget):
    """Dedicated tab for all DeepLabCut project management and pose estimation."""

    navigate_pipeline = pyqtSignal()

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self._worker: SubprocessWorker | None = None
        self._keypoint_combos: dict[str, QComboBox] = {}
        self._keypoint_object_checks: dict[str, QCheckBox] = {}
        self._build()
        self._refresh_recent()
        self._refresh_project_status()
        QTimer.singleShot(0, self._detect_and_show_status)
        QTimer.singleShot(50, self._try_preload_keypoints)

    # ── Layout ───────────────────────────────────────────────────────────────

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(12)

        title = QLabel("DLC Setup")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        outer.addWidget(title)

        subtitle = QLabel(
            "Configure your DeepLabCut project, label frames, train a model, "
            "and run pose estimation before proceeding to the Pipeline."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color:#555;")
        outer.addWidget(subtitle)

        # ── Zone 1: Status banner ────────────────────────────────────────────
        self._banner_frame = QFrame()
        self._banner_frame.setObjectName("statusBanner")
        banner_lay = QHBoxLayout(self._banner_frame)
        banner_lay.setContentsMargins(14, 10, 14, 10)
        self._banner_label = QLabel("")
        self._banner_label.setWordWrap(True)
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

        # ── Zone 2: DLC Project ───────────────────────────────────────────────
        proj_box = QGroupBox("DLC Project")
        proj_box.setStyleSheet("QGroupBox{font-weight:bold;color:#333;}")
        pl = QVBoxLayout(proj_box)

        path_row = QHBoxLayout()
        self._path_le = QLineEdit()
        self._path_le.setReadOnly(True)
        self._path_le.setPlaceholderText("No DLC project linked yet…")
        self._path_le.setToolTip(
            "The root directory of your DLC project.\n"
            "Must contain a config.yaml file."
        )
        self._path_le.textChanged.connect(self._on_path_changed)
        path_row.addWidget(self._path_le, stretch=1)

        browse_btn = QPushButton("Browse…")
        browse_btn.setToolTip("Select an existing DLC config.yaml to load that project")
        browse_btn.clicked.connect(self._browse_project)
        path_row.addWidget(browse_btn)
        pl.addLayout(path_row)

        self._project_status = QLabel("")
        self._project_status.setWordWrap(True)
        pl.addWidget(self._project_status)

        self._create_link = _ClickableLabel("＋ Create a new DLC project from scratch")
        self._create_link.setStyleSheet(
            "color:#1a73e8;text-decoration:underline;font-size:11px;"
        )
        self._create_link.setCursor(Qt.PointingHandCursor)
        self._create_link.setToolTip("Create a brand-new DLC project directory")
        self._create_link.clicked.connect(self._create_project)
        pl.addWidget(self._create_link)

        outer.addWidget(proj_box)

        # ── Evaluation results panel (hidden until eval results found) ───────
        self._build_eval_panel(outer)

        # Hidden — kept only so _refresh_recent()/_load_from_recent() keep working
        self._recent_combo = QComboBox()
        self._recent_combo.hide()
        self._recent_combo.currentIndexChanged.connect(self._load_from_recent)

        # ── Import existing project + keypoint mapping ───────────────────────
        self._build_import_section(outer)
        self._build_keypoint_panel(outer)

        sep1 = QFrame(); sep1.setFrameShape(QFrame.HLine)
        sep1.setStyleSheet("color:#e0e0e0;")
        outer.addWidget(sep1)

        # ── Zone 3: Primary action — Run Pose Estimation ─────────────────────
        self._btn_analyze = QPushButton("🎯  Run Pose Estimation")
        self._btn_analyze.setToolTip(
            "Run the trained DLC model on all videos to generate pose CSV files."
        )
        self._btn_analyze.setMinimumHeight(52)
        self._btn_analyze.setStyleSheet(
            "QPushButton{background-color:#4E79A7;color:white;border-radius:6px;"
            "font-weight:bold;font-size:13pt;}"
            "QPushButton:hover{background-color:#3d6291;}"
            "QPushButton:disabled{background-color:#b0bec5;}"
        )
        self._btn_analyze.clicked.connect(self._run_pose_estimation)
        outer.addWidget(self._btn_analyze)

        analyze_hint = QLabel(
            "Requires a trained DLC model. Use the advanced options below to train first."
        )
        analyze_hint.setStyleSheet("color:#888;font-style:italic;font-size:11px;")
        analyze_hint.setWordWrap(True)
        outer.addWidget(analyze_hint)

        # ── Zone 4: Advanced options (collapsed by default) ──────────────────
        self._advanced_header = _ClickableLabel("Advanced options  ▾")
        self._advanced_header.setStyleSheet("color:#555;font-size:11px;padding:4px 0;")
        self._advanced_header.setCursor(Qt.PointingHandCursor)
        self._advanced_header.clicked.connect(self._toggle_advanced)
        outer.addWidget(self._advanced_header)

        self._advanced_frame = QFrame()
        self._advanced_frame.setObjectName("advancedFrame")
        self._advanced_frame.setStyleSheet(
            "QFrame#advancedFrame{background:#f5f5f5;border:1px solid #e0e0e0;"
            "border-radius:4px;}"
        )
        adv_lay = QVBoxLayout(self._advanced_frame)
        adv_lay.setContentsMargins(8, 8, 8, 8)
        adv_lay.setSpacing(8)

        # Sub-section A — pretrained model
        pre_row = QHBoxLayout()
        pre_row.addWidget(QLabel("Pretrained:"))
        self._pretrained_combo = QComboBox()
        self._pretrained_combo.setToolTip(
            "Available pretrained models in pretrained/\n"
            "Download from GitHub Releases if empty."
        )
        self._refresh_pretrained()
        pre_row.addWidget(self._pretrained_combo, stretch=1)
        use_pre_btn = QPushButton("Use Pretrained")
        use_pre_btn.setToolTip(
            "Load a pretrained model and run pose estimation — no training required"
        )
        use_pre_btn.setMinimumWidth(140)
        use_pre_btn.clicked.connect(self._use_pretrained)
        pre_row.addWidget(use_pre_btn)
        adv_lay.addLayout(pre_row)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet("color:#e0e0e0;")
        adv_lay.addWidget(sep2)

        # Sub-section B — step by step (for new projects)
        step_lbl = QLabel("Step-by-step for new projects:")
        step_lbl.setStyleSheet("font-weight:bold;color:#333;")
        adv_lay.addWidget(step_lbl)

        step_grid = QGridLayout()
        step_grid.setSpacing(8)

        def _dlc_btn(label, tip, slot):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.setMinimumHeight(34)
            b.setMinimumWidth(130)
            b.clicked.connect(slot)
            return b

        self._btn_extract = _dlc_btn(
            "Extract Frames",
            "Extract frames from your videos for labeling (kmeans sampling).",
            lambda: self._run_dlc("--", "extract_frames"),
        )
        self._btn_label = _dlc_btn(
            "Open Labeling GUI",
            "Launch the Napari labeling interface for the next unlabeled video.",
            self._open_labeling,
        )
        self._btn_train = _dlc_btn(
            "Train Model",
            "Train the ResNet50 DLC model on your labeled frames.\nThis can take 30 min – 2 hrs with a GPU.",
            lambda: self._run_dlc_subprocess(["setup_dlc_training.py", "--train"]),
        )
        self._btn_evaluate = _dlc_btn(
            "Evaluate Model",
            "Evaluate the trained model and produce accuracy metrics (mAP).",
            lambda: self._run_dlc_subprocess(["setup_dlc_training.py", "--evaluate"]),
        )
        step_grid.addWidget(self._btn_extract, 0, 0)
        step_grid.addWidget(self._btn_label, 0, 1)
        step_grid.addWidget(self._btn_train, 1, 0)
        step_grid.addWidget(self._btn_evaluate, 1, 1)
        adv_lay.addLayout(step_grid)

        step_hint = QLabel(
            "New project? Extract frames → label keypoints → train → evaluate → "
            "Run Pose Estimation"
        )
        step_hint.setWordWrap(True)
        step_hint.setStyleSheet("color:#888;font-size:11px;")
        adv_lay.addWidget(step_hint)

        # Sub-section C — train-first toggle
        self._train_first = QCheckBox(
            "Train model before running pose estimation (new projects)"
        )
        self._train_first.setChecked(False)
        self._train_first.setToolTip(
            "When checked, clicking 'Run Pose Estimation' will:\n"
            "  1. Run DLC training first\n"
            "  2. Then run inference on all videos\n\n"
            "Leave unchecked if you already have a trained model."
        )
        adv_lay.addWidget(self._train_first)

        self._advanced_frame.hide()
        outer.addWidget(self._advanced_frame)

        # ── Zone 5: Log (collapsed by default) ───────────────────────────────
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

        # ── Zone 6: Bottom bar ────────────────────────────────────────────────
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

    # ── Collapsible sections / banner ───────────────────────────────────────

    def _toggle_advanced(self):
        visible = not self._advanced_frame.isVisible()
        self._advanced_frame.setVisible(visible)
        self._advanced_header.setText("Advanced options  ▴" if visible else "Advanced options  ▾")

    def _toggle_log(self):
        visible = not self._log.isVisible()
        self._log.setVisible(visible)
        self._log_header.setText("Hide log  ▴" if visible else "Show log  ▾")

    def _copy_log(self):
        from PyQt5.QtWidgets import QApplication
        QApplication.clipboard().setText(self._log.toPlainText())

    def _update_import_visibility(self):
        """Show the 'Import Existing DLC Project' section only when no project is linked."""
        linked = bool(self._path_le.text().strip())
        self._import_box.setVisible(not linked)

    def _update_banner(self):
        """Refresh the Zone 1 status banner based on _detect_and_show_status() results."""
        project_path = getattr(self, "_dlc_project_path", None)
        csv_count = getattr(self, "_dlc_csv_count", 0)

        if csv_count > 0:
            self._banner_frame.setStyleSheet(
                "QFrame#statusBanner{background:#e8f5e9;border:1px solid #a5d6a7;"
                "border-radius:6px;}"
            )
            self._banner_label.setText(
                f"✓  Pose estimation complete — {csv_count} video(s) have CSV files.<br>"
                "You do not need to redo this step unless adding new videos."
            )
            self._banner_btn.show()
        elif project_path:
            self._banner_frame.setStyleSheet(
                "QFrame#statusBanner{background:#fff8e1;border:1px solid #ffe082;"
                "border-radius:6px;}"
            )
            self._banner_label.setText(
                "⚠  DLC project linked but no pose CSVs found.<br>"
                "Run Pose Estimation below to generate them."
            )
            self._banner_btn.hide()
        else:
            self._banner_frame.setStyleSheet(
                "QFrame#statusBanner{background:#e3f2fd;border:1px solid #90caf9;"
                "border-radius:6px;}"
            )
            self._banner_label.setText(
                "ℹ  Link your DLC project below to get started.<br>"
                "If you already have CSV or H5 pose files, go to Settings → "
                "Pose Data Source instead."
            )
            self._banner_btn.hide()
        self._banner_label.setTextFormat(Qt.RichText)

        if hasattr(self, "_bottom_proceed_btn"):
            self._bottom_proceed_btn.setVisible(csv_count == 0)

    # ── Import section builders ───────────────────────────────────────────────

    def _build_import_section(self, layout: QVBoxLayout):
        """'Import Existing DLC Project' group box, inserted at the top of the tools area."""
        import_box = QGroupBox("Import Existing DLC Project")
        import_box.setStyleSheet("QGroupBox{font-weight:bold;color:#333;}")
        self._import_box = import_box
        il = QVBoxLayout(import_box)
        il.setSpacing(8)

        desc = QLabel(
            "If you have already trained a DLC model, point VIEB to your config.yaml file. "
            "VIEB only needs the config.yaml and the dlc-models/ folder in the same directory."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#555;")
        il.addWidget(desc)

        path_row = QHBoxLayout()
        self._import_path_le = QLineEdit()
        self._import_path_le.setReadOnly(True)
        self._import_path_le.setPlaceholderText("No DLC project linked yet…")
        # Pre-fill from vieb_config (may already be set from a previous session)
        try:
            cur = get_dlc_project_path()
            if cur:
                self._import_path_le.setText(cur)
        except Exception:
            pass
        path_row.addWidget(self._import_path_le, stretch=1)

        browse_import_btn = QPushButton("Browse…")
        browse_import_btn.setToolTip("Select a config.yaml from your trained DLC project")
        browse_import_btn.clicked.connect(self._on_import_browse)
        path_row.addWidget(browse_import_btn)
        il.addLayout(path_row)

        layout.addWidget(import_box)

    def _build_keypoint_panel(self, layout: QVBoxLayout):
        """Keypoint role mapping panel — hidden until a project is successfully imported."""
        self._keypoint_panel = QGroupBox(
            "Keypoint Roles (optional — improves postural feature quality)"
        )
        self._keypoint_panel.hide()
        kp_outer = QVBoxLayout(self._keypoint_panel)
        kp_outer.setSpacing(8)

        # Scroll area hosts the two-column grid (replaced on each import)
        self._keypoint_scroll = QScrollArea()
        self._keypoint_scroll.setWidgetResizable(True)
        self._keypoint_scroll.setMaximumHeight(260)
        self._keypoint_scroll.setFrameShape(QFrame.NoFrame)
        kp_outer.addWidget(self._keypoint_scroll)

        save_btn = QPushButton("Save Keypoint Mapping")
        save_btn.setToolTip(
            "Write the mapping to config.json under 'keypoint_roles' "
            "(used by feature_extraction.py for correct postural scalars)"
        )
        save_btn.clicked.connect(self._save_keypoint_mapping)
        kp_outer.addWidget(save_btn, alignment=Qt.AlignRight)

        layout.addWidget(self._keypoint_panel)

    def _build_eval_panel(self, layout: QVBoxLayout):
        """'Model Performance' panel — hidden until DLC evaluation results are found."""
        self._eval_panel = QFrame()
        self._eval_panel.setObjectName("evalPanel")
        self._eval_panel.setStyleSheet(
            "QFrame#evalPanel{background:#f8f9fa;border:1px solid #e0e0e0;border-radius:6px;}"
        )
        eval_lay = QVBoxLayout(self._eval_panel)
        eval_lay.setContentsMargins(12, 12, 12, 12)
        eval_lay.setSpacing(8)

        title_row = QHBoxLayout()
        eval_title = QLabel("Model Performance")
        eval_title.setStyleSheet("font-weight:bold;color:#333;background:transparent;border:none;")
        title_row.addWidget(eval_title)
        title_row.addStretch()
        eval_subtitle = QLabel("Best snapshot")
        eval_subtitle.setStyleSheet("color:#888;font-size:11px;background:transparent;border:none;")
        title_row.addWidget(eval_subtitle)
        eval_lay.addLayout(title_row)

        cards_row = QHBoxLayout()
        self._eval_card_test_map, card1 = self._make_stat_card("--", "Test mAP")
        self._eval_card_test_rmse, card2 = self._make_stat_card("--", "Test RMSE")
        self._eval_card_train_map, card3 = self._make_stat_card("--", "Train mAP")
        self._eval_card_epochs, card4 = self._make_stat_card("--", "Epochs")
        for card in (card1, card2, card3, card4):
            cards_row.addWidget(card)
        cards_row.addStretch()
        eval_lay.addLayout(cards_row)

        eval_note = QLabel(
            "Evaluated on held-out test frames. "
            "RMSE = mean keypoint position error in pixels."
        )
        eval_note.setWordWrap(True)
        eval_note.setStyleSheet("color:#888;font-size:11px;background:transparent;border:none;")
        eval_lay.addWidget(eval_note)

        self._eval_thumb_row = QHBoxLayout()
        eval_lay.addLayout(self._eval_thumb_row)

        self._eval_thumb_label = QLabel("Predicted keypoints on test frames")
        self._eval_thumb_label.setStyleSheet("color:#888;font-size:11px;background:transparent;border:none;")
        self._eval_thumb_label.hide()
        eval_lay.addWidget(self._eval_thumb_label)

        self._eval_panel.hide()
        layout.addWidget(self._eval_panel)

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

    # ── Import logic ──────────────────────────────────────────────────────────

    def _on_import_browse(self):
        """Open a file dialog to select config.yaml from an existing DLC project."""
        start_dir = self._import_path_le.text().strip() or str(Path.home())
        config_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select DLC config.yaml",
            start_dir,
            "config.yaml (config.yaml);;All files (*)",
        )
        if config_file:
            self._do_import_config(config_file)

    def _do_import_config(self, config_file: str):
        """Execute the full import sequence for a selected config.yaml."""
        # a. Parse YAML
        try:
            import yaml as _yaml
            with open(config_file, encoding="utf-8") as fh:
                dlc_cfg = _yaml.safe_load(fh)
        except Exception as exc:
            QMessageBox.warning(self, "Parse Error", f"Could not read config.yaml:\n{exc}")
            return

        if not isinstance(dlc_cfg, dict):
            QMessageBox.warning(
                self, "Invalid Config",
                "The selected file does not appear to be a valid DLC config.yaml."
            )
            return

        project_dir = os.path.dirname(os.path.abspath(config_file))

        # b. Warn if dlc-models/ is absent (non-fatal)
        if not os.path.isdir(os.path.join(project_dir, "dlc-models")):
            QMessageBox.warning(
                self,
                "dlc-models/ Not Found",
                "dlc-models/ folder not found next to config.yaml. The model weights may be "
                "missing. VIEB can still run analysis if the CSVs already exist in raw_videos/.",
            )

        # c. Persist the project path via vieb_config
        set_dlc_project_path(project_dir)

        # d. Also write it into the GUI's live config dict and save to config.json
        self.cfg["dlc_project_path"] = project_dir
        _save_cfg(self.cfg)

        # e. Update the read-only display field
        self._import_path_le.setText(project_dir)

        # f. Success notification
        QMessageBox.information(self, "DLC Project Linked", "DLC project linked successfully.")

        # Sync the existing project-path line edit and status so the rest of the
        # UI (pose-estimation buttons, _validate_project, etc.) picks it up too.
        self._path_le.setText(project_dir)
        _register_project(project_dir)
        self._refresh_recent()

        # Show keypoint mapping panel
        bodyparts = dlc_cfg.get("bodyparts", [])
        if bodyparts:
            self._populate_keypoint_panel(bodyparts)

    # ── Keypoint mapping ──────────────────────────────────────────────────────

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

    def _populate_keypoint_panel(self, bodyparts: list):
        """Build (or rebuild) the three-column keypoint → role / object grid and show the panel."""
        self._keypoint_combos = {}
        self._keypoint_object_checks = {}

        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(4, 4, 4, 4)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(5)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 2)
        grid.setColumnStretch(2, 0)

        # Header row
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
            matched = self._match_role(name)
            combo.setCurrentIndex(
                _ROLE_OPTIONS.index(matched) if matched in _ROLE_OPTIONS else 0
            )

            chk = QCheckBox()
            chk.setToolTip(
                "Check if this keypoint tracks an object, not a body part.\n"
                "Object keypoints contribute to distance features but not posture calculations."
            )

            # Wire checkbox to disable/enable the role combo
            def _make_handler(c: QComboBox) -> object:
                state = {"prev": "— unassigned —"}

                def handler(checked: bool):
                    if checked:
                        state["prev"] = c.currentText() if c.isEnabled() else "— unassigned —"
                        c.clear()
                        c.addItem("— object point —")
                        c.setEnabled(False)
                    else:
                        c.clear()
                        for role in _ROLE_OPTIONS:
                            c.addItem(role)
                        prev = state["prev"]
                        idx = _ROLE_OPTIONS.index(prev) if prev in _ROLE_OPTIONS else 0
                        c.setCurrentIndex(idx)
                        c.setEnabled(True)

                return handler

            chk.toggled.connect(_make_handler(combo))

            grid.addWidget(lbl, row_idx, 0)
            grid.addWidget(combo, row_idx, 1)
            grid.addWidget(chk, row_idx, 2)
            self._keypoint_combos[name] = combo
            self._keypoint_object_checks[name] = chk

        # Pad so the grid doesn't stretch weirdly when there are few keypoints
        grid.setRowStretch(len(bodyparts) + 1, 1)

        self._keypoint_scroll.setWidget(container)
        self._keypoint_panel.show()

    def _save_keypoint_mapping(self):
        """Write keypoint_roles and object_keypoints to config.json."""
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

    def _try_preload_keypoints(self):
        """On startup, show the keypoint panel if a valid project is already linked."""
        try:
            project_dir = get_dlc_project_path()
            if not project_dir:
                return
            config_yaml = os.path.join(project_dir, "config.yaml")
            if not os.path.exists(config_yaml):
                return
            import yaml as _yaml
            with open(config_yaml, encoding="utf-8") as fh:
                dlc_cfg = _yaml.safe_load(fh)
            bodyparts = dlc_cfg.get("bodyparts", []) if isinstance(dlc_cfg, dict) else []
            if bodyparts:
                # Apply any previously saved role assignments as defaults
                saved_roles: dict = self.cfg.get("keypoint_roles", {})
                saved_objects: list = self.cfg.get("object_keypoints", [])
                self._populate_keypoint_panel(bodyparts)
                for name, combo in self._keypoint_combos.items():
                    if name in saved_roles and saved_roles[name] in _ROLE_OPTIONS:
                        combo.setCurrentIndex(_ROLE_OPTIONS.index(saved_roles[name]))
                # Restore object flags (setting the checkbox triggers the handler)
                for name, chk in self._keypoint_object_checks.items():
                    if name in saved_objects:
                        chk.setChecked(True)
            self._refresh_eval_panel(project_dir)
        except Exception:
            pass  # Non-critical startup enhancement — never crash here

    # ── Project management ────────────────────────────────────────────────────

    def _refresh_recent(self):
        self._recent_combo.blockSignals(True)
        self._recent_combo.clear()
        self._recent_combo.addItem("— select a recent project —")
        for p in _load_projects():
            label = f"{p.get('name', '?')}  ({p.get('added', '')})"
            self._recent_combo.addItem(label, p.get("path", ""))
        self._recent_combo.blockSignals(False)

        # Pre-fill from vieb_config if no explicit path is set
        if not self._path_le.text().strip():
            try:
                import vieb_config
                cur = vieb_config.get_dlc_project_path()
                if cur:
                    self._path_le.setText(cur)
            except Exception:
                pass

    def _load_from_recent(self, idx: int):
        if idx <= 0:
            return
        path = self._recent_combo.itemData(idx)
        if path:
            self._path_le.setText(path)

    def _on_path_changed(self, text: str):
        self._refresh_project_status()

    def _detect_and_show_status(self):
        """Check whether DLC has already been run and update the status banner (Zone 1)."""
        project_path = None
        csv_count = 0
        labels_count = 0

        # Check 1: config.json explicit path
        try:
            import vieb_config
            project_path = vieb_config.get_dlc_project_path()
        except Exception:
            pass

        # Check 2: auto-discovery
        if project_path is None:
            dlc_dir = _find_dlc_project()
            if dlc_dir:
                project_path = str(dlc_dir)

        # Check 3: labels.npy files already exist
        labels_count = len(list((RESULTS / "shared").glob("*_labels.npy"))) if (RESULTS / "shared").exists() else 0

        # Check 4: DLC CSVs in raw_videos/
        raw_dir = Path(self.cfg.get("raw_videos_dir", str(ROOT / "raw_videos")))
        if raw_dir.exists():
            csv_count = len(list(raw_dir.glob("*DLC*.csv")))

        self._dlc_project_path = project_path
        self._dlc_csv_count = csv_count
        self._dlc_labels_count = labels_count

        if project_path and not self._path_le.text().strip():
            self._path_le.setText(project_path)

        self._update_banner()
        if project_path:
            self._refresh_eval_panel(project_path)
        else:
            self._eval_panel.hide()

    @staticmethod
    def _load_eval_results(dlc_project_path: str) -> dict | None:
        """Return the best-snapshot row from a DLC CombinedEvaluation-results.csv, or None."""
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

    def _refresh_eval_panel(self, project_path: str) -> None:
        """Populate and show the Model Performance panel, or hide it if no results exist."""
        results = self._load_eval_results(project_path) if project_path else None
        if not results:
            self._eval_panel.hide()
            return

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

        def _set_card(label: QLabel, text: str, color: str | None = None):
            label.setText(text)
            base = "font-size:16pt;font-weight:bold;background:transparent;border:none;"
            label.setStyleSheet(base + f"color:{color};" if color else base + "color:#333;")

        test_map = results.get("test mAP")
        test_rmse = results.get("test rmse_pcutoff")
        train_map = results.get("train mAP")
        epochs = results.get("Training epochs")

        if test_map is not None:
            _set_card(self._eval_card_test_map, f"{float(test_map):.1f}%", _map_color(float(test_map)))
        if test_rmse is not None:
            _set_card(self._eval_card_test_rmse, f"{float(test_rmse):.2f}px", _rmse_color(float(test_rmse)))
        if train_map is not None:
            _set_card(self._eval_card_train_map, f"{float(train_map):.1f}%", _map_color(float(train_map)))
        if epochs is not None:
            _set_card(self._eval_card_epochs, f"{int(epochs)}")

        self._refresh_eval_thumbnails(project_path)
        self._eval_panel.show()

    def _refresh_eval_thumbnails(self, project_path: str) -> None:
        """Populate the labeled-image thumbnail strip below the stat cards."""
        while self._eval_thumb_row.count():
            item = self._eval_thumb_row.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        pattern = "evaluation-results-pytorch/iteration-*/*/LabeledImages_*/*.png"
        pngs = sorted(Path(project_path).glob(pattern))[:3]
        if not pngs:
            self._eval_thumb_label.hide()
            return

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
            self._eval_thumb_row.addWidget(thumb)
        self._eval_thumb_row.addStretch()
        self._eval_thumb_label.show()

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

    def _refresh_project_status(self):
        path = self._path_le.text().strip()
        if not path:
            self._project_status.setText("")
            self._update_import_visibility()
            return
        config_yaml = os.path.join(path, "config.yaml")
        if not os.path.isdir(path):
            self._project_status.setText("⚠ Directory not found.")
            self._project_status.setStyleSheet("color:#c62828;")
        elif not os.path.exists(config_yaml):
            self._project_status.setText("⚠ config.yaml not found in this directory.")
            self._project_status.setStyleSheet("color:#c62828;")
        else:
            self._project_status.setText(f"✓ Valid DLC project  —  {config_yaml}")
            self._project_status.setStyleSheet("color:#2e7d32;")
            # Persist to vieb_config so rest of the app can find it
            try:
                import vieb_config
                vieb_config.set_dlc_project_path(path)
            except Exception:
                pass
            _register_project(path)
            self._refresh_recent()
        self._update_import_visibility()

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
                cfg = _yaml.safe_load(f)
            if not isinstance(cfg, dict) or "bodyparts" not in cfg:
                QMessageBox.warning(
                    self,
                    "Invalid DLC Config",
                    f"The file does not appear to be a valid DLC config.yaml\n({config_file})",
                )
                return
        except Exception as exc:
            QMessageBox.warning(self, "Parse Error", f"Could not read config.yaml:\n{exc}")
            return
        self._path_le.setText(os.path.dirname(config_file))

    def _create_project(self):
        dlg = _CreateProjectDialog(self.cfg, self)
        if dlg.exec_() == QDialog.Accepted and dlg.result_path:
            self._path_le.setText(dlg.result_path)

    # ── Pretrained models ─────────────────────────────────────────────────────

    def _refresh_pretrained(self):
        self._pretrained_combo.clear()
        try:
            from pretrained_manager import list_available_pretrained
            models = list_available_pretrained()
        except Exception:
            models = []
        if models:
            for m in models:
                self._pretrained_combo.addItem(m.get("model_name", "?"))
        else:
            self._pretrained_combo.addItem("(no pretrained models found)")

    def _use_pretrained(self):
        name = self._pretrained_combo.currentText()
        if not name or name.startswith("("):
            QMessageBox.information(
                self,
                "No Model",
                "No pretrained models found in pretrained/\n"
                "Download a model from GitHub Releases and unzip it into pretrained/",
            )
            return
        self._run_dlc_subprocess(["setup_dlc_training.py", "--use-pretrained", name])

    # ── DLC subprocess helpers ────────────────────────────────────────────────

    def _run_dlc_subprocess(self, args: list[str]):
        if self._worker and self._worker.isRunning():
            self._log_human("⚠ A task is already running. Wait for it to finish.")
            return
        self._dlc_error_shown = False
        self._set_buttons_enabled(False)
        dlc_python = self.cfg.get("dlc_python") or sys.executable
        self._worker = SubprocessWorker(args, python_exe=dlc_python)
        self._worker.log.connect(self._on_raw_log)
        self._worker.done.connect(self._on_worker_done)
        self._worker.start()

    def _run_dlc(self, _unused, action: str):
        """Thin wrapper used by buttons that map to CLI flags."""
        mapping = {
            "extract_frames": ["setup_dlc_training.py"],  # default flow
        }
        args = mapping.get(action, [])
        if args:
            self._run_dlc_subprocess(args)

    def _open_labeling(self):
        self._show_labeling_guide(before_launch=True)
        self._run_dlc_subprocess(["setup_dlc_training.py", "--label"])

    def _run_pose_estimation(self):
        """Run training first (if toggled), then pose estimation."""
        if not self._validate_project():
            return
        if self._train_first.isChecked():
            if not self._validate_model(warn=False):
                reply = QMessageBox.question(
                    self,
                    "Train First?",
                    "Train model first is checked. This will run training then pose estimation.\n\n"
                    "Training can take 30 min – 2 hrs. Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return
            self._log_human("⏳ Training model first, then running pose estimation…")
            self._train_then_analyze = True
            self._run_dlc_subprocess(["setup_dlc_training.py", "--train"])
        else:
            if not self._validate_model(warn=True):
                return
            self._train_then_analyze = False
            self._log_human("⏳ Running pose estimation on all videos…")
            self._run_dlc_subprocess(["setup_dlc_training.py", "--analyze"])

    def _validate_project(self) -> bool:
        path = self._path_le.text().strip()
        if not path or not os.path.exists(os.path.join(path, "config.yaml")):
            QMessageBox.warning(
                self,
                "No DLC Project",
                "Please select a valid DLC project directory first.\n"
                "Use 'Browse…' or 'Create New Project…'",
            )
            return False
        return True

    def _validate_model(self, warn: bool = True) -> bool:
        """Return True if a trained DLC model snapshot exists."""
        path = self._path_le.text().strip()
        if not path:
            return False
        snapshots = list(Path(path).glob("dlc-models/**/train/snapshot-*.index"))
        if not snapshots:
            if warn:
                reply = QMessageBox.question(
                    self,
                    "No Trained Model Found",
                    f"No trained model found at:\n  {path}/dlc-models/\n\n"
                    "Would you like to train now?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply == QMessageBox.Yes:
                    self._train_first.setChecked(True)
                    self._run_pose_estimation()
            return False
        return True

    def _on_worker_done(self, ok: bool):
        self._set_buttons_enabled(True)
        self._detect_and_show_status()
        if ok:
            self._log_human("✓ Task completed successfully.")
            path = self._path_le.text().strip()
            if path:
                self._refresh_eval_panel(path)
            # If train-then-analyze: launch analyze now
            if getattr(self, "_train_then_analyze", False):
                self._train_then_analyze = False
                self._log_human("⏳ Training done — now running pose estimation…")
                self._run_dlc_subprocess(["setup_dlc_training.py", "--analyze"])
        else:
            self._check_dlc_error("")
            self._log_human("✕ Task failed — check the log above for details.")

    def _check_dlc_error(self, text: str) -> bool:
        """Detect a missing-DeepLabCut/torch import error and surface a helpful message."""
        if "ModuleNotFoundError: No module named 'deeplabcut'" in text or \
                "ModuleNotFoundError: No module named deeplabcut" in text or \
                "ModuleNotFoundError: No module named 'torch'" in text or \
                "ModuleNotFoundError: No module named torch" in text:
            if not getattr(self, "_dlc_error_shown", False):
                self._dlc_error_shown = True
                self._log.append(
                    f"<pre style='color:#ffb300;'>{_DLC_NOT_INSTALLED_MSG}</pre>"
                )
                sb = self._log.verticalScrollBar()
                sb.setValue(sb.maximum())
                QMessageBox.warning(self, "DeepLabCut Not Installed", _DLC_NOT_INSTALLED_MSG)
            return True
        return False

    def _set_buttons_enabled(self, enabled: bool):
        for b in (self._btn_extract, self._btn_label, self._btn_train,
                  self._btn_evaluate, self._btn_analyze):
            b.setEnabled(enabled)

    # ── Logging ───────────────────────────────────────────────────────────────

    def _on_raw_log(self, text: str):
        human = _translate_log(text)
        if human is not None:
            self._log.insertPlainText(human + "\n")
        else:
            self._log.insertPlainText(text)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())
        self._check_dlc_error(text)

    def _log_human(self, msg: str):
        self._log.insertPlainText(msg + "\n")
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ── Labeling guide (P11) ──────────────────────────────────────────────────

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
