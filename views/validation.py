from __future__ import annotations
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QKeySequence, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QShortcut, QSizePolicy, QSlider,
    QSpinBox, QSplitter, QStackedWidget,
    QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from _utils import ROOT, RESULTS, CLIPS, VALIDATION_DIR, _open_folder, _save_cfg, _CV2, _MPL

if _CV2:
    import cv2
if _MPL:
    from _utils import Figure, FigureCanvas, mpl_cm
    from _widgets import MplCanvas

from _widgets import VideoPlayer, KinematicsPanel

import characterize

BASE_DIR = Path(__file__).parent.parent.resolve()

# ---------------------------------------------------------------------------
# Clip Reviewer palette + helpers
# ---------------------------------------------------------------------------

_CAT_PALETTE = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
]


def _lighten(hex_color: str, factor: float = 0.45) -> str:
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    return "#{:02x}{:02x}{:02x}".format(
        int(r + (255 - r) * factor),
        int(g + (255 - g) * factor),
        int(b + (255 - b) * factor),
    )


def _get_state_name(state_id: int) -> str:
    """Return 'State N' or 'State N — Name' if config.json has state_names."""
    try:
        cfg_path = BASE_DIR / "config.json"
        if cfg_path.exists():
            import json
            with open(cfg_path) as fh:
                data = json.load(fh)
            name = data.get("state_names", {}).get(str(state_id), "")
            if name:
                return f"State {state_id} — {name}"
    except Exception:
        pass
    return f"State {state_id}"


# ---------------------------------------------------------------------------
# Background workers for classifier training / prediction
# ---------------------------------------------------------------------------

class _TrainWorker(QThread):
    done = pyqtSignal(dict)

    def __init__(self, annotations_path: str, features_index: dict,
                 shared_dir: str, output_path: str):
        super().__init__()
        self._ann_path = annotations_path
        self._fi = features_index
        self._shared = shared_dir
        self._out = output_path

    def run(self):
        try:
            result = characterize.train_classifier(
                self._ann_path, self._fi, self._shared, self._out
            )
        except Exception as exc:
            result = {"trained": False, "reason": str(exc)}
        self.done.emit(result)


class _PredictWorker(QThread):
    done = pyqtSignal(object)  # pd.DataFrame

    def __init__(self, classifier_path: str, shared_dir: str,
                 all_clips: dict, annotations_path: str, output_path: str):
        super().__init__()
        self._clf = classifier_path
        self._shared = shared_dir
        self._clips = all_clips
        self._ann = annotations_path
        self._out = output_path

    def run(self):
        try:
            df = characterize.predict_clips(
                self._clf, self._shared, self._clips, self._ann, self._out
            )
        except Exception:
            df = pd.DataFrame(
                columns=["clip_path", "state_id", "predicted_label", "confidence"]
            )
        self.done.emit(df)


# ---------------------------------------------------------------------------
# Clip Reviewer widget
# ---------------------------------------------------------------------------

class _ClipReviewerWidget(QWidget):
    """Session-based clip annotation tool."""
    navigate_help = pyqtSignal(str)

    _ANN_PATH  = str(RESULTS / "annotations" / "annotations.csv")
    _PRED_PATH = str(RESULTS / "annotations" / "predictions.csv")
    _CLF_PATH  = str(RESULTS / "annotations" / "classifier.pkl")

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._categories: list[dict] = []        # [{"name": str, "color": str}]
        self._chip_widgets: dict[str, QWidget] = {}
        self._annotations: dict[str, str] = {}   # {clip_path: label}
        self._predictions: dict[str, tuple] = {} # {clip_path: (label, conf)}
        self._clips_dict: dict[int, list] = {}   # {state_id: [clip_path]}
        self._all_clips_flat: list[str] = []
        self._session_clips: list[str] = []
        self._session_idx: int = 0
        self._session_active: bool = False
        self._shortcuts: list = []               # keep QShortcuts alive
        self._train_worker: _TrainWorker | None = None
        self._predict_worker: _PredictWorker | None = None
        self._build()
        self._load_state()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack)

        start_w = QWidget()
        self._build_start_panel(start_w)
        self._stack.addWidget(start_w)   # index 0

        review_w = QWidget()
        self._build_review_panel(review_w)
        self._stack.addWidget(review_w)  # index 1

        self._stack.setCurrentIndex(0)

    def _build_start_panel(self, parent: QWidget):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(40, 32, 40, 32)
        layout.setSpacing(14)
        layout.setAlignment(Qt.AlignTop)

        _cr_title_row = QHBoxLayout()
        title = QLabel("Clip Reviewer")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        _cr_title_row.addWidget(title)
        _cr_hbtn = QPushButton("?")
        _cr_hbtn.setFixedSize(20, 20)
        _cr_hbtn.setFlat(True)
        _cr_hbtn.setToolTip("Open Help for Clip Reviewer")
        _cr_hbtn.setCursor(Qt.PointingHandCursor)
        _cr_hbtn.setStyleSheet(
            "QPushButton{border:1px solid #aaa;border-radius:10px;color:#555;"
            "background:#f5f5f5;font-size:10px;font-weight:bold;}"
            "QPushButton:hover{background:#e8f0fe;color:#1a73e8;border-color:#1a73e8;}"
        )
        _cr_hbtn.clicked.connect(lambda: self.navigate_help.emit("clip_reviewer"))
        _cr_title_row.addWidget(_cr_hbtn)
        _cr_title_row.addStretch()
        layout.addLayout(_cr_title_row)
        layout.addSpacing(4)

        layout.addWidget(QLabel("Define your categories:"))

        self._cat_input = QLineEdit()
        self._cat_input.setPlaceholderText(
            "Type a category name and press Enter or comma to add.  "
            "Example: Success, Failure"
        )
        self._cat_input.returnPressed.connect(self._add_cat_from_input)
        self._cat_input.textChanged.connect(self._on_cat_input_changed)
        layout.addWidget(self._cat_input)

        # Tag chips — HBoxLayout inside a scroll area
        chips_scroll = QScrollArea()
        chips_scroll.setWidgetResizable(True)
        chips_scroll.setFixedHeight(52)
        chips_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        chips_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        chips_scroll.setStyleSheet("border:none;background:transparent;")

        chips_container = QWidget()
        self._chips_layout = QHBoxLayout(chips_container)
        self._chips_layout.setContentsMargins(4, 4, 4, 4)
        self._chips_layout.setSpacing(8)
        self._chips_layout.addStretch()
        chips_scroll.setWidget(chips_container)
        layout.addWidget(chips_scroll)

        # Shuffle seed
        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Shuffle seed (0 = random each session):"))
        self._seed_spin = QSpinBox()
        self._seed_spin.setRange(0, 99999)
        self._seed_spin.setValue(int(self.cfg.get("reviewer_seed", 0)))
        self._seed_spin.setMaximumWidth(80)
        seed_row.addWidget(self._seed_spin)
        seed_row.addStretch()
        layout.addLayout(seed_row)

        # Info label for no-clips case
        self._no_clips_lbl = QLabel(
            "No clips found. Run 'Generate Clips' from the Pipeline tab first."
        )
        self._no_clips_lbl.setStyleSheet(
            "color:#856404;background:#fff3cd;padding:8px;border-radius:4px;"
        )
        self._no_clips_lbl.setWordWrap(True)
        self._no_clips_lbl.setVisible(False)
        layout.addWidget(self._no_clips_lbl)

        self._start_btn = QPushButton("Start Session")
        self._start_btn.setEnabled(False)
        self._start_btn.setMinimumHeight(44)
        self._start_btn.clicked.connect(self._start_session)
        layout.addWidget(self._start_btn)
        layout.addStretch()

    def _build_review_panel(self, parent: QWidget):
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter, stretch=1)

        # ── Left: video player (65%) ──────────────────────────────────────
        left = QWidget()
        left_lay = QVBoxLayout(left)
        left_lay.setContentsMargins(4, 4, 4, 4)
        left_lay.setSpacing(4)

        self._review_player = VideoPlayer()
        self._review_player._loop_btn.setChecked(True)
        left_lay.addWidget(self._review_player, stretch=1)

        self._state_id_lbl = QLabel("")
        self._state_id_lbl.setAlignment(Qt.AlignCenter)
        self._state_id_lbl.setStyleSheet("color:#888;font-size:11px;")
        left_lay.addWidget(self._state_id_lbl)

        # "All clips reviewed" overlay (hidden by default)
        self._done_overlay = QWidget()
        done_lay = QVBoxLayout(self._done_overlay)
        done_lay.setAlignment(Qt.AlignCenter)
        done_msg = QLabel("All clips reviewed.\nEnd session or continue from beginning.")
        done_msg.setAlignment(Qt.AlignCenter)
        done_msg.setStyleSheet("font-size:14px;color:#333;")
        done_lay.addWidget(done_msg)
        self._restart_btn = QPushButton("Restart from beginning")
        self._restart_btn.clicked.connect(self._restart_session)
        done_lay.addWidget(self._restart_btn)
        self._done_overlay.setVisible(False)
        left_lay.addWidget(self._done_overlay)

        splitter.addWidget(left)

        # ── Right: controls (35%) ─────────────────────────────────────────
        right = QWidget()
        right.setMinimumWidth(240)
        right_lay = QVBoxLayout(right)
        right_lay.setContentsMargins(8, 8, 8, 8)
        right_lay.setSpacing(6)

        # Clip counter + progress bar
        self._clip_counter_lbl = QLabel("Clip 0 of 0")
        self._clip_counter_lbl.setFont(QFont("Arial", 11))
        right_lay.addWidget(self._clip_counter_lbl)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFormat("%p% annotated")
        right_lay.addWidget(self._progress_bar)

        # Distribution display
        dist_title = QLabel("Distribution:")
        dist_title.setFont(QFont("Arial", 10, QFont.Bold))
        right_lay.addWidget(dist_title)

        self._dist_widget = QWidget()
        self._dist_layout = QVBoxLayout(self._dist_widget)
        self._dist_layout.setContentsMargins(0, 0, 0, 0)
        self._dist_layout.setSpacing(3)
        right_lay.addWidget(self._dist_widget)

        right_lay.addSpacing(4)

        # Category buttons (rebuilt on session start)
        self._cat_btns_widget = QWidget()
        self._cat_btns_layout = QVBoxLayout(self._cat_btns_widget)
        self._cat_btns_layout.setContentsMargins(0, 0, 0, 0)
        self._cat_btns_layout.setSpacing(4)
        right_lay.addWidget(self._cat_btns_widget)

        right_lay.addSpacing(4)

        # Navigation buttons
        nav_row = QHBoxLayout()
        self._skip_btn = QPushButton("Skip")
        self._skip_btn.setStyleSheet("background:#777;color:white;")
        self._skip_btn.clicked.connect(self._skip)
        nav_row.addWidget(self._skip_btn)
        self._back_btn = QPushButton("Back")
        self._back_btn.clicked.connect(self._back)
        nav_row.addWidget(self._back_btn)
        right_lay.addLayout(nav_row)

        self._end_session_btn = QPushButton("End Session")
        self._end_session_btn.clicked.connect(self._end_session)
        right_lay.addWidget(self._end_session_btn)

        # Train Classifier
        right_lay.addSpacing(8)
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#ccc;")
        right_lay.addWidget(sep)

        self._train_btn = QPushButton("Train Classifier")
        self._train_btn.setToolTip("Need at least 5 clips per category to train")
        self._train_btn.setVisible(False)
        self._train_btn.clicked.connect(self._train_classifier)
        right_lay.addWidget(self._train_btn)

        self._retrain_btn = QPushButton("Re-train Classifier")
        self._retrain_btn.setToolTip("Retrain classifier with updated annotations")
        self._retrain_btn.setVisible(False)
        self._retrain_btn.clicked.connect(self._train_classifier)
        right_lay.addWidget(self._retrain_btn)

        right_lay.addStretch()

        splitter.addWidget(right)
        splitter.setSizes([650, 350])

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _load_state(self):
        """Load existing annotations, predictions, categories from config."""
        self._annotations = characterize.load_annotations(self._ANN_PATH)

        # Load predictions
        try:
            if os.path.exists(self._PRED_PATH):
                df = pd.read_csv(self._PRED_PATH)
                for _, row in df.iterrows():
                    self._predictions[str(row["clip_path"])] = (
                        str(row["predicted_label"]),
                        float(row.get("confidence", 0.0)),
                    )
        except Exception:
            pass

        # Restore categories from config
        for name in self.cfg.get("reviewer_categories", []):
            self._add_category(name)

        # Load clips from disk
        self._refresh_clips()

    def _refresh_clips(self):
        self._clips_dict = characterize.load_clips(str(CLIPS))
        self._all_clips_flat = [c for clips in self._clips_dict.values() for c in clips]
        has_clips = len(self._all_clips_flat) > 0
        self._no_clips_lbl.setVisible(not has_clips)
        self._update_start_btn()

    # ------------------------------------------------------------------
    # Category management
    # ------------------------------------------------------------------

    def _on_cat_input_changed(self, text: str):
        if text.endswith(","):
            name = text[:-1].strip()
            if name:
                self._add_category(name)
            self._cat_input.clear()

    def _add_cat_from_input(self):
        name = self._cat_input.text().strip().rstrip(",")
        if name:
            self._add_category(name)
        self._cat_input.clear()

    def _add_category(self, name: str):
        name = name.strip()
        if not name:
            return
        if any(c["name"] == name for c in self._categories):
            return
        if len(self._categories) >= len(_CAT_PALETTE):
            return

        idx = len(self._categories)
        color = _CAT_PALETTE[idx % len(_CAT_PALETTE)]
        self._categories.append({"name": name, "color": color})

        # Build chip widget
        chip = QWidget()
        chip_lay = QHBoxLayout(chip)
        chip_lay.setContentsMargins(10, 4, 6, 4)
        chip_lay.setSpacing(4)
        chip.setStyleSheet(
            f"background:{color};border-radius:12px;"
        )

        lbl = QLabel(name)
        lbl.setStyleSheet("color:white;font-weight:bold;background:transparent;border:none;")
        chip_lay.addWidget(lbl)

        x_btn = QPushButton("×")
        x_btn.setFixedSize(18, 18)
        x_btn.setStyleSheet(
            "background:transparent;color:white;border:none;"
            "font-size:14px;font-weight:bold;padding:0;"
        )
        x_btn.clicked.connect(lambda _, n=name: self._remove_category(n))
        chip_lay.addWidget(x_btn)

        # Insert before the trailing stretch
        insert_pos = max(0, self._chips_layout.count() - 1)
        self._chips_layout.insertWidget(insert_pos, chip)
        self._chip_widgets[name] = chip

        self._update_start_btn()

    def _remove_category(self, name: str):
        if name in self._chip_widgets:
            w = self._chip_widgets.pop(name)
            self._chips_layout.removeWidget(w)
            w.setParent(None)
            w.deleteLater()
        self._categories = [c for c in self._categories if c["name"] != name]
        self._update_start_btn()

    def _update_start_btn(self):
        ok = len(self._categories) >= 2 and len(self._all_clips_flat) > 0
        self._start_btn.setEnabled(ok)

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _start_session(self):
        seed = self._seed_spin.value()

        # Save categories to config
        self.cfg["reviewer_categories"] = [c["name"] for c in self._categories]
        self.cfg["reviewer_seed"] = seed
        _save_cfg(self.cfg)

        # Warn if classifier categories have changed
        self._check_classifier_compat()

        # Reload clips in case new ones were generated
        self._refresh_clips()
        if not self._all_clips_flat:
            return

        # Shuffle then sort: unannotated first, annotated at end
        shuffled = characterize.shuffle_clips(
            self._clips_dict, seed if seed > 0 else None
        )
        unannotated = [c for c in shuffled if c not in self._annotations]
        annotated   = [c for c in shuffled if c in self._annotations]
        self._session_clips = unannotated + annotated
        self._session_idx = 0
        self._session_active = True

        # Rebuild category buttons + shortcuts
        self._rebuild_cat_buttons()

        self._stack.setCurrentIndex(1)
        self._show_current_clip()
        self._update_distribution()
        self._update_train_btn()

    def _end_session(self):
        self._session_active = False
        self._clear_shortcuts()
        self._stack.setCurrentIndex(0)

    def _restart_session(self):
        self._session_idx = 0
        self._done_overlay.setVisible(False)
        self._review_player.setVisible(True)
        self._state_id_lbl.setVisible(True)
        self._show_current_clip()

    def _check_classifier_compat(self):
        if not os.path.exists(self._CLF_PATH):
            return
        try:
            import joblib
            saved = joblib.load(self._CLF_PATH)
            saved_classes = set(saved.get("classes", []))
            current_cats = {c["name"] for c in self._categories}
            if saved_classes and saved_classes != current_cats:
                QMessageBox.warning(
                    self,
                    "Categories Changed",
                    "Categories changed since last training. Re-train classifier.",
                )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Review actions
    # ------------------------------------------------------------------

    def _rebuild_cat_buttons(self):
        # Clear existing buttons
        while self._cat_btns_layout.count():
            item = self._cat_btns_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._clear_shortcuts()

        for i, cat in enumerate(self._categories):
            btn = QPushButton(f"{i + 1}.  {cat['name']}")
            btn.setMinimumHeight(40)
            btn.setStyleSheet(
                f"background:{cat['color']};color:white;"
                "font-weight:bold;border-radius:4px;"
            )
            btn.clicked.connect(lambda _, name=cat["name"]: self._annotate(name))
            self._cat_btns_layout.addWidget(btn)

            # Keyboard shortcut 1–9
            if i < 9:
                sc = QShortcut(QKeySequence(str(i + 1)), self)
                sc.activated.connect(lambda name=cat["name"]: self._annotate(name))
                self._shortcuts.append(sc)

    def _clear_shortcuts(self):
        for sc in self._shortcuts:
            sc.setEnabled(False)
            sc.deleteLater()
        self._shortcuts.clear()

    def _show_current_clip(self):
        if self._session_idx >= len(self._session_clips):
            # All clips reviewed
            self._review_player.setVisible(False)
            self._state_id_lbl.setVisible(False)
            self._done_overlay.setVisible(True)
            self._clip_counter_lbl.setText(
                f"Clip {len(self._session_clips)} of {len(self._session_clips)}"
            )
            return

        self._done_overlay.setVisible(False)
        self._review_player.setVisible(True)
        self._state_id_lbl.setVisible(True)

        clip = self._session_clips[self._session_idx]
        n = len(self._session_clips)
        self._clip_counter_lbl.setText(f"Clip {self._session_idx + 1} of {n}")

        # State ID label
        try:
            sid = int(Path(clip).parent.name.split("_")[1])
            self._state_id_lbl.setText(_get_state_name(sid))
        except Exception:
            self._state_id_lbl.setText("")

        if os.path.exists(clip):
            self._review_player.load(clip)
            self._review_player.play()
        else:
            self._review_player._display.setText(f"File not found:\n{clip}")

    def _annotate(self, label: str):
        if self._session_idx >= len(self._session_clips):
            return
        clip = self._session_clips[self._session_idx]
        self._annotations[clip] = label
        characterize.save_annotations({clip: label}, self._ANN_PATH)
        self._advance()
        self._update_distribution()
        self._update_train_btn()

    def _skip(self):
        self._advance()

    def _back(self):
        if self._session_idx > 0:
            self._session_idx -= 1
            self._show_current_clip()

    def _advance(self):
        self._session_idx += 1
        self._show_current_clip()

    # ------------------------------------------------------------------
    # Distribution display
    # ------------------------------------------------------------------

    def _update_distribution(self):
        # Clear existing rows
        while self._dist_layout.count():
            item = self._dist_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
            elif item.layout():
                _clear_layout(item.layout())

        dist = characterize.get_clip_distribution(
            self._annotations,
            self._all_clips_flat,
            predictions=self._predictions if self._predictions else None,
        )
        total = max(dist["total"], 1)

        # Update progress bar
        pct_annotated = int(dist["annotated"] / total * 100)
        self._progress_bar.setValue(pct_annotated)

        cat_map = {c["name"]: c["color"] for c in self._categories}

        def _add_bar(label: str, count: int, color: str, italic: bool = False):
            row = QHBoxLayout()
            bar = QProgressBar()
            bar.setRange(0, total)
            bar.setValue(count)
            bar.setFormat("")
            bar.setFixedHeight(12)
            bar.setStyleSheet(
                f"QProgressBar {{border:1px solid #ccc;border-radius:3px;background:#eee;}}"
                f"QProgressBar::chunk {{background:{color};border-radius:2px;}}"
            )
            row.addWidget(bar, stretch=1)
            pct = count / total * 100
            text = f"  {label}  {pct:.0f}%  ({count})"
            lbl = QLabel(text)
            lbl.setMinimumWidth(180)
            if italic:
                lbl.setStyleSheet("color:#888;font-style:italic;font-size:11px;")
            else:
                lbl.setStyleSheet("font-size:11px;")
            row.addWidget(lbl)

            row_w = QWidget()
            row_w.setLayout(row)
            self._dist_layout.addWidget(row_w)

        for cat in self._categories:
            n = dist["by_label"].get(cat["name"], 0)
            _add_bar(cat["name"], n, cat_map.get(cat["name"], "#888"))

            # Predicted row
            predicted = dist.get("by_label_predicted", {})
            n_pred = predicted.get(cat["name"], 0)
            if n_pred > 0:
                _add_bar(
                    f"{cat['name']} (pred.)",
                    n_pred,
                    _lighten(cat_map.get(cat["name"], "#888"), 0.45),
                    italic=True,
                )

        # Unannotated
        _add_bar("Unannotated", dist["unannotated"], "#cccccc")

    # ------------------------------------------------------------------
    # Train Classifier
    # ------------------------------------------------------------------

    def _update_train_btn(self):
        from collections import Counter
        label_counts = Counter(self._annotations.values())
        cats_with_ann = [c for c in self._categories if label_counts.get(c["name"], 0) > 0]

        if len(cats_with_ann) < 2:
            self._train_btn.setVisible(False)
            self._retrain_btn.setVisible(False)
            return

        self._train_btn.setVisible(True)

        min_count = min(label_counts.get(c["name"], 0) for c in self._categories)
        if min_count < 5:
            self._train_btn.setEnabled(False)
            self._train_btn.setToolTip("Need at least 5 clips per category")
        else:
            self._train_btn.setEnabled(True)
            self._train_btn.setToolTip("")

        self._retrain_btn.setVisible(os.path.exists(self._CLF_PATH))

    def _train_classifier(self):
        if self._train_worker and self._train_worker.isRunning():
            return

        # Progress dialog
        from PyQt5.QtWidgets import QProgressDialog
        prog = QProgressDialog("Training classifier…", None, 0, 0, self)
        prog.setWindowTitle("Training")
        prog.setModal(True)
        prog.setMinimumDuration(0)
        prog.show()

        shared_dir = str(RESULTS / "shared")
        fi_path = str(RESULTS / "features" / "index.json")
        features_index: dict = {}
        try:
            import json as _j
            with open(fi_path) as fh:
                features_index = _j.load(fh)
        except Exception:
            pass

        self._train_worker = _TrainWorker(
            self._ANN_PATH, features_index, shared_dir, self._CLF_PATH
        )

        def _on_done(report: dict):
            prog.close()
            self._on_train_done(report)

        self._train_worker.done.connect(_on_done)
        self._train_worker.start()

    def _on_train_done(self, report: dict):
        if not report.get("trained"):
            QMessageBox.warning(
                self, "Training Failed",
                report.get("reason", "Could not train classifier."),
            )
            return

        apply = self._show_train_results(report)
        self._update_train_btn()

        if apply:
            self._run_predict()

    def _show_train_results(self, report: dict) -> bool:
        """Show accuracy + confusion matrix. Return True if user wants to apply predictions."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Classifier Results")
        dlg.setMinimumWidth(400)
        layout = QVBoxLayout(dlg)

        # Accuracy
        acc = report.get("accuracy")
        if acc is not None:
            color = "green" if acc >= 0.80 else "orange"
            acc_lbl = QLabel(f"Cross-validation accuracy: {acc:.1%}")
            acc_lbl.setStyleSheet(
                f"color:{color};font-size:18px;font-weight:bold;"
            )
        else:
            acc_lbl = QLabel("Accuracy: N/A (fewer than 10 samples for CV)")
            acc_lbl.setStyleSheet("font-size:14px;color:#555;")
        acc_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(acc_lbl)

        # Confusion matrix
        classes = report.get("classes", [])
        cm_data = report.get("confusion_matrix", [])
        if classes and cm_data:
            cm_group = QGroupBox("Confusion Matrix (training data)")
            grid = QGridLayout(cm_group)
            for j, cls in enumerate(classes):
                lbl = QLabel(cls)
                lbl.setFont(QFont("Arial", 9, QFont.Bold))
                lbl.setAlignment(Qt.AlignCenter)
                grid.addWidget(lbl, 0, j + 1)
            for i, cls in enumerate(classes):
                lbl = QLabel(cls)
                lbl.setFont(QFont("Arial", 9, QFont.Bold))
                lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                grid.addWidget(lbl, i + 1, 0)
                for j, val in enumerate(cm_data[i]):
                    cell = QLabel(str(val))
                    cell.setAlignment(Qt.AlignCenter)
                    cell.setStyleSheet(
                        "padding:6px 12px;background:#f0f0f0;"
                        "border:1px solid #ddd;"
                    )
                    grid.addWidget(cell, i + 1, j + 1)
            layout.addWidget(cm_group)

        layout.addSpacing(8)
        q_lbl = QLabel("Apply predictions to unannotated clips?")
        q_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(q_lbl)

        btn_row = QHBoxLayout()
        yes_btn = QPushButton("Yes, apply predictions")
        no_btn  = QPushButton("No thanks")
        btn_row.addWidget(yes_btn)
        btn_row.addWidget(no_btn)
        layout.addLayout(btn_row)

        result = {"apply": False}
        yes_btn.clicked.connect(lambda: (result.__setitem__("apply", True), dlg.accept()))
        no_btn.clicked.connect(dlg.reject)
        dlg.exec_()
        return result["apply"]

    def _run_predict(self):
        if self._predict_worker and self._predict_worker.isRunning():
            return

        from PyQt5.QtWidgets import QProgressDialog
        prog = QProgressDialog("Applying predictions…", None, 0, 0, self)
        prog.setWindowTitle("Predicting")
        prog.setModal(True)
        prog.setMinimumDuration(0)
        prog.show()

        shared_dir = str(RESULTS / "shared")
        self._predict_worker = _PredictWorker(
            self._CLF_PATH, shared_dir, self._clips_dict,
            self._ANN_PATH, self._PRED_PATH,
        )

        def _on_done(df):
            prog.close()
            self._on_predict_done(df)

        self._predict_worker.done.connect(_on_done)
        self._predict_worker.start()

    def _on_predict_done(self, df: "pd.DataFrame"):
        # Reload predictions
        self._predictions.clear()
        if not df.empty:
            for _, row in df.iterrows():
                self._predictions[str(row["clip_path"])] = (
                    str(row["predicted_label"]),
                    float(row.get("confidence", 0.0)),
                )

        self._update_distribution()

        n = len(df)
        QMessageBox.information(
            self,
            "Predictions Saved",
            f"Predictions saved to results/annotations/predictions.csv.\n"
            f"{n} clips predicted.\n\n"
            "These are model predictions, not human labels.",
        )


def _clear_layout(layout):
    """Recursively remove all widgets and sub-layouts from a layout."""
    while layout.count():
        item = layout.takeAt(0)
        if item.widget():
            item.widget().deleteLater()
        elif item.layout():
            _clear_layout(item.layout())


class _ValidationPlayer(VideoPlayer):
    """VideoPlayer with loop-on default, 2x speed option, and frame counter."""

    def _build(self):
        super()._build()
        self._speed_combo.addItem("2x")
        self._loop_btn.setChecked(True)
        self._frame_lbl = QLabel("Frame: 0 / 0")
        self._frame_lbl.setAlignment(Qt.AlignCenter)
        self._frame_lbl.setStyleSheet("color:#888;font-size:11px;")
        self.layout().addWidget(self._frame_lbl)

    def _show(self, idx):
        super()._show(idx)
        self._frame_lbl.setText(f"Frame: {self._cur + 1} / {self._total}")

    def load(self, path: str):
        super().load(path)
        self._frame_lbl.setText(f"Frame: 1 / {self._total}")


class ValidationView(QWidget):
    navigate_to_pipeline = pyqtSignal()
    navigate_help = pyqtSignal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._data = {}
        self._current_clip: Path | None = None
        self._state_id: int | None = None
        # Advanced tab state (mirrors original frame-sampling implementation)
        self._adv_sample = None
        self._adv_cursor = 0
        self._adv_pose_cache: dict = {}
        self._adv_cap_cache: dict = {}
        self._adv_feature_cache: dict = {}
        self._adv_label_map = {
            Qt.Key_F: "freeze", Qt.Key_W: "walk", Qt.Key_G: "groom",
            Qt.Key_R: "rear", Qt.Key_O: "other", Qt.Key_S: "skip",
        }
        self.setFocusPolicy(Qt.StrongFocus)
        self._build()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(10)

        title = QLabel("Validation")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        outer.addWidget(title)
        outer.addSpacing(5)

        self._tabs = QTabWidget()
        outer.addWidget(self._tabs, stretch=1)

        # Clip Reviewer tab (first — new)
        self._clip_reviewer = _ClipReviewerWidget(self.cfg)
        self._clip_reviewer.navigate_help.connect(self.navigate_help.emit)
        self._tabs.insertTab(0, self._clip_reviewer, "Clip Reviewer")

        self._watch_widget = QWidget()
        self._tabs.addTab(self._watch_widget, "Video Watching")
        self._build_watching()

        self._adv_widget = QWidget()
        self._tabs.addTab(self._adv_widget, "Frame Sampling (Advanced)")
        self._build_sampling()

        self._tabs.setCurrentIndex(0)

    def _build_watching(self):
        layout = QHBoxLayout(self._watch_widget)
        layout.setSpacing(12)

        # ---- Left panel (fixed 300px, scrollable) ----
        left_scroll = QScrollArea()
        left_scroll.setFixedWidth(312)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(8)

        lbl_title = QLabel("Behavioral State Validation")
        lbl_title.setFont(QFont("Arial", 13, QFont.Bold))
        lbl_title.setWordWrap(True)
        ll.addWidget(lbl_title)

        subtitle = QLabel(
            "Watch each clip and label what behavior the mouse is performing. "
            "Your labels validate that the discovered states correspond to real behaviors."
        )
        subtitle.setStyleSheet("color:gray;font-style:italic;font-size:11px;")
        subtitle.setWordWrap(True)
        ll.addWidget(subtitle)

        ll.addSpacing(4)

        rater_row = QHBoxLayout()
        rater_row.addWidget(QLabel("Rater:"))
        self._rater_edit = QLineEdit(self.cfg.get("rater_name", "Rater 1"))
        self._rater_edit.editingFinished.connect(self._save_rater)
        rater_row.addWidget(self._rater_edit)
        ll.addLayout(rater_row)

        ll.addWidget(QLabel("Select state to validate:"))
        self._state_combo = QComboBox()
        self._state_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ll.addWidget(self._state_combo)

        self._load_btn = QPushButton("Load Random Clip")
        self._load_btn.clicked.connect(self._load_random_clip)
        ll.addWidget(self._load_btn)

        self._clip_name_lbl = QLabel("")
        self._clip_name_lbl.setStyleSheet("color:gray;font-size:11px;")
        self._clip_name_lbl.setWordWrap(True)
        ll.addWidget(self._clip_name_lbl)

        ll.addSpacing(4)

        for btn_text, label in [
            ("[F]  Freeze", "freeze"),
            ("[W]  Walk",   "walk"),
            ("[G]  Groom",  "groom"),
            ("[R]  Rear",   "rear"),
            ("[O]  Other",  "other"),
        ]:
            btn = QPushButton(btn_text)
            btn.setMinimumHeight(44)
            btn.clicked.connect(lambda _, l=label: self._save_label(l))
            ll.addWidget(btn)

        skip_btn = QPushButton("[S]  Skip")
        skip_btn.setMinimumHeight(36)
        skip_btn.clicked.connect(self._skip_clip)
        ll.addWidget(skip_btn)

        self._counter_lbl = QLabel("Labeled: 0 clips")
        self._counter_lbl.setStyleSheet("font-weight:bold;")
        ll.addWidget(self._counter_lbl)

        self._progress_lbl = QLabel("")
        self._progress_lbl.setStyleSheet("color:#555;font-size:11px;")
        self._progress_lbl.setWordWrap(True)
        ll.addWidget(self._progress_lbl)

        ll.addWidget(QLabel("Summary:"))
        self._summary_table = QTableWidget(0, 4)
        self._summary_table.setHorizontalHeaderLabels(["State", "N labeled", "Top label", "Agreement"])
        self._summary_table.horizontalHeader().setStretchLastSection(True)
        self._summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._summary_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._summary_table.setMaximumHeight(140)
        ll.addWidget(self._summary_table)

        self._export_btn = QPushButton("Export Validation Report")
        self._export_btn.clicked.connect(self._export_report)
        ll.addWidget(self._export_btn)

        ll.addStretch()
        left_scroll.setWidget(left)
        layout.addWidget(left_scroll)

        # ---- Right panel (fills remaining width): 70% video, 30% kinematics ----
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)

        self._player = _ValidationPlayer()
        self._player._display.setText("Select a state and click Load Random Clip")
        rl.addWidget(self._player, stretch=7)

        self._kin_panel = KinematicsPanel(parent=right)
        rl.addWidget(self._kin_panel, stretch=3)

        self._existing_lbl = QLabel("")
        self._existing_lbl.setAlignment(Qt.AlignCenter)
        self._existing_lbl.setStyleSheet("color:#888;font-style:italic;font-size:12px;")
        rl.addWidget(self._existing_lbl)

        # Cursor timer — updates kinematic panel at ~30fps
        self._kin_cursor_timer = QTimer(self)
        self._kin_cursor_timer.timeout.connect(self._update_kin_cursor)
        self._kin_cursor_timer.start(33)

        layout.addWidget(right, stretch=1)

    def _build_sampling(self):
        """Original frame-sampling implementation, preserved unchanged."""
        ll = QVBoxLayout(self._adv_widget)

        note = QLabel(
            "Advanced validation for paper figures.\n"
            "Requires characterize.py to have been run."
        )
        note.setStyleSheet(
            "background:#fff3cd;color:#856404;padding:10px;"
            "border-radius:4px;margin-bottom:6px;"
        )
        ll.addWidget(note)

        split = QHBoxLayout()
        split.setSpacing(10)
        ll.addLayout(split, stretch=2)

        left = QGroupBox("Frame Sampler")
        adv_ll = QVBoxLayout(left)
        self._adv_video_combo = QComboBox()
        self._adv_state_combo = QComboBox()
        self._adv_n_slider = QSlider(Qt.Horizontal)
        self._adv_n_slider.setRange(10, 200)
        self._adv_n_slider.setValue(50)
        self._adv_n_lbl = QLabel("Frames: 50")
        self._adv_n_slider.valueChanged.connect(lambda v: self._adv_n_lbl.setText(f"Frames: {v}"))
        self._adv_sample_btn = QPushButton("Sample Frames From Video")
        self._adv_sample_btn.clicked.connect(self._adv_sample_frames)
        adv_ll.addWidget(QLabel("Video to Validate"))
        adv_ll.addWidget(self._adv_video_combo)
        adv_ll.addWidget(QLabel("State"))
        adv_ll.addWidget(self._adv_state_combo)
        adv_ll.addWidget(self._adv_n_lbl)
        adv_ll.addWidget(self._adv_n_slider)
        adv_ll.addWidget(self._adv_sample_btn)
        self._adv_progress_lbl = QLabel("0 of 0 frames labeled")
        adv_ll.addWidget(self._adv_progress_lbl)
        split.addWidget(left, stretch=1)

        center = QGroupBox("Frame Display")
        cl = QVBoxLayout(center)
        self._adv_frame = QLabel("Select a video and sample frames to begin", alignment=Qt.AlignCenter)
        self._adv_frame.setMinimumSize(320, 240)
        self._adv_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._adv_frame.setStyleSheet("background:#111;color:#999;")
        cl.addWidget(self._adv_frame)
        self._adv_frame_info = QLabel("State: - | kinematics: -")
        self._adv_frame_info.setWordWrap(True)
        cl.addWidget(self._adv_frame_info)
        shortcuts = QLabel("Shortcuts: F=freeze  W=walk  G=groom  R=rear  O=other  S=skip")
        shortcuts.setStyleSheet("color:#666;")
        shortcuts.setWordWrap(True)
        cl.addWidget(shortcuts)
        split.addWidget(center, stretch=2)

        right = QGroupBox("Label Assignment")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 16, 8, 8)
        for name in ("Freeze", "Walk", "Groom", "Rear", "Other"):
            b = QPushButton(name)
            b.setMinimumHeight(44)
            b.clicked.connect(lambda _, n=name.lower(): self._adv_assign(n))
            rl.addWidget(b)
        skip = QPushButton("Skip")
        skip.setMinimumHeight(44)
        skip.clicked.connect(lambda: self._adv_assign("skip"))
        rl.addWidget(skip)
        rl.addStretch()
        split.addWidget(right, stretch=1)

        bottom = QGroupBox("Results")
        bottom.setMinimumHeight(180)
        bl_outer = QHBoxLayout(bottom)
        if _MPL:
            self._adv_cm_canvas = MplCanvas(figsize=(4, 2.2))
            bl_outer.addWidget(self._adv_cm_canvas, stretch=3)
        else:
            self._adv_cm_canvas = None
            bl_outer.addWidget(QLabel("Install matplotlib for confusion matrix heatmap."), stretch=3)
        bl_right = QVBoxLayout()
        self._adv_agree = QLabel("Agreement per state: -")
        self._adv_agree.setWordWrap(True)
        bl_right.addWidget(self._adv_agree)
        self._adv_export_btn = QPushButton("Export labels CSV")
        self._adv_export_btn.clicked.connect(self._adv_export)
        bl_right.addWidget(self._adv_export_btn)
        bl_right.addStretch()
        bl_outer.addLayout(bl_right, stretch=1)
        ll.addWidget(bottom)

    # ------------------------------------------------------------------
    # Data update (called from main GUI on reload)
    # ------------------------------------------------------------------

    def update_data(self, data: dict):
        self._data = data
        self._refresh_state_combo(data)
        self._refresh_summary()
        self._refresh_adv_combos(data)

    def _refresh_state_combo(self, data: dict):
        ci = data.get("cluster_info")
        clips_root = BASE_DIR / "clips"

        if ci is None:
            self._state_combo.clear()
            self._state_combo.addItem("No cluster data — run pipeline first", -1)
            self._load_btn.setEnabled(False)
            return

        if not clips_root.exists():
            self._load_btn.setEnabled(False)
            self._player._display.setText(
                "No clips directory found.\n\n"
                "Generate clips first:\n  python characterize.py --clips"
            )
        else:
            self._load_btn.setEnabled(True)

        n = int(ci.get("n_clusters", 0))
        ss = data.get("state_summary")
        label_map: dict[int, str] = {}
        if ss is not None and "state" in ss.columns and "heuristic_label" in ss.columns:
            for _, row in ss.iterrows():
                sid = int(row["state"])
                raw = str(row.get("heuristic_label", ""))
                parts = raw.split(":", 1)
                label_map[sid] = parts[1].strip() if len(parts) == 2 else raw

        prev = self._state_combo.currentData()
        self._state_combo.clear()
        for sid in range(n):
            hint = label_map.get(sid, "")
            text = f"State {sid} ({hint})" if hint else f"State {sid}"
            self._state_combo.addItem(text, sid)

        if prev is not None:
            for i in range(self._state_combo.count()):
                if self._state_combo.itemData(i) == prev:
                    self._state_combo.setCurrentIndex(i)
                    break

    def _refresh_adv_combos(self, data: dict):
        ci = data.get("cluster_info")
        lpf = data.get("labels_per_frame")
        if ci is None or lpf is None or lpf.empty:
            self._adv_sample_btn.setEnabled(False)
            self._adv_progress_lbl.setText("Run Characterization + Clip Export to generate this data.")
            return
        self._adv_sample_btn.setEnabled(True)
        n = int(ci.get("n_clusters", 0))
        summary = data.get("summary")
        dominant = -1
        if summary is not None:
            dominant = max(
                [(i, float(summary.get(f"state_{i}_frac", pd.Series([0])).mean())) for i in range(n)],
                key=lambda x: x[1],
            )[0]
        self._adv_state_combo.clear()
        for sid in range(n):
            if sid != dominant:
                self._adv_state_combo.addItem(f"State {sid}", sid)
        self._adv_video_combo.clear()
        if "stem" in lpf.columns:
            for s in sorted(lpf["stem"].dropna().astype(str).unique().tolist()):
                self._adv_video_combo.addItem(s, s)
        sample = data.get("validation_sample")
        if sample is not None and not sample.empty:
            cur_video = self._adv_video_combo.currentData() if self._adv_video_combo.count() else None
            cur_state = self._adv_state_combo.currentData() if self._adv_state_combo.count() else None
            resumed = sample
            if cur_video is not None and "stem" in sample.columns:
                resumed = resumed[resumed["stem"].astype(str) == str(cur_video)]
            if cur_state is not None and "cluster_label" in resumed.columns:
                resumed = resumed[resumed["cluster_label"] == int(cur_state)]
            self._adv_sample = resumed.reset_index(drop=True) if not resumed.empty else None
            if self._adv_sample is not None:
                self._adv_cursor = int((self._adv_sample["manual_label"].fillna("") != "").sum())
                self._adv_show_current()
                self._adv_refresh_results()

    # ------------------------------------------------------------------
    # Video-watching tab actions
    # ------------------------------------------------------------------

    def _save_rater(self):
        name = self._rater_edit.text().strip() or "Rater 1"
        self.cfg["rater_name"] = name
        _save_cfg(self.cfg)

    def _load_random_clip(self):
        state_id = self._state_combo.currentData()
        if state_id is None or state_id < 0:
            return

        clips_dir = BASE_DIR / "clips" / f"state_{state_id}"
        if not clips_dir.exists():
            self._player._display.setText(
                f"No clips found for State {state_id}.\n\n"
                "Generate clips first:\n  python characterize.py --clips"
            )
            self._clip_name_lbl.setText("")
            self._existing_lbl.setText("")
            self._kin_panel.clear()
            return

        clips = list(clips_dir.glob("*.mp4"))
        if not clips:
            self._player._display.setText(f"clips/state_{state_id}/ is empty.")
            self._kin_panel.clear()
            return

        clip = random.choice(clips)
        self._current_clip = clip
        self._state_id = state_id
        self._clip_name_lbl.setText(clip.name)
        self._player.load(str(clip))
        self._player.play()
        self._check_existing_label()
        self._load_clip_kinematics(clip)

    def _save_label(self, label: str):
        if self._current_clip is None:
            return
        rater = self._rater_edit.text().strip() or "Rater 1"
        row = {
            "state_id": self._state_id,
            "clip_filename": self._current_clip.name,
            "label": label,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "rater": rater,
        }
        labels_path = BASE_DIR / "results" / "validation" / "labels.csv"
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        if labels_path.exists():
            df = pd.read_csv(labels_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(labels_path, index=False)
        self._refresh_summary()
        self._load_random_clip()

    def _skip_clip(self):
        self._load_random_clip()

    def _check_existing_label(self):
        if self._current_clip is None:
            self._existing_lbl.setText("")
            return
        labels_path = BASE_DIR / "results" / "validation" / "labels.csv"
        if not labels_path.exists():
            self._existing_lbl.setText("")
            return
        df = pd.read_csv(labels_path)
        match = df[
            (df["state_id"] == self._state_id) &
            (df["clip_filename"] == self._current_clip.name)
        ]
        if not match.empty:
            self._existing_lbl.setText(f"Previously labeled: {match.iloc[-1]['label']}")
        else:
            self._existing_lbl.setText("")

    def _load_clip_kinematics(self, clip: Path):
        """Load feature data for the current clip in a background thread."""
        fi = self._data.get("feature_index") or {}
        if not fi:
            self._kin_panel.clear()
            return
        # Pick the first available feature file as a proxy for the clip's kinematics
        feat_path = None
        for vstem, info in fi.items():
            fp = (info.get("features_path") or "").replace("\\", "/")
            if fp and os.path.exists(fp):
                feat_path = fp
                break
        if feat_path is None:
            self._kin_panel.clear()
            return

        n_frames = self._player._total if hasattr(self._player, "_total") else 300
        import threading
        def _bg():
            self._kin_panel.load_clip(feat_path, 0, n_frames)
        threading.Thread(target=_bg, daemon=True).start()

    def _update_kin_cursor(self):
        """Called by QTimer at ~30fps to update the kinematic cursor position."""
        if hasattr(self, "_player") and hasattr(self._player, "_cur"):
            self._kin_panel.set_frame(self._player._cur)

    def _refresh_summary(self):
        labels_path = BASE_DIR / "results" / "validation" / "labels.csv"
        if not labels_path.exists():
            self._counter_lbl.setText("Labeled: 0 clips")
            self._progress_lbl.setText("")
            self._summary_table.setRowCount(0)
            return
        df = pd.read_csv(labels_path)
        self._counter_lbl.setText(f"Labeled: {len(df)} clips")

        rows = []
        progress_parts = []
        for sid, grp in df.groupby("state_id"):
            n = len(grp)
            vc = grp["label"].value_counts()
            top = vc.index[0] if not vc.empty else "-"
            agree = f"{100 * vc.iloc[0] / n:.0f}%" if not vc.empty else "-"
            rows.append((int(sid), n, top, agree))
            progress_parts.append(f"State {sid}: {n} clips labeled")

        self._progress_lbl.setText("  |  ".join(progress_parts))
        self._summary_table.setRowCount(len(rows))
        for r, (sid, n, top, agree) in enumerate(rows):
            self._summary_table.setItem(r, 0, QTableWidgetItem(str(sid)))
            self._summary_table.setItem(r, 1, QTableWidgetItem(str(n)))
            self._summary_table.setItem(r, 2, QTableWidgetItem(top))
            self._summary_table.setItem(r, 3, QTableWidgetItem(agree))
        self._summary_table.resizeColumnsToContents()

    def _export_report(self):
        labels_path = BASE_DIR / "results" / "validation" / "labels.csv"
        if not labels_path.exists():
            QMessageBox.information(self, "Validation", "No labels to export yet.")
            return
        df = pd.read_csv(labels_path)
        summary_rows = []
        for sid, grp in df.groupby("state_id"):
            n = len(grp)
            vc = grp["label"].value_counts()
            summary_rows.append({
                "state_id": sid,
                "n_labeled": n,
                "top_label": vc.index[0] if not vc.empty else "-",
                "agreement_pct": round(100 * vc.iloc[0] / n, 1) if not vc.empty else None,
            })
        out_dir = BASE_DIR / "results" / "validation"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "validation_summary.csv"
        pd.DataFrame(summary_rows).to_csv(out, index=False)
        QMessageBox.information(self, "Validation", f"Report saved to:\n{out}")

    def _run_characterize_clips(self):
        script = BASE_DIR / "characterize.py"
        try:
            kwargs = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
            subprocess.Popen([sys.executable, str(script), "--clips"],
                             cwd=str(BASE_DIR), **kwargs)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not launch characterize.py:\n{e}")

    # ------------------------------------------------------------------
    # Key shortcuts (active for whichever tab is current)
    # ------------------------------------------------------------------

    def keyPressEvent(self, e):
        if self._tabs.currentIndex() == 0:
            label_map = {
                Qt.Key_F: "freeze", Qt.Key_W: "walk", Qt.Key_G: "groom",
                Qt.Key_R: "rear", Qt.Key_O: "other",
            }
            if e.key() in label_map:
                self._save_label(label_map[e.key()])
                e.accept()
                return
            if e.key() == Qt.Key_S:
                self._skip_clip()
                e.accept()
                return
        else:
            if e.key() in self._adv_label_map:
                self._adv_assign(self._adv_label_map[e.key()])
                e.accept()
                return
        super().keyPressEvent(e)

    # ------------------------------------------------------------------
    # Advanced tab — original frame-sampling methods (prefixed _adv_)
    # ------------------------------------------------------------------

    def _adv_sample_frames(self):
        lpf = self._data.get("labels_per_frame")
        fi = self._data.get("feature_index") or {}
        if lpf is None or lpf.empty:
            return
        stem = self._adv_video_combo.currentData()
        if not stem:
            QMessageBox.information(self, "Validation", "Select a video to validate.")
            return
        sid = int(self._adv_state_combo.currentData())
        n = int(self._adv_n_slider.value())
        sub = lpf[(lpf["state"] == sid) & (lpf["stem"].astype(str) == str(stem))]
        if sub.empty:
            QMessageBox.information(self, "Validation",
                                    "No frames available for this state in selected video.")
            return
        if len(sub) > n:
            sub = sub.sample(n=n, random_state=42)
        sub = sub.copy()
        sub.rename(columns={"frame": "frame_idx"}, inplace=True)
        sub["cluster_label"] = sub["state"]
        sub["manual_label"] = ""
        sub["timestamp"] = ""
        for idx, row in sub.iterrows():
            s = row["stem"]
            info = fi.get(s, {}) if isinstance(fi, dict) else {}
            sub.at[idx, "video_path"] = info.get(
                "video_path", str(ROOT / "raw_videos" / f"{s}.mp4"))
            sub.at[idx, "csv_path"] = info.get("csv_path", "")
        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        sub.to_csv(VALIDATION_DIR / "current_sample.csv", index=False)
        self._adv_sample = sub.reset_index(drop=True)
        self._adv_cursor = 0
        self._adv_show_current()
        self._adv_refresh_results()

    def _adv_load_pose(self, csv_path):
        if csv_path in self._adv_pose_cache:
            return self._adv_pose_cache[csv_path]
        try:
            from main import load_pose
            pose, conf, _ = load_pose(csv_path)
            self._adv_pose_cache[csv_path] = (pose, conf)
            return pose, conf
        except Exception:
            return None, None

    def _adv_draw_frame(self, row):
        if not _CV2:
            self._adv_frame.setText(
                "opencv-python is required to display frames.\n"
                "Install it with:  pip install opencv-python"
            )
            return
        video = row.get("video_path", "")
        frame_idx = int(row.get("frame_idx", 0))
        csv_path = row.get("csv_path", "")
        if video and not os.path.isabs(video):
            video = str(ROOT / video)
        if not video or not os.path.exists(video):
            self._adv_frame.setText(
                f"Video not found:\n{video}\n\n"
                "Check that raw_videos/ contains the .mp4 files."
            )
            return
        cap = self._adv_cap_cache.get(video)
        if cap is None:
            cap = cv2.VideoCapture(video)
            self._adv_cap_cache[video] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            self._adv_frame.setText(f"Could not read frame {frame_idx} from:\n{os.path.basename(video)}")
            return
        pose, _ = self._adv_load_pose(csv_path)
        if pose is not None and frame_idx < len(pose):
            pts = pose[frame_idx]
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
            ]
            for i, pt in enumerate(pts):
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, colors[i], -1)
            for a, b in [(2, 3), (3, 6), (0, 1)]:
                cv2.line(frame, tuple(np.int32(pts[a])), tuple(np.int32(pts[b])), (255, 255, 255), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        mxw, mxh = self._adv_frame.width(), self._adv_frame.height()
        sc = min(mxw / w, mxh / h)
        nw, nh = int(w * sc), int(h * sc)
        frame = cv2.resize(frame, (nw, nh))
        buf = frame.copy()
        img = QImage(buf.data, nw, nh, 3 * nw, QImage.Format_RGB888)
        self._adv_frame.setPixmap(QPixmap.fromImage(img))

    def _adv_kinematic_text(self, row):
        fi = self._data.get("feature_index") or {}
        stem = row.get("stem", "")
        frame_idx = int(row.get("frame_idx", 0))
        info = fi.get(stem, {}) if isinstance(fi, dict) else {}
        fp = info.get("features_path", "")
        if not fp:
            return "-"
        arr = self._adv_feature_cache.get(fp)
        if arr is None and Path(fp).exists():
            arr = np.load(fp)
            self._adv_feature_cache[fp] = arr
        if arr is None or frame_idx >= len(arr):
            return "-"
        feat = arr[frame_idx]
        return f"speed={feat[36]:.3f}, ang_vel={feat[39]:.3f}, entropy={feat[40]:.3f}"

    def _adv_show_current(self):
        if self._adv_sample is None or self._adv_sample.empty:
            self._adv_frame.setText("No sample loaded.")
            return
        unl = self._adv_sample["manual_label"].fillna("") == ""
        if unl.sum() == 0:
            self._adv_frame.setText("All frames labeled.")
            return
        self._adv_cursor = int(self._adv_sample.index[unl][0])
        row = self._adv_sample.loc[self._adv_cursor]
        self._adv_draw_frame(row)
        self._adv_frame_info.setText(
            f"State {int(row.get('cluster_label', -1))} | "
            f"frame {int(row.get('frame_idx', 0))} | {self._adv_kinematic_text(row)}"
        )
        done = int((self._adv_sample["manual_label"].fillna("") != "").sum())
        self._adv_progress_lbl.setText(f"{done} of {len(self._adv_sample)} frames labeled")

    def _adv_assign(self, manual_label: str):
        if self._adv_sample is None or self._adv_sample.empty:
            return
        self._adv_sample.at[self._adv_cursor, "manual_label"] = manual_label
        self._adv_sample.at[self._adv_cursor, "timestamp"] = datetime.now().isoformat(timespec="seconds")
        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        self._adv_sample.to_csv(VALIDATION_DIR / "current_sample.csv", index=False)
        self._adv_sample.to_csv(VALIDATION_DIR / "frame_labels.csv", index=False)
        self._adv_refresh_results()
        self._adv_show_current()

    def _adv_refresh_results(self):
        if self._adv_sample is None or self._adv_sample.empty:
            return
        done = self._adv_sample[self._adv_sample["manual_label"].fillna("") != ""]
        if done.empty:
            self._adv_agree.setText("Agreement per state: -")
            if self._adv_cm_canvas:
                self._adv_cm_canvas.ax.clear()
                self._adv_cm_canvas.ax.text(0.5, 0.5, "No labels yet", ha="center", va="center")
                self._adv_cm_canvas.draw()
            return
        cros = pd.crosstab(done["cluster_label"], done["manual_label"])
        if self._adv_cm_canvas:
            self._adv_cm_canvas.ax.clear()
            self._adv_cm_canvas.ax.imshow(cros.values, aspect="auto", cmap="Blues")
            self._adv_cm_canvas.ax.set_xticks(range(len(cros.columns)))
            self._adv_cm_canvas.ax.set_xticklabels(cros.columns, rotation=45, ha="right")
            self._adv_cm_canvas.ax.set_yticks(range(len(cros.index)))
            self._adv_cm_canvas.ax.set_yticklabels(cros.index)
            self._adv_cm_canvas.ax.set_xlabel("Manual Label")
            self._adv_cm_canvas.ax.set_ylabel("Cluster")
            self._adv_cm_canvas.fig.tight_layout()
            self._adv_cm_canvas.draw()
        agreements = []
        for sid, grp in done.groupby("cluster_label"):
            top = grp["manual_label"].value_counts().max()
            agreements.append(f"S{sid}: {100 * top / len(grp):.1f}%")
        self._adv_agree.setText("Agreement per state: " + ", ".join(agreements))

    def _adv_export(self):
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
