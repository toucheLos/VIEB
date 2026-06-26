from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QPushButton, QScrollArea, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QToolButton,
    QVBoxLayout, QWidget,
)

import vieb_config as _vc
from _utils import RESULTS, _MPL, _save_cfg
from _workers import SubprocessWorker
from views.analysis import TerminalBox, _placeholder, _section_title

if _MPL:
    from _widgets import MplCanvas

_BEHAVIORAL_CATEGORIES = [
    "Freezing", "Walking", "Grooming", "Rearing",
    "Running", "Exploring", "Other",
]
_TECHNICAL_CATEGORIES = [
    "Low Velocity", "High Velocity",
    "Low Elongation", "High Elongation",
    "Short Bouts", "Long Bouts",
    "High Angular Vel.", "Low Angular Vel.",
    "High Rearing", "Low Rearing",
]

_CHIP_STYLE = (
    "QPushButton{background:#f0f0f0;border:1px solid #ccc;border-radius:12px;"
    "padding:4px 10px;font-size:11px;}"
    "QPushButton:checked{background:#4E79A7;color:white;border-color:#4E79A7;}"
    "QPushButton:hover:!checked{background:#e0e8f0;}"
)
_TECH_CHIP_STYLE = (
    "QPushButton{background:#f5f5f5;border:1px solid #ddd;border-radius:12px;"
    "padding:3px 8px;font-size:10px;color:#666;}"
    "QPushButton:checked{background:#76B7B2;color:white;border-color:#76B7B2;}"
    "QPushButton:hover:!checked{background:#e8f0f0;}"
)
_CARD_STYLE = (
    "QFrame{background:#F8F9FA;border:1px solid #E0E0E0;border-radius:6px;"
    "padding:6px 10px;}"
)


# ---------------------------------------------------------------------------
# Small stat card
# ---------------------------------------------------------------------------

def _stat_card(title: str) -> tuple[QFrame, QLabel, QLabel]:
    """Return (frame, title_label, value_label)."""
    frame = QFrame()
    frame.setStyleSheet(_CARD_STYLE)
    fl = QVBoxLayout(frame)
    fl.setContentsMargins(6, 4, 6, 4)
    fl.setSpacing(1)
    t = QLabel(title)
    t.setStyleSheet("font-size:10px; color:#888;")
    v = QLabel("—")
    v.setStyleSheet("font-size:14px; font-weight:bold; color:#1A1A1A;")
    fl.addWidget(t)
    fl.addWidget(v)
    return frame, t, v


# ---------------------------------------------------------------------------
# StateCharacterizationView
# ---------------------------------------------------------------------------

class StateCharacterizationView(QWidget):
    """Browse discovered behavioral states: kinematic profiles, context
    enrichment, free-text labeling, and example clips."""

    worker_running = pyqtSignal(bool)

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.cfg = cfg or {}
        self._data: dict = {}
        self._worker = None
        self._state_ids: list[int] = []
        self._heuristic_labels: dict[int, str] = {}
        self._kin_metrics: dict[str, float] = {}
        self._ref_width = 900
        self._build()

    # ─────────────────────────────────────────────────────────── build ──

    def _make_header(self, title: str, run_label: str, run_slot, terminal: TerminalBox) -> QWidget:
        outer = QWidget()
        lay = QVBoxLayout(outer)
        lay.setContentsMargins(0, 0, 0, 6)
        lay.setSpacing(4)

        top = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setFont(QFont("Arial", 14, QFont.Bold))
        top.addWidget(lbl)
        top.addStretch()
        run_btn = QPushButton(run_label)
        run_btn.setFixedHeight(30)
        run_btn.clicked.connect(run_slot)
        top.addWidget(run_btn)
        lay.addLayout(top)
        lay.addWidget(terminal)
        return outer

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(8)

        self._terminal = TerminalBox()
        hdr = self._make_header(
            "State Characterization", "Run Characterization",
            lambda: self._run_command(["state_characterizer.py"], self._terminal),
            self._terminal,
        )
        lay.addWidget(hdr)

        self._splitter = QSplitter(Qt.Horizontal)
        splitter = self._splitter

        # ── Left: state list ──────────────────────────────────────────────
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 8, 0)
        ll.setSpacing(4)

        toggle_row = QHBoxLayout()
        toggle_row.setContentsMargins(0, 0, 0, 0)
        self._panel_title = QLabel("States")
        self._panel_title.setStyleSheet(
            "font-weight:bold; font-size:12px; color:#1A1A1A;"
        )
        toggle_row.addWidget(self._panel_title)
        toggle_row.addStretch()
        self._collapse_btn = QPushButton("«")
        self._collapse_btn.setFixedSize(22, 22)
        self._collapse_btn.setCursor(Qt.PointingHandCursor)
        self._collapse_btn.setToolTip("Collapse state list")
        self._collapse_btn.setStyleSheet(
            "QPushButton{background:transparent;color:#9B9B9B;border:none;font-size:13px;}"
            "QPushButton:hover{color:#1A1A1A;background:rgba(0,0,0,0.05);border-radius:4px;}"
        )
        self._collapse_btn.clicked.connect(self._toggle_state_panel)
        toggle_row.addWidget(self._collapse_btn)
        ll.addLayout(toggle_row)

        self._search = QLineEdit()
        self._search.setPlaceholderText("Search states…")
        self._search.textChanged.connect(self._on_search)
        ll.addWidget(self._search)

        sort_row = QHBoxLayout()
        sort_row.setSpacing(4)
        self._sort_lbl = QLabel("Sort:")
        self._sort_lbl.setStyleSheet("font-size:11px; color:#666;")
        sort_row.addWidget(self._sort_lbl)
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["State ID", "Speed", "Elongation", "Bout Duration", "Angular Velocity"])
        self._sort_combo.setStyleSheet("font-size:11px;")
        self._sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        sort_row.addWidget(self._sort_combo, stretch=1)
        self._sort_asc = QToolButton()
        self._sort_asc.setText("↑")
        self._sort_asc.setCheckable(True)
        self._sort_asc.setChecked(True)
        self._sort_asc.setToolTip("Toggle ascending / descending")
        self._sort_asc.setFixedWidth(26)
        self._sort_asc.toggled.connect(self._on_sort_changed)
        sort_row.addWidget(self._sort_asc)
        self._sort_row_widget = QWidget()
        self._sort_row_widget.setLayout(sort_row)
        ll.addWidget(self._sort_row_widget)

        self._state_list = QListWidget()
        self._state_list.currentRowChanged.connect(self._on_state_selected)
        ll.addWidget(self._state_list, stretch=1)

        self._collapsed_list = QListWidget()
        self._collapsed_list.setStyleSheet(
            "QListWidget{border:none; font-size:11px; font-weight:bold;}"
            "QListWidget::item{padding:2px 4px; text-align:center;}"
            "QListWidget::item:selected{background:#4E79A7; color:white;"
            "border-radius:3px;}"
        )
        self._collapsed_list.setFixedWidth(40)
        self._collapsed_list.currentRowChanged.connect(self._on_collapsed_selected)
        self._collapsed_list.hide()
        ll.addWidget(self._collapsed_list, stretch=1)

        self._left_panel = left
        self._panel_collapsed = False
        self._left_expanded_width = 240
        self._collapse_threshold = 100
        left.setMinimumWidth(56)
        splitter.addWidget(left)

        # ── Right: scrollable detail panel ───────────────────────────────
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 0, 4, 8)
        rl.setSpacing(8)

        self._detail_placeholder = _placeholder("Select a state from the list.")
        rl.addWidget(self._detail_placeholder)

        # ── Stat cards row ──
        cards_row = QHBoxLayout()
        cards_row.setSpacing(8)
        self._card_frames: list[QFrame] = []
        self._card_values: dict[str, QLabel] = {}
        for key, title in [
            ("occupancy", "Occupancy"),
            ("n_bouts", "# Bouts"),
            ("mean_dur", "Mean Duration"),
            ("median_dur", "Median Duration"),
        ]:
            frame, _, val_lbl = _stat_card(title)
            self._card_values[key] = val_lbl
            self._card_frames.append(frame)
            cards_row.addWidget(frame, stretch=1)
        self._cards_widget = QWidget()
        self._cards_widget.setLayout(cards_row)
        self._cards_widget.hide()
        rl.addWidget(self._cards_widget)

        # ── Profile tabs (kinematic chart + top features + group enrichment) ──
        self._profile_tabs = QTabWidget()
        self._profile_tabs.setStyleSheet(
            "QTabBar::tab{padding:4px 10px;font-size:11px;}"
            "QTabBar::tab:selected{background:#fff;border-bottom:2px solid #4E79A7;}"
        )
        self._profile_tabs.hide()

        # Tab 0: Kinematic Profile
        kin_page = QWidget()
        kin_lay = QVBoxLayout(kin_page)
        kin_lay.setContentsMargins(4, 4, 4, 4)
        if _MPL:
            self._kin_canvas = MplCanvas(figsize=(6, 2.5))
            kin_lay.addWidget(self._kin_canvas)
        else:
            self._kin_canvas = None
            kin_lay.addWidget(_placeholder("Install matplotlib to see kinematic profiles."))
        self._profile_tabs.addTab(kin_page, "Kinematic Profile")

        # Tab 1: Top Features
        feat_page = QWidget()
        feat_lay = QVBoxLayout(feat_page)
        feat_lay.setContentsMargins(4, 4, 4, 4)
        feat_desc = QLabel(
            "Features with the highest z-scores distinguish this state from the global mean. "
            "Positive = above average; negative = below average."
        )
        feat_desc.setWordWrap(True)
        feat_desc.setStyleSheet("color:#666; font-size:11px; padding-bottom:4px;")
        feat_lay.addWidget(feat_desc)

        self._feat_table = QTableWidget(0, 3)
        self._feat_table.setHorizontalHeaderLabels(["Feature", "Z-Score", "Direction"])
        self._feat_table.horizontalHeader().setStretchLastSection(False)
        self._feat_table.horizontalHeader().setSectionResizeMode(0, self._feat_table.horizontalHeader().Stretch)
        self._feat_table.horizontalHeader().setSectionResizeMode(1, self._feat_table.horizontalHeader().ResizeToContents)
        self._feat_table.horizontalHeader().setSectionResizeMode(2, self._feat_table.horizontalHeader().ResizeToContents)
        self._feat_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._feat_table.setSelectionMode(QTableWidget.NoSelection)
        self._feat_table.verticalHeader().setVisible(False)
        self._feat_table.setStyleSheet(
            "QTableWidget{border:none;font-size:11px;}"
            "QTableWidget::item{padding:2px 6px;}"
        )
        self._feat_no_data = _placeholder(
            "Feature z-scores not available.\n"
            "Click 'Run Characterization' to compute them."
        )
        feat_lay.addWidget(self._feat_no_data)
        feat_lay.addWidget(self._feat_table)
        self._feat_table.hide()
        self._profile_tabs.addTab(feat_page, "Top Features")

        # Tab 2: Group Enrichment
        enrich_page = QWidget()
        enrich_lay = QVBoxLayout(enrich_page)
        enrich_lay.setContentsMargins(4, 4, 4, 4)
        enrich_desc = QLabel(
            "Fraction of bouts in this state by experimental group. "
            "Enrichment ratio > 1 means this state occurs more than expected in that group."
        )
        enrich_desc.setWordWrap(True)
        enrich_desc.setStyleSheet("color:#666; font-size:11px; padding-bottom:4px;")
        enrich_lay.addWidget(enrich_desc)

        self._enrich_table = QTableWidget(0, 4)
        self._enrich_table.setHorizontalHeaderLabels(["Variable", "Group", "Fraction", "Enrichment"])
        self._enrich_table.horizontalHeader().setSectionResizeMode(0, self._enrich_table.horizontalHeader().ResizeToContents)
        self._enrich_table.horizontalHeader().setSectionResizeMode(1, self._enrich_table.horizontalHeader().ResizeToContents)
        self._enrich_table.horizontalHeader().setSectionResizeMode(2, self._enrich_table.horizontalHeader().ResizeToContents)
        self._enrich_table.horizontalHeader().setSectionResizeMode(3, self._enrich_table.horizontalHeader().Stretch)
        self._enrich_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._enrich_table.setSelectionMode(QTableWidget.NoSelection)
        self._enrich_table.verticalHeader().setVisible(False)
        self._enrich_table.setStyleSheet(
            "QTableWidget{border:none;font-size:11px;}"
            "QTableWidget::item{padding:2px 6px;}"
        )
        self._enrich_no_data = _placeholder(
            "Group enrichment not available.\n"
            "Click 'Run Characterization' after clustering to compute enrichment."
        )
        enrich_lay.addWidget(self._enrich_no_data)
        enrich_lay.addWidget(self._enrich_table)
        self._enrich_table.hide()
        self._profile_tabs.addTab(enrich_page, "Group Enrichment")

        rl.addWidget(self._profile_tabs)

        # ── Heuristic label hint ──
        self._heuristic_lbl = QLabel("")
        self._heuristic_lbl.setStyleSheet(
            "color:#555; font-size:11px; font-style:italic; padding:2px 0;"
        )
        self._heuristic_lbl.setWordWrap(True)
        self._heuristic_lbl.hide()
        rl.addWidget(self._heuristic_lbl)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#DCDCDC;")
        rl.addWidget(sep)

        # ── Clip controls ──
        clip_row = QHBoxLayout()
        self._load_clip_btn = QPushButton("Load Clip")
        self._load_clip_btn.setEnabled(False)
        self._load_clip_btn.clicked.connect(self._load_clip)
        clip_row.addWidget(self._load_clip_btn)
        self._autoplay_cb = QCheckBox("Autoplay")
        self._autoplay_cb.setChecked(True)
        clip_row.addWidget(self._autoplay_cb)
        self._clip_status = QLabel("")
        self._clip_status.setStyleSheet("color:#888; font-size:11px;")
        clip_row.addWidget(self._clip_status, stretch=1)
        rl.addLayout(clip_row)

        # ── Label section ──
        self._lbl_title = QLabel("Label this state:")
        self._lbl_title.setStyleSheet("font-weight:bold; font-size:12px; color:#333;")
        rl.addWidget(self._lbl_title)

        input_row = QHBoxLayout()
        self._label_input = QLineEdit()
        self._label_input.setPlaceholderText("Free-text label, or select a category below")
        input_row.addWidget(self._label_input, stretch=1)
        self._save_btn = QPushButton("Save")
        self._save_btn.clicked.connect(self._save_state_label)
        input_row.addWidget(self._save_btn)
        self._back_btn = QPushButton("← Back")
        self._back_btn.setToolTip("Go back to previous state")
        self._back_btn.clicked.connect(self._go_back)
        input_row.addWidget(self._back_btn)
        self._save_next_btn = QPushButton("Save && Next")
        self._save_next_btn.setToolTip("Save label and advance to next state")
        self._save_next_btn.clicked.connect(self._save_and_next)
        input_row.addWidget(self._save_next_btn)
        rl.addLayout(input_row)

        self._cat_group = QButtonGroup(self)
        self._cat_group.setExclusive(True)
        self._cat_buttons: dict[str, QPushButton] = {}

        cat_row = QHBoxLayout()
        cat_row.setSpacing(4)
        self._cat_label = QLabel("Category:")
        cat_row.addWidget(self._cat_label)
        for name in _BEHAVIORAL_CATEGORIES:
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setStyleSheet(_CHIP_STYLE)
            self._cat_group.addButton(btn)
            cat_row.addWidget(btn)
            self._cat_buttons[name] = btn
        self._more_btn = QToolButton()
        self._more_btn.setText("More ▸")
        self._more_btn.setCheckable(True)
        self._more_btn.setStyleSheet("font-size:11px; color:#666; border:none;")
        self._more_btn.toggled.connect(self._toggle_technical_cats)
        cat_row.addWidget(self._more_btn)
        cat_row.addStretch()
        rl.addLayout(cat_row)

        self._tech_row_widget = QWidget()
        tech_inner = QHBoxLayout(self._tech_row_widget)
        tech_inner.setContentsMargins(0, 0, 0, 0)
        tech_inner.setSpacing(4)
        for name in _TECHNICAL_CATEGORIES:
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setStyleSheet(_TECH_CHIP_STYLE)
            self._cat_group.addButton(btn)
            tech_inner.addWidget(btn)
            self._cat_buttons[name] = btn
        tech_inner.addStretch()
        self._tech_row_widget.hide()
        rl.addWidget(self._tech_row_widget)

        self._custom_cat_row = QHBoxLayout()
        self._custom_cat_row.setSpacing(4)
        self._custom_cat_chips_layout = self._custom_cat_row
        self._cat_input = QLineEdit()
        self._cat_input.setPlaceholderText("New category…")
        self._cat_input.setMaximumWidth(160)
        self._cat_input.returnPressed.connect(self._add_custom_category)
        self._custom_cat_row.addWidget(self._cat_input)
        add_cat_btn = QPushButton("+")
        add_cat_btn.setFixedWidth(28)
        add_cat_btn.setToolTip("Add custom category")
        add_cat_btn.clicked.connect(self._add_custom_category)
        self._custom_cat_row.addWidget(add_cat_btn)
        self._custom_cat_row.addStretch()
        rl.addLayout(self._custom_cat_row)
        self._restore_custom_categories()

        try:
            from _widgets import VideoPlayer
            self._player = VideoPlayer()
            rl.addWidget(self._player, stretch=1)
        except Exception:
            self._player = None
            rl.addWidget(_placeholder(
                "Video player unavailable.\nInstall opencv-python to enable."
            ))

        scroll.setWidget(right)
        splitter.addWidget(scroll)
        splitter.setSizes([self._left_expanded_width, 660])
        splitter.splitterMoved.connect(self._on_splitter_moved)
        lay.addWidget(splitter, stretch=1)

    # ────────────────────────────────────────────────── Data loading ──

    def update_data(self, data: dict) -> None:
        self._data = data
        self._load()

    def refresh(self, data: dict) -> None:
        self._data = data
        self._load()

    def _load(self) -> None:
        ss = self._data.get("state_summary")
        ctx = self._data.get("context_report")

        self._state_list.clear()
        self._state_ids = []
        self._load_clip_btn.setEnabled(False)
        self._clip_status.setText("")
        self._detail_placeholder.show()
        self._cards_widget.hide()
        self._profile_tabs.hide()
        self._heuristic_lbl.hide()

        if ss is None or ss.empty:
            self._state_list.addItem("No data — run state_characterizer.py")
            return

        id_col = next(
            (c for c in ("state_id", "cluster_id", "state") if c in ss.columns), None
        )
        if id_col is None:
            self._state_list.addItem("Missing state_id column")
            return

        ctx_enrich: dict[int, str] = {}
        if ctx is not None and not ctx.empty and id_col in ctx.columns:
            for _, row in ctx.iterrows():
                sid = int(row.get(id_col, -1))
                enrich = str(row.get("enriched_context", ""))
                if enrich and enrich != "nan":
                    ctx_enrich[sid] = enrich

        self._id_col = id_col
        self._ctx_enrich = ctx_enrich
        self._compute_heuristic_labels(ss, id_col)
        self._populate_list()

    def _compute_heuristic_labels(self, ss: pd.DataFrame, id_col: str) -> None:
        self._heuristic_labels = {}
        ci = self._data.get("cluster_info") or {}
        centers = ci.get("cluster_centers", [])
        if not centers:
            return
        try:
            from compare import _generate_kinematic_labels
        except ImportError:
            return
        n_kp = 8
        try:
            import json as _json
            idx_path = RESULTS / "features" / "index.json"
            if idx_path.exists():
                with open(idx_path, encoding="utf-8") as f:
                    n_kp = _json.load(f).get("_meta", {}).get("n_keypoints", 8)
        except Exception:
            pass
        bout_stats: dict[int, dict] = {}
        for _, row in ss.iterrows():
            sid = int(row.get(id_col, -1))
            dur = row.get("mean_bout_dur_sec")
            if dur is not None and not pd.isna(dur):
                bout_stats[sid] = {"mean_dur": float(dur)}
            else:
                bout_stats[sid] = {"mean_dur": None}
        self._heuristic_labels = _generate_kinematic_labels(
            centers, bout_stats, n_keypoints=n_kp,
        )

    _SORT_COLS = [
        None,
        "mean_centroid_speed",
        "mean_elongation",
        "mean_bout_dur_sec",
        "mean_angular_vel",
    ]

    def _populate_list(self) -> None:
        ss = self._data.get("state_summary")
        if ss is None or ss.empty:
            return
        id_col = getattr(self, "_id_col", None)
        if id_col is None:
            return
        sort_idx = self._sort_combo.currentIndex()
        sort_col = self._SORT_COLS[sort_idx]
        ascending = self._sort_asc.isChecked()

        if sort_col and sort_col in ss.columns:
            df = ss.sort_values(sort_col, ascending=ascending, na_position="last")
        else:
            df = ss.sort_values(id_col, ascending=ascending)

        self._state_list.clear()
        self._state_ids = []
        for _, row in df.iterrows():
            sid = int(row.get(id_col, -1))
            base = f"State {sid}"
            trait = self._heuristic_labels.get(sid, "")
            if trait and trait != base:
                label = f"{base}  —  {trait}"
            else:
                label = base
            self._state_list.addItem(label)
            self._state_ids.append(sid)
        if self._panel_collapsed:
            self._sync_collapsed_list()

    def _on_sort_changed(self, _=None) -> None:
        self._sort_asc.setText("↑" if self._sort_asc.isChecked() else "↓")
        self._populate_list()

    def _on_search(self, text: str) -> None:
        for i in range(self._state_list.count()):
            item = self._state_list.item(i)
            item.setHidden(bool(text) and text.lower() not in item.text().lower())

    def _enrich_kinematic_columns(self, ss: pd.DataFrame) -> pd.DataFrame:
        needed = {
            "mean_centroid_speed", "mean_angular_vel", "mean_elongation",
            "mean_rearing_score", "mean_movement_entropy",
        }
        existing = {c.lower() for c in ss.columns}
        if needed.issubset(existing):
            return ss

        ci = self._data.get("cluster_info") or {}
        centers = ci.get("cluster_centers", [])
        if not centers:
            return ss

        from compare import _extract_kinematic_values
        id_col = getattr(self, "_id_col", None)
        if id_col is None:
            return ss

        n_kp = 8
        try:
            import json as _json
            idx_path = RESULTS / "features" / "index.json"
            if idx_path.exists():
                with open(idx_path, encoding="utf-8") as f:
                    n_kp = _json.load(f).get("_meta", {}).get("n_keypoints", 8)
        except Exception:
            pass

        kin_rows = []
        for _, row in ss.iterrows():
            sid = int(row.get(id_col, -1))
            if 0 <= sid < len(centers):
                vals = _extract_kinematic_values(centers[sid], n_keypoints=n_kp)
            else:
                vals = {}
            vals[id_col] = sid
            kin_rows.append(vals)

        kin_df = pd.DataFrame(kin_rows)
        for col in kin_df.columns:
            if col != id_col and col.lower() not in existing:
                ss[col] = kin_df[col].values
        return ss

    def _on_state_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._state_ids):
            return
        sid = self._state_ids[row]
        self._detail_placeholder.hide()
        self._load_clip_btn.setEnabled(True)
        self._load_clip_btn.setProperty("_sid", sid)

        ss = self._data.get("state_summary")
        if ss is None:
            return
        id_col = next(
            (c for c in ("state_id", "cluster_id", "state") if c in ss.columns), None
        )
        if id_col is None:
            return

        ss = self._enrich_kinematic_columns(ss)
        self._data["state_summary"] = ss

        rows = ss[ss[id_col] == sid]
        if rows.empty:
            return
        r = rows.iloc[0]

        # ── Stat cards ──
        self._update_stat_cards(sid, r, ss, id_col)

        # ── Kinematic profile chart ──
        if _MPL and self._kin_canvas:
            kinematic_metrics = [
                ("mean_centroid_speed",   "Speed"),
                ("mean_angular_vel",      "Angular Velocity"),
                ("mean_elongation",       "Elongation"),
                ("mean_rearing_score",    "Rearing Score"),
                ("mean_bout_dur_sec",     "Bout Duration"),
                ("mean_movement_entropy", "Movement Variability"),
            ]
            cols_lower = {c.lower(): c for c in ss.columns}

            metrics = {}
            for col_name, display in kinematic_metrics:
                col = cols_lower.get(col_name.lower())
                if col is None:
                    continue
                v = r.get(col, None)
                if v is None or pd.isna(v):
                    continue
                min_v = ss[col].min()
                max_v = ss[col].max()
                if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
                    continue
                metrics[display] = (float(v) - float(min_v)) / (float(max_v) - float(min_v))

            self._kin_metrics = metrics
            self._draw_kinematic_chart()

        # ── Top features table ──
        self._update_feat_table(sid)

        # ── Group enrichment table ──
        self._update_enrich_table(sid)

        # ── Heuristic label hint ──
        hint = self._heuristic_labels.get(sid, "")
        if hint and hint != f"State {sid}":
            self._heuristic_lbl.setText(f"Kinematic signature: {hint}")
            self._heuristic_lbl.show()
        else:
            self._heuristic_lbl.hide()

        self._profile_tabs.show()
        self._cards_widget.show()

        # ── Restore saved label ──
        saved = self._load_saved_state_labels().get(sid, {})
        self._label_input.setText(saved.get("label", ""))
        cat = saved.get("category", "")
        self._cat_group.setExclusive(False)
        for btn in self._cat_buttons.values():
            btn.setChecked(False)
        self._cat_group.setExclusive(True)
        if cat and cat in self._cat_buttons:
            self._cat_buttons[cat].setChecked(True)
            if cat in _TECHNICAL_CATEGORIES:
                self._more_btn.setChecked(True)

    def _update_stat_cards(self, sid: int, r, ss: pd.DataFrame, id_col: str) -> None:
        """Populate the stat cards for the selected state."""
        # Occupancy — from global_fraction or n_bouts-derived
        occ = None
        for col in ("global_fraction", "fraction", "occupancy"):
            v = r.get(col)
            if v is not None and not (isinstance(v, float) and v != v):
                occ = float(v)
                break

        # Also try duration_summary if available
        dur_df = self._data.get("duration_summary")
        if dur_df is not None and not dur_df.empty and "state_id" in dur_df.columns:
            dur_row = dur_df[dur_df["state_id"] == sid]
            if not dur_row.empty:
                dr = dur_row.iloc[0]
                n_bouts = int(dr.get("n_bouts", 0)) if not pd.isna(dr.get("n_bouts", float("nan"))) else None
                mean_sec = dr.get("mean_sec")
                median_sec = dr.get("median_sec")
            else:
                n_bouts = None
                mean_sec = None
                median_sec = None
        else:
            n_bouts_v = r.get("n_bouts")
            n_bouts = int(n_bouts_v) if n_bouts_v is not None and not pd.isna(n_bouts_v) else None
            mean_sec = r.get("mean_bout_dur_sec")
            median_sec = r.get("median_bout_dur_sec")

        # Occupancy fallback from state_summary
        if occ is None:
            for col in [c for c in ss.columns if "frac" in c.lower() or "occ" in c.lower()]:
                v = r.get(col)
                if v is not None and not (isinstance(v, float) and v != v):
                    occ = float(v)
                    break

        self._card_values["occupancy"].setText(
            f"{occ * 100:.1f}%" if occ is not None else "—"
        )
        self._card_values["n_bouts"].setText(
            str(n_bouts) if n_bouts is not None else "—"
        )
        self._card_values["mean_dur"].setText(
            f"{float(mean_sec):.2f}s" if mean_sec is not None and not (isinstance(mean_sec, float) and mean_sec != mean_sec) else "—"
        )
        self._card_values["median_dur"].setText(
            f"{float(median_sec):.2f}s" if median_sec is not None and not (isinstance(median_sec, float) and median_sec != median_sec) else "—"
        )

    def _update_feat_table(self, sid: int) -> None:
        """Populate the top-features table from feature_zscores data."""
        zdf = self._data.get("feature_zscores")
        if zdf is None or zdf.empty or "state_id" not in zdf.columns:
            self._feat_table.hide()
            self._feat_no_data.show()
            return

        row_mask = zdf["state_id"] == sid
        if not row_mask.any():
            self._feat_table.hide()
            self._feat_no_data.show()
            return

        z_row = zdf[row_mask].iloc[0]
        feat_cols = [c for c in zdf.columns if c != "state_id"]
        if not feat_cols:
            self._feat_table.hide()
            self._feat_no_data.show()
            return

        z_vals = [(col, float(z_row[col])) for col in feat_cols
                  if not pd.isna(z_row[col])]
        z_vals.sort(key=lambda x: abs(x[1]), reverse=True)
        top_n = z_vals[:10]

        self._feat_table.setRowCount(len(top_n))
        for row_i, (fname, zval) in enumerate(top_n):
            self._feat_table.setItem(row_i, 0, QTableWidgetItem(fname))
            zscore_item = QTableWidgetItem(f"{zval:+.3f}")
            zscore_item.setTextAlignment(Qt.AlignCenter)
            self._feat_table.setItem(row_i, 1, zscore_item)
            dir_text = "▲ above avg" if zval > 0 else "▼ below avg"
            dir_item = QTableWidgetItem(dir_text)
            dir_item.setTextAlignment(Qt.AlignCenter)
            dir_item.setForeground(
                self._feat_table.palette().highlight() if zval > 0
                else self._feat_table.palette().dark()
            )
            self._feat_table.setItem(row_i, 2, dir_item)

        self._feat_table.resizeRowsToContents()
        self._feat_no_data.hide()
        self._feat_table.show()

    def _update_enrich_table(self, sid: int) -> None:
        """Populate the group enrichment table."""
        ge = self._data.get("group_enrichment")
        if ge is None or ge.empty or "state_id" not in ge.columns:
            # Fall back to context_report
            ctx = self._data.get("context_report")
            if ctx is not None and not ctx.empty:
                self._update_enrich_from_context_report(sid, ctx)
            else:
                self._enrich_table.hide()
                self._enrich_no_data.show()
            return

        mask = ge["state_id"] == sid
        sub = ge[mask].copy()
        if sub.empty:
            self._enrich_table.hide()
            self._enrich_no_data.show()
            return

        self._enrich_table.setRowCount(len(sub))
        for row_i, (_, row) in enumerate(sub.iterrows()):
            self._enrich_table.setItem(row_i, 0, QTableWidgetItem(str(row.get("group_variable", ""))))
            self._enrich_table.setItem(row_i, 1, QTableWidgetItem(str(row.get("group_value", ""))))
            frac = row.get("fraction")
            frac_item = QTableWidgetItem(
                f"{float(frac)*100:.1f}%" if frac is not None and not pd.isna(frac) else "—"
            )
            frac_item.setTextAlignment(Qt.AlignCenter)
            self._enrich_table.setItem(row_i, 2, frac_item)
            enr = row.get("enrichment_ratio")
            enr_item = QTableWidgetItem(
                f"{float(enr):.2f}×" if enr is not None and not pd.isna(enr) else "—"
            )
            enr_item.setTextAlignment(Qt.AlignCenter)
            self._enrich_table.setItem(row_i, 3, enr_item)

        self._enrich_table.resizeRowsToContents()
        self._enrich_no_data.hide()
        self._enrich_table.show()

    def _update_enrich_from_context_report(self, sid: int, ctx: pd.DataFrame) -> None:
        """Show context enrichment from context_report.csv as fallback."""
        id_col = next(
            (c for c in ("state_id", "cluster_id", "state") if c in ctx.columns), None
        )
        if id_col is None:
            self._enrich_table.hide()
            self._enrich_no_data.show()
            return

        ctx_cols = [c for c in ctx.columns
                    if c.startswith("context_") and c.endswith("_frac")]
        if not ctx_cols:
            self._enrich_table.hide()
            self._enrich_no_data.show()
            return

        mask = ctx[id_col] == sid
        if not mask.any():
            self._enrich_table.hide()
            self._enrich_no_data.show()
            return

        row = ctx[mask].iloc[0]
        rows_data = []
        for col in ctx_cols:
            v = row.get(col)
            if v is None or pd.isna(v):
                continue
            ctx_name = col.replace("context_", "").replace("_frac", "")
            rows_data.append(("context", ctx_name, float(v)))

        if not rows_data:
            self._enrich_table.hide()
            self._enrich_no_data.show()
            return

        self._enrich_table.setRowCount(len(rows_data))
        for row_i, (var, val, frac) in enumerate(rows_data):
            self._enrich_table.setItem(row_i, 0, QTableWidgetItem(var))
            self._enrich_table.setItem(row_i, 1, QTableWidgetItem(val))
            frac_item = QTableWidgetItem(f"{frac*100:.1f}%")
            frac_item.setTextAlignment(Qt.AlignCenter)
            self._enrich_table.setItem(row_i, 2, frac_item)
            self._enrich_table.setItem(row_i, 3, QTableWidgetItem("—"))

        self._enrich_table.resizeRowsToContents()
        self._enrich_no_data.hide()
        self._enrich_table.show()

    # ───────────────────────────── Responsive scaling ──

    def _scale_factor(self) -> float:
        return max(0.6, min(1.6, self.width() / self._ref_width))

    def _draw_kinematic_chart(self) -> None:
        if not (_MPL and self._kin_canvas):
            return
        canvas = self._kin_canvas
        canvas.fig.clf()
        ax = canvas.fig.add_subplot(111)
        s = self._scale_factor()
        metrics = self._kin_metrics

        if metrics:
            norm_v = list(metrics.values())
            y = np.arange(len(metrics))
            ax.barh(y, norm_v, color="#4E79A7", alpha=0.85)
            ax.set_yticks(y)
            ax.set_yticklabels(list(metrics.keys()), fontsize=max(7, int(9 * s)))
            ax.set_xlim(0, 1)
            ax.set_xlabel(
                "Normalized value (min–max across states)",
                fontsize=max(7, int(9 * s)),
            )
            ax.tick_params(axis="x", labelsize=max(6, int(8 * s)))
        else:
            ax.text(
                0.5, 0.5, "No kinematic data",
                ha="center", va="center", transform=ax.transAxes, color="#999",
                fontsize=max(8, int(10 * s)),
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        canvas.fig.tight_layout()
        canvas.draw()

    def _apply_scaled_styles(self) -> None:
        s = self._scale_factor()
        chip_fs = max(8, int(11 * s))
        chip_pad_h = max(3, int(4 * s))
        chip_pad_w = max(6, int(10 * s))
        chip_r = max(8, int(12 * s))
        chip_style = (
            f"QPushButton{{background:#f0f0f0;border:1px solid #ccc;"
            f"border-radius:{chip_r}px;padding:{chip_pad_h}px {chip_pad_w}px;"
            f"font-size:{chip_fs}px;}}"
            f"QPushButton:checked{{background:#4E79A7;color:white;border-color:#4E79A7;}}"
            f"QPushButton:hover:!checked{{background:#e0e8f0;}}"
        )
        tech_fs = max(7, int(10 * s))
        tech_pad_h = max(2, int(3 * s))
        tech_pad_w = max(5, int(8 * s))
        tech_style = (
            f"QPushButton{{background:#f5f5f5;border:1px solid #ddd;"
            f"border-radius:{chip_r}px;padding:{tech_pad_h}px {tech_pad_w}px;"
            f"font-size:{tech_fs}px;color:#666;}}"
            f"QPushButton:checked{{background:#76B7B2;color:white;border-color:#76B7B2;}}"
            f"QPushButton:hover:!checked{{background:#e8f0f0;}}"
        )
        for name, btn in self._cat_buttons.items():
            if name in _BEHAVIORAL_CATEGORIES:
                btn.setStyleSheet(chip_style)
            elif name in _TECHNICAL_CATEGORIES:
                btn.setStyleSheet(tech_style)
            else:
                btn.setStyleSheet(chip_style)
        lbl_fs = max(9, int(12 * s))
        self._lbl_title.setStyleSheet(
            f"font-weight:bold; font-size:{lbl_fs}px; color:#333;"
        )
        cat_fs = max(9, int(11 * s))
        self._cat_label.setStyleSheet(f"font-size:{cat_fs}px;")
        more_fs = max(8, int(11 * s))
        self._more_btn.setStyleSheet(
            f"font-size:{more_fs}px; color:#666; border:none;"
        )
        btn_fs = max(9, int(12 * s))
        btn_style = f"font-size:{btn_fs}px;"
        self._save_btn.setStyleSheet(btn_style)
        self._save_next_btn.setStyleSheet(btn_style)
        self._back_btn.setStyleSheet(btn_style)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_scaled_styles()
        if self._kin_metrics:
            self._draw_kinematic_chart()

    # ──────────────────────────── State panel collapse ──

    def _toggle_state_panel(self) -> None:
        self._set_collapsed(not self._panel_collapsed)

    def _set_collapsed(self, collapsed: bool) -> None:
        if collapsed == self._panel_collapsed:
            return
        self._panel_collapsed = collapsed

        self._search.setVisible(not collapsed)
        self._sort_row_widget.setVisible(not collapsed)
        self._state_list.setVisible(not collapsed)
        self._panel_title.setVisible(not collapsed)
        self._collapsed_list.setVisible(collapsed)

        if collapsed:
            self._collapse_btn.setText("»")
            self._collapse_btn.setToolTip("Expand state list")
            sizes = self._splitter.sizes()
            total = sum(sizes)
            self._splitter.setSizes([56, total - 56])
            self._sync_collapsed_list()
        else:
            self._collapse_btn.setText("«")
            self._collapse_btn.setToolTip("Collapse state list")
            sizes = self._splitter.sizes()
            total = sum(sizes)
            self._splitter.setSizes([self._left_expanded_width,
                                     total - self._left_expanded_width])
            row = self._collapsed_list.currentRow()
            if 0 <= row < self._state_list.count():
                self._state_list.setCurrentRow(row)

    def _on_splitter_moved(self, _pos: int, _index: int) -> None:
        left_width = self._splitter.sizes()[0]
        if left_width < self._collapse_threshold and not self._panel_collapsed:
            self._set_collapsed(True)
        elif left_width >= self._collapse_threshold and self._panel_collapsed:
            self._set_collapsed(False)

    def _sync_collapsed_list(self) -> None:
        self._collapsed_list.clear()
        for sid in self._state_ids:
            self._collapsed_list.addItem(str(sid))
        row = self._state_list.currentRow()
        if 0 <= row < self._collapsed_list.count():
            self._collapsed_list.blockSignals(True)
            self._collapsed_list.setCurrentRow(row)
            self._collapsed_list.blockSignals(False)

    def _on_collapsed_selected(self, row: int) -> None:
        if 0 <= row < self._state_list.count():
            self._state_list.setCurrentRow(row)

    def _load_saved_state_labels(self) -> dict[int, dict]:
        p = RESULTS / "validation" / "state_labels.csv"
        if not p.exists():
            return {}
        try:
            df = pd.read_csv(p)
            return {
                int(row.get("state_id", -1)): {
                    "label": str(row.get("label", "")),
                    "category": str(row.get("category", "")),
                }
                for _, row in df.iterrows()
            }
        except Exception:
            return {}

    def _save_state_label(self) -> None:
        row = self._state_list.currentRow()
        if row < 0 or row >= len(self._state_ids):
            return
        sid = self._state_ids[row]
        existing = self._load_saved_state_labels()
        checked = self._cat_group.checkedButton()
        category = checked.text() if checked else ""
        label = self._label_input.text().strip()
        if not label and category:
            label = category
        existing[sid] = {"label": label, "category": category}
        p = RESULTS / "validation" / "state_labels.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"state_id": k, "label": v["label"], "category": v["category"]}
            for k, v in sorted(existing.items())
        ]
        pd.DataFrame(rows).to_csv(p, index=False)

    def _load_clip(self) -> None:
        sid = self._load_clip_btn.property("_sid")
        if sid is None:
            return
        clip_dir = Path(_vc.get_clips_dir()) / f"state_{sid}"
        if not clip_dir.exists():
            self._clip_status.setText(
                f"No clips found for state {sid} — run: python generate_clips.py"
            )
            return
        clips = list(clip_dir.glob("*.mp4"))
        if not clips:
            self._clip_status.setText(f"No .mp4 files in clips/state_{sid}/")
            return
        chosen = random.choice(clips)
        self._clip_status.setText(chosen.name)
        if self._player:
            try:
                self._player.load(str(chosen))
                if self._autoplay_cb.isChecked():
                    self._player.play()
            except Exception:
                self._clip_status.setText(f"Error loading {chosen.name}")

    def _toggle_technical_cats(self, expanded: bool) -> None:
        self._tech_row_widget.setVisible(expanded)
        self._more_btn.setText("More ▾" if expanded else "More ▸")

    def _save_and_next(self) -> None:
        self._save_state_label()
        current_row = self._state_list.currentRow()
        next_row = current_row + 1
        while next_row < self._state_list.count():
            if not self._state_list.item(next_row).isHidden():
                break
            next_row += 1
        if next_row >= self._state_list.count():
            self._clip_status.setText("All states labeled!")
            return
        self._state_list.setCurrentRow(next_row)
        self._load_clip()

    def _go_back(self) -> None:
        current_row = self._state_list.currentRow()
        prev_row = current_row - 1
        while prev_row >= 0:
            if not self._state_list.item(prev_row).isHidden():
                break
            prev_row -= 1
        if prev_row < 0:
            self._clip_status.setText("Already at first state.")
            return
        self._state_list.setCurrentRow(prev_row)
        self._load_clip()

    def _add_custom_category(self) -> None:
        name = self._cat_input.text().strip()
        if not name or name in self._cat_buttons:
            self._cat_input.clear()
            return
        btn = QPushButton(name)
        btn.setCheckable(True)
        btn.setStyleSheet(_CHIP_STYLE)
        self._cat_group.addButton(btn)
        stretch = self._custom_cat_chips_layout.takeAt(
            self._custom_cat_chips_layout.count() - 1,
        )
        self._custom_cat_chips_layout.insertWidget(
            self._custom_cat_chips_layout.count(), btn,
        )
        if stretch:
            self._custom_cat_chips_layout.addItem(stretch)
        self._cat_buttons[name] = btn
        self._cat_input.clear()
        custom = self.cfg.get("state_categories", [])
        if name not in custom:
            custom.append(name)
            self.cfg["state_categories"] = custom
            _save_cfg(self.cfg)
        self._apply_scaled_styles()

    def _restore_custom_categories(self) -> None:
        for name in self.cfg.get("state_categories", []):
            if name not in self._cat_buttons:
                btn = QPushButton(name)
                btn.setCheckable(True)
                btn.setStyleSheet(_CHIP_STYLE)
                self._cat_group.addButton(btn)
                stretch = self._custom_cat_chips_layout.takeAt(
                    self._custom_cat_chips_layout.count() - 1,
                )
                self._custom_cat_chips_layout.insertWidget(
                    self._custom_cat_chips_layout.count(), btn,
                )
                if stretch:
                    self._custom_cat_chips_layout.addItem(stretch)
                self._cat_buttons[name] = btn

    # ─────────────────────────────────────── Command runner ──

    def _run_command(self, args: list[str], terminal: TerminalBox) -> None:
        if self._worker and self._worker.isRunning():
            return
        terminal.set_command("python " + " ".join(str(a) for a in args))
        self._worker = SubprocessWorker(args)
        self._worker.log.connect(terminal.append_output)
        self._worker.done.connect(self._on_run_done)
        self.worker_running.emit(True)
        self._worker.start()

    def stop_worker(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()

    def _on_run_done(self, ok: bool) -> None:
        self.worker_running.emit(False)
        self._load()
