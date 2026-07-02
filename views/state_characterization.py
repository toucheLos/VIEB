from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QMessageBox, QPushButton, QScrollArea, QSizePolicy,
    QSplitter, QTabWidget, QTableWidget, QTableWidgetItem, QToolButton,
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

_CHIP_STYLE = (
    "QPushButton{background:#f0f0f0;border:1px solid #ccc;border-radius:12px;"
    "padding:4px 10px;font-size:11px;}"
    "QPushButton:checked{background:#4E79A7;color:white;border-color:#4E79A7;}"
    "QPushButton:hover:!checked{background:#e0e8f0;}"
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
        self._running_command = ""
        self._state_ids: list[int] = []
        self._heuristic_labels: dict[int, str] = {}
        self._kin_metrics: dict[str, float] = {}
        self._pending_command: list | None = None
        self._saved_labels: dict[int, dict] = {}
        self._current_sid: int | None = None
        self._current_clip_path: Path | None = None
        self._last_run_kind: str = ""
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
        run_both_btn = QPushButton("Run + Generate")
        run_both_btn.setFixedHeight(30)
        run_both_btn.setToolTip("Run characterization then generate exemplar clips sequentially")
        run_both_btn.clicked.connect(self._run_characterize_and_generate)
        top.addWidget(run_both_btn)
        run_btn = QPushButton(run_label)
        run_btn.setFixedHeight(30)
        run_btn.clicked.connect(run_slot)
        top.addWidget(run_btn)
        lay.addLayout(top)
        lay.addWidget(terminal)
        return outer

    def _build_notice_banner(self) -> QFrame:
        """Dismissible, non-blocking notice shown after generation completes."""
        banner = QFrame()
        banner.setStyleSheet(
            "QFrame{background:#EDF7ED;border:1px solid #C3E6C3;border-radius:6px;}"
        )
        row = QHBoxLayout(banner)
        row.setContentsMargins(10, 6, 6, 6)
        row.setSpacing(8)
        self._notice_lbl = QLabel("")
        self._notice_lbl.setWordWrap(True)
        self._notice_lbl.setStyleSheet("color:#1E5620; font-size:11px; border:none; background:transparent;")
        row.addWidget(self._notice_lbl, stretch=1)
        close_btn = QPushButton("×")
        close_btn.setFixedSize(20, 20)
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.setToolTip("Dismiss")
        close_btn.setStyleSheet(
            "QPushButton{background:transparent;color:#5A805A;border:none;font-size:15px;font-weight:bold;}"
            "QPushButton:hover{color:#1E5620;}"
        )
        close_btn.clicked.connect(self._dismiss_notice)
        row.addWidget(close_btn, alignment=Qt.AlignTop)
        self._notice_timer = QTimer(self)
        self._notice_timer.setSingleShot(True)
        self._notice_timer.timeout.connect(self._dismiss_notice)
        return banner

    def _show_notice(self, text: str) -> None:
        self._notice_lbl.setText(text)
        self._notice_banner.show()
        self._notice_timer.start(12000)

    def _dismiss_notice(self) -> None:
        self._notice_timer.stop()
        self._notice_banner.hide()

    # Metric columns surfaced in the extremes panel: (column, display name).
    _EXTREME_METRICS = [
        ("mean_centroid_speed", "Speed"),
        ("mean_angular_vel", "Angular Velocity"),
        ("mean_elongation", "Elongation"),
        ("mean_bout_dur_sec", "Bout Duration"),
        ("mean_rearing_score", "Rearing"),
        ("mean_movement_entropy", "Movement Variability"),
    ]

    def _build_metric_extremes_panel(self) -> QFrame:
        """Compact panel listing the highest/lowest state for each kinematic metric."""
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame{background:#F8F9FA;border:1px solid #E0E0E0;border-radius:6px;}"
        )
        vl = QVBoxLayout(frame)
        vl.setContentsMargins(10, 6, 10, 6)
        vl.setSpacing(2)

        title = QLabel("Metric Extremes")
        title.setStyleSheet(
            "font-weight:bold; font-size:11px; color:#555; letter-spacing:0.5px;"
            "background:transparent;border:none;"
        )
        vl.addWidget(title)

        self._metrics_body = QLabel("")
        self._metrics_body.setWordWrap(True)
        self._metrics_body.setStyleSheet(
            "font-size:11px; color:#333; background:transparent; border:none;"
        )
        self._metrics_body.setTextFormat(Qt.RichText)
        vl.addWidget(self._metrics_body)
        return frame

    def _update_metric_extremes(self) -> None:
        """Fill the metric-extremes panel from state_summary (highest/lowest per metric)."""
        ss = self._data.get("state_summary")
        id_col = getattr(self, "_id_col", None)
        if ss is None or getattr(ss, "empty", True) or not id_col:
            self._metrics_panel.hide()
            return
        ss = self._enrich_kinematic_columns(ss)
        self._data["state_summary"] = ss
        cols_lower = {c.lower(): c for c in ss.columns}

        lines: list[str] = []
        for col_name, display in self._EXTREME_METRICS:
            col = cols_lower.get(col_name.lower())
            if col is None:
                continue
            sub = ss[[id_col, col]].dropna()
            if sub.empty:
                continue
            hi = sub.loc[sub[col].idxmax()]
            lo = sub.loc[sub[col].idxmin()]
            lines.append(
                f"<b>{display}</b> — highest: State {int(hi[id_col])} "
                f"({float(hi[col]):.3g}), lowest: State {int(lo[id_col])} "
                f"({float(lo[col]):.3g})"
            )
        if not lines:
            self._metrics_panel.hide()
            return
        self._metrics_body.setText("<br>".join(lines))
        self._metrics_panel.show()

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

        self._notice_banner = self._build_notice_banner()
        self._notice_banner.hide()
        lay.addWidget(self._notice_banner)

        self._metrics_panel = self._build_metric_extremes_panel()
        self._metrics_panel.hide()
        lay.addWidget(self._metrics_panel)

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

        # State-to-state navigation is intentionally the state list only — no
        # prev/next state buttons (see plan Part 1).
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
        right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
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

        # ── Exemplar controls (generation only — clip playback lives below the
        #    video player, see Part 3a) ──
        clip_row = QHBoxLayout()
        self._gen_exemplars_btn = QPushButton("Generate State Exemplars")
        self._gen_exemplars_btn.clicked.connect(
            lambda: self._run_command(["generate_clips.py"], self._terminal)
        )
        clip_row.addWidget(self._gen_exemplars_btn)
        clip_row.addStretch(1)
        rl.addLayout(clip_row)

        self._exemplar_placeholder = _placeholder(
            "No clips for this state yet.\n"
            "Click 'Generate State Exemplars' to export clips for each state."
        )
        self._exemplar_table = QTableWidget(0, 2)
        self._exemplar_table.setHorizontalHeaderLabels(["Clip", "Type"])
        header = self._exemplar_table.horizontalHeader()
        header.setSectionResizeMode(0, header.Stretch)
        header.setSectionResizeMode(1, header.ResizeToContents)
        self._exemplar_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._exemplar_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._exemplar_table.setSelectionMode(QTableWidget.SingleSelection)
        self._exemplar_table.verticalHeader().setVisible(False)
        self._exemplar_table.setMaximumHeight(150)
        self._exemplar_table.itemSelectionChanged.connect(self._on_exemplar_selected)
        self._exemplar_table.doubleClicked.connect(lambda _idx: self._load_clip())
        self._exemplar_table.hide()
        rl.addWidget(self._exemplar_placeholder)
        rl.addWidget(self._exemplar_table)

        # ── Label section ──
        self._lbl_title = QLabel("Label this state:")
        self._lbl_title.setStyleSheet("font-weight:bold; font-size:12px; color:#333;")
        rl.addWidget(self._lbl_title)

        input_row = QHBoxLayout()
        self._label_input = QLineEdit()
        self._label_input.setPlaceholderText("Free-text label, or select a category below")
        input_row.addWidget(self._label_input, stretch=1)
        self._save_btn = QPushButton("Save")
        self._save_btn.setToolTip("Save this state's label; the list row updates in place")
        self._save_btn.clicked.connect(self._save_state_label)
        input_row.addWidget(self._save_btn)
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
        cat_row.addStretch()
        rl.addLayout(cat_row)

        self._custom_cat_row = QHBoxLayout()
        self._custom_cat_row.setSpacing(4)
        self._custom_cat_chips_layout = self._custom_cat_row
        self._cat_input = QLineEdit()
        self._cat_input.setPlaceholderText("New category name…")
        self._cat_input.setMaximumWidth(200)
        self._cat_input.returnPressed.connect(self._add_custom_category)
        self._custom_cat_row.addWidget(self._cat_input)
        add_cat_btn = QPushButton("＋ Add Custom Category")
        add_cat_btn.setToolTip("Add a custom category")
        add_cat_btn.clicked.connect(self._add_custom_category)
        self._custom_cat_row.addWidget(add_cat_btn)
        self._custom_cat_row.addStretch()
        rl.addLayout(self._custom_cat_row)
        self._restore_custom_categories()

        try:
            from _widgets import VideoPlayer
            self._player = VideoPlayer()
            self._player.video_finished.connect(self._on_video_finished)
            rl.addWidget(self._player, stretch=1)
        except Exception:
            self._player = None
            rl.addWidget(_placeholder(
                "Video player unavailable.\nInstall opencv-python to enable."
            ))

        # ── Clip navigation + export, directly below the video player (Autoplay
        #    sits just under the player's Loop control) ──
        clip_nav_row = QHBoxLayout()
        clip_nav_row.setSpacing(6)
        self._autoplay_cb = QCheckBox("Autoplay")
        self._autoplay_cb.setChecked(True)
        self._autoplay_cb.setToolTip("Automatically play the next clip when one ends")
        clip_nav_row.addWidget(self._autoplay_cb)
        self._prev_clip_btn = QPushButton("◀ Previous Clip")
        self._prev_clip_btn.setToolTip("Play the previous exemplar clip for this state")
        self._prev_clip_btn.setEnabled(False)
        self._prev_clip_btn.clicked.connect(self._prev_clip)
        clip_nav_row.addWidget(self._prev_clip_btn)
        self._next_clip_btn = QPushButton("Next Clip ▶")
        self._next_clip_btn.setToolTip("Play the next exemplar clip for this state")
        self._next_clip_btn.setEnabled(False)
        self._next_clip_btn.clicked.connect(self._next_clip)
        clip_nav_row.addWidget(self._next_clip_btn)
        clip_nav_row.addStretch(1)
        self._clip_status = QLabel("")
        self._clip_status.setStyleSheet("color:#888; font-size:11px;")
        clip_nav_row.addWidget(self._clip_status, stretch=1)
        self._export_clip_btn = QPushButton("Export Clip")
        self._export_clip_btn.setToolTip("Copy the current clip to results/exports/")
        self._export_clip_btn.setEnabled(False)
        self._export_clip_btn.clicked.connect(self._export_current_clip)
        clip_nav_row.addWidget(self._export_clip_btn)
        rl.addLayout(clip_nav_row)

        scroll.setWidget(right)
        splitter.addWidget(scroll)
        splitter.setSizes([self._left_expanded_width, 660])
        splitter.splitterMoved.connect(self._on_splitter_moved)
        lay.addWidget(splitter, stretch=1)

    # ────────────────────────────────────────────────── Data loading ──

    def update_data(self, data: dict) -> None:
        self._data = self._merge_incoming_data(data)
        self._load()

    def refresh(self, data: dict) -> None:
        self._data = self._merge_incoming_data(data)
        self._load()

    @staticmethod
    def _n_rows(df) -> int:
        return 0 if df is None or getattr(df, "empty", True) else len(df)

    def _merge_incoming_data(self, data: dict) -> dict:
        """Never let a partial/lightweight payload shrink the state list.

        The app's DataLoader only fills ``state_summary`` in its full (non-
        lightweight) pass; a lightweight refresh would otherwise wipe the list.
        Retain the last good ``state_summary`` (and companion frames) when the
        incoming payload lacks it or has fewer states than we already show.
        """
        merged = dict(data or {})
        prev = self._data or {}
        if self._n_rows(merged.get("state_summary")) < self._n_rows(prev.get("state_summary")):
            merged["state_summary"] = prev.get("state_summary")
            for k in ("context_report", "feature_zscores", "duration_summary",
                      "group_enrichment", "cluster_info"):
                if merged.get(k) is None and prev.get(k) is not None:
                    merged[k] = prev.get(k)
        return merged

    def _read_characterization_from_disk(self) -> dict:
        """Return a copy of self._data with characterization outputs re-read from disk."""
        def _csv(rel: str) -> pd.DataFrame | None:
            p = RESULTS / rel
            try:
                return pd.read_csv(p) if p.exists() else None
            except Exception:
                return None

        def _json(rel: str) -> dict:
            p = RESULTS / rel
            try:
                import json as _json_mod
                return _json_mod.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
            except Exception:
                return {}

        data = dict(self._data)
        data["state_summary"] = _csv("characterization/state_summary.csv")
        data["context_report"] = _csv("characterization/context_report.csv")
        data["feature_zscores"] = _csv("characterization/state_feature_zscores.csv")
        data["duration_summary"] = _csv("characterization/state_duration_summary.csv")
        data["group_enrichment"] = _csv("characterization/state_group_enrichment.csv")
        data["cluster_info"] = _json("shared/cluster_info.json")
        return data

    def _load_from_disk(self) -> None:
        """Read characterization outputs from disk and fully refresh the view."""
        self._data = self._read_characterization_from_disk()
        self._load()

    def on_cluster_changed(self) -> None:
        """Re-read from disk after the active cluster run changes; flag stale clips.

        Clips on disk belong to whichever run generated them, so after switching
        runs they may be stale for the new clustering. We never auto-regenerate —
        just refresh what's shown and prompt the user if clips look outdated.
        """
        self._load_from_disk()
        if self._clips_are_stale():
            self._show_notice(
                "Clips may be stale for this cluster run — click "
                "'Generate State Exemplars' to rebuild clips for the current clustering."
            )

    def _clips_are_stale(self) -> bool:
        """True when no clips exist, or the newest clip predates cluster_info.json."""
        try:
            clips_root = Path(_vc.get_clips_dir())
        except Exception:
            return False
        state_dirs = list(clips_root.glob("state_*")) if clips_root.is_dir() else []
        if not state_dirs:
            return True
        ci = RESULTS / "shared" / "cluster_info.json"
        if not ci.exists():
            return False
        newest = 0.0
        for d in state_dirs:
            for f in d.glob("*.mp4"):
                try:
                    newest = max(newest, f.stat().st_mtime)
                except OSError:
                    continue
        return newest > 0 and newest < ci.stat().st_mtime

    def _soft_refresh_after_generation(self, ok: bool) -> None:
        """Refresh characterization/exemplar data after a generate/characterize run
        WITHOUT rebuilding the state list or dropping saved labels (Part 5)."""
        # Re-read fresh outputs, merged so the full state list is never shrunk.
        self._data = self._merge_incoming_data(self._read_characterization_from_disk())

        if not self._state_ids:
            # First-ever population: safe to build the list from scratch.
            self._load()
        else:
            ss = self._data.get("state_summary")
            id_col = getattr(self, "_id_col", None)
            if ss is not None and not ss.empty and id_col:
                self._compute_heuristic_labels(ss, id_col)
            self._saved_labels = self._load_saved_state_labels()
            # Update each row's text in place (traits may change; saved labels kept).
            for i, sid in enumerate(self._state_ids):
                text, tooltip = self._state_row_display(sid)
                item = self._state_list.item(i)
                if item is not None:
                    item.setText(text)
                    item.setToolTip(tooltip)
            if self._panel_collapsed:
                self._sync_collapsed_list()
            self._update_metric_extremes()
            # Refresh the currently-selected state's kinematics/exemplars in place.
            cur = self._state_list.currentRow()
            if 0 <= cur < len(self._state_ids):
                self._on_state_selected(cur)

        if ok:
            self._show_generation_notice()

    def _show_generation_notice(self) -> None:
        """Non-blocking summary of what the last run produced (Part 5)."""
        kind = self._last_run_kind
        if "generate_clips" in kind:
            # States that actually have clips on disk.
            sids = sorted(sid for sid in self._state_ids if self._clips_for_state(sid))
            head = "Generated clips for"
        else:
            sids = sorted(self._state_ids)
            head = "Ran characterization on"
        n = len(sids)
        if not n:
            self._show_notice("Generation complete.")
            return
        examples = []
        for sid in sids[:4]:
            trait = self._heuristic_labels.get(sid, "")
            examples.append(f"State {sid} ({trait})" if trait else f"State {sid}")
        more = f", +{n - len(examples)} more" if n > len(examples) else ""
        self._show_notice(f"{head} {n} state(s) — " + ", ".join(examples) + more)

    def _load(self) -> None:
        ss = self._data.get("state_summary")
        ctx = self._data.get("context_report")

        self._state_list.clear()
        self._state_ids = []
        self._current_sid = None
        self._clear_current_clip("")
        self._detail_placeholder.show()
        self._cards_widget.hide()
        self._profile_tabs.hide()
        self._heuristic_lbl.hide()
        self._exemplar_placeholder.show()
        self._exemplar_table.hide()
        self._exemplar_table.setRowCount(0)

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
        self._saved_labels = self._load_saved_state_labels()
        self._compute_heuristic_labels(ss, id_col)
        self._update_metric_extremes()
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
            text, tooltip = self._state_row_display(sid)
            self._state_list.addItem(text)
            if tooltip:
                self._state_list.item(self._state_list.count() - 1).setToolTip(tooltip)
            self._state_ids.append(sid)
        if self._panel_collapsed:
            self._sync_collapsed_list()

    def _state_row_display(self, sid: int) -> tuple[str, str]:
        """Return (row_text, tooltip) for a state.

        A user-saved label is shown as the row text once saved; the auto-
        generated kinematic description is preserved as the tooltip.
        """
        base = f"State {sid}"
        trait = self._heuristic_labels.get(sid, "")
        saved = str((self._saved_labels.get(sid) or {}).get("label", "")).strip()
        display = saved or (trait if trait and trait != base else "")
        text = f"{base}  —  {display}" if display else base
        tooltip = f"Kinematic signature: {trait}" if trait and trait != base else ""
        return text, tooltip

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
        self._current_sid = sid
        self._detail_placeholder.hide()

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

        self._populate_exemplars(sid)

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
        for btn in self._cat_buttons.values():
            btn.setStyleSheet(chip_style)
        lbl_fs = max(9, int(12 * s))
        self._lbl_title.setStyleSheet(
            f"font-weight:bold; font-size:{lbl_fs}px; color:#333;"
        )
        cat_fs = max(9, int(11 * s))
        self._cat_label.setStyleSheet(f"font-size:{cat_fs}px;")
        btn_fs = max(9, int(12 * s))
        btn_style = f"font-size:{btn_fs}px;"
        self._save_btn.setStyleSheet(btn_style)

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

        # Update the list row in place (no full reload / re-selection).
        self._saved_labels = existing
        text, tooltip = self._state_row_display(sid)
        item = self._state_list.item(row)
        if item is not None:
            item.setText(text)
            item.setToolTip(tooltip)
        if self._panel_collapsed:
            self._sync_collapsed_list()

    # ─────────────────────────────────── State clips ──

    def _clips_for_state(self, sid: int) -> list[Path]:
        """All exported clips for a state, from clips/state_<sid>/*.mp4.

        Mirrors characterize.load_clips() but scoped to one state directory so
        every generated clip (longest/typical/context/representative) is shown.
        """
        try:
            d = Path(_vc.get_clips_dir()) / f"state_{sid}"
        except Exception:
            return []
        if not d.is_dir():
            return []
        return sorted(d.glob("*.mp4"))

    @staticmethod
    def _clip_type_label(name: str) -> str:
        n = name.lower()
        if n.startswith("longest"):
            return "Longest bout"
        if n.startswith("typical"):
            return "Typical"
        if n.startswith("context_"):
            parts = name.split("_")
            return f"Context {parts[1]}" if len(parts) >= 3 else "Context"
        if n.startswith("clip"):
            return "Representative"
        return "Clip"

    def _clear_current_clip(self, status: str = "") -> None:
        """Reset clip state (no clip loaded) and disable clip nav/export."""
        self._current_clip_path = None
        self._clip_status.setText(status)
        self._export_clip_btn.setEnabled(False)
        self._prev_clip_btn.setEnabled(False)
        self._next_clip_btn.setEnabled(False)

    def _update_clip_nav_buttons(self) -> None:
        n = self._exemplar_table.rowCount()
        row = self._exemplar_table.currentRow()
        self._prev_clip_btn.setEnabled(n > 1 and row > 0)
        self._next_clip_btn.setEnabled(n > 1 and 0 <= row < n - 1)

    def _play_clip(self, path: Path | None) -> None:
        """Central clip loader: track the current clip, load it, honor Autoplay."""
        if path is None or not str(path):
            return
        if not path.exists():
            self._current_clip_path = None
            self._export_clip_btn.setEnabled(False)
            self._clip_status.setText(f"Clip file not found: {path}")
            return
        self._current_clip_path = path
        self._clip_status.setText(path.name)
        self._export_clip_btn.setEnabled(True)
        if self._player:
            try:
                self._player.load(str(path))
                if self._autoplay_cb.isChecked():
                    self._player.play()
            except Exception:
                self._clip_status.setText(f"Error loading {path.name}")

    def _populate_exemplars(self, sid: int) -> None:
        clips = self._clips_for_state(sid)
        self._exemplar_table.setRowCount(0)
        if not clips:
            self._exemplar_placeholder.show()
            self._exemplar_table.hide()
            self._clear_current_clip("")
            return

        self._exemplar_placeholder.hide()
        self._exemplar_table.show()
        self._exemplar_table.setSortingEnabled(False)
        self._exemplar_table.setRowCount(len(clips))
        for row_i, path in enumerate(clips):
            clip_item = QTableWidgetItem(path.name)
            clip_item.setData(Qt.UserRole, str(path))
            self._exemplar_table.setItem(row_i, 0, clip_item)
            self._exemplar_table.setItem(row_i, 1, QTableWidgetItem(self._clip_type_label(path.name)))
        self._exemplar_table.setSortingEnabled(True)
        # Auto-load the first clip when a state is selected.
        self._exemplar_table.selectRow(0)
        self._load_clip()
        self._update_clip_nav_buttons()

    def _selected_exemplar_path(self) -> Path | None:
        row = self._exemplar_table.currentRow()
        if row < 0:
            return None
        item = self._exemplar_table.item(row, 0)
        if item is None:
            return None
        raw = str(item.data(Qt.UserRole) or "")
        return Path(raw) if raw else None

    def _on_exemplar_selected(self) -> None:
        # Selecting a clip row loads it immediately (single source of clip playback).
        self._update_clip_nav_buttons()
        self._load_clip()

    def _load_clip(self) -> None:
        chosen = self._selected_exemplar_path()
        if chosen is None or not str(chosen):
            return
        self._play_clip(chosen)

    def _export_current_clip(self) -> None:
        """Copy the currently loaded clip to results/exports/ (single click)."""
        src = self._current_clip_path
        if src is None or not src.exists():
            QMessageBox.information(self, "Export Clip", "No clip is currently loaded.")
            return
        export_dir = RESULTS / "exports"
        try:
            export_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.warning(self, "Export Clip", f"Could not create export folder:\n{exc}")
            return
        sid = self._current_sid
        prefix = f"state_{sid}_" if sid is not None else ""
        dest = export_dir / f"{prefix}{src.name}"
        stem, suffix = dest.stem, dest.suffix
        counter = 1
        while dest.exists():
            dest = export_dir / f"{stem}_{counter}{suffix}"
            counter += 1
        try:
            shutil.copy2(src, dest)
        except Exception as exc:
            QMessageBox.warning(self, "Export Clip", f"Export failed:\n{exc}")
            return
        QMessageBox.information(self, "Clip Exported", f"Saved to:\n{dest}")

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

    # ─────────────────────────────────────── Clip cycling ──
    # (state-to-state navigation is via the state list only — see Part 1)

    def _prev_clip(self) -> None:
        row = self._exemplar_table.currentRow()
        if row > 0:
            # selectRow triggers _on_exemplar_selected, which loads the clip.
            self._exemplar_table.selectRow(row - 1)

    def _next_clip(self) -> None:
        row = self._exemplar_table.currentRow()
        if row < self._exemplar_table.rowCount() - 1:
            self._exemplar_table.selectRow(row + 1)

    def _on_video_finished(self) -> None:
        """When a clip ends and Autoplay is on, advance to the next clip (wrapping)."""
        if not self._autoplay_cb.isChecked():
            return
        n = self._exemplar_table.rowCount()
        if n <= 0:
            return
        if n == 1:
            # Single clip: replay it.
            self._play_clip(self._selected_exemplar_path())
            return
        nxt = (self._exemplar_table.currentRow() + 1) % n
        # selectRow triggers _on_exemplar_selected → load + play.
        self._exemplar_table.selectRow(nxt)

    # ─────────────────────────────────────── Command runner ──

    def _run_characterize_and_generate(self) -> None:
        self._pending_command = ["generate_clips.py"]
        self._run_command(["state_characterizer.py"], self._terminal)

    def _run_command(self, args: list[str], terminal: TerminalBox) -> None:
        if self._worker and self._worker.isRunning():
            return
        command = "python " + " ".join(str(a) for a in args)
        terminal.set_command(command)
        self._running_command = command
        self._last_run_kind = str(args[0]) if args else ""
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
        self._running_command = ""
        if ok and self._pending_command:
            next_cmd = self._pending_command
            self._pending_command = None
            self._run_command(next_cmd, self._terminal)
            return
        self._pending_command = None
        # Non-destructive: merge fresh outputs into the existing list instead of
        # rebuilding it (which dropped states/labels). See Part 5.
        self._soft_refresh_after_generation(ok)
