from __future__ import annotations

import os
import random
import shutil
import tempfile
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QListWidget, QMessageBox, QPushButton,
    QScrollArea, QSplitter, QStackedWidget, QTableWidget,
    QTableWidgetItem, QTextEdit, QToolButton, QVBoxLayout, QWidget,
)

from _utils import CLIPS, RESULTS, ROOT, _MPL, _state_colors
from _workers import SubprocessWorker

if _MPL:
    from _utils import Figure, FigureCanvas, mpl_cm, mpimg
    from _widgets import MplCanvas

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COHORT_COLORS = ["#4E79A7", "#E07B39", "#59A14F", "#B07AA1"]

_TAB_LABELS = [
    "Column Mapping",
    "Comparison Report",
    "State Characterization",
    "Cohort Analysis",
    "Quantification",
    "Fear Index",
    "Jess Correlation",
    "Event Alignment",
]


# ---------------------------------------------------------------------------
# TerminalBox
# ---------------------------------------------------------------------------

class TerminalBox(QWidget):
    """Dark terminal strip: shows last command and live stdout."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cmd = ""
        self._lines: deque[str] = deque(maxlen=20)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)

        self._edit = QTextEdit()
        self._edit.setReadOnly(True)
        self._edit.setFixedHeight(80)
        self._edit.setStyleSheet(
            "QTextEdit {"
            "  background:#1E1E1E; color:#E0E0E0;"
            "  font-family:'Consolas','Courier New',monospace; font-size:9pt;"
            "  border:1px solid #333; border-radius:3px; padding:4px;"
            "}"
        )
        lay.addWidget(self._edit, stretch=1)

        copy_btn = QToolButton()
        copy_btn.setText("⧉")   # ⧉
        copy_btn.setFixedSize(26, 26)
        copy_btn.setToolTip("Copy command to clipboard")
        copy_btn.clicked.connect(self._copy)
        copy_btn.setStyleSheet(
            "QToolButton { background:#2A2A2A; color:#AAA; border:none; font-size:13px; }"
            "QToolButton:hover { background:#3A3A3A; }"
        )
        lay.addWidget(copy_btn, alignment=Qt.AlignTop)

    def set_command(self, cmd: str) -> None:
        self._cmd = cmd
        self._lines.clear()
        self._refresh()

    def append_output(self, text: str) -> None:
        for line in text.splitlines():
            if line.strip():
                self._lines.append(line)
        self._refresh()

    def _refresh(self) -> None:
        parts = []
        if self._cmd:
            c = self._cmd.replace("&", "&amp;").replace("<", "&lt;")
            parts.append(f'<span style="color:#4EC9B0;">$ {c}</span>')
        for ln in self._lines:
            e = ln.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            parts.append(f'<span style="color:#E0E0E0;">{e}</span>')
        self._edit.setHtml("<br>".join(parts))
        sb = self._edit.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _copy(self) -> None:
        if self._cmd:
            QApplication.clipboard().setText(self._cmd)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _placeholder(msg: str) -> QLabel:
    lbl = QLabel(msg)
    lbl.setAlignment(Qt.AlignCenter)
    lbl.setWordWrap(True)
    lbl.setStyleSheet("color:#888; font-style:italic; padding:20px;")
    return lbl


def _section_title(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setFont(QFont("Arial", 11, QFont.Bold))
    lbl.setStyleSheet("color:#1A1A1A; padding-top:8px; padding-bottom:2px;")
    return lbl


# ---------------------------------------------------------------------------
# AnalysisView
# ---------------------------------------------------------------------------

class AnalysisView(QWidget):
    worker_running = pyqtSignal(bool)

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.cfg = cfg or {}
        self._data: dict = {}
        self._worker = None
        self._jess_df = None
        self._current_tab = 0
        self._t2_state_ids: list[int] = []
        # Each entry is True when that tab needs a redraw.
        # Starts True so the first visit always renders.
        self._tab_dirty = [True] * 8
        import vieb_config as _vc
        self._cond_a = _vc.get_condition_a_label()
        self._cond_b = _vc.get_condition_b_label()
        self._metric_label = _vc.get_primary_metric_label()
        self._build()

    # ─────────────────────────────────────────────────────────── build ──

    def _build_tab_names(self) -> list[str]:
        return [
            "Column Mapping",
            "Comparison Report",
            "State Characterization",
            "Cohort Analysis",
            "Quantification",
            self._metric_label,
            "Jess Correlation",
            "Event Alignment",
        ]

    def _build(self) -> None:
        root_lay = QHBoxLayout(self)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(0)

        # ── Vertical tab bar ─────────────────────────────────────────────
        self._tab_list = QListWidget()
        self._tab_list.setFixedWidth(160)
        self._tab_list.setSpacing(1)
        self._tab_list.setStyleSheet("""
            QListWidget {
                background: #F4F4F4;
                border: none;
                border-right: 1px solid #DCDCDC;
                padding-top: 8px;
                outline: none;
            }
            QListWidget::item {
                padding: 10px 14px;
                color: #333;
                font-size: 12px;
            }
            QListWidget::item:selected {
                background: #1a73e8;
                color: white;
            }
            QListWidget::item:hover:!selected {
                background: #E8E8E8;
            }
        """)
        for label in self._build_tab_names():
            self._tab_list.addItem(label)
        self._tab_list.setCurrentRow(0)
        self._tab_list.currentRowChanged.connect(self._switch_tab)
        root_lay.addWidget(self._tab_list)

        # ── Content stack ────────────────────────────────────────────────
        self._stack = QStackedWidget()
        builders = [
            self._build_tab0,
            self._build_tab1,
            self._build_tab2,
            self._build_tab3,
            self._build_tab4,
            self._build_tab5,
            self._build_tab6,
            self._build_tab7,
        ]
        for build in builders:
            page = build()
            self._stack.addWidget(page)

        root_lay.addWidget(self._stack, stretch=1)

    # ─────────────────────────────────── Shared header builder ──

    def _make_header(
        self,
        title: str,
        run_label: str,
        run_slot,
        terminal: TerminalBox,
        extra_widget: QWidget | None = None,
    ) -> QWidget:
        outer = QWidget()
        lay = QVBoxLayout(outer)
        lay.setContentsMargins(0, 0, 0, 6)
        lay.setSpacing(4)

        top = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setFont(QFont("Arial", 14, QFont.Bold))
        top.addWidget(lbl)
        if extra_widget:
            top.addWidget(extra_widget)
        top.addStretch()
        run_btn = QPushButton(run_label)
        run_btn.setFixedHeight(30)
        run_btn.clicked.connect(run_slot)
        top.addWidget(run_btn)
        lay.addLayout(top)
        lay.addWidget(terminal)
        return outer

    # ─────────────────────────────────────────── Tab 0: Column Mapping ──

    def _build_tab0(self) -> QWidget:
        from views.metadata_mapper import MetadataMapperWidget
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(0)
        self._mapper_widget = MetadataMapperWidget(self.cfg, parent=page)

        def _on_saved(col_map):
            self.cfg["column_map"] = col_map

        self._mapper_widget.mapping_saved.connect(_on_saved)
        lay.addWidget(self._mapper_widget)
        return page

    def _load_tab0(self) -> None:
        pass  # widget is self-contained; nothing to reload from data

    # ──────────────────────────────────────── Tab 1: Comparison Report ──

    def _build_tab1(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(8)

        self._t1_terminal = TerminalBox()
        hdr = self._make_header(
            "Comparison Report", "Run Report",
            lambda: self._run_command(["compare.py", "--report"], self._t1_terminal),
            self._t1_terminal,
        )
        lay.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setSpacing(16)

        self._t1_placeholder = _placeholder(
            "No comparison data found.\n"
            "Run pipeline stages 1–8 first, then click 'Run Report'.\n\n"
            "python compare.py --report"
        )
        cl.addWidget(self._t1_placeholder)

        if _MPL:
            cl.addWidget(_section_title(
                f"State Occupancy: {self._cond_a} vs {self._cond_b}"
            ))
            self._t1_ctx_canvas = MplCanvas(figsize=(9, 3))
            self._t1_ctx_canvas.setMinimumHeight(260)
            cl.addWidget(self._t1_ctx_canvas)

            cl.addWidget(_section_title(
                "Behavioral State Dynamics Across Conditioning Days"
            ))
            self._t1_day_canvas = MplCanvas(figsize=(9, 3))
            self._t1_day_canvas.setMinimumHeight(260)
            cl.addWidget(self._t1_day_canvas)

            cl.addWidget(_section_title("Condition Discrimination per State"))
            self._t1_disc_canvas = MplCanvas(figsize=(9, 3))
            self._t1_disc_canvas.setMinimumHeight(260)
            cl.addWidget(self._t1_disc_canvas)
        else:
            self._t1_ctx_canvas = self._t1_day_canvas = self._t1_disc_canvas = None
            cl.addWidget(_placeholder("Install matplotlib to view charts."))

        cl.addStretch()
        scroll.setWidget(content)
        lay.addWidget(scroll, stretch=1)
        return page

    # ──────────────────────────────── Tab 2: State Characterization ──

    def _build_tab2(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(8)

        self._t2_terminal = TerminalBox()
        hdr = self._make_header(
            "State Characterization", "Run Characterization",
            lambda: self._run_command(["characterize.py"], self._t2_terminal),
            self._t2_terminal,
        )
        lay.addWidget(hdr)

        splitter = QSplitter(Qt.Horizontal)

        # Left: state list
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 8, 0)
        ll.setSpacing(4)
        self._t2_search = QLineEdit()
        self._t2_search.setPlaceholderText("Search states…")
        self._t2_search.textChanged.connect(self._on_t2_search)
        ll.addWidget(self._t2_search)

        sort_row = QHBoxLayout()
        sort_row.setSpacing(4)
        sort_lbl = QLabel("Sort:")
        sort_lbl.setStyleSheet("font-size:11px; color:#666;")
        sort_row.addWidget(sort_lbl)
        self._t2_sort_combo = QComboBox()
        self._t2_sort_combo.addItems(["State ID", "Speed", "Elongation", "Bout Duration", "Angular Velocity"])
        self._t2_sort_combo.setStyleSheet("font-size:11px;")
        self._t2_sort_combo.currentIndexChanged.connect(self._on_t2_sort_changed)
        sort_row.addWidget(self._t2_sort_combo, stretch=1)
        self._t2_sort_asc = QToolButton()
        self._t2_sort_asc.setText("↑")
        self._t2_sort_asc.setCheckable(True)
        self._t2_sort_asc.setChecked(True)
        self._t2_sort_asc.setToolTip("Toggle ascending / descending")
        self._t2_sort_asc.setFixedWidth(26)
        self._t2_sort_asc.toggled.connect(self._on_t2_sort_changed)
        sort_row.addWidget(self._t2_sort_asc)
        ll.addLayout(sort_row)

        self._t2_state_list = QListWidget()
        self._t2_state_list.currentRowChanged.connect(self._on_t2_state_selected)
        ll.addWidget(self._t2_state_list, stretch=1)
        left.setFixedWidth(240)
        splitter.addWidget(left)

        # Right: detail
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 0, 0, 0)
        rl.setSpacing(8)

        self._t2_detail_placeholder = _placeholder("Select a state from the list.")
        rl.addWidget(self._t2_detail_placeholder)

        if _MPL:
            rl.addWidget(_section_title("Kinematic Profile"))
            self._t2_kin_canvas = MplCanvas(figsize=(6, 2.5))
            self._t2_kin_canvas.setMaximumHeight(200)
            self._t2_kin_canvas.hide()
            rl.addWidget(self._t2_kin_canvas)
        else:
            self._t2_kin_canvas = None

        lbl_row = QHBoxLayout()
        lbl_row.addWidget(QLabel("Label:"))
        self._t2_label_input = QLineEdit()
        self._t2_label_input.setPlaceholderText("Free-text label")
        lbl_row.addWidget(self._t2_label_input, stretch=1)
        self._t2_label_combo = QComboBox()
        self._t2_label_combo.addItems(["Freeze", "Walk", "Groom", "Rear", "Explore", "Other"])
        lbl_row.addWidget(self._t2_label_combo)
        save_btn = QPushButton("Save")
        save_btn.setFixedWidth(56)
        save_btn.clicked.connect(self._save_state_label)
        lbl_row.addWidget(save_btn)
        rl.addLayout(lbl_row)

        clip_row = QHBoxLayout()
        self._t2_load_clip_btn = QPushButton("Load Clip")
        self._t2_load_clip_btn.setEnabled(False)
        self._t2_load_clip_btn.clicked.connect(self._load_clip)
        clip_row.addWidget(self._t2_load_clip_btn)
        self._t2_clip_status = QLabel("")
        self._t2_clip_status.setStyleSheet("color:#888; font-size:11px;")
        clip_row.addWidget(self._t2_clip_status, stretch=1)
        rl.addLayout(clip_row)

        try:
            from _widgets import VideoPlayer
            self._t2_player = VideoPlayer()
            rl.addWidget(self._t2_player, stretch=1)
        except Exception:
            self._t2_player = None
            rl.addWidget(_placeholder(
                "Video player unavailable.\nInstall opencv-python to enable."
            ))

        splitter.addWidget(right)
        lay.addWidget(splitter, stretch=1)
        return page

    # ──────────────────────────────────────── Tab 3: Cohort Analysis ──

    def _build_tab3(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(8)

        self._t3_terminal = TerminalBox()

        # Header row with run controls
        top = QHBoxLayout()
        t3_title = QLabel("Cohort Analysis")
        t3_title.setFont(QFont("Arial", 14, QFont.Bold))
        top.addWidget(t3_title)

        top.addWidget(QLabel("  Group by:"))
        self._t3_groupby = QComboBox()
        self._t3_groupby.addItems([
            "age_treatment", "treatment", "age", "sex",
            "genotype_treatment", "age_sex", "full",
        ])
        top.addWidget(self._t3_groupby)

        cohort_p = self.cfg.get("cohort_csv_path", "")
        self._t3_cohort_lbl = QLabel(
            os.path.basename(cohort_p) if cohort_p else "no cohort file"
        )
        self._t3_cohort_lbl.setStyleSheet("color:#666; font-size:11px;")
        top.addWidget(self._t3_cohort_lbl)

        change_btn = QPushButton("Change")
        change_btn.setFixedHeight(26)
        change_btn.clicked.connect(self._change_cohort_file)
        top.addWidget(change_btn)

        top.addStretch()
        run3_btn = QPushButton("Run Cohort Analysis")
        run3_btn.setFixedHeight(30)
        run3_btn.clicked.connect(
            lambda: self._run_command(self._cohort_cmd(), self._t3_terminal)
        )
        top.addWidget(run3_btn)
        lay.addLayout(top)
        lay.addWidget(self._t3_terminal)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setSpacing(16)

        self._t3_placeholder = _placeholder(
            "No quantification data found.\n\n"
            "python compare.py --quantify --cohort cohort_normalized.csv"
        )
        cl.addWidget(self._t3_placeholder)

        if _MPL:
            cl.addWidget(_section_title("Behavioral State Occupancy by Cohort"))
            self._t3_profiles_canvas = MplCanvas(figsize=(9, 3.5))
            self._t3_profiles_canvas.setMinimumHeight(280)
            cl.addWidget(self._t3_profiles_canvas)

            cl.addWidget(_section_title("Behavioral Phenotype Summary by Cohort"))
            self._t3_metrics_canvas = MplCanvas(figsize=(9, 3))
            self._t3_metrics_canvas.setMinimumHeight(250)
            cl.addWidget(self._t3_metrics_canvas)

            cl.addWidget(_section_title("Per-Animal Behavioral Profile"))
            self._t3_heatmap_canvas = MplCanvas(figsize=(9, 4))
            self._t3_heatmap_canvas.setMinimumHeight(300)
            cl.addWidget(self._t3_heatmap_canvas)
        else:
            self._t3_profiles_canvas = None
            self._t3_metrics_canvas = None
            self._t3_heatmap_canvas = None
            cl.addWidget(_placeholder("Install matplotlib to view charts."))

        cl.addWidget(_section_title("Notable Distinctions (FDR p < 0.05)"))
        self._t3_sig_table = QTableWidget(0, 5)
        self._t3_sig_table.setHorizontalHeaderLabels(
            ["State", "Label", "Cohorts", "Fold Change", "p FDR"]
        )
        self._t3_sig_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._t3_sig_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._t3_sig_table.setMaximumHeight(200)
        cl.addWidget(self._t3_sig_table)

        cl.addStretch()
        scroll.setWidget(content)
        lay.addWidget(scroll, stretch=1)
        return page

    # ──────────────────────────────────────── Tab 4: Quantification ──

    def _build_tab4(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(8)

        self._t4_terminal = TerminalBox()

        top = QHBoxLayout()
        t4_title = QLabel("Quantification")
        t4_title.setFont(QFont("Arial", 14, QFont.Bold))
        top.addWidget(t4_title)
        top.addStretch()
        self._t4_export_btn = QPushButton("Export CSV")
        self._t4_export_btn.setFixedHeight(30)
        self._t4_export_btn.clicked.connect(self._export_master_table)
        top.addWidget(self._t4_export_btn)
        cohort_p = self.cfg.get("cohort_csv_path", "cohort_normalized.csv") or "cohort_normalized.csv"
        run4_btn = QPushButton("Run Quantification")
        run4_btn.setFixedHeight(30)
        run4_btn.clicked.connect(
            lambda: self._run_command(
                ["compare.py", "--quantify", "--cohort",
                 self.cfg.get("cohort_csv_path", "cohort_normalized.csv") or "cohort_normalized.csv"],
                self._t4_terminal,
            )
        )
        top.addWidget(run4_btn)
        lay.addLayout(top)
        lay.addWidget(self._t4_terminal)

        self._t4_placeholder = _placeholder(
            "No master table found.\n\n"
            "python compare.py --quantify --cohort cohort_normalized.csv"
        )
        lay.addWidget(self._t4_placeholder)

        self._t4_table = QTableWidget(0, 0)
        self._t4_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._t4_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._t4_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._t4_table.setSortingEnabled(True)
        self._t4_table.hide()
        lay.addWidget(self._t4_table, stretch=1)
        return page

    # ──────────────────────────────────────────── Tab 5: Fear Index ──

    def _build_tab5(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(8)

        self._t5_terminal = TerminalBox()
        hdr = self._make_header(
            self._metric_label, f"Run {self._metric_label}",
            lambda: self._run_command(
                ["fear_index.py", "--cohort",
                 self.cfg.get("cohort_csv_path", "cohort_normalized.csv") or "cohort_normalized.csv"],
                self._t5_terminal,
            ),
            self._t5_terminal,
        )
        lay.addWidget(hdr)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setSpacing(16)

        self._t5_placeholder = _placeholder(
            "No fear index data found.\n\n"
            "python fear_index.py --cohort cohort_normalized.csv"
        )
        cl.addWidget(self._t5_placeholder)

        if _MPL:
            cl.addWidget(_section_title(f"{self._metric_label} per Animal (cohort-normalized)"))
            self._t5_animals_canvas = MplCanvas(figsize=(9, 5))
            self._t5_animals_canvas.setMinimumHeight(350)
            cl.addWidget(self._t5_animals_canvas)

            cl.addWidget(_section_title(f"Mean {self._metric_label} by Cohort"))
            self._t5_cohort_canvas = MplCanvas(figsize=(9, 3))
            self._t5_cohort_canvas.setMinimumHeight(260)
            cl.addWidget(self._t5_cohort_canvas)
        else:
            self._t5_animals_canvas = None
            self._t5_cohort_canvas = None
            cl.addWidget(_placeholder("Install matplotlib to view charts."))

        cl.addWidget(_section_title("Cohort Summary"))
        self._t5_stats_table = QTableWidget(0, 5)
        self._t5_stats_table.setHorizontalHeaderLabels(
            ["Cohort", "N Animals", "Mean Fear Index", "CI Low", "CI High"]
        )
        self._t5_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._t5_stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._t5_stats_table.setMaximumHeight(180)
        cl.addWidget(self._t5_stats_table)

        cl.addStretch()
        scroll.setWidget(content)
        lay.addWidget(scroll, stretch=1)
        return page

    # ────────────────────────────────────── Tab 6: Jess Correlation ──

    def _build_tab6(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(8)

        self._t6_terminal = TerminalBox()

        top = QHBoxLayout()
        t6_title = QLabel("Jess Correlation")
        t6_title.setFont(QFont("Arial", 14, QFont.Bold))
        top.addWidget(t6_title)
        top.addStretch()
        self._t6_run_btn = QPushButton("Run Correlation")
        self._t6_run_btn.setFixedHeight(30)
        self._t6_run_btn.setEnabled(False)
        self._t6_run_btn.clicked.connect(self._run_jess)
        top.addWidget(self._t6_run_btn)
        lay.addLayout(top)
        lay.addWidget(self._t6_terminal)

        # Import section
        imp_row = QHBoxLayout()
        self._t6_import_btn = QPushButton("Import Jess Data (.csv / .xlsx)")
        self._t6_import_btn.clicked.connect(self._import_jess)
        imp_row.addWidget(self._t6_import_btn)
        self._t6_status_lbl = QLabel("No file loaded")
        self._t6_status_lbl.setStyleSheet("color:#888; font-size:11px;")
        imp_row.addWidget(self._t6_status_lbl, stretch=1)
        lay.addLayout(imp_row)

        self._t6_match_lbl = QLabel("")
        self._t6_match_lbl.setStyleSheet("color:#1b5e20; font-size:11px;")
        lay.addWidget(self._t6_match_lbl)

        fmt_lbl = QLabel(
            "Expected format: CSV/XLSX with columns: animal_id, GluA1, GluA2, NMDA1, …"
        )
        fmt_lbl.setStyleSheet("color:#777; font-size:11px; font-style:italic;")
        lay.addWidget(fmt_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setSpacing(16)

        self._t6_placeholder = _placeholder(
            "Import Jess protein data above, then click 'Run Correlation'."
        )
        cl.addWidget(self._t6_placeholder)

        if _MPL:
            cl.addWidget(_section_title(
                "Behavioral × Protein Correlation (Pearson r)"
            ))
            self._t6_heatmap_canvas = MplCanvas(figsize=(9, 4))
            self._t6_heatmap_canvas.setMinimumHeight(300)
            cl.addWidget(self._t6_heatmap_canvas)

            cl.addWidget(_section_title(
                "Strongest Behavioral-Molecular Associations"
            ))
            self._t6_top_canvas = MplCanvas(figsize=(9, 3.5))
            self._t6_top_canvas.setMinimumHeight(280)
            cl.addWidget(self._t6_top_canvas)
        else:
            self._t6_heatmap_canvas = None
            self._t6_top_canvas = None
            cl.addWidget(_placeholder("Install matplotlib to view charts."))

        note = QLabel(
            "Pearson r with FDR correction (Benjamini-Hochberg). "
            "Full results in results/quantification/jess_correlations.csv."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#666; font-size:10px; font-style:italic;")
        cl.addWidget(note)

        cl.addStretch()
        scroll.setWidget(content)
        lay.addWidget(scroll, stretch=1)
        return page

    # ─────────────────────────────────────────────────── Tab switching ──

    _TAB_LOADERS = None  # populated after first call to avoid forward-ref issues

    def _get_loaders(self):
        return [
            self._load_tab0,
            self._load_tab1, self._load_tab2, self._load_tab3,
            self._load_tab4, self._load_tab5, self._load_tab6,
            self._load_tab7,
        ]

    def _switch_tab(self, idx: int) -> None:
        if idx < 0 or idx >= self._stack.count():
            return
        self._current_tab = idx
        self._stack.setCurrentIndex(idx)
        # Only render if dirty — skip if the chart is already up-to-date.
        if self._tab_dirty[idx]:
            self._get_loaders()[idx]()
            self._tab_dirty[idx] = False

    def _load_current_tab(self) -> None:
        """Reload the visible tab unconditionally (e.g. after a pipeline run)."""
        idx = self._current_tab
        if 0 <= idx < self._stack.count():
            self._get_loaders()[idx]()
            self._tab_dirty[idx] = False

    def _mark_all_dirty(self) -> None:
        """Flag every tab for redraw on next visit."""
        self._tab_dirty = [True] * 8

    # ────────────────────────────────────────────────── Data loading ──

    def update_data(self, data: dict) -> None:
        if data is not self._data:
            self._data = data
            loaders = self._get_loaders()
            for i, loader in enumerate(loaders):
                loader()
                self._tab_dirty[i] = False

    def refresh(self, data: dict) -> None:
        """Reload only the currently visible tab with fresh data; others reload lazily on switch."""
        self._data = data
        self._mark_all_dirty()
        self._load_current_tab()

    # ─────────────────────── Tab 1 loader ──

    def _load_tab1(self) -> None:
        summary = self._data.get("summary")
        ci = self._data.get("cluster_info") or {}
        n = int(ci.get("n_clusters", 0))

        has_data = (
            summary is not None and not summary.empty and n > 0
        )
        self._t1_placeholder.setVisible(not has_data)

        if not has_data or not _MPL:
            return

        state_cols = [
            f"state_{i}_frac" for i in range(n)
            if f"state_{i}_frac" in summary.columns
        ]
        if not state_cols:
            return
        state_ids = [int(c.split("_")[1]) for c in state_cols]

        # — Section 1: context occupancy —
        canvas = self._t1_ctx_canvas
        canvas.fig.clf()
        ax = canvas.fig.add_subplot(111)
        if "context" in summary.columns:
            ga = summary[summary["context"].astype(str) == "A"]
            gb = summary[summary["context"].astype(str) == "B"]
            if not ga.empty and not gb.empty:
                x = np.arange(len(state_ids))
                w = 0.36
                ax.bar(x - w / 2,
                       [ga[c].mean() for c in state_cols], w,
                       yerr=[ga[c].sem() for c in state_cols],
                       color="#E63946", alpha=0.85, label=self._cond_a,
                       capsize=3, zorder=3)
                ax.bar(x + w / 2,
                       [gb[c].mean() for c in state_cols], w,
                       yerr=[gb[c].sem() for c in state_cols],
                       color="#4361EE", alpha=0.85, label=self._cond_b,
                       capsize=3, zorder=3)
                ax.set_xticks(x)
                ax.set_xticklabels([f"S{i}" for i in state_ids], fontsize=8)
                ax.set_ylabel("Mean Fraction")
                ax.legend(fontsize=9)
            else:
                ax.text(0.5, 0.5, "No A/B context rows in summary_table",
                        ha="center", va="center", transform=ax.transAxes, color="#999")
        else:
            ax.text(0.5, 0.5, "No 'context' column in summary_table",
                    ha="center", va="center", transform=ax.transAxes, color="#999")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, color="#EEEEEE", zorder=0)
        canvas.fig.tight_layout()
        canvas.draw()

        # — Section 2: state by day —
        canvas2 = self._t1_day_canvas
        canvas2.fig.clf()
        ax2 = canvas2.fig.add_subplot(111)
        if "day" in summary.columns:
            by_day = summary.groupby("day")[state_cols].mean()
            variances = summary[state_cols].var()
            top5 = variances.nlargest(5).index.tolist()
            colors = _state_colors(n)
            for col in top5:
                sid = int(col.split("_")[1])
                c = colors[min(sid, len(colors) - 1)]
                ax2.plot(by_day.index, by_day[col], marker="o",
                         label=f"S{sid}", color=c, linewidth=1.8, markersize=4)
            ax2.set_xlabel("Day")
            ax2.set_ylabel("Mean Fraction")
            ax2.legend(fontsize=8, ncol=3)
        else:
            ax2.text(0.5, 0.5, "No 'day' column in summary_table",
                     ha="center", va="center", transform=ax2.transAxes, color="#999")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.yaxis.grid(True, color="#EEEEEE", zorder=0)
        canvas2.fig.tight_layout()
        canvas2.draw()

        # — Section 3: discrimination —
        canvas3 = self._t1_disc_canvas
        canvas3.fig.clf()
        ax3 = canvas3.fig.add_subplot(111)
        if "context" in summary.columns:
            ga = summary[summary["context"].astype(str) == "A"]
            gb = summary[summary["context"].astype(str) == "B"]
            if not ga.empty and not gb.empty:
                disc = np.array(
                    [ga[c].mean() - gb[c].mean() for c in state_cols]
                )
                order = np.argsort(disc)[::-1]
                s_ord = [state_ids[i] for i in order]
                d_ord = disc[order]
                colors3 = ["#E63946" if v >= 0 else "#4361EE" for v in d_ord]
                ax3.bar(range(len(s_ord)), d_ord, color=colors3, alpha=0.85)
                ax3.set_xticks(range(len(s_ord)))
                ax3.set_xticklabels([f"S{i}" for i in s_ord], fontsize=8)
                ax3.axhline(0, color="#999", linewidth=0.8)
                ax3.set_ylabel("Mean A − Mean B")
            else:
                ax3.text(0.5, 0.5, "No A/B context rows",
                         ha="center", va="center", transform=ax3.transAxes, color="#999")
        else:
            ax3.text(0.5, 0.5, "No 'context' column",
                     ha="center", va="center", transform=ax3.transAxes, color="#999")
        ax3.spines["top"].set_visible(False)
        ax3.spines["right"].set_visible(False)
        ax3.yaxis.grid(True, color="#EEEEEE", zorder=0)
        canvas3.fig.tight_layout()
        canvas3.draw()

    # ─────────────────────── Tab 2 loader + helpers ──

    def _load_tab2(self) -> None:
        ss = self._data.get("state_summary")
        ctx = self._data.get("context_report")

        self._t2_state_list.clear()
        self._t2_state_ids = []
        self._t2_load_clip_btn.setEnabled(False)
        self._t2_clip_status.setText("")
        self._t2_detail_placeholder.show()
        if self._t2_kin_canvas:
            self._t2_kin_canvas.hide()

        if ss is None or ss.empty:
            self._t2_state_list.addItem("No data — run characterize.py")
            return

        id_col = next(
            (c for c in ("state_id", "cluster_id", "state") if c in ss.columns), None
        )
        if id_col is None:
            self._t2_state_list.addItem("Missing state_id column")
            return

        ctx_enrich: dict[int, str] = {}
        if ctx is not None and not ctx.empty and id_col in ctx.columns:
            for _, row in ctx.iterrows():
                sid = int(row.get(id_col, -1))
                enrich = str(row.get("enriched_context", ""))
                if enrich and enrich != "nan":
                    ctx_enrich[sid] = enrich

        self._t2_id_col = id_col
        self._t2_ctx_enrich = ctx_enrich
        self._populate_t2_list()

    _T2_SORT_COLS = [
        None,                    # State ID
        "mean_centroid_speed",   # Speed
        "mean_elongation",       # Elongation
        "mean_bout_dur_sec",     # Bout Duration
        "mean_angular_vel",      # Angular Velocity
    ]

    def _populate_t2_list(self) -> None:
        ss = self._data.get("state_summary")
        if ss is None or ss.empty:
            return
        id_col = getattr(self, "_t2_id_col", None)
        if id_col is None:
            return
        ctx_enrich = getattr(self, "_t2_ctx_enrich", {})

        sort_idx = self._t2_sort_combo.currentIndex()
        sort_col = self._T2_SORT_COLS[sort_idx]
        ascending = self._t2_sort_asc.isChecked()

        if sort_col and sort_col in ss.columns:
            df = ss.sort_values(sort_col, ascending=ascending, na_position="last")
        else:
            df = ss.sort_values(id_col, ascending=ascending)

        self._t2_state_list.clear()
        self._t2_state_ids = []
        for _, row in df.iterrows():
            sid = int(row.get(id_col, -1))
            label = str(row.get("heuristic_label", f"State {sid}"))
            speed = row.get("mean_centroid_speed", None)
            enrich = ctx_enrich.get(sid, "")
            speed_str = (
                f"  spd={speed:.3f}" if speed is not None and not pd.isna(speed) else ""
            )
            badge = f"  [{enrich}]" if enrich else ""
            self._t2_state_list.addItem(f"S{sid}: {label}{speed_str}{badge}")
            self._t2_state_ids.append(sid)

    def _on_t2_sort_changed(self, _=None) -> None:
        self._t2_sort_asc.setText("↑" if self._t2_sort_asc.isChecked() else "↓")
        self._populate_t2_list()

    def _on_t2_search(self, text: str) -> None:
        for i in range(self._t2_state_list.count()):
            item = self._t2_state_list.item(i)
            item.setHidden(bool(text) and text.lower() not in item.text().lower())

    def _on_t2_state_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._t2_state_ids):
            return
        sid = self._t2_state_ids[row]
        self._t2_detail_placeholder.hide()
        self._t2_load_clip_btn.setEnabled(True)
        self._t2_load_clip_btn.setProperty("_sid", sid)

        ss = self._data.get("state_summary")
        if ss is None:
            return
        id_col = next(
            (c for c in ("state_id", "cluster_id", "state") if c in ss.columns), None
        )
        if id_col is None:
            return
        rows = ss[ss[id_col] == sid]
        if rows.empty:
            return
        r = rows.iloc[0]

        if _MPL and self._t2_kin_canvas:
            self._t2_kin_canvas.show()
            canvas = self._t2_kin_canvas
            canvas.fig.clf()
            ax = canvas.fig.add_subplot(111)
            metrics = {}
            for display, col in [
                ("Mean Speed", "mean_centroid_speed"),
                ("Angular Vel.", "mean_angular_vel"),
                ("Bout Duration", "mean_bout_dur_sec"),
                ("Elongation", "mean_elongation"),
            ]:
                v = r.get(col, None)
                if v is not None and not pd.isna(v):
                    metrics[display] = float(v)
            if metrics:
                vals = list(metrics.values())
                max_v = max(abs(v) for v in vals) or 1.0
                norm_v = [v / max_v for v in vals]
                y = np.arange(len(metrics))
                colors = ["#4E79A7" if v >= 0 else "#E63946" for v in norm_v]
                ax.barh(y, norm_v, color=colors, alpha=0.85)
                ax.set_yticks(y)
                ax.set_yticklabels(list(metrics.keys()), fontsize=9)
                ax.set_xlabel("Normalized value (relative to max)")
                ax.axvline(0, color="#999", linewidth=0.8)
            else:
                ax.text(0.5, 0.5, "No kinematic data",
                        ha="center", va="center", transform=ax.transAxes, color="#999")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            canvas.fig.tight_layout()
            canvas.draw()

        saved = self._load_saved_state_labels().get(sid, {})
        self._t2_label_input.setText(saved.get("label", ""))
        cat = saved.get("category", "")
        if cat:
            idx = self._t2_label_combo.findText(cat)
            if idx >= 0:
                self._t2_label_combo.setCurrentIndex(idx)

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
        row = self._t2_state_list.currentRow()
        if row < 0 or row >= len(self._t2_state_ids):
            return
        sid = self._t2_state_ids[row]
        existing = self._load_saved_state_labels()
        existing[sid] = {
            "label": self._t2_label_input.text().strip(),
            "category": self._t2_label_combo.currentText(),
        }
        p = RESULTS / "validation" / "state_labels.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"state_id": k, "label": v["label"], "category": v["category"]}
            for k, v in sorted(existing.items())
        ]
        pd.DataFrame(rows).to_csv(p, index=False)

    def _load_clip(self) -> None:
        sid = self._t2_load_clip_btn.property("_sid")
        if sid is None:
            return
        clip_dir = CLIPS / f"state_{sid}"
        if not clip_dir.exists():
            self._t2_clip_status.setText(
                f"No clips found for state {sid} — run: python generate_clips.py"
            )
            return
        clips = list(clip_dir.glob("*.mp4"))
        if not clips:
            self._t2_clip_status.setText(f"No .mp4 files in clips/state_{sid}/")
            return
        chosen = random.choice(clips)
        self._t2_clip_status.setText(chosen.name)
        if self._t2_player:
            try:
                self._t2_player.load(str(chosen))
            except Exception:
                self._t2_clip_status.setText(f"Error loading {chosen.name}")

    # ─────────────────────── Tab 3 loader + helpers ──

    def _cohort_cmd(self) -> list[str]:
        gb = self._t3_groupby.currentText()
        cohort_p = self.cfg.get("cohort_csv_path", "") or "cohort_normalized.csv"
        cmd = ["cohort_analysis.py", "--groupby", gb, "--cohort", cohort_p]
        out_dir = self.cfg.get("cohort_output_dir", "").strip()
        if out_dir:
            cmd += ["--output", out_dir]
        return cmd

    def _change_cohort_file(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Cohort File", "",
            "Data files (*.csv *.xlsx *.xls);;All files (*.*)"
        )
        if not p:
            return
        self.cfg["cohort_csv_path"] = p
        self._t3_cohort_lbl.setText(os.path.basename(p))
        try:
            from _utils import _save_cfg
            _save_cfg(self.cfg)
        except Exception:
            pass

    def _get_phenotype_metrics(self) -> list[tuple[str, str]]:
        """Ordered preference list for the three phenotype summary panels.
        First column that exists and has non-null values wins each slot."""
        return [
            ("discrimination_ratio_mean", "Discrimination Ratio"),
            ("context_discrimination",    "Condition Discrimination"),
            ("fear_auc",                  f"{self._cond_a} AUC"),
            ("fear_index",                self._metric_label),
            ("behavioral_diversity",      "Behavioral Diversity"),
            ("transition_entropy_A",      "Transition Entropy (A)"),
        ]

    # Columns that are identity / categorical — excluded from heatmap
    _HEATMAP_EXCLUDE = frozenset({
        "animal_id", "cohort_label", "sex", "age_group", "genotype",
        "treatment", "n_sessions", "n_days", "freeze_state",
        "dominant_fear_state", "dominant_safety_state",
        "n_A_sessions", "n_B_sessions", "n_fear_states",
        "fit_success", "n_states_used_A", "n_states_used_B",
        "discrimination_ratio_peak_day",
    })

    def _load_tab3(self) -> None:
        # Primary data: master_table.csv (generated by compare.py --quantify)
        master_path = RESULTS / "quantification" / "master_table.csv"
        master: pd.DataFrame | None = None
        if master_path.exists():
            try:
                master = pd.read_csv(master_path)
                master["animal_id"] = master["animal_id"].astype(str)
                if master.empty or "cohort_label" not in master.columns:
                    master = None
            except Exception:
                master = None

        has_master = master is not None
        self._t3_placeholder.setVisible(not has_master)
        if not has_master or not _MPL:
            return

        # — Plot 1: cohort state profiles PNG (from cohort_analysis.py) —
        _out = self.cfg.get("cohort_output_dir", "").strip()
        cohort_dir = Path(_out) if _out else RESULTS / "cohort"
        profiles_png = cohort_dir / "cohort_state_profiles.png"
        if profiles_png.exists() and self._t3_profiles_canvas:
            try:
                self._t3_profiles_canvas.fig.clf()
                ax = self._t3_profiles_canvas.fig.add_subplot(111)
                img = mpimg.imread(str(profiles_png))
                ax.imshow(img)
                ax.axis("off")
                self._t3_profiles_canvas.fig.tight_layout()
                self._t3_profiles_canvas.draw()
            except Exception:
                pass

        # — Plot 2: Behavioral Phenotype Summary —
        self._render_phenotype_summary(master)

        # — Plot 3: Per-Animal Behavioral Profile —
        self._render_animal_heatmap(master)

        # — Notable distinctions table —
        sig_csv = cohort_dir / "cohort_significant_states.csv"
        if sig_csv.exists():
            try:
                sig = pd.read_csv(sig_csv)
                self._t3_sig_table.setRowCount(len(sig))
                for ri, row in sig.reset_index(drop=True).iterrows():
                    def _cell(val):
                        return QTableWidgetItem(
                            f"{val:.3f}" if isinstance(val, float) else str(val)
                        )
                    self._t3_sig_table.setItem(ri, 0, _cell(row.get("state_id", "")))
                    self._t3_sig_table.setItem(ri, 1, _cell(row.get("label", "")))
                    self._t3_sig_table.setItem(ri, 2, _cell(row.get("cohorts", "")))
                    self._t3_sig_table.setItem(ri, 3, _cell(row.get("fold_change", "")))
                    self._t3_sig_table.setItem(ri, 4, _cell(row.get("p_fdr", "")))
            except Exception:
                pass

    def _render_phenotype_summary(self, master: pd.DataFrame) -> None:
        """Three-panel bar chart: one bar per cohort per metric, dots overlaid."""
        canvas = self._t3_metrics_canvas
        if not canvas:
            return

        # Pick up to 3 metrics in priority order
        metrics: list[tuple[str, str]] = []
        seen_labels: set[str] = set()
        for col, label in self._get_phenotype_metrics():
            if col in master.columns and master[col].notna().any():
                if label not in seen_labels:
                    metrics.append((col, label))
                    seen_labels.add(label)
            if len(metrics) == 3:
                break

        canvas.fig.clf()
        if not metrics:
            ax = canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "No behavioral scalars found in master_table.csv\n"
                    "Run: python compare.py --quantify --cohort cohort_normalized.csv",
                    ha="center", va="center", transform=ax.transAxes,
                    color="#888", fontsize=9)
            canvas.draw()
            return

        cohort_groups = sorted(master["cohort_label"].dropna().unique())
        n_panels = len(metrics)
        axes = canvas.fig.subplots(1, n_panels)
        if n_panels == 1:
            axes = [axes]

        np.random.seed(42)
        for ax, (col, label) in zip(axes, metrics):
            for gi, grp in enumerate(cohort_groups):
                vals = master[master["cohort_label"] == grp][col].dropna().values
                if not len(vals):
                    continue
                color = COHORT_COLORS[gi % len(COHORT_COLORS)]
                mean_v = float(vals.mean())
                se_v = float(vals.std() / np.sqrt(len(vals))) if len(vals) > 1 else 0.0

                ax.bar(gi, mean_v, color=color, alpha=0.82, width=0.6, zorder=2)
                ax.errorbar(gi, mean_v, yerr=se_v, color="black",
                            linewidth=1.5, capsize=4, zorder=3)
                jitter = np.random.uniform(-0.12, 0.12, len(vals))
                ax.scatter(gi + jitter, vals, color=color, s=35, zorder=4,
                           edgecolors="white", linewidth=0.5, alpha=0.9)

            if "discrimination" in col or "disc" in col:
                ax.axhline(0, color="#9B9B9B", linewidth=0.8,
                           linestyle="--", zorder=1)

            ax.set_xticks(range(len(cohort_groups)))
            # Shorten long cohort labels: keep last two underscore-segments
            short = []
            for g in cohort_groups:
                parts = g.split("_")
                short.append("\n".join(parts[-2:]) if len(parts) >= 2 else g)
            ax.set_xticklabels(short, fontsize=7)
            ax.set_ylabel(label, fontsize=9)
            ax.set_title(label, fontsize=9, fontweight="bold", loc="left")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.yaxis.grid(True, color="#EEEEEE", zorder=0)
            ax.set_axisbelow(True)

        canvas.fig.tight_layout()
        canvas.draw()

    def _render_animal_heatmap(self, master: pd.DataFrame) -> None:
        """Rows = animals, cols = behavioral scalars, color = z-score."""
        canvas = self._t3_heatmap_canvas
        if not canvas:
            return

        # Select numeric behavioral scalar columns (no per-state fractions)
        scalar_cols = [
            c for c in master.columns
            if c not in self._HEATMAP_EXCLUDE
            and not any(c.startswith(f"s{i}_") for i in range(30))
            and pd.api.types.is_numeric_dtype(master[c])
            and master[c].notna().sum() >= 3
            and master[c].std() > 0
        ]

        canvas.fig.clf()
        if len(scalar_cols) < 2:
            ax = canvas.fig.add_subplot(111)
            ax.text(0.5, 0.5, "Not enough numeric columns for heatmap",
                    ha="center", va="center", transform=ax.transAxes, color="#888")
            canvas.draw()
            return

        # Sort: cohort then discrimination_ratio_mean descending
        sort_col = ("discrimination_ratio_mean"
                    if "discrimination_ratio_mean" in master.columns
                    else scalar_cols[0])
        df = master.sort_values(
            ["cohort_label", sort_col],
            ascending=[True, False],
            na_position="last",
        ).reset_index(drop=True)

        # Z-score each scalar column
        z = df[scalar_cols].copy().astype(float)
        for col in scalar_cols:
            std = float(z[col].std())
            if std > 0:
                z[col] = (z[col] - z[col].mean()) / std
            else:
                z[col] = 0.0
        z = z.fillna(0)

        n_animals = len(df)
        vmax = min(3.0, float(np.abs(z.values).max()) or 1.0)

        ax = canvas.fig.add_subplot(111)
        im = ax.imshow(
            z.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
            aspect="auto", interpolation="nearest",
        )

        ax.set_xticks(range(len(scalar_cols)))
        ax.set_xticklabels(scalar_cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n_animals))
        ax.set_yticklabels(df["animal_id"].astype(str).tolist(), fontsize=7)
        ax.set_xlabel("Behavioral Metric", fontsize=9)

        # White lines between cohorts
        prev_cohort = None
        for i, cohort in enumerate(df["cohort_label"].values):
            if cohort != prev_cohort and i > 0:
                ax.axhline(i - 0.5, color="white", linewidth=2.0)
            prev_cohort = cohort

        canvas.fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Z-score")
        canvas.fig.tight_layout()
        canvas.draw()

    # ─────────────────────── Tab 4 loader + helpers ──

    def _load_tab4(self) -> None:
        p = RESULTS / "quantification" / "master_table.csv"
        if not p.exists():
            self._t4_placeholder.show()
            self._t4_table.hide()
            return
        try:
            df = pd.read_csv(p)
            self._t4_placeholder.hide()
            self._t4_table.show()
            self._t4_table.setSortingEnabled(False)
            self._t4_table.setRowCount(len(df))
            self._t4_table.setColumnCount(len(df.columns))
            self._t4_table.setHorizontalHeaderLabels(list(df.columns))
            for ri, row in df.iterrows():
                for ci, val in enumerate(row):
                    self._t4_table.setItem(ri, ci, QTableWidgetItem(str(val)))
            self._t4_table.setSortingEnabled(True)
        except Exception as exc:
            self._t4_placeholder.setText(f"Error loading master_table.csv:\n{exc}")
            self._t4_placeholder.show()
            self._t4_table.hide()

    def _export_master_table(self) -> None:
        p = RESULTS / "quantification" / "master_table.csv"
        if not p.exists():
            QMessageBox.warning(self, "No Data", "master_table.csv not found.")
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Export Master Table", "master_table.csv", "CSV (*.csv)"
        )
        if dest:
            shutil.copy2(str(p), dest)
            QMessageBox.information(self, "Exported", f"Saved to {dest}")

    # ─────────────────────── Tab 5 loader ──

    def _load_tab5(self) -> None:
        p = RESULTS / "quantification" / "fear_index.csv"
        if not p.exists():
            self._t5_placeholder.show()
            if self._t5_animals_canvas:
                self._t5_animals_canvas.hide()
            if self._t5_cohort_canvas:
                self._t5_cohort_canvas.hide()
            return

        try:
            df = pd.read_csv(p)
        except Exception:
            return

        self._t5_placeholder.hide()
        if not _MPL:
            return

        cohort_col = next(
            (c for c in ("cohort", "group", "treatment") if c in df.columns), None
        )

        # Plot 1: horizontal bar per animal
        canvas1 = self._t5_animals_canvas
        canvas1.show()
        canvas1.fig.clf()
        ax1 = canvas1.fig.add_subplot(111)
        if "fear_index" in df.columns and "animal_id" in df.columns:
            df_s = df.sort_values("fear_index", ascending=True)
            groups_list = list(df[cohort_col].unique()) if cohort_col else []
            bar_colors = []
            for _, row in df_s.iterrows():
                if cohort_col and row[cohort_col] in groups_list:
                    gi = groups_list.index(row[cohort_col])
                    bar_colors.append(COHORT_COLORS[gi % len(COHORT_COLORS)])
                else:
                    bar_colors.append("#4E79A7")
            ax1.barh(range(len(df_s)), df_s["fear_index"].values,
                     color=bar_colors, alpha=0.85)
            ax1.set_yticks(range(len(df_s)))
            ax1.set_yticklabels(df_s["animal_id"].astype(str).tolist(), fontsize=7)
            ax1.axvline(0, color="#999", linewidth=1)
            ax1.set_xlabel(self._metric_label)
        else:
            ax1.text(0.5, 0.5, "Need fear_index and animal_id columns",
                     ha="center", va="center", transform=ax1.transAxes, color="#999")
        ax1.spines["top"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        canvas1.fig.tight_layout()
        canvas1.draw()

        # Plot 2: Mean Fear Index by Cohort (from cohort_fear_profiles.csv)
        profiles_p = RESULTS / "quantification" / "cohort_fear_profiles.csv"
        profiles_df = None
        if profiles_p.exists():
            try:
                profiles_df = pd.read_csv(profiles_p)
            except Exception:
                pass

        canvas2 = self._t5_cohort_canvas
        canvas2.show()
        canvas2.fig.clf()
        ax2 = canvas2.fig.add_subplot(111)
        if profiles_df is not None and not profiles_df.empty:
            cohort_col2 = next(
                (c for c in ("cohort_label", "cohort", "group", "treatment")
                 if c in profiles_df.columns), None
            )
            if cohort_col2 and "mean_fear_index" in profiles_df.columns:
                groups = profiles_df[cohort_col2].tolist()
                means = profiles_df["mean_fear_index"].tolist()
                ci_lo = profiles_df.get("fear_index_ci_lo", pd.Series([None] * len(groups))).tolist()
                ci_hi = profiles_df.get("fear_index_ci_hi", pd.Series([None] * len(groups))).tolist()
                for gi, (grp, mean) in enumerate(zip(groups, means)):
                    color = COHORT_COLORS[gi % len(COHORT_COLORS)]
                    yerr_lo = (mean - ci_lo[gi]) if ci_lo[gi] is not None and not pd.isna(ci_lo[gi]) else 0
                    yerr_hi = (ci_hi[gi] - mean) if ci_hi[gi] is not None and not pd.isna(ci_hi[gi]) else 0
                    ax2.bar(gi, mean, color=color, alpha=0.85,
                            yerr=[[yerr_lo], [yerr_hi]], capsize=4, error_kw={"elinewidth": 1.5})
                    # overlay individual animal points from fear_index.csv if cohort col present
                    if cohort_col and cohort_col in df.columns and str(grp) in df[cohort_col].astype(str).values:
                        sub = df[df[cohort_col].astype(str) == str(grp)]["fear_index"].dropna()
                        for v in sub:
                            ax2.scatter(gi, v, color="#555", s=18, zorder=5, alpha=0.6)
                ax2.set_xticks(range(len(groups)))
                ax2.set_xticklabels([str(g) for g in groups], fontsize=9)
                ax2.axhline(0, color="#999", linewidth=0.8)
                ax2.set_ylabel(f"Mean {self._metric_label}")
            else:
                ax2.text(0.5, 0.5, "Missing cohort_label or mean_fear_index columns",
                         ha="center", va="center", transform=ax2.transAxes, color="#999")
        else:
            ax2.text(0.5, 0.5, "Run fear_index.py to generate cohort profiles",
                     ha="center", va="center", transform=ax2.transAxes, color="#999")
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)
        ax2.yaxis.grid(True, color="#EEEEEE", zorder=0)
        canvas2.fig.tight_layout()
        canvas2.draw()

        # Cohort summary table (from cohort_fear_profiles.csv)
        if profiles_df is not None and not profiles_df.empty:
            cohort_col2 = next(
                (c for c in ("cohort_label", "cohort", "group", "treatment")
                 if c in profiles_df.columns), None
            )
            cols = [cohort_col2 or profiles_df.columns[0],
                    "n_animals", "mean_fear_index", "fear_index_ci_lo", "fear_index_ci_hi"]
            display_rows = profiles_df[[c for c in cols if c in profiles_df.columns]]
            self._t5_stats_table.setRowCount(len(display_rows))
            for ri, row in display_rows.reset_index(drop=True).iterrows():
                for ci, val in enumerate(row):
                    self._t5_stats_table.setItem(
                        ri, ci,
                        QTableWidgetItem(f"{val:.4f}" if isinstance(val, float) else str(val))
                    )

    # ─────────────────────── Tab 6 loader + helpers ──

    def _import_jess(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self, "Import Jess Data", "",
            "Data files (*.csv *.xlsx *.xls);;All files (*.*)"
        )
        if not p:
            return
        try:
            ext = os.path.splitext(p)[1].lower()
            df = pd.read_excel(p) if ext in (".xlsx", ".xls") else pd.read_csv(p)
        except Exception as e:
            QMessageBox.warning(self, "Import Error", str(e))
            return
        if "animal_id" not in df.columns:
            QMessageBox.warning(
                self, "Invalid Format",
                "File must have an 'animal_id' column."
            )
            return
        self._jess_df = df
        n_prot = len([c for c in df.columns if c != "animal_id"])
        self._t6_status_lbl.setText(f"{len(df)} animals, {n_prot} proteins loaded")
        self._t6_status_lbl.setStyleSheet("color:#1b5e20; font-size:11px;")

        summary = self._data.get("summary")
        if summary is not None and "animal_id" in summary.columns:
            behav = set(summary["animal_id"].astype(str).unique())
            jess = set(df["animal_id"].astype(str).unique())
            n_match = len(behav & jess)
            self._t6_match_lbl.setText(f"{n_match}/{len(behav)} animals matched")
            self._t6_run_btn.setEnabled(n_match >= 3)
        else:
            self._t6_run_btn.setEnabled(True)

        self.cfg["jess_csv_path"] = p
        try:
            from _utils import _save_cfg
            _save_cfg(self.cfg)
        except Exception:
            pass

    def _run_jess(self) -> None:
        if self._jess_df is None:
            return
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            jess_path = tf.name
        try:
            self._jess_df.to_csv(jess_path, index=False)
            self._run_command(["compare.py", "--jess", jess_path], self._t6_terminal)
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def _load_tab6(self) -> None:
        p = RESULTS / "quantification" / "jess_correlations.csv"
        if not p.exists():
            self._t6_placeholder.show()
            if self._t6_heatmap_canvas:
                self._t6_heatmap_canvas.hide()
            if self._t6_top_canvas:
                self._t6_top_canvas.hide()
            return

        self._t6_placeholder.hide()
        if not _MPL:
            return

        # Heatmap PNG
        heatmap_png = RESULTS / "quantification" / "correlation_heatmap.png"
        if heatmap_png.exists() and self._t6_heatmap_canvas:
            try:
                self._t6_heatmap_canvas.show()
                self._t6_heatmap_canvas.fig.clf()
                ax = self._t6_heatmap_canvas.fig.add_subplot(111)
                img = mpimg.imread(str(heatmap_png))
                ax.imshow(img)
                ax.axis("off")
                self._t6_heatmap_canvas.fig.tight_layout()
                self._t6_heatmap_canvas.draw()
            except Exception:
                pass

        # Top correlations bar chart
        try:
            corr = pd.read_csv(p)
        except Exception:
            return

        if self._t6_top_canvas and not corr.empty:
            self._t6_top_canvas.show()
            canvas = self._t6_top_canvas
            canvas.fig.clf()
            ax = canvas.fig.add_subplot(111)
            r_col = next(
                (c for c in ("pearson_r", "r", "correlation") if c in corr.columns), None
            )
            if r_col:
                top15 = corr.loc[corr[r_col].abs().nlargest(15).index]
                labels = []
                for _, row in top15.iterrows():
                    bv = row.get("behavioral_var", row.get("feature", "?"))
                    prot = row.get("jess_protein", row.get("protein", "?"))
                    r_val = row.get(r_col, 0)
                    labels.append(f"{bv} × {prot}  r={r_val:.2f}")
                vals = top15[r_col].values
                bar_colors = ["#E63946" if v >= 0 else "#4361EE" for v in vals]
                y = np.arange(len(labels))
                ax.barh(y, vals, color=bar_colors, alpha=0.85)
                ax.set_yticks(y)
                ax.set_yticklabels(labels, fontsize=7)
                ax.axvline(0, color="#999", linewidth=0.8)
                ax.set_xlabel("Pearson r")
            else:
                ax.text(0.5, 0.5, "No r column found in jess_correlations.csv",
                        ha="center", va="center", transform=ax.transAxes, color="#999")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            canvas.fig.tight_layout()
            canvas.draw()

    # ────────────────────────────────────── Tab 7: Event Alignment ──

    def _build_tab7(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(8)

        self._t7_terminal = TerminalBox()
        hdr = self._make_header(
            "Event Alignment", "Run Event Alignment",
            lambda: self._run_command(["compare.py", "--event-align"], self._t7_terminal),
            self._t7_terminal,
        )
        lay.addWidget(hdr)

        # Panel shown when no event column is configured
        self._t7_no_event_panel = QWidget()
        nc_lay = QVBoxLayout(self._t7_no_event_panel)
        nc_lay.addStretch()
        nc_msg = QLabel(
            "Configure an event column in Column Mapping "
            "to enable event alignment analysis."
        )
        nc_msg.setAlignment(Qt.AlignCenter)
        nc_msg.setWordWrap(True)
        nc_msg.setStyleSheet("color:#888; font-style:italic; padding:20px;")
        nc_lay.addWidget(nc_msg)
        open_mapping_btn = QPushButton("Open Column Mapping")
        open_mapping_btn.setFixedWidth(200)
        open_mapping_btn.clicked.connect(lambda: self._tab_list.setCurrentRow(0))
        nc_lay.addWidget(open_mapping_btn, alignment=Qt.AlignCenter)
        nc_lay.addStretch()
        lay.addWidget(self._t7_no_event_panel, stretch=1)

        # Data panel (scroll area with chart + table)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setSpacing(16)

        self._t7_placeholder = _placeholder(
            "No event alignment data found.\n\n"
            "Click 'Run Event Alignment' or run:\n"
            "python compare.py --event-align"
        )
        cl.addWidget(self._t7_placeholder)

        if _MPL:
            cl.addWidget(_section_title("Behavioral State Occupancy by Trial Outcome"))
            self._t7_chart_canvas = MplCanvas(figsize=(9, 3.5))
            self._t7_chart_canvas.setMinimumHeight(280)
            cl.addWidget(self._t7_chart_canvas)
        else:
            self._t7_chart_canvas = None
            cl.addWidget(_placeholder("Install matplotlib to view charts."))

        cl.addWidget(_section_title("Event Contrast Summary"))
        self._t7_contrast_table = QTableWidget(0, 5)
        self._t7_contrast_table.setHorizontalHeaderLabels(
            ["Label A", "Label B", "Contrast Magnitude",
             "Dominant State A", "Dominant State B"]
        )
        self._t7_contrast_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )
        self._t7_contrast_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._t7_contrast_table.setMaximumHeight(180)
        cl.addWidget(self._t7_contrast_table)

        cl.addStretch()
        scroll.setWidget(content)
        self._t7_data_panel = scroll
        lay.addWidget(self._t7_data_panel, stretch=1)

        return page

    def _load_tab7(self) -> None:
        event_col = self.cfg.get("column_map", {}).get("event", "")
        has_event_col = bool(event_col)
        self._t7_no_event_panel.setVisible(not has_event_col)
        self._t7_data_panel.setVisible(has_event_col)

        if not has_event_col:
            return

        profiles_p = RESULTS / "quantification" / "peri_event_profiles.csv"
        has_data = profiles_p.exists()
        self._t7_placeholder.setVisible(not has_data)
        if self._t7_chart_canvas:
            self._t7_chart_canvas.setVisible(has_data)

        if not has_data or not _MPL:
            return

        try:
            df = pd.read_csv(profiles_p)
            self._render_event_chart(df)
        except Exception:
            pass

        contrast_p = RESULTS / "quantification" / "event_contrast.csv"
        if contrast_p.exists():
            try:
                contrast = pd.read_csv(contrast_p)
                cols = ["label_A", "label_B", "contrast_magnitude",
                        "dominant_state_A", "dominant_state_B"]
                self._t7_contrast_table.setRowCount(len(contrast))
                for ri, row in contrast.reset_index(drop=True).iterrows():
                    for ci, col in enumerate(cols):
                        v = row.get(col, "")
                        self._t7_contrast_table.setItem(
                            ri, ci,
                            QTableWidgetItem(
                                f"{v:.4f}" if isinstance(v, float) else str(v)
                            )
                        )
            except Exception:
                pass

    def _render_event_chart(self, df: pd.DataFrame) -> None:
        canvas = self._t7_chart_canvas
        if not canvas:
            return

        canvas.fig.clf()
        ax = canvas.fig.add_subplot(111)

        state_cols = [
            c for c in df.columns
            if c.startswith("state_") and c.endswith("_frac")
        ]
        if not state_cols or df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, color="#999")
            canvas.fig.tight_layout()
            canvas.draw()
            return

        state_ids = [int(c.split("_")[1]) for c in state_cols]
        n_events = len(df)
        x = np.arange(len(state_ids))
        w = 0.8 / max(1, n_events)
        colors = _state_colors(max(n_events, 2))

        for ei, (_, row) in enumerate(df.iterrows()):
            fracs = [row[c] for c in state_cols]
            offset = (ei - n_events / 2 + 0.5) * w
            color = colors[min(ei, len(colors) - 1)]
            ax.bar(x + offset, fracs, w, color=color, alpha=0.85,
                   label=str(row["event_label"]))

        ax.set_xticks(x)
        ax.set_xticklabels([f"S{i}" for i in state_ids], fontsize=8)
        ax.set_ylabel("Mean Occupancy Fraction")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, color="#EEEEEE", zorder=0)
        canvas.fig.tight_layout()
        canvas.draw()

    # ───────────────────────────────────────── Label refresh ──

    def refresh_labels(self) -> None:
        """Re-read vocabulary labels from config and re-render the current tab."""
        import vieb_config as _vc
        self._cond_a = _vc.get_condition_a_label()
        self._cond_b = _vc.get_condition_b_label()
        self._metric_label = _vc.get_primary_metric_label()
        self._load_current_tab()

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
        # Pipeline output changed — all tabs need a fresh render on next visit.
        self._mark_all_dirty()
        self._load_current_tab()
