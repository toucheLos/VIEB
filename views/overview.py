from __future__ import annotations
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFrame,
    QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton, QRadioButton,
    QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from _utils import ROOT, RESULTS, _fmt_ts, _MPL, _CV2
from _widgets import _Card, MplCanvas

if _MPL:
    from _utils import mpl_cm
    from matplotlib.patches import Patch as _MplPatch


class OverviewView(QWidget):
    export_requested = pyqtSignal()
    navigate_pipeline = pyqtSignal()
    navigate_settings = pyqtSignal()
    navigate_to_animal = pyqtSignal(str)
    cohort_path_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._banner_dismissed = False
        self._data = {}
        self._cohort_df = None
        self._dom_state = None
        self._distinctions_expanded = False

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        root_lay = QVBoxLayout(self)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.addWidget(scroll)

        content = QWidget()
        scroll.setWidget(content)
        lay = QVBoxLayout(content)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(16)

        # Getting Started banner
        self._banner = QFrame()
        self._banner.setObjectName("gsBanner")
        self._banner.setStyleSheet(
            "QFrame#gsBanner{background:#e8f0fe;border:1px solid #c8dafc;border-radius:8px;}"
        )
        bl = QVBoxLayout(self._banner)
        bl.setContentsMargins(16, 12, 16, 12)
        bl.setSpacing(6)
        hdr_row = QHBoxLayout()
        hdr_lbl = QLabel("  Getting Started")
        hdr_lbl.setFont(QFont("Arial", 11, QFont.Bold))
        hdr_lbl.setStyleSheet("color:#1a73e8;background:transparent;border:none;")
        hdr_row.addWidget(hdr_lbl)
        hdr_row.addStretch()
        dismiss_btn = QPushButton("✕")
        dismiss_btn.setFlat(True)
        dismiss_btn.setFixedSize(22, 22)
        dismiss_btn.setStyleSheet(
            "QPushButton{color:#555;border:none;background:transparent;font-size:11px;}"
            "QPushButton:hover{color:#c62828;}"
        )
        dismiss_btn.clicked.connect(self._dismiss_banner)
        hdr_row.addWidget(dismiss_btn)
        bl.addLayout(hdr_row)
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color:#c8dafc;background:transparent;border:none;border-top:1px solid #c8dafc;")
        bl.addWidget(sep)
        for num, title, desc in [
            ("Step 1", "Videos",        "Copy your .mp4 files into raw_videos/"),
            ("Step 2", "Settings",       "Set FPS and arena bounds (Settings)"),
            ("Step 3", "Pose estimation","Use a pretrained model or run DeepLabCut (Pipeline)"),
            ("Step 4", "Run Pipeline",   "Open the Pipeline tab and run all stages"),
            ("Step 5", "Explore",        "Browse States, Overview, Advanced views"),
        ]:
            lbl = QLabel(f"<b>{num}: {title}</b> — {desc}")
            lbl.setStyleSheet("background:transparent;border:none;color:#202020;")
            lbl.setWordWrap(True)
            bl.addWidget(lbl)
        acts_row = QHBoxLayout()
        go_settings = QPushButton("Open Settings")
        go_settings.setFixedHeight(28)
        go_settings.clicked.connect(self.navigate_settings.emit)
        go_pipeline = QPushButton("Open Pipeline")
        go_pipeline.setFixedHeight(28)
        go_pipeline.clicked.connect(self.navigate_pipeline.emit)
        acts_row.addWidget(go_settings)
        acts_row.addWidget(go_pipeline)
        acts_row.addStretch()
        bl.addLayout(acts_row)
        lay.addWidget(self._banner)

        # Previous Session banner
        self._prev_banner = QFrame()
        self._prev_banner.setObjectName("prevBanner")
        self._prev_banner.setStyleSheet(
            "QFrame#prevBanner{background:#fff3cd;border:1px solid #ffc107;border-radius:8px;}"
        )
        pb_lay = QHBoxLayout(self._prev_banner)
        pb_lay.setContentsMargins(16, 10, 16, 10)
        pb_lay.setSpacing(10)
        self._prev_lbl = QLabel("Previous analysis results available.")
        self._prev_lbl.setStyleSheet("background:transparent;border:none;color:#664d03;")
        pb_lay.addWidget(self._prev_lbl, stretch=1)
        self._prev_load_btn = QPushButton("Load Previous Session")
        self._prev_load_btn.setFixedHeight(28)
        pb_lay.addWidget(self._prev_load_btn)
        _pb_dismiss = QPushButton("✕")
        _pb_dismiss.setFlat(True)
        _pb_dismiss.setFixedSize(22, 22)
        _pb_dismiss.setStyleSheet(
            "QPushButton{color:#664d03;border:none;background:transparent;font-size:11px;}"
            "QPushButton:hover{color:#c62828;}"
        )
        _pb_dismiss.clicked.connect(self._prev_banner.hide)
        pb_lay.addWidget(_pb_dismiss)
        self._prev_banner.hide()
        lay.addWidget(self._prev_banner)

        # Title + Export
        top = QHBoxLayout()
        title_lbl = QLabel("Overview")
        title_lbl.setFont(QFont("Arial", 18, QFont.Bold))
        top.addWidget(title_lbl)
        top.addStretch()
        self._export_btn = QPushButton("Export Results")
        self._export_btn.clicked.connect(self.export_requested.emit)
        top.addWidget(self._export_btn)
        lay.addLayout(top)

        # 4 stat cards
        card_row = QHBoxLayout()
        self._c_videos = _Card("Total Videos")
        self._c_frames = _Card("Total Frames")
        self._c_states = _Card("States")
        self._c_noise  = _Card("Noise %")
        for c in (self._c_videos, self._c_frames, self._c_states, self._c_noise):
            card_row.addWidget(c)
        lay.addLayout(card_row)

        # All-states bar chart — header row with view selector + dominant toggle
        states_hdr = QHBoxLayout()
        states_title = QLabel("State Occupancy — All Videos")
        states_title.setFont(QFont("Arial", 12, QFont.Bold))
        states_hdr.addWidget(states_title)
        states_hdr.addSpacing(16)
        self._rb_all = QRadioButton("All Animals")
        self._rb_all.setChecked(True)
        self._rb_cohort = QRadioButton("By Cohort")
        self._rb_mouse = QRadioButton("By Mouse")
        _rb_grp = QButtonGroup(self)
        for _rb in (self._rb_all, self._rb_cohort, self._rb_mouse):
            _rb_grp.addButton(_rb)
            _rb.toggled.connect(self._on_view_mode_changed)
            states_hdr.addWidget(_rb)
        states_hdr.addStretch()
        self._incl_dominant_chk = QCheckBox("Include dominant state")
        self._incl_dominant_chk.setChecked(False)
        self._incl_dominant_chk.stateChanged.connect(lambda _: self._render_states_bar())
        states_hdr.addWidget(self._incl_dominant_chk)
        lay.addLayout(states_hdr)

        # Secondary control row — "Group by" dropdown, only visible in By Cohort mode
        bar_ctrl_row = QHBoxLayout()
        self._bar_cohort_lbl = QLabel("Group by:")
        bar_ctrl_row.addWidget(self._bar_cohort_lbl)
        self._bar_group_combo = QComboBox()
        self._bar_group_combo.addItems(["Treatment", "Age", "Sex", "Age × Treatment"])
        self._bar_group_combo.currentIndexChanged.connect(lambda _: self._render_states_bar())
        bar_ctrl_row.addWidget(self._bar_group_combo)
        self._bar_cohort_msg = QLabel("Upload cohort file in Settings to enable cohort view")
        self._bar_cohort_msg.setStyleSheet("color:#e65100;font-size:11px;")
        bar_ctrl_row.addWidget(self._bar_cohort_msg)
        bar_ctrl_row.addStretch()
        self._bar_cohort_lbl.hide()
        self._bar_group_combo.hide()
        self._bar_cohort_msg.hide()
        lay.addLayout(bar_ctrl_row)
        if _MPL:
            self._canvas_states = MplCanvas(figsize=(10, 2.5))
            lay.addWidget(self._canvas_states)
        else:
            self._canvas_states = None
            lay.addWidget(QLabel("Install matplotlib to view chart."))

        # Controls: cohort grouping + upload
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Group by:"))
        self._group_combo = QComboBox()
        self._group_combo.addItems([
            "Age × Treatment", "Sex", "Genotype", "Treatment",
            "Age Group", "Genotype × Treatment",
        ])
        self._group_combo.currentIndexChanged.connect(self._rerender)
        ctrl_row.addWidget(self._group_combo)
        ctrl_row.addSpacing(16)
        self._indiv_chk = QCheckBox("Show individual animals")
        self._indiv_chk.stateChanged.connect(self._rerender)
        ctrl_row.addWidget(self._indiv_chk)
        ctrl_row.addStretch()
        upload_btn = QPushButton("Upload cohort file")
        upload_btn.setFixedHeight(26)
        upload_btn.setToolTip(
            "Accepts Excel (.xlsx) or CSV files.\n"
            "Columns: animal_id, sex, age_group, genotype, treatment"
        )
        upload_btn.clicked.connect(self._upload_cohort_csv)
        ctrl_row.addWidget(upload_btn)
        self._cohort_status_lbl = QLabel("No cohort file loaded")
        self._cohort_status_lbl.setStyleSheet("color:#777;font-size:11px;")
        ctrl_row.addWidget(self._cohort_status_lbl)
        lay.addLayout(ctrl_row)

        # Middle: Fear Conditioning line chart
        mid_title = QLabel("Fear Conditioning — Context Discrimination")
        mid_title.setFont(QFont("Arial", 12, QFont.Bold))
        lay.addWidget(mid_title)
        if _MPL:
            self._canvas_disc = MplCanvas(figsize=(10, 3))
            lay.addWidget(self._canvas_disc)
        else:
            self._canvas_disc = None
            lay.addWidget(QLabel("Install matplotlib to view charts."))

        # Bottom: two side-by-side charts
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(16)

        left_col = QVBoxLayout()
        if _MPL:
            self._canvas_occ = MplCanvas(figsize=(5, 3))
            left_col.addWidget(self._canvas_occ)
        else:
            self._canvas_occ = None
            left_col.addWidget(QLabel("Install matplotlib."))
        left_col.addStretch()

        right_col = QVBoxLayout()
        if _MPL:
            self._canvas_fear = MplCanvas(figsize=(5, 3))
            right_col.addWidget(self._canvas_fear)
        else:
            self._canvas_fear = None
            right_col.addWidget(QLabel("Install matplotlib."))
        right_col.addStretch()

        bottom_row.addLayout(left_col, stretch=1)
        bottom_row.addLayout(right_col, stretch=1)
        lay.addLayout(bottom_row)

        # Notable Distinctions (collapsible)
        nd_header = QHBoxLayout()
        self._nd_toggle_btn = QPushButton("▶  Show Notable Distinctions")
        self._nd_toggle_btn.setFlat(True)
        self._nd_toggle_btn.setStyleSheet(
            "font-size:12px;color:#1a73e8;text-align:left;border:none;background:transparent;"
        )
        self._nd_toggle_btn.clicked.connect(self._toggle_distinctions)
        nd_header.addWidget(self._nd_toggle_btn)
        nd_header.addStretch()
        nd_header.addWidget(QLabel("Compare:"))
        self._nd_grp_a = QComboBox()
        self._nd_grp_a.setFixedWidth(130)
        self._nd_grp_b = QComboBox()
        self._nd_grp_b.setFixedWidth(130)
        nd_header.addWidget(self._nd_grp_a)
        nd_header.addWidget(QLabel("vs"))
        nd_header.addWidget(self._nd_grp_b)
        lay.addLayout(nd_header)

        self._nd_frame = QFrame()
        self._nd_frame.setVisible(False)
        nd_fl = QVBoxLayout(self._nd_frame)
        nd_fl.setContentsMargins(0, 0, 0, 0)
        self._nd_table = QTableWidget(0, 8)
        self._nd_table.setHorizontalHeaderLabels([
            "State", "Label", "Cohort A", "Cohort B",
            "Mean A", "Mean B", "Fold Change", "p-value",
        ])
        self._nd_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._nd_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._nd_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._nd_table.setMaximumHeight(260)
        nd_fl.addWidget(self._nd_table)
        self._nd_footnote = QLabel()
        self._nd_footnote.setStyleSheet("color:#777;font-style:italic;font-size:10px;")
        self._nd_footnote.setWordWrap(True)
        nd_fl.addWidget(self._nd_footnote)
        lay.addWidget(self._nd_frame)

        self._nd_grp_a.currentIndexChanged.connect(self._render_distinctions)
        self._nd_grp_b.currentIndexChanged.connect(self._render_distinctions)

        self._run_lbl = QLabel("Last run: -")
        self._run_lbl.setStyleSheet("color:#777;")
        lay.addWidget(self._run_lbl)
        lay.addStretch()

    def _upload_cohort_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Cohort File", "",
            "Cohort files (*.csv *.xlsx *.xls);;CSV files (*.csv);;Excel files (*.xlsx *.xls)"
        )
        if not path:
            return
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in (".xlsx", ".xls", ".xlsm"):
                from cohort_loader import load_cohort_excel
                cohort_df = load_cohort_excel(path)
                norm_path = str(ROOT / "cohort_normalized.csv")
                cohort_df.to_csv(norm_path, index=False)
                emit_path = norm_path
            else:
                cohort_df = pd.read_csv(path)
                emit_path = path
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Could not load cohort file:\n{e}")
            return
        if "animal_id" not in cohort_df.columns:
            QMessageBox.warning(self, "Invalid File",
                "File must have an 'animal_id' column.\n"
                "Excel: columns Animal, Treatment, Sex, Age, Genotype\n"
                "CSV:   columns animal_id, sex, age_group, genotype, treatment")
            return
        self._cohort_df = cohort_df
        self._cohort_status_lbl.setText(self._cohort_status_text())
        self._bar_cohort_msg.hide()
        self.cohort_path_changed.emit(emit_path)
        self._rerender()

    def _cohort_status_text(self):
        if self._cohort_df is None:
            return "No cohort file loaded"
        df = self._cohort_df
        if all(c in df.columns for c in ("genotype", "sex", "treatment")):
            return (f"{len(df)} animals | {df['genotype'].nunique()} genotypes | "
                    f"{df['treatment'].nunique()} treatments")
        return f"{len(df)} animals in cohort"

    @staticmethod
    def _age_bin(age_weeks):
        try:
            w = float(age_weeks)
        except (TypeError, ValueError):
            return "Unknown"
        if w < 8:
            return "Young (<8 wk)"
        elif w <= 12:
            return "Adult (8-12 wk)"
        return "Old (>12 wk)"

    def _merge_cohort(self, per_animal_df):
        pa = per_animal_df.copy()
        pa["animal_id"] = pa["animal_id"].astype(str)
        cohort = self._cohort_df
        if cohort is None:
            for col in ("sex", "age_group", "age_weeks", "genotype", "treatment", "cohort_label"):
                pa[col] = "Unknown"
            return pa
        cohort = cohort.copy()
        cohort["animal_id"] = cohort["animal_id"].astype(str)
        cols_to_merge = [c for c in ("sex", "age_group", "age_weeks", "genotype", "treatment", "cohort_label")
                         if c in cohort.columns]
        pa = pa.merge(cohort[["animal_id"] + cols_to_merge], on="animal_id", how="left")
        for col in ("sex", "genotype", "treatment", "age_group", "cohort_label"):
            if col in pa.columns:
                pa[col] = pa[col].fillna("Unknown")
        if "age_weeks" in pa.columns:
            pa["age_weeks"] = pd.to_numeric(pa["age_weeks"], errors="coerce")
        else:
            pa["age_weeks"] = float("nan")
        return pa

    def _get_group_col(self, pa):
        pa = pa.copy()
        grp = self._group_combo.currentText()
        n = len(pa)

        def _col(name):
            if name in pa.columns:
                return pa[name].fillna("Unknown").astype(str)
            return pd.Series(["Unknown"] * n, index=pa.index)

        if grp == "Sex":
            pa["_group"] = _col("sex")
        elif grp == "Genotype":
            pa["_group"] = _col("genotype")
        elif grp == "Treatment":
            pa["_group"] = _col("treatment")
        elif grp == "Age Group":
            if "age_group" in pa.columns and (pa["age_group"] != "Unknown").any():
                pa["_group"] = _col("age_group")
            elif "age_weeks" in pa.columns:
                pa["_group"] = pa["age_weeks"].apply(self._age_bin)
            else:
                pa["_group"] = "Unknown"
        elif grp == "Genotype × Treatment":
            pa["_group"] = _col("genotype") + " × " + _col("treatment")
        else:  # Age x Treatment (default)
            if "age_group" in pa.columns and (pa["age_group"] != "Unknown").any():
                age = _col("age_group")
            elif "age_weeks" in pa.columns:
                age = pa["age_weeks"].apply(self._age_bin).astype(str)
            else:
                age = pd.Series(["Unknown"] * n, index=pa.index)
            pa["_group"] = age + " × " + _col("treatment")
        return pa

    def _on_view_mode_changed(self):
        is_cohort = self._rb_cohort.isChecked()
        self._bar_cohort_lbl.setVisible(is_cohort)
        self._bar_group_combo.setVisible(is_cohort)
        self._bar_cohort_msg.setVisible(is_cohort and self._cohort_df is None)
        self._render_states_bar()

    def _get_bar_group_col(self, pa):
        pa = pa.copy()
        grp = self._bar_group_combo.currentText()
        n = len(pa)

        def _col(name):
            if name in pa.columns:
                return pa[name].fillna("Unknown").astype(str)
            return pd.Series(["Unknown"] * n, index=pa.index)

        if grp == "Treatment":
            pa["_group"] = _col("treatment")
        elif grp == "Age":
            if "age_group" in pa.columns and (pa["age_group"] != "Unknown").any():
                pa["_group"] = _col("age_group")
            elif "age_weeks" in pa.columns:
                pa["_group"] = pa["age_weeks"].apply(self._age_bin)
            else:
                pa["_group"] = "Unknown"
        elif grp == "Sex":
            pa["_group"] = _col("sex")
        else:  # Age × Treatment
            if "age_group" in pa.columns and (pa["age_group"] != "Unknown").any():
                age = _col("age_group")
            elif "age_weeks" in pa.columns:
                age = pa["age_weeks"].apply(self._age_bin).astype(str)
            else:
                age = pd.Series(["Unknown"] * n, index=pa.index)
            pa["_group"] = age + " × " + _col("treatment")
        return pa

    @staticmethod
    def _state_colors(state_ids):
        if not _MPL:
            return {}
        palette = mpl_cm.tab20(np.linspace(0, 1, max(len(state_ids), 1)))
        return {sid: tuple(palette[i]) for i, sid in enumerate(state_ids)}

    def _compute_dominant_state(self):
        ci = self._data.get("cluster_info")
        if ci and "dominant_state" in ci:
            return int(ci["dominant_state"])
        summary = self._data.get("summary")
        if summary is None:
            return None
        state_cols = [c for c in summary.columns if c.startswith("state_") and c.endswith("_frac")]
        if not state_cols:
            return None
        means = summary[state_cols].mean()
        return int(means.idxmax().split("_")[1])

    def _active_state_cols(self):
        ci = self._data.get("cluster_info")
        summary = self._data.get("summary")
        if ci is None or summary is None:
            return []
        n = int(ci.get("n_clusters", 0))
        cols = [f"state_{i}_frac" for i in range(n) if f"state_{i}_frac" in summary.columns]
        if self._dom_state is not None:
            cols = [c for c in cols if int(c.split("_")[1]) != self._dom_state]
        return cols

    def _disc_per_animal_day(self):
        summary = self._data.get("summary")
        if summary is None:
            return pd.DataFrame()
        ctx_col = next((c for c in ("context", "Context", "ctx") if c in summary.columns), None)
        if ctx_col is None or "day" not in summary.columns or "animal_id" not in summary.columns:
            return pd.DataFrame()
        ctx_vals = summary[ctx_col].dropna().astype(str).unique()
        ctx_A = [v for v in ctx_vals if v.upper().startswith("A")]
        ctx_B = [v for v in ctx_vals if v.upper().startswith("B")]
        if not ctx_A or not ctx_B:
            return pd.DataFrame()
        state_cols = self._active_state_cols()
        if not state_cols:
            return pd.DataFrame()
        a_mean = summary[summary[ctx_col].isin(ctx_A)][state_cols].mean()
        b_mean = summary[summary[ctx_col].isin(ctx_B)][state_cols].mean()
        diff = a_mean - b_mean
        fear_col = diff.idxmax() if not diff.empty else state_cols[0]
        rows = []
        for (animal, day), grp in summary.groupby(["animal_id", "day"]):
            a_rows = grp[grp[ctx_col].isin(ctx_A)]
            b_rows = grp[grp[ctx_col].isin(ctx_B)]
            if a_rows.empty or b_rows.empty:
                continue
            fa = float(a_rows[fear_col].mean())
            fb = float(b_rows[fear_col].mean())
            disc = (fa - fb) / (fa + fb + 1e-6)
            rows.append({"animal_id": str(animal), "day": int(day), "disc_ratio": disc})
        return pd.DataFrame(rows)

    def _rerender(self):
        self._render_states_bar()
        self._render_disc()
        self._render_occupancy()
        self._render_fear_enriched()
        if self._distinctions_expanded:
            self._render_distinctions()

    def _render_states_bar(self):
        canvas = self._canvas_states
        if not canvas or not _MPL:
            return
        canvas.fig.clf()
        ax = canvas.fig.add_subplot(111)
        summary = self._data.get("summary")
        ci = self._data.get("cluster_info")
        if summary is None or ci is None:
            ax.text(0.5, 0.5, "No data — run pipeline stages 2–8 first.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.axis("off")
            canvas.draw()
            return
        n = int(ci.get("n_clusters", 0))
        all_cols = [f"state_{i}_frac" for i in range(n) if f"state_{i}_frac" in summary.columns]
        if not all_cols:
            canvas.draw()
            return
        # Dominant state filtering controlled by checkbox
        if self._incl_dominant_chk.isChecked() or self._dom_state is None:
            state_cols = all_cols
        else:
            state_cols = [c for c in all_cols if int(c.split("_")[1]) != self._dom_state]
        if not state_cols:
            state_cols = all_cols  # fallback: never show empty chart
        # Update cohort message visibility
        self._bar_cohort_msg.setVisible(self._rb_cohort.isChecked() and self._cohort_df is None)
        if self._rb_cohort.isChecked():
            self._render_states_by_cohort(ax, summary, state_cols)
        elif self._rb_mouse.isChecked():
            self._render_states_by_mouse(ax, summary, state_cols)
        else:
            self._render_states_all_animals(ax, summary, state_cols)
        canvas.fig.tight_layout()
        canvas.draw()

    def _render_states_all_animals(self, ax, summary, state_cols):
        means = summary[state_cols].mean()
        state_ids = [int(c.split("_")[1]) for c in means.index]
        values = means.values
        colors = self._state_colors(state_ids)
        bar_colors = [colors.get(sid, (0.29, 0.56, 0.85, 1.0)) for sid in state_ids]
        ax.bar(state_ids, values, color=bar_colors, width=0.7)
        ax.set_xlabel("State ID", fontsize=9)
        ax.set_ylabel("Mean Fraction of Session", fontsize=9)
        ax.set_title("State Occupancy — All Animals", fontsize=10, fontweight="bold")
        if state_ids:
            ax.set_xticks(state_ids)
        ax.tick_params(labelsize=8)

    def _render_states_by_cohort(self, ax, summary, state_cols):
        if self._cohort_df is None:
            ax.text(0.5, 0.5,
                    "Upload cohort file in Settings to enable cohort view.",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="#e65100")
            ax.axis("off")
            return
        if "animal_id" not in summary.columns:
            ax.text(0.5, 0.5, "No animal_id column.", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")
            return
        per_animal = summary.groupby("animal_id")[state_cols].mean().reset_index()
        per_animal = self._merge_cohort(per_animal)
        per_animal = self._get_bar_group_col(per_animal)
        # Keep top 4 groups by N animals
        group_counts = per_animal["_group"].value_counts()
        top_groups = group_counts.nlargest(4).index.tolist()
        per_animal = per_animal[per_animal["_group"].isin(top_groups)]
        groups = sorted(top_groups)
        if not groups:
            ax.text(0.5, 0.5, "No cohort groups found.", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")
            return
        state_ids = [int(c.split("_")[1]) for c in state_cols]
        palette = mpl_cm.tab10(np.linspace(0, 1, max(len(groups), 1)))
        g_colors = {g: palette[i] for i, g in enumerate(groups)}
        bar_w = 0.7 / max(len(groups), 1)
        offsets = np.linspace(-(len(groups) - 1) * bar_w / 2,
                              (len(groups) - 1) * bar_w / 2, len(groups))
        for gi, g in enumerate(groups):
            gdf = per_animal[per_animal["_group"] == g]
            means = gdf[state_cols].mean().values
            sems = gdf[state_cols].sem().fillna(0).values
            x = np.array(state_ids, dtype=float) + offsets[gi]
            n_g = len(gdf)
            ax.bar(x, means, width=bar_w, label=f"{g} (N={n_g})",
                   color=g_colors[g], yerr=sems, capsize=3,
                   error_kw={"elinewidth": 0.8, "ecolor": "#444"})
        ax.set_xlabel("State ID", fontsize=9)
        ax.set_ylabel("Mean Fraction of Session", fontsize=9)
        ax.set_title("State Occupancy — By Cohort", fontsize=10, fontweight="bold")
        if state_ids:
            ax.set_xticks(state_ids)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)

    def _render_states_by_mouse(self, ax, summary, state_cols):
        if "animal_id" not in summary.columns:
            ax.text(0.5, 0.5, "No animal_id column.", ha="center", va="center",
                    transform=ax.transAxes)
            ax.axis("off")
            return
        per_animal = summary.groupby("animal_id")[state_cols].mean().reset_index()
        per_animal = self._merge_cohort(per_animal)
        if self._cohort_df is not None:
            per_animal = self._get_bar_group_col(per_animal)
            groups = sorted(per_animal["_group"].dropna().unique().tolist())
        else:
            per_animal["_group"] = "All Animals"
            groups = ["All Animals"]
        per_animal = per_animal.sort_values("animal_id")
        palette = mpl_cm.tab10(np.linspace(0, 1, max(len(groups), 1)))
        g_colors = {g: palette[i] for i, g in enumerate(groups)}
        state_ids = [int(c.split("_")[1]) for c in state_cols]
        n_animals = len(per_animal)
        bar_w = max(0.7 / max(n_animals, 1), 0.005)
        for ai, (_, row) in enumerate(per_animal.iterrows()):
            grp = row.get("_group", groups[0] if groups else "All Animals")
            color = g_colors.get(grp, (0.5, 0.5, 0.5, 1.0))
            offset = (ai - n_animals / 2 + 0.5) * bar_w
            vals = [float(row.get(c, 0)) for c in state_cols]
            ax.bar(
                np.array(state_ids, dtype=float) + offset,
                vals, width=bar_w,
                color=(*tuple(color[:3]), 0.7),
                linewidth=0,
            )
        ax.set_xlabel("State ID", fontsize=9)
        ax.set_ylabel("Mean Fraction of Session", fontsize=9)
        ax.set_title("State Occupancy — By Mouse", fontsize=10, fontweight="bold")
        if state_ids:
            ax.set_xticks(state_ids)
        ax.tick_params(labelsize=8)
        if len(groups) > 1:
            handles = [_MplPatch(color=tuple(g_colors[g][:3]), label=g) for g in groups]
            ax.legend(handles=handles, fontsize=7)

    def _render_disc(self):
        canvas = self._canvas_disc
        if not canvas or not _MPL:
            return
        canvas.fig.clf()
        ax = canvas.fig.add_subplot(111)
        disc_df = self._disc_per_animal_day()
        if disc_df.empty:
            ax.text(0.5, 0.5, "No data — run pipeline stages 1–8 first.",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.axis("off")
            canvas.draw()
            return
        per_animal = disc_df.groupby("animal_id")["disc_ratio"].mean().reset_index()
        per_animal = self._merge_cohort(per_animal)
        per_animal = self._get_group_col(per_animal)
        groups = sorted(per_animal["_group"].dropna().unique().tolist())
        palette = mpl_cm.tab10(np.linspace(0, 1, max(len(groups), 1)))
        g_colors = {g: palette[i] for i, g in enumerate(groups)}
        disc_df["animal_id"] = disc_df["animal_id"].astype(str)
        merged = disc_df.merge(
            per_animal[["animal_id", "_group"]].drop_duplicates(), on="animal_id", how="left"
        )
        merged["_group"] = merged["_group"].fillna("Unknown")
        days = sorted(merged["day"].unique())
        if self._indiv_chk.isChecked():
            for _aid, adf in merged.groupby("animal_id"):
                grp = adf["_group"].iloc[0]
                c = g_colors.get(grp, (0.5, 0.5, 0.5, 1.0))
                ax.plot(adf.sort_values("day")["day"],
                        adf.sort_values("day")["disc_ratio"],
                        color=(*c[:3], 0.25), linewidth=0.8, zorder=1)
        for g in groups:
            gdf = merged[merged["_group"] == g]
            day_mean = gdf.groupby("day")["disc_ratio"].mean()
            ax.plot(day_mean.index, day_mean.values,
                    color=g_colors[g], linewidth=2.5, marker="o",
                    markersize=4, label=g, zorder=2)
        ax.axhline(0.0, color="#999", linewidth=1.0, linestyle="-")
        ax.axhline(0.2, color="#bbb", linewidth=1.0, linestyle="--")
        ax.text(0.99, 0.03, "chance", color="#999", fontsize=7,
                ha="right", va="bottom", transform=ax.transAxes)
        ax.text(0.99, 0.24, "learning criterion", color="#bbb",
                fontsize=7, ha="right", va="bottom", transform=ax.transAxes)
        ax.set_xlabel("Day", fontsize=9)
        ax.set_ylabel("Discrimination Ratio", fontsize=9)
        ax.set_ylim(-1.05, 1.05)
        if days:
            ax.set_xlim(min(days) - 0.3, max(days) + 0.3)
        if groups:
            ax.legend(fontsize=8, loc="upper left")
        canvas.fig.tight_layout()
        canvas.draw()

    def _render_occupancy(self):
        canvas = self._canvas_occ
        if not canvas or not _MPL:
            return
        canvas.fig.clf()
        ax = canvas.fig.add_subplot(111)
        summary = self._data.get("summary")
        ci = self._data.get("cluster_info")
        if summary is None or ci is None or "animal_id" not in summary.columns:
            ax.text(0.5, 0.5, "No data.", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            canvas.draw()
            return
        state_cols = self._active_state_cols()
        if not state_cols:
            canvas.draw()
            return
        per_animal = summary.groupby("animal_id")[state_cols].mean().reset_index()
        per_animal = self._merge_cohort(per_animal)
        per_animal = self._get_group_col(per_animal)
        groups = sorted(per_animal["_group"].dropna().unique().tolist())
        if not groups:
            canvas.draw()
            return
        state_ids = [int(c.split("_")[1]) for c in state_cols]
        palette = mpl_cm.tab10(np.linspace(0, 1, max(len(groups), 1)))
        bar_w = 0.8 / max(len(groups), 1)
        offsets = np.linspace(-(len(groups) - 1) * bar_w / 2,
                              (len(groups) - 1) * bar_w / 2, len(groups))
        for gi, g in enumerate(groups):
            gdf = per_animal[per_animal["_group"] == g]
            means = gdf[state_cols].mean().values
            sems = gdf[state_cols].sem().fillna(0).values
            x = np.array(state_ids, dtype=float) + offsets[gi]
            ax.bar(x, means, width=bar_w, label=g,
                   color=palette[gi], yerr=sems, capsize=2,
                   error_kw={"elinewidth": 0.8, "ecolor": "#444"})
        ax.set_xlabel("State ID", fontsize=8)
        ax.set_ylabel("Mean Fraction", fontsize=8)
        ax.set_title("State Occupancy by Cohort", fontsize=9, fontweight="bold")
        if state_ids:
            ax.set_xticks(state_ids)
        ax.tick_params(labelsize=7)
        if groups:
            ax.legend(fontsize=7)
        canvas.fig.tight_layout()
        canvas.draw()

    def _render_fear_enriched(self):
        canvas = self._canvas_fear
        if not canvas or not _MPL:
            return
        canvas.fig.clf()
        ax = canvas.fig.add_subplot(111)
        summary = self._data.get("summary")
        ci = self._data.get("cluster_info")
        if summary is None or ci is None:
            ax.text(0.5, 0.5, "No data.", ha="center", va="center", transform=ax.transAxes)
            ax.axis("off")
            canvas.draw()
            return
        state_cols = self._active_state_cols()
        if not state_cols:
            canvas.draw()
            return
        ctx_col = next((c for c in ("context", "Context", "ctx") if c in summary.columns), None)
        if ctx_col is None:
            ax.text(0.5, 0.5, "No context column in data.", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.axis("off")
            canvas.draw()
            return
        ctx_vals = summary[ctx_col].dropna().astype(str).unique()
        ctx_A = [v for v in ctx_vals if v.upper().startswith("A")]
        ctx_B = [v for v in ctx_vals if v.upper().startswith("B")]
        if not ctx_A or not ctx_B:
            ax.text(0.5, 0.5, "Contexts A and B not found.", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9)
            ax.axis("off")
            canvas.draw()
            return
        a_mean = summary[summary[ctx_col].isin(ctx_A)][state_cols].mean()
        b_mean = summary[summary[ctx_col].isin(ctx_B)][state_cols].mean()
        diffs = (a_mean - b_mean).sort_values(ascending=False)
        state_ids = [int(c.split("_")[1]) for c in diffs.index]
        vals = diffs.values
        bar_colors = ["#c62828" if v >= 0 else "#1565c0" for v in vals]
        ax.bar(state_ids, vals, color=bar_colors)
        ax.axhline(0, color="#555", linewidth=0.8)
        ax.set_xlabel("State ID", fontsize=8)
        ax.set_ylabel("A − B fraction diff", fontsize=8)
        ax.set_title("Fear-Enriched States", fontsize=9, fontweight="bold")
        if state_ids:
            ax.set_xticks(state_ids)
        ax.tick_params(labelsize=7)
        ax.annotate("Positive = more in Context A (fear)", xy=(0.02, 0.96),
                    xycoords="axes fraction", fontsize=7, color="#666", va="top")
        canvas.fig.tight_layout()
        canvas.draw()

    def _toggle_distinctions(self):
        self._distinctions_expanded = not self._distinctions_expanded
        self._nd_frame.setVisible(self._distinctions_expanded)
        self._nd_toggle_btn.setText(
            "▼  Hide Notable Distinctions"
            if self._distinctions_expanded
            else "▶  Show Notable Distinctions"
        )
        if self._distinctions_expanded:
            self._render_distinctions()

    def _update_group_dropdowns(self, groups):
        for combo in (self._nd_grp_a, self._nd_grp_b):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(groups)
            idx = combo.findText(current)
            combo.setCurrentIndex(max(idx, 0))
            combo.blockSignals(False)
        if len(groups) >= 2:
            self._nd_grp_a.setCurrentIndex(0)
            self._nd_grp_b.setCurrentIndex(min(1, len(groups) - 1))

    def _render_distinctions(self):
        self._nd_table.setRowCount(0)
        summary = self._data.get("summary")
        ci = self._data.get("cluster_info")
        state_summary = self._data.get("state_summary")
        if summary is None or ci is None:
            self._nd_footnote.setText("No data available.")
            return
        state_cols = self._active_state_cols()
        if not state_cols or "animal_id" not in summary.columns:
            return
        per_animal = summary.groupby("animal_id")[state_cols].mean().reset_index()
        per_animal = self._merge_cohort(per_animal)
        per_animal = self._get_group_col(per_animal)
        grp_a_name = self._nd_grp_a.currentText()
        grp_b_name = self._nd_grp_b.currentText()
        if not grp_a_name or not grp_b_name or grp_a_name == grp_b_name:
            return
        a_df = per_animal[per_animal["_group"] == grp_a_name]
        b_df = per_animal[per_animal["_group"] == grp_b_name]
        n_a, n_b = len(a_df), len(b_df)
        if n_a < 2 or n_b < 2:
            self._nd_footnote.setText(f"Need ≥2 animals per group (N={n_a} vs N={n_b}).")
            return
        try:
            from scipy.stats import mannwhitneyu
            _has_scipy = True
        except ImportError:
            _has_scipy = False
        heuristic_labels = {}
        if (state_summary is not None and "state_id" in state_summary.columns
                and "heuristic_label" in state_summary.columns):
            for _, row in state_summary.iterrows():
                heuristic_labels[int(row["state_id"])] = str(row.get("heuristic_label", ""))
        rows = []
        for col in state_cols:
            sid = int(col.split("_")[1])
            a_vals = a_df[col].dropna().values
            b_vals = b_df[col].dropna().values
            if len(a_vals) < 2 or len(b_vals) < 2:
                continue
            mean_a = float(a_vals.mean())
            mean_b = float(b_vals.mean())
            fold = mean_a / (mean_b + 1e-9)
            if _has_scipy:
                _, pval = mannwhitneyu(a_vals, b_vals, alternative="two-sided")
            else:
                pval = float("nan")
            if not _has_scipy or pval < 0.05:
                rows.append({
                    "state": sid, "label": heuristic_labels.get(sid, ""),
                    "mean_a": mean_a, "mean_b": mean_b,
                    "fold": fold, "p": pval,
                })
        rows.sort(key=lambda r: r["fold"], reverse=True)
        self._nd_table.setRowCount(len(rows))
        for ri, r in enumerate(rows):
            self._nd_table.setItem(ri, 0, QTableWidgetItem(str(r["state"])))
            self._nd_table.setItem(ri, 1, QTableWidgetItem(r["label"]))
            self._nd_table.setItem(ri, 2, QTableWidgetItem(grp_a_name))
            self._nd_table.setItem(ri, 3, QTableWidgetItem(grp_b_name))
            self._nd_table.setItem(ri, 4, QTableWidgetItem(f"{r['mean_a']:.3f}"))
            self._nd_table.setItem(ri, 5, QTableWidgetItem(f"{r['mean_b']:.3f}"))
            self._nd_table.setItem(ri, 6, QTableWidgetItem(f"{r['fold']:.2f}×"))
            p_str = f"{r['p']:.3f}" if not math.isnan(r["p"]) else "n/a"
            self._nd_table.setItem(ri, 7, QTableWidgetItem(p_str))
        foot = (f"Uncorrected p-values.  N={n_a} vs N={n_b}.\n"
                "For corrected analysis see Advanced → Cohort Analysis.")
        if not _has_scipy:
            foot += "  (Install scipy for p-values.)"
        self._nd_footnote.setText(foot)

    def _dismiss_banner(self):
        self._banner_dismissed = True
        self._banner.hide()

    def _update_banner(self, data):
        if self._banner_dismissed:
            return
        self._banner.setVisible(data.get("summary") is None)

    def show_load_banner(self, data, on_load, has_results: bool):
        if not self._banner_dismissed:
            self._banner.setVisible(not has_results)
        if not has_results:
            self._prev_banner.hide()
            return
        p = RESULTS / "comparison" / "summary_table.csv"
        date_str = _fmt_ts(p.stat().st_mtime) if p.exists() else "a previous session"
        self._prev_lbl.setText(f"Results from {date_str} available on disk.")
        try:
            self._prev_load_btn.clicked.disconnect()
        except (RuntimeError, TypeError):
            pass
        self._prev_load_btn.clicked.connect(lambda: (on_load(), self._prev_banner.hide()))
        self._prev_banner.show()

    def update_data(self, data):
        self._prev_banner.hide()
        self._update_banner(data)
        self._data = data
        cohort = data.get("cohort")
        if cohort is not None:
            self._cohort_df = cohort
            self._bar_cohort_msg.setVisible(self._rb_cohort.isChecked() and self._cohort_df is None)
        summary = data.get("summary")
        ci = data.get("cluster_info")
        fi = data.get("feature_index")
        if summary is None:
            self._c_videos.set("-")
            self._cohort_status_lbl.setText(self._cohort_status_text())
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
            noise = (1 - float(summary[state_cols].sum(axis=1).mean())) * 100
            self._c_noise.set(f"{noise:.1f}%")
        self._dom_state = self._compute_dominant_state()
        self._cohort_status_lbl.setText(self._cohort_status_text())
        active_cols = self._active_state_cols()
        if "animal_id" in summary.columns and active_cols:
            pa = summary.groupby("animal_id")[active_cols].mean().reset_index()
            pa = self._merge_cohort(pa)
            pa = self._get_group_col(pa)
            groups = sorted(pa["_group"].dropna().unique().tolist())
            self._update_group_dropdowns(groups)
        p = RESULTS / "comparison" / "summary_table.csv"
        if p.exists():
            self._run_lbl.setText(f"Last run: {_fmt_ts(p.stat().st_mtime)}")
        self._rerender()
