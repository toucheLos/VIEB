from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFrame,
    QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton,
    QScrollArea, QSpinBox, QStackedWidget, QTabWidget, QTableWidget,
    QTableWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from _utils import ROOT, RESULTS, CLIPS, _open_folder, _MPL

if _MPL:
    from _utils import mpl_cm, Figure, FigureCanvas, plt
    from _widgets import MplCanvas


class _CohortWorker(QThread):
    """Background worker that runs behavioral_fingerprint.py or plot_cohort.py."""
    log  = pyqtSignal(str)
    done = pyqtSignal(bool)

    def __init__(self, args: list[str]):
        super().__init__()
        self._args = args

    def run(self):
        import subprocess
        p = subprocess.Popen(
            [sys.executable, *self._args],
            cwd=str(ROOT),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        assert p.stdout is not None
        for line in p.stdout:
            self.log.emit(line)
        self.done.emit(p.wait() == 0)


class CohortAnalysisView(QWidget):
    """Three-tab Cohort Analysis panel: Fingerprints · Models · Deviation Scores."""
    navigate_to_animal = pyqtSignal(str)

    def __init__(self, cfg: dict):
        super().__init__()
        self._cfg     = cfg
        self._data    = {}
        self._worker  = None
        self._heatmap_animals: list[int] = []
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(10)

        hdr = QHBoxLayout()
        title = QLabel("Cohort Analysis")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        hdr.addWidget(title)
        hdr.addStretch()
        self._run_all_btn = QPushButton("Run All")
        self._run_all_btn.setToolTip(
            "Build fingerprints, compute deviation scores, fit forward + reverse models, "
            "generate plots.\nRequires a cohort file to be uploaded in Overview."
        )
        self._run_all_btn.clicked.connect(self._run_all)
        hdr.addWidget(self._run_all_btn)
        lay.addLayout(hdr)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(120)
        self._log.setStyleSheet("background:#111;color:#ddd;font-family:monospace;font-size:11px;")
        lay.addWidget(self._log)

        tabs = QTabWidget()
        tabs.addTab(self._build_tab_fingerprints(), "Fingerprints")
        tabs.addTab(self._build_tab_models(),       "Models")
        tabs.addTab(self._build_tab_deviation(),    "Deviation Scores")
        lay.addWidget(tabs)

    def _build_tab_fingerprints(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Color rows by:"))
        self._heat_color_combo = QComboBox()
        self._heat_color_combo.addItems(["cohort_label", "genotype", "treatment", "sex", "age_group"])
        self._heat_color_combo.currentIndexChanged.connect(self._refresh_heatmap)
        ctrl.addWidget(self._heat_color_combo)
        ctrl.addStretch()
        self._fp_run_btn = QPushButton("Build Fingerprints")
        self._fp_run_btn.clicked.connect(self._run_fingerprints)
        ctrl.addWidget(self._fp_run_btn)
        lay.addLayout(ctrl)

        self._heat_placeholder = QLabel(
            "No fingerprint data.\nClick 'Build Fingerprints' or 'Run All' to compute."
        )
        self._heat_placeholder.setAlignment(Qt.AlignCenter)
        self._heat_placeholder.setStyleSheet("color:#888;font-size:13px;")

        self._heat_scroll = QScrollArea()
        self._heat_scroll.setWidgetResizable(True)
        if _MPL:
            self._heat_canvas = MplCanvas(figsize=(14, 8))
            self._heat_canvas.mpl_connect("button_press_event", self._on_heatmap_click)
            self._heat_scroll.setWidget(self._heat_canvas)
        else:
            self._heat_canvas = None
            self._heat_scroll.setWidget(QLabel("Install matplotlib to view heatmap."))

        self._heat_stack = QStackedWidget()
        self._heat_stack.addWidget(self._heat_placeholder)
        self._heat_stack.addWidget(self._heat_scroll)
        lay.addWidget(self._heat_stack)
        return w

    def _build_tab_models(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)

        btn_row = QHBoxLayout()
        self._fwd_btn = QPushButton("Run Forward Model  (cohort → behavior)")
        self._fwd_btn.clicked.connect(self._run_forward)
        self._rev_btn = QPushButton("Run Reverse Model  (behavior → cohort)")
        self._rev_btn.clicked.connect(self._run_reverse)
        btn_row.addWidget(self._fwd_btn)
        btn_row.addWidget(self._rev_btn)
        btn_row.addStretch()

        self._rev_target_combo = QComboBox()
        self._rev_target_combo.addItems(["treatment", "genotype", "sex", "age_group"])
        btn_row.addWidget(QLabel("Target:"))
        btn_row.addWidget(self._rev_target_combo)
        lay.addLayout(btn_row)

        if _MPL:
            self._model_canvas = MplCanvas(figsize=(12, 5))
            lay.addWidget(self._model_canvas)
        else:
            self._model_canvas = None
            lay.addWidget(QLabel("Install matplotlib to view model charts."))

        self._model_placeholder = QLabel(
            "Run Forward or Reverse model to display results.\n"
            "Requires fingerprints + cohort file."
        )
        self._model_placeholder.setAlignment(Qt.AlignCenter)
        self._model_placeholder.setStyleSheet("color:#888;font-size:12px;")
        lay.addWidget(self._model_placeholder)
        return w

    def _build_tab_deviation(self) -> QWidget:
        w   = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(8, 8, 8, 8)

        btn_row = QHBoxLayout()
        self._dev_run_btn = QPushButton("Compute Deviation Scores")
        self._dev_run_btn.clicked.connect(self._run_deviation)
        btn_row.addWidget(self._dev_run_btn)
        self._dev_export_btn = QPushButton("Export CSV")
        self._dev_export_btn.clicked.connect(self._export_deviation)
        btn_row.addWidget(self._dev_export_btn)
        btn_row.addStretch()
        lay.addLayout(btn_row)

        self._dev_table = QTableWidget()
        self._dev_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._dev_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._dev_table.setSortingEnabled(True)
        self._dev_table.cellDoubleClicked.connect(self._on_dev_row_clicked)
        lay.addWidget(self._dev_table)
        return w

    def update_data(self, data: dict):
        self._data = data
        self._refresh_heatmap()
        self._refresh_deviation_table()
        self._refresh_model_view()

    def _cohort_file_arg(self) -> list[str]:
        p = self._cfg.get("cohort_csv_path", "")
        if p and os.path.exists(p):
            return ["--cohort", p]
        norm = str(ROOT / "cohort_normalized.csv")
        if os.path.exists(norm):
            return ["--cohort", norm]
        return []

    def _refresh_heatmap(self):
        fp_raw = self._data.get("fingerprints")
        if fp_raw is None or fp_raw.empty or not _MPL or self._heat_canvas is None:
            self._heat_stack.setCurrentIndex(0)
            return

        fp = fp_raw.copy()
        if "animal_id" in fp.columns:
            fp = fp.set_index("animal_id")
        fp.index = fp.index.astype(int)

        cohort = self._data.get("cohort")
        dev    = self._data.get("deviation_scores")
        color_by = self._heat_color_combo.currentText()

        sort_df = pd.DataFrame({"animal_id": fp.index.tolist()})
        if cohort is not None and "cohort_label" in cohort.columns:
            cmap = dict(zip(cohort["animal_id"].astype(int), cohort["cohort_label"]))
            sort_df["_cl"] = sort_df["animal_id"].map(cmap).fillna("?")
        else:
            sort_df["_cl"] = "?"
        if dev is not None and "composite_z" in dev.columns:
            zmap = dict(zip(dev["animal_id"].astype(int), dev["composite_z"]))
            sort_df["_z"] = sort_df["animal_id"].map(zmap).fillna(0.0)
        else:
            sort_df["_z"] = 0.0
        sort_df = sort_df.sort_values(["_cl", "_z"], ascending=[True, False])
        row_order = list(sort_df["animal_id"])
        self._heatmap_animals = row_order

        from behavioral_fingerprint import _sorted_feature_order
        feat_cols = _sorted_feature_order([c for c in fp.columns])
        feat_cols = [c for c in feat_cols if c in fp.columns]
        fp_sorted = fp.loc[row_order, feat_cols]
        fp_z = (fp_sorted - fp_sorted.mean()) / fp_sorted.std().clip(lower=1e-10)
        fp_z = fp_z.clip(-3, 3)

        canvas = self._heat_canvas
        canvas.fig.clf()
        ax = canvas.fig.add_subplot(111)
        ax.imshow(fp_z.values, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
                  interpolation="nearest")
        ax.set_yticks(range(len(row_order)))
        ax.set_yticklabels([str(a) for a in row_order], fontsize=6)
        ax.set_xticks([])
        ax.set_xlabel("Behavioral features (grouped)", fontsize=8)
        ax.set_title("Behavioral Fingerprint Heatmap — click a row to open Animal Explorer",
                     fontsize=9)
        canvas.fig.tight_layout()
        canvas.draw()
        self._heat_stack.setCurrentIndex(1)

    def _on_heatmap_click(self, event):
        if event.inaxes is None or event.ydata is None:
            return
        row = int(round(float(event.ydata)))
        if 0 <= row < len(self._heatmap_animals):
            self.navigate_to_animal.emit(str(self._heatmap_animals[row]))

    def _refresh_deviation_table(self):
        dev = self._data.get("deviation_scores")
        if dev is None or dev.empty:
            self._dev_table.setRowCount(0)
            return

        display_cols = ["animal_id", "cohort_label", "composite_z",
                        "context_A_z", "context_B_z", "most_deviant_feature"]
        display_cols = [c for c in display_cols if c in dev.columns]
        dev_sorted = dev.sort_values("composite_z", ascending=False).reset_index(drop=True)

        self._dev_table.setColumnCount(len(display_cols))
        self._dev_table.setHorizontalHeaderLabels(display_cols)
        self._dev_table.setRowCount(len(dev_sorted))
        self._dev_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

        RED_BG = QColor(255, 220, 220)
        RED_FG = QColor(180, 0, 0)

        for row_i, (_, row) in enumerate(dev_sorted.iterrows()):
            is_outlier = float(row.get("composite_z", 0)) > 2.0
            for col_i, col in enumerate(display_cols):
                val  = row[col]
                text = f"{val:.3f}" if isinstance(val, float) else str(val)
                item = QTableWidgetItem(text)
                if is_outlier:
                    item.setBackground(RED_BG)
                    item.setForeground(RED_FG)
                self._dev_table.setItem(row_i, col_i, item)

    def _on_dev_row_clicked(self, row, _col):
        id_col = next((i for i, h in enumerate(
            [self._dev_table.horizontalHeaderItem(c).text()
             for c in range(self._dev_table.columnCount())]
        ) if h == "animal_id"), None)
        if id_col is not None:
            item = self._dev_table.item(row, id_col)
            if item:
                self.navigate_to_animal.emit(item.text())

    def _export_deviation(self):
        dev = self._data.get("deviation_scores")
        if dev is None or dev.empty:
            QMessageBox.information(self, "Export", "No deviation scores loaded yet.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save deviation scores", "deviation_scores.csv", "CSV files (*.csv)"
        )
        if path:
            dev.to_csv(path, index=False)
            QMessageBox.information(self, "Export", f"Saved to {path}")

    def _refresh_model_view(self):
        rev = self._data.get("reverse_results")
        fwd_path = ROOT / "results" / "comparison" / "forward_model_weights.csv"

        has_fwd = fwd_path.exists()
        has_rev = rev is not None and bool(rev)

        if not has_fwd and not has_rev:
            if self._model_canvas:
                self._model_canvas.fig.clf()
                ax = self._model_canvas.fig.add_subplot(111)
                ax.text(0.5, 0.5,
                        "Run Forward or Reverse Model to see results.",
                        ha="center", va="center", transform=ax.transAxes, fontsize=11)
                ax.axis("off")
                self._model_canvas.draw()
            return

        if has_rev and self._model_canvas and _MPL:
            self._draw_feature_importance(rev)

    def _draw_feature_importance(self, rev: dict):
        canvas = self._model_canvas
        if canvas is None:
            return
        canvas.fig.clf()
        targets = list(rev.keys())
        n_t = len(targets)
        if n_t == 0:
            return

        axes = canvas.fig.subplots(1, n_t)
        if n_t == 1:
            axes = [axes]

        for ax, tgt in zip(axes, targets):
            res   = rev[tgt]
            top10 = res.get("top10_features", [])
            acc   = res.get("loo_accuracy", float("nan"))
            if not top10:
                continue
            feats = [d["feature"]    for d in top10]
            imps  = [d["importance"] for d in top10]
            colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(feats)))
            y_pos  = range(len(feats))
            ax.barh(list(y_pos), imps, color=colors[::-1])
            ax.set_yticks(list(y_pos))
            ax.set_yticklabels([f[:28] for f in feats[::-1]], fontsize=7)
            ax.set_xlabel("Importance", fontsize=8)
            ax.set_title(f"→ {tgt}  (LOO acc={acc:.2f})", fontsize=9)
            ax.invert_yaxis()

        canvas.fig.tight_layout()
        canvas.draw()
        self._model_placeholder.hide()

    def _start_worker(self, script: str, extra_args: list[str]):
        if self._worker and self._worker.isRunning():
            self._log.append("[busy] Another job is running — please wait.")
            return
        self._log.clear()
        args = [script] + extra_args + self._cohort_file_arg()
        self._worker = _CohortWorker(args)
        self._worker.log.connect(lambda t: self._log.insertPlainText(t))
        self._worker.done.connect(self._on_worker_done)
        self._worker.start()

    def _on_worker_done(self, ok: bool):
        self._log.append("\n[OK]" if ok else "\n[FAILED]")

    def _run_all(self):
        self._start_worker("behavioral_fingerprint.py", ["--all", "--plots"])

    def _run_fingerprints(self):
        self._start_worker("behavioral_fingerprint.py", ["--fingerprints"])

    def _run_deviation(self):
        self._start_worker("behavioral_fingerprint.py", ["--deviation"])

    def _run_forward(self):
        self._start_worker("behavioral_fingerprint.py", ["--forward"])

    def _run_reverse(self):
        tgt = self._rev_target_combo.currentText()
        self._start_worker("behavioral_fingerprint.py", ["--reverse", "--target", tgt])


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
        top.addWidget(QLabel("Context"))
        self._ctx_combo = QComboBox()
        self._ctx_combo.setMinimumWidth(70)
        self._ctx_combo.setToolTip(
            "Filter sessions by context.\n'All' shows every session for this animal."
        )
        self._ctx_combo.currentTextChanged.connect(lambda _: self._update(self._combo.currentText()))
        top.addWidget(self._ctx_combo)
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
        self._ctx_combo.blockSignals(True)
        prev_ctx = self._ctx_combo.currentText()
        self._ctx_combo.clear()
        self._ctx_combo.addItem("All")
        if "context" in summary.columns:
            for ctx in sorted(summary["context"].dropna().astype(str).unique()):
                self._ctx_combo.addItem(ctx)
        idx = self._ctx_combo.findText(prev_ctx)
        self._ctx_combo.setCurrentIndex(max(0, idx))
        self._ctx_combo.blockSignals(False)
        if animals:
            self._update(animals[0])

    def select_animal(self, animal_id: str):
        idx = self._combo.findText(str(animal_id))
        if idx >= 0:
            self._combo.setCurrentIndex(idx)

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

        ctx_filter = self._ctx_combo.currentText()
        if ctx_filter and ctx_filter != "All" and "context" in sub.columns:
            sub = sub[sub["context"].astype(str) == ctx_filter].copy()

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
                        plot_df = summary.copy()
                        if ctx_filter and ctx_filter != "All" and "context" in plot_df.columns:
                            plot_df = plot_df[plot_df["context"].astype(str) == ctx_filter]
                        colors = mpl_cm.tab20(np.linspace(0, 1, max(1, plot_df["animal_id"].nunique())))
                        for i, (aid, grp) in enumerate(plot_df.groupby("animal_id")):
                            if grp.empty:
                                continue
                            by_day = grp.groupby("day")[col].mean()
                            self._line.ax.plot(
                                by_day.index, by_day.values,
                                label=str(aid), color=colors[i % len(colors)],
                                linewidth=1.4, alpha=0.85,
                            )
                        ctx_suffix = f" — Context {ctx_filter}" if ctx_filter != "All" else ""
                        self._line.ax.set_title(f"State {sid} Occupancy - All Animals{ctx_suffix}")
                        self._line.ax.legend(fontsize=7, ncol=2)
                else:
                    by_day = sub.groupby("day")[state_cols].mean()
                    colors = mpl_cm.tab20(np.linspace(0, 1, max(1, len(state_cols))))
                    for i, col in enumerate(state_cols[:20]):
                        sid = int(col.split("_")[1])
                        self._line.ax.plot(by_day.index, by_day[col], label=f"S{sid}", color=colors[i], linewidth=1.4)
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
            if tt is not None and "animal_id" in tt.columns and ci:
                sub_t = tt[tt["animal_id"].astype(str) == str(animal_id)]
                if ctx_filter and ctx_filter != "All" and "context" in sub_t.columns:
                    sub_t = sub_t[sub_t["context"].astype(str) == ctx_filter]
                n = int(ci.get("n_clusters", 0))
                cols = [f"trans_{i}_{j}" for i in range(n) for j in range(n) if f"trans_{i}_{j}" in sub_t.columns]
                if cols:
                    mean_vals = sub_t[cols].mean().values
                    mat = mean_vals.reshape(n, n)
                    im = self._heat_canvas.ax.imshow(mat, cmap="Blues", aspect="auto")
                    self._heat_canvas.ax.set_title("Mean Transition Matrix")
                    self._heat_canvas.ax.set_xlabel("To")
                    self._heat_canvas.ax.set_ylabel("From")
                    self._heat_canvas.fig.colorbar(im, ax=self._heat_canvas.ax, fraction=0.046, pad=0.04)
            self._heat_canvas.fig.tight_layout()
            self._heat_canvas.draw()


class AdvancedView(QWidget):
    """Tabbed advanced analysis: Cohort Analysis, Animal Explorer."""
    navigate_to_animal = pyqtSignal(str)

    _DESC_TEXTS = {
        "Cohort Analysis": (
            "Behavioral fingerprints encode each animal as a vector of state occupancies, "
            "bout durations, and motif frequencies. The heatmap shows z-scored values — "
            "red = above cohort mean, blue = below. Use this to identify outlier animals "
            "and compare cohort profiles at the feature level."
        ),
        "Animal Explorer": (
            "Select an animal to view its state occupancy across days, session-by-session "
            "breakdown, discrimination trajectory, and transition matrix. Individual animal "
            "data is most useful for identifying outliers and understanding within-cohort variability."
        ),
    }

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or {}
        self._dismissed_descs: set = set()
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._tabs = QTabWidget()
        lay.addWidget(self._tabs)

        self._cav = CohortAnalysisView(self.cfg)
        self._cav.navigate_to_animal.connect(self.navigate_to_animal.emit)
        self._av = AnimalExplorerView()

        tab_defs = [
            ("Cohort Analysis", self._cav),
            ("Animal Explorer", self._av),
        ]
        for name, view in tab_defs:
            wrapper = QWidget()
            wl = QVBoxLayout(wrapper)
            wl.setContentsMargins(0, 0, 0, 0)
            wl.setSpacing(0)
            desc_key = f"advanced_desc_dismissed_{name.replace(' ', '_')}"
            if not self.cfg.get(desc_key, False):
                desc_frame = QFrame()
                desc_frame.setObjectName(f"descFrame_{name.replace(' ', '_')}")
                desc_frame.setStyleSheet(
                    "QFrame{background:#f5f5f5;border-bottom:1px solid #e0e0e0;}"
                )
                df_lay = QHBoxLayout(desc_frame)
                df_lay.setContentsMargins(16, 8, 8, 8)
                desc_lbl = QLabel(self._DESC_TEXTS.get(name, ""))
                desc_lbl.setWordWrap(True)
                desc_lbl.setStyleSheet(
                    "color:#555;font-style:italic;font-size:11px;"
                    "background:transparent;border:none;"
                )
                df_lay.addWidget(desc_lbl, stretch=1)
                x_btn = QPushButton("✕")
                x_btn.setFlat(True)
                x_btn.setFixedSize(22, 22)
                x_btn.setStyleSheet(
                    "QPushButton{color:#888;border:none;background:transparent;}"
                    "QPushButton:hover{color:#c62828;}"
                )
                def _make_dismiss(frame, key):
                    def _dismiss():
                        frame.hide()
                        self.cfg[key] = True
                        try:
                            from _utils import _save_cfg
                            _save_cfg(self.cfg)
                        except Exception:
                            pass
                    return _dismiss
                x_btn.clicked.connect(_make_dismiss(desc_frame, desc_key))
                df_lay.addWidget(x_btn)
                wl.addWidget(desc_frame)
            wl.addWidget(view, stretch=1)
            self._tabs.addTab(wrapper, name)

    def update_data(self, data):
        self._cav.update_data(data)
        self._av.update_data(data)

    def select_animal(self, animal_id: str):
        for i in range(self._tabs.count()):
            if self._tabs.tabText(i) == "Animal Explorer":
                self._tabs.setCurrentIndex(i)
                break
        self._av.select_animal(animal_id)
