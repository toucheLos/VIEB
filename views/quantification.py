from __future__ import annotations
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QAbstractItemView, QFileDialog, QHBoxLayout, QHeaderView,
    QLabel, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem,
    QTabWidget, QVBoxLayout, QWidget,
)

from _utils import RESULTS, _MPL

if _MPL:
    from _utils import mpimg
    from _widgets import MplCanvas


class QuantificationView(QWidget):
    """Two-tab view: Master Table, Import Jess."""
    pipeline_run_requested = pyqtSignal(str)

    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or {}
        self._data = {}
        self._jess_df = None
        self._corr_df = None
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        tabs = QTabWidget()
        lay.addWidget(tabs)

        # ── Tab 1: Master Table ───────────────────────────────────────────────
        t1 = QWidget()
        t1l = QVBoxLayout(t1)
        t1l.setContentsMargins(20, 16, 20, 16)
        t1l.setSpacing(10)

        t1_hdr = QHBoxLayout()
        t1_title = QLabel("Master Quantification Table")
        t1_title.setFont(QFont("Arial", 14, QFont.Bold))
        t1_hdr.addWidget(t1_title)
        t1_hdr.addStretch()
        self._quant_run_btn = QPushButton("Run Quantification")
        self._quant_run_btn.setFixedHeight(28)
        self._quant_run_btn.clicked.connect(lambda: self.pipeline_run_requested.emit("--quantify"))
        t1_hdr.addWidget(self._quant_run_btn)
        self._quant_export_btn = QPushButton("Export")
        self._quant_export_btn.setFixedHeight(28)
        self._quant_export_btn.clicked.connect(self._export_master_table)
        t1_hdr.addWidget(self._quant_export_btn)
        t1l.addLayout(t1_hdr)

        self._quant_no_data_lbl = QLabel(
            "No quantification data found.\n"
            "Click 'Run Quantification' to generate results/quantification/master_table.csv"
        )
        self._quant_no_data_lbl.setAlignment(Qt.AlignCenter)
        self._quant_no_data_lbl.setStyleSheet("color:#888;font-style:italic;")
        t1l.addWidget(self._quant_no_data_lbl)

        self._master_table = QTableWidget(0, 0)
        self._master_table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self._master_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._master_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._master_table.setSortingEnabled(True)
        self._master_table.hide()
        t1l.addWidget(self._master_table, stretch=1)
        tabs.addTab(t1, "Master Table")

        # ── Tab 2: Import Jess ────────────────────────────────────────────────
        t2 = QWidget()
        t2l = QVBoxLayout(t2)
        t2l.setContentsMargins(20, 16, 20, 16)
        t2l.setSpacing(12)

        jess_hdr = QHBoxLayout()
        jess_title = QLabel("Import Jess Protein Data")
        jess_title.setFont(QFont("Arial", 14, QFont.Bold))
        jess_hdr.addWidget(jess_title)
        jess_hdr.addStretch()
        t2l.addLayout(jess_hdr)

        jess_desc = QLabel(
            "Expected format: CSV or Excel with columns: animal_id, protein_1, protein_2, …\n"
            "Each protein column should be a numeric measurement. "
            "Rows are matched to behavioral data by animal_id."
        )
        jess_desc.setWordWrap(True)
        jess_desc.setStyleSheet("color:#555;font-style:italic;")
        t2l.addWidget(jess_desc)

        import_row = QHBoxLayout()
        self._jess_import_btn = QPushButton("Import Jess Data (.csv / .xlsx)")
        self._jess_import_btn.setFixedHeight(30)
        self._jess_import_btn.clicked.connect(self._import_jess)
        import_row.addWidget(self._jess_import_btn)
        self._jess_status_lbl = QLabel("No file loaded")
        self._jess_status_lbl.setStyleSheet("color:#777;font-size:11px;")
        import_row.addWidget(self._jess_status_lbl, stretch=1)
        t2l.addLayout(import_row)

        self._jess_match_lbl = QLabel("")
        self._jess_match_lbl.setStyleSheet("color:#1b5e20;font-size:11px;")
        t2l.addWidget(self._jess_match_lbl)

        corr_row = QHBoxLayout()
        self._jess_run_btn = QPushButton("Run Correlation")
        self._jess_run_btn.setFixedHeight(28)
        self._jess_run_btn.setEnabled(False)
        self._jess_run_btn.clicked.connect(self._run_jess_corr)
        corr_row.addWidget(self._jess_run_btn)
        self._jess_export_btn = QPushButton("Export Correlations")
        self._jess_export_btn.setFixedHeight(28)
        self._jess_export_btn.setEnabled(False)
        self._jess_export_btn.clicked.connect(self._export_jess_corr)
        corr_row.addWidget(self._jess_export_btn)
        corr_row.addStretch()
        t2l.addLayout(corr_row)

        if _MPL:
            self._jess_canvas = MplCanvas(figsize=(10, 4))
            t2l.addWidget(self._jess_canvas, stretch=1)
        else:
            self._jess_canvas = None
            t2l.addWidget(QLabel("Install matplotlib to view heatmap."))

        jess_tbl_lbl = QLabel("Top 20 Correlations")
        jess_tbl_lbl.setFont(QFont("Arial", 10, QFont.Bold))
        t2l.addWidget(jess_tbl_lbl)
        self._jess_table = QTableWidget(0, 7)
        self._jess_table.setHorizontalHeaderLabels([
            "Behavioral Var", "Protein", "Pearson r", "Pearson p",
            "Spearman rho", "Spearman p", "N pairs",
        ])
        self._jess_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._jess_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._jess_table.setSortingEnabled(True)
        self._jess_table.setMaximumHeight(220)
        t2l.addWidget(self._jess_table)

        self._jess_note_lbl = QLabel(
            "Statistical note: Pearson r assumes bivariate normality; "
            "Spearman rho is rank-based and more robust to outliers. "
            "Multiple-comparison correction (FDR) is recommended for large panels."
        )
        self._jess_note_lbl.setWordWrap(True)
        self._jess_note_lbl.setStyleSheet("color:#666;font-size:10px;font-style:italic;")
        t2l.addWidget(self._jess_note_lbl)

        tabs.addTab(t2, "Import Jess")

    # ── Data update ────────────────────────────────────────────────────────────

    def update_data(self, data):
        self._data = data
        self._refresh_master_table()

    def _refresh_master_table(self):
        quant_path = RESULTS / "quantification" / "master_table.csv"
        if quant_path.exists():
            try:
                df = pd.read_csv(quant_path)
                self._quant_no_data_lbl.hide()
                self._master_table.show()
                self._master_table.setRowCount(len(df))
                self._master_table.setColumnCount(len(df.columns))
                self._master_table.setHorizontalHeaderLabels(list(df.columns))
                for ri, row in df.iterrows():
                    for ci, val in enumerate(row):
                        self._master_table.setItem(ri, ci, QTableWidgetItem(str(val)))
                return
            except Exception:
                pass
        self._master_table.hide()
        self._quant_no_data_lbl.show()

    def _export_master_table(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Master Table", "", "CSV files (*.csv)")
        if not path:
            return
        quant_path = RESULTS / "quantification" / "master_table.csv"
        if quant_path.exists():
            try:
                import shutil as _shutil
                _shutil.copy(str(quant_path), path)
                QMessageBox.information(self, "Exported", f"Saved to {path}")
            except Exception as e:
                QMessageBox.warning(self, "Export Error", str(e))
        else:
            QMessageBox.warning(self, "No Data", "No master table to export.")

    def _import_jess(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Jess Data", "",
            "Data files (*.csv *.xlsx *.xls);;CSV files (*.csv);;Excel files (*.xlsx *.xls)"
        )
        if not path:
            return
        try:
            ext = os.path.splitext(path)[1].lower()
            if ext in (".xlsx", ".xls"):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.warning(self, "Load Error", f"Could not load file:\n{e}")
            return
        if "animal_id" not in df.columns:
            QMessageBox.warning(self, "Invalid Format",
                "File must have an 'animal_id' column.\n"
                "Other columns should be protein measurements.")
            return
        self._jess_df = df
        n_prot = len([c for c in df.columns if c != "animal_id"])
        self._jess_status_lbl.setText(f"{len(df)} animals, {n_prot} proteins loaded")
        self._jess_status_lbl.setStyleSheet("color:#1b5e20;font-size:11px;")
        summary = self._data.get("summary")
        if summary is not None and "animal_id" in summary.columns:
            behav_ids = set(summary["animal_id"].astype(str).unique())
            jess_ids = set(df["animal_id"].astype(str).unique())
            n_match = len(behav_ids & jess_ids)
            self._jess_match_lbl.setText(f"{n_match}/{len(behav_ids)} animals matched")
            self._jess_run_btn.setEnabled(n_match >= 3)
        else:
            self._jess_run_btn.setEnabled(True)

    def _run_jess_corr(self):
        if self._jess_df is None:
            return
        summary = self._data.get("summary")
        if summary is None:
            QMessageBox.warning(self, "No Data", "Run pipeline first to generate behavioral data.")
            return
        try:
            from quantify import run_jess_correlation
        except ImportError:
            QMessageBox.warning(self, "Missing Module",
                "quantify.py not found. Please ensure it exists in the project directory.")
            return
        master_path = RESULTS / "quantification" / "master_table.csv"
        if not master_path.exists():
            QMessageBox.warning(self, "No Master Table",
                "Generate master_table.csv first (Run Quantification tab).")
            return
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tf:
            jess_path = tf.name
        self._jess_df.to_csv(jess_path, index=False)
        out_dir = str(RESULTS / "quantification")
        try:
            result = run_jess_correlation(str(master_path), jess_path, out_dir)
            self._corr_df = result
            self._jess_export_btn.setEnabled(True)
            self._render_jess_results(result)
        except Exception as e:
            QMessageBox.warning(self, "Correlation Error", str(e))
        finally:
            try:
                os.unlink(jess_path)
            except Exception:
                pass

    def _render_jess_results(self, result):
        if result is None or result.empty:
            return
        canvas = self._jess_canvas
        heatmap_path = RESULTS / "quantification" / "correlation_heatmap.png"
        if canvas and _MPL and heatmap_path.exists():
            canvas.fig.clf()
            ax = canvas.fig.add_subplot(111)
            try:
                img = mpimg.imread(str(heatmap_path))
                ax.imshow(img)
                ax.axis("off")
            except Exception:
                pass
            canvas.draw()
        top20 = result.nlargest(min(20, len(result)), "pearson_r")
        self._jess_table.setRowCount(len(top20))
        for ri, row in top20.reset_index(drop=True).iterrows():
            self._jess_table.setItem(ri, 0, QTableWidgetItem(str(row.get("behavioral_var", ""))))
            self._jess_table.setItem(ri, 1, QTableWidgetItem(str(row.get("jess_protein", ""))))
            self._jess_table.setItem(ri, 2, QTableWidgetItem(f"{row.get('pearson_r', 0):.3f}"))
            self._jess_table.setItem(ri, 3, QTableWidgetItem(f"{row.get('pearson_p', 1):.4f}"))
            self._jess_table.setItem(ri, 4, QTableWidgetItem(f"{row.get('spearman_rho', 0):.3f}"))
            self._jess_table.setItem(ri, 5, QTableWidgetItem(f"{row.get('spearman_p', 1):.4f}"))
            self._jess_table.setItem(ri, 6, QTableWidgetItem(str(row.get("n_pairs", ""))))

    def _export_jess_corr(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Correlations", "", "CSV files (*.csv)")
        if not path or self._corr_df is None:
            return
        try:
            self._corr_df.to_csv(path, index=False)
            QMessageBox.information(self, "Exported", f"Saved to {path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", str(e))
