from __future__ import annotations
import json
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QFileDialog, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton,
    QScrollArea, QSpinBox, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from _utils import ROOT, RESULTS, _DEFAULT_CFG, _save_cfg, _load_cfg, _open_folder


class SettingsView(QWidget):
    settings_changed = pyqtSignal(dict)
    navigate_help = pyqtSignal(str)

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)

        top_row = QHBoxLayout()
        t = QLabel("Settings")
        t.setFont(QFont("Arial", 18, QFont.Bold))
        top_row.addWidget(t)
        _hbtn = QPushButton("?")
        _hbtn.setFixedSize(20, 20)
        _hbtn.setFlat(True)
        _hbtn.setToolTip("Open Help for Settings")
        _hbtn.setCursor(Qt.PointingHandCursor)
        _hbtn.setStyleSheet(
            "QPushButton{border:1px solid #aaa;border-radius:10px;color:#555;"
            "background:#f5f5f5;font-size:10px;font-weight:bold;}"
            "QPushButton:hover{background:#e8f0fe;color:#1a73e8;border-color:#1a73e8;}"
        )
        _hbtn.clicked.connect(lambda: self.navigate_help.emit("settings"))
        top_row.addWidget(_hbtn)
        top_row.addStretch()
        lay.addLayout(top_row)
        form = QGridLayout()
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)
        r = 0

        def _help_btn(title, body):
            """Small (?) button that opens an info dialog."""
            b = QPushButton("?")
            b.setFixedSize(20, 20)
            b.setFlat(True)
            b.setStyleSheet(
                "QPushButton{border:1px solid #aaa;border-radius:10px;color:#555;"
                "background:#f5f5f5;font-size:10px;font-weight:bold;}"
                "QPushButton:hover{background:#e8f0fe;color:#1a73e8;border-color:#1a73e8;}"
            )
            b.setToolTip(body)
            b.clicked.connect(lambda: QMessageBox.information(None, title, body))
            return b

        def row(label_text, widget, tooltip=""):
            nonlocal r
            lbl = QLabel(label_text)
            if tooltip:
                lbl.setToolTip(tooltip)
                widget.setToolTip(tooltip)
            form.addWidget(lbl, r, 0)
            if tooltip:
                hw = QHBoxLayout()
                hw.setContentsMargins(0, 0, 0, 0)
                hw.setSpacing(4)
                hw.addWidget(widget)
                hw.addWidget(_help_btn(label_text, tooltip))
                form.addLayout(hw, r, 1)
            else:
                form.addWidget(widget, r, 1)
            r += 1

        def dir_row(label_text, key, tooltip=""):
            nonlocal r
            le = QLineEdit(self.cfg.get(key, ""))
            if tooltip:
                le.setToolTip(tooltip)
            browse = QPushButton("Browse...")
            browse.clicked.connect(lambda: self._browse(le))
            h = QHBoxLayout()
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(4)
            h.addWidget(le)
            h.addWidget(browse)
            if tooltip:
                h.addWidget(_help_btn(label_text, tooltip))
            lbl = QLabel(label_text)
            if tooltip:
                lbl.setToolTip(tooltip)
            form.addWidget(lbl, r, 0)
            form.addLayout(h, r, 1)
            r += 1
            return le

        ab = cfg.get("arena_bounds", _DEFAULT_CFG["arena_bounds"])
        self._xmin = QSpinBox(); self._xmin.setRange(0, 9999); self._xmin.setValue(ab["x_min"])
        self._ymin = QSpinBox(); self._ymin.setRange(0, 9999); self._ymin.setValue(ab["y_min"])
        self._xmax = QSpinBox(); self._xmax.setRange(0, 9999); self._xmax.setValue(ab["x_max"])
        self._ymax = QSpinBox(); self._ymax.setRange(0, 9999); self._ymax.setValue(ab["y_max"])

        _arena_tip = (
            "Pixel coordinates of the arena boundary in the raw video frame.\n"
            "Used to compute distance-to-wall features in feature extraction.\n"
            "Set to the full frame size (e.g. 0–1280, 0–960) if unsure."
        )
        row("Arena x_min", self._xmin, _arena_tip)
        row("Arena y_min", self._ymin, _arena_tip)
        row("Arena x_max", self._xmax, _arena_tip)
        row("Arena y_max", self._ymax, _arena_tip)

        self._results = dir_row(
            "Results directory", "results_dir",
            "Where all pipeline output files are saved: feature arrays, cluster models,\n"
            "comparison plots, and characterization CSVs.\n"
            "Default: results/ inside the VIEB project folder."
        )
        self._raw = dir_row(
            "Raw videos directory", "raw_videos_dir",
            "Folder containing your .mp4 video files and their DLC pose CSV files.\n"
            "DLC CSVs must be in the same folder as the corresponding .mp4.\n"
            "Default: raw_videos/ inside the VIEB project folder."
        )

        # ── Metadata CSV ──────────────────────────────────────────────────
        self._meta_csv = QLineEdit(self.cfg.get("metadata_csv_path", ""))
        self._meta_csv.setPlaceholderText("Path to metadata.csv …")
        meta_browse = QPushButton("Browse...")
        meta_browse.setFixedWidth(80)
        meta_browse.clicked.connect(self._browse_meta_csv)
        _meta_row_widget = QWidget()
        _meta_row_h = QHBoxLayout(_meta_row_widget)
        _meta_row_h.setContentsMargins(0, 0, 0, 0)
        _meta_row_h.setSpacing(4)
        _meta_row_h.addWidget(self._meta_csv)
        _meta_row_h.addWidget(meta_browse)
        lbl_meta = QLabel("Metadata CSV")
        lbl_meta.setToolTip("CSV file with one row per video: filename, animal_id, day, context, etc.")
        form.addWidget(lbl_meta, r, 0)
        form.addWidget(_meta_row_widget, r, 1)
        r += 1

        map_btn = QPushButton("Configure Column Mapping…")
        map_btn.setToolTip("Map your CSV column names to VIEB concepts (animal_id, day, context …)")
        map_btn.clicked.connect(self._open_mapper)
        form.addWidget(QLabel(""), r, 0)
        form.addWidget(map_btn, r, 1)
        r += 1

        gen_meta_btn = QPushButton("Generate Metadata Template…")
        gen_meta_btn.setToolTip(
            "Scan raw_videos/ and/or the configured H5 pose file and build a\n"
            "metadata.csv template with inferred filename/date/box/experiment/\n"
            "day/context/animal_id columns. no_shock and fear are left blank."
        )
        gen_meta_btn.clicked.connect(self._open_metadata_generator)
        form.addWidget(QLabel(""), r, 0)
        form.addWidget(gen_meta_btn, r, 1)
        r += 1

        validate_meta_btn = QPushButton("Validate Metadata…")
        validate_meta_btn.setToolTip(
            "Check metadata.csv for blank 'animal_id' or 'context' values\n"
            "before running feature extraction / comparison."
        )
        validate_meta_btn.clicked.connect(self._validate_metadata)
        form.addWidget(QLabel(""), r, 0)
        form.addWidget(validate_meta_btn, r, 1)
        r += 1

        # ── H5 Pose Source ──────────────────────────────────────────────
        h5_sep_lbl = QLabel("Pose Data Source")
        h5_sep_lbl.setFont(QFont("Arial", 10, QFont.Bold))
        h5_sep_lbl.setStyleSheet("color:#555; padding-top:10px; padding-bottom:2px;")
        form.addWidget(h5_sep_lbl, r, 0, 1, 2)
        r += 1

        self._pose_source = QComboBox()
        self._pose_source.addItems(["csv", "h5"])
        self._pose_source.setCurrentText(self.cfg.get("pose_source", "csv"))
        row(
            "Pose source", self._pose_source,
            "csv: per-video DLC pose CSVs in raw_videos/ (default).\n"
            "h5: a single shared H5 file containing pose data for multiple\n"
            "animals/trials, matched to metadata.csv rows by filename/animal_id."
        )

        self._h5_group = QWidget()
        h5_form = QGridLayout(self._h5_group)
        h5_form.setContentsMargins(0, 0, 0, 0)
        h5_form.setHorizontalSpacing(10)
        h5_form.setVerticalSpacing(8)

        self._h5_path = QLineEdit(self.cfg.get("h5_path", ""))
        self._h5_path.setPlaceholderText("Path to .h5 pose file …")
        h5_browse = QPushButton("Browse...")
        h5_browse.setFixedWidth(80)
        h5_browse.clicked.connect(self._browse_h5_path)
        h5_path_row = QWidget()
        h5_path_h = QHBoxLayout(h5_path_row)
        h5_path_h.setContentsMargins(0, 0, 0, 0)
        h5_path_h.setSpacing(4)
        h5_path_h.addWidget(self._h5_path)
        h5_path_h.addWidget(h5_browse)
        h5_form.addWidget(QLabel("H5 file"), 0, 0)
        h5_form.addWidget(h5_path_row, 0, 1)

        self._h5_manifest = QLineEdit(self.cfg.get("h5_manifest_path", ""))
        self._h5_manifest.setPlaceholderText("(optional) manifest CSV: animal_id/filename -> h5_key")
        h5_manifest_browse = QPushButton("Browse...")
        h5_manifest_browse.setFixedWidth(80)
        h5_manifest_browse.clicked.connect(self._browse_h5_manifest)
        h5_manifest_row = QWidget()
        h5_manifest_h = QHBoxLayout(h5_manifest_row)
        h5_manifest_h.setContentsMargins(0, 0, 0, 0)
        h5_manifest_h.setSpacing(4)
        h5_manifest_h.addWidget(self._h5_manifest)
        h5_manifest_h.addWidget(h5_manifest_browse)
        h5_form.addWidget(QLabel("H5 manifest (optional)"), 1, 0)
        h5_form.addWidget(h5_manifest_row, 1, 1)

        self._h5_key_combo = QComboBox()
        self._h5_key_combo.setEditable(True)
        existing_h5_key = self.cfg.get("h5_key", "")
        if existing_h5_key:
            self._h5_key_combo.addItem(existing_h5_key)
        h5_form.addWidget(QLabel("H5 key (default)"), 2, 0)
        h5_form.addWidget(self._h5_key_combo, 2, 1)

        self._h5_source_col_combo = QComboBox()
        self._h5_source_col_combo.setEditable(True)
        existing_source_col = self.cfg.get("h5_source_col", "")
        if existing_source_col:
            self._h5_source_col_combo.addItem(existing_source_col)
        h5_form.addWidget(QLabel("H5 source column"), 3, 0)
        h5_form.addWidget(self._h5_source_col_combo, 3, 1)

        h5_detect_btn = QPushButton("Detect")
        h5_detect_btn.setToolTip(
            "Open the H5 file and auto-populate the available keys and columns."
        )
        h5_detect_btn.clicked.connect(self._on_detect_h5)
        h5_form.addWidget(QLabel(""), 4, 0)
        h5_form.addWidget(h5_detect_btn, 4, 1)

        self._h5_detect_summary = QLabel("")
        self._h5_detect_summary.setWordWrap(True)
        self._h5_detect_summary.setStyleSheet("color:#666; font-size:11px;")
        h5_form.addWidget(self._h5_detect_summary, 5, 0, 1, 2)

        form.addWidget(self._h5_group, r, 0, 1, 2)
        r += 1

        def _toggle_h5_group(text):
            self._h5_group.setVisible(text == "h5")

        self._pose_source.currentTextChanged.connect(_toggle_h5_group)
        _toggle_h5_group(self._pose_source.currentText())

        self._ctx_groups = QLineEdit(str(self.cfg.get("context_groups", "A,B,C")))
        row(
            "Context groups (comma-separated)", self._ctx_groups,
            "Labels for the experimental contexts in your metadata.csv 'context' column.\n"
            "Example: 'A,B,C' for three contexts (A=conditioned, B=test, C=novel).\n"
            "Must exactly match the values in the context column of metadata.csv."
        )

        # Context descriptions — serialize dict as "A=shock context,B=safe context,..."
        _ctx_desc_dict = self.cfg.get("context_descriptions", {})
        _ctx_desc_str = ",".join(f"{k}={v}" for k, v in _ctx_desc_dict.items())
        self._ctx_desc = QLineEdit(_ctx_desc_str)
        _ctx_desc_tip = (
            "Comma-separated list of context=description pairs matching your metadata.csv\n"
            "context column. Used to label plots and reports.\n"
            "Example: A=shock context,B=safe context,C=novel context"
        )
        self._ctx_desc.setToolTip(_ctx_desc_tip)
        row("Context descriptions", self._ctx_desc, _ctx_desc_tip)

        # ── Experiment Labels ──────────────────────────────────────────────
        sep_lbl = QLabel("Experiment Labels")
        sep_lbl.setFont(QFont("Arial", 10, QFont.Bold))
        sep_lbl.setStyleSheet("color:#555; padding-top:10px; padding-bottom:2px;")
        form.addWidget(sep_lbl, r, 0, 1, 2)
        r += 1

        self._cond_a_le = QLineEdit(self.cfg.get("condition_a_label", ""))
        self._cond_a_le.setPlaceholderText("Auto-detected from metadata")
        row(
            "Condition A label", self._cond_a_le,
            "Label for your primary condition (e.g. 'Fear Context', 'Perturbed', 'Drug'). "
            "Used in all plots and reports. Leave blank to auto-detect.",
        )

        self._cond_b_le = QLineEdit(self.cfg.get("condition_b_label", ""))
        self._cond_b_le.setPlaceholderText("Auto-detected from metadata")
        row(
            "Condition B label", self._cond_b_le,
            "Label for your comparison condition (e.g. 'Safe Context', 'Normal', 'Vehicle').",
        )

        self._metric_label_le = QLineEdit(self.cfg.get("primary_metric_label", ""))
        self._metric_label_le.setPlaceholderText("Fear Index")
        row(
            "Primary metric label", self._metric_label_le,
            "Label for the primary per-animal scalar shown in Overview and Quantification "
            "(e.g. 'Fear Index', 'Adaptation Score', 'Learning Index').",
        )

        self._fps = QSpinBox()
        self._fps.setRange(1, 256)
        self._fps.setValue(int(self.cfg.get("fps", 30)))
        row(
            "FPS", self._fps,
            "Frame rate of your videos in frames per second.\n"
            "Used to convert frame counts to seconds in all reports and bout durations.\n"
            "Typical values: 25 (PAL), 30 (NTSC), 60 (high-speed)."
        )

        self._umap_dims = QSpinBox()
        self._umap_dims.setRange(2, 50)
        self._umap_dims.setValue(int(self.cfg.get("umap_dims", 10)))
        row(
            "UMAP dimensions", self._umap_dims,
            "Number of UMAP output dimensions before HDBSCAN clustering.\n"
            "Lower values (3–5) run faster and produce coarser clusters.\n"
            "Higher values (10–15) preserve more structure. Default: 10."
        )

        self._hdbscan_min_samples = QSpinBox()
        self._hdbscan_min_samples.setRange(0, 500)
        self._hdbscan_min_samples.setValue(int(self.cfg.get("hdbscan_min_samples", 0)))
        row(
            "HDBSCAN min_samples", self._hdbscan_min_samples,
            "HDBSCAN min_samples controls how conservative cluster borders are.\n"
            "0 = use the same value as min_cluster_size (recommended default).\n"
            "Lower values produce more clusters with softer borders."
        )

        lay.addLayout(form)

        save = QPushButton("Save Settings")
        save.clicked.connect(self._save)
        lay.addWidget(save)
        lay.addStretch()

    def _browse(self, le):
        d = QFileDialog.getExistingDirectory(self, "Select Directory", le.text())
        if d:
            le.setText(d)

    def _browse_meta_csv(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select Metadata CSV", self._meta_csv.text(), "CSV files (*.csv)"
        )
        if p:
            self._meta_csv.setText(p)

    def _browse_h5_path(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select H5 Pose File", self._h5_path.text(), "HDF5 files (*.h5 *.hdf5)"
        )
        if p:
            self._h5_path.setText(p)

    def _browse_h5_manifest(self):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select H5 Manifest CSV", self._h5_manifest.text(), "CSV files (*.csv)"
        )
        if p:
            self._h5_manifest.setText(p)

    def _on_detect_h5(self):
        h5_path = self._h5_path.text().strip()
        if not h5_path:
            QMessageBox.warning(self, "Detect H5", "Enter or browse to an H5 file first.")
            return
        try:
            from pose_io import inspect_h5
            info = inspect_h5(h5_path)
        except Exception as e:
            QMessageBox.warning(self, "Detect H5", f"Could not read H5 file:\n{e}")
            return

        keys = info.get("keys", [])
        if not keys:
            QMessageBox.warning(self, "Detect H5", "No keys found in this H5 file.")
            return

        self._h5_key_combo.clear()
        self._h5_key_combo.addItems(keys)

        first_details = info.get("details", {}).get(keys[0], {})
        columns = first_details.get("columns") or list(first_details.get("datasets", {}).keys())
        self._h5_source_col_combo.clear()
        self._h5_source_col_combo.addItems(columns)

        n_frames = first_details.get("n_frames")
        summary = f"Found {len(keys)} key(s). '{keys[0]}': "
        if columns:
            summary += f"{len(columns)} column(s)"
        if n_frames is not None:
            summary += f", {n_frames} frames"
        self._h5_detect_summary.setText(summary)

    def _open_metadata_generator(self):
        from metadata_generator import generate_metadata_template, write_metadata_csv

        raw_videos_dir = self._raw.text().strip() or None
        h5_path = self._h5_path.text().strip() or None

        try:
            df = generate_metadata_template(raw_videos_dir=raw_videos_dir, h5_path=h5_path)
        except Exception as e:
            QMessageBox.warning(self, "Generate Metadata", f"Could not scan inputs:\n{e}")
            return

        if df.empty:
            QMessageBox.information(
                self, "Generate Metadata",
                "No videos or H5 keys found to generate metadata from.\n"
                "Set 'Raw videos directory' and/or 'H5 file' first."
            )
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Generate Metadata Template")
        dlg.setMinimumSize(720, 420)
        lay = QVBoxLayout(dlg)

        info_lbl = QLabel(
            f"{len(df)} row(s) inferred. 'no_shock' and 'fear' are left blank for you to fill in."
        )
        lay.addWidget(info_lbl)

        table = QTableWidget(len(df), len(df.columns))
        table.setHorizontalHeaderLabels(list(df.columns))
        for i, (_, row_data) in enumerate(df.iterrows()):
            for j, col in enumerate(df.columns):
                table.setItem(i, j, QTableWidgetItem(str(row_data[col])))
        table.resizeColumnsToContents()
        lay.addWidget(table)

        btn_row = QHBoxLayout()
        save_btn = QPushButton("Save As…")
        cancel_btn = QPushButton("Cancel")
        btn_row.addStretch()
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(save_btn)
        lay.addLayout(btn_row)

        def _save_template():
            default_path = self._meta_csv.text().strip() or str(Path(ROOT) / "metadata.csv")
            p, _ = QFileDialog.getSaveFileName(
                dlg, "Save Metadata Template", default_path, "CSV files (*.csv)"
            )
            if not p:
                return
            write_metadata_csv(df, p)
            self._meta_csv.setText(p)
            from views.metadata_mapper import _autodetect_columns
            self.cfg["metadata_csv_path"] = p
            self.cfg["column_map"] = _autodetect_columns(p)
            QMessageBox.information(dlg, "Generate Metadata", f"Saved to {p}")
            _open_folder(Path(p).parent)
            dlg.accept()

        save_btn.clicked.connect(_save_template)
        cancel_btn.clicked.connect(dlg.reject)
        dlg.exec_()

    def _validate_metadata(self):
        from metadata_generator import validate_metadata_csv

        meta_path = self._meta_csv.text().strip() or str(Path(ROOT) / "metadata.csv")
        report = validate_metadata_csv(meta_path)

        if report["valid"]:
            QMessageBox.information(
                self, "Validate Metadata",
                f"{meta_path}\n\nLooks good — 'animal_id' and 'context' are filled in for all rows."
            )
            return

        details = "\n".join(f"- {m}" for m in report["messages"])
        QMessageBox.warning(
            self, "Validate Metadata",
            f"{meta_path}\n\nIssues found:\n{details}\n\n"
            "Fill in these rows before running feature extraction / comparison."
        )

    def _open_mapper(self):
        from views.metadata_mapper import MetadataMapperWidget
        self.cfg["metadata_csv_path"] = self._meta_csv.text().strip()
        dlg = QDialog(self)
        dlg.setWindowTitle("Metadata Column Mapping")
        dlg.setMinimumSize(560, 520)
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(0, 0, 0, 0)
        mapper = MetadataMapperWidget(self.cfg, parent=dlg)

        def _on_saved(col_map):
            self.cfg["column_map"] = col_map
            self.settings_changed.emit(self.cfg)
            dlg.accept()

        mapper.mapping_saved.connect(_on_saved)
        lay.addWidget(mapper)
        dlg.exec_()

    def _save(self):
        self.cfg["arena_bounds"] = {
            "x_min": self._xmin.value(),
            "y_min": self._ymin.value(),
            "x_max": self._xmax.value(),
            "y_max": self._ymax.value(),
        }
        self.cfg["results_dir"] = self._results.text()
        self.cfg["raw_videos_dir"] = self._raw.text()
        self.cfg["metadata_csv_path"] = self._meta_csv.text().strip()
        self.cfg["pose_source"] = self._pose_source.currentText()
        self.cfg["h5_path"] = self._h5_path.text().strip()
        self.cfg["h5_manifest_path"] = self._h5_manifest.text().strip()
        self.cfg["h5_key"] = self._h5_key_combo.currentText().strip()
        self.cfg["h5_source_col"] = self._h5_source_col_combo.currentText().strip()
        self.cfg["condition_a_label"] = self._cond_a_le.text().strip()
        self.cfg["condition_b_label"] = self._cond_b_le.text().strip()
        self.cfg["primary_metric_label"] = self._metric_label_le.text().strip()
        self.cfg["context_groups"] = self._ctx_groups.text().strip() or "A,B,C"
        # Parse "A=shock context,B=safe context" → {"A": "shock context", ...}
        ctx_desc_raw = self._ctx_desc.text().strip()
        ctx_desc_dict: dict = {}
        if ctx_desc_raw:
            for part in ctx_desc_raw.split(","):
                if "=" in part:
                    k, _, v = part.partition("=")
                    k, v = k.strip(), v.strip()
                    if k:
                        ctx_desc_dict[k] = v
        self.cfg["context_descriptions"] = ctx_desc_dict
        self.cfg["fps"] = self._fps.value()
        self.cfg["umap_dims"] = self._umap_dims.value()
        self.cfg["hdbscan_min_samples"] = self._hdbscan_min_samples.value()
        _save_cfg(self.cfg)
        self.settings_changed.emit(self.cfg)
        QMessageBox.information(self, "Settings", "Saved.")
