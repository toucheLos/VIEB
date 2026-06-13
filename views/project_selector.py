"""ProjectSelectorDialog and NewProjectDialog for VIEB multi-project support."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from _utils import APP_CONFIG_PATH, ROOT

# Full default config for new projects (canonical set of keys)
_NEW_PROJECT_DEFAULTS: dict = {
    "arena_bounds": {"x_min": 0, "y_min": 0, "x_max": 1280, "y_max": 960},
    "fps": 30,
    "window_size": [1280, 800],
    "last_view": "Overview",
    "min_cluster_size": 2000,
    "collapse_threshold": 0.5,
    "use_wavelets": True,
    "enable_state_collapse": False,
    "export_clips": False,
    "onboarding_complete": False,
    "project_name": "VIEB Project",
    "last_completed_stage": "",
    "stage_status": {},
    "stage_last_run": {},
    "context_groups": "A,B,C",
    "context_descriptions": {},
    "cohort_csv_path": "",
    "metadata_csv_path": "",
    "hdbscan_min_samples": 0,
    "umap_dims": 10,
    "validate": False,
    "min_confidence": 0.7,
    "diagnose_mcs": "",
    "umap_sweep": False,
    "hdbscan_jobs": 1,
    "dlc_project_path": "",
    "raw_videos_dir": "",
    "results_dir": "",
    "column_map": {
        "animal_id":  "animal_id",
        "day":        "day",
        "context":    "context",
        "experiment": "experiment",
        "cohort":     "",
        "event":      "",
    },
    "object_keypoints": [],
    "condition_a_label": "",
    "condition_b_label": "",
    "primary_metric_label": "",
    "pose_source": "csv",
    "h5_path": "",
    "h5_key": "",
    "h5_manifest_path": "",
    "h5_source_col": "",
}


def load_app_config() -> dict:
    if APP_CONFIG_PATH.exists():
        try:
            return json.loads(APP_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"projects": [], "active_project": "", "default_project": ""}


def save_app_config(app_cfg: dict) -> None:
    APP_CONFIG_PATH.write_text(json.dumps(app_cfg, indent=2), encoding="utf-8")


def _slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "_", slug)
    return slug.strip("_") or "project"


class ProjectSelectorDialog(QDialog):
    """Shows all registered projects and lets the user pick one to open."""

    def __init__(self, app_cfg: dict, parent=None):
        super().__init__(parent)
        self.app_cfg = app_cfg
        self.selected_path: str = ""
        self.setWindowTitle("Select Project")
        self.setMinimumSize(620, 420)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(14)

        title = QLabel("Select Project")
        title.setFont(QFont("Arial", 15, QFont.Bold))
        lay.addWidget(title)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        holder = QWidget()
        card_lay = QVBoxLayout(holder)
        card_lay.setSpacing(8)
        card_lay.setContentsMargins(0, 0, 0, 0)

        active = self.app_cfg.get("active_project", "")
        for proj in self.app_cfg.get("projects", []):
            card_lay.addWidget(self._make_card(proj, proj.get("path") == active))
        card_lay.addStretch()
        scroll.setWidget(holder)
        lay.addWidget(scroll, stretch=1)

        bot = QHBoxLayout()
        self._new_btn = QPushButton("New Project")
        self._new_btn.clicked.connect(self._open_new_project)
        bot.addWidget(self._new_btn)
        bot.addStretch()
        self._default_cb = QCheckBox("Set as default (don't ask again)")
        bot.addWidget(self._default_cb)
        lay.addLayout(bot)

    def _make_card(self, proj: dict, is_active: bool) -> QWidget:
        card = QWidget()
        border_color = "#4E79A7" if is_active else "#E5E5E5"
        card.setStyleSheet(
            f"QWidget{{background:#fff;border:1px solid {border_color};"
            "border-radius:6px;}}"
        )
        row = QHBoxLayout(card)
        row.setContentsMargins(14, 10, 14, 10)

        info = QVBoxLayout()
        name_lbl = QLabel(proj.get("name", "Unnamed"))
        name_lbl.setFont(QFont("Arial", 11, QFont.Bold))
        path_lbl = QLabel(proj.get("path", ""))
        path_lbl.setStyleSheet("color:#888;font-size:11px;")
        last_lbl = QLabel(f"Last opened: {proj.get('last_opened', '-')}")
        last_lbl.setStyleSheet("color:#aaa;font-size:10px;")
        info.addWidget(name_lbl)
        info.addWidget(path_lbl)
        info.addWidget(last_lbl)
        row.addLayout(info, stretch=1)

        if is_active:
            badge = QLabel("active")
            badge.setStyleSheet(
                "color:#4E79A7;font-size:10px;font-weight:600;"
                "border:1px solid #4E79A7;border-radius:3px;padding:1px 6px;"
            )
            row.addWidget(badge)

        open_btn = QPushButton("Open")
        open_btn.setFixedWidth(68)
        open_btn.clicked.connect(lambda _=None, p=proj: self._select(p))
        row.addWidget(open_btn)
        return card

    def _select(self, proj: dict):
        path = proj.get("path", "")
        if not Path(path).exists():
            QMessageBox.warning(
                self, "Missing Project",
                f"Project folder not found:\n{path}"
            )
            return
        self.selected_path = path
        self.app_cfg["active_project"] = path
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        for p in self.app_cfg.get("projects", []):
            if p.get("path") == path:
                p["last_opened"] = now
        if self._default_cb.isChecked():
            self.app_cfg["default_project"] = path
        save_app_config(self.app_cfg)
        self.accept()

    def _open_new_project(self):
        dlg = NewProjectDialog(self.app_cfg, self)
        if dlg.exec_() == QDialog.Accepted:
            self.selected_path = dlg.created_path
            self.app_cfg = load_app_config()
            self.accept()


class NewProjectDialog(QDialog):
    """Create a new VIEB project directory with its own config.json."""

    def __init__(self, app_cfg: dict, parent=None):
        super().__init__(parent)
        self.app_cfg = app_cfg
        self.created_path: str = ""
        self._folder_manually_set = False
        self.setWindowTitle("New Project")
        self.setMinimumWidth(540)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(12)

        title = QLabel("New Project")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        lay.addWidget(title)

        # ── Pose data source ────────────────────────────────────────────
        source_lbl = QLabel("Do you have a CSV, H5, or DLC project?")
        source_lbl.setFont(QFont("Arial", 10, QFont.Bold))
        lay.addWidget(source_lbl)

        source_row = QHBoxLayout()
        self._radio_csv = QRadioButton("CSV (DeepLabCut per-video CSVs)")
        self._radio_h5 = QRadioButton("H5 (single pose file)")
        self._radio_dlc = QRadioButton("DLC project (config.yaml + train)")
        self._radio_csv.setChecked(True)
        self._source_group = QButtonGroup(self)
        for rb in (self._radio_csv, self._radio_h5, self._radio_dlc):
            self._source_group.addButton(rb)
            source_row.addWidget(rb)
        source_row.addStretch()
        lay.addLayout(source_row)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)

        self._name = QLineEdit()
        self._name.setPlaceholderText("e.g. Luna Fear Conditioning")
        self._name.textChanged.connect(self._on_name_changed)
        form.addRow("Project name *:", self._name)

        self._folder = QLineEdit()
        self._folder.setPlaceholderText(str(ROOT / "projects" / "my_project"))
        self._folder.textEdited.connect(self._on_folder_edited)
        folder_btn = QPushButton("Browse...")
        folder_btn.setFixedWidth(80)
        folder_btn.clicked.connect(self._browse_folder)
        form.addRow("Project folder *:", self._make_row(self._folder, folder_btn))

        self._raw = QLineEdit()
        self._raw.setPlaceholderText("Optional — set later in Settings")
        raw_btn = QPushButton("Browse...")
        raw_btn.setFixedWidth(80)
        raw_btn.clicked.connect(lambda: self._browse_dir(self._raw))
        self._raw_label = "Raw videos directory:"
        form.addRow(self._raw_label, self._make_row(self._raw, raw_btn))

        self._meta = QLineEdit()
        self._meta.setPlaceholderText("Optional — set later in Settings")
        meta_btn = QPushButton("Browse...")
        meta_btn.setFixedWidth(80)
        meta_btn.clicked.connect(
            lambda: self._browse_file(self._meta, "CSV files (*.csv)")
        )
        self._meta_label = "Metadata CSV:"
        form.addRow(self._meta_label, self._make_row(self._meta, meta_btn))

        self._gen_meta_cb = QCheckBox("Generate metadata template from H5 keys after creating project")
        self._gen_meta_label = ""
        form.addRow(self._gen_meta_label, self._gen_meta_cb)

        self._dlc = QLineEdit()
        self._dlc.setPlaceholderText("Optional — set later in DLC Setup")
        dlc_btn = QPushButton("Browse...")
        dlc_btn.setFixedWidth(80)
        dlc_btn.clicked.connect(
            lambda: self._browse_file(
                self._dlc, "YAML files (*.yaml *.yml);;All files (*)"
            )
        )
        self._dlc_label = "DLC config.yaml:"
        form.addRow(self._dlc_label, self._make_row(self._dlc, dlc_btn))

        # ── H5 pose file fields ─────────────────────────────────────────
        self._h5_path = QLineEdit()
        self._h5_path.setPlaceholderText("Path to .h5 pose file")
        h5_btn = QPushButton("Browse...")
        h5_btn.setFixedWidth(80)
        h5_btn.clicked.connect(
            lambda: self._browse_file(self._h5_path, "HDF5 files (*.h5 *.hdf5)")
        )
        self._h5_path_label = "H5 file *:"
        form.addRow(self._h5_path_label, self._make_row(self._h5_path, h5_btn))

        self._h5_manifest = QLineEdit()
        self._h5_manifest.setPlaceholderText("Optional — maps animal_id/filename to h5_key")
        h5_manifest_btn = QPushButton("Browse...")
        h5_manifest_btn.setFixedWidth(80)
        h5_manifest_btn.clicked.connect(
            lambda: self._browse_file(self._h5_manifest, "CSV files (*.csv)")
        )
        self._h5_manifest_label = "H5 manifest:"
        form.addRow(self._h5_manifest_label, self._make_row(self._h5_manifest, h5_manifest_btn))

        self._h5_key_combo = QComboBox()
        self._h5_key_combo.setEditable(True)
        self._h5_key_label = "H5 key (default):"
        form.addRow(self._h5_key_label, self._h5_key_combo)

        self._h5_source_col_combo = QComboBox()
        self._h5_source_col_combo.setEditable(True)
        self._h5_source_col_label = "H5 source column:"
        form.addRow(self._h5_source_col_label, self._h5_source_col_combo)

        self._h5_detect_btn = QPushButton("Detect")
        self._h5_detect_btn.setToolTip("Open the H5 file and auto-populate the available keys and columns.")
        self._h5_detect_btn.clicked.connect(self._on_detect_h5)
        self._h5_detect_label = ""
        form.addRow(self._h5_detect_label, self._h5_detect_btn)

        self._h5_summary = QLabel("")
        self._h5_summary.setWordWrap(True)
        self._h5_summary.setStyleSheet("color:#666; font-size:11px;")
        self._h5_summary_label = ""
        form.addRow(self._h5_summary_label, self._h5_summary)

        lay.addLayout(form)
        self._form = form

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("Create Project")
        btns.accepted.connect(self._create)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

        for rb in (self._radio_csv, self._radio_h5, self._radio_dlc):
            rb.toggled.connect(self._update_visible_fields)
        self._update_visible_fields()

    @staticmethod
    def _make_row(le: QLineEdit, btn: QPushButton) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(le)
        h.addWidget(btn)
        return w

    def _on_name_changed(self, name: str):
        if self._folder_manually_set:
            return
        if name.strip():
            self._folder.setText(str(ROOT / "projects" / _slugify(name)))
        else:
            self._folder.clear()

    def _on_folder_edited(self):
        # Mark that the user has overridden the auto-generated path
        self._folder_manually_set = bool(self._folder.text().strip())

    def _browse_folder(self):
        start = self._folder.text() or str(ROOT / "projects")
        d = QFileDialog.getExistingDirectory(self, "Select Project Folder", start)
        if d:
            self._folder.setText(d)
            self._folder_manually_set = True

    def _browse_dir(self, le: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, "Select Directory",
                                              le.text() or str(ROOT))
        if d:
            le.setText(d)

    def _browse_file(self, le: QLineEdit, filter_str: str):
        p, _ = QFileDialog.getOpenFileName(
            self, "Select File", le.text() or str(ROOT), filter_str
        )
        if p:
            le.setText(p)

    def _set_row_visible(self, field_widget: QWidget, visible: bool):
        label = self._form.labelForField(field_widget)
        if label is not None:
            label.setVisible(visible)
        field_widget.setVisible(visible)

    def _update_visible_fields(self):
        is_h5 = self._radio_h5.isChecked()
        is_csv = self._radio_csv.isChecked()
        is_dlc = self._radio_dlc.isChecked()

        # widgets carry their own row container (returned by _make_row), so
        # toggle visibility on the row container, not the inner QLineEdit
        self._set_row_visible(self._raw.parentWidget(), is_csv or is_dlc)
        self._set_row_visible(self._meta.parentWidget(), True)
        self._set_row_visible(self._gen_meta_cb, is_h5)
        self._set_row_visible(self._dlc.parentWidget(), is_dlc)

        for w in (
            self._h5_path.parentWidget(),
            self._h5_manifest.parentWidget(),
            self._h5_key_combo,
            self._h5_source_col_combo,
            self._h5_detect_btn,
            self._h5_summary,
        ):
            self._set_row_visible(w, is_h5)

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
        self._h5_summary.setText(summary)

    def _create(self):
        name = self._name.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation", "Project name is required.")
            return

        is_h5 = self._radio_h5.isChecked()
        is_dlc = self._radio_dlc.isChecked()

        h5_path = self._h5_path.text().strip()
        h5_key = self._h5_key_combo.currentText().strip()
        h5_source_col = self._h5_source_col_combo.currentText().strip()
        h5_manifest = self._h5_manifest.text().strip()

        if is_h5:
            if not h5_path:
                QMessageBox.warning(self, "Validation", "H5 file is required for H5 projects.")
                return
            if not Path(h5_path).exists():
                QMessageBox.warning(self, "Validation", f"H5 file not found:\n{h5_path}")
                return
            try:
                from pose_io import inspect_h5
                info = inspect_h5(h5_path)
            except Exception as e:
                QMessageBox.warning(self, "Validation", f"Could not open H5 file:\n{e}")
                return

            keys = info.get("keys", [])
            if not keys:
                QMessageBox.warning(self, "Validation", "H5 file contains no usable keys.")
                return

            xy_re = re.compile(r"(_x$|_y$|/x$|/y$|^x$|^y$)", re.IGNORECASE)
            first_details = info["details"].get(keys[0], {})
            cols = first_details.get("columns") or []
            datasets = list(first_details.get("datasets", {}).keys())
            has_xy = any(xy_re.search(str(c)) for c in cols) or bool(
                first_details.get("shape")
            ) or any(name.lower() in ("coords", "x", "y", "xy", "pose") for name in datasets)
            if not has_xy and not datasets and not cols:
                QMessageBox.warning(
                    self, "Validation",
                    f"Key '{keys[0]}' in this H5 file doesn't look like pose data "
                    "(no x/y-like columns or datasets found)."
                )
                return

        if is_dlc:
            dlc_yaml = self._dlc.text().strip()
            if dlc_yaml and not Path(dlc_yaml).exists():
                QMessageBox.warning(self, "Validation", f"DLC config.yaml not found:\n{dlc_yaml}")
                return

        folder_text = self._folder.text().strip()
        if not folder_text:
            folder_text = str(ROOT / "projects" / _slugify(name))
        folder = Path(folder_text)

        if folder.exists():
            QMessageBox.warning(
                self, "Folder Exists",
                f"Folder already exists:\n{folder}\n\nChoose a different location."
            )
            return

        try:
            folder.mkdir(parents=True)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Could not create folder:\n{exc}")
            return

        # Build project config
        cfg: dict = json.loads(json.dumps(_NEW_PROJECT_DEFAULTS))
        cfg["project_name"] = name
        cfg["results_dir"] = str(folder / "results")
        raw = self._raw.text().strip()
        cfg["raw_videos_dir"] = raw if raw else str(folder / "raw_videos")

        meta_csv_path = self._meta.text().strip()

        if is_h5:
            cfg["pose_source"] = "h5"
            cfg["h5_path"] = h5_path
            cfg["h5_key"] = h5_key
            cfg["h5_source_col"] = h5_source_col
            cfg["h5_manifest_path"] = h5_manifest

            if not meta_csv_path and self._gen_meta_cb.isChecked():
                from metadata_generator import generate_metadata_template, write_metadata_csv
                df = generate_metadata_template(h5_path=h5_path)
                meta_csv_path = str(folder / "metadata.csv")
                write_metadata_csv(df, meta_csv_path)
        else:
            cfg["pose_source"] = "csv"

        cfg["metadata_csv_path"] = meta_csv_path
        if meta_csv_path and Path(meta_csv_path).exists():
            from views.metadata_mapper import _autodetect_columns
            detected = _autodetect_columns(meta_csv_path)
            cfg["column_map"].update(detected)

        if is_dlc:
            dlc_yaml = self._dlc.text().strip()
            if dlc_yaml:
                cfg["dlc_project_path"] = str(Path(dlc_yaml).parent)

        (folder / "config.json").write_text(
            json.dumps(cfg, indent=2), encoding="utf-8"
        )

        # Register in app_config
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.app_cfg.setdefault("projects", []).append(
            {"name": name, "path": str(folder), "last_opened": now}
        )
        self.app_cfg["active_project"] = str(folder)
        save_app_config(self.app_cfg)

        self.created_path = str(folder)
        self.accept()
