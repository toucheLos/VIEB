"""ProjectSelectorDialog and NewProjectDialog for VIEB multi-project support."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
import csv

import project_manager as _pm

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QButtonGroup,
    QCheckBox,
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
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from _utils import APP_CONFIG_PATH, ROOT

LIGHT_DIALOG_STYLE = """
QDialog {
    background: #ffffff;
    color: #202124;
}
QLabel {
    color: #202124;
    background: transparent;
}
QLineEdit, QScrollArea {
    background: #ffffff;
    color: #202124;
    border: 1px solid #DADCE0;
}
QRadioButton, QCheckBox {
    color: #202124;
    background: transparent;
}
QPushButton {
    background: #F8F9FA;
    color: #202124;
    border: 1px solid #DADCE0;
    border-radius: 4px;
    padding: 5px 10px;
}
QPushButton:hover {
    background: #F1F3F4;
}
QPushButton[primary="true"] {
    background: #1A73E8;
    color: #ffffff;
    border-color: #1A73E8;
}
"""


def _existing_directory(parent, title: str, start: str) -> str:
    return QFileDialog.getExistingDirectory(
        parent,
        title,
        start,
        QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog,
    )


def _open_file(parent, title: str, start: str, filter_str: str):
    return QFileDialog.getOpenFileName(
        parent,
        title,
        start,
        filter_str,
        options=QFileDialog.DontUseNativeDialog,
    )


def _ensure_project_scaffold(folder: Path) -> None:
    (folder / "raw_videos").mkdir(exist_ok=True)
    (folder / "results").mkdir(exist_ok=True)
    (folder / "logs").mkdir(exist_ok=True)
    metadata = folder / "metadata.csv"
    if not metadata.exists():
        with metadata.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([
                "session_id", "stem", "subject_id", "animal_id",
                "context", "condition", "day", "timepoint",
            ])


def _apply_nested_paths(cfg: dict, folder: Path) -> dict:
    cfg["paths"] = {
        "raw_videos": cfg.get("raw_videos_dir") or str(folder / "raw_videos"),
        "pose_files": cfg.get("pose_files_dir", ""),
        "pose_h5": cfg.get("h5_path") or None,
        "metadata": cfg.get("metadata_csv_path") or str(folder / "metadata.csv"),
        "results": cfg.get("results_dir") or str(folder / "results"),
        "external_data_root": cfg.get("external_data_root", ""),
    }
    cfg.setdefault("metadata_schema", {})
    cfg.setdefault("analysis_groups", [])
    cfg.setdefault("ui_panels", {})
    cfg.setdefault("pipeline_settings", {})
    return _pm.normalize_project_config(cfg, folder)

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
    "manifest_path": "",
    "h5_source_col": "",
    "h5_frame_col": "Frame Number",
}


def load_app_config() -> dict:
    if APP_CONFIG_PATH.exists():
        try:
            return json.loads(APP_CONFIG_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"projects": [], "active_project": "", "default_project": ""}


def save_app_config(app_cfg: dict) -> None:
    active = app_cfg.get("active_project")
    if active:
        recent = [p for p in app_cfg.get("recent_projects", []) if p != active]
        app_cfg["recent_projects"] = [active, *recent][:20]
    APP_CONFIG_PATH.write_text(json.dumps(app_cfg, indent=2), encoding="utf-8")


def _slugify(name: str) -> str:
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_-]+", "_", slug)
    return slug.strip("_") or "project"


class WelcomeDialog(QDialog):
    """Minimal two-screen onboarding dialog for creating a new project.

    Screen 1 asks about pose-tracking data (CSV / H5 / none).
    Screen 2 asks for a project name and creates the project on disk.
    """

    def __init__(self, app_cfg: dict, parent=None, first_launch: bool = True):
        super().__init__(parent)
        self.app_cfg = app_cfg
        self.first_launch = first_launch
        self.created_path: str = ""
        self.pose_source: str = "none"
        self.setWindowTitle("Welcome to VIEB" if first_launch else "New Project")
        self.setMinimumWidth(480)
        self.setStyleSheet(LIGHT_DIALOG_STYLE)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 24, 28, 24)
        lay.setSpacing(14)

        self._stack = QStackedWidget()
        lay.addWidget(self._stack)
        self._stack.addWidget(self._build_screen1())
        self._stack.addWidget(self._build_screen2())

    def _build_screen1(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(10)

        title = QLabel("Welcome to VIEB" if self.first_launch else "New Project")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        v.addWidget(title)

        if self.first_launch:
            subtitle = QLabel("Let's get you set up in under a minute.")
            subtitle.setStyleSheet("color:#777;")
            v.addWidget(subtitle)

        question = QLabel("Do you have pose-tracking data?")
        question.setFont(QFont("Arial", 11, QFont.Bold))
        v.addWidget(question)

        self._rb_csv = QRadioButton("I have CSV files from DeepLabCut (one per video)")
        self._rb_h5 = QRadioButton("I have a single H5 file (all sessions combined)")
        self._rb_none = QRadioButton("I haven't run pose tracking yet")
        self._source_group = QButtonGroup(self)
        for rb in (self._rb_csv, self._rb_h5, self._rb_none):
            rb.setStyleSheet("font-size:13px;padding:8px;")
            self._source_group.addButton(rb)
            v.addWidget(rb)
        self._source_group.buttonToggled.connect(self._on_source_toggled)

        v.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._next_btn = QPushButton("Next →")
        self._next_btn.setEnabled(False)
        self._next_btn.setProperty("primary", "true")
        self._next_btn.clicked.connect(lambda: self._stack.setCurrentIndex(1))
        btn_row.addWidget(self._next_btn)
        v.addLayout(btn_row)
        return w

    def _on_source_toggled(self, *_args):
        self._next_btn.setEnabled(self._source_group.checkedButton() is not None)

    def _build_screen2(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setSpacing(10)

        title = QLabel("Name your project")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        v.addWidget(title)

        self._name = QLineEdit()
        self._name.setPlaceholderText("e.g. Spence Lab Open Field")
        self._name.textChanged.connect(self._on_name_changed)
        v.addWidget(self._name)

        self._folder_preview = QLabel("")
        self._folder_preview.setStyleSheet("color:#888;font-size:11px;")
        v.addWidget(self._folder_preview)

        self._dup_error = QLabel("")
        self._dup_error.setStyleSheet("color:#c62828;font-size:11px;")
        self._dup_error.setWordWrap(True)
        v.addWidget(self._dup_error)

        v.addStretch()

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self._create_btn = QPushButton("Create Project →")
        self._create_btn.setEnabled(False)
        self._create_btn.setProperty("primary", "true")
        self._create_btn.clicked.connect(self._create)
        btn_row.addWidget(self._create_btn)
        v.addLayout(btn_row)
        return w

    def _on_name_changed(self, name: str):
        name = name.strip()
        if not name:
            self._folder_preview.setText("")
            self._dup_error.setText("")
            self._create_btn.setEnabled(False)
            return
        slug = _slugify(name)
        self._folder_preview.setText(f"Will be created at: projects/{slug}")
        if self._slug_exists(slug):
            self._dup_error.setText(
                f"A project named '{slug}' already exists. Choose a different name."
            )
            self._create_btn.setEnabled(False)
        else:
            self._dup_error.setText("")
            self._create_btn.setEnabled(True)

    def _slug_exists(self, slug: str) -> bool:
        for proj in self.app_cfg.get("projects", []):
            if _slugify(proj.get("name", "")) == slug:
                return True
        return (ROOT / "projects" / slug).exists()

    def _create(self):
        name = self._name.text().strip()
        if not name:
            return
        slug = _slugify(name)
        if self._slug_exists(slug):
            self._dup_error.setText(
                f"A project named '{slug}' already exists. Choose a different name."
            )
            self._create_btn.setEnabled(False)
            return

        folder = ROOT / "projects" / slug
        try:
            folder.mkdir(parents=True)
        except Exception as exc:
            QMessageBox.critical(self, "Error", f"Could not create folder:\n{exc}")
            return

        if self._rb_csv.isChecked():
            self.pose_source = "csv"
        elif self._rb_h5.isChecked():
            self.pose_source = "h5"
        else:
            self.pose_source = "none"

        cfg: dict = json.loads(json.dumps(_NEW_PROJECT_DEFAULTS))
        cfg["project_name"] = name
        cfg["results_dir"] = str(folder / "results")
        cfg["raw_videos_dir"] = str(folder / "raw_videos")
        cfg["pose_source"] = self.pose_source

        _ensure_project_scaffold(folder)
        _pm.write_project_config(folder, _apply_nested_paths(cfg, folder))

        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.app_cfg.setdefault("projects", []).append(
            {"name": name, "path": str(folder), "last_opened": now}
        )
        self.app_cfg["active_project"] = str(folder)
        save_app_config(self.app_cfg)

        self.created_path = str(folder)
        self.accept()


class ProjectSelectorDialog(QDialog):
    """Shows all registered projects and lets the user pick one to open."""

    def __init__(self, app_cfg: dict, parent=None):
        super().__init__(parent)
        self.app_cfg = app_cfg
        self.selected_path: str = ""
        self.setWindowTitle("Select Project")
        self.setMinimumSize(620, 420)
        self.setStyleSheet(LIGHT_DIALOG_STYLE)
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
        card.setObjectName("projectCard")
        border_color = "#4E79A7" if is_active else "#E5E5E5"
        card.setStyleSheet(
            f"QWidget#projectCard{{background:#fff;border:1px solid {border_color};"
            "border-radius:6px;}}"
            "QWidget#projectCard QLabel{border:none;background:transparent;}"
            "QWidget#projectCard QPushButton{border:1px solid #DADCE0;}"
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
        self.setWindowTitle("New Project")
        self.setMinimumWidth(540)
        self.setStyleSheet(LIGHT_DIALOG_STYLE)
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 24, 24, 24)
        lay.setSpacing(12)

        title = QLabel("New Project")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        lay.addWidget(title)

        subtitle = QLabel(
            "Create the project first. You can add raw videos, pose files, or H5 data "
            "afterward from Stage 0."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color:#5f6368;")
        lay.addWidget(subtitle)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(8)

        self._name = QLineEdit()
        self._name.setPlaceholderText("e.g. Spence Lab Open Field")
        self._name.textChanged.connect(self._on_name_changed)
        form.addRow("Project name *:", self._name)

        self._folder_preview = QLabel("")
        self._folder_preview.setStyleSheet("color:#888;font-size:11px;")
        self._folder_preview_label = ""
        form.addRow(self._folder_preview_label, self._folder_preview)

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

        lay.addLayout(form)
        self._form = form

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.button(QDialogButtonBox.Ok).setText("Create Project")
        btns.accepted.connect(self._create)
        btns.rejected.connect(self.reject)
        lay.addWidget(btns)

    @staticmethod
    def _make_row(le: QLineEdit, btn: QPushButton) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(le)
        h.addWidget(btn)
        return w

    def _on_name_changed(self, name: str):
        name = name.strip()
        if name:
            slug = _slugify(name)
            self._folder_preview.setText(f"Will be created at: projects/{slug}")
        else:
            self._folder_preview.setText("")

    def _browse_dir(self, le: QLineEdit):
        d = _existing_directory(self, "Select Directory", le.text() or str(ROOT))
        if d:
            le.setText(d)

    def _browse_file(self, le: QLineEdit, filter_str: str):
        p, _ = _open_file(self, "Select File", le.text() or str(ROOT), filter_str)
        if p:
            le.setText(p)

    def _create(self):
        name = self._name.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation", "Project name is required.")
            return

        slug = _slugify(name)
        for proj in self.app_cfg.get("projects", []):
            if _slugify(proj.get("name", "")) == slug:
                QMessageBox.warning(
                    self, "Duplicate Project",
                    f"A project named '{name}' already exists.\n"
                    "Choose a different name."
                )
                return

        folder = ROOT / "projects" / slug

        if folder.exists():
            QMessageBox.warning(
                self, "Folder Exists",
                f"Folder already exists:\n{folder}\n\nChoose a different name."
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
        cfg["pose_source"] = "none"

        meta_csv_path = self._meta.text().strip()

        # Auto-generate a starter metadata.csv when no metadata CSV was given
        # and we have raw videos to infer rows from.
        meta_generated = False
        meta_rows = 0
        if not meta_csv_path and raw:
            from metadata_generator import generate_metadata_template, write_metadata_csv
            df = generate_metadata_template(
                raw_videos_dir=cfg["raw_videos_dir"],
            )
            if not df.empty:
                meta_csv_path = str(folder / "metadata.csv")
                write_metadata_csv(df, meta_csv_path)
                meta_generated = True
                meta_rows = len(df)
                print(
                    f"Generated metadata.csv with {meta_rows} rows — "
                    "open it to fill in context/experiment columns"
                )

        cfg["metadata_csv_path"] = meta_csv_path
        if meta_csv_path and Path(meta_csv_path).exists():
            from views.metadata_mapper import _autodetect_columns
            detected = _autodetect_columns(meta_csv_path)
            cfg["column_map"].update(detected)

        _ensure_project_scaffold(folder)
        _pm.write_project_config(folder, _apply_nested_paths(cfg, folder))

        # Register in app_config
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.app_cfg.setdefault("projects", []).append(
            {"name": name, "path": str(folder), "last_opened": now}
        )
        self.app_cfg["active_project"] = str(folder)
        save_app_config(self.app_cfg)

        self.created_path = str(folder)

        msg = f"Project '{name}' created at:\n{folder}"
        if meta_generated:
            msg += (
                f"\n\nA starter metadata.csv ({meta_rows} rows) was created in your "
                "project folder. Open it to fill in context and experiment columns "
                "before running the pipeline."
            )
        QMessageBox.information(self, "Project Created", msg)

        self.accept()
