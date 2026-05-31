"""ProjectSelectorDialog and NewProjectDialog for VIEB multi-project support."""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
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
        form.addRow("Raw videos directory:", self._make_row(self._raw, raw_btn))

        self._meta = QLineEdit()
        self._meta.setPlaceholderText("Optional — set later in Settings")
        meta_btn = QPushButton("Browse...")
        meta_btn.setFixedWidth(80)
        meta_btn.clicked.connect(
            lambda: self._browse_file(self._meta, "CSV files (*.csv)")
        )
        form.addRow("Metadata CSV:", self._make_row(self._meta, meta_btn))

        self._dlc = QLineEdit()
        self._dlc.setPlaceholderText("Optional — set later in DLC Setup")
        dlc_btn = QPushButton("Browse...")
        dlc_btn.setFixedWidth(80)
        dlc_btn.clicked.connect(
            lambda: self._browse_file(
                self._dlc, "YAML files (*.yaml *.yml);;All files (*)"
            )
        )
        form.addRow("DLC config.yaml:", self._make_row(self._dlc, dlc_btn))

        lay.addLayout(form)

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

    def _create(self):
        name = self._name.text().strip()
        if not name:
            QMessageBox.warning(self, "Validation", "Project name is required.")
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
        cfg["metadata_csv_path"] = meta_csv_path
        if meta_csv_path and Path(meta_csv_path).exists():
            from views.metadata_mapper import _autodetect_columns
            detected = _autodetect_columns(meta_csv_path)
            cfg["column_map"].update(detected)

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
