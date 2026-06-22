"""MetadataMapperWidget — maps user CSV column names to VIEB concepts."""

from __future__ import annotations

import csv
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from _utils import _load_cfg, _save_cfg

# ---------------------------------------------------------------------------
# VIEB concept definitions
# ---------------------------------------------------------------------------

# (concept_key, display_label, is_required)
_CONCEPTS: list[tuple[str, str, bool]] = [
    ("session_id",  "Session / File ID",    True),
    ("animal_id",   "Animal ID",            False),
    ("day",         "Day / Session",        False),
    ("context",     "Context / Condition",  False),
    ("experiment",  "Experiment",           False),
    ("cohort",      "Cohort / Group",       False),
    ("event",       "Event / Outcome",      False),
]

_NOT_MAPPED = "— not mapped —"

# Aliases for auto-detection: concept → candidate column names (lower-case)
_ALIASES: dict[str, list[str]] = {
    "session_id": ["session_id", "filename", "source_file", "video_file", "file", "recording_file"],
    "animal_id":  ["animal_id", "animal", "mouse_id", "rat_id", "subject"],
    "day":        ["day", "session", "session_num", "trial_day"],
    "context":    ["context", "condition", "ctx", "group_condition"],
    "experiment": ["experiment", "exp", "paradigm", "task"],
    "cohort":     ["cohort", "group", "treatment", "genotype", "condition_group"],
    "event":      ["event", "outcome", "trial_outcome", "result"],
}


# ---------------------------------------------------------------------------
# Standalone helpers (importable without Qt)
# ---------------------------------------------------------------------------

def _autodetect_columns(csv_path: str) -> dict:
    """Read the CSV header and return a partial column_map dict.

    Keys are VIEB concept names; values are the matched CSV column names.
    Only matched concepts are included — callers supply defaults for the rest.
    """
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.reader(fh)
            headers = next(reader, [])
    except Exception:
        return {}

    lower_to_original = {h.lower(): h for h in headers}
    result: dict[str, str] = {}
    for concept, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in lower_to_original:
                result[concept] = lower_to_original[alias]
                break
    return result


def _read_csv_headers(csv_path: str) -> list[str]:
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            return next(csv.reader(fh), [])
    except Exception:
        return []


def _read_csv_preview(csv_path: str, n_rows: int = 5) -> tuple[list[str], list[list[str]]]:
    try:
        with open(csv_path, newline="", encoding="utf-8-sig") as fh:
            reader = csv.reader(fh)
            headers = next(reader, [])
            rows = [row for _, row in zip(range(n_rows), reader)]
        return headers, rows
    except Exception:
        return [], []


# ---------------------------------------------------------------------------
# MetadataMapperWidget
# ---------------------------------------------------------------------------

class MetadataMapperWidget(QWidget):
    """Two-column mapping table: VIEB concept ↔ user's CSV column name."""

    mapping_saved = pyqtSignal(dict)

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._combos: dict[str, QComboBox] = {}
        self._optional_cols: QLineEdit | None = None
        self._analysis_groups: QLineEdit | None = None
        self._correlation_cols: QLineEdit | None = None
        self._preview_table: QTableWidget | None = None
        self._build()

    # ── Build UI ────────────────────────────────────────────────────────────

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(12)

        title = QLabel("Metadata Column Mapping")
        title.setFont(QFont("Arial", 13, QFont.Bold))
        lay.addWidget(title)

        subtitle = QLabel(
            "Tell VIEB which columns in your metadata CSV correspond to each concept.\n"
            "Required fields (*) must be mapped before running analysis."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color:#555; font-size:11px;")
        lay.addWidget(subtitle)

        csv_path = self.cfg.get("metadata_csv_path", "")
        if not csv_path or not Path(csv_path).exists():
            placeholder = QLabel(
                "Select a metadata CSV in Settings to configure column mapping."
            )
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setWordWrap(True)
            placeholder.setStyleSheet(
                "color:#888; font-style:italic; padding:30px; font-size:12px;"
            )
            lay.addWidget(placeholder)
            lay.addStretch()
            return

        self._csv_path = csv_path
        headers = _read_csv_headers(csv_path)

        # ── Mapping grid ──────────────────────────────────────────────────
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(6)

        hdr_concept = QLabel("VIEB Concept")
        hdr_concept.setFont(QFont("Arial", 10, QFont.Bold))
        hdr_col = QLabel("Your CSV column")
        hdr_col.setFont(QFont("Arial", 10, QFont.Bold))
        grid.addWidget(hdr_concept, 0, 0)
        grid.addWidget(hdr_col, 0, 1)

        schema = self.cfg.get("metadata_schema", {}) if isinstance(self.cfg.get("metadata_schema"), dict) else {}
        current_map = dict(schema.get("column_map", {}))
        current_map.update(self.cfg.get("column_map", {}))
        auto = _autodetect_columns(csv_path)

        for row_idx, (concept, label, required) in enumerate(_CONCEPTS, start=1):
            display = f"{label} *" if required else label
            lbl = QLabel(display)
            if required:
                lbl.setStyleSheet("font-weight: bold;")
            grid.addWidget(lbl, row_idx, 0)

            combo = QComboBox()
            if not required:
                combo.addItem(_NOT_MAPPED)
            for h in headers:
                combo.addItem(h)
            self._combos[concept] = combo
            grid.addWidget(combo, row_idx, 1)

            # Priority: saved config > auto-detected > first header (required only)
            saved = current_map.get(concept, "")
            if saved and saved != _NOT_MAPPED and saved in headers:
                combo.setCurrentText(saved)
            elif concept in auto and auto[concept] in headers:
                combo.setCurrentText(auto[concept])
            elif not required:
                combo.setCurrentIndex(0)  # "— not mapped —"

        lay.addLayout(grid)

        schema = self.cfg.get("metadata_schema", {}) if isinstance(self.cfg.get("metadata_schema"), dict) else {}
        optional_cols = schema.get("optional_columns", {})
        if isinstance(optional_cols, dict):
            optional_text = ", ".join(optional_cols.keys())
        else:
            optional_text = ""
        self._optional_cols = QLineEdit(optional_text)
        self._optional_cols.setPlaceholderText("fear, sex, genotype, treatment, protein_A")
        lay.addWidget(QLabel("Optional experimental columns (comma-separated):"))
        lay.addWidget(self._optional_cols)

        groups = schema.get("analysis_groups", [])
        group_cols = [str(g.get("column", "")) for g in groups if isinstance(g, dict) and g.get("enabled", True)]
        if not group_cols:
            group_cols = ["context", "animal_id", "day", "experiment", "fear"]
        self._analysis_groups = QLineEdit(", ".join(group_cols))
        self._analysis_groups.setPlaceholderText("context, animal_id, treatment, timepoint")
        self._analysis_groups.setToolTip(
            "Grouping columns for state occupancy plots. 'context' also enables "
            "transition matrices and motif enrichment; other groups can be edited "
            "in config.json for plot-specific control."
        )
        lay.addWidget(QLabel("Analysis groups / report columns:"))
        lay.addWidget(self._analysis_groups)

        correlations = schema.get("correlations", [])
        corr_cols: list[str] = []
        if not isinstance(correlations, list):
            correlations = []
        for corr in correlations:
            if isinstance(corr, dict):
                corr_cols.extend(str(c) for c in corr.get("columns", []))
        self._correlation_cols = QLineEdit(", ".join(dict.fromkeys(corr_cols)))
        self._correlation_cols.setPlaceholderText("protein_A, protein_B")
        lay.addWidget(QLabel("Continuous columns for correlations (comma-separated):"))
        lay.addWidget(self._correlation_cols)

        # ── Action buttons ────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        detect_btn = QPushButton("Detect columns")
        detect_btn.setToolTip("Re-read the CSV header and auto-fill matched concepts")
        detect_btn.clicked.connect(self._detect)
        btn_row.addWidget(detect_btn)
        btn_row.addStretch()

        save_btn = QPushButton("Save Mapping")
        save_btn.setStyleSheet(
            "QPushButton { background:#1a73e8; color:white; padding:6px 18px; "
            "border-radius:4px; font-weight:bold; border:none; }"
            "QPushButton:hover { background:#1558b0; }"
        )
        save_btn.clicked.connect(self._save)
        btn_row.addWidget(save_btn)
        lay.addLayout(btn_row)

        # ── Preview table ─────────────────────────────────────────────────
        preview_lbl = QLabel("Preview — first 5 rows (mapped columns only):")
        preview_lbl.setFont(QFont("Arial", 10, QFont.Bold))
        preview_lbl.setStyleSheet("margin-top: 6px;")
        lay.addWidget(preview_lbl)

        self._preview_table = QTableWidget()
        self._preview_table.setAlternatingRowColors(True)
        self._preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._preview_table.setSelectionMode(QTableWidget.NoSelection)
        self._preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._preview_table.setMaximumHeight(160)
        lay.addWidget(self._preview_table)

        # Refresh preview whenever a combo changes
        for combo in self._combos.values():
            combo.currentIndexChanged.connect(self._refresh_preview)

        self._refresh_preview()
        lay.addStretch()

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _refresh_preview(self) -> None:
        if self._preview_table is None:
            return
        csv_path = self.cfg.get("metadata_csv_path", "")
        if not csv_path or not Path(csv_path).exists():
            return

        all_headers, all_rows = _read_csv_preview(csv_path)
        if not all_headers:
            self._preview_table.setRowCount(0)
            self._preview_table.setColumnCount(0)
            return

        # Collect (vieb_label, csv_col) for mapped concepts only
        mapped: list[tuple[str, str]] = []
        for concept, label, _ in _CONCEPTS:
            combo = self._combos.get(concept)
            if combo:
                val = combo.currentText()
                if val and val != _NOT_MAPPED:
                    mapped.append((label, val))

        if not mapped:
            self._preview_table.setRowCount(0)
            self._preview_table.setColumnCount(0)
            return

        vieb_labels = [m[0] for m in mapped]
        col_names = [m[1] for m in mapped]
        self._preview_table.setColumnCount(len(col_names))
        self._preview_table.setHorizontalHeaderLabels(
            [f"{lbl}\n({col})" for lbl, col in zip(vieb_labels, col_names)]
        )

        col_indices: list[int | None] = []
        for col in col_names:
            try:
                col_indices.append(all_headers.index(col))
            except ValueError:
                col_indices.append(None)

        self._preview_table.setRowCount(len(all_rows))
        for r, row_data in enumerate(all_rows):
            for c, idx in enumerate(col_indices):
                val = row_data[idx] if idx is not None and idx < len(row_data) else ""
                self._preview_table.setItem(r, c, QTableWidgetItem(str(val)))

    def _detect(self) -> None:
        """Re-run auto-detection and update combos."""
        csv_path = self.cfg.get("metadata_csv_path", "")
        if not csv_path or not Path(csv_path).exists():
            return
        headers = _read_csv_headers(csv_path)
        auto = _autodetect_columns(csv_path)
        for concept, _, _ in _CONCEPTS:
            combo = self._combos.get(concept)
            if combo and concept in auto and auto[concept] in headers:
                combo.setCurrentText(auto[concept])

    def _save(self) -> None:
        # Validate required fields
        for concept, label, required in _CONCEPTS:
            combo = self._combos.get(concept)
            val = combo.currentText() if combo else ""
            if required and (not val or val == _NOT_MAPPED):
                QMessageBox.warning(
                    self, "Validation",
                    f"'{label}' is a required field — please select a column."
                )
                return

        column_map: dict[str, str] = {}
        for concept, _, _ in _CONCEPTS:
            combo = self._combos.get(concept)
            val = combo.currentText() if combo else ""
            column_map[concept] = "" if val == _NOT_MAPPED else val

        optional_names = []
        if self._optional_cols is not None:
            optional_names = [x.strip() for x in self._optional_cols.text().split(",") if x.strip()]
        optional_columns = {name: name for name in optional_names}

        group_names = []
        if self._analysis_groups is not None:
            group_names = [x.strip() for x in self._analysis_groups.text().split(",") if x.strip()]
        analysis_groups = []
        for col in group_names:
            plots = ["state_fraction"]
            if col == "context":
                plots.extend(["transition_matrix", "motif_enrichment"])
            if col == "animal_id":
                plots.append("trajectory")
            analysis_groups.append({
                "name": col.replace("_", " ").title(),
                "column": col,
                "enabled": True,
                "plots": plots,
                "optional": col not in ("session_id",),
            })

        corr_names = []
        if self._correlation_cols is not None:
            corr_names = [x.strip() for x in self._correlation_cols.text().split(",") if x.strip()]
        correlations = []
        if corr_names:
            correlations.append({
                "name": "Configured correlations",
                "columns": corr_names,
                "targets": ["state_fraction", "motif_frequency"],
                "enabled": True,
            })

        self.cfg["column_map"] = column_map
        self.cfg["metadata_schema"] = {
            "id_column": column_map.get("session_id", "filename") or "filename",
            "column_map": column_map,
            "optional_columns": optional_columns,
            "analysis_groups": analysis_groups,
            "correlations": correlations,
        }
        self.cfg["optional_report_columns"] = optional_names or ["fear"]
        _save_cfg(self.cfg)
        self.mapping_saved.emit(self.cfg["metadata_schema"])
        QMessageBox.information(self, "Column Mapping", "Column mapping saved.")
