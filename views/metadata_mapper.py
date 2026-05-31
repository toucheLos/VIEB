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
    ("animal_id",   "Animal ID",            True),
    ("day",         "Day / Session",        True),
    ("context",     "Context / Condition",  True),
    ("experiment",  "Experiment",           False),
    ("cohort",      "Cohort / Group",       False),
    ("event",       "Event / Outcome",      False),
]

_NOT_MAPPED = "— not mapped —"

# Aliases for auto-detection: concept → candidate column names (lower-case)
_ALIASES: dict[str, list[str]] = {
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

        current_map = self.cfg.get("column_map", {})
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

        self.cfg["column_map"] = column_map
        _save_cfg(self.cfg)
        self.mapping_saved.emit(column_map)
        QMessageBox.information(self, "Column Mapping", "Column mapping saved.")
