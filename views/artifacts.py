from __future__ import annotations

import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import pandas as pd

from PyQt5.QtCore import QSize, Qt, QThread, QTimer, QUrl, pyqtSignal
from PyQt5.QtGui import QColor, QDesktopServices, QFont, QIcon, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QFrame, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QToolButton,
    QTreeWidget, QTreeWidgetItem, QTextEdit, QVBoxLayout, QWidget,
)

from _utils import CLIPS, RESULTS, _open_folder
from artifact_scanner import (
    scan_artifacts, build_publication_bundle, format_size, format_time,
)
from views.analysis import _section_title
from views.state_characterization import _CARD_STYLE

PREVIEW_CSV_ROWS = 100
PREVIEW_TEXT_BYTES = 64_000
SMALL_JSON_BYTES = 256_000
ROW_BATCH_SIZE = 150
BINARY_TYPES = {"Model", "NumPy", "HDF5"}
VISUAL_TYPES = {"Image", "Video", "HTML"}
DATA_TYPES = {"CSV", "JSON", "Text", "YAML", "Excel", "NumPy", "HDF5", "Model", "Other", "PDF"}
VIRTUAL_CATEGORIES = {"Plots", "Raw Tables", "Models / Binary"}
MAX_BROWSER_ITEMS = 1000
MAX_GALLERY_ITEMS = 200
BINARY_SUFFIXES = {
    ".pkl", ".pt", ".pth", ".ckpt", ".npy", ".npy.gz", ".h5", ".hdf5",
    ".mp4", ".avi", ".mov",
}

ALL_CATEGORIES_LABEL = "All Files"

CATEGORY_ORDER = [
    "Summary",
    "States",
    "State Characterization",
    "Video Stories",
    "Motifs",
    "Transitions",
    "Comparison",
    "Diagnostics",
    "Quantification",
    "Cluster Runs",
    "Features",
    "Clips",
    "Raw Tables",
    "Metadata",
    "Models / Binary",
    "Plots",
]

CATEGORY_DEFAULTS = {
    "State Characterization": [
        "state_occupancy.png",
        "state_duration_summary.png",
        "state_feature_profiles.png",
        "state_feature_zscores.png",
    ],
    "Comparison": [
        "state_by_context.png",
        "state_by_day.png",
        "state_by_animal.png",
        "animal_trajectories.png",
        "transition_by_context.png",
    ],
    "Diagnostics": ["cluster_overview.png", "umap_embedding_by_state.png"],
    "Motifs": ["motif_heatmap.png"],
    "Transitions": ["transition_by_context.png", "transition_matrix.png", "transition_table.csv"],
    "Video Stories": ["story_timeline.png", "timeline.png", "video_stories.csv"],
    "Quantification": [
        "contrast_bars.png",
        "contrast_heatmap.png",
        "contrast_magnitude.png",
        "contrast_scatter.png",
        "master_table.csv",
    ],
}

FILE_PURPOSES = {
    "video_stories.csv": (
        "One row per video/session summarizing the state story: dominant state, "
        "transitions, entropy, and sequence metrics."
    ),
    "video_story_bouts.csv": (
        "One row per state step/bout in each video, with state id, start/end "
        "time, and duration."
    ),
    "subject_journeys.csv": (
        "One row per subject/timepoint summarizing how the subject's story "
        "changes across the experiment."
    ),
    "state_summary.csv": (
        "State-level summary table: occupancy, bout counts, durations, and "
        "feature summaries."
    ),
    "summary_table.csv": "Session-level state fractions: one row per video/session.",
    "bouts.csv": (
        "Continuous state-step/bout table used for stories, durations, motifs, "
        "and clips."
    ),
    "motif_sequences.csv": "Observed motif occurrences derived from state-step sequences.",
    "motif_context_enrichment.csv": "Motifs enriched by timepoint, condition, or context.",
    "transition_table.csv": "State-to-state transition counts/frequencies per session or group.",
    "cluster_info.json": (
        "Cluster model metadata: number of states, centers, parameters, and diagnostics."
    ),
    "index.json": "Feature extraction manifest mapping sessions to feature arrays and source files.",
}

_SIDEBAR_STYLE = """
    QListWidget {
        background: #F4F4F4;
        border: none;
        border-right: 1px solid #DCDCDC;
        padding-top: 4px;
        outline: none;
    }
    QListWidget::item {
        padding: 9px 14px;
        color: #333;
        font-size: 12px;
        min-height: 20px;
    }
    QListWidget::item:selected {
        background: #1a73e8;
        color: white;
    }
    QListWidget::item:hover:!selected {
        background: #E8E8E8;
    }
"""


def binary_preview_disabled(file_type: str, filename: str) -> bool:
    suffixes = "".join(s.lower() for s in Path(filename).suffixes)
    return (
        file_type in BINARY_TYPES
        or file_type == "Video"
        or any(suffixes.endswith(s) for s in BINARY_SUFFIXES)
    )


def is_visual_artifact(art: dict) -> bool:
    return art.get("file_type") in VISUAL_TYPES


def is_data_artifact(art: dict) -> bool:
    return not is_visual_artifact(art)


def artifact_purpose(art: dict) -> str:
    return FILE_PURPOSES.get(Path(art.get("name", "")).name, "")


def artifact_section_name(art: dict) -> str:
    return "Visuals" if is_visual_artifact(art) else "Data / Tables"


class ArtifactScanWorker(QThread):
    done = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, results_dir: str, clips_dir: str | None):
        super().__init__()
        self._results_dir = results_dir
        self._clips_dir = clips_dir

    def run(self):
        import time
        t0 = time.perf_counter()
        try:
            artifacts = scan_artifacts(self._results_dir, clips_dir=self._clips_dir)
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        print(f"[timing] Artifacts scan: {(time.perf_counter() - t0) * 1000:.1f} ms ({len(artifacts)} files)")
        self.done.emit(artifacts)


class ArtifactsView(QWidget):
    worker_running = pyqtSignal(bool)

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.cfg = cfg or {}
        self._data: dict = {}
        self._artifacts: list[dict] = []
        self._artifact_by_rel: dict[str, dict] = {}
        self._filtered: list[dict] = []
        self._worker = None
        self._running_command = ""
        self._pending_rows: list[dict] = []
        self._row_timer = QTimer(self)
        self._row_timer.timeout.connect(self._insert_next_rows)
        self._pending_category: str | None = None
        self._ignore_browser_selection = False
        self._current_rel_path: str | None = None
        self._action_buttons: dict[str, QPushButton] = {}
        # row -> category label ("All Files" or a real category name)
        self._nav_row_category: dict[int, str] = {}
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # ── Category sub-nav sidebar ─────────────────────────────────────
        self._cat_nav = QListWidget()
        self._cat_nav.setFixedWidth(210)
        self._cat_nav.setMinimumWidth(210)
        self._cat_nav.setMaximumWidth(210)
        self._cat_nav.setSpacing(0)
        self._cat_nav.setUniformItemSizes(True)
        self._cat_nav.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._cat_nav.setTextElideMode(Qt.ElideRight)
        self._cat_nav.setStyleSheet(_SIDEBAR_STYLE)
        self._cat_nav.currentRowChanged.connect(self._on_nav_row_changed)
        root.addWidget(self._cat_nav)

        # ── Right panel ───────────────────────────────────────────────────
        right = QVBoxLayout()
        right.setContentsMargins(16, 16, 16, 16)
        right.setSpacing(8)

        # ── Header ───────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        title = QLabel("Artifacts")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        hdr.addWidget(title)
        hdr.addStretch()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedHeight(28)
        refresh_btn.clicked.connect(self._scan)
        hdr.addWidget(refresh_btn)
        right.addLayout(hdr)

        # ── Summary card ─────────────────────────────────────────────────
        self._summary_card = QFrame()
        self._summary_card.setStyleSheet(_CARD_STYLE)
        card_lay = QVBoxLayout(self._summary_card)
        card_lay.setContentsMargins(6, 4, 6, 4)
        card_lay.setSpacing(1)
        self._summary_lbl = QLabel("")
        self._summary_lbl.setStyleSheet(
            "font-size:13px; font-weight:bold; color:#1A1A1A; border:none; background:transparent;"
        )
        card_lay.addWidget(self._summary_lbl)
        right.addWidget(self._summary_card)

        # ── Filter bar (secondary refinement within a category) ───────────
        filt = QHBoxLayout()
        filt.addWidget(QLabel("Search:"))
        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter by name or path...")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_filters)
        self._search.setFixedWidth(220)
        filt.addWidget(self._search)

        filt.addWidget(QLabel("Type:"))
        self._type_filter = QComboBox()
        self._type_filter.addItem("All")
        for t in ("CSV", "JSON", "Image", "Video", "PDF", "NumPy",
                  "HDF5", "Model", "Excel", "Text", "YAML", "HTML", "Other"):
            self._type_filter.addItem(t)
        self._type_filter.currentTextChanged.connect(self._apply_filters)
        filt.addWidget(self._type_filter)
        self._clip_type_filter = QComboBox()
        self._clip_type_filter.addItems(["All clip types", "longest", "typical", "context", "story", "motif"])
        self._clip_type_filter.currentTextChanged.connect(self._apply_filters)
        self._clip_type_filter.hide()
        filt.addWidget(self._clip_type_filter)
        self._clip_state_filter = QLineEdit()
        self._clip_state_filter.setPlaceholderText("State id")
        self._clip_state_filter.setFixedWidth(80)
        self._clip_state_filter.textChanged.connect(self._apply_filters)
        self._clip_state_filter.hide()
        filt.addWidget(self._clip_state_filter)
        filt.addStretch()
        right.addLayout(filt)

        # ── Splitter: file table + preview ───────────────────────────────
        splitter = QSplitter(Qt.Vertical)

        browser = QWidget()
        browser_lay = QVBoxLayout(browser)
        browser_lay.setContentsMargins(0, 0, 0, 0)
        browser_lay.setSpacing(4)
        view_row = QHBoxLayout()
        view_row.addWidget(_section_title("Files"))
        view_row.addStretch()
        self._table_view_btn = QToolButton()
        self._table_view_btn.setText("Table")
        self._table_view_btn.setCheckable(True)
        self._table_view_btn.setChecked(True)
        self._table_view_btn.clicked.connect(lambda: self._set_browser_mode("table"))
        view_row.addWidget(self._table_view_btn)
        self._gallery_view_btn = QToolButton()
        self._gallery_view_btn.setText("Gallery")
        self._gallery_view_btn.setCheckable(True)
        self._gallery_view_btn.clicked.connect(lambda: self._set_browser_mode("gallery"))
        view_row.addWidget(self._gallery_view_btn)
        browser_lay.addLayout(view_row)

        self._browser_stack = QStackedWidget()
        self._file_tree = QTreeWidget()
        self._file_tree.setColumnCount(6)
        self._file_tree.setHeaderLabels(["Name", "Category", "Type", "Size", "Modified", "Path"])
        self._file_tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in (1, 2, 3, 4):
            self._file_tree.header().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self._file_tree.header().setSectionResizeMode(5, QHeaderView.Stretch)
        self._file_tree.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._file_tree.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._file_tree.currentItemChanged.connect(self._on_tree_selection_changed)
        self._file_tree.itemDoubleClicked.connect(lambda *_: self._open_file())
        self._browser_stack.addWidget(self._file_tree)

        self._gallery = QListWidget()
        self._gallery.setViewMode(QListWidget.IconMode)
        self._gallery.setResizeMode(QListWidget.Adjust)
        self._gallery.setMovement(QListWidget.Static)
        self._gallery.setIconSize(QSize(160, 120))
        self._gallery.setGridSize(QSize(190, 165))
        self._gallery.setWordWrap(True)
        self._gallery.currentItemChanged.connect(self._on_gallery_selection_changed)
        self._gallery.itemDoubleClicked.connect(lambda *_: self._open_file())
        self._browser_stack.addWidget(self._gallery)
        browser_lay.addWidget(self._browser_stack)
        splitter.addWidget(browser)

        # Preview pane — titled card, consistent with Analysis's plot/detail panes
        preview_card = QFrame()
        preview_card.setStyleSheet(_CARD_STYLE)
        preview_card_lay = QVBoxLayout(preview_card)
        preview_card_lay.setContentsMargins(6, 4, 6, 6)
        preview_card_lay.setSpacing(4)
        preview_card_lay.addWidget(_section_title("Preview"))
        self._preview_details = QLabel("")
        self._preview_details.setWordWrap(True)
        self._preview_details.setStyleSheet(
            "color:#444; font-size:11px; padding:2px 4px; border:none; background:transparent;"
        )
        preview_card_lay.addWidget(self._preview_details)

        self._preview_stack = QStackedWidget()

        self._preview_empty = QLabel("Select a file to preview")
        self._preview_empty.setAlignment(Qt.AlignCenter)
        self._preview_empty.setStyleSheet(
            "color:#999; font-style:italic; padding:20px; border:none; background:transparent;"
        )
        self._preview_stack.addWidget(self._preview_empty)

        self._preview_text = QTextEdit()
        self._preview_text.setReadOnly(True)
        self._preview_text.setStyleSheet(
            "font-family:'Consolas','Courier New',monospace; font-size:10pt;"
        )
        self._preview_stack.addWidget(self._preview_text)

        self._preview_table = QTableWidget()
        self._preview_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._preview_stack.addWidget(self._preview_table)

        self._preview_image = QLabel()
        self._preview_image.setAlignment(Qt.AlignCenter)
        self._preview_image.setStyleSheet("background:#f0f0f0; padding:8px; border:none;")
        self._preview_stack.addWidget(self._preview_image)

        self._preview_info = QLabel()
        self._preview_info.setAlignment(Qt.AlignCenter)
        self._preview_info.setWordWrap(True)
        self._preview_info.setStyleSheet("color:#555; padding:20px; border:none; background:transparent;")
        self._preview_stack.addWidget(self._preview_info)

        self._preview_stack.setCurrentWidget(self._preview_empty)
        preview_card_lay.addWidget(self._preview_stack)
        splitter.addWidget(preview_card)
        splitter.setSizes([400, 200])
        right.addWidget(splitter, stretch=1)

        # ── Bottom action bar ─────────────────────────────────────────────
        btn_row = QHBoxLayout()
        for label, slot in [
            ("Open File", self._open_file),
            ("Reveal in Folder", self._reveal_file),
            ("Save As…", self._save_as),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(30)
            btn.clicked.connect(slot)
            btn_row.addWidget(btn)
            self._action_buttons[label] = btn
        btn_row.addStretch()
        for label, slot in [
            ("Export Selected", self._export_selected),
            ("Export Category", self._export_category),
            ("Export All as ZIP", self._export_all),
            ("Publication Bundle", self._export_publication),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(30)
            btn.clicked.connect(slot)
            btn_row.addWidget(btn)
            self._action_buttons[label] = btn
        right.addLayout(btn_row)

        right_widget = QWidget()
        right_widget.setLayout(right)
        root.addWidget(right_widget, stretch=1)
        self._update_action_buttons()

    # ----------------------------------------------------------- data hooks
    def update_data(self, data: dict) -> None:
        self._data = data

    def refresh(self, data: dict | None = None) -> None:
        if isinstance(data, dict):
            self._data = data
        self._scan()

    def select_category(self, category: str) -> None:
        """Public hook for other views to navigate here with a category
        preselected (e.g. Video Stories -> Artifacts). If a scan is already
        in flight or hasn't happened yet, the category is applied once
        _on_scan_done sees it among the freshly scanned categories."""
        self._pending_category = category
        if self._select_nav_row_for_category(category):
            self._pending_category = None
        self._apply_filters()

    # -------------------------------------------------------------- scanning
    def _scan(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._summary_lbl.setText("Scanning artifacts...")
        self._row_timer.stop()
        self._pending_rows = []
        self._file_tree.clear()
        self._gallery.clear()
        self._current_rel_path = None
        self._preview_stack.setCurrentWidget(self._preview_empty)
        self._update_action_buttons()
        results_dir, clips_dir = self._current_artifact_roots()
        self._worker = ArtifactScanWorker(str(results_dir), str(clips_dir) if clips_dir else None)
        self._worker.done.connect(self._on_scan_done)
        self._worker.failed.connect(self._on_scan_failed)
        self._running_command = "artifact scan"
        self.worker_running.emit(True)
        self._worker.start()

    def _current_artifact_roots(self) -> tuple[Path, Path | None]:
        """Resolve active-project artifact roots at scan time, not import time."""
        try:
            import vieb_config as _vc
            results_dir = Path(_vc.get_results_dir())
            clips_dir = Path(_vc.get_clips_dir())
        except Exception:
            results_dir = RESULTS
            clips_dir = CLIPS
        return results_dir, clips_dir if clips_dir.is_dir() else None

    def _on_scan_failed(self, message: str) -> None:
        self.worker_running.emit(False)
        self._running_command = ""
        self._summary_lbl.setText("Artifact scan failed")
        self._show_info(f"Artifact scan failed:\n{message}")

    def _on_scan_done(self, artifacts: list[dict]) -> None:
        self.worker_running.emit(False)
        self._running_command = ""
        self._artifacts = artifacts
        self._artifact_by_rel = {a["rel_path"]: a for a in self._artifacts}

        counts = self._category_counts()
        categories = [cat for cat in CATEGORY_ORDER if counts.get(cat)]
        categories.extend(sorted(cat for cat in counts if cat not in categories))

        current = self._current_category()

        self._cat_nav.blockSignals(True)
        self._rebuild_nav(categories, counts)

        if self._pending_category and self._pending_category in categories:
            self._select_nav_row_for_category(self._pending_category)
            self._pending_category = None
        elif not (current and self._select_nav_row_for_category(current)):
            self._cat_nav.setCurrentRow(0)
        self._cat_nav.blockSignals(False)

        self._apply_filters()

    # ------------------------------------------------------------ sidebar nav
    def _category_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        raw_tables = 0
        plots = 0
        binaries = 0
        for art in self._artifacts:
            cat = art["category"]
            counts[cat] = counts.get(cat, 0) + 1
            if art["file_type"] == "Image" or art["file_type"] in {"Video", "HTML"}:
                plots += 1
            if art["file_type"] in {"CSV", "JSON", "Text", "YAML", "Excel"}:
                raw_tables += 1
            if art["file_type"] in BINARY_TYPES:
                binaries += 1
        if plots:
            counts["Plots"] = plots
        if raw_tables:
            counts["Raw Tables"] = raw_tables
        if binaries:
            counts["Models / Binary"] = binaries
        return counts

    def _rebuild_nav(self, categories: list[str], counts: dict[str, int]) -> None:
        labels = [f"{ALL_CATEGORIES_LABEL} ({len(self._artifacts)})", "CATEGORIES"]
        labels.extend(f"{cat} ({counts[cat]})" for cat in categories)
        existing = [self._cat_nav.item(i).text() for i in range(self._cat_nav.count())]
        if existing == labels:
            return

        self._cat_nav.clear()
        self._nav_row_category = {}

        total = len(self._artifacts)
        all_item = QListWidgetItem(f"{ALL_CATEGORIES_LABEL} ({total})")
        all_item.setSizeHint(QSize(210, 38))
        self._cat_nav.addItem(all_item)
        self._nav_row_category[0] = ALL_CATEGORIES_LABEL

        header = QListWidgetItem("CATEGORIES")
        header.setFlags(Qt.ItemIsEnabled)
        font = header.font()
        font.setPointSize(8)
        font.setBold(True)
        header.setFont(font)
        header.setForeground(QColor("#999"))
        header.setBackground(QColor("#EBEBEB"))
        header.setSizeHint(QSize(210, 26))
        self._cat_nav.addItem(header)

        for cat in categories:
            row = self._cat_nav.count()
            item = QListWidgetItem(f"{cat} ({counts[cat]})")
            item.setSizeHint(QSize(210, 38))
            self._cat_nav.addItem(item)
            self._nav_row_category[row] = cat

    def _select_nav_row_for_category(self, category: str) -> bool:
        for row, cat in self._nav_row_category.items():
            if cat == category:
                self._cat_nav.setCurrentRow(row)
                return True
        return False

    def _current_category(self) -> str:
        row = self._cat_nav.currentRow()
        return self._nav_row_category.get(row, ALL_CATEGORIES_LABEL)

    def _on_nav_row_changed(self, row: int) -> None:
        if row not in self._nav_row_category:
            # Section header row is not selectable content — bounce to the
            # nearest real row instead of leaving the filter stuck.
            fallback = 0 if row < 0 else min(row + 1, self._cat_nav.count() - 1)
            if fallback in self._nav_row_category:
                self._cat_nav.setCurrentRow(fallback)
            return
        self._apply_filters()

    # ------------------------------------------------------------ filtering
    def _apply_filters(self) -> None:
        search = self._search.text().strip().lower()
        cat = self._current_category()
        ftype = self._type_filter.currentText()
        self._clip_type_filter.setVisible(cat == "Clips")
        self._clip_state_filter.setVisible(cat == "Clips")

        filtered = self._artifacts_for_category(cat)
        if ftype != "All":
            filtered = [a for a in filtered if a["file_type"] == ftype]
        if search:
            filtered = [
                a for a in filtered
                if search in a["name"].lower() or search in a["rel_path"].lower()
            ]
        if cat == "Clips":
            clip_type = self._clip_type_filter.currentText()
            state = self._clip_state_filter.text().strip().lower()
            if clip_type != "All clip types":
                filtered = [a for a in filtered if self._clip_type_matches(a, clip_type)]
            if state:
                token = f"state_{state}"
                filtered = [a for a in filtered if token in a["rel_path"].replace("\\", "/").lower()]

        self._filtered = filtered
        self._populate_browser(filtered)

        total_size = sum(a["size_bytes"] for a in filtered)
        if cat == "Video Stories":
            self._summary_lbl.setText(self._video_stories_summary(filtered, total_size))
        else:
            self._summary_lbl.setText(
                f"{cat} — {len(filtered)} files, {format_size(total_size)}"
            )

    def _video_stories_summary(self, artifacts: list[dict], total_size: int) -> str:
        visual_count = sum(1 for a in artifacts if is_visual_artifact(a))
        story_count = self._cheap_csv_row_count(artifacts, "video_stories.csv")
        subject_count = self._cheap_csv_unique_count(artifacts, "subject_journeys.csv", "subject_id")
        bout_count = self._cheap_csv_row_count(artifacts, "video_story_bouts.csv")
        parts = [
            f"Video Stories — {len(artifacts)} files, {format_size(total_size)}",
            f"{visual_count} story visual files",
        ]
        if story_count is not None:
            parts.append(f"{story_count} videos/stories")
        if subject_count is not None:
            parts.append(f"{subject_count} subjects")
        if bout_count is not None:
            parts.append(f"{bout_count} state steps/bouts")
        if visual_count == 0:
            parts.append(
                "No story timeline images/clips found. Open Analysis -> Sequences -> "
                "Video Stories to inspect timelines interactively or generate story visualizations."
            )
        else:
            parts.append("Open Analysis -> Sequences -> Video Stories to inspect timelines interactively.")
        return " | ".join(parts)

    def _cheap_csv_row_count(self, artifacts: list[dict], name: str) -> int | None:
        art = next((a for a in artifacts if a["name"] == name), None)
        if not art or art["size_bytes"] > 20_000_000:
            return None
        try:
            with open(art["abs_path"], "r", encoding="utf-8", errors="replace") as f:
                return max(sum(1 for _ in f) - 1, 0)
        except OSError:
            return None

    def _cheap_csv_unique_count(self, artifacts: list[dict], name: str, column: str) -> int | None:
        art = next((a for a in artifacts if a["name"] == name), None)
        if not art or art["size_bytes"] > 20_000_000:
            return None
        try:
            df = pd.read_csv(art["abs_path"], usecols=[column])
            return int(df[column].nunique())
        except Exception:
            return None

    def _clip_type_matches(self, art: dict, clip_type: str) -> bool:
        rel = art["rel_path"].replace("\\", "/").lower()
        name = art["name"].lower()
        if clip_type == "story":
            return "/stories/" in rel
        if clip_type == "motif":
            return "motif" in rel or "motif" in name
        return clip_type in name or f"/{clip_type}" in rel

    def _artifacts_for_category(self, cat: str) -> list[dict]:
        if cat == ALL_CATEGORIES_LABEL:
            return list(self._artifacts)
        if cat == "Plots":
            return [a for a in self._artifacts if is_visual_artifact(a)]
        if cat == "Raw Tables":
            return [a for a in self._artifacts if a["file_type"] in {"CSV", "JSON", "Text", "YAML", "Excel"}]
        if cat == "Models / Binary":
            return [a for a in self._artifacts if a["file_type"] in BINARY_TYPES]
        return [a for a in self._artifacts if a["category"] == cat]

    def _populate_table(self, artifacts: list[dict]) -> None:
        self._populate_browser(artifacts)

    def _populate_browser(self, artifacts: list[dict]) -> None:
        self._ignore_browser_selection = True
        self._row_timer.stop()
        self._pending_rows = list(artifacts)
        self._file_tree.clear()
        self._gallery.clear()
        self._insert_next_rows()
        self._populate_gallery(artifacts)
        self._ignore_browser_selection = False
        self._auto_select_preview(artifacts)
        self._update_action_buttons()

    def _insert_next_rows(self) -> None:
        artifacts = self._pending_rows[:MAX_BROWSER_ITEMS]
        self._pending_rows = []
        sections: dict[str, list[dict]] = {"Visuals": [], "Data / Tables": []}
        for art in artifacts:
            sections[artifact_section_name(art)].append(art)
        for section_name in ("Visuals", "Data / Tables"):
            section_files = sections[section_name]
            if not section_files:
                continue
            display_section = section_name
            if self._current_category() == "Video Stories":
                display_section = "Story Visuals" if section_name == "Visuals" else "Story Tables"
            item = QTreeWidgetItem([f"{display_section} ({len(section_files)})", "", "", "", "", ""])
            font = item.font(0)
            font.setBold(True)
            item.setFont(0, font)
            item.setForeground(0, QColor("#333"))
            item.setBackground(0, QColor("#F0F0F0"))
            self._file_tree.addTopLevelItem(item)
            for art in sorted(section_files, key=lambda a: (a["file_type"], a["rel_path"].lower())):
                child = QTreeWidgetItem([
                    art["name"],
                    art["category"],
                    art["file_type"],
                    format_size(art["size_bytes"]),
                    format_time(art["modified_ts"]),
                    art["rel_path"],
                ])
                child.setData(0, Qt.UserRole, art["rel_path"])
                purpose = artifact_purpose(art)
                if purpose:
                    child.setToolTip(0, purpose)
                item.addChild(child)
            item.setExpanded(True)
        cat = self._current_category()
        total_size = sum(a["size_bytes"] for a in self._filtered)
        if len(self._filtered) > MAX_BROWSER_ITEMS:
            self._summary_lbl.setText(
                f"{cat} — showing first {MAX_BROWSER_ITEMS} of {len(self._filtered)} files, {format_size(total_size)}"
            )
        else:
            self._summary_lbl.setText(
                f"{cat} — {len(self._filtered)} files, {format_size(total_size)}"
            )

    def _populate_gallery(self, artifacts: list[dict]) -> None:
        visuals = [a for a in artifacts if is_visual_artifact(a)]
        for art in visuals[:MAX_GALLERY_ITEMS]:
            item = QListWidgetItem(art["name"])
            item.setData(Qt.UserRole, art["rel_path"])
            if art["file_type"] == "Image":
                pix = QPixmap(art["abs_path"])
                if not pix.isNull():
                    item.setIcon(QIcon(pix.scaled(160, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)))
            else:
                item.setIcon(self.style().standardIcon(self.style().SP_FileIcon))
            purpose = artifact_purpose(art)
            if purpose:
                item.setToolTip(purpose)
            self._gallery.addItem(item)
        if len(visuals) > MAX_GALLERY_ITEMS:
            item = QListWidgetItem(f"Showing first {MAX_GALLERY_ITEMS} visual artifacts")
            item.setFlags(Qt.ItemIsEnabled)
            self._gallery.addItem(item)

    # ----------------------------------------------------------- selection
    def _selected_artifact(self) -> dict | None:
        if self._browser_stack.currentWidget() is self._gallery:
            item = self._gallery.currentItem()
            if item is not None:
                rel_path = item.data(Qt.UserRole)
                return self._artifact_by_rel.get(rel_path) if rel_path else None
        else:
            item = self._file_tree.currentItem()
            if item is not None:
                rel_path = item.data(0, Qt.UserRole)
                return self._artifact_by_rel.get(rel_path) if rel_path else None
        return self._artifact_by_rel.get(self._current_rel_path) if self._current_rel_path else None

    def _selected_artifacts(self) -> list[dict]:
        if self._browser_stack.currentWidget() is self._gallery:
            items = self._gallery.selectedItems()
            rels = [item.data(Qt.UserRole) for item in items]
            selected = [self._artifact_by_rel[rel] for rel in rels if rel in self._artifact_by_rel]
            if not selected and self._current_rel_path in self._artifact_by_rel:
                return [self._artifact_by_rel[self._current_rel_path]]
            return selected

        out: list[dict] = []
        for item in self._file_tree.selectedItems():
            rel = item.data(0, Qt.UserRole)
            if rel in self._artifact_by_rel:
                out.append(self._artifact_by_rel[rel])
        return out

    def _on_tree_selection_changed(self, item: QTreeWidgetItem | None, _prev: QTreeWidgetItem | None) -> None:
        if self._ignore_browser_selection:
            return
        art = self._selected_artifact()
        if not art:
            self._current_rel_path = None
            self._preview_stack.setCurrentWidget(self._preview_empty)
            self._update_action_buttons()
            return
        self._current_rel_path = art["rel_path"]
        self._preview_file(art)
        self._sync_gallery_selection(art["rel_path"])
        self._update_action_buttons()

    def _on_gallery_selection_changed(self, item: QListWidgetItem | None, _prev: QListWidgetItem | None) -> None:
        if self._ignore_browser_selection:
            return
        art = self._selected_artifact()
        if not art:
            self._current_rel_path = None
            self._update_action_buttons()
            return
        self._current_rel_path = art["rel_path"]
        self._preview_file(art)
        self._sync_tree_selection(art["rel_path"])
        self._update_action_buttons()

    def _set_browser_mode(self, mode: str) -> None:
        self._table_view_btn.setChecked(mode == "table")
        self._gallery_view_btn.setChecked(mode == "gallery")
        self._browser_stack.setCurrentWidget(self._gallery if mode == "gallery" else self._file_tree)
        self._update_action_buttons()

    def _sync_tree_selection(self, rel_path: str) -> None:
        for i in range(self._file_tree.topLevelItemCount()):
            root = self._file_tree.topLevelItem(i)
            for j in range(root.childCount()):
                child = root.child(j)
                if child.data(0, Qt.UserRole) == rel_path:
                    self._ignore_browser_selection = True
                    self._file_tree.setCurrentItem(child)
                    self._ignore_browser_selection = False
                    return

    def _sync_gallery_selection(self, rel_path: str) -> None:
        for i in range(self._gallery.count()):
            item = self._gallery.item(i)
            if item.data(Qt.UserRole) == rel_path:
                self._ignore_browser_selection = True
                self._gallery.setCurrentItem(item)
                self._ignore_browser_selection = False
                return

    def _auto_select_preview(self, artifacts: list[dict]) -> None:
        art = self._best_default_artifact(artifacts)
        if not art:
            self._current_rel_path = None
            self._preview_details.clear()
            self._preview_details.hide()
            self._preview_empty.setText("No previewable artifacts found in this category")
            self._preview_stack.setCurrentWidget(self._preview_empty)
            return
        self._current_rel_path = art["rel_path"]
        self._sync_tree_selection(art["rel_path"])
        self._sync_gallery_selection(art["rel_path"])
        self._preview_file(art)

    def _best_default_artifact(self, artifacts: list[dict]) -> dict | None:
        if not artifacts:
            return None
        cat = self._current_category()
        by_name = {a["name"].lower(): a for a in artifacts}
        by_rel = {a["rel_path"].replace("\\", "/").lower(): a for a in artifacts}
        for name in CATEGORY_DEFAULTS.get(cat, []):
            key = name.lower()
            if key in by_name:
                return by_name[key]
            for rel, art in by_rel.items():
                if rel.endswith(key):
                    return art
        for art in artifacts:
            if art["file_type"] == "Image":
                return art
        for art in artifacts:
            if art["file_type"] == "Video":
                return art
        for art in artifacts:
            if art["file_type"] in {"CSV", "Excel"}:
                return art
        for art in artifacts:
            if art["file_type"] in {"JSON", "Text", "YAML", "HTML"}:
                return art
        return artifacts[0]

    # ------------------------------------------------------------- preview
    def _preview_file(self, art: dict) -> None:
        t0 = time.perf_counter()
        ftype = art["file_type"]
        path = art["abs_path"]
        purpose = artifact_purpose(art)
        self._preview_details.setText(purpose)
        self._preview_details.setVisible(bool(purpose))

        def _metadata_text(message: str | None = None) -> str:
            lines = [
                f"File: {art['name']}",
                f"Type: {ftype}",
                f"Size: {format_size(art['size_bytes'])}",
                f"Modified: {format_time(art['modified_ts'])}",
                f"Path: {path}",
            ]
            if purpose:
                lines.extend(["", f"Purpose: {purpose}"])
            if message:
                lines.extend(["", message])
            return "\n".join(lines)

        try:
            if binary_preview_disabled(ftype, path):
                if ftype == "Video":
                    self._show_info(
                        _metadata_text(
                            "Video preview uses the system player. Use Open File, Reveal in Folder, or Export."
                        )
                    )
                    return
                self._show_info(
                    _metadata_text(
                        "Binary artifact preview is disabled. Use Open File, Reveal in Folder, or Export."
                    )
                )
                return

            if ftype == "CSV" or ftype == "Excel":
                if ftype == "Excel":
                    df = pd.read_excel(path, nrows=PREVIEW_CSV_ROWS)
                else:
                    df = pd.read_csv(path, nrows=PREVIEW_CSV_ROWS)
                self._preview_table.setRowCount(len(df))
                self._preview_table.setColumnCount(len(df.columns))
                self._preview_table.setHorizontalHeaderLabels(list(df.columns))
                for ri, row in df.iterrows():
                    for ci, val in enumerate(row):
                        self._preview_table.setItem(
                            ri, ci, QTableWidgetItem(str(val)),
                        )
                self._preview_table.setToolTip(
                    f"Preview limited to first {PREVIEW_CSV_ROWS} rows. "
                    "Use Open File or Export for the full file."
                )
                self._preview_stack.setCurrentWidget(self._preview_table)
                self._preview_details.setText(
                    "\n".join(
                        part for part in [
                            purpose,
                            f"Previewing first {min(len(df), PREVIEW_CSV_ROWS)} rows and {len(df.columns)} columns.",
                        ]
                        if part
                    )
                )
                self._preview_details.setVisible(True)

            elif ftype in {"JSON", "YAML"}:
                if art["size_bytes"] <= SMALL_JSON_BYTES:
                    with open(path, encoding="utf-8", errors="replace") as f:
                        if ftype == "JSON":
                            data = json.load(f)
                            text = json.dumps(data, indent=2)
                        else:
                            text = f.read(PREVIEW_TEXT_BYTES)
                    if len(text.encode("utf-8", errors="replace")) > PREVIEW_TEXT_BYTES:
                        text = text[:PREVIEW_TEXT_BYTES] + (
                            f"\n\nLarge file preview limited to first {PREVIEW_TEXT_BYTES} bytes. "
                            "Use Export/Open for full file."
                        )
                else:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read(PREVIEW_TEXT_BYTES)
                    text += (
                        f"\n\nLarge file preview limited to first {PREVIEW_TEXT_BYTES} bytes. "
                        "Use Export/Open for full file."
                    )
                self._preview_text.setPlainText(text)
                self._preview_stack.setCurrentWidget(self._preview_text)

            elif ftype == "Image":
                pix = QPixmap(path)
                if not pix.isNull():
                    available = self._preview_stack.size()
                    scaled = pix.scaled(
                        available.width() - 20,
                        available.height() - 20,
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                    self._preview_image.setPixmap(scaled)
                    self._preview_stack.setCurrentWidget(self._preview_image)
                else:
                    self._show_info("Cannot display image.")

            elif ftype == "Video":
                self._show_info(
                    _metadata_text("Double-click the row to open in system player.")
                )

            elif ftype == "PDF":
                self._show_info(
                    _metadata_text("Double-click the row to open in system viewer.")
                )

            elif ftype == "HTML":
                self._show_info(
                    _metadata_text("Open externally to view this HTML artifact in a browser.")
                )

            else:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    head = f.read(PREVIEW_TEXT_BYTES)
                if head.strip():
                    if art["size_bytes"] > PREVIEW_TEXT_BYTES:
                        head += (
                            f"\n\nLarge file preview limited to first {PREVIEW_TEXT_BYTES} bytes. "
                            "Use Export/Open for full file."
                        )
                    self._preview_text.setPlainText(head)
                    self._preview_stack.setCurrentWidget(self._preview_text)
                else:
                    self._show_info(_metadata_text())
        except Exception as e:
            self._show_info(f"Error previewing {ftype}:\n{e}")
        finally:
            print(
                f"[timing] Artifact preview {ftype}: "
                f"{(time.perf_counter() - t0) * 1000:.1f} ms"
            )

    def _show_info(self, text: str) -> None:
        self._preview_info.setText(text)
        self._preview_stack.setCurrentWidget(self._preview_info)

    def _update_action_buttons(self) -> None:
        has_selection = self._selected_artifact() is not None
        has_category = self._current_category() != ALL_CATEGORIES_LABEL
        for label in ("Open File", "Reveal in Folder", "Save As…", "Export Selected"):
            if label in self._action_buttons:
                self._action_buttons[label].setEnabled(has_selection)
        if "Export Category" in self._action_buttons:
            self._action_buttons["Export Category"].setEnabled(has_category)

    # ------------------------------------------------------------- actions
    def _open_file(self) -> None:
        art = self._selected_artifact()
        if art:
            QDesktopServices.openUrl(QUrl.fromLocalFile(art["abs_path"]))

    def _reveal_file(self) -> None:
        art = self._selected_artifact()
        if art:
            _open_folder(os.path.dirname(art["abs_path"]))

    def _save_as(self) -> None:
        art = self._selected_artifact()
        if not art:
            return
        dest, _ = QFileDialog.getSaveFileName(
            self, "Save As", art["name"], "All files (*.*)",
        )
        if dest:
            shutil.copy2(art["abs_path"], dest)
            QMessageBox.information(self, "Saved", f"Copied to {dest}")

    def _export_selected(self) -> None:
        selected = self._selected_artifacts()
        if not selected:
            QMessageBox.information(self, "Export", "Select files first.")
            return

        dest, _ = QFileDialog.getSaveFileName(
            self, "Export Selected", "selected_results.zip", "ZIP (*.zip)",
        )
        if not dest:
            return

        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
            for a in selected:
                zf.write(a["abs_path"], a["rel_path"])
        QMessageBox.information(
            self, "Exported", f"{len(selected)} files saved to {dest}",
        )

    def _export_category(self) -> None:
        cat = self._current_category()
        if cat == ALL_CATEGORIES_LABEL:
            QMessageBox.information(
                self, "Export Category",
                "Select a category from the sidebar first.",
            )
            return
        cat_files = self._artifacts_for_category(cat)
        if not cat_files:
            QMessageBox.information(self, "Export", f"No files in {cat}.")
            return

        dest, _ = QFileDialog.getSaveFileName(
            self, f"Export {cat}", f"{cat.lower()}_results.zip", "ZIP (*.zip)",
        )
        if not dest:
            return

        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
            for a in cat_files:
                zf.write(a["abs_path"], a["rel_path"])
        QMessageBox.information(
            self, "Exported", f"{len(cat_files)} {cat} files saved to {dest}",
        )

    def _export_all(self) -> None:
        dest, _ = QFileDialog.getSaveFileName(
            self, "Export All Results", "vieb_results.zip", "ZIP (*.zip)",
        )
        if not dest:
            return

        from _workers import ArtifactExportWorker
        self._worker = ArtifactExportWorker("all", dest)
        self._worker.done.connect(self._on_export_done)
        self._running_command = "artifact export: all"
        self.worker_running.emit(True)
        self._worker.start()

    def _export_publication(self) -> None:
        dest, _ = QFileDialog.getSaveFileName(
            self, "Export Publication Bundle",
            "publication_bundle.zip", "ZIP (*.zip)",
        )
        if not dest:
            return

        from _workers import ArtifactExportWorker
        self._worker = ArtifactExportWorker("publication", dest)
        self._worker.done.connect(self._on_export_done)
        self._running_command = "artifact export: publication"
        self.worker_running.emit(True)
        self._worker.start()

    def _on_export_done(self, ok: bool) -> None:
        self.worker_running.emit(False)
        self._running_command = ""
        QMessageBox.information(
            self, "Export",
            "Export complete." if ok else "Export failed — see log.",
        )
