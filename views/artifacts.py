from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd

from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QDesktopServices, QFont, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMessageBox, QPushButton,
    QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget,
)

from _utils import CLIPS, RESULTS, _open_folder
from artifact_scanner import (
    scan_artifacts, build_publication_bundle, format_size, format_time,
)


class ArtifactsView(QWidget):
    worker_running = pyqtSignal(bool)

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.cfg = cfg or {}
        self._data: dict = {}
        self._artifacts: list[dict] = []
        self._filtered: list[dict] = []
        self._worker = None
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(8)

        # ── Header ───────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        title = QLabel("Results Browser")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        hdr.addWidget(title)
        hdr.addStretch()
        self._summary_lbl = QLabel("")
        self._summary_lbl.setStyleSheet("color:#666; font-size:11px;")
        hdr.addWidget(self._summary_lbl)
        root.addLayout(hdr)

        # ── Filter bar ───────────────────────────────────────────────────
        filt = QHBoxLayout()
        filt.addWidget(QLabel("Search:"))
        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter by name or path...")
        self._search.setClearButtonEnabled(True)
        self._search.textChanged.connect(self._apply_filters)
        self._search.setFixedWidth(220)
        filt.addWidget(self._search)

        filt.addWidget(QLabel("Category:"))
        self._cat_filter = QComboBox()
        self._cat_filter.addItem("All")
        self._cat_filter.currentTextChanged.connect(self._apply_filters)
        filt.addWidget(self._cat_filter)

        filt.addWidget(QLabel("Type:"))
        self._type_filter = QComboBox()
        self._type_filter.addItem("All")
        for t in ("CSV", "JSON", "Image", "Video", "PDF", "NumPy",
                  "Model", "Excel", "Other"):
            self._type_filter.addItem(t)
        self._type_filter.currentTextChanged.connect(self._apply_filters)
        filt.addWidget(self._type_filter)

        filt.addStretch()
        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedHeight(28)
        refresh_btn.clicked.connect(self._scan)
        filt.addWidget(refresh_btn)
        root.addLayout(filt)

        # ── Splitter: file table + preview ───────────────────────────────
        splitter = QSplitter(Qt.Vertical)

        self._table = QTableWidget(0, 6)
        self._table.setHorizontalHeaderLabels(
            ["Name", "Category", "Type", "Size", "Modified", "Path"]
        )
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in (1, 2, 3, 4):
            self._table.horizontalHeader().setSectionResizeMode(
                col, QHeaderView.ResizeToContents,
            )
        self._table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSortingEnabled(True)
        self._table.currentCellChanged.connect(self._on_selection_changed)
        self._table.doubleClicked.connect(lambda _: self._open_file())
        splitter.addWidget(self._table)

        # Preview pane
        self._preview_stack = QStackedWidget()

        self._preview_empty = QLabel("Select a file to preview")
        self._preview_empty.setAlignment(Qt.AlignCenter)
        self._preview_empty.setStyleSheet(
            "color:#999; font-style:italic; padding:20px;"
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
        self._preview_image.setStyleSheet("background:#f0f0f0; padding:8px;")
        self._preview_stack.addWidget(self._preview_image)

        self._preview_info = QLabel()
        self._preview_info.setAlignment(Qt.AlignCenter)
        self._preview_info.setWordWrap(True)
        self._preview_info.setStyleSheet("color:#555; padding:20px;")
        self._preview_stack.addWidget(self._preview_info)

        self._preview_stack.setCurrentWidget(self._preview_empty)
        splitter.addWidget(self._preview_stack)
        splitter.setSizes([400, 200])
        root.addWidget(splitter, stretch=1)

        # ── Bottom buttons ───────────────────────────────────────────────
        btn_row = QHBoxLayout()
        for label, slot in [
            ("Open File", self._open_file),
            ("Reveal in Folder", self._reveal_file),
            ("Save As…", self._save_as),
            ("Export Selected", self._export_selected),
            ("Export Category", self._export_category),
            ("Export All as ZIP", self._export_all),
            ("Publication Bundle", self._export_publication),
        ]:
            btn = QPushButton(label)
            btn.setFixedHeight(30)
            btn.clicked.connect(slot)
            btn_row.addWidget(btn)
        root.addLayout(btn_row)

    # ----------------------------------------------------------- data hooks
    def update_data(self, data: dict) -> None:
        self._data = data
        self._scan()

    def refresh(self, data: dict) -> None:
        self._data = data
        self._scan()

    # -------------------------------------------------------------- scanning
    def _scan(self) -> None:
        clips_dir = str(CLIPS) if CLIPS.is_dir() else None
        self._artifacts = scan_artifacts(str(RESULTS), clips_dir=clips_dir)

        categories = sorted(set(a["category"] for a in self._artifacts))
        current = self._cat_filter.currentText()
        self._cat_filter.blockSignals(True)
        self._cat_filter.clear()
        self._cat_filter.addItem("All")
        for c in categories:
            self._cat_filter.addItem(c)
        if current in categories or current == "All":
            self._cat_filter.setCurrentText(current)
        self._cat_filter.blockSignals(False)

        self._apply_filters()

    # ------------------------------------------------------------ filtering
    def _apply_filters(self) -> None:
        search = self._search.text().strip().lower()
        cat = self._cat_filter.currentText()
        ftype = self._type_filter.currentText()

        filtered = self._artifacts
        if cat != "All":
            filtered = [a for a in filtered if a["category"] == cat]
        if ftype != "All":
            filtered = [a for a in filtered if a["file_type"] == ftype]
        if search:
            filtered = [
                a for a in filtered
                if search in a["name"].lower() or search in a["rel_path"].lower()
            ]

        self._filtered = filtered
        self._populate_table(filtered)

        total_size = sum(a["size_bytes"] for a in filtered)
        self._summary_lbl.setText(
            f"{len(filtered)} files, {format_size(total_size)}"
        )

    def _populate_table(self, artifacts: list[dict]) -> None:
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(artifacts))
        for ri, a in enumerate(artifacts):
            self._table.setItem(ri, 0, QTableWidgetItem(a["name"]))
            self._table.setItem(ri, 1, QTableWidgetItem(a["category"]))
            self._table.setItem(ri, 2, QTableWidgetItem(a["file_type"]))
            size_item = QTableWidgetItem(format_size(a["size_bytes"]))
            size_item.setData(Qt.UserRole, a["size_bytes"])
            self._table.setItem(ri, 3, size_item)
            self._table.setItem(ri, 4, QTableWidgetItem(format_time(a["modified_ts"])))
            self._table.setItem(ri, 5, QTableWidgetItem(a["rel_path"]))
        self._table.setSortingEnabled(True)

    # ----------------------------------------------------------- selection
    def _selected_artifact(self) -> dict | None:
        row = self._table.currentRow()
        if row < 0:
            return None
        rel_item = self._table.item(row, 5)
        if not rel_item:
            return None
        rel_path = rel_item.text()
        for a in self._artifacts:
            if a["rel_path"] == rel_path:
                return a
        return None

    def _selected_artifacts(self) -> list[dict]:
        rows = sorted(set(idx.row() for idx in self._table.selectedIndexes()))
        out: list[dict] = []
        for row in rows:
            rel_item = self._table.item(row, 5)
            if not rel_item:
                continue
            rel = rel_item.text()
            for a in self._artifacts:
                if a["rel_path"] == rel:
                    out.append(a)
                    break
        return out

    def _on_selection_changed(
        self, row: int, _col: int, _prev_row: int, _prev_col: int,
    ) -> None:
        art = self._selected_artifact()
        if not art:
            self._preview_stack.setCurrentWidget(self._preview_empty)
            return
        self._preview_file(art)

    # ------------------------------------------------------------- preview
    def _preview_file(self, art: dict) -> None:
        ftype = art["file_type"]
        path = art["abs_path"]

        if ftype == "CSV" or ftype == "Excel":
            try:
                if ftype == "Excel":
                    df = pd.read_excel(path, nrows=20)
                else:
                    df = pd.read_csv(path, nrows=20)
                self._preview_table.setRowCount(len(df))
                self._preview_table.setColumnCount(len(df.columns))
                self._preview_table.setHorizontalHeaderLabels(list(df.columns))
                for ri, row in df.iterrows():
                    for ci, val in enumerate(row):
                        self._preview_table.setItem(
                            ri, ci, QTableWidgetItem(str(val)),
                        )
                self._preview_stack.setCurrentWidget(self._preview_table)
            except Exception as e:
                self._show_info(f"Error reading {ftype}:\n{e}")

        elif ftype == "JSON":
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                text = json.dumps(data, indent=2)
                if len(text) > 12_000:
                    text = text[:12_000] + "\n\n... (truncated)"
                self._preview_text.setPlainText(text)
                self._preview_stack.setCurrentWidget(self._preview_text)
            except Exception as e:
                self._show_info(f"Error reading JSON:\n{e}")

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
                f"Video: {art['name']}\n"
                f"Size: {format_size(art['size_bytes'])}\n\n"
                "Double-click the row to open in system player."
            )

        elif ftype == "PDF":
            self._show_info(
                f"PDF: {art['name']}\n"
                f"Size: {format_size(art['size_bytes'])}\n\n"
                "Double-click the row to open in system viewer."
            )

        else:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    head = f.read(4000)
                if head.strip():
                    self._preview_text.setPlainText(head)
                    self._preview_stack.setCurrentWidget(self._preview_text)
                else:
                    self._show_info(
                        f"{ftype} file: {art['name']}\n"
                        f"Size: {format_size(art['size_bytes'])}\n"
                        f"Modified: {format_time(art['modified_ts'])}"
                    )
            except Exception:
                self._show_info(
                    f"{ftype} file: {art['name']}\n"
                    f"Size: {format_size(art['size_bytes'])}\n"
                    f"Modified: {format_time(art['modified_ts'])}"
                )

    def _show_info(self, text: str) -> None:
        self._preview_info.setText(text)
        self._preview_stack.setCurrentWidget(self._preview_info)

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
        cat = self._cat_filter.currentText()
        if cat == "All":
            QMessageBox.information(
                self, "Export Category",
                "Select a category from the filter first.",
            )
            return
        cat_files = [a for a in self._artifacts if a["category"] == cat]
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
        self.worker_running.emit(True)
        self._worker.start()

    def _on_export_done(self, ok: bool) -> None:
        self.worker_running.emit(False)
        QMessageBox.information(
            self, "Export",
            "Export complete." if ok else "Export failed — see log.",
        )
