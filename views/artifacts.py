from __future__ import annotations

import json
import os
import shutil
import time
import zipfile
from pathlib import Path

import pandas as pd

from PyQt5.QtCore import Qt, QThread, QTimer, QUrl, pyqtSignal
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

PREVIEW_CSV_ROWS = 100
PREVIEW_TEXT_BYTES = 64_000
SMALL_JSON_BYTES = 256_000
ROW_BATCH_SIZE = 150
BINARY_TYPES = {"Model", "NumPy", "HDF5"}
BINARY_SUFFIXES = {".pkl", ".pt", ".pth", ".ckpt", ".npy", ".npy.gz", ".h5", ".hdf5"}


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
        self._filtered: list[dict] = []
        self._worker = None
        self._pending_rows: list[dict] = []
        self._row_timer = QTimer(self)
        self._row_timer.timeout.connect(self._insert_next_rows)
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(8)

        # ── Header ───────────────────────────────────────────────────────
        hdr = QHBoxLayout()
        title = QLabel("Artifacts")
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
                  "HDF5", "Model", "Excel", "Other"):
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

    def refresh(self, data: dict | None = None) -> None:
        if isinstance(data, dict):
            self._data = data
        self._scan()

    # -------------------------------------------------------------- scanning
    def _scan(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            return
        self._summary_lbl.setText("Scanning artifacts...")
        self._row_timer.stop()
        self._pending_rows = []
        self._table.setRowCount(0)
        self._preview_stack.setCurrentWidget(self._preview_empty)
        results_dir, clips_dir = self._current_artifact_roots()
        self._worker = ArtifactScanWorker(str(results_dir), str(clips_dir) if clips_dir else None)
        self._worker.done.connect(self._on_scan_done)
        self._worker.failed.connect(self._on_scan_failed)
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
        self._summary_lbl.setText("Artifact scan failed")
        self._show_info(f"Artifact scan failed:\n{message}")

    def _on_scan_done(self, artifacts: list[dict]) -> None:
        self.worker_running.emit(False)
        self._artifacts = artifacts

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
        self._row_timer.stop()
        self._pending_rows = list(artifacts)
        self._table.setSortingEnabled(False)
        self._table.setRowCount(0)
        self._insert_next_rows()
        if self._pending_rows:
            self._summary_lbl.setText(
                f"Loading {min(len(artifacts), ROW_BATCH_SIZE)} of {len(artifacts)} files..."
            )
            self._row_timer.start(0)
        else:
            self._table.setSortingEnabled(True)

    def _insert_next_rows(self) -> None:
        if not self._pending_rows:
            self._row_timer.stop()
            self._table.setSortingEnabled(True)
            return

        batch = self._pending_rows[:ROW_BATCH_SIZE]
        del self._pending_rows[:ROW_BATCH_SIZE]
        start = self._table.rowCount()
        self._table.setRowCount(start + len(batch))

        for offset, a in enumerate(batch):
            ri = start + offset
            self._table.setItem(ri, 0, QTableWidgetItem(a["name"]))
            self._table.setItem(ri, 1, QTableWidgetItem(a["category"]))
            self._table.setItem(ri, 2, QTableWidgetItem(a["file_type"]))
            size_item = QTableWidgetItem(format_size(a["size_bytes"]))
            size_item.setData(Qt.UserRole, a["size_bytes"])
            self._table.setItem(ri, 3, size_item)
            self._table.setItem(ri, 4, QTableWidgetItem(format_time(a["modified_ts"])))
            self._table.setItem(ri, 5, QTableWidgetItem(a["rel_path"]))

        total_size = sum(a["size_bytes"] for a in self._filtered)
        if self._pending_rows:
            loaded = len(self._filtered) - len(self._pending_rows)
            self._summary_lbl.setText(
                f"Loading {loaded} of {len(self._filtered)} files, {format_size(total_size)}"
            )
        else:
            self._row_timer.stop()
            self._table.setSortingEnabled(True)
            self._summary_lbl.setText(
                f"{len(self._filtered)} files, {format_size(total_size)}"
            )

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
        t0 = time.perf_counter()
        ftype = art["file_type"]
        path = art["abs_path"]
        path_obj = Path(path)
        suffixes = "".join(s.lower() for s in path_obj.suffixes)

        def _metadata_text(message: str | None = None) -> str:
            lines = [
                f"File: {art['name']}",
                f"Type: {ftype}",
                f"Size: {format_size(art['size_bytes'])}",
                f"Modified: {format_time(art['modified_ts'])}",
                f"Path: {path}",
            ]
            if message:
                lines.extend(["", message])
            return "\n".join(lines)

        try:
            if ftype in BINARY_TYPES or any(suffixes.endswith(s) for s in BINARY_SUFFIXES):
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

            elif ftype == "JSON":
                if art["size_bytes"] <= SMALL_JSON_BYTES:
                    with open(path, encoding="utf-8") as f:
                        data = json.load(f)
                    text = json.dumps(data, indent=2)
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
