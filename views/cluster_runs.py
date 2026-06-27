from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QAbstractItemView, QFrame, QHBoxLayout, QHeaderView, QInputDialog,
    QLabel, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSpinBox,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from _utils import ROOT, RESULTS, _save_cfg, _load_cfg
from cluster_run_manager import (
    ClusterRunConfig,
    ClusterRunManifest,
    ClusterRunManager,
)


def _manager() -> ClusterRunManager:
    cfg_path = ROOT / "config.json"
    return ClusterRunManager(RESULTS, config_path=cfg_path)


def _load_manifest(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Individual run card
# ---------------------------------------------------------------------------

class _RunCard(QFrame):
    """Card widget for one run (current or saved)."""

    save_requested = pyqtSignal()
    delete_requested = pyqtSignal()
    activate_requested = pyqtSignal()
    rename_requested = pyqtSignal()

    def __init__(self, manifest: dict, is_current: bool = False, parent=None):
        super().__init__(parent)
        self.manifest = manifest
        self.is_current = is_current
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame{background:#FFFFFF;border:1px solid #E5E5E5;border-radius:6px;padding:4px;}"
        )
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)

        m = self.manifest
        run_id = m.get("run_id", "—")
        date = m.get("date", "—")
        n_clusters = m.get("n_clusters", "—")
        mean_conf = m.get("mean_confidence", None)
        noise_frac = m.get("noise_frac", None)
        mcs = m.get("min_cluster_size", "—")
        umap_dims = m.get("umap_dims", "—")
        ms_req = m.get("min_samples_requested", None)
        ms_res = m.get("min_samples_resolved", m.get("hdbscan_min_samples", 0))
        saved = bool(m.get("saved", False))

        id_row = QHBoxLayout()
        id_lbl = QLabel(run_id)
        id_lbl.setFont(QFont("Consolas", 11, QFont.Bold))
        id_row.addWidget(id_lbl)
        id_row.addStretch()
        if self.is_current and saved:
            ck = QLabel("✓ saved")
            ck.setStyleSheet("color:#2e7d32;font-weight:600;font-size:11px;")
            id_row.addWidget(ck)
        lay.addLayout(id_row)

        meta_parts = [f"Date: {date}", f"Clusters: {n_clusters}"]
        if mean_conf is not None:
            meta_parts.append(f"Conf: {float(mean_conf):.3f}")
        if noise_frac is not None:
            meta_parts.append(f"Noise: {float(noise_frac) * 100:.1f}%")
        lay.addWidget(QLabel("  ·  ".join(meta_parts)))

        param_parts = [f"mcs={mcs}", f"umap={umap_dims}"]
        if ms_req is not None and ms_req == 0:
            param_parts.append(f"min_samples=Auto (→{ms_res})")
        elif ms_res:
            param_parts.append(f"min_samples={ms_res}")
        param_lbl = QLabel("  ·  ".join(param_parts))
        param_lbl.setStyleSheet("color:#666;font-size:11px;")
        lay.addWidget(param_lbl)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        _btn_style = (
            "QPushButton{border:1px solid #bbb;border-radius:4px;padding:0 12px;background:#f5f5f5;}"
            "QPushButton:hover{background:#e0e0e0;}"
        )

        if self.is_current:
            save_btn = QPushButton("Save")
            save_btn.setFixedHeight(26)
            save_btn.setStyleSheet(
                "QPushButton{background:#2e7d32;color:#fff;border:none;border-radius:4px;padding:0 12px;font-weight:600;}"
                "QPushButton:hover{background:#1b5e20;}"
                "QPushButton:disabled{background:#a5d6a7;color:#fff;}"
            )
            save_btn.setEnabled(not saved)
            save_btn.clicked.connect(self.save_requested.emit)
            btn_row.addWidget(save_btn)

            rename_btn = QPushButton("Rename")
            rename_btn.setFixedHeight(26)
            rename_btn.setStyleSheet(_btn_style)
            rename_btn.clicked.connect(self.rename_requested.emit)
            btn_row.addWidget(rename_btn)

            del_btn = QPushButton("Delete")
            del_btn.setFixedHeight(26)
            del_btn.setStyleSheet(
                "QPushButton{background:#c62828;color:#fff;border:none;border-radius:4px;padding:0 12px;}"
                "QPushButton:hover{background:#b71c1c;}"
            )
            del_btn.clicked.connect(self.delete_requested.emit)
            btn_row.addWidget(del_btn)
        else:
            act_btn = QPushButton("Set Active")
            act_btn.setFixedHeight(26)
            act_btn.setStyleSheet(
                "QPushButton{background:#1565c0;color:#fff;border:none;border-radius:4px;padding:0 12px;font-weight:600;}"
                "QPushButton:hover{background:#0d47a1;}"
            )
            act_btn.clicked.connect(self.activate_requested.emit)
            btn_row.addWidget(act_btn)

            rename_btn = QPushButton("Rename")
            rename_btn.setFixedHeight(26)
            rename_btn.setStyleSheet(_btn_style)
            rename_btn.clicked.connect(self.rename_requested.emit)
            btn_row.addWidget(rename_btn)

            del_btn = QPushButton("Delete")
            del_btn.setFixedHeight(26)
            del_btn.setStyleSheet(
                "QPushButton{background:#c62828;color:#fff;border:none;border-radius:4px;padding:0 12px;}"
                "QPushButton:hover{background:#b71c1c;}"
            )
            del_btn.clicked.connect(self.delete_requested.emit)
            btn_row.addWidget(del_btn)

        btn_row.addStretch()
        lay.addLayout(btn_row)


# ---------------------------------------------------------------------------
# Run Queue Table
# ---------------------------------------------------------------------------

class _QueueTable(QFrame):
    """Editable table of ClusterRunConfig objects for queued execution."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame{background:#FFFFFF;border:1px solid #E5E5E5;border-radius:6px;}"
        )
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        hdr = QLabel("Run Queue")
        hdr.setFont(QFont("Arial", 13, QFont.Bold))
        hdr.setStyleSheet("color:#444;")
        lay.addWidget(hdr)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels([
            "min_cluster_size", "min_samples", "umap_dims", "hdbscan_sample",
        ])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setFixedHeight(160)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        lay.addWidget(self._table)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        _btn_style = (
            "QPushButton{border:1px solid #bbb;border-radius:4px;padding:4px 12px;background:#f5f5f5;}"
            "QPushButton:hover{background:#e0e0e0;}"
            "QPushButton:disabled{background:#eee;color:#aaa;}"
        )

        add_btn = QPushButton("Add Config")
        add_btn.setStyleSheet(_btn_style)
        add_btn.clicked.connect(self._add_row)
        btn_row.addWidget(add_btn)

        dup_btn = QPushButton("Duplicate")
        dup_btn.setStyleSheet(_btn_style)
        dup_btn.clicked.connect(self._duplicate_row)
        btn_row.addWidget(dup_btn)

        rm_btn = QPushButton("Remove")
        rm_btn.setStyleSheet(_btn_style)
        rm_btn.clicked.connect(self._remove_row)
        btn_row.addWidget(rm_btn)

        btn_row.addStretch()
        lay.addLayout(btn_row)

    def _add_row(self, mcs=2000, ms=0, umap=10, sample=300000):
        r = self._table.rowCount()
        self._table.insertRow(r)
        self._table.setItem(r, 0, QTableWidgetItem(str(mcs)))
        auto_item = QTableWidgetItem("Auto" if ms == 0 else str(ms))
        self._table.setItem(r, 1, auto_item)
        self._table.setItem(r, 2, QTableWidgetItem(str(umap)))
        self._table.setItem(r, 3, QTableWidgetItem(str(sample)))

    def _duplicate_row(self):
        row = self._table.currentRow()
        if row < 0:
            return
        vals = self._read_row(row)
        if vals:
            self._add_row(*vals)

    def _remove_row(self):
        row = self._table.currentRow()
        if row >= 0:
            self._table.removeRow(row)

    def _read_row(self, row: int) -> tuple | None:
        try:
            mcs = int(self._table.item(row, 0).text())
            ms_text = self._table.item(row, 1).text().strip()
            ms = 0 if ms_text.lower() == "auto" else int(ms_text)
            umap = int(self._table.item(row, 2).text())
            sample = int(self._table.item(row, 3).text())
            return mcs, ms, umap, sample
        except (ValueError, AttributeError):
            return None

    def get_configs(self) -> list[ClusterRunConfig]:
        configs = []
        for r in range(self._table.rowCount()):
            vals = self._read_row(r)
            if vals:
                mcs, ms, umap, sample = vals
                configs.append(ClusterRunConfig(
                    min_cluster_size=mcs,
                    min_samples=ms,
                    umap_dims=umap,
                    hdbscan_sample=sample,
                ))
        return configs

    def set_enabled(self, enabled: bool):
        self._table.setEnabled(enabled)


# ---------------------------------------------------------------------------
# Run Comparison Table
# ---------------------------------------------------------------------------

class _ComparisonTable(QFrame):
    """Read-only table comparing all saved runs."""

    activate_requested = pyqtSignal(str)

    _COLUMNS = [
        ("run_id", "Run ID"),
        ("date", "Date"),
        ("status", "Status"),
        ("n_clusters", "States"),
        ("min_cluster_size", "MCS"),
        ("min_samples_requested", "MS Req"),
        ("min_samples_resolved", "MS Res"),
        ("umap_dims", "UMAP"),
        ("hdbscan_sample", "Sample"),
        ("noise_frac", "Noise %"),
        ("largest_state_occupancy", "Largest %"),
        ("mean_confidence", "Conf"),
        ("health_status", "Health"),
        ("warnings_count", "Warns"),
        ("assignment_method", "Assign"),
        ("runtime_seconds", "Time (s)"),
        ("notes", "Notes"),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame{background:#FFFFFF;border:1px solid #E5E5E5;border-radius:6px;}"
        )
        self._active_run = ""
        self._build()

    def _build(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(6)

        hdr_row = QHBoxLayout()
        hdr = QLabel("Run Comparison")
        hdr.setFont(QFont("Arial", 13, QFont.Bold))
        hdr.setStyleSheet("color:#444;")
        hdr_row.addWidget(hdr)
        hdr_row.addStretch()

        self._activate_btn = QPushButton("Set Active")
        self._activate_btn.setFixedHeight(26)
        self._activate_btn.setStyleSheet(
            "QPushButton{background:#1565c0;color:#fff;border:none;border-radius:4px;padding:0 12px;font-weight:600;}"
            "QPushButton:hover{background:#0d47a1;}"
            "QPushButton:disabled{background:#90caf9;color:#fff;}"
        )
        self._activate_btn.setEnabled(False)
        self._activate_btn.clicked.connect(self._on_activate_clicked)
        hdr_row.addWidget(self._activate_btn)
        lay.addLayout(hdr_row)

        col_labels = [c[1] for c in self._COLUMNS]
        self._table = QTableWidget(0, len(col_labels))
        self._table.setHorizontalHeaderLabels(col_labels)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for i in range(1, len(col_labels)):
            self._table.horizontalHeader().setSectionResizeMode(i, QHeaderView.ResizeToContents)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setMinimumHeight(120)
        self._table.currentCellChanged.connect(self._on_selection_changed)
        lay.addWidget(self._table)

    def refresh(self, manifests: list[ClusterRunManifest], active_run: str):
        self._active_run = active_run
        self._manifests = manifests
        self._table.setRowCount(0)

        for m in manifests:
            row = self._table.rowCount()
            self._table.insertRow(row)
            d = m.to_dict()

            for col_idx, (key, _label) in enumerate(self._COLUMNS):
                val = d.get(key, "")
                if key == "noise_frac" and isinstance(val, (int, float)):
                    text = f"{val * 100:.1f}%"
                elif key == "largest_state_occupancy" and isinstance(val, (int, float)):
                    text = f"{val * 100:.1f}%"
                elif key == "mean_confidence" and isinstance(val, (int, float)):
                    text = f"{val:.3f}"
                elif key == "runtime_seconds" and isinstance(val, (int, float)) and val > 0:
                    text = f"{val:.0f}"
                elif key == "min_samples_requested" and val == 0:
                    text = "Auto"
                else:
                    text = str(val) if val else ""

                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                if m.run_id == active_run:
                    item.setBackground(QColor("#E3F2FD"))

                if key == "health_status":
                    if val == "failed":
                        item.setForeground(QColor("#c62828"))
                    elif val == "suspicious":
                        item.setForeground(QColor("#e65100"))
                    elif val == "good":
                        item.setForeground(QColor("#2e7d32"))

                self._table.setItem(row, col_idx, item)

        self._activate_btn.setEnabled(False)

    def _on_selection_changed(self, row, _col, _prev_row, _prev_col):
        if 0 <= row < len(self._manifests):
            m = self._manifests[row]
            self._activate_btn.setEnabled(
                m.status == "completed" and m.run_id != self._active_run
            )
        else:
            self._activate_btn.setEnabled(False)

    def _on_activate_clicked(self):
        row = self._table.currentRow()
        if 0 <= row < len(self._manifests):
            self.activate_requested.emit(self._manifests[row].run_id)


# ---------------------------------------------------------------------------
# Main view
# ---------------------------------------------------------------------------

class ClusterRunsView(QWidget):
    """View for browsing, saving, and restoring versioned cluster runs."""

    run_activated = pyqtSignal()
    cluster_changed = pyqtSignal()
    queue_start_requested = pyqtSignal(list)  # list[dict] of configs

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._queue_worker = None
        self._build()
        self.refresh()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(12)

        title = QLabel("Cluster Runs")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        outer.addWidget(title)

        # --- Queue section ---
        self._queue_table = _QueueTable()
        outer.addWidget(self._queue_table)

        queue_action_row = QHBoxLayout()
        queue_action_row.setSpacing(6)

        self._start_queue_btn = QPushButton("Start Queue")
        self._start_queue_btn.setFixedHeight(30)
        self._start_queue_btn.setStyleSheet(
            "QPushButton{background:#2e7d32;color:#fff;border:none;border-radius:4px;padding:0 16px;font-weight:600;}"
            "QPushButton:hover{background:#1b5e20;}"
            "QPushButton:disabled{background:#a5d6a7;color:#fff;}"
        )
        self._start_queue_btn.clicked.connect(self._start_queue)
        queue_action_row.addWidget(self._start_queue_btn)

        self._stop_queue_btn = QPushButton("Stop After Current")
        self._stop_queue_btn.setFixedHeight(30)
        self._stop_queue_btn.setStyleSheet(
            "QPushButton{background:#e65100;color:#fff;border:none;border-radius:4px;padding:0 16px;font-weight:600;}"
            "QPushButton:hover{background:#bf360c;}"
            "QPushButton:disabled{background:#ffcc80;color:#fff;}"
        )
        self._stop_queue_btn.setEnabled(False)
        self._stop_queue_btn.clicked.connect(self._stop_queue)
        queue_action_row.addWidget(self._stop_queue_btn)

        queue_action_row.addStretch()

        self._queue_progress = QLabel("")
        self._queue_progress.setStyleSheet("color:#555;font-style:italic;")
        queue_action_row.addWidget(self._queue_progress)

        outer.addLayout(queue_action_row)

        # --- Divider ---
        div1 = QFrame()
        div1.setFrameShape(QFrame.HLine)
        div1.setStyleSheet("color:#E5E5E5;background:#E5E5E5;border:none;max-height:1px;")
        outer.addWidget(div1)

        # --- Current run ---
        self._current_area = QWidget()
        self._current_layout = QVBoxLayout(self._current_area)
        self._current_layout.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._current_area)

        # --- Divider ---
        div2 = QFrame()
        div2.setFrameShape(QFrame.HLine)
        div2.setStyleSheet("color:#E5E5E5;background:#E5E5E5;border:none;max-height:1px;")
        outer.addWidget(div2)

        # --- Comparison table ---
        self._comparison = _ComparisonTable()
        self._comparison.activate_requested.connect(self._activate_run)
        outer.addWidget(self._comparison)

        # --- Divider ---
        div3 = QFrame()
        div3.setFrameShape(QFrame.HLine)
        div3.setStyleSheet("color:#E5E5E5;background:#E5E5E5;border:none;max-height:1px;")
        outer.addWidget(div3)

        # --- Saved runs (card list) ---
        saved_hdr = QLabel("Saved Runs")
        saved_hdr.setFont(QFont("Arial", 13, QFont.Bold))
        saved_hdr.setStyleSheet("color:#444;")
        outer.addWidget(saved_hdr)

        self._no_saved_lbl = QLabel("No saved runs yet. Click Save on the current run to preserve it.")
        self._no_saved_lbl.setStyleSheet("color:#888;font-style:italic;")
        outer.addWidget(self._no_saved_lbl)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self._saved_host = QWidget()
        self._saved_vl = QVBoxLayout(self._saved_host)
        self._saved_vl.setContentsMargins(0, 0, 0, 0)
        self._saved_vl.setSpacing(8)
        self._saved_vl.addStretch()
        scroll.setWidget(self._saved_host)
        outer.addWidget(scroll, stretch=1)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def refresh(self):
        self._refresh_current()
        self._refresh_saved()
        self._refresh_comparison()

    def set_queue_running(self, running: bool):
        self._start_queue_btn.setEnabled(not running)
        self._stop_queue_btn.setEnabled(running)
        self._queue_table.set_enabled(not running)

    def set_queue_progress(self, index: int, total: int):
        self._queue_progress.setText(f"Running {index + 1}/{total}...")

    def clear_queue_progress(self):
        self._queue_progress.setText("")

    # ------------------------------------------------------------------
    # Current run
    # ------------------------------------------------------------------

    def _refresh_current(self):
        lay = self._current_layout
        while lay.count():
            item = lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        shared = RESULTS / "shared"
        manifest_path = shared / "run_manifest.json"
        if not manifest_path.exists():
            lbl = QLabel("No current run. Run --cluster to create one.")
            lbl.setStyleSheet("color:#888;font-style:italic;")
            lay.addWidget(lbl)
            return

        manifest = _load_manifest(manifest_path)
        if manifest is None:
            lbl = QLabel("Could not read current run manifest.")
            lbl.setStyleSheet("color:#c62828;")
            lay.addWidget(lbl)
            return

        hdr = QLabel("Current Run")
        hdr.setFont(QFont("Arial", 13, QFont.Bold))
        hdr.setStyleSheet("color:#444;")
        lay.addWidget(hdr)

        card = _RunCard(manifest, is_current=True)
        card.save_requested.connect(self._save_current_run)
        card.delete_requested.connect(self._delete_current_run)
        card.rename_requested.connect(self._rename_current_run)
        lay.addWidget(card)

    # ------------------------------------------------------------------
    # Saved runs list
    # ------------------------------------------------------------------

    def _refresh_saved(self):
        vl = self._saved_vl
        while vl.count() > 1:
            item = vl.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        mgr = _manager()
        all_runs = mgr.list_runs()
        saved_runs = [(m.run_id, m.to_dict()) for m in all_runs if m.saved]

        if not saved_runs:
            self._no_saved_lbl.show()
        else:
            self._no_saved_lbl.hide()

        for run_name, manifest in saved_runs:
            card = _RunCard(manifest, is_current=False)
            card.activate_requested.connect(lambda _rn=run_name: self._activate_run(_rn))
            card.delete_requested.connect(lambda _rn=run_name: self._delete_saved_run(_rn))
            card.rename_requested.connect(lambda _rn=run_name: self._rename_saved_run(_rn))
            vl.insertWidget(vl.count() - 1, card)

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------

    def _refresh_comparison(self):
        mgr = _manager()
        manifests = mgr.list_runs()
        active = mgr.get_active_run()
        self._comparison.refresh(manifests, active)

    # ------------------------------------------------------------------
    # Queue actions
    # ------------------------------------------------------------------

    def _start_queue(self):
        configs = self._queue_table.get_configs()
        if not configs:
            QMessageBox.warning(self, "Empty Queue", "Add at least one configuration to the queue.")
            return
        self.queue_start_requested.emit([c.to_dict() for c in configs])

    def _stop_queue(self):
        if self._queue_worker:
            self._queue_worker.stop_after_current()
        self._stop_queue_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _save_current_run(self):
        shared = RESULTS / "shared"
        manifest_path = shared / "run_manifest.json"
        if not manifest_path.exists():
            return

        manifest = _load_manifest(manifest_path)
        if manifest is None:
            return

        manifest["saved"] = True
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        run_id = manifest.get("run_id", "")
        if run_id:
            mgr = _manager()
            mgr.save_run(run_id)

        self.cfg["current_run_saved"] = True
        _save_cfg(self.cfg)

        QMessageBox.information(self, "Run Saved", f"Run saved as {run_id}.")
        self.refresh()

    def _delete_current_run(self):
        reply = QMessageBox.question(
            self, "Delete Current Run",
            "Delete the current unsaved run? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        shared = RESULTS / "shared"
        cluster_files = (
            list(shared.glob("*.npy"))
            + list(shared.glob("*.pkl"))
            + [shared / "cluster_info.json", shared / "run_manifest.json",
               shared / "validation_report.json"]
        )
        for p in cluster_files:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

        self.cfg["current_run_saved"] = False
        self.cfg["current_run_id"] = ""
        self.cfg["active_cluster_run"] = ""
        _save_cfg(self.cfg)
        self.refresh()

    def _activate_run(self, run_name: str):
        mgr = _manager()
        try:
            mgr.set_active_run(run_name)
        except FileNotFoundError:
            QMessageBox.warning(self, "Not Found", f"Run directory not found: {run_name}")
            return

        manifest = mgr.load_run_manifest(run_name)
        run_id = manifest.run_id if manifest else run_name
        self.cfg["current_run_id"] = run_id
        self.cfg["active_cluster_run"] = run_id
        self.cfg["current_run_saved"] = True
        _save_cfg(self.cfg)

        self.refresh()
        self.run_activated.emit()
        self.cluster_changed.emit()

    def _rename_current_run(self):
        shared = RESULTS / "shared"
        manifest_path = shared / "run_manifest.json"
        if not manifest_path.exists():
            return
        manifest = _load_manifest(manifest_path)
        if manifest is None:
            return

        old_id = manifest.get("run_id", "")
        new_name, ok = QInputDialog.getText(
            self, "Rename Run", "New name:", text=old_id
        )
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        if new_name == old_id:
            return

        runs_dir = RESULTS / "runs"
        if (runs_dir / new_name).exists():
            QMessageBox.warning(self, "Name Taken", f"A saved run named '{new_name}' already exists.")
            return

        manifest["run_id"] = new_name
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        old_dir = runs_dir / old_id
        if old_dir.is_dir():
            new_dir = runs_dir / new_name
            old_dir.rename(new_dir)
            saved_manifest_path = new_dir / "run_manifest.json"
            saved_manifest = _load_manifest(saved_manifest_path)
            if saved_manifest is not None:
                saved_manifest["run_id"] = new_name
                saved_manifest_path.write_text(json.dumps(saved_manifest, indent=2), encoding="utf-8")

        if self.cfg.get("current_run_id") == old_id:
            self.cfg["current_run_id"] = new_name
            _save_cfg(self.cfg)
        if self.cfg.get("active_cluster_run") == old_id:
            self.cfg["active_cluster_run"] = new_name
            _save_cfg(self.cfg)

        self.refresh()

    def _rename_saved_run(self, run_name: str):
        runs_dir = RESULTS / "runs"
        run_dir = runs_dir / run_name
        if not run_dir.is_dir():
            QMessageBox.warning(self, "Not Found", f"Run directory not found: {run_name}")
            return

        new_name, ok = QInputDialog.getText(
            self, "Rename Run", "New name:", text=run_name
        )
        if not ok or not new_name.strip():
            return
        new_name = new_name.strip()
        if new_name == run_name:
            return

        if (runs_dir / new_name).exists():
            QMessageBox.warning(self, "Name Taken", f"A run named '{new_name}' already exists.")
            return

        new_dir = runs_dir / new_name
        run_dir.rename(new_dir)

        saved_manifest_path = new_dir / "run_manifest.json"
        saved_manifest = _load_manifest(saved_manifest_path)
        if saved_manifest is not None:
            saved_manifest["run_id"] = new_name
            saved_manifest_path.write_text(json.dumps(saved_manifest, indent=2), encoding="utf-8")

        current_manifest_path = RESULTS / "shared" / "run_manifest.json"
        current_manifest = _load_manifest(current_manifest_path)
        if current_manifest and current_manifest.get("run_id") == run_name:
            current_manifest["run_id"] = new_name
            current_manifest_path.write_text(json.dumps(current_manifest, indent=2), encoding="utf-8")

        if self.cfg.get("current_run_id") == run_name:
            self.cfg["current_run_id"] = new_name
            _save_cfg(self.cfg)
        if self.cfg.get("active_cluster_run") == run_name:
            self.cfg["active_cluster_run"] = new_name
            _save_cfg(self.cfg)

        self.refresh()

    def _delete_saved_run(self, run_name: str):
        reply = QMessageBox.question(
            self, "Delete Saved Run",
            f"Delete run '{run_name}'? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        mgr = _manager()
        mgr.delete_run(run_name)

        current_run_id = self.cfg.get("current_run_id", "")
        active_run = self.cfg.get("active_cluster_run", "")
        run_manifest = _load_manifest(RESULTS / "shared" / "run_manifest.json")
        shared_run_id = run_manifest.get("run_id", "") if run_manifest else ""

        if run_name in (shared_run_id, current_run_id, active_run):
            shared = RESULTS / "shared"
            for p in (list(shared.glob("*.npy")) + list(shared.glob("*.pkl"))
                      + [shared / "cluster_info.json", shared / "run_manifest.json",
                         shared / "validation_report.json"]):
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
            self.cfg["current_run_id"] = ""
            self.cfg["current_run_saved"] = False
            self.cfg["active_cluster_run"] = ""
            _save_cfg(self.cfg)

        self.refresh()
