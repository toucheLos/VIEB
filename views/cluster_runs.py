from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QVBoxLayout, QWidget,
)

from _utils import ROOT, RESULTS, _save_cfg, _load_cfg


def _shared_dir() -> Path:
    return RESULTS / "shared"


def _runs_dir() -> Path:
    return RESULTS / "runs"


def _load_manifest(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _next_run_n(runs_dir: Path) -> int:
    max_n = 0
    if runs_dir.is_dir():
        for name in os.listdir(runs_dir):
            if name.startswith("run_") and (runs_dir / name).is_dir():
                try:
                    n = int(name.split("_")[1])
                    max_n = max(max_n, n)
                except (IndexError, ValueError):
                    pass
    return max_n + 1


# ---------------------------------------------------------------------------
# Individual run card
# ---------------------------------------------------------------------------

class _RunCard(QFrame):
    """Card widget for one run (current or saved)."""

    save_requested = pyqtSignal()
    delete_requested = pyqtSignal()
    activate_requested = pyqtSignal()

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
        hms = m.get("hdbscan_min_samples", 0)
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
        if hms:
            param_parts.append(f"min_samples={hms}")
        param_lbl = QLabel("  ·  ".join(param_parts))
        param_lbl.setStyleSheet("color:#666;font-size:11px;")
        lay.addWidget(param_lbl)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

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
# Main view
# ---------------------------------------------------------------------------

class ClusterRunsView(QWidget):
    """View for browsing, saving, and restoring versioned cluster runs."""

    cluster_changed = pyqtSignal()

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._build()
        self.refresh()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 24, 24, 24)
        outer.setSpacing(12)

        title = QLabel("Cluster Runs")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        outer.addWidget(title)

        self._current_area = QWidget()
        self._current_layout = QVBoxLayout(self._current_area)
        self._current_layout.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self._current_area)

        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet("color:#E5E5E5;background:#E5E5E5;border:none;max-height:1px;")
        outer.addWidget(div)

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

    # ------------------------------------------------------------------
    # Current run
    # ------------------------------------------------------------------

    def _refresh_current(self):
        lay = self._current_layout
        while lay.count():
            item = lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        manifest_path = _shared_dir() / "run_manifest.json"
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
        lay.addWidget(card)

    # ------------------------------------------------------------------
    # Saved runs list
    # ------------------------------------------------------------------

    def _refresh_saved(self):
        vl = self._saved_vl
        # Remove all but the trailing stretch
        while vl.count() > 1:
            item = vl.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        runs_dir = _runs_dir()
        saved_runs: list[tuple[str, dict]] = []
        if runs_dir.is_dir():
            for entry in sorted(os.listdir(runs_dir)):
                run_path = runs_dir / entry
                if not run_path.is_dir():
                    continue
                mp = run_path / "run_manifest.json"
                m = _load_manifest(mp)
                if m and m.get("saved", False):
                    saved_runs.append((entry, m))

        if not saved_runs:
            self._no_saved_lbl.show()
        else:
            self._no_saved_lbl.hide()

        for run_name, manifest in saved_runs:
            card = _RunCard(manifest, is_current=False)
            card.activate_requested.connect(lambda _rn=run_name: self._activate_run(_rn))
            card.delete_requested.connect(lambda _rn=run_name: self._delete_saved_run(_rn))
            vl.insertWidget(vl.count() - 1, card)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _save_current_run(self):
        manifest_path = _shared_dir() / "run_manifest.json"
        if not manifest_path.exists():
            return

        manifest = _load_manifest(manifest_path)
        if manifest is None:
            return

        manifest["saved"] = True
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        run_id = manifest.get("run_id", "")
        if run_id:
            run_dir = _runs_dir() / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            for fname in os.listdir(_shared_dir()):
                src = _shared_dir() / fname
                if src.is_file():
                    shutil.copy2(src, run_dir / fname)

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

        shared = _shared_dir()
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
        _save_cfg(self.cfg)
        self.refresh()

    def _activate_run(self, run_name: str):
        run_dir = _runs_dir() / run_name
        if not run_dir.is_dir():
            QMessageBox.warning(self, "Not Found", f"Run directory not found: {run_name}")
            return

        shared = _shared_dir()
        shared.mkdir(parents=True, exist_ok=True)

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

        for fname in os.listdir(run_dir):
            src = run_dir / fname
            if src.is_file():
                shutil.copy2(src, shared / fname)

        manifest = _load_manifest(run_dir / "run_manifest.json")
        run_id = manifest.get("run_id", run_name) if manifest else run_name
        self.cfg["current_run_id"] = run_id
        self.cfg["current_run_saved"] = True
        _save_cfg(self.cfg)

        self.refresh()
        self.cluster_changed.emit()

    def _delete_saved_run(self, run_name: str):
        reply = QMessageBox.question(
            self, "Delete Saved Run",
            f"Delete run '{run_name}'? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        run_dir = _runs_dir() / run_name
        try:
            shutil.rmtree(run_dir)
        except Exception as exc:
            QMessageBox.warning(self, "Error", f"Could not delete run: {exc}")
            return

        current_run_id = self.cfg.get("current_run_id", "")
        run_manifest = _load_manifest(_shared_dir() / "run_manifest.json")
        active_run_id = run_manifest.get("run_id", "") if run_manifest else ""

        if run_name == active_run_id or run_name == current_run_id:
            shared = _shared_dir()
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
            _save_cfg(self.cfg)

        self.refresh()
