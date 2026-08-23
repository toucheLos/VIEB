from __future__ import annotations

import json
from pathlib import Path

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QFrame, QHBoxLayout, QInputDialog, QLabel, QMessageBox, QPushButton,
    QScrollArea, QVBoxLayout, QWidget,
)

from _utils import ROOT, RESULTS, _save_cfg
from cluster_run_manager import ClusterRunManager


def _manager() -> ClusterRunManager:
    return ClusterRunManager(RESULTS, config_path=ROOT / "config.json")


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
# Main view
# ---------------------------------------------------------------------------

class ClusterRunsView(QWidget):
    """View for browsing, saving, and restoring versioned cluster runs."""

    run_activated = pyqtSignal()
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

        self._diag_frame = QFrame()
        self._diag_frame.setFrameShape(QFrame.StyledPanel)
        self._diag_frame.setStyleSheet(
            "QFrame{background:#FAFAFA;border:1px solid #E0E0E0;border-radius:6px;}"
        )
        df_lay = QVBoxLayout(self._diag_frame)
        df_lay.setContentsMargins(16, 12, 16, 12)
        df_lay.setSpacing(8)

        diag_hdr = QHBoxLayout()
        diag_title = QLabel("Clustering Diagnostics")
        diag_title.setFont(QFont("Arial", 12, QFont.Bold))
        diag_hdr.addWidget(diag_title)
        diag_hdr.addStretch()
        df_lay.addLayout(diag_hdr)

        self._diag_params = QLabel("")
        self._diag_params.setWordWrap(True)
        self._diag_params.setStyleSheet(
            "font-size:11px;color:#444;font-family:monospace;border:none;background:transparent;"
        )
        df_lay.addWidget(self._diag_params)

        self._diag_warnings_lay = QVBoxLayout()
        self._diag_warnings_lay.setSpacing(4)
        df_lay.addLayout(self._diag_warnings_lay)

        self._diag_frame.hide()
        outer.addWidget(self._diag_frame)

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
        self._refresh_diagnostics()
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

        manifest_path = RESULTS / "shared" / "run_manifest.json"
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
    # Diagnostics panel
    # ------------------------------------------------------------------

    def _refresh_diagnostics(self):
        manifest = _load_manifest(RESULTS / "shared" / "run_manifest.json") or {}
        ci       = _load_manifest(RESULTS / "shared" / "cluster_info.json") or {}

        if not manifest and not ci:
            self._diag_frame.hide()
            return

        feat_meta: dict = {}
        idx_path = RESULTS / "features" / "index.json"
        if idx_path.exists():
            idx = _load_manifest(idx_path)
            if idx:
                raw_meta = idx.get("_meta", {})
                # Inline schema migration: feature_count → n_features
                feat_meta = dict(raw_meta)
                if "n_features" not in feat_meta and "feature_count" in feat_meta:
                    feat_meta["n_features"] = feat_meta["feature_count"]

        def _v(m1, m2, key, default="—"):
            v = m1.get(key)
            if v is None:
                v = m2.get(key)
            return v if v is not None else default

        n_clusters  = _v(manifest, ci, "n_clusters")
        noise_pct   = manifest.get("noise_frac") or ci.get("noise_frac") or ci.get("low_confidence_frac") or 0
        mean_conf   = manifest.get("mean_confidence") or ci.get("mean_confidence") or 0
        umap_dims   = _v(manifest, ci, "umap_dims")
        mcs         = _v(manifest, ci, "min_cluster_size")
        ms_req      = manifest.get("min_samples_requested")
        ms_res      = manifest.get("min_samples_resolved", manifest.get("hdbscan_min_samples"))
        assign      = manifest.get("assignment_method", ci.get("assignment_method", "—"))
        health      = manifest.get("health_status", "—")

        # n_features: index.json _meta is primary; manifest is fallback (written since this fix)
        n_features = feat_meta.get("n_features")
        if n_features is None:
            n_features = manifest.get("n_features", ci.get("n_features", "—"))

        # use_wavelets: index.json _meta is primary; manifest fallback; None = unknown
        use_wavelets = feat_meta.get("use_wavelets")
        if use_wavelets is None:
            uw_fallback = manifest.get("use_wavelets")
            if uw_fallback is not None:
                use_wavelets = uw_fallback

        if ms_req == 0 and ms_res:
            ms_text = f"min_samples=Auto (→{ms_res})"
        elif ms_res:
            ms_text = f"min_samples={ms_res}"
        else:
            ms_text = ""

        if use_wavelets is True:
            wav_text = "yes"
        elif use_wavelets is False:
            wav_text = "no"
        else:
            wav_text = "—"

        health_colors = {"good": "#2e7d32", "suspicious": "#e65100", "failed": "#c62828"}
        health_color = health_colors.get(str(health), "#555")

        line1 = f"States: {n_clusters}   Noise: {float(noise_pct) * 100:.1f}%   Conf: {float(mean_conf):.3f}"
        line2_parts = [f"mcs={mcs}", f"umap={umap_dims}"]
        if ms_text:
            line2_parts.append(ms_text)
        line2 = "   ".join(line2_parts)
        line3_parts = [f"Features: {n_features}", f"Wavelets: {wav_text}", f"Assign: {assign}"]
        line3 = "   ".join(line3_parts)

        self._diag_params.setText(f"{line1}\n{line2}\n{line3}")

        while self._diag_warnings_lay.count():
            item = self._diag_warnings_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        health_lbl = QLabel(f"Health: {health}")
        health_lbl.setStyleSheet(f"color:{health_color};font-size:11px;font-weight:600;padding:2px 0;border:none;background:transparent;")
        self._diag_warnings_lay.addWidget(health_lbl)

        warnings = manifest.get("warnings", [])
        if isinstance(warnings, str):
            warnings = [warnings]
        elif not isinstance(warnings, list):
            warnings = []
        if not warnings:
            ok_lbl = QLabel("No warnings.")
            ok_lbl.setStyleSheet("color:#2e7d32;font-size:11px;padding:2px 0;border:none;background:transparent;")
            self._diag_warnings_lay.addWidget(ok_lbl)
        for w in warnings:
            if isinstance(w, dict):
                level   = w.get("level", "info")
                message = w.get("message", "")
                action  = w.get("action")
            else:
                level, message, action = "warning", str(w), None
            if level == "error":
                color, icon = "#c62828", "!"
            elif level == "warning":
                color, icon = "#e65100", "*"
            else:
                color, icon = "#1565c0", "i"
            lbl = QLabel(f"  {icon}  {message}")
            lbl.setWordWrap(True)
            lbl.setStyleSheet(f"color:{color};font-size:11px;padding:2px 0;border:none;background:transparent;")
            if action:
                lbl.setToolTip(action)
            self._diag_warnings_lay.addWidget(lbl)

        self._diag_frame.show()

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
        saved_runs = [(m.run_id, m.to_dict()) for m in mgr.list_runs() if m.saved]

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

        run_id = manifest.get("run_id", "")
        if not run_id:
            return

        _manager().save_run(run_id)

        manifest["saved"] = True
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        self.cfg["current_run_saved"] = True
        _save_cfg(self.cfg)

        QMessageBox.information(self, "Run Saved", f"Run saved as {run_id}.")
        self.refresh()

    def _delete_current_run(self):
        reply = QMessageBox.question(
            self, "Delete Current Run",
            "Delete the current run? This cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        shared = RESULTS / "shared"
        for p in (list(shared.glob("*.npy")) + list(shared.glob("*.pkl"))
                  + [shared / "cluster_info.json", shared / "run_manifest.json",
                     shared / "validation_report.json"]):
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

        m = mgr.load_run_manifest(run_name)
        run_id = m.run_id if m else run_name
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
        new_name, ok = QInputDialog.getText(self, "Rename Run", "New name:", text=old_id)
        if not ok or not new_name.strip() or new_name.strip() == old_id:
            return
        new_name = new_name.strip()

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
            saved_mp = new_dir / "run_manifest.json"
            saved_m = _load_manifest(saved_mp)
            if saved_m is not None:
                saved_m["run_id"] = new_name
                saved_mp.write_text(json.dumps(saved_m, indent=2), encoding="utf-8")

        for key in ("current_run_id", "active_cluster_run"):
            if self.cfg.get(key) == old_id:
                self.cfg[key] = new_name
        _save_cfg(self.cfg)
        self.refresh()

    def _rename_saved_run(self, run_name: str):
        runs_dir = RESULTS / "runs"
        if not (runs_dir / run_name).is_dir():
            QMessageBox.warning(self, "Not Found", f"Run directory not found: {run_name}")
            return

        new_name, ok = QInputDialog.getText(self, "Rename Run", "New name:", text=run_name)
        if not ok or not new_name.strip() or new_name.strip() == run_name:
            return
        new_name = new_name.strip()

        if (runs_dir / new_name).exists():
            QMessageBox.warning(self, "Name Taken", f"A run named '{new_name}' already exists.")
            return

        (runs_dir / run_name).rename(runs_dir / new_name)

        saved_mp = runs_dir / new_name / "run_manifest.json"
        saved_m = _load_manifest(saved_mp)
        if saved_m is not None:
            saved_m["run_id"] = new_name
            saved_mp.write_text(json.dumps(saved_m, indent=2), encoding="utf-8")

        current_mp = RESULTS / "shared" / "run_manifest.json"
        current_m = _load_manifest(current_mp)
        if current_m and current_m.get("run_id") == run_name:
            current_m["run_id"] = new_name
            current_mp.write_text(json.dumps(current_m, indent=2), encoding="utf-8")

        for key in ("current_run_id", "active_cluster_run"):
            if self.cfg.get(key) == run_name:
                self.cfg[key] = new_name
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

        _manager().delete_run(run_name)

        run_m = _load_manifest(RESULTS / "shared" / "run_manifest.json")
        shared_id = run_m.get("run_id", "") if run_m else ""
        if run_name in (shared_id, self.cfg.get("current_run_id", ""), self.cfg.get("active_cluster_run", "")):
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
