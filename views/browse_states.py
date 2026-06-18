from __future__ import annotations
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QIcon, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView,
    QLabel, QMessageBox, QPushButton, QScrollArea, QSlider,
    QSplitter, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout, QWidget,
)

import vieb_config as _vc

from _utils import ROOT, RESULTS, _open_folder, _fmt_ts, _state_colors, _CV2, _MPL, _thumb_from_video
from _widgets import VideoPlayer, MplCanvas, KinematicsPanel

if _CV2:
    import cv2
if _MPL:
    from _utils import Figure, FigureCanvas, mpl_cm
    import numpy as np


def _thumb_at_frame(video_path: str, frame: int, size=(140, 90)):
    """Extract a single thumbnail frame from a video at the given frame index."""
    if not _CV2:
        return None
    try:
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, frm = cap.read()
        cap.release()
        if not ret:
            return None
        frm = cv2.resize(frm, size)
        frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
        img = QImage(frm.data, size[0], size[1], 3 * size[0], QImage.Format_RGB888)
        return QPixmap.fromImage(img.copy())
    except Exception:
        return None


class StateDetailDialog(QDialog):
    clips_generated = pyqtSignal()
    request_clip_generation = pyqtSignal(int)

    def __init__(self, sid, s_row=None, c_row=None, bouts_df=None, cfg=None, parent=None):
        super().__init__(parent)
        self.sid = sid
        self.s_row = s_row
        self.c_row = c_row
        self.bouts_df = bouts_df
        self.cfg = cfg or {}
        self._player = None
        self._clip_sections = []
        self.setWindowTitle(f"State {sid} Details")
        self.resize(840, 580)
        self.setMinimumSize(520, 360)
        self.setSizeGripEnabled(True)
        self.setModal(False)
        self._build()

    def _clip_groups(self):
        sd = Path(_vc.get_clips_dir()) / f"state_{self.sid}"
        if not sd.exists():
            return {}, False
        return {
            "Longest Bouts": sorted(sd.glob("longest_*.mp4"))[:5],
            "Most Typical": sorted(sd.glob("typical_*.mp4"))[:5],
            "Context-Enriched": sorted(sd.glob("context_*.mp4"))[:5],
        }, True

    def _meta_for(self, clip: Path):
        if self.bouts_df is None or self.bouts_df.empty:
            return "duration: -, bout: -, context: -, animal: -, day: -"
        sub = self.bouts_df[self.bouts_df["state"] == self.sid]
        if sub.empty:
            return "duration: -, bout: -, context: -, animal: -, day: -"
        if clip.name.startswith("context_"):
            c = clip.name.split("_")[1]
            sub = sub[sub["context"].astype(str) == c]
        sub = sub.sort_values("duration_sec", ascending=False)
        row = sub.iloc[0]
        return (
            f"duration: {float(row.get('duration_sec', 0)):.2f}s, "
            f"bout: {int(row.get('end_frame', 0) - row.get('start_frame', 0) + 1)} fr, "
            f"context: {row.get('context', '-')}, animal: {row.get('animal_id', '-')}, day: {row.get('day', '-')}"
        )

    def _build_kinematics_panel(self):
        panel = QFrame()
        panel.setObjectName("kinPanel")
        panel.setMinimumWidth(280)
        panel.setStyleSheet("QFrame#kinPanel{background:#f8f8f8;border:1px solid #e0e0e0;}")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(8)

        title = QLabel("State Profile")
        title.setFont(QFont("Arial", 11, QFont.Bold))
        lay.addWidget(title)

        subtitle = QLabel("Mean across all frames in this state")
        subtitle.setStyleSheet("color:#888;font-size:10px;font-style:italic;")
        lay.addWidget(subtitle)

        if _MPL and self.s_row is not None:
            metrics = {
                "Speed":      float(self.s_row.get("mean_centroid_speed", 0) or 0),
                "AngVel":     float(self.s_row.get("mean_angular_vel", 0) or 0),
                "BodyLen":    float(self.s_row.get("mean_body_length_px", 0) or 0),
                "Elongation": float(self.s_row.get("mean_elongation", 0) or 0),
                "Entropy":    float(self.s_row.get("mean_entropy", 0) or 0),
                "BoutSec":    float(self.s_row.get("mean_bout_dur_sec", 0) or 0),
            }
            c = MplCanvas(figsize=(3.5, 2.5))
            c.ax.barh(list(metrics.keys()), list(metrics.values()), color="#4a90d9")
            c.ax.set_xlabel("Value", fontsize=7)
            c.ax.tick_params(labelsize=7)
            c.fig.tight_layout()
            lay.addWidget(c)

            tbl = QWidget()
            gl = QGridLayout(tbl)
            gl.setContentsMargins(0, 4, 0, 0)
            gl.setSpacing(3)
            mono = QFont("Courier New", 9)
            rows_data = [
                ("Speed",         f"{metrics['Speed']:.1f} px/s"),
                ("Angular Vel",   f"{metrics['AngVel']:.2f} rad/f"),
                ("Body Length",   f"{metrics['BodyLen']:.0f} px"),
                ("Elongation",    f"{metrics['Elongation']:.2f}"),
                ("Entropy",       f"{metrics['Entropy']:.2f}"),
                ("Bout Duration", f"{metrics['BoutSec']:.1f} s"),
            ]
            for i, (name, val_str) in enumerate(rows_data):
                name_lbl = QLabel(name)
                name_lbl.setStyleSheet("color:#444;font-size:9px;")
                val_lbl = QLabel(val_str)
                val_lbl.setFont(mono)
                val_lbl.setStyleSheet("color:#222;font-size:9px;")
                val_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                gl.addWidget(name_lbl, i, 0)
                gl.addWidget(val_lbl, i, 1)
            lay.addWidget(tbl)
        else:
            no_data = QLabel("Run Characterize to view state profile.")
            no_data.setAlignment(Qt.AlignCenter)
            no_data.setStyleSheet("color:#888;font-style:italic;")
            no_data.setWordWrap(True)
            lay.addWidget(no_data)

        lay.addStretch()
        return panel

    def _start_generate(self):
        self.request_clip_generation.emit(self.sid)
        QMessageBox.information(
            self,
            "Background Job Started",
            "Clip generation is running in the background. You can keep navigating the app.",
        )
        self.close()

    def _build_clips_panel(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        groups, has_any = self._clip_groups()
        self._clip_sections = []
        if has_any and _CV2:
            self._player = VideoPlayer()
            lay.addWidget(self._player)
            for sec, clips in groups.items():
                g = QGroupBox(sec)
                gl = QVBoxLayout(g)
                row_host = QWidget()
                row = QHBoxLayout(row_host)
                row.setContentsMargins(8, 4, 8, 4)
                row.setSpacing(10)
                section_cards = []
                if not clips:
                    row.addWidget(QLabel("No clips in this section."))
                for clip in clips:
                    card = QWidget()
                    cl = QVBoxLayout(card)
                    cl.setContentsMargins(4, 4, 4, 4)
                    cl.setSpacing(4)
                    b = QPushButton("")
                    pm = _thumb_from_video(clip, size=(360, 220))
                    if pm is not None:
                        b.setIcon(QIcon(pm))
                        b.setIconSize(pm.size())
                    b.setMinimumSize(120, 80)
                    b.clicked.connect(lambda _, p=clip: self._player.load(str(p)))
                    name_lbl = QLabel(clip.name)
                    name_lbl.setAlignment(Qt.AlignCenter)
                    name_lbl.setWordWrap(True)
                    meta_lbl = QLabel(self._meta_for(clip))
                    meta_lbl.setWordWrap(True)
                    cl.addWidget(b)
                    cl.addWidget(name_lbl)
                    cl.addWidget(meta_lbl)
                    row.addWidget(card)
                    section_cards.append(
                        {
                            "card": card,
                            "button": b,
                            "pixmap": pm,
                        }
                    )
                row.addStretch()
                sc = QScrollArea()
                sc.setWidgetResizable(True)
                sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
                sc.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
                sc.setWidget(row_host)
                gl.addWidget(sc)
                lay.addWidget(g)
                self._clip_sections.append({"scroll": sc, "cards": section_cards})
            first = next((c for clips in groups.values() for c in clips), None)
            if first:
                self._player.load(str(first))
            QTimer.singleShot(0, self._update_clip_card_sizes)
        else:
            lay.addWidget(QLabel("Clips are not available for this state yet."))
            self._gen_btn = QPushButton("Generate clips for this state")
            self._gen_btn.clicked.connect(self._start_generate)
            lay.addWidget(self._gen_btn)
            lay.addWidget(QLabel("Generation runs in background so you can continue browsing."))
        return w

    def _update_clip_card_sizes(self):
        if not self._clip_sections:
            return
        for sec in self._clip_sections:
            cards = sec["cards"]
            if not cards:
                continue
            viewport_w = sec["scroll"].viewport().width()
            n_cards = len(cards)
            avail = max(220, viewport_w - (n_cards + 1) * 10)
            card_w = max(150, min(300, int(avail / max(1, n_cards))))
            thumb_w = int(card_w * 0.92)
            thumb_h = max(88, int(thumb_w * 0.62))
            for item in cards:
                card = item["card"]
                btn = item["button"]
                pm = item["pixmap"]
                card.setFixedWidth(card_w)
                btn.setFixedSize(thumb_w, thumb_h)
                if pm is not None:
                    scaled = pm.scaled(
                        max(40, thumb_w - 8),
                        max(30, thumb_h - 8),
                        Qt.KeepAspectRatio,
                        Qt.SmoothTransformation,
                    )
                    btn.setIcon(QIcon(scaled))
                    btn.setIconSize(scaled.size())

    def _build(self):
        lay = QVBoxLayout(self)
        hdr = QLabel(f"State {self.sid}")
        hdr.setFont(QFont("Arial", 14, QFont.Bold))
        lay.addWidget(hdr)
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_clips_panel())
        splitter.addWidget(self._build_kinematics_panel())
        splitter.setSizes([600, 400])
        lay.addWidget(splitter, stretch=1)
        btn = QDialogButtonBox(QDialogButtonBox.Close)
        btn.rejected.connect(self.reject)
        lay.addWidget(btn)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._update_clip_card_sizes()


class StateCard(QFrame):
    clicked = pyqtSignal(int)

    def __init__(self, sid, s_row=None, c_row=None, strength=None):
        super().__init__()
        self.sid = sid
        self.setFrameShape(QFrame.StyledPanel)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(200, 235)
        self.setStyleSheet(
            "QFrame{background:#fff;border:none;border-radius:8px;}"
            "QFrame:hover{background:#f5f9ff;}"
        )
        lay = QVBoxLayout(self)
        th = QLabel(alignment=Qt.AlignCenter)
        th.setFixedSize(170, 100)
        th.setStyleSheet("background:#eee;border-radius:4px;")
        clip_dir = Path(_vc.get_clips_dir()) / f"state_{sid}"
        pm = _thumb_from_video(next(iter(clip_dir.glob("*.mp4")), None)) if clip_dir.exists() else None
        if pm:
            th.setPixmap(pm)
        else:
            th.setText(f"State {sid}")
        lay.addWidget(th, alignment=Qt.AlignCenter)
        lab = QLabel(f"State {sid}")
        lab.setFont(QFont("Arial", 10, QFont.Bold))
        lay.addWidget(lab)
        if s_row is not None:
            hl = str(s_row.get("heuristic_label", ""))
            lay.addWidget(QLabel((hl[:32] + "...") if len(hl) > 32 else hl))
        if c_row is not None:
            scores = {
                "A": float(c_row.get("A_enrichment", 0) or 0),
                "B": float(c_row.get("B_enrichment", 0) or 0),
                "C": float(c_row.get("C_enrichment", 0) or 0),
            }
            best = max(scores, key=scores.get)
            lay.addWidget(QLabel(f"Context {best} enriched"))
        if strength is not None and not np.isnan(strength):
            lay.addWidget(QLabel(f"Strength: {100 * float(strength):.1f}%"))
        lay.addStretch()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.sid)


class BrowseStatesView(QWidget):
    navigate_to_pipeline = pyqtSignal()
    request_clip_generation = pyqtSignal(int)

    _CLIPS_PER_PAGE = 20
    _CLIPS_PER_ROW = 4
    _SORT_OPTIONS = ["By ID", "By occupancy", "By bout duration", "By context enrichment"]

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._data = {}
        self._detail_dialogs = []
        self._selected_sid = None
        self._clip_page = 0
        self._clip_files: list = []
        self._state_order: list = []
        self._kin_individual = False
        self._build()

    def _build(self):
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # LEFT PANEL
        left = QFrame()
        left.setFixedWidth(280)
        left.setFrameShape(QFrame.StyledPanel)
        left.setStyleSheet("QFrame{border-right:1px solid #ddd;background:#fafafa;}")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(10, 14, 10, 10)
        ll.setSpacing(8)

        lbl_title = QLabel("Browse States")
        lbl_title.setFont(QFont("Arial", 14, QFont.Bold))
        ll.addWidget(lbl_title)

        # Search
        from PyQt5.QtWidgets import QLineEdit
        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("Search by label…")
        self._search_box.textChanged.connect(self._rebuild_list)
        ll.addWidget(self._search_box)

        # Sort
        sort_row = QHBoxLayout()
        sort_row.addWidget(QLabel("Sort:"))
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(self._SORT_OPTIONS)
        self._sort_combo.currentIndexChanged.connect(self._rebuild_list)
        sort_row.addWidget(self._sort_combo, stretch=1)
        ll.addLayout(sort_row)

        # State list
        self._state_list_scroll = QScrollArea()
        self._state_list_scroll.setWidgetResizable(True)
        self._state_list_scroll.setFrameShape(QFrame.NoFrame)
        self._state_list_widget = QWidget()
        self._state_list_layout = QVBoxLayout(self._state_list_widget)
        self._state_list_layout.setContentsMargins(0, 0, 0, 0)
        self._state_list_layout.setSpacing(4)
        self._state_list_layout.addStretch()
        self._state_list_scroll.setWidget(self._state_list_widget)
        ll.addWidget(self._state_list_scroll, stretch=1)

        self._no_data_btn = QPushButton("Run Characterization + Clip Export")
        self._no_data_btn.clicked.connect(self.navigate_to_pipeline.emit)
        self._no_data_btn.hide()
        ll.addWidget(self._no_data_btn)

        outer.addWidget(left)

        # RIGHT PANEL
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(20, 14, 20, 14)
        rl.setSpacing(10)

        # State title
        self._detail_title = QLabel("Select a state from the list")
        self._detail_title.setFont(QFont("Arial", 16, QFont.Bold))
        rl.addWidget(self._detail_title)

        self._detail_subtitle = QLabel("")
        self._detail_subtitle.setStyleSheet("color:#666;")
        rl.addWidget(self._detail_subtitle)

        # Kinematics header: label + toggle
        kin_hdr = QHBoxLayout()
        kin_lbl = QLabel("Kinematics")
        kin_lbl.setStyleSheet("font-weight:bold;font-size:10px;color:#444;")
        kin_hdr.addWidget(kin_lbl)
        kin_hdr.addStretch()
        self._kin_toggle = QPushButton("Individual")
        self._kin_toggle.setCheckable(True)
        self._kin_toggle.setFixedHeight(22)
        self._kin_toggle.setFixedWidth(80)
        self._kin_toggle.setStyleSheet(
            "QPushButton{font-size:9px;border:1px solid #bbb;border-radius:3px;"
            "background:#f5f5f5;padding:0 6px;color:#333;}"
            "QPushButton:checked{background:#1976D2;color:white;border-color:#1565C0;}"
        )
        self._kin_toggle.toggled.connect(self._on_kin_toggle)
        kin_hdr.addWidget(self._kin_toggle)
        rl.addLayout(kin_hdr)

        # Per-video stem selector (only visible in Individual mode)
        self._kin_stem_combo = QComboBox()
        self._kin_stem_combo.setFixedHeight(24)
        self._kin_stem_combo.setStyleSheet("font-size:9px;")
        self._kin_stem_combo.currentIndexChanged.connect(self._on_kin_stem_changed)
        self._kin_stem_combo.hide()
        rl.addWidget(self._kin_stem_combo)

        # Compact kinematic chart
        if _MPL:
            self._kin_canvas = MplCanvas(figsize=(8, 1.6))
            self._kin_canvas.setMaximumHeight(110)
            rl.addWidget(self._kin_canvas)
        else:
            self._kin_canvas = None

        # Filter bar
        flt_row = QHBoxLayout()
        flt_row.addWidget(QLabel("Animals:"))
        self._flt_animal = QComboBox()
        self._flt_animal.setFixedWidth(120)
        self._flt_animal.currentIndexChanged.connect(self._refresh_clips)
        flt_row.addWidget(self._flt_animal)
        flt_row.addWidget(QLabel("Context:"))
        self._flt_context = QComboBox()
        self._flt_context.setFixedWidth(90)
        self._flt_context.currentIndexChanged.connect(self._refresh_clips)
        flt_row.addWidget(self._flt_context)
        flt_row.addWidget(QLabel("Day:"))
        self._flt_day = QComboBox()
        self._flt_day.setFixedWidth(70)
        self._flt_day.currentIndexChanged.connect(self._refresh_clips)
        flt_row.addWidget(self._flt_day)
        flt_row.addStretch()
        self._gen_clips_btn = QPushButton("Generate Clips")
        self._gen_clips_btn.setFixedHeight(26)
        self._gen_clips_btn.clicked.connect(self._on_generate_clips)
        self._gen_clips_btn.hide()
        flt_row.addWidget(self._gen_clips_btn)
        rl.addLayout(flt_row)

        # Clip grid (scrollable area)
        self._clips_scroll = QScrollArea()
        self._clips_scroll.setWidgetResizable(True)
        self._clips_scroll.setFrameShape(QFrame.NoFrame)
        self._clips_container = QWidget()
        self._clips_grid = QGridLayout(self._clips_container)
        self._clips_grid.setSpacing(10)
        self._clips_scroll.setWidget(self._clips_container)
        rl.addWidget(self._clips_scroll, stretch=1)

        # Pagination
        page_row = QHBoxLayout()
        self._prev_page_btn = QPushButton("◀ Previous")
        self._prev_page_btn.setFixedHeight(26)
        self._prev_page_btn.clicked.connect(self._prev_page)
        self._page_lbl = QLabel("Page 1 of 1")
        self._page_lbl.setAlignment(Qt.AlignCenter)
        self._next_page_btn = QPushButton("Next ▶")
        self._next_page_btn.setFixedHeight(26)
        self._next_page_btn.clicked.connect(self._next_page)
        page_row.addWidget(self._prev_page_btn)
        page_row.addWidget(self._page_lbl, stretch=1)
        page_row.addWidget(self._next_page_btn)
        rl.addLayout(page_row)

        outer.addWidget(right, stretch=1)

    # ── Data update ────────────────────────────────────────────────────────────

    def update_data(self, data):
        self._data = data
        self._rebuild_list()
        if self._selected_sid is not None:
            self._show_state(self._selected_sid)

    def refresh(self, data):
        """Rescan the clips directory and reload the state card list with fresh data."""
        self.update_data(data)

    # ── Left panel: build state cards list ────────────────────────────────────

    def _state_info(self):
        """Return list of (sid, label, mean_frac, mean_bout_dur, max_ctx_enrich) sorted by current sort."""
        ci = self._data.get("cluster_info")
        ss = self._data.get("state_summary")
        cr = self._data.get("context_report")
        summary = self._data.get("summary")
        if ci is None:
            return []
        n = int(ci.get("n_clusters", 0))
        rows = []
        for sid in range(n):
            label = ""
            mean_frac = 0.0
            mean_bout = 0.0
            max_enrich = 0.0
            if ss is not None:
                ss_col = "state_id" if "state_id" in ss.columns else ("state" if "state" in ss.columns else None)
                if ss_col:
                    r = ss[ss[ss_col] == sid]
                    if not r.empty:
                        label = str(r.iloc[0].get("heuristic_label", ""))
                        mean_bout = float(r.iloc[0].get("mean_bout_dur", 0) or 0)
            if summary is not None:
                col = f"state_{sid}_frac"
                if col in summary.columns:
                    mean_frac = float(summary[col].mean())
            if cr is not None:
                cr_col = "state_id" if "state_id" in cr.columns else ("state" if "state" in cr.columns else None)
                if cr_col:
                    r = cr[cr[cr_col] == sid]
                    if not r.empty:
                        for ek in ("A_enrichment", "B_enrichment", "C_enrichment"):
                            v = r.iloc[0].get(ek, 0) or 0
                            try:
                                max_enrich = max(max_enrich, float(v))
                            except (TypeError, ValueError):
                                pass
            rows.append((sid, label, mean_frac, mean_bout, max_enrich))
        return rows

    def _rebuild_list(self):
        # Remove existing items except the trailing stretch
        while self._state_list_layout.count() > 1:
            item = self._state_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        ci = self._data.get("cluster_info")
        if ci is None:
            self._no_data_btn.show()
            self._detail_title.setText("Select a state from the list")
            self._detail_subtitle.setText("")
            return
        self._no_data_btn.hide()

        rows = self._state_info()
        query = self._search_box.text().strip().lower()
        if query:
            rows = [(s, l, f, b, e) for s, l, f, b, e in rows if query in l.lower() or query in str(s)]

        sort_key = self._sort_combo.currentText()
        if sort_key == "By occupancy":
            rows.sort(key=lambda x: x[2], reverse=True)
        elif sort_key == "By bout duration":
            rows.sort(key=lambda x: x[3], reverse=True)
        elif sort_key == "By context enrichment":
            rows.sort(key=lambda x: x[4], reverse=True)
        else:  # By ID
            rows.sort(key=lambda x: x[0])

        self._state_order = [r[0] for r in rows]

        for sid, label, mean_frac, mean_bout, max_enrich in rows:
            card = _StateListCard(sid, label, mean_frac)
            card.clicked.connect(self._on_list_card_clicked)
            if sid == self._selected_sid:
                card.setStyleSheet(
                    "QFrame{background:#e3f2fd;border:1px solid #90caf9;border-radius:6px;}"
                )
            self._state_list_layout.insertWidget(self._state_list_layout.count() - 1, card)

    def _on_list_card_clicked(self, sid):
        self._selected_sid = sid
        self._rebuild_list()
        self._show_state(sid)

    # ── Right panel: show state detail ────────────────────────────────────────

    def _show_state(self, sid):
        ss = self._data.get("state_summary")
        cr = self._data.get("context_report")
        summary = self._data.get("summary")
        lpf = self._data.get("labels_per_frame")

        label = f"State {sid}"
        subtitle_parts = []
        if ss is not None:
            ss_col = "state_id" if "state_id" in ss.columns else ("state" if "state" in ss.columns else None)
            if ss_col:
                r = ss[ss[ss_col] == sid]
                if not r.empty:
                    hl = str(r.iloc[0].get("heuristic_label", ""))
                    if hl:
                        label = f"State {sid} — {hl}"

        if summary is not None and "animal_id" in summary.columns:
            col = f"state_{sid}_frac"
            if col in summary.columns:
                n_animals = int((summary[col] > 0.01).sum())
                n_sessions = int(len(summary))
                subtitle_parts.append(f"Found in {n_animals} animals across {n_sessions} sessions")

        self._detail_title.setText(label)
        self._detail_subtitle.setText("  |  ".join(subtitle_parts) if subtitle_parts else "")

        if self._kin_individual:
            self._populate_kin_stems(sid)
        self._render_kinematics(sid)
        self._populate_filters(sid)
        self._refresh_clips()

    # ── Kinematics toggle & per-video rendering ───────────────────────────────

    def _on_kin_toggle(self, checked: bool):
        self._kin_individual = checked
        self._kin_stem_combo.setVisible(checked)
        if self._selected_sid is not None:
            if checked:
                self._populate_kin_stems(self._selected_sid)
            self._render_kinematics(self._selected_sid)

    def _on_kin_stem_changed(self, _idx: int):
        if self._kin_individual and self._selected_sid is not None:
            self._render_kinematics(self._selected_sid)

    def _populate_kin_stems(self, sid: int):
        self._kin_stem_combo.blockSignals(True)
        self._kin_stem_combo.clear()
        summary = self._data.get("summary")
        col = f"state_{sid}_frac"
        entries: list[tuple[str, str]] = []
        if summary is not None and col in summary.columns:
            stem_col = next((c for c in ("filename", "stem") if c in summary.columns), None)
            if stem_col:
                sub = summary[summary[col] > 0]
                for _, row in sub.iterrows():
                    stem = str(row[stem_col]).replace(".mp4", "")
                    animal = str(row.get("animal_id", ""))
                    ctx    = str(row.get("context", ""))
                    day    = str(row.get("day", ""))
                    label  = " · ".join(p for p in (animal, ctx, f"Day {day}" if day else "") if p)
                    entries.append((stem, label or stem))
        if not entries:
            shared = RESULTS / "shared"
            for f in sorted(shared.glob("*_labels.npy")):
                stem = f.name.replace("_labels.npy", "")
                entries.append((stem, stem))
        for stem, label in entries:
            self._kin_stem_combo.addItem(label, userData=stem)
        self._kin_stem_combo.blockSignals(False)

    _KIN_FEATURE_NAMES = {
        "Speed":       "centroid_speed",
        "Elongation":  "elongation",
        "Ang Vel":     "angular_velocity",
        "Entropy":     "movement_entropy",
        "Rearing":     "rearing_score",
    }

    @staticmethod
    def _load_feature_name_index() -> dict:
        """Return {feature_name: column_index} from index.json metadata."""
        import json
        from ml.feature_extraction import resolve_feature_indices
        idx_path = RESULTS / "features" / "index.json"
        if idx_path.exists():
            try:
                with open(idx_path) as f:
                    meta = json.load(f).get("_meta", {})
                names = meta.get("feature_names", [])
                if names:
                    return resolve_feature_indices(names)
            except Exception:
                pass
        return {}

    def _compute_per_video_kinematics(self, stem: str, sid: int) -> dict | None:
        labels_path = RESULTS / "shared" / f"{stem}_labels.npy"
        feats_path  = RESULTS / "features" / f"{stem}_features.npy"
        if not labels_path.exists() or not feats_path.exists():
            return None
        labels = np.load(str(labels_path))
        mask   = labels == sid
        if not mask.any():
            return None
        feats = np.load(str(feats_path))
        if len(feats) != len(labels):
            return None
        state_feats = feats[mask]
        feat_idx = self._load_feature_name_index()
        return {
            display_name: float(np.mean(np.abs(state_feats[:, idx])))
            for display_name, feat_name in self._KIN_FEATURE_NAMES.items()
            for idx in [feat_idx.get(feat_name)]
            if idx is not None and idx < state_feats.shape[1]
        }

    def _render_kinematics(self, sid):
        if not self._kin_canvas or not _MPL:
            return
        self._kin_canvas.fig.clf()
        ax = self._kin_canvas.fig.add_subplot(111)
        if self._kin_individual:
            self._render_kin_individual(sid, ax)
        else:
            self._render_kin_mean(sid, ax)
        self._kin_canvas.fig.tight_layout(pad=0.5)
        self._kin_canvas.draw()

    def _render_kin_mean(self, sid, ax):
        ss = self._data.get("state_summary")
        if ss is None:
            ax.axis("off"); return
        ss_col = "state_id" if "state_id" in ss.columns else ("state" if "state" in ss.columns else None)
        if ss_col is None:
            ax.axis("off"); return
        r = ss[ss[ss_col] == sid]
        if r.empty:
            ax.axis("off"); return
        row = r.iloc[0]
        kin_cols = [c for c in ("mean_speed", "mean_acceleration", "mean_body_length",
                                "mean_head_angle", "mean_tail_angle")
                    if c in ss.columns]
        if not kin_cols:
            ax.axis("off"); return
        vals = []
        for c in kin_cols:
            try:
                vals.append(float(row[c]))
            except (TypeError, ValueError):
                vals.append(0.0)
        short = [c.replace("mean_", "").replace("_", " ") for c in kin_cols]
        colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2", "#D32F2F"][:len(kin_cols)]
        ax.barh(short, vals, color=colors)
        ax.set_xlabel("Mean (all videos)", fontsize=7)
        ax.tick_params(labelsize=7)

    def _render_kin_individual(self, sid, ax):
        stem = self._kin_stem_combo.currentData()
        if not stem:
            ax.axis("off"); return
        metrics = self._compute_per_video_kinematics(stem, sid)
        if not metrics:
            ax.text(0.5, 0.5, "No data for this video", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="#888")
            ax.axis("off"); return
        colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2", "#D32F2F"][:len(metrics)]
        ax.barh(list(metrics.keys()), list(metrics.values()), color=colors)
        ax.set_xlabel("Mean (this video)", fontsize=7)
        ax.tick_params(labelsize=7)

    def _populate_filters(self, sid):
        summary = self._data.get("summary")
        for combo in (self._flt_animal, self._flt_context, self._flt_day):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("All")
            combo.blockSignals(False)
        if summary is None:
            return
        if "animal_id" in summary.columns:
            for aid in sorted(summary["animal_id"].dropna().astype(str).unique()):
                self._flt_animal.addItem(aid)
        ctx_col = next((c for c in ("context", "Context") if c in summary.columns), None)
        if ctx_col:
            for ctx in sorted(summary[ctx_col].dropna().astype(str).unique()):
                self._flt_context.addItem(ctx)
        if "day" in summary.columns:
            for d in sorted(summary["day"].dropna().astype(str).unique()):
                self._flt_day.addItem(str(d))

    def _resolve_stem_video(self, stem: str) -> str | None:
        """Return the absolute path to the source .mp4 for a stem, or None."""
        raw_dir = _vc.get_raw_videos_dir()
        if raw_dir:
            p = os.path.join(raw_dir, f"{stem}.mp4")
            if os.path.exists(p):
                return p
        fi = self._data.get("feature_index") or {}
        if stem in fi:
            vp = fi[stem].get("video_path", "")
            if vp:
                for candidate in (
                    vp,
                    os.path.join(os.path.dirname(os.path.dirname(RESULTS)), vp),
                    os.path.join(raw_dir or "", os.path.basename(vp)),
                ):
                    if os.path.exists(candidate):
                        return candidate
        return None

    def _refresh_clips(self):
        sid = self._selected_sid
        if sid is None:
            self._gen_clips_btn.hide()
            self._clear_clip_grid()
            return

        flt_animal = self._flt_animal.currentText()
        flt_ctx    = self._flt_context.currentText()
        flt_day    = self._flt_day.currentText()

        # ── Bouts-based approach: one entry per source video ───────────────
        bouts = self._data.get("bouts")
        if bouts is not None and not bouts.empty and "state" in bouts.columns:
            sb = bouts[bouts["state"] == sid]
            if flt_animal != "All":
                sb = sb[sb.get("animal_id", pd.Series(dtype=str)).astype(str) == flt_animal] \
                    if "animal_id" in sb.columns else sb
            if flt_ctx != "All":
                sb = sb[sb.get("context", pd.Series(dtype=str)).astype(str) == flt_ctx] \
                    if "context" in sb.columns else sb
            if flt_day != "All":
                sb = sb[sb.get("day", pd.Series(dtype=str)).astype(str) == flt_day] \
                    if "day" in sb.columns else sb

            if not sb.empty:
                # One representative bout per stem (longest)
                per_stem = (
                    sb.sort_values("duration_sec", ascending=False)
                    .drop_duplicates("stem")
                    .reset_index(drop=True)
                )
                entries = []
                for _, row in per_stem.iterrows():
                    stem = str(row["stem"])
                    vp = self._resolve_stem_video(stem)
                    if not vp:
                        continue
                    entries.append({
                        "stem":        stem,
                        "video_path":  vp,
                        "start_frame": int(row.get("start_frame", 0)),
                        "end_frame":   int(row.get("end_frame", 0)),
                        "context":     str(row.get("context", "")),
                        "animal_id":   str(row.get("animal_id", "")),
                        "day":         str(row.get("day", "")),
                        "duration_sec": float(row.get("duration_sec", 0)),
                    })
                if entries:
                    self._clip_files = entries
                    self._clip_page = 0
                    self._gen_clips_btn.hide()
                    self._render_clip_page()
                    return

            # No entries after filtering
            self._clip_files = []
            self._clip_page = 0
            self._clear_clip_grid()
            msg = "No videos match the current filters for this state." if (
                flt_animal != "All" or flt_ctx != "All" or flt_day != "All"
            ) else "No bouts found for this state.\nRe-run:  python characterize.py"
            no_lbl = QLabel(msg)
            no_lbl.setAlignment(Qt.AlignCenter)
            no_lbl.setStyleSheet("color:#888;font-style:italic;")
            self._clips_grid.addWidget(no_lbl, 0, 0, 1, self._CLIPS_PER_ROW)
            self._page_lbl.setText("Page 0 of 0")
            self._prev_page_btn.setEnabled(False)
            self._next_page_btn.setEnabled(False)
            self._gen_clips_btn.hide()
            return

        # ── Fallback: pre-generated clip files ────────────────────────────
        clip_dir = Path(os.path.abspath(str(Path(_vc.get_clips_dir()) / f"state_{sid}")))
        if not clip_dir.exists() or not list(clip_dir.glob("*.mp4")):
            self._clip_files = []
            self._clip_page = 0
            self._clear_clip_grid()
            lpf = self._data.get("labels_per_frame")
            has_frames = (
                lpf is not None and not lpf.empty
                and "state" in lpf.columns
                and int(sid) in lpf["state"].values
            )
            if has_frames:
                self._gen_clips_btn.show()
                msg = "Clips not yet generated.\nRun:  python characterize.py --clips"
            else:
                self._gen_clips_btn.hide()
                msg = "No frames assigned to this state in any video."
            no_lbl = QLabel(msg)
            no_lbl.setAlignment(Qt.AlignCenter)
            no_lbl.setStyleSheet("color:#888;font-style:italic;")
            self._clips_grid.addWidget(no_lbl, 0, 0, 1, self._CLIPS_PER_ROW)
            self._page_lbl.setText("Page 0 of 0")
            self._prev_page_btn.setEnabled(False)
            self._next_page_btn.setEnabled(False)
            return
        self._gen_clips_btn.hide()
        all_clips = sorted(clip_dir.glob("*.mp4"), key=lambda p: p.name)
        filtered = []
        for p in all_clips:
            name = p.stem
            if flt_animal != "All" and flt_animal not in name:
                continue
            if flt_ctx != "All" and flt_ctx not in name:
                continue
            if flt_day != "All" and flt_day not in name:
                continue
            filtered.append(p)
        self._clip_files = filtered
        self._clip_page = 0
        self._render_clip_page()

    def _load_metadata(self):
        if not hasattr(self, "_metadata_cache"):
            meta_path = Path(os.path.abspath(str(ROOT / "metadata.csv")))
            if meta_path.exists():
                try:
                    self._metadata_cache = pd.read_csv(meta_path)
                except Exception:
                    self._metadata_cache = None
            else:
                self._metadata_cache = None
        return self._metadata_cache

    def _clip_meta(self, fp: Path) -> dict:
        stem = fp.stem
        parts = stem.split("_")

        context = "unknown"
        if parts[0] == "context" and len(parts) >= 2:
            context = parts[1]

        animal_id = "unknown"
        meta = self._load_metadata()
        if meta is not None and "filename" in meta.columns:
            meta_stems = meta["filename"].str.replace(r"\.mp4$", "", regex=True)
            match = meta[meta_stems == stem]
            if not match.empty:
                animal_id = str(match.iloc[0].get("animal_id", "unknown"))
                ctx_val = str(match.iloc[0].get("context", ""))
                if ctx_val:
                    context = ctx_val

        duration = 0.0
        if _CV2:
            try:
                abs_path = os.path.abspath(str(fp))
                cap = cv2.VideoCapture(abs_path)
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                cap.release()
                if n_frames > 0:
                    duration = n_frames / fps
            except Exception:
                pass

        return {"animal_id": animal_id, "context": context, "duration": duration}

    def _render_clip_page(self):
        self._clear_clip_grid()
        files = self._clip_files
        if not files:
            self._page_lbl.setText("Page 0 of 0")
            self._prev_page_btn.setEnabled(False)
            self._next_page_btn.setEnabled(False)
            return

        # Bout-dict entries: render all at once (no pagination cap)
        if isinstance(files[0], dict):
            self._page_lbl.setText(f"{len(files)} videos")
            self._prev_page_btn.setEnabled(False)
            self._next_page_btn.setEnabled(False)
            for i, entry in enumerate(files):
                row_idx = i // self._CLIPS_PER_ROW
                col_idx = i % self._CLIPS_PER_ROW
                mid = (entry["start_frame"] + entry["end_frame"]) // 2
                pm = _thumb_at_frame(entry["video_path"], mid)
                self._clips_grid.addWidget(
                    self._make_bout_cell(entry, pm), row_idx, col_idx
                )
            return

        # Fallback: pre-generated clip Path list with pagination
        total_pages = max(1, math.ceil(len(files) / self._CLIPS_PER_PAGE))
        self._clip_page = max(0, min(self._clip_page, total_pages - 1))
        start = self._clip_page * self._CLIPS_PER_PAGE
        end   = start + self._CLIPS_PER_PAGE
        self._page_lbl.setText(f"Page {self._clip_page + 1} of {total_pages}")
        self._prev_page_btn.setEnabled(self._clip_page > 0)
        self._next_page_btn.setEnabled(self._clip_page < total_pages - 1)
        for i, fp in enumerate(files[start:end]):
            row_idx = i // self._CLIPS_PER_ROW
            col_idx = i % self._CLIPS_PER_ROW
            fp = Path(os.path.abspath(str(fp)))
            thumb_lbl = QLabel()
            thumb_lbl.setFixedSize(140, 90)
            thumb_lbl.setAlignment(Qt.AlignCenter)
            thumb_lbl.setStyleSheet("background:#222;border-radius:4px;color:#aaa;font-size:10px;")
            pm = _thumb_from_video(fp)
            if pm:
                thumb_lbl.setPixmap(pm)
            else:
                thumb_lbl.setText(fp.stem[:18])
            thumb_lbl.setCursor(Qt.PointingHandCursor)
            fp_cap = fp
            thumb_lbl.mousePressEvent = lambda _e, p=fp_cap: self._play_clip(p)
            meta = self._clip_meta(fp)
            name_lbl = QLabel(fp.name)
            name_lbl.setAlignment(Qt.AlignCenter)
            name_lbl.setStyleSheet("font-size:9px;color:#888;")
            animal_lbl = QLabel(f"Animal: {meta['animal_id']}")
            animal_lbl.setAlignment(Qt.AlignCenter)
            animal_lbl.setStyleSheet("font-size:9px;color:#444;")
            ctx_lbl = QLabel(f"Context: {meta['context']}")
            ctx_lbl.setAlignment(Qt.AlignCenter)
            ctx_lbl.setStyleSheet("font-size:9px;color:#444;")
            dur_text = f"{meta['duration']:.1f}s" if meta["duration"] > 0 else "-"
            dur_lbl = QLabel(f"Duration: {dur_text}")
            dur_lbl.setAlignment(Qt.AlignCenter)
            dur_lbl.setStyleSheet("font-size:9px;color:#444;")
            cell = QWidget()
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(2, 2, 2, 2)
            cl.setSpacing(2)
            cl.addWidget(thumb_lbl, alignment=Qt.AlignCenter)
            cl.addWidget(name_lbl, alignment=Qt.AlignCenter)
            cl.addWidget(animal_lbl, alignment=Qt.AlignCenter)
            cl.addWidget(ctx_lbl, alignment=Qt.AlignCenter)
            cl.addWidget(dur_lbl, alignment=Qt.AlignCenter)
            self._clips_grid.addWidget(cell, row_idx, col_idx)

    def _make_bout_cell(self, entry: dict, pm) -> QWidget:
        thumb_lbl = QLabel()
        thumb_lbl.setFixedSize(140, 90)
        thumb_lbl.setAlignment(Qt.AlignCenter)
        thumb_lbl.setStyleSheet("background:#222;border-radius:4px;color:#aaa;font-size:10px;")
        if pm:
            thumb_lbl.setPixmap(pm)
        else:
            thumb_lbl.setText(entry["stem"][-20:])
        thumb_lbl.setCursor(Qt.PointingHandCursor)
        thumb_lbl.mousePressEvent = lambda _e, e=entry: self._play_bout(e)

        animal = entry.get("animal_id", "") or "—"
        ctx    = entry.get("context", "")   or "—"
        day    = entry.get("day", "")       or "—"
        dur    = entry.get("duration_sec", 0)

        cell = QWidget()
        cl = QVBoxLayout(cell)
        cl.setContentsMargins(2, 2, 2, 2)
        cl.setSpacing(2)
        cl.addWidget(thumb_lbl, alignment=Qt.AlignCenter)
        for text, style in (
            (f"Animal: {animal}",          "font-size:9px;color:#444;"),
            (f"Ctx: {ctx}  Day: {day}",    "font-size:9px;color:#444;"),
            (f"Longest bout: {dur:.1f}s",  "font-size:9px;color:#888;"),
        ):
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(style)
            cl.addWidget(lbl, alignment=Qt.AlignCenter)
        return cell

    def _play_bout(self, entry: dict):
        """Open source video seeked to the representative bout for this entry."""
        vp = entry.get("video_path", "")
        if not vp or not os.path.isfile(vp):
            QMessageBox.warning(self, "File Not Found",
                                f"Source video not found:\n{vp}")
            return
        try:
            dlg = QDialog(self)
            dlg.setWindowTitle(f"{entry.get('animal_id','?')} · {entry.get('context','?')} · Day {entry.get('day','?')}")
            lay = QVBoxLayout(dlg)
            lay.setContentsMargins(4, 4, 4, 4)
            lay.setSpacing(4)
            player = VideoPlayer(parent=dlg)
            lay.addWidget(player, stretch=7)
            kin_panel = KinematicsPanel(parent=dlg)
            lay.addWidget(kin_panel, stretch=3)
            player.load(vp)
            # Seek to start of the representative bout
            start = entry.get("start_frame", 0)
            if start > 0 and hasattr(player, "_cap") and player._cap:
                player._cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                player._cur = start
            stem = entry["stem"]
            fi = self._data.get("feature_index") or {}
            if stem in fi:
                feat_path = fi[stem].get("features_path", "")
                if feat_path and os.path.exists(feat_path):
                    import threading
                    end = entry.get("end_frame", start + 300)
                    threading.Thread(
                        target=kin_panel.load_clip,
                        args=(feat_path, start, end),
                        daemon=True
                    ).start()
                    cursor_timer = QTimer(kin_panel)
                    cursor_timer.timeout.connect(
                        lambda: kin_panel.set_frame(player._cur) if hasattr(player, "_cur") else None
                    )
                    cursor_timer.start(33)
                    kin_panel._cursor_timer = cursor_timer
            dlg.resize(700, 620)
            dlg.finished.connect(lambda: _cleanup_clip_dialog(player, kin_panel))
            dlg.exec_()
        except Exception as exc:
            QMessageBox.warning(self, "Playback Error", str(exc))

    def _clear_clip_grid(self):
        while self._clips_grid.count():
            item = self._clips_grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _play_clip(self, path: Path):
        abs_path = os.path.abspath(str(path))
        if not os.path.isfile(abs_path):
            QMessageBox.warning(
                self,
                "File Not Found",
                f"Could not find the clip at:\n{abs_path}",
            )
            return
        try:
            dlg = QDialog(self)
            dlg.setWindowTitle(os.path.basename(abs_path))
            lay = QVBoxLayout(dlg)
            lay.setContentsMargins(4, 4, 4, 4)
            lay.setSpacing(4)

            player = VideoPlayer(parent=dlg)
            lay.addWidget(player, stretch=7)

            kin_panel = KinematicsPanel(parent=dlg)
            lay.addWidget(kin_panel, stretch=3)

            player.load(abs_path)

            # Load kinematics for this clip
            self._attach_kinematics(kin_panel, path, player)

            dlg.resize(700, 620)
            dlg.finished.connect(lambda: _cleanup_clip_dialog(player, kin_panel))
            dlg.exec_()
        except Exception as exc:
            QMessageBox.warning(self, "Playback Error", str(exc))

    def _attach_kinematics(self, kin_panel: "KinematicsPanel", clip_path: Path, player: "VideoPlayer"):
        """Load feature file for the clip and start cursor timer."""
        fi = self._data.get("feature_index") or {}
        if not fi:
            return
        stem = clip_path.stem
        # Try to find the source video stem from clip filename
        # Clip filenames: longest_NN.mp4, typical_NN.mp4, context_X_NN.mp4
        # We need to find features for any video in the same state.
        # For simplicity, load features for the first available stem in the index.
        feat_path = None
        for vstem, info in fi.items():
            fp = info.get("features_path", "").replace("\\", "/")
            if os.path.exists(fp):
                feat_path = fp
                break
        if feat_path is None:
            return

        # Estimate frame range from the clip video length
        start_frame = 0
        end_frame = player._total if hasattr(player, "_total") else 300

        import threading
        def _load_and_show():
            kin_panel.load_clip(feat_path, start_frame, end_frame)

        t = threading.Thread(target=_load_and_show, daemon=True)
        t.start()

        # Cursor timer — fires at 30fps and passes current frame to panel
        cursor_timer = QTimer(kin_panel)
        def _update_cursor():
            if hasattr(player, "_cur"):
                kin_panel.set_frame(player._cur)
        cursor_timer.timeout.connect(_update_cursor)
        cursor_timer.start(33)  # ~30fps
        kin_panel._cursor_timer = cursor_timer  # keep alive


def _cleanup_clip_dialog(player, kin_panel):
    """Stop timers and release resources when clip dialog closes."""
    if hasattr(kin_panel, "_cursor_timer"):
        kin_panel._cursor_timer.stop()
    player.pause()
    if player._cap:
        player._cap.release()

    def _prev_page(self):
        self._clip_page = max(0, self._clip_page - 1)
        self._render_clip_page()

    def _next_page(self):
        total = max(1, math.ceil(len(self._clip_files) / self._CLIPS_PER_PAGE))
        self._clip_page = min(total - 1, self._clip_page + 1)
        self._render_clip_page()

    def _on_generate_clips(self):
        if self._selected_sid is not None:
            self.request_clip_generation.emit(self._selected_sid)


class _StateListCard(QFrame):
    """Compact list card for the left panel of BrowseStatesView."""
    clicked = pyqtSignal(int)

    def __init__(self, sid: int, label: str, mean_frac: float):
        super().__init__()
        self.sid = sid
        self.setCursor(Qt.PointingHandCursor)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "QFrame{background:#fff;border:1px solid #e0e0e0;border-radius:6px;}"
            "QFrame:hover{background:#f5f9ff;border-color:#90caf9;}"
        )
        self.setFixedHeight(58)
        ll = QHBoxLayout(self)
        ll.setContentsMargins(10, 6, 10, 6)
        ll.setSpacing(8)
        badge = QLabel(str(sid))
        badge.setFixedSize(28, 28)
        badge.setAlignment(Qt.AlignCenter)
        badge.setStyleSheet(
            f"background:{_state_colors(max(sid + 1, 10))[sid] if sid < 20 else '#607D8B'};"
            "border-radius:14px;color:white;font-weight:bold;font-size:10px;"
        )
        ll.addWidget(badge)
        info_col = QVBoxLayout()
        info_col.setSpacing(0)
        name_lbl = QLabel(label if label else f"State {sid}")
        name_lbl.setFont(QFont("Arial", 9, QFont.Bold))
        info_col.addWidget(name_lbl)
        frac_lbl = QLabel(f"{mean_frac * 100:.1f}% mean occupancy")
        frac_lbl.setStyleSheet("color:#888;font-size:9px;")
        info_col.addWidget(frac_lbl)
        ll.addLayout(info_col, stretch=1)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.clicked.emit(self.sid)
