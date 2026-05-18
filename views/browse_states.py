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
    QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit,
    QVBoxLayout, QWidget,
)

from _utils import ROOT, RESULTS, CLIPS, _open_folder, _fmt_ts, _state_colors, _CV2, _MPL, _thumb_from_video
from _widgets import VideoPlayer, MplCanvas, KinematicsPanel

if _CV2:
    import cv2
if _MPL:
    from _utils import Figure, FigureCanvas, mpl_cm
    import numpy as np


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
        sd = CLIPS / f"state_{self.sid}"
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

    def _build_kinematics(self, tabs):
        w = QWidget()
        lay = QVBoxLayout(w)
        if _MPL and self.s_row is not None:
            c = MplCanvas(figsize=(6, 3))
            metrics = {
                "Speed": self.s_row.get("mean_centroid_speed", 0),
                "AngVel": self.s_row.get("mean_angular_vel", 0),
                "BodyLen": self.s_row.get("mean_body_length_px", 0),
                "Elongation": self.s_row.get("mean_elongation", 0),
                "Entropy": self.s_row.get("mean_entropy", 0),
                "BoutSec": self.s_row.get("mean_bout_dur_sec", 0),
            }
            c.ax.bar(list(metrics.keys()), [float(v or 0) for v in metrics.values()], color="#4a90d9")
            c.ax.set_title("Kinematic Profile")
            c.fig.tight_layout()
            lay.addWidget(c)
        else:
            lay.addWidget(QLabel("Run Characterization + Clip Export to generate state kinematics."))
        tabs.addTab(w, "Kinematics")

    def _start_generate(self):
        self.request_clip_generation.emit(self.sid)
        QMessageBox.information(
            self,
            "Background Job Started",
            "Clip generation is running in the background. You can keep navigating the app.",
        )
        self.close()

    def _build_clips(self, tabs):
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
        tabs.addTab(w, "Video Clips")

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
        tabs = QTabWidget()
        self._build_kinematics(tabs)
        self._build_clips(tabs)
        lay.addWidget(tabs)
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
        pm = _thumb_from_video(next(iter((CLIPS / f"state_{sid}").glob("*.mp4")), None)) if (CLIPS / f"state_{sid}").exists() else None
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

        self._render_kinematics(sid)
        self._populate_filters(sid)
        self._refresh_clips()

    def _render_kinematics(self, sid):
        if not self._kin_canvas or not _MPL:
            return
        ss = self._data.get("state_summary")
        self._kin_canvas.fig.clf()
        ax = self._kin_canvas.fig.add_subplot(111)
        if ss is None:
            ax.axis("off")
            self._kin_canvas.draw()
            return
        ss_col = "state_id" if "state_id" in ss.columns else ("state" if "state" in ss.columns else None)
        if ss_col is None:
            ax.axis("off")
            self._kin_canvas.draw()
            return
        r = ss[ss[ss_col] == sid]
        if r.empty:
            ax.axis("off")
            self._kin_canvas.draw()
            return
        row = r.iloc[0]
        kin_cols = [c for c in ("mean_speed", "mean_acceleration", "mean_body_length",
                                "mean_head_angle", "mean_tail_angle")
                    if c in ss.columns]
        if not kin_cols:
            ax.axis("off")
            self._kin_canvas.draw()
            return
        vals = []
        for c in kin_cols:
            try:
                vals.append(float(row[c]))
            except (TypeError, ValueError):
                vals.append(0.0)
        short = [c.replace("mean_", "").replace("_", " ") for c in kin_cols]
        colors = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2", "#D32F2F"][:len(kin_cols)]
        ax.barh(short, vals, color=colors)
        ax.set_xlabel("Value", fontsize=7)
        ax.tick_params(labelsize=7)
        self._kin_canvas.fig.tight_layout(pad=0.5)
        self._kin_canvas.draw()

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

    def _refresh_clips(self):
        sid = self._selected_sid
        if sid is None:
            self._gen_clips_btn.hide()
            self._clear_clip_grid()
            return
        clip_dir = Path(os.path.abspath(str(CLIPS / f"state_{sid}")))
        if not clip_dir.exists() or not list(clip_dir.glob("*.mp4")):
            self._clip_files = []
            self._clip_page = 0
            self._clear_clip_grid()
            lpf = self._data.get("labels_per_frame")
            has_frames = (
                lpf is not None
                and not lpf.empty
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
        # Apply filters
        flt_animal = self._flt_animal.currentText()
        flt_ctx = self._flt_context.currentText()
        flt_day = self._flt_day.currentText()
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
        total_pages = max(1, math.ceil(len(files) / self._CLIPS_PER_PAGE))
        self._clip_page = max(0, min(self._clip_page, total_pages - 1))
        start = self._clip_page * self._CLIPS_PER_PAGE
        end = start + self._CLIPS_PER_PAGE
        page_files = files[start:end]
        self._page_lbl.setText(f"Page {self._clip_page + 1} of {total_pages}")
        self._prev_page_btn.setEnabled(self._clip_page > 0)
        self._next_page_btn.setEnabled(self._clip_page < total_pages - 1)
        for i, fp in enumerate(page_files):
            row = i // self._CLIPS_PER_ROW
            col = i % self._CLIPS_PER_ROW
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
            self._clips_grid.addWidget(cell, row, col)

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
