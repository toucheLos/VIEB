from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QLineEdit, QListWidget, QPushButton,
    QSplitter, QToolButton, QVBoxLayout, QWidget,
)

import vieb_config as _vc
from _utils import RESULTS, _MPL
from _workers import SubprocessWorker
from views.analysis import TerminalBox, _placeholder, _section_title

if _MPL:
    from _widgets import MplCanvas


# ---------------------------------------------------------------------------
# StateCharacterizationView
# ---------------------------------------------------------------------------

class StateCharacterizationView(QWidget):
    """Browse discovered behavioral states: kinematic profiles, context
    enrichment, free-text labeling, and example clips."""

    worker_running = pyqtSignal(bool)

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.cfg = cfg or {}
        self._data: dict = {}
        self._worker = None
        self._state_ids: list[int] = []
        self._build()

    # ─────────────────────────────────────────────────────────── build ──

    def _make_header(self, title: str, run_label: str, run_slot, terminal: TerminalBox) -> QWidget:
        outer = QWidget()
        lay = QVBoxLayout(outer)
        lay.setContentsMargins(0, 0, 0, 6)
        lay.setSpacing(4)

        top = QHBoxLayout()
        lbl = QLabel(title)
        lbl.setFont(QFont("Arial", 14, QFont.Bold))
        top.addWidget(lbl)
        top.addStretch()
        run_btn = QPushButton(run_label)
        run_btn.setFixedHeight(30)
        run_btn.clicked.connect(run_slot)
        top.addWidget(run_btn)
        lay.addLayout(top)
        lay.addWidget(terminal)
        return outer

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(20, 16, 20, 16)
        lay.setSpacing(8)

        self._terminal = TerminalBox()
        hdr = self._make_header(
            "State Characterization", "Run Characterization",
            lambda: self._run_command(["characterize.py"], self._terminal),
            self._terminal,
        )
        lay.addWidget(hdr)

        splitter = QSplitter(Qt.Horizontal)

        # Left: state list
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 8, 0)
        ll.setSpacing(4)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search states…")
        self._search.textChanged.connect(self._on_search)
        ll.addWidget(self._search)

        sort_row = QHBoxLayout()
        sort_row.setSpacing(4)
        sort_lbl = QLabel("Sort:")
        sort_lbl.setStyleSheet("font-size:11px; color:#666;")
        sort_row.addWidget(sort_lbl)
        self._sort_combo = QComboBox()
        self._sort_combo.addItems(["State ID", "Speed", "Elongation", "Bout Duration", "Angular Velocity"])
        self._sort_combo.setStyleSheet("font-size:11px;")
        self._sort_combo.currentIndexChanged.connect(self._on_sort_changed)
        sort_row.addWidget(self._sort_combo, stretch=1)
        self._sort_asc = QToolButton()
        self._sort_asc.setText("↑")
        self._sort_asc.setCheckable(True)
        self._sort_asc.setChecked(True)
        self._sort_asc.setToolTip("Toggle ascending / descending")
        self._sort_asc.setFixedWidth(26)
        self._sort_asc.toggled.connect(self._on_sort_changed)
        sort_row.addWidget(self._sort_asc)
        ll.addLayout(sort_row)

        self._state_list = QListWidget()
        self._state_list.currentRowChanged.connect(self._on_state_selected)
        ll.addWidget(self._state_list, stretch=1)
        left.setFixedWidth(240)
        splitter.addWidget(left)

        # Right: detail
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 0, 0, 0)
        rl.setSpacing(8)

        self._detail_placeholder = _placeholder("Select a state from the list.")
        rl.addWidget(self._detail_placeholder)

        if _MPL:
            rl.addWidget(_section_title("Kinematic Profile"))
            self._kin_canvas = MplCanvas(figsize=(6, 2.5))
            self._kin_canvas.setMaximumHeight(200)
            self._kin_canvas.hide()
            rl.addWidget(self._kin_canvas)
        else:
            self._kin_canvas = None

        lbl_row = QHBoxLayout()
        lbl_row.addWidget(QLabel("Label:"))
        self._label_input = QLineEdit()
        self._label_input.setPlaceholderText("Free-text label")
        lbl_row.addWidget(self._label_input, stretch=1)
        self._label_combo = QComboBox()
        self._label_combo.addItems(["Freeze", "Walk", "Groom", "Rear", "Explore", "Other"])
        lbl_row.addWidget(self._label_combo)
        save_btn = QPushButton("Save")
        save_btn.setFixedWidth(56)
        save_btn.clicked.connect(self._save_state_label)
        lbl_row.addWidget(save_btn)
        rl.addLayout(lbl_row)

        clip_row = QHBoxLayout()
        self._load_clip_btn = QPushButton("Load Clip")
        self._load_clip_btn.setEnabled(False)
        self._load_clip_btn.clicked.connect(self._load_clip)
        clip_row.addWidget(self._load_clip_btn)
        self._clip_status = QLabel("")
        self._clip_status.setStyleSheet("color:#888; font-size:11px;")
        clip_row.addWidget(self._clip_status, stretch=1)
        rl.addLayout(clip_row)

        try:
            from _widgets import VideoPlayer
            self._player = VideoPlayer()
            rl.addWidget(self._player, stretch=1)
        except Exception:
            self._player = None
            rl.addWidget(_placeholder(
                "Video player unavailable.\nInstall opencv-python to enable."
            ))

        splitter.addWidget(right)
        lay.addWidget(splitter, stretch=1)

    # ────────────────────────────────────────────────── Data loading ──

    def update_data(self, data: dict) -> None:
        self._data = data
        self._load()

    def refresh(self, data: dict) -> None:
        self._data = data
        self._load()

    def _load(self) -> None:
        ss = self._data.get("state_summary")
        ctx = self._data.get("context_report")

        self._state_list.clear()
        self._state_ids = []
        self._load_clip_btn.setEnabled(False)
        self._clip_status.setText("")
        self._detail_placeholder.show()
        if self._kin_canvas:
            self._kin_canvas.hide()

        if ss is None or ss.empty:
            self._state_list.addItem("No data — run characterize.py")
            return

        id_col = next(
            (c for c in ("state_id", "cluster_id", "state") if c in ss.columns), None
        )
        if id_col is None:
            self._state_list.addItem("Missing state_id column")
            return

        ctx_enrich: dict[int, str] = {}
        if ctx is not None and not ctx.empty and id_col in ctx.columns:
            for _, row in ctx.iterrows():
                sid = int(row.get(id_col, -1))
                enrich = str(row.get("enriched_context", ""))
                if enrich and enrich != "nan":
                    ctx_enrich[sid] = enrich

        self._id_col = id_col
        self._ctx_enrich = ctx_enrich
        self._populate_list()

    _SORT_COLS = [
        None,                    # State ID
        "mean_centroid_speed",   # Speed
        "mean_elongation",       # Elongation
        "mean_bout_dur_sec",     # Bout Duration
        "mean_angular_vel",      # Angular Velocity
    ]

    def _populate_list(self) -> None:
        ss = self._data.get("state_summary")
        if ss is None or ss.empty:
            return
        id_col = getattr(self, "_id_col", None)
        if id_col is None:
            return
        ctx_enrich = getattr(self, "_ctx_enrich", {})

        sort_idx = self._sort_combo.currentIndex()
        sort_col = self._SORT_COLS[sort_idx]
        ascending = self._sort_asc.isChecked()

        if sort_col and sort_col in ss.columns:
            df = ss.sort_values(sort_col, ascending=ascending, na_position="last")
        else:
            df = ss.sort_values(id_col, ascending=ascending)

        self._state_list.clear()
        self._state_ids = []
        for _, row in df.iterrows():
            sid = int(row.get(id_col, -1))
            label = str(row.get("heuristic_label", f"State {sid}"))
            speed = row.get("mean_centroid_speed", None)
            enrich = ctx_enrich.get(sid, "")
            speed_str = (
                f"  spd={speed:.3f}" if speed is not None and not pd.isna(speed) else ""
            )
            badge = f"  [{enrich}]" if enrich else ""
            self._state_list.addItem(f"S{sid}: {label}{speed_str}{badge}")
            self._state_ids.append(sid)

    def _on_sort_changed(self, _=None) -> None:
        self._sort_asc.setText("↑" if self._sort_asc.isChecked() else "↓")
        self._populate_list()

    def _on_search(self, text: str) -> None:
        for i in range(self._state_list.count()):
            item = self._state_list.item(i)
            item.setHidden(bool(text) and text.lower() not in item.text().lower())

    def _on_state_selected(self, row: int) -> None:
        if row < 0 or row >= len(self._state_ids):
            return
        sid = self._state_ids[row]
        self._detail_placeholder.hide()
        self._load_clip_btn.setEnabled(True)
        self._load_clip_btn.setProperty("_sid", sid)

        ss = self._data.get("state_summary")
        if ss is None:
            return
        id_col = next(
            (c for c in ("state_id", "cluster_id", "state") if c in ss.columns), None
        )
        if id_col is None:
            return
        rows = ss[ss[id_col] == sid]
        if rows.empty:
            return
        r = rows.iloc[0]

        if _MPL and self._kin_canvas:
            self._kin_canvas.show()
            canvas = self._kin_canvas
            canvas.fig.clf()
            ax = canvas.fig.add_subplot(111)

            # Fixed kinematic metrics, matched case-insensitively against
            # whatever columns state_summary.csv actually has.
            kinematic_metrics = [
                ("centroid_speed",   "Speed"),
                ("angular_velocity", "Angular Velocity"),
                ("body_elongation",  "Elongation"),
                ("rearing_score",    "Rearing Score"),
                ("bout_duration_s",  "Bout Duration"),
            ]
            cols_lower = {c.lower(): c for c in ss.columns}

            metrics = {}
            for col_name, display in kinematic_metrics:
                col = cols_lower.get(col_name.lower())
                if col is None:
                    continue
                v = r.get(col, None)
                if v is None or pd.isna(v):
                    continue
                max_v = ss[col].max()
                if pd.isna(max_v) or max_v == 0:
                    continue
                metrics[display] = float(v) / float(max_v)

            if metrics:
                norm_v = list(metrics.values())
                y = np.arange(len(metrics))
                ax.barh(y, norm_v, color="#4E79A7", alpha=0.85)
                ax.set_yticks(y)
                ax.set_yticklabels(list(metrics.keys()), fontsize=9)
                ax.set_xlim(0, 1)
                ax.set_xlabel("Normalized value (relative to max across states)")
            else:
                ax.text(0.5, 0.5, "No kinematic data",
                        ha="center", va="center", transform=ax.transAxes, color="#999")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            canvas.fig.tight_layout()
            canvas.draw()

        saved = self._load_saved_state_labels().get(sid, {})
        self._label_input.setText(saved.get("label", ""))
        cat = saved.get("category", "")
        if cat:
            idx = self._label_combo.findText(cat)
            if idx >= 0:
                self._label_combo.setCurrentIndex(idx)

    def _load_saved_state_labels(self) -> dict[int, dict]:
        p = RESULTS / "validation" / "state_labels.csv"
        if not p.exists():
            return {}
        try:
            df = pd.read_csv(p)
            return {
                int(row.get("state_id", -1)): {
                    "label": str(row.get("label", "")),
                    "category": str(row.get("category", "")),
                }
                for _, row in df.iterrows()
            }
        except Exception:
            return {}

    def _save_state_label(self) -> None:
        row = self._state_list.currentRow()
        if row < 0 or row >= len(self._state_ids):
            return
        sid = self._state_ids[row]
        existing = self._load_saved_state_labels()
        existing[sid] = {
            "label": self._label_input.text().strip(),
            "category": self._label_combo.currentText(),
        }
        p = RESULTS / "validation" / "state_labels.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {"state_id": k, "label": v["label"], "category": v["category"]}
            for k, v in sorted(existing.items())
        ]
        pd.DataFrame(rows).to_csv(p, index=False)

    def _load_clip(self) -> None:
        sid = self._load_clip_btn.property("_sid")
        if sid is None:
            return
        clip_dir = Path(_vc.get_clips_dir()) / f"state_{sid}"
        if not clip_dir.exists():
            self._clip_status.setText(
                f"No clips found for state {sid} — run: python generate_clips.py"
            )
            return
        clips = list(clip_dir.glob("*.mp4"))
        if not clips:
            self._clip_status.setText(f"No .mp4 files in clips/state_{sid}/")
            return
        chosen = random.choice(clips)
        self._clip_status.setText(chosen.name)
        if self._player:
            try:
                self._player.load(str(chosen))
            except Exception:
                self._clip_status.setText(f"Error loading {chosen.name}")

    # ─────────────────────────────────────── Command runner ──

    def _run_command(self, args: list[str], terminal: TerminalBox) -> None:
        if self._worker and self._worker.isRunning():
            return
        terminal.set_command("python " + " ".join(str(a) for a in args))
        self._worker = SubprocessWorker(args)
        self._worker.log.connect(terminal.append_output)
        self._worker.done.connect(self._on_run_done)
        self.worker_running.emit(True)
        self._worker.start()

    def stop_worker(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()

    def _on_run_done(self, ok: bool) -> None:
        self.worker_running.emit(False)
        self._load()
