from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QHBoxLayout, QLabel, QLineEdit,
    QListWidget, QPushButton, QSplitter, QToolButton, QVBoxLayout, QWidget,
)

import vieb_config as _vc
from _utils import RESULTS, _MPL, _save_cfg
from _workers import SubprocessWorker
from views.analysis import TerminalBox, _placeholder, _section_title

if _MPL:
    from _widgets import MplCanvas

_BEHAVIORAL_CATEGORIES = [
    "Freezing", "Walking", "Grooming", "Rearing",
    "Running", "Exploring", "Other",
]
_TECHNICAL_CATEGORIES = [
    "Low Velocity", "High Velocity",
    "Low Elongation", "High Elongation",
    "Short Bouts", "Long Bouts",
    "High Angular Vel.", "Low Angular Vel.",
    "High Rearing", "Low Rearing",
]

_CHIP_STYLE = (
    "QPushButton{background:#f0f0f0;border:1px solid #ccc;border-radius:12px;"
    "padding:4px 10px;font-size:11px;}"
    "QPushButton:checked{background:#4E79A7;color:white;border-color:#4E79A7;}"
    "QPushButton:hover:!checked{background:#e0e8f0;}"
)
_TECH_CHIP_STYLE = (
    "QPushButton{background:#f5f5f5;border:1px solid #ddd;border-radius:12px;"
    "padding:3px 8px;font-size:10px;color:#666;}"
    "QPushButton:checked{background:#76B7B2;color:white;border-color:#76B7B2;}"
    "QPushButton:hover:!checked{background:#e8f0f0;}"
)

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
        self._heuristic_labels: dict[int, str] = {}
        self._kin_metrics: dict[str, float] = {}
        self._ref_width = 900
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

        self._splitter = QSplitter(Qt.Horizontal)
        splitter = self._splitter

        # Left: state list panel (expanded + collapsed views)
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 8, 0)
        ll.setSpacing(4)

        # Collapse / expand toggle row
        toggle_row = QHBoxLayout()
        toggle_row.setContentsMargins(0, 0, 0, 0)
        self._panel_title = QLabel("States")
        self._panel_title.setStyleSheet(
            "font-weight:bold; font-size:12px; color:#1A1A1A;"
        )
        toggle_row.addWidget(self._panel_title)
        toggle_row.addStretch()
        self._collapse_btn = QPushButton("«")
        self._collapse_btn.setFixedSize(22, 22)
        self._collapse_btn.setCursor(Qt.PointingHandCursor)
        self._collapse_btn.setToolTip("Collapse state list")
        self._collapse_btn.setStyleSheet(
            "QPushButton{background:transparent;color:#9B9B9B;border:none;font-size:13px;}"
            "QPushButton:hover{color:#1A1A1A;background:rgba(0,0,0,0.05);border-radius:4px;}"
        )
        self._collapse_btn.clicked.connect(self._toggle_state_panel)
        toggle_row.addWidget(self._collapse_btn)
        ll.addLayout(toggle_row)

        # Expanded controls (hidden when collapsed)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search states…")
        self._search.textChanged.connect(self._on_search)
        ll.addWidget(self._search)

        sort_row = QHBoxLayout()
        sort_row.setSpacing(4)
        self._sort_lbl = QLabel("Sort:")
        self._sort_lbl.setStyleSheet("font-size:11px; color:#666;")
        sort_row.addWidget(self._sort_lbl)
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
        self._sort_row_widget = QWidget()
        self._sort_row_widget.setLayout(sort_row)
        ll.addWidget(self._sort_row_widget)

        self._state_list = QListWidget()
        self._state_list.currentRowChanged.connect(self._on_state_selected)
        ll.addWidget(self._state_list, stretch=1)

        # Collapsed view: narrow column with just state numbers
        self._collapsed_list = QListWidget()
        self._collapsed_list.setStyleSheet(
            "QListWidget{border:none; font-size:11px; font-weight:bold;}"
            "QListWidget::item{padding:2px 4px; text-align:center;}"
            "QListWidget::item:selected{background:#4E79A7; color:white;"
            "border-radius:3px;}"
        )
        self._collapsed_list.setFixedWidth(40)
        self._collapsed_list.currentRowChanged.connect(self._on_collapsed_selected)
        self._collapsed_list.hide()
        ll.addWidget(self._collapsed_list, stretch=1)

        self._left_panel = left
        self._panel_collapsed = False
        self._left_expanded_width = 240
        self._collapse_threshold = 100
        left.setMinimumWidth(56)
        splitter.addWidget(left)

        # Right: detail
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 0, 0, 0)
        rl.setSpacing(8)

        self._detail_placeholder = _placeholder("Select a state from the list.")
        rl.addWidget(self._detail_placeholder)

        if _MPL:
            self._kin_title = _section_title("Kinematic Profile")
            rl.addWidget(self._kin_title)
            self._kin_canvas = MplCanvas(figsize=(6, 2.5))
            self._kin_canvas.hide()
            rl.addWidget(self._kin_canvas)
        else:
            self._kin_canvas = None
            self._kin_title = None

        # ── Clip controls ──
        clip_row = QHBoxLayout()
        self._load_clip_btn = QPushButton("Load Clip")
        self._load_clip_btn.setEnabled(False)
        self._load_clip_btn.clicked.connect(self._load_clip)
        clip_row.addWidget(self._load_clip_btn)
        self._autoplay_cb = QCheckBox("Autoplay")
        self._autoplay_cb.setChecked(True)
        self._autoplay_cb.setToolTip("Automatically play clips when loaded")
        clip_row.addWidget(self._autoplay_cb)
        self._clip_status = QLabel("")
        self._clip_status.setStyleSheet("color:#888; font-size:11px;")
        clip_row.addWidget(self._clip_status, stretch=1)
        rl.addLayout(clip_row)

        # ── Label section ──
        self._lbl_title = QLabel("Label this state:")
        self._lbl_title.setStyleSheet("font-weight:bold; font-size:12px; color:#333;")
        rl.addWidget(self._lbl_title)

        input_row = QHBoxLayout()
        self._label_input = QLineEdit()
        self._label_input.setPlaceholderText("Free-text label, or select a category below")
        input_row.addWidget(self._label_input, stretch=1)
        self._save_btn = QPushButton("Save")
        self._save_btn.clicked.connect(self._save_state_label)
        input_row.addWidget(self._save_btn)
        self._back_btn = QPushButton("← Back")
        self._back_btn.setToolTip("Go back to previous state")
        self._back_btn.clicked.connect(self._go_back)
        input_row.addWidget(self._back_btn)
        self._save_next_btn = QPushButton("Save && Next")
        self._save_next_btn.setToolTip("Save label and advance to next state")
        self._save_next_btn.clicked.connect(self._save_and_next)
        input_row.addWidget(self._save_next_btn)
        rl.addLayout(input_row)

        # Category chips
        self._cat_group = QButtonGroup(self)
        self._cat_group.setExclusive(True)
        self._cat_buttons: dict[str, QPushButton] = {}

        cat_row = QHBoxLayout()
        cat_row.setSpacing(4)
        self._cat_label = QLabel("Category:")
        cat_row.addWidget(self._cat_label)
        for name in _BEHAVIORAL_CATEGORIES:
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setStyleSheet(_CHIP_STYLE)
            self._cat_group.addButton(btn)
            cat_row.addWidget(btn)
            self._cat_buttons[name] = btn
        self._more_btn = QToolButton()
        self._more_btn.setText("More ▸")
        self._more_btn.setCheckable(True)
        self._more_btn.setStyleSheet("font-size:11px; color:#666; border:none;")
        self._more_btn.toggled.connect(self._toggle_technical_cats)
        cat_row.addWidget(self._more_btn)
        cat_row.addStretch()
        rl.addLayout(cat_row)

        self._tech_row_widget = QWidget()
        tech_inner = QHBoxLayout(self._tech_row_widget)
        tech_inner.setContentsMargins(0, 0, 0, 0)
        tech_inner.setSpacing(4)
        for name in _TECHNICAL_CATEGORIES:
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setStyleSheet(_TECH_CHIP_STYLE)
            self._cat_group.addButton(btn)
            tech_inner.addWidget(btn)
            self._cat_buttons[name] = btn
        tech_inner.addStretch()
        self._tech_row_widget.hide()
        rl.addWidget(self._tech_row_widget)

        # Custom category row
        self._custom_cat_row = QHBoxLayout()
        self._custom_cat_row.setSpacing(4)
        self._custom_cat_chips_layout = self._custom_cat_row
        self._cat_input = QLineEdit()
        self._cat_input.setPlaceholderText("New category…")
        self._cat_input.setMaximumWidth(160)
        self._cat_input.returnPressed.connect(self._add_custom_category)
        self._custom_cat_row.addWidget(self._cat_input)
        add_cat_btn = QPushButton("+")
        add_cat_btn.setFixedWidth(28)
        add_cat_btn.setToolTip("Add custom category")
        add_cat_btn.clicked.connect(self._add_custom_category)
        self._custom_cat_row.addWidget(add_cat_btn)
        self._custom_cat_row.addStretch()
        rl.addLayout(self._custom_cat_row)
        self._restore_custom_categories()

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
        splitter.setSizes([self._left_expanded_width, 660])
        splitter.splitterMoved.connect(self._on_splitter_moved)
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
        self._compute_heuristic_labels(ss, id_col)
        self._populate_list()

    def _compute_heuristic_labels(self, ss: pd.DataFrame, id_col: str) -> None:
        self._heuristic_labels = {}
        ci = self._data.get("cluster_info") or {}
        centers = ci.get("cluster_centers", [])
        if not centers:
            return
        try:
            from compare import _generate_kinematic_labels
        except ImportError:
            return
        n_kp = 8
        try:
            import json as _json
            idx_path = RESULTS / "features" / "index.json"
            if idx_path.exists():
                with open(idx_path, encoding="utf-8") as f:
                    n_kp = _json.load(f).get("_meta", {}).get("n_keypoints", 8)
        except Exception:
            pass
        bout_stats: dict[int, dict] = {}
        for _, row in ss.iterrows():
            sid = int(row.get(id_col, -1))
            dur = row.get("mean_bout_dur_sec")
            if dur is not None and not pd.isna(dur):
                bout_stats[sid] = {"mean_dur": float(dur)}
            else:
                bout_stats[sid] = {"mean_dur": None}
        self._heuristic_labels = _generate_kinematic_labels(
            centers, bout_stats, n_keypoints=n_kp,
        )

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
            base = f"State {sid}"
            trait = self._heuristic_labels.get(sid, "")
            if trait and trait != base:
                label = f"{base}  —  {trait}"
            else:
                label = base
            self._state_list.addItem(label)
            self._state_ids.append(sid)
        if self._panel_collapsed:
            self._sync_collapsed_list()

    def _on_sort_changed(self, _=None) -> None:
        self._sort_asc.setText("↑" if self._sort_asc.isChecked() else "↓")
        self._populate_list()

    def _on_search(self, text: str) -> None:
        for i in range(self._state_list.count()):
            item = self._state_list.item(i)
            item.setHidden(bool(text) and text.lower() not in item.text().lower())

    def _enrich_kinematic_columns(self, ss: pd.DataFrame) -> pd.DataFrame:
        """Add kinematic columns from cluster_info centers when missing."""
        needed = {
            "mean_centroid_speed", "mean_angular_vel", "mean_elongation",
            "mean_rearing_score", "mean_movement_entropy",
        }
        existing = {c.lower() for c in ss.columns}
        if needed.issubset(existing):
            return ss

        ci = self._data.get("cluster_info") or {}
        centers = ci.get("cluster_centers", [])
        if not centers:
            return ss

        from compare import _extract_kinematic_values
        id_col = getattr(self, "_id_col", None)
        if id_col is None:
            return ss

        n_kp = 8
        try:
            import json as _json
            idx_path = RESULTS / "features" / "index.json"
            if idx_path.exists():
                with open(idx_path, encoding="utf-8") as f:
                    n_kp = _json.load(f).get("_meta", {}).get("n_keypoints", 8)
        except Exception:
            pass

        kin_rows = []
        for _, row in ss.iterrows():
            sid = int(row.get(id_col, -1))
            if 0 <= sid < len(centers):
                vals = _extract_kinematic_values(centers[sid], n_keypoints=n_kp)
            else:
                vals = {}
            vals[id_col] = sid
            kin_rows.append(vals)

        kin_df = pd.DataFrame(kin_rows)
        for col in kin_df.columns:
            if col != id_col and col.lower() not in existing:
                ss[col] = kin_df[col].values
        return ss

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

        ss = self._enrich_kinematic_columns(ss)
        self._data["state_summary"] = ss

        rows = ss[ss[id_col] == sid]
        if rows.empty:
            return
        r = rows.iloc[0]

        if _MPL and self._kin_canvas:
            kinematic_metrics = [
                ("mean_centroid_speed",     "Speed"),
                ("mean_angular_vel",        "Angular Velocity"),
                ("mean_elongation",         "Elongation"),
                ("mean_rearing_score",      "Rearing Score"),
                ("mean_bout_dur_sec",       "Bout Duration"),
                ("mean_movement_entropy",   "Movement Variability"),
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
                min_v = ss[col].min()
                max_v = ss[col].max()
                if pd.isna(min_v) or pd.isna(max_v) or max_v == min_v:
                    continue
                metrics[display] = (float(v) - float(min_v)) / (float(max_v) - float(min_v))

            self._kin_metrics = metrics
            self._draw_kinematic_chart()

        saved = self._load_saved_state_labels().get(sid, {})
        self._label_input.setText(saved.get("label", ""))
        cat = saved.get("category", "")
        self._cat_group.setExclusive(False)
        for btn in self._cat_buttons.values():
            btn.setChecked(False)
        self._cat_group.setExclusive(True)
        if cat and cat in self._cat_buttons:
            self._cat_buttons[cat].setChecked(True)
            if cat in _TECHNICAL_CATEGORIES:
                self._more_btn.setChecked(True)

    # ───────────────────────────── Responsive scaling ──

    def _scale_factor(self) -> float:
        return max(0.6, min(1.6, self.width() / self._ref_width))

    def _draw_kinematic_chart(self) -> None:
        if not (_MPL and self._kin_canvas):
            return
        self._kin_canvas.show()
        canvas = self._kin_canvas
        canvas.fig.clf()
        ax = canvas.fig.add_subplot(111)
        s = self._scale_factor()
        metrics = self._kin_metrics

        if metrics:
            norm_v = list(metrics.values())
            y = np.arange(len(metrics))
            ax.barh(y, norm_v, color="#4E79A7", alpha=0.85)
            ax.set_yticks(y)
            ax.set_yticklabels(list(metrics.keys()), fontsize=max(7, int(9 * s)))
            ax.set_xlim(0, 1)
            ax.set_xlabel(
                "Normalized value (min–max across states)",
                fontsize=max(7, int(9 * s)),
            )
            ax.tick_params(axis="x", labelsize=max(6, int(8 * s)))
        else:
            ax.text(
                0.5, 0.5, "No kinematic data",
                ha="center", va="center", transform=ax.transAxes, color="#999",
                fontsize=max(8, int(10 * s)),
            )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        canvas.fig.tight_layout()
        canvas.draw()

    def _apply_scaled_styles(self) -> None:
        s = self._scale_factor()
        chip_fs = max(8, int(11 * s))
        chip_pad_h = max(3, int(4 * s))
        chip_pad_w = max(6, int(10 * s))
        chip_r = max(8, int(12 * s))
        chip_style = (
            f"QPushButton{{background:#f0f0f0;border:1px solid #ccc;"
            f"border-radius:{chip_r}px;padding:{chip_pad_h}px {chip_pad_w}px;"
            f"font-size:{chip_fs}px;}}"
            f"QPushButton:checked{{background:#4E79A7;color:white;border-color:#4E79A7;}}"
            f"QPushButton:hover:!checked{{background:#e0e8f0;}}"
        )
        tech_fs = max(7, int(10 * s))
        tech_pad_h = max(2, int(3 * s))
        tech_pad_w = max(5, int(8 * s))
        tech_style = (
            f"QPushButton{{background:#f5f5f5;border:1px solid #ddd;"
            f"border-radius:{chip_r}px;padding:{tech_pad_h}px {tech_pad_w}px;"
            f"font-size:{tech_fs}px;color:#666;}}"
            f"QPushButton:checked{{background:#76B7B2;color:white;border-color:#76B7B2;}}"
            f"QPushButton:hover:!checked{{background:#e8f0f0;}}"
        )
        for name, btn in self._cat_buttons.items():
            if name in _BEHAVIORAL_CATEGORIES:
                btn.setStyleSheet(chip_style)
            elif name in _TECHNICAL_CATEGORIES:
                btn.setStyleSheet(tech_style)
            else:
                btn.setStyleSheet(chip_style)
        lbl_fs = max(9, int(12 * s))
        self._lbl_title.setStyleSheet(
            f"font-weight:bold; font-size:{lbl_fs}px; color:#333;"
        )
        cat_fs = max(9, int(11 * s))
        self._cat_label.setStyleSheet(f"font-size:{cat_fs}px;")
        more_fs = max(8, int(11 * s))
        self._more_btn.setStyleSheet(
            f"font-size:{more_fs}px; color:#666; border:none;"
        )
        btn_fs = max(9, int(12 * s))
        btn_style = f"font-size:{btn_fs}px;"
        self._save_btn.setStyleSheet(btn_style)
        self._save_next_btn.setStyleSheet(btn_style)
        self._back_btn.setStyleSheet(btn_style)
        if self._kin_title:
            title_fs = max(9, int(11 * s))
            self._kin_title.setFont(QFont("Arial", title_fs, QFont.Bold))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._apply_scaled_styles()
        if self._kin_metrics:
            self._draw_kinematic_chart()

    # ──────────────────────────── State panel collapse ──

    def _toggle_state_panel(self) -> None:
        self._set_collapsed(not self._panel_collapsed)

    def _set_collapsed(self, collapsed: bool) -> None:
        if collapsed == self._panel_collapsed:
            return
        self._panel_collapsed = collapsed

        self._search.setVisible(not collapsed)
        self._sort_row_widget.setVisible(not collapsed)
        self._state_list.setVisible(not collapsed)
        self._panel_title.setVisible(not collapsed)
        self._collapsed_list.setVisible(collapsed)

        if collapsed:
            self._collapse_btn.setText("»")
            self._collapse_btn.setToolTip("Expand state list")
            sizes = self._splitter.sizes()
            total = sum(sizes)
            self._splitter.setSizes([56, total - 56])
            self._sync_collapsed_list()
        else:
            self._collapse_btn.setText("«")
            self._collapse_btn.setToolTip("Collapse state list")
            sizes = self._splitter.sizes()
            total = sum(sizes)
            self._splitter.setSizes([self._left_expanded_width,
                                     total - self._left_expanded_width])
            row = self._collapsed_list.currentRow()
            if 0 <= row < self._state_list.count():
                self._state_list.setCurrentRow(row)

    def _on_splitter_moved(self, _pos: int, _index: int) -> None:
        left_width = self._splitter.sizes()[0]
        if left_width < self._collapse_threshold and not self._panel_collapsed:
            self._set_collapsed(True)
        elif left_width >= self._collapse_threshold and self._panel_collapsed:
            self._set_collapsed(False)

    def _sync_collapsed_list(self) -> None:
        self._collapsed_list.clear()
        for sid in self._state_ids:
            self._collapsed_list.addItem(str(sid))
        row = self._state_list.currentRow()
        if 0 <= row < self._collapsed_list.count():
            self._collapsed_list.blockSignals(True)
            self._collapsed_list.setCurrentRow(row)
            self._collapsed_list.blockSignals(False)

    def _on_collapsed_selected(self, row: int) -> None:
        if 0 <= row < self._state_list.count():
            self._state_list.setCurrentRow(row)

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
        checked = self._cat_group.checkedButton()
        category = checked.text() if checked else ""
        label = self._label_input.text().strip()
        if not label and category:
            label = category
        existing[sid] = {"label": label, "category": category}
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
                if self._autoplay_cb.isChecked():
                    self._player.play()
            except Exception:
                self._clip_status.setText(f"Error loading {chosen.name}")

    def _toggle_technical_cats(self, expanded: bool) -> None:
        self._tech_row_widget.setVisible(expanded)
        self._more_btn.setText("More ▾" if expanded else "More ▸")

    def _save_and_next(self) -> None:
        self._save_state_label()
        current_row = self._state_list.currentRow()
        next_row = current_row + 1
        while next_row < self._state_list.count():
            if not self._state_list.item(next_row).isHidden():
                break
            next_row += 1
        if next_row >= self._state_list.count():
            self._clip_status.setText("All states labeled!")
            return
        self._state_list.setCurrentRow(next_row)
        self._load_clip()

    def _go_back(self) -> None:
        current_row = self._state_list.currentRow()
        prev_row = current_row - 1
        while prev_row >= 0:
            if not self._state_list.item(prev_row).isHidden():
                break
            prev_row -= 1
        if prev_row < 0:
            self._clip_status.setText("Already at first state.")
            return
        self._state_list.setCurrentRow(prev_row)
        self._load_clip()

    def _add_custom_category(self) -> None:
        name = self._cat_input.text().strip()
        if not name or name in self._cat_buttons:
            self._cat_input.clear()
            return
        btn = QPushButton(name)
        btn.setCheckable(True)
        btn.setStyleSheet(_CHIP_STYLE)
        self._cat_group.addButton(btn)
        stretch = self._custom_cat_chips_layout.takeAt(
            self._custom_cat_chips_layout.count() - 1,
        )
        self._custom_cat_chips_layout.insertWidget(
            self._custom_cat_chips_layout.count(), btn,
        )
        if stretch:
            self._custom_cat_chips_layout.addItem(stretch)
        self._cat_buttons[name] = btn
        self._cat_input.clear()
        custom = self.cfg.get("state_categories", [])
        if name not in custom:
            custom.append(name)
            self.cfg["state_categories"] = custom
            _save_cfg(self.cfg)
        self._apply_scaled_styles()

    def _restore_custom_categories(self) -> None:
        for name in self.cfg.get("state_categories", []):
            if name not in self._cat_buttons:
                btn = QPushButton(name)
                btn.setCheckable(True)
                btn.setStyleSheet(_CHIP_STYLE)
                self._cat_group.addButton(btn)
                stretch = self._custom_cat_chips_layout.takeAt(
                    self._custom_cat_chips_layout.count() - 1,
                )
                self._custom_cat_chips_layout.insertWidget(
                    self._custom_cat_chips_layout.count(), btn,
                )
                if stretch:
                    self._custom_cat_chips_layout.addItem(stretch)
                self._cat_buttons[name] = btn

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
