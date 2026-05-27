from __future__ import annotations
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QComboBox, QFileDialog, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMessageBox,
    QPushButton, QSizePolicy, QSlider,
    QTabWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from _utils import ROOT, RESULTS, VALIDATION_DIR, _open_folder, _save_cfg, _CV2, _MPL

if _CV2:
    import cv2
if _MPL:
    from _utils import Figure, FigureCanvas, mpl_cm
    from _widgets import MplCanvas

from _widgets import VideoPlayer, KinematicsPanel

BASE_DIR = Path(__file__).parent.parent.resolve()


class _ValidationPlayer(VideoPlayer):
    """VideoPlayer with loop-on default, 2x speed option, and frame counter."""

    def _build(self):
        super()._build()
        self._speed_combo.addItem("2x")
        self._loop_btn.setChecked(True)
        self._frame_lbl = QLabel("Frame: 0 / 0")
        self._frame_lbl.setAlignment(Qt.AlignCenter)
        self._frame_lbl.setStyleSheet("color:#888;font-size:11px;")
        self.layout().addWidget(self._frame_lbl)

    def _show(self, idx):
        super()._show(idx)
        self._frame_lbl.setText(f"Frame: {self._cur + 1} / {self._total}")

    def load(self, path: str):
        super().load(path)
        self._frame_lbl.setText(f"Frame: 1 / {self._total}")


class ValidationView(QWidget):
    navigate_to_pipeline = pyqtSignal()

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._data = {}
        self._current_clip: Path | None = None
        self._state_id: int | None = None
        # Advanced tab state (mirrors original frame-sampling implementation)
        self._adv_sample = None
        self._adv_cursor = 0
        self._adv_pose_cache: dict = {}
        self._adv_cap_cache: dict = {}
        self._adv_feature_cache: dict = {}
        self._adv_label_map = {
            Qt.Key_F: "freeze", Qt.Key_W: "walk", Qt.Key_G: "groom",
            Qt.Key_R: "rear", Qt.Key_O: "other", Qt.Key_S: "skip",
        }
        self.setFocusPolicy(Qt.StrongFocus)
        self._build()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(10)

        title = QLabel("Validation")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        outer.addWidget(title)
        outer.addSpacing(5)

        self._tabs = QTabWidget()
        outer.addWidget(self._tabs, stretch=1)

        self._watch_widget = QWidget()
        self._tabs.addTab(self._watch_widget, "Video Watching")
        self._build_watching()

        self._adv_widget = QWidget()
        self._tabs.addTab(self._adv_widget, "Frame Sampling (Advanced)")
        self._build_sampling()

    def _build_watching(self):
        layout = QHBoxLayout(self._watch_widget)
        layout.setSpacing(12)

        # ---- Left panel (fixed 300px) ----
        left = QWidget()
        left.setFixedWidth(300)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.setSpacing(8)

        lbl_title = QLabel("Behavioral State Validation")
        lbl_title.setFont(QFont("Arial", 13, QFont.Bold))
        lbl_title.setWordWrap(True)
        ll.addWidget(lbl_title)

        subtitle = QLabel(
            "Watch each clip and label what behavior the mouse is performing. "
            "Your labels validate that the discovered states correspond to real behaviors."
        )
        subtitle.setStyleSheet("color:gray;font-style:italic;font-size:11px;")
        subtitle.setWordWrap(True)
        ll.addWidget(subtitle)

        ll.addSpacing(4)

        rater_row = QHBoxLayout()
        rater_row.addWidget(QLabel("Rater:"))
        self._rater_edit = QLineEdit(self.cfg.get("rater_name", "Rater 1"))
        self._rater_edit.editingFinished.connect(self._save_rater)
        rater_row.addWidget(self._rater_edit)
        ll.addLayout(rater_row)

        ll.addWidget(QLabel("Select state to validate:"))
        self._state_combo = QComboBox()
        self._state_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ll.addWidget(self._state_combo)

        self._load_btn = QPushButton("Load Random Clip")
        self._load_btn.clicked.connect(self._load_random_clip)
        ll.addWidget(self._load_btn)

        self._clip_name_lbl = QLabel("")
        self._clip_name_lbl.setStyleSheet("color:gray;font-size:11px;")
        self._clip_name_lbl.setWordWrap(True)
        ll.addWidget(self._clip_name_lbl)

        ll.addSpacing(4)

        for btn_text, label in [
            ("[F]  Freeze", "freeze"),
            ("[W]  Walk",   "walk"),
            ("[G]  Groom",  "groom"),
            ("[R]  Rear",   "rear"),
            ("[O]  Other",  "other"),
        ]:
            btn = QPushButton(btn_text)
            btn.setMinimumHeight(44)
            btn.clicked.connect(lambda _, l=label: self._save_label(l))
            ll.addWidget(btn)

        skip_btn = QPushButton("[S]  Skip")
        skip_btn.setMinimumHeight(36)
        skip_btn.clicked.connect(self._skip_clip)
        ll.addWidget(skip_btn)

        self._counter_lbl = QLabel("Labeled: 0 clips")
        self._counter_lbl.setStyleSheet("font-weight:bold;")
        ll.addWidget(self._counter_lbl)

        self._progress_lbl = QLabel("")
        self._progress_lbl.setStyleSheet("color:#555;font-size:11px;")
        self._progress_lbl.setWordWrap(True)
        ll.addWidget(self._progress_lbl)

        ll.addWidget(QLabel("Summary:"))
        self._summary_table = QTableWidget(0, 4)
        self._summary_table.setHorizontalHeaderLabels(["State", "N labeled", "Top label", "Agreement"])
        self._summary_table.horizontalHeader().setStretchLastSection(True)
        self._summary_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._summary_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._summary_table.setMaximumHeight(140)
        ll.addWidget(self._summary_table)

        self._export_btn = QPushButton("Export Validation Report")
        self._export_btn.clicked.connect(self._export_report)
        ll.addWidget(self._export_btn)

        ll.addStretch()
        layout.addWidget(left)

        # ---- Right panel (fills remaining width): 70% video, 30% kinematics ----
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(4)

        self._player = _ValidationPlayer()
        self._player._display.setText("Select a state and click Load Random Clip")
        rl.addWidget(self._player, stretch=7)

        self._kin_panel = KinematicsPanel(parent=right)
        rl.addWidget(self._kin_panel, stretch=3)

        self._existing_lbl = QLabel("")
        self._existing_lbl.setAlignment(Qt.AlignCenter)
        self._existing_lbl.setStyleSheet("color:#888;font-style:italic;font-size:12px;")
        rl.addWidget(self._existing_lbl)

        # Cursor timer — updates kinematic panel at ~30fps
        self._kin_cursor_timer = QTimer(self)
        self._kin_cursor_timer.timeout.connect(self._update_kin_cursor)
        self._kin_cursor_timer.start(33)

        layout.addWidget(right, stretch=1)

    def _build_sampling(self):
        """Original frame-sampling implementation, preserved unchanged."""
        ll = QVBoxLayout(self._adv_widget)

        note = QLabel(
            "Advanced validation for paper figures.\n"
            "Requires characterize.py to have been run."
        )
        note.setStyleSheet(
            "background:#fff3cd;color:#856404;padding:10px;"
            "border-radius:4px;margin-bottom:6px;"
        )
        ll.addWidget(note)

        split = QHBoxLayout()
        split.setSpacing(10)
        ll.addLayout(split, stretch=2)

        left = QGroupBox("Frame Sampler")
        adv_ll = QVBoxLayout(left)
        self._adv_video_combo = QComboBox()
        self._adv_state_combo = QComboBox()
        self._adv_n_slider = QSlider(Qt.Horizontal)
        self._adv_n_slider.setRange(10, 200)
        self._adv_n_slider.setValue(50)
        self._adv_n_lbl = QLabel("Frames: 50")
        self._adv_n_slider.valueChanged.connect(lambda v: self._adv_n_lbl.setText(f"Frames: {v}"))
        self._adv_sample_btn = QPushButton("Sample Frames From Video")
        self._adv_sample_btn.clicked.connect(self._adv_sample_frames)
        adv_ll.addWidget(QLabel("Video to Validate"))
        adv_ll.addWidget(self._adv_video_combo)
        adv_ll.addWidget(QLabel("State"))
        adv_ll.addWidget(self._adv_state_combo)
        adv_ll.addWidget(self._adv_n_lbl)
        adv_ll.addWidget(self._adv_n_slider)
        adv_ll.addWidget(self._adv_sample_btn)
        self._adv_progress_lbl = QLabel("0 of 0 frames labeled")
        adv_ll.addWidget(self._adv_progress_lbl)
        split.addWidget(left, stretch=1)

        center = QGroupBox("Frame Display")
        cl = QVBoxLayout(center)
        self._adv_frame = QLabel("Select a video and sample frames to begin", alignment=Qt.AlignCenter)
        self._adv_frame.setMinimumSize(320, 240)
        self._adv_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._adv_frame.setStyleSheet("background:#111;color:#999;")
        cl.addWidget(self._adv_frame)
        self._adv_frame_info = QLabel("State: - | kinematics: -")
        cl.addWidget(self._adv_frame_info)
        shortcuts = QLabel("Shortcuts: F=freeze, W=walk, G=groom, R=rear, O=other, S=skip")
        shortcuts.setStyleSheet("color:#666;")
        cl.addWidget(shortcuts)
        split.addWidget(center, stretch=2)

        right = QGroupBox("Label Assignment")
        rl = QVBoxLayout(right)
        rl.setContentsMargins(8, 16, 8, 8)
        for name in ("Freeze", "Walk", "Groom", "Rear", "Other"):
            b = QPushButton(name)
            b.setMinimumHeight(44)
            b.clicked.connect(lambda _, n=name.lower(): self._adv_assign(n))
            rl.addWidget(b)
        skip = QPushButton("Skip")
        skip.setMinimumHeight(44)
        skip.clicked.connect(lambda: self._adv_assign("skip"))
        rl.addWidget(skip)
        rl.addStretch()
        split.addWidget(right, stretch=1)

        bottom = QGroupBox("Results")
        bottom.setMinimumHeight(180)
        bl_outer = QHBoxLayout(bottom)
        if _MPL:
            self._adv_cm_canvas = MplCanvas(figsize=(4, 2.2))
            bl_outer.addWidget(self._adv_cm_canvas, stretch=3)
        else:
            self._adv_cm_canvas = None
            bl_outer.addWidget(QLabel("Install matplotlib for confusion matrix heatmap."), stretch=3)
        bl_right = QVBoxLayout()
        self._adv_agree = QLabel("Agreement per state: -")
        self._adv_agree.setWordWrap(True)
        bl_right.addWidget(self._adv_agree)
        self._adv_export_btn = QPushButton("Export labels CSV")
        self._adv_export_btn.clicked.connect(self._adv_export)
        bl_right.addWidget(self._adv_export_btn)
        bl_right.addStretch()
        bl_outer.addLayout(bl_right, stretch=1)
        ll.addWidget(bottom)

    # ------------------------------------------------------------------
    # Data update (called from main GUI on reload)
    # ------------------------------------------------------------------

    def update_data(self, data: dict):
        self._data = data
        self._refresh_state_combo(data)
        self._refresh_summary()
        self._refresh_adv_combos(data)

    def _refresh_state_combo(self, data: dict):
        ci = data.get("cluster_info")
        clips_root = BASE_DIR / "clips"

        if ci is None:
            self._state_combo.clear()
            self._state_combo.addItem("No cluster data — run pipeline first", -1)
            self._load_btn.setEnabled(False)
            return

        if not clips_root.exists():
            self._load_btn.setEnabled(False)
            self._player._display.setText(
                "No clips directory found.\n\n"
                "Generate clips first:\n  python characterize.py --clips"
            )
        else:
            self._load_btn.setEnabled(True)

        n = int(ci.get("n_clusters", 0))
        ss = data.get("state_summary")
        label_map: dict[int, str] = {}
        if ss is not None and "state" in ss.columns and "heuristic_label" in ss.columns:
            for _, row in ss.iterrows():
                sid = int(row["state"])
                raw = str(row.get("heuristic_label", ""))
                parts = raw.split(":", 1)
                label_map[sid] = parts[1].strip() if len(parts) == 2 else raw

        prev = self._state_combo.currentData()
        self._state_combo.clear()
        for sid in range(n):
            hint = label_map.get(sid, "")
            text = f"State {sid} ({hint})" if hint else f"State {sid}"
            self._state_combo.addItem(text, sid)

        if prev is not None:
            for i in range(self._state_combo.count()):
                if self._state_combo.itemData(i) == prev:
                    self._state_combo.setCurrentIndex(i)
                    break

    def _refresh_adv_combos(self, data: dict):
        ci = data.get("cluster_info")
        lpf = data.get("labels_per_frame")
        if ci is None or lpf is None or lpf.empty:
            self._adv_sample_btn.setEnabled(False)
            self._adv_progress_lbl.setText("Run Characterization + Clip Export to generate this data.")
            return
        self._adv_sample_btn.setEnabled(True)
        n = int(ci.get("n_clusters", 0))
        summary = data.get("summary")
        dominant = -1
        if summary is not None:
            dominant = max(
                [(i, float(summary.get(f"state_{i}_frac", pd.Series([0])).mean())) for i in range(n)],
                key=lambda x: x[1],
            )[0]
        self._adv_state_combo.clear()
        for sid in range(n):
            if sid != dominant:
                self._adv_state_combo.addItem(f"State {sid}", sid)
        self._adv_video_combo.clear()
        if "stem" in lpf.columns:
            for s in sorted(lpf["stem"].dropna().astype(str).unique().tolist()):
                self._adv_video_combo.addItem(s, s)
        sample = data.get("validation_sample")
        if sample is not None and not sample.empty:
            cur_video = self._adv_video_combo.currentData() if self._adv_video_combo.count() else None
            cur_state = self._adv_state_combo.currentData() if self._adv_state_combo.count() else None
            resumed = sample
            if cur_video is not None and "stem" in sample.columns:
                resumed = resumed[resumed["stem"].astype(str) == str(cur_video)]
            if cur_state is not None and "cluster_label" in resumed.columns:
                resumed = resumed[resumed["cluster_label"] == int(cur_state)]
            self._adv_sample = resumed.reset_index(drop=True) if not resumed.empty else None
            if self._adv_sample is not None:
                self._adv_cursor = int((self._adv_sample["manual_label"].fillna("") != "").sum())
                self._adv_show_current()
                self._adv_refresh_results()

    # ------------------------------------------------------------------
    # Video-watching tab actions
    # ------------------------------------------------------------------

    def _save_rater(self):
        name = self._rater_edit.text().strip() or "Rater 1"
        self.cfg["rater_name"] = name
        _save_cfg(self.cfg)

    def _load_random_clip(self):
        state_id = self._state_combo.currentData()
        if state_id is None or state_id < 0:
            return

        clips_dir = BASE_DIR / "clips" / f"state_{state_id}"
        if not clips_dir.exists():
            self._player._display.setText(
                f"No clips found for State {state_id}.\n\n"
                "Generate clips first:\n  python characterize.py --clips"
            )
            self._clip_name_lbl.setText("")
            self._existing_lbl.setText("")
            self._kin_panel.clear()
            return

        clips = list(clips_dir.glob("*.mp4"))
        if not clips:
            self._player._display.setText(f"clips/state_{state_id}/ is empty.")
            self._kin_panel.clear()
            return

        clip = random.choice(clips)
        self._current_clip = clip
        self._state_id = state_id
        self._clip_name_lbl.setText(clip.name)
        self._player.load(str(clip))
        self._player.play()
        self._check_existing_label()
        self._load_clip_kinematics(clip)

    def _save_label(self, label: str):
        if self._current_clip is None:
            return
        rater = self._rater_edit.text().strip() or "Rater 1"
        row = {
            "state_id": self._state_id,
            "clip_filename": self._current_clip.name,
            "label": label,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "rater": rater,
        }
        labels_path = BASE_DIR / "results" / "validation" / "labels.csv"
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        if labels_path.exists():
            df = pd.read_csv(labels_path)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            df = pd.DataFrame([row])
        df.to_csv(labels_path, index=False)
        self._refresh_summary()
        self._load_random_clip()

    def _skip_clip(self):
        self._load_random_clip()

    def _check_existing_label(self):
        if self._current_clip is None:
            self._existing_lbl.setText("")
            return
        labels_path = BASE_DIR / "results" / "validation" / "labels.csv"
        if not labels_path.exists():
            self._existing_lbl.setText("")
            return
        df = pd.read_csv(labels_path)
        match = df[
            (df["state_id"] == self._state_id) &
            (df["clip_filename"] == self._current_clip.name)
        ]
        if not match.empty:
            self._existing_lbl.setText(f"Previously labeled: {match.iloc[-1]['label']}")
        else:
            self._existing_lbl.setText("")

    def _load_clip_kinematics(self, clip: Path):
        """Load feature data for the current clip in a background thread."""
        fi = self._data.get("feature_index") or {}
        if not fi:
            self._kin_panel.clear()
            return
        # Pick the first available feature file as a proxy for the clip's kinematics
        feat_path = None
        for vstem, info in fi.items():
            fp = (info.get("features_path") or "").replace("\\", "/")
            if fp and os.path.exists(fp):
                feat_path = fp
                break
        if feat_path is None:
            self._kin_panel.clear()
            return

        n_frames = self._player._total if hasattr(self._player, "_total") else 300
        import threading
        def _bg():
            self._kin_panel.load_clip(feat_path, 0, n_frames)
        threading.Thread(target=_bg, daemon=True).start()

    def _update_kin_cursor(self):
        """Called by QTimer at ~30fps to update the kinematic cursor position."""
        if hasattr(self, "_player") and hasattr(self._player, "_cur"):
            self._kin_panel.set_frame(self._player._cur)

    def _refresh_summary(self):
        labels_path = BASE_DIR / "results" / "validation" / "labels.csv"
        if not labels_path.exists():
            self._counter_lbl.setText("Labeled: 0 clips")
            self._progress_lbl.setText("")
            self._summary_table.setRowCount(0)
            return
        df = pd.read_csv(labels_path)
        self._counter_lbl.setText(f"Labeled: {len(df)} clips")

        rows = []
        progress_parts = []
        for sid, grp in df.groupby("state_id"):
            n = len(grp)
            vc = grp["label"].value_counts()
            top = vc.index[0] if not vc.empty else "-"
            agree = f"{100 * vc.iloc[0] / n:.0f}%" if not vc.empty else "-"
            rows.append((int(sid), n, top, agree))
            progress_parts.append(f"State {sid}: {n} clips labeled")

        self._progress_lbl.setText("  |  ".join(progress_parts))
        self._summary_table.setRowCount(len(rows))
        for r, (sid, n, top, agree) in enumerate(rows):
            self._summary_table.setItem(r, 0, QTableWidgetItem(str(sid)))
            self._summary_table.setItem(r, 1, QTableWidgetItem(str(n)))
            self._summary_table.setItem(r, 2, QTableWidgetItem(top))
            self._summary_table.setItem(r, 3, QTableWidgetItem(agree))
        self._summary_table.resizeColumnsToContents()

    def _export_report(self):
        labels_path = BASE_DIR / "results" / "validation" / "labels.csv"
        if not labels_path.exists():
            QMessageBox.information(self, "Validation", "No labels to export yet.")
            return
        df = pd.read_csv(labels_path)
        summary_rows = []
        for sid, grp in df.groupby("state_id"):
            n = len(grp)
            vc = grp["label"].value_counts()
            summary_rows.append({
                "state_id": sid,
                "n_labeled": n,
                "top_label": vc.index[0] if not vc.empty else "-",
                "agreement_pct": round(100 * vc.iloc[0] / n, 1) if not vc.empty else None,
            })
        out_dir = BASE_DIR / "results" / "validation"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / "validation_summary.csv"
        pd.DataFrame(summary_rows).to_csv(out, index=False)
        QMessageBox.information(self, "Validation", f"Report saved to:\n{out}")

    def _run_characterize_clips(self):
        script = BASE_DIR / "characterize.py"
        try:
            kwargs = {}
            if sys.platform == "win32":
                kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
            subprocess.Popen([sys.executable, str(script), "--clips"],
                             cwd=str(BASE_DIR), **kwargs)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not launch characterize.py:\n{e}")

    # ------------------------------------------------------------------
    # Key shortcuts (active for whichever tab is current)
    # ------------------------------------------------------------------

    def keyPressEvent(self, e):
        if self._tabs.currentIndex() == 0:
            label_map = {
                Qt.Key_F: "freeze", Qt.Key_W: "walk", Qt.Key_G: "groom",
                Qt.Key_R: "rear", Qt.Key_O: "other",
            }
            if e.key() in label_map:
                self._save_label(label_map[e.key()])
                e.accept()
                return
            if e.key() == Qt.Key_S:
                self._skip_clip()
                e.accept()
                return
        else:
            if e.key() in self._adv_label_map:
                self._adv_assign(self._adv_label_map[e.key()])
                e.accept()
                return
        super().keyPressEvent(e)

    # ------------------------------------------------------------------
    # Advanced tab — original frame-sampling methods (prefixed _adv_)
    # ------------------------------------------------------------------

    def _adv_sample_frames(self):
        lpf = self._data.get("labels_per_frame")
        fi = self._data.get("feature_index") or {}
        if lpf is None or lpf.empty:
            return
        stem = self._adv_video_combo.currentData()
        if not stem:
            QMessageBox.information(self, "Validation", "Select a video to validate.")
            return
        sid = int(self._adv_state_combo.currentData())
        n = int(self._adv_n_slider.value())
        sub = lpf[(lpf["state"] == sid) & (lpf["stem"].astype(str) == str(stem))]
        if sub.empty:
            QMessageBox.information(self, "Validation",
                                    "No frames available for this state in selected video.")
            return
        if len(sub) > n:
            sub = sub.sample(n=n, random_state=42)
        sub = sub.copy()
        sub.rename(columns={"frame": "frame_idx"}, inplace=True)
        sub["cluster_label"] = sub["state"]
        sub["manual_label"] = ""
        sub["timestamp"] = ""
        for idx, row in sub.iterrows():
            s = row["stem"]
            info = fi.get(s, {}) if isinstance(fi, dict) else {}
            sub.at[idx, "video_path"] = info.get(
                "video_path", str(ROOT / "raw_videos" / f"{s}.mp4"))
            sub.at[idx, "csv_path"] = info.get("csv_path", "")
        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        sub.to_csv(VALIDATION_DIR / "current_sample.csv", index=False)
        self._adv_sample = sub.reset_index(drop=True)
        self._adv_cursor = 0
        self._adv_show_current()
        self._adv_refresh_results()

    def _adv_load_pose(self, csv_path):
        if csv_path in self._adv_pose_cache:
            return self._adv_pose_cache[csv_path]
        try:
            from main import load_pose
            pose, conf, _ = load_pose(csv_path)
            self._adv_pose_cache[csv_path] = (pose, conf)
            return pose, conf
        except Exception:
            return None, None

    def _adv_draw_frame(self, row):
        if not _CV2:
            self._adv_frame.setText(
                "opencv-python is required to display frames.\n"
                "Install it with:  pip install opencv-python"
            )
            return
        video = row.get("video_path", "")
        frame_idx = int(row.get("frame_idx", 0))
        csv_path = row.get("csv_path", "")
        if video and not os.path.isabs(video):
            video = str(ROOT / video)
        if not video or not os.path.exists(video):
            self._adv_frame.setText(
                f"Video not found:\n{video}\n\n"
                "Check that raw_videos/ contains the .mp4 files."
            )
            return
        cap = self._adv_cap_cache.get(video)
        if cap is None:
            cap = cv2.VideoCapture(video)
            self._adv_cap_cache[video] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            self._adv_frame.setText(f"Could not read frame {frame_idx} from:\n{os.path.basename(video)}")
            return
        pose, _ = self._adv_load_pose(csv_path)
        if pose is not None and frame_idx < len(pose):
            pts = pose[frame_idx]
            colors = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                (255, 0, 255), (0, 255, 255), (255, 128, 0), (128, 0, 255),
            ]
            for i, pt in enumerate(pts):
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 4, colors[i], -1)
            for a, b in [(2, 3), (3, 6), (0, 1)]:
                cv2.line(frame, tuple(np.int32(pts[a])), tuple(np.int32(pts[b])), (255, 255, 255), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        mxw, mxh = self._adv_frame.width(), self._adv_frame.height()
        sc = min(mxw / w, mxh / h)
        nw, nh = int(w * sc), int(h * sc)
        frame = cv2.resize(frame, (nw, nh))
        buf = frame.copy()
        img = QImage(buf.data, nw, nh, 3 * nw, QImage.Format_RGB888)
        self._adv_frame.setPixmap(QPixmap.fromImage(img))

    def _adv_kinematic_text(self, row):
        fi = self._data.get("feature_index") or {}
        stem = row.get("stem", "")
        frame_idx = int(row.get("frame_idx", 0))
        info = fi.get(stem, {}) if isinstance(fi, dict) else {}
        fp = info.get("features_path", "")
        if not fp:
            return "-"
        arr = self._adv_feature_cache.get(fp)
        if arr is None and Path(fp).exists():
            arr = np.load(fp)
            self._adv_feature_cache[fp] = arr
        if arr is None or frame_idx >= len(arr):
            return "-"
        feat = arr[frame_idx]
        return f"speed={feat[36]:.3f}, ang_vel={feat[39]:.3f}, entropy={feat[40]:.3f}"

    def _adv_show_current(self):
        if self._adv_sample is None or self._adv_sample.empty:
            self._adv_frame.setText("No sample loaded.")
            return
        unl = self._adv_sample["manual_label"].fillna("") == ""
        if unl.sum() == 0:
            self._adv_frame.setText("All frames labeled.")
            return
        self._adv_cursor = int(self._adv_sample.index[unl][0])
        row = self._adv_sample.loc[self._adv_cursor]
        self._adv_draw_frame(row)
        self._adv_frame_info.setText(
            f"State {int(row.get('cluster_label', -1))} | "
            f"frame {int(row.get('frame_idx', 0))} | {self._adv_kinematic_text(row)}"
        )
        done = int((self._adv_sample["manual_label"].fillna("") != "").sum())
        self._adv_progress_lbl.setText(f"{done} of {len(self._adv_sample)} frames labeled")

    def _adv_assign(self, manual_label: str):
        if self._adv_sample is None or self._adv_sample.empty:
            return
        self._adv_sample.at[self._adv_cursor, "manual_label"] = manual_label
        self._adv_sample.at[self._adv_cursor, "timestamp"] = datetime.now().isoformat(timespec="seconds")
        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        self._adv_sample.to_csv(VALIDATION_DIR / "current_sample.csv", index=False)
        self._adv_sample.to_csv(VALIDATION_DIR / "frame_labels.csv", index=False)
        self._adv_refresh_results()
        self._adv_show_current()

    def _adv_refresh_results(self):
        if self._adv_sample is None or self._adv_sample.empty:
            return
        done = self._adv_sample[self._adv_sample["manual_label"].fillna("") != ""]
        if done.empty:
            self._adv_agree.setText("Agreement per state: -")
            if self._adv_cm_canvas:
                self._adv_cm_canvas.ax.clear()
                self._adv_cm_canvas.ax.text(0.5, 0.5, "No labels yet", ha="center", va="center")
                self._adv_cm_canvas.draw()
            return
        cros = pd.crosstab(done["cluster_label"], done["manual_label"])
        if self._adv_cm_canvas:
            self._adv_cm_canvas.ax.clear()
            self._adv_cm_canvas.ax.imshow(cros.values, aspect="auto", cmap="Blues")
            self._adv_cm_canvas.ax.set_xticks(range(len(cros.columns)))
            self._adv_cm_canvas.ax.set_xticklabels(cros.columns, rotation=45, ha="right")
            self._adv_cm_canvas.ax.set_yticks(range(len(cros.index)))
            self._adv_cm_canvas.ax.set_yticklabels(cros.index)
            self._adv_cm_canvas.ax.set_xlabel("Manual Label")
            self._adv_cm_canvas.ax.set_ylabel("Cluster")
            self._adv_cm_canvas.fig.tight_layout()
            self._adv_cm_canvas.draw()
        agreements = []
        for sid, grp in done.groupby("cluster_label"):
            top = grp["manual_label"].value_counts().max()
            agreements.append(f"S{sid}: {100 * top / len(grp):.1f}%")
        self._adv_agree.setText("Agreement per state: " + ", ".join(agreements))

    def _adv_export(self):
        p = VALIDATION_DIR / "frame_labels.csv"
        if not p.exists():
            QMessageBox.information(self, "Validation", "No labels to export yet.")
            return
        d = QFileDialog.getExistingDirectory(self, "Select Destination", str(ROOT))
        if not d:
            return
        dst = Path(d) / "frame_labels.csv"
        shutil.copy2(p, dst)
        QMessageBox.information(self, "Validation", f"Exported to {dst}")
