from __future__ import annotations
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPixmap
from PyQt5.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QFrame,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton,
    QScrollArea, QSlider, QSpinBox, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget,
)

from _utils import ROOT, RESULTS, VALIDATION_DIR, _open_folder, _CV2, _MPL

if _CV2:
    import cv2
if _MPL:
    from _utils import Figure, FigureCanvas, mpl_cm
    from _widgets import MplCanvas


class ValidationView(QWidget):
    navigate_to_pipeline = pyqtSignal()

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._data = {}
        self._sample = None
        self._cursor = 0
        self._pose_cache = {}
        self._cap_cache = {}
        self._feature_cache = {}
        self._label_map = {
            Qt.Key_F: "freeze",
            Qt.Key_W: "walk",
            Qt.Key_G: "groom",
            Qt.Key_R: "rear",
            Qt.Key_O: "other",
            Qt.Key_S: "skip",
        }
        self.setFocusPolicy(Qt.StrongFocus)
        self._build()

    def _build(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(10)
        title = QLabel("Validation")
        title.setFont(QFont("Arial", 18, QFont.Bold))
        outer.addWidget(title)
        outer.addSpacing(5)

        split = QHBoxLayout()
        split.setSpacing(10)
        outer.addLayout(split, stretch=1)

        left = QGroupBox("Frame Sampler")
        ll = QVBoxLayout(left)
        self._video_combo = QComboBox()
        self._state_combo = QComboBox()
        self._n_slider = QSlider(Qt.Horizontal)
        self._n_slider.setRange(10, 200)
        self._n_slider.setValue(50)
        self._n_lbl = QLabel("Frames: 50")
        self._n_slider.valueChanged.connect(lambda v: self._n_lbl.setText(f"Frames: {v}"))
        self._sample_btn = QPushButton("Sample Frames From Video")
        self._sample_btn.clicked.connect(self._sample_frames)
        ll.addWidget(QLabel("Video to Validate"))
        ll.addWidget(self._video_combo)
        ll.addWidget(QLabel("State"))
        ll.addWidget(self._state_combo)
        ll.addWidget(self._n_lbl)
        ll.addWidget(self._n_slider)
        ll.addWidget(self._sample_btn)
        self._progress_lbl = QLabel("0 of 0 frames labeled")
        ll.addWidget(self._progress_lbl)
        split.addWidget(left, stretch=1)

        center = QGroupBox("Frame Display")
        cl = QVBoxLayout(center)
        self._frame = QLabel("Select a video and sample frames to begin", alignment=Qt.AlignCenter)
        self._frame.setMinimumSize(540, 360)
        self._frame.setStyleSheet("background:#111;color:#999;")
        cl.addWidget(self._frame)
        self._frame_info = QLabel("State: - | kinematics: -")
        cl.addWidget(self._frame_info)
        self._shortcuts = QLabel("Shortcuts: F=freeze, W=walk, G=groom, R=rear, O=other, S=skip")
        self._shortcuts.setStyleSheet("color:#666;")
        cl.addWidget(self._shortcuts)
        split.addWidget(center, stretch=2)

        right = QGroupBox("Label Assignment")
        rl = QVBoxLayout(right)
        for name in ("Freeze", "Walk", "Groom", "Rear", "Other"):
            b = QPushButton(name)
            b.setMinimumHeight(44)
            b.clicked.connect(lambda _, n=name.lower(): self._assign(n))
            rl.addWidget(b)
        skip = QPushButton("Skip")
        skip.setMinimumHeight(44)
        skip.clicked.connect(lambda: self._assign("skip"))
        rl.addWidget(skip)
        rl.addStretch()
        split.addWidget(right, stretch=1)

        bottom = QGroupBox("Results")
        bl = QVBoxLayout(bottom)
        if _MPL:
            self._cm_canvas = MplCanvas(figsize=(5, 3))
            bl.addWidget(self._cm_canvas)
        else:
            self._cm_canvas = None
            bl.addWidget(QLabel("Install matplotlib for confusion matrix heatmap."))
        self._agree = QLabel("Agreement per state: -")
        bl.addWidget(self._agree)
        self._export_btn = QPushButton("Export labels CSV")
        self._export_btn.clicked.connect(self._export_validation)
        bl.addWidget(self._export_btn)
        outer.addWidget(bottom)

    def update_data(self, data):
        self._data = data
        ci = data.get("cluster_info")
        lpf = data.get("labels_per_frame")
        if ci is None or lpf is None or lpf.empty:
            self._sample_btn.setEnabled(False)
            self._progress_lbl.setText("Run Characterization + Clip Export to generate this data.")
            return
        self._sample_btn.setEnabled(True)
        n = int(ci.get("n_clusters", 0))
        summary = data.get("summary")
        dominant = -1
        if summary is not None:
            dominant = max(
                [(i, float(summary.get(f"state_{i}_frac", pd.Series([0])).mean())) for i in range(n)],
                key=lambda x: x[1],
            )[0]
        self._state_combo.clear()
        for sid in range(n):
            if sid != dominant:
                self._state_combo.addItem(f"State {sid}", sid)
        self._video_combo.clear()
        if "stem" in lpf.columns:
            stems = sorted(lpf["stem"].dropna().astype(str).unique().tolist())
            for s in stems:
                self._video_combo.addItem(s, s)
        sample = data.get("validation_sample")
        if sample is not None and not sample.empty:
            # Resume previous work only when same selected video/state exists in sample
            current_video = self._video_combo.currentData() if self._video_combo.count() else None
            current_state = self._state_combo.currentData() if self._state_combo.count() else None
            resumed = sample
            if current_video is not None and "stem" in sample.columns:
                resumed = resumed[resumed["stem"].astype(str) == str(current_video)]
            if current_state is not None and "cluster_label" in resumed.columns:
                resumed = resumed[resumed["cluster_label"] == int(current_state)]
            self._sample = resumed.reset_index(drop=True) if not resumed.empty else None
            if self._sample is not None:
                self._cursor = int((self._sample["manual_label"].fillna("") != "").sum())
                self._show_current()
                self._refresh_results()

    def _sample_frames(self):
        lpf = self._data.get("labels_per_frame")
        fi = self._data.get("feature_index") or {}
        if lpf is None or lpf.empty:
            return
        stem = self._video_combo.currentData()
        if not stem:
            QMessageBox.information(self, "Validation", "Select a video to validate.")
            return
        sid = int(self._state_combo.currentData())
        n = int(self._n_slider.value())
        sub = lpf[(lpf["state"] == sid) & (lpf["stem"].astype(str) == str(stem))]
        if sub.empty:
            QMessageBox.information(self, "Validation", "No frames available for this state in selected video.")
            return
        if len(sub) > n:
            sub = sub.sample(n=n, random_state=42)
        sub = sub.copy()
        sub.rename(columns={"frame": "frame_idx"}, inplace=True)
        sub["cluster_label"] = sub["state"]
        sub["manual_label"] = ""
        sub["timestamp"] = ""
        for idx, row in sub.iterrows():
            stem = row["stem"]
            info = fi.get(stem, {}) if isinstance(fi, dict) else {}
            sub.at[idx, "video_path"] = info.get("video_path", str(ROOT / "raw_videos" / f"{stem}.mp4"))
            sub.at[idx, "csv_path"] = info.get("csv_path", "")
        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        sub.to_csv(VALIDATION_DIR / "current_sample.csv", index=False)
        self._sample = sub.reset_index(drop=True)
        self._cursor = 0
        self._show_current()
        self._refresh_results()

    def _load_pose(self, csv_path):
        if csv_path in self._pose_cache:
            return self._pose_cache[csv_path]
        try:
            from pose_io import load_pose

            pose, conf, _ = load_pose(csv_path)
            self._pose_cache[csv_path] = (pose, conf)
            return pose, conf
        except Exception:
            return None, None

    def _draw_frame(self, row):
        if not _CV2:
            self._frame.setText(
                "opencv-python is required to display frames.\n"
                "Install it with:  pip install opencv-python"
            )
            return
        video = row.get("video_path", "")
        frame_idx = int(row.get("frame_idx", 0))
        csv_path = row.get("csv_path", "")
        # Resolve relative paths against the project root so they work regardless
        # of the process working directory when the GUI was launched.
        if video and not os.path.isabs(video):
            video = str(ROOT / video)
        if not video or not os.path.exists(video):
            self._frame.setText(
                f"Video not found:\n{video}\n\n"
                "Check that raw_videos/ contains the .mp4 files."
            )
            return
        cap = self._cap_cache.get(video)
        if cap is None:
            cap = cv2.VideoCapture(video)
            self._cap_cache[video] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            self._frame.setText(f"Could not read frame {frame_idx} from:\n{os.path.basename(video)}")
            return
        pose, _ = self._load_pose(csv_path)
        if pose is not None and frame_idx < len(pose):
            pts = pose[frame_idx]
            colors = [
                (255, 0, 0),
                (0, 255, 0),
                (0, 0, 255),
                (255, 255, 0),
                (255, 0, 255),
                (0, 255, 255),
                (255, 128, 0),
                (128, 0, 255),
            ]
            for i, pt in enumerate(pts):
                x, y = int(pt[0]), int(pt[1])
                cv2.circle(frame, (x, y), 4, colors[i], -1)
            edges = [(2, 3), (3, 6), (0, 1)]
            for a, b in edges:
                pa = tuple(np.int32(pts[a]))
                pb = tuple(np.int32(pts[b]))
                cv2.line(frame, pa, pb, (255, 255, 255), 2)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        mxw, mxh = self._frame.width(), self._frame.height()
        sc = min(mxw / w, mxh / h)
        nw, nh = int(w * sc), int(h * sc)
        frame = cv2.resize(frame, (nw, nh))
        buf = frame.copy()
        img = QImage(buf.data, nw, nh, 3 * nw, QImage.Format_RGB888)
        self._frame.setPixmap(QPixmap.fromImage(img))

    def _kinematic_text(self, row):
        fi = self._data.get("feature_index") or {}
        stem = row.get("stem", "")
        frame_idx = int(row.get("frame_idx", 0))
        info = fi.get(stem, {}) if isinstance(fi, dict) else {}
        fp = info.get("features_path", "")
        if not fp:
            return "-"
        arr = self._feature_cache.get(fp)
        if arr is None and Path(fp).exists():
            arr = np.load(fp)
            self._feature_cache[fp] = arr
        if arr is None or frame_idx >= len(arr):
            return "-"
        feat = arr[frame_idx]
        return f"speed={feat[36]:.3f}, ang_vel={feat[39]:.3f}, entropy={feat[40]:.3f}"

    def _show_current(self):
        if self._sample is None or self._sample.empty:
            self._frame.setText("No sample loaded.")
            return
        unl = self._sample["manual_label"].fillna("") == ""
        if unl.sum() == 0:
            self._frame.setText("All frames labeled.")
            return
        idxs = self._sample.index[unl]
        self._cursor = int(idxs[0])
        row = self._sample.loc[self._cursor]
        self._draw_frame(row)
        self._frame_info.setText(
            f"State {int(row.get('cluster_label', -1))} | "
            f"frame {int(row.get('frame_idx', 0))} | {self._kinematic_text(row)}"
        )
        done = int((self._sample["manual_label"].fillna("") != "").sum())
        self._progress_lbl.setText(f"{done} of {len(self._sample)} frames labeled")

    def _assign(self, manual_label):
        if self._sample is None or self._sample.empty:
            return
        self._sample.at[self._cursor, "manual_label"] = manual_label
        self._sample.at[self._cursor, "timestamp"] = datetime.now().isoformat(timespec="seconds")
        VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
        self._sample.to_csv(VALIDATION_DIR / "current_sample.csv", index=False)
        self._sample.to_csv(VALIDATION_DIR / "frame_labels.csv", index=False)
        self._refresh_results()
        self._show_current()

    def _refresh_results(self):
        if self._sample is None or self._sample.empty:
            return
        done = self._sample[self._sample["manual_label"].fillna("") != ""]
        if done.empty:
            self._agree.setText("Agreement per state: -")
            if self._cm_canvas:
                self._cm_canvas.ax.clear()
                self._cm_canvas.ax.text(0.5, 0.5, "No labels yet", ha="center", va="center")
                self._cm_canvas.draw()
            return
        cros = pd.crosstab(done["cluster_label"], done["manual_label"])
        if self._cm_canvas:
            self._cm_canvas.ax.clear()
            self._cm_canvas.ax.imshow(cros.values, aspect="auto", cmap="Blues")
            self._cm_canvas.ax.set_xticks(range(len(cros.columns)))
            self._cm_canvas.ax.set_xticklabels(cros.columns, rotation=45, ha="right")
            self._cm_canvas.ax.set_yticks(range(len(cros.index)))
            self._cm_canvas.ax.set_yticklabels(cros.index)
            self._cm_canvas.ax.set_xlabel("Manual Label")
            self._cm_canvas.ax.set_ylabel("Cluster")
            self._cm_canvas.fig.tight_layout()
            self._cm_canvas.draw()
        agreements = []
        for sid, grp in done.groupby("cluster_label"):
            top = grp["manual_label"].value_counts().max()
            agreements.append(f"S{sid}: {100 * top / len(grp):.1f}%")
        self._agree.setText("Agreement per state: " + ", ".join(agreements))

    def _export_validation(self):
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

    def keyPressEvent(self, e):
        if e.key() in self._label_map:
            self._assign(self._label_map[e.key()])
            e.accept()
            return
        super().keyPressEvent(e)
