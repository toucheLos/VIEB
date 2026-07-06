"""Video Stories: per-video state-sequence browser (Analysis > Video Stories).

Reads results/sequences/video_stories.csv + video_story_bouts.csv (written by
sequence_artifacts.build_sequence_artifacts, run automatically inside
compare.py's cmd_report()). Falls back to results/characterization/bouts.csv
+ results/comparison/summary_table.csv when the sequence artifacts are
missing. Never raises on missing/malformed data.
"""

from __future__ import annotations

import ast
import json
import math
import os
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QFormLayout, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

import vieb_config as _vc
from _utils import RESULTS, _MPL, _state_colors
from _workers import SubprocessWorker
from generate_clips import _export_clip, _resolve_video_path
from views.analysis import TerminalBox, _placeholder, _section_title, _scroll_content_widget

if _MPL:
    from _widgets import MplCanvas
from _widgets import VideoPlayer


# ---------------------------------------------------------------------------
# Pure-logic helpers — no Qt, directly unit-testable.
# ---------------------------------------------------------------------------

_LEGACY_BOUT_RENAME = {"stem": "video_id", "animal_id": "subject_id", "day": "timepoint", "context": "condition"}


def _is_blank(value) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    return text == "" or text.lower() == "nan"


def _entropy(vector: np.ndarray) -> float:
    arr = np.maximum(np.asarray(vector, dtype=float), 0)
    total = float(arr.sum())
    if total <= 0:
        return float("nan")
    p = arr / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def load_state_labels(results_dir) -> dict[int, dict]:
    """Read results/validation/state_labels.csv — the persistence layer
    already built for State Characterization labels. Same file, same
    read logic as views/state_characterization.py:_load_saved_state_labels."""
    p = Path(results_dir) / "validation" / "state_labels.csv"
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
        result = {}
        for _, row in df.iterrows():
            label, category = row.get("label", ""), row.get("category", "")
            result[int(row.get("state_id", -1))] = {
                "label": "" if pd.isna(label) else str(label),
                "category": "" if pd.isna(category) else str(category),
            }
        return result
    except Exception:
        return {}


def order_timepoints(values, results_dir) -> list:
    """Sort distinct timepoint values using results/analysis_design.json's
    time_order (baseline < week2 < ... per analysis_design._time_key) when
    present, falling back to a natural sort otherwise."""
    clean = [v for v in values if not _is_blank(v)]
    time_order = None
    design_path = Path(results_dir) / "analysis_design.json"
    if design_path.exists():
        try:
            design = json.loads(design_path.read_text(encoding="utf-8"))
            time_order = design.get("time_order")
        except Exception:
            time_order = None
    available = {str(v): v for v in clean}
    ordered = []
    if time_order:
        for item in time_order:
            key = str(item)
            if key in available:
                ordered.append(available[key])
    seen = {str(v) for v in ordered}
    rest = [v for v in clean if str(v) not in seen]
    try:
        rest = sorted(rest)
    except TypeError:
        rest = sorted(rest, key=str)
    return ordered + rest


def build_fallback_stories(bouts_raw: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Adapt legacy results/characterization/bouts.csv + results/comparison/
    summary_table.csv into the video_stories/video_story_bouts shape, for
    projects that haven't re-run compare.py --report since stories shipped."""
    bouts = bouts_raw.rename(columns=_LEGACY_BOUT_RENAME).copy()
    bouts["video_id"] = bouts["video_id"].astype(str)
    bouts["confidence_mean"] = np.nan
    for col in ("subject_id", "timepoint", "condition"):
        if col not in bouts.columns:
            bouts[col] = ""

    def _col_state_id(col: str) -> int:
        try:
            return int(str(col).split("_")[1])
        except Exception:
            return -1

    state_cols = sorted(
        (c for c in summary.columns if str(c).startswith("state_") and str(c).endswith("_frac")),
        key=_col_state_id,
    )
    state_ids = [_col_state_id(c) for c in state_cols]
    summary = summary.rename(columns={"stem": "video_id"}).copy()
    summary["video_id"] = summary["video_id"].astype(str)
    summary_by_id = summary.drop_duplicates("video_id").set_index("video_id")

    story_rows = []
    for video_id, grp in bouts.groupby("video_id"):
        grp = grp.sort_values("start_frame")
        durations = pd.to_numeric(grp["duration_sec"], errors="coerce").fillna(0).tolist()
        n_bouts = len(grp)
        n_transitions = max(0, n_bouts - 1)
        duration_sec = float(pd.to_numeric(grp["end_sec"], errors="coerce").max() or 0.0)
        dominant_state = float("nan")
        entropy = float("nan")
        if video_id in summary_by_id.index and state_cols:
            vec = pd.to_numeric(summary_by_id.loc[video_id, state_cols], errors="coerce").fillna(0).to_numpy(dtype=float)
            if vec.sum() > 0:
                dominant_state = state_ids[int(np.argmax(vec))]
                entropy = _entropy(vec)
        story_rows.append({
            "video_id": video_id,
            "subject_id": grp["subject_id"].iloc[0] if "subject_id" in grp.columns else "",
            "timepoint": grp["timepoint"].iloc[0] if "timepoint" in grp.columns else "",
            "condition": grp["condition"].iloc[0] if "condition" in grp.columns else "",
            "duration_sec": duration_sec,
            "dominant_state": dominant_state,
            "state_entropy": entropy,
            "n_bouts": n_bouts,
            "n_transitions": n_transitions,
            "transition_rate": (n_transitions / duration_sec) if duration_sec > 0 else float("nan"),
            "mean_bout_duration": float(np.mean(durations)) if durations else float("nan"),
            "short_bout_fraction": float(np.mean(np.array(durations) < 0.5)) if durations else float("nan"),
            "state_sequence_rle": "",
            "top_motifs": "",
        })
    stories = pd.DataFrame(story_rows)
    return stories, bouts


def load_story_data(results_dir) -> tuple[pd.DataFrame | None, pd.DataFrame | None, str | None]:
    """Load native sequence artifacts, falling back to legacy characterization
    outputs. Never raises. Returns (stories_df, bouts_df, source) where
    source is "native", "fallback", or None when nothing usable is found."""
    results_dir = Path(results_dir)
    stories_path = results_dir / "sequences" / "video_stories.csv"
    bouts_path = results_dir / "sequences" / "video_story_bouts.csv"
    if stories_path.exists() and bouts_path.exists():
        try:
            stories = pd.read_csv(stories_path)
            bouts = pd.read_csv(bouts_path)
            if not stories.empty:
                return stories, bouts, "native"
        except Exception:
            pass
    try:
        legacy_bouts_path = results_dir / "characterization" / "bouts.csv"
        summary_path = results_dir / "comparison" / "summary_table.csv"
        if legacy_bouts_path.exists() and summary_path.exists():
            legacy_bouts = pd.read_csv(legacy_bouts_path)
            summary = pd.read_csv(summary_path)
            stories, bouts = build_fallback_stories(legacy_bouts, summary)
            if not stories.empty:
                return stories, bouts, "fallback"
    except Exception:
        pass
    return None, None, None


def load_subject_journeys(results_dir) -> pd.DataFrame | None:
    p = Path(results_dir) / "sequences" / "subject_journeys.csv"
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        return df if not df.empty else None
    except Exception:
        return None


def find_bout_at_time(bouts_df: pd.DataFrame, t: float) -> dict | None:
    """Return the bout row (as a dict) whose [start_sec, end_sec] contains t,
    or None. A range-containment check, not a pixel hit-test, so arbitrarily
    short bouts stay clickable."""
    if bouts_df is None or bouts_df.empty:
        return None
    for _, row in bouts_df.sort_values("start_sec").iterrows():
        if float(row["start_sec"]) <= t <= float(row["end_sec"]):
            return row.to_dict()
    return None


def compute_clip_window(
    start_sec: float,
    end_sec: float,
    *,
    target_clip_sec: float = 5.0,
    min_clip_sec: float = 3.0,
    pad_before_sec: float = 1.0,
    pad_after_sec: float = 1.0,
    video_duration_sec: float | None = None,
) -> tuple[float, float]:
    """Fixed-window clip bounds centered on a bout, not raw bout duration.
    Clamping at either edge shifts the window (preserving as much of the
    target/min length as possible) rather than just truncating it."""
    center = (start_sec + end_sec) / 2.0
    half = target_clip_sec / 2.0
    win_start = min(center - half, start_sec - pad_before_sec)
    win_end = max(center + half, end_sec + pad_after_sec)
    if win_end - win_start < min_clip_sec:
        deficit = min_clip_sec - (win_end - win_start)
        win_start -= deficit / 2.0
        win_end += deficit / 2.0
    if win_start < 0:
        win_end -= win_start  # shift right by the overshoot
        win_start = 0.0
    if video_duration_sec is not None and win_end > float(video_duration_sec):
        win_start -= win_end - float(video_duration_sec)
        win_end = float(video_duration_sec)
        win_start = max(0.0, win_start)
    return win_start, win_end


def resolve_source_video(video_id: str, feature_index: dict | None = None) -> str | None:
    """Resolve a video_id/stem to a local source .mp4 path, or None. Mirrors
    the resolution order already used by generate_clips._resolve_video_path
    and views/browse_states.py:_resolve_stem_video."""
    raw_dir = _vc.get_raw_videos_dir()
    if raw_dir:
        candidate = os.path.join(raw_dir, f"{video_id}.mp4")
        if os.path.exists(candidate):
            return candidate
    if feature_index and video_id in feature_index:
        vp = feature_index[video_id].get("video_path") if isinstance(feature_index[video_id], dict) else None
        resolved = _resolve_video_path(vp)
        if resolved:
            return resolved
    return None


def parse_top_motifs(text: str) -> list[tuple[tuple[int, ...], int]]:
    """Parse video_stories.csv's top_motifs string: ';'-joined 'motif:count'
    pairs, motif written as a Python tuple repr (same format as motifs.csv)."""
    if not text or not isinstance(text, str):
        return []
    out = []
    for part in text.split(";"):
        part = part.strip()
        if not part or ":" not in part:
            continue
        motif_str, _, count_str = part.rpartition(":")
        try:
            motif = tuple(ast.literal_eval(motif_str))
            count = int(count_str)
        except Exception:
            continue
        out.append((motif, count))
    return out


def bout_motif_membership(states: list[int], idx: int, top_motifs: list[tuple[tuple[int, ...], int]]) -> list[tuple[int, ...]]:
    """Which top-motif n-grams (if any) the bout at position idx participates in."""
    motif_set = {m for m, _ in top_motifs}
    n = len(states)
    hits = []
    for size in (2, 3):
        for start in range(max(0, idx - size + 1), min(idx, n - size) + 1):
            window = tuple(states[start:start + size])
            if window in motif_set:
                hits.append(window)
    return hits


def format_duration(seconds) -> str:
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return "—"
    m, s = divmod(int(round(max(0.0, float(seconds)))), 60)
    return f"{m:d}:{s:02d}"


def format_number(value, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{float(value):.{digits}f}"


def format_label_value(value) -> str:
    """Display a subject/timepoint/condition value without a spurious
    trailing '.0' — these columns are frequently read from CSV as float64
    (e.g. subject_id 526 -> 526.0) even though they're really identifiers."""
    text = str(value)
    try:
        f = float(text)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return text


def _normalize_identifier_columns(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """Strip spurious trailing '.0' from subject/timepoint/condition columns
    (frequently read from CSV as float64 even though they're identifiers),
    applied once at load time so every downstream comparison, dropdown, and
    display sees the same cleaned string consistently."""
    if df is None:
        return None
    for col in ("subject_id", "timepoint", "condition"):
        if col in df.columns:
            df[col] = df[col].map(format_label_value)
    return df


def state_label_text(sid, state_labels: dict) -> str:
    try:
        sid_int = int(sid)
    except (TypeError, ValueError):
        return "State — unlabeled"
    info = state_labels.get(sid_int) or {}
    label = str(info.get("label") or "").strip()
    if label:
        return f"State {sid_int} — tentative label: {label}"
    return f"State {sid_int} — unlabeled"


def compute_state_occupancy(bouts_for_video: pd.DataFrame) -> dict[int, float]:
    """Return {state_id: occupancy_fraction} over the story's total duration."""
    if bouts_for_video is None or bouts_for_video.empty:
        return {}
    total = float(pd.to_numeric(bouts_for_video["duration_sec"], errors="coerce").fillna(0).sum())
    if total <= 0:
        return {}
    out = {}
    for sid, grp in bouts_for_video.groupby("state"):
        try:
            sid_int = int(sid)
        except (TypeError, ValueError):
            continue
        out[sid_int] = float(pd.to_numeric(grp["duration_sec"], errors="coerce").fillna(0).sum()) / total
    return out


# ---------------------------------------------------------------------------
# Small one-off clip generation worker
# ---------------------------------------------------------------------------

class _ClipWorker(QThread):
    done = pyqtSignal(bool, str)

    def __init__(self, video_path: str, start_frame: int, end_frame: int, out_path: str, fps: float, window_sec: float):
        super().__init__()
        self.video_path = video_path
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.out_path = out_path
        self.fps = fps
        self.window_sec = window_sec

    def run(self):
        try:
            if os.path.isfile(self.out_path):
                self.done.emit(True, self.out_path)
                return
            ok = _export_clip(
                self.video_path, self.start_frame, self.end_frame, self.out_path,
                fps=self.fps, pad_to_secs=self.window_sec, max_secs=self.window_sec,
            )
            self.done.emit(bool(ok), self.out_path if ok else "Clip export failed.")
        except Exception:
            self.done.emit(False, traceback.format_exc())


# ---------------------------------------------------------------------------
# VideoStoriesView
# ---------------------------------------------------------------------------

class VideoStoriesView(QWidget):
    """Browse per-video behavioral state stories: timeline, summary, clips."""

    worker_running = pyqtSignal(bool)

    _CLIP_TARGET_SEC = 5.0
    _CLIP_MIN_SEC = 3.0
    _CLIP_PAD_BEFORE_SEC = 1.0
    _CLIP_PAD_AFTER_SEC = 1.0
    _LABEL_WIDTH_FRACTION = 0.015

    def __init__(self, cfg: dict | None = None):
        super().__init__()
        self.cfg = cfg or {}
        self._worker = None
        self._running_command = ""
        self._clip_worker = None
        self._stories: pd.DataFrame | None = None
        self._bouts: pd.DataFrame | None = None
        self._journeys: pd.DataFrame | None = None
        self._state_labels: dict = {}
        self._feature_index: dict = {}
        self._source: str | None = None
        self._current_bouts: pd.DataFrame | None = None
        self._current_story: dict | None = None
        self._subject_active = False
        self._timepoint_active = False
        self._condition_active = False
        self._build()
        self.refresh()

    # ─────────────────────────────────────────────────────────── build ──

    def _build(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 16, 20, 16)
        outer.setSpacing(8)

        top = QHBoxLayout()
        title = QLabel("Video Stories")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        top.addWidget(title)
        top.addStretch()
        self._run_btn = QPushButton("Run compare.py --report")
        self._run_btn.setFixedHeight(30)
        self._run_btn.clicked.connect(
            lambda: self._run_command(["compare.py", "--report"], self._terminal)
        )
        top.addWidget(self._run_btn)
        outer.addLayout(top)

        self._terminal = TerminalBox()
        self._terminal.setVisible(False)
        outer.addWidget(self._terminal)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        content, cl = _scroll_content_widget()
        self._content_layout = cl

        self._placeholder_lbl = _placeholder(
            "Video stories have not been generated yet.\nRun: python compare.py --report"
        )
        cl.addWidget(self._placeholder_lbl)

        self._selector_widget = QWidget()
        sel_lay = QHBoxLayout(self._selector_widget)
        sel_lay.setContentsMargins(0, 0, 0, 0)
        self._subject_combo = QComboBox()
        self._timepoint_combo = QComboBox()
        self._condition_combo = QComboBox()
        self._video_combo = QComboBox()
        self._subject_row = self._add_selector(sel_lay, "Subject", self._subject_combo)
        self._timepoint_row = self._add_selector(sel_lay, "Timepoint", self._timepoint_combo)
        self._condition_row = self._add_selector(sel_lay, "Condition", self._condition_combo)
        self._video_row = self._add_selector(sel_lay, "Video", self._video_combo)
        sel_lay.addStretch()
        cl.addWidget(self._selector_widget)
        self._selector_widget.setVisible(False)

        self._subject_combo.currentIndexChanged.connect(self._on_subject_changed)
        self._timepoint_combo.currentIndexChanged.connect(self._on_timepoint_changed)
        self._condition_combo.currentIndexChanged.connect(self._on_condition_changed)
        self._video_combo.currentIndexChanged.connect(self._on_video_changed)

        cl.addWidget(_section_title("Story Summary"))
        self._summary_frame = QFrame()
        self._summary_form = QFormLayout(self._summary_frame)
        cl.addWidget(self._summary_frame)

        cl.addWidget(_section_title("Timeline"))
        if _MPL:
            self._timeline_canvas = MplCanvas(figsize=(9, 1.6))
            self._timeline_canvas.setMinimumHeight(110)
            self._timeline_canvas.mpl_connect("button_press_event", self._on_timeline_click)
            cl.addWidget(self._timeline_canvas)
        else:
            self._timeline_canvas = None
            cl.addWidget(_placeholder("Install matplotlib to view the timeline."))

        cl.addWidget(_section_title("State Legend"))
        self._legend_widget = QWidget()
        self._legend_layout = QGridLayout(self._legend_widget)
        self._legend_layout.setHorizontalSpacing(16)
        cl.addWidget(self._legend_widget)

        cl.addStretch()
        scroll.setWidget(content)
        outer.addWidget(scroll, stretch=1)

    @staticmethod
    def _add_selector(layout: QHBoxLayout, label: str, combo: QComboBox) -> QWidget:
        row = QWidget()
        row_lay = QVBoxLayout(row)
        row_lay.setContentsMargins(0, 0, 12, 0)
        row_lay.setSpacing(2)
        lbl = QLabel(label)
        lbl.setStyleSheet("color:#666; font-size:11px;")
        row_lay.addWidget(lbl)
        combo.setMinimumWidth(140)
        row_lay.addWidget(combo)
        layout.addWidget(row)
        return row

    # ─────────────────────────────────────────────────────────── data ──

    def refresh(self) -> None:
        stories, bouts, source = load_story_data(RESULTS)
        stories, bouts = _normalize_identifier_columns(stories), _normalize_identifier_columns(bouts)
        self._stories, self._bouts, self._source = stories, bouts, source
        self._journeys = _normalize_identifier_columns(load_subject_journeys(RESULTS))
        self._state_labels = load_state_labels(RESULTS)
        self._feature_index = self._load_feature_index()

        has_data = stories is not None and not stories.empty
        self._placeholder_lbl.setVisible(not has_data)
        self._selector_widget.setVisible(has_data)
        self._summary_frame.setVisible(has_data)
        if self._timeline_canvas is not None:
            self._timeline_canvas.setVisible(has_data)
        self._legend_widget.setVisible(has_data)
        if not has_data:
            return
        self._populate_subject_combo()

    @staticmethod
    def _load_feature_index() -> dict:
        p = RESULTS / "features" / "index.json"
        if not p.exists():
            return {}
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _distinct(self, col: str, df: pd.DataFrame | None = None) -> list:
        df = self._stories if df is None else df
        if df is None or col not in df.columns:
            return []
        return sorted({str(v) for v in df[col].tolist() if not _is_blank(v)})

    def _populate_subject_combo(self) -> None:
        subjects = self._distinct("subject_id")
        self._subject_active = len(subjects) > 1
        self._subject_row.setVisible(self._subject_active)
        self._subject_combo.blockSignals(True)
        self._subject_combo.clear()
        if self._subject_active:
            self._subject_combo.addItems(subjects)
        self._subject_combo.blockSignals(False)
        self._populate_timepoint_combo()

    def _filtered_stories(self, upto: str) -> pd.DataFrame:
        df = self._stories
        if df is None:
            return pd.DataFrame()
        if upto in ("timepoint", "condition", "video") and self._subject_active:
            subj = self._subject_combo.currentText()
            if subj:
                df = df[df["subject_id"].astype(str) == subj]
        if upto in ("condition", "video") and self._timepoint_active:
            tp = self._timepoint_combo.currentText()
            if tp:
                df = df[df["timepoint"].astype(str) == tp]
        if upto == "video" and self._condition_active:
            cond = self._condition_combo.currentText()
            if cond:
                df = df[df["condition"].astype(str) == cond]
        return df

    def _populate_timepoint_combo(self) -> None:
        df = self._filtered_stories("timepoint")
        timepoints = order_timepoints(self._distinct("timepoint", df), RESULTS)
        self._timepoint_active = len(timepoints) > 1
        self._timepoint_row.setVisible(self._timepoint_active)
        self._timepoint_combo.blockSignals(True)
        self._timepoint_combo.clear()
        if self._timepoint_active:
            self._timepoint_combo.addItems([str(v) for v in timepoints])
        self._timepoint_combo.blockSignals(False)
        self._populate_condition_combo()

    def _populate_condition_combo(self) -> None:
        df = self._filtered_stories("condition")
        conditions = self._distinct("condition", df)
        self._condition_active = len(conditions) > 1
        self._condition_row.setVisible(self._condition_active)
        self._condition_combo.blockSignals(True)
        self._condition_combo.clear()
        if self._condition_active:
            self._condition_combo.addItems(conditions)
        self._condition_combo.blockSignals(False)
        self._populate_video_combo()

    def _populate_video_combo(self) -> None:
        df = self._filtered_stories("video")
        videos = sorted({str(v) for v in df["video_id"].tolist()}) if df is not None and not df.empty else []
        self._video_combo.blockSignals(True)
        self._video_combo.clear()
        self._video_combo.addItems(videos)
        self._video_combo.blockSignals(False)
        if videos:
            self._load_story(videos[0])
        else:
            self._current_story = None
            self._current_bouts = None

    def _on_subject_changed(self, _idx: int) -> None:
        self._populate_timepoint_combo()

    def _on_timepoint_changed(self, _idx: int) -> None:
        self._populate_condition_combo()

    def _on_condition_changed(self, _idx: int) -> None:
        self._populate_video_combo()

    def _on_video_changed(self, _idx: int) -> None:
        video_id = self._video_combo.currentText()
        if video_id:
            self._load_story(video_id)

    # ─────────────────────────────────────────────────────── story load ──

    def _load_story(self, video_id: str) -> None:
        stories = self._stories
        bouts = self._bouts
        if stories is None or bouts is None:
            return
        rows = stories[stories["video_id"].astype(str) == str(video_id)]
        if rows.empty:
            return
        self._current_story = rows.iloc[0].to_dict()
        story_bouts = bouts[bouts["video_id"].astype(str) == str(video_id)].copy()
        story_bouts = story_bouts.sort_values("start_frame" if "start_frame" in story_bouts.columns else "start_sec")
        self._current_bouts = story_bouts
        self._render_summary()
        self._render_timeline()
        self._render_legend()

    # ─────────────────────────────────────────────────────────── render ──

    def _render_summary(self) -> None:
        while self._summary_form.rowCount():
            self._summary_form.removeRow(0)
        story = self._current_story or {}
        if not story:
            return

        def add(label: str, value: str) -> None:
            self._summary_form.addRow(QLabel(label), QLabel(value))

        add("Video / session:", str(story.get("video_id", "—")))
        subject = story.get("subject_id")
        if not _is_blank(subject):
            add("Subject:", format_label_value(subject))
        timepoint = story.get("timepoint")
        if not _is_blank(timepoint):
            add("Timepoint:", format_label_value(timepoint))
        condition = story.get("condition")
        if not _is_blank(condition):
            add("Condition:", format_label_value(condition))
        add("Duration:", format_duration(story.get("duration_sec")))
        add("Dominant state:", state_label_text(story.get("dominant_state"), self._state_labels)
            if not _is_blank(story.get("dominant_state")) else "—")
        add("Number of bouts:", str(int(story.get("n_bouts", 0)) if not _is_blank(story.get("n_bouts")) else "—"))
        add("Number of transitions:", str(int(story.get("n_transitions", 0)) if not _is_blank(story.get("n_transitions")) else "—"))
        add("Transition rate:", format_number(story.get("transition_rate")) + " /s")
        add("Mean bout duration:", format_number(story.get("mean_bout_duration"), digits=2) + " s")
        add("Short-bout fraction:", format_number(story.get("short_bout_fraction")))
        add("State entropy:", format_number(story.get("state_entropy")))

        top_motifs = parse_top_motifs(str(story.get("top_motifs", "")))
        if top_motifs:
            text = ", ".join(f"{m} ×{c}" for m, c in top_motifs[:5])
        else:
            text = "not available"
        add("Top motifs:", text)

        add("Distance from baseline:", self._distance_from_baseline_text(story))

    def _distance_from_baseline_text(self, story: dict) -> str:
        if self._journeys is None:
            return "not available"
        subject = story.get("subject_id")
        timepoint = story.get("timepoint")
        if _is_blank(subject) or _is_blank(timepoint):
            return "not available"
        rows = self._journeys[
            (self._journeys["subject_id"].astype(str) == str(subject))
            & (self._journeys["timepoint"].astype(str) == str(timepoint))
        ]
        if rows.empty:
            return "not available"
        val = rows.iloc[0].get("distance_from_baseline")
        return format_number(val)

    def _render_timeline(self) -> None:
        if self._timeline_canvas is None:
            return
        ax = self._timeline_canvas.ax
        ax.clear()
        bouts = self._current_bouts
        story = self._current_story or {}
        duration = float(story.get("duration_sec") or 0.0)
        if bouts is None or bouts.empty or duration <= 0:
            ax.text(0.5, 0.5, "No bouts to display", ha="center", va="center", transform=ax.transAxes)
            self._timeline_canvas.draw()
            return

        n_states = int(bouts["state"].max()) + 1 if not bouts.empty else 1
        colors = _state_colors(max(n_states, 10))
        spans = []
        facecolors = []
        for _, row in bouts.iterrows():
            start = float(row["start_sec"])
            span = float(row["end_sec"]) - start
            spans.append((start, span))
            sid = int(row["state"])
            facecolors.append(colors[sid] if sid < len(colors) else "#607D8B")
        ax.broken_barh(spans, (0, 1), facecolors=facecolors, edgecolor="white", linewidth=0.5)

        label_threshold = duration * self._LABEL_WIDTH_FRACTION
        for _, row in bouts.iterrows():
            dur = float(row["duration_sec"])
            if dur >= label_threshold:
                mid = float(row["start_sec"]) + dur / 2.0
                ax.text(mid, 0.5, str(int(row["state"])), ha="center", va="center", fontsize=7, color="white")

        ax.set_xlim(0, duration)
        ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_xlabel("Time (s)")
        self._timeline_canvas.draw()

    def _render_legend(self) -> None:
        while self._legend_layout.count():
            item = self._legend_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        bouts = self._current_bouts
        if bouts is None or bouts.empty:
            return
        occupancy = compute_state_occupancy(bouts)
        n_states = max(occupancy.keys(), default=0) + 1
        colors = _state_colors(max(n_states, 10))
        for row_idx, sid in enumerate(sorted(occupancy)):
            swatch = QLabel()
            swatch.setFixedSize(14, 14)
            c = colors[sid] if sid < len(colors) else (0.4, 0.4, 0.4, 1.0)
            rgb = tuple(int(255 * x) for x in c[:3])
            swatch.setStyleSheet(f"background: rgb{rgb}; border-radius:3px;")
            text = QLabel(f"{state_label_text(sid, self._state_labels)} — {occupancy[sid] * 100:.1f}%")
            self._legend_layout.addWidget(swatch, row_idx, 0)
            self._legend_layout.addWidget(text, row_idx, 1)

    # ────────────────────────────────────────────────────── click-to-play ──

    def _on_timeline_click(self, event) -> None:
        if event.inaxes is None or event.xdata is None:
            return
        bouts = self._current_bouts
        if bouts is None or bouts.empty:
            return
        bout = find_bout_at_time(bouts, float(event.xdata))
        if bout is None:
            return
        self._open_segment_dialog(bout)

    def _open_segment_dialog(self, bout: dict) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle(f"State {int(bout['state'])} — {format_duration(bout.get('start_sec'))}–{format_duration(bout.get('end_sec'))}")
        lay = QVBoxLayout(dlg)

        story = self._current_story or {}
        ordered_bouts = self._current_bouts.reset_index(drop=True) if self._current_bouts is not None else pd.DataFrame()
        states = [int(s) for s in ordered_bouts["state"].tolist()] if not ordered_bouts.empty else []
        motif_hits = []
        if not ordered_bouts.empty and "start_frame" in ordered_bouts.columns:
            match = ordered_bouts.index[ordered_bouts["start_frame"] == bout.get("start_frame")]
            if len(match):
                top_motifs = parse_top_motifs(str(story.get("top_motifs", "")))
                motif_hits = bout_motif_membership(states, int(match[0]), top_motifs)

        header = QLabel(
            f"State: {int(bout['state'])}\n"
            f"Start: {format_duration(bout.get('start_sec'))}   "
            f"End: {format_duration(bout.get('end_sec'))}   "
            f"Duration: {format_duration(bout.get('duration_sec'))}\n"
            f"Confidence: {format_number(bout.get('confidence_mean')) if not _is_blank(bout.get('confidence_mean')) else 'not available'}\n"
            f"Motif membership: {', '.join(str(m) for m in motif_hits) if motif_hits else 'none'}"
        )
        lay.addWidget(header)

        video_id = str(story.get("video_id", ""))
        video_path = resolve_source_video(video_id, self._feature_index)
        if not video_path:
            lay.addWidget(QLabel("Source video unavailable locally for this session."))
            dlg.resize(420, 220)
            dlg.exec_()
            return

        status_lbl = QLabel("Generating clip…")
        lay.addWidget(status_lbl)
        player = VideoPlayer(parent=dlg)
        player.setVisible(False)
        lay.addWidget(player, stretch=1)
        dlg.resize(700, 560)

        fps = float(self.cfg.get("fps", 30.0) or 30.0)
        win_start, win_end = compute_clip_window(
            float(bout["start_sec"]), float(bout["end_sec"]),
            target_clip_sec=self._CLIP_TARGET_SEC, min_clip_sec=self._CLIP_MIN_SEC,
            pad_before_sec=self._CLIP_PAD_BEFORE_SEC, pad_after_sec=self._CLIP_PAD_AFTER_SEC,
            video_duration_sec=story.get("duration_sec"),
        )
        start_frame = int(round(win_start * fps))
        end_frame = int(round(win_end * fps))
        window_sec = max(win_end - win_start, self._CLIP_MIN_SEC)
        out_dir = Path(_vc.get_clips_dir()) / "stories" / video_id
        out_path = str(out_dir / f"f{start_frame}-{end_frame}.mp4")

        def on_done(ok: bool, path_or_error: str) -> None:
            status_lbl.setVisible(False)
            if ok:
                player.setVisible(True)
                player.load(path_or_error)
                player.play()
            else:
                status_lbl.setVisible(True)
                status_lbl.setText(f"Could not generate clip: {path_or_error}")

        self._clip_worker = _ClipWorker(video_path, start_frame, end_frame, out_path, fps, window_sec)
        self._clip_worker.done.connect(on_done)
        self._clip_worker.start()

        def cleanup() -> None:
            player.pause()
            if player._cap:
                player._cap.release()

        dlg.finished.connect(cleanup)
        dlg.exec_()

    # ─────────────────────────────────────────────────── command runner ──

    def _run_command(self, args: list[str], terminal: TerminalBox) -> None:
        if self._worker and self._worker.isRunning():
            return
        terminal.setVisible(True)
        command = "python " + " ".join(str(a) for a in args)
        terminal.set_command(command)
        self._running_command = command
        self._worker = SubprocessWorker(args)
        self._worker.log.connect(terminal.append_output)
        self._worker.done.connect(self._on_run_done)
        self.worker_running.emit(True)
        self._worker.start()

    def _on_run_done(self, ok: bool) -> None:
        self._running_command = ""
        self.worker_running.emit(False)
        if ok:
            self.refresh()
