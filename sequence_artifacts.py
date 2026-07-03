"""Build State -> Motif -> Story -> Journey sequence artifacts."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STORY_BOUT_COLUMNS = [
    "video_id",
    "subject_id",
    "timepoint",
    "condition",
    "state",
    "start_frame",
    "end_frame",
    "start_sec",
    "end_sec",
    "duration_sec",
    "confidence_mean",
]

VIDEO_STORY_COLUMNS = [
    "video_id",
    "subject_id",
    "timepoint",
    "condition",
    "duration_sec",
    "dominant_state",
    "state_entropy",
    "n_bouts",
    "n_transitions",
    "transition_rate",
    "mean_bout_duration",
    "short_bout_fraction",
    "state_sequence_rle",
    "top_motifs",
]

SUBJECT_JOURNEY_COLUMNS = [
    "subject_id",
    "timepoint",
    "distance_from_baseline",
    "dominant_state",
    "state_entropy",
    "transition_rate",
    "state_occupancy_vector",
    "story_similarity_to_baseline",
]


def _state_cols(summary: pd.DataFrame) -> list[str]:
    cols = [c for c in summary.columns if str(c).startswith("state_") and str(c).endswith("_frac")]

    def _key(col: str) -> int:
        try:
            return int(str(col).split("_")[1])
        except Exception:
            return 10**9

    return sorted(cols, key=_key)


def _pick_axis(summary: pd.DataFrame, design: dict, key: str, fallback: str) -> str | None:
    col = design.get(key)
    if col and col in summary.columns:
        return str(col)
    if fallback in summary.columns:
        return fallback
    return None


def _condition_col(summary: pd.DataFrame, design: dict) -> str | None:
    for col in design.get("condition_cols") or []:
        if col in summary.columns:
            return str(col)
    if "context" in summary.columns:
        return "context"
    return None


def _value(row: pd.Series, col: str | None) -> Any:
    if not col:
        return ""
    val = row.get(col, "")
    if pd.isna(val):
        return ""
    return val


def _entropy(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = np.maximum(arr, 0)
    total = float(arr.sum())
    if total <= 0:
        return float("nan")
    p = arr / total
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def _dominant_state_from_vector(values: np.ndarray) -> int | float:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0 or np.all(np.isnan(arr)):
        return float("nan")
    arr = np.nan_to_num(arr, nan=-1.0)
    if np.max(arr) < 0:
        return float("nan")
    return int(np.argmax(arr))


def _motif_counts(states: list[int], n: int) -> Counter[tuple[int, ...]]:
    counts: Counter[tuple[int, ...]] = Counter()
    if len(states) < n:
        return counts
    for i in range(len(states) - n + 1):
        motif = tuple(int(s) for s in states[i:i + n])
        if len(set(motif)) > 1:
            counts[motif] += 1
    return counts


def _top_motifs(states: list[int], limit: int = 5) -> str:
    counts: Counter[tuple[int, ...]] = Counter()
    counts.update(_motif_counts(states, 2))
    counts.update(_motif_counts(states, 3))
    repeated = [(motif, count) for motif, count in counts.items() if count > 1]
    repeated.sort(key=lambda item: (-item[1], len(item[0]), item[0]))
    return ";".join(f"{motif}:{count}" for motif, count in repeated[:limit])


def _rle_string(bouts: list[dict]) -> str:
    return "|".join(
        f"{int(row['state'])}:{int(row['end_frame']) - int(row['start_frame']) + 1}"
        for row in bouts
    )


def _ordered_timepoints(values: list[Any], design: dict) -> list[Any]:
    available = {str(v): v for v in values}
    ordered = []
    for item in design.get("time_order") or []:
        key = str(item)
        if key in available:
            ordered.append(available[key])
    seen = {str(v) for v in ordered}
    rest = [v for v in values if str(v) not in seen]
    try:
        rest = sorted(rest)
    except TypeError:
        rest = sorted(rest, key=lambda v: str(v))
    return ordered + rest


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def _safe_float(value: float) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return float("nan")
    return round(float(value), 6)


def build_sequence_artifacts(
    summary: pd.DataFrame,
    design: dict,
    results_dir: str | os.PathLike,
    *,
    fps: float = 30.0,
    n_clusters: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Write video story bouts, video stories, and subject journeys."""
    results = Path(results_dir)
    shared_dir = results / "shared"
    out_dir = results / "sequences"
    out_dir.mkdir(parents=True, exist_ok=True)

    fps = max(float(fps or 30.0), 1e-9)
    state_cols = _state_cols(summary)
    if n_clusters is None:
        n_clusters = len(state_cols)

    subject_col = _pick_axis(summary, design, "subject_col", "animal_id")
    time_col = _pick_axis(summary, design, "time_col", "day")
    condition_col = _condition_col(summary, design)

    bout_rows: list[dict] = []
    story_rows: list[dict] = []
    story_vectors: dict[str, np.ndarray] = {}

    if "stem" not in summary.columns:
        video_bouts = pd.DataFrame(columns=STORY_BOUT_COLUMNS)
        video_stories = pd.DataFrame(columns=VIDEO_STORY_COLUMNS)
        journeys = pd.DataFrame(columns=SUBJECT_JOURNEY_COLUMNS)
        video_bouts.to_csv(out_dir / "video_story_bouts.csv", index=False)
        video_stories.to_csv(out_dir / "video_stories.csv", index=False)
        journeys.to_csv(out_dir / "subject_journeys.csv", index=False)
        return {"video_story_bouts": video_bouts, "video_stories": video_stories, "subject_journeys": journeys}

    meta_by_stem = summary.drop_duplicates("stem").set_index("stem")

    for stem in summary["stem"].dropna().astype(str).tolist():
        labels_path = shared_dir / f"{stem}_labels.npy"
        if stem not in meta_by_stem.index or not labels_path.exists():
            continue
        labels = np.load(labels_path).astype(np.int32)
        probs_path = shared_dir / f"{stem}_probs.npy"
        if probs_path.exists():
            probs = np.load(probs_path).astype(float)
            if len(probs) != len(labels):
                probs = np.ones(len(labels), dtype=float)
        else:
            probs = np.ones(len(labels), dtype=float)

        meta = meta_by_stem.loc[stem]
        subject_id = _value(meta, subject_col)
        timepoint = _value(meta, time_col)
        condition = _value(meta, condition_col)

        video_bouts: list[dict] = []
        i = 0
        while i < len(labels):
            if labels[i] < 0:
                i += 1
                continue
            state = int(labels[i])
            j = i + 1
            while j < len(labels) and labels[j] == state:
                j += 1
            row = {
                "video_id": stem,
                "subject_id": subject_id,
                "timepoint": timepoint,
                "condition": condition,
                "state": state,
                "start_frame": int(i),
                "end_frame": int(j - 1),
                "start_sec": float(i / fps),
                "end_sec": float((j - 1) / fps),
                "duration_sec": float((j - i) / fps),
                "confidence_mean": float(np.nanmean(probs[i:j])) if j > i else float("nan"),
            }
            video_bouts.append(row)
            bout_rows.append(row)
            i = j

        valid = labels[labels >= 0]
        counts = np.bincount(valid, minlength=max(int(n_clusters or 0), int(valid.max()) + 1 if len(valid) else 0))
        vector = counts.astype(float) / max(1, int(counts.sum())) if len(counts) else np.array([], dtype=float)
        story_vectors[stem] = vector
        duration_sec = float(len(labels) / fps)
        states = [int(row["state"]) for row in video_bouts]
        durations = [float(row["duration_sec"]) for row in video_bouts]
        n_bouts = len(video_bouts)
        n_transitions = max(0, n_bouts - 1)
        story_rows.append({
            "video_id": stem,
            "subject_id": subject_id,
            "timepoint": timepoint,
            "condition": condition,
            "duration_sec": duration_sec,
            "dominant_state": _dominant_state_from_vector(vector),
            "state_entropy": _safe_float(_entropy(vector)),
            "n_bouts": n_bouts,
            "n_transitions": n_transitions,
            "transition_rate": _safe_float(n_transitions / duration_sec if duration_sec > 0 else float("nan")),
            "mean_bout_duration": _safe_float(float(np.mean(durations)) if durations else float("nan")),
            "short_bout_fraction": _safe_float(float(np.mean(np.asarray(durations) < 0.5)) if durations else float("nan")),
            "state_sequence_rle": _rle_string(video_bouts),
            "top_motifs": _top_motifs(states),
        })

    video_story_bouts = pd.DataFrame(bout_rows, columns=STORY_BOUT_COLUMNS)
    video_stories = pd.DataFrame(story_rows, columns=VIDEO_STORY_COLUMNS)

    journey_rows: list[dict] = []
    if not video_stories.empty and state_cols:
        summary_vectors = summary.set_index("stem")
        journey_inputs = video_stories.copy()
        journey_inputs["_duration_weight"] = pd.to_numeric(journey_inputs["duration_sec"], errors="coerce").fillna(0)

        for subject, subject_stories in journey_inputs.groupby("subject_id", dropna=False):
            time_values = subject_stories["timepoint"].dropna().tolist()
            time_order = _ordered_timepoints(time_values, design)
            baseline_vector: np.ndarray | None = None
            for timepoint in time_order:
                grp = subject_stories[subject_stories["timepoint"].astype(str) == str(timepoint)]
                stems = grp["video_id"].astype(str).tolist()
                vectors = []
                weights = []
                for stem in stems:
                    if stem in summary_vectors.index:
                        vec = pd.to_numeric(summary_vectors.loc[stem, state_cols], errors="coerce").fillna(0).to_numpy(dtype=float)
                    else:
                        vec = story_vectors.get(stem, np.zeros(len(state_cols), dtype=float))
                    vectors.append(vec)
                    weights.append(float(grp.loc[grp["video_id"].astype(str) == stem, "_duration_weight"].iloc[0]))
                if not vectors:
                    continue
                w = np.asarray(weights, dtype=float)
                if float(w.sum()) <= 0:
                    w = np.ones(len(vectors), dtype=float)
                vector = np.average(np.vstack(vectors), axis=0, weights=w)
                total = float(np.nansum(np.maximum(vector, 0)))
                if total > 0:
                    vector = np.maximum(vector, 0) / total
                if baseline_vector is None:
                    baseline_vector = vector.copy()
                distance = float(np.sum(np.abs(vector - baseline_vector)) / 2.0)
                journey_rows.append({
                    "subject_id": subject,
                    "timepoint": timepoint,
                    "distance_from_baseline": _safe_float(distance),
                    "dominant_state": _dominant_state_from_vector(vector),
                    "state_entropy": _safe_float(_entropy(vector)),
                    "transition_rate": _safe_float(
                        np.average(
                            pd.to_numeric(grp["transition_rate"], errors="coerce").fillna(0).to_numpy(dtype=float),
                            weights=w,
                        )
                    ),
                    "state_occupancy_vector": json.dumps([round(float(x), 6) for x in vector]),
                    "story_similarity_to_baseline": _safe_float(_cosine_similarity(vector, baseline_vector)),
                })

    subject_journeys = pd.DataFrame(journey_rows, columns=SUBJECT_JOURNEY_COLUMNS)

    video_story_bouts.to_csv(out_dir / "video_story_bouts.csv", index=False)
    video_stories.to_csv(out_dir / "video_stories.csv", index=False)
    subject_journeys.to_csv(out_dir / "subject_journeys.csv", index=False)
    print(f"Sequence bouts saved: results/sequences/video_story_bouts.csv  ({len(video_story_bouts)} bouts)")
    print(f"Video stories saved: results/sequences/video_stories.csv  ({len(video_stories)} videos)")
    print(f"Subject journeys saved: results/sequences/subject_journeys.csv  ({len(subject_journeys)} rows)")
    return {
        "video_story_bouts": video_story_bouts,
        "video_stories": video_stories,
        "subject_journeys": subject_journeys,
    }
