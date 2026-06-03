"""
characterize.py — Clip Reviewer data layer for VIEB
====================================================
Provides the backend API used by the Clip Reviewer GUI in views/validation.py.

Public API:
    load_clips(clips_dir)          → {state_id: [clip_path, ...]}
    load_annotations(path)         → {clip_path: label_string}
    save_annotations(ann, path)    → writes/updates annotations.csv
    get_clip_distribution(ann, all_clips, predictions=None) → distribution dict
    shuffle_clips(all_clips, seed) → shuffled flat list
    train_classifier(...)          → trains RandomForest, returns report dict
    predict_clips(...)             → applies classifier, returns predictions DataFrame

Annotations are persisted in results/annotations/annotations.csv:
    clip_path, state_id, assigned_label, timestamp

Predictions (model outputs, never mixed with human labels):
    results/annotations/predictions.csv:
    clip_path, state_id, predicted_label, confidence
"""

from __future__ import annotations

import json
import os
import random
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import vieb_config as _vc


# ---------------------------------------------------------------------------
# Clip loading
# ---------------------------------------------------------------------------

def load_clips(clips_dir: str) -> dict:
    """Return {state_id: [clip_path, ...]} from clips/state_N/ subdirectories."""
    result: dict[int, list[str]] = {}
    clips_dir_path = Path(clips_dir)
    if not clips_dir_path.exists():
        return result
    for state_dir in sorted(clips_dir_path.iterdir()):
        if not state_dir.is_dir():
            continue
        if not state_dir.name.startswith("state_"):
            continue
        try:
            state_id = int(state_dir.name.split("_")[1])
        except (ValueError, IndexError):
            continue
        clips = sorted(str(p) for p in state_dir.glob("*.mp4"))
        if clips:
            result[state_id] = clips
    return result


# ---------------------------------------------------------------------------
# Annotation persistence
# ---------------------------------------------------------------------------

def load_annotations(annotations_path: str) -> dict:
    """Return {clip_path: label_string} from annotations.csv."""
    p = Path(annotations_path)
    if not p.exists():
        return {}
    try:
        df = pd.read_csv(p)
        if "clip_path" not in df.columns or "assigned_label" not in df.columns:
            return {}
        return {
            str(row["clip_path"]): str(row["assigned_label"])
            for _, row in df.iterrows()
            if str(row.get("assigned_label", "")).strip()
        }
    except Exception:
        return {}


def save_annotations(annotations: dict, annotations_path: str) -> None:
    """
    Write/update annotations.csv — never overwrites; appends or updates by
    clip_path key.  State_id is inferred from the parent directory name
    (clips/state_3/longest_01.mp4 → state_id=3).
    """
    p = Path(annotations_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Load existing rows keyed by clip_path
    rows: dict[str, dict] = {}
    if p.exists():
        try:
            df = pd.read_csv(p)
            for _, row in df.iterrows():
                rows[str(row["clip_path"])] = {
                    "clip_path":      str(row["clip_path"]),
                    "state_id":       int(row.get("state_id", -1)),
                    "assigned_label": str(row.get("assigned_label", "")),
                    "timestamp":      str(row.get("timestamp", "")),
                }
        except Exception:
            pass

    now = datetime.now().isoformat(timespec="seconds")
    for clip_path, label in annotations.items():
        clip_path = str(clip_path)
        state_id = -1
        try:
            parent_name = Path(clip_path).parent.name
            if parent_name.startswith("state_"):
                state_id = int(parent_name.split("_")[1])
        except Exception:
            pass

        if clip_path in rows:
            rows[clip_path]["assigned_label"] = label
            rows[clip_path]["timestamp"] = now
        else:
            rows[clip_path] = {
                "clip_path":      clip_path,
                "state_id":       state_id,
                "assigned_label": label,
                "timestamp":      now,
            }

    df_out = pd.DataFrame(
        list(rows.values()),
        columns=["clip_path", "state_id", "assigned_label", "timestamp"],
    )
    df_out.to_csv(p, index=False)


# ---------------------------------------------------------------------------
# Distribution statistics
# ---------------------------------------------------------------------------

def get_clip_distribution(
    annotations: dict,
    all_clips,
    predictions: dict | None = None,
) -> dict:
    """
    Compute clip distribution across categories.

    Parameters
    ----------
    annotations : {clip_path: label_string}
    all_clips   : flat list of clip paths  OR  dict {state_id: [clip_path]}
    predictions : optional {clip_path: (label_string, confidence)}
                  Predictions for clips NOT already in annotations.

    Returns
    -------
    {
      "total":            int,
      "annotated":        int,
      "unannotated":      int,
      "by_label":         {label: count},
      "by_label_pct":     {label: fraction_of_total, "unannotated": fraction},
      "by_label_predicted": {label: count}   ← only when predictions is provided
    }
    """
    if isinstance(all_clips, dict):
        flat = [c for clips in all_clips.values() for c in clips]
    else:
        flat = list(all_clips)

    total = len(flat)
    annotated_set = set(annotations.keys())

    by_label: dict[str, int] = {}
    for clip in flat:
        label = annotations.get(clip)
        if label:
            by_label[label] = by_label.get(label, 0) + 1

    annotated = sum(by_label.values())
    unannotated = total - annotated

    by_label_pct: dict[str, float] = {
        label: count / total for label, count in by_label.items()
    } if total else {}
    by_label_pct["unannotated"] = unannotated / total if total else 0.0

    result: dict = {
        "total":        total,
        "annotated":    annotated,
        "unannotated":  unannotated,
        "by_label":     by_label,
        "by_label_pct": by_label_pct,
    }

    if predictions is not None:
        by_label_predicted: dict[str, int] = {}
        for clip in flat:
            if clip in annotated_set:
                continue
            pred = predictions.get(clip)
            if pred:
                pred_label = pred[0]
                by_label_predicted[pred_label] = by_label_predicted.get(pred_label, 0) + 1
        result["by_label_predicted"] = by_label_predicted

    return result


# ---------------------------------------------------------------------------
# Shuffle
# ---------------------------------------------------------------------------

def shuffle_clips(all_clips, seed=None) -> list:
    """
    Return a randomly shuffled flat list of all clip paths across all states.

    Parameters
    ----------
    all_clips : dict {state_id: [clip_path]} or flat list
    seed      : int or None (None = random each call)
    """
    if isinstance(all_clips, dict):
        flat = [c for clips in all_clips.values() for c in clips]
    else:
        flat = list(all_clips)

    rng = random.Random(seed if seed else None)
    rng.shuffle(flat)
    return flat


# ---------------------------------------------------------------------------
# Supervised learning backbone
# ---------------------------------------------------------------------------

def train_classifier(
    annotations_path: str,
    features_index: dict,
    shared_dir: str,
    output_path: str,
) -> dict:
    """
    Train a Random Forest classifier on annotated clips.

    Uses the mean UMAP embedding of each state as the feature vector
    (computed by applying the saved umap_reducer to cluster_centers).

    Steps
    -----
    1. Load annotations.csv — get state_id → label mapping
    2. Load cluster_info.json for cluster centers (standardized feature space)
    3. Apply umap_reducer.pkl to cluster centers → UMAP embeddings per state
    4. Build X (n_annotated_states, n_umap_dims) and y (label strings)
    5. Train RandomForestClassifier(n_estimators=100, random_state=42)
    6. Compute cross-validation accuracy (5-fold, only if n_samples >= 10)
    7. Save classifier to output_path via joblib.dump
    8. Return training report dict

    Returns {"trained": False, "reason": ...} when data are insufficient.
    """
    try:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import confusion_matrix as _sk_cm
    except ImportError as e:
        return {"trained": False, "reason": f"Missing dependency: {e}"}

    # ── Load annotations ───────────────────────────────────────────────────
    ann_path = Path(annotations_path)
    if not ann_path.exists():
        return {"trained": False, "reason": "Not enough annotations"}

    try:
        ann_df = pd.read_csv(ann_path)
    except Exception:
        return {"trained": False, "reason": "Not enough annotations"}

    # Check clip-level label counts (each individual clip must count)
    all_annotations = load_annotations(annotations_path)
    label_counts = Counter(all_annotations.values())
    if len(label_counts) < 2:
        return {"trained": False, "reason": "Not enough annotations"}
    if any(c < 5 for c in label_counts.values()):
        return {"trained": False, "reason": "Not enough annotations"}

    # Build state_id → most common label
    state_label_lists: dict[int, list[str]] = {}
    for _, row in ann_df.iterrows():
        sid = int(row.get("state_id", -1))
        label = str(row.get("assigned_label", "")).strip()
        if sid >= 0 and label:
            state_label_lists.setdefault(sid, []).append(label)

    state_label: dict[int, str] = {
        sid: Counter(labels).most_common(1)[0][0]
        for sid, labels in state_label_lists.items()
    }

    if len(set(state_label.values())) < 2:
        return {"trained": False, "reason": "Not enough annotations"}

    # ── Load cluster info ──────────────────────────────────────────────────
    cluster_info_path = os.path.join(shared_dir, "cluster_info.json")
    if not os.path.exists(cluster_info_path):
        return {"trained": False, "reason": "cluster_info.json not found — run compare.py --cluster first"}

    with open(cluster_info_path) as f:
        cluster_info = json.load(f)
    centers = np.array(cluster_info["cluster_centers"])  # (n_states, n_features)

    # ── Compute state-level embeddings ─────────────────────────────────────
    # Prefer UMAP embeddings (10D); fall back to standardized features
    umap_path = os.path.join(shared_dir, "umap_reducer.pkl")
    state_features: dict[int, np.ndarray] = {}

    if os.path.exists(umap_path):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                umap_reducer = joblib.load(umap_path)
                umap_centers = umap_reducer.transform(centers)  # (n_states, 10)
            for sid in range(len(centers)):
                state_features[sid] = umap_centers[sid]
        except Exception:
            for sid in range(len(centers)):
                state_features[sid] = centers[sid]
    else:
        for sid in range(len(centers)):
            state_features[sid] = centers[sid]

    # ── Build training matrix ──────────────────────────────────────────────
    X_list, y_list = [], []
    for sid, label in state_label.items():
        if sid in state_features:
            X_list.append(state_features[sid])
            y_list.append(label)

    if len(X_list) < 2 or len(set(y_list)) < 2:
        return {"trained": False, "reason": "Not enough annotations"}

    X = np.array(X_list)
    y = y_list

    # ── Train ──────────────────────────────────────────────────────────────
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    accuracy = None
    if len(X) >= 10:
        try:
            scores = cross_val_score(clf, X, y, cv=min(5, len(X)))
            accuracy = float(scores.mean())
        except Exception:
            pass

    clf.fit(X, y)
    y_pred = clf.predict(X)
    cm = _sk_cm(y, y_pred, labels=list(clf.classes_)).tolist()

    # ── Persist ────────────────────────────────────────────────────────────
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "clf":            clf,
            "state_features": state_features,
            "classes":        list(clf.classes_),
        },
        output_path,
    )

    report = {
        "accuracy":            accuracy,
        "n_train":             len(X),
        "classes":             list(clf.classes_),
        "feature_importances": clf.feature_importances_.tolist(),
        "confusion_matrix":    cm,
        "trained":             True,
    }

    report_path = Path(output_path).parent / "training_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    return report


def predict_clips(
    classifier_path: str,
    shared_dir: str,
    all_clips,
    annotations_path: str,
    output_path: str,
) -> "pd.DataFrame":
    """
    Apply trained classifier to all unannotated clips.

    Uses the same state-level UMAP embedding as train_classifier.
    Only predicts clips NOT already in annotations.csv.
    Saves predictions to output_path as predictions.csv.

    Columns: clip_path, state_id, predicted_label, confidence
    """
    try:
        import joblib
    except ImportError:
        return pd.DataFrame(columns=["clip_path", "state_id", "predicted_label", "confidence"])

    if not os.path.exists(classifier_path):
        return pd.DataFrame(columns=["clip_path", "state_id", "predicted_label", "confidence"])

    try:
        saved = joblib.load(classifier_path)
    except Exception:
        return pd.DataFrame(columns=["clip_path", "state_id", "predicted_label", "confidence"])

    clf = saved["clf"]
    state_features: dict = saved.get("state_features", {})

    existing = load_annotations(annotations_path)

    # Flatten all_clips → [(state_id, clip_path)]
    if isinstance(all_clips, dict):
        flat_pairs = [(sid, clip) for sid, clips in all_clips.items() for clip in clips]
    else:
        flat_pairs = []
        for clip in all_clips:
            sid = -1
            try:
                pname = Path(clip).parent.name
                if pname.startswith("state_"):
                    sid = int(pname.split("_")[1])
            except Exception:
                pass
            flat_pairs.append((sid, str(clip)))

    rows = []
    for state_id, clip_path in flat_pairs:
        if str(clip_path) in existing:
            continue  # already human-labeled
        if state_id not in state_features:
            continue

        feat = np.array(state_features[state_id]).reshape(1, -1)
        try:
            pred_label = clf.predict(feat)[0]
            proba = clf.predict_proba(feat)[0]
            confidence = float(proba.max())
        except Exception:
            continue

        rows.append({
            "clip_path":       str(clip_path),
            "state_id":        state_id,
            "predicted_label": pred_label,
            "confidence":      round(confidence, 3),
        })

    df = pd.DataFrame(
        rows,
        columns=["clip_path", "state_id", "predicted_label", "confidence"],
    ) if rows else pd.DataFrame(
        columns=["clip_path", "state_id", "predicted_label", "confidence"]
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df
