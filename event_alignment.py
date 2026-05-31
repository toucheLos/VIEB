"""
event_alignment.py — Peri-event behavioral state analysis for VIEB

Analyses discrete experiments (trials with outcomes) by grouping videos by
event label and computing mean state occupancy per group.  Works at session
level (one outcome per video); sub-session timestamped events are a stub.
"""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_results_dir() -> str:
    try:
        import vieb_config as _vc
        return _vc.get_results_dir()
    except Exception:
        return os.path.join(os.getcwd(), "results")


def _get_shared_dir(index: dict) -> str:
    try:
        import vieb_config as _vc
        return os.path.join(_vc.get_results_dir(), "shared")
    except Exception:
        pass
    # Fallback: derive from first feature file path
    for k, v in index.items():
        if k == "_meta":
            continue
        fp = v.get("features_path", "")
        if fp:
            return str(Path(fp).parent.parent / "shared")
    return os.path.join(os.getcwd(), "results", "shared")


def _get_output_dir(index: dict) -> str:
    try:
        import vieb_config as _vc
        return os.path.join(_vc.get_results_dir(), "quantification")
    except Exception:
        shared = _get_shared_dir(index)
        return str(Path(shared).parent / "quantification")


def _get_n_clusters(shared_dir: str) -> int:
    ci_path = os.path.join(shared_dir, "cluster_info.json")
    if not os.path.exists(ci_path):
        return 0
    try:
        with open(ci_path) as f:
            return int(json.load(f).get("n_clusters", 0))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Part 1 — Event detection
# ---------------------------------------------------------------------------

def load_events(metadata_path: str, column_map: dict) -> pd.DataFrame | None:
    """Load events from metadata.csv using the column_map event key.

    Returns a DataFrame with columns: stem, event_label [, animal_id]
    Returns None if no event column is configured or the column is absent.
    """
    event_col = column_map.get("event", "")
    if not event_col:
        return None

    try:
        meta = pd.read_csv(metadata_path)
    except Exception:
        return None

    if event_col not in meta.columns:
        return None

    meta = meta.copy()

    if "filename" in meta.columns:
        meta["stem"] = meta["filename"].str.replace(r"\.mp4$", "", regex=True)
    elif "stem" not in meta.columns:
        return None

    meta = meta.dropna(subset=[event_col])
    if meta.empty:
        return None

    keep = ["stem", event_col]
    animal_col = column_map.get("animal_id", "animal_id") or "animal_id"
    if animal_col and animal_col in meta.columns:
        keep.append(animal_col)

    result = meta[keep].copy()
    rename = {event_col: "event_label"}
    if animal_col and animal_col != "animal_id" and animal_col in result.columns:
        rename[animal_col] = "animal_id"
    result = result.rename(columns=rename)

    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Part 2 — Peri-event state profiles
# ---------------------------------------------------------------------------

def compute_peri_event_profiles(
    index: dict,
    events_df: pd.DataFrame,
    window_sec: float = 5.0,
    fps: float = 30.0,
    min_confidence: float = 0.0,
) -> dict:
    """Compute mean state occupancy per event label (session-level events).

    Parameters
    ----------
    index : dict
        Feature index from results/features/index.json.
    events_df : pd.DataFrame
        Output of load_events(): columns stem, event_label [, animal_id].
    window_sec : float
        Time window for sub-session events (reserved; not yet used).
    fps : float
        Video frame rate (reserved for sub-session events).
    min_confidence : float
        HDBSCAN soft probability threshold; 0 = no filtering.

    Returns
    -------
    dict : {event_label: np.ndarray shape (n_states,)}
    """
    shared_dir = _get_shared_dir(index)
    n_clusters = _get_n_clusters(shared_dir)
    if n_clusters == 0:
        print("[WARN] No cluster_info.json found or n_clusters=0. Run --cluster first.")
        return {}

    # Internal tracking for CSV output
    _session_info: dict[str, dict] = {}

    result_profiles: dict[str, np.ndarray] = {}

    for label, group in events_df.groupby("event_label"):
        state_sums = np.zeros(n_clusters, dtype=np.float64)
        n_sessions = 0

        for _, row in group.iterrows():
            stem = str(row["stem"])
            labels_path = os.path.join(shared_dir, f"{stem}_labels.npy")
            if not os.path.exists(labels_path):
                print(f"  [SKIP] No labels for {stem}")
                continue
            labels = np.load(labels_path)

            probs_path = os.path.join(shared_dir, f"{stem}_probs.npy")
            if min_confidence > 0 and os.path.exists(probs_path):
                probs = np.load(probs_path)
                valid = (labels >= 0) & (probs >= min_confidence)
            else:
                valid = labels >= 0

            denom = int(valid.sum())
            if denom == 0:
                continue

            for k in range(n_clusters):
                state_sums[k] += float((labels[valid] == k).sum()) / denom
            n_sessions += 1

        if n_sessions == 0:
            continue

        fractions = state_sums / n_sessions
        n_animals = (
            int(group["animal_id"].nunique())
            if "animal_id" in group.columns
            else n_sessions
        )
        result_profiles[label] = fractions
        _session_info[label] = {"n_sessions": n_sessions, "n_animals": n_animals}

    # Save peri_event_profiles.csv
    try:
        output_dir = _get_output_dir(index)
        os.makedirs(output_dir, exist_ok=True)
        rows = []
        for label, fracs in result_profiles.items():
            info = _session_info[label]
            row: dict = {
                "event_label": label,
                "n_sessions": info["n_sessions"],
                "n_animals": info["n_animals"],
            }
            for k, frac in enumerate(fracs):
                row[f"state_{k}_frac"] = round(float(frac), 6)
            rows.append(row)
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(output_dir, "peri_event_profiles.csv"), index=False
            )
            print("Peri-event profiles saved → results/quantification/peri_event_profiles.csv")
    except Exception as e:
        print(f"[WARN] Could not save peri_event_profiles.csv: {e}")

    return result_profiles


# ---------------------------------------------------------------------------
# Part 3 — Event contrast vector
# ---------------------------------------------------------------------------

def compute_event_contrast(peri_event_profiles: dict) -> dict:
    """Compute pairwise contrast vectors between event labels.

    Parameters
    ----------
    peri_event_profiles : dict
        {event_label: np.ndarray shape (n_states,)} — output of
        compute_peri_event_profiles().

    Returns
    -------
    dict : {"{label_A}_vs_{label_B}": {"contrast_vector": np.ndarray,
                                        "contrast_magnitude": float,
                                        "label_A": str, "label_B": str,
                                        "dominant_state_A": int,
                                        "dominant_state_B": int}}
    """
    labels = sorted(peri_event_profiles.keys())
    if len(labels) < 2:
        print("[WARN] Need at least 2 event labels for contrast computation.")
        return {}

    results: dict = {}
    for label_a, label_b in itertools.combinations(labels, 2):
        fa = np.asarray(peri_event_profiles[label_a], dtype=np.float64)
        fb = np.asarray(peri_event_profiles[label_b], dtype=np.float64)
        cv = fa - fb
        magnitude = float(np.linalg.norm(cv) / np.sqrt(2))
        key = f"{label_a}_vs_{label_b}"
        results[key] = {
            "contrast_vector": cv,
            "contrast_magnitude": magnitude,
            "label_A": label_a,
            "label_B": label_b,
            "dominant_state_A": int(np.argmax(fa)),
            "dominant_state_B": int(np.argmax(fb)),
        }

    # Save event_contrast.csv
    try:
        res_dir = _get_results_dir()
        output_dir = os.path.join(res_dir, "quantification")
        os.makedirs(output_dir, exist_ok=True)
        rows = [
            {
                "label_A": info["label_A"],
                "label_B": info["label_B"],
                "contrast_magnitude": round(info["contrast_magnitude"], 6),
                "dominant_state_A": info["dominant_state_A"],
                "dominant_state_B": info["dominant_state_B"],
            }
            for info in results.values()
        ]
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(output_dir, "event_contrast.csv"), index=False
            )
            print("Event contrast saved → results/quantification/event_contrast.csv")
    except Exception as e:
        print(f"[WARN] Could not save event_contrast.csv: {e}")

    return results
