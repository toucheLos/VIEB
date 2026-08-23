"""
state_characterizer.py — Per-state characterization outputs for VIEB.

Computes interpretable summaries for each discovered behavioral state and
saves them to results/characterization/. This is a pure-Python backend
module; it has no CLI entry point and no GUI dependencies.

Outputs
-------
state_feature_profiles.csv   one row per state, raw feature means
state_feature_zscores.csv    one row per state, z-scored feature means
                             (value = how many std above global mean)
state_duration_summary.csv   per-state bout duration percentiles
state_group_enrichment.csv   fractional enrichment by metadata groups
state_characterization.json  combined summary dict (subset of above)

Public API
----------
    run(results_dir, shared_dir, features_dir, metadata_path=None, fps=30)
    load_outputs(results_dir) -> dict
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run(
    results_dir: str | Path,
    shared_dir: str | Path,
    features_dir: str | Path,
    metadata_path: Optional[str | Path] = None,
    fps: float = 30.0,
) -> dict:
    """
    Compute and save per-state characterization outputs.

    Returns a summary dict with keys: n_states, saved_files, errors.
    Missing inputs are handled gracefully — whatever can be computed will be.
    """
    results_dir = Path(results_dir)
    shared_dir = Path(shared_dir)
    features_dir = Path(features_dir)
    char_dir = results_dir / "characterization"
    char_dir.mkdir(parents=True, exist_ok=True)

    errors: list[str] = []
    saved: list[str] = []

    # ── 1. Load feature names ───────────────────────────────────────────────
    feature_names = _load_feature_names(features_dir)

    # ── 2. Load cluster info ────────────────────────────────────────────────
    cluster_info = _load_cluster_info(shared_dir)
    if not cluster_info:
        errors.append("cluster_info.json not found — run compare.py --cluster first")
        return {"n_states": 0, "saved_files": saved, "errors": errors}

    centers = np.array(cluster_info["cluster_centers"], dtype=np.float64)
    n_states = len(centers)
    n_features = centers.shape[1]

    # ── 3. Align feature names to center dimensions ─────────────────────────
    if len(feature_names) != n_features:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # ── 4. Feature z-scores (cluster centers ARE in standardized space) ─────
    # Each center value = (state_mean - global_mean) / global_std
    # i.e., the z-score of that state for that feature
    zscores_df = pd.DataFrame(
        centers,
        columns=feature_names,
    )
    zscores_df.insert(0, "state_id", list(range(n_states)))

    p = char_dir / "state_feature_zscores.csv"
    zscores_df.to_csv(p, index=False)
    saved.append(str(p))

    # ── 5. Raw feature profiles (denormalize using preprocessor stats) ──────
    profiles_df = _compute_raw_profiles(centers, feature_names, shared_dir)
    profiles_df.insert(0, "state_id", list(range(n_states)))
    p = char_dir / "state_feature_profiles.csv"
    profiles_df.to_csv(p, index=False)
    saved.append(str(p))

    # ── 6. Duration summary ─────────────────────────────────────────────────
    bouts_df = _load_bouts(char_dir)
    duration_df = _compute_duration_summary(bouts_df, n_states)
    p = char_dir / "state_duration_summary.csv"
    duration_df.to_csv(p, index=False)
    saved.append(str(p))

    # ── 7. Group enrichment ─────────────────────────────────────────────────
    metadata = _load_metadata(metadata_path)
    enrichment_df = _compute_group_enrichment(bouts_df, metadata, n_states, fps)
    p = char_dir / "state_group_enrichment.csv"
    enrichment_df.to_csv(p, index=False)
    saved.append(str(p))

    # ── 8. Movement poles ───────────────────────────────────────────────────
    n_kp = 8
    try:
        idx_path = features_dir / "index.json"
        if idx_path.exists():
            n_kp = int(json.loads(idx_path.read_text(encoding="utf-8"))
                       .get("_meta", {}).get("n_keypoints", 8))
    except Exception:
        pass

    poles = compute_movement_poles(centers, feature_names, bouts_df, n_keypoints=n_kp, fps=fps)
    p = char_dir / "poles.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(poles, fh, indent=2)
    saved.append(str(p))

    poles_rows = [
        {
            "pole_name":    "low_motion",
            "type":         poles["low_motion"]["type"],
            "state_id":     poles["low_motion"]["state_id"],
            "speed_zscore": poles["low_motion"]["speed_zscore"],
            "n_bouts":      poles["low_motion"]["n_bouts"],
            "warning":      poles["low_motion"]["warning"],
        },
        {
            "pole_name":    "high_motion",
            "type":         poles["high_motion"]["type"],
            "state_id":     poles["high_motion"]["state_id"],
            "speed_zscore": poles["high_motion"]["speed_zscore"],
            "n_bouts":      poles["high_motion"]["n_bouts"],
            "warning":      poles["high_motion"]["warning"],
        },
    ]
    p = char_dir / "movement_poles.csv"
    pd.DataFrame(poles_rows).to_csv(p, index=False)
    saved.append(str(p))

    # ── 9. Top distinguishing features per state ────────────────────────────
    top_features = _compute_top_features(centers, feature_names, n=5)

    # ── 10. Combined JSON summary ───────────────────────────────────────────
    summary = {
        "n_states": n_states,
        "n_features": n_features,
        "feature_names": feature_names,
        "states": {},
    }

    for sid in range(n_states):
        state_dur = duration_df[duration_df["state_id"] == sid]
        dur_row = state_dur.iloc[0].to_dict() if not state_dur.empty else {}

        summary["states"][str(sid)] = {
            "state_id": sid,
            "top_positive_features": top_features[sid]["positive"],
            "top_negative_features": top_features[sid]["negative"],
            "n_bouts": int(dur_row.get("n_bouts", 0)),
            "mean_bout_sec": _safe_float(dur_row.get("mean_sec")),
            "median_bout_sec": _safe_float(dur_row.get("median_sec")),
        }

    p = char_dir / "state_characterization.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    saved.append(str(p))

    return {"n_states": n_states, "saved_files": saved, "errors": errors}


# ---------------------------------------------------------------------------
# Load outputs (for UI consumption)
# ---------------------------------------------------------------------------

def load_outputs(results_dir: str | Path) -> dict:
    """Load all characterization outputs. Returns empty DataFrames for missing files."""
    char_dir = Path(results_dir) / "characterization"
    out: dict = {}

    for fname, key in [
        ("state_feature_profiles.csv",  "feature_profiles"),
        ("state_feature_zscores.csv",   "feature_zscores"),
        ("state_duration_summary.csv",  "duration_summary"),
        ("state_group_enrichment.csv",  "group_enrichment"),
    ]:
        p = char_dir / fname
        try:
            out[key] = pd.read_csv(p) if p.exists() else pd.DataFrame()
        except Exception:
            out[key] = pd.DataFrame()

    p = char_dir / "state_characterization.json"
    try:
        out["characterization"] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        out["characterization"] = {}

    p = char_dir / "poles.json"
    try:
        out["poles"] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    except Exception:
        out["poles"] = {}

    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_feature_names(features_dir: Path) -> list[str]:
    p = features_dir / "index.json"
    if not p.exists():
        return []
    try:
        idx = json.loads(p.read_text(encoding="utf-8"))
        return list(idx.get("_meta", {}).get("feature_names", []))
    except Exception:
        return []


def _load_cluster_info(shared_dir: Path) -> dict:
    p = shared_dir / "cluster_info.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_bouts(char_dir: Path) -> pd.DataFrame:
    p = char_dir / "bouts.csv"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _load_metadata(metadata_path) -> pd.DataFrame:
    if not metadata_path:
        return pd.DataFrame()
    p = Path(metadata_path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, dtype=str).fillna("")
    except Exception:
        return pd.DataFrame()


def _compute_raw_profiles(
    centers: np.ndarray,
    feature_names: list[str],
    shared_dir: Path,
) -> pd.DataFrame:
    """Denormalize cluster centers back to raw feature space if preprocessor available."""
    try:
        import joblib
        pp_path = shared_dir / "preprocessor.pkl"
        if pp_path.exists():
            pp = joblib.load(pp_path)
            # BehaviorPreprocessor.transform uses scaler internally
            scaler = getattr(pp, "scaler", None)
            if scaler is not None:
                raw = scaler.inverse_transform(centers)
                return pd.DataFrame(raw, columns=feature_names)
    except Exception:
        pass
    # Fallback: z-scores as profiles (centers are already standardized)
    return pd.DataFrame(centers, columns=feature_names)


def _compute_duration_summary(bouts_df: pd.DataFrame, n_states: int) -> pd.DataFrame:
    """Per-state bout duration statistics."""
    rows = []
    for sid in range(n_states):
        if bouts_df.empty or "state" not in bouts_df.columns:
            grp = pd.Series(dtype=float)
        else:
            col = "duration_sec"
            if col not in bouts_df.columns:
                grp = pd.Series(dtype=float)
            else:
                grp = bouts_df.loc[bouts_df["state"] == sid, col].dropna()

        rows.append({
            "state_id":   sid,
            "n_bouts":    len(grp),
            "mean_sec":   float(grp.mean())   if len(grp) else float("nan"),
            "median_sec": float(grp.median()) if len(grp) else float("nan"),
            "p25_sec":    float(grp.quantile(0.25)) if len(grp) else float("nan"),
            "p75_sec":    float(grp.quantile(0.75)) if len(grp) else float("nan"),
            "p5_sec":     float(grp.quantile(0.05)) if len(grp) else float("nan"),
            "p95_sec":    float(grp.quantile(0.95)) if len(grp) else float("nan"),
            "min_sec":    float(grp.min())    if len(grp) else float("nan"),
            "max_sec":    float(grp.max())    if len(grp) else float("nan"),
        })
    return pd.DataFrame(rows)


def _compute_group_enrichment(
    bouts_df: pd.DataFrame,
    metadata: pd.DataFrame,
    n_states: int,
    fps: float,
) -> pd.DataFrame:
    """
    Fractional enrichment of each state by metadata group variables.

    For each group variable (context, day, animal_id, experiment) that appears
    in bouts_df, computes the fraction of bouts in each group that belong to
    each state, relative to the global state fraction.

    Output columns: state_id, group_variable, group_value, fraction, enrichment_ratio
    """
    if bouts_df.empty or "state" not in bouts_df.columns:
        return pd.DataFrame(
            columns=["state_id", "group_variable", "group_value",
                     "fraction", "enrichment_ratio"]
        )

    group_cols = [c for c in ("context", "day", "animal_id", "experiment")
                  if c in bouts_df.columns]
    if not group_cols:
        # Try merging metadata by stem
        if not metadata.empty and "stem" in bouts_df.columns:
            stem_col = "stem"
            # Normalize metadata column names
            meta_cols = set(metadata.columns)
            for gc in ("context", "day", "animal_id", "experiment"):
                if gc in meta_cols:
                    bouts_df = bouts_df.merge(
                        metadata[["stem", gc]].drop_duplicates("stem"),
                        on="stem", how="left",
                    )
                    group_cols.append(gc)

    if not group_cols:
        return pd.DataFrame(
            columns=["state_id", "group_variable", "group_value",
                     "fraction", "enrichment_ratio"]
        )

    # Global state fractions
    state_counts = bouts_df["state"].value_counts(normalize=True)
    global_frac: dict[int, float] = {
        sid: float(state_counts.get(sid, 0.0)) for sid in range(n_states)
    }

    rows = []
    for gcol in group_cols:
        col = bouts_df[gcol].astype(str)
        groups = [v for v in col.unique() if v and v not in ("", "nan", "None")]
        for gval in sorted(groups):
            mask = col == gval
            sub = bouts_df.loc[mask, "state"]
            if sub.empty:
                continue
            grp_frac = sub.value_counts(normalize=True)
            for sid in range(n_states):
                frac = float(grp_frac.get(sid, 0.0))
                gf = global_frac.get(sid, 0.0)
                enrich = (frac / gf) if gf > 1e-9 else float("nan")
                rows.append({
                    "state_id":        sid,
                    "group_variable":  gcol,
                    "group_value":     gval,
                    "fraction":        round(frac, 6),
                    "enrichment_ratio": round(enrich, 4) if not (enrich != enrich) else float("nan"),
                })

    return pd.DataFrame(rows, columns=["state_id", "group_variable", "group_value",
                                        "fraction", "enrichment_ratio"])


def _compute_top_features(
    centers: np.ndarray,
    feature_names: list[str],
    n: int = 5,
) -> dict[int, dict]:
    """Return top-n positive and negative distinguishing features per state."""
    n_states = len(centers)
    result: dict[int, dict] = {}

    for sid in range(n_states):
        z = centers[sid]
        indexed = sorted(enumerate(z), key=lambda x: x[1], reverse=True)
        positive = [
            {"feature": feature_names[i], "zscore": round(float(v), 3)}
            for i, v in indexed[:n]
            if v > 0
        ]
        negative = [
            {"feature": feature_names[i], "zscore": round(float(v), 3)}
            for i, v in reversed(indexed[-n:])
            if v < 0
        ]
        result[sid] = {"positive": positive, "negative": negative}

    return result


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return None if f != f else f  # NaN → None for JSON
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Movement poles
# ---------------------------------------------------------------------------

def _get_centroid_speed_index(feature_names: list[str], n_keypoints: int) -> Optional[int]:
    """Find centroid_speed column index.
    Primary: search feature_names for 'centroid_speed'.
    Fallback: positional formula (standard VIEB model: base + 0).
    """
    for i, name in enumerate(feature_names):
        if "centroid_speed" in name.lower():
            return i
    if n_keypoints > 0:
        return n_keypoints + n_keypoints * (n_keypoints - 1) // 2
    return None


def compute_movement_poles(
    centers: np.ndarray,
    feature_names: list[str],
    bouts_df: pd.DataFrame,
    n_keypoints: int = 8,
    fps: float = 30.0,
) -> dict:
    """Select low- and high-motion pole states from cluster centers.

    Centers are z-scored; lower centroid_speed z-score = slower state.
    Returns {"low_motion": {...}, "high_motion": {...}}.
    """
    n_states = len(centers)
    _unavailable = {
        "low_motion":  {"type": "unavailable", "state_id": None, "speed_zscore": None, "n_bouts": 0, "warning": "centroid_speed feature not found"},
        "high_motion": {"type": "unavailable", "state_id": None, "speed_zscore": None, "n_bouts": 0, "warning": "centroid_speed feature not found"},
    }

    if n_states == 0:
        return _unavailable

    speed_idx = _get_centroid_speed_index(feature_names, n_keypoints)
    if speed_idx is None or speed_idx >= centers.shape[1]:
        return _unavailable

    speed_vals = centers[:, speed_idx]
    sorted_asc = np.argsort(speed_vals)

    low_id  = int(sorted_asc[0])
    high_id = int(sorted_asc[-1])

    global_std = float(speed_vals.std()) if n_states > 1 else 1.0

    def _bout_count(sid: int) -> int:
        if bouts_df.empty or "state" not in bouts_df.columns:
            return 0
        return int((bouts_df["state"] == sid).sum())

    def _ambiguity_warning(pole: str, ordered_ids: np.ndarray) -> str:
        if len(ordered_ids) < 2:
            return ""
        gap = abs(float(speed_vals[ordered_ids[1]]) - float(speed_vals[ordered_ids[0]]))
        if global_std > 1e-8 and gap < 0.3 * global_std:
            label = "low" if pole == "low" else "high"
            return f"Multiple states with similar {label} motion; selection may be ambiguous"
        return ""

    low_warn  = _ambiguity_warning("low",  sorted_asc)
    high_warn = _ambiguity_warning("high", sorted_asc[::-1])

    return {
        "low_motion": {
            "type":         "state",
            "state_id":     low_id,
            "speed_zscore": round(float(speed_vals[low_id]), 4),
            "n_bouts":      _bout_count(low_id),
            "warning":      low_warn,
        },
        "high_motion": {
            "type":         "state",
            "state_id":     high_id,
            "speed_zscore": round(float(speed_vals[high_id]), 4),
            "n_bouts":      _bout_count(high_id),
            "warning":      high_warn,
        },
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import vieb_config as _vc

    results = _vc.get_results_dir()
    shared = str(Path(results) / "shared")
    features = str(Path(results) / "features")
    metadata = _vc.get_metadata_path()

    try:
        fps = float(_vc.get_fps())
    except Exception:
        fps = 30.0

    print(f"[state_characterizer] results : {results}")
    print(f"[state_characterizer] shared  : {shared}")
    print(f"[state_characterizer] features: {features}")
    print(f"[state_characterizer] metadata: {metadata}")

    out = run(results, shared, features, metadata_path=metadata, fps=fps)
    if out["errors"]:
        for e in out["errors"]:
            print(f"[state_characterizer] ERROR: {e}")
    for f in out["saved_files"]:
        print(f"[state_characterizer] saved: {f}")
    print(f"[state_characterizer] done — {out['n_states']} states characterized")
