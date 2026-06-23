#!/usr/bin/env python3
"""
diagnose_clusters.py — HDBSCAN (and optionally UMAP) parameter sweep for VIEB.

Usage
-----
    python diagnose_clusters.py
        Quick sweep of min_cluster_size on a <=200k-frame sample.

    python diagnose_clusters.py --umap-dims 2 --min-samples 5
        Re-embed to 2-D, sweep MCS with min_samples fixed at 5.

    python diagnose_clusters.py --full --silhouette
        Use up to 1M frames and compute slower silhouette scores.

    python diagnose_clusters.py --umap-sweep
        Also sweep UMAP n_neighbors on a 50k-frame sample.

    python diagnose_clusters.py --min-cluster-sizes 100,200,500,1000,2000
        Custom comma-separated list of min_cluster_size values.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

_DEFAULT_MCS = [50, 100, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
_QUICK_MAX_FRAMES = 200_000
_FULL_MAX_FRAMES = 1_000_000
_DEFAULT_MAX_FRAMES = _QUICK_MAX_FRAMES
_UMAP_NEIGHBORS = [10, 15, 30, 50]
_DEFAULT_SEED = 42
_DIAG_DIR = "results/diagnostics"
_CACHE_DIR = os.path.join(_DIAG_DIR, "cache")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pooled_features(max_frames: int | None = None, seed: int = 42):
    """Load standardized pooled feature matrix from results/features/."""
    index_path = "results/features/index.json"
    if not os.path.exists(index_path):
        sys.exit("Missing results/features/index.json — run compare.py --extract first.")
    with open(index_path) as f:
        index = json.load(f)
    if not index:
        sys.exit("Feature index is empty — run compare.py --extract first.")

    stems = sorted(k for k in index.keys() if k != "_meta")
    parts = []
    for stem in stems:
        p = index[stem]["features_path"].replace("\\", "/")
        if not os.path.exists(p):
            continue
        parts.append(np.load(p).astype(np.float32))

    pooled = np.vstack(parts)

    if max_frames and len(pooled) > max_frames:
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(pooled), max_frames, replace=False))
        pooled = pooled[idx]

    return pooled


def _index_signature(index_path: str) -> str:
    """Small cache-busting signature for the feature index."""
    try:
        stat = os.stat(index_path)
        return f"{int(stat.st_mtime)}_{stat.st_size}"
    except OSError:
        return "missing"


def _cache_token(value: object) -> str:
    return "all" if value is None else str(value)


def _load_feature_metadata() -> dict:
    """Load feature availability metadata from results/features/index.json."""
    index_path = "results/features/index.json"
    if not os.path.exists(index_path):
        return {
            "feature_count": 0,
            "feature_names": [],
            "semantic_features": [],
            "keypoint_groups": {},
            "skipped_features": {},
            "scale_normalization_detected": False,
        }

    with open(index_path) as f:
        index = json.load(f)

    meta = index.get("_meta", {})
    first_entry = next((v for k, v in index.items() if k != "_meta"), {})
    keypoint_groups = meta.get("keypoint_groups", {}) or {}
    semantic_features = meta.get("semantic_features", []) or []
    skipped_features = meta.get("skipped_features", {}) or {}
    feature_names = meta.get("feature_names", []) or []
    scale_normalization_detected = _detect_scale_normalization(index)

    if not skipped_features and keypoint_groups:
        try:
            from ml.feature_extraction import PoseFeatureExtractor
            requirements = PoseFeatureExtractor._SEMANTIC_FEATURE_GROUPS
        except Exception:
            requirements = {}
        for feat_name, required_groups in requirements.items():
            if feat_name in semantic_features:
                continue
            missing = [
                group for group in required_groups
                if not keypoint_groups.get(group, {}).get("resolved", False)
            ]
            if missing:
                skipped_features[feat_name] = f"missing groups: {', '.join(missing)}"

    return {
        "feature_count": int(meta.get("n_features", first_entry.get("n_features", 0)) or 0),
        "feature_names": feature_names,
        "semantic_features": semantic_features,
        "keypoint_groups": keypoint_groups,
        "skipped_features": skipped_features,
        "scale_normalization_detected": scale_normalization_detected,
    }


def _detect_scale_normalization(index: dict) -> bool:
    """Return True only when index metadata clearly records body/scale normalization."""
    indicators = (
        "scale_normalized",
        "scale_normalization",
        "body_size_normalized",
        "body_size_normalization",
        "distance_normalized",
        "speed_normalized",
        "normalized_by_body_size",
    )
    meta = index.get("_meta", {}) or {}

    def has_positive_indicator(obj: object) -> bool:
        if not isinstance(obj, dict):
            return False
        for key, value in obj.items():
            key_l = str(key).lower()
            if any(ind in key_l for ind in indicators):
                if isinstance(value, bool):
                    if value:
                        return True
                elif isinstance(value, str):
                    if value.strip().lower() in {"true", "yes", "body_size", "scale", "normalized"}:
                        return True
                elif value:
                    return True
        return False

    if has_positive_indicator(meta):
        return True

    for key, value in index.items():
        if key == "_meta":
            continue
        if has_positive_indicator(value):
            return True

    return False


def _load_or_build_scaled(max_frames: int | None, seed: int = _DEFAULT_SEED,
                          use_cache: bool = True):
    """Load sampled standardized matrix, caching the result for repeat diagnostics."""
    index_path = "results/features/index.json"
    if use_cache:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        sig = _index_signature(index_path)
        cache_path = os.path.join(
            _CACHE_DIR,
            f"scaled_max{_cache_token(max_frames)}_seed{seed}_idx{sig}.npz",
        )
        if os.path.exists(cache_path):
            print(f"  Loading cached standardized matrix -> {cache_path}")
            with np.load(cache_path) as data:
                return np.asarray(data["scaled"], dtype=np.float32)
    else:
        cache_path = ""

    pooled_raw = _load_pooled_features(max_frames=max_frames, seed=seed)
    print(f"  {len(pooled_raw):,} frames × {pooled_raw.shape[1]} features loaded")

    print("\nStandardizing features…")
    scaled = _standardize(pooled_raw)

    if use_cache and cache_path:
        np.savez(cache_path, scaled=scaled.astype(np.float32))
        print(f"  Cached standardized matrix -> {cache_path}")
    return scaled


def _standardize(pooled):
    """Load or fit the shared BehaviorPreprocessor."""
    from ml import BehaviorPreprocessor
    prep_path = "results/shared/preprocessor.pkl"
    if os.path.exists(prep_path):
        prep = BehaviorPreprocessor.load(prep_path)
        return np.asarray(prep.transform(pooled), dtype=np.float32)

    prep = BehaviorPreprocessor(use_pca=False)
    return np.asarray(prep.fit_transform(pooled), dtype=np.float32)


def _try_cuml_umap():
    """Return cuML UMAP class if available, else None."""
    import platform
    if platform.system() == "Windows":
        return None  # cuML has no Windows wheels; requires WSL2
    try:
        from cuml.manifold import UMAP as CuUMAP
        import cupy as cp
        cp.cuda.runtime.getDeviceCount()
        return CuUMAP
    except Exception:
        return None


def _get_umap_embedding(scaled, max_fit_frames: int = 200_000, seed: int = 42,
                        n_components: int = 10, n_neighbors: int = 30,
                        max_frames: int | None = None, use_cache: bool = True):
    """
    Return UMAP embedding.

    Priority:
    1. Load existing results/shared/umap_reducer.pkl (when n_neighbors==30 and load succeeds).
    2. cuML GPU UMAP if available (Linux + CUDA).
    3. CPU umap-learn with n_jobs=-1 (all cores, no fixed seed so parallelism is enabled).
    """
    cache_path = ""
    if use_cache:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(
            _CACHE_DIR,
            "umap_"
            f"max{_cache_token(max_frames)}_seed{seed}_"
            f"dims{n_components}_neighbors{n_neighbors}.npz",
        )
        if os.path.exists(cache_path):
            print(f"  Loading cached UMAP embedding -> {cache_path}")
            with np.load(cache_path) as data:
                return np.asarray(data["embedding"], dtype=np.float32)

    reducer_path = "results/shared/umap_reducer.pkl"
    _loaded_reducer = None
    # Only reuse the saved reducer when the requested geometry matches the saved one (10-D, n_neighbors=30)
    if os.path.exists(reducer_path) and n_neighbors == 30 and n_components == 10:
        try:
            import joblib
            _loaded_reducer = joblib.load(reducer_path)
            print(f"  Using saved UMAP reducer (n_components={n_components}, n_neighbors={n_neighbors})...")
            emb = _loaded_reducer.transform(scaled)
        except Exception as e:
            print(f"  Could not load saved reducer ({e}); recomputing UMAP...")
            _loaded_reducer = None

    if _loaded_reducer is None:
        import multiprocessing
        n_cpu = multiprocessing.cpu_count()

        CuUMAP = _try_cuml_umap()
        if CuUMAP is not None:
            UMAPClass = CuUMAP
            umap_kwargs = dict(n_components=n_components, n_neighbors=n_neighbors,
                               min_dist=0.0)
            backend = "GPU (cuML)"
        else:
            try:
                import umap as umap_lib
            except ImportError:
                sys.exit("umap-learn not installed.  Run: pip install umap-learn")
            UMAPClass = umap_lib.UMAP
            # Omit random_state so umap-learn can use n_jobs > 1.
            umap_kwargs = dict(n_components=n_components, n_neighbors=n_neighbors,
                               min_dist=0.0, n_jobs=n_cpu, low_memory=True)
            backend = f"CPU ({n_cpu} cores, parallel)"

        n_total = len(scaled)
        n_fit = min(max_fit_frames, n_total)
        rng = np.random.default_rng(seed)
        fit_idx = np.sort(rng.choice(n_total, n_fit, replace=False)) if n_total > n_fit else np.arange(n_total)
        fit_data = scaled[fit_idx]

        print(f"  Fitting UMAP [{backend}] (n_neighbors={n_neighbors}, "
              f"n_components={n_components}) on {n_fit:,}-frame sample…")
        reducer = UMAPClass(**umap_kwargs)
        try:
            reducer.fit(fit_data)
            emb = reducer.transform(scaled)
        except Exception as gpu_err:
            if backend.startswith("GPU"):
                print(f"  GPU UMAP failed ({gpu_err}); falling back to CPU…")
                import multiprocessing
                try:
                    import umap as umap_lib
                except ImportError:
                    sys.exit("umap-learn not installed.  Run: pip install umap-learn")
                n_cpu = multiprocessing.cpu_count()
                umap_kwargs_cpu = dict(n_components=n_components, n_neighbors=n_neighbors,
                                       min_dist=0.0, n_jobs=n_cpu, low_memory=True)
                reducer = umap_lib.UMAP(**umap_kwargs_cpu)
                reducer.fit(fit_data)
                emb = reducer.transform(scaled)
            else:
                raise

    if hasattr(emb, "to_numpy"):
        emb = emb.to_numpy()
    elif hasattr(emb, "get"):
        emb = emb.get()
    emb = np.asarray(emb, dtype=np.float32)
    if use_cache and cache_path:
        np.savez(cache_path, embedding=emb)
        print(f"  Cached UMAP embedding -> {cache_path}")
    return emb


def _run_hdbscan(emb, mcs: int, min_samples: int | None = None,
                 core_dist_n_jobs: int = 1):
    """Fit HDBSCAN and return (labels, n_clusters, noise_frac, dom_frac, mean_size, elapsed)."""
    try:
        import hdbscan as hdbscan_lib
    except ImportError:
        sys.exit("hdbscan not installed.  Run: pip install hdbscan")

    # min_samples defaults to min_cluster_size when not specified.
    # Setting it much lower than min_cluster_size (e.g. 5) keeps more frames in
    # clusters and reduces the noise fraction without requiring larger clusters.
    ms = min_samples if min_samples is not None else mcs

    t0 = time.time()
    model = hdbscan_lib.HDBSCAN(min_cluster_size=mcs, min_samples=ms,
                                  cluster_selection_method="eom",
                                  core_dist_n_jobs=core_dist_n_jobs)
    model.fit(emb)
    labels = np.asarray(model.labels_, dtype=np.int32)
    elapsed = time.time() - t0

    valid = labels[labels >= 0]
    n_clusters = int(len(np.unique(valid))) if len(valid) else 0
    noise_frac = float((labels == -1).sum()) / len(labels)

    if n_clusters == 0:
        return labels, 0, noise_frac, 1.0, 0.0, elapsed

    counts = np.bincount(valid, minlength=n_clusters)
    dom_frac = float(counts.max()) / len(labels)
    non_dom = counts[counts != counts.max()]
    mean_size = float(non_dom.mean()) if len(non_dom) > 0 else 0.0

    return labels, n_clusters, noise_frac, dom_frac, mean_size, elapsed


def _silhouette(emb, labels, max_per_cluster: int = 500, max_total: int = 10_000, seed: int = 42):
    """
    Stratified silhouette score computed on the UMAP embedding.

    Sampling strategy:
    - Take up to max_per_cluster frames from each cluster (all frames if cluster is smaller).
    - Cap total sample at max_total.
    - Requires >= 2 clusters with at least 1 sample each.

    This avoids NaN from uniform random samples that happen to miss small clusters.
    """
    from sklearn.metrics import silhouette_score

    valid_mask = labels >= 0
    valid_emb = emb[valid_mask]
    valid_lbl = labels[valid_mask]
    unique_clusters = np.unique(valid_lbl)
    n_classes = len(unique_clusters)
    if n_classes < 2 or len(valid_lbl) == 0:
        return float("nan")

    rng = np.random.default_rng(seed)
    parts = []
    for c in unique_clusters:
        c_idx = np.where(valid_lbl == c)[0]
        n_take = min(len(c_idx), max_per_cluster)
        parts.append(rng.choice(c_idx, n_take, replace=False))

    sample_idx = np.concatenate(parts)
    if len(sample_idx) > max_total:
        sample_idx = rng.choice(sample_idx, max_total, replace=False)

    s_emb = valid_emb[sample_idx]
    s_lbl = valid_lbl[sample_idx]
    # Ensure every selected cluster still has ≥1 sample after capping
    if len(np.unique(s_lbl)) < 2:
        return float("nan")

    try:
        return float(silhouette_score(s_emb, s_lbl, metric="euclidean"))
    except Exception:
        return float("nan")


def _recommend(rows: list[dict]) -> dict | None:
    """Pick the best row: n_clusters in [8,25], dom_frac < 0.5, noise_frac < 0.20."""
    rows = [r for r in rows if not r.get("error")]
    if not rows:
        return None

    candidates = [r for r in rows
                  if 8 <= r["n_clusters"] <= 25
                  and r["dom_frac"] < 0.5
                  and r["noise_frac"] < 0.20]
    if not candidates:
        # Relax: dom_frac < 0.6, n_clusters >= 6
        candidates = [r for r in rows
                      if r["n_clusters"] >= 6
                      and r["dom_frac"] < 0.6
                      and r["noise_frac"] < 0.20]
    if not candidates:
        candidates = rows
    # Among candidates, prefer: most clusters, then lowest dom_frac
    return max(candidates, key=lambda r: (r["n_clusters"], -r["dom_frac"]))


def _effective_min_samples(mcs: int, min_samples: int | None) -> int:
    return int(min_samples if min_samples is not None else mcs)


def _recommended_compare_command(rec: dict | None, umap_dims: int,
                                 min_samples: int | None,
                                 warnings: list[str] | None = None) -> str:
    if not rec:
        return ""

    params = _recommended_parameters(rec, umap_dims, min_samples, warnings)

    return (
        "python compare.py --cluster"
        f" --umap-dims {params['umap_dims']}"
        f" --min-cluster-size {params['min_cluster_size']}"
        f" --hdbscan-min-samples {params['hdbscan_min_samples']}"
        f" --hdbscan-sample {params['hdbscan_sample']}"
    )


def _recommended_parameters(rec: dict | None, umap_dims: int,
                            min_samples: int | None,
                            warnings: list[str] | None = None) -> dict:
    if not rec:
        return {}

    warning_text = " ".join(warnings or [])
    recommended_dims = int(umap_dims)
    if (
        umap_dims == 2
        or "Try --umap-dims 5 or --umap-dims 8 or --umap-dims 10." in warning_text
        or "5–10D UMAP" in warning_text
    ):
        recommended_dims = 8

    recommended_ms = _effective_min_samples(rec["mcs"], min_samples)
    if (
        "Try lowering min_samples independently from min_cluster_size" in warning_text
        or "Try --umap-dims 5 or --umap-dims 8 or --umap-dims 10." in warning_text
    ):
        recommended_ms = 50

    return {
        "umap_dims": recommended_dims,
        "min_cluster_size": int(rec["mcs"]),
        "hdbscan_min_samples": int(recommended_ms),
        "hdbscan_sample": 300_000,
    }


def _build_warning_strings(umap_dims: int, n_clusters: int, noise_fraction: float,
                           largest_state_occupancy: float,
                           feature_meta: dict | None = None) -> list[str]:
    """Generate project-agnostic clustering health warnings."""
    warnings: list[str] = []
    feature_meta = feature_meta or {}
    skipped_features = feature_meta.get("skipped_features", {}) or {}
    semantic_features = feature_meta.get("semantic_features", []) or []

    if umap_dims == 2:
        warnings.append(
            "2D UMAP is useful for visualization but often too compressed for clustering. "
            "Try 5–10D UMAP for clustering."
        )
    if n_clusters <= 3:
        warnings.append(
            "Try --umap-dims 5 or --umap-dims 8 or --umap-dims 10."
        )
    if largest_state_occupancy > 0.85:
        warnings.append(
            "Try lowering min_samples independently from min_cluster_size, e.g. min_samples 50–100."
        )
    if noise_fraction == 0 and largest_state_occupancy > 0.85:
        warnings.append(
            "0 noise with one dominant state may indicate overly conservative or overly smoothed clustering."
        )
    if not feature_meta.get("scale_normalization_detected", False):
        warnings.append(
            "Scale/body-size normalization was not detected. If animals, camera scale, or body size "
            "vary across sessions, raw distance features may dominate clustering."
        )

    n_skipped = len(skipped_features) if isinstance(skipped_features, dict) else len(skipped_features)
    n_semantic_total = n_skipped + len(semantic_features)
    if n_skipped and (n_skipped >= 2 or n_skipped >= max(1, n_semantic_total / 2)):
        warnings.append(
            "Many semantic features were skipped; check the keypoint group mapping."
        )
    return warnings


def _build_diagnostic_json(rows: list[dict], rec: dict | None, umap_dims: int,
                           min_samples: int | None, n_neighbors: int,
                           max_frames: int | None, seed: int,
                           silhouette_enabled: bool) -> dict:
    feature_meta = _load_feature_metadata()
    best = rec or next((r for r in rows if not r.get("error")), None)
    if best:
        n_clusters = int(best["n_clusters"])
        noise_fraction = float(best["noise_frac"])
        largest_state_occupancy = float(best["dom_frac"])
    else:
        n_clusters = 0
        noise_fraction = float("nan")
        largest_state_occupancy = float("nan")

    warnings = _build_warning_strings(
        umap_dims,
        n_clusters,
        noise_fraction if not math.isnan(noise_fraction) else 0.0,
        largest_state_occupancy if not math.isnan(largest_state_occupancy) else 0.0,
        feature_meta,
    )

    tested_mcs = [int(r["mcs"]) for r in rows]
    recommended_mcs = int(rec["mcs"]) if rec else None
    recommended_params = _recommended_parameters(rec, umap_dims, min_samples, warnings)
    command = _recommended_compare_command(rec, umap_dims, min_samples, warnings)
    umap_dims_likely_too_low = bool(umap_dims == 2 or n_clusters <= 3)
    dominant_state_collapse_detected = bool(
        not math.isnan(largest_state_occupancy) and largest_state_occupancy > 0.85
    )

    return {
        "tested_parameters": {
            "umap_dims": int(umap_dims),
            "n_neighbors": int(n_neighbors),
            "min_samples": min_samples,
            "effective_min_samples": (
                _effective_min_samples(recommended_mcs, min_samples)
                if recommended_mcs else min_samples
            ),
            "min_cluster_size_values": tested_mcs,
            "max_frames": max_frames,
            "seed": seed,
            "silhouette_enabled": silhouette_enabled,
        },
        "recommended_parameters": recommended_params,
        "umap_dims": int(umap_dims),
        "n_neighbors": int(n_neighbors),
        "min_samples": _effective_min_samples(recommended_mcs, min_samples) if recommended_mcs else min_samples,
        "tested_min_cluster_size_values": tested_mcs,
        "recommended_min_cluster_size": recommended_mcs,
        "n_clusters": n_clusters,
        "noise_fraction": round(noise_fraction, 6) if not math.isnan(noise_fraction) else None,
        "largest_state_occupancy": (
            round(largest_state_occupancy, 6) if not math.isnan(largest_state_occupancy) else None
        ),
        "umap_dims_likely_too_low": umap_dims_likely_too_low,
        "dominant_state_collapse_detected": dominant_state_collapse_detected,
        "dominant_state_warning": dominant_state_collapse_detected,
        "low_state_count_warning": bool(n_clusters <= 3),
        "feature_count": feature_meta["feature_count"],
        "semantic_features": feature_meta["semantic_features"],
        "keypoint_groups": feature_meta["keypoint_groups"],
        "skipped_features": feature_meta["skipped_features"],
        "scale_normalization_detected": feature_meta["scale_normalization_detected"],
        "max_frames": max_frames,
        "seed": seed,
        "silhouette_enabled": silhouette_enabled,
        "recommended_next_command": command,
        "warnings": warnings,
    }


def _save_diagnostic_json(diagnostics: dict, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Diagnostics JSON saved -> {out_path}")


def _print_table(rows: list[dict]):
    header = f"{'MCS':>6}  {'n_clusters':>10}  {'noise%':>7}  {'dom%':>6}  {'mean_size':>10}  {'silhouette':>11}  {'time_s':>6}  status"
    print(header)
    print("-" * len(header))
    for r in rows:
        if r.get("error"):
            print(f"{r['mcs']:>6}  {'failed':>10}  {'n/a':>7}  {'n/a':>6}  "
                  f"{'n/a':>10}  {'n/a':>11}  {r['elapsed']:>5.1f}s  {r['error']}")
            continue

        sil = f"{r['silhouette']:11.3f}" if not math.isnan(r["silhouette"]) else "        n/a"
        print(f"{r['mcs']:>6}  {r['n_clusters']:>10}  "
              f"{r['noise_frac']*100:>6.1f}%  {r['dom_frac']*100:>5.1f}%  "
              f"{r['mean_size']:>10.0f}  {sil}  {r['elapsed']:>5.1f}s  ok")


def _save_plot(rows: list[dict], out_path: str):
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend; safe in headless/WSL environments
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    rows = [r for r in rows if not r.get("error")]
    if not rows:
        print("No successful rows available — skipping plot.")
        return

    mcs_vals = [r["mcs"] for r in rows]
    n_cls    = [r["n_clusters"] for r in rows]
    dom_pct  = [r["dom_frac"] * 100 for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 4))
    color1 = "#1a73e8"
    color2 = "#c62828"

    ax1.set_xlabel("min_cluster_size")
    ax1.set_ylabel("n_clusters", color=color1)
    ax1.plot(mcs_vals, n_cls, "o-", color=color1, label="n_clusters")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.axhline(8, color=color1, linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.axhline(25, color=color1, linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.set_ylabel("dominant state %", color=color2)
    ax2.plot(mcs_vals, dom_pct, "s--", color=color2, label="dominant %")
    ax2.tick_params(axis="y", labelcolor=color2)
    ax2.axhline(50, color=color2, linestyle=":", linewidth=0.8)

    plt.title("Cluster Quality vs min_cluster_size")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved -> {out_path}")


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def cmd_sweep(mcs_list: list[int], umap_dims: int = 10, min_samples: int | None = None,
              max_frames: int | None = _DEFAULT_MAX_FRAMES, hdbscan_jobs: int = 1,
              silhouette_enabled: bool = False, n_neighbors: int = 30,
              seed: int = _DEFAULT_SEED, use_cache: bool = True):
    print("=" * 60)
    print("VIEB — Cluster Diagnostic (HDBSCAN sweep)")
    print(f"  umap_dims={umap_dims}  min_samples={'auto (=mcs)' if min_samples is None else min_samples}")
    print(f"  n_neighbors={n_neighbors}  hdbscan_jobs={hdbscan_jobs}")
    print(f"  silhouette={'on' if silhouette_enabled else 'off'}")
    max_frames_label = "all" if max_frames is None else f"{max_frames:,}"
    print(f"  max_frames={max_frames_label} (subsample for speed/memory)")
    print("=" * 60)

    print("\nLoading features…")
    scaled = _load_or_build_scaled(max_frames=max_frames, seed=seed, use_cache=use_cache)
    print(f"  Standardized matrix: {scaled.shape[0]:,} frames × {scaled.shape[1]} features")

    print(f"\nBuilding UMAP embedding (n_components={umap_dims})…")
    emb = _get_umap_embedding(
        scaled,
        n_components=umap_dims,
        n_neighbors=n_neighbors,
        max_frames=max_frames,
        seed=seed,
        use_cache=use_cache,
    )
    print(f"  Embedding shape: {emb.shape}")

    rows = []
    ms_label = f"min_samples={min_samples}" if min_samples is not None else "min_samples=mcs"
    print(f"\nSweeping {len(mcs_list)} min_cluster_size values ({ms_label})…\n")
    for mcs in mcs_list:
        print(f"  mcs={mcs}…", end=" ", flush=True)
        t0 = time.time()
        try:
            labels, n_cls, noise_frac, dom_frac, mean_size, elapsed = _run_hdbscan(
                emb, mcs, min_samples=min_samples, core_dist_n_jobs=hdbscan_jobs
            )
        except Exception as exc:
            elapsed = time.time() - t0
            detail = str(exc).strip().splitlines()[0] if str(exc).strip() else "no details"
            err = f"{type(exc).__name__}: {detail}"
            print(f"failed after {elapsed:.1f}s ({err})")
            rows.append({
                "mcs": mcs,
                "n_clusters": np.nan,
                "noise_frac": np.nan,
                "dom_frac": np.nan,
                "mean_size": np.nan,
                "silhouette": np.nan,
                "elapsed": elapsed,
                "error": err,
            })
            continue

        sil = _silhouette(emb, labels, seed=seed) if silhouette_enabled else float("nan")
        sil_str = f"{sil:.3f}" if not math.isnan(sil) else "n/a"
        print(f"n_clusters={n_cls}, dom={dom_frac*100:.1f}%, "
              f"noise={noise_frac*100:.1f}%, sil={sil_str} ({elapsed:.1f}s)")
        rows.append({
            "mcs": mcs,
            "n_clusters": n_cls,
            "noise_frac": noise_frac,
            "dom_frac": dom_frac,
            "mean_size": mean_size,
            "silhouette": sil,
            "elapsed": elapsed,
            "error": "",
        })

    rows.sort(key=lambda r: (-1 if r.get("error") else r["n_clusters"]), reverse=True)

    print("\n" + "=" * 60)
    print("Results (sorted by n_clusters):")
    print("=" * 60)
    _print_table(rows)

    rec = _recommend(rows)
    diagnostics = _build_diagnostic_json(
        rows,
        rec,
        umap_dims=umap_dims,
        min_samples=min_samples,
        n_neighbors=n_neighbors,
        max_frames=max_frames,
        seed=seed,
        silhouette_enabled=silhouette_enabled,
    )
    print("\n" + "=" * 60)
    if rec:
        print(f"RECOMMENDATION: min_cluster_size = {rec['mcs']}")
        print(f"  -> n_clusters={rec['n_clusters']}, dominant={rec['dom_frac']*100:.1f}%, "
              f"noise={rec['noise_frac']*100:.1f}%")
        if diagnostics["warnings"]:
            print("\nWarnings:")
            for warning in diagnostics["warnings"]:
                print(f"  - {warning}")
        print(f"\nRun: {diagnostics['recommended_next_command']}")
    else:
        print("No ideal setting found. Review the table manually.")
    print("=" * 60)

    os.makedirs("results/shared", exist_ok=True)
    os.makedirs(_DIAG_DIR, exist_ok=True)
    csv_path = "results/shared/cluster_diagnostic.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nTable saved -> {csv_path}")
    diag_csv_path = os.path.join(_DIAG_DIR, "cluster_diagnostic.csv")
    pd.DataFrame(rows).to_csv(diag_csv_path, index=False)
    print(f"Table saved -> {diag_csv_path}")

    _save_plot(rows, "results/shared/cluster_diagnostic.png")
    _save_plot(rows, os.path.join(_DIAG_DIR, "cluster_diagnostic.png"))
    _save_diagnostic_json(diagnostics, os.path.join(_DIAG_DIR, "cluster_diagnostics.json"))
    return rec


# ---------------------------------------------------------------------------
# UMAP sweep
# ---------------------------------------------------------------------------

def cmd_umap_sweep(rec_mcs: int | None, umap_dims: int = 10, min_samples: int | None = None,
                   hdbscan_jobs: int = 1, silhouette_enabled: bool = False,
                   seed: int = _DEFAULT_SEED, use_cache: bool = True):
    print("\n" + "=" * 60)
    print("VIEB — UMAP n_neighbors Sweep")
    print(f"  umap_dims={umap_dims}  min_samples={'auto' if min_samples is None else min_samples}")
    print("=" * 60)

    mcs = rec_mcs if rec_mcs else 500
    print(f"Using HDBSCAN min_cluster_size={mcs} (from previous sweep or default)")

    print("\nLoading features (50k-frame sample for UMAP sweep)…")
    scaled = _load_or_build_scaled(max_frames=50_000, seed=seed, use_cache=use_cache)
    print(f"  Standardized matrix: {scaled.shape[0]:,} frames × {scaled.shape[1]} features")

    umap_rows = []
    for nn in _UMAP_NEIGHBORS:
        print(f"\n  n_neighbors={nn}…")
        emb = _get_umap_embedding(scaled, max_fit_frames=50_000, n_neighbors=nn,
                                  n_components=umap_dims, max_frames=50_000,
                                  seed=seed, use_cache=use_cache)
        labels, n_cls, noise_frac, dom_frac, mean_size, elapsed = _run_hdbscan(
            emb, mcs, min_samples=min_samples, core_dist_n_jobs=hdbscan_jobs
        )
        sil = _silhouette(emb, labels, seed=seed) if silhouette_enabled else float("nan")
        sil_str = f"{sil:.3f}" if not math.isnan(sil) else "n/a"
        print(f"    n_clusters={n_cls}, dom={dom_frac*100:.1f}%, noise={noise_frac*100:.1f}%, "
              f"sil={sil_str}, time={elapsed:.1f}s")
        umap_rows.append({
            "n_neighbors": nn,
            "mcs": mcs,
            "n_clusters": n_cls,
            "noise_frac": noise_frac,
            "dom_frac": dom_frac,
            "mean_size": mean_size,
            "silhouette": sil,
            "elapsed": elapsed,
        })

    print("\n" + "=" * 60)
    print("UMAP Sweep Results:")
    print(f"{'n_neighbors':>11}  {'n_clusters':>10}  {'noise%':>7}  {'dom%':>6}  {'silhouette':>11}")
    print("-" * 55)
    for r in umap_rows:
        sil_s = f"{r['silhouette']:11.3f}" if not math.isnan(r["silhouette"]) else "        n/a"
        print(f"{r['n_neighbors']:>11}  {r['n_clusters']:>10}  "
              f"{r['noise_frac']*100:>6.1f}%  {r['dom_frac']*100:>5.1f}%  {sil_s}")

    candidates = [r for r in umap_rows if 8 <= r["n_clusters"] <= 25]
    if not candidates:
        candidates = umap_rows
    best = min(candidates, key=lambda r: r["dom_frac"])
    print(f"\nBEST UMAP setting: n_neighbors={best['n_neighbors']}")
    print(f"  -> n_clusters={best['n_clusters']}, dominant={best['dom_frac']*100:.1f}%")
    effective_ms = _effective_min_samples(mcs, min_samples)
    print(
        "\nRun: python compare.py --cluster"
        f" --umap-dims {umap_dims}"
        f" --min-cluster-size {mcs}"
        f" --hdbscan-min-samples {effective_ms}"
        " --hdbscan-sample 300000"
    )
    print("=" * 60)

    os.makedirs(_DIAG_DIR, exist_ok=True)
    for csv_path in (
        "results/shared/umap_diagnostic.csv",
        os.path.join(_DIAG_DIR, "umap_diagnostic.csv"),
    ):
        pd.DataFrame(umap_rows).to_csv(csv_path, index=False)
        print(f"Table saved -> {csv_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--umap-sweep", action="store_true",
                        help="Also test different UMAP n_neighbors values (50k-frame sample)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true",
                      help=f"Use quick diagnostics sample (default, max {_QUICK_MAX_FRAMES:,} frames)")
    mode.add_argument("--full", action="store_true",
                      help=f"Use full diagnostics sample (max {_FULL_MAX_FRAMES:,} frames)")
    parser.add_argument("--umap-dims", type=int, default=10,
                        help="UMAP n_components for the diagnostic embedding (default: 10). "
                             "Use 2 for visualization checks; try 5–10 for clustering.")
    parser.add_argument("--n-neighbors", type=int, default=30,
                        help="UMAP n_neighbors for the diagnostic embedding (default: 30)")
    parser.add_argument("--min-samples", type=int, default=None,
                        help="Fix HDBSCAN min_samples across the sweep (default: equals mcs). "
                             "Try --min-samples 5 to cut noise without growing clusters.")
    parser.add_argument("--min-cluster-sizes", type=str, default=None,
                        help="Comma-separated list of min_cluster_size values to test "
                             f"(default: {','.join(map(str, _DEFAULT_MCS))})")
    parser.add_argument("--max-frames", type=int, default=None,
                        help=f"Subsample this many frames for the diagnostic sweep "
                             f"(default quick: {_QUICK_MAX_FRAMES:,}; full: {_FULL_MAX_FRAMES:,}). "
                             "Use 0 for all frames.")
    parser.add_argument("--hdbscan-jobs", type=int, default=1,
                        help="Parallel jobs for HDBSCAN core-distance computation "
                             "(default: 1, lower memory; raise only if you have RAM headroom).")
    sil_group = parser.add_mutually_exclusive_group()
    sil_group.add_argument("--silhouette", action="store_true",
                           help="Compute silhouette scores (slower)")
    sil_group.add_argument("--no-silhouette", action="store_true",
                           help="Skip silhouette scores (default)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable diagnostics .npz cache under results/diagnostics/cache/")
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED,
                        help=f"Random seed for frame sampling (default: {_DEFAULT_SEED})")
    args = parser.parse_args()

    mcs_list = _DEFAULT_MCS
    if args.min_cluster_sizes:
        try:
            mcs_list = [int(x.strip()) for x in args.min_cluster_sizes.split(",") if x.strip()]
        except ValueError:
            sys.exit("--min-cluster-sizes must be a comma-separated list of integers")

    if args.max_frames is not None:
        max_frames = args.max_frames if args.max_frames > 0 else None
    elif args.full:
        max_frames = _FULL_MAX_FRAMES
    else:
        max_frames = _QUICK_MAX_FRAMES

    silhouette_enabled = bool(args.silhouette)

    rec = cmd_sweep(mcs_list, umap_dims=args.umap_dims, min_samples=args.min_samples,
                    max_frames=max_frames, hdbscan_jobs=args.hdbscan_jobs,
                    silhouette_enabled=silhouette_enabled,
                    n_neighbors=args.n_neighbors, seed=args.seed,
                    use_cache=not args.no_cache)
    rec_mcs = rec["mcs"] if rec else None

    if args.umap_sweep:
        cmd_umap_sweep(rec_mcs, umap_dims=args.umap_dims, min_samples=args.min_samples,
                       hdbscan_jobs=args.hdbscan_jobs,
                       silhouette_enabled=silhouette_enabled,
                       seed=args.seed, use_cache=not args.no_cache)


if __name__ == "__main__":
    main()
