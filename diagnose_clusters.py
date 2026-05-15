#!/usr/bin/env python3
"""
diagnose_clusters.py — HDBSCAN (and optionally UMAP) parameter sweep for VIEB.

Usage
-----
    python diagnose_clusters.py
        Sweep min_cluster_size on the existing 10-D UMAP embedding.

    python diagnose_clusters.py --umap-dims 2 --min-samples 5
        Re-embed to 2-D, sweep MCS with min_samples fixed at 5.

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
_DEFAULT_MAX_FRAMES = 1_000_000
_UMAP_NEIGHBORS = [10, 15, 30, 50]


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

    stems = sorted(index.keys())
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
                        n_components: int = 10, n_neighbors: int = 30):
    """
    Return UMAP embedding.

    Priority:
    1. Load existing results/shared/umap_reducer.pkl (when n_neighbors==30 and load succeeds).
    2. cuML GPU UMAP if available (Linux + CUDA).
    3. CPU umap-learn with n_jobs=-1 (all cores, no fixed seed so parallelism is enabled).
    """
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
        reducer.fit(fit_data)
        emb = reducer.transform(scaled)

    if hasattr(emb, "to_numpy"):
        emb = emb.to_numpy()
    elif hasattr(emb, "get"):
        emb = emb.get()
    return np.asarray(emb, dtype=np.float32)


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
              max_frames: int = _DEFAULT_MAX_FRAMES, hdbscan_jobs: int = 1):
    print("=" * 60)
    print("VIEB — Cluster Diagnostic (HDBSCAN sweep)")
    print(f"  umap_dims={umap_dims}  min_samples={'auto (=mcs)' if min_samples is None else min_samples}")
    print(f"  hdbscan_jobs={hdbscan_jobs}")
    max_frames_label = "all" if max_frames is None else f"{max_frames:,}"
    print(f"  max_frames={max_frames_label} (subsample for speed/memory)")
    print("=" * 60)

    print("\nLoading features…")
    pooled_raw = _load_pooled_features(max_frames=max_frames)
    print(f"  {len(pooled_raw):,} frames × {pooled_raw.shape[1]} features loaded")

    print("\nStandardizing features…")
    scaled = _standardize(pooled_raw)

    print(f"\nBuilding UMAP embedding (n_components={umap_dims})…")
    emb = _get_umap_embedding(scaled, n_components=umap_dims)
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

        sil = _silhouette(emb, labels)
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
    print("\n" + "=" * 60)
    if rec:
        ms_flag = f" --hdbscan-min-samples {min_samples}" if min_samples is not None else ""
        dims_flag = f" --umap-dims {umap_dims}" if umap_dims != 10 else ""
        print(f"RECOMMENDATION: min_cluster_size = {rec['mcs']}")
        print(f"  -> n_clusters={rec['n_clusters']}, dominant={rec['dom_frac']*100:.1f}%, "
              f"noise={rec['noise_frac']*100:.1f}%")
        print(f"\nRun: python compare.py --cluster --min-cluster-size {rec['mcs']}{ms_flag}{dims_flag}")
    else:
        print("No ideal setting found. Review the table manually.")
    print("=" * 60)

    os.makedirs("results/shared", exist_ok=True)
    csv_path = "results/shared/cluster_diagnostic.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"\nTable saved -> {csv_path}")

    _save_plot(rows, "results/shared/cluster_diagnostic.png")
    return rec


# ---------------------------------------------------------------------------
# UMAP sweep
# ---------------------------------------------------------------------------

def cmd_umap_sweep(rec_mcs: int | None, umap_dims: int = 10, min_samples: int | None = None,
                   hdbscan_jobs: int = 1):
    print("\n" + "=" * 60)
    print("VIEB — UMAP n_neighbors Sweep")
    print(f"  umap_dims={umap_dims}  min_samples={'auto' if min_samples is None else min_samples}")
    print("=" * 60)

    mcs = rec_mcs if rec_mcs else 500
    print(f"Using HDBSCAN min_cluster_size={mcs} (from previous sweep or default)")

    print("\nLoading features (50k-frame sample for UMAP sweep)…")
    pooled_raw = _load_pooled_features(max_frames=50_000)
    print(f"  {len(pooled_raw):,} frames × {pooled_raw.shape[1]} features")

    print("\nStandardizing…")
    scaled = _standardize(pooled_raw)

    umap_rows = []
    for nn in _UMAP_NEIGHBORS:
        print(f"\n  n_neighbors={nn}…")
        emb = _get_umap_embedding(scaled, max_fit_frames=50_000, n_neighbors=nn,
                                  n_components=umap_dims)
        labels, n_cls, noise_frac, dom_frac, mean_size, elapsed = _run_hdbscan(
            emb, mcs, min_samples=min_samples, core_dist_n_jobs=hdbscan_jobs
        )
        sil = _silhouette(emb, labels)
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
    ms_flag = f" --hdbscan-min-samples {min_samples}" if min_samples is not None else ""
    dims_flag = f" --umap-dims {umap_dims}" if umap_dims != 10 else ""
    print(f"\nRun: python compare.py --cluster --min-cluster-size {mcs}"
          f" --umap-dims {best['n_neighbors']}{ms_flag}{dims_flag}")
    print("=" * 60)

    csv_path = "results/shared/umap_diagnostic.csv"
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
    parser.add_argument("--umap-dims", type=int, default=10,
                        help="UMAP n_components for the diagnostic embedding (default: 10). "
                             "Try 2 or 3 — density is more meaningful in low dimensions.")
    parser.add_argument("--min-samples", type=int, default=None,
                        help="Fix HDBSCAN min_samples across the sweep (default: equals mcs). "
                             "Try --min-samples 5 to cut noise without growing clusters.")
    parser.add_argument("--min-cluster-sizes", type=str, default=None,
                        help="Comma-separated list of min_cluster_size values to test "
                             f"(default: {','.join(map(str, _DEFAULT_MCS))})")
    parser.add_argument("--max-frames", type=int, default=_DEFAULT_MAX_FRAMES,
                        help=f"Subsample this many frames for the diagnostic sweep "
                             f"(default: {_DEFAULT_MAX_FRAMES:,}). Use 0 for all frames.")
    parser.add_argument("--hdbscan-jobs", type=int, default=1,
                        help="Parallel jobs for HDBSCAN core-distance computation "
                             "(default: 1, lower memory; raise only if you have RAM headroom).")
    args = parser.parse_args()

    mcs_list = _DEFAULT_MCS
    if args.min_cluster_sizes:
        try:
            mcs_list = [int(x.strip()) for x in args.min_cluster_sizes.split(",") if x.strip()]
        except ValueError:
            sys.exit("--min-cluster-sizes must be a comma-separated list of integers")

    max_frames = args.max_frames if args.max_frames > 0 else None

    rec = cmd_sweep(mcs_list, umap_dims=args.umap_dims, min_samples=args.min_samples,
                    max_frames=max_frames, hdbscan_jobs=args.hdbscan_jobs)
    rec_mcs = rec["mcs"] if rec else None

    if args.umap_sweep:
        cmd_umap_sweep(rec_mcs, umap_dims=args.umap_dims, min_samples=args.min_samples,
                       hdbscan_jobs=args.hdbscan_jobs)


if __name__ == "__main__":
    main()
