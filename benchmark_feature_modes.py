"""benchmark_feature_modes.py — Compare pose feature representations (Part E).

Runs the full pipeline (--extract, --cluster, --report) once per
--feature-mode against whichever project is currently active
(project_manager's active-project mechanism — see vieb_config.py), and
collects n_states, noise_fraction, mean_confidence, repeatability R,
modularity Q / possible-split-state count, and runtime/memory cost into one
comparison table.

Loops over feature modes only, NOT datasets — Luna vs Spence are just
"whichever project is currently active." Run this script once per project,
switching the active project (via the GUI's project selector or
project_manager) between runs, to build separate comparison tables for
each lab's data.

Usage
-----
    python benchmark_feature_modes.py
    python benchmark_feature_modes.py --modes default,shape_space
    python benchmark_feature_modes.py --skip-existing

This script does NOT declare a winner — it only produces the comparison
table in results/benchmark/feature_mode_comparison.csv. Which
representation to use is a scientific judgment call for the researcher,
not something to automate.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
ALL_MODES = ["default", "shape_space", "delay_embedding", "topological"]


def _results_dir() -> str:
    import vieb_config as _vc
    return _vc.get_results_dir()


def _project_name() -> str:
    try:
        import project_manager as _pm
        return _pm.get_active_project(ROOT).name
    except Exception:
        return "unknown_project"


def _mode_paths(mode: str) -> dict:
    """Reuse compare.py's own directory-isolation helpers (single source of truth)."""
    import compare as _compare
    return {
        "shared": _compare._shared_dir(mode),
        "diagnostics": _compare._diagnostics_dir(mode),
    }


def _run_stage(stage_flag: str, mode: str, extra_args: list[str]) -> float:
    """Run one pipeline stage as a subprocess (matches how a user would
    actually invoke it), return wall-clock seconds."""
    cmd = [sys.executable, os.path.join(ROOT, "compare.py"), stage_flag,
           "--feature-mode", mode] + extra_args
    print(f"  $ {' '.join(cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        raise RuntimeError(f"{stage_flag} --feature-mode {mode} failed (exit {result.returncode})")
    return elapsed


def _peak_rss_mb() -> float:
    """Peak RSS of this process's completed children, in MB (Linux: KB units)."""
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    if sys.platform == "darwin":
        return usage.ru_maxrss / (1024 * 1024)
    return usage.ru_maxrss / 1024


def benchmark_one_mode(mode: str, skip_existing: bool = False) -> dict:
    print(f"\n=== feature_mode: {mode} ===")
    paths = _mode_paths(mode)

    cluster_info_path = os.path.join(paths["shared"], "cluster_info.json")
    if skip_existing and os.path.exists(cluster_info_path):
        print(f"  Skipping (already exists: {cluster_info_path})")

    extract_t = _run_stage("--extract", mode, [])
    cluster_t = _run_stage("--cluster", mode, [])
    report_t = _run_stage("--report", mode, [])

    with open(cluster_info_path) as f:
        cluster_info = json.load(f)

    manifest_path = os.path.join(paths["shared"], "run_manifest.json")
    manifest = {}
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            manifest = json.load(f)

    validation_stats = {}
    vs_path = os.path.join(paths["diagnostics"], "validation_stats.json")
    if os.path.exists(vs_path):
        with open(vs_path) as f:
            validation_stats = json.load(f)

    repeatability = validation_stats.get("repeatability", {})
    modularity = validation_stats.get("modularity", {})

    return {
        "feature_mode": mode,
        "project_name": _project_name(),
        "n_states": cluster_info.get("n_clusters"),
        "noise_frac": manifest.get("noise_frac"),
        "mean_confidence": cluster_info.get("mean_confidence"),
        "repeatability_mean_R": repeatability.get("mean_R"),
        "modularity_Q": modularity.get("modularity_Q"),
        "n_possible_split_states": len(modularity.get("possible_split_states", []) or []),
        "extract_runtime_sec": round(extract_t, 2),
        "cluster_runtime_sec": round(cluster_t, 2),
        "report_runtime_sec": round(report_t, 2),
        "total_runtime_sec": round(extract_t + cluster_t + report_t, 2),
        "peak_rss_mb": round(_peak_rss_mb(), 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--modes", type=str, default=",".join(ALL_MODES),
                         help=f"Comma-separated feature modes to benchmark (default: all — {','.join(ALL_MODES)})")
    parser.add_argument("--skip-existing", action="store_true",
                         help="Print a note (but still re-run) when a mode's cluster_info.json already exists")
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = set(modes) - set(ALL_MODES)
    if unknown:
        sys.exit(f"Unknown feature mode(s): {sorted(unknown)}. Available: {ALL_MODES}")

    rows = []
    for mode in modes:
        try:
            rows.append(benchmark_one_mode(mode, skip_existing=args.skip_existing))
        except Exception as e:
            print(f"[error] feature_mode={mode} failed: {e}")
            rows.append({"feature_mode": mode, "project_name": _project_name(), "error": str(e)})

    out_dir = os.path.join(_results_dir(), "benchmark")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "feature_mode_comparison.csv")

    df = pd.DataFrame(rows)
    if os.path.exists(out_path):
        existing = pd.read_csv(out_path)
        existing = existing[~existing["feature_mode"].isin(df["feature_mode"])]
        df = pd.concat([existing, df], ignore_index=True)
    df.to_csv(out_path, index=False)

    print(f"\n=== Comparison table (results/benchmark/feature_mode_comparison.csv) ===")
    print(df.to_string(index=False))
    print("\nNo winner is declared — review the table and decide which representation "
          "to use for this project.")


if __name__ == "__main__":
    main()
