"""Benchmark v2's representation against v1's default feature set.

Same project, same HDBSCAN parameters, reporting largest_state_frac, n_states
and state_entropy side by side. **No winner is declared here** -- the harness
prints both columns and the caveats that qualify them, and stops.

Three things keep the comparison honest:

  * Both arms are scored by `representation.metrics.cluster_metrics`. v1's
    stored diagnostics are never read, because v1's on-disk `state_entropy` is
    normalised and its state fractions are taken over total frames including
    noise -- scoring the arms differently would let a definitional gap look
    like a result.
  * The architectural asymmetry is printed, not hidden: v1 reduces with UMAP
    before HDBSCAN, v2 clusters the delay embedding directly. "Same HDBSCAN
    parameters" across different dimensionalities is a convention, not a
    neutral fact.
  * The v2 arm needs continuous per-frame pose. If none is present the harness
    says so and exits without inventing a comparison.

Usage:
    python -m benchmark.compare_v1_v2 --project /path/to/project
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from representation.metrics import cluster_metrics, speed_diagnostics  # noqa: E402
from representation.pipeline import run as run_v2  # noqa: E402

POSE_PATTERNS = ("*.h5", "*DLC*.csv", "*filtered*.csv")


def find_pose_files(project_dir):
    """Locate continuous per-frame pose output for the v2 arm."""
    roots = [
        os.path.join(project_dir, "results", "pose"),
        os.path.join(project_dir, "pose"),
        os.path.join(project_dir, "raw_videos"),
        project_dir,
    ]
    found = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for pattern in POSE_PATTERNS:
            found.extend(glob.glob(os.path.join(root, pattern)))
    return sorted(set(found))


def load_v1_labels(project_dir):
    """Read v1's per-recording cluster labels from results/shared/."""
    shared = os.path.join(project_dir, "results", "shared")
    paths = sorted(glob.glob(os.path.join(shared, "*_labels.npy")))
    if not paths:
        return None
    return np.concatenate([np.load(p) for p in paths])


def load_v1_features(project_dir):
    """Read v1's extracted feature matrices, one per recording."""
    feat_dir = os.path.join(project_dir, "results", "features")
    paths = sorted(glob.glob(os.path.join(feat_dir, "*_features.npy")))
    return [np.load(p) for p in paths] if paths else None


def run_v1_arm(project_dir, min_cluster_size, min_samples, recluster=False):
    """Score v1's arm.

    By default this reuses v1's existing labels, which is what "v1's default
    feature set" means in practice. `recluster` instead re-runs HDBSCAN on v1's
    features with the v2 parameters -- a fairer parameter match, but no longer
    v1's actual pipeline (it skips v1's UMAP reduction and HMM smoothing).
    """
    if recluster:
        features = load_v1_features(project_dir)
        if not features:
            return None, "no v1 features found in results/features/"
        from representation.cluster import cluster

        pooled = np.concatenate(features, axis=0)
        pooled = (pooled - pooled.mean(0)) / (pooled.std(0) + 1e-9)
        labels, _ = cluster(pooled, min_cluster_size, min_samples)
        return {"labels": labels, "scores": pooled,
                "source": "re-clustered v1 features"}, None

    labels = load_v1_labels(project_dir)
    if labels is None:
        return None, "no v1 labels found in results/shared/"
    return {"labels": labels, "scores": None,
            "source": "v1 stored labels (UMAP + HDBSCAN + HMM)"}, None


def run_v2_arm(pose_files, min_cluster_size, min_samples, n_lags, lag_stride,
               latent_method="pca"):
    """Load pose and run the v2 pipeline with the given latent space."""
    from representation.pose_loader import load_sessions

    sessions, bodyparts, _ = load_sessions(pose_files)
    if not sessions:
        return None, "no pose sessions loaded"

    result = run_v2(sessions, bodyparts=bodyparts, n_lags=n_lags,
                    lag_stride=lag_stride, min_cluster_size=min_cluster_size,
                    min_samples=min_samples, latent_method=latent_method)
    return result, None


def format_report(columns, params, notes):
    """Render every arm side by side. States facts, draws no conclusion.

    `columns` is an ordered mapping of arm name -> metrics dict (or None).
    """
    lines = []
    add = lines.append
    names = list(columns)
    width = 18

    add("=" * 78)
    add("VIEB representation benchmark -- v1 features vs v2 latent spaces")
    add("=" * 78)
    add("")
    add("Parameters (identical across arms):")
    for k, v in params.items():
        add(f"  {k:<22} {v}")
    add("")
    add("ARCHITECTURAL ASYMMETRY -- read before comparing:")
    add("  v1        : 91 engineered features -> UMAP -> HDBSCAN (+ HMM)")
    add("  v2-PCA    : aligned pose -> pooled PCA -> delay embed -> HDBSCAN")
    add("  v2-diffusion: same, with a diffusion map in place of PCA")
    add("  Identical min_cluster_size/min_samples across different")
    add("  dimensionalities is a convention, not a neutral comparison.")
    add("")

    header = f"{'metric':<28}" + "".join(f"{n:>{width}}" for n in names)
    add(header)
    add("-" * len(header))
    for label, key, conv in [
        ("n_states", "n_states", None),
        ("noise_frac", "noise_frac", None),
        ("largest_state_frac", "largest_state_frac", "v1_convention"),
        ("state_entropy", "state_entropy", "v1_convention"),
        ("largest_state_frac (clean)", "largest_state_frac", "clustered_only"),
        ("state_entropy (clean)", "state_entropy", "clustered_only"),
    ]:
        row = "".join(f"{_fmt(_get(columns[n], key, conv)):>{width}}"
                      for n in names)
        add(f"{label:<28}{row}")

    add("")
    add("All columns are computed by the same function. v1's stored")
    add("diagnostics are not read: its on-disk state_entropy is normalised")
    add("and its fractions are over total frames including noise.")
    add("Diffusion distances additionally depend on epsilon, alpha and t, so")
    add("they are not comparable across different diffusion settings.")

    if notes:
        add("")
        add("Notes:")
        for n in notes:
            add(f"  - {n}")

    add("")
    add("No winner is declared. Interpretation is left to the reader.")
    add("=" * 78)
    return "\n".join(lines)


def _get(metrics, key, convention):
    if metrics is None:
        return None
    return metrics[convention][key] if convention else metrics.get(key)


def _fmt(v):
    if v is None:
        return "n/a"
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", default=".", help="project directory")
    ap.add_argument("--min-cluster-size", type=int, default=50)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument("--n-lags", type=int, default=4)
    ap.add_argument("--lag-stride", type=int, default=2)
    ap.add_argument("--recluster-v1", action="store_true",
                    help="re-cluster v1 features with the v2 parameters "
                         "instead of reusing v1's stored labels")
    ap.add_argument("--json", help="also write the results as JSON")
    args = ap.parse_args(argv)

    notes = []
    params = {
        "min_cluster_size": args.min_cluster_size,
        "min_samples": args.min_samples,
        "n_lags": args.n_lags,
        "lag_stride": args.lag_stride,
        "window_frames": args.n_lags * args.lag_stride + 1,
    }

    v1_result, v1_err = run_v1_arm(args.project, args.min_cluster_size,
                                   args.min_samples, args.recluster_v1)
    v1_metrics = None
    if v1_err:
        notes.append(f"v1 arm unavailable: {v1_err}")
    else:
        v1_metrics = cluster_metrics(v1_result["labels"])
        notes.append(f"v1 source: {v1_result['source']}")

    pose_files = find_pose_files(args.project)
    columns = {"v1": v1_metrics, "v2-PCA": None, "v2-diffusion": None}

    if not pose_files:
        notes.append(
            "both v2 arms SKIPPED: no continuous per-frame pose found. The v2 "
            "pipeline starts from raw keypoints, so it cannot run on v1's "
            "derived features. Run DLC --analyze to produce pose, then re-run."
        )
    else:
        for column, method in (("v2-PCA", "pca"),
                               ("v2-diffusion", "diffusion")):
            result, err = run_v2_arm(pose_files, args.min_cluster_size,
                                     args.min_samples, args.n_lags,
                                     args.lag_stride, latent_method=method)
            if err:
                notes.append(f"{column} unavailable: {err}")
                continue
            columns[column] = result["metrics"]
            latent = result["report"]["latent"]
            notes.append(f"{column}: {latent['n_components']} components")
            if method == "diffusion":
                notes.append(
                    f"{column}: epsilon={latent['epsilon']:.4g}, "
                    f"alpha={latent['alpha']}, t={latent['diffusion_time']}, "
                    f"{latent['n_landmarks']} landmarks")
            speed = result["speed_diagnostics"]
            if speed and speed.get("noise_speed_ratio"):
                notes.append(
                    f"{column}: noise/clustered speed ratio "
                    f"{speed['noise_speed_ratio']:.2f} "
                    f"(>1 supports the density-duration account)")

    print(format_report(columns, params, notes))

    if args.json:
        with open(args.json, "w") as fh:
            json.dump({"params": params, "columns": columns, "notes": notes},
                      fh, indent=2)

    # Missing pose is the expected state today, not an error.
    return 0


if __name__ == "__main__":
    sys.exit(main())
