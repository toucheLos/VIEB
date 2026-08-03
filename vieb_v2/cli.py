"""VIEB v2 representation pipeline -- command line interface.

Usage:
    python -m vieb_v2.cli doctor                              # environment / GPU preflight
    python -m vieb_v2.cli tune    --pose results/pose         # suggest --lag-stride
    python -m vieb_v2.cli run     --pose results/pose         # align + pca + embed + cluster
    python -m vieb_v2.cli align   --pose results/pose         # or run the stages separately
    python -m vieb_v2.cli pca
    python -m vieb_v2.cli embed   --n-lags 4 --lag-stride 2
    python -m vieb_v2.cli cluster --min-cluster-size 50 --gpu on
    python -m vieb_v2.cli sweep   --min-cluster-sizes 25,50,100,200
    python -m vieb_v2.cli benchmark --project /path/to/vieb

Each stage checkpoints to --out as .npz, so sweeping HDBSCAN parameters does not
redo alignment and PCA over millions of frames.

GPU (--gpu auto|on|off) applies to HDBSCAN only. The pooled PCA is a 14x14
eigenproblem and gains nothing from a device. "on" fails immediately if the GPU
backend is unusable rather than silently spending a batch allocation on CPU.

Exit codes:
    0  completed
    1  ran, but needs attention (no pose data, degenerate clustering)
    2  hard failure
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from representation import gpu, keypoints  # noqa: E402
from representation import tune as tune_mod  # noqa: E402
from representation.align import align_all, null_leakage  # noqa: E402
from representation.cluster import cluster as run_cluster  # noqa: E402
from representation.cluster import seed_stability  # noqa: E402
from representation.embed import embed_all  # noqa: E402
from representation.metrics import cluster_metrics, speed_diagnostics  # noqa: E402
from representation.pooled_pca import PooledPCA  # noqa: E402
from representation.pose_loader import find_pose_files, load_sessions  # noqa: E402

ALIGNED, SCORES, EMBEDDED, LABELS = (
    "aligned.npz", "scores.npz", "embedded.npz", "labels.npz")

OK, NEEDS_ATTENTION, FAILED = 0, 1, 2


# ---------------------------------------------------------------- utilities

def _log(msg=""):
    print(f"[vieb2] {msg}" if msg else "", flush=True)


def _save(path, arrays, meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    np.savez_compressed(path, meta=json.dumps(_jsonable(meta)), **arrays)
    _log(f"wrote {path}")


def _load(path):
    if not os.path.exists(path):
        raise _Attention(
            f"missing {path} -- run the earlier stage first, or use `run`")
    data = np.load(path, allow_pickle=False)
    return data, json.loads(str(data["meta"]))


class _Attention(Exception):
    """Recoverable: the command could not proceed but nothing is broken."""


def _pack(sessions):
    """Store ragged per-recording arrays without losing the boundaries.

    A flattened blob would let a later stage delay-embed across two animals,
    so the lengths travel with the data.
    """
    return {"stacked": np.concatenate(sessions, axis=0),
            "lengths": np.array([len(s) for s in sessions])}


def _unpack(data):
    return np.split(data["stacked"], np.cumsum(data["lengths"])[:-1])


def _load_pose(args):
    paths = find_pose_files(args.pose)
    if not paths:
        raise _Attention(
            f"no pose files under {args.pose!r}.\n"
            f"  The v2 pipeline starts from continuous per-frame keypoints,\n"
            f"  so DLC must run first:\n"
            f"      python setup_dlc_training.py --analyze"
        )
    if args.limit:
        paths = paths[: args.limit]

    _log(f"loading {len(paths)} pose file(s) from {args.pose}")
    sessions, bodyparts, skipped = load_sessions(paths, max_gap=args.max_gap)
    for path, why in skipped:
        _log(f"  skipped {os.path.basename(path)}: {why}")
    if not sessions:
        raise _Attention("no usable recordings after loading")
    _log(f"{len(sessions)} recording(s), "
         f"{sum(len(p) for p, _ in sessions):,} frames")
    _log(f"bodyparts: {bodyparts}")
    return sessions, bodyparts


# ------------------------------------------------------------------ commands

def cmd_doctor(args):
    """Preflight, so a batch job is not queued against a broken environment."""
    report = gpu.report()

    _log(f"python           {sys.version.split()[0]}")
    _log(f"numpy            {np.__version__}")
    for name in ("hdbscan", "cuml", "cupy", "pandas", "h5py", "tables"):
        try:
            mod = __import__(name)
            _log(f"{name:<16} {getattr(mod, '__version__', 'ok')}")
        except Exception:
            _log(f"{name:<16} MISSING")
    _log()

    _log(f"GPU device       {report['device'] or 'none visible'}")
    _log(f"HDBSCAN on GPU   {report['hdbscan_gpu']}"
         + (f"  ({report['hdbscan_gpu_reason']})"
            if report["hdbscan_gpu_reason"] else ""))
    _log(f"cupy linalg      {report['cupy_linalg']}"
         + (f"  ({report['cupy_linalg_reason']})"
            if report["cupy_linalg_reason"] else ""))
    if report["forced_cpu"]:
        _log("VIEB_FORCE_CPU is set -- all GPU paths disabled")
    _log()

    _log("GPU is used for HDBSCAN only. The pooled PCA is a 14x14")
    _log("eigenproblem, so cupy is not needed by the pipeline; it is")
    _log("reported because a broken cupy signals a misconfigured CUDA env.")
    _log("cuml HDBSCAN is a separate implementation from the CPU package.")
    _log("Measured on a 3568x35 embedding the two agree on the partition")
    _log("(adjusted Rand index 1.0000) but number the clusters differently,")
    _log("so a state id is only meaningful within one run. Untested at the")
    _log("~2.3M-frame scale of a full project. The backend used is recorded.")

    if report["loader_hint"]:
        _log()
        for line in report["loader_hint"].splitlines():
            _log(line)

    if args.json:
        print(json.dumps(report, indent=2))
    return OK


def cmd_align(args):
    sessions, bodyparts = _load_pose(args)
    kept = [b for b in bodyparts
            if str(b).strip().lower() not in keypoints.DROPPED_BODYPARTS]
    dropped = [b for b in bodyparts if b not in kept]
    selected = [keypoints.select(p, c, bodyparts)[:2] for p, c in sessions]
    _log(f"dropped keypoints {dropped} -> K={len(kept)}, D={2 * len(kept)}, "
         f"rank ceiling {2 * len(kept) - 3}")

    t0 = time.time()
    aligned, reference = align_all(selected)
    flat = np.concatenate([a.reshape(a.shape[0], -1) for a in aligned])
    leak = null_leakage(flat)
    _log(f"aligned {len(aligned)} recording(s) in {time.time() - t0:.1f}s")
    _log(f"null-direction leakage {leak:.3%} -- variance that confidence "
         f"weighting put back into the nominally-null directions")
    if leak > 0.05:
        _log("  WARNING: high leakage means the PCs are partly tracking "
             "likelihood noise rather than pose")

    _save(os.path.join(args.out, ALIGNED),
          {**_pack([a.reshape(a.shape[0], -1) for a in aligned]),
           "reference": reference},
          {"bodyparts": kept, "dropped": dropped, "n_keypoints": len(kept),
           "expected_rank": 2 * len(kept) - 3, "null_leakage": leak})
    return OK


def cmd_pca(args):
    data, meta = _load(os.path.join(args.out, ALIGNED))
    aligned = _unpack(data)

    t0 = time.time()
    pca = PooledPCA(var_threshold=args.var_threshold).fit(aligned)
    scores = pca.transform_all(aligned)
    report = pca.spectrum_report()

    _log(f"pooled PCA over {sum(len(a) for a in aligned):,} frames from "
         f"{len(aligned)} recording(s) in {time.time() - t0:.1f}s")
    _log(f"  one basis for the whole project -- never per recording")
    _log(f"  q={report['n_components']} components "
         f"({report['explained_variance']:.1%} variance)")

    nonzero, expected = report["n_nonzero_directions"], meta["expected_rank"]
    _log(f"  {nonzero} non-null directions, {expected} expected "
         f"(rank 2K-3)")
    if nonzero > expected:
        # Uniform weights give exactly 2K-3; per-frame confidence weighting
        # partially refills the rotation null, so the extra directions hold
        # likelihood fluctuation rather than pose.
        _log(f"  the extra {nonzero - expected} carry confidence-weighting "
             f"noise, not pose (leakage {meta['null_leakage']:.3%})")
        if report["n_components"] > expected:
            _log(f"  WARNING: q={report['n_components']} exceeds the rank "
                 f"ceiling -- lower --var-threshold so noise directions are "
                 f"not retained")

    _save(os.path.join(args.out, SCORES),
          {**_pack(scores), "components": pca.components_, "mean": pca.mean_},
          {**meta, "pca": report})
    return OK


def cmd_embed(args):
    data, meta = _load(os.path.join(args.out, SCORES))
    scores = _unpack(data)

    embedded, index = embed_all(scores, args.n_lags, args.lag_stride)
    window = args.n_lags * args.lag_stride + 1
    if embedded.shape[0] == 0:
        raise _Attention(
            f"no frames survived a {window}-frame window -- "
            f"lower --n-lags or --lag-stride")

    dropped = sum(len(s) for s in scores) - embedded.shape[0]
    _log(f"delay embedding {embedded.shape[0]:,} x {embedded.shape[1]}")
    _log(f"  window {window} frames = {window / args.fps:.2f}s at {args.fps}fps")
    _log(f"  dropped {dropped:,} frames at recording boundaries "
         f"(lags never cross one)")

    _save(os.path.join(args.out, EMBEDDED),
          {"embedded": embedded, "index": index},
          {**meta, "n_lags": args.n_lags, "lag_stride": args.lag_stride,
           "fps": args.fps, "window_frames": window})
    return OK


def cmd_cluster(args):
    data, meta = _load(os.path.join(args.out, EMBEDDED))
    embedded, index = data["embedded"], data["index"]

    use_gpu = gpu.resolve(args.gpu)
    sample = None if args.hdbscan_sample <= 0 else args.hdbscan_sample
    if sample and embedded.shape[0] > sample:
        _log(f"fitting on a {sample:,}-point subsample, then labelling all "
             f"{embedded.shape[0]:,} by approximate_predict")

    t0 = time.time()
    labels, probs, backend = run_cluster(
        embedded, args.min_cluster_size, args.min_samples,
        use_gpu=use_gpu, return_backend=True, sample=sample,
    )
    _log(f"HDBSCAN [{backend}] on {embedded.shape[0]:,} points "
         f"in {time.time() - t0:.1f}s")

    metrics = cluster_metrics(labels)
    speed = speed_diagnostics(labels, embedded)
    _report_metrics(metrics, speed)

    stability = None
    if args.stability:
        stability = seed_stability(embedded, args.min_cluster_size,
                                   args.min_samples, n_repeats=args.stability)
        _log(f"seed stability n_states={stability['n_states']} "
             f"(mean {stability['mean']:.1f}, sd {stability['std']:.2f})")
        _log("  wide spread means states are found by luck, not structure")

    _save(os.path.join(args.out, LABELS),
          {"labels": labels, "probabilities": probs, "index": index},
          {**meta, "min_cluster_size": args.min_cluster_size,
           "min_samples": args.min_samples, "hdbscan_backend": backend,
           "hdbscan_sample": sample, "metrics": metrics,
           "speed": speed, "seed_stability": stability})

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(_jsonable({"metrics": metrics, "speed": speed,
                                 "backend": backend}), fh, indent=2)
        _log(f"wrote {args.json}")

    if metrics["n_states"] < 2:
        _log("only one state found -- try a smaller --min-cluster-size")
        return NEEDS_ATTENTION
    return OK


def cmd_run(args):
    for step in (cmd_align, cmd_pca, cmd_embed, cmd_cluster):
        code = step(args)
        if code:
            return code
    _log(f"done -- artifacts in {args.out}")
    return OK


def cmd_tune(args):
    scores_path = os.path.join(args.out, SCORES)
    if not os.path.exists(scores_path):
        _log("no scores checkpoint found; running align + pca first")
        cmd_align(args)
        cmd_pca(args)

    data, _ = _load(scores_path)
    result = tune_mod.suggest(_unpack(data), fps=args.fps,
                              max_lag=args.max_lag, n_lags=args.n_lags)

    _log(f"tau from autocorrelation (1/e crossing) : "
         f"{result['tau_autocorrelation']}")
    _log(f"tau from mutual information (first min) : "
         f"{result['tau_mutual_information']}")
    _log(f"  {result['estimator_agreement']}")
    _log()
    _log(f"suggested:  --n-lags {result['n_lags']} "
         f"--lag-stride {result['tau_suggested']}")
    _log(f"  window {result['window_frames']} frames = "
         f"{result['window_seconds']:.2f}s at {result['fps']}fps")
    _log()
    _log("reference timescales: stride 0.1-0.2s, rear ~0.5s, freeze seconds")
    _log("avoid a stride commensurate with a periodic behavior's period,")
    _log("where the lag block degenerates into a copy of the current frame")

    if args.json:
        print(json.dumps(result, indent=2))
    return OK


def cmd_sweep(args):
    """Sweep min_cluster_size against one embedding.

    A single parameter point is not evidence: equal HDBSCAN parameters across
    representations of different dimensionality is a convention, not a fair
    match, so the honest comparison is a curve.
    """
    data, _ = _load(os.path.join(args.out, EMBEDDED))
    embedded = data["embedded"]
    use_gpu = gpu.resolve(args.gpu)
    sample = None if args.hdbscan_sample <= 0 else args.hdbscan_sample

    rows = []
    print(f"\n{'min_cluster_size':>17} {'n_states':>9} {'noise':>8} "
          f"{'largest':>8} {'entropy':>8} {'backend':>8}")
    print("-" * 62)
    for size in [int(s) for s in str(args.min_cluster_sizes).split(",")]:
        labels, _, backend = run_cluster(embedded, size, args.min_samples,
                                         use_gpu=use_gpu, return_backend=True,
                                         sample=sample)
        m = cluster_metrics(labels)
        clean = m["clustered_only"]
        print(f"{size:>17} {m['n_states']:>9} {m['noise_frac']:>7.1%} "
              f"{clean['largest_state_frac']:>7.1%} "
              f"{clean['state_entropy']:>8.3f} {backend:>8}")
        rows.append({"min_cluster_size": size, "backend": backend, **m})
    print()

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(_jsonable(rows), fh, indent=2)
        _log(f"wrote {args.json}")
    return OK


def cmd_benchmark(args):
    from benchmark.compare_v1_v2 import main as bench_main

    argv = ["--project", args.project,
            "--min-cluster-size", str(args.min_cluster_size),
            "--n-lags", str(args.n_lags),
            "--lag-stride", str(args.lag_stride)]
    if args.json:
        argv += ["--json", args.json]
    return bench_main(argv)


# ------------------------------------------------------------------ reporting

def _report_metrics(metrics, speed):
    v1, clean = metrics["v1_convention"], metrics["clustered_only"]
    _log(f"n_states={metrics['n_states']}  "
         f"noise_frac={metrics['noise_frac']:.4f}")
    _log(f"  largest_state_frac  {v1['largest_state_frac']:.4f} (v1 conv.)   "
         f"{clean['largest_state_frac']:.4f} (clustered only)")
    _log(f"  state_entropy       {v1['state_entropy']:.4f} (v1 conv.)   "
         f"{clean['state_entropy']:.4f} (clustered only)")

    ratio = speed.get("noise_speed_ratio") if speed else None
    if ratio:
        _log(f"  noise/clustered speed ratio {ratio:.2f}")
        _log("    >1 supports the density-duration account: the frames that")
        _log("    failed to cluster are the fast ones. Near 1 means that")
        _log("    account does not explain this result.")


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    return obj


# --------------------------------------------------------------------- parser

def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m vieb_v2.cli",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    def pose_args(sp):
        sp.add_argument("--pose", default="results/pose",
                        help="directory holding DLC pose output (default: %(default)s)")
        sp.add_argument("--limit", type=int, default=None,
                        help="use only the first N recordings (smoke test)")
        sp.add_argument("--max-gap", type=int, default=None,
                        help="longest run of missing frames to interpolate")

    def out_args(sp):
        sp.add_argument("--out", default="results/v2",
                        help="artifact directory (default: %(default)s)")

    def gpu_args(sp):
        sp.add_argument("--gpu", default="auto", choices=["auto", "on", "off"],
                        help="GPU for HDBSCAN; 'on' fails if unusable")

    def embed_args(sp):
        sp.add_argument("--n-lags", type=int, default=4)
        sp.add_argument("--lag-stride", type=int, default=2)
        sp.add_argument("--fps", type=float, default=30.0)

    def cluster_args(sp):
        sp.add_argument("--min-cluster-size", type=int, default=50)
        sp.add_argument("--min-samples", type=int, default=None)
        sp.add_argument("--hdbscan-sample", type=int, default=300_000,
                        help="fit on at most N points, then label the rest "
                             "by approximate_predict; 0 disables")
        sp.add_argument("--stability", type=int, default=0, metavar="N",
                        help="re-cluster N subsamples to test state stability")
        sp.add_argument("--json", default=None, help="write metrics as JSON")

    sp = sub.add_parser("doctor", help="environment and GPU preflight")
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("align", help="drop noisy keypoints, align pose")
    pose_args(sp), out_args(sp), gpu_args(sp)
    sp.set_defaults(func=cmd_align)

    sp = sub.add_parser("pca", help="pooled PCA across all recordings")
    out_args(sp), gpu_args(sp)
    sp.add_argument("--var-threshold", type=float, default=0.95)
    sp.set_defaults(func=cmd_pca)

    sp = sub.add_parser("embed", help="delay-embed the PC scores")
    out_args(sp), gpu_args(sp), embed_args(sp)
    sp.set_defaults(func=cmd_embed)

    sp = sub.add_parser("cluster", help="HDBSCAN on the embedding")
    out_args(sp), gpu_args(sp), cluster_args(sp)
    sp.set_defaults(func=cmd_cluster)

    sp = sub.add_parser("run", help="align + pca + embed + cluster")
    pose_args(sp), out_args(sp), gpu_args(sp), embed_args(sp), cluster_args(sp)
    sp.add_argument("--var-threshold", type=float, default=0.95)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("tune", help="suggest --lag-stride from the data")
    pose_args(sp), out_args(sp), gpu_args(sp)
    sp.add_argument("--var-threshold", type=float, default=0.95)
    sp.add_argument("--n-lags", type=int, default=4)
    sp.add_argument("--fps", type=float, default=30.0)
    sp.add_argument("--max-lag", type=int, default=60)
    sp.add_argument("--json", action="store_true")
    sp.set_defaults(func=cmd_tune)

    sp = sub.add_parser("sweep", help="sweep min_cluster_size on one embedding")
    out_args(sp), gpu_args(sp)
    sp.add_argument("--min-cluster-sizes", default="25,50,100,200,400")
    sp.add_argument("--min-samples", type=int, default=None)
    sp.add_argument("--hdbscan-sample", type=int, default=300_000)
    sp.add_argument("--json", default=None)
    sp.set_defaults(func=cmd_sweep)

    sp = sub.add_parser("benchmark", help="compare against v1")
    sp.add_argument("--project", default=".")
    sp.add_argument("--min-cluster-size", type=int, default=50)
    sp.add_argument("--n-lags", type=int, default=4)
    sp.add_argument("--lag-stride", type=int, default=2)
    sp.add_argument("--json", default=None)
    sp.set_defaults(func=cmd_benchmark)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except _Attention as exc:
        _log(str(exc))
        return NEEDS_ATTENTION
    except RuntimeError as exc:     # e.g. --gpu on with no usable device
        _log(f"error: {exc}")
        return FAILED


if __name__ == "__main__":
    sys.exit(main())
