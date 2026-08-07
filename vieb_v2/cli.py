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
`doctor` reports whether a GPU works here and, from the driver version, which
pinned RAPIDS stack to install; `doctor --print-packages` emits that stack as
pip arguments, which is how hpc/install_gpu.sh builds its venv.

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

from representation import checkpoints, gpu, keypoints, koopman  # noqa: E402
from representation import run_registry  # noqa: E402
from representation import tune as tune_mod  # noqa: E402
from representation.align import align_all, null_leakage  # noqa: E402
from representation.cluster import cluster as run_cluster  # noqa: E402
from representation.cluster import seed_stability  # noqa: E402
from representation.embed import embed_all  # noqa: E402
from representation.metrics import cluster_metrics, speed_diagnostics  # noqa: E402
from representation.pipeline import LATENT_METHODS, make_latent  # noqa: E402
from representation.pose_loader import find_pose_files, load_sessions  # noqa: E402

ALIGNED = checkpoints.ALIGNED
SCORES = checkpoints.SCORES
EMBEDDED = checkpoints.EMBEDDED
LABELS = checkpoints.LABELS

OK, NEEDS_ATTENTION, FAILED = 0, 1, 2


# ---------------------------------------------------------------- utilities

def _log(msg=""):
    print(f"[vieb2] {msg}" if msg else "", flush=True)


def _save(path, arrays, meta):
    checkpoints.save(path, arrays, meta)
    _log(f"wrote {path}")


def _load(path):
    try:
        return checkpoints.load(path)
    except FileNotFoundError:
        raise _Attention(
            f"missing {path} -- run the earlier stage first, or use `run`")


class _Attention(Exception):
    """Recoverable: the command could not proceed but nothing is broken."""


_pack = checkpoints.pack
_unpack = checkpoints.unpack


def _load_pose(args):
    paths = find_pose_files(args.pose)
    if not paths:
        raise _Attention(
            f"no pose files under {args.pose!r}.\n"
            f"  The v2 pipeline starts from continuous per-frame keypoints,\n"
            f"  so DLC must run first:\n"
            f"      python setup_dlc_training.py --analyze"
        )

    # Deduplicate before anything else. DLC exports the same recording as both
    # .h5 and .csv, and treating those as two recordings double-counts them in
    # every pooled statistic downstream -- most damagingly in the stationary
    # measure of the transfer operator, which is literally an occupancy count.
    n_found = len(paths)
    paths, duplicates, ambiguous = recordings.dedupe(paths)
    if duplicates:
        _log(f"deduplicated {n_found} file(s) -> {len(paths)} recording(s) "
             f"({sum(len(v) for v in duplicates.values())} duplicate export(s) "
             f"dropped, .h5 preferred)")
    if ambiguous:
        _log(f"  WARNING: {len(ambiguous)} recording(s) had several files of "
             f"the same format; kept one by sort order: {ambiguous[:3]}")
    if args.limit:
        paths = paths[: args.limit]

    _log(f"loading {len(paths)} pose file(s) from {args.pose}")
    sessions, bodyparts, skipped, kept_paths = load_sessions(
        paths, max_gap=args.max_gap, return_paths=True)
    for path, why in skipped:
        _log(f"  skipped {os.path.basename(path)}: {why}")
    if not sessions:
        raise _Attention("no usable recordings after loading")
    _log(f"{len(sessions)} recording(s), "
         f"{sum(len(p) for p, _ in sessions):,} frames")
    _log(f"bodyparts: {bodyparts}")
    return sessions, bodyparts, kept_paths


def _save_manifest(out_dir, kept_paths, lengths, n_found, n_skipped):
    """Persist which recording each row came from, and what it decodes to.

    Without this the only identifier a checkpoint carries is position in a
    `sorted()` glob, which shifts the moment a file becomes unreadable and
    cannot be reconstructed afterwards.
    """
    rows, report = recordings.build_manifest(kept_paths, count_frames=False)
    recordings.attach_lengths(rows, lengths)
    report.update({"n_files_found": n_found, "n_skipped_on_load": n_skipped,
                   "n_frames_after_load": int(np.sum(lengths))})
    recordings.save_manifest(out_dir, rows, report)
    _log(f"manifest: {report['n_recordings']} recording(s), "
         f"{report['n_parsed']} parsed from filename, "
         f"{report['n_frames_after_load']:,} frames")
    if report["n_unparsed"]:
        _log(f"  {report['n_unparsed']} recording(s) did not match the "
             f"filename pattern: {report['unparsed_ids'][:3]}")
    return rows, report


# ------------------------------------------------------------------ commands

def cmd_doctor(args):
    """Preflight, so a batch job is not queued against a broken environment."""
    report = gpu.report()

    if args.print_packages:
        # Machine-readable: hpc/install_gpu.sh pipes this into pip, which is
        # why nothing else may touch stdout here. Needs only the stdlib, so it
        # still answers before cuml/cupy exist -- no chicken-and-egg.
        packages = report["recommended_stack_packages"]
        if not packages:
            print(f"[vieb2] {report['stack_message']}", file=sys.stderr)
            return NEEDS_ATTENTION
        print("\n".join(packages))
        return OK

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

    _log(f"NVIDIA driver     {report['driver'] or 'not detected'}"
         + (f"  (CUDA {report['driver_cuda']})" if report["driver_cuda"] else ""))
    if report["driver_gpu_name"]:
        _log(f"driver GPU name   {report['driver_gpu_name']}")
    if report["driver_error"] and not report["driver_ok"]:
        _log(f"nvidia-smi        {report['driver_error'].splitlines()[0][:80]}")
    _log(f"recommended stack {report['recommended_stack_label'] or 'none'}")
    _log(report["stack_message"])
    if report["recommended_stack_packages"] and not report["hdbscan_gpu"]:
        _log("to build a venv for this stack:  ./hpc/install_gpu.sh")
        _log("  (a separate python/3.11.4 venv -- the RAPIDS pins do not all")
        _log("   publish wheels for the 3.13 the default venv is built on)")
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
    sessions, bodyparts, kept_paths = _load_pose(args)
    kept = [b for b in bodyparts
            if str(b).strip().lower() not in keypoints.DROPPED_BODYPARTS]
    dropped = [b for b in bodyparts if b not in kept]
    selected = [keypoints.select(p, c, bodyparts)[:2] for p, c in sessions]
    _log(f"dropped keypoints {dropped} -> K={len(kept)}, D={2 * len(kept)}, "
         f"rank ceiling {2 * len(kept) - 3}")

    t0 = time.time()
    full = align_all_full(selected)
    aligned, reference = full["aligned"], full["reference"]
    flat = np.concatenate([a.reshape(a.shape[0], -1) for a in aligned])
    leak = null_leakage(flat)
    _log(f"aligned {len(aligned)} recording(s) in {time.time() - t0:.1f}s")
    _log(f"null-direction leakage {leak:.3%} -- variance that confidence "
         f"weighting put back into the nominally-null directions")
    if leak > 0.05:
        _log("  WARNING: high leakage means the PCs are partly tracking "
             "likelihood noise rather than pose")

    lengths = [len(a) for a in aligned]
    _save(os.path.join(args.out, ALIGNED),
          {**_pack([a.reshape(a.shape[0], -1) for a in aligned]),
           "reference": reference},
          {"bodyparts": kept, "dropped": dropped, "n_keypoints": len(kept),
           "expected_rank": 2 * len(kept) - 3, "null_leakage": leak})

    # Arena position and heading, which the alignment removes. Saved beside the
    # aligned pose rather than folded into it, so the pose-only arm stays
    # exactly what it was and the two can be compared.
    frame = [np.column_stack([c, t]).astype(np.float32)
             for c, t in zip(full["centroid"], full["theta"])]
    _save(os.path.join(args.out, POSE_FRAME), _pack(frame),
          {"columns": ["centroid_x", "centroid_y", "theta"],
           "units": ["pixels", "pixels", "radians"],
           "note": "heading = -theta; theta is the rotation applied to the "
                   "centred pose to bring it onto the reference"})
    _log(f"saved arena position and heading for {len(frame)} recording(s)")

    _save_manifest(args.out, kept_paths, lengths, len(kept_paths),
                   len(kept_paths) - len(aligned))
    return OK


def cmd_latent(args):
    """Reduce aligned pose with PCA or diffusion maps -- the same slot."""
    data, meta = _load(os.path.join(args.out, ALIGNED))
    aligned = _unpack(data)
    method = args.latent_method

    t0 = time.time()
    latent = make_latent(method, args.var_threshold, args.n_components,
                         args.alpha, args.epsilon, args.diffusion_time,
                         args.n_landmarks).fit(aligned)
    scores = latent.transform_all(aligned)
    report = latent.spectrum_report()

    _log(f"latent [{method}] over {sum(len(a) for a in aligned):,} frames "
         f"from {len(aligned)} recording(s) in {time.time() - t0:.1f}s")
    _log("  one basis for the whole project -- never per recording")
    _log(f"  q={report['n_components']} components")

    if method == "pca":
        _log(f"  {report['explained_variance']:.1%} of variance retained")
        _report_rank(report, meta)
    else:
        # A diffusion coordinate is not interpretable without the bandwidth,
        # alpha and diffusion time that produced it.
        _log(f"  epsilon={report['epsilon']:.4g} (10th percentile of pairwise "
             f"squared distance), alpha={report['alpha']}, "
             f"t={report['diffusion_time']}")
        _log(f"  fitted on {report['n_landmarks']:,} landmarks, extended to "
             f"all frames by Nystrom")
        if report["spectral_gap"] is not None:
            _log(f"  spectral gap {report['spectral_gap']:.4f}")
        if report["alpha"] < 1.0:
            _log("  alpha<1 lets sampling density distort the embedding: "
                 "densely sampled (slow) regions get compressed")
        _report_operator_health(report)
        _report_degenerate_extensions(report)

    _save(os.path.join(args.out, SCORES),
          {**_pack(scores)},
          {**meta, "latent_method": method, "latent": report, "pca": report})
    return OK


def _report_rank(report, meta):
    nonzero, expected = report["n_nonzero_directions"], meta["expected_rank"]
    _log(f"  {nonzero} non-null directions, {expected} expected (rank 2K-3)")
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


def _report_operator_health(report):
    """Report landmark pruning and the stationary-eigenvector count.

    These are the two numbers that distinguish a diffusion map from an
    expensive lookup table. n_trivial_eigenvectors > 1 means the operator was
    disconnected and the coordinates encode component membership rather than
    geometry -- the failure that once sent a 36h GPU job down the CPU path with
    nothing in the log to say so.
    """
    pruned = report.get("n_landmarks_pruned", 0)
    kept = report.get("n_landmarks", 0)
    if pruned:
        _log(f"  pruned {pruned:,} isolated landmark(s), {kept:,} kept "
             f"(epsilon {report.get('epsilon', float('nan')):.1f})")

    trivial = report.get("n_trivial_eigenvectors", 1)
    if trivial > 1:
        _log(f"  WARNING: {trivial} eigenvalues equal 1.0 where a connected "
             f"operator has exactly 1 -- the embedding reflects which "
             f"disconnected piece a frame landed in, not the manifold. Raise "
             f"--epsilon or --n-landmarks before trusting these states.")


def _report_degenerate_extensions(report):
    """Warn when Nystrom extended a frame from an empty landmark neighborhood.

    n_degenerate_extensions > 0 usually means a single mistracked keypoint put
    a frame far outside epsilon_'s neighborhood, not that the animal jumped --
    see representation/diffusion.py's module docstring. Those frames get a
    defined (zero) embedding rather than crashing, but a high fraction is
    worth knowing about.
    """
    n = report.get("n_degenerate_extensions", 0)
    if not n:
        return
    frac = report.get("degenerate_extension_frac", 0.0)
    _log(f"  WARNING: {n:,} frame(s) ({frac:.4%}) fell outside every "
         f"landmark's neighborhood and got a zero embedding -- likely "
         f"single mistracked keypoints, not real jumps. Raise --epsilon "
         f"or check upstream tracking/alignment if this fraction is large.")


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

    # Record the run so the Cluster Runs page shows it alongside GUI runs.
    run_registry.record(
        args.out,
        latent_method=meta.get("latent_method", "pca"),
        params={"min_cluster_size": args.min_cluster_size,
                "min_samples": args.min_samples,
                "n_lags": meta.get("n_lags"),
                "lag_stride": meta.get("lag_stride"),
                "hdbscan_backend": backend},
        metrics=metrics,
        report=meta.get("latent"),
        source="cli",
    )

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(_jsonable({"metrics": metrics, "speed": speed,
                                 "backend": backend}), fh, indent=2)
        _log(f"wrote {args.json}")

    if metrics["n_states"] < 2:
        _log("only one state found -- try a smaller --min-cluster-size")
        return NEEDS_ATTENTION
    return OK


def cmd_koopman(args):
    """Behavioral states as basins of attraction -- no clustering is run.

    Reads `scores.npz`, not `embedded.npz`: the Koopman operator is fitted on
    consecutive frames of the latent itself, so the delay embedding HDBSCAN
    needs would be a second, redundant history. That is also why the label
    array is longer than HDBSCAN's -- no boundary window is dropped -- and why
    the two must be joined on `index`, never positionally.

    The state count is an output here, not a parameter. `--n-regions` is the
    one free knob that could manufacture it, so sweep it before believing it.
    """
    data, meta = _load(os.path.join(args.out, SCORES))
    scores = _unpack(data)

    t0 = time.time()
    result = koopman.extract_topology(
        scores, fps=args.fps, n_regions=args.n_regions,
        v_percentile=args.v_percentile, knn=args.knn, seed=args.seed,
        min_edge_frac=args.min_edge_frac, min_edge_count=args.min_edge_count,
        coherence_tol=args.coherence_tol, knn_sample=args.knn_sample,
    )
    report = result["report"]
    labels = result["labels"]
    _log(f"koopman topology over {labels.size:,} frames from "
         f"{len(scores)} recording(s) in {time.time() - t0:.1f}s")
    _report_topology(report)

    metrics = cluster_metrics(labels)
    speed = speed_diagnostics(labels, np.concatenate(scores, axis=0))
    _report_metrics(metrics, speed)

    path = koopman.save_topology(
        os.path.join(args.out, checkpoints.KOOPMAN_LABELS), result)
    _log(f"wrote {path}")

    run_registry.record(
        args.out,
        latent_method=meta.get("latent_method", "pca"),
        params={"n_regions": report["n_regions"], "fps": args.fps,
                "v_percentile": args.v_percentile,
                "coherence_tol": args.coherence_tol,
                "min_edge_frac": args.min_edge_frac,
                "min_edge_count": args.min_edge_count,
                "knn": args.knn, "knn_sample": args.knn_sample,
                "method": "koopman"},
        metrics=metrics,
        report=report,
        source="cli",
    )

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(_jsonable({"report": report, "metrics": metrics,
                                 "speed": speed}), fh, indent=2)
        _log(f"wrote {args.json}")

    if report["n_attractors"] < 2:
        _log("fewer than two attractors -- the flow did not decompose; try a "
             "larger --n-regions or a lower --min-edge-frac")
        return NEEDS_ATTENTION
    return OK


def _report_topology(report):
    _log(f"  {report['n_attractors']} attractor(s): "
         f"{report['n_fixed_points']} fixed point(s), "
         f"{report['n_limit_cycles']} limit cycle(s)")
    _log(f"  over {report['n_regions']} regions -- the state count is an "
         f"output, but sweep --n-regions before trusting it")
    for period, ok in zip(report["limit_cycle_period_s"],
                          report["limit_cycle_plausible"]):
        if period:
            _log(f"    cycle period {period:.3f}s = {1.0 / period:.2f} Hz"
                 f"  plausible={ok}")
    frac = report["transition_fraction"]
    _log(f"  transition_fraction {frac:.4f} (frames near a separatrix)")
    if frac >= 0.5:
        _log("    WARNING: a majority of frames sit near a separatrix, so the "
             "partition is doing the talking, not the animal")
    if report["knn_subsampled"]:
        _log(f"  separatrix kNN index fitted on a {report['knn_sample']:,}-frame "
             f"subsample; every frame was still queried against it")
    _log(f"  global DMD rank {report['global_rank']}, "
         f"residual {report['global_residual']:.4g}, "
         f"{report['n_growing_modes']} growing mode(s)")


def cmd_run(args):
    for step in (cmd_align, cmd_latent, cmd_embed, cmd_cluster):
        code = step(args)
        if code:
            return code
    _log(f"done -- artifacts in {args.out}")
    return OK


def cmd_compare_latents(args):
    """Run both latents from one aligned checkpoint and print them side by side.

    Reusing a single alignment is the point: the two arms then differ only in
    the reducer, so any difference in the metrics is attributable to that and
    not to a re-run of everything upstream.
    """
    data, meta = _load(os.path.join(args.out, ALIGNED))
    aligned = _unpack(data)
    use_gpu = gpu.resolve(args.gpu)
    sample = None if args.hdbscan_sample <= 0 else args.hdbscan_sample

    results = {}
    for method in LATENT_METHODS:
        _log(f"--- {method} ---")
        t0 = time.time()
        latent = make_latent(method, args.var_threshold, args.n_components,
                             args.alpha, args.epsilon, args.diffusion_time,
                             args.n_landmarks).fit(aligned)
        scores = latent.transform_all(aligned)
        latent_report = latent.spectrum_report()
        if method != "pca":
            _report_operator_health(latent_report)
            _report_degenerate_extensions(latent_report)

        embedded, _ = embed_all(scores, args.n_lags, args.lag_stride)
        if embedded.shape[0] == 0:
            raise _Attention(f"{method}: no frames survived delay embedding")

        labels, _, backend = run_cluster(
            embedded, args.min_cluster_size, args.min_samples,
            use_gpu=use_gpu, return_backend=True, sample=sample)
        results[method] = {
            "metrics": cluster_metrics(labels),
            "speed": speed_diagnostics(labels, embedded),
            "latent": latent_report,
            "backend": backend,
            "seconds": time.time() - t0,
        }
        _log(f"  {time.time() - t0:.1f}s, backend {backend}")

    _print_latent_comparison(results, args)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(_jsonable(results), fh, indent=2)
        _log(f"wrote {args.json}")
    return OK


def _print_latent_comparison(results, args):
    methods = list(results)
    print()
    print("=" * 72)
    print("Latent space comparison -- identical alignment, identical HDBSCAN")
    print("=" * 72)
    print(f"  min_cluster_size {args.min_cluster_size}   n_lags {args.n_lags}"
          f"   lag_stride {args.lag_stride}")
    print()
    header = f"{'metric':<28}" + "".join(f"{m:>20}" for m in methods)
    print(header)
    print("-" * len(header))

    def row(label, fn):
        print(f"{label:<28}" + "".join(f"{fn(results[m]):>20}" for m in methods))

    def fmt(v):
        return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float)
                                        else str(v))

    row("n_states", lambda r: fmt(r["metrics"]["n_states"]))
    row("noise_frac", lambda r: fmt(r["metrics"]["noise_frac"]))
    row("largest_state_frac",
        lambda r: fmt(r["metrics"]["clustered_only"]["largest_state_frac"]))
    row("state_entropy",
        lambda r: fmt(r["metrics"]["clustered_only"]["state_entropy"]))
    row("noise_speed_ratio",
        lambda r: fmt(r["speed"].get("noise_speed_ratio")))
    row("n_components", lambda r: fmt(r["latent"]["n_components"]))
    row("seconds", lambda r: fmt(round(r["seconds"], 1)))
    print()
    print("PCA distances are exact but linear. Diffusion distances are")
    print("nonlinear and defined (random-walk connectivity), but depend on")
    print("epsilon, alpha and t -- they are not comparable across settings.")
    print("No winner is declared.")
    print("=" * 72)


def cmd_compare_koopman(args):
    """HDBSCAN states beside Koopman basins, for both latents.

    Four arms from two out-dirs. Every arm is scored with the *same* metrics --
    Koopman's `-1` is already `metrics.NOISE_LABEL` -- so nothing here is
    special-cased for the method that produced the labels.
    """
    arms, joins = {}, {}
    for latent, out in (("pca", args.pca_out), ("diffusion", args.diffusion_out)):
        if not out:
            continue
        hdb = _score_labels(out, LABELS, EMBEDDED, "embedded")
        koop = _score_labels(out, checkpoints.KOOPMAN_LABELS, SCORES, None)
        if hdb:
            arms[f"{latent}-HDBSCAN"] = hdb
        if koop:
            arms[f"{latent}-Koopman"] = koop
        if hdb and koop:
            joins[latent] = _join_on_index(hdb, koop)

    if not arms:
        raise _Attention(
            f"no labels found under {args.pca_out!r} or {args.diffusion_out!r}"
            f" -- run `cluster` and `koopman` first")

    _print_koopman_comparison(arms, joins)

    if args.json:
        payload = {
            "arms": {k: {"metrics": v["metrics"], "speed": v["speed"],
                         "report": v["report"]} for k, v in arms.items()},
            "index_join": joins,
        }
        with open(args.json, "w") as fh:
            json.dump(_jsonable(payload), fh, indent=2)
        _log(f"wrote {args.json}")
    return OK


def _score_labels(out, labels_name, points_name, points_key):
    """Load one arm's labels and score them against the space they live in."""
    path = os.path.join(out, labels_name)
    if not os.path.exists(path):
        _log(f"skipping {path} -- not found")
        return None

    data, meta = checkpoints.load(path)
    labels, index = data["labels"], data["index"]

    pdata, _ = checkpoints.load(os.path.join(out, points_name))
    points = (pdata[points_key] if points_key
              else np.concatenate(_unpack(pdata), axis=0))

    return {"labels": labels, "index": index, "n_frames": int(labels.size),
            "metrics": cluster_metrics(labels),
            "speed": speed_diagnostics(labels, points),
            "report": {k: v for k, v in meta.items()
                       if k in ("n_attractors", "n_fixed_points",
                                "n_limit_cycles", "n_regions",
                                "transition_fraction", "knn_subsampled",
                                "knn_sample", "limit_cycle_period_s",
                                "limit_cycle_plausible", "min_cluster_size",
                                "hdbscan_backend", "hdbscan_sample")}}


def _join_on_index(hdb, koop):
    """Align two label arrays on (recording, frame) -- never positionally.

    Koopman labels every frame of `scores.npz`; HDBSCAN labels the delay-
    embedded frames, which are fewer by one window per recording. Comparing
    them row by row would silently offset every recording after the first.
    """
    def key(index):
        # One int64 per frame. Frame counts are ~10^4, so a 10^6 stride keeps
        # recordings from colliding while staying far inside int64.
        return index[:, 0].astype(np.int64) * 1_000_000 + index[:, 1]

    hk, kk = key(hdb["index"]), key(koop["index"])
    order = np.argsort(kk)
    pos = np.searchsorted(kk[order], hk)
    pos[pos >= kk.size] = 0
    matched = kk[order][pos] == hk

    hl = hdb["labels"][matched]
    kl = koop["labels"][order][pos[matched]]

    both = (hl >= 0) & (kl >= 0)
    agree_noise = float(((hl < 0) == (kl < 0)).mean()) if hl.size else None
    return {
        "n_hdbscan_frames": int(hdb["n_frames"]),
        "n_koopman_frames": int(koop["n_frames"]),
        "n_joined": int(matched.sum()),
        "n_dropped_by_delay_window": int(koop["n_frames"] - matched.sum()),
        "both_assigned_frac": float(both.mean()) if hl.size else None,
        "noise_agreement_frac": agree_noise,
        "adjusted_rand": _adjusted_rand(hl[both], kl[both]),
    }


def _adjusted_rand(a, b):
    """ARI between two partitions, on the frames both assigned."""
    if a.size == 0:
        return None
    try:
        from sklearn.metrics import adjusted_rand_score
    except ImportError:
        return None
    return float(adjusted_rand_score(a, b))


def _print_koopman_comparison(arms, joins):
    names = list(arms)
    print()
    print("=" * 78)
    print("HDBSCAN states vs Koopman basins -- same alignment, same latents")
    print("=" * 78)
    header = f"{'metric':<26}" + "".join(f"{n:>26}" for n in names)
    print(header)
    print("-" * len(header))

    def fmt(v):
        return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float)
                                        else str(v))

    def row(label, fn):
        print(f"{label:<26}" + "".join(f"{fmt(fn(arms[n])):>26}" for n in names))

    row("n_states", lambda r: r["metrics"]["n_states"])
    row("n_frames", lambda r: r["n_frames"])
    row("noise_frac", lambda r: r["metrics"]["noise_frac"])
    row("largest_state_frac",
        lambda r: r["metrics"]["clustered_only"]["largest_state_frac"])
    row("state_entropy",
        lambda r: r["metrics"]["clustered_only"]["state_entropy"])
    row("noise_speed_ratio", lambda r: r["speed"].get("noise_speed_ratio"))
    row("size_speed_rank_corr", lambda r: r["speed"].get("size_speed_rank_corr"))
    row("n_fixed_points", lambda r: r["report"].get("n_fixed_points"))
    row("n_limit_cycles", lambda r: r["report"].get("n_limit_cycles"))
    row("n_regions", lambda r: r["report"].get("n_regions"))

    for latent, j in joins.items():
        print()
        print(f"{latent}: joined on (recording, frame), not position")
        print(f"  {j['n_koopman_frames']:,} koopman frames, "
              f"{j['n_hdbscan_frames']:,} hdbscan frames, "
              f"{j['n_joined']:,} joined")
        print(f"  {j['n_dropped_by_delay_window']:,} frames exist only in "
              f"koopman -- the delay window hdbscan drops per recording")
        print(f"  both assigned {fmt(j['both_assigned_frac'])}   "
              f"noise agreement {fmt(j['noise_agreement_frac'])}   "
              f"ARI {fmt(j['adjusted_rand'])}")

    print()
    print("noise means different things per column: HDBSCAN's -1 is a frame")
    print("that failed to cluster, Koopman's is a frame near a separatrix.")
    print("They share a value, not a meaning. The Koopman flow field is")
    print("unvalidated on this data (no v_coherence run), so these basins are")
    print("exploratory. No winner is declared.")
    print("=" * 78)


def cmd_tune(args):
    scores_path = os.path.join(args.out, SCORES)
    if not os.path.exists(scores_path):
        _log("no scores checkpoint found; running align + pca first")
        cmd_align(args)
        cmd_latent(args)

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

    def latent_args(sp):
        sp.add_argument("--latent-method", default="pca",
                        choices=list(LATENT_METHODS),
                        help="latent space builder (default: %(default)s)")
        sp.add_argument("--var-threshold", type=float, default=0.95,
                        help="PCA only: variance to retain")
        sp.add_argument("--n-components", type=int, default=8,
                        help="diffusion only: number of coordinates")
        sp.add_argument("--alpha", type=float, default=1.0,
                        help="diffusion only: density normalisation; 1.0 "
                             "(Laplace-Beltrami) removes most of the sampling-"
                             "density effect")
        sp.add_argument("--epsilon", default="auto",
                        help="diffusion only: kernel bandwidth, or 'auto'")
        sp.add_argument("--diffusion-time", type=int, default=1,
                        help="diffusion only: random-walk steps")
        sp.add_argument("--n-landmarks", type=int, default=3000,
                        help="diffusion only: points the operator is built on")

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
    sp.add_argument("--print-packages", action="store_true",
                    help="print only the pinned packages of the RAPIDS stack "
                         "matching the detected driver, one per line, for pip")
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("align", help="drop noisy keypoints, align pose")
    pose_args(sp), out_args(sp), gpu_args(sp)
    sp.set_defaults(func=cmd_align)

    for name, helptext in (("latent", "pooled PCA or diffusion maps"),
                           ("pca", "alias for `latent` (backwards compatible)")):
        sp = sub.add_parser(name, help=helptext)
        out_args(sp), gpu_args(sp), latent_args(sp)
        sp.set_defaults(func=cmd_latent)

    sp = sub.add_parser("embed", help="delay-embed the PC scores")
    out_args(sp), gpu_args(sp), embed_args(sp)
    sp.set_defaults(func=cmd_embed)

    sp = sub.add_parser("cluster", help="HDBSCAN on the embedding")
    out_args(sp), gpu_args(sp), cluster_args(sp)
    sp.set_defaults(func=cmd_cluster)

    # No gpu_args: this path is pure numpy/sklearn. The GPU accelerates
    # HDBSCAN only, and no clustering is run here.
    sp = sub.add_parser("koopman",
                        help="basins of attraction as states (no clustering)")
    out_args(sp)
    sp.add_argument("--fps", type=float, default=30.0)
    sp.add_argument("--n-regions", type=int, default=48,
                    help="Voronoi cells the flow is partitioned into; the one "
                         "free parameter that could manufacture the state "
                         "count, so sweep it (default: %(default)s)")
    sp.add_argument("--v-percentile", type=float, default=25.0,
                    help="||v|| percentile below which a region counts as slow")
    sp.add_argument("--coherence-tol", type=float, default=0.5,
                    help="direction coherence separating a cycle from a freeze")
    sp.add_argument("--min-edge-frac", type=float, default=0.05,
                    help="share of outgoing mass an edge needs to survive")
    sp.add_argument("--min-edge-count", type=int, default=3,
                    help="absolute frame count an edge needs to survive; a "
                         "sparse region's stray frame can clear the share floor")
    sp.add_argument("--knn", type=int, default=12,
                    help="neighbours voting on whether a frame is a transition")
    sp.add_argument("--knn-sample", type=int, default=1_000_000, metavar="N",
                    help="fit the separatrix neighbour index on at most N "
                         "frames, then query all of them against it; 0 forces "
                         "the exact all-frames index (default: %(default)s)")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument("--json", default=None, help="write the report as JSON")
    sp.set_defaults(func=cmd_koopman)

    sp = sub.add_parser("compare-koopman",
                        help="HDBSCAN states vs Koopman basins, both latents")
    sp.add_argument("--pca-out", default=None,
                    help="out-dir holding the PCA arm's checkpoints")
    sp.add_argument("--diffusion-out", default=None,
                    help="out-dir holding the diffusion arm's checkpoints")
    sp.add_argument("--json", default=None, help="write the comparison as JSON")
    sp.set_defaults(func=cmd_compare_koopman)

    sp = sub.add_parser("run", help="align + latent + embed + cluster")
    pose_args(sp), out_args(sp), gpu_args(sp), embed_args(sp)
    latent_args(sp), cluster_args(sp)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("compare-latents",
                        help="run PCA and diffusion from one alignment")
    out_args(sp), gpu_args(sp), embed_args(sp), latent_args(sp)
    sp.add_argument("--min-cluster-size", type=int, default=50)
    sp.add_argument("--min-samples", type=int, default=None)
    sp.add_argument("--hdbscan-sample", type=int, default=300_000)
    sp.add_argument("--json", default=None)
    sp.set_defaults(func=cmd_compare_latents)

    sp = sub.add_parser("tune", help="suggest --lag-stride from the data")
    pose_args(sp), out_args(sp), gpu_args(sp), latent_args(sp)
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
