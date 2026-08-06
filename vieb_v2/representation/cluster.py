"""HDBSCAN on the delay-embedded coordinates.

Run directly on the embedding -- no UMAP. That is a deliberate asymmetry with
v1, which reduces with UMAP first; it is recorded in the benchmark output rather
than quietly equalised.

The -1 noise label is preserved and never force-assigned. Filling it in (by
nearest cluster, or by HMM smoothing) would destroy the one measurement that
tests the density-duration confound: whether unclustered frames are
systematically faster than clustered ones. Force-assignment would hide exactly
the detection bias the representation is meant to be evaluated against.
"""

from __future__ import annotations

import warnings

import numpy as np

from .metrics import NOISE_LABEL, cluster_metrics, speed_diagnostics


class GPUFallbackWarning(UserWarning):
    """Raised when GPU HDBSCAN failed at fit time and CPU took over."""


def cluster(embedded, min_cluster_size=50, min_samples=None, metric="euclidean",
            use_gpu=False, return_backend=False, sample=None, batch_size=25_000,
            seed=0):
    """Cluster delay-embedded coordinates. Returns (labels, probabilities).

    Labels are -1 for noise, preserved as-is.

    With use_gpu this runs cuml's GPU HDBSCAN -- a separate implementation, not
    a faster copy. Measured on a 3568x35 embedding the two backends produce the
    *same partition* (adjusted Rand index 1.0000) but *different integer label
    values*: the grouping agrees, the cluster numbering is permuted. So the
    backend is safe to switch, but a state id is only meaningful within one
    run, which is why the backend used is recorded alongside the labels.
    Agreement at full project scale (~2.3M frames) is untested.

    Falls back to CPU if cuml is missing or the run fails.

    `sample` caps how many points the clusterer is *fitted* on; the rest are
    labelled by approximate_predict in batches. A full VIEB project is ~2.3M
    frames, which neither backend will fit directly, so this mirrors v1's
    strategy (compare.py:1305, :1500) with the same 300k / 25k defaults.
    """
    embedded = np.asarray(embedded, dtype=np.float64)
    if embedded.ndim != 2:
        raise ValueError(f"embedded must be (N, D), got {embedded.shape}")
    if embedded.shape[0] == 0:
        empty = (np.empty(0, dtype=int), np.empty(0))
        return (*empty, "none") if return_backend else empty

    from .gpu import explicitly_requested, resolve

    n = embedded.shape[0]
    subsample = sample is not None and n > int(sample)
    if subsample:
        rng = np.random.default_rng(seed)
        fit_idx = np.sort(rng.choice(n, size=int(sample), replace=False))
        fit_data = embedded[fit_idx]
    else:
        fit_data = embedded

    backend = "cpu"
    result = None

    if resolve(use_gpu):
        try:
            result = _fit_gpu(fit_data, embedded, min_cluster_size, min_samples,
                              subsample, batch_size)
            backend = "gpu"
        except Exception as exc:
            # gpu.resolve() only proves the backend *can* run a toy problem; it
            # cannot predict a failure on this particular embedding. Under
            # "--gpu on" the caller has already said a CPU fallback is not an
            # acceptable outcome, so honour that here too -- a fallback that
            # silently downgrades a 36h GPU allocation to single-core CPU is
            # exactly the failure resolve() exists to prevent, just deferred to
            # fit time. Under "auto" the fallback is the point, but it is
            # announced rather than swallowed: a run that quietly took the slow
            # path is indistinguishable from one that never had a GPU.
            if explicitly_requested(use_gpu):
                raise RuntimeError(
                    f"GPU HDBSCAN was requested with --gpu on but failed while "
                    f"fitting {fit_data.shape[0]} x {fit_data.shape[1]} points: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            warnings.warn(
                f"GPU HDBSCAN failed ({type(exc).__name__}: {exc}); "
                f"falling back to CPU, which is much slower at this scale.",
                GPUFallbackWarning, stacklevel=2,
            )
            result = None

    if result is None:
        result = _fit_cpu(fit_data, embedded, min_cluster_size, min_samples,
                          metric, subsample, batch_size)

    labels, probs = result
    return (labels, probs, backend) if return_backend else (labels, probs)


def _fit_gpu(fit_data, all_data, min_cluster_size, min_samples, subsample,
             batch_size):
    from cuml.cluster import hdbscan as cuh

    model = cuh.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=(None if min_samples is None else int(min_samples)),
        prediction_data=subsample,
    ).fit(fit_data.astype(np.float32))

    if not subsample:
        labels = np.asarray(model.labels_)
        probs = np.asarray(getattr(model, "probabilities_",
                                   np.ones(labels.shape)))
        return labels, probs

    return _predict_batched(
        lambda chunk: cuh.approximate_predict(model, chunk.astype(np.float32)),
        all_data, batch_size,
    )


def _fit_cpu(fit_data, all_data, min_cluster_size, min_samples, metric,
             subsample, batch_size):
    import hdbscan

    model = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
        metric=metric,
        prediction_data=subsample,
    ).fit(fit_data)

    if not subsample:
        labels = model.labels_
        probs = getattr(model, "probabilities_", np.ones(labels.shape))
        return np.asarray(labels), np.asarray(probs)

    from hdbscan import approximate_predict

    return _predict_batched(
        lambda chunk: approximate_predict(model, chunk), all_data, batch_size,
    )


def _predict_batched(predict, data, batch_size):
    """Label every point in batches, so the full project never lands in memory
    at once."""
    labels = np.empty(data.shape[0], dtype=int)
    probs = np.empty(data.shape[0], dtype=float)
    for start in range(0, data.shape[0], batch_size):
        stop = min(start + batch_size, data.shape[0])
        batch_labels, batch_probs = predict(data[start:stop])
        labels[start:stop] = np.asarray(batch_labels)
        probs[start:stop] = np.asarray(batch_probs)
    return labels, probs


def cluster_with_diagnostics(embedded, min_cluster_size=50, min_samples=None,
                             use_gpu=False, sample=None):
    """Cluster and report the metrics plus the confound diagnostics."""
    labels, probs, backend = cluster(
        embedded, min_cluster_size, min_samples, use_gpu=use_gpu,
        return_backend=True, sample=sample,
    )
    if labels.size == 0:
        return {"labels": labels, "probabilities": probs, "metrics": None,
                "speed": None, "backend": backend}
    return {
        "labels": labels,
        "probabilities": probs,
        "backend": backend,
        "metrics": cluster_metrics(labels),
        "speed": speed_diagnostics(labels, embedded),
    }


def seed_stability(embedded, min_cluster_size=50, min_samples=None,
                   n_repeats=5, subsample_frac=0.8, seed=0):
    """Re-cluster on repeated random subsamples and report the spread.

    Instability is itself the signature of the confound. In simulation a real
    density peak was recovered on every seed (100% x6), while a marginal fast
    behavior swung between 15% and 100% recovery. So a large variance in
    `n_states` across subsamples is evidence that clusters are being found or
    missed by luck rather than by structure.
    """
    embedded = np.asarray(embedded, dtype=np.float64)
    n = embedded.shape[0]
    if n < 2:
        return {"n_states": [], "mean": None, "std": None}

    rng = np.random.default_rng(seed)
    counts = []
    for _ in range(n_repeats):
        idx = rng.choice(n, size=max(2, int(n * subsample_frac)), replace=False)
        labels, _ = cluster(embedded[idx], min_cluster_size, min_samples)
        counts.append(int(np.unique(labels[labels >= 0]).size))

    return {
        "n_states": counts,
        "mean": float(np.mean(counts)),
        "std": float(np.std(counts)),
        "n_repeats": n_repeats,
        "subsample_frac": subsample_frac,
    }


__all__ = ["cluster", "cluster_with_diagnostics", "seed_stability", "NOISE_LABEL"]
