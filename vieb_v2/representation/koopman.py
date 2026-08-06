"""Koopman-style topology of the latent flow: fixed points, cycles, basins.

Every other decomposition in this codebase defines a behavior as a *density
peak* -- HDBSCAN finds regions where frames pile up. This module defines it as a
*basin of attraction*: estimate the flow on the latent manifold, read off its
attractors, and label each frame by which attractor its trajectory falls into.
The number of behaviors is then an output of the topology rather than a
parameter (`min_cluster_size` in `cluster.py`, the HDP truncation in
keypoint-MoSeq).

The comparison with keypoint-MoSeq is the point. MoSeq's per-syllable `A_k` is
already a local *linear* approximation to the flow in one region, with the
syllable count pre-committed. Fitting local operators directly and taking their
topology drops both the piecewise-linearity assumption and the pre-specified
state count.

Three things are worth knowing before reading further.

**Snapshot pairs must not straddle a recording boundary.** A pair
(p_t, p_{t+1}) spanning the end of one animal's session and the start of
another's is a fabricated transition, and DMD would fit it as real dynamics.
`snapshot_pairs` therefore takes a *list* of per-recording arrays and rejects a
concatenated one outright, the same guard `embed.embed_all` uses and for the
same reason -- the constraint lives in the signature, where it cannot be
forgotten, rather than in a comment.

**Local operators are affine, not linear.** A region of the manifold sits at
some offset from the origin, and `p -> A p` cannot represent a flow that
circulates about a point other than 0. Fitting `p_{t+1} = A p_t + c` instead is
what lets a limit-cycle arc report rotation rather than a spurious contraction.
This is implemented by centering each snapshot matrix on its own mean before
calling `dmd`: the least-squares `A` of the affine fit is exactly the
least-squares `A` of the linear fit on doubly-centered data, so the global DMD
of Part 1 and the local operators share one code path.

**An attractor is a terminal strongly-connected component** of the
region-transition graph -- a set of regions the trajectory can move within but
never leave. That is the textbook definition of a recurrent set for a discrete
transition system, and it makes the fixed-point/limit-cycle distinction
structural (one absorbing region with near-zero flow, versus several regions
traversed cyclically) rather than a matter of eigenvalue thresholds alone. It
matters here because a stable spiral and a stable cycle *both* have complex
eigenvalues of modulus near 1; only the flow magnitude and the cycle structure
tell them apart.

Basin labels use `-1` for "near a separatrix", deliberately matching
`metrics.NOISE_LABEL`, so `cluster_metrics` and `speed_diagnostics` score basin
labels unchanged.
"""

from __future__ import annotations

import os

import numpy as np

from .metrics import NOISE_LABEL

# A region needs more snapshot pairs than it has dimensions before a local
# operator means anything; below this the fit is interpolation, not estimation.
_PAIRS_PER_DIM = 3


# ---------------------------------------------------------------------------
# Part 1 -- global DMD
# ---------------------------------------------------------------------------

def snapshot_pairs(sessions, region_ids=None, region=None):
    """Build (X, X') from per-recording latent trajectories.

    sessions : list of (T_r, q) arrays -- one per recording, never concatenated

    Returns two (q, M) matrices whose columns are paired consecutive frames.
    Pairs are formed strictly within a recording, so no column of X' is the
    successor of a different animal's last frame.

    `region_ids` / `region` restrict the pairs to those *originating* in one
    region. Deliberately not both endpoints: a local operator approximates the
    flow map *on* a region, and the image of a region is not the region. On a
    limit cycle the per-frame step routinely exceeds the width of a cell, so
    requiring both endpoints inside yields almost no pairs for exactly the fast
    regions whose rotation is the thing worth measuring -- the operator would
    then exist only where the trajectory is slow, which is the opposite of what
    the topology needs. The recording-boundary guarantee is unaffected; pairs
    are still formed strictly within one recording.
    """
    if isinstance(sessions, np.ndarray):
        raise TypeError(
            "sessions must be a list of per-recording arrays, not a single "
            "concatenated array; concatenating first would pair the last frame "
            "of one recording with the first frame of the next and fit that "
            "fabricated jump as dynamics"
        )
    if not isinstance(sessions, (list, tuple)):
        raise TypeError(f"sessions must be a list, got {type(sessions).__name__}")

    lefts, rights = [], []
    for r, scores in enumerate(sessions):
        scores = np.asarray(scores, dtype=np.float64)
        if scores.ndim != 2:
            raise ValueError(f"recording {r} must be (T, q), got {scores.shape}")
        if scores.shape[0] < 2:
            continue

        keep = np.ones(scores.shape[0] - 1, dtype=bool)
        if region is not None:
            keep = np.asarray(region_ids[r])[:-1] == region
            if not keep.any():
                continue

        lefts.append(scores[:-1][keep])
        rights.append(scores[1:][keep])

    if not lefts:
        q = int(np.asarray(sessions[0]).shape[1]) if len(sessions) else 0
        empty = np.empty((q, 0))
        return empty, empty.copy()

    return (np.concatenate(lefts, axis=0).T,
            np.concatenate(rights, axis=0).T)


def dmd(X, Xp, rank=None, energy=0.9999):
    """Best-fit linear operator with X' ~= A X, by the standard SVD-based DMD.

        X = U S V*        A_tilde = U* X' V S^-1        modes = U w

    for eigenvectors w of A_tilde. The full A is never formed; A_tilde carries
    the same spectrum on the leading subspace.

    The rank actually used and the relative residual are both returned, so a
    fit that is degenerate (a region whose motion is effectively
    one-dimensional, say) is visible rather than silently reported as a
    confident spectrum.
    """
    X = np.asarray(X, dtype=np.float64)
    Xp = np.asarray(Xp, dtype=np.float64)
    if X.shape != Xp.shape:
        raise ValueError(f"X {X.shape} and X' {Xp.shape} must match")
    if X.size == 0 or X.shape[1] == 0:
        return {"eigenvalues": np.empty(0, dtype=complex),
                "modes": np.empty((X.shape[0], 0), dtype=complex),
                "amplitudes": np.empty(0, dtype=complex),
                "operator": np.empty((0, 0)), "rank": 0, "residual": None}

    U, s, Vh = np.linalg.svd(X, full_matrices=False)

    keep = s > (s.max() * 1e-12) if s.size and s.max() > 0 else np.zeros(0, bool)
    r = int(keep.sum())
    if r == 0:
        return {"eigenvalues": np.empty(0, dtype=complex),
                "modes": np.empty((X.shape[0], 0), dtype=complex),
                "amplitudes": np.empty(0, dtype=complex),
                "operator": np.empty((0, 0)), "rank": 0, "residual": None}

    if energy is not None and energy < 1.0:
        cumulative = np.cumsum(s[:r] ** 2) / np.sum(s[:r] ** 2)
        r = int(np.searchsorted(cumulative, energy) + 1)
    if rank is not None:
        r = min(int(rank), r)
    r = max(1, min(r, s.size))

    Ur, sr, Vr = U[:, :r], s[:r], Vh[:r].conj().T
    a_tilde = Ur.conj().T @ Xp @ Vr @ np.diag(1.0 / sr)
    eigenvalues, w = np.linalg.eig(a_tilde)
    modes = Ur @ w

    reconstructed = Ur @ a_tilde @ (Ur.conj().T @ X)
    denominator = np.linalg.norm(Xp)
    residual = (float(np.linalg.norm(Xp - reconstructed) / denominator)
                if denominator > 0 else None)

    amplitudes = np.linalg.lstsq(modes, X[:, 0].astype(complex), rcond=None)[0]

    return {"eigenvalues": eigenvalues, "modes": modes,
            "amplitudes": amplitudes, "operator": a_tilde,
            "rank": r, "residual": residual}


def classify_modes(eigenvalues, fps, unit_tol=0.05):
    """Interpret each eigenvalue as a behavior-scale claim.

      |lambda| ~ 1  persistent oscillation -- gait, grooming rhythm. These are
                    the limit cycles.
      |lambda| < 1  a decaying transient; a movement that settles.
      |lambda| > 1  a growing mode. Real behavior is bounded, so on real data
                    this means numerical trouble or a nonstationarity, and it is
                    flagged rather than reported as a behavior.

    Frequencies come back in Hz precisely so that a mode claiming to be gait can
    be checked against known rodent stride rates instead of taken on faith.
    """
    eigenvalues = np.atleast_1d(np.asarray(eigenvalues, dtype=complex))
    out = []
    for lam in eigenvalues:
        magnitude = float(abs(lam))
        angle = float(np.angle(lam))
        cycles_per_frame = abs(angle) / (2.0 * np.pi)
        frequency = cycles_per_frame * float(fps)

        if magnitude > 1.0 + unit_tol:
            band = "growing"
        elif magnitude < 1.0 - unit_tol:
            band = "decaying"
        else:
            band = "oscillatory"

        out.append({
            "eigenvalue_real": float(lam.real),
            "eigenvalue_imag": float(lam.imag),
            "magnitude": magnitude,
            "angle_rad": angle,
            "frequency_hz": frequency,
            "period_frames": (1.0 / cycles_per_frame) if cycles_per_frame > 0 else None,
            "period_s": (1.0 / frequency) if frequency > 0 else None,
            "band": band,
            "flagged": band == "growing",
            "oscillatory": abs(lam.imag) > 1e-9,
        })
    return out


# ---------------------------------------------------------------------------
# Part 2 -- local linearisation
# ---------------------------------------------------------------------------

def partition(sessions, n_regions=48, method="kmeans", seed=0):
    """Split the latent space into regions, returned per recording.

    A single global operator would assume one linear law governs the whole
    manifold, which is the assumption this module exists to escape. Regions are
    the unit of local linearisation; they are not behaviors, and there are
    deliberately far more of them than there will be attractors.

    `method="kmeans"` uses MiniBatchKMeans -- the micro-cluster stage this
    approach was specified against. `method="knn"` is kept as the alternative
    partition; the choice is a parameter so a micro-cluster assignment computed
    elsewhere can be substituted without touching this module.
    """
    stacked = np.concatenate([np.asarray(s, dtype=np.float64) for s in sessions],
                             axis=0)
    lengths = [len(s) for s in sessions]
    n_regions = int(min(n_regions, max(1, stacked.shape[0] // _PAIRS_PER_DIM)))

    if method == "kmeans":
        centers = _minibatch_kmeans(stacked, n_regions, seed)
        ids = _nearest(stacked, centers)
    elif method == "knn":
        rng = np.random.default_rng(seed)
        idx = rng.choice(stacked.shape[0], size=n_regions, replace=False)
        centers = stacked[idx]
        ids = _nearest(stacked, centers)
    else:
        raise ValueError(f"method must be 'kmeans' or 'knn'; got {method!r}")

    ids = np.asarray(ids, dtype=int)
    return list(np.split(ids, np.cumsum(lengths)[:-1])), centers


def _minibatch_kmeans(points, k, seed, n_iter=120, batch_size=2048):
    """MiniBatchKMeans, in numpy, with sklearn used when it is installed.

    The sibling modules here are deliberately numpy-native --
    `metrics._rank_corr` hand-rolls Spearman rather than import scipy -- and a
    partition into micro-regions is not where a hard dependency earns its
    place. sklearn is still preferred when present because it is faster and is
    already a declared dependency; the fallback exists so the topology can be
    verified in an environment that lacks it.
    """
    try:
        from sklearn.cluster import MiniBatchKMeans

        return MiniBatchKMeans(n_clusters=k, random_state=int(seed), n_init=10,
                               batch_size=batch_size).fit(points).cluster_centers_
    except ImportError:
        pass

    rng = np.random.default_rng(seed)
    centers = _kmeans_plusplus(points, k, rng)
    counts = np.zeros(k)

    for _ in range(n_iter):
        n = points.shape[0]
        batch = points[rng.choice(n, size=min(batch_size, n), replace=False)]
        assignment = _nearest(batch, centers)
        for j in np.unique(assignment):
            mask = assignment == j
            counts[j] += mask.sum()
            # Per-center learning rate: the running mean of everything this
            # center has ever been assigned, which is what makes the updates
            # converge rather than oscillate with the batch draw.
            centers[j] += (mask.sum() / counts[j]) * (batch[mask].mean(axis=0)
                                                      - centers[j])
    return centers


def _kmeans_plusplus(points, k, rng):
    """D^2 seeding. Random init leaves whole arms of the manifold unclaimed,
    which shows up directly as a missing attractor."""
    centers = np.empty((k, points.shape[1]))
    centers[0] = points[rng.integers(points.shape[0])]
    closest = ((points - centers[0]) ** 2).sum(axis=1)

    for i in range(1, k):
        total = closest.sum()
        if total <= 0:
            centers[i] = points[rng.integers(points.shape[0])]
        else:
            centers[i] = points[rng.choice(points.shape[0], p=closest / total)]
        closest = np.minimum(closest, ((points - centers[i]) ** 2).sum(axis=1))
    return centers


def _nearest(points, centers, chunk=4096):
    out = np.empty(points.shape[0], dtype=int)
    for start in range(0, points.shape[0], chunk):
        block = points[start:start + chunk]
        d = ((block[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        out[start:start + chunk] = d.argmin(axis=1)
    return out


def local_operators(sessions, region_ids, n_regions, fps, unit_tol=0.05):
    """Fit one affine Koopman operator per region.

    Nonlinear globally, linear locally, with no pre-committed number of
    regimes. Each region reports its eigenvalues, dominant frequency, mean flow
    magnitude ||v||, and whether it is contracting (an attractor), expanding, or
    oscillatory.

    Centering both snapshot matrices makes the fit affine -- see the module
    docstring. Without it, an arc of a limit cycle offset from the origin
    reports contraction toward the origin instead of rotation about the cycle.
    """
    regions = []
    for r in range(n_regions):
        X, Xp = snapshot_pairs(sessions, region_ids, region=r)
        record = {
            "region": r,
            "n_pairs": int(X.shape[1]),
            "n_frames": int(sum(int((ids == r).sum()) for ids in region_ids)),
            "v_norm": _flow_magnitude(sessions, region_ids, r),
            "coherence": _direction_coherence(sessions, region_ids, r),
            "eigenvalues": [],
            "dominant_frequency_hz": None,
            "dominant_period_frames": None,
            "regime": "unknown",
            "residual": None,
            "modes": [],
        }

        q = X.shape[0]
        if X.shape[1] >= max(2, _PAIRS_PER_DIM * q):
            fit = dmd(X - X.mean(axis=1, keepdims=True),
                      Xp - Xp.mean(axis=1, keepdims=True))
            modes = classify_modes(fit["eigenvalues"], fps, unit_tol)
            record["eigenvalues"] = [complex(v) for v in fit["eigenvalues"]]
            record["modes"] = modes
            record["residual"] = fit["residual"]
            record["regime"] = _regime(modes, unit_tol)

            oscillating = [m for m in modes if m["oscillatory"]
                           and m["frequency_hz"] > 0]
            if oscillating:
                dominant = max(oscillating, key=lambda m: m["magnitude"])
                record["dominant_frequency_hz"] = dominant["frequency_hz"]
                record["dominant_period_frames"] = dominant["period_frames"]

        regions.append(record)
    return regions


def _regime(modes, unit_tol):
    """Contracting means every mode decays -- the region is an attractor."""
    if not modes:
        return "unknown"
    magnitudes = [m["magnitude"] for m in modes]
    if max(magnitudes) > 1.0 + unit_tol:
        return "expanding"
    if max(magnitudes) < 1.0 - unit_tol:
        return "contracting"
    return "oscillatory"


def _flow_magnitude(sessions, region_ids, r):
    """Mean ||p_{t+1} - p_t|| over frames leaving this region.

    Every outgoing pair counts, not only the region-internal ones: a region the
    trajectory crosses quickly should report a large ||v|| even though few of
    its pairs stay inside it.
    """
    steps = []
    for scores, ids in zip(sessions, region_ids):
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape[0] < 2:
            continue
        mask = np.asarray(ids)[:-1] == r
        if mask.any():
            steps.append(np.linalg.norm(np.diff(scores, axis=0)[mask], axis=1))
    return float(np.concatenate(steps).mean()) if steps else 0.0


def _direction_coherence(sessions, region_ids, r):
    """Mean cosine between consecutive step vectors in this region.

    This is what actually separates a stable spiral from a stable limit cycle,
    and neither the eigenvalues nor ||v|| can do it. Both have complex
    eigenvalues of modulus near 1; both can be slow. What differs is whether the
    motion *goes anywhere*: on a cycle consecutive steps are nearly parallel
    (coherence -> 1), while at a fixed point the residual motion is
    noise-driven and uncorrelated (coherence -> 0).

    Being a ratio of like quantities it is scale-free, so unlike a ||v||
    threshold it does not need calibrating to the latent's arbitrary units.
    """
    cosines = []
    for scores, ids in zip(sessions, region_ids):
        scores = np.asarray(scores, dtype=np.float64)
        if scores.shape[0] < 3:
            continue
        steps = np.diff(scores, axis=0)
        mask = np.asarray(ids)[:-2] == r
        if not mask.any():
            continue
        first, second = steps[:-1][mask], steps[1:][mask]
        norms = np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1)
        good = norms > 1e-12
        if good.any():
            cosines.append((first[good] * second[good]).sum(axis=1) / norms[good])
    return float(np.concatenate(cosines).mean()) if cosines else 0.0


# ---------------------------------------------------------------------------
# Part 3 -- topology
# ---------------------------------------------------------------------------

def region_graph(region_ids, n_regions, min_edge_frac=0.05, min_edge_count=3):
    """Transition counts between regions, as an adjacency matrix.

    An edge survives only if it carries at least `min_edge_frac` of the
    region's outgoing mass *and* is observed at least `min_edge_count` times.
    Both floors are needed, and the absolute one is not redundant: a sparse
    region in a transient part of the manifold owns a large, thinly-populated
    Voronoi cell, so a single stray frame can be several percent of its
    outgoing mass and clear a share threshold on its own.

    This is the difference between recovering the topology and not. One
    noise-driven excursion that survives pruning welds two genuinely separate
    basins into one recurrent set, and every attractor downstream of it
    disappears -- the flow is meant to be read from where the trajectory
    *reliably* goes, not from anywhere it was ever observed to go once.
    """
    counts = np.zeros((n_regions, n_regions), dtype=float)
    for ids in region_ids:
        ids = np.asarray(ids)
        if ids.size < 2:
            continue
        np.add.at(counts, (ids[:-1], ids[1:]), 1.0)

    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = np.where(totals > 0, counts / totals, 0.0)
    keep = (share >= min_edge_frac) & (counts >= min_edge_count)
    return np.where(keep, counts, 0.0)


def _sccs(adjacency):
    """Strongly connected components, by Kosaraju. Iterative: a project-scale
    region graph is small, but recursion depth is not worth risking."""
    n = adjacency.shape[0]
    successors = [np.flatnonzero(adjacency[i] > 0).tolist() for i in range(n)]
    predecessors = [np.flatnonzero(adjacency[:, i] > 0).tolist() for i in range(n)]

    seen, order = np.zeros(n, dtype=bool), []
    for start in range(n):
        if seen[start]:
            continue
        stack = [(start, iter(successors[start]))]
        seen[start] = True
        while stack:
            node, children = stack[-1]
            advanced = False
            for child in children:
                if not seen[child]:
                    seen[child] = True
                    stack.append((child, iter(successors[child])))
                    advanced = True
                    break
            if not advanced:
                order.append(stack.pop()[0])

    assigned, components = np.full(n, -1, dtype=int), []
    for node in reversed(order):
        if assigned[node] >= 0:
            continue
        component, stack = [], [node]
        assigned[node] = len(components)
        while stack:
            current = stack.pop()
            component.append(current)
            for parent in predecessors[current]:
                if assigned[parent] < 0:
                    assigned[parent] = len(components)
                    stack.append(parent)
        components.append(sorted(component))
    return components, assigned


def attractors(adjacency, regions, fps, v_threshold, cycle_min_regions=3,
               coherence_tol=0.5, min_frames=0):
    """Terminal SCCs of the region graph, classified.

    A terminal component is one the trajectory can enter but not leave; that is
    what "attractor" means for a discrete transition system, and it is what
    makes the state count an output rather than a parameter.

    Classification deliberately does not rest on eigenvalues: a stable spiral
    and a stable limit cycle both have complex eigenvalues of modulus near 1, so
    the spectrum alone cannot tell a freeze from a gait. `||v|| < threshold`
    alone is no better, because the threshold is a percentile of an arbitrary
    latent scale and a fixed point sits inside the very population that defines
    it. Direction coherence is the discriminator that survives both problems --
    see `_direction_coherence`. `v_threshold` is still applied, as 3a specifies,
    but it is not what the classification hangs on.
    """
    components, assigned = _sccs(adjacency)

    found = []
    for cid, component in enumerate(components):
        leaves = any(assigned[j] != cid
                     for i in component
                     for j in np.flatnonzero(adjacency[i] > 0))
        if leaves:
            continue

        members = [regions[i] for i in component]
        frames = sum(m["n_frames"] for m in members)
        # A recurrent set the trajectory entered twice is a measurement
        # artifact, not a behavior. Without this floor every stray pair of
        # frames that happens to be mutually reachable becomes a state.
        if frames <= max(0, min_frames):
            continue

        weights = [max(1, m["n_frames"]) for m in members]
        v_mean = float(np.average([m["v_norm"] for m in members], weights=weights))
        coherence = float(np.average([m["coherence"] for m in members],
                                     weights=weights))
        contracting = [m for m in members if m["regime"] == "contracting"]
        periods = [m["dominant_period_frames"] for m in members
                   if m["dominant_period_frames"]]

        is_cycle = bool(periods
                        and coherence >= coherence_tol
                        and len(component) >= cycle_min_regions)
        is_fixed = (not is_cycle
                    and (coherence < coherence_tol or v_mean <= v_threshold))

        kind = "limit_cycle" if is_cycle else "fixed_point" if is_fixed else "recurrent"

        found.append({
            "attractor": len(found),
            "kind": kind,
            "regions": component,
            "n_regions": len(component),
            "attractor_frames": frames,
            "v_norm": v_mean,
            "coherence": coherence,
            "contracting_regions": len(contracting),
            "max_magnitude": max(
                (m["magnitude"] for mem in members for m in mem["modes"]),
                default=None),
            "eigen_period_frames": float(np.median(periods)) if periods else None,
            "eigen_period_s": (float(np.median(periods)) / float(fps)
                               if periods else None),
        })
    return found, assigned


def _empirical_period(region_ids, component, fps):
    """Return time of the trajectory to the busiest region of a cycle.

    The eigenvalue period is a local rotation rate; this is the period actually
    observed. They agree only when the flow circulates uniformly, so reporting
    both is what makes a disagreement visible instead of averaged away.
    """
    members = set(int(r) for r in component)
    counts = {}
    for ids in region_ids:
        for r in members:
            counts[r] = counts.get(r, 0) + int((np.asarray(ids) == r).sum())
    if not counts:
        return None
    busiest = max(counts, key=counts.get)

    gaps = []
    for ids in region_ids:
        ids = np.asarray(ids)
        visits = (ids == busiest).astype(int)
        if visits.sum() < 2:
            continue
        starts = np.flatnonzero(np.diff(np.r_[0, visits]) == 1)
        if starts.size >= 2:
            gaps.append(np.diff(starts))
    if not gaps:
        return None

    period_frames = float(np.median(np.concatenate(gaps)))
    return {"period_frames": period_frames,
            "period_s": period_frames / float(fps),
            "frequency_hz": float(fps) / period_frames if period_frames > 0 else None}


def basins(adjacency, found, n_regions):
    """Which attractor each region flows into.

    This is the forward integration of 3c done on the empirical trajectory: a
    region's basin is the attractor reachable from it in the transition graph.
    A region reaching two different attractors is left unassigned -- it sits on
    a separatrix, and picking one would invent a boundary.
    """
    component_of_attractor = {}
    for a in found:
        for r in a["regions"]:
            component_of_attractor[r] = a["attractor"]

    successors = [np.flatnonzero(adjacency[i] > 0).tolist()
                  for i in range(n_regions)]

    basin = np.full(n_regions, NOISE_LABEL, dtype=int)
    for start in range(n_regions):
        reached, seen, stack = set(), {start}, [start]
        while stack:
            node = stack.pop()
            if node in component_of_attractor:
                reached.add(component_of_attractor[node])
                continue
            for child in successors[node]:
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        if len(reached) == 1:
            basin[start] = reached.pop()
    return basin


def separatrices(stacked, labels, knn=12, seed=0, index_sample=1_000_000,
                 query_chunk=500_000):
    """Frames whose neighbours do not agree on an attractor.

    3d asks for points where forward integration is unstable under small
    perturbation. On empirical data the perturbation is already present: a
    frame's k nearest neighbours *are* nearby initial conditions that were
    integrated forward. If they land in different basins, the frame is near a
    separatrix -- a transition, and the direct analogue of keypoint-MoSeq's
    model-free changepoints.

    Above `index_sample` frames the neighbour index is fitted on a seeded
    subsample and every frame is queried against it, rather than fitting on all
    N. This mirrors what `--hdbscan-sample` already does for clustering: a
    fit-and-query over all pairs is what does not scale, not the query itself.
    Every frame still receives a label -- only the pool of possible neighbours
    is subsampled. Set `index_sample=0` to force the exact all-frames index.

    Returns (near_separatrix mask, agreement fraction).
    """
    n = stacked.shape[0]
    k = int(min(max(2, knn) + 1, n))

    sub = None
    if index_sample and n > index_sample:
        rng = np.random.default_rng(seed)
        sub = np.sort(rng.choice(n, size=int(index_sample), replace=False))
        points, index_labels = stacked[sub], labels[sub]
    else:
        points, index_labels = stacked, labels

    query = _neighbour_query(points, k)

    near = np.empty(n, dtype=bool)
    agreement = np.empty(n, dtype=np.float32)
    for start in range(0, n, max(1, int(query_chunk))):
        stop = min(start + max(1, int(query_chunk)), n)
        idx = query(stacked[start:stop], k)

        # A frame is not its own neighbour. On the full index the self-match is
        # always column 0; on a subsampled one it is present only when the
        # frame was itself sampled, so it is located rather than assumed.
        if sub is None:
            keep = idx[:, 1:]
        else:
            keep = _drop_self(idx, sub[idx] == np.arange(start, stop)[:, None])

        neighbour_labels = index_labels[keep]
        block = labels[start:stop]
        agreement[start:stop] = (neighbour_labels == block[:, None]).mean(axis=1)
        near[start:stop] = (block < 0) | (_n_distinct(neighbour_labels) > 1)

    return near, agreement.astype(np.float32)


def _drop_self(idx, is_self):
    """Drop one self-match per row, or the furthest neighbour when there is none.

    Keeps every row the same width, so a frame that happened to be sampled into
    the index is not left with one neighbour fewer than one that was not.
    """
    n, k = idx.shape
    pos = np.where(is_self.any(axis=1), is_self.argmax(axis=1), k - 1)
    mask = np.ones((n, k), dtype=bool)
    mask[np.arange(n), pos] = False
    return idx[mask].reshape(n, k - 1)


def _n_distinct(neighbour_labels):
    """Per-row count of distinct non-negative labels.

    Sorting beats a per-row `np.unique` loop by orders of magnitude at project
    scale, where this runs over tens of millions of rows.
    """
    # Sorting alone separates the classes -- every negative label sorts ahead of
    # every valid one -- so no sentinel is needed. A sentinel would have to be
    # representable in the label dtype, and under NumPy 2's weak promotion an
    # out-of-range one is silently wrapped rather than upcast, which reads as
    # noise being a distinct state.
    ordered = np.sort(neighbour_labels, axis=1)
    valid = ordered >= 0
    changed = np.empty(ordered.shape, dtype=bool)
    changed[:, 0] = valid[:, 0]
    changed[:, 1:] = valid[:, 1:] & (ordered[:, 1:] != ordered[:, :-1])
    return changed.sum(axis=1)


def _n_jobs():
    """Cores to use: the scheduler's allocation where there is one, else all."""
    for var in ("SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS"):
        try:
            n = int(os.environ.get(var, ""))
        except ValueError:
            continue
        if n > 0:
            return n
    return -1


def _neighbour_query(points, k):
    """A reusable `(block, k) -> indices` query, sklearn when available.

    The index is built once and queried in chunks: at project scale the build
    dominates, so refitting per chunk would cost more than the queries do. The
    brute-force fallback is O(N*M) and is meant for verification-scale data.
    """
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError:
        pass
    else:
        # Threading the query is not a micro-optimisation here: it is the whole
        # cost at project scale (28.6M frames against a 1M-point index in 9-D
        # measures ~150 min single-threaded), and sklearn defaults to one core.
        # -1 would mean *every* core on the box, which on a shared batch node
        # ignores the cgroup the scheduler allocated -- four such jobs on one
        # 64-core node oversubscribe it 4x. SLURM_CPUS_PER_TASK is that
        # allocation, so it wins where it is set.
        fitted = NearestNeighbors(n_neighbors=k, n_jobs=_n_jobs()).fit(points)
        return lambda block, k_: fitted.kneighbors(block, n_neighbors=k_)[1]

    squared = (points ** 2).sum(axis=1)

    def brute(block, k_):
        d = (squared[None, :] - 2.0 * (block @ points.T)
             + (block ** 2).sum(axis=1)[:, None])
        nearest = np.argpartition(d, kth=k_ - 1, axis=1)[:, :k_]
        order = np.take_along_axis(d, nearest, axis=1).argsort(axis=1)
        return np.take_along_axis(nearest, order, axis=1)

    return brute


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def extract_topology(sessions, fps=30.0, n_regions=48, v_percentile=25.0,
                     unit_tol=0.05, knn=12, seed=0, partition_method="kmeans",
                     min_edge_frac=0.05, min_edge_count=3, cycle_min_regions=3,
                     coherence_tol=0.5, min_attractor_frac=0.001,
                     plausible_hz=(0.5, 12.0), knn_sample=1_000_000):
    """Full pipeline: flow -> local operators -> topology -> basin labels.

    The returned `labels` are the behavioral states. No clustering is run.
    """
    if not isinstance(sessions, (list, tuple)) or not sessions:
        raise TypeError("sessions must be a non-empty list of per-recording arrays")

    sessions = [np.asarray(s, dtype=np.float64) for s in sessions]
    stacked = np.concatenate(sessions, axis=0)
    index = np.concatenate(
        [np.stack([np.full(len(s), r), np.arange(len(s))], axis=1)
         for r, s in enumerate(sessions)], axis=0).astype(int)

    # Part 1 -- one global operator, reported for reference. It is not used to
    # build the topology; it is the baseline the local fits improve on.
    X, Xp = snapshot_pairs(sessions)
    global_fit = dmd(X, Xp)
    global_modes = classify_modes(global_fit["eigenvalues"], fps, unit_tol)

    # Part 2 -- local operators.
    region_ids, centers = partition(sessions, n_regions, partition_method, seed)
    n_regions = int(centers.shape[0])
    regions = local_operators(sessions, region_ids, n_regions, fps, unit_tol)

    # Part 3 -- topology.
    weights = [max(1, r["n_frames"]) for r in regions]
    v_threshold = float(np.percentile(
        np.repeat([r["v_norm"] for r in regions], weights), v_percentile))

    adjacency = region_graph(region_ids, n_regions, min_edge_frac,
                             min_edge_count)
    found, assigned = attractors(adjacency, regions, fps, v_threshold,
                                 cycle_min_regions, coherence_tol,
                                 int(min_attractor_frac * stacked.shape[0]))

    for a in found:
        # Only a cycle has a period. A fixed point's return time to its own
        # region is 1 frame, and reporting that as a frequency would invent a
        # rhythm where the animal is motionless.
        empirical = (_empirical_period(region_ids, a["regions"], fps)
                     if a["kind"] == "limit_cycle" else None)
        a["empirical_period_frames"] = empirical["period_frames"] if empirical else None
        a["empirical_period_s"] = empirical["period_s"] if empirical else None
        a["frequency_hz"] = empirical["frequency_hz"] if empirical else None
        if a["kind"] == "limit_cycle" and a["frequency_hz"]:
            low, high = plausible_hz
            a["plausible"] = bool(low <= a["frequency_hz"] <= high)
        else:
            a["plausible"] = None

    basin_of_region = basins(adjacency, found, n_regions)
    labels = np.concatenate([basin_of_region[np.asarray(ids)]
                             for ids in region_ids]).astype(np.int32)

    near, agreement = separatrices(stacked, labels, knn, seed,
                                   index_sample=knn_sample)
    labels = labels.copy()
    labels[near] = NOISE_LABEL

    for a in found:
        a["basin_frames"] = int((labels == a["attractor"]).sum())

    fixed = [a for a in found if a["kind"] == "fixed_point"]
    cycles = [a for a in found if a["kind"] == "limit_cycle"]

    return {
        "labels": labels,
        "probabilities": agreement,
        "index": index,
        "attractors": found,
        "fixed_points": fixed,
        "limit_cycles": cycles,
        "regions": regions,
        "region_ids": region_ids,
        "region_centers": centers,
        "adjacency": adjacency,
        "basin_of_region": basin_of_region,
        "global_modes": global_modes,
        "report": {
            "n_attractors": len(found),
            "n_fixed_points": len(fixed),
            "n_limit_cycles": len(cycles),
            "fixed_point_basin_frames": [a["basin_frames"] for a in fixed],
            "limit_cycle_period_s": [a["empirical_period_s"] for a in cycles],
            "limit_cycle_plausible": [a["plausible"] for a in cycles],
            "transition_fraction": float(near.mean()),
            "n_regions": n_regions,
            "knn": int(knn),
            # Recorded, not implied: above this many frames the separatrix
            # neighbour index is fitted on a subsample, so a reader can tell an
            # exact result from an approximate one.
            "knn_sample": int(knn_sample),
            "knn_subsampled": bool(knn_sample and stacked.shape[0] > knn_sample),
            "v_threshold": v_threshold,
            "coherence_tol": coherence_tol,
            "min_attractor_frac": min_attractor_frac,
            "fps": float(fps),
            "global_rank": global_fit["rank"],
            "global_residual": global_fit["residual"],
            "n_growing_modes": sum(1 for m in global_modes if m["flagged"]),
            "unit_tol": unit_tol,
            "min_edge_frac": min_edge_frac,
            "min_edge_count": min_edge_count,
            "clustering": "none -- states are basins of attraction",
        },
    }


def save_topology(path, result):
    """Write basin labels in the shape `checkpoints.LABELS` already uses.

    Same three arrays as `labels.npz`, so basins drop into the slot HDBSCAN
    labels occupy and existing consumers need no change. `-1` means near a
    separatrix rather than HDBSCAN noise, which is why the meaning is recorded
    in `meta` rather than left to the reader.
    """
    from . import checkpoints

    meta = dict(result["report"])
    meta["noise_label_means"] = "near separatrix (transition), not HDBSCAN noise"
    meta["attractors"] = [
        {k: v for k, v in a.items() if k != "regions"} for a in result["attractors"]
    ]
    return checkpoints.save(
        path,
        {"labels": result["labels"].astype(np.int32),
         "probabilities": result["probabilities"].astype(np.float32),
         "index": result["index"]},
        meta,
    )


__all__ = [
    "snapshot_pairs", "dmd", "classify_modes", "partition", "local_operators",
    "region_graph", "attractors", "basins", "separatrices", "extract_topology",
    "save_topology", "NOISE_LABEL",
]
