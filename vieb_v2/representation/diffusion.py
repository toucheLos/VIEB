"""Diffusion maps -- a nonlinear alternative to pooled PCA.

Why this and not UMAP: the embedding comes from the eigenvectors of a diffusion
operator, so Euclidean distance in the embedding *is* diffusion distance -- the
probability a random walk connects two points through the data manifold within
t steps. UMAP's distances have no such definition. And because this is an
eigenproblem rather than a stochastic optimisation, the result is deterministic
given the kernel bandwidth: two runs on the same data give the same answer,
which matters when the whole point is to compare representations.

The alpha parameter is load-bearing. Coifman's alpha-normalisation at alpha=1
recovers the Laplace-Beltrami operator, making the embedding geometry
independent of how densely each region was sampled.

Measured on a 1-D manifold whose two halves have equal true arc length but very
unequal sampling -- 1500 points in one half, 150 in the other, the
density-duration confound in miniature. Coordinate spread should be equal if
density has been removed:

    alpha    rank-corr with true coord    dense/sparse coordinate spread
    0.0              0.9999                        0.001
    1.0              0.9999                        0.254      (target 1.0)

So it is the *densely* sampled region -- the slow behavior -- that gets
compressed almost to a point without alpha-normalisation, and alpha=1 recovers
roughly 250x of that. Note the direction: high sampling density collapses a
region in the embedding rather than inflating it, because a random walk mixes
quickly through a well-connected neighbourhood.

Two honest limits. alpha=1 reduces the density effect but does not remove it
(0.254, not 1.0). And HDBSCAN downstream is still density-based and still sees
more points in slow regions, so this mitigates the confound in the
*representation*, never in the clusterer.

The failure mode to know about is connectivity, not bandwidth tuning. If any
landmark is unreachable at the chosen bandwidth it becomes its own connected
component, and every component adds an eigenvalue of exactly 1.0 whose
eigenvector is a spike on the points in it. 1.0 being the operator's largest
possible eigenvalue, those spikes outrank every real mode and fill the retained
set, turning the embedding into a component-membership lookup -- on real data
this collapsed 20k frames onto 29 distinct points. `_prune_isolated` drops
those landmarks before the operator is built and `spectrum_report` publishes
both `n_landmarks_pruned` and `n_trivial_eigenvectors`, the latter being the
health check: 1 means connected, more means the coordinates are not geometry.

Interface mirrors PooledPCA so the two are swappable at the same pipeline slot.
"""

from __future__ import annotations

import warnings

import numpy as np

_TRIVIAL_EIGENVALUE_TOL = 1e-8


class DegenerateOperatorWarning(UserWarning):
    """The diffusion operator stayed disconnected after landmark pruning."""


class DiffusionMap:
    """Diffusion map with Nystrom out-of-sample extension.

    Attributes after `fit`:
      landmarks_        (M, D)  points the operator was built on
      eigenvalues_      (q,)    non-trivial eigenvalues, descending
      psi_              (M, q)  eigenvectors at the landmarks
      epsilon_          float   kernel bandwidth actually used
      n_components_     int     q
    """

    # Quantile of pairwise squared distances used as the default bandwidth.
    # Measured on a non-uniformly sampled 1-D manifold: 5-25% all recover the
    # geometry (rank-corr 0.9999), below 2% the graph disconnects and the
    # embedding collapses (0.02), and the median -- a common default elsewhere
    # -- short-circuits the manifold entirely (0.09).
    EPSILON_PERCENTILE = 10.0

    # Minimum off-diagonal kernel mass ("effective neighbours") a landmark must
    # have to stay in the operator. A landmark below this is isolated, and an
    # isolated landmark is not a harmless outlier: it forms its own connected
    # component, which contributes an eigenvalue of exactly 1.0 whose
    # eigenvector is a spike on that single point. Since 1.0 is the largest
    # eigenvalue the operator can have, those spikes sort to the *top* of the
    # spectrum and evict every real manifold mode from the retained set.
    #
    # Measured on the 28.6M-frame project (3000 landmarks, alpha=1, default
    # bandwidth), varying only this threshold:
    #
    #   threshold   kept   epsilon   lambda_1   lambda_8   #lambda=1   unique/20k
    #     none      3000     501.4   1.000000   1.000000       30            29
    #     1e-3      2948     493.0   1.000000   0.999443        1          4691
    #     0.1       2905     486.1   0.999892   0.995458        1         17869
    #     1.0       2858     478.6   0.988172   0.934651        1         19973
    #
    # Without pruning, 20k frames collapse onto 29 distinct embedded points --
    # the coordinates had become component indicators rather than geometry, and
    # HDBSCAN downstream cannot cluster that. At 1.0 the spectrum decays
    # properly, exactly one trivial eigenvector remains (as theory requires for
    # a connected operator), and the embedding is continuous again.
    #
    # Note what this does NOT do: it does not widen the bandwidth. Connecting
    # the same graph by raising epsilon instead needs the 90th percentile of
    # pairwise distances -- 12.7x the default -- which is deep in the
    # short-circuit regime the EPSILON_PERCENTILE note above warns about. The
    # cheaper and more honest fix is to drop the handful of outlier poses that
    # no bandwidth was ever going to connect.
    MIN_NEIGHBOR_MASS = 1.0

    # Pruning is iterative because removing a landmark lowers its neighbours'
    # mass; these bound the work and refuse to gut the operator silently.
    MAX_PRUNE_ROUNDS = 20
    MAX_PRUNE_FRACTION = 0.25

    def __init__(self, n_components=8, alpha=1.0, epsilon="auto",
                 diffusion_time=1, n_landmarks=3000, random_state=0,
                 use_gpu=False, min_neighbor_mass=None):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if diffusion_time < 0:
            raise ValueError("diffusion_time must be >= 0")
        self.n_components = int(n_components)
        self.alpha = float(alpha)
        self.epsilon = epsilon
        self.diffusion_time = diffusion_time
        self.n_landmarks = int(n_landmarks)
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.min_neighbor_mass = (self.MIN_NEIGHBOR_MASS
                                  if min_neighbor_mass is None
                                  else float(min_neighbor_mass))
        self.backend_ = "cpu"

    # ------------------------------------------------------------------ fit

    def fit(self, sessions):
        """Fit on frames pooled across all recordings.

        Pooling is as mandatory here as for PCA: a per-recording operator would
        give each animal its own manifold, and no coordinate would be
        comparable across them.
        """
        if not sessions:
            raise ValueError("no sessions given")
        pooled = np.concatenate([_flatten(s) for s in sessions], axis=0)
        if pooled.shape[0] < 2:
            raise ValueError("need at least 2 frames to fit a diffusion map")

        landmarks = self._choose_landmarks(pooled)
        sq = _sq_dists(landmarks, landmarks)

        # Drop landmarks the kernel cannot reach before building the operator:
        # a single isolated point costs a retained component (see
        # MIN_NEIGHBOR_MASS).
        keep, self.epsilon_ = self._prune_isolated(sq)
        self.n_landmarks_pruned_ = int((~keep).sum())
        if self.n_landmarks_pruned_:
            landmarks = landmarks[keep]
            sq = sq[np.ix_(keep, keep)]
        self.landmarks_ = landmarks

        kernel = np.exp(-sq / self.epsilon_)

        # Coifman alpha-normalisation: divide out the sampling density so the
        # operator reflects manifold geometry rather than where points happen
        # to be concentrated.
        degree = kernel.sum(axis=1)
        if self.alpha > 0:
            scale = degree ** self.alpha
            kernel = kernel / np.outer(scale, scale)

        # Eigendecompose the symmetric conjugate S = D^-1/2 W D^-1/2 rather
        # than the row-stochastic P = D^-1 W. They are similar, so the spectrum
        # is identical, but S is symmetric and eigh is stable where a general
        # eig on P is not.
        d = kernel.sum(axis=1)
        d_inv_sqrt = 1.0 / np.sqrt(np.maximum(d, 1e-300))
        sym = kernel * np.outer(d_inv_sqrt, d_inv_sqrt)
        sym = (sym + sym.T) / 2.0

        vals, vecs = np.linalg.eigh(sym)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]

        # Recover the eigenvectors of P.
        psi = vecs * d_inv_sqrt[:, None]

        # Drop the trivial stationary eigenvectors (lambda = 1). They carry no
        # information and silently cost a component if kept.
        keep = self._non_trivial(vals)
        self.n_trivial_eigenvectors_ = int((~keep).sum())
        if self.n_trivial_eigenvectors_ > 1:
            # A connected operator has exactly one. More means pruning did not
            # fully connect it, and every extra one is a component-indicator
            # spike occupying a slot a real mode should have had.
            warnings.warn(
                f"the diffusion operator is disconnected: "
                f"{self.n_trivial_eigenvectors_} eigenvalues equal 1.0 where a "
                f"connected operator has exactly 1, so the embedding is "
                f"driven by component membership rather than geometry. Raise "
                f"--epsilon, raise --n-landmarks, or check upstream alignment.",
                DegenerateOperatorWarning, stacklevel=2,
            )
        vals, psi = vals[keep], psi[:, keep]

        q = min(self.n_components, vals.size)
        self.eigenvalues_ = vals[:q]
        self.psi_ = psi[:, :q]
        self.n_components_ = q
        self._row_sums = d          # for the Nystrom extension
        self._degree = degree
        self._n_extended = 0
        self._n_degenerate_extensions = 0
        return self

    def _choose_landmarks(self, pooled):
        """Subsample the points the operator is built on.

        A full project is ~2.3M frames; an N x N kernel is 5e12 entries. Fitting
        on landmarks and extending by Nystrom is what makes this tractable, and
        it is also what lets `transform` work per recording after a pooled fit.
        Sampling is evenly spaced in time rather than random so every recording
        contributes proportionally.
        """
        n = pooled.shape[0]
        if n <= self.n_landmarks:
            return pooled.copy()
        idx = np.linspace(0, n - 1, self.n_landmarks).round().astype(int)
        return pooled[np.unique(idx)]

    def _resolve_epsilon(self, sq_dists):
        if self.epsilon != "auto":
            eps = float(self.epsilon)
            if eps <= 0:
                raise ValueError("epsilon must be > 0")
            return eps
        # A low quantile of pairwise squared distances: large enough to keep
        # the neighbourhood graph connected, small enough not to short-circuit
        # across folds of the manifold. Logged, because a diffusion distance is
        # meaningless without the bandwidth that produced it.
        off_diagonal = sq_dists[~np.eye(sq_dists.shape[0], dtype=bool)]
        eps = float(np.percentile(off_diagonal, self.EPSILON_PERCENTILE))
        if eps <= 0:
            positive = off_diagonal[off_diagonal > 0]
            eps = float(positive.min()) if positive.size else 1.0
        return eps

    def _prune_isolated(self, sq):
        """Drop landmarks with too little kernel mass to the rest.

        Returns (keep_mask, epsilon). Iterative because removing a landmark
        lowers the mass of whatever it was attached to, and the bandwidth is
        re-resolved each round since it is a quantile of the surviving
        pairwise distances.
        """
        keep = np.ones(sq.shape[0], dtype=bool)
        floor = max(self.n_components + 2, 3)
        eps = self._resolve_epsilon(sq)

        for _ in range(self.MAX_PRUNE_ROUNDS):
            idx = np.where(keep)[0]
            sub = sq[np.ix_(idx, idx)]
            eps = self._resolve_epsilon(sub)
            # Subtract the self-kernel (exp(0) = 1) so this counts neighbours,
            # not the point itself -- an isolated landmark scores 0, not 1.
            mass = np.exp(-sub / eps).sum(axis=1) - 1.0
            bad = mass < self.min_neighbor_mass
            if not bad.any():
                return keep, eps

            # Checked before the floor below, not after: wanting to drop this
            # many landmarks is itself the diagnosis, and silently keeping the
            # unpruned set would hand back the degenerate operator this method
            # exists to prevent.
            pruned = int((~keep).sum()) + int(bad.sum())
            if pruned > self.MAX_PRUNE_FRACTION * keep.size:
                raise ValueError(
                    f"pruning isolated landmarks would drop "
                    f"{pruned}/{keep.size} of them "
                    f"(>{self.MAX_PRUNE_FRACTION:.0%}), which means the kernel "
                    f"bandwidth is far too small for this data rather than "
                    f"that a few poses are outliers. Raise --epsilon or "
                    f"--n-landmarks."
                )

            if idx.size - int(bad.sum()) < floor:
                # Too few would survive to build an operator at all. Stop and
                # let the lambda = 1 warning describe what came out.
                return keep, eps

            keep[idx[bad]] = False

        # Rounds exhausted without converging. Re-resolve so the bandwidth
        # matches the landmarks actually kept rather than the set one round
        # earlier -- every other exit above already returns a consistent pair.
        idx = np.where(keep)[0]
        return keep, self._resolve_epsilon(sq[np.ix_(idx, idx)])

    @staticmethod
    def _non_trivial(vals):
        """Mask out every stationary eigenvector (lambda = 1).

        Deliberately not "the first constant one". On a connected operator
        lambda = 1 has multiplicity 1 and its eigenvector is constant, so the
        two descriptions agree. On a *disconnected* one the multiplicity equals
        the number of components and eigh returns an arbitrary basis of that
        eigenspace -- whose vectors are component indicators, not constants.
        Testing for constancy therefore recognised none of them and let all but
        one through, which is how a fragmented operator used to reach HDBSCAN
        disguised as an embedding.
        """
        return np.abs(vals - 1.0) >= _TRIVIAL_EIGENVALUE_TOL

    # ------------------------------------------------------------ transform

    def transform(self, session, batch_size=20_000):
        """Project one recording via the Nystrom extension. -> (T, q)

        psi_k(x) = lambda_k^-1 * sum_j P(x, j) psi_k(j)

        Batched so a multi-million-frame recording never materialises a full
        kernel against the landmarks at once.
        """
        x = _flatten(session)
        out = np.empty((x.shape[0], self.n_components_))
        for start in range(0, x.shape[0], batch_size):
            stop = min(start + batch_size, x.shape[0])
            out[start:stop] = self._extend(x[start:stop])
        return out * (self.eigenvalues_ ** self.diffusion_time)

    def _extend(self, x):
        kernel = np.exp(-_sq_dists(x, self.landmarks_) / self.epsilon_)

        # A query point far enough outside epsilon_'s neighborhood underflows
        # exp(-d/epsilon_) to exactly 0.0 against every landmark -- almost
        # always one mistracked keypoint in an otherwise normal frame (see the
        # module docstring), not a real jump, but real data hits this at
        # multi-million-frame scale. Counted so the caller can see how often.
        raw_degree = kernel.sum(axis=1)
        degenerate = raw_degree <= 0.0
        if degenerate.any():
            self._n_degenerate_extensions += int(degenerate.sum())
        self._n_extended += x.shape[0]

        if self.alpha > 0:
            # Use the landmark degrees so new points sit in the same
            # normalisation as the fitted operator. Floored before the power,
            # not after: 0.0 ** alpha is still 0.0, which would turn a
            # degenerate row's division into 0/0 = NaN instead of 0/eps = 0.
            own = np.maximum(raw_degree, 1e-300) ** self.alpha
            kernel = kernel / np.outer(own, self._degree ** self.alpha)

        row = kernel.sum(axis=1, keepdims=True)
        transition = kernel / np.maximum(row, 1e-300)

        safe = np.where(np.abs(self.eigenvalues_) < 1e-300, 1e-300,
                        self.eigenvalues_)
        return (transition @ self.psi_) / safe

    def transform_all(self, sessions):
        return [self.transform(s) for s in sessions]

    def fit_transform(self, sessions):
        return self.fit(sessions).transform_all(sessions)

    # --------------------------------------------------------------- report

    def spectrum_report(self):
        """Loggable summary.

        Bandwidth, alpha and diffusion time are included because a diffusion
        distance means nothing without them -- unlike a PCA component, the
        coordinate is not interpretable on its own.
        """
        return {
            "method": "diffusion",
            "backend": self.backend_,
            "n_components": self.n_components_,
            "alpha": self.alpha,
            "epsilon": self.epsilon_,
            "diffusion_time": self.diffusion_time,
            "n_landmarks": int(self.landmarks_.shape[0]),
            # Landmarks the kernel could not reach, dropped before the operator
            # was built, and how many stationary eigenvectors came out. The
            # second is the honest health check: 1 means connected, more means
            # the coordinates are component indicators (see _non_trivial).
            "n_landmarks_pruned": self.n_landmarks_pruned_,
            "n_trivial_eigenvectors": self.n_trivial_eigenvectors_,
            "eigenvalues": [float(v) for v in self.eigenvalues_],
            "spectral_gap": (
                float(self.eigenvalues_[0] - self.eigenvalues_[1])
                if self.eigenvalues_.size > 1 else None
            ),
            # Named to match PooledPCA's key so callers can log either without
            # branching; for diffusion this is eigenvalue share, not variance.
            "explained_variance": float(
                self.eigenvalues_.sum() / max(self.eigenvalues_.sum(), 1e-300)
            ),
            "n_nonzero_directions": int((self.eigenvalues_ > 1e-12).sum()),
            # Points Nystrom-extended from an all-zero landmark kernel -- see
            # _extend's comment. Zero until transform/transform_all is called.
            "n_degenerate_extensions": self._n_degenerate_extensions,
            "degenerate_extension_frac": (
                self._n_degenerate_extensions / self._n_extended
                if self._n_extended else 0.0
            ),
        }


def _sq_dists(a, b):
    """Pairwise squared Euclidean distances, clipped at zero."""
    d2 = (a ** 2).sum(1)[:, None] + (b ** 2).sum(1)[None, :] - 2.0 * a @ b.T
    return np.maximum(d2, 0.0)


def _flatten(session):
    x = np.asarray(session, dtype=np.float64)
    if x.ndim == 3:
        return x.reshape(x.shape[0], -1)
    if x.ndim == 2:
        return x
    raise ValueError(f"expected (T, K, 2) or (T, D), got {x.shape}")
