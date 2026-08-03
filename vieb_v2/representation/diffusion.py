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

Interface mirrors PooledPCA so the two are swappable at the same pipeline slot.
"""

from __future__ import annotations

import numpy as np

_TRIVIAL_EIGENVALUE_TOL = 1e-8


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

    def __init__(self, n_components=8, alpha=1.0, epsilon="auto",
                 diffusion_time=1, n_landmarks=3000, random_state=0,
                 use_gpu=False):
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

        self.landmarks_ = self._choose_landmarks(pooled)
        sq = _sq_dists(self.landmarks_, self.landmarks_)
        self.epsilon_ = self._resolve_epsilon(sq)

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

        # Drop the trivial stationary eigenvector (lambda = 1, psi constant).
        # It carries no information and silently costs a component if kept.
        keep = self._non_trivial(vals, psi)
        vals, psi = vals[keep], psi[:, keep]

        q = min(self.n_components, vals.size)
        self.eigenvalues_ = vals[:q]
        self.psi_ = psi[:, :q]
        self.n_components_ = q
        self._row_sums = d          # for the Nystrom extension
        self._degree = degree
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

    @staticmethod
    def _non_trivial(vals, psi):
        """Mask out the constant stationary eigenvector."""
        keep = np.ones(vals.size, dtype=bool)
        for i in range(vals.size):
            if abs(vals[i] - 1.0) < _TRIVIAL_EIGENVALUE_TOL:
                column = psi[:, i]
                spread = column.std() / (np.abs(column).mean() + 1e-300)
                if spread < 1e-6:      # constant up to numerical noise
                    keep[i] = False
                    break
        return keep

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

        if self.alpha > 0:
            # Use the landmark degrees so new points sit in the same
            # normalisation as the fitted operator.
            own = kernel.sum(axis=1) ** self.alpha
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
