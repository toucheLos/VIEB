"""PCA pooled across every recording in a project.

The pooling is the whole point, and it is a correctness requirement rather than
a performance one. Fit per recording and you get (mu_r, U_r) per animal, so two
frames with *identical* aligned pose map to *different* coordinates unless the
bases happen to coincide. Cluster labels would then name a different region of
pose space in every video and no cross-animal comparison would mean anything.
One global chart makes equality of pose imply equality of coordinates -- the
formal content of "a state must mean the same thing everywhere".

The API takes a list of per-recording arrays for that reason: there is no way to
call `fit` on a single recording and get something the rest of the pipeline will
accept as a project-level basis.
"""

from __future__ import annotations

import numpy as np


class PooledPCA:
    """Pooled PCA over aligned pose.

    Attributes after `fit`:
      mean_             (D,)      pooled mean
      components_       (q, D)    retained eigenvectors, descending variance
      eigenvalues_      (D,)      full spectrum, descending
      n_components_     int       q, selected by variance threshold
      explained_variance_ratio_   per-component fraction of total variance
    """

    def __init__(self, var_threshold=0.95, max_components=None):
        if not 0.0 < var_threshold <= 1.0:
            raise ValueError("var_threshold must be in (0, 1]")
        self.var_threshold = float(var_threshold)
        self.max_components = max_components

    def fit(self, sessions):
        """Fit on frames pooled across all recordings.

        sessions : list of (T_r, K, 2) or (T_r, D) arrays
        """
        if not sessions:
            raise ValueError("no sessions given")
        pooled = np.concatenate([_flatten(s) for s in sessions], axis=0)
        if pooled.shape[0] < 2:
            raise ValueError("need at least 2 frames to fit PCA")

        self.mean_ = pooled.mean(axis=0)
        centered = pooled - self.mean_
        cov = np.cov(centered.T)
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]

        # eigh can return small negatives for the null directions.
        vals = np.clip(vals, 0.0, None)
        total = vals.sum()
        self.eigenvalues_ = vals
        self.explained_variance_ratio_ = (
            vals / total if total > 0 else np.zeros_like(vals)
        )

        cumulative = np.cumsum(self.explained_variance_ratio_)
        q = int(np.searchsorted(cumulative, self.var_threshold) + 1)
        q = min(q, len(vals))
        if self.max_components is not None:
            q = min(q, int(self.max_components))

        # Never retain a numerically-null direction, whatever the threshold
        # asks for: with confidence weighting these hold likelihood noise, and
        # they are the directions whitening would amplify without bound.
        nonzero = int((self.explained_variance_ratio_ > 1e-12).sum())
        q = max(1, min(q, nonzero))

        self.n_components_ = q
        self.components_ = vecs[:, :q].T
        return self

    def transform(self, session):
        """Project one recording onto the shared basis. -> (T, q)"""
        x = _flatten(session)
        return (x - self.mean_) @ self.components_.T

    def transform_all(self, sessions):
        return [self.transform(s) for s in sessions]

    def fit_transform(self, sessions):
        return self.fit(sessions).transform_all(sessions)

    def spectrum_report(self):
        """Loggable summary. The brief requires the component count be logged;
        the spectrum is logged with it so the choice is auditable rather than a
        magic number."""
        return {
            "n_components": self.n_components_,
            "var_threshold": self.var_threshold,
            "explained_variance": float(
                self.explained_variance_ratio_[: self.n_components_].sum()
            ),
            "eigenvalues": [float(v) for v in self.eigenvalues_],
            "explained_variance_ratio": [
                float(v) for v in self.explained_variance_ratio_
            ],
            "n_nonzero_directions": int(
                (self.explained_variance_ratio_ > 1e-12).sum()
            ),
        }


def _flatten(session):
    x = np.asarray(session, dtype=np.float64)
    if x.ndim == 3:
        return x.reshape(x.shape[0], -1)
    if x.ndim == 2:
        return x
    raise ValueError(f"expected (T, K, 2) or (T, D), got {x.shape}")
