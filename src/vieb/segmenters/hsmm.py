"""The ``hsmm`` arm: AR emissions from jax_moseq, durations from ``_hsmm_negbin``.

This module is an adapter and a Gibbs loop, nothing else. The AR emission model,
its MNIW conjugate priors and the forward-backward are jax_moseq's and are called
unmodified; the loop below differs from ``jax_moseq.models.arhmm.resample_model``
in exactly three places, all of them the duration model:

===========================  ==========================================
``resample_hdp_transitions``  -> ``resample_transitions``: segment-level
                                 counts, ``kappa=0``, **zero diagonal**
``resample_discrete_stateseqs`` -> ``resample_stateseqs``: the NegBin
                                 embedding, then collapse
*(nothing)*                   -> ``resample_stay_probs`` + ``resample_r``
===========================  ==========================================

``resample_ar_params`` is called with the collapsed ``z`` and never learns that
the embedding exists.

The sticky-HMM baseline of §4.2/§4.3 is not reimplemented here either: it is
``jax_moseq.models.arhmm`` itself, so "the HSMM beats the sticky HMM" compares
against MoSeq's own sampler rather than against my reading of it.

Frames the model declines to label
----------------------------------
An AR(``nlags``) emission has no prediction for the first ``nlags`` frames of a
recording, so those are ``-1`` — the absence of a state, per the contract, not a
state. It is about 0.05% of Luna. keypoint-MoSeq pads them with the first
syllable instead; that is a display convenience, and copying it here would put
three fabricated frames at the head of every recording.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..data.dataset import PoseDataset
from ..registry import SEGMENTERS
from ._hsmm_negbin import (
    R_MAX,
    duration_summary,
    marginal_log_likelihood,
    negbin_mean,
    negbin_mode,
    resample_r,
    resample_stateseqs,
    resample_stay_probs,
    resample_transitions,
    segment_durations,
)
from .base import make_segmentation

#: The Luna keypoint-MoSeq baseline, from ``~/moseq/luna_demo/config.yml``.
#: Every one of these is matched so the comparison isolates the duration model.
BASELINE = {
    "num_states": 100,
    "latent_dim": 10,
    "nlags": 3,
    "alpha": 5.7,
    "gamma": 1000.0,
    "S_0_scale": 0.01,
    "K_0_scale": 10.0,
    "kappa_sticky": 1e6,
    "fit_sample_size": 100,
}


def _enable_x64() -> None:
    """kpms fits in float64; the duration counts need it too."""
    import jax

    jax.config.update("jax_enable_x64", True)


def batch_by_recording(X: np.ndarray, data: PoseDataset):
    """``(n_frames, d)`` -> ``(N, T_max, d)`` + mask, one row per recording.

    One row per recording is what makes every boundary guarantee in this arm
    structural rather than checked: a segment cannot span a boundary because the
    boundary is the end of the row, and an AR lag cannot cross one because
    ``get_lags`` never leaves the row.
    """
    X = np.asarray(X, dtype=np.float64)
    slices = [sl for _, sl in data.slices()]
    lengths = [sl.stop - sl.start for sl in slices]
    T = max(lengths)
    out = np.zeros((len(slices), T, X.shape[1]), dtype=np.float64)
    mask = np.zeros((len(slices), T), dtype=np.float64)
    for i, sl in enumerate(slices):
        n = sl.stop - sl.start
        out[i, :n] = X[sl]
        mask[i, :n] = 1.0
    return out, mask, slices


def _hypparams(num_states, latent_dim, nlags, alpha, gamma, kappa,
               S_0_scale, K_0_scale):
    from jax_moseq.models.arhmm.initialize import init_hyperparams

    return init_hyperparams(
        trans_hypparams={"num_states": num_states, "alpha": alpha,
                         "gamma": gamma, "kappa": kappa},
        ar_hypparams={"latent_dim": latent_dim, "nlags": nlags,
                      "S_0_scale": S_0_scale, "K_0_scale": K_0_scale},
    )


def fit_sticky_hmm(x, mask, *, seed=0, num_iters=100, **kw):
    """The baseline, run through ``jax_moseq.models.arhmm`` unmodified.

    Exists so §4.2 and §4.3 compare against MoSeq's own sampler.
    """
    import jax.random as jr
    from jax_moseq.models import arhmm

    _enable_x64()
    cfg = {**BASELINE, **kw}
    hyp = _hypparams(cfg["num_states"], cfg["latent_dim"], cfg["nlags"],
                     cfg["alpha"], cfg["gamma"], cfg["kappa_sticky"],
                     cfg["S_0_scale"], cfg["K_0_scale"])
    model = arhmm.init_model(data={"x": x, "mask": mask}, hypparams=hyp,
                             seed=jr.PRNGKey(seed))
    for _ in range(num_iters):
        model = arhmm.resample_model(data={"x": x, "mask": mask}, **model)
    return model


def fit_hsmm(x, mask, *, seed=0, num_iters=100, r_max=R_MAX,
             stay_prior=(1.0, 1.0), r_every=1, **kw):
    """Gibbs, with the duration model. Returns ``(params, z, diagnostics)``."""
    import jax.numpy as jnp
    import jax.random as jr
    from jax_moseq.models.arhmm.gibbs import resample_ar_params
    from jax_moseq.models.arhmm.initialize import init_ar_params
    from jax_moseq.utils.transitions import init_hdp_transitions

    _enable_x64()
    cfg = {**BASELINE, **kw}
    K, nlags = cfg["num_states"], cfg["nlags"]
    hyp = _hypparams(K, cfg["latent_dim"], nlags, cfg["alpha"], cfg["gamma"],
                     0.0, cfg["S_0_scale"], cfg["K_0_scale"])
    key = jr.PRNGKey(seed)

    key, k1, k2 = jr.split(key, 3)
    betas, pi = init_hdp_transitions(k1, **hyp["trans_hypparams"])
    pi = pi * (1.0 - jnp.eye(K))
    pi = pi / jnp.clip(pi.sum(axis=1, keepdims=True), 1e-12)
    Ab, Q = init_ar_params(k2, **hyp["ar_hypparams"])

    # Start geometric (r=1) so the duration model has to *earn* a mode away from
    # 1 rather than being handed one by the initialization. §4.3 depends on this.
    p = jnp.full(K, 0.9)
    r = jnp.ones(K, dtype=jnp.int32)

    history = []
    for it in range(num_iters):
        key, ks, kp, kr, kt, ka = jr.split(key, 6)
        z, z_emb = resample_stateseqs(ks, x, mask, Ab, Q, pi, p, r, r_max)
        p, n_stay, n_move = resample_stay_probs(
            kp, z_emb, mask, K, r_max, *stay_prior, nlags=nlags
        )
        if it % r_every == 0:
            states, durs = segment_durations(z, mask, nlags=nlags)
            r_new, p_new = resample_r(kr, durs, states, p, r, K, r_max,
                                      stay_prior=stay_prior)
            r, p = jnp.asarray(r_new), jnp.asarray(p_new)
        betas, pi = resample_transitions(
            kt, z, mask, betas, cfg["alpha"], cfg["gamma"], nlags=nlags
        )
        Ab, Q = resample_ar_params(
            ka, mask=mask, x=x, z=z, **hyp["ar_hypparams"]
        )
        history.append({
            "iter": it,
            "n_used": int(np.unique(np.asarray(z)).size),
            "r_mean": float(np.mean(np.asarray(r))),
            "p_mean": float(np.mean(np.asarray(p))),
        })

    params = {"pi": pi, "betas": betas, "Ab": Ab, "Q": Q, "p": p, "r": r}
    return params, z, {"history": history, "hypparams": hyp}


@SEGMENTERS.register("hsmm")
class HSMMSegmenter:
    """Explicit-duration AR-HSMM with a zero-diagonal transition matrix."""

    name = "hsmm"
    version = "1.0-negbin"

    def __init__(self, *, num_states=100, latent_dim=10, nlags=3, alpha=5.7,
                 gamma=1000.0, S_0_scale=0.01, K_0_scale=10.0, r_max=R_MAX,
                 num_iters=500, predict_iters=20, fit_sample_size=100,
                 stay_prior=(1.0, 1.0), r_every=1):
        self.num_states = int(num_states)
        self.latent_dim = int(latent_dim)
        self.nlags = int(nlags)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.S_0_scale = float(S_0_scale)
        self.K_0_scale = float(K_0_scale)
        self.r_max = int(r_max)
        self.num_iters = int(num_iters)
        self.predict_iters = int(predict_iters)
        self.fit_sample_size = int(fit_sample_size)
        self.stay_prior = tuple(stay_prior)
        self.r_every = int(r_every)
        self.params_ = None
        self.report_: dict = {}

    # -- contract ---------------------------------------------------------

    def _cfg(self) -> dict:
        return {
            "num_states": self.num_states, "latent_dim": self.latent_dim,
            "nlags": self.nlags, "alpha": self.alpha, "gamma": self.gamma,
            "S_0_scale": self.S_0_scale, "K_0_scale": self.K_0_scale,
        }

    def fit(self, X: np.ndarray, data: PoseDataset, *, seed: int = 0) -> None:
        """Fit on a subsample of recordings, matching MoSeq's ``FIT_SAMPLE_SIZE``."""
        sub = self._fit_subset(data, seed)
        x, mask, _ = batch_by_recording(np.asarray(X)[sub["rows"]], sub["data"])
        params, _, diag = fit_hsmm(
            x, mask, seed=seed, num_iters=self.num_iters, r_max=self.r_max,
            stay_prior=self.stay_prior, r_every=self.r_every, **self._cfg()
        )
        self.params_ = params
        self.seed_ = seed
        self.report_ = {
            "fit_recordings": sub["ids"],
            "n_fit_frames": int(mask.sum()),
            "history": diag["history"][-10:],
        }

    def predict(self, X: np.ndarray, data: PoseDataset):
        if self.params_ is None:
            raise RuntimeError("fit() must be called before predict()")

        import jax.random as jr

        x, mask, slices = batch_by_recording(X, data)
        pm = self.params_
        key = jr.PRNGKey(int(getattr(self, "seed_", 0)) + 1)
        for _ in range(max(1, self.predict_iters)):
            key, ks = jr.split(key)
            z, z_emb = resample_stateseqs(
                ks, x, mask, pm["Ab"], pm["Q"], pm["pi"], pm["p"], pm["r"],
                self.r_max,
            )
        z = np.asarray(z)

        labels = np.full(data.n_frames, -1, dtype=np.int32)
        for i, sl in enumerate(slices):
            n = sl.stop - sl.start
            # z is offset by nlags: z[i, t] labels frame t + nlags.
            labels[sl.start + self.nlags: sl.stop] = z[i, : n - self.nlags]

        states, durs = segment_durations(z, mask, nlags=self.nlags)
        extra = {
            "duration_summary": duration_summary(
                states, durs, self.num_states, data.fps
            ),
            "r": np.asarray(pm["r"]).tolist(),
            "p": np.asarray(pm["p"]).tolist(),
            "negbin_mode_frames": [
                negbin_mode(int(rk), float(pk))
                for rk, pk in zip(np.asarray(pm["r"]), np.asarray(pm["p"]))
            ],
            "negbin_mean_frames": [
                negbin_mean(int(rk), float(pk))
                for rk, pk in zip(np.asarray(pm["r"]), np.asarray(pm["p"]))
            ],
            "r_max": self.r_max,
            "unlabeled_reason": f"first {self.nlags} frames per recording (AR lag)",
            **self.report_,
        }
        return make_segmentation(labels, data, **extra)

    # -- diagnostics ------------------------------------------------------

    def held_out_log_likelihood(self, X: np.ndarray, data: PoseDataset) -> float:
        """Marginal log-likelihood of held-out data under the fitted HSMM."""
        x, mask, _ = batch_by_recording(X, data)
        pm = self.params_
        return marginal_log_likelihood(
            x, mask, pm["Ab"], pm["Q"], pm["pi"], pm["p"], pm["r"], self.r_max
        )

    def _fit_subset(self, data: PoseDataset, seed: int) -> dict:
        ids = list(data.recording_ids)
        if len(ids) <= self.fit_sample_size:
            keep = ids
        else:
            rng = np.random.default_rng(seed)
            keep = [ids[i] for i in sorted(
                rng.choice(len(ids), self.fit_sample_size, replace=False))]
        sub = data.subset(keep)
        rows = np.concatenate([
            np.arange(sl.start, sl.stop)
            for rid, sl in data.slices() if rid in set(keep)
        ])
        return {"data": sub, "ids": keep, "rows": rows}

    def get_params(self) -> dict:
        return {
            **self._cfg(),
            "r_max": self.r_max, "num_iters": self.num_iters,
            "predict_iters": self.predict_iters,
            "fit_sample_size": self.fit_sample_size,
            "stay_prior": list(self.stay_prior), "r_every": self.r_every,
            "duration": "negbin", "transition_diagonal": "zero",
            "kappa": None,
            "r_sampling": "metropolis on integer grid 1..r_max, +/-1 proposal "
                          "with Hastings correction at the edges",
        }

    def save(self, path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        pm = self.params_ or {}
        np.savez(path / "params.npz", **{k: np.asarray(v) for k, v in pm.items()})
        (path / "hsmm.json").write_text(json.dumps(
            {"params": self.get_params(), "report": self.report_,
             "version": self.version, "seed": int(getattr(self, "seed_", 0))},
            indent=2, default=str))

    @classmethod
    def load(cls, path):
        path = Path(path)
        meta = json.loads((path / "hsmm.json").read_text())
        kw = {k: v for k, v in meta["params"].items()
              if k not in {"duration", "transition_diagonal", "kappa",
                           "r_sampling"}}
        kw["stay_prior"] = tuple(kw.get("stay_prior", (1.0, 1.0)))
        seg = cls(**kw)
        with np.load(path / "params.npz") as f:
            seg.params_ = {k: f[k] for k in f.files}
        seg.report_ = meta.get("report", {})
        seg.seed_ = int(meta.get("seed", 0))
        return seg
