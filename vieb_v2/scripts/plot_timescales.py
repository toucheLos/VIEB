"""Plot the implied-timescale curves that decide this branch.

Two guides are drawn on every panel because both artifact regions have to be
visible for the plateau claim to mean anything:

  t = tau      the resolution floor. A timescale at or below its own lag is not
               measured, it is the near-identity artifact -- at small lag every
               eigenvalue degenerates toward 1 and the curve rises with tau by
               construction.
  horizon      the longest timescale these recordings can carry. Beyond it the
               rows of P are noisy copies of pi and every timescale grows
               linearly again, for reasons that have nothing to do with the
               animal.

A plateau, if there is one, lives between the two. Drawing them means a reader
can check the claim rather than take it.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _arr(x):
    return np.array([[np.nan if v is None else v for v in row] for row in x],
                    dtype=float)


def panel(ax, res, n_processes, title):
    taus = np.asarray(res["taus_s"], dtype=float)
    its = _arr(res["its_s"])
    horizon = res.get("horizon_s")

    ax.fill_between(taus, taus, taus.max() * 1e3, color="0.92", zorder=0)
    ax.plot(taus, taus, color="0.55", lw=1, ls="--", zorder=1,
            label=r"$t=\tau$ (resolution floor)")
    if horizon:
        ax.axhline(horizon, color="0.55", lw=1, ls=":", zorder=1,
                   label=f"horizon ({horizon:.0f}s)")

    boot = res.get("bootstrap")
    if boot:
        b_tau = np.asarray(boot["taus_s"], dtype=float)
        lo, hi = _arr(boot["lo"]), _arr(boot["hi"])

    colors = plt.cm.viridis(np.linspace(0, 0.85, n_processes))
    for p in range(1, n_processes + 1):
        if p >= its.shape[1]:
            break
        ax.plot(taus, its[:, p], "-o", ms=3, lw=1.4, color=colors[p - 1],
                label=f"$t_{{{p + 1}}}$", zorder=3)
        if boot and p < lo.shape[1]:
            ok = np.isfinite(lo[:, p]) & np.isfinite(hi[:, p])
            if ok.any():
                ax.fill_between(b_tau[ok], lo[ok, p], hi[ok, p],
                                color=colors[p - 1], alpha=0.25, lw=0, zorder=2)

    # Shade only the widest plateau. Overlaying one band per process stacks
    # alpha into a gradient that reads as structure in the data.
    gate = res.get("gate", {})
    widest = None
    for e in gate.get("processes", {}).values():
        pl = e.get("plateau")
        if pl and (widest is None or pl["tau_ratio"] > widest["tau_ratio"]):
            widest = pl
    if widest:
        ax.axvspan(widest["tau_lo_s"], widest["tau_hi_s"], color="tab:orange",
                   alpha=0.13, zorder=0, label="widest plateau")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"lag $\tau$ (s)")
    ax.set_ylabel("implied timescale (s)")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(taus.min(), taus.max())
    finite = its[:, 1:n_processes + 1]
    finite = finite[np.isfinite(finite)]
    if finite.size:
        ax.set_ylim(max(finite.min() * 0.5, taus.min() * 0.5),
                    finite.max() * 2)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9, ncol=2)


def has_ck(res):
    """Is there anything to draw in a CK panel for this arm?

    When the gate finds no plateau there is no tau*, so no CK test was run and
    the panel would be an empty box. Asked before the figure is laid out, so
    the row can be dropped rather than reserved and left blank.
    """
    for key, field in (("ck", "err"), ("holdout_ck", "err_holdout")):
        block = res.get(key)
        if block and block.get("ok"):
            if any(r.get("ok") and field in r for r in block["rows"]):
                return True
    return False


def ck_panel(ax, res, title):
    ck, held = res.get("ck"), res.get("holdout_ck")
    drew = False
    if ck and ck.get("ok"):
        rows = [r for r in ck["rows"] if r.get("ok")]
        if rows:
            ax.plot([r["n"] for r in rows], [r["err"] for r in rows],
                    "-o", ms=4, color="tab:blue", label="pooled")
            drew = True
    if held and held.get("ok"):
        rows = [r for r in held["rows"] if r.get("ok")]
        if rows:
            ax.plot([r["n"] for r in rows], [r["err_holdout"] for r in rows],
                    "-s", ms=4, color="tab:red", label="held out")
            drew = True
    if not drew:
        ax.text(0.5, 0.5, "no Chapman-Kolmogorov test\n(no plateau, no tau*)",
                ha="center", va="center", transform=ax.transAxes, fontsize=8,
                color="0.4")
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        ax.set_xlabel(r"$n$  (comparing $P(n\tau)$ with $P(\tau)^n$)")
        ax.set_ylabel(r"$\pi$-weighted TV error")
        ax.axhline(0.05, color="0.6", lw=1, ls=":")
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=7)
        ax.grid(alpha=0.25, lw=0.4)
    ax.set_title(title, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory holding the json")
    ap.add_argument("--tags", default="channels,pose_only")
    ap.add_argument("--processes", type=int, default=5)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()

    tags = [t for t in args.tags.split(",") if t]
    found = []
    for t in tags:
        p = os.path.join(args.out, f"timescales_{t}.json")
        if os.path.exists(p):
            with open(p) as fh:
                found.append((t, json.load(fh)))
        else:
            print(f"  (missing {p})")
    if not found:
        raise SystemExit("no timescales_*.json found")

    # Drop the CK row entirely when no arm has a tau*. A row of empty boxes
    # spends half the figure saying nothing, and the verdict it would have
    # carried belongs on the curve that produced it.
    ck_row = any(has_ck(res) for _, res in found)
    nrows = 2 if ck_row else 1
    fig, axes = plt.subplots(nrows, len(found),
                             figsize=(6.0 * len(found), 4.2 * nrows),
                             squeeze=False)
    for j, (tag, res) in enumerate(found):
        label = ("pose PCs + restored channels" if tag == "channels"
                 else "pose PCs only (control)" if tag == "pose_only" else tag)
        n = res.get("n_states")
        verdict = res.get("gate", {}).get("verdict", "")
        title = f"{label}\n{n} microstates, {res.get('n_recordings')} recordings"
        if not ck_row and verdict:
            title += "\n" + textwrap.fill(verdict, 52)
        panel(axes[0][j], res, args.processes, title)
        if ck_row:
            ck_panel(axes[1][j], res,
                     "Chapman-Kolmogorov\n" + textwrap.fill(verdict, 52))

    fig.suptitle("Implied timescales of the transfer operator\n"
                 "reversibilized, so these are an upper bound "
                 "for irreversible dynamics", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = args.save or os.path.join(args.out, "implied_timescales.png")
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
