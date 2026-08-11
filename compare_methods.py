#!/usr/bin/env python
"""Load every VUS-1 run and put the arms in one table.

This is the consumer §7.3 requires: if an arm's output cannot be read here, it
has not emitted valid VUS-1 and it is not in the comparison.

It reads only `run_manifest.json` and `bouts.parquet`. It never imports a
segmenter, never re-derives a label, and never recomputes a bout — bout
construction is the harness's job precisely so that a difference in run-length
encoding cannot show up as a difference in bout duration, which is one of the
things being measured.

    python compare_methods.py --runs ~/vieb-runs
    python compare_methods.py --runs ~/vieb-runs --csv table.csv

Two refusals, both deliberate:

  * a run with `git_dirty: true` cannot be reproduced from its recorded sha, so
    it is flagged and excluded from `--strict` output.
  * arms whose recording ids overlap by less than `--min-overlap` are not
    comparable. Ids that disagree are how a join silently drops rows and still
    prints a table, so the check runs before any comparison is reported.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vieb.io.vus1 import BOUTS_NAME, MANIFEST_NAME, RunManifest, read_bouts  # noqa: E402


def find_runs(root: Path) -> list[Path]:
    """Every directory holding both VUS-1 files, at any depth under ``root``."""
    return sorted(
        p.parent for p in Path(root).expanduser().rglob(MANIFEST_NAME)
        if (p.parent / BOUTS_NAME).exists()
    )


def summarize(run_dir: Path) -> dict:
    """One row: what the arm was, and what it produced."""
    manifest = RunManifest.read(run_dir / MANIFEST_NAME)
    bouts = read_bouts(run_dir / BOUTS_NAME)

    durations = (bouts["end_frame"] - bouts["start_frame"]).to_numpy()
    recordings = bouts["recording_id"].astype(str).unique()
    assigned = int(durations.sum())
    counts = bouts.groupby("state")["end_frame"].count()

    occupancy = (
        bouts.assign(d=durations).groupby("state")["d"].sum() / assigned
        if assigned else pd.Series(dtype=float)
    )

    return {
        "run": run_dir.name,
        "representation": manifest.representation,
        "segmenter": manifest.segmenter,
        "arm": f"{manifest.representation}-{manifest.segmenter}",
        "dataset": manifest.dataset,
        "fps": manifest.fps,
        "n_states": manifest.n_states if manifest.n_states is not None else int(counts.size),
        "n_recordings": len(recordings),
        "n_bouts": int(len(bouts)),
        "unassigned_frac": manifest.unassigned_frac,
        "largest_state_frac": float(occupancy.max()) if len(occupancy) else float("nan"),
        # Seconds, not frames — the whole point of carrying fps in the manifest.
        "median_bout_s": float(np.median(durations) / manifest.fps) if len(durations) else float("nan"),
        "mean_bout_s": float(durations.mean() / manifest.fps) if len(durations) else float("nan"),
        "seed": manifest.seed,
        "git_sha": (manifest.git_sha or "")[:12],
        "git_dirty": manifest.git_dirty,
        "repr_hash": (manifest.repr_hash or "")[:19],
        "config_hash": (manifest.config_hash or "")[:19],
        "wall_clock_s": manifest.wall_clock_s,
        "_recording_ids": set(recordings),
        "_path": str(run_dir),
    }


def check_overlap(rows: list[dict], minimum: float) -> list[str]:
    """Every pair of arms must share recording ids, or they are not comparable."""
    problems = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            sa, sb = a["_recording_ids"], b["_recording_ids"]
            if not sa or not sb:
                continue
            overlap = len(sa & sb) / max(len(sa), len(sb))
            if overlap < minimum:
                problems.append(
                    f"{a['arm']} and {b['arm']} share only {overlap:.1%} of their "
                    f"recording ids ({len(sa & sb)} of {max(len(sa), len(sb))}). "
                    f"This is normalization drift, not missing data."
                )
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="~/vieb-runs",
                    help="run store root (default: ~/vieb-runs)")
    ap.add_argument("--csv", help="also write the table here")
    ap.add_argument("--min-overlap", type=float, default=0.90,
                    help="minimum recording-id overlap between any two arms")
    ap.add_argument("--strict", action="store_true",
                    help="exclude dirty-tree runs and fail on an overlap problem")
    args = ap.parse_args(argv)

    runs = find_runs(Path(args.runs))
    if not runs:
        print(f"no VUS-1 runs under {args.runs}", file=sys.stderr)
        return 1

    rows, unreadable = [], []
    for run in runs:
        try:
            rows.append(summarize(run))
        except Exception as exc:
            unreadable.append((run, f"{type(exc).__name__}: {exc}"))

    if unreadable:
        print(f"\n{len(unreadable)} run(s) did not load as VUS-1:", file=sys.stderr)
        for run, why in unreadable:
            print(f"  {run}: {why}", file=sys.stderr)

    dirty = [r for r in rows if r["git_dirty"]]
    if dirty:
        print(f"\n{len(dirty)} run(s) were produced from a dirty tree and cannot be "
              f"reproduced from their recorded sha:", file=sys.stderr)
        for r in dirty:
            print(f"  {r['run']} ({r['arm']})", file=sys.stderr)
        if args.strict:
            rows = [r for r in rows if not r["git_dirty"]]

    if not rows:
        print("nothing left to compare", file=sys.stderr)
        return 1

    problems = check_overlap(rows, args.min_overlap)
    if problems:
        print("\nrecording-id overlap problems:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        if args.strict:
            return 2

    display = [c for c in rows[0] if not c.startswith("_")]
    table = pd.DataFrame(rows)[display].sort_values(["representation", "segmenter"])

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print()
        print(table.to_string(index=False))

    if args.csv:
        table.to_csv(args.csv, index=False)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
