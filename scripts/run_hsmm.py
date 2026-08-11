#!/usr/bin/env python
"""Run the ``hsmm`` arm end to end and emit VUS-1.

    python scripts/run_hsmm.py --limit 6 --num-iters 50    # smoke
    python scripts/run_hsmm.py --num-iters 500             # the real fit

Representation and segmenter are both resolved by name through the registry, so
this script contains no model code and no scoring code — ``run_arm`` owns the
manifest, the bout encoding and the run directory, exactly as it does for the
other arms. That is the point: an arm that scores itself is not comparable to one
that does not.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vieb.compare.runner import ArmSpec, run_arm  # noqa: E402
from vieb.data.loaders import load_dataset  # noqa: E402

DEFAULT_POSE = "~/dlc-training/raw_videos"
DEFAULT_MOSEQ = "~/moseq/luna_demo"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pose", default=DEFAULT_POSE)
    ap.add_argument("--moseq", default=DEFAULT_MOSEQ)
    ap.add_argument("--store", default="~/vieb-runs")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--limit", type=int, default=None,
                    help="use only the first N recordings (smoke tests)")
    ap.add_argument("--latent-dim", type=int, default=10)
    ap.add_argument("--num-states", type=int, default=100)
    ap.add_argument("--num-iters", type=int, default=500)
    ap.add_argument("--predict-iters", type=int, default=20)
    ap.add_argument("--fit-sample-size", type=int, default=100)
    ap.add_argument("--r-max", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="gpu")
    args = ap.parse_args(argv)

    data, report = load_dataset(
        Path(args.pose).expanduser(), fps=args.fps, limit=args.limit,
        dataset="luna",
    )
    print(f"loaded {data.n_recordings} recordings / {data.n_frames:,} frames")
    print(json.dumps({k: v for k, v in report.items()
                      if not isinstance(v, (list, dict))}, indent=2))

    spec = ArmSpec(
        representation="moseq_latent",
        segmenter="hsmm",
        representation_params={"source": args.moseq, "latent_dim": args.latent_dim},
        segmenter_params={
            "num_states": args.num_states, "latent_dim": args.latent_dim,
            "num_iters": args.num_iters, "predict_iters": args.predict_iters,
            "fit_sample_size": args.fit_sample_size, "r_max": args.r_max,
        },
        seed=args.seed,
    )
    result = run_arm(spec, data, store=Path(args.store).expanduser(),
                     device=args.device)

    print(f"\nwrote {result.run_dir}")
    print(f"  n_states        {result.segmentation.n_states}")
    print(f"  unassigned_frac {result.segmentation.unassigned_frac:.6f}")
    print(f"  wall_clock_s    {result.wall_clock_s:.1f}")

    ds = result.segmentation.extra.get("duration_summary", [])
    occupied = [d for d in ds if d.get("n_bouts", 0) > 0]
    off_one = [d for d in occupied if not d.get("mode_is_one_frame", True)]
    print(f"\n  states with bouts        {len(occupied)}")
    print(f"  duration mode off 1 frame {len(off_one)} / {len(occupied)}")
    for d in sorted(occupied, key=lambda x: -x["n_bouts"])[:10]:
        print(f"    state {d['state']:3d}  n={d['n_bouts']:7d}  mode={d['mode_frames']:5d}f "
              f"({d['mode_s']:.2f}s)  median={d['median_frames']:7.1f}f  cv={d['cv']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
