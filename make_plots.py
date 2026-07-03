#!/usr/bin/env python3
"""Generate dynamic VIEB report plots from current pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import vieb_config as _vc
from analysis_design import detect_analysis_design
from report_plots import generate_mode_driven_plots


def _csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def main() -> None:
    results = Path(_vc.get_results_dir())
    summary = _csv(results / "comparison" / "summary_table.csv")
    if summary is None:
        raise SystemExit("Missing results/comparison/summary_table.csv. Run compare.py --report first.")

    design_path = results / "analysis_design.json"
    if design_path.exists():
        design = json.loads(design_path.read_text(encoding="utf-8"))
    else:
        design = detect_analysis_design(summary, _vc._load_config())
        design_path.write_text(json.dumps(design, indent=2, default=str), encoding="utf-8")
        print(f"Analysis design saved: {design_path}")

    transition = _csv(results / "comparison" / "transition_table.csv")
    bouts = _csv(results / "characterization" / "bouts.csv")
    generate_mode_driven_plots(summary, transition, bouts, design, results)


if __name__ == "__main__":
    main()
