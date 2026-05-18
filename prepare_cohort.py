#!/usr/bin/env python3
"""
prepare_cohort.py — Normalize a cohort Excel file and match against VIEB metadata.

Usage
-----
python prepare_cohort.py --input cohort.xlsx \\
                         --output cohort_normalized.csv \\
                         --metadata metadata.csv

Prints a match report showing how many animals overlap between the cohort
file and VIEB's metadata.csv.  Saves the normalized cohort as a plain CSV
that all downstream scripts (compare.py --cohort, gui.py) can read.
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

import cohort_loader


def main():
    parser = argparse.ArgumentParser(
        description="Normalize cohort Excel and match to VIEB metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input",    required=True,  metavar="FILE",
                        help="Path to cohort Excel or CSV file")
    parser.add_argument("--output",   required=True,  metavar="CSV",
                        help="Output path for normalized CSV")
    parser.add_argument("--metadata", default="metadata.csv", metavar="CSV",
                        help="Path to VIEB metadata.csv (default: metadata.csv)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"[ERROR] Input file not found: {args.input}")

    # Load and normalize cohort
    print(f"Loading cohort file: {args.input}")
    cohort_df = cohort_loader.load_cohort_excel(args.input)
    print(f"  {len(cohort_df)} animals loaded")

    summary = cohort_loader.get_cohort_summary(cohort_df)
    print(f"\n  Cohort summary:")
    print(f"    Animals:    {summary['n_animals']}")
    print(f"    Genotypes:  {summary['n_genotypes']}  {sorted(summary['by_genotype'].keys())}")
    print(f"    Age groups: {summary['n_age_groups']}  {sorted(summary['by_age_group'].keys())}")
    print(f"    Treatments: {summary['n_treatments']}  {sorted(summary['by_treatment'].keys())}")
    print(f"    Sexes:      {summary['n_sexes']}  {sorted(summary['by_sex'].keys())}")
    print(f"    Cohorts:    {summary['n_cohorts']}  (unique cohort_labels)")

    # Write normalized CSV
    cohort_df.to_csv(args.output, index=False)
    print(f"\nNormalized cohort saved to: {args.output}")

    # Match report against metadata
    if not os.path.exists(args.metadata):
        print(f"\n[SKIP] metadata.csv not found at '{args.metadata}' — skipping match report.")
        return

    meta = pd.read_csv(args.metadata)
    if "animal_id" not in meta.columns:
        print(f"\n[SKIP] metadata.csv has no 'animal_id' column — skipping match report.")
        return

    meta_ids   = set(pd.to_numeric(meta["animal_id"], errors="coerce").dropna().astype(int).unique())
    cohort_ids = set(cohort_df["animal_id"].unique())

    matched          = meta_ids & cohort_ids
    unmatched_meta   = sorted(meta_ids   - cohort_ids)
    unmatched_cohort = sorted(cohort_ids - meta_ids)

    print(f"\n--- Match report ---")
    print(f"  Animals in cohort file:   {len(cohort_ids)}")
    print(f"  Animals in VIEB metadata: {len(meta_ids)}")
    print(f"  Matched (both sides):     {len(matched)}")

    if unmatched_meta:
        print(f"\n  In metadata.csv but NOT in cohort ({len(unmatched_meta)} animals):")
        for aid in unmatched_meta:
            print(f"    {aid}")

    if unmatched_cohort:
        print(f"\n  In cohort but NOT in metadata ({len(unmatched_cohort)} animals):")
        for aid in unmatched_cohort:
            print(f"    {aid}")

    if not unmatched_meta and not unmatched_cohort:
        print("\n  All animal IDs match perfectly.")


if __name__ == "__main__":
    main()
