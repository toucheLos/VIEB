"""
export_results.py — CLI for browsing and exporting VIEB results
================================================================

Usage:
    python export_results.py --list
    python export_results.py --list --category Motifs
    python export_results.py --all --out vieb_results.zip
    python export_results.py --category motifs --out motif_results.zip
    python export_results.py --publication-bundle --out publication_bundle.zip
"""

import argparse
import os
import sys
import zipfile

import vieb_config as _vc
from artifact_scanner import (
    scan_artifacts, build_publication_bundle,
    format_size, format_time,
)


def _results_dir():
    return _vc.get_results_dir()


def _clips_dir():
    return _vc.get_clips_dir()


def cmd_list(category: str | None = None):
    clips = _clips_dir()
    clips_arg = clips if os.path.isdir(clips) else None
    artifacts = scan_artifacts(_results_dir(), clips_dir=clips_arg)
    if category:
        artifacts = [
            a for a in artifacts
            if a["category"].lower() == category.lower()
        ]

    if not artifacts:
        print("No artifacts found.")
        return

    categories = sorted(set(a["category"] for a in artifacts))
    print(f"{'Name':<40} {'Category':<14} {'Type':<8} {'Size':<10} {'Modified'}")
    print("-" * 95)
    for a in sorted(artifacts, key=lambda x: (x["category"], x["name"])):
        print(
            f"{a['name']:<40} {a['category']:<14} {a['file_type']:<8} "
            f"{format_size(a['size_bytes']):<10} {format_time(a['modified_ts'])}"
        )

    total_size = sum(a["size_bytes"] for a in artifacts)
    print(f"\n{len(artifacts)} files, {format_size(total_size)} total")
    if not category:
        print(f"Categories: {', '.join(categories)}")


def cmd_export_all(out_path: str):
    clips = _clips_dir()
    clips_arg = clips if os.path.isdir(clips) else None
    artifacts = scan_artifacts(_results_dir(), clips_dir=clips_arg)
    if not artifacts:
        sys.exit("No artifacts found.")

    skipped = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for a in artifacts:
            if not os.path.exists(a["abs_path"]):
                print(f"  [skip] {a['rel_path']} (file missing)")
                skipped += 1
                continue
            zf.write(a["abs_path"], a["rel_path"])
    written = len(artifacts) - skipped
    print(f"Exported {written} files to {out_path}")
    if skipped:
        print(f"  ({skipped} files skipped — missing on disk)")


def cmd_export_category(category: str, out_path: str):
    clips = _clips_dir()
    clips_arg = clips if os.path.isdir(clips) else None
    artifacts = scan_artifacts(_results_dir(), clips_dir=clips_arg)
    filtered = [
        a for a in artifacts
        if a["category"].lower() == category.lower()
    ]
    if not filtered:
        available = sorted(set(a["category"] for a in artifacts))
        sys.exit(
            f"No artifacts in category '{category}'.\n"
            f"Available: {', '.join(available) if available else '(none)'}"
        )

    skipped = 0
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for a in filtered:
            if not os.path.exists(a["abs_path"]):
                print(f"  [skip] {a['rel_path']} (file missing)")
                skipped += 1
                continue
            zf.write(a["abs_path"], a["rel_path"])
    written = len(filtered) - skipped
    print(f"Exported {written} {category} files to {out_path}")
    if skipped:
        print(f"  ({skipped} files skipped — missing on disk)")


def cmd_publication_bundle(out_path: str):
    build_publication_bundle(_results_dir(), out_path)
    print(f"Publication bundle saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Browse and export VIEB results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--list", action="store_true",
                        help="List all result artifacts")
    parser.add_argument("--category", type=str, default=None,
                        help="Filter by category (with --list or --out)")
    parser.add_argument("--all", action="store_true", dest="export_all",
                        help="Export all artifacts as ZIP")
    parser.add_argument("--publication-bundle", action="store_true",
                        help="Export curated publication bundle")
    parser.add_argument("--out", type=str, default=None,
                        help="Output ZIP path")
    args = parser.parse_args()

    if args.list:
        cmd_list(category=args.category)
    elif args.export_all:
        if not args.out:
            sys.exit("--out is required with --all")
        cmd_export_all(args.out)
    elif args.publication_bundle:
        if not args.out:
            sys.exit("--out is required with --publication-bundle")
        cmd_publication_bundle(args.out)
    elif args.category and args.out:
        cmd_export_category(args.category, args.out)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
