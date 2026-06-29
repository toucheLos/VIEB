"""
artifact_scanner.py — Scan and categorize VIEB result files
============================================================
Pure Python module (no Qt). Used by the Artifacts browser UI
and the export_results.py CLI.
"""

import fnmatch
import os
import zipfile
from datetime import datetime


_EXT_TYPE = {
    ".csv": "CSV",
    ".tsv": "CSV",
    ".json": "JSON",
    ".png": "Image",
    ".jpg": "Image",
    ".jpeg": "Image",
    ".svg": "Image",
    ".pdf": "PDF",
    ".mp4": "Video",
    ".avi": "Video",
    ".mov": "Video",
    ".npy": "NumPy",
    ".gz": "NumPy",
    ".pkl": "Model",
    ".pt": "Model",
    ".h5": "HDF5",
    ".hdf5": "HDF5",
    ".xlsx": "Excel",
    ".xls": "Excel",
}

# Ordered list — first match wins.  Longer/more-specific patterns come first.
_CATEGORY_RULES: list[tuple[str, str]] = [
    # Summary
    ("comparison/summary_table.csv", "Summary"),
    ("comparison/animal_scalars.csv", "Summary"),
    ("metadata_schema_report.json", "Summary"),
    # State characterization
    ("characterization/state_occupancy.png", "State Characterization"),
    ("characterization/state_summary.csv", "State Characterization"),
    ("characterization/state_exemplars.csv", "State Characterization"),
    ("characterization/labels_per_frame.csv", "State Characterization"),
    ("characterization/context_report.csv", "State Characterization"),
    ("characterization/state_", "State Characterization"),
    ("validation/state_labels.csv", "States"),
    # Bouts
    ("characterization/bouts.csv", "Bouts"),
    ("motifs/bouts.csv", "Bouts"),
    # Motifs
    ("comparison/motifs.csv", "Motifs"),
    ("comparison/motif_", "Motifs"),
    ("motifs/", "Motifs"),
    # Transitions
    ("comparison/transition_", "Transitions"),
    # Comparison
    ("comparison/contrast_vector_comparison.png", "Comparison"),
    ("comparison/state_by_", "Comparison"),
    # Diagnostics
    ("diagnostics/", "Diagnostics"),
    ("shared/cluster_info.json", "Diagnostics"),
    ("shared/run_manifest.json", "Diagnostics"),
    ("shared/validation_report.json", "Diagnostics"),
    # Metadata / config
    ("shared/preprocessor.pkl", "Metadata"),
    ("shared/umap_reducer.pkl", "Metadata"),
    ("shared/clusterer.pkl", "Metadata"),
    ("features/index.json", "Metadata"),
]

_SKIP_PATTERNS = ("_labels.npy", "_probs.npy", "_features.npy")


def categorize_file(rel_path: str) -> tuple[str, str]:
    """Return (category, file_type) for a relative path under results/."""
    normalized = rel_path.replace("\\", "/")
    ext = os.path.splitext(normalized)[1].lower()
    file_type = _EXT_TYPE.get(ext, "Other")

    for pattern, category in _CATEGORY_RULES:
        if pattern in normalized:
            return category, file_type

    if file_type == "Image":
        return "Plots", file_type
    if file_type == "Video":
        return "Clips", file_type

    return "Raw Tables", file_type


def scan_artifacts(
    results_dir: str,
    include_bulk: bool = False,
    clips_dir: str | None = None,
) -> list[dict]:
    """Walk results_dir (and optionally clips_dir) and return file metadata.

    By default skips per-video label/prob/feature .npy files
    (thousands of files). Set include_bulk=True to include them.
    """
    artifacts: list[dict] = []
    results_dir = os.path.abspath(results_dir)

    dirs_to_scan: list[tuple[str, str]] = []
    if os.path.isdir(results_dir):
        dirs_to_scan.append((results_dir, results_dir))
    if clips_dir:
        clips_abs = os.path.abspath(clips_dir)
        if os.path.isdir(clips_abs) and clips_abs != results_dir:
            dirs_to_scan.append((clips_abs, clips_abs))

    for scan_root, base_for_rel in dirs_to_scan:
        for dirpath, _dirnames, filenames in os.walk(scan_root):
            for fname in sorted(filenames):
                abs_path = os.path.join(dirpath, fname)
                rel_path = os.path.relpath(abs_path, base_for_rel)

                if not include_bulk and any(rel_path.endswith(p) for p in _SKIP_PATTERNS):
                    continue

                if scan_root != results_dir:
                    category = "Clips"
                    ext = os.path.splitext(fname)[1].lower()
                    file_type = _EXT_TYPE.get(ext, "Other")
                    rel_path = os.path.join("clips", rel_path)
                else:
                    category, file_type = categorize_file(rel_path)

                try:
                    stat = os.stat(abs_path)
                    size_bytes = stat.st_size
                    modified_ts = stat.st_mtime
                except OSError:
                    size_bytes = 0
                    modified_ts = 0.0

                artifacts.append({
                    "name": fname,
                    "category": category,
                    "file_type": file_type,
                    "size_bytes": size_bytes,
                    "modified_ts": modified_ts,
                    "rel_path": rel_path,
                    "abs_path": abs_path,
                })

    return artifacts


_PUBLICATION_FILES = [
    "comparison/summary_table.csv",
    "characterization/state_summary.csv",
    "characterization/state_exemplars.csv",
    "characterization/state_occupancy.png",
    "comparison/motifs.csv",
    "comparison/bout_duration_by_context.csv",
    "comparison/transition_table.csv",
    "comparison/contrast_vector_comparison.png",
    "motifs/motif_summary.csv",
    "motifs/motif_context_enrichment.csv",
    "motifs/motif_exemplars.csv",
    "diagnostics/cluster_overview.png",
    "diagnostics/umap_embedding_by_state.png",
    "diagnostics/cluster_diagnostics.json",
    "shared/cluster_info.json",
    "shared/run_manifest.json",
]

_PUBLICATION_GLOBS = [
    ("comparison", "*.png"),
    ("diagnostics", "*.png"),
]


def build_publication_bundle(
    results_dir: str,
    out_path: str,
    metadata_csv: str | None = None,
) -> None:
    """Create a ZIP with curated publication-ready files.

    Skips files that don't exist — never errors on missing outputs.
    """
    results_dir = os.path.abspath(results_dir)
    added: set[str] = set()

    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for rel in _PUBLICATION_FILES:
            abs_p = os.path.join(results_dir, rel)
            if os.path.exists(abs_p):
                zf.write(abs_p, rel)
                added.add(rel)

        for subdir, pattern in _PUBLICATION_GLOBS:
            full_dir = os.path.join(results_dir, subdir)
            if not os.path.isdir(full_dir):
                continue
            for fname in sorted(os.listdir(full_dir)):
                if fnmatch.fnmatch(fname, pattern):
                    rel = os.path.join(subdir, fname).replace("\\", "/")
                    if rel not in added:
                        zf.write(os.path.join(full_dir, fname), rel)
                        added.add(rel)

        if metadata_csv and os.path.exists(metadata_csv):
            zf.write(metadata_csv, "metadata.csv")
        else:
            project_root = os.path.dirname(results_dir)
            meta_path = os.path.join(project_root, "metadata.csv")
            if os.path.exists(meta_path):
                zf.write(meta_path, "metadata.csv")


def format_size(size_bytes: int) -> str:
    """Human-readable file size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def format_time(ts: float) -> str:
    """Format a timestamp to readable string."""
    if ts <= 0:
        return ""
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
