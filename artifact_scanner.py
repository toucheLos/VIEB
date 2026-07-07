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
    ".txt": "Text",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".png": "Image",
    ".jpg": "Image",
    ".jpeg": "Image",
    ".gif": "Image",
    ".webp": "Image",
    ".svg": "Image",
    ".html": "HTML",
    ".htm": "HTML",
    ".pdf": "PDF",
    ".mp4": "Video",
    ".avi": "Video",
    ".mov": "Video",
    ".npy": "NumPy",
    ".npz": "NumPy",
    ".gz": "NumPy",
    ".pkl": "Model",
    ".pt": "Model",
    ".pth": "Model",
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
    # Features and shared model artifacts
    ("features/", "Features"),
    ("shared/cluster_info.json", "Diagnostics"),
    ("shared/validation_report.json", "Diagnostics"),
    ("shared/run_manifest.json", "Metadata"),
    ("shared/preprocessor.pkl", "Metadata"),
    ("shared/umap_reducer.pkl", "Metadata"),
    ("shared/clusterer.pkl", "Metadata"),
    ("shared/", "Cluster Runs"),
    # State characterization
    ("characterization/state_occupancy.png", "State Characterization"),
    ("characterization/state_duration_summary.png", "State Characterization"),
    ("characterization/state_feature_profiles.png", "State Characterization"),
    ("characterization/state_feature_zscores.png", "State Characterization"),
    ("characterization/state_summary.csv", "State Characterization"),
    ("characterization/bouts.csv", "State Characterization"),
    ("characterization/labels_per_frame.csv", "State Characterization"),
    ("characterization/context_report.csv", "State Characterization"),
    ("characterization/state_", "State Characterization"),
    ("characterization/", "State Characterization"),
    ("validation/state_labels.csv", "States"),
    # Motifs
    ("comparison/motifs.csv", "Motifs"),
    ("comparison/motif_", "Motifs"),
    ("motifs/", "Motifs"),
    # Video Stories (sequence artifacts)
    ("sequences/", "Video Stories"),
    # Transitions
    ("comparison/transition_table.csv", "Transitions"),
    ("comparison/transition_", "Transitions"),
    # Comparison
    ("comparison/contrast_vector_comparison.png", "Comparison"),
    ("comparison/bout_duration_by_context.csv", "Comparison"),
    ("comparison/state_by_", "Comparison"),
    ("comparison/animal_trajectories.png", "Comparison"),
    ("comparison/", "Comparison"),
    # Diagnostics
    ("diagnostics/", "Diagnostics"),
    # Cluster Runs (saved run snapshots)
    ("runs/", "Cluster Runs"),
    # Quantification
    ("quantification/", "Quantification"),
    # Metadata / config
    ("metadata", "Metadata"),
    ("run_manifest.json", "Metadata"),
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
    if file_type in {"Model", "NumPy", "HDF5"}:
        return "Models / Binary", file_type

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
                    ext = os.path.splitext(fname)[1].lower()
                    file_type = _EXT_TYPE.get(ext, "Other")
                    category = (
                        "Video Stories"
                        if rel_path.replace("\\", "/").startswith("stories/")
                        else "Clips"
                    )
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
    "comparison/transition_table.csv",
    "comparison/state_by_day.png",
    "comparison/state_by_context.png",
    "comparison/state_by_experiment.png",
    "comparison/state_by_fear.png",
    "comparison/state_by_animal.png",
    "comparison/animal_trajectories.png",
    "comparison/motif_heatmap.png",
    "comparison/bout_duration_by_context.csv",
    "comparison/contrast_vector_comparison.png",
    "characterization/state_summary.csv",
    "characterization/state_occupancy.png",
    "characterization/state_duration_summary.png",
    "characterization/state_feature_profiles.png",
    "characterization/state_feature_zscores.csv",
    "characterization/state_feature_zscores.png",
    "characterization/context_report.csv",
    "comparison/motifs.csv",
    "motifs/motif_summary.csv",
    "motifs/motif_context_enrichment.csv",
    "motifs/motif_exemplars.csv",
    "sequences/video_story_bouts.csv",
    "sequences/video_stories.csv",
    "sequences/subject_journeys.csv",
    "diagnostics/cluster_overview.png",
    "diagnostics/umap_embedding_by_state.png",
    "diagnostics/cluster_diagnostics.json",
    "shared/cluster_info.json",
    "shared/run_manifest.json",
    "quantification/master_table.csv",
    "quantification/contrast_vectors.csv",
    "quantification/cohort_contrast_stats.csv",
    "quantification/contrast_bars.png",
    "quantification/contrast_heatmap.png",
    "quantification/contrast_magnitude.png",
    "quantification/contrast_scatter.png",
]

_PUBLICATION_GLOBS = [
    ("comparison", "*.png"),
    ("characterization", "*.png"),
    ("diagnostics", "*.png"),
    ("quantification", "*.png"),
    ("clips", "*.mp4"),
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
