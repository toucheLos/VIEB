import json
import os
import sys
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from artifact_scanner import (
    scan_artifacts, categorize_file, build_publication_bundle, format_size,
)


# ──────────────────────────────────────────── categorize_file ──

def test_categorize_csv_png_json_mp4():
    assert categorize_file("comparison/summary_table.csv") == ("Summary", "CSV")
    assert categorize_file("comparison/motifs.csv") == ("Motifs", "CSV")
    assert categorize_file("diagnostics/cluster_overview.png") == ("Diagnostics", "Image")
    assert categorize_file("diagnostics/umap_embedding_by_state.png") == ("Diagnostics", "Image")
    assert categorize_file("motifs/motif_sequences.csv") == ("Motifs", "CSV")
    assert categorize_file("sequences/video_stories.csv") == ("Video Stories", "CSV")
    assert categorize_file("comparison/transition_table.csv") == ("Transitions", "CSV")
    assert categorize_file("comparison/state_by_context.png") == ("Comparison", "Image")
    assert categorize_file("comparison/contrast_vector_comparison.png") == ("Comparison", "Image")
    assert categorize_file("shared/cluster_info.json") == ("Diagnostics", "JSON")
    assert categorize_file("characterization/bouts.csv") == ("State Characterization", "CSV")


def test_categorize_states():
    assert categorize_file("characterization/state_summary.csv") == ("State Characterization", "CSV")
    assert categorize_file("characterization/context_report.csv") == ("State Characterization", "CSV")
    assert categorize_file("characterization/state_occupancy.png") == ("State Characterization", "Image")
    assert categorize_file("validation/state_labels.csv") == ("States", "CSV")


def test_categorize_cluster_runs():
    assert categorize_file("runs/run_001_20260101_0000_mcs2000_umap10/cluster_info.json") == ("Cluster Runs", "JSON")
    assert categorize_file("runs/run_001_20260101_0000_mcs2000_umap10/clusterer.pkl") == ("Cluster Runs", "Model")
    assert categorize_file("runs/run_001_20260101_0000_mcs2000_umap10/run_manifest.json") == ("Cluster Runs", "JSON")


def test_categorize_metadata():
    assert categorize_file("shared/preprocessor.pkl") == ("Metadata", "Model")
    assert categorize_file("features/index.json") == ("Features", "JSON")


def test_categorize_video_as_clips():
    assert categorize_file("some_dir/clip.mp4")[0] == "Clips"
    assert categorize_file("some_dir/clip.avi")[0] == "Clips"
    assert categorize_file("some_dir/clip.mov")[0] == "Clips"


def test_categorize_mov_extension():
    cat, ftype = categorize_file("clips/state_0/longest_01.mov")
    assert ftype == "Video"
    assert cat == "Clips"


# ───────────────────────────────────────────── scan_artifacts ──

def test_scan_finds_files(tmp_path):
    (tmp_path / "comparison").mkdir()
    (tmp_path / "comparison" / "summary_table.csv").write_text("a,b\n1,2")
    (tmp_path / "characterization").mkdir()
    (tmp_path / "characterization" / "state_occupancy.png").write_bytes(b"PNG")
    (tmp_path / "diagnostics").mkdir()
    (tmp_path / "diagnostics" / "cluster_diagnostics.json").write_text("{}")

    artifacts = scan_artifacts(str(tmp_path))
    assert len(artifacts) == 3
    categories = {a["category"] for a in artifacts}
    assert "Summary" in categories
    assert "State Characterization" in categories
    assert "Diagnostics" in categories


def test_scan_skips_bulk_npy(tmp_path):
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "video1_labels.npy").write_bytes(b"x")
    (tmp_path / "shared" / "video1_probs.npy").write_bytes(b"x")
    (tmp_path / "shared" / "cluster_info.json").write_text("{}")

    artifacts = scan_artifacts(str(tmp_path), include_bulk=False)
    names = [a["name"] for a in artifacts]
    assert "cluster_info.json" in names
    assert "video1_labels.npy" not in names


def test_scan_includes_bulk_when_requested(tmp_path):
    (tmp_path / "shared").mkdir()
    (tmp_path / "shared" / "video1_labels.npy").write_bytes(b"x")
    artifacts = scan_artifacts(str(tmp_path), include_bulk=True)
    names = [a["name"] for a in artifacts]
    assert "video1_labels.npy" in names


def test_scan_categorizes_story_clips_separately_from_state_clips(tmp_path):
    results_dir = tmp_path / "results"
    clips_dir = tmp_path / "clips"
    results_dir.mkdir()
    (clips_dir / "stories" / "vid1").mkdir(parents=True)
    (clips_dir / "state_0").mkdir(parents=True)
    (clips_dir / "stories" / "vid1" / "f0-150.mp4").write_bytes(b"x")
    (clips_dir / "state_0" / "longest_01.mp4").write_bytes(b"x")

    artifacts = scan_artifacts(str(results_dir), clips_dir=str(clips_dir))
    by_rel = {a["rel_path"].replace("\\", "/"): a["category"] for a in artifacts}
    assert by_rel["clips/stories/vid1/f0-150.mp4"] == "Video Stories"
    assert by_rel["clips/state_0/longest_01.mp4"] == "Clips"


def test_scan_empty_dir(tmp_path):
    artifacts = scan_artifacts(str(tmp_path))
    assert artifacts == []


def test_scan_nonexistent_dir():
    artifacts = scan_artifacts("/nonexistent/path")
    assert artifacts == []


def test_scan_includes_clips_dir(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "comparison").mkdir()
    (results / "comparison" / "summary_table.csv").write_text("a,b\n1,2")

    clips = tmp_path / "clips"
    (clips / "state_0").mkdir(parents=True)
    (clips / "state_0" / "longest_01.mp4").write_bytes(b"vid")

    artifacts = scan_artifacts(str(results), clips_dir=str(clips))
    categories = {a["category"] for a in artifacts}
    assert "Clips" in categories
    assert "Summary" in categories
    clip_arts = [a for a in artifacts if a["category"] == "Clips"]
    assert clip_arts[0]["rel_path"].startswith("clips/")


# ────────────────────────────────────────── publication_bundle ──

def test_publication_bundle(tmp_path):
    results = tmp_path / "results"
    (results / "comparison").mkdir(parents=True)
    (results / "comparison" / "summary_table.csv").write_text("a,b\n1,2")
    (results / "comparison" / "state_by_context.png").write_bytes(b"PNG")
    (results / "shared").mkdir()
    (results / "shared" / "run_manifest.json").write_text('{"run_id":"test"}')
    (results / "shared" / "cluster_info.json").write_text('{"n":5}')
    (results / "motifs").mkdir()
    (results / "motifs" / "motif_summary.csv").write_text("x,y\n1,2")
    (results / "sequences").mkdir()
    (results / "sequences" / "video_story_bouts.csv").write_text("video_id,state\nv1,0")
    (results / "sequences" / "video_stories.csv").write_text("video_id,n_bouts\nv1,1")
    (results / "sequences" / "subject_journeys.csv").write_text("subject_id,timepoint\ns1,0")
    (results / "characterization").mkdir()
    (results / "characterization" / "state_occupancy.png").write_bytes(b"PNG")
    (results / "diagnostics").mkdir()
    (results / "diagnostics" / "umap_embedding_by_state.png").write_bytes(b"PNG")
    (results / "comparison" / "contrast_vector_comparison.png").write_bytes(b"PNG")

    out = str(tmp_path / "pub.zip")
    build_publication_bundle(str(results), out)

    assert os.path.exists(out)
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "comparison/summary_table.csv" in names
        assert "shared/run_manifest.json" in names
        assert "shared/cluster_info.json" in names
        assert "comparison/state_by_context.png" in names
        assert "characterization/state_occupancy.png" in names
        assert "diagnostics/umap_embedding_by_state.png" in names
        assert "comparison/contrast_vector_comparison.png" in names
        assert "motifs/motif_summary.csv" in names
        assert "sequences/video_story_bouts.csv" in names
        assert "sequences/video_stories.csv" in names
        assert "sequences/subject_journeys.csv" in names


def test_publication_bundle_skips_missing(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    out = str(tmp_path / "pub.zip")
    build_publication_bundle(str(results), out)
    assert os.path.exists(out)
    with zipfile.ZipFile(out) as zf:
        assert zf.namelist() == []


def test_publication_bundle_includes_metadata(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (tmp_path / "metadata.csv").write_text("filename,animal_id\na.mp4,m1")
    out = str(tmp_path / "pub.zip")
    build_publication_bundle(str(results), out)
    with zipfile.ZipFile(out) as zf:
        assert "metadata.csv" in zf.namelist()


# ──────────────────────────────────────── export / zip paths ──

def test_export_preserves_relative_paths(tmp_path):
    results = tmp_path / "results"
    (results / "motifs").mkdir(parents=True)
    (results / "motifs" / "bouts.csv").write_text("a,b\n1,2")
    (results / "motifs" / "motif_summary.csv").write_text("x,y\n3,4")

    artifacts = scan_artifacts(str(results))
    out = str(tmp_path / "test.zip")
    with zipfile.ZipFile(out, "w") as zf:
        for a in artifacts:
            zf.write(a["abs_path"], a["rel_path"])

    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
        assert "motifs/bouts.csv" in names
        assert "motifs/motif_summary.csv" in names


def test_export_zip_contains_expected(tmp_path):
    results = tmp_path / "results"
    (results / "comparison").mkdir(parents=True)
    (results / "comparison" / "summary_table.csv").write_text("a,b\n1,2")
    (results / "comparison" / "motifs.csv").write_text("m,t\n(0,1),bigram")
    (results / "diagnostics").mkdir()
    (results / "diagnostics" / "cluster_overview.png").write_bytes(b"PNG")

    artifacts = scan_artifacts(str(results))
    out = str(tmp_path / "all.zip")
    with zipfile.ZipFile(out, "w") as zf:
        for a in artifacts:
            zf.write(a["abs_path"], a["rel_path"])

    with zipfile.ZipFile(out) as zf:
        assert len(zf.namelist()) == 3


# ─────────────────────────────────────────────── format_size ──

def test_format_size():
    assert format_size(500) == "500 B"
    assert "KB" in format_size(2048)
    assert "MB" in format_size(5 * 1024 * 1024)
    assert "GB" in format_size(2 * 1024 * 1024 * 1024)
