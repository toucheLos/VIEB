import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from diagnose_clusters import _build_warning_strings, _recommended_compare_command


def test_cluster_diagnostic_warnings_are_project_agnostic():
    warnings = _build_warning_strings(
        umap_dims=2,
        n_clusters=2,
        noise_fraction=0.0,
        largest_state_occupancy=0.9,
        feature_meta={
            "semantic_features": [],
            "skipped_features": {
                "rearing_score": "missing groups: head, tail",
                "head_angle": "missing groups: head, tail",
            },
        },
    )

    assert any("2D UMAP is useful for visualization" in w for w in warnings)
    assert any("--umap-dims 5" in w for w in warnings)
    assert any("Clustering may be collapsed" in w for w in warnings)
    assert any("0 noise with one dominant state" in w for w in warnings)
    assert any("keypoint group mapping" in w for w in warnings)


def test_recommended_command_uses_compare_cli_flags():
    rec = {"mcs": 200}
    command = _recommended_compare_command(
        rec,
        umap_dims=2,
        min_samples=None,
        warnings=["Clustering may be collapsed. Try 5–10D UMAP and smaller min_samples."],
    )

    assert command == (
        "python compare.py --cluster --umap-dims 10 "
        "--min-cluster-size 200 --hdbscan-min-samples 10"
    )
