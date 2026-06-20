"""Tests for motif discovery output contracts."""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def test_cmd_motifs_writes_expected_outputs(tmp_path, monkeypatch):
    project_dir = tmp_path / "project"
    results_dir = project_dir / "results"
    char_dir = results_dir / "characterization"
    shared_dir = results_dir / "shared"
    comparison_dir = results_dir / "comparison"
    motifs_dir = results_dir / "motifs"
    char_dir.mkdir(parents=True)
    shared_dir.mkdir(parents=True)
    project_dir.mkdir(exist_ok=True)

    metadata_rows = []
    bouts_rows = []
    stem_states = {
        "a1": ("A", "1", [0, 1, 0, 1]),
        "a2": ("A", "2", [0, 1, 0, 1]),
        "a3": ("A", "3", [0, 1, 0, 1]),
        "a4": ("A", "4", [0, 1, 0, 1]),
        "b1": ("B", "5", [2, 2, 2, 2]),
        "b2": ("B", "6", [2, 2, 2, 2]),
        "b3": ("B", "7", [2, 2, 2, 2]),
        "b4": ("B", "8", [2, 2, 2, 2]),
    }
    for stem, (context, animal_id, states) in stem_states.items():
        metadata_rows.append(
            {
                "filename": f"{stem}.mp4",
                "context": context,
                "animal_id": animal_id,
                "day": 1,
                "experiment": "CFC",
            }
        )
        for i, state in enumerate(states):
            bouts_rows.append(
                {
                    "stem": stem,
                    "state": state,
                    "start_frame": i * 10,
                    "end_frame": i * 10 + 9,
                    "start_sec": float(i),
                    "end_sec": float(i) + 0.3,
                    "duration_sec": 0.3,
                    "context": context,
                    "animal_id": animal_id,
                    "day": 1,
                    "experiment": "CFC",
                }
            )

    metadata_path = project_dir / "metadata.csv"
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    pd.DataFrame(bouts_rows).to_csv(char_dir / "bouts.csv", index=False)
    with open(shared_dir / "cluster_info.json", "w") as f:
        json.dump({"n_clusters": 3}, f)

    config_path = project_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "results_dir": str(results_dir),
                "metadata_csv_path": str(metadata_path),
            }
        )
    )
    app_config_path = tmp_path / "app_config.json"
    app_config_path.write_text(json.dumps({"active_project": str(project_dir)}))

    import vieb_config as vc
    monkeypatch.setattr(vc, "_APP_CONFIG_PATH", str(app_config_path))
    monkeypatch.setattr(vc, "_CONFIG_PATH", str(config_path))

    import compare
    import behavioral_fingerprint as bf

    compare.cmd_motifs()

    motifs_path = comparison_dir / "motifs.csv"
    assert motifs_path.exists()
    motifs = pd.read_csv(motifs_path)
    assert {"type", "motif", "enrichment_ratio", "flagged", "enriched_context", "p_value", "count_total"} <= set(motifs.columns)
    assert ((motifs["type"] == "bigram") & (motifs["motif"] == "(0, 1)")).any()

    top = bf._get_top_motifs(motifs, n=5)
    assert (0, 1) in top

    assert (motifs_dir / "bouts.csv").exists()
    assert (motifs_dir / "motif_sequences.csv").exists()
    assert (motifs_dir / "motif_summary.csv").exists()
    assert (motifs_dir / "motif_context_enrichment.csv").exists()
