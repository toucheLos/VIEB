import ast
from pathlib import Path


def _load_stages():
    src = Path(__file__).resolve().parents[1] / "user_interface.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "STAGES":
                    return ast.literal_eval(node.value)
    raise AssertionError("STAGES assignment not found")


def test_stage_zero_is_onboarding_not_project_onboarding():
    stages = _load_stages()

    assert stages[0]["id"] == 0
    assert stages[0]["name"] == "Onboarding"
    assert "Project Onboarding" not in stages[0]["name"]


def test_pipeline_stage_ids_are_in_product_order():
    stages = _load_stages()

    assert [(s["id"], s["name"]) for s in stages] == [
        (0, "Onboarding"),
        (1, "Pose Estimation / DLC Analysis"),
        (2, "Feature Extraction"),
        (3, "Preprocessing · UMAP · Clustering · Smoothing"),
        (4, "State Collapsing (optional)"),
        (5, "Report Generation"),
        (6, "Per-Animal Scalars"),
        (7, "Motif Discovery"),
        (8, "Generate Clips"),
        (9, "Add Videos"),
    ]
