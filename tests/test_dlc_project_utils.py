from __future__ import annotations

from datetime import date
from pathlib import Path

import dlc_project_utils as dpu
import vieb_config


def test_default_dlc_task_name_starts_with_dlc_project_and_date():
    assert (
        dpu.default_dlc_task_name("Luna Project", date(2026, 6, 30))
        == "DLC-Luna-Project-2026-06-30"
    )


def test_normalize_dlc_project_path_accepts_config_yaml(tmp_path):
    project = tmp_path / "DLC-Luna-2026-06-30-Carlos-2026-06-30"
    project.mkdir()
    config = project / "config.yaml"
    config.write_text("Task: DLC-Luna\n", encoding="utf-8")

    assert dpu.normalize_dlc_project_path(config) == project.resolve()
    assert dpu.dlc_config_path(config) == config.resolve()
    assert dpu.is_valid_dlc_project(config)


def test_has_trained_dlc_model_accepts_tensorflow_snapshots(tmp_path):
    project = tmp_path / "DLC-Luna-2026-06-30"
    train_dir = project / "dlc-models" / "iteration-0" / "DLC-trainset95shuffle2" / "train"
    train_dir.mkdir(parents=True)
    snapshot = train_dir / "snapshot-100.index"
    snapshot.write_text("", encoding="utf-8")

    assert dpu.has_trained_dlc_model(project)
    assert dpu.find_trained_dlc_snapshots(project) == [snapshot]


def test_has_trained_dlc_model_accepts_pytorch_snapshots(tmp_path):
    project = tmp_path / "DLC-Luna-2026-06-30"
    train_dir = (
        project
        / "dlc-models-pytorch"
        / "iteration-0"
        / "DLC-trainset95shuffle2"
        / "train"
    )
    train_dir.mkdir(parents=True)
    snapshot = train_dir / "snapshot-best-030.pt"
    snapshot.write_text("", encoding="utf-8")

    assert dpu.has_trained_dlc_model(project)
    assert dpu.find_trained_dlc_snapshots(project) == [snapshot]


def test_has_trained_dlc_model_false_without_snapshots(tmp_path):
    project = tmp_path / "DLC-Luna-2026-06-30"
    project.mkdir()
    (project / "config.yaml").write_text("Task: DLC-Luna\n", encoding="utf-8")

    assert not dpu.has_trained_dlc_model(project)
    assert dpu.find_trained_dlc_snapshots(project) == []


def test_discover_dlc_projects_supports_new_legacy_and_existing_dlc_names(tmp_path):
    new = tmp_path / "DLC-Luna-2026-06-30"
    legacy = tmp_path / "VIEB-Carlos-2026-02-11"
    accidental = tmp_path / "Luna DLC-Carlos-2026-06-30"
    unrelated = tmp_path / "Analysis"
    for folder in (new, legacy, accidental, unrelated):
        folder.mkdir()
    for folder in (new, legacy, accidental):
        (folder / "config.yaml").write_text("Task: test\n", encoding="utf-8")

    discovered = dpu.discover_dlc_projects(tmp_path)

    assert discovered == [new.resolve(), legacy.resolve(), accidental.resolve()]


def test_vieb_config_normalizes_explicit_config_yaml(monkeypatch, tmp_path):
    project = tmp_path / "DLC-Luna-2026-06-30"
    project.mkdir()
    (project / "config.yaml").write_text("Task: DLC-Luna\n", encoding="utf-8")

    monkeypatch.setattr(
        vieb_config,
        "_load_config",
        lambda: {"dlc_project_path": str(project / "config.yaml")},
    )
    monkeypatch.setattr(vieb_config, "PROJECT_ROOT", str(tmp_path))

    assert vieb_config.get_dlc_project_path() == str(project.resolve())


def test_gitignore_ignores_generated_dlc_outputs():
    text = Path(".gitignore").read_text(encoding="utf-8")
    assert "DLC*" in text
    assert "**/DLC*" in text
