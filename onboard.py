#!/usr/bin/env python
"""
VIEB Stage 0 — Headless Project Onboarder
------------------------------------------
A single-command, GUI-free copy of the Stage 0 onboarding panel
(``Stage0ReadinessPanel`` in ``user_interface.py``). It creates/selects a
project, generates ``metadata.csv`` from detected raw videos, and prints the
same readiness checklist the GUI shows — so VIEB can be onboarded on a headless
HPC cluster.

Usage:
    python onboard.py                       # onboard the project in this repo
    python onboard.py --path /scratch/exp1  # target / create a project elsewhere
    python onboard.py --check               # read-only readiness report (no writes)
    python onboard.py --force               # regenerate metadata.csv even if present
    python onboard.py --json                # machine-readable summary (for sbatch logs)

Exit codes:
    0  ready for the pipeline (metadata valid, no blank required fields)
    1  onboarded but needs manual attention (blank fields / incomplete / mapping)
    2  hard failure (no data source, invalid project, metadata could not be built)

Only Stage 0 is performed here — nothing from Stage 1+ (pose estimation,
feature extraction, clustering, …) is run. This script imports no Qt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import project_manager as _pm

ROOT = Path(__file__).resolve().parent
APP_CONFIG_PATH = ROOT / "app_config.json"

# ---------------------------------------------------------------------------
# Wording copied verbatim from Stage0ReadinessPanel (user_interface.py).
# Kept in sync so the CLI and GUI describe the same states identically.
# ---------------------------------------------------------------------------
STATUS_LABEL = {
    "ready_for_stage1": "Ready",
    "ready_for_stage2": "Ready",
    "incomplete": "Needs Setup",
    "needs_data": "Needs Data",
    "needs_metadata": "Needs Metadata",
    "needs_column_mapping": "Needs Mapping",
    "legacy_paths": "Path Issue",
    "auto_detected": "Detected",
    "no_project": "No Project",
    "picker_required": "Choose Project",
}

ACTION_TEXT = {
    "no_project":
        "No project selected. Choose or create a project to begin.",
    "picker_required":
        "Multiple projects detected. Choose one to continue.",
    "auto_detected":
        "A project was detected. Activate it to continue.",
    "legacy_paths":
        "This project points to old repo-root metadata/results. "
        "VIEB can migrate those into the project and keep raw videos external.",
    "needs_data":
        "No session-defining data source was found. "
        "Add raw videos, pose CSVs, H5 pose data, or a metadata manifest.",
    "needs_metadata":
        "Metadata is missing. VIEB can create it from detected session files.",
    "needs_column_mapping":
        "Metadata was found but the session identifier column "
        "could not be determined.",
    "incomplete":
        "Check the project config to complete readiness.",
    "ready_for_stage1":
        "Project ready for pose estimation. Continue to Stage 1.",
    "ready_for_stage2":
        "Pose-tracking data and metadata are ready. You can skip Stage 1.",
}

# ASCII markers so output is safe on any terminal/encoding (HPC, Windows cp1252).
_CHECK_MARKER = {"green": "[ok]", "yellow": "[warn]", "red": "[fail]"}

# States that mean "onboarded and ready to run the pipeline".
_READY_STATES = {"ready_for_stage1", "ready_for_stage2"}
# States that are a soft "needs manual attention" (exit 1).
_ATTENTION_STATES = {"incomplete", "needs_column_mapping"}


def _classify(project: Path, root: Path, validation=None):
    """Post-selection half of ``Stage0ReadinessPanel._determine_state``.

    Given a concrete project directory, return ``(state, validation, ctx)``
    using the exact ordering the GUI uses. Shared by the target-centric and
    discovery paths of ``determine_state``.
    """
    if validation is None:
        validation = _pm.validate_project(project, root)
    ctx: dict = {}
    paths = validation.paths or _pm.resolve_project_paths_for_project(
        project, validation.config, root
    )
    ctx["paths"] = paths
    if any(info.origin == "invalid_repo_root_fallback" for info in paths.values()):
        return ("legacy_paths", validation, ctx)

    source = validation.source or _pm.session_source_status(
        project, validation.config, root
    )
    ctx["source"] = source
    has_raw = source.get("raw_videos", 0) > 0
    has_pose = source.get("pose_csvs", 0) > 0 or bool(source.get("h5"))
    if not has_raw and not has_pose:
        return ("needs_data", validation, ctx)

    meta = paths["metadata"].path
    if not meta.exists():
        return ("needs_metadata", validation, ctx)

    mapping = _pm.column_mapping_status(project, validation.config)
    ctx["mapping"] = mapping
    if not mapping.get("mapped"):
        return ("needs_column_mapping", validation, ctx)

    if not _pm.onboarding_complete(project):
        return ("incomplete", validation, ctx)
    if has_raw and not has_pose:
        return ("ready_for_stage1", validation, ctx)
    return ("ready_for_stage2", validation, ctx)


def determine_state(root: Path, app_config_path: Path, target: Path,
                    allow_discovery: bool = True):
    """Target-centric headless port of ``Stage0ReadinessPanel._determine_state``.

    Classifies the project at ``target`` directly (so ``--check --path X``
    reports X, not whatever happens to be the active project). When ``target``
    is not yet a project and ``allow_discovery`` is set (i.e. no explicit
    ``--path`` was given), defers to project discovery for guidance
    (``auto_detected`` / ``picker_required`` / ``no_project``). With an explicit
    ``--path`` that isn't a project, reports ``no_project`` for that path rather
    than silently reporting some other detected project.

    Returns ``(state, selected, validation, ctx)``.
    """
    target = Path(target).resolve()
    if (target / _pm.PROJECT_CONFIG_NAME).exists() or _pm.is_valid_project(target):
        state, validation, ctx = _classify(target, root)
        return (state, None, validation, ctx)

    if not allow_discovery:
        return ("no_project", None, None, {})

    selected = _pm.select_startup_project(root, app_config_path)
    if not selected.active_project:
        if selected.action == "picker_required":
            return ("picker_required", selected, None, {})
        return ("no_project", selected, None, {})
    validation = _pm.validate_project(selected.active_project, root)
    if selected.action == "auto_selected" and not _pm.onboarding_complete(
        selected.active_project
    ):
        return ("auto_detected", selected, validation, {})
    state, validation, ctx = _classify(selected.active_project, root, validation)
    return (state, selected, validation, ctx)


def _looks_like_legacy_root(target: Path) -> bool:
    """Repo-root legacy layout: raw videos and/or metadata live at the root."""
    if target.resolve() != ROOT.resolve():
        return False
    return (target / "metadata.csv").exists() or (target / "raw_videos").is_dir()


def _ensure_project(target: Path, name: str, raw_videos: str | None) -> Path:
    """Create/register a project at ``target`` if one is not already there.

    A project is "already there" only when it has a ``config.json`` — the mere
    presence of a ``raw_videos/`` folder makes ``is_valid_project`` return True
    but leaves the project without a config, so guard on the config file.
    """
    if (target / _pm.PROJECT_CONFIG_NAME).exists():
        return target
    if _looks_like_legacy_root(target):
        print(f"No active project — registering existing repo-root layout as "
              f"project '{name}'.")
        return _pm.register_legacy_project(ROOT, APP_CONFIG_PATH, name=name)
    paths = {"raw_videos": str(Path(raw_videos).resolve())} if raw_videos else None
    print(f"No project at {target} — creating new project '{name}'.")
    return _pm.create_project(
        target, name, paths=paths, repo_root=ROOT, app_config_path=APP_CONFIG_PATH
    )


# Columns whose blanks are expected (filled in by the researcher later, not by
# onboarding) — noted, but described as manual rather than as an error.
_MANUAL_FILL_COLUMNS = {"fear"}


def _blank_field_report(meta_path: Path) -> list[str]:
    """Return human-readable lines for blank cells in the generated metadata.

    Filename-inferred metadata frequently leaves experimental columns
    (``context``, ``experiment``, ``day``, ``fear`` …) blank. These are not
    validation failures, but they must be filled before the pipeline runs, so
    scan every column except ``filename`` and report per-column blank counts.
    """
    if not meta_path or not meta_path.exists():
        return []
    import pandas as pd

    try:
        df = pd.read_csv(meta_path, dtype=str).fillna("")
    except Exception:
        return []
    lines: list[str] = []
    for col in df.columns:
        if col == "filename":
            continue
        n_blank = int((df[col].astype(str).str.strip() == "").sum())
        if not n_blank:
            continue
        suffix = " (fill in manually when scored)" if col in _MANUAL_FILL_COLUMNS else ""
        lines.append(f"'{col}': {n_blank} row(s) blank{suffix}")
    return lines


def _print_report(state, selected, validation, ctx, meta_path, target, as_json: bool) -> int:
    """Print the readiness report and return the process exit code."""
    blanks = _blank_field_report(meta_path) if meta_path else []

    if state in _READY_STATES and not blanks:
        code = 0
    elif state in _READY_STATES or state in _ATTENTION_STATES or blanks:
        code = 1
    else:
        code = 2

    project_name = validation.project_name if validation else "No project selected"
    source_msg = ""
    if validation is not None:
        source = ctx.get("source") or validation.source or _pm.session_source_status(
            validation.path, validation.config, ROOT
        )
        source_msg = source.get("message", "")

    if as_json:
        print(json.dumps({
            "state": state,
            "status": STATUS_LABEL.get(state, state),
            "project_name": project_name,
            "project_path": str(validation.path) if validation else None,
            "metadata_path": str(meta_path) if meta_path else None,
            "data_summary": source_msg,
            "next_action": ACTION_TEXT.get(state, ""),
            "checks": [c.__dict__ for c in (validation.checks if validation else [])],
            "blank_fields": blanks,
            "exit_code": code,
        }, indent=2))
        return code

    print("=" * 68)
    print(f"VIEB Stage 0 Onboarding — {STATUS_LABEL.get(state, state)}")
    print("=" * 68)
    print(f"Project : {project_name}")
    if validation is not None:
        print(f"Path    : {validation.path}")
    if source_msg:
        print(f"Data    : {source_msg}")
    print()

    if validation is not None:
        print("Readiness checklist:")
        for chk in validation.checks:
            marker = _CHECK_MARKER.get(chk.status, "[?]")
            detail = f" — {chk.message}" if chk.message else ""
            print(f"  {marker} {chk.label}{detail}")
        print()

    action = ACTION_TEXT.get(state, "")
    if action:
        print(f"Next: {action}")

    if state in ("no_project", "picker_required"):
        print("To create/onboard a project here, run:")
        print(f"  python onboard.py --path {target} --name <project-name>")

    if blanks:
        print()
        print("Note: metadata generated, but some fields are blank and must be")
        print("      filled in before running the pipeline:")
        for line in blanks:
            print(f"  - {line}")
        print("Edit the file, then re-check:")
        if meta_path:
            print(f"  <edit> {meta_path}")
        print("  python onboard.py --check")

    print()
    verdict = {0: "READY", 1: "NEEDS ATTENTION", 2: "NOT READY"}[code]
    print(f"Result: {verdict} (exit {code})")
    return code


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Headless VIEB Stage 0 project onboarder.",
    )
    parser.add_argument(
        "--path", default=None,
        help="Project directory to onboard / create (default: current directory).",
    )
    parser.add_argument(
        "--name", default=None,
        help="Project name for a newly created project (default: directory name).",
    )
    parser.add_argument(
        "--raw-videos", dest="raw_videos", default=None,
        help="Raw videos source directory when creating a new project.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Regenerate metadata.csv even if it already exists.",
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Read-only readiness report; make no changes.",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit a machine-readable JSON summary.",
    )
    args = parser.parse_args()

    if args.path:
        # Explicit --path: operate strictly on this directory.
        target = Path(args.path).expanduser().resolve()
        allow_discovery = False
    else:
        # No --path: prefer the active/detected project (like the GUI); fall
        # back to the current directory so a fresh setup with no active
        # project can be created in place.
        active = None
        try:
            active = _pm.select_startup_project(ROOT, APP_CONFIG_PATH).active_project
        except Exception:
            active = None
        target = Path(active).resolve() if active else Path.cwd().resolve()
        allow_discovery = True
    name = args.name or target.name

    if not args.check:
        try:
            project = _ensure_project(target, name, args.raw_videos)
            _pm.set_active_project(project, ROOT, APP_CONFIG_PATH)
            _pm.ensure_project_metadata(project, overwrite=args.force)
        except _pm.ProjectSelectionError as exc:
            print(f"Onboarding failed: {exc}", file=sys.stderr)
            return 2
        except Exception as exc:  # noqa: BLE001 - surface any setup error cleanly
            print(f"Onboarding failed: {exc}", file=sys.stderr)
            return 2

    state, selected, validation, ctx = determine_state(
        ROOT, APP_CONFIG_PATH, target, allow_discovery=allow_discovery
    )
    meta_path = None
    if validation is not None:
        paths = ctx.get("paths") or validation.paths
        if paths and "metadata" in paths:
            meta_path = paths["metadata"].path

    return _print_report(state, selected, validation, ctx, meta_path, target, args.json)


if __name__ == "__main__":
    sys.exit(main())
