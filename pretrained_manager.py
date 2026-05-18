"""
pretrained_manager.py — Manage pretrained DLC pose models for VIEB.

Pretrained model packages live in pretrained/<model_name>/ and contain:
  - snapshot-*  (model weights)
  - pose_cfg.yaml
  - pretrained_info.json  (metadata)

A model package is self-contained and works without internet access after download.
Download packages from the GitHub Releases page and unzip into pretrained/.
"""

from __future__ import annotations

import datetime
import getpass
import glob
import json
import os
import platform
import shutil
import sys
from typing import Dict, List, Optional

import vieb_config

_PRETRAINED_DIR = os.path.join(vieb_config.PROJECT_ROOT, "pretrained")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_available_pretrained() -> List[Dict]:
    """
    Scan pretrained/ for model packages.  Each package must contain a
    pretrained_info.json to be listed.

    Returns a list of info dicts, one per model.
    """
    if not os.path.isdir(_PRETRAINED_DIR):
        return []

    models = []
    for entry in sorted(os.listdir(_PRETRAINED_DIR)):
        model_dir = os.path.join(_PRETRAINED_DIR, entry)
        info_path = os.path.join(model_dir, "pretrained_info.json")
        if os.path.isdir(model_dir) and os.path.exists(info_path):
            try:
                with open(info_path, encoding="utf-8") as f:
                    info = json.load(f)
                info["_path"] = model_dir
                models.append(info)
            except (json.JSONDecodeError, OSError):
                pass  # malformed package — skip silently
    return models


def load_pretrained_model(model_name: str, target_videos_dir: str) -> str:
    """
    Set up a new DLC project from a pretrained model package.

    Steps:
      1. Locate the pretrained package in pretrained/<model_name>/
      2. Create a new DLC project directory:
           VIEB-<username>-<YYYY-MM-DD>/  (standard DLC naming)
      3. Copy weights and config files into the correct sub-paths
      4. Rewrite config.yaml with correct absolute paths for this machine
      5. Persist the new project path to config.json

    Returns the path to the newly created DLC project directory.
    """
    model_dir = os.path.join(_PRETRAINED_DIR, model_name)
    _check_model_exists(model_name, model_dir)

    info = _load_info(model_dir)
    keypoints = info.get("keypoints", [])
    dlc_version = info.get("dlc_version", "2.3.x")

    # Build a project name using the current user and today's date
    username = _safe_username()
    date_str = datetime.date.today().strftime("%Y-%m-%d")
    project_name = f"VIEB-{username}-{date_str}"
    project_dir = os.path.join(vieb_config.PROJECT_ROOT, project_name)

    # If a project with this name already exists, extend the name
    if os.path.exists(project_dir):
        project_dir = project_dir + "_pretrained"
    os.makedirs(project_dir, exist_ok=True)

    # ---- Copy pose weights ----
    # DLC expects weights at:
    #   <project>/dlc-models/iteration-0/<project>-trainset95shuffle2/train/snapshot-*
    shuffle_name = f"{project_name}-trainset95shuffle2"
    train_dir = os.path.join(project_dir, "dlc-models", "iteration-0",
                             shuffle_name, "train")
    os.makedirs(train_dir, exist_ok=True)

    weight_files = (
        glob.glob(os.path.join(model_dir, "snapshot-*"))
        + glob.glob(os.path.join(model_dir, "*.index"))
        + glob.glob(os.path.join(model_dir, "*.data-*"))
        + glob.glob(os.path.join(model_dir, "*.meta"))
        + glob.glob(os.path.join(model_dir, "checkpoint"))
    )
    if not weight_files:
        print(
            f"\n[VIEB] Error: No model weight files found in {model_dir}\n"
            "Expected: snapshot-* files (TensorFlow checkpoints)\n"
            f"Fix: Re-download the {model_name} package from GitHub Releases.\n"
        )
        sys.exit(1)

    for src in weight_files:
        shutil.copy2(src, train_dir)
    print(f"  Copied {len(weight_files)} weight file(s) → {train_dir}")

    # ---- Copy pose_cfg.yaml ----
    pose_cfg_src = os.path.join(model_dir, "pose_cfg.yaml")
    if os.path.exists(pose_cfg_src):
        pose_cfg_dst = os.path.join(train_dir, "pose_cfg.yaml")
        shutil.copy2(pose_cfg_src, pose_cfg_dst)
        # Patch the init_weights path to point at the copied snapshot
        _patch_pose_cfg(pose_cfg_dst, train_dir)
        print(f"  Copied pose_cfg.yaml")

    # ---- Write config.yaml ----
    videos_abs = os.path.abspath(target_videos_dir)
    _write_dlc_config(
        project_dir=project_dir,
        project_name=project_name,
        username=username,
        keypoints=keypoints,
        videos_dir=videos_abs,
        shuffle_name=shuffle_name,
    )
    print(f"  Wrote config.yaml")

    # ---- Persist to config.json ----
    vieb_config.set_dlc_project_path(project_dir)
    print(f"\nDLC project configured: {project_dir}")
    print(f"Path saved to config.json as 'dlc_project_path'.")

    return project_dir


def analyze_with_pretrained(
    model_name: str,
    videos_dir: str,
    output_dir: str,
) -> List[str]:
    """
    Run DLC pose estimation on all .mp4 videos in videos_dir using a pretrained
    model (no training required).

    The DLC project must already have been set up via load_pretrained_model().
    Output CSVs are written alongside the videos (DLC default behaviour).

    Returns a list of CSV paths that were created.
    """
    try:
        import deeplabcut
    except ImportError:
        print(
            "\n[VIEB] Error: DeepLabCut is not installed.\n"
            "Expected: deeplabcut importable in the current environment.\n"
            "Fix: pip install 'deeplabcut[tf]'  OR  pip install -e '.[tracking]'\n"
        )
        sys.exit(1)

    dlc_config = vieb_config.require_dlc_project_path()
    config_yaml = os.path.join(dlc_config, "config.yaml")

    video_files = glob.glob(os.path.join(videos_dir, "*.mp4"))
    if not video_files:
        print(
            f"\n[VIEB] Error: No .mp4 files found in {videos_dir}\n"
            "Expected: At least one .mp4 video\n"
            "Fix: Copy your videos into raw_videos/ and re-run.\n"
        )
        sys.exit(1)

    print(f"Running pose estimation on {len(video_files)} video(s)...")
    try:
        import torch
        hw = detect_hw()
        batchsize = hw["batch_size"]
    except ImportError:
        batchsize = 4

    deeplabcut.analyze_videos(
        config_yaml,
        video_files,
        shuffle=2,
        save_as_csv=True,
        destfolder=output_dir if output_dir != videos_dir else None,
        batchsize=batchsize,
    )

    csv_paths = glob.glob(os.path.join(output_dir, "*.csv"))
    return csv_paths


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_model_exists(model_name: str, model_dir: str) -> None:
    if not os.path.isdir(model_dir):
        available = [m["model_name"] for m in list_available_pretrained()]
        print(
            f"\n[VIEB] Error: Pretrained model '{model_name}' not found.\n"
            f"Expected: Directory at {model_dir}\n"
        )
        if available:
            print(f"Available models: {', '.join(available)}")
            print(f"Fix: python setup_dlc_training.py --use-pretrained {available[0]}")
        else:
            print(
                "No pretrained models found in pretrained/\n"
                "Fix: Download a model from GitHub Releases and unzip it into pretrained/\n"
                "     Example: pretrained/mouse_8kp_v1/"
            )
        sys.exit(1)


def _load_info(model_dir: str) -> dict:
    info_path = os.path.join(model_dir, "pretrained_info.json")
    if not os.path.exists(info_path):
        return {}
    with open(info_path, encoding="utf-8") as f:
        return json.load(f)


def _safe_username() -> str:
    try:
        name = getpass.getuser()
        # Keep only alphanumeric + underscore to be safe in directory names
        import re
        name = re.sub(r"[^a-zA-Z0-9_]", "", name) or "user"
        return name
    except Exception:
        return "user"


def detect_hw() -> dict:
    try:
        import torch
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            return {"device": "cuda", "batch_size": 16 if vram_gb >= 16 else 8}
    except ImportError:
        pass
    return {"device": "cpu", "batch_size": 4}


def _patch_pose_cfg(pose_cfg_path: str, train_dir: str) -> None:
    """Update init_weights in pose_cfg.yaml to point at the snapshot in train_dir."""
    import yaml as _yaml

    try:
        with open(pose_cfg_path, encoding="utf-8") as f:
            cfg = _yaml.safe_load(f) or {}

        snapshots = glob.glob(os.path.join(train_dir, "snapshot-*.index"))
        if snapshots:
            # Take the highest-numbered snapshot
            snapshots.sort()
            snapshot_base = snapshots[-1].replace(".index", "")
            cfg["init_weights"] = snapshot_base

        with open(pose_cfg_path, "w", encoding="utf-8") as f:
            _yaml.dump(cfg, f, default_flow_style=False)
    except Exception:
        pass  # Non-fatal: DLC will fall back to its own path resolution


def _write_dlc_config(
    project_dir: str,
    project_name: str,
    username: str,
    keypoints: List[str],
    videos_dir: str,
    shuffle_name: str,
) -> None:
    """Write a minimal DLC config.yaml with correct absolute paths."""
    import yaml as _yaml

    # DLC expects bodyparts, skeleton, and some global settings
    if not keypoints:
        keypoints = [
            "left_ear", "right_ear", "nose", "center",
            "left_hip", "right_hip", "tail_base", "tail_tip",
        ]

    config = {
        "Task": "VIEB",
        "scorer": username,
        "date": datetime.date.today().strftime("%B%Y"),
        "multianimalproject": False,
        "identity": None,
        "project_path": project_dir,
        "video_sets": {},
        "bodyparts": keypoints,
        "start": 0,
        "stop": 1,
        "numframes2pick": 20,
        "skeleton": [],
        "skeleton_color": "black",
        "pcutoff": 0.6,
        "dotsize": 12,
        "alphavalue": 0.7,
        "colormap": "rainbow",
        "TrainingFraction": [0.95],
        "iteration": 0,
        "default_net_type": "resnet_50",
        "default_augmenter": "default",
        "snapshotindex": -1,
        "batch_size": 8,
        "cropping": False,
        "x1": 0,
        "x2": 640,
        "y1": 0,
        "y2": 480,
        "corner2move2": [50, 50],
        "move2corner": True,
    }

    config_path = os.path.join(project_dir, "config.yaml")
    with open(config_path, "w", encoding="utf-8") as f:
        _yaml.dump(config, f, default_flow_style=False, sort_keys=False)
