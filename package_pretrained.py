"""
package_pretrained.py — Package the pretrained DLC model for distribution.

Run this once from the original researcher's machine after training and evaluation:

    python package_pretrained.py --output pretrained/mouse_8kp_v1/

This copies:
  - Trained model weights (snapshot-* files)
  - pose_cfg.yaml
  - A pretrained_info.json manifest

The output directory can then be zipped and attached to a GitHub Release so
new users can download and use the model without training.

Users install a downloaded package by unzipping into pretrained/:
    unzip mouse_8kp_v1.zip -d pretrained/

Then they run:
    python setup_dlc_training.py --use-pretrained mouse_8kp_v1
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys

import vieb_config


def find_best_snapshot(dlc_project: str) -> tuple[str, list[str]]:
    """
    Locate the best (highest-numbered or specifically named best) DLC snapshot.

    Returns (snapshot_prefix, [all_files_belonging_to_that_snapshot]).
    """
    # Look for a snapshot named snapshot-best or the highest snapshot-N
    patterns = [
        os.path.join(dlc_project, "dlc-models", "**", "train", "snapshot-*.index"),
    ]
    all_index_files = []
    for pattern in patterns:
        all_index_files.extend(glob.glob(pattern, recursive=True))

    if not all_index_files:
        print(
            f"\n[VIEB] Error: No snapshot weight files found in {dlc_project}\n"
            "Expected: snapshot-*.index files under dlc-models/\n"
            "Fix: Run  python setup_dlc_training.py --train  and wait for training to complete.\n"
        )
        sys.exit(1)

    # Prefer snapshot named "best" or the highest iteration number
    def _sort_key(p):
        name = os.path.basename(p)
        if "best" in name:
            return 10 ** 9  # highest priority
        m = re.search(r"snapshot-(\d+)", name)
        return int(m.group(1)) if m else 0

    best_index = sorted(all_index_files, key=_sort_key)[-1]
    prefix = best_index.replace(".index", "")

    # Gather all files belonging to this snapshot (TF checkpoint format)
    sibling_dir = os.path.dirname(best_index)
    snapshot_name = os.path.basename(prefix)
    related = [
        f for f in glob.glob(os.path.join(sibling_dir, "*"))
        if os.path.basename(f).startswith(snapshot_name)
        or os.path.basename(f) == "checkpoint"
    ]

    return prefix, related


def find_pose_cfg(dlc_project: str) -> str | None:
    """Return path to pose_cfg.yaml (inside train/ directory), or None."""
    candidates = glob.glob(
        os.path.join(dlc_project, "dlc-models", "**", "train", "pose_cfg.yaml"),
        recursive=True,
    )
    return candidates[0] if candidates else None


def read_mAP(dlc_project: str) -> float:
    """Try to extract mAP from evaluation results CSV, or return 0.0."""
    eval_csvs = glob.glob(
        os.path.join(dlc_project, "evaluation-results*", "**", "*.csv"),
        recursive=True,
    )
    for csv_path in eval_csvs:
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            # DLC evaluation CSVs typically have a column containing 'mAP' or 'AP'
            for col in df.columns:
                if "map" in col.lower() or col.lower() == "ap":
                    val = df[col].dropna()
                    if len(val):
                        return float(val.iloc[-1])
        except Exception:
            continue
    return 0.0


def main():
    parser = argparse.ArgumentParser(description="Package a pretrained DLC model for distribution")
    parser.add_argument("--output", default="pretrained/mouse_8kp_v1",
                        help="Output directory (default: pretrained/mouse_8kp_v1)")
    parser.add_argument("--model-name", default=None,
                        help="Model name (default: basename of --output)")
    parser.add_argument("--species", default="mouse")
    parser.add_argument("--trained-on", default="222 fear conditioning videos, C57BL/6 mice")
    parser.add_argument("--vieb-version", default="1.0")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    dlc_project = vieb_config.require_dlc_project_path()
    print(f"DLC project: {dlc_project}")

    model_name = args.model_name or os.path.basename(args.output.rstrip("/\\"))
    output_dir = os.path.abspath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output:      {output_dir}\n")

    # ---- Detect DLC version ----
    try:
        import deeplabcut
        dlc_version = deeplabcut.__version__
    except ImportError:
        dlc_version = "unknown"
    print(f"DLC version: {dlc_version}")

    # ---- Locate and copy weights ----
    print("\nLocating best snapshot...")
    snapshot_prefix, snapshot_files = find_best_snapshot(dlc_project)
    print(f"  Best snapshot: {os.path.basename(snapshot_prefix)}")
    print(f"  Files: {len(snapshot_files)}")

    for src in snapshot_files:
        dst = os.path.join(output_dir, os.path.basename(src))
        shutil.copy2(src, dst)
        print(f"  Copied: {os.path.basename(src)}")

    # ---- Copy pose_cfg.yaml (stripped of absolute paths) ----
    pose_cfg = find_pose_cfg(dlc_project)
    if pose_cfg:
        import yaml
        with open(pose_cfg, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        # Remove absolute paths — pretrained_manager.py will rewrite them on install
        for key in ["project_path", "init_weights", "snapshot_prefix"]:
            cfg.pop(key, None)

        # Replace any absolute path in init_weights if present
        if "init_weights" in cfg:
            cfg["init_weights"] = os.path.basename(str(cfg["init_weights"]))

        out_pose_cfg = os.path.join(output_dir, "pose_cfg.yaml")
        with open(out_pose_cfg, "w", encoding="utf-8") as f:
            yaml.dump(cfg, f, default_flow_style=False)
        print(f"\nCopied pose_cfg.yaml (paths stripped)")
    else:
        print("\nWARNING: pose_cfg.yaml not found — skipping.")

    # ---- Read mAP ----
    mAP = read_mAP(dlc_project)
    print(f"\nmAP from evaluation: {mAP}")

    # ---- Read keypoints from DLC config ----
    import yaml
    with open(os.path.join(dlc_project, "config.yaml"), encoding="utf-8") as f:
        dlc_cfg = yaml.safe_load(f) or {}
    keypoints = dlc_cfg.get("bodyparts", [
        "left_ear", "right_ear", "nose", "center",
        "left_hip", "right_hip", "tail_base", "tail_tip",
    ])

    # ---- Write manifest ----
    manifest = {
        "model_name": model_name,
        "species": args.species,
        "keypoints": keypoints,
        "trained_on": args.trained_on,
        "vieb_version": args.vieb_version,
        "dlc_version": dlc_version,
        "mAP": round(mAP, 4),
        "notes": args.notes,
    }
    manifest_path = os.path.join(output_dir, "pretrained_info.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written: {manifest_path}")
    print(json.dumps(manifest, indent=2))

    print(f"\n=== Done ===")
    print(f"Package ready at: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Zip the directory:  cd pretrained && zip -r {model_name}.zip {model_name}/")
    print(f"  2. Attach the zip to a GitHub Release.")
    print(f"  3. Update README.md 'Pretrained Models' table with mAP={mAP:.3f}")
    print(f"\nUsers install it with:")
    print(f"  unzip {model_name}.zip -d pretrained/")
    print(f"  python setup_dlc_training.py --use-pretrained {model_name}")


if __name__ == "__main__":
    main()
