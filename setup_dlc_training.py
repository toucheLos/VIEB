"""
DLC Training Setup Script for VIEB

Run this after placing your videos in raw_videos/.
It will:
1. Add all videos in raw_videos/videos/ to the DLC config
2. Extract frames for labeling
3. Launch the labeling GUI
4. After labeling, create the training dataset and start training

Usage:
    python setup_dlc_training.py              # Run full setup (add videos, extract, label)
    python setup_dlc_training.py --train      # Create training dataset and train (after labeling)
    python setup_dlc_training.py --evaluate   # Evaluate trained model
    python setup_dlc_training.py --analyze    # Analyze all videos with trained model
"""

import os
import sys
import glob
import argparse
import yaml
from contextlib import contextmanager


@contextmanager
def prevent_sleep():
    """Prevent Windows from sleeping during long-running tasks (like caffeinate on Mac)."""
    try:
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        print("  (sleep prevention active)")
    except AttributeError:
        pass  # non-Windows
    try:
        yield
    finally:
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)  # release
        except AttributeError:
            pass

# --- Python version check ---
if sys.version_info < (3, 10):
    print(f"WARNING: Python {sys.version_info.major}.{sys.version_info.minor} detected. Python 3.10+ is required.")
    sys.exit(1)
elif sys.version_info >= (3, 13):
    print(f"WARNING: Python {sys.version_info.major}.{sys.version_info.minor} may have compatibility issues. Python 3.10-3.12 is recommended.")

# --- Hardware detection ---
def detect_hardware():
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            batch_size = 16 if vram_gb >= 16 else 8
            print(f"GPU detected: {gpu_name} ({vram_gb:.1f} GB VRAM) — using batch size {batch_size}")
            return {"device": "cuda", "batch_size": batch_size}
        else:
            import multiprocessing
            cpus = multiprocessing.cpu_count()
            print(f"No GPU detected — using CPU ({cpus} cores), batch size 4. Training will be slow.")
            return {"device": "cpu", "batch_size": 4}
    except ImportError:
        print("WARNING: torch not installed — cannot detect hardware.")
        return {"device": "cpu", "batch_size": 4}

HW = detect_hardware()

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DLC_CONFIG = os.path.join(PROJECT_ROOT, "VIEB-Carlos-2026-02-11", "config.yaml")
RAW_VIDEOS_DIR = os.path.join(PROJECT_ROOT, "raw_videos")


def add_videos_to_config():
    """Register all mp4 videos in raw_videos/ to the DLC config."""
    with open(DLC_CONFIG, "r") as f:
        config = yaml.safe_load(f)

    video_files = glob.glob(os.path.join(RAW_VIDEOS_DIR, "*.mp4"))
    if not video_files:
        print("No .mp4 files found in raw_videos/. Add your videos there first.")
        sys.exit(1)

    # Remove stale paths that no longer exist on this machine
    stale = [p for p in config.get("video_sets", {}) if not os.path.isfile(p)]
    for p in stale:
        del config["video_sets"][p]
    if stale:
        print(f"  Removed {len(stale)} stale video path(s) from config.")

    existing = set(config.get("video_sets", {}).keys())
    added = 0

    for vpath in video_files:
        abs_path = os.path.abspath(vpath)
        if abs_path not in existing:
            config.setdefault("video_sets", {})[abs_path] = {"crop": "0, 640, 0, 480"}
            added += 1
            print(f"  Added: {os.path.basename(vpath)}")

    if added > 0 or stale:
        with open(DLC_CONFIG, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"\n{added} new video(s) registered in DLC config.")
    else:
        print("All videos already registered in DLC config.")

    print(f"Total videos in config: {len(config['video_sets'])}")
    return video_files


def extract_frames():
    """Extract frames from all videos for labeling."""
    import deeplabcut

    print("\nExtracting frames for labeling (kmeans selection)...")
    deeplabcut.extract_frames(
        DLC_CONFIG,
        mode="automatic",
        algo="kmeans",
        userfeedback=False,
    )
    print("Frame extraction complete.")


QUEUE_FILE = os.path.join(PROJECT_ROOT, "VIEB-Carlos-2026-02-11", "labeling_queue.txt")


def _get_queue():
    """Load or create a random shuffled queue of all video folders that have frames."""
    import random
    labeled_dir = os.path.join(PROJECT_ROOT, "VIEB-Carlos-2026-02-11", "labeled-data")
    has_frames = sorted([
        d for d in os.listdir(labeled_dir)
        if os.path.isdir(os.path.join(labeled_dir, d))
        and any(f.endswith(".png") for f in os.listdir(os.path.join(labeled_dir, d)))
    ])

    if os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE) as f:
            queue = [line.strip() for line in f if line.strip()]
        # Add any new folders not yet in the queue
        existing = set(queue)
        new = [d for d in has_frames if d not in existing]
        if new:
            random.shuffle(new)
            queue.extend(new)
            with open(QUEUE_FILE, "w") as f:
                f.write("\n".join(queue) + "\n")
            print(f"  Added {len(new)} new video(s) to queue.")
    else:
        queue = list(has_frames)
        random.shuffle(queue)
        with open(QUEUE_FILE, "w") as f:
            f.write("\n".join(queue) + "\n")
        print(f"  Created labeling queue with {len(queue)} videos (randomized).")

    return queue


def _is_labeled(folder_name):
    labeled_dir = os.path.join(PROJECT_ROOT, "VIEB-Carlos-2026-02-11", "labeled-data")
    folder = os.path.join(labeled_dir, folder_name)
    return any(f.startswith("CollectedData") and f.endswith(".h5") for f in os.listdir(folder))


def label_frames(index=None):
    """Launch the DLC labeling GUI for one video at a time.

    index: 1-based position in the queue, or None to open the next unlabeled video.
    """
    import deeplabcut, napari

    queue = _get_queue()
    total = len(queue)

    if index is not None:
        if index < 1 or index > total:
            print(f"ERROR: Index {index} out of range (queue has {total} videos).")
            sys.exit(1)
        video_name = queue[index - 1]
        status = "already labeled" if _is_labeled(video_name) else "unlabeled"
        print(f"\nOpening #{index}/{total}: {video_name}  [{status}]")
    else:
        # Find the next unlabeled video in queue order
        labeled_count = sum(1 for d in queue if _is_labeled(d))
        next_entry = next(((i + 1, d) for i, d in enumerate(queue) if not _is_labeled(d)), None)
        if next_entry is None:
            print(f"All {total} videos are labeled!")
            return
        index, video_name = next_entry
        print(f"\nProgress: {labeled_count}/{total} labeled.")
        print(f"Opening #{index}/{total}: {video_name}")

    print("\nLabel all 8 keypoints on each frame:")
    print("  left_ear, right_ear, nose, center,")
    print("  left_hip, right_hip, tail_base, tail_tip")
    print("\nSave with Ctrl+S before closing.\n")
    deeplabcut.label_frames(DLC_CONFIG, video_name)
    napari.run()


def check_labeled_data():
    """Verify that at least some frames have been labeled before training."""
    labeled_dir = os.path.join(PROJECT_ROOT, "VIEB-Carlos-2026-02-11", "labeled-data")
    h5_files = glob.glob(os.path.join(labeled_dir, "*", "CollectedData_Carlos.h5"))
    if not h5_files:
        print("\nERROR: No labeled data found!")
        print("You must label frames before training. Run:")
        print("  python setup_dlc_training.py --label")
        print("\nThis opens the napari GUI where you click on each keypoint per frame.")
        print("Save your work in the GUI, then close it, and re-run --train.")
        sys.exit(1)
    print(f"\nFound labeled data in {len(h5_files)} video(s).")


def create_training_dataset():
    """Create the training dataset from labeled frames."""
    import deeplabcut

    print("\nCreating training dataset...")
    deeplabcut.create_training_dataset(DLC_CONFIG, Shuffles=[2], userfeedback=False)
    print("Training dataset created.")


def train():
    """Train the DLC network."""
    import deeplabcut

    print("\nStarting training (this may take 30min - 2hrs)...")
    print("You can stop early with Ctrl+C — DLC saves snapshots periodically.\n")
    with prevent_sleep():
        deeplabcut.train_network(
            DLC_CONFIG,
            shuffle=2,
            maxiters=200000,
            displayiters=1000,
            saveiters=10000,
            batch_size=HW["batch_size"],
        )
    print("Training complete.")


def evaluate():
    """Evaluate the trained model."""
    import deeplabcut

    print("\nEvaluating model...")
    deeplabcut.evaluate_network(DLC_CONFIG, Shuffles=[2], plotting=True)
    print("Evaluation complete. Check the evaluation-results folder.")


def analyze_videos():
    """Run pose estimation on all videos."""
    import deeplabcut

    video_files = glob.glob(os.path.join(RAW_VIDEOS_DIR, "*.mp4"))
    print(f"\nAnalyzing {len(video_files)} video(s)...")
    with prevent_sleep():
        deeplabcut.analyze_videos(
            DLC_CONFIG,
            video_files,
            shuffle=2,
            save_as_csv=True,
            batchsize=HW["batch_size"],
        )
    print("Analysis complete. CSV outputs saved alongside videos.")


def main():
    parser = argparse.ArgumentParser(description="DLC Training Setup for VIEB")
    parser.add_argument("--label", nargs="?", const=0, type=int, metavar="N",
                        help="Open labeling GUI. No arg = next unlabeled. Pass a number (e.g. --label 39) to jump to that position in the queue.")
    parser.add_argument("--train", action="store_true", help="Create dataset and train (run after labeling)")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained model")
    parser.add_argument("--analyze", action="store_true", help="Analyze all videos with trained model")
    args = parser.parse_args()

    if args.label is not None:
        label_frames(index=args.label if args.label != 0 else None)
    elif args.train:
        check_labeled_data()
        create_training_dataset()
        train()
    elif args.evaluate:
        evaluate()
    elif args.analyze:
        analyze_videos()
    else:
        # Default: setup flow
        print("=== VIEB DLC Training Setup ===\n")
        print(f"DLC config: {DLC_CONFIG}")
        print(f"Videos dir: {RAW_VIDEOS_DIR}\n")

        print("Step 1: Registering videos...")
        add_videos_to_config()

        print("\nStep 2: Extracting frames...")
        extract_frames()

        print("\nStep 3: Labeling...")
        label_frames()

        print("\n=== Labeling complete! ===")
        print("Now run:  python setup_dlc_training.py --train")


if __name__ == "__main__":
    main()
