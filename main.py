"""
VIEB — Video Interpreter for Experimental Behavior
====================================================
Behavioral analysis pipeline: pose data → feature extraction → clustering → anomaly detection.

Usage
-----
Single video:
    python main.py --video raw_videos/my_video.mp4

All videos at once:
    python main.py --all

Options:
    --video PATH        Path to a single .mp4 video (DLC CSV must exist alongside it)
    --all               Process every video in raw_videos/ that has a DLC CSV
    --output DIR        Output directory (default: results/<video_stem>/)
    --fps FLOAT         Frame rate (default: 30.0)
    --n-clusters INT    Number of behavioral states (default: auto-detect)
    --no-anomaly        Skip anomaly detection (faster)
    --no-temporal       Skip LSTM temporal model (faster)
    --dlc-config PATH   DLC config.yaml (default: auto-detected)
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


# Locate DLC config

def _find_dlc_config(hint=None):
    if hint:
        return hint
    candidates = glob.glob("VIEB-*/config.yaml") + glob.glob("**/config.yaml", recursive=False)
    if not candidates:
        sys.exit("ERROR: Could not find DLC config.yaml. Pass --dlc-config explicitly.")
    return candidates[0]


def _load_bodyparts(config_path):
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("bodyparts", [])


# ---------------------------------------------------------------------------
# Load DLC pose data
# ---------------------------------------------------------------------------

def _find_dlc_csv(video_path):
    """Find the DLC-generated CSV for a given video file."""
    stem = os.path.splitext(video_path)[0]
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # DLC saves CSVs alongside the video with the scorer name appended
    patterns = [
        f"{stem}*.csv",
        os.path.join(video_dir, f"{video_name}*.csv"),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        # Exclude any metadata.csv or similar
        matches = [m for m in matches if "metadata" not in os.path.basename(m).lower()]
        if matches:
            return matches[0]
    return None


def load_pose(csv_path):
    """
    Load DLC CSV output into pose array.

    Returns
    -------
    pose : np.ndarray  shape (T, K, 2)  — x, y per keypoint per frame
    conf : np.ndarray  shape (T, K)     — likelihood per keypoint per frame
    bodyparts : list[str]
    """
    df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
    bodyparts = df.columns.get_level_values(1).unique().tolist()
    scorer = df.columns.get_level_values(0)[0]

    T = len(df)
    K = len(bodyparts)
    pose = np.zeros((T, K, 2))
    conf = np.zeros((T, K))

    for k, bp in enumerate(bodyparts):
        pose[:, k, 0] = df[(scorer, bp, "x")].values
        pose[:, k, 1] = df[(scorer, bp, "y")].values
        if (scorer, bp, "likelihood") in df.columns:
            conf[:, k] = df[(scorer, bp, "likelihood")].values
        else:
            conf[:, k] = 1.0

    return pose, conf, bodyparts


# ---------------------------------------------------------------------------
# Core analysis pipeline
# ---------------------------------------------------------------------------

def analyze_video(video_path, output_dir, fps=30.0, n_clusters=None,
                  run_anomaly=True, run_temporal=True, bodypart_names=None):
    from ml import (
        PoseFeatureExtractor,
        BehaviorPreprocessor,
        BehaviorClusterer,
        AnomalyDetector,
        BehaviorAnalyzer,
    )

    print(f"\n{'='*60}")
    print(f"Video: {os.path.basename(video_path)}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    # --- Find DLC CSV ---
    csv_path = _find_dlc_csv(video_path)
    if csv_path is None:
        print(f"  SKIP: No DLC CSV found for {os.path.basename(video_path)}")
        print(f"  Run:  python setup_dlc_training.py --analyze  first.")
        return None

    print(f"  Loading: {os.path.basename(csv_path)}")
    pose, conf, bodyparts = load_pose(csv_path)
    print(f"  Frames: {pose.shape[0]},  Keypoints: {pose.shape[1]}")

    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Feature extraction ---
    print("\n[1/4] Extracting behavioral features...")
    extractor = PoseFeatureExtractor(fps=fps)
    features_dict = extractor.extract_features(pose, confidence=conf)
    features_flat = extractor._flatten_features(features_dict)
    print(f"      Feature vector size: {features_flat.shape[1]}")

    # --- 2. Preprocessing ---
    print("[2/4] Preprocessing (normalize + PCA)...")
    preprocessor = BehaviorPreprocessor(use_pca=True, pca_variance=0.95)
    features_norm = preprocessor.fit_transform(features_flat)
    preprocessor.save(os.path.join(output_dir, "preprocessor.pkl"))
    print(f"      Reduced to {features_norm.shape[1]} components")

    # --- 3. Behavioral state clustering ---
    print("[3/4] Discovering behavioral states (clustering)...")
    clusterer = BehaviorClusterer(
        method="kmeans",
        n_clusters=n_clusters,
        auto_tune=(n_clusters is None),
        max_clusters=12,
    )
    cluster_labels = clusterer.fit_predict(features_norm)
    n_found = len(np.unique(cluster_labels[cluster_labels >= 0]))
    print(f"      Found {n_found} behavioral states")
    clusterer.visualize_clusters(
        features_norm,
        method="pca",
        save_path=os.path.join(output_dir, "clusters_pca.png"),
    )
    clusterer.plot_temporal_states(fps=fps, save_path=os.path.join(output_dir, "ethogram.png"))

    # --- 4. Anomaly detection ---
    anomaly_scores = None
    is_anomaly = None
    if run_anomaly:
        print("[4/4] Training anomaly detector (autoencoder)...")
        detector = AnomalyDetector(input_dim=features_norm.shape[1])
        detector.train(features_norm, n_epochs=50, verbose=False)
        is_anomaly, anomaly_scores = detector.detect_anomalies(features_norm)
        detector.save(os.path.join(output_dir, "anomaly_detector.pt"))
        detector.plot_reconstruction_errors(
            features_norm,
            save_path=os.path.join(output_dir, "anomaly_scores.png"),
        )
        pct = is_anomaly.mean() * 100
        print(f"      Anomalous frames: {pct:.1f}%")
    else:
        print("[4/4] Anomaly detection skipped (--no-anomaly)")

    # --- Analysis & report ---
    print("\nGenerating report...")
    analyzer = BehaviorAnalyzer(fps=fps)
    analyzer.analyze_behavioral_states(cluster_labels, extractor.get_feature_names(pose.shape[1]), features_flat)
    if anomaly_scores is not None:
        analyzer.analyze_anomalies(is_anomaly, anomaly_scores, features_flat)
    analyzer.analyze_temporal_transitions(cluster_labels)
    analyzer.plot_behavior_summary(
        cluster_labels,
        anomaly_scores if anomaly_scores is not None else np.zeros(len(cluster_labels)),
        save_path=os.path.join(output_dir, "behavior_summary.png"),
    )
    analyzer.export_results(output_dir)
    analyzer.generate_report(os.path.join(output_dir, "analysis_report.txt"))

    print(f"\nDone. Results saved to: {output_dir}")
    return output_dir


# CLI

def main():
    parser = argparse.ArgumentParser(
        description="VIEB behavioral analysis pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--video", metavar="PATH",
                        help="Path to a single .mp4 video to analyze")
    parser.add_argument("--all", action="store_true",
                        help="Analyze all videos in raw_videos/ that have a DLC CSV")
    parser.add_argument("--output", metavar="DIR",
                        help="Output directory (default: results/<video_stem>/)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Video frame rate (default: 30.0)")
    parser.add_argument("--n-clusters", type=int, default=None,
                        help="Number of behavioral states (default: auto-detect)")
    parser.add_argument("--no-anomaly", action="store_true",
                        help="Skip anomaly detection")
    parser.add_argument("--no-temporal", action="store_true",
                        help="Skip LSTM temporal model")
    parser.add_argument("--dlc-config", metavar="PATH",
                        help="Path to DLC config.yaml (auto-detected if omitted)")
    args = parser.parse_args()

    if not args.video and not args.all:
        parser.print_help()
        sys.exit(1)

    config_path = _find_dlc_config(args.dlc_config)

    if args.all:
        videos = sorted(glob.glob("raw_videos/*.mp4"))
        if not videos:
            sys.exit("No .mp4 files found in raw_videos/")
        print(f"Found {len(videos)} videos. Processing those with DLC output...")
        results = []
        for video in videos:
            if _find_dlc_csv(video) is None:
                continue
            stem = os.path.splitext(os.path.basename(video))[0]
            out = args.output or os.path.join("results", stem)
            result = analyze_video(
                video, out,
                fps=args.fps,
                n_clusters=args.n_clusters,
                run_anomaly=not args.no_anomaly,
                run_temporal=not args.no_temporal,
            )
            if result:
                results.append(result)
        print(f"\nCompleted {len(results)} videos. Results in results/")
    else:
        if not os.path.exists(args.video):
            sys.exit(f"ERROR: Video not found: {args.video}")
        stem = os.path.splitext(os.path.basename(args.video))[0]
        out = args.output or os.path.join("results", stem)
        analyze_video(
            args.video, out,
            fps=args.fps,
            n_clusters=args.n_clusters,
            run_anomaly=not args.no_anomaly,
            run_temporal=not args.no_temporal,
        )


if __name__ == "__main__":
    main()
