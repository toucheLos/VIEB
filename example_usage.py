"""
example_usage.py

Example usage of VIEB for behavioral analysis.

This script demonstrates how to use VIEB programmatically
for analyzing mouse behavior from DeepLabCut pose data.
"""

import numpy as np
from pathlib import Path

# Import VIEB components
from tracking.deeplabcut_backend import DeepLabCutTracker
from ml import (
    PoseFeatureExtractor,
    BehaviorPreprocessor,
    BehaviorClusterer,
    AnomalyDetector,
    TemporalBehaviorModel,
    TemporalDataSplitter,
    BehaviorAnalyzer
)


def example_full_pipeline():
    """
    Complete example: video → pose → features → ML → insights
    """
    print("=" * 80)
    print("VIEB Example: Complete Behavioral Analysis Pipeline")
    print("=" * 80)
    print()

    # Configuration
    VIDEO_PATH = "path/to/your/mouse_video.mp4"
    DLC_CONFIG = "path/to/deeplabcut/config.yaml"
    OUTPUT_DIR = "./example_output"
    FPS = 30.0

    # Create output directory
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Step 1: Extract poses with DeepLabCut
    # -------------------------------------------------------------------------
    print("Step 1: Extracting poses with DeepLabCut...")

    tracker = DeepLabCutTracker({
        "dlc_config_path": DLC_CONFIG,
        "shuffle": 1,
        "pcutoff": 0.6
    })

    pose_data = tracker.analyze_video(VIDEO_PATH, output_dir=OUTPUT_DIR)
    pose = pose_data["pose"]
    confidence = pose_data["confidence"]

    print(f"  ✓ Extracted {pose.shape[0]} frames, {pose.shape[1]} keypoints")
    print()

    # -------------------------------------------------------------------------
    # Step 2: Extract behavioral features
    # -------------------------------------------------------------------------
    print("Step 2: Extracting behavioral features...")

    extractor = PoseFeatureExtractor(fps=FPS, smooth_window=5, feature_window=30)
    features = extractor.extract_features(pose, confidence)
    feature_matrix = features["flattened"]

    print(f"  ✓ Extracted {feature_matrix.shape[1]} features per frame")
    print()

    # -------------------------------------------------------------------------
    # Step 3: Preprocess features
    # -------------------------------------------------------------------------
    print("Step 3: Preprocessing features...")

    preprocessor = BehaviorPreprocessor(
        scaler_type="standard",
        use_pca=False,
        remove_outliers=True
    )

    features_normalized = preprocessor.fit_transform(feature_matrix)
    preprocessor.save(f"{OUTPUT_DIR}/preprocessor.pkl")

    print(f"  ✓ Normalized features saved")
    print()

    # -------------------------------------------------------------------------
    # Step 4: Discover behavioral states (clustering)
    # -------------------------------------------------------------------------
    print("Step 4: Discovering behavioral states...")

    clusterer = BehaviorClusterer(
        method="kmeans",
        n_clusters=None,  # Auto-detect optimal number
        auto_tune=True,
        max_clusters=15
    )

    cluster_labels = clusterer.fit_predict(features_normalized)

    print(f"  ✓ Discovered {clusterer.n_clusters} behavioral states")

    # Evaluate clustering quality
    metrics = clusterer.evaluate(features_normalized)
    print(f"  - Silhouette score: {metrics['silhouette_score']:.3f}")
    print()

    # Visualize
    clusterer.visualize_clusters(
        features_normalized,
        save_path=f"{OUTPUT_DIR}/clusters.png"
    )
    print(f"  ✓ Cluster visualization saved")
    print()

    # -------------------------------------------------------------------------
    # Step 5: Detect anomalous behaviors
    # -------------------------------------------------------------------------
    print("Step 5: Detecting anomalies...")

    detector = AnomalyDetector(
        input_dim=features_normalized.shape[1],
        latent_dim=16,
        hidden_dims=[128, 64, 32]
    )

    # Train autoencoder
    history = detector.train(
        features_normalized,
        n_epochs=100,
        batch_size=256,
        patience=10,
        verbose=False
    )

    print(f"  ✓ Autoencoder trained (final loss: {history['val_loss'][-1]:.6f})")

    # Detect anomalies
    is_anomaly, anomaly_scores = detector.detect_anomalies(features_normalized)

    print(f"  - Anomaly rate: {np.mean(is_anomaly):.2%}")
    print(f"  - Threshold: {detector.threshold:.6f}")

    # Save model
    detector.save(f"{OUTPUT_DIR}/anomaly_detector.pt")
    print(f"  ✓ Model saved")
    print()

    # -------------------------------------------------------------------------
    # Step 6: Temporal sequence analysis
    # -------------------------------------------------------------------------
    print("Step 6: Analyzing temporal dynamics...")

    # Create sequences (1 second windows)
    sequence_length = int(FPS)  # 1 second
    sequences, _ = TemporalDataSplitter.create_sequences(
        features_normalized,
        sequence_length=sequence_length,
        stride=5
    )

    print(f"  - Created {len(sequences)} sequences")

    # Train LSTM
    temporal_model = TemporalBehaviorModel(
        input_dim=features_normalized.shape[1],
        hidden_dim=64,
        num_layers=2,
        task="embedding"
    )

    history = temporal_model.train(
        sequences,
        n_epochs=50,
        batch_size=32,
        patience=10,
        verbose=False
    )

    print(f"  ✓ LSTM trained (final loss: {history['val_loss'][-1]:.4f})")

    # Get sequence embeddings
    embeddings = temporal_model.get_embeddings(sequences)
    print(f"  - Learned {embeddings.shape[1]}-dimensional embeddings")

    temporal_model.save(f"{OUTPUT_DIR}/temporal_model.pt")
    print(f"  ✓ Model saved")
    print()

    # -------------------------------------------------------------------------
    # Step 7: Comprehensive analysis
    # -------------------------------------------------------------------------
    print("Step 7: Generating analysis report...")

    analyzer = BehaviorAnalyzer(fps=FPS)

    # Analyze behavioral states
    state_analysis = analyzer.analyze_behavioral_states(
        cluster_labels,
        extractor.get_feature_names(n_keypoints=pose.shape[1]),
        feature_matrix
    )

    print(f"  ✓ Analyzed {state_analysis['n_states']} behavioral states")

    # Analyze anomalies
    anomaly_analysis = analyzer.analyze_anomalies(
        is_anomaly,
        anomaly_scores,
        features_normalized
    )

    print(f"  ✓ Detected {anomaly_analysis['n_anomaly_bouts']} anomalous bouts")

    # Analyze transitions
    transition_analysis = analyzer.analyze_temporal_transitions(cluster_labels)

    print(f"  ✓ Analyzed state transitions")

    # Generate visualizations
    analyzer.plot_behavior_summary(
        cluster_labels,
        anomaly_scores=anomaly_scores,
        save_path=f"{OUTPUT_DIR}/summary.png"
    )

    analyzer.plot_transition_matrix(
        np.array(transition_analysis["transition_matrix"]),
        transition_analysis["state_labels"],
        save_path=f"{OUTPUT_DIR}/transitions.png"
    )

    print(f"  ✓ Visualizations generated")
    print()

    # -------------------------------------------------------------------------
    # Step 8: Export results
    # -------------------------------------------------------------------------
    print("Step 8: Exporting results...")

    analyzer.export_results(OUTPUT_DIR)
    analyzer.generate_report(f"{OUTPUT_DIR}/report.txt")

    print(f"  ✓ Results exported to {OUTPUT_DIR}")
    print()

    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print()
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"  - Discovered {state_analysis['n_states']} behavioral states")
    print(f"  - Detected {anomaly_analysis['n_anomaly_bouts']} anomalies")
    print(f"  - Trained models saved for reuse")
    print()


def example_load_existing_pose():
    """
    Example: Load previously extracted pose data instead of running DLC
    """
    print("Example: Loading existing pose data")
    print("-" * 80)

    # Load pose data from DLC CSV output
    tracker = DeepLabCutTracker({"dlc_config_path": "config.yaml"})
    pose_data = tracker.load_outputs("path/to/dlc_output_dir/")

    # Continue with feature extraction
    extractor = PoseFeatureExtractor(fps=30.0)
    features = extractor.extract_features(
        pose_data["pose"],
        pose_data["confidence"]
    )

    print(f"Loaded {pose_data['pose'].shape[0]} frames")
    print(f"Extracted {features['flattened'].shape[1]} features")


def example_compare_conditions():
    """
    Example: Compare behavior between different experimental conditions
    """
    print("Example: Comparing behavioral conditions")
    print("-" * 80)

    # Analyze control group
    control_features = np.load("control_features.npy")
    clusterer = BehaviorClusterer(method="kmeans", n_clusters=8)
    control_labels = clusterer.fit_predict(control_features)

    # Analyze treatment group using same clustering
    treatment_features = np.load("treatment_features.npy")
    treatment_labels = clusterer.predict(treatment_features)

    # Compare state distributions
    control_dist = np.bincount(control_labels) / len(control_labels)
    treatment_dist = np.bincount(treatment_labels) / len(treatment_labels)

    print("State distribution comparison:")
    for i in range(len(control_dist)):
        print(f"  State {i}: Control={control_dist[i]:.2%}, "
              f"Treatment={treatment_dist[i]:.2%}")


def example_reuse_trained_models():
    """
    Example: Reuse previously trained models on new data
    """
    print("Example: Reusing trained models")
    print("-" * 80)

    # Load preprocessor
    preprocessor = BehaviorPreprocessor.load("output/preprocessor.pkl")

    # Load anomaly detector
    detector = AnomalyDetector(input_dim=50)  # Must match original
    detector.load("output/anomaly_detector.pt")

    # Apply to new data
    new_features = np.random.randn(1000, 50)  # Replace with actual features
    new_features_norm = preprocessor.transform(new_features)
    is_anomaly, scores = detector.detect_anomalies(new_features_norm)

    print(f"Detected {np.sum(is_anomaly)} anomalies in new data")


if __name__ == "__main__":
    print("VIEB Usage Examples")
    print()
    print("This script contains example usage patterns.")
    print("Uncomment the example you want to run.")
    print()

    # Uncomment to run examples:

    # example_full_pipeline()
    # example_load_existing_pose()
    # example_compare_conditions()
    # example_reuse_trained_models()

    print("Edit this file to enable specific examples.")
