"""
ml - Machine Learning pipeline for VIEB behavioral analysis.

This package provides tools for discovering subtle patterns in mouse behavior
from pose tracking data.

Main components:
- feature_extraction: Extract behavioral features from raw poses
- preprocessing: Normalize and prepare data for ML
- clustering: Discover discrete behavioral states
- anomaly_detection: Detect unusual/rare behaviors
- sequence_models: Learn temporal dynamics
- analysis: High-level analysis and visualization
"""

from .feature_extraction import PoseFeatureExtractor
from .preprocessing import BehaviorPreprocessor, TemporalDataSplitter, normalize_pose_to_body_frame
from .clustering import BehaviorClusterer
from .anomaly_detection import AnomalyDetector, BehaviorAutoencoder
from .sequence_models import TemporalBehaviorModel, BehaviorLSTM
from .analysis import BehaviorAnalyzer

__all__ = [
    "PoseFeatureExtractor",
    "BehaviorPreprocessor",
    "TemporalDataSplitter",
    "normalize_pose_to_body_frame",
    "BehaviorClusterer",
    "AnomalyDetector",
    "BehaviorAutoencoder",
    "TemporalBehaviorModel",
    "BehaviorLSTM",
    "BehaviorAnalyzer",
]
