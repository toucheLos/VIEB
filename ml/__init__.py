"""
ml - Machine Learning pipeline for VIEB behavioral analysis.

Main components:
- feature_extraction: Extract behavioral features from raw poses
- preprocessing: Normalize and prepare data for ML
- clustering: Discover discrete behavioral states
- anomaly_detection: Detect unusual/rare behaviors
- analysis: High-level analysis and visualization
"""

from .feature_extraction import PoseFeatureExtractor, resolve_feature_indices
from .preprocessing import BehaviorPreprocessor
from .clustering import BehaviorClusterer
from .analysis import BehaviorAnalyzer

__all__ = [
    "PoseFeatureExtractor",
    "resolve_feature_indices",
    "BehaviorPreprocessor",
    "BehaviorClusterer",
    "AnomalyDetector",
    "BehaviorAnalyzer",
]
