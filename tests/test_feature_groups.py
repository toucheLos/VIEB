"""Tests for config-driven keypoint groups in PoseFeatureExtractor."""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml.feature_extraction import PoseFeatureExtractor  # noqa: E402


def test_default_group_resolution_matches_luna_role_defaults():
    bodypart_names = [
        "left_ear",
        "right_ear",
        "nose",
        "center",
        "left_hip",
        "right_hip",
        "tail_base",
        "tail_tip",
    ]
    extractor = PoseFeatureExtractor(
        fps=30.0,
        use_wavelets=False,
        keypoint_roles={},
        bodypart_names=bodypart_names,
    )

    assert extractor._role_idx == extractor._DEFAULT_ROLE_IDX
    assert extractor._available_semantic == {"rearing_score", "head_angle"}

    report = extractor.get_feature_availability_report()
    assert report["groups"]["head"]["resolved"] is True
    assert report["groups"]["head"]["keypoints"] == ["nose", "left_ear", "right_ear"]
    assert report["groups"]["head"]["indices"] == [2, 0, 1]
    assert report["groups"]["tail"]["indices"] == [6, 7]
    assert report["available_features"] == ["head_angle", "rearing_score"]
    assert report["skipped_features"] == {}


def test_missing_head_group_skips_semantic_features():
    bodypart_names = [
        "left_ear",
        "right_ear",
        "nose",
        "center",
        "left_hip",
        "right_hip",
        "tail_base",
        "tail_tip",
    ]
    keypoint_roles = {
        "head": [],
        "body_center": ["center"],
        "hips": ["left_hip", "right_hip"],
        "tail": ["tail_base", "tail_tip"],
        "forepaws": [],
        "hindpaws": [],
    }
    extractor = PoseFeatureExtractor(
        fps=30.0,
        use_wavelets=False,
        keypoint_roles=keypoint_roles,
        bodypart_names=bodypart_names,
    )

    report = extractor.get_feature_availability_report()
    assert report["groups"]["head"]["resolved"] is False
    assert report["available_features"] == []
    assert report["skipped_features"] == {
        "rearing_score": "missing groups: head",
        "head_angle": "missing groups: head",
    }

    pose = np.zeros((5, len(bodypart_names), 2), dtype=np.float32)
    features = extractor.extract_features(pose)
    meta = extractor.get_feature_meta(n_keypoints=len(bodypart_names))
    assert "rearing_score" not in features
    assert "head_angle" not in features
    assert "rearing_score" not in meta["feature_names"]
    assert "head_angle" not in meta["feature_names"]
