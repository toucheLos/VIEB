"""The v1 engineered feature set: 51 features, or 91 with Morlet wavelets.

Adapter over ``ml.feature_extraction.PoseFeatureExtractor``. Two things about it
differ from every other representation here, and both are preserved rather than
harmonized because changing either would change what the ``vieb_v1`` arm computes:

- **It does not align.** No centroid is subtracted and no rotation is applied, so
  unlike ``identity``/``pca``/``diffusion`` this space still contains translation
  and heading — which is to say it still contains locomotion. That makes it the
  one VIEB representation that has not thrown away the channel freezing is
  defined by, and it is worth knowing that when reading the comparison.
- **Its windows are in frames, not seconds.** ``smooth_window=5`` and
  ``feature_window=30`` are 0.17 s and 1.0 s at Luna's 30 fps but 0.02 s and
  0.12 s at Spence's 250. They are exposed here as seconds *and* frames so a new
  run can be specified portably, defaulting to the frame values the existing
  outputs were produced with.

The extractor runs per recording, so no window crosses a seam. It does have two
different edge conventions internally — ``_compute_temporal_features`` truncates
its window at the head of a recording while ``_compute_movement_entropy``
zero-fills — which is a real inconsistency, reported rather than fixed.
"""

from __future__ import annotations

import numpy as np

from ..data.dataset import PoseDataset
from ..registry import REPRESENTATIONS
from .base import BaseRepresentation

#: v1's defaults, in frames, at the 30 fps they were chosen for.
DEFAULT_SMOOTH_FRAMES = 5
DEFAULT_FEATURE_FRAMES = 30


@REPRESENTATIONS.register("engineered91")
class Engineered91Representation(BaseRepresentation):
    """91 engineered features (51 without wavelets), one row per frame.

    Parameters:
      ``use_wavelets``     include the 40 Morlet wavelet features (default True).
      ``smooth_seconds``   Savitzky-Golay width; overrides ``smooth_frames``.
      ``feature_seconds``  temporal-statistics window; overrides ``feature_frames``.
      ``keypoint_roles``   config-driven anatomical groups, see the extractor.
    """

    name = "engineered91"

    def __init__(
        self,
        use_wavelets: bool = True,
        smooth_frames: int = DEFAULT_SMOOTH_FRAMES,
        feature_frames: int = DEFAULT_FEATURE_FRAMES,
        smooth_seconds: float | None = None,
        feature_seconds: float | None = None,
        keypoint_roles: dict | None = None,
    ):
        self.use_wavelets = bool(use_wavelets)
        self.smooth_frames = int(smooth_frames)
        self.feature_frames = int(feature_frames)
        self.smooth_seconds = smooth_seconds
        self.feature_seconds = feature_seconds
        self.keypoint_roles = keypoint_roles
        self.report_: dict = {}

    def fit_transform(self, data: PoseDataset) -> np.ndarray:
        from ml.feature_extraction import PoseFeatureExtractor

        smooth, window = self._windows(data)
        extractor = PoseFeatureExtractor(
            fps=data.fps,
            smooth_window=smooth,
            feature_window=window,
            use_wavelets=self.use_wavelets,
            keypoint_roles=self.keypoint_roles,
            bodypart_names=list(data.keypoint_names),
        )

        blocks = []
        for _, sl in data.slices():
            pose = np.asarray(data.keypoints[sl], dtype=np.float64)
            conf = None if data.confidence is None else data.confidence[sl]
            feats = extractor.extract_features(pose, conf)
            blocks.append(np.asarray(feats["flattened"]))

        X = np.concatenate(blocks, axis=0)
        self.report_["n_features"] = int(X.shape[1])
        self.report_["smooth_frames"] = smooth
        self.report_["feature_frames"] = window
        try:
            self.report_["feature_names"] = list(
                extractor.get_feature_names(data.n_keypoints)
            )
        except Exception:
            self.report_["feature_names"] = []
        self._names = self.report_["feature_names"]
        return self._check_output(X, data)

    def _windows(self, data: PoseDataset) -> tuple[int, int]:
        """Resolve both windows, preferring the seconds-valued form.

        This is the §6c conversion: a config in seconds means the same real
        duration on a 30 fps rig and a 250 fps one.
        """
        smooth = (
            data.seconds_to_frames(self.smooth_seconds)
            if self.smooth_seconds is not None else self.smooth_frames
        )
        window = (
            data.seconds_to_frames(self.feature_seconds)
            if self.feature_seconds is not None else self.feature_frames
        )
        # Savitzky-Golay needs an odd window strictly greater than its polyorder.
        if smooth % 2 == 0:
            smooth += 1
        return smooth, window

    def get_params(self) -> dict:
        return {
            "use_wavelets": self.use_wavelets,
            "smooth_frames": self.smooth_frames,
            "feature_frames": self.feature_frames,
            "smooth_seconds": self.smooth_seconds,
            "feature_seconds": self.feature_seconds,
            "keypoint_roles": self.keypoint_roles,
        }

    @property
    def channel_names(self) -> list[str]:
        return list(getattr(self, "_names", []))
