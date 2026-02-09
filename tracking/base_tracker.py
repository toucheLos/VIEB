"""
base_tracker.py

Defines the abstract interface that all tracking backends
must follow in VIEB.

This ensures that downstream behavior analysis code is completely
agnostic to *how* tracking is performed (DeepLabCut, SLEAP, etc.).
"""

from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
from typing import List


class Tracker(ABC):
    """
    Abstract base class for all pose-tracking backends.

    Any concrete tracker must implement:
      - track(): run pose estimation on a video
      - keypoint_names(): return ordered keypoint labels
    """

    @abstractmethod
    def track(self, video_path: Path) -> np.ndarray:
        """
        Run pose estimation on a single video.

        video_path : Path
            Filesystem path to the input video.

        Returns
        poses : np.ndarray
            Array of shape:
                (num_frames, num_keypoints, 2)

            Where:
                poses[t, k, 0] = x coordinate
                poses[t, k, 1] = y coordinate

        Notes
        - Coordinates are in pixel space
        - No normalization (centering/rotation) should happen here
        - Confidence filtering is backend-specific
        """
        raise NotImplementedError

    @abstractmethod
    def keypoint_names(self) -> List[str]:
        """
        Return the ordered list of keypoint names.
        The order MUST match the second axis of the array returned by `track()`.
        Example:
            ["nose", "left_ear", "right_ear", "tail_base", ...]

        Returns
        names : list[str]
        """
        raise NotImplementedError
