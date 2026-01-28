"""
deeplabcut_backend.py

DeepLabCut adapter for VIEB.

This module:
  - Lazily imports DeepLabCut (DLC)
  - Runs DLC inference
  - Loads and sanitizes DLC output
  - Converts everything into a clean NumPy array

No DeepLabCut-specific objects leak outside this file.
"""

from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

# Lazy import: only fails if this backend is actually used
try:
    import deeplabcut
except ImportError as e:
    raise RuntimeError(
        "DeepLabCut is not installed.\n"
        "Install it with:\n"
        "  pip install vieb[tracking]"
    ) from e

from .base_tracker import Tracker


class DeepLabCutTracker(Tracker):
    """
    DeepLabCut-based pose tracker.

    Parameters
    config_path : Path
        Path to the DeepLabCut project config.yaml
    shuffle : int, optional
        DLC shuffle index to use (default = 1)
    pcutoff : float, optional
        Likelihood cutoff for keypoint confidence
    use_cached : bool, optional
        If True, reuse existing DLC output if present
    """

    def __init__(
        self,
        config_path: Path,
        shuffle: int = 1,
        pcutoff: float = 0.9,
        use_cached: bool = True,
    ):
        self.config_path = Path(config_path)
        self.shuffle = shuffle
        self.pcutoff = pcutoff
        self.use_cached = use_cached

        if not self.config_path.exists():
            raise FileNotFoundError(f"DLC config not found: {self.config_path}")

    # Public API

    def track(self, video_path: Path) -> np.ndarray:
        """
        Run DeepLabCut inference on a video and return poses.

        Parameters
        ----------
        video_path : Path
            Path to input video.

        Returns
        -------
        poses : np.ndarray
            Shape: (frames, keypoints, 2)
        """
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(video_path)

        # Check for existing output (cache)
        output_file = self._find_output_file(video_path)

        if output_file is None or not self.use_cached:
            deeplabcut.analyze_videos(
                str(self.config_path),
                [str(video_path)],
                shuffle=self.shuffle,
                auto_track=True,
                save_as_csv=False,
            )
            output_file = self._find_output_file(video_path)

        if output_file is None:
            raise RuntimeError("DeepLabCut did not produce an output file.")

        return self._load_and_clean_output(output_file)

    def keypoint_names(self) -> List[str]:
        """
        Load keypoint (bodypart) names from the DLC config file.

        Returns
        -------
        names : list[str]
        """
        import yaml

        with open(self.config_path, "r") as f:
            cfg = yaml.safe_load(f)

        return cfg["bodyparts"]

    # Internal helpers

    def _find_output_file(self, video_path: Path) -> Path | None:
        """
        Locate the DeepLabCut output .h5 file for a given video.

        Returns
        -------
        Path or None
        """
        candidates = list(video_path.parent.glob("*.h5"))
        return candidates[0] if candidates else None

    def _load_and_clean_output(self, h5_path: Path) -> np.ndarray:
        """
        Load DLC output and convert to a clean NumPy array.

        Steps:
          - Load pandas MultiIndex DataFrame
          - Extract x, y, likelihood
          - Mask low-confidence points
          - Interpolate missing values

        Returns
        -------
        poses : np.ndarray
            Shape: (frames, keypoints, 2)
        """
        df = pd.read_hdf(h5_path)

        scorer = df.columns.levels[0][0]
        bodyparts = df.columns.levels[1]

        all_keypoints = []

        for bp in bodyparts:
            x = df[scorer][bp]["x"].to_numpy()
            y = df[scorer][bp]["y"].to_numpy()
            p = df[scorer][bp]["likelihood"].to_numpy()

            # Mask low-confidence detections
            x[p < self.pcutoff] = np.nan
            y[p < self.pcutoff] = np.nan

            # Interpolate missing values (linear, frame-wise)
            x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
            y = pd.Series(y).interpolate(limit_direction="both").to_numpy()

            all_keypoints.append(np.stack([x, y], axis=-1))

        # Stack into (frames, keypoints, 2)
        poses = np.stack(all_keypoints, axis=1)

        return poses
