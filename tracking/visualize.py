"""
visualize.py

Visualization utilities for pose tracking and behavioral inspection.

Supports:
- Keypoint overlays on video
- Skeleton visualization
- Trajectory plots
"""

from typing import Dict, List, Optional, Tuple
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt


# Video overlay visualization

def overlay_pose_on_video(
    video_path: str,
    pose_data: Dict,
    output_path: Optional[str] = None,
    skeleton: Optional[List[Tuple[int, int]]] = None,
    point_radius: int = 4,
    confidence_threshold: float = 0.6,
):
    """
    Overlay tracked keypoints on a video.

    Parameters
    ----------
    video_path : str
        Path to input video.
    pose_data : dict
        Output dictionary from a tracker.
    output_path : str, optional
        Where to save the annotated video.
    skeleton : list of (i, j), optional
        Keypoint index pairs to draw skeleton lines.
    point_radius : int
        Radius of keypoint dots.
    confidence_threshold : float
        Minimum confidence required to draw a keypoint.
    """

    pose = pose_data["pose"]
    confidence = pose_data["confidence"]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    else:
        writer = None

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= pose.shape[0]:
            break

        for k in range(pose.shape[1]):
            if confidence[frame_idx, k] < confidence_threshold:
                continue

            x, y = pose[frame_idx, k]
            if np.isnan(x) or np.isnan(y):
                continue

            cv2.circle(
                frame,
                (int(x), int(y)),
                point_radius,
                (0, 255, 0),
                -1
            )

        if skeleton:
            for i, j in skeleton:
                if (
                    confidence[frame_idx, i] >= confidence_threshold
                    and confidence[frame_idx, j] >= confidence_threshold
                ):
                    xi, yi = pose[frame_idx, i]
                    xj, yj = pose[frame_idx, j]
                    if not any(np.isnan([xi, yi, xj, yj])):
                        cv2.line(
                            frame,
                            (int(xi), int(yi)),
                            (int(xj), int(yj)),
                            (255, 0, 0),
                            2
                        )

        if writer:
            writer.write(frame)
        else:
            cv2.imshow("Pose Overlay", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


# Trajectory visualization

def plot_trajectories(
    pose_data: Dict,
    keypoint_indices: Optional[List[int]] = None,
    invert_y: bool = True,
):
    """
    Plot 2D trajectories of selected keypoints.

    Parameters
    ----------
    pose_data : dict
        Tracker output dictionary.
    keypoint_indices : list[int], optional
        Which keypoints to plot (default: all).
    invert_y : bool
        Invert y-axis to match image coordinates.
    """

    pose = pose_data["pose"]
    keypoints = pose_data["keypoints"]

    if keypoint_indices is None:
        keypoint_indices = list(range(pose.shape[1]))

    plt.figure(figsize=(6, 6))

    for k in keypoint_indices:
        traj = pose[:, k, :]
        plt.plot(traj[:, 0], traj[:, 1], label=keypoints[k])

    if invert_y:
        plt.gca().invert_yaxis()

    plt.xlabel("X position (px)")
    plt.ylabel("Y position (px)")
    plt.title("Mouse Trajectories")
    plt.legend()
    plt.axis("equal")
    plt.show()
