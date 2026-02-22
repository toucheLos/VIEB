from tracking.deeplabcut_backend import DeepLabCutTracker
import numpy as np

# CONFIG (adjust paths)

config = {
    "dlc_config_path": "./dlc_config.yaml",
    "shuffle": 1,
    "video_type": "mp4",
    "pcutoff": 0.6,
}

video_path = "../raw_videos/20241113_Box_1_CFC_Day_0_(Context_A)_9001.mp4"

# RUN TRACKER

tracker = DeepLabCutTracker(config)
outputs = tracker.analyze_video(video_path)

# INSPECT OUTPUTS

pose = outputs["pose"]
confidence = outputs["confidence"]
keypoints = outputs["keypoints"]
metadata = outputs["metadata"]

print("=== DeepLabCut Output Verification ===")
print(f"Frames (T): {pose.shape[0]}")
print(f"Keypoints (K): {pose.shape[1]}")
print(f"Dimensions (D): {pose.shape[2]}")
print()
print("Keypoints:", keypoints)
print()
print("Pose array dtype:", pose.dtype)
print("Confidence array dtype:", confidence.dtype)
print()
print("Metadata:", metadata)

# Confidence

print("Any NaNs in pose:", np.isnan(pose).any())
print("Pose min:", np.nanmin(pose))
print("Pose max:", np.nanmax(pose))

print("Confidence range:",
      confidence.min(),
      confidence.max())

# Output new video which highlights the keypoints

from tracking.visualize import overlay_pose_on_video

overlay_pose_on_video(
    video_path,
    outputs,
    output_path="debug_overlay.mp4",
)

# Verify frame alignment

import cv2

cap = cv2.VideoCapture(video_path)
video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print("Video frames:", video_frames)
print("Pose frames:", pose.shape[0])