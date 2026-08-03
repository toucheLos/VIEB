"""Keypoint roles and selection.

The 8-point Luna lab mouse model, as labelled in the DLC project:

    left_ear, nose, right_ear, center, right_hip, left_hip, tail_base, tail_tip

`tail_tip` is dropped from the v2 representation entirely -- not merely excluded
from the alignment fit. It is the noisiest tracked point and carries little
signal for the behavioral states of interest, and its jitter would otherwise
enter both the Procrustes fit and the PCA coordinates.

Dropping it takes K from 8 to 7, so D = 2K = 14 and the aligned covariance has
rank 2K - 3 = 11 (see docs/REPRESENTATION_MATH.md section 1.4b).
"""

from __future__ import annotations

# Canonical ordering used when bodypart names are unavailable.
DEFAULT_BODYPARTS = [
    "left_ear",
    "right_ear",
    "nose",
    "center",
    "left_hip",
    "right_hip",
    "tail_base",
    "tail_tip",
]

# Excluded from the v2 representation. Kept as a set so adding another noisy
# point later is a one-line change rather than an edit to the selection logic.
DROPPED_BODYPARTS = frozenset({"tail_tip"})


def select_indices(bodyparts):
    """Return indices of `bodyparts` to keep, in their original order.

    Matching is case-insensitive and ignores surrounding whitespace, since
    DLC column names are researcher-entered and inconsistently cased.
    """
    keep = []
    for i, name in enumerate(bodyparts):
        if str(name).strip().lower() not in DROPPED_BODYPARTS:
            keep.append(i)
    if not keep:
        raise ValueError(
            f"all {len(bodyparts)} keypoints were dropped; "
            f"check bodypart names: {list(bodyparts)}"
        )
    return keep


def select(pose, conf, bodyparts):
    """Drop excluded keypoints from a (pose, conf, bodyparts) triple.

    pose : (T, K, 2)
    conf : (T, K) or None
    Returns (pose, conf, bodyparts) restricted to the kept keypoints.
    """
    idx = select_indices(bodyparts)
    kept_names = [bodyparts[i] for i in idx]
    return pose[:, idx, :], (None if conf is None else conf[:, idx]), kept_names
